//! Readers for the fixture files under `fixtures/`. The library's tests embed the same
//! files with `include_str!`; the trainer reads them from disk so a run can point at
//! another directory.

use std::path::Path;

use anyhow::Context;
use serde::Deserialize;
use serde::de::DeserializeOwned;

#[derive(Debug, Deserialize)]
pub struct ParserFixture {
    pub cases: Vec<ParserCase>,
}

#[derive(Debug, Deserialize)]
pub struct ParserCase {
    pub name: String,
    pub country: String,
    pub input: String,
    pub components: Vec<ExpectedComponent>,
}

#[derive(Debug, Deserialize)]
pub struct ExpectedComponent {
    pub label: String,
    pub text: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Deserialize)]
pub struct DetectorFixture {
    pub cases: Vec<DetectorCase>,
}

#[derive(Debug, Deserialize)]
pub struct DetectorCase {
    pub name: String,
    #[serde(default)]
    pub country: Option<String>,
    pub input: String,
    pub expected: Vec<ExpectedSpan>,
    #[serde(default)]
    pub must_not: Vec<MustNot>,
}

#[derive(Debug, Deserialize)]
pub struct ExpectedSpan {
    pub kind: String,
    pub text: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Deserialize)]
pub struct MustNot {
    pub kind: String,
    pub text: String,
}

#[derive(Debug, Deserialize)]
pub struct GrouperFixture {
    pub cases: Vec<GrouperCase>,
}

#[derive(Debug, Deserialize)]
pub struct GrouperCase {
    pub name: String,
    pub input: String,
    pub entities: Vec<ExpectedSpan>,
    pub contacts: Vec<ExpectedContact>,
    #[serde(default)]
    pub unassigned: Vec<usize>,
}

#[derive(Debug, Deserialize)]
pub struct ExpectedContact {
    #[serde(default)]
    pub person: Option<usize>,
    #[serde(default)]
    pub org: Option<usize>,
    #[serde(default)]
    pub addresses: Vec<usize>,
    #[serde(default)]
    pub emails: Vec<usize>,
    #[serde(default)]
    pub phones: Vec<usize>,
}

/// Every `*.json` in `dir`, sorted by name, as `(file stem, parsed)`.
pub fn load_dir<T: DeserializeOwned>(dir: &Path) -> anyhow::Result<Vec<(String, T)>> {
    let mut paths: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    paths.sort();
    paths
        .into_iter()
        .map(|p| {
            let text =
                std::fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
            let parsed =
                serde_json::from_str(&text).with_context(|| format!("parsing {}", p.display()))?;
            let stem = p
                .file_stem()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            Ok((stem, parsed))
        })
        .collect()
}

/// Fails when any expected span's offsets do not slice to its own `text`, so a typo in a
/// fixture is reported as such instead of as a baseline or model error.
pub fn check_offsets(
    parser: &[(String, ParserFixture)],
    detector: &[(String, DetectorFixture)],
    grouper: &[(String, GrouperFixture)],
) -> anyhow::Result<usize> {
    let mut checked = 0;
    let slices = |input: &str, start: usize, end: usize| input.get(start..end).map(str::to_string);
    for (file, fixture) in parser {
        for case in &fixture.cases {
            for c in &case.components {
                checked += 1;
                if slices(&case.input, c.start, c.end).as_deref() != Some(c.text.as_str()) {
                    anyhow::bail!(
                        "parser/{file}: `{}` component {:?} does not slice to its text",
                        case.name,
                        c.text
                    );
                }
            }
        }
    }
    for (file, fixture) in detector {
        for case in &fixture.cases {
            for e in &case.expected {
                checked += 1;
                if slices(&case.input, e.start, e.end).as_deref() != Some(e.text.as_str()) {
                    anyhow::bail!(
                        "detector/{file}: `{}` span {:?} does not slice to its text",
                        case.name,
                        e.text
                    );
                }
            }
        }
    }
    for (file, fixture) in grouper {
        for case in &fixture.cases {
            for e in &case.entities {
                checked += 1;
                if slices(&case.input, e.start, e.end).as_deref() != Some(e.text.as_str()) {
                    anyhow::bail!(
                        "grouper/{file}: `{}` entity {:?} does not slice to its text",
                        case.name,
                        e.text
                    );
                }
            }
        }
    }
    Ok(checked)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
    }

    /// Every fixture offset must slice its own `text`, or a baseline failure is really a typo.
    #[test]
    fn fixture_offsets_slice_their_text() {
        let f = root().join("fixtures");
        let parser = load_dir::<ParserFixture>(&f.join("parser")).unwrap();
        let detector = load_dir::<DetectorFixture>(&f.join("detector")).unwrap();
        let grouper = load_dir::<GrouperFixture>(&f.join("grouper")).unwrap();
        let checked = check_offsets(&parser, &detector, &grouper).unwrap();
        assert!(checked > 40, "only {checked} spans checked");
    }
}
