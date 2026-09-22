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
        let mut checked = 0;
        for (_, fixture) in load_dir::<ParserFixture>(&f.join("parser")).unwrap() {
            for case in &fixture.cases {
                for c in &case.components {
                    assert_eq!(&case.input[c.start..c.end], c.text, "{}", case.name);
                    checked += 1;
                }
            }
        }
        for (_, fixture) in load_dir::<DetectorFixture>(&f.join("detector")).unwrap() {
            for case in &fixture.cases {
                for e in &case.expected {
                    assert_eq!(&case.input[e.start..e.end], e.text, "{}", case.name);
                    checked += 1;
                }
            }
        }
        for (_, fixture) in load_dir::<GrouperFixture>(&f.join("grouper")).unwrap() {
            for case in &fixture.cases {
                for e in &case.entities {
                    assert_eq!(&case.input[e.start..e.end], e.text, "{}", case.name);
                    checked += 1;
                }
            }
        }
        assert!(
            checked > 40,
            "expected the full fixture set, checked {checked}"
        );
    }
}
