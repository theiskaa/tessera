//! Fixture-driven tests for `fixtures/markdown/`, natively and in wasm: the selection, the
//! rules-only detection path, and in wasm the UTF-16 slicing of a real JavaScript string.

#![cfg(feature = "markdown")]

mod common;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

use serde::Deserialize;
use tessera::markdown::{Selection, select};
use tessera::{Config, Entity, Format, Kind, MarkdownOptions, Query, Tessera};

const FIXTURES: &[(&str, &str)] = &[
    (
        "headings",
        include_str!("../../fixtures/markdown/headings.json"),
    ),
    (
        "paragraphs",
        include_str!("../../fixtures/markdown/paragraphs.json"),
    ),
    ("lists", include_str!("../../fixtures/markdown/lists.json")),
    (
        "blockquotes",
        include_str!("../../fixtures/markdown/blockquotes.json"),
    ),
    ("links", include_str!("../../fixtures/markdown/links.json")),
    (
        "tables",
        include_str!("../../fixtures/markdown/tables.json"),
    ),
    ("code", include_str!("../../fixtures/markdown/code.json")),
    ("html", include_str!("../../fixtures/markdown/html.json")),
    (
        "front_matter",
        include_str!("../../fixtures/markdown/front_matter.json"),
    ),
];

#[derive(Deserialize)]
struct File {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    options: Options,
    input: String,
    blocks: Vec<Range>,
    skip: Vec<Range>,
    links: Vec<Link>,
    expected: Vec<Expected>,
    #[serde(default)]
    must_not: Vec<MustNot>,
}

#[derive(Deserialize)]
struct Options {
    include_code: bool,
    include_html: bool,
    gfm_tables: bool,
}

#[derive(Deserialize)]
struct Range {
    start: usize,
    end: usize,
}

#[derive(Deserialize)]
struct Link {
    target: String,
    start: usize,
    end: usize,
    text_start: usize,
    text_end: usize,
}

#[derive(Deserialize)]
struct Expected {
    kind: String,
    text: String,
    start: usize,
    end: usize,
    normalized: Option<String>,
    region: Option<String>,
    min_confidence: f32,
}

#[derive(Deserialize)]
struct MustNot {
    kind: String,
    text: String,
}

fn cases() -> impl Iterator<Item = (&'static str, Case)> {
    FIXTURES.iter().flat_map(|(file, json)| {
        let parsed: File = serde_json::from_str(json).unwrap_or_else(|e| panic!("{file}: {e}"));
        parsed.cases.into_iter().map(move |c| (*file, c))
    })
}

fn options(o: &Options) -> MarkdownOptions {
    MarkdownOptions {
        include_code: o.include_code,
        include_html: o.include_html,
        gfm_tables: o.gfm_tables,
    }
}

fn query(case: &Case) -> Query<'static> {
    Query {
        format: Format::Markdown(options(&case.options)),
        ..Query::default()
    }
}

fn rules_only() -> Tessera {
    Tessera::load(
        &[],
        Config {
            kinds: Kind::Email | Kind::Phone,
            expected_checksum: None,
        },
    )
    .unwrap()
}

fn pairs(ranges: &[Range]) -> Vec<(usize, usize)> {
    ranges.iter().map(|r| (r.start, r.end)).collect()
}

fn inside_scan_outside_skip(e: &Entity, s: &Selection) -> bool {
    s.scan.iter().any(|&(a, b)| a <= e.start && e.end <= b)
        && !s.skip.iter().any(|&(a, b)| e.start < b && a < e.end)
}

#[test]
fn selection_matches_fixture() {
    for (file, case) in cases() {
        let got = select(&case.input, &options(&case.options));
        let label = format!("{file}/{}", case.name);
        assert_eq!(got.scan, pairs(&case.blocks), "{label}: scan");
        assert_eq!(got.skip, pairs(&case.skip), "{label}: skip");
        let links: Vec<(&str, usize, usize, usize, usize)> = got
            .links
            .iter()
            .map(|l| (l.dest.as_str(), l.start, l.end, l.text_start, l.text_end))
            .collect();
        let want: Vec<(&str, usize, usize, usize, usize)> = case
            .links
            .iter()
            .map(|l| (l.target.as_str(), l.start, l.end, l.text_start, l.text_end))
            .collect();
        assert_eq!(links, want, "{label}: links");
    }
}

#[test]
fn rules_only_detect_matches_fixture() {
    let t = rules_only();
    for (file, case) in cases() {
        let label = format!("{file}/{}", case.name);
        let got = t
            .detect(&case.input, &query(&case))
            .unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(
            got.len(),
            case.expected.len(),
            "{label}: count; got {got:?}"
        );
        for (e, want) in got.iter().zip(&case.expected) {
            assert_eq!(
                e.kind.as_str(),
                want.kind,
                "{label}: kind at {}",
                want.start
            );
            assert_eq!(
                (e.start, e.end),
                (want.start, want.end),
                "{label}: span of {}",
                want.text
            );
            assert_eq!(e.text(&case.input), want.text, "{label}: text");
            assert_eq!(
                e.normalized, want.normalized,
                "{label}: normalized of {}",
                want.text
            );
            assert_eq!(
                e.region.as_deref(),
                want.region.as_deref(),
                "{label}: region of {}",
                want.text
            );
            assert!(
                e.confidence >= want.min_confidence,
                "{label}: confidence {} of {}",
                e.confidence,
                want.text
            );
        }
        for m in &case.must_not {
            assert!(
                !got.iter()
                    .any(|e| e.kind.as_str() == m.kind && e.text(&case.input) == m.text),
                "{label}: must not find {} {}",
                m.kind,
                m.text
            );
        }
        let selection = select(&case.input, &options(&case.options));
        for e in &got {
            assert!(
                inside_scan_outside_skip(e, &selection),
                "{label}: {} escapes the mask",
                e.text(&case.input)
            );
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[test]
fn utf16_offsets_slice_the_js_string() {
    let t = rules_only();
    for (file, case) in cases() {
        let got = t.detect(&case.input, &query(&case)).unwrap();
        let js = js_sys::JsString::from(case.input.as_str());
        for e in &got {
            let a = case.input[..e.start].encode_utf16().count() as u32;
            let b = case.input[..e.end].encode_utf16().count() as u32;
            let sliced: String = js.slice(a, b).into();
            assert_eq!(
                sliced,
                e.text(&case.input),
                "{file}/{}: utf-16 slice",
                case.name
            );
        }
    }
}
