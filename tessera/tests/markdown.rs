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
    let international = (
        "international",
        INTERNATIONAL_MD.to_string(),
        MarkdownOptions::default(),
    );
    let inputs = cases()
        .map(|(file, case)| (file, case.input, options(&case.options)))
        .chain([international]);
    for (file, input, opts) in inputs {
        let query = Query {
            format: Format::Markdown(opts),
            ..Query::default()
        };
        let got = t.detect(&input, &query).unwrap();
        assert!(!got.is_empty(), "{file}");
        let js = js_sys::JsString::from(input.as_str());
        for e in &got {
            let a = input[..e.start].encode_utf16().count() as u32;
            let b = input[..e.end].encode_utf16().count() as u32;
            let sliced: String = js.slice(a, b).into();
            assert_eq!(sliced, e.text(&input), "{file}: utf-16 slice");
        }
    }
}

/// With `SEGMENT_BYTES` every fixture is one segment, so this proves the public plumbing; the
/// markdown module's unit tests cut the same fixtures at every blank line.
#[test]
fn segmented_selection_equals_whole_selection() {
    for (file, case) in cases() {
        let opts = options(&case.options);
        let mut joined = Selection::default();
        for (_, s) in tessera::markdown::segments(&case.input, &opts) {
            joined.scan.extend(s.scan);
            joined.skip.extend(s.skip);
            joined.links.extend(s.links);
        }
        joined.skip.sort_unstable();
        assert_eq!(joined, select(&case.input, &opts), "{file}/{}", case.name);
    }
}

const INTERNATIONAL_MD: &str = include_str!("../../fixtures/markdown/international.md");
const INTERNATIONAL_JSON: &str = include_str!("../../fixtures/markdown/international.json");

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn every_markdown_fixture_is_listed() {
    let mut listed: Vec<(&str, ())> = FIXTURES.iter().map(|(name, _)| (*name, ())).collect();
    listed.push(("international", ()));
    common::assert_lists_every_fixture("markdown", &listed);
}

#[derive(Deserialize)]
struct Document {
    options: Options,
    blocks: Vec<Range>,
    skip: Vec<Range>,
    links: Vec<Link>,
    entities: Vec<GoldEntity>,
    contacts: Vec<GoldContact>,
    unassigned: Vec<usize>,
    #[serde(default)]
    must_not: Vec<MustNot>,
    #[serde(default)]
    known_failures: Vec<String>,
}

#[derive(Deserialize)]
struct GoldEntity {
    kind: String,
    text: String,
    start: usize,
    end: usize,
    normalized: Option<String>,
    region: Option<String>,
}

#[derive(Deserialize)]
struct GoldContact {
    person: Option<usize>,
    org: Option<usize>,
    addresses: Vec<usize>,
    emails: Vec<usize>,
    phones: Vec<usize>,
}

fn international() -> Document {
    common::parse("international.json", INTERNATIONAL_JSON)
}

fn found(entities: &[&Entity], kind: &str, text: &str) -> bool {
    entities
        .iter()
        .any(|e| e.kind.as_str() == kind && e.text(INTERNATIONAL_MD) == text)
}

#[test]
fn international_selection_and_rules() {
    let doc = international();
    let opts = options(&doc.options);
    let selection = select(INTERNATIONAL_MD, &opts);
    assert_eq!(selection.scan, pairs(&doc.blocks));
    assert_eq!(selection.skip, pairs(&doc.skip));
    let links: Vec<(&str, usize, usize, usize, usize)> = selection
        .links
        .iter()
        .map(|l| (l.dest.as_str(), l.start, l.end, l.text_start, l.text_end))
        .collect();
    let want: Vec<(&str, usize, usize, usize, usize)> = doc
        .links
        .iter()
        .map(|l| (l.target.as_str(), l.start, l.end, l.text_start, l.text_end))
        .collect();
    assert_eq!(links, want);

    let query = Query {
        format: Format::Markdown(opts),
        ..Query::default()
    };
    let got = rules_only().detect(INTERNATIONAL_MD, &query).unwrap();
    let want: Vec<&GoldEntity> = doc
        .entities
        .iter()
        .filter(|e| e.kind == "email" || e.kind == "phone")
        .collect();
    assert_eq!(got.len(), want.len(), "rule entities: {got:?}");
    for (e, w) in got.iter().zip(&want) {
        assert_eq!(
            (e.kind.as_str(), e.start, e.end),
            (w.kind.as_str(), w.start, w.end),
            "{}",
            w.text
        );
        assert_eq!(e.text(INTERNATIONAL_MD), w.text);
        assert_eq!(e.normalized, w.normalized, "{}", w.text);
        assert_eq!(e.region, w.region, "{}", w.text);
    }
    let got: Vec<&Entity> = got.iter().collect();
    for m in &doc.must_not {
        assert!(!found(&got, &m.kind, &m.text), "found {}", m.text);
    }
}

/// Every way the model's output departs from the gold document, as readable lines. Extra
/// entities outside every gold span are reported, not listed: they are quality findings.
fn model_departures(doc: &Document, out: &tessera::Extraction) -> Vec<String> {
    let all: Vec<&Entity> = out
        .contacts
        .iter()
        .flat_map(|c| c.entities())
        .chain(&out.unassigned)
        .collect();
    let is_gold = |e: &Entity| {
        doc.entities
            .iter()
            .any(|w| e.kind.as_str() == w.kind && (e.start, e.end) == (w.start, w.end))
    };
    let text = |e: &Entity| e.text(INTERNATIONAL_MD).to_string();
    let mut departures = Vec::new();
    for w in &doc.entities {
        if !all
            .iter()
            .any(|e| e.kind.as_str() == w.kind && (e.start, e.end) == (w.start, w.end))
        {
            departures.push(format!(
                "missing {} {:?} at {}..{}",
                w.kind, w.text, w.start, w.end
            ));
        }
    }
    for m in &doc.must_not {
        if found(&all, &m.kind, &m.text) {
            departures.push(format!("found {} {:?}", m.kind, m.text));
        }
    }
    let span = |i: usize| (doc.entities[i].start, doc.entities[i].end);
    let spans = |ids: &[usize]| ids.iter().map(|&i| span(i)).collect::<Vec<_>>();
    let sorted = |es: &[Entity]| {
        let mut v: Vec<(usize, usize)> = es.iter().map(|e| (e.start, e.end)).collect();
        v.sort_unstable();
        v
    };
    for (n, gold) in doc.contacts.iter().enumerate() {
        let person = gold.person.map(span);
        let Some(c) = out
            .contacts
            .iter()
            .find(|c| c.person.as_ref().map(|p| (p.start, p.end)) == person)
        else {
            departures.push(format!("contact {n}: no contact with person {person:?}"));
            continue;
        };
        if sorted(&c.emails) != spans(&gold.emails) {
            departures.push(format!("contact {n}: emails {:?}", sorted(&c.emails)));
        }
        if sorted(&c.phones) != spans(&gold.phones) {
            departures.push(format!("contact {n}: phones {:?}", sorted(&c.phones)));
        }
        if let Some(org) = gold.org
            && c.org.as_ref().map(|o| (o.start, o.end)) != Some(span(org))
        {
            departures.push(format!(
                "contact {n}: org {:?}, expected {:?}",
                c.org.as_ref().map(text),
                doc.entities[org].text
            ));
        }
        for &a in &gold.addresses {
            if !c.addresses.iter().any(|x| (x.start, x.end) == span(a)) {
                departures.push(format!(
                    "contact {n}: address {:?} not attached",
                    doc.entities[a].text
                ));
            }
        }
    }
    let unassigned_gold: Vec<(usize, usize)> = out
        .unassigned
        .iter()
        .filter(|e| is_gold(e))
        .map(|e| (e.start, e.end))
        .collect();
    let gold_unassigned = spans(&doc.unassigned);
    if unassigned_gold != gold_unassigned {
        departures.push(format!("unassigned gold entities {unassigned_gold:?}"));
    }
    let extra: Vec<String> = all
        .iter()
        .filter(|e| !is_gold(e))
        .map(|e| format!("{} {:?}", e.kind.as_str(), text(e)))
        .collect();
    if !extra.is_empty() {
        common::report!("international: extra entities: {}", extra.join(", "));
    }
    if out.contacts.len() > doc.contacts.len() {
        common::report!(
            "international: {} extra contacts",
            out.contacts.len() - doc.contacts.len()
        );
    }
    departures
}

/// The gold document stays as a person would label it; `known_failures` lists the model's
/// current departures from it, so a regression and an improvement both fail until the list is
/// updated in the same change.
#[test]
fn international_contacts_with_model() {
    let doc = international();
    let t = common::load_all();
    let query = Query {
        format: Format::Markdown(options(&doc.options)),
        country_hint: &["GB", "DE", "US"],
        ..Query::default()
    };
    let out = t.extract_contacts(INTERNATIONAL_MD, &query).unwrap();
    assert_eq!(model_departures(&doc, &out), doc.known_failures);
}
