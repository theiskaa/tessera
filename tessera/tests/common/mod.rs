//! Fixture loading shared by the integration tests. Fixtures, golden vectors, and the bundle
//! are embedded at compile time, so the same test binaries run natively and in a browser,
//! where there is no filesystem.

#![allow(dead_code)]

use serde::Deserialize;

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

/// Prints a line in the test output: stderr natively, the browser console under wasm, where
/// `wasm-bindgen-test` relays it.
#[allow(unused_macros)]
macro_rules! report {
    ($($arg:tt)*) => {{
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_test::console_log!($($arg)*);
        #[cfg(not(target_arch = "wasm32"))]
        eprintln!($($arg)*);
    }};
}
#[allow(unused_imports)]
pub(crate) use report;

/// The release bundle, embedded.
pub const BUNDLE: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../models/tessera-v1.safetensors"
));

/// The bundle's integrity string, `sha256-<hex>`.
pub fn bundle_checksum() -> &'static str {
    include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../models/tessera-v1.sha256"
    ))
    .trim()
}

/// Asserts that `listed` names every JSON file in `fixtures/<dir>/`. A browser cannot list a
/// directory, so the fixture lists below spell out their files, and this catches one added on disk
/// but not to its list.
#[cfg(not(target_arch = "wasm32"))]
pub fn assert_lists_every_fixture(dir: &str, listed: &[(&str, impl Sized)]) {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../fixtures")
        .join(dir);
    let mut on_disk: Vec<String> = std::fs::read_dir(&path)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter(|name| name.ends_with(".json"))
        .collect();
    on_disk.sort();
    let mut listed: Vec<String> = listed
        .iter()
        .map(|(name, _)| format!("{name}.json"))
        .collect();
    listed.sort();
    assert_eq!(listed, on_disk, "fixtures/{dir}");
}

/// Parses an embedded JSON document, naming it in the panic when it is malformed.
pub fn parse<T: serde::de::DeserializeOwned>(name: &str, json: &str) -> T {
    serde_json::from_str(json).unwrap_or_else(|e| panic!("{name}: {e}"))
}

#[derive(Deserialize)]
pub struct TokenizerFixture {
    pub cases: Vec<TokenizerCase>,
}

#[derive(Deserialize)]
pub struct TokenizerCase {
    pub name: String,
    pub input: String,
    pub tokens: Vec<ExpectedToken>,
}

#[derive(Deserialize)]
pub struct ExpectedToken {
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub class: String,
    pub script: String,
    #[serde(default)]
    pub utf16_start: Option<u32>,
    #[serde(default)]
    pub utf16_end: Option<u32>,
}

pub fn tokenizer_fixtures() -> Vec<(&'static str, TokenizerFixture)> {
    [
        (
            "latin",
            include_str!("../../../fixtures/tokenizer/latin.json"),
        ),
        (
            "scripts",
            include_str!("../../../fixtures/tokenizer/scripts.json"),
        ),
        (
            "mixed",
            include_str!("../../../fixtures/tokenizer/mixed.json"),
        ),
    ]
    .into_iter()
    .map(|(name, src)| (name, parse(name, src)))
    .collect()
}

#[derive(Deserialize)]
pub struct RulesFixture {
    pub cases: Vec<RulesCase>,
}

#[derive(Deserialize)]
pub struct RulesCase {
    pub name: String,
    pub input: String,
    #[serde(default)]
    pub country_hint: Vec<String>,
    pub expected: Vec<ExpectedEntity>,
    #[serde(default)]
    pub synthetic_invalid: bool,
}

#[derive(Deserialize)]
pub struct ExpectedEntity {
    pub kind: String,
    pub text: String,
    pub start: usize,
    pub end: usize,
    #[serde(default)]
    pub normalized: Option<String>,
    #[serde(default)]
    pub region: Option<String>,
    pub min_confidence: f32,
}

pub fn rules_fixtures() -> Vec<(&'static str, RulesFixture)> {
    [
        ("email", include_str!("../../../fixtures/rules/email.json")),
        (
            "phone-us",
            include_str!("../../../fixtures/rules/phone-us.json"),
        ),
        (
            "phone-gb",
            include_str!("../../../fixtures/rules/phone-gb.json"),
        ),
        (
            "phone-de",
            include_str!("../../../fixtures/rules/phone-de.json"),
        ),
        (
            "phone-ge",
            include_str!("../../../fixtures/rules/phone-ge.json"),
        ),
        (
            "phone-jp",
            include_str!("../../../fixtures/rules/phone-jp.json"),
        ),
        (
            "documents",
            include_str!("../../../fixtures/rules/documents.json"),
        ),
    ]
    .into_iter()
    .map(|(name, src)| (name, parse(name, src)))
    .collect()
}

pub fn load_tessera() -> tessera::Tessera {
    load_with(tessera::Kind::Address | tessera::Kind::Email | tessera::Kind::Phone)
}

/// The versioned bundle loaded for every kind, both networks built.
pub fn load_all() -> tessera::Tessera {
    load_with(tessera::Kind::all())
}

fn load_with(kinds: tessera::KindSet) -> tessera::Tessera {
    tessera::Tessera::load(
        BUNDLE,
        tessera::Config {
            kinds,
            expected_checksum: Some(bundle_checksum()),
        },
    )
    .expect("bundle loads")
}

#[derive(Deserialize)]
pub struct Golden {
    pub input: GoldenInput,
    pub int8_logits: Vec<Vec<f32>>,
    pub decoded: Vec<u8>,
    pub tolerance: f32,
    /// Detector goldens only: whether each position lies inside an email or phone.
    #[serde(default)]
    pub masked: Vec<bool>,
}

#[derive(Deserialize)]
pub struct GoldenInput {
    pub text: String,
    pub tokens: Vec<(usize, usize)>,
    pub features: Vec<GoldenFeatures>,
}

#[derive(Deserialize)]
pub struct GoldenFeatures {
    pub ngram_ids: Vec<u32>,
    pub script: u8,
    pub shape: u8,
    pub flags: u32,
}

#[derive(Deserialize)]
pub struct ParserFixture {
    pub cases: Vec<ParserCase>,
}

#[derive(Deserialize)]
pub struct ParserCase {
    pub name: String,
    pub country: String,
    pub input: String,
    pub components: Vec<ExpectedComponent>,
    #[serde(default)]
    pub expected_failure: Option<String>,
}

#[derive(Deserialize)]
pub struct ExpectedComponent {
    pub label: String,
    pub text: String,
    pub start: usize,
    pub end: usize,
}

/// Parser fixtures, with every component's `text` checked against its offsets so a mistyped
/// offset fails here rather than as a model error.
pub fn parser_fixtures() -> Vec<(&'static str, ParserFixture)> {
    [
        ("gb", include_str!("../../../fixtures/parser/gb.json")),
        ("de", include_str!("../../../fixtures/parser/de.json")),
        ("ge", include_str!("../../../fixtures/parser/ge.json")),
        ("us", include_str!("../../../fixtures/parser/us.json")),
        ("jp", include_str!("../../../fixtures/parser/jp.json")),
    ]
    .into_iter()
    .map(|(name, src)| {
        let fixture: ParserFixture = parse(name, src);
        for case in &fixture.cases {
            let tokens = tessera::internal::tokenize(&case.input);
            let starts: Vec<usize> = tokens.iter().map(|t| t.start).collect();
            let ends: Vec<usize> = tokens.iter().map(|t| t.end).collect();
            for c in &case.components {
                assert_eq!(
                    case.input.get(c.start..c.end),
                    Some(c.text.as_str()),
                    "fixture {name}: `{}` offsets of {}",
                    case.name,
                    c.label
                );
                // A span that cuts a token can never be predicted: the model labels tokens.
                assert!(
                    starts.contains(&c.start) && ends.contains(&c.end),
                    "fixture {name}: `{}` {} {:?} cuts a token",
                    case.name,
                    c.label,
                    c.text
                );
            }
        }
        (name, fixture)
    })
    .collect()
}

#[derive(Deserialize)]
pub struct DetectorFixture {
    pub cases: Vec<DetectorCase>,
}

#[derive(Deserialize)]
pub struct DetectorCase {
    pub name: String,
    #[serde(default)]
    pub country: Option<String>,
    pub input: String,
    pub expected: Vec<DetectorExpected>,
    #[serde(default)]
    pub must_not: Vec<MustNot>,
    /// A case the current model gets wrong; the test asserts it still fails, so a fix is noticed.
    #[serde(default)]
    pub known_failure: bool,
}

#[derive(Deserialize)]
pub struct DetectorExpected {
    pub kind: String,
    pub text: String,
    pub start: usize,
    pub end: usize,
}

#[derive(Deserialize)]
pub struct MustNot {
    pub kind: String,
    pub text: String,
}

/// Detector fixtures, with every expected entity's `text` checked against its offsets.
pub fn detector_fixtures() -> Vec<(&'static str, DetectorFixture)> {
    [
        (
            "georgian",
            include_str!("../../../fixtures/detector/georgian.json"),
        ),
        (
            "japanese",
            include_str!("../../../fixtures/detector/japanese.json"),
        ),
        (
            "letterheads-de",
            include_str!("../../../fixtures/detector/letterheads-de.json"),
        ),
        (
            "negatives",
            include_str!("../../../fixtures/detector/negatives.json"),
        ),
        (
            "prose-en",
            include_str!("../../../fixtures/detector/prose-en.json"),
        ),
        (
            "signatures",
            include_str!("../../../fixtures/detector/signatures.json"),
        ),
        (
            "signatures-gb",
            include_str!("../../../fixtures/detector/signatures-gb.json"),
        ),
        (
            "tables",
            include_str!("../../../fixtures/detector/tables.json"),
        ),
    ]
    .into_iter()
    .map(|(name, src)| {
        let fixture: DetectorFixture = parse(name, src);
        for case in &fixture.cases {
            for e in &case.expected {
                assert_eq!(
                    case.input.get(e.start..e.end),
                    Some(e.text.as_str()),
                    "fixture {name}: `{}` offsets of {} {:?}",
                    case.name,
                    e.kind,
                    e.text
                );
            }
        }
        (name, fixture)
    })
    .collect()
}
