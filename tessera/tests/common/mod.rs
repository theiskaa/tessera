//! Fixture loading shared by the integration tests. Tokenizer, rule, and parser fixtures are
//! embedded with `include_str!` so they need no filesystem; the bundle and its golden vectors
//! are read from `models/`, so those tests run natively until Milestone 3 embeds them for wasm.

#![allow(dead_code)]

use serde::Deserialize;

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
    .map(|(name, src)| {
        (
            name,
            serde_json::from_str(src).unwrap_or_else(|e| panic!("fixture {name}: {e}")),
        )
    })
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
    .map(|(name, src)| {
        (
            name,
            serde_json::from_str(src).unwrap_or_else(|e| panic!("fixture {name}: {e}")),
        )
    })
    .collect()
}

fn models_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../models")
}

pub fn bundle_bytes() -> Vec<u8> {
    std::fs::read(models_dir().join("tessera-v1.safetensors"))
        .expect("models/tessera-v1.safetensors")
}

pub fn bundle_checksum() -> String {
    std::fs::read_to_string(models_dir().join("tessera-v1.sha256"))
        .expect("models/tessera-v1.sha256")
        .trim()
        .to_string()
}

pub fn load_tessera() -> tessera::Tessera {
    let checksum = bundle_checksum();
    tessera::Tessera::load(
        &bundle_bytes(),
        tessera::Config {
            kinds: tessera::Kind::Address | tessera::Kind::Email | tessera::Kind::Phone,
            expected_checksum: Some(&checksum),
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

/// Every `.json` file under `models/golden/<net>/`, sorted by name.
pub fn golden_files(net: &str) -> Vec<std::path::PathBuf> {
    let mut files: Vec<_> = std::fs::read_dir(models_dir().join("golden").join(net))
        .expect("golden directory")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    files.sort();
    files
}

pub fn load_json<T: serde::de::DeserializeOwned>(path: &std::path::Path) -> T {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
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
        let fixture: ParserFixture =
            serde_json::from_str(src).unwrap_or_else(|e| panic!("fixture {name}: {e}"));
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
