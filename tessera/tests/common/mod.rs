//! Fixture loading shared by the integration tests. Fixtures are embedded with
//! `include_str!` so the same tests run in wasm, which has no filesystem.

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
