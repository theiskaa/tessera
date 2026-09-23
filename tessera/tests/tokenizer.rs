//! Fixture-driven tokenizer tests, natively and in the browser.

mod common;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

use tessera::internal::{Script, TokenClass, tokenize, utf16_offsets};

fn class_name(c: TokenClass) -> &'static str {
    match c {
        TokenClass::Alpha => "alpha",
        TokenClass::Digit => "digit",
        TokenClass::Alnum => "alnum",
        TokenClass::Punct => "punct",
        TokenClass::Space => "space",
        TokenClass::Newline => "newline",
        TokenClass::Other => "other",
    }
}

fn script_name(s: Script) -> &'static str {
    match s {
        Script::Latin => "latin",
        Script::Cyrillic => "cyrillic",
        Script::Georgian => "georgian",
        Script::Arabic => "arabic",
        Script::Hebrew => "hebrew",
        Script::Greek => "greek",
        Script::Han => "han",
        Script::Hiragana => "hiragana",
        Script::Katakana => "katakana",
        Script::Hangul => "hangul",
        Script::Thai => "thai",
        Script::Devanagari => "devanagari",
        Script::Other => "other",
    }
}

#[test]
fn fixtures_match() {
    for (file, fixture) in common::tokenizer_fixtures() {
        for case in &fixture.cases {
            let text = &case.input;
            let tokens = tokenize(text);
            let u16 = utf16_offsets(text);
            let joined: String = tokens.iter().map(|t| &text[t.start..t.end]).collect();
            assert_eq!(
                &joined, text,
                "{file}/{}: tokens must reproduce input",
                case.name
            );

            let got: Vec<(String, usize, usize, &str, &str)> = tokens
                .iter()
                .map(|t| {
                    (
                        text[t.start..t.end].to_string(),
                        t.start,
                        t.end,
                        class_name(t.class),
                        script_name(t.script),
                    )
                })
                .collect();
            let want: Vec<(String, usize, usize, &str, &str)> = case
                .tokens
                .iter()
                .map(|t| {
                    (
                        t.text.clone(),
                        t.start,
                        t.end,
                        t.class.as_str(),
                        t.script.as_str(),
                    )
                })
                .collect();
            assert_eq!(got, want, "{file}/{}", case.name);

            for t in &case.tokens {
                #[cfg(target_arch = "wasm32")]
                {
                    let js = js_sys::JsString::from(text.as_str());
                    let sliced: String = js.slice(u16[t.start], u16[t.end]).into();
                    assert_eq!(sliced, t.text, "{file}/{}: js slice", case.name);
                }
                assert_eq!(
                    &text[t.start..t.end],
                    t.text,
                    "{file}/{}: offsets slice text",
                    case.name
                );
                if let (Some(a), Some(b)) = (t.utf16_start, t.utf16_end) {
                    assert_eq!(
                        (u16[t.start], u16[t.end]),
                        (a, b),
                        "{file}/{}: utf16 for {:?}",
                        case.name,
                        t.text
                    );
                    let units: Vec<u16> = text.encode_utf16().collect();
                    let sliced = String::from_utf16(&units[a as usize..b as usize]).unwrap();
                    assert_eq!(sliced, t.text, "{file}/{}: utf16 slice", case.name);
                }
            }
        }
    }
}
