//! `tessera::json::answer`, the in-process form of `tessera json`, against the committed model
//! directory.
#![cfg(all(feature = "json", not(target_arch = "wasm32")))]

use std::path::Path;

use serde_json::json;
use tessera::json::{Fault, answer};

const MODELS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models");

#[test]
fn a_request_is_answered_in_process() {
    let request =
        json!({"operation": "detect", "text": "Write to jordan@acme.example", "kinds": ["email"]});
    let response = answer(Path::new(MODELS), request.to_string().as_bytes()).unwrap();
    assert_eq!(
        serde_json::to_string(&response).unwrap(),
        r#"{"model":"tessera","operation":"detect","entities":[{"kind":"email","text":"jordan@acme.example","start":9,"end":28,"confidence":0.99,"review_recommended":false,"source":"rules","normalized":"jordan@acme.example"}]}"#
    );
}

#[test]
fn the_request_is_blamed_before_the_model_is_read() {
    let fault = answer(
        Path::new("/nonexistent"),
        br#"{"operation":"detect","text":"x","colour":1}"#,
    )
    .unwrap_err();
    assert!(matches!(fault, Fault::Request(m) if m.contains("unknown field")));
    let fault = answer(
        Path::new("/nonexistent"),
        br#"{"operation":"detect","text":"x"}"#,
    )
    .unwrap_err();
    assert!(matches!(fault, Fault::Run(m) if m.contains("bundle.json")));
}

#[test]
fn a_request_without_a_country_hint_gets_the_one_the_bundle_expects() {
    let phone = |hint: serde_json::Value| {
        let mut request =
            json!({"operation": "detect", "text": "Call (701) 555-0142 today", "kinds": ["phone"]});
        if !hint.is_null() {
            request["country_hint"] = hint;
        }
        answer(Path::new(MODELS), request.to_string().as_bytes()).unwrap()["entities"]
            .as_array()
            .unwrap()
            .len()
    };
    assert_eq!(
        phone(serde_json::Value::Null),
        1,
        "the bundle's US policy applies"
    );
    assert_eq!(phone(json!(["US"])), 1);
    assert_eq!(phone(json!([])), 0, "an explicit empty hint still wins");
}
