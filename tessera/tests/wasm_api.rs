//! The JavaScript bindings: object shape, UTF-16 offsets that slice the JS string, option
//! validation, and error codes. Browser-only; natively this file compiles to nothing.

#![cfg(all(target_arch = "wasm32", feature = "wasm"))]

mod common;

use js_sys::{Array, Function, JsString, Object, Promise, Reflect, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use wasm_bindgen_test::wasm_bindgen_test as test;

use tessera::internal::display_confidence;
use tessera::wasm::{JsTessera, create_instance};
use tessera::{Error, Query};

fn prop(obj: &JsValue, key: &str) -> JsValue {
    Reflect::get(obj, &JsValue::from_str(key)).unwrap()
}

fn has(obj: &JsValue, key: &str) -> bool {
    Reflect::has(obj, &JsValue::from_str(key)).unwrap()
}

fn string(obj: &JsValue, key: &str) -> String {
    prop(obj, key)
        .as_string()
        .unwrap_or_else(|| panic!("`{key}` is not a string"))
}

fn number(obj: &JsValue, key: &str) -> f64 {
    prop(obj, key)
        .as_f64()
        .unwrap_or_else(|| panic!("`{key}` is not a number"))
}

fn options(json: &str) -> JsValue {
    js_sys::JSON::parse(json).unwrap()
}

fn utf16_len(s: &str) -> f64 {
    s.encode_utf16().count() as f64
}

fn text(s: &str) -> JsValue {
    JsValue::from_str(s)
}

fn load(bytes: &[u8], json: &str) -> Result<JsTessera, JsValue> {
    JsTessera::load(&Uint8Array::from(bytes).into(), Some(options(json)))
}

fn address_options() -> String {
    format!(
        r#"{{"kinds":["address"],"integrity":"{}"}}"#,
        common::bundle_checksum()
    )
}

fn address_tessera() -> JsTessera {
    load(common::BUNDLE, &address_options()).unwrap()
}

fn rules_tessera() -> JsTessera {
    load(&[], r#"{"kinds":["email","phone"]}"#).unwrap()
}

fn kind_labels(tessera: &JsTessera) -> Vec<String> {
    tessera
        .kinds()
        .iter()
        .map(|k| k.as_string().unwrap())
        .collect()
}

async fn resolved(promise: Promise) -> JsValue {
    JsFuture::from(promise).await.unwrap()
}

async fn rejected(promise: Promise) -> JsValue {
    JsFuture::from(promise)
        .await
        .expect_err("the promise should reject")
}

fn assert_tessera_error(err: &JsValue, code: &str) {
    assert!(err.is_instance_of::<js_sys::Error>(), "{err:?}");
    assert_eq!(string(err, "name"), "TesseraError");
    assert_eq!(string(err, "code"), code);
    assert!(!string(err, "message").is_empty());
}

fn assert_type_error(err: &JsValue) {
    assert!(err.is_instance_of::<js_sys::TypeError>(), "{err:?}");
    assert!(!has(err, "code"), "a caller bug carries no library code");
}

/// Addresses whose components sit after a surrogate pair or a combining mark, where UTF-8 byte,
/// UTF-16 unit, and code point offsets all differ.
const WIDE_ADDRESSES: [&str; 3] = [
    "📍 Flat 2, 10 Downing Street, London SW1A 2AA",
    "Rue de l'E\u{301}glise 5, 75001 Paris",
    "𠮷田ビル 3階, 1-1 Chiyoda, Tokyo 100-0001",
];

async fn assert_parse_matches(
    native: &tessera::Tessera,
    js: &JsTessera,
    input: &str,
    include_uncertain: bool,
    label: &str,
) {
    let query = Query {
        include_uncertain,
        ..Query::default()
    };
    let want = native.parse_address(input, &query).unwrap();
    let opts = options(&format!(r#"{{"includeUncertain":{include_uncertain}}}"#));
    let got = resolved(js.parse_address(&text(input), Some(opts))).await;

    assert_eq!(string(&got, "kind"), "address", "{label}");
    assert_eq!(string(&got, "text"), input, "{label}");
    assert_eq!(number(&got, "start"), 0.0, "{label}");
    assert_eq!(number(&got, "end"), utf16_len(input), "{label}");
    assert_eq!(string(&got, "source"), "model", "{label}");
    assert_eq!(
        number(&got, "confidence"),
        display_confidence(want.confidence),
        "{label}"
    );
    assert_eq!(
        prop(&got, "reviewRecommended").as_bool(),
        Some(want.review_recommended),
        "{label}"
    );
    assert!(!has(&got, "normalized") && !has(&got, "region"), "{label}");

    let js_text = JsString::from(input);
    let components: Array = prop(&got, "components").dyn_into().unwrap();
    assert_eq!(
        components.length() as usize,
        want.components.len(),
        "{label}"
    );
    for (c, w) in components.iter().zip(&want.components) {
        let (start, end) = (number(&c, "start"), number(&c, "end"));
        assert_eq!(string(&c, "label"), w.label.as_str(), "{label}");
        assert_eq!(string(&c, "text"), w.text(input), "{label}");
        assert_eq!(start, utf16_len(&input[..w.start]), "{label}");
        assert_eq!(end, utf16_len(&input[..w.end]), "{label}");
        let sliced: String = js_text.slice(start as u32, end as u32).into();
        assert_eq!(sliced, w.text(input), "{label}");
        assert_eq!(
            number(&c, "confidence"),
            display_confidence(w.confidence),
            "{label}"
        );
    }
}

#[test]
async fn parse_address_matches_native_with_utf16_offsets() {
    let native = common::load_tessera();
    let js = address_tessera();
    for (file, fixture) in common::parser_fixtures() {
        for case in &fixture.cases {
            for include_uncertain in [false, true] {
                let label = format!("{file}/{} uncertain={include_uncertain}", case.name);
                assert_parse_matches(&native, &js, &case.input, include_uncertain, &label).await;
            }
        }
    }
}

#[test]
async fn offsets_after_surrogate_pairs_and_combining_marks() {
    let native = common::load_tessera();
    let js = address_tessera();
    for input in WIDE_ADDRESSES {
        // The comparison only means something if a component lies past the wide character.
        let last = native
            .parse_address(input, &Query::default())
            .unwrap()
            .components
            .last()
            .map(|c| c.start)
            .unwrap_or_else(|| panic!("{input}: no components"));
        assert_ne!(utf16_len(&input[..last]), last as f64, "{input}");
        for include_uncertain in [false, true] {
            assert_parse_matches(&native, &js, input, include_uncertain, input).await;
        }
    }
}

#[test]
async fn georgian_offsets_are_utf16() {
    let input = "რუსთაველის გამზირი 14, თბილისი 0108";
    let got = resolved(address_tessera().parse_address(&text(input), None)).await;
    let end = number(&got, "end");
    assert_eq!(end, utf16_len(input));
    assert_ne!(end, input.len() as f64);
}

#[test]
async fn empty_address_has_empty_components() {
    let got = resolved(address_tessera().parse_address(&text(""), Some(JsValue::NULL))).await;
    let components: Array = prop(&got, "components").dyn_into().unwrap();
    assert_eq!(components.length(), 0);
    assert_eq!(number(&got, "confidence"), 0.0);
    assert_eq!(number(&got, "end"), 0.0);
}

#[test]
async fn detect_matches_native_with_utf16_offsets() {
    let hint = ["GB"];
    let query = Query {
        country_hint: &hint,
        ..Query::default()
    };
    let native = tessera::Tessera::load(
        &[],
        tessera::Config {
            kinds: tessera::Kind::Email | tessera::Kind::Phone,
            expected_checksum: None,
        },
    )
    .unwrap();
    let js = rules_tessera();
    for input in [
        "😀 თბილისი: write to nino@kavkaz-freight.example, tel 020 7946 0958 🇬🇪",
        "📞 020 7946 0958 or 📧 nino@kavkaz-freight.example",
        "Cafe\u{301} 👩‍👩‍👧 +44 20 7946 0958",
    ] {
        let want = native.detect(input, &query).unwrap();
        assert!(
            want.iter().any(|e| e.kind == tessera::Kind::Phone),
            "{input}: no phone"
        );
        let got: Array =
            resolved(js.detect(&text(input), Some(options(r#"{"countryHint":["GB"]}"#))))
                .await
                .dyn_into()
                .unwrap();
        assert_eq!(got.length() as usize, want.len(), "{input}");
        let js_text = JsString::from(input);
        for (e, w) in got.iter().zip(&want) {
            let (start, end) = (number(&e, "start"), number(&e, "end"));
            assert_eq!(string(&e, "kind"), w.kind.as_str(), "{input}");
            assert_eq!(string(&e, "text"), w.text(input), "{input}");
            assert_eq!(start, utf16_len(&input[..w.start]), "{input}");
            assert_eq!(end, utf16_len(&input[..w.end]), "{input}");
            let sliced: String = js_text.slice(start as u32, end as u32).into();
            assert_eq!(sliced, w.text(input), "{input}");
            assert_eq!(string(&e, "source"), "rules", "{input}");
            assert_eq!(prop(&e, "normalized").as_string(), w.normalized, "{input}");
            assert_eq!(prop(&e, "region").as_string(), w.region, "{input}");
            assert!(
                !has(&e, "components"),
                "{input}: components only on addresses"
            );
        }
    }
}

#[test]
async fn rules_only_needs_no_bundle() {
    let tessera = rules_tessera();
    assert_eq!(kind_labels(&tessera), ["email", "phone"]);
    let result: Array = resolved(tessera.detect(
        &text("Write to nino@kavkaz-freight.example"),
        Some(options("{}")),
    ))
    .await
    .dyn_into()
    .unwrap();
    assert_eq!(result.length(), 1);
    let entity = result.get(0);
    assert_eq!(string(&entity, "normalized"), "nino@kavkaz-freight.example");
    assert!(!has(&entity, "components"));
    assert!(!has(&entity, "region"));
}

#[test]
fn kinds_come_back_in_taxonomy_order() {
    let tessera = load(&[], r#"{"kinds":["phone","email"]}"#).unwrap();
    assert_eq!(kind_labels(&tessera), ["email", "phone"]);
}

#[test]
async fn load_accepts_an_array_buffer() {
    let buffer = Uint8Array::from(common::BUNDLE).buffer();
    let tessera = JsTessera::load(&buffer.into(), Some(options(&address_options()))).unwrap();
    let got =
        resolved(tessera.parse_address(&text("10 Downing Street, London SW1A 2AA"), None)).await;
    let components: Array = prop(&got, "components").dyn_into().unwrap();
    assert!(components.length() > 0);
}

#[test]
fn load_rejects_what_is_not_bytes() {
    for bytes in [
        JsValue::UNDEFINED,
        text("bundle"),
        Array::of2(&1.into(), &2.into()).into(),
    ] {
        let err = JsTessera::load(&bytes, Some(options(r#"{"kinds":["email"]}"#)))
            .err()
            .unwrap();
        assert_type_error(&err);
    }
}

#[test]
async fn absent_options_mean_defaults() {
    let tessera = rules_tessera();
    for opts in [
        None,
        Some(JsValue::UNDEFINED),
        Some(JsValue::NULL),
        Some(options(r#"{"countryHint":null,"includeUncertain":null}"#)),
    ] {
        let result: Array = resolved(tessera.detect(&text("a@b.example"), opts))
            .await
            .dyn_into()
            .unwrap();
        assert_eq!(result.length(), 1);
    }
}

#[test]
async fn bad_options_are_type_errors() {
    for json in [
        r#"{"kinds":"address"}"#,
        r#"{"kinds":["Address"]}"#,
        r#"{"kinds":[1]}"#,
        r#"{"kinds":[]}"#,
        r#"{"integrity":5}"#,
        r#""address""#,
    ] {
        assert_type_error(&load(&[], json).err().unwrap());
    }
    let tessera = rules_tessera();
    for json in [
        r#"{"countryHint":"GB"}"#,
        r#"{"countryHint":["GB",7]}"#,
        r#"{"includeUncertain":"yes"}"#,
        "42",
    ] {
        assert_type_error(
            &rejected(tessera.detect(&text("a@b.example"), Some(options(json)))).await,
        );
    }
    let address = address_tessera();
    assert_type_error(
        &rejected(address.parse_address(
            &text("1 Main St"),
            Some(options(r#"{"includeUncertain":1}"#)),
        ))
        .await,
    );
}

#[test]
async fn a_sparse_hint_array_fails_without_a_copy() {
    let opts = Object::new();
    let sparse = Array::new_with_length(1_000_000_000);
    Reflect::set(&opts, &text("countryHint"), &sparse).unwrap();
    let err = rejected(rules_tessera().detect(&text("a@b.example"), Some(opts.into()))).await;
    assert_type_error(&err);
}

#[test]
async fn a_non_string_text_rejects() {
    let rules = rules_tessera();
    let address = address_tessera();
    for bad in [JsValue::from(42), JsValue::UNDEFINED, JsValue::NULL] {
        assert_type_error(&rejected(rules.detect(&bad, None)).await);
        assert_type_error(&rejected(address.parse_address(&bad, None)).await);
    }
}

#[test]
fn wrong_checksum_has_a_code() {
    let err = load(common::BUNDLE, r#"{"integrity":"sha256-0000"}"#)
        .err()
        .unwrap();
    assert_tessera_error(&err, "CHECKSUM_MISMATCH");
    assert!(!has(&err, "stage"));
}

#[test]
fn missing_network_is_bundle_invalid() {
    let err = load(&[], r#"{"kinds":["address"]}"#).err().unwrap();
    assert_tessera_error(&err, "BUNDLE_INVALID");
    // Every kind is the default, and the shipped bundle has no detector for person and org.
    let err = JsTessera::load(&Uint8Array::from(common::BUNDLE).into(), None)
        .err()
        .unwrap();
    assert_tessera_error(&err, "BUNDLE_INVALID");
}

#[test]
async fn oversized_address_rejects() {
    let input = "x ".repeat(300);
    let err = rejected(address_tessera().parse_address(&text(&input), None)).await;
    assert_tessera_error(&err, "INPUT_TOO_LARGE");
}

#[test]
async fn unserved_operation_rejects_with_a_stage() {
    let err = rejected(address_tessera().detect(&text("a@b.example"), None)).await;
    assert_tessera_error(&err, "INFERENCE");
    assert_eq!(string(&err, "stage"), "detect");
    let err = rejected(rules_tessera().parse_address(&text("1 Main St"), None)).await;
    assert_tessera_error(&err, "INFERENCE");
    assert_eq!(string(&err, "stage"), "parse");
}

#[test]
fn every_error_has_a_stable_code() {
    for (err, code) in [
        (Error::BundleInvalid, "BUNDLE_INVALID"),
        (Error::ChecksumMismatch, "CHECKSUM_MISMATCH"),
        (Error::UnsupportedVersion, "UNSUPPORTED_VERSION"),
        (Error::InputTooLarge, "INPUT_TOO_LARGE"),
        (Error::Inference { stage: "group" }, "INFERENCE"),
    ] {
        let message = err.to_string();
        let js = JsValue::from(err);
        assert_tessera_error(&js, code);
        assert_eq!(string(&js, "message"), message);
        assert_eq!(has(&js, "stage"), code == "INFERENCE");
    }
}

#[test]
async fn disposed_instance_rejects() {
    let mut tessera = address_tessera();
    tessera.dispose();
    tessera.dispose();
    assert_eq!(tessera.kinds().length(), 0);
    let err = rejected(tessera.parse_address(&text("1 Main St"), None)).await;
    assert_tessera_error(&err, "DISPOSED");
    let err = rejected(tessera.detect(&text("a@b.example"), None)).await;
    assert_tessera_error(&err, "DISPOSED");
}

/// A `blob:` URL serving `bytes`, so a fetch succeeds without the test server knowing the file.
fn blob_url(bytes: &[u8]) -> String {
    let make = Function::new_with_args("bytes", "return URL.createObjectURL(new Blob([bytes]))");
    make.call1(&JsValue::NULL, &Uint8Array::from(bytes))
        .unwrap()
        .as_string()
        .unwrap()
}

fn with_url(url: &str, integrity: &str) -> JsValue {
    options(&format!(
        r#"{{"kinds":["address"],"modelUrl":"{url}","integrity":"{integrity}"}}"#
    ))
}

/// Runs `create_instance` with the global `fetch` replaced by `stand_in`. No other test can see
/// the stand-in because wasm-bindgen-test runs tests one at a time (`CONCURRENCY = 1` in its
/// runtime), so even a stand-in that is awaited is gone before the next test starts. Callers whose
/// stand-in is never called also settle on their first poll, which would keep them safe without
/// that guarantee.
async fn create_with_fetch(stand_in: &JsValue, opts: JsValue) -> Result<JsTessera, JsValue> {
    let global = js_sys::global();
    let key = text("fetch");
    let real = Reflect::get(&global, &key).unwrap();
    Reflect::set(&global, &key, stand_in).unwrap();
    let result = create_instance(opts).await;
    Reflect::set(&global, &key, &real).unwrap();
    result
}

#[test]
async fn create_from_a_url_parses() {
    let url = blob_url(common::BUNDLE);
    let tessera = create_instance(with_url(&url, common::bundle_checksum()))
        .await
        .unwrap();
    let got =
        resolved(tessera.parse_address(&text("221B Baker Street, London NW1 6XE"), None)).await;
    assert_eq!(string(&got, "kind"), "address");
    let components: Array = prop(&got, "components").dyn_into().unwrap();
    assert!(components.length() > 0);
}

#[test]
async fn create_from_bytes_parses() {
    let array = Uint8Array::from(common::BUNDLE);
    for bytes in [JsValue::from(array.clone()), array.buffer().into()] {
        let opts = options(&address_options());
        Reflect::set(&opts, &text("modelBytes"), &bytes).unwrap();
        let tessera = create_instance(opts).await.unwrap();
        let got = resolved(tessera.parse_address(&text("10 Downing Street, London"), None)).await;
        assert_eq!(string(&got, "kind"), "address");
    }
}

#[test]
async fn model_bytes_win_over_the_url() {
    let never = Function::new_no_args("throw new Error('fetch was called')");
    let opts = with_url(
        "http://127.0.0.1:1/never.safetensors",
        common::bundle_checksum(),
    );
    Reflect::set(
        &opts,
        &text("modelBytes"),
        &Uint8Array::from(common::BUNDLE),
    )
    .unwrap();
    assert!(create_with_fetch(&never, opts).await.is_ok());
}

#[test]
async fn a_fetched_bundle_is_verified() {
    let url = blob_url(common::BUNDLE);
    let err = create_instance(with_url(&url, "sha256-0000"))
        .await
        .err()
        .unwrap();
    assert_tessera_error(&err, "CHECKSUM_MISMATCH");
}

#[test]
async fn http_404_is_model_fetch_failed_with_status() {
    // The wasm-bindgen-test server serves the test page and answers unknown paths with 404.
    let url = "/does-not-exist.safetensors";
    let err = create_instance(with_url(url, common::bundle_checksum()))
        .await
        .err()
        .unwrap();
    assert_tessera_error(&err, "MODEL_FETCH_FAILED");
    let message = string(&err, "message");
    assert!(
        message.contains(url) && message.contains("http 404"),
        "{message}"
    );
}

#[test]
async fn unreachable_url_is_model_fetch_failed() {
    let url = "http://127.0.0.1:1/none.safetensors";
    let err = create_instance(with_url(url, common::bundle_checksum()))
        .await
        .err()
        .unwrap();
    assert_tessera_error(&err, "MODEL_FETCH_FAILED");
    assert!(string(&err, "message").starts_with(url));
}

#[test]
async fn rules_only_skips_fetch() {
    let never = Function::new_no_args("throw new Error('fetch was called')");
    let opts = options(r#"{"kinds":["email"],"modelUrl":"http://127.0.0.1:1/never"}"#);
    let tessera = create_with_fetch(&never, opts).await.unwrap();
    assert_eq!(kind_labels(&tessera), ["email"]);
}

#[test]
async fn missing_model_source_is_bundle_invalid() {
    let err = create_instance(options(r#"{"kinds":["phone","address"]}"#))
        .await
        .err()
        .unwrap();
    assert_tessera_error(&err, "BUNDLE_INVALID");
    assert_eq!(
        string(&err, "message"),
        "modelUrl or modelBytes is required for kinds [address, phone]"
    );
}

#[test]
async fn no_fetch_is_unsupported_runtime() {
    let err = create_with_fetch(&JsValue::UNDEFINED, with_url("/bundle", "sha256-0000"))
        .await
        .err()
        .unwrap();
    assert_tessera_error(&err, "UNSUPPORTED_RUNTIME");
}

#[test]
async fn bad_create_options_are_type_errors() {
    for json in [
        r#"{"kinds":["address"],"modelUrl":5}"#,
        r#"{"kinds":["address"],"modelBytes":"bytes"}"#,
        r#"{"kinds":["email"],"modelBytes":[1,2]}"#,
        r#"{"kinds":[]}"#,
        r#"{"kinds":["address"],"modelBytes":null,"integrity":7}"#,
    ] {
        assert_type_error(&create_instance(options(json)).await.err().unwrap());
    }
}

#[test]
async fn awaited_fetch_failures_are_model_fetch_failed() {
    let url = "/stand-in.safetensors";
    for (body, expected) in [
        ("return Promise.reject('offline')", Some("offline")),
        (
            "return Promise.resolve({})",
            Some("fetch did not return a Response"),
        ),
        (
            "return Promise.resolve(new Response('', { status: 500 }))",
            Some("http 500"),
        ),
        // A body read twice fails, with a message that differs between engines.
        (
            "return (async () => { const r = new Response('x'); await r.arrayBuffer(); return r; })()",
            None,
        ),
    ] {
        let stand_in = Function::new_no_args(body);
        let err = create_with_fetch(&stand_in, with_url(url, common::bundle_checksum()))
            .await
            .err()
            .unwrap();
        assert_tessera_error(&err, "MODEL_FETCH_FAILED");
        let message = string(&err, "message");
        match expected {
            Some(detail) => assert_eq!(message, format!("{url}: {detail}")),
            None => assert!(
                message.starts_with(&format!("{url}: ")) && message.len() > url.len() + 2,
                "{message}"
            ),
        }
    }
}
