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

/// Asserts that `js.detect` returns what `native.detect` does, with UTF-16 offsets that slice
/// the JS string to each entity's text.
async fn assert_detect_matches(
    native: &tessera::Tessera,
    js: &JsTessera,
    input: &str,
    hint: Option<&str>,
    label: &str,
) {
    let hints: Vec<&str> = hint.into_iter().collect();
    let query = Query {
        country_hint: &hints,
        ..Query::default()
    };
    let want = native.detect(input, &query).unwrap();
    let opts = options(&format!(r#"{{"countryHint":{hints:?}}}"#));
    let got: Array = resolved(js.detect(&text(input), Some(opts)))
        .await
        .dyn_into()
        .unwrap();
    assert_eq!(got.length() as usize, want.len(), "{label}");
    let js_text = JsString::from(input);
    for (e, w) in got.iter().zip(&want) {
        let (start, end) = (number(&e, "start"), number(&e, "end"));
        assert_eq!(string(&e, "kind"), w.kind.as_str(), "{label}");
        assert_eq!(string(&e, "text"), w.text(input), "{label}");
        assert_eq!(start, utf16_len(&input[..w.start]), "{label}");
        assert_eq!(end, utf16_len(&input[..w.end]), "{label}");
        let sliced: String = js_text.slice(start as u32, end as u32).into();
        assert_eq!(sliced, w.text(input), "{label}");
        assert_eq!(string(&e, "source"), w.source.as_str(), "{label}");
        assert_eq!(
            number(&e, "confidence"),
            display_confidence(w.confidence),
            "{label}"
        );
        assert_eq!(prop(&e, "normalized").as_string(), w.normalized, "{label}");
        assert_eq!(prop(&e, "region").as_string(), w.region, "{label}");
        assert_eq!(
            has(&e, "components"),
            w.kind == tessera::Kind::Address,
            "{label}: components only on addresses"
        );
    }
}

#[test]
async fn detect_matches_native_with_utf16_offsets() {
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
        let found = native
            .detect(
                input,
                &Query {
                    country_hint: &["GB"],
                    ..Query::default()
                },
            )
            .unwrap();
        assert!(
            found.iter().any(|e| e.kind == tessera::Kind::Phone),
            "{input}: no phone"
        );
        assert_detect_matches(&native, &js, input, Some("GB"), input).await;
    }
}

#[test]
async fn detector_fixtures_match_native_with_utf16_offsets() {
    let native = common::load_all();
    let js = load(
        common::BUNDLE,
        &format!(r#"{{"integrity":"{}"}}"#, common::bundle_checksum()),
    )
    .unwrap();
    for (file, fixture) in common::detector_fixtures() {
        for case in &fixture.cases {
            let label = format!("{file}/{}", case.name);
            let hint = case.country.as_deref();
            assert_detect_matches(&native, &js, &case.input, hint, &label).await;
        }
    }
}

/// One JS entity against its native twin: kind, text, and offsets converted to UTF-16.
fn assert_entity(js: &JsValue, native: &tessera::Entity, input: &str, label: &str) {
    assert_eq!(string(js, "kind"), native.kind.as_str(), "{label}");
    assert_eq!(string(js, "text"), native.text(input), "{label}");
    assert_eq!(
        number(js, "start"),
        utf16_len(&input[..native.start]),
        "{label}"
    );
    assert_eq!(
        number(js, "end"),
        utf16_len(&input[..native.end]),
        "{label}"
    );
}

#[test]
async fn extract_contacts_matches_native_with_utf16_offsets() {
    let native = common::load_all();
    let js = load(
        common::BUNDLE,
        &format!(r#"{{"integrity":"{}"}}"#, common::bundle_checksum()),
    )
    .unwrap();
    let query = Query {
        country_hint: &["GE"],
        ..Query::default()
    };
    for input in [
        "Thanks, see you on Monday.\n\nNino Beridze\nKavkaz Freight LLC\n14 Rustaveli Avenue, Tbilisi 0108, Georgia\n+995 32 212 3456\nnino@kavkaz-freight.example",
        "მადლობა, ორშაბათს შევხვდებით.\n\nნინო ბერიძე\nშპს კავკაზ ფრეითი\nრუსთაველის გამზირი 14, თბილისი 0108, საქართველო\n+995 32 212 3456\nnino@kavkaz-freight.example",
    ] {
        let want = native.extract_contacts(input, &query).unwrap();
        let got =
            resolved(js.extract_contacts(&text(input), Some(options(r#"{"countryHint":["GE"]}"#))))
                .await;
        let contacts: Array = prop(&got, "contacts").dyn_into().unwrap();
        assert_eq!(contacts.length() as usize, want.contacts.len(), "{input}");
        for (c, w) in contacts.iter().zip(&want.contacts) {
            assert_eq!(number(&c, "start"), utf16_len(&input[..w.start]), "{input}");
            assert_eq!(number(&c, "end"), utf16_len(&input[..w.end]), "{input}");
            assert_eq!(
                prop(&c, "reviewRecommended").as_bool(),
                Some(w.review_recommended)
            );
            for (key, one) in [("person", &w.person), ("org", &w.org)] {
                match one {
                    Some(e) => assert_entity(&prop(&c, key), e, input, key),
                    None => assert!(!has(&c, key), "{input}: {key} should be absent"),
                }
            }
            for (key, list) in [
                ("addresses", &w.addresses),
                ("emails", &w.emails),
                ("phones", &w.phones),
            ] {
                let arr: Array = prop(&c, key).dyn_into().unwrap();
                assert_eq!(arr.length() as usize, list.len(), "{input}: {key}");
                for (e, n) in arr.iter().zip(list) {
                    assert_entity(&e, n, input, key);
                }
            }
        }
        let unassigned: Array = prop(&got, "unassigned").dyn_into().unwrap();
        assert_eq!(
            unassigned.length() as usize,
            want.unassigned.len(),
            "{input}"
        );
        for (e, n) in unassigned.iter().zip(&want.unassigned) {
            assert_entity(&e, n, input, "unassigned");
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

/// Arrays whose reads throw: an element getter, a `Proxy` whose `get` trap throws, and a revoked
/// `Proxy`, which `Array.isArray` itself throws on.
fn hostile_arrays() -> Vec<JsValue> {
    let make = Function::new_no_args(
        "const getter = ['GB'];
         Object.defineProperty(getter, 0, { get() { throw new Error('getter'); } });
         const trap = new Proxy(['GB'], { get() { throw new Error('trap'); } });
         const { proxy, revoke } = Proxy.revocable([], {});
         revoke();
         return [getter, trap, proxy];",
    );
    let made: Array = make.call0(&JsValue::NULL).unwrap().dyn_into().unwrap();
    made.iter().collect()
}

/// A `Proxy` over a `Uint8Array` whose `getPrototypeOf` trap, which `instanceof` runs, throws.
fn hostile_bytes() -> JsValue {
    Function::new_no_args(
        "return new Proxy(new Uint8Array(4), { getPrototypeOf() { throw new Error('trap'); } });",
    )
    .call0(&JsValue::NULL)
    .unwrap()
}

/// `name(...args)` on the JS object, through the generated glue as a page would call it.
fn call_js(obj: &JsValue, name: &str, args: &[JsValue]) -> Result<JsValue, JsValue> {
    let method: Function = prop(obj, name).dyn_into().unwrap();
    method.apply(obj, &args.iter().collect::<Array>())
}

#[test]
async fn throwing_option_reads_reject_and_leave_the_instance_usable() {
    let tessera: JsValue = rules_tessera().into();
    for hints in hostile_arrays() {
        let opts = Object::new();
        Reflect::set(&opts, &text("countryHint"), &hints).unwrap();
        let promise = call_js(&tessera, "detect", &[text("a@b.example"), opts.into()])
            .expect("detect returns a promise instead of throwing");
        assert_type_error(&rejected(promise.dyn_into().unwrap()).await);
    }
    let getter =
        Function::new_no_args("return { get includeUncertain() { throw new Error('getter'); } };")
            .call0(&JsValue::NULL)
            .unwrap();
    let promise = call_js(&tessera, "detect", &[text("a@b.example"), getter]).unwrap();
    assert_type_error(&rejected(promise.dyn_into().unwrap()).await);

    let found = call_js(&tessera, "detect", &[text("a@b.example")]).unwrap();
    let found: Array = resolved(found.dyn_into().unwrap())
        .await
        .dyn_into()
        .unwrap();
    assert_eq!(found.length(), 1);
    call_js(&tessera, "dispose", &[]).expect("the instance is not left borrowed");
    let later = call_js(&tessera, "detect", &[text("a@b.example")]).unwrap();
    assert_tessera_error(&rejected(later.dyn_into().unwrap()).await, "DISPOSED");
}

#[test]
async fn throwing_create_options_are_type_errors() {
    for kinds in hostile_arrays() {
        let opts = Object::new();
        Reflect::set(&opts, &text("kinds"), &kinds).unwrap();
        assert_type_error(&create_instance(opts.clone().into()).await.err().unwrap());
        assert_type_error(
            &JsTessera::load(&Uint8Array::new_with_length(0), Some(opts.into()))
                .err()
                .unwrap(),
        );
    }
    let opts = options(r#"{"kinds":["address"]}"#);
    Reflect::set(&opts, &text("modelBytes"), &hostile_bytes()).unwrap();
    assert_type_error(&create_instance(opts).await.err().unwrap());
    assert_type_error(&JsTessera::load(&hostile_bytes(), None).err().unwrap());
}

#[test]
async fn a_lone_surrogate_survives_in_result_texts() {
    let input: JsString = js_sys::JSON::parse(r#""10 Downing\ud800 Street, London SW1A 2AA""#)
        .unwrap()
        .dyn_into()
        .unwrap();
    let entity = resolved(address_tessera().parse_address(&input, None)).await;
    let slice = |obj: &JsValue| {
        JsValue::from(input.slice(number(obj, "start") as u32, number(obj, "end") as u32))
    };
    assert_eq!(prop(&entity, "text"), slice(&entity));
    assert!(
        prop(&entity, "text")
            .as_string()
            .unwrap()
            .contains('\u{fffd}'),
        "the entity should span the surrogate"
    );
    assert_ne!(
        JsValue::from(input.slice(0, input.length())),
        JsValue::from_str(&String::from(&input)),
        "the input really holds a lone surrogate"
    );
    let components: Array = prop(&entity, "components").dyn_into().unwrap();
    assert!(components.length() > 0);
    for c in components.iter() {
        assert_eq!(prop(&c, "text"), slice(&c));
    }
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
}

#[test]
fn every_kind_is_the_default_and_the_shipped_bundle_serves_it() {
    let tessera = JsTessera::load(&Uint8Array::from(common::BUNDLE).into(), None).unwrap();
    assert_eq!(
        kind_labels(&tessera),
        ["person", "org", "address", "email", "phone"]
    );
}

#[test]
async fn oversized_address_rejects() {
    let input = "x ".repeat(300);
    let err = rejected(address_tessera().parse_address(&text(&input), None)).await;
    assert_tessera_error(&err, "INPUT_TOO_LARGE");
}

#[test]
async fn unserved_operation_rejects_with_a_stage() {
    let parser_only = load(&parser_only_bundle(), r#"{"kinds":["address"]}"#).unwrap();
    let err = rejected(parser_only.detect(&text("a@b.example"), None)).await;
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

/// A `blob:` URL, revoked when dropped; keep it alive until whatever loads it has finished.
struct BlobUrl(String);

impl BlobUrl {
    fn new(part: &JsValue, kind: &str) -> BlobUrl {
        let make = Function::new_with_args(
            "part, type",
            "return URL.createObjectURL(new Blob([part], { type }))",
        );
        let url = make.call2(&JsValue::NULL, part, &text(kind)).unwrap();
        BlobUrl(url.as_string().unwrap())
    }
}

impl std::ops::Deref for BlobUrl {
    type Target = str;
    fn deref(&self) -> &str {
        &self.0
    }
}

impl Drop for BlobUrl {
    fn drop(&mut self) {
        let revoke = Function::new_with_args("url", "URL.revokeObjectURL(url)");
        revoke.call1(&JsValue::NULL, &text(&self.0)).unwrap();
    }
}

/// A `blob:` URL serving `bytes`, so a fetch succeeds without the test server knowing the file.
fn blob_url(bytes: &[u8]) -> BlobUrl {
    BlobUrl::new(&Uint8Array::from(bytes).into(), "application/octet-stream")
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
        r#"{"kinds":["email"],"worker":"yes"}"#,
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

/// The shipped `worker.js` as a `blob:` module run against this test binary, and the URL of the
/// binary's wasm. The runner builds tests with `--target web` and serves the glue as
/// `/wasm-bindgen-test.js`, which exports the same `Tessera` class as the package's `tessera.js`,
/// so only the import is rewritten; a `blob:` module cannot resolve a relative import.
struct TestWorker {
    script: BlobUrl,
    wasm_url: String,
}

impl TestWorker {
    fn new() -> TestWorker {
        let origin = prop(&prop(&js_sys::global(), "location"), "origin")
            .as_string()
            .unwrap();
        let source = include_str!("../js/worker.js");
        let rewritten = source.replace(
            r#""./tessera.js""#,
            &format!(r#""{origin}/wasm-bindgen-test.js""#),
        );
        assert_ne!(
            rewritten, source,
            "worker.js no longer imports ./tessera.js"
        );
        TestWorker {
            script: BlobUrl::new(&text(&rewritten), "text/javascript"),
            wasm_url: format!("{origin}/wasm-bindgen-test_bg.wasm"),
        }
    }
}

/// `options` with the worker switched on, or explicitly off, and the URLs it needs. The returned
/// `TestWorker` must outlive `create_instance`.
fn in_worker(json: &str, worker: bool) -> (JsValue, TestWorker) {
    worker_on(options(json), worker)
}

fn worker_on(opts: JsValue, worker: bool) -> (JsValue, TestWorker) {
    let urls = TestWorker::new();
    Reflect::set(&opts, &text("worker"), &JsValue::from_bool(worker)).unwrap();
    Reflect::set(&opts, &text("workerUrl"), &text(&urls.script)).unwrap();
    Reflect::set(&opts, &text("wasmUrl"), &text(&urls.wasm_url)).unwrap();
    (opts, urls)
}

fn with_bundle(opts: JsValue) -> JsValue {
    Reflect::set(
        &opts,
        &text("modelBytes"),
        &Uint8Array::from(common::BUNDLE),
    )
    .unwrap();
    opts
}

fn json(value: &JsValue) -> String {
    js_sys::JSON::stringify(value).unwrap().into()
}

/// Patches `Worker.prototype` to count terminations and remember the last worker posted to; the
/// glue calls both through the prototype. Tests run one at a time, so only the installing test
/// sees the patch. A panic in wasm does not unwind, so tests restore it before asserting.
struct WorkerSpy(JsValue);

impl WorkerSpy {
    fn install() -> WorkerSpy {
        let install = Function::new_no_args(
            "const proto = Worker.prototype;
             const { terminate, postMessage } = proto;
             const seen = { terminated: 0, last: null };
             proto.terminate = function () { seen.terminated++; return terminate.call(this); };
             proto.postMessage = function (...args) { seen.last = this; return postMessage.apply(this, args); };
             return { seen, restore() { Object.assign(proto, { terminate, postMessage }); } };",
        );
        WorkerSpy(install.call0(&JsValue::NULL).unwrap())
    }

    fn terminated(&self) -> f64 {
        number(&prop(&self.0, "seen"), "terminated")
    }

    fn last(&self) -> JsValue {
        prop(&prop(&self.0, "seen"), "last")
    }

    fn restore(&self) {
        let restore: Function = prop(&self.0, "restore").dyn_into().unwrap();
        restore.call0(&self.0).unwrap();
    }
}

async fn settled(promise: Promise) -> Result<JsValue, JsValue> {
    JsFuture::from(promise).await
}

#[test]
async fn worker_and_inline_parse_identically() {
    let (opts, _inline_worker) = in_worker(&address_options(), false);
    let inline = create_instance(with_bundle(opts)).await.unwrap();
    // Fetched bytes are moved into the worker and a caller's bytes are copied; cover both.
    let bundle = blob_url(common::BUNDLE);
    let (fetched, _fetched_worker) = worker_on(with_url(&bundle, common::bundle_checksum()), true);
    let (given, _given_worker) = in_worker(&address_options(), true);
    let given = with_bundle(given);
    let caller_bytes: Uint8Array = prop(&given, "modelBytes").dyn_into().unwrap();
    let workers = [
        create_instance(fetched).await.unwrap(),
        create_instance(given).await.unwrap(),
    ];
    assert_eq!(
        caller_bytes.byte_length() as usize,
        common::BUNDLE.len(),
        "a caller's modelBytes must not be detached"
    );
    let mut inputs: Vec<String> = common::parser_fixtures()
        .into_iter()
        .flat_map(|(_, f)| f.cases.into_iter().map(|c| c.input))
        .collect();
    inputs.extend(WIDE_ADDRESSES.iter().map(|s| s.to_string()));
    for worker in &workers {
        assert_eq!(kind_labels(worker), ["address"]);
        for input in &inputs {
            for opts in [r#"{}"#, r#"{"includeUncertain":true}"#] {
                let want = resolved(inline.parse_address(&text(input), Some(options(opts)))).await;
                let got = resolved(worker.parse_address(&text(input), Some(options(opts)))).await;
                assert_eq!(json(&got), json(&want), "{input} {opts}");
            }
        }
    }
}

#[test]
async fn worker_and_inline_detect_identically() {
    let kinds = r#"{"kinds":["email","phone"]}"#;
    let (opts, _inline_worker) = in_worker(kinds, false);
    let inline = create_instance(opts).await.unwrap();
    let (opts, _worker) = in_worker(kinds, true);
    let worker = create_instance(opts).await.unwrap();
    assert_eq!(kind_labels(&worker), ["email", "phone"]);
    for (_, fixture) in common::rules_fixtures() {
        for case in &fixture.cases {
            let hint: Vec<String> = case.country_hint.iter().map(|h| format!("{h:?}")).collect();
            let opts = format!(r#"{{"countryHint":[{}]}}"#, hint.join(","));
            let want = resolved(inline.detect(&text(&case.input), Some(options(&opts)))).await;
            let got = resolved(worker.detect(&text(&case.input), Some(options(&opts)))).await;
            assert_eq!(json(&got), json(&want), "{}", case.name);
        }
    }
}

#[test]
async fn errors_cross_the_worker_boundary_with_their_code() {
    let (opts, _bad) = in_worker(r#"{"kinds":["address"],"integrity":"sha256-0000"}"#, true);
    let err = create_instance(with_bundle(opts)).await.err().unwrap();
    assert_tessera_error(&err, "CHECKSUM_MISMATCH");

    let (opts, _good) = in_worker(&address_options(), true);
    let worker = create_instance(with_bundle(opts)).await.unwrap();
    let err = rejected(worker.parse_address(&text(&"x ".repeat(300)), None)).await;
    assert_tessera_error(&err, "INPUT_TOO_LARGE");
    // Arguments are checked before posting, so a caller bug stays a TypeError.
    assert_type_error(&rejected(worker.parse_address(&JsValue::from(7), None)).await);
    let bad = options(r#"{"includeUncertain":"yes"}"#);
    assert_type_error(&rejected(worker.parse_address(&text("1 Main St"), Some(bad))).await);

    let (opts, _parser_only) = in_worker(r#"{"kinds":["address"]}"#, true);
    Reflect::set(
        &opts,
        &text("modelBytes"),
        &Uint8Array::from(parser_only_bundle().as_slice()),
    )
    .unwrap();
    let worker = create_instance(opts).await.unwrap();
    let err = rejected(worker.detect(&text("a@b.example"), None)).await;
    assert_tessera_error(&err, "INFERENCE");
    assert_eq!(string(&err, "stage"), "detect");
}

/// The shipped bundle with its manifest listing only the parser, as a Milestone 2 bundle did:
/// an address instance loads from it and serves `parseAddress` but not `detect`.
fn parser_only_bundle() -> Vec<u8> {
    let bytes = common::BUNDLE;
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&bytes[8..8 + n]).unwrap().replacen(
        r#""nets":"parser,detector""#,
        r#""nets":"parser""#,
        1,
    );
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&bytes[8 + n..]);
    out
}

#[test]
async fn a_worker_whose_module_fails_to_load_is_unsupported_runtime() {
    let (opts, _worker) = in_worker(&address_options(), true);
    Reflect::set(&opts, &text("wasmUrl"), &text("/no-such-module.wasm")).unwrap();
    let err = create_instance(with_bundle(opts)).await.err().unwrap();
    assert_tessera_error(&err, "UNSUPPORTED_RUNTIME");
}

/// What `dispose_terminates_the_worker` observes, gathered before any assertion.
struct DisposeObservations {
    before: f64,
    after_dispose: f64,
    in_flight: Result<JsValue, JsValue>,
    later: Result<JsValue, JsValue>,
    kinds_after: u32,
    after_second_dispose: f64,
    after_drop: f64,
}

async fn observe_dispose(spy: &WorkerSpy) -> Result<DisposeObservations, JsValue> {
    let (opts, _address) = in_worker(&address_options(), true);
    let mut worker = create_instance(with_bundle(opts)).await?;
    let before = spy.terminated();
    let in_flight = worker.parse_address(&text("10 Downing Street, London"), None);
    worker.dispose();
    let after_dispose = spy.terminated();
    let in_flight = settled(in_flight).await;
    let later = settled(worker.detect(&text("a@b.example"), None)).await;
    let kinds_after = worker.kinds().length();
    worker.dispose();
    let after_second_dispose = spy.terminated();
    // Dropping the instance, which JavaScript's `free()` does, terminates the worker as well.
    let (opts, _rules) = in_worker(r#"{"kinds":["email"]}"#, true);
    drop(create_instance(opts).await?);
    Ok(DisposeObservations {
        before,
        after_dispose,
        in_flight,
        later,
        kinds_after,
        after_second_dispose,
        after_drop: spy.terminated(),
    })
}

#[test]
async fn dispose_terminates_the_worker() {
    let spy = WorkerSpy::install();
    let observed = observe_dispose(&spy).await;
    spy.restore();
    let seen = observed.unwrap();
    assert_eq!(seen.before, 0.0);
    assert_eq!(seen.after_dispose, 1.0);
    assert_tessera_error(&seen.in_flight.unwrap_err(), "DISPOSED");
    assert_tessera_error(&seen.later.unwrap_err(), "DISPOSED");
    assert_eq!(seen.kinds_after, 0);
    assert_eq!(
        seen.after_second_dispose, 1.0,
        "disposing twice terminates once"
    );
    assert_eq!(seen.after_drop, 2.0);
}

/// What `a_freed_instance_answers_waiting_calls` observes, gathered before any assertion.
struct FreeObservations {
    after_free: f64,
    waiting: Result<JsValue, JsValue>,
    after_reply: f64,
}

async fn observe_free(spy: &WorkerSpy) -> Result<FreeObservations, JsValue> {
    let (opts, _worker) = in_worker(&address_options(), true);
    let worker = create_instance(with_bundle(opts)).await?;
    let waiting = worker.parse_address(&text("10 Downing Street, London"), None);
    // What `free()` or the glue's `FinalizationRegistry` does to an instance.
    drop(worker);
    let after_free = spy.terminated();
    let waiting = settled(waiting).await;
    Ok(FreeObservations {
        after_free,
        waiting,
        after_reply: spy.terminated(),
    })
}

#[test]
async fn a_freed_instance_answers_waiting_calls() {
    let spy = WorkerSpy::install();
    let observed = observe_free(&spy).await;
    spy.restore();
    let seen = observed.unwrap();
    assert_eq!(
        seen.after_free, 0.0,
        "the worker outlives a call still waiting"
    );
    let parsed = seen.waiting.expect("a waiting call is answered after free");
    let want =
        resolved(address_tessera().parse_address(&text("10 Downing Street, London"), None)).await;
    assert_eq!(json(&parsed), json(&want));
    assert_eq!(
        seen.after_reply, 1.0,
        "the worker stops after the last reply"
    );
}

/// What `worker_events_reject_waiting_calls` observes, gathered before any assertion.
struct EventObservations {
    unreadable: Result<JsValue, JsValue>,
    after_unreadable: Result<JsValue, JsValue>,
    stopped: Result<JsValue, JsValue>,
    after_stopped: Result<JsValue, JsValue>,
}

async fn observe_events(spy: &WorkerSpy) -> Result<EventObservations, JsValue> {
    let (opts, _worker) = in_worker(r#"{"kinds":["email"]}"#, true);
    let tessera = create_instance(opts).await?;
    let worker = spy.last();
    // Dispatched synchronously after the post, so each event lands before the real reply.
    let dispatch = Function::new_with_args(
        "worker, type",
        "worker.dispatchEvent(type === 'error'
           ? new ErrorEvent('error', { message: 'the worker crashed' })
           : new MessageEvent(type))",
    );
    let waiting = tessera.detect(&text("a@b.example"), None);
    dispatch.call2(&JsValue::NULL, &worker, &text("messageerror"))?;
    let unreadable = settled(waiting).await;
    let after_unreadable = settled(tessera.detect(&text("a@b.example"), None)).await;
    let waiting = tessera.detect(&text("a@b.example"), None);
    dispatch.call2(&JsValue::NULL, &worker, &text("error"))?;
    let stopped = settled(waiting).await;
    let after_stopped = settled(tessera.detect(&text("a@b.example"), None)).await;
    Ok(EventObservations {
        unreadable,
        after_unreadable,
        stopped,
        after_stopped,
    })
}

#[test]
async fn worker_events_reject_waiting_calls() {
    let spy = WorkerSpy::install();
    let observed = observe_events(&spy).await;
    spy.restore();
    let seen = observed.unwrap();
    let unreadable = seen.unreadable.unwrap_err();
    assert_tessera_error(&unreadable, "INFERENCE");
    assert_eq!(string(&unreadable, "stage"), "worker");
    let found: Array = seen.after_unreadable.unwrap().dyn_into().unwrap();
    assert_eq!(
        found.length(),
        1,
        "an unreadable reply does not stop the worker"
    );
    for err in [seen.stopped.unwrap_err(), seen.after_stopped.unwrap_err()] {
        assert_tessera_error(&err, "UNSUPPORTED_RUNTIME");
        assert_eq!(string(&err, "message"), "the worker crashed");
    }
}

#[test]
async fn worker_needs_the_entry_urls() {
    for missing in ["workerUrl", "wasmUrl"] {
        let (opts, _worker) = in_worker(&address_options(), true);
        let opts = with_bundle(opts);
        Reflect::delete_property(opts.unchecked_ref::<Object>(), &text(missing)).unwrap();
        let err = create_instance(opts).await.err().unwrap();
        assert_tessera_error(&err, "UNSUPPORTED_RUNTIME");
        assert!(string(&err, "message").starts_with(missing));
    }
}

#[test]
async fn a_worker_script_that_fails_to_load_rejects() {
    let (opts, _worker) = in_worker(&address_options(), true);
    let opts = with_bundle(opts);
    Reflect::set(&opts, &text("workerUrl"), &text("/no-such-worker.js")).unwrap();
    let err = create_instance(opts).await.err().unwrap();
    assert_tessera_error(&err, "UNSUPPORTED_RUNTIME");
}

/// Parses with `worker: true` while `globalThis.Worker` throws the `SecurityError` a cross-origin
/// script raises, and returns the constructor's call count, the instance, and its result.
async fn observe_blocked_worker() -> Result<(f64, JsTessera, JsValue), JsValue> {
    let block = Function::new_no_args(
        "const { Worker } = globalThis;
         const seen = { calls: 0 };
         globalThis.Worker = function () {
           seen.calls++;
           throw new DOMException('the script is on another origin', 'SecurityError');
         };
         return { seen, restore() { globalThis.Worker = Worker; } };",
    );
    let patch = block.call0(&JsValue::NULL)?;
    let (opts, _worker) = in_worker(&address_options(), true);
    let created = create_instance(with_bundle(opts)).await;
    let restore: Function = prop(&patch, "restore").dyn_into()?;
    restore.call0(&patch)?;
    let tessera = created?;
    let parsed = settled(tessera.parse_address(&text("10 Downing Street, London"), None)).await?;
    Ok((number(&prop(&patch, "seen"), "calls"), tessera, parsed))
}

#[test]
async fn a_worker_the_page_may_not_start_runs_inline() {
    let (calls, tessera, parsed) = observe_blocked_worker().await.unwrap();
    assert_eq!(calls, 1.0);
    assert_eq!(kind_labels(&tessera), ["address"]);
    let inline = address_tessera();
    let want = resolved(inline.parse_address(&text("10 Downing Street, London"), None)).await;
    assert_eq!(json(&parsed), json(&want));
}
