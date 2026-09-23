//! Fixture-driven tests for the rules layer.

mod common;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

use tessera::internal::scan_email;

#[test]
fn email_fixtures() {
    for (file, fixture) in common::rules_fixtures()
        .into_iter()
        .filter(|(n, _)| *n == "email")
    {
        for case in &fixture.cases {
            let text = &case.input;
            let got = scan_email(text);
            let got_spans: Vec<(&str, usize, usize)> = got
                .iter()
                .map(|e| (e.kind.as_str(), e.start, e.end))
                .collect();
            let want_spans: Vec<(&str, usize, usize)> = case
                .expected
                .iter()
                .map(|e| (e.kind.as_str(), e.start, e.end))
                .collect();
            assert_eq!(got_spans, want_spans, "{file}/{}", case.name);
            for (g, w) in got.iter().zip(&case.expected) {
                assert_eq!(&text[g.start..g.end], w.text, "{file}/{}", case.name);
                assert_eq!(g.normalized, w.normalized, "{file}/{}", case.name);
                assert!(
                    g.confidence >= w.min_confidence,
                    "{file}/{}: confidence {} < {}",
                    case.name,
                    g.confidence,
                    w.min_confidence
                );
            }
        }
    }
}

#[cfg(feature = "phone-metadata")]
#[test]
fn phone_fixtures() {
    for (file, fixture) in common::rules_fixtures()
        .into_iter()
        .filter(|(n, _)| n.starts_with("phone-"))
    {
        for case in &fixture.cases {
            let hints: Vec<&str> = case.country_hint.iter().map(String::as_str).collect();
            let got = tessera::internal::scan_phone(&case.input, &hints);
            let got_spans: Vec<(usize, usize)> = got.iter().map(|e| (e.start, e.end)).collect();
            let want_spans: Vec<(usize, usize)> =
                case.expected.iter().map(|e| (e.start, e.end)).collect();
            assert_eq!(got_spans, want_spans, "{file}/{}", case.name);
            for (g, w) in got.iter().zip(&case.expected) {
                assert_eq!(&case.input[g.start..g.end], w.text, "{file}/{}", case.name);
                assert_eq!(g.normalized, w.normalized, "{file}/{}", case.name);
                assert_eq!(g.region, w.region, "{file}/{}", case.name);
                assert!(
                    g.confidence >= w.min_confidence,
                    "{file}/{}: {} < {}",
                    case.name,
                    g.confidence,
                    w.min_confidence
                );
            }
        }
    }
}

use tessera::{Config, Error, Kind, Query, Tessera};

#[cfg(feature = "phone-metadata")]
#[test]
fn detect_rules_only() {
    let t = Tessera::load(
        &[],
        Config {
            kinds: Kind::Email | Kind::Phone,
            expected_checksum: None,
        },
    )
    .unwrap();
    let text = "Nino Beridze\nKavkaz Freight LLC\n+995 32 212 3456\nnino@kavkaz-freight.example\n";
    let out = t
        .detect(
            text,
            &Query {
                country_hint: &["GE"],
                include_uncertain: false,
            },
        )
        .unwrap();
    let got: Vec<(&str, &str)> = out
        .iter()
        .map(|e| (e.kind.as_str(), &text[e.start..e.end]))
        .collect();
    assert_eq!(
        got,
        vec![
            ("phone", "+995 32 212 3456"),
            ("email", "nino@kavkaz-freight.example")
        ]
    );
    assert!(out.iter().all(|e| !e.review_recommended));

    let only_email = Tessera::load(
        b"garbage",
        Config {
            kinds: Kind::Email.into(),
            expected_checksum: None,
        },
    )
    .unwrap();
    let out = only_email.detect(text, &Query::default()).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].kind, Kind::Email);
}

/// Whole documents full of numbers that are not phones: order, ticket and invoice numbers,
/// dates, prices, tracking codes, IBANs, and register numbers. Only the contacts may come out.
#[cfg(feature = "phone-metadata")]
#[test]
fn document_fixtures() {
    for (file, fixture) in common::rules_fixtures()
        .into_iter()
        .filter(|(n, _)| *n == "documents")
    {
        for case in &fixture.cases {
            let hints: Vec<&str> = case.country_hint.iter().map(String::as_str).collect();
            let found = tessera::internal::scan_rules(&case.input, &hints);
            for (g, w) in found.iter().zip(&case.expected) {
                assert_eq!(
                    g.region, w.region,
                    "{file}/{}: region of {}",
                    case.name, w.text
                );
                assert!(
                    g.confidence >= w.min_confidence,
                    "{file}/{}: confidence of {}",
                    case.name,
                    w.text
                );
            }
            let got: Vec<(&str, usize, usize, Option<&str>)> = found
                .iter()
                .map(|e| (e.kind.as_str(), e.start, e.end, e.normalized.as_deref()))
                .collect();
            let want: Vec<(&str, usize, usize, Option<&str>)> = case
                .expected
                .iter()
                .map(|e| {
                    let normalized = e
                        .normalized
                        .as_deref()
                        .or((e.kind == "email").then_some(e.text.as_str()));
                    (e.kind.as_str(), e.start, e.end, normalized)
                })
                .collect();
            assert_eq!(got, want, "{file}/{}", case.name);
        }
    }
}

#[test]
fn model_kinds_are_typed_errors() {
    assert_eq!(
        Tessera::load(
            &[],
            Config {
                kinds: Kind::all(),
                expected_checksum: None
            }
        )
        .err(),
        Some(Error::BundleInvalid)
    );
    let t = Tessera::load(
        &[],
        Config {
            kinds: Kind::Email.into(),
            expected_checksum: None,
        },
    )
    .unwrap();
    assert_eq!(
        t.parse_address("x", &Query::default()).err(),
        Some(Error::Inference { stage: "parse" })
    );
    assert_eq!(
        t.extract_contacts("x", &Query::default()).err(),
        Some(Error::Inference { stage: "group" })
    );
    assert_eq!(t.detect("", &Query::default()).unwrap(), Vec::new());
}

#[test]
fn overlap_email_wins() {
    let t = Tessera::load(
        &[],
        Config {
            kinds: Kind::Email | Kind::Phone,
            expected_checksum: None,
        },
    )
    .unwrap();
    let text = "+442079460958@x.example";
    let out = t
        .detect(
            text,
            &Query {
                country_hint: &["GB"],
                include_uncertain: false,
            },
        )
        .unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].kind, Kind::Email);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn the_fixture_list_names_every_rules_file() {
    common::assert_lists_every_fixture("rules", &common::rules_fixtures());
}
