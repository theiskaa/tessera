//! Loading the weight bundle: the committed file loads, and each way a bundle can be wrong
//! gives its typed error rather than a panic.

mod common;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

use tessera::{Config, Error, Kind, Tessera};

fn load(bytes: &[u8], checksum: Option<&str>) -> Result<Tessera, Error> {
    Tessera::load(
        bytes,
        Config {
            kinds: Kind::Address | Kind::Email | Kind::Phone,
            expected_checksum: checksum,
        },
    )
}

#[test]
fn committed_bundle_loads() {
    let t = load(common::BUNDLE, Some(common::bundle_checksum())).unwrap();
    assert!(t.model_version().is_some_and(|v| v.starts_with("0.2.")));
}

#[test]
fn wrong_checksum() {
    let bad = format!("sha256-{}", "0".repeat(64));
    assert_eq!(
        load(common::BUNDLE, Some(&bad)).unwrap_err(),
        Error::ChecksumMismatch
    );
}

#[test]
fn truncated() {
    let bytes = common::BUNDLE;
    for n in [0, 7, 100, bytes.len() - 1] {
        assert_eq!(
            load(&bytes[..n], None).unwrap_err(),
            Error::BundleInvalid,
            "{n}"
        );
    }
}

#[test]
fn header_length_overflow() {
    let mut bytes = common::BUNDLE.to_vec();
    bytes[..8].copy_from_slice(&u64::MAX.to_le_bytes());
    assert_eq!(load(&bytes, None).unwrap_err(), Error::BundleInvalid);
}

/// The bundle with its header rewritten. Data offsets count from the end of the header, so
/// the header may change length.
fn with_header(edit: impl Fn(&str) -> String) -> Vec<u8> {
    let bytes = common::BUNDLE;
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = edit(std::str::from_utf8(&bytes[8..8 + n]).unwrap());
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&bytes[8 + n..]);
    out
}

fn with_header_edit(from: &str, to: &str) -> Vec<u8> {
    with_header(|h| {
        assert!(h.contains(from), "header has no {from}");
        h.replacen(from, to, 1)
    })
}

#[test]
fn newer_format_is_unsupported() {
    let bytes = with_header_edit(r#""format":"1""#, r#""format":"2""#);
    assert_eq!(load(&bytes, None).unwrap_err(), Error::UnsupportedVersion);
}

#[test]
fn a_later_model_series_is_unsupported() {
    let bytes = with_header_edit(r#""model_version":"0.2."#, r#""model_version":"0.3."#);
    assert_eq!(load(&bytes, None).unwrap_err(), Error::UnsupportedVersion);
}

/// The bundle with its model version replaced by `version`.
fn with_model_version(version: &str) -> Vec<u8> {
    with_header(|h| {
        let key = r#""model_version":""#;
        let at = h.find(key).expect("header has a model version") + key.len();
        let end = at + h[at..].find('"').expect("version is closed");
        format!("{}{version}{}", &h[..at], &h[end..])
    })
}

#[test]
fn the_series_is_read_before_the_rest_of_the_version() {
    let result = |v: &str| load(&with_model_version(v), None).map(|_| ());
    assert_eq!(result("0.2.7"), Ok(()));
    assert_eq!(result("0.3.0-rc1"), Err(Error::UnsupportedVersion));
    assert_eq!(result("0.1.9"), Err(Error::UnsupportedVersion));
    assert_eq!(result("1.0"), Err(Error::UnsupportedVersion));
    assert_eq!(result("0.2"), Err(Error::BundleInvalid));
    assert_eq!(result("0.2.0-rc1"), Err(Error::BundleInvalid));
    assert_eq!(result("garbage"), Err(Error::BundleInvalid));
    assert_eq!(result(""), Err(Error::BundleInvalid));
}

#[test]
fn offsets_that_disagree_with_the_shape_are_invalid() {
    let bytes = with_header_edit(r#""data_offsets":[0,"#, r#""data_offsets":[9,"#);
    assert_eq!(load(&bytes, None).unwrap_err(), Error::BundleInvalid);
}

#[test]
fn rules_only_needs_no_bundle() {
    let t = Tessera::load(
        &[],
        Config {
            kinds: Kind::Email | Kind::Phone,
            expected_checksum: None,
        },
    )
    .unwrap();
    assert_eq!(t.model_version(), None);
}

#[test]
fn model_kinds_need_a_bundle() {
    assert_eq!(load(&[], None).unwrap_err(), Error::BundleInvalid);
}

#[test]
fn each_way_a_header_can_be_wrong() {
    let cases: [(&str, &str, Error); 10] = [
        (r#""dtype":"I8""#, r#""dtype":"F32""#, Error::BundleInvalid),
        (
            "parser.head.bias",
            "parser.head.bias2",
            Error::BundleInvalid,
        ),
        (
            "parser.head.weight.scale",
            "parser.head.weight.scal",
            Error::BundleInvalid,
        ),
        (
            r#""nets":"parser""#,
            r#""nets":"detector""#,
            Error::BundleInvalid,
        ),
        (
            r#"\"flag_bits\":23"#,
            r#"\"flag_bits\":24"#,
            Error::UnsupportedVersion,
        ),
        ("B-house_number", "B-house_numbers", Error::BundleInvalid),
        (
            r#"{"__metadata__""#,
            r#"{"__metadata__":{},"__metadata__""#,
            Error::BundleInvalid,
        ),
        (
            r#"\"ngram_sizes\":[2,3,4]"#,
            r#"\"ngram_sizes\":[0,3,4]"#,
            Error::BundleInvalid,
        ),
        (
            r#"\"ngram_sizes\":[2,3,4]"#,
            r#"\"ngram_sizes\":[2,2,4]"#,
            Error::BundleInvalid,
        ),
        (
            r#"\"hash_buckets\":32768"#,
            r#"\"hash_buckets\":0"#,
            Error::BundleInvalid,
        ),
    ];
    for (from, to, want) in cases {
        let got = load(&with_header_edit(from, to), None).map(|_| ());
        assert_eq!(got, Err(want), "{from} -> {to}");
    }
}

/// Whenever a damaged bundle still loads, parsing must return rather than panic.
#[test]
fn a_bundle_that_loads_never_panics_on_parse() {
    let bytes = common::BUNDLE;
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&bytes[8..8 + n]).unwrap().to_string();
    let fc_start = header.find("feature_config").unwrap();
    let fc_end = fc_start + header[fc_start..].find("}\"").unwrap();
    let in_shape = |i: usize| {
        header[..i]
            .rfind("\"shape\":[")
            .is_some_and(|s| !header[s..i].contains(']'))
    };
    let positions: Vec<usize> = header
        .char_indices()
        .filter(|&(i, c)| c.is_ascii_digit() && ((fc_start..fc_end).contains(&i) || in_shape(i)))
        .map(|(i, _)| i)
        .collect();
    assert!(positions.len() > 30, "found {} digits", positions.len());
    let inputs = [
        "",
        "10 Downing Street, London SW1A 2AA",
        "რუსთაველის გამზირი 14",
    ];
    let mut loaded = 0;
    for &i in &positions {
        for digit in ["0", "1", "9", "99"] {
            if header[i..i + 1] == *digit {
                continue;
            }
            let edited = with_header(|h| format!("{}{digit}{}", &h[..i], &h[i + 1..]));
            if let Ok(t) = load(&edited, None) {
                loaded += 1;
                for text in inputs {
                    let _ = t.parse_address(text, &tessera::Query::default());
                }
            }
        }
    }
    assert!(
        loaded > 0,
        "no edited header loaded, so nothing was exercised"
    );
}

/// Scales and biases are the f32 tensors; filled with non-finite or extreme values, the
/// bundle still loads, and parsing must return valid spans rather than panic.
#[test]
fn hostile_float_weights_do_not_panic() {
    let bytes = common::BUNDLE;
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n]).unwrap();
    let ranges: Vec<(usize, usize)> = header
        .as_object()
        .unwrap()
        .values()
        .filter(|v| v["dtype"] == "F32")
        .map(|v| {
            let o = &v["data_offsets"];
            (
                o[0].as_u64().unwrap() as usize,
                o[1].as_u64().unwrap() as usize,
            )
        })
        .collect();
    assert!(ranges.len() > 10);
    let text = "Flat 4, 221B Baker Street, London NW1 6XE";
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
        let mut hostile = bytes.to_vec();
        for &(begin, end) in &ranges {
            for chunk in hostile[8 + n + begin..8 + n + end].chunks_exact_mut(4) {
                chunk.copy_from_slice(&value.to_le_bytes());
            }
        }
        let t = load(&hostile, None).unwrap();
        let e = t.parse_address(text, &tessera::Query::default()).unwrap();
        for c in &e.components {
            assert!(text.get(c.start..c.end).is_some(), "{value}: {c:?}");
        }
    }
}

#[test]
fn kinds_without_their_network_are_invalid() {
    for kinds in [Kind::Person.into(), Kind::Org | Kind::Address, Kind::all()] {
        let got = Tessera::load(
            common::BUNDLE,
            Config {
                kinds,
                expected_checksum: None,
            },
        );
        assert_eq!(got.map(|_| ()), Err(Error::BundleInvalid), "{kinds:?}");
    }
}

#[test]
fn an_instance_without_address_does_not_parse() {
    let t = Tessera::load(
        common::BUNDLE,
        Config {
            kinds: Kind::Email | Kind::Phone,
            expected_checksum: None,
        },
    )
    .unwrap();
    assert!(matches!(
        t.parse_address("10 Downing Street", &tessera::Query::default()),
        Err(Error::Inference { .. })
    ));
}

#[test]
fn tessera_is_shared_across_threads() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Tessera>();
}

/// Trailing spaces are valid header padding, so the real header grown to the cap must load
/// and one byte more must not.
#[test]
fn the_header_cap_is_exactly_64_kib() {
    let pad_to = |len: usize| {
        with_header(|h| {
            let mut h = h.trim_end().to_string();
            h.extend(std::iter::repeat_n(' ', len - h.len()));
            h
        })
    };
    assert!(load(&pad_to(64 * 1024), None).is_ok());
    assert_eq!(
        load(&pad_to(64 * 1024 + 1), None).map(|_| ()),
        Err(Error::BundleInvalid)
    );
}

#[test]
fn a_newer_bundle_is_unsupported_even_with_other_changes() {
    let bytes = with_header(|h| {
        h.replacen(r#""format":"1""#, r#""format":"2""#, 1)
            .replacen(r#""report_url":"""#, r#""report":"""#, 1)
    });
    assert_eq!(
        load(&bytes, None).map(|_| ()),
        Err(Error::UnsupportedVersion)
    );
}

#[test]
fn a_model_instance_does_not_detect_until_the_detector_ships() {
    let tessera = common::load_tessera();
    assert!(matches!(
        tessera.detect("mail me at a@example.com", &tessera::Query::default()),
        Err(Error::Inference { stage: "detect" })
    ));
}
