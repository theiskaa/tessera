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
    assert_eq!(t.model_version(), Some("0.4.0"));
    let metadata = header(common::BUNDLE)["__metadata__"].clone();
    assert_eq!(metadata["format"], "2");
    assert_eq!(metadata["status"], "experimental");
    assert_eq!(metadata["model_scope"], "US");
    assert_eq!(metadata["detector_training_updates"], "6160");
    assert_eq!(metadata["detector_training_complete"], "true");
    assert_eq!(metadata["learning_gate_passed"], "true");
    assert_eq!(metadata["detector_input_policy"], "known_us");
    assert_eq!(metadata["fresh_unseen_evaluation_pending"], "true");
    assert_eq!(metadata["general_accuracy_claim"], "false");
    assert_eq!(metadata["tokenizer_contract"], "tessera-tokenize-legacy-v1");
    assert_eq!(
        metadata["decoder_contract"],
        "tessera-bio-context96-address-continuation-v1"
    );
    assert_eq!(
        metadata["detector_architecture"],
        tessera::internal::CONTEXT96_RMS_CONTRACT
    );
    assert_eq!(
        metadata["detector_feature_config"],
        r#"{"contract":"tessera-detector-tab-cells25-v1","flag_bits":25}"#
    );
    assert_eq!(
        metadata["detector_postprocess_contract"],
        "address_labeled_fields_v1"
    );
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
    let bytes = with_header_edit(r#""format":"2""#, r#""format":"3""#);
    assert_eq!(load(&bytes, None).unwrap_err(), Error::UnsupportedVersion);
}

#[test]
fn a_later_model_series_is_unsupported() {
    let bytes = with_header_edit(r#""model_version":"0.4."#, r#""model_version":"0.5."#);
    assert_eq!(load(&bytes, None).unwrap_err(), Error::UnsupportedVersion);
}

fn header(bytes: &[u8]) -> serde_json::Value {
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    serde_json::from_slice(&bytes[8..8 + n]).unwrap()
}

fn with_metadata_in(
    bytes: &[u8],
    edit: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>),
) -> Vec<u8> {
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let mut header = header(bytes);
    edit(header["__metadata__"].as_object_mut().unwrap());
    let header = serde_json::to_vec(&header).unwrap();
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&header);
    out.extend_from_slice(&bytes[8 + n..]);
    out
}

fn with_metadata(edit: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>)) -> Vec<u8> {
    with_metadata_in(common::BUNDLE, edit)
}

/// Reuse tensor values in a synthetic six-block format-1 fixture, with compacted offsets. The
/// fixture keeps the legacy 23 detector flags: the tab-cell flags are the projection's last two
/// input columns, so each row drops them.
fn legacy_bundle() -> Vec<u8> {
    let n = u64::from_le_bytes(common::BUNDLE[..8].try_into().unwrap()) as usize;
    let mut header = header(common::BUNDLE);
    let entries = header.as_object_mut().unwrap();
    entries.retain(|name, _| !name.starts_with("detector.block6."));
    let metadata = entries["__metadata__"].as_object_mut().unwrap();
    metadata.insert("format".into(), "1".into());
    metadata.insert("model_version".into(), "0.2.0".into());
    for key in [
        "detector_architecture",
        "tokenizer_contract",
        "decoder_contract",
        "detector_feature_config",
        "detector_postprocess_contract",
        "detector_input_policy",
    ] {
        metadata.remove(key);
    }
    let proj = &mut entries["detector.proj.weight"]["shape"];
    let (rows, columns) = (
        proj[0].as_u64().unwrap() as usize,
        proj[1].as_u64().unwrap() as usize,
    );
    let legacy_columns = columns - 2;
    *proj = serde_json::json!([rows, legacy_columns]);
    let mut tensors: Vec<_> = entries
        .iter_mut()
        .filter(|(name, _)| *name != "__metadata__")
        .collect();
    tensors.sort_by_key(|(_, entry)| entry["data_offsets"][0].as_u64().unwrap());
    let mut data = Vec::new();
    for (name, entry) in tensors {
        let start = entry["data_offsets"][0].as_u64().unwrap() as usize;
        let end = entry["data_offsets"][1].as_u64().unwrap() as usize;
        let offset = data.len();
        let values = &common::BUNDLE[8 + n + start..8 + n + end];
        if name == "detector.proj.weight" {
            assert_eq!(values.len(), rows * columns);
            for row in values.chunks(columns) {
                data.extend_from_slice(&row[..legacy_columns]);
            }
        } else {
            data.extend_from_slice(values);
        }
        entry["data_offsets"] = serde_json::json!([offset, data.len()]);
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&header);
    out.extend_from_slice(&data);
    out
}

#[test]
fn explicit_legacy_contracts_load_with_the_next_patch_version() {
    let original = legacy_bundle();
    let bytes = with_metadata_in(&original, |m| {
        m.insert("model_version".into(), "0.2.1".into());
        m.insert(
            "tokenizer_contract".into(),
            "tessera-tokenize-legacy-v1".into(),
        );
        m.insert("decoder_contract".into(), "tessera-bio-legacy-v1".into());
    });
    let t = load(&bytes, None).unwrap();
    assert_eq!(t.model_version(), Some("0.2.1"));
    let legacy = load(&original, None).unwrap();
    let text = "14 Wharf Road, Leeds LS1 4AP";
    assert_eq!(
        t.parse_address(text, &tessera::Query::default()).unwrap(),
        legacy
            .parse_address(text, &tessera::Query::default())
            .unwrap()
    );
}

#[test]
fn incompatible_or_partial_contracts_fail_before_network_loading() {
    let original = legacy_bundle();
    let tagged = |tokenizer: Option<&str>, decoder: Option<&str>| {
        with_metadata_in(&original, |m| {
            m.insert("model_version".into(), "0.2.1".into());
            if let Some(id) = tokenizer {
                m.insert("tokenizer_contract".into(), id.into());
            }
            if let Some(id) = decoder {
                m.insert("decoder_contract".into(), id.into());
            }
        })
    };
    for bytes in [
        tagged(Some("future-tokenizer"), Some("tessera-bio-legacy-v1")),
        tagged(Some("tessera-tokenize-legacy-v1"), Some("future-decoder")),
    ] {
        assert_eq!(load(&bytes, None).unwrap_err(), Error::UnsupportedVersion);
    }
    for bytes in [
        tagged(Some("tessera-tokenize-legacy-v1"), None),
        tagged(None, Some("tessera-bio-legacy-v1")),
    ] {
        assert_eq!(load(&bytes, None).unwrap_err(), Error::BundleInvalid);
    }
    assert_eq!(
        load(&tagged(None, None), None).unwrap_err(),
        Error::UnsupportedVersion
    );
    let malformed = with_metadata_in(&original, |m| {
        m.insert("model_version".into(), "0.2.1".into());
        m.insert("tokenizer_contract".into(), serde_json::Value::Null);
        m.insert("decoder_contract".into(), "tessera-bio-legacy-v1".into());
    });
    assert_eq!(load(&malformed, None).unwrap_err(), Error::BundleInvalid);
    let legacy_with_wrong_id = with_metadata_in(&original, |m| {
        m.insert("tokenizer_contract".into(), "future-tokenizer".into());
        m.insert("decoder_contract".into(), "tessera-bio-legacy-v1".into());
    });
    assert_eq!(
        load(&legacy_with_wrong_id, None).unwrap_err(),
        Error::UnsupportedVersion
    );
}

#[test]
fn context_bundle_requires_its_graph_and_input_contracts() {
    for (key, value, error) in [
        (
            "decoder_contract",
            serde_json::json!("tessera-bio-legacy-v1"),
            Error::UnsupportedVersion,
        ),
        (
            "tokenizer_contract",
            serde_json::json!("future-tokenizer"),
            Error::UnsupportedVersion,
        ),
        (
            "detector_architecture",
            serde_json::json!("future-graph"),
            Error::UnsupportedVersion,
        ),
        (
            "detector_architecture",
            serde_json::Value::Null,
            Error::BundleInvalid,
        ),
        (
            "decoder_contract",
            serde_json::Value::Null,
            Error::BundleInvalid,
        ),
    ] {
        let bytes = with_metadata(|m| {
            m.insert(key.into(), value);
        });
        assert_eq!(load(&bytes, None).unwrap_err(), error, "{key}");
    }
    for key in [
        "tokenizer_contract",
        "decoder_contract",
        "detector_architecture",
    ] {
        let bytes = with_metadata(|m| {
            m.remove(key);
        });
        assert_eq!(
            load(&bytes, None).unwrap_err(),
            Error::BundleInvalid,
            "{key}"
        );
    }
    let missing_contracts = with_metadata(|m| {
        m.remove("tokenizer_contract");
        m.remove("decoder_contract");
    });
    assert_eq!(
        load(&missing_contracts, None).unwrap_err(),
        Error::UnsupportedVersion
    );
    let legacy_with_context_graph = with_metadata_in(&legacy_bundle(), |m| {
        m.insert(
            "detector_architecture".into(),
            tessera::internal::CONTEXT96_RMS_CONTRACT.into(),
        );
    });
    assert_eq!(
        load(&legacy_with_context_graph, None).unwrap_err(),
        Error::BundleInvalid
    );
}

#[test]
fn the_series_is_read_before_the_rest_of_the_version() {
    let legacy = legacy_bundle();
    let result = |bytes: &[u8], v: &str| {
        load(
            &with_metadata_in(bytes, |m| {
                m.insert("model_version".into(), v.into());
            }),
            None,
        )
        .map(|_| ())
    };
    assert_eq!(result(&legacy, "0.2.0"), Ok(()));
    assert_eq!(result(&legacy, "0.2.7"), Err(Error::UnsupportedVersion));
    assert_eq!(result(&legacy, "0.3.0-rc1"), Err(Error::UnsupportedVersion));
    assert_eq!(result(common::BUNDLE, "0.3.0"), Ok(()));
    assert_eq!(result(common::BUNDLE, "0.3.7"), Ok(()));
    assert_eq!(result(common::BUNDLE, "0.4.0"), Ok(()));
    assert_eq!(result(common::BUNDLE, "0.4.7"), Ok(()));
    assert_eq!(
        result(common::BUNDLE, "0.4.0-rc1"),
        Err(Error::BundleInvalid)
    );
    assert_eq!(
        result(common::BUNDLE, "0.5.0-rc1"),
        Err(Error::UnsupportedVersion)
    );
    for (v, error) in [
        ("0.1.9", Error::UnsupportedVersion),
        ("1.0", Error::UnsupportedVersion),
        ("0.3", Error::BundleInvalid),
        ("0.4", Error::BundleInvalid),
        ("0.3.0-rc1", Error::BundleInvalid),
        ("garbage", Error::BundleInvalid),
        ("", Error::BundleInvalid),
    ] {
        assert_eq!(result(common::BUNDLE, v), Err(error), "{v}");
    }
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
    let cases: [(&str, &str, Error); 12] = [
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
            r#""nets":"parser,detector""#,
            r#""nets":"detector""#,
            Error::BundleInvalid,
        ),
        (
            "detector.head.bias",
            "detector.head.bias2",
            Error::BundleInvalid,
        ),
        ("B-PERSON", "B-PERSONS", Error::BundleInvalid),
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
    let fc_start = header.find("\"feature_config\"").unwrap();
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
            for chunk in hostile[8 + n + begin..8 + n + end].as_chunks_mut::<4>().0 {
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
    let parser_only = with_header_edit(r#""nets":"parser,detector""#, r#""nets":"parser""#);
    for kinds in [Kind::Person.into(), Kind::Org | Kind::Address, Kind::all()] {
        let load = |bytes: &[u8]| {
            Tessera::load(
                bytes,
                Config {
                    kinds,
                    expected_checksum: None,
                },
            )
            .map(|_| ())
        };
        assert_eq!(load(common::BUNDLE), Ok(()), "{kinds:?}");
        assert_eq!(load(&parser_only), Err(Error::BundleInvalid), "{kinds:?}");
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
        h.replacen(r#""format":"2""#, r#""format":"3""#, 1)
            .replacen(r#""report_url":"""#, r#""report":"""#, 1)
    });
    assert_eq!(
        load(&bytes, None).map(|_| ()),
        Err(Error::UnsupportedVersion)
    );
}

#[test]
fn a_model_instance_detects_with_the_shipped_detector() {
    let tessera = common::load_tessera();
    let found = tessera
        .detect("mail me at a@example.com", &tessera::Query::default())
        .unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].kind, Kind::Email);
}
