//! Synthetic quantized runtime controls; no authored gold or private checkpoint dependencies.

use super::Parts;
use crate::detector_features::{DetectorFeatureContract, featurize_detector};
use crate::features::{FeatureConfig, TokenFeatures, flag, is_content};
use crate::model::{self, Tagger, bio, context, kernels, weights::Bundle};
use crate::token::tokenize;
use crate::{Config, Error, Kind, Tessera};
use serde_json::{Map, Value, json};

const BUCKETS: usize = 8;
const PREFIX: usize = 48 + 8 + 8;
const LEGACY_INPUT: usize = PREFIX + 23;
const TAB_INPUT: usize = PREFIX + 25;
const LEFT: &str = "Name\tPosition\nMary Vice\tPresident";
const RIGHT: &str = "Name\tPosition\nMary\tVice President";

fn q(parts: &mut Parts, name: &str, shape: &[usize], channels: usize, data: &[u8]) {
    assert_eq!(shape.iter().product::<usize>(), data.len());
    parts.add(name, "I8", shape, data);
    let scales: Vec<_> = std::iter::repeat_n(1.0f32, channels)
        .flat_map(f32::to_le_bytes)
        .collect();
    parts.add(&format!("{name}.scale"), "F32", &[channels], &scales);
}

fn floats(parts: &mut Parts, name: &str, values: &[f32]) {
    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    parts.add(name, "F32", &[values.len()], &bytes);
}

fn net(
    parts: &mut Parts,
    name: &str,
    hidden: usize,
    input: usize,
    labels: usize,
    blocks: usize,
    tab_weights: bool,
) {
    for (suffix, rows, width) in [
        ("ngram", BUCKETS + 1, 48),
        ("script", model::SCRIPT_ROWS, 8),
        ("shape", model::SHAPE_ROWS, 8),
    ] {
        q(
            parts,
            &format!("{name}.embed.{suffix}"),
            &[rows, width],
            width,
            &vec![0; rows * width],
        );
    }
    let mut projection = vec![0; hidden * input];
    if name == "detector" {
        projection[2 * input + PREFIX + 3] = 1;
        if tab_weights {
            assert_eq!(input, TAB_INPUT);
            projection[PREFIX + 23] = 1;
            projection[input + PREFIX + 24] = 1;
        }
    }
    q(
        parts,
        &format!("{name}.proj.weight"),
        &[hidden, input],
        hidden,
        &projection,
    );
    floats(parts, &format!("{name}.proj.bias"), &vec![0.0; hidden]);
    let mut head = vec![0; labels * hidden];
    if name == "detector" {
        head[hidden] = 2;
        head[3 * hidden + 1] = 2;
        head[5 * hidden + 2] = 2;
    }
    q(
        parts,
        &format!("{name}.head.weight"),
        &[labels, hidden],
        labels,
        &head,
    );
    let mut bias = vec![0.0; labels];
    bias[0] = 1.0;
    floats(parts, &format!("{name}.head.bias"), &bias);
    for i in 0..blocks {
        q(
            parts,
            &format!("{name}.block{i}.conv.weight"),
            &[hidden, hidden, 3],
            hidden,
            &vec![0; hidden * hidden * 3],
        );
        floats(
            parts,
            &format!("{name}.block{i}.conv.bias"),
            &vec![0.0; hidden],
        );
    }
}

pub(super) fn fixture(
    detector_input: usize,
    parser_input: usize,
    metadata: Option<Value>,
    tab_weights: bool,
) -> Vec<u8> {
    let mut parts = Parts {
        header: Map::new(),
        data: Vec::new(),
    };
    parts.header.insert("__metadata__".into(), json!({
        "format":"2", "model_version":"0.3.0",
        "tokenizer_contract":model::TOKENIZER_CONTRACT,
        "decoder_contract":context::CONTEXT96_DECODER_CONTRACT,
        "detector_architecture":context::CONTEXT96_RMS_CONTRACT,
        "nets":"parser,detector", "detector_labels":serde_json::to_string(&bio::detector_label_strings()).unwrap(),
        "parser_labels":serde_json::to_string(&bio::parser_label_strings()).unwrap(),
        "feature_config":json!({"ngram_sizes":[2,3,4],"hash_buckets":BUCKETS,"hash_seed":0,
            "max_ngrams_per_token":64,"flag_bits":23,"script_rows":model::SCRIPT_ROWS,"shape_rows":model::SHAPE_ROWS}).to_string(),
        "phone_metadata_version":"synthetic-runtime-control", "supported_regions":"[]", "experimental_regions":"[]",
        "training_snapshot":"synthetic-tab-width-control", "report_url":""
    }));
    if let Some(metadata) = metadata {
        parts.metadata().insert(
            "detector_feature_config".into(),
            json!(metadata.to_string()),
        );
    }
    net(
        &mut parts,
        "parser",
        2,
        parser_input,
        model::PARSER_LABELS,
        4,
        false,
    );
    net(
        &mut parts,
        "detector",
        96,
        detector_input,
        model::DETECTOR_LABELS,
        7,
        tab_weights,
    );
    parts.write()
}

fn tab_metadata() -> Value {
    json!({"contract":DetectorFeatureContract::TabCells25.name(),"flag_bits":25})
}

fn features(text: &str, contract: DetectorFeatureContract) -> Vec<TokenFeatures> {
    let tokens = tokenize(text);
    let fc = FeatureConfig {
        ngram_sizes: vec![2, 3, 4],
        hash_buckets: BUCKETS as u32,
        hash_seed: 0,
    };
    featurize_detector(text, &tokens, &[], None, &fc, None, contract)
        .into_iter()
        .zip(tokens)
        .filter(|(_, t)| is_content(t))
        .map(|(f, _)| f)
        .collect()
}

fn decoded(text: &str, logits: &[f32]) -> Vec<(Kind, usize, usize, u32)> {
    let tokens = tokenize(text);
    let retained: Vec<_> = tokens
        .iter()
        .enumerate()
        .filter(|(_, t)| is_content(t))
        .map(|(i, _)| i)
        .collect();
    let bounds: Vec<_> = retained
        .iter()
        .map(|&i| (tokens[i].start, tokens[i].end))
        .collect();
    let breaks = crate::chunk::paragraph_breaks(&tokens, &retained);
    let mut probabilities = logits.to_vec();
    kernels::softmax_rows(&mut probabilities, model::DETECTOR_LABELS);
    for row in probabilities.chunks(model::DETECTOR_LABELS) {
        assert!(row.iter().all(|p| p.is_finite() && (0.0..=1.0).contains(p)));
        assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }
    bio::decode_detector_with_text(
        text,
        &bounds,
        &probabilities,
        &vec![false; bounds.len()],
        &breaks,
    )
    .into_iter()
    .map(|s| {
        let start = bounds[s.first].0;
        let raw_end = bounds[s.last].1;
        let end = if s.kind == Kind::Address {
            crate::internal::normalized_us_address_end(text, start, raw_end)
        } else {
            raw_end
        };
        (s.kind, start, end, s.confidence.to_bits())
    })
    .collect()
}

#[test]
fn zero_extension_pairs_real_runtime_forward_and_keeps_parser_at87() {
    let legacy_bytes = fixture(LEGACY_INPUT, LEGACY_INPUT, None, false);
    let tab_bytes = fixture(TAB_INPUT, LEGACY_INPUT, Some(tab_metadata()), false);
    let legacy = Bundle::parse(&legacy_bytes, None).unwrap();
    let tab = Bundle::parse(&tab_bytes, None).unwrap();
    assert_eq!(
        legacy.manifest.detector_feature_contract,
        DetectorFeatureContract::Legacy23
    );
    assert_eq!(
        tab.manifest.detector_feature_contract,
        DetectorFeatureContract::TabCells25
    );
    assert_eq!(tab.manifest.flag_bits, 23);
    assert!(
        tab.take_i8("detector.proj.weight", &[96, TAB_INPUT], 0)
            .is_ok()
    );
    assert!(
        tab.take_i8("parser.proj.weight", &[2, LEGACY_INPUT], 0)
            .is_ok()
    );
    let old = Tagger::detector(&legacy, None).unwrap();
    let new = Tagger::detector(&tab, None).unwrap();
    let old_parser = Tagger::parser(&legacy).unwrap();
    let new_parser = Tagger::parser(&tab).unwrap();
    assert!(old.uses_address_continuation() && new.uses_address_continuation());
    assert_eq!(old.context_margin(), new.context_margin());
    for text in [LEFT, RIGHT] {
        let original = features(text, DetectorFeatureContract::Legacy23);
        let extended = features(text, DetectorFeatureContract::TabCells25);
        let a = old.forward(&original).unwrap();
        let b = new.forward(&extended).unwrap();
        assert_eq!(
            a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(decoded(text, &a), decoded(text, &b));
        assert_eq!(
            old_parser.forward(&original).unwrap(),
            new_parser.forward(&original).unwrap()
        );
        assert_eq!(new_parser.forward(&extended), Err(Error::BundleInvalid));
        assert_eq!(old.forward(&extended), Err(Error::BundleInvalid));
    }
}

#[test]
fn tab_metadata_drives_public_input_selection_and_both_new_columns_affect_forward() {
    let bytes = fixture(TAB_INPUT, LEGACY_INPUT, Some(tab_metadata()), true);
    let bundle = Bundle::parse(&bytes, None).unwrap();
    let detector = Tagger::detector(&bundle, None).unwrap();
    let runtime = Tessera::load(
        &bytes,
        Config {
            kinds: Kind::Person | Kind::Org,
            expected_checksum: None,
        },
    )
    .unwrap();
    assert_eq!(
        features(LEFT, DetectorFeatureContract::Legacy23),
        features(RIGHT, DetectorFeatureContract::Legacy23)
    );
    let left = features(LEFT, DetectorFeatureContract::TabCells25);
    let right = features(RIGHT, DetectorFeatureContract::TabCells25);
    assert_ne!(left, right);
    assert_ne!(
        detector.forward(&left).unwrap(),
        detector.forward(&right).unwrap()
    );
    for bit in [flag::AFTER_TAB, flag::BEFORE_TAB] {
        assert!(left.iter().any(|f| f.flags & bit != 0));
        let mut suppressed = left.clone();
        for f in &mut suppressed {
            f.flags &= !bit;
        }
        assert_ne!(
            detector.forward(&left).unwrap(),
            detector.forward(&suppressed).unwrap()
        );
    }
    let trace = runtime.detect_trace(LEFT).unwrap();
    assert_eq!(trace.features, left);
    assert_eq!(trace.logits, detector.forward(&left).unwrap());
}

#[test]
fn optional_metadata_and_tensor_widths_never_silently_fallback() {
    let config = || Config {
        kinds: Kind::Person | Kind::Org | Kind::Address,
        expected_checksum: None,
    };
    let missing = fixture(TAB_INPUT, LEGACY_INPUT, None, false);
    assert_eq!(
        Bundle::parse(&missing, None)
            .unwrap()
            .manifest
            .detector_feature_contract,
        DetectorFeatureContract::Legacy23
    );
    assert_eq!(
        Tessera::load(&missing, config()).unwrap_err(),
        Error::BundleInvalid
    );
    for (detector_width, parser_width, metadata) in [
        (LEGACY_INPUT, LEGACY_INPUT, tab_metadata()),
        (TAB_INPUT, TAB_INPUT, tab_metadata()),
        (
            TAB_INPUT,
            LEGACY_INPUT,
            json!({"contract":"unknown","flag_bits":25}),
        ),
        (
            TAB_INPUT,
            LEGACY_INPUT,
            json!({"contract":DetectorFeatureContract::TabCells25.name(),"flag_bits":23}),
        ),
        (TAB_INPUT, LEGACY_INPUT, json!(false)),
    ] {
        let malformed = fixture(detector_width, parser_width, Some(metadata), false);
        assert!(Tessera::load(&malformed, config()).is_err());
    }
}

#[test]
fn committed_bundle_omits_optional_contract_and_still_defaults_legacy23() {
    let bytes = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../models/tessera-v1.safetensors"
    ))
    .unwrap();
    let parts = Parts::read(&bytes);
    assert!(
        !parts.header["__metadata__"]
            .as_object()
            .unwrap()
            .contains_key("detector_feature_config")
    );
    let bundle = Bundle::parse(&bytes, None).unwrap();
    assert_eq!(
        bundle.manifest.detector_feature_contract,
        DetectorFeatureContract::Legacy23
    );
    assert_eq!(bundle.manifest.flag_bits, 23);
}
