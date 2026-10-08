//! Synthetic native and runtime parity and padding tests for the context graph.

use std::collections::BTreeMap;

use burn::backend::NdArray;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use serde_json::json;

use crate::dataset::{ParserBatch, ParserBatcher};
use crate::diagnostic_operator::ForwardOperator;

fn bundle_bytes(
    q: &[crate::quantize::QTensor],
    biases: &[crate::quantize::F32Tensor],
    metadata: BTreeMap<String, String>,
) -> Vec<u8> {
    let mut tensors = BTreeMap::new();
    for tensor in q {
        tensors.insert(
            tensor.name.clone(),
            (
                "I8",
                tensor.shape.clone(),
                tensor.data.iter().map(|&x| x as u8).collect::<Vec<_>>(),
            ),
        );
        tensors.insert(
            format!("{}.scale", tensor.name),
            (
                "F32",
                vec![tensor.scales.len()],
                tensor.scales.iter().flat_map(|x| x.to_le_bytes()).collect(),
            ),
        );
    }
    for tensor in biases {
        tensors.insert(
            tensor.name.clone(),
            (
                "F32",
                tensor.shape.clone(),
                tensor.data.iter().flat_map(|x| x.to_le_bytes()).collect(),
            ),
        );
    }
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), json!(metadata));
    let mut payload = Vec::new();
    for (name, (dtype, shape, bytes)) in tensors {
        let start = payload.len();
        payload.extend(bytes);
        header.insert(
            name,
            json!({"dtype":dtype,"shape":shape,"data_offsets":[start,payload.len()]}),
        );
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend(payload);
    bytes
}

fn max_difference(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    assert!(a.iter().chain(b).all(|x| x.is_finite()));
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0., f32::max)
}

#[test]
#[ignore = "slow synthetic native and runtime parity test"]
fn context96_native_parity_and_padding() {
    let device = Default::default();
    <NdArray as Backend>::seed(&device, 8172);
    let mut cfg = crate::config::load(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/detector-shared.toml"),
    )
    .unwrap();
    cfg.features.hash_buckets = 16;
    cfg.net.architecture = Some(tessera::internal::CONTEXT96_RMS_NAME.into());
    cfg.net.dilations = tessera::internal::CONTEXT96_RMS_DILATIONS.to_vec();
    cfg.net.dropout = 0.;
    cfg.net.ngram_from = None;
    cfg.net.finetune_ngram = false;
    cfg.validate().unwrap();
    let net = cfg.detector_net_config();
    let initialized = net.init::<NdArray>(&device);
    let parameters = crate::quantize::extract(&initialized, "detector");
    assert_eq!(parameters.len(), 21);
    let (q, biases, dequantized) = crate::quantize::quantize_all(&parameters);
    let model =
        crate::quantize::inject::<NdArray>(&net, &dequantized, "detector", &device).unwrap();
    let mut metadata = crate::export::metadata(
        &cfg,
        "synthetic parity only; no training",
        "synthetic fixture",
    );
    metadata.insert("nets".into(), "detector".into());
    let bytes = bundle_bytes(&q, &biases, metadata);
    let runtime = tessera::Tessera::load(
        &bytes,
        tessera::Config {
            kinds: tessera::Kind::Person.into(),
            expected_checksum: None,
        },
    )
    .unwrap();
    let short_text = "alpha ".repeat(193);
    let long_text = "beta ".repeat(257);
    let fc = cfg.features.to_tessera();
    let short = crate::detector::encode_document(&short_text, &[], &fc).unwrap();
    let long = crate::detector::encode_document(&long_text, &[], &fc).unwrap();
    assert_eq!(short.token_spans.len(), 193);
    assert_eq!(long.token_spans.len(), 257);
    let operator = ForwardOperator::ContextRmsV2;
    let short_native = super::logits(&model, operator, 7, &short, &device)
        .unwrap()
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let long_native = super::logits(&model, operator, 7, &long, &device)
        .unwrap()
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let short_runtime = tessera::internal::detect_trace(&runtime, &short_text).unwrap();
    let long_runtime = tessera::internal::detect_trace(&runtime, &long_text).unwrap();
    assert_eq!(short_runtime.features.len(), 193);
    assert_eq!(long_runtime.features.len(), 257);
    let short_difference = max_difference(&short_native, &short_runtime.logits);
    let long_difference = max_difference(&long_native, &long_runtime.logits);
    assert!(
        short_difference <= crate::export::GOLDEN_TOLERANCE as f32,
        "{short_difference}"
    );
    assert!(
        long_difference <= crate::export::GOLDEN_TOLERANCE as f32,
        "{long_difference}"
    );
    let batch: ParserBatch<NdArray> = ParserBatcher.batch(vec![short.clone(), long], &device);
    let output = operator
        .forward(
            &model,
            batch.ngram_ids.clone(),
            batch.script.clone(),
            batch.shape.clone(),
            batch.flags.clone(),
            batch.mask.clone(),
        )
        .unwrap();
    let [_, length, labels] = output.dims();
    assert_eq!(length, 257);
    assert_eq!(labels, 7);
    let values = output.into_data().to_vec::<f32>().unwrap();
    let padding_difference = max_difference(&short_native, &values[..193 * 7]);
    assert!(padding_difference <= 1e-5, "{padding_difference}");
    let poisoned = batch.flags.clone().mask_fill(
        batch.mask.clone().unsqueeze_dim::<3>(2).equal_elem(0.),
        f32::NAN,
    );
    let poisoned = operator
        .forward(
            &model,
            batch.ngram_ids.clone(),
            batch.script.clone(),
            batch.shape.clone(),
            poisoned,
            batch.mask.clone(),
        )
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let poisoned_difference = max_difference(&short_native, &poisoned[..193 * 7]);
    assert!(poisoned_difference <= 1e-5, "{poisoned_difference}");
    assert!(
        ForwardOperator::Standard
            .forward(
                &model,
                batch.ngram_ids.clone(),
                batch.script.clone(),
                batch.shape.clone(),
                batch.flags.clone(),
                batch.mask.clone()
            )
            .is_err()
    );
    assert!(
        ForwardOperator::ResidualRmsV1
            .forward(
                &model,
                batch.ngram_ids,
                batch.script,
                batch.shape,
                batch.flags,
                batch.mask
            )
            .is_err()
    );
}

#[test]
fn reviewed_tab_cell_goldens_match_the_library_trace_with_their_country_hint() {
    let device = Default::default();
    <NdArray as Backend>::seed(&device, 9301);
    let mut cfg = crate::config::load(
        &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/detector-shared.toml"),
    )
    .unwrap();
    cfg.features.hash_buckets = 16;
    cfg.net.architecture = Some(tessera::internal::CONTEXT96_RMS_NAME.into());
    cfg.net.dilations = tessera::internal::CONTEXT96_RMS_DILATIONS.to_vec();
    cfg.net.dropout = 0.;
    cfg.net.ngram_from = None;
    cfg.net.finetune_ngram = false;
    cfg.net.detector_features = Some(crate::config::DetectorFeatures::TabCells25);
    cfg.net.detector_postprocess = Some(crate::config::DetectorPostprocess::AddressLabeledFieldsV1);
    let net = cfg.detector_net_config();
    let f32_model = net.init::<NdArray>(&device);
    let (q, biases, dequantized) =
        crate::quantize::quantize_all(&crate::quantize::extract(&f32_model, "detector"));
    let int8_model =
        crate::quantize::inject::<NdArray>(&net, &dequantized, "detector", &device).unwrap();
    let mut metadata = crate::export::metadata(&cfg, "synthetic golden parity", "synthetic");
    metadata.insert("nets".into(), "detector".into());
    metadata.insert("model_version".into(), "0.4.0".into());
    let runtime = tessera::Tessera::load(
        &bundle_bytes(&q, &biases, metadata),
        tessera::Config {
            kinds: tessera::Kind::Person.into(),
            expected_checksum: None,
        },
    )
    .unwrap();
    let policy = crate::detector::DetectorInputPolicy::KnownUs;
    let mut unhinted_differs = false;
    for (name, text) in super::DETECTOR_GOLDEN_TEXTS {
        let case = super::detector_golden(
            "0.4.0",
            &cfg,
            Some(policy),
            (&f32_model, &int8_model),
            text,
            &device,
        )
        .unwrap();
        assert_eq!(case["input_policy"], "known_us", "{name}");
        assert_eq!(case["country_hint"], json!(["US"]), "{name}");
        assert_eq!(
            case["detector_postprocess"], "address_labeled_fields_v1",
            "{name}"
        );
        let hint: Vec<&str> = case["country_hint"]
            .as_array()
            .unwrap()
            .iter()
            .map(|h| h.as_str().unwrap())
            .collect();
        let trace = runtime.detect_trace_with_country_hint(text, &hint).unwrap();
        let tokens: Vec<(usize, usize)> =
            serde_json::from_value(case["input"]["tokens"].clone()).unwrap();
        assert_eq!(trace.token_spans, tokens, "{name}");
        for (i, (got, want)) in trace
            .features
            .iter()
            .zip(case["input"]["features"].as_array().unwrap())
            .enumerate()
        {
            assert_eq!(json!(got.ngram_ids), want["ngram_ids"], "{name} {i}");
            assert_eq!(
                json!([got.script, got.shape, got.flags]),
                json!([want["script"], want["shape"], want["flags"]]),
                "{name} {i}"
            );
        }
        assert_eq!(json!(trace.masked), case["masked"], "{name}");
        let want: Vec<f32> = serde_json::from_value::<Vec<Vec<f32>>>(case["int8_logits"].clone())
            .unwrap()
            .concat();
        let difference = max_difference(&trace.logits, &want);
        assert!(
            difference <= crate::export::GOLDEN_TOLERANCE as f32,
            "{name}: {difference}"
        );
        assert_eq!(json!(trace.decoded), case["decoded"], "{name}");
        let unhinted = tessera::internal::detect_trace(&runtime, text).unwrap();
        unhinted_differs |= json!(unhinted.masked) != case["masked"];
    }
    assert!(
        unhinted_differs,
        "some golden must depend on the declared country hint"
    );
}
