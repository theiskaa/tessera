//! Synthetic bundle controls for the complete public detector postprocessing path.

use super::{Parts, tab_cell_runtime_tests};
use crate::detector_postprocess::DetectorPostprocessContract;
use crate::model::{DETECTOR_LABELS, weights::Bundle};
use crate::{Config, Kind, Query, Source, Tessera};
use serde_json::json;

const TEXT: &str =
    "PO Box 549\nVinton IA 52349\nPhysical Address\n101 First Street\nVinton IA 52349";

fn replace(parts: &mut Parts, name: &str, bytes: &[u8]) {
    let offsets = &parts.header[name]["data_offsets"];
    let start = offsets[0].as_u64().unwrap() as usize;
    let end = offsets[1].as_u64().unwrap() as usize;
    assert_eq!(end - start, bytes.len());
    parts.data[start..end].copy_from_slice(bytes);
}

fn fixture(contract: Option<DetectorPostprocessContract>) -> Vec<u8> {
    let mut parts = Parts::read(&tab_cell_runtime_tests::fixture(87, 87, None, false));
    let mut projection = vec![0; 96 * 87];
    projection[64 + 2] = 1;
    projection[64 + 5] = 1;
    replace(&mut parts, "detector.proj.weight", &projection);
    let mut projection_bias = vec![0.0f32; 96];
    projection_bias[0] = -1.0;
    replace(
        &mut parts,
        "detector.proj.bias",
        &projection_bias
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    let mut head = vec![0; DETECTOR_LABELS * 96];
    head[5 * 96] = 2;
    replace(&mut parts, "detector.head.weight", &head);
    // ALL_UPPER and LINE_START together select only PO; IA stays an I-ADDRESS.
    let bias = [0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 10.0];
    replace(
        &mut parts,
        "detector.head.bias",
        &bias
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    if let Some(contract) = contract {
        parts.metadata().insert(
            "detector_postprocess_contract".into(),
            json!(contract.name()),
        );
    }
    parts.write()
}

#[test]
fn bundle_metadata_reaches_public_detect_and_preserves_missing_default() {
    let mut outputs = Vec::new();
    let mut payloads = Vec::new();
    for contract in [
        None,
        Some(DetectorPostprocessContract::AddressContinuationV1),
        Some(DetectorPostprocessContract::AddressLabeledFieldsV1),
    ] {
        let bytes = fixture(contract);
        payloads.push(Parts::read(&bytes).data);
        let bundle = Bundle::parse(&bytes, None).unwrap();
        assert_eq!(bundle.manifest.detector_postprocess_contract, contract);
        let runtime = Tessera::load(
            &bytes,
            Config {
                kinds: Kind::Address.into(),
                expected_checksum: None,
            },
        )
        .unwrap();
        assert_eq!(
            runtime
                .model
                .as_ref()
                .unwrap()
                .detector_postprocess_contract,
            contract
        );
        let detected = runtime
            .detect(
                TEXT,
                &Query {
                    country_hint: &["US"],
                    ..Query::default()
                },
            )
            .unwrap();
        for entity in &detected {
            assert_eq!(entity.kind, Kind::Address);
            assert_eq!(entity.source, Source::Model);
            assert!(entity.confidence.is_finite() && entity.confidence >= 0.85);
            assert!(!entity.review_recommended);
            assert_eq!(entity.text(TEXT), &TEXT[entity.start..entity.end]);
        }
        outputs.push(detected);
    }
    assert!(payloads.windows(2).all(|pair| pair[0] == pair[1]));
    assert_eq!(outputs[0], outputs[1]);
    assert_eq!(outputs[0].len(), 1);
    assert_eq!(outputs[0][0].text(TEXT), TEXT);
    assert_eq!(outputs[2].len(), 2);
    assert_eq!(outputs[2][0].text(TEXT), "PO Box 549\nVinton IA 52349");
    assert_eq!(
        outputs[2][1].text(TEXT),
        "101 First Street\nVinton IA 52349"
    );
    assert!(outputs[2][0].end < outputs[2][1].start);
    assert!(outputs[2].iter().all(|entity| {
        entity
            .components
            .iter()
            .all(|component| component.start >= entity.start && component.end <= entity.end)
    }));
}
