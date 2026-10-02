use super::loader::{mapping_crosslink, markers};
use super::manifest::{review, verify_correction};
use super::*;

fn pinned(path: &Path) -> Pin {
    Pin {
        path: path.to_string_lossy().into_owned(),
        sha256: hash(path).unwrap(),
    }
}

#[test]
fn legacy_config_omits_typed_settings_and_authored_fits_refuse() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/detector-shared.toml");
    let mut cfg = crate::config::load(&path).unwrap();
    assert!(
        serde_json::to_value(&cfg).unwrap()["detector"]
            .get("typed_synthetic")
            .is_none()
    );
    cfg.detector.as_mut().unwrap().silver.clear();
    refuse_fit(&cfg).unwrap();
    cfg.detector.as_mut().unwrap().typed_synthetic = Some(Settings {
        manifest: Pin {
            path: "/not/read/by/fit/guard.json".to_owned(),
            sha256: "a".repeat(64),
        },
        corrected_real_sources: None,
    });
    assert!(refuse_fit(&cfg).is_err());
}

#[test]
fn typed_config_refuses_missing_hash_relative_path_and_unknown_fields() {
    assert!(serde_json::from_value::<Settings>(json!({"manifest":{"path":"/x"}})).is_err());
    assert!(
        Settings {
            manifest: Pin {
                path: "relative.json".to_owned(),
                sha256: "a".repeat(64)
            },
            corrected_real_sources: None,
        }
        .validate()
        .is_err()
    );
    assert!(
        Settings {
            manifest: Pin {
                path: "/x.json".to_owned(),
                sha256: "A".repeat(64)
            },
            corrected_real_sources: None,
        }
        .validate()
        .is_err()
    );
    assert!(
        serde_json::from_value::<Settings>(
            json!({"manifest":{"path":"/x","sha256":"a".repeat(64)},"allow_fit":true})
        )
        .is_err()
    );
}

#[test]
fn reviewed_artifact_requires_actual_hash_and_exact_crosslink() {
    let dir = tempfile::tempdir().unwrap();
    let data = dir.path().join("packet.json");
    std::fs::write(&data, "{}").unwrap();
    let artifact = pinned(&data);
    let p = dir.path().join("review.json");
    std::fs::write(
        &p,
        serde_json::to_vec(&json!({"passed":true,"blockers":[],"reviewed_files":{}})).unwrap(),
    )
    .unwrap();
    assert!(review(&pinned(&p), &[&artifact]).is_err());
    std::fs::write(&p,serde_json::to_vec(&json!({"passed":true,"blockers":[],"reviewed_files":[{"path":artifact.path,"sha256":artifact.sha256}]})).unwrap()).unwrap();
    review(&pinned(&p), &[&artifact]).unwrap();
    std::fs::write(&data, "{\"changed\":true}").unwrap();
    assert!(review(&pinned(&p), &[&artifact]).is_err());
}

#[test]
fn real_silver_rejects_authored_and_split_family_markers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("annotations.jsonl");
    let fc = FeatureConfig {
        ngram_sizes: vec![2],
        hash_buckets: 64,
        hash_seed: 0,
    };
    for marker in [
        json!({"origin":"synthetic_authored"}),
        json!({"split":"train"}),
        json!({"family":"authored_address"}),
        json!({"source":"synthetic-training-candidate"}),
        json!({"source_document_key":"authored:test-example"}),
        json!({"scenario_provenance":"authored synthetic documentary scenario"}),
        json!({"narrative_family":"mail_sentence"}),
    ] {
        let mut row = json!({"text":"ordinary text","country":"US","entities":[]});
        row.as_object_mut()
            .unwrap()
            .extend(marker.as_object().unwrap().clone());
        std::fs::write(&path, serde_json::to_vec(&row).unwrap()).unwrap();
        assert!(
            crate::detector::load_silver(&[path.to_string_lossy().into_owned()], &fc, 512).is_err()
        );
        let mut cfg = crate::config::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/detector-shared.toml"),
        )
        .unwrap();
        cfg.detector.as_mut().unwrap().silver = vec![path.to_string_lossy().into_owned()];
        assert!(refuse_fit(&cfg).is_err());
    }
}

#[test]
fn shared_dedup_retains_first_seen_and_refuses_conflicting_labels() {
    let mut seen = AnnotationDedup::default();
    let empty = Vec::new();
    assert!(seen.insert("Jane Doe", &empty, "first").unwrap());
    assert!(!seen.insert("Jane Doe", &empty, "copy").unwrap());
    assert!(
        seen.insert(
            "Jane Doe",
            &[crate::detector::KindSpan {
                kind: 0,
                start: 0,
                end: 8
            }],
            "conflict"
        )
        .is_err()
    );
}

#[test]
fn split_origin_family_and_native_mapping_substitutions_refuse() {
    markers(
        Some("synthetic_authored"),
        Some("train"),
        Some("future_address_family"),
        "future_address_family",
    )
    .unwrap();
    for (origin, split, family) in [
        (Some("real_silver"), Some("train"), Some("f")),
        (Some("synthetic_authored"), Some("valid"), Some("f")),
        (Some("synthetic_authored"), Some("train"), Some("other")),
        (None, Some("train"), Some("f")),
    ] {
        assert!(markers(origin, split, family, "f").is_err());
    }
    let original = json!({"encoded_index":0,"raw_packet_rows":[0,1],"family":"f","model_labels":[],"content_tokens":3});
    let mut map = original.clone();
    map["prototype_encoded_index"] = json!(170000);
    mapping_crosslink(&map, &original).unwrap();
    for key in [
        "encoded_index",
        "raw_packet_rows",
        "family",
        "model_labels",
        "content_tokens",
    ] {
        let mut substituted = map.clone();
        substituted[key] = Value::Null;
        assert!(mapping_crosslink(&substituted, &original).is_err());
    }
    let mut extra = map.clone();
    extra["unapproved_append"] = json!(true);
    assert!(mapping_crosslink(&extra, &original).is_err());
}

#[test]
fn partial_batch_dose_uses_actual_native_denominator_lr_and_synthetic_boundary() {
    let cfg = crate::config::load(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/detector-shared.toml"),
    )
    .unwrap();
    let fc = FeatureConfig {
        ngram_sizes: vec![2],
        hash_buckets: 64,
        hash_seed: 0,
    };
    let base = crate::detector::annotated_document(
        "Lena Fox",
        "US",
        r#"[{"kind":"person","start":0,"end":8}]"#,
        &fc,
    )
    .unwrap();
    let authored = crate::detector::annotated_document(
        "Archives Unit",
        "US",
        r#"[{"kind":"org","start":0,"end":13}]"#,
        &fc,
    )
    .unwrap();
    let real = crate::detector::annotated_document("pending", "US", "[]", &fc).unwrap();
    let items = vec![base.enc, authored.enc, real.enc];
    let batches = vec![vec![0, 1, 2], vec![1]];
    let actual = preview(
        &cfg,
        &items,
        &batches,
        1,
        2,
        &[json!({"origin":"synthetic_authored"})],
        7000,
    )
    .unwrap();
    let weights = &cfg.detector.as_ref().unwrap().class_weights;
    let d0 = crate::fullmix_exposure::denominator(&items, &batches[0], weights).unwrap();
    let d1 = crate::fullmix_exposure::denominator(&items, &batches[1], weights).unwrap();
    assert_eq!(actual["planned_updates"], 2);
    assert_eq!(actual["authored_documents"][0]["presentations"], 2);
    assert_eq!(actual["aggregate_exposure"]["synthetic"]["draws"], 3);
    assert_eq!(actual["aggregate_exposure"]["real"]["draws"], 1);
    assert_eq!(
        actual["authored_documents"][0]["inverse_native_denominator_sum"],
        1.0 / d0 + 1.0 / d1
    );
    assert_eq!(
        actual["authored_documents"][0]["lr_inverse_native_denominator_sum"],
        crate::train::lr_at(0, 7000, cfg.train.learning_rate, cfg.train.warmup_steps) / d0
            + crate::train::lr_at(1, 7000, cfg.train.learning_rate, cfg.train.warmup_steps) / d1
    );
    assert_eq!(actual["optimizer_updates_executed"], 0);
}

#[test]
fn exact_correction_refuses_text_metadata_or_extra_row_edits() {
    let dir = tempfile::tempdir().unwrap();
    let old = dir.path().join("old.jsonl");
    let new = dir.path().join("new.jsonl");
    let original =
        json!({"id":"synthetic-test","country":"US","text":"BLS","entities":[],"source":"test"});
    std::fs::write(&old, serde_json::to_vec(&original).unwrap()).unwrap();
    let mut corrected = original.clone();
    corrected["entities"] = json!([{"kind":"org","start":0,"end":3}]);
    std::fs::write(&new, serde_json::to_vec(&corrected).unwrap()).unwrap();
    let change = json!({"original_path":old,"candidate_path":new,"rows":1,"physical_row_1_based":1,"id":"synthetic-test","text_sha256":crate::export::sha256_hex(b"BLS"),"action":"add","entity":{"kind":"org","start":0,"end":3}});
    verify_correction(&change).unwrap();
    corrected["source"] = json!("changed provenance");
    std::fs::write(&new, serde_json::to_vec(&corrected).unwrap()).unwrap();
    assert!(verify_correction(&change).is_err());
}
