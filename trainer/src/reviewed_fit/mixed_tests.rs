use super::draws::complete_passes;
use super::*;
use std::collections::BTreeMap;

#[test]
fn origin_tags_cannot_substitute_native_and_authored() {
    assert!(
        serde_json::from_value::<Index>(json!({"origin":"native","packet":0,"row":0})).is_err()
    );
    assert!(
        serde_json::from_value::<Index>(json!({"origin":"authored_train_only","row":0})).is_err()
    );
    assert!(serde_json::from_str::<Postprocess>("\"unregistered\"").is_err());
}

#[test]
fn authored_owner_must_be_train_only_and_cannot_equal_dev() {
    let native = vec!["native-parent".into()];
    let dev = vec!["dev-card".into()];
    let identity = json!({"origin_type":"authored_train_only","base_variant_group":{"native_name":"native-parent"}});
    validate_authored_owner(
        "authored",
        "complete authored text",
        &identity,
        &native,
        &dev,
        &["complete DEV"],
    )
    .unwrap();
    assert!(validate_authored_owner("dev-card", "other", &identity, &native, &dev, &[]).is_err());
    assert!(
        validate_authored_owner(
            "authored",
            "complete DEV",
            &identity,
            &native,
            &dev,
            &["complete DEV"]
        )
        .is_err()
    );
    let mut unsupported = identity.clone();
    unsupported["origin_type"] = json!("native");
    assert!(
        validate_authored_owner("authored", "other", &unsupported, &native, &dev, &[]).is_err()
    );
    unsupported = identity;
    unsupported["base_variant_group"]["native_name"] = json!("dev-card");
    assert!(
        validate_authored_owner("authored", "other", &unsupported, &native, &dev, &[]).is_err()
    );
}

#[test]
fn complete_passes_expose_all_owners_exactly_and_reject_partial_or_zero_budgets() {
    let groups = BTreeMap::from([
        ("native".into(), vec![0, 1, 2]),
        ("authored".into(), vec![3, 4]),
    ]);
    let budget = || sampling::Budget {
        seed: 7,
        updates: 6,
        size: 2,
    };
    let batches = complete_passes(&groups, 5, budget()).unwrap();
    assert_eq!(batches, complete_passes(&groups, 5, budget()).unwrap());
    assert_eq!(
        sampling::validate_batches(&batches, 5, 6, 2).unwrap(),
        [2; 5]
    );
    for pass in batches.chunks(3) {
        assert_eq!(
            pass.iter().flatten().copied().collect::<BTreeSet<_>>(),
            (0..5).collect()
        );
    }
    assert!(
        complete_passes(
            &groups,
            5,
            sampling::Budget {
                seed: 7,
                updates: 5,
                size: 2
            }
        )
        .is_err()
    );
    assert!(
        complete_passes(
            &groups,
            5,
            sampling::Budget {
                seed: 7,
                updates: 6,
                size: 0
            }
        )
        .is_err()
    );
    let malformed = BTreeMap::from([("all".into(), vec![0, 0, 2, 3, 4])]);
    assert!(complete_passes(&malformed, 5, budget()).is_err());
}

#[test]
fn mixed_canonical_receipt_binds_owners_labels_and_all_features() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("canonical.json");
    let record = json!({"native_documents":1,"authored_documents":1,"train_inputs":[
        {"origin":{"origin":"native","row":0},"features":{"flags":[0],"labels":[1]}},
        {"origin":{"origin":"authored_train_only","packet":0,"row":0},"features":{"flags":[8388608],"labels":[0]}}
    ]});
    let bytes = serde_json::to_vec(&record).unwrap();
    std::fs::write(&path, &bytes).unwrap();
    let receipt = Receipt {
        path: path.clone(),
        sha256: digest(&bytes),
    };
    verify_canonical(&receipt, &record).unwrap();
    for (field, value) in [("labels", json!([2])), ("flags", json!([0]))] {
        let mut changed = record.clone();
        changed["train_inputs"][1]["features"][field] = value;
        assert!(verify_canonical(&receipt, &changed).is_err());
    }
    let mut changed = record.clone();
    changed["train_inputs"][1]["origin"] = json!({"origin":"native","row":1});
    assert!(verify_canonical(&receipt, &changed).is_err());
    changed = record.clone();
    changed["authored_documents"] = json!(2);
    assert!(verify_canonical(&receipt, &changed).is_err());
    std::fs::write(&path, b"{}").unwrap();
    assert!(verify_canonical(&receipt, &record).is_err());
}

#[test]
fn token_accounting_reports_flag_only_activity_and_rejects_positive_rule_gold() {
    let mut doc = DetectorDoc {
        text: "Mary 5551234567".into(),
        gold: vec![],
        breaks: vec![false; 2],
        enc: Encoded {
            token_spans: vec![(0, 4), (5, 15)],
            ngram_ids: vec![vec![1]; 2],
            script: vec![0; 2],
            shape: vec![0; 2],
            flags: vec![0, tessera::internal::flag::IN_RULE_SPAN],
            labels: vec![1, 0],
            country: "US".into(),
        },
    };
    let report = token_counts(&doc, &[1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0], None).unwrap();
    assert_eq!(report["all_label_counts"], json!([1, 1, 0, 0, 0, 0, 0]));
    assert_eq!(report["active_label_counts"], json!([0, 1, 0, 0, 0, 0, 0]));
    assert_eq!(report["excluded_rule_o_tokens"], 1);
    assert_eq!(report["active_weighted_mass"], 2.0);
    doc.enc.labels[1] = 3;
    assert!(token_counts(&doc, &[1.0; 7], None).is_err());
    for bad in [vec![1.0; 6], vec![0.0; 7], vec![f32::NAN; 7]] {
        assert!(weights(&bad).is_err());
    }
}

#[test]
fn metadata_can_measure_reference_weights_but_execution_needs_both_authorities() {
    let value = json!({"scope":"reviewed-native-authored-train-only-v1","repository_root":"/source", "authored":[],
        "native_documents":340,"authored_documents":56,"dev_documents":100,"postprocess":"address_labeled_fields_v1",
        "canonical_preflight":null,"loss_weights":null});
    let mut settings: Settings = serde_json::from_value(value).unwrap();
    validate_native_counts(&settings, 340, 100).unwrap();
    assert!(validate_native_counts(&settings, 339, 100).is_err());
    assert!(validate_native_counts(&settings, 340, 99).is_err());
    assert!(settings.execution_ready().is_err());
    settings.canonical_preflight = Some(Receipt {
        path: "/canonical".into(),
        sha256: "a".repeat(64),
    });
    assert!(settings.execution_ready().is_err());
    settings.loss_weights = Some(vec![1.0; 7]);
    settings.execution_ready().unwrap();
    settings.loss_weights = Some(vec![f32::INFINITY; 7]);
    assert!(settings.execution_ready().is_err());
}

#[test]
fn mixed_draws_map_to_native_rows_by_explicit_owner_not_flat_position() {
    let loaded = LoadedMixed {
        authored: Vec::new(),
        indices: vec![
            Index::AuthoredTrainOnly { packet: 0, row: 0 },
            Index::Native { row: 1 },
            Index::Native { row: 0 },
        ],
        record: json!({}),
        dependencies: Vec::new(),
    };
    assert_eq!(
        loaded.native_owners(&[2, 0, 1]).unwrap(),
        vec![Some(0), None, Some(1)]
    );
    assert!(loaded.native_owners(&[3]).is_err());
    let resolved = supervision::Resolved::from_rows(BTreeMap::from([(0, vec![true, false])]));
    let item = |n: usize| Encoded {
        token_spans: (0..n as u32).map(|i| (i, i + 1)).collect(),
        ngram_ids: vec![vec![1]; n],
        script: vec![0; n],
        shape: vec![0; n],
        flags: vec![0; n],
        labels: vec![0; n],
        country: "US".into(),
    };
    let rows = resolved
        .batch(
            &loaded.native_owners(&[1, 2, 0]).unwrap(),
            &[item(1), item(2), item(2)],
        )
        .unwrap();
    assert_eq!(
        rows,
        vec![vec![false], vec![true, false], vec![false, false]]
    );
}

#[test]
fn token_accounting_adds_reviewed_ambiguity_only_when_declared() {
    let doc = DetectorDoc {
        text: "Respondent Mary".into(),
        gold: vec![],
        breaks: vec![false; 2],
        enc: Encoded {
            token_spans: vec![(0, 10), (11, 15)],
            ngram_ids: vec![vec![1]; 2],
            script: vec![0; 2],
            shape: vec![0; 2],
            flags: vec![0, 0],
            labels: vec![0, 1],
            country: "US".into(),
        },
    };
    let plain = token_counts(&doc, &[1.0; 7], None).unwrap();
    assert!(plain.get("excluded_reviewed_ambiguity_o_tokens").is_none());
    let reviewed = token_counts(&doc, &[1.0; 7], Some(&[true, false])).unwrap();
    assert_eq!(reviewed["excluded_reviewed_ambiguity_o_tokens"], 1);
    assert_eq!(
        reviewed["active_label_counts"],
        json!([0, 1, 0, 0, 0, 0, 0])
    );
    assert_eq!(reviewed["all_label_counts"], plain["all_label_counts"]);
    assert!(token_counts(&doc, &[1.0; 7], Some(&[false, true])).is_err());
    assert!(token_counts(&doc, &[1.0; 7], Some(&[true])).is_err());
}
