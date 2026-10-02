use super::*;

fn pin(path: &Path) -> Receipt {
    Receipt {
        path: path.to_path_buf(),
        sha256: crate::export::sha256_hex(&std::fs::read(path).unwrap()),
    }
}

#[test]
fn objective_rejects_formula_group_coefficient_and_unknown_field_changes() {
    println!(
        "objective_identity_sha256={}",
        objective_identity().unwrap()
    );
    let receipt = json!({"path":"/not/read","sha256":"a".repeat(64)});
    let value = json!({"schema":"native_hard_objective_v1","objective_identity_sha256":objective_identity().unwrap(),
        "selector_draft":receipt,"native_selector_proof":receipt,"hard_dose_preview":receipt,
        "corrected_sources":receipt,"corrected_gold":receipt,"measured_batch_source":receipt,
        "source_reviews":[],"native_dose_reviews":[],"feature_sha256":"a","class_weights_sha256":"b",
        "groups":GROUPS,"coefficients":[0.0625,0.0625,0.0625]});
    serde_json::from_value::<Contract>(value.clone())
        .unwrap()
        .validate_identity()
        .unwrap();
    for (key, bad) in [
        ("schema", json!("legacy")),
        ("objective_identity_sha256", json!("a".repeat(64))),
        ("groups", json!(["org_contrast", "person", "address"])),
        ("coefficients", json!([0.0625, 0.0625, 0.125])),
    ] {
        let mut changed = value.clone();
        changed[key] = bad;
        assert!(
            serde_json::from_value::<Contract>(changed)
                .unwrap()
                .validate_identity()
                .is_err()
        );
    }
    let mut unknown = value;
    unknown["enable"] = json!(true);
    assert!(serde_json::from_value::<Contract>(unknown).is_err());
}

#[test]
fn same_prepared_batch_records_only_after_success_and_simulation_is_truthful() {
    let dir = tempfile::tempdir().unwrap();
    let rows = Rows::new(vec![
        NativeRow::new(
            0,
            vec![1, 0],
            Some([vec![1., 0.], vec![0., 0.], vec![0., 0.]]),
        )
        .unwrap(),
    ])
    .unwrap();
    let prepared = PreparedBatch::collate(
        &rows,
        &[0],
        &[1., 3., 2., 3., 2., 2., 1.5],
        Some([0.0625; 3]),
        0,
        0.01,
    )
    .unwrap();
    let expected = vec![prepared.record().clone()];
    let mut completed = Completed {
        rows,
        accounting: Accounting::selected_only(expected, std::collections::BTreeSet::from([(0, 0)]))
            .unwrap(),
        records: vec![],
        selected: vec![
            json!({"name":"fixture","native_index":0,"token_index":0,"group":"person","native_label":1,"span":[0,1]}),
        ],
        expected: json!({"batches":[]}),
        bindings: json!({}),
        run: dir.path().to_path_buf(),
        log: std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dir.path().join("completed-batches.jsonl"))
            .unwrap(),
        simulation: true,
    };
    completed
        .prepare(&[0], &[1., 3., 2., 3., 2., 2., 1.5], 0.01)
        .unwrap();
    assert!(completed.records.is_empty());
    assert!(completed.accounting.presentations().is_empty());
    assert!(
        completed
            .prepare(&[0], &[1., 3., 2., 3., 2., 2., 1.5], 0.02)
            .is_err()
    );
    assert!(completed.records.is_empty());
    completed.record(&prepared).unwrap();
    assert_eq!(completed.accounting.presentations()[&0], 1);
    assert_eq!(completed.accounting.cumulative_coefficients().count(), 1);
    assert!(!dir.path().join("completed-exposure.json").exists());
    completed.write().unwrap();
    let proof: Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("completed-exposure.json")).unwrap())
            .unwrap();
    assert_eq!(proof["simulation"], true);
    assert_eq!(proof["optimizer_updates_executed"], 0);
    assert_eq!(proof["simulated_records"], 1);
    assert_eq!(proof["complete"], false);
    assert_eq!(
        std::fs::read_to_string(dir.path().join("completed-batches.jsonl"))
            .unwrap()
            .lines()
            .count(),
        1
    );
    assert!(completed.record(&prepared).is_err());
    assert_eq!(completed.records.len(), 1);
}

fn completion_fixture(dir: &Path) -> (Value, Value) {
    let dose = json!({"base":0.1,"auxiliary":[0.0625,0.,0.],"total":0.1625,"lr_base":0.001,
        "lr_auxiliary":[0.000625,0.,0.],"lr_total":0.001625});
    let tokens = json!([{"index":0,"span":[0,3],"native_label":1,"group":"person","dose":dose}]);
    let expected = json!({"batches":vec![json!({"fixture":"pure reader; no optimizer executed"});4000],
        "rows":[{"name":"fixture","actual_native_index":0,"presentations":4000,"selected_tokens":tokens}]});
    let path = dir.join("dose.json");
    std::fs::write(&path, serde_json::to_vec(&expected).unwrap()).unwrap();
    let pin = serde_json::to_value(pin(&path)).unwrap();
    let contract = json!({"objective_identity_sha256":objective_identity().unwrap(),"selector_draft":pin,
        "native_selector_proof":pin,"hard_dose_preview":pin,"corrected_sources":pin,"corrected_gold":pin});
    let manifest = json!({"scope":crate::fullmix_rms::HARD_SCOPE,"config":pin,"exposure":pin,"typed_diagnostic":{"hard_objective":contract}});
    let proof = json!({"scope":COMPLETED_SCOPE,"complete":true,"optimizer_updates_executed":4000,"planned_updates":4000,
        "simulation":false,"execution":"native optimizer success","simulated_records":0,
        "bindings":{"objective_identity_sha256":objective_identity().unwrap(),"config":pin,"base_native_preview":pin,
            "selector_draft":pin,"native_selector_proof":pin,"hard_dose_preview":pin,"corrected_sources":pin,"corrected_gold":pin},
        "batches":expected["batches"],"completed_batch_sequence_sha256":crate::export::sha256_hex(&serde_json::to_vec(&expected["batches"]).unwrap()),
        "planned_batch_sequence_sha256":crate::export::sha256_hex(&serde_json::to_vec(&expected["batches"]).unwrap()),
        "selected_full_parent_rows":1,"presentations":{"0":4000},"selected_tokens":[{"name":"fixture","native_index":0,"token_index":0,"group":"person","native_label":1,"span":[0,3],"dose":dose}]});
    (proof, manifest)
}

#[test]
fn checkpoint_reader_refuses_incomplete_foreign_and_mutated_objective_evidence() {
    let dir = tempfile::tempdir().unwrap();
    let (proof, manifest) = completion_fixture(dir.path());
    validate_completed(&proof, &manifest).unwrap();
    for (field, bad) in [
        ("complete", json!(false)),
        ("simulation", json!(true)),
        ("simulated_records", json!(4000)),
        ("optimizer_updates_executed", json!(3999)),
        ("execution", json!("simulation")),
        ("scope", json!("completed-native-typed-fullmix-exposure-v1")),
    ] {
        let mut changed = proof.clone();
        changed[field] = bad;
        assert!(validate_completed(&changed, &manifest).is_err());
    }
    for field in [
        "objective_identity_sha256",
        "config",
        "selector_draft",
        "native_selector_proof",
        "hard_dose_preview",
        "corrected_sources",
        "corrected_gold",
        "base_native_preview",
    ] {
        let mut changed = proof.clone();
        changed["bindings"][field] = json!("foreign");
        assert!(validate_completed(&changed, &manifest).is_err());
    }
    for field in [
        "name",
        "native_index",
        "token_index",
        "group",
        "native_label",
        "span",
        "dose",
    ] {
        let mut changed = proof.clone();
        changed["selected_tokens"][0][field] = json!("mutated");
        assert!(validate_completed(&changed, &manifest).is_err());
    }
    let mut extra = proof.clone();
    extra["selected_tokens"]
        .as_array_mut()
        .unwrap()
        .push(proof["selected_tokens"][0].clone());
    assert!(validate_completed(&extra, &manifest).is_err());
    let mut missing = proof.clone();
    missing["selected_tokens"] = json!([]);
    assert!(validate_completed(&missing, &manifest).is_err());
    let mut reordered = proof.clone();
    reordered["batches"][0] = json!({"changed":"labelmask"});
    reordered["completed_batch_sequence_sha256"] = json!(crate::export::sha256_hex(
        &serde_json::to_vec(&reordered["batches"]).unwrap()
    ));
    assert!(validate_completed(&reordered, &manifest).is_err());
    let mut changed = proof;
    changed["presentations"]["0"] = json!(3999);
    assert!(validate_completed(&changed, &manifest).is_err());
}

#[test]
fn persisted_fractional_native_dose_and_scheduled_lrs_match_reader_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let rows = Rows::new(vec![
        NativeRow::new(
            0,
            vec![1, 0, 1],
            Some([vec![1., 0., 0.], vec![0.; 3], vec![0.; 3]]),
        )
        .unwrap(),
    ])
    .unwrap();
    let mut expected_records = Vec::new();
    let mut dose = json!({"base":0.,"auxiliary":[0.,0.,0.],"total":0.,"lr_base":0.,"lr_auxiliary":[0.,0.,0.],"lr_total":0.});
    for step in 0..4000 {
        let lr = crate::train::lr_at(step, 7000, 0.0001, 500);
        let prepared = PreparedBatch::collate(
            &rows,
            &[0, 0],
            &[1., 3., 2., 3., 2., 2., 1.5],
            Some([0.0625; 3]),
            step,
            lr,
        )
        .unwrap();
        for slot in 0..2 {
            let coefficient =
                serde_json::to_value(prepared.token_coefficients(slot, 0).unwrap()).unwrap();
            for field in ["base", "total", "lr_base", "lr_total"] {
                dose[field] =
                    json!(dose[field].as_f64().unwrap() + coefficient[field].as_f64().unwrap());
            }
            for field in ["auxiliary", "lr_auxiliary"] {
                for group in 0..3 {
                    dose[field][group] = json!(
                        dose[field][group].as_f64().unwrap()
                            + coefficient[field][group].as_f64().unwrap()
                    );
                }
            }
        }
        expected_records.push(prepared.record().clone());
    }
    let expected_raw = json!({"batches":expected_records,"rows":[{"name":"fractional","actual_native_index":0,
        "presentations":8000,"selected_tokens":[{"index":0,"span":[0,3],"native_label":1,"group":"person","dose":dose}]}]});
    let path = dir.path().join("fractional-planned.json");
    std::fs::write(&path, serde_json::to_vec_pretty(&expected_raw).unwrap()).unwrap();
    let expected: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let pin = serde_json::to_value(pin(&path)).unwrap();
    let bindings = json!({"objective_identity_sha256":objective_identity().unwrap(),"config":pin,"base_native_preview":pin,
        "selector_draft":pin,"native_selector_proof":pin,"hard_dose_preview":pin,"corrected_sources":pin,"corrected_gold":pin});
    let mut completed = Completed {
        rows,
        accounting: Accounting::selected_only(
            expected_records,
            std::collections::BTreeSet::from([(0, 0)]),
        )
        .unwrap(),
        records: vec![],
        selected: vec![
            json!({"name":"fractional","native_index":0,"token_index":0,"group":"person","native_label":1,"span":[0,3]}),
        ],
        expected: expected.clone(),
        bindings,
        run: dir.path().to_path_buf(),
        log: std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(dir.path().join("completed-batches.jsonl"))
            .unwrap(),
        simulation: true,
    };
    for step in 0..4000 {
        let prepared = completed
            .prepare(
                &[0, 0],
                &[1., 3., 2., 3., 2., 2., 1.5],
                crate::train::lr_at(step, 7000, 0.0001, 500),
            )
            .unwrap();
        completed.record(&prepared).unwrap();
    }
    completed.write().unwrap();
    let mut parsed: Value =
        serde_json::from_slice(&std::fs::read(dir.path().join("completed-exposure.json")).unwrap())
            .unwrap();
    assert_eq!(parsed["simulation"], true);
    assert_eq!(parsed["optimizer_updates_executed"], 0);
    assert_eq!(
        parsed["selected_tokens"][0]["dose"],
        expected["rows"][0]["selected_tokens"][0]["dose"]
    );
    assert_eq!(parsed["batches"], expected["batches"]);
    assert_eq!(
        parsed["completed_batch_sequence_sha256"],
        crate::export::sha256_hex(&serde_json::to_vec(&expected["batches"]).unwrap())
    );
    let manifest = json!({"scope":crate::fullmix_rms::HARD_SCOPE,"config":pin,"exposure":pin,"typed_diagnostic":{"hard_objective":{
        "objective_identity_sha256":objective_identity().unwrap(),"selector_draft":pin,"native_selector_proof":pin,
        "hard_dose_preview":pin,"corrected_sources":pin,"corrected_gold":pin}}});
    assert!(validate_completed(&parsed, &manifest).is_err());
    // Only an in-memory reader fixture receives executed metadata; the persisted proof stays a simulation.
    parsed["scope"] = json!(COMPLETED_SCOPE);
    parsed["simulation"] = json!(false);
    parsed["execution"] = json!("native optimizer success");
    parsed["optimizer_updates_executed"] = json!(4000);
    parsed["simulated_records"] = json!(0);
    parsed["complete"] = json!(true);
    validate_completed(&parsed, &manifest).unwrap();
}
