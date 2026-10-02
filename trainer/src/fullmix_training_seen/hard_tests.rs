use super::*;

fn pin(path: PathBuf) -> Receipt {
    Receipt {
        sha256: crate::fullmix_rms::hash(&path).unwrap(),
        path,
    }
}

fn support_plan() -> Value {
    json!({"schema":SCHEMA,"model_gold_support":{
        "authored":{"person":600,"org":600,"address":120},
        "real":{"person":9817,"org":1912,"address":600}},
        "declared_real_gold_support":{"person":9817,"org":1912,"address":600,"email":226,"phone":335}})
}

fn binding() -> Binding {
    let receipt = json!({"path":"/fixture/artifact","sha256":"a".repeat(64)});
    let mut value = serde_json::Map::new();
    for key in [
        "config",
        "typed_manifest",
        "source_successor_receipt",
        "membership",
        "base_native_preview",
        "selector_draft",
        "native_selector_proof",
        "hard_dose_preview",
        "hard_objective",
        "learning_acceptance",
        "learning_spec",
        "learning_invariants",
        "same_gold_baseline",
        "learning_checker",
        "learning_checker_successor",
        "promotion_checker",
    ] {
        value.insert(key.to_owned(), receipt.clone());
    }
    for key in [
        "source_delta_reviews",
        "production_source_reviews",
        "native_dose_reviews",
        "evaluation_reviews",
        "selected_real_sources",
    ] {
        value.insert(key.to_owned(), json!([]));
    }
    serde_json::from_value(Value::Object(value)).unwrap()
}

#[test]
fn explicit_version_and_all_five_supports_are_required() {
    validate_support(&support_plan()).unwrap();
    for field in ["person", "org", "address", "email", "phone"] {
        let mut altered = support_plan();
        altered["declared_real_gold_support"][field] = json!(0);
        assert!(validate_support(&altered).is_err());
    }
    for org in [1900, 1904, 1911, 1913] {
        let mut altered = support_plan();
        altered["model_gold_support"]["real"]["org"] = json!(org);
        assert!(validate_support(&altered).is_err());
    }
    for schema in [
        Value::Null,
        json!("training_seen_evaluation_v4_1904_v1"),
        json!("unknown"),
    ] {
        let mut altered = support_plan();
        altered["schema"] = schema;
        assert!(
            supports(&altered, &json!({}))
                .unwrap_err()
                .to_string()
                .contains("unsupported v5")
        );
    }
    assert_eq!(
        super::super::legacy_bindings::supports(&json!({}), &json!({})).unwrap(),
        [9817, 1900, 600]
    );
}

#[test]
fn binding_refuses_missing_fields_unknown_fields_and_changed_receipts() {
    let value = serde_json::to_value(binding()).unwrap();
    for key in value.as_object().unwrap().keys() {
        let mut altered = value.clone();
        altered.as_object_mut().unwrap().remove(key);
        assert!(serde_json::from_value::<Binding>(altered).is_err(), "{key}");
    }
    let mut altered = value;
    altered["allow_legacy_gold"] = json!(true);
    assert!(serde_json::from_value::<Binding>(altered).is_err());
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("artifact.json");
    std::fs::write(&path, b"{}").unwrap();
    let pinned = pin(path.clone());
    let review = json!({"reviewed_files":{path.to_string_lossy():pinned.sha256}});
    reviewed(&pinned, &review).unwrap();
    assert!(reviewed(&pinned, &json!({"reviewed_files":{}})).is_err());
    std::fs::write(path, b"{\"changed\":true}").unwrap();
    assert!(reviewed(&pinned, &review).is_err());
}

#[test]
fn independent_receipts_must_be_distinct_and_bind_required_artifacts() {
    let dir = tempfile::tempdir().unwrap();
    let artifact_path = dir.path().join("artifact");
    std::fs::write(&artifact_path, b"data").unwrap();
    let artifact = pin(artifact_path);
    let mut reviews = Vec::new();
    let mut files = serde_json::Map::new();
    for i in 0..2 {
        let path = dir.path().join(format!("review{i}.json"));
        std::fs::write(
            &path,
            serde_json::to_vec(&json!({"passed":true,"blockers":[],"fit_approved":false,
            "reviewer":i,"reviewed_files":{artifact.path.to_string_lossy():artifact.sha256}}))
            .unwrap(),
        )
        .unwrap();
        let receipt = pin(path);
        files.insert(
            receipt.path.to_string_lossy().into_owned(),
            json!(receipt.sha256),
        );
        reviews.push(receipt);
    }
    let review = json!({"reviewed_files":files});
    review_pair(&reviews, &[&artifact], &review).unwrap();
    let missing = Receipt {
        path: dir.path().join("unreviewed"),
        sha256: "b".repeat(64),
    };
    assert!(review_pair(&reviews, &[&missing], &review).is_err());
    let repeated: Vec<Receipt> = serde_json::from_value(json!([reviews[0], reviews[0]])).unwrap();
    assert!(review_pair(&repeated, &[&artifact], &review).is_err());
}

#[test]
fn exact_five_kind_utf8_gold_preserves_rules_and_rejects_overlap() {
    let source = json!({"text":"é Joe Unit Road a@b.co 123","entities":[
        {"kind":"person","start":3,"end":6},{"kind":"org","start":7,"end":11},
        {"kind":"address","start":12,"end":16},{"kind":"email","start":17,"end":23},
        {"kind":"phone","start":24,"end":27}]});
    let document = json!({"expected":[
        {"kind":"person","start":3,"end":6,"text":"Joe"},
        {"kind":"org","start":7,"end":11,"text":"Unit"},
        {"kind":"address","start":12,"end":16,"text":"Road"},
        {"kind":"email","start":17,"end":23,"text":"a@b.co"},
        {"kind":"phone","start":24,"end":27,"text":"123"}]});
    assert_eq!(source_entities(&source, &document).unwrap(), [1; 5]);
    for i in 0..5 {
        let mut altered = document.clone();
        altered["expected"].as_array_mut().unwrap().remove(i);
        assert!(source_entities(&source, &altered).is_err());
    }
    let mut altered = source.clone();
    altered["entities"][0]["start"] = json!(1);
    assert!(source_entities(&altered, &document).is_err());
    altered = source;
    altered["entities"][1]["start"] = json!(5);
    assert!(source_entities(&altered, &document).is_err());
}

#[test]
fn selected_gold_hash_uses_selector_key_order_and_preserves_annotation_order() {
    let first: Value =
        serde_json::from_str(r#"[{"kind":"person","start":0,"end":3,"text":"Joe"}]"#).unwrap();
    let reordered: Value =
        serde_json::from_str(r#"[{"text":"Joe","end":3,"start":0,"kind":"person"}]"#).unwrap();
    assert_ne!(
        serde_json::to_vec(&first).unwrap(),
        serde_json::to_vec(&reordered).unwrap()
    );
    assert_eq!(
        expected_hash(&first).unwrap(),
        expected_hash(&reordered).unwrap()
    );
    assert_eq!(
        canonical(&first),
        serde_json::from_str::<Value>(r#"[{"end":3,"kind":"person","start":0,"text":"Joe"}]"#)
            .unwrap()
    );
    let two = json!([first[0],{"kind":"person","start":4,"end":7,"text":"Ann"}]);
    let reversed = json!([two[1], two[0]]);
    assert_ne!(
        expected_hash(&two).unwrap(),
        expected_hash(&reversed).unwrap()
    );
}

fn native_fixture() -> (Binding, Value, Value, Value, Value, Vec<Value>, Vec<Value>) {
    let binding = binding();
    let gold_hash =
        expected_hash(&json!([{"kind":"person","start":0,"end":3,"text":"Joe"}])).unwrap();
    let gold: Vec<_> = (0..541)
        .map(|i| json!({"name":format!("parent{i}"),"input":"Joe","expected":[{"kind":"person","start":0,"end":3,"text":"Joe"}]}))
        .collect();
    let masks:Vec<_>=gold.iter().map(|row|json!({"name":row["name"],"text_sha256":crate::export::sha256_hex(b"Joe"),
        "expected_sha256":gold_hash,"native_encoding_sha256":"encoding","binary_masks_sha256":"masks"})).collect();
    let selected: Vec<_> = gold
        .iter()
        .enumerate()
        .map(|(i, row)| {
            json!({"name":row["name"],"actual_native_index":170000+i,
        "native_encoding_sha256":"encoding","binary_masks_sha256":"masks","presentations":1})
        })
        .collect();
    let batches: Vec<_> = (0..4000)
        .map(|i| {
            json!({"optimizer_step_index":i,"indices":[1,2],
        "native_weighted_token_denominator":3.0,"scheduled_learning_rate":0.0001})
        })
        .collect();
    let hard_batches: Vec<_> = (0..4000)
        .map(|i| {
            json!({"update_index":i,"ordered_native_indices":[1,2],
        "base_denominator":3.0,"scheduled_learning_rate":0.0001})
        })
        .collect();
    let base = json!({"config_sha256":binding.config.sha256,"silver":{"unreachable_spans":0,"documents":1799,
        "pieces":1799,"by_country":{"US":{"person":9817,"org":1912,"address":600}}},
        "typed_synthetic":{"uniform_synthetic_pool":170000,"full_synthetic_boundary":170569,"real_documents":1799,
        "planned_updates":4000,"model_initialized":false,"model_forwards":0,"optimizer_updates_executed":0,
        "batch_sequence_sha256":"sequence","batches":batches}});
    let draft = json!({"groups":GROUPS,"lambda_each_proposal":0.0625,"model_operations":0,"training_ready":false,
        "records":vec![Value::Null;541],"evidence":[]});
    let proof = json!({"draft":binding.selector_draft,"evidence":[],"passed":true,"full_parent_rows":541,
        "group_order":GROUPS,"lambda_each_proposal":0.0625,"model_initialized":false,"model_forwards":0,
        "optimizer_updates":0,"training_ready":false,"rows":masks});
    let summaries: Vec<_> = GROUPS
        .iter()
        .map(|group| {
            json!({"group":group,"positive":{"selected_tokens":1,
        "zero_base_dose_tokens":0,"zero_auxiliary_dose_tokens":0},"O":{"selected_tokens":1,
        "zero_base_dose_tokens":0,"zero_auxiliary_dose_tokens":0}})
        })
        .collect();
    let dose = json!({"passed":true,"fit_approved":false,"training_ready":false,"model_initialized":false,
        "model_forwards":0,"optimizer_updates":0,"inputs":[binding.config,binding.selector_draft,binding.native_selector_proof],
        "native_pool_documents":172368,"uniform_synthetic_pool":170000,"authored_documents":569,"real_documents":1799,
        "selected_full_parent_rows":541,"planned_updates":4000,"learning_rate_horizon":7000,"group_order":GROUPS,
        "lambda_each":0.0625,"base_batch_sequence_sha256":"sequence","base_denominators_learning_rates_order_identical":true,
        "full_original_native_context_features_breaks_labels_verified":true,"batches":hard_batches,"rows":selected,"summaries":summaries});
    (binding, base, draft, proof, dose, vec![], gold)
}

#[test]
fn native_preview_refuses_changed_budget_order_denominator_and_learning_rate() {
    let (binding, base, draft, proof, dose, real, authored) = native_fixture();
    validate_native(&binding, &base, &draft, &proof, &dose, &real, &authored).unwrap();
    for (field, value) in [
        ("planned_updates", json!(3999)),
        ("native_pool_documents", json!(172367)),
        (
            "base_denominators_learning_rates_order_identical",
            json!(false),
        ),
        ("lambda_each", json!(0.125)),
    ] {
        let mut altered = dose.clone();
        altered[field] = value;
        assert!(
            validate_native(&binding, &base, &draft, &proof, &altered, &real, &authored).is_err()
        );
    }
    for (field, value) in [
        ("ordered_native_indices", json!([2, 1])),
        ("base_denominator", json!(4.0)),
        ("scheduled_learning_rate", json!(0.0002)),
        ("update_index", json!(1)),
    ] {
        let mut altered = dose.clone();
        altered["batches"][0][field] = value;
        assert!(
            validate_native(&binding, &base, &draft, &proof, &altered, &real, &authored).is_err()
        );
    }
}

#[test]
fn native_selected_parents_reject_historical_index_mask_hash_duplicate_and_zero_dose() {
    let (binding, base, draft, proof, dose, real, authored) = native_fixture();
    for (field, value) in [
        ("actual_native_index", json!(170569)),
        ("binary_masks_sha256", json!("changed")),
        ("native_encoding_sha256", json!("changed")),
        ("presentations", json!(0)),
        ("name", json!("absent")),
    ] {
        let mut altered = dose.clone();
        altered["rows"][0][field] = value;
        assert!(
            validate_native(&binding, &base, &draft, &proof, &altered, &real, &authored).is_err()
        );
    }
    let mut altered = proof.clone();
    altered["rows"][1] = altered["rows"][0].clone();
    assert!(validate_native(&binding, &base, &draft, &altered, &dose, &real, &authored).is_err());
    altered = dose.clone();
    altered["summaries"][0]["O"]["zero_auxiliary_dose_tokens"] = json!(1);
    assert!(validate_native(&binding, &base, &draft, &proof, &altered, &real, &authored).is_err());
}

#[test]
fn full_source_fixture_rejects_reordered_slots_membership_and_rule_gold() {
    let dir = tempfile::tempdir().unwrap();
    let mut entities = Vec::new();
    let mut expected = Vec::new();
    let mut text = String::new();
    for (kind, count) in KINDS.iter().zip([9817, 1912, 600, 226, 335]) {
        for _ in 0..count {
            let start = text.len();
            text.push_str("x ");
            entities.push(json!({"kind":kind,"start":start,"end":start+1}));
            expected.push(json!({"kind":kind,"start":start,"end":start+1,"text":"x"}));
        }
    }
    let mut sources = Vec::new();
    let mut selected = Vec::new();
    let mut gold = Vec::new();
    let mut members = Vec::new();
    let mut files = serde_json::Map::new();
    for slot in 0..34 {
        let path = dir.path().join(format!("source{slot}.jsonl"));
        let rows:Vec<_>=(0..if slot==0 {1766} else {1}).map(|physical| {
            let row=if slot==0 && physical==0 {json!({"id":format!("{slot}-{physical}"),
                "country":"US","text":text,"entities":entities})} else {
                json!({"id":format!("{slot}-{physical}"),"country":"US","text":"empty","entities":[]})};
            gold.push(json!({"name":format!("seen-real-{slot:02}-{physical:04}"),"country":"US",
                "input":row["text"],"expected":if slot==0 && physical==0 {json!(expected)} else {json!([])}}));
            row
        }).collect();
        let bytes = rows
            .iter()
            .map(|row| serde_json::to_string(row).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&path, bytes).unwrap();
        let receipt = pin(path.clone());
        for (physical, row) in rows.iter().enumerate() {
            members.push(json!({"name":format!("seen-real-{slot:02}-{physical:04}"),"id":row["id"],
                "source_slot":slot,"physical_row_1_based":physical+1,"native_index":170569+members.len(),
                "source":receipt}));
        }
        files.insert(path.to_string_lossy().into_owned(), json!(receipt.sha256));
        selected.push(path.to_string_lossy().into_owned());
        sources.push(Source {
            source_slot: slot,
            path,
            sha256: receipt.sha256,
        });
    }
    let membership = json!({"real":members});
    let review = json!({"reviewed_files":files});
    validate_sources(&sources, &selected, &gold, &membership, &review).unwrap();
    let mut reordered = selected.clone();
    reordered.swap(0, 1);
    assert!(validate_sources(&sources, &reordered, &gold, &membership, &review).is_err());
    let mut altered = membership.clone();
    altered["real"][0]["native_index"] = json!(170000);
    assert!(validate_sources(&sources, &selected, &gold, &altered, &review).is_err());
    let mut altered = gold.clone();
    altered[0]["expected"].as_array_mut().unwrap().pop();
    assert!(validate_sources(&sources, &selected, &altered, &membership, &review).is_err());
    altered = gold;
    altered.swap(0, 1);
    assert!(validate_sources(&sources, &selected, &altered, &membership, &review).is_err());
}

fn learning_lineage_entry(sha256: &str, scope: &str, marker: &str) -> (Receipt, Value) {
    (
        Receipt {
            path: PathBuf::from(format!("/metadata/{marker}.json")),
            sha256: sha256.to_owned(),
        },
        json!({"scope":scope,"marker":marker}),
    )
}

#[test]
fn original_learning_freezes_use_exact_pins_in_either_evidence_order() {
    let scope = "frozen training-seen learning acceptance; no unseen accuracy claim";
    let spec = "6007235876d2a008376b66edea3ce07361294460757cb78e29e7d637ba2866fd";
    let acceptance = "f4b1c2bbe6a2a4683ab27bed3ba99dcf793be2fe6dc7916b28156ee9497ee9a2";
    let acceptance_scope =
        "next candidate promotion requirements fixed before changing training data or fitting";
    let mut evidence = vec![
        learning_lineage_entry(spec, scope, "spec"),
        learning_lineage_entry(
            "61e971cb27e1021b6666b8dda1204fbcc1ea339b99e9a22cafb0c8443115ec7e",
            scope,
            "result",
        ),
        learning_lineage_entry(acceptance, acceptance_scope, "acceptance"),
    ];
    for _ in 0..2 {
        assert_eq!(
            original_learning_freeze(&evidence, spec, scope).unwrap()["marker"],
            "spec"
        );
        assert_eq!(
            original_learning_freeze(&evidence, acceptance, acceptance_scope).unwrap()["marker"],
            "acceptance"
        );
        evidence.reverse();
    }
}

#[test]
fn original_learning_freezes_reject_missing_duplicate_or_wrong_scope_without_fallback() {
    for (sha, scope) in [
        (
            "6007235876d2a008376b66edea3ce07361294460757cb78e29e7d637ba2866fd",
            "frozen training-seen learning acceptance; no unseen accuracy claim",
        ),
        (
            "f4b1c2bbe6a2a4683ab27bed3ba99dcf793be2fe6dc7916b28156ee9497ee9a2",
            "next candidate promotion requirements fixed before changing training data or fitting",
        ),
    ] {
        assert!(original_learning_freeze(&[], sha, scope).is_err());
        let fallback = vec![learning_lineage_entry(&"a".repeat(64), scope, "same-scope")];
        assert!(original_learning_freeze(&fallback, sha, scope).is_err());
        let duplicate = vec![
            learning_lineage_entry(sha, scope, "first"),
            learning_lineage_entry(sha, scope, "second"),
        ];
        assert!(original_learning_freeze(&duplicate, sha, scope).is_err());
        let wrong = vec![learning_lineage_entry(sha, "wrong scope", "wrong")];
        assert!(original_learning_freeze(&wrong, sha, scope).is_err());
    }
}
