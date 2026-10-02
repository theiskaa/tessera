use super::*;

fn pin(path: &Path) -> Pin {
    Pin {
        path: path.to_string_lossy().into_owned(),
        sha256: hash(path).unwrap(),
    }
}

fn config() -> Config {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/detector-shared.toml");
    let mut cfg = crate::config::load(&path).unwrap();
    let detector = cfg.detector.as_mut().unwrap();
    detector.silver = (0..34).map(|i| format!("/source-{i}.jsonl")).collect();
    detector.typed_synthetic = Some(Settings {
        manifest: Pin {
            path: "/unchanged-manifest.json".to_owned(),
            sha256: "a".repeat(64),
        },
        corrected_real_sources: None,
    });
    cfg
}

fn corrected(prior: &Config) -> Config {
    let mut value = serde_json::to_value(prior).unwrap();
    value["name"] = json!("data-only-successor");
    for slot in [0, 6, 13, 18, 21, 22] {
        value["detector"]["silver"][slot] = json!(format!("/corrected-{slot}.jsonl"));
    }
    value["detector"]["typed_synthetic"]["corrected_real_sources"] =
        json!({"path":"/data-only-receipt.json","sha256":"b".repeat(64)});
    serde_json::from_value(value).unwrap()
}

#[test]
fn absent_successor_preserves_legacy_settings_bytes_and_invalid_settings_refuse() {
    let old = json!({"manifest":{"path":"/manifest.json","sha256":"a".repeat(64)}});
    let settings: Settings = serde_json::from_value(old.clone()).unwrap();
    assert_eq!(
        serde_json::to_vec(&settings).unwrap(),
        serde_json::to_vec(&old).unwrap()
    );
    assert!(verify(&config()).unwrap().is_none());
    for successor in [
        json!({"path":"relative","sha256":"a".repeat(64)}),
        json!({"path":"/receipt","sha256":"A".repeat(64)}),
    ] {
        let mut value = old.clone();
        value["corrected_real_sources"] = successor;
        assert!(
            serde_json::from_value::<Settings>(value)
                .unwrap()
                .validate()
                .is_err()
        );
    }
}

#[test]
fn config_whitelist_retains_generation_net_train_features_and_all_sampling_rules() {
    let prior = config();
    let cfg = corrected(&prior);
    let selected = cfg.detector.as_ref().unwrap().silver.clone();
    verify_config(&cfg, &prior, &selected).unwrap();
    let value = serde_json::to_value(&cfg).unwrap();
    for pointer in [
        "/seed",
        "/features",
        "/net",
        "/train",
        "/generate",
        "/names",
        "/data",
        "/detector/silver_repeat",
        "/detector/silver_repeats",
        "/detector/typed_synthetic/manifest/sha256",
    ] {
        let mut bad = value.clone();
        let target = bad.pointer_mut(pointer).unwrap();
        match pointer {
            "/seed" | "/detector/silver_repeat" => *target = json!(9876),
            "/detector/typed_synthetic/manifest/sha256" => *target = json!("c".repeat(64)),
            "/detector/silver_repeats" => *target = json!(vec![2; 34]),
            "/generate" | "/names" => {
                *target = if target.is_null() {
                    json!({"unknown":true})
                } else {
                    Value::Null
                }
            }
            _ => target["unexpected_rule"] = json!(true),
        }
        if let Ok(bad) = serde_json::from_value::<Config>(bad) {
            assert!(verify_config(&bad, &prior, &selected).is_err(), "{pointer}");
        }
    }
    let mut bad = serde_json::to_value(&cfg).unwrap();
    bad["detector"]["silver"][3] = json!("/undeclared-source.jsonl");
    let bad: Config = serde_json::from_value(bad).unwrap();
    assert!(verify_config(&bad, &prior, &selected).is_err());
}

#[test]
fn receipt_schema_and_unreviewed_prior_hash_refuse_before_prior_data_read() {
    let dir = tempfile::tempdir().unwrap();
    let absent = json!({"path":"/not-read.json","sha256":"a".repeat(64)});
    let value = json!({"schema":"reviewed_real_source_successor_v1","prior_config":absent,
        "source_deltas":absent,"reviews":[absent,absent],"gold":absent});
    for field in ["unknown", "schema", "prior_hash"] {
        let mut bad = value.clone();
        if field == "prior_hash" {
            bad["prior_config"]["sha256"] = json!("b".repeat(64));
        } else {
            bad[field] = json!("unsupported");
        }
        if field == "unknown" {
            assert!(serde_json::from_value::<Successor>(bad).is_err());
        } else {
            let path = dir.path().join("receipt.json");
            std::fs::write(&path, serde_json::to_vec(&bad).unwrap()).unwrap();
            let mut cfg = config();
            cfg.detector
                .as_mut()
                .unwrap()
                .typed_synthetic
                .as_mut()
                .unwrap()
                .corrected_real_sources = Some(pin(&path));
            assert!(
                verify(&cfg)
                    .unwrap_err()
                    .to_string()
                    .contains("exact reviewed prior config")
            );
        }
    }
}

#[test]
fn source_contract_counts_zero_35_duplicate_slots_and_aliased_replacements_refuse() {
    let dir = tempfile::tempdir().unwrap();
    let mut prior = Vec::new();
    let mut contracts = Vec::new();
    let text = "Agency";
    for slot in 0..34 {
        let before = dir.path().join(format!("before-{slot}.jsonl"));
        let after = dir.path().join(format!("after-{slot}.jsonl"));
        std::fs::write(
            &before,
            format!(
                "{}\n",
                json!({"id":"row","country":"US","text":text,"entities":[]})
            ),
        )
        .unwrap();
        let entity = json!({"kind":"org","start":0,"end":6});
        std::fs::write(
            &after,
            format!(
                "{}\n",
                json!({"id":"row","country":"US","text":text,"entities":[entity]})
            ),
        )
        .unwrap();
        prior.push(before.to_string_lossy().into_owned());
        contracts.push(json!({"schema":"entities_only_source_delta_v1","before":pin(&before),"after":pin(&after),"rows":1,
            "changes":[{"physical_row_1_based":1,"id":"row","text_sha256":crate::export::sha256_hex(text.as_bytes()),"added":[entity],"removed":[]}]}));
    }
    for count in [1, 6, 34] {
        let result = source_selection(&prior, &contracts[..count]).unwrap();
        assert_eq!(result.files.len(), 2 * count);
        assert_eq!(&result.sources[count..], &prior[count..]);
    }
    assert!(source_selection(&prior, &[]).is_err());
    let mut too_many = contracts.clone();
    too_many.push(contracts[0].clone());
    assert!(source_selection(&prior, &too_many).is_err());
    assert!(source_selection(&prior, &[contracts[0].clone(), contracts[0].clone()]).is_err());
    let mut alias = contracts[1].clone();
    alias["after"] = contracts[0]["after"].clone();
    assert!(source_selection(&prior, &[contracts[0].clone(), alias]).is_err());
}

#[test]
fn two_review_maps_or_checked_lists_bind_all_required_files_and_refuse_mutations() {
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("required.json");
    std::fs::write(&file, b"reviewed").unwrap();
    let required = pin(&file);
    let bindings = json!({required.path.clone():required.sha256.clone()});
    let a = json!({"passed":true,"blockers":[],"fit_approved":false,"reviewed_files":bindings});
    let b = json!({"passed":true,"blockers":[],"fit_approved":false,"checked_hashes":[required]});
    let pa = dir.path().join("a.json");
    let pb = dir.path().join("b.json");
    std::fs::write(&pa, serde_json::to_vec(&a).unwrap()).unwrap();
    std::fs::write(&pb, serde_json::to_vec(&b).unwrap()).unwrap();
    verify_reviews(&[pin(&pa), pin(&pb)], std::slice::from_ref(&required)).unwrap();
    assert!(verify_reviews(&[pin(&pa), pin(&pa)], std::slice::from_ref(&required)).is_err());
    for field in [
        "passed",
        "blockers",
        "fit_approved",
        "missing_fit",
        "binding",
        "ambiguous",
    ] {
        let mut bad = a.clone();
        match field {
            "passed" => bad[field] = json!(false),
            "blockers" => bad[field] = json!(["unresolved"]),
            "fit_approved" => bad[field] = json!(true),
            "missing_fit" => {
                bad.as_object_mut().unwrap().remove("fit_approved");
            }
            "binding" => bad["reviewed_files"] = json!({}),
            _ => bad["checked_hashes"] = json!([required]),
        }
        std::fs::write(&pa, serde_json::to_vec(&bad).unwrap()).unwrap();
        assert!(
            verify_reviews(&[pin(&pa), pin(&pb)], std::slice::from_ref(&required)).is_err(),
            "{field}"
        );
    }
}

#[test]
fn gold_requires_all_1799_source_rows_and_all_five_kinds_with_exact_utf8_spans() {
    let dir = tempfile::tempdir().unwrap();
    let entities = json!([
        {"kind":"person","start":0,"end":2}, {"kind":"org","start":3,"end":4},
        {"kind":"address","start":5,"end":6}, {"kind":"email","start":7,"end":8},
        {"kind":"phone","start":9,"end":10}]);
    let text = "É B C D E";
    let expected = entities
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            let mut e = e.clone();
            e["text"] = json!(
                text.get(
                    e["start"].as_u64().unwrap() as usize..e["end"].as_u64().unwrap() as usize
                )
                .unwrap()
            );
            e
        })
        .collect::<Vec<_>>();
    let mut selected = Vec::new();
    let mut gold = Vec::new();
    for slot in 0..34 {
        let count = if slot == 0 { 1766 } else { 1 };
        let mut source = String::new();
        for index in 0..count {
            source.push_str(&format!(
                "{}\n",
                json!({"id":format!("row-{index}"),"text":text,
                "country":"US","entities":entities})
            ));
            gold.push(
                json!({"name":format!("seen-real-{slot:02}-{index:04}"),"country":"US",
                "input":text,"expected":expected}),
            );
        }
        let path = dir.path().join(format!("source-{slot}.jsonl"));
        std::fs::write(&path, source).unwrap();
        selected.push(path.to_string_lossy().into_owned());
    }
    let path = dir.path().join("gold.jsonl");
    for mutation in [
        "none",
        "phone",
        "email",
        "text",
        "country",
        "extra",
        "duplicate",
        "utf8",
        "surface",
    ] {
        let mut rows = gold.clone();
        match mutation {
            "none" => {}
            "phone" => {
                rows[0]["expected"].as_array_mut().unwrap().pop();
            }
            "email" => {
                rows[0]["expected"].as_array_mut().unwrap().remove(3);
            }
            "text" => rows[0]["input"] = json!("different"),
            "country" => rows[0]["country"] = json!("UK"),
            "extra" => rows.push(rows[0].clone()),
            "duplicate" => rows[1] = rows[0].clone(),
            "utf8" => rows[0]["expected"][0]["start"] = json!(1),
            _ => rows[0]["expected"][0]["text"] = json!("different"),
        }
        let data = rows.iter().map(|r| format!("{r}\n")).collect::<String>();
        std::fs::write(&path, data).unwrap();
        let result = verify_gold(&pin(&path), &selected);
        if mutation == "none" {
            result.unwrap();
        } else {
            assert!(result.is_err(), "{mutation}");
        }
    }
}
