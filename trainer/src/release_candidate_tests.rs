use super::*;
use clap::Parser;

fn checkpoint() -> Value {
    json!({"scope":crate::fullmix_rms::HARD_SCOPE,"optimizer_updates_executed":4000,
        "checkpoint_file":"checkpoints/diagnostic-step-4000.mpk","selection":"snapshot",
        "parameter_sha256":"a".repeat(64),"objective_identity_sha256":crate::fullmix_hard::objective_identity().unwrap(),
        "release_quality_claim":false})
}

#[test]
fn promotion_requires_explicit_waiver_and_a_new_destination() {
    let cli = [
        "trainer",
        "promote",
        "--run",
        "final",
        "--training-run",
        "native",
        "--evaluation",
        "evaluation.json",
        "--out",
        "stage",
    ];
    assert!(crate::Cli::try_parse_from(cli).is_err());
    let mut accepted = cli.to_vec();
    accepted.push("--accept-experimental-quality");
    assert!(crate::Cli::try_parse_from(accepted).is_ok());
    let dir = tempfile::tempdir().unwrap();
    let absent = dir.path().join("absent");
    assert!(
        promote(&absent, &absent, &absent, &absent, false)
            .unwrap_err()
            .to_string()
            .contains("accept-experimental-quality")
    );
    assert!(
        promote(&absent, &absent, &absent, dir.path(), true)
            .unwrap_err()
            .to_string()
            .contains("already exists")
    );
    assert!(!absent.exists());
    let evidence = dir.path().join("evaluation.json");
    std::fs::write(&evidence, b"{}").unwrap();
    assert!(
        promote(dir.path(), dir.path(), &evidence, &absent, true)
            .unwrap_err()
            .to_string()
            .contains("outside the original runs")
    );
    assert!(!absent.exists());
}

#[test]
fn release_selects_only_the_native_final_snapshot() {
    let proof = checkpoint();
    final_checkpoint(&proof).unwrap();
    for (key, value) in [
        ("optimizer_updates_executed", json!(3999)),
        ("optimizer_updates_executed", json!(true)),
        ("checkpoint_file", json!("best.mpk")),
        ("selection", json!("development-selected")),
        ("scope", json!(crate::fullmix_rms::CONTEXT_SCOPE)),
        ("objective_identity_sha256", json!("b".repeat(64))),
    ] {
        let mut changed = proof.clone();
        changed[key] = value;
        assert!(final_checkpoint(&changed).is_err(), "{key}");
    }
}

#[test]
fn bounded_completion_requires_both_enforced_learning_groups() {
    let summary = json!({"steps":4000,"training_complete":false,"learning_check_passed":true,
        "learning_check_scope":"final bounded checkpoint"});
    let mut group = json!({"step":4000,"start_step":4000,"min_recall":0.5,"enforced":true,"passed":true,"failed_kinds":[],"per_kind":{}});
    for kind in ["address", "org", "person"] {
        group["per_kind"][kind] = json!({"gold":1,"exact":{"recall":0.5}});
    }
    let result = json!({"step":4000,"diagnostic_scope":crate::fullmix_rms::HARD_SCOPE,
        "passed":true,"enforced":true,"source_groups":{"real":group,"synthetic":group}});
    learning(&summary, &result).unwrap();
    for role in ["real", "synthetic"] {
        let mut changed = result.clone();
        changed["source_groups"][role]["passed"] = json!(false);
        assert!(learning(&summary, &changed).is_err());
        changed["source_groups"][role] = Value::Null;
        assert!(learning(&summary, &changed).is_err());
        let mut changed = result.clone();
        changed["source_groups"][role]["per_kind"]["org"]["exact"]["recall"] = json!(0.49);
        assert!(learning(&summary, &changed).is_err());
    }
    let mut partial = summary;
    partial["steps"] = json!(3999);
    assert!(learning(&partial, &result).is_err());
}

#[test]
fn release_closures_reject_tampering_and_source_path_substitution() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("checkpoint");
    std::fs::write(&path, b"immutable checkpoint").unwrap();
    let staged = BTreeMap::from([(PathBuf::from("checkpoint"), hash(&path).unwrap())]);
    let original = BTreeMap::from([(std::fs::canonicalize(&path).unwrap(), hash(&path).unwrap())]);
    let release = Release {
        schema: SCHEMA.into(),
        experimental_quality_accepted: true,
        release_artifact_complete: true,
        training_complete: false,
        training_seen_95_percent_gate_passed: false,
        fresh_unseen_evaluation_pending: true,
        general_accuracy_claim: false,
        optimizer_updates_executed: 4000,
        checkpoint: checkpoint(),
        source_run: dir.path().into(),
        training_run: dir.path().into(),
        evaluation: Pin {
            path: path.clone(),
            sha256: hash(&path).unwrap(),
        },
        staged_files: staged.clone(),
        original_files: original.clone(),
    };
    source_binding(&release, &path, Path::new("checkpoint")).unwrap();
    let foreign = dir.path().join("foreign");
    std::fs::write(&foreign, b"immutable checkpoint").unwrap();
    assert!(source_binding(&release, &foreign, Path::new("checkpoint")).is_err());
    assert!(source_binding(&release, &path, Path::new("foreign")).is_err());
    verify_hashes(Some(dir.path()), &staged).unwrap();
    verify_hashes(None, &original).unwrap();
    assert!(verify_hashes(None, &staged).is_err());
    assert!(
        verify_hashes(
            Some(dir.path()),
            &BTreeMap::from([(PathBuf::from("../checkpoint"), hash(&path).unwrap())])
        )
        .is_err()
    );
    std::fs::write(&path, b"changed checkpoint").unwrap();
    assert!(verify_hashes(Some(dir.path()), &staged).is_err());
    assert!(verify_hashes(None, &original).is_err());
}

#[test]
fn both_native_evaluations_must_bind_the_same_checkpoint_config_and_gold() {
    let dir = tempfile::tempdir().unwrap();
    for file in FINAL_FILES {
        std::fs::write(dir.path().join(file), file.as_bytes()).unwrap();
    }
    let mut gold = json!({});
    let proof = checkpoint();
    let mut report = json!({"release_ready":false,"fresh_unseen_release_evaluation_still_required":true,
        "training_seen_95_percent_gate_passed":false,"candidate_provenance":{},"evidence":[]});
    for role in ["authored", "real"] {
        let path = dir.path().join(format!("{role}.jsonl"));
        std::fs::write(&path, role).unwrap();
        gold[role] = json!({"path":path,"sha256":hash(&path).unwrap()});
        let mut provenance = json!({"best_sha256":hash(&dir.path().join("best.mpk")).unwrap(),
            "config_sha256":hash(&dir.path().join("config.toml")).unwrap(),"gold_sha256":hash(&path).unwrap(),
            "checkpoint":proof,"loaded_parameter_sha256":proof["parameter_sha256"],
            "diagnostic_scope":crate::fullmix_rms::HARD_SCOPE,"preflight":false,
            "architecture":crate::diagnostic_operator::CONTEXT_NAME,"decoder_contract":tessera::internal::CONTEXT96_DECODER_CONTRACT,
            "evaluation_role":"TRAINING-SEEN final4000 fitting diagnostic","release_quality_claim":false});
        for (key, file) in [
            ("diagnostic_sha256", "diagnostic.json"),
            ("selection_sha256", "selection.json"),
            ("fullmix_manifest_sha256", "fullmix.json"),
            ("operator_sha256", "context-operator.json"),
            ("checkpoint_receipt_sha256", "checkpoint.json"),
        ] {
            provenance[key] = json!(hash(&dir.path().join(file)).unwrap());
        }
        report["candidate_provenance"][role] = provenance.clone();
        let subdir = dir.path().join(role);
        std::fs::create_dir(&subdir).unwrap();
        let confidence = subdir.join("confidence.json");
        std::fs::write(&confidence, json!({"provenance":provenance}).to_string()).unwrap();
        report["evidence"]
            .as_array_mut()
            .unwrap()
            .push(json!({"path":confidence,"sha256":hash(&confidence).unwrap()}));
    }
    evaluation_provenance(&report, &proof, dir.path(), &gold).unwrap();
    assert_eq!(evaluated_files(&report).unwrap().len(), 2);
    for key in [
        "best_sha256",
        "config_sha256",
        "gold_sha256",
        "loaded_parameter_sha256",
    ] {
        let mut changed = report.clone();
        changed["candidate_provenance"]["real"][key] = json!("foreign");
        assert!(evaluation_provenance(&changed, &proof, dir.path(), &gold).is_err());
        assert!(evaluated_files(&changed).is_err());
    }
}

#[test]
fn release_requires_the_exact_named_evidence_closure() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("final");
    let training = dir.path().join("fit/native");
    let stage = dir.path().join("stage");
    let mut staged = BTreeMap::new();
    let mut originals = BTreeMap::new();
    let mut preserve = |path: PathBuf, destination: PathBuf| {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"evidence").unwrap();
        copy_file(&path, &destination, &stage, &mut staged, &mut originals).unwrap();
    };
    for file in FINAL_FILES {
        preserve(source.join(file), Path::new("source/final").join(file));
    }
    for file in SHIPPING_FILES {
        preserve(source.join(file), file.into());
    }
    for file in training_files() {
        preserve(
            training.join(file),
            Path::new("source/training/native").join(file),
        );
    }
    preserve(
        training.parent().unwrap().join("exit.json"),
        "source/training/exit.json".into(),
    );
    let evaluation = dir.path().join("evaluation.json");
    preserve(evaluation.clone(), "source/evaluation.json".into());
    let mut report = json!({"candidate_provenance":{},"evidence":[]});
    for role in ["real", "authored"] {
        let path = dir.path().join(role).join("confidence.json");
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let provenance = json!({"role":role});
        std::fs::write(&path, json!({"provenance":provenance}).to_string()).unwrap();
        report["candidate_provenance"][role] = provenance;
        report["evidence"]
            .as_array_mut()
            .unwrap()
            .push(json!({"path":path,"sha256":hash(&path).unwrap()}));
        copy_file(
            &path,
            &Path::new("source").join(format!("evaluation-{role}.json")),
            &stage,
            &mut staged,
            &mut originals,
        )
        .unwrap();
    }
    staged.insert("summary.json".into(), "f".repeat(64));
    let mut release = Release {
        schema: SCHEMA.into(),
        experimental_quality_accepted: true,
        release_artifact_complete: true,
        training_complete: false,
        training_seen_95_percent_gate_passed: false,
        fresh_unseen_evaluation_pending: true,
        general_accuracy_claim: false,
        optimizer_updates_executed: 4000,
        checkpoint: checkpoint(),
        source_run: source.clone(),
        training_run: training.clone(),
        evaluation: Pin {
            path: evaluation.clone(),
            sha256: hash(&evaluation).unwrap(),
        },
        staged_files: staged,
        original_files: originals,
    };
    exact_closure(&release, &report).unwrap();
    let removed = release
        .staged_files
        .remove(Path::new("summary.json"))
        .unwrap();
    assert!(exact_closure(&release, &report).is_err());
    release.staged_files.insert("summary.json".into(), removed);
    let key = std::fs::canonicalize(source.join("best.mpk")).unwrap();
    let removed = release.original_files.remove(&key).unwrap();
    assert!(exact_closure(&release, &report).is_err());
    release.original_files.insert(key, removed);
    release.source_run = training.clone();
    assert!(exact_closure(&release, &report).is_err());
    release.source_run = source.clone();
    release.training_run = source;
    assert!(exact_closure(&release, &report).is_err());
    release.training_run = training;
    release.evaluation.path = release.source_run.join("cases.jsonl");
    assert!(exact_closure(&release, &report).is_err());
    release.evaluation.path = evaluation;
    release
        .staged_files
        .insert("unexpected".into(), "f".repeat(64));
    assert!(exact_closure(&release, &report).is_err());
}
