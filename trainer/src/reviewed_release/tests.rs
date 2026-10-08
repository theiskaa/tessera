use super::*;
use crate::reviewed_data::Receipt;
use crate::reviewed_fit::MIXED_SCOPE;
use clap::Parser;
use serde_json::json;

const UPDATES: usize = 3;

/// A synthetic completed mixed fit whose evidence chain, snapshot and checkpoint all agree.
pub(crate) struct Fixture {
    _root: tempfile::TempDir,
    pub(crate) base: PathBuf,
    pub(crate) run: PathBuf,
    pub(crate) fit: PathBuf,
    pub(crate) data: PathBuf,
}

fn config(base: &Path) -> Config {
    let mut cfg = crate::config::learning_test_config();
    cfg.net.architecture = Some(tessera::internal::CONTEXT96_RMS_NAME.into());
    cfg.net.hidden = 96;
    cfg.net.dilations = tessera::internal::CONTEXT96_RMS_DILATIONS.to_vec();
    cfg.net.detector_features = Some(crate::config::DetectorFeatures::TabCells25);
    cfg.net.detector_postprocess = Some(crate::config::DetectorPostprocess::AddressLabeledFieldsV1);
    cfg.names = None;
    cfg.generate = None;
    cfg.augment.copies = 0;
    cfg.data.source.clear();
    cfg.data.processed.clear();
    cfg.data.manifests.clear();
    cfg.data.coverage_exceptions.clear();
    cfg.data.train_per_country = 0;
    cfg.data.valid_per_country = 0;
    cfg.data.test_per_country = 0;
    cfg.train.diagnostic_schedule_steps = None;
    cfg.train.gradient_clip_norm = Some(1.0);
    let detector = cfg.detector.as_mut().unwrap();
    detector.silver.clear();
    detector.silver_repeats.clear();
    detector.silver_repeat = 1;
    detector.synthetic_per_epoch = None;
    detector.typed_synthetic = None;
    detector.learning_check = None;
    cfg.reviewed_native = Some(crate::reviewed_train::Settings {
        recipe: Receipt {
            path: base.join("recipe.json"),
            sha256: "a".repeat(64),
        },
    });
    cfg.validate().unwrap();
    cfg
}

fn write(path: &Path, value: &Value) -> String {
    let bytes = serde_json::to_vec_pretty(value).unwrap();
    std::fs::write(path, &bytes).unwrap();
    crate::export::sha256_hex(&bytes)
}

/// Rewrite the learning proof and the completion record that binds it.
fn rebind_proof(fit: &Path, edit: impl FnOnce(&mut Value)) {
    let mut proof = evidence::read(&fit.join("learning-proof.json")).unwrap();
    edit(&mut proof);
    let sha = write(&fit.join("learning-proof.json"), &proof);
    let mut completion = evidence::read(&fit.join("completion.json")).unwrap();
    completion["final"]["learning_sanity_proof"] = proof;
    completion["learning_proof_sha256"] = sha.into();
    write(&fit.join("completion.json"), &completion);
}

pub(crate) fn fit() -> Fixture {
    let root = tempfile::tempdir().unwrap();
    let base = root.path().canonicalize().unwrap();
    let run = base.join("run");
    let fit = run.join("fit");
    std::fs::create_dir_all(fit.join("checkpoints")).unwrap();
    let cfg = config(&base);
    std::fs::write(run.join("config.toml"), toml::to_string(&cfg).unwrap()).unwrap();
    let config_sha = hash(&run.join("config.toml")).unwrap();
    write(
        &run.join("manifest.json"),
        &json!({"scope":MIXED_SCOPE,"updates":UPDATES}),
    );
    let manifest_sha = hash(&run.join("manifest.json")).unwrap();
    let data = base.join("dev.jsonl");
    std::fs::write(&data, b"reviewed rows\n").unwrap();
    for file in [evidence::checkpoint_file(UPDATES).as_str(), "best.mpk"] {
        std::fs::write(fit.join(file), b"final weights").unwrap();
    }
    let checkpoint = hash(&fit.join("best.mpk")).unwrap();
    let parameters = "b".repeat(64);
    let snapshot = write(
        &fit.join("input_snapshot.json"),
        &json!({"scope":"reviewed-native-authored-full-input-snapshot-v1",
            "identity":{"manifest":{"path":run.join("manifest.json"),"sha256":manifest_sha},
                "native":{"validation":{"config_sha256":config_sha,"code_files":{},
                    "binary_sha256":"c".repeat(64),"input_policy":"known_us","dev_documents":1,
                    "canonical_inputs":{}}}},
            "files":{data.display().to_string():hash(&data).unwrap(),
                run.join("manifest.json").display().to_string():manifest_sha,
                run.join("config.toml").display().to_string():config_sha},
            "tokenizer_contract":tessera::internal::TOKENIZER_CONTRACT,
            "decoder_contract":cfg.decoder_contract(),"postprocess":"address_labeled_fields_v1"}),
    );
    let score = write(
        &fit.join(evidence::score_file(UPDATES)),
        &json!({"scope":MIXED_SCOPE,"step":UPDATES,"parameter_sha256":parameters,
            "development_used_for_selection":false,"separate_frozen_dev_observation":{}}),
    );
    let event = json!({"scope":MIXED_SCOPE,"optimizer_updates_executed":UPDATES,
        "checkpoint_file":evidence::checkpoint_file(UPDATES),"checkpoint_sha256":checkpoint,
        "parameter_sha256":parameters,"native_roundtrip_verified":true,"all_parameters_finite":true,
        "score_file":evidence::score_file(UPDATES),"score_sha256":score});
    let ledger = format!("{}\n{event}\n", json!({"step":0}));
    std::fs::write(fit.join("checkpoint-events.jsonl"), &ledger).unwrap();
    let batches = "{}\n".repeat(UPDATES);
    std::fs::write(fit.join("completed-batches.jsonl"), &batches).unwrap();
    let proof = json!({"passed":true,"parameters_changed":true,
        "checkpoint_sha256":checkpoint,"parameter_sha256":parameters});
    let proof_sha = write(&fit.join("learning-proof.json"), &proof);
    write(
        &fit.join("diagnostic.json"),
        &json!({"diagnostic_scope":MIXED_SCOPE,"preflight_only":false,"export_allowed":false}),
    );
    write(
        &fit.join("manifest.json"),
        &json!({"scope":MIXED_SCOPE,"config":{"path":run.join("config.toml"),"sha256":config_sha},
            "updates":UPDATES,"snapshots":[0,UPDATES],
            "mixed_inputs":{"postprocess":"address_labeled_fields_v1"}}),
    );
    write(
        &fit.join("completion.json"),
        &json!({"scope":MIXED_SCOPE,"optimizer_updates_executed":UPDATES,"fixed_updates":UPDATES,
            "final":{"final_checkpoint":event,"learning_sanity_proof":proof},
            "dev_target_met":false,"learning_sanity_passed":true,
            "checkpoint_ledger_sha256":crate::export::sha256_hex(ledger.as_bytes()),
            "completed_batches_sha256":crate::export::sha256_hex(batches.as_bytes()),
            "input_snapshot_sha256":snapshot,"best_sha256":checkpoint,
            "learning_proof_sha256":proof_sha,"source_inputs_unchanged":true,
            "export_allowed":false,"general_accuracy_claim":false}),
    );
    Fixture {
        _root: root,
        base,
        run,
        fit,
        data,
    }
}

/// A stage copied through the real closure step, up to its quantization gate.
fn copied(fixture: &Fixture) -> (Checked, PathBuf, Closure) {
    let out = fixture.base.join("stage");
    let checked = check(&fixture.run, &out, true).unwrap();
    std::fs::create_dir(&out).unwrap();
    let closure = copy_closure(&checked, &out).unwrap();
    (checked, out, closure)
}

/// Gate synthetic weights through the shared quantization gate with the given DEV F1 values.
fn gate(out: &Path, (f32_f1, int8_f1): (f64, f64)) -> anyhow::Result<Quantization> {
    let pending = out.join("quantized.pending.safetensors");
    crate::quantize::write_safetensors(&pending, &[], &[], None).unwrap();
    let initial = json!({
        "config_sha256": hash(&out.join("config.toml")).unwrap(),
        "best_sha256": hash(&out.join("best.mpk")).unwrap(),
        "input_snapshot_sha256": hash(&out.join("input_snapshot.json")).unwrap(),
    });
    let quantization = Quantization {
        dev_documents: 1,
        f32_macro_exact_f1: f32_f1,
        int8_macro_exact_f1: int8_f1,
        f32_micro_f1: 0.95,
        int8_micro_f1: 0.95,
        max_f1_drop: crate::quantize::MAX_F1_DROP,
    };
    let mut summary = quantization.summary();
    crate::quantize::publish_scored(out, &pending, &initial, (f32_f1, int8_f1), &mut summary)?;
    Ok(quantization)
}

/// A stage built through the real closure and approval steps, with a synthetic passing gate in
/// place of the model and DEV scoring that a synthetic checkpoint cannot provide.
pub(crate) fn staged() -> (Fixture, PathBuf) {
    let fixture = fit();
    let (checked, out, closure) = copied(&fixture);
    let quantization = gate(&out, (0.9, 0.899)).unwrap();
    approve(&checked, &out, closure, quantization).unwrap();
    (fixture, out)
}

/// Rewrite the fit's input snapshot and the completion hash that binds it.
fn rebind_snapshot(fit: &Path, edit: impl FnOnce(&mut Value)) {
    let mut snapshot = evidence::read(&fit.join("input_snapshot.json")).unwrap();
    edit(&mut snapshot);
    let sha = write(&fit.join("input_snapshot.json"), &snapshot);
    let mut completion = evidence::read(&fit.join("completion.json")).unwrap();
    completion["input_snapshot_sha256"] = sha.into();
    write(&fit.join("completion.json"), &completion);
}

/// List one fit-time code identity entry, with `listed` as the snapshot's hash of its path.
fn list_code(fixture: &Fixture, name: &str, identity: &str, listed: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .canonicalize()
        .unwrap()
        .join(name);
    rebind_snapshot(&fixture.fit, |snapshot| {
        snapshot["identity"]["native"]["validation"]["code_files"][name] = identity.into();
        snapshot["files"][path.display().to_string()] = listed.into();
    });
}

/// Change one stage file, require export's verification to refuse it, then restore it.
fn rejects(out: &Path, path: &Path, edit: impl FnOnce(&mut Vec<u8>)) {
    let original = std::fs::read(path).unwrap();
    let mut changed = original.clone();
    edit(&mut changed);
    std::fs::write(path, changed).unwrap();
    assert!(verified_config(out).is_err(), "{}", path.display());
    std::fs::write(path, original).unwrap();
    verified_config(out).unwrap();
}

fn replace(from: &str, to: &str) -> impl FnOnce(&mut Vec<u8>) {
    let (from, to) = (from.to_owned(), to.to_owned());
    move |bytes| {
        let text = String::from_utf8(bytes.clone()).unwrap();
        assert!(text.contains(&from), "{from}");
        *bytes = text.replace(&from, &to).into_bytes();
    }
}

fn refusal(fixture: &Fixture, accepted: bool) -> String {
    let out = fixture.base.join("stage");
    let error = stage(&fixture.run, &out, accepted).unwrap_err().to_string();
    assert!(!out.exists(), "{error}");
    error
}

#[test]
fn release_requires_the_explicit_waiver_flag_and_a_new_destination() {
    let cli = [
        "trainer",
        "reviewed-release",
        "--fit",
        "fit",
        "--out",
        "stage",
    ];
    assert!(crate::Cli::try_parse_from(cli).is_err());
    let mut accepted = cli.to_vec();
    accepted.push("--accept-experimental-quality");
    assert!(crate::Cli::try_parse_from(accepted).is_ok());
    let fixture = fit();
    assert!(refusal(&fixture, false).contains("accept-experimental-quality"));
    let error = stage(&fixture.run, &fixture.base, true)
        .unwrap_err()
        .to_string();
    assert!(error.contains("already exists"), "{error}");
    let inside = fixture.run.join("stage");
    let error = stage(&fixture.run, &inside, true).unwrap_err().to_string();
    assert!(error.contains("outside the fit"), "{error}");
    assert!(!inside.exists());
}

#[test]
fn release_refuses_a_fit_whose_learning_proof_failed() {
    let fixture = fit();
    check(&fixture.run, &fixture.base.join("stage"), true).unwrap();
    rebind_proof(&fixture.fit, |proof| proof["passed"] = false.into());
    assert!(refusal(&fixture, true).contains("learning gate"));
    let fixture = fit();
    rebind_proof(&fixture.fit, |proof| {
        proof["checkpoint_sha256"] = "d".repeat(64).into()
    });
    assert!(refusal(&fixture, true).contains("learning gate"));
}

#[test]
fn release_refuses_a_changed_or_unexecuted_final_checkpoint() {
    let fixture = fit();
    std::fs::write(
        fixture.fit.join(evidence::checkpoint_file(UPDATES)),
        b"other weights",
    )
    .unwrap();
    assert!(refusal(&fixture, true).contains("fixed final checkpoint differs"));
    let fixture = fit();
    std::fs::write(fixture.fit.join("best.mpk"), b"other weights").unwrap();
    assert!(refusal(&fixture, true).contains("best.mpk differs"));
    let fixture = fit();
    let path = fixture.fit.join("completion.json");
    let mut completion = evidence::read(&path).unwrap();
    completion["optimizer_updates_executed"] = (UPDATES - 1).into();
    write(&path, &completion);
    assert!(refusal(&fixture, true).contains("fixed final update count"));
}

#[test]
fn release_refuses_a_run_manifest_with_other_updates() {
    let fixture = fit();
    write(
        &fixture.run.join("manifest.json"),
        &json!({"scope":MIXED_SCOPE,"updates":UPDATES + 1}),
    );
    assert!(refusal(&fixture, true).contains("receipt changed"));
}

#[test]
fn release_refuses_changed_fit_inputs() {
    let fixture = fit();
    std::fs::write(&fixture.data, b"changed rows\n").unwrap();
    assert!(refusal(&fixture, true).contains("receipt changed"));
}

#[test]
fn approval_records_the_waiver_without_an_accuracy_claim() {
    let (_fixture, out) = staged();
    let approval = verify(&out).unwrap();
    assert!(is_approval(&approval));
    assert_eq!(approval["accepted_experimental_quality"], true);
    assert_eq!(approval["learning_gate_passed"], true);
    assert_eq!(approval["general_accuracy_claim"], false);
    assert_eq!(approval["fit_export_allowed"], false);
    assert_eq!(approval["optimizer_updates_executed"], UPDATES);
    assert_eq!(
        crate::release_candidate::verify(&out).unwrap(),
        Some(approval)
    );
    let mut meta = BTreeMap::new();
    bind_metadata(&mut meta, &verify(&out).unwrap(), &out).unwrap();
    assert_eq!(meta["detector_training_updates"], UPDATES.to_string());
    assert_eq!(meta["detector_input_policy"], "known_us");
    assert_eq!(meta["model_version"], MODEL_VERSION);
    assert_eq!(meta["dev_target_met"], "false");
    assert_eq!(meta["general_accuracy_claim"], "false");
    assert_eq!(meta["evaluation_scope"], EVALUATION_SCOPE);
    assert_eq!(
        meta["release_approval_sha256"],
        hash(&out.join("release.json")).unwrap()
    );
    assert!(!meta.contains_key("training_seen_95_percent_gate_passed"));
}

#[test]
fn approval_rejects_claims_and_failed_quantization() {
    let (_fixture, out) = staged();
    let path = out.join("release.json");
    let original = evidence::read(&path).unwrap();
    for (key, value) in [
        ("general_accuracy_claim", json!(true)),
        ("fit_export_allowed", json!(true)),
        ("learning_gate_passed", json!(false)),
        ("optimizer_updates_executed", json!(UPDATES + 1)),
    ] {
        let mut changed = original.clone();
        changed[key] = value;
        write(&path, &changed);
        assert!(verify(&out).is_err(), "{key}");
    }
    let mut changed = original.clone();
    changed["quantization"]["int8_macro_exact_f1"] = json!(0.8);
    write(&path, &changed);
    assert!(verify(&out).is_err());
    write(&path, &original);
    verify(&out).unwrap();
}

#[test]
fn only_fit_time_sources_are_exempt_from_input_reverification() {
    let fixture = fit();
    list_code(
        &fixture,
        "trainer/src/reviewed_release_deleted_fixture.rs",
        &"e".repeat(64),
        &"e".repeat(64),
    );
    list_code(
        &fixture,
        "trainer/src/main.rs",
        &"f".repeat(64),
        &"f".repeat(64),
    );
    list_code(&fixture, "Cargo.lock", &"f".repeat(64), &"f".repeat(64));
    check(&fixture.run, &fixture.base.join("stage"), true).unwrap();
    for (name, identity, listed) in [
        ("data/reviewed_release_fixture.jsonl", "e", "e"),
        ("trainer/src/reviewed_release_mismatch.rs", "a", "b"),
        ("trainer/src/../../reviewed_release_outside.rs", "e", "e"),
        ("trainer/src/reviewed_release_fixture.txt", "e", "e"),
        ("trainer/build.rs", "e", "e"),
    ] {
        let fixture = fit();
        list_code(&fixture, name, &identity.repeat(64), &listed.repeat(64));
        assert!(
            check(&fixture.run, &fixture.base.join("stage"), true).is_err(),
            "{name}"
        );
    }
}

#[test]
fn a_failed_int8_gate_leaves_no_exportable_stage() {
    let fixture = fit();
    let (_checked, out, _closure) = copied(&fixture);
    let error = gate(&out, (0.9, 0.89)).unwrap_err().to_string();
    assert!(error.contains("over the"), "{error}");
    assert!(!out.join("quantized.safetensors").exists());
    assert!(out.join("quantized.rejected.safetensors").exists());
    assert_eq!(
        evidence::read(&out.join("quantize.json")).unwrap()["passed"],
        false
    );
    assert!(crate::quantize::verify_gate(&out).is_err());
    withdraw(&out).unwrap();
    assert!(out.join("quantize.json").exists() && !out.join("release.json").exists());
    assert!(gate(&out, (0.9, f64::NAN)).is_err());
}

#[test]
fn withdrawing_a_failed_stage_removes_any_approval_and_unreadable_or_passing_gate() {
    let (_fixture, out) = staged();
    withdraw(&out).unwrap();
    assert!(!out.join("release.json").exists() && !out.join("quantize.json").exists());
    std::fs::write(out.join("quantize.json"), b"{not json").unwrap();
    withdraw(&out).unwrap();
    assert!(!out.join("quantize.json").exists());
}

#[test]
fn quantize_refuses_a_reviewed_stage_without_touching_its_gate() {
    let (_fixture, out) = staged();
    let error = crate::quantize::run::<burn::backend::NdArray>(&out, &Default::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("reviewed release stage"), "{error}");
    verified_config(&out).unwrap();
}

#[test]
fn export_verification_rejects_every_tampered_stage_artifact() {
    let (fixture, out) = staged();
    verified_config(&out).unwrap();
    rejects(
        &out,
        &out.join("config.toml"),
        replace("address_labeled_fields_v1", "address_continuation_v1"),
    );
    rejects(
        &out,
        &out.join("config.toml"),
        replace("tab_cells25", "legacy23"),
    );
    rejects(
        &out,
        &out.join(crate::diagnostic_operator::CONTEXT_FILE),
        |bytes| bytes.push(b' '),
    );
    rejects(&out, &out.join("quantize.json"), replace("0.899", "0.898"));
    rejects(&out, &fixture.fit.join("best.mpk"), |bytes| {
        bytes.push(b' ')
    });
    rejects(&out, &fixture.run.join("manifest.json"), |bytes| {
        bytes.push(b' ')
    });
    let release = out.join("release.json");
    for (key, value) in [
        ("dev_target_met", json!(true)),
        ("evaluation_scope", json!("independent heldout evaluation")),
        ("scope", json!("reviewed-fit-general-release-v1")),
        ("input_policy", json!("auto_text")),
    ] {
        rejects(&out, &release, |bytes| {
            let mut approval: Value = serde_json::from_slice(bytes).unwrap();
            approval[key] = value;
            *bytes = serde_json::to_vec_pretty(&approval).unwrap();
        });
    }
    std::fs::write(out.join("extra.txt"), b"extra").unwrap();
    let extra = hash(&out.join("extra.txt")).unwrap();
    rejects(&out, &release, |bytes| {
        let mut approval: Value = serde_json::from_slice(bytes).unwrap();
        approval["staged_files"]["extra.txt"] = extra.into();
        *bytes = serde_json::to_vec_pretty(&approval).unwrap();
    });
    rejects(&out, &release, |bytes| {
        let mut approval: Value = serde_json::from_slice(bytes).unwrap();
        approval["staged_files"]
            .as_object_mut()
            .unwrap()
            .remove("quantize.json");
        *bytes = serde_json::to_vec_pretty(&approval).unwrap();
    });
}

/// A random-weight TabCells25 detector checkpoint staged with one DEV document.
fn model_stage() -> (tempfile::TempDir, Config, crate::reviewed_data::Cohort) {
    use burn::module::Module;
    let dir = tempfile::tempdir().unwrap();
    let mut cfg = config(dir.path());
    cfg.features.hash_buckets = 16;
    let device = Default::default();
    <burn::backend::NdArray as burn::prelude::Backend>::seed(&device, 4127);
    cfg.detector_net_config()
        .init::<burn::backend::NdArray>(&device)
        .save_file(
            dir.path().join("best"),
            &burn::record::NamedMpkFileRecorder::<burn::record::FullPrecisionSettings>::new(),
        )
        .unwrap();
    std::fs::write(dir.path().join("config.toml"), b"config").unwrap();
    std::fs::write(dir.path().join("input_snapshot.json"), b"inputs").unwrap();
    let text = "Grace Liu\n1200 Market Street, Philadelphia, PA 19107\n(215) 555-0142";
    let enc = crate::detector::encode_document_with_feature_contract(
        text,
        &[],
        &cfg.features.to_tessera(),
        crate::detector::DetectorInputPolicy::KnownUs,
        cfg.detector_feature_contract(),
    )
    .unwrap();
    let doc = crate::detector::DetectorDoc {
        text: text.into(),
        enc,
        gold: Vec::new(),
        breaks: crate::detector::breaks_of(text),
    };
    let dev = crate::reviewed_data::Cohort::unreviewed_for_test(vec!["dev-0".into()], vec![doc]);
    (dir, cfg, dev)
}

#[test]
fn quantization_refuses_weights_that_do_not_reproduce_the_recorded_dev_score() {
    let (dir, cfg, dev) = model_stage();
    let device = Default::default();
    let (model, parameters) = crate::span_scores::load_checkpoint(
        &cfg.detector_net_config(),
        &dir.path().join("best.mpk"),
        &device,
    )
    .unwrap();
    let postprocess = cfg.net.detector_postprocess;
    let recorded = crate::cohort_fit::score_with_postprocess(
        &model,
        &dev,
        &device,
        crate::detector::DetectorInputPolicy::KnownUs,
        postprocess.unwrap(),
    )
    .unwrap();
    let scoring = |recorded| quantized::Scoring {
        cfg: &cfg,
        dev: &dev,
        policy: crate::detector::DetectorInputPolicy::KnownUs,
        postprocess,
        recorded,
        parameter_sha256: &parameters,
    };
    let mut changed = recorded.clone();
    changed["filtered"]["documents"] = json!(2);
    let error = quantized::run(dir.path(), &scoring(&changed))
        .unwrap_err()
        .to_string();
    assert!(error.contains("f32 DEV rescoring differs"), "{error}");
    for file in [
        "quantize.json",
        "quantized.safetensors",
        "quantized.pending.safetensors",
    ] {
        assert!(!dir.path().join(file).exists(), "{file}");
    }
    let quantization = quantized::run(dir.path(), &scoring(&recorded)).unwrap();
    assert_eq!(quantization.dev_documents, 1);
    quantization
        .verify(&evidence::read(&dir.path().join("quantize.json")).unwrap())
        .unwrap();
    assert!(dir.path().join("quantized.safetensors").exists());
}
