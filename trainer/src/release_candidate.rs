//! Explicit experimental promotion of an immutable bounded checkpoint.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const SCHEMA: &str = "experimental_release_artifact_v1";
const FINAL_FILES: &[&str] = &[
    "best.mpk",
    "config.toml",
    "input_snapshot.json",
    "context-operator.json",
    "diagnostic.json",
    "checkpoint.json",
    "checkpoint-events.jsonl",
    "completed-exposure.json",
    "completed-batches.jsonl",
    "fullmix.json",
    "selection.json",
    "learning_probe.json",
    "cases.jsonl",
];
const SHIPPING_FILES: &[&str] = &[
    "best.mpk",
    "config.toml",
    "input_snapshot.json",
    "context-operator.json",
];

#[derive(Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Pin {
    path: PathBuf,
    sha256: String,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Release {
    schema: String,
    experimental_quality_accepted: bool,
    release_artifact_complete: bool,
    training_complete: bool,
    training_seen_95_percent_gate_passed: bool,
    fresh_unseen_evaluation_pending: bool,
    general_accuracy_claim: bool,
    optimizer_updates_executed: usize,
    checkpoint: Value,
    source_run: PathBuf,
    training_run: PathBuf,
    evaluation: Pin,
    staged_files: BTreeMap<PathBuf, String>,
    original_files: BTreeMap<PathBuf, String>,
}

/// SHA-256 of a file's exact bytes.
pub(crate) fn hash(path: &Path) -> anyhow::Result<String> {
    Ok(crate::export::sha256_hex(
        &std::fs::read(path).with_context(|| format!("reading {}", path.display()))?,
    ))
}

fn read(path: &Path) -> anyhow::Result<Value> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

fn pinned(value: &Value) -> anyhow::Result<Pin> {
    let pin: Pin = serde_json::from_value(value.clone())?;
    ensure!(
        pin.path.is_absolute() && hash(&pin.path)? == pin.sha256,
        "release evidence pin differs"
    );
    Ok(pin)
}

fn final_checkpoint(proof: &Value) -> anyhow::Result<()> {
    ensure!(
        proof["scope"] == crate::fullmix_rms::HARD_SCOPE
            && proof["optimizer_updates_executed"] == 4000
            && proof["checkpoint_file"] == "checkpoints/diagnostic-step-4000.mpk"
            && proof["selection"] == "snapshot"
            && proof["objective_identity_sha256"] == crate::fullmix_hard::objective_identity()?
            && proof["release_quality_claim"] == false,
        "experimental promotion requires the actual final4000 hard snapshot"
    );
    Ok(())
}

fn learning(summary: &Value, last: &Value) -> anyhow::Result<()> {
    ensure!(
        summary["steps"] == 4000
            && summary["training_complete"] == false
            && summary["learning_check_passed"] == true
            && summary["learning_check_scope"] == "final bounded checkpoint"
            && last["step"] == 4000
            && last["diagnostic_scope"] == crate::fullmix_rms::HARD_SCOPE
            && last["passed"] == true
            && last["enforced"] == true,
        "final bounded checkpoint lacks its enforced learning check"
    );
    for group in ["real", "synthetic"] {
        let proof = &last["source_groups"][group];
        ensure!(
            proof["step"] == 4000
                && proof["start_step"] == 4000
                && proof["min_recall"] == 0.5
                && proof["enforced"] == true
                && proof["passed"] == true
                && proof["failed_kinds"] == json!([]),
            "final learning group failed: {group}"
        );
        for kind in ["address", "org", "person"] {
            let result = &proof["per_kind"][kind];
            ensure!(
                result["gold"].as_u64().is_some_and(|count| count > 0)
                    && result["exact"]["recall"]
                        .as_f64()
                        .is_some_and(|recall| (0.5..=1.0).contains(&recall)),
                "final learning recall failed: {group}/{kind}"
            );
        }
    }
    Ok(())
}

fn evaluation_provenance(
    report: &Value,
    proof: &Value,
    run: &Path,
    gold: &Value,
) -> anyhow::Result<()> {
    ensure!(
        report["release_ready"] == false
            && report["fresh_unseen_release_evaluation_still_required"] == true
            && report["training_seen_95_percent_gate_passed"].is_boolean(),
        "evaluation must retain the unresolved release quality gates"
    );
    for role in ["authored", "real"] {
        let provenance = &report["candidate_provenance"][role];
        let gold = pinned(&gold[role])?;
        ensure!(
            provenance["best_sha256"] == hash(&run.join("best.mpk"))?
                && provenance["config_sha256"] == hash(&run.join("config.toml"))?
                && provenance["gold_sha256"] == gold.sha256
                && provenance["checkpoint"] == *proof
                && provenance["loaded_parameter_sha256"] == proof["parameter_sha256"]
                && provenance["diagnostic_scope"] == crate::fullmix_rms::HARD_SCOPE
                && provenance["preflight"] == false
                && provenance["architecture"] == crate::diagnostic_operator::CONTEXT_NAME
                && provenance["decoder_contract"] == tessera::internal::CONTEXT96_DECODER_CONTRACT
                && provenance["evaluation_role"] == "TRAINING-SEEN final4000 fitting diagnostic"
                && provenance["release_quality_claim"] == false,
            "evaluation is not bound to this final checkpoint: {role}"
        );
        for (field, file) in [
            ("diagnostic_sha256", "diagnostic.json"),
            ("selection_sha256", "selection.json"),
            ("fullmix_manifest_sha256", "fullmix.json"),
            ("operator_sha256", "context-operator.json"),
            ("checkpoint_receipt_sha256", "checkpoint.json"),
        ] {
            ensure!(
                provenance[field] == hash(&run.join(file))?,
                "evaluation source binding differs: {field}"
            );
        }
    }
    Ok(())
}

fn validate_source(run: &Path, training: &Path, evaluation: &Path) -> anyhow::Result<Value> {
    let cfg = crate::config::load(&run.join("config.toml"))?;
    ensure!(
        cfg.context96_rms() && cfg.net_name() == "detector",
        "promotion requires the named context detector"
    );
    crate::train::verify_input_snapshot(run)?;
    let proof = crate::fullmix_rms::checkpoint_provenance(run)?;
    final_checkpoint(&proof)?;
    let manifest = read(&run.join("fullmix.json"))?;
    crate::fullmix_hard::validate_completed(
        &read(&run.join("completed-exposure.json"))?,
        &manifest,
    )?;
    ensure!(
        proof["completed_exposure_sha256"] == hash(&run.join("completed-exposure.json"))?,
        "completed objective differs from snapshot"
    );
    let checkpoint = training.join("checkpoints/diagnostic-step-4000.mpk");
    ensure!(
        hash(&checkpoint)? == hash(&run.join("best.mpk"))?
            && hash(&checkpoint.with_extension("json"))? == hash(&run.join("checkpoint.json"))?,
        "selected release weights differ from native final snapshot"
    );
    for file in FINAL_FILES
        .iter()
        .filter(|&&file| file != "best.mpk" && file != "checkpoint.json")
    {
        ensure!(
            hash(&run.join(file))? == hash(&training.join(file))?,
            "native source differs: {file}"
        );
    }
    ensure!(
        read(
            &training
                .parent()
                .context("training run parent absent")?
                .join("exit.json")
        )?["native_exit_code"]
            == 0,
        "native training did not finish its bounded invocation"
    );
    let metrics = std::fs::read(training.join("learning_metrics.jsonl"))?;
    ensure!(metrics.ends_with(b"\n"), "learning metrics are truncated");
    let last = metrics
        .split(|&byte| byte == b'\n')
        .rev()
        .find(|line| !line.is_empty())
        .context("learning metrics absent")?;
    learning(
        &read(&training.join("summary.json"))?,
        &serde_json::from_slice(last)?,
    )?;
    let plan = pinned(&manifest["typed_diagnostic"]["seen_plan"])?;
    let report = read(evaluation)?;
    evaluation_provenance(&report, &proof, run, &read(&plan.path)?["gold"])?;
    evaluated_files(&report)?;
    Ok(proof)
}

fn evaluated_files(report: &Value) -> anyhow::Result<BTreeMap<String, Pin>> {
    let mut files = BTreeMap::new();
    for evidence in report["evidence"]
        .as_array()
        .context("evaluation evidence absent")?
    {
        let pin = pinned(evidence)?;
        if pin
            .path
            .file_name()
            .is_some_and(|name| name == "confidence.json")
        {
            let confidence = read(&pin.path)?;
            for role in ["authored", "real"] {
                if confidence["provenance"] == report["candidate_provenance"][role] {
                    ensure!(
                        files.insert(role.to_owned(), pin.clone()).is_none(),
                        "duplicate native evaluation role"
                    );
                }
            }
        }
    }
    ensure!(
        files.len() == 2,
        "both native evaluation confidence files are required"
    );
    Ok(files)
}

/// Require every staged (relative to `base`) or original (absolute) file to keep its hash.
pub(crate) fn verify_hashes(
    base: Option<&Path>,
    files: &BTreeMap<PathBuf, String>,
) -> anyhow::Result<()> {
    ensure!(!files.is_empty(), "release artifact hash closure absent");
    for (path, expected) in files {
        let actual_path = if let Some(base) = base {
            ensure!(
                !path.as_os_str().is_empty()
                    && path
                        .components()
                        .all(|part| matches!(part, Component::Normal(_))),
                "invalid staged release path"
            );
            base.join(path)
        } else {
            ensure!(path.is_absolute(), "original release path must be absolute");
            path.clone()
        };
        ensure!(
            hash(&actual_path)? == *expected,
            "release artifact changed: {}",
            actual_path.display()
        );
    }
    Ok(())
}

fn training_files() -> impl Iterator<Item = &'static str> {
    FINAL_FILES
        .iter()
        .filter(|&&file| file != "best.mpk" && file != "checkpoint.json")
        .copied()
        .chain([
            "summary.json",
            "learning_metrics.jsonl",
            "checkpoints/diagnostic-step-4000.mpk",
            "checkpoints/diagnostic-step-4000.json",
        ])
}

fn source_binding(release: &Release, source: &Path, staged: &Path) -> anyhow::Result<()> {
    let source = std::fs::canonicalize(source)?;
    let original = release
        .original_files
        .get(&source)
        .context("original release file binding absent")?;
    ensure!(
        release.staged_files.get(staged) == Some(original),
        "staged file is not bound to its named original"
    );
    Ok(())
}

fn exact_closure(release: &Release, evaluation: &Value) -> anyhow::Result<()> {
    let mut staged = BTreeSet::from([PathBuf::from("summary.json")]);
    let mut originals = BTreeSet::new();
    let mut bind = |source: PathBuf, path: PathBuf| -> anyhow::Result<()> {
        source_binding(release, &source, &path)?;
        originals.insert(std::fs::canonicalize(source)?);
        staged.insert(path);
        Ok(())
    };
    for file in FINAL_FILES {
        bind(
            release.source_run.join(file),
            Path::new("source/final").join(file),
        )?;
    }
    for file in SHIPPING_FILES {
        bind(release.source_run.join(file), file.into())?;
    }
    for file in training_files() {
        bind(
            release.training_run.join(file),
            Path::new("source/training/native").join(file),
        )?;
    }
    bind(
        release
            .training_run
            .parent()
            .context("training parent absent")?
            .join("exit.json"),
        "source/training/exit.json".into(),
    )?;
    bind(
        release.evaluation.path.clone(),
        "source/evaluation.json".into(),
    )?;
    for (role, evidence) in evaluated_files(evaluation)? {
        bind(
            evidence.path,
            Path::new("source").join(format!("evaluation-{role}.json")),
        )?;
    }
    for evidence in evaluation["evidence"]
        .as_array()
        .context("evaluation evidence absent")?
    {
        let pin = pinned(evidence)?;
        let path = std::fs::canonicalize(pin.path)?;
        ensure!(
            release.original_files.get(&path) == Some(&pin.sha256),
            "evaluation evidence absent from release closure"
        );
        originals.insert(path);
    }
    ensure!(
        staged == release.staged_files.keys().cloned().collect()
            && originals == release.original_files.keys().cloned().collect(),
        "release closure keys differ from required source artifacts"
    );
    Ok(())
}

/// Revalidate an experimental or reviewed-fit stage before quantization or export; ordinary
/// runs return None.
pub(crate) fn verify(run: &Path) -> anyhow::Result<Option<Value>> {
    let path = run.join("release.json");
    if !path.try_exists()? {
        return Ok(None);
    }
    let bytes = std::fs::read(path)?;
    if crate::reviewed_release::declares(&bytes)? {
        return crate::reviewed_release::verify(run).map(Some);
    }
    let release: Release = serde_json::from_slice(&bytes)?;
    ensure!(
        release.schema == SCHEMA
            && release.experimental_quality_accepted
            && release.release_artifact_complete
            && !release.training_complete
            && release.fresh_unseen_evaluation_pending
            && !release.general_accuracy_claim
            && release.optimizer_updates_executed == 4000
            && release.source_run.is_absolute()
            && release.training_run.is_absolute()
            && release.evaluation.path.is_absolute()
            && !run.join("diagnostic.json").try_exists()?
            && !run.join(crate::diagnostic_operator::FILE).try_exists()?,
        "invalid experimental release authority"
    );
    verify_hashes(Some(run), &release.staged_files)?;
    verify_hashes(None, &release.original_files)?;
    let source = run.join("source/final");
    let proof = validate_source(
        &source,
        &run.join("source/training/native"),
        &run.join("source/evaluation.json"),
    )?;
    ensure!(
        proof == release.checkpoint,
        "release checkpoint proof differs"
    );
    for file in SHIPPING_FILES {
        ensure!(
            hash(&run.join(file))? == hash(&source.join(file))?,
            "shipping artifact differs from its preserved source"
        );
    }
    let evaluation = read(&run.join("source/evaluation.json"))?;
    exact_closure(&release, &evaluation)?;
    ensure!(
        hash(&release.evaluation.path)? == release.evaluation.sha256
            && hash(&run.join("source/evaluation.json"))? == release.evaluation.sha256
            && release.training_seen_95_percent_gate_passed
                == evaluation["training_seen_95_percent_gate_passed"]
                    .as_bool()
                    .context("quality gate absent")?,
        "experimental quality waiver differs from preserved evaluation"
    );
    let summary = read(&run.join("summary.json"))?;
    ensure!(
        release.staged_files.get(Path::new("summary.json"))
            == Some(&hash(&run.join("summary.json"))?)
            && summary["training_complete"] == false
            && summary["release_artifact_complete"] == true
            && summary["steps"] == 4000
            && summary["experimental_quality_accepted"] == true
            && summary["learning_check_passed"] == true
            && summary["learning_check_scope"] == "final bounded checkpoint"
            && summary["general_accuracy_claim"] == false,
        "release summary misstates bounded training"
    );
    Ok(Some(serde_json::to_value(release)?))
}

/// Copy one original into the stage and record both of its hash bindings.
pub(crate) fn copy_file(
    source: &Path,
    relative: &Path,
    out: &Path,
    staged: &mut BTreeMap<PathBuf, String>,
    originals: &mut BTreeMap<PathBuf, String>,
) -> anyhow::Result<()> {
    let source = std::fs::canonicalize(source)?;
    let destination = out.join(relative);
    std::fs::create_dir_all(
        destination
            .parent()
            .context("stage destination parent absent")?,
    )?;
    std::fs::copy(&source, &destination)?;
    let sha = hash(&source)?;
    ensure!(hash(&destination)? == sha, "copy changed release evidence");
    staged.insert(relative.to_path_buf(), sha.clone());
    originals.insert(source, sha);
    Ok(())
}

/// Preserve original evidence and stage the exact final snapshot after an explicit quality waiver.
pub(crate) fn promote(
    run: &Path,
    training: &Path,
    evaluation: &Path,
    out: &Path,
    accepted: bool,
) -> anyhow::Result<()> {
    ensure!(accepted, "promotion requires --accept-experimental-quality");
    ensure!(!out.try_exists()?, "promotion output already exists");
    let run = std::fs::canonicalize(run)?;
    let training = std::fs::canonicalize(training)?;
    let evaluation = std::fs::canonicalize(evaluation)?;
    let destination = std::fs::canonicalize(
        out.parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new(".")),
    )?
    .join(out.file_name().context("promotion output name absent")?);
    ensure!(
        !destination.starts_with(&run) && !destination.starts_with(&training),
        "promotion output must be outside the original runs"
    );
    let checkpoint = validate_source(&run, &training, &evaluation)?;
    let report = read(&evaluation)?;
    let evaluated = evaluated_files(&report)?;
    let mut staged = BTreeMap::new();
    let mut originals = BTreeMap::new();
    std::fs::create_dir(out)?;
    for file in FINAL_FILES {
        copy_file(
            &run.join(file),
            &Path::new("source/final").join(file),
            out,
            &mut staged,
            &mut originals,
        )?;
    }
    for file in SHIPPING_FILES {
        copy_file(
            &run.join(file),
            Path::new(file),
            out,
            &mut staged,
            &mut originals,
        )?;
    }
    for file in training_files() {
        copy_file(
            &training.join(file),
            &Path::new("source/training/native").join(file),
            out,
            &mut staged,
            &mut originals,
        )?;
    }
    copy_file(
        &training
            .parent()
            .context("training parent absent")?
            .join("exit.json"),
        Path::new("source/training/exit.json"),
        out,
        &mut staged,
        &mut originals,
    )?;
    copy_file(
        &evaluation,
        Path::new("source/evaluation.json"),
        out,
        &mut staged,
        &mut originals,
    )?;
    for (role, evidence) in evaluated {
        copy_file(
            &evidence.path,
            &Path::new("source").join(format!("evaluation-{role}.json")),
            out,
            &mut staged,
            &mut originals,
        )?;
    }
    for evidence in report["evidence"]
        .as_array()
        .context("evaluation evidence absent")?
    {
        let pin = pinned(evidence)?;
        originals.insert(std::fs::canonicalize(pin.path)?, pin.sha256);
    }
    let summary = json!({"training_complete":false,"release_artifact_complete":true,"steps":4000,
        "experimental_quality_accepted":true,"learning_check_passed":true,
        "learning_check_scope":"final bounded checkpoint","general_accuracy_claim":false});
    std::fs::write(
        out.join("summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    staged.insert("summary.json".into(), hash(&out.join("summary.json"))?);
    let release = Release {
        schema: SCHEMA.into(),
        experimental_quality_accepted: true,
        release_artifact_complete: true,
        training_complete: false,
        training_seen_95_percent_gate_passed: report["training_seen_95_percent_gate_passed"]
            .as_bool()
            .context("quality gate absent")?,
        fresh_unseen_evaluation_pending: true,
        general_accuracy_claim: false,
        optimizer_updates_executed: 4000,
        checkpoint,
        source_run: run,
        training_run: training,
        evaluation: Pin {
            sha256: hash(&evaluation)?,
            path: evaluation,
        },
        staged_files: staged,
        original_files: originals,
    };
    std::fs::write(
        out.join("release.json"),
        serde_json::to_vec_pretty(&release)?,
    )?;
    verify(out)?.context("release stage absent")?;
    Ok(())
}

#[cfg(test)]
#[path = "release_candidate_tests.rs"]
mod tests;
