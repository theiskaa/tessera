//! A completed fixed-final fit: its executed update count, passing learning proof and the hash
//! chain that binds both to one final checkpoint.

use std::path::{Path, PathBuf};

use anyhow::{Context, ensure};
use serde_json::Value;

use crate::config::DetectorPostprocess;
use crate::export::sha256_hex;
use crate::release_candidate::hash;
use crate::reviewed_data::{Receipt, valid_sha};
use crate::reviewed_fit::{MIXED_SCOPE, SCOPE};

/// Fit evidence a stage preserves under `source/fit`, besides the final DEV score file.
pub(super) const FILES: &[&str] = &[
    "completion.json",
    "learning-proof.json",
    "diagnostic.json",
    "manifest.json",
    "checkpoint-events.jsonl",
    "completed-batches.jsonl",
];

/// The final DEV observation written at the fixed final update.
pub(super) fn score_file(updates: usize) -> String {
    format!("scores-step-{updates}.json")
}

/// The native checkpoint written at the fixed final update.
pub(super) fn checkpoint_file(updates: usize) -> String {
    format!("checkpoints/step-{updates}.mpk")
}

/// Where a fit's evidence, final checkpoint and input snapshot are found.
pub(super) struct Paths {
    evidence: PathBuf,
    checkpoint: Option<PathBuf>,
    snapshot: PathBuf,
}

impl Paths {
    /// The original fit directory, whose final checkpoint is also copied to `best.mpk`.
    pub(super) fn fit(fit: &Path) -> Self {
        Self {
            evidence: fit.to_path_buf(),
            checkpoint: None,
            snapshot: fit.join("input_snapshot.json"),
        }
    }

    /// A stage, whose `best.mpk` and `input_snapshot.json` are the fit's exact bytes.
    pub(super) fn staged(run: &Path) -> Self {
        Self {
            evidence: run.join("source/fit"),
            checkpoint: Some(run.join("best.mpk")),
            snapshot: run.join("input_snapshot.json"),
        }
    }
}

/// What a verified fit binds; the approval records it and export recomputes it.
pub(super) struct Evidence {
    pub(super) scope: String,
    pub(super) updates: usize,
    pub(super) checkpoint_sha256: String,
    pub(super) parameter_sha256: String,
    pub(super) dev_target_met: bool,
    pub(super) completion_sha256: String,
    pub(super) learning_proof_sha256: String,
    pub(super) manifest_sha256: String,
    pub(super) input_snapshot_sha256: String,
    pub(super) config: Receipt,
    pub(super) postprocess: Option<DetectorPostprocess>,
}

pub(super) fn read(path: &Path) -> anyhow::Result<Value> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("parsing {}", path.display()))
}

fn last_line(bytes: &[u8]) -> anyhow::Result<Value> {
    let line = bytes
        .split(|&byte| byte == b'\n')
        .rev()
        .find(|line| !line.is_empty())
        .context("checkpoint ledger is empty")?;
    Ok(serde_json::from_slice(line)?)
}

fn lines(bytes: &[u8]) -> usize {
    bytes.iter().filter(|&&byte| byte == b'\n').count()
}

/// Refuse a fit unless it executed its fixed final update count, its learning proof passed for
/// that exact checkpoint, and every completion hash still matches the files it names.
pub(super) fn verify(paths: &Paths) -> anyhow::Result<Evidence> {
    let dir = &paths.evidence;
    let completion = read(&dir.join("completion.json"))?;
    let manifest = read(&dir.join("manifest.json"))?;
    let diagnostic = read(&dir.join("diagnostic.json"))?;
    let proof = read(&dir.join("learning-proof.json"))?;
    let scope = completion["scope"]
        .as_str()
        .context("fit completion has no scope")?
        .to_owned();
    ensure!(
        [SCOPE, MIXED_SCOPE].contains(&scope.as_str())
            && manifest["scope"] == scope.as_str()
            && diagnostic["diagnostic_scope"] == scope.as_str()
            && diagnostic["preflight_only"] == false,
        "not a completed reviewed fixed-final fit"
    );
    let updates = manifest["updates"]
        .as_u64()
        .and_then(|updates| usize::try_from(updates).ok())
        .filter(|&updates| updates > 0)
        .context("fit manifest declares no updates")?;
    let snapshots = manifest["snapshots"]
        .as_array()
        .context("fit manifest declares no snapshots")?;
    ensure!(
        completion["optimizer_updates_executed"] == updates
            && completion["fixed_updates"] == updates
            && snapshots.last() == Some(&Value::from(updates)),
        "fit did not execute its fixed final update count"
    );
    let last = &completion["final"]["final_checkpoint"];
    let checkpoint = paths
        .checkpoint
        .clone()
        .unwrap_or_else(|| dir.join(checkpoint_file(updates)));
    let checkpoint_sha256 = hash(&checkpoint)?;
    let score = dir.join(score_file(updates));
    ensure!(
        last["scope"] == scope.as_str()
            && last["optimizer_updates_executed"] == updates
            && last["checkpoint_file"] == checkpoint_file(updates)
            && last["checkpoint_sha256"] == checkpoint_sha256.as_str()
            && completion["best_sha256"] == checkpoint_sha256.as_str()
            && last["native_roundtrip_verified"] == true
            && last["all_parameters_finite"] == true
            && last["score_file"] == score_file(updates)
            && last["score_sha256"] == hash(&score)?.as_str(),
        "fixed final checkpoint differs from its completion record"
    );
    if paths.checkpoint.is_none() {
        ensure!(
            hash(&dir.join("best.mpk"))? == checkpoint_sha256,
            "fit best.mpk differs from its fixed final checkpoint"
        );
    }
    let parameter_sha256 = last["parameter_sha256"]
        .as_str()
        .filter(|sha| valid_sha(sha))
        .context("fixed final checkpoint has no parameter identity")?
        .to_owned();
    let learning_proof_sha256 = hash(&dir.join("learning-proof.json"))?;
    ensure!(
        proof["passed"] == true
            && proof["parameters_changed"] == true
            && completion["learning_sanity_passed"] == true
            && proof == completion["final"]["learning_sanity_proof"]
            && proof["checkpoint_sha256"] == checkpoint_sha256.as_str()
            && proof["parameter_sha256"] == parameter_sha256.as_str()
            && completion["learning_proof_sha256"] == learning_proof_sha256.as_str(),
        "fit learning gate did not pass for its fixed final checkpoint"
    );
    let input_snapshot_sha256 = hash(&paths.snapshot)?;
    let ledger = std::fs::read(dir.join("checkpoint-events.jsonl"))?;
    let batches = std::fs::read(dir.join("completed-batches.jsonl"))?;
    ensure!(
        completion["input_snapshot_sha256"] == input_snapshot_sha256.as_str()
            && completion["checkpoint_ledger_sha256"] == sha256_hex(&ledger).as_str()
            && ledger.ends_with(b"\n")
            && lines(&ledger) == snapshots.len()
            && last_line(&ledger)? == *last
            && completion["completed_batches_sha256"] == sha256_hex(&batches).as_str()
            && batches.ends_with(b"\n")
            && lines(&batches) == updates
            && completion["source_inputs_unchanged"] == true
            && completion["export_allowed"] == false
            && completion["general_accuracy_claim"] == false
            && diagnostic["export_allowed"] == false,
        "fit completion ledgers or declarations differ"
    );
    let scores = read(&score)?;
    ensure!(
        scores["scope"] == scope.as_str()
            && scores["step"] == updates
            && scores["parameter_sha256"] == parameter_sha256.as_str()
            && scores["development_used_for_selection"] == false
            && scores["separate_frozen_dev_observation"].is_object(),
        "final DEV observation is not bound to the fixed final checkpoint"
    );
    let postprocess: Option<DetectorPostprocess> = manifest
        .get("mixed_inputs")
        .map(|mixed| serde_json::from_value(mixed["postprocess"].clone()))
        .transpose()?;
    ensure!(
        postprocess.is_some() == (scope == MIXED_SCOPE),
        "fit postprocess declaration differs from its scope"
    );
    Ok(Evidence {
        scope,
        updates,
        checkpoint_sha256,
        parameter_sha256,
        dev_target_met: completion["dev_target_met"]
            .as_bool()
            .context("fit completion has no DEV target result")?,
        completion_sha256: hash(&dir.join("completion.json"))?,
        learning_proof_sha256,
        manifest_sha256: hash(&dir.join("manifest.json"))?,
        input_snapshot_sha256,
        config: serde_json::from_value(manifest["config"].clone())?,
        postprocess,
    })
}
