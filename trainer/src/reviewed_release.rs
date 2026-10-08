//! `trainer reviewed-release`: stage a completed reviewed fixed-final fit for export under an
//! explicit experimental quality waiver.
//!
//! A reviewed fit records `export_allowed: false`, and that declaration keeps its meaning: this
//! route is a separate opt-in approval, recorded as such. The stage holds the fit's exact final
//! checkpoint, int8 weights gated on the fit's own DEV documents, the fit's own input snapshot,
//! copies of its completion and learning evidence, and `release.json`, which export rechecks in
//! full before it writes any bundle.
//!
//! One exception to reverifying the fit's input snapshot: the trainer and library sources and
//! the workspace `Cargo.toml`/`Cargo.lock` files it lists as fit-time code identity are recorded,
//! not reverified, because a release runs current export code; every other listed file,
//! including the fit's own binary, must still match. See `inputs::verify`.

mod evidence;
mod inputs;
mod quantized;
#[cfg(test)]
pub(crate) mod tests;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::{Config, Task};
use crate::detector::DetectorInputPolicy;
use crate::release_candidate::{copy_file, hash, verify_hashes};
use evidence::Evidence;
use inputs::Inputs;
use quantized::Quantization;

const SCHEMA: &str = "reviewed_fit_release_artifact_v1";
const SCOPE: &str = "reviewed-fit-experimental-release-v1";
const EVALUATION_SCOPE: &str = "reviewed fixed-final fit; DEV never selected updates or checkpoints within the fit, but DEV observations informed the choice among runs and data variants, so DEV scores are not an unbiased estimate; learning gate is a TRAIN sanity check; no fresh unseen evaluation";
/// Model series of a reviewed-fit release bundle.
const MODEL_VERSION: &str = "0.4.0";
const GENERATED: &[&str] = &[
    crate::diagnostic_operator::CONTEXT_FILE,
    "quantized.safetensors",
    "quantize.json",
];

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Approval {
    schema: String,
    scope: String,
    accepted_experimental_quality: bool,
    learning_gate_passed: bool,
    general_accuracy_claim: bool,
    fit_export_allowed: bool,
    dev_target_met: bool,
    fresh_unseen_evaluation_pending: bool,
    evaluation_scope: String,
    fit: PathBuf,
    fit_scope: String,
    input_policy: DetectorInputPolicy,
    config: PathBuf,
    recipe_manifest: PathBuf,
    optimizer_updates_executed: usize,
    checkpoint_sha256: String,
    parameter_sha256: String,
    completion_sha256: String,
    learning_proof_sha256: String,
    manifest_sha256: String,
    input_snapshot_sha256: String,
    verified_input_files: usize,
    training_code_identity_sha256: String,
    training_binary_sha256: String,
    quantization: Quantization,
    staged_files: BTreeMap<PathBuf, String>,
    original_files: BTreeMap<PathBuf, String>,
}

/// A fit that passed every check that needs no model, ready to stage.
struct Checked {
    fit: PathBuf,
    config: PathBuf,
    cfg: Config,
    evidence: Evidence,
    inputs: Inputs,
}

type Closure = (BTreeMap<PathBuf, String>, BTreeMap<PathBuf, String>);

/// Whether release bytes declare a reviewed-fit approval rather than the legacy experimental one.
pub(crate) fn declares(bytes: &[u8]) -> anyhow::Result<bool> {
    let value: Value = serde_json::from_slice(bytes)?;
    Ok(value["schema"] == SCHEMA)
}

/// Whether a run directory declares a reviewed-fit stage; its approval still needs `verify`.
pub(crate) fn staged(run: &Path) -> anyhow::Result<bool> {
    let path = run.join("release.json");
    Ok(path.try_exists()? && declares(&std::fs::read(path)?)?)
}

/// Whether a verified approval is a reviewed-fit approval.
pub(crate) fn is_approval(approval: &Value) -> bool {
    approval["schema"] == SCHEMA
}

/// Parse a reviewed-route config, which the legacy loader refuses for its feature contract and
/// postprocess; only a verified reviewed stage may use this.
fn parse_config(bytes: &[u8], path: &Path) -> anyhow::Result<Config> {
    let cfg: Config = toml::from_str(std::str::from_utf8(bytes)?)
        .with_context(|| format!("parsing {}", path.display()))?;
    cfg.validate()
        .with_context(|| format!("validating {}", path.display()))?;
    ensure!(
        cfg.reviewed_native.is_some() && cfg.task == Task::Detector && cfg.context96_rms(),
        "reviewed release requires a reviewed Context96 detector config"
    );
    Ok(cfg)
}

fn load_config(run: &Path) -> anyhow::Result<Config> {
    let path = run.join("config.toml");
    parse_config(
        &std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?,
        &path,
    )
}

/// The fit run's recipe manifest must be the one the fit recorded, declaring the same updates.
fn recipe_manifest(path: &Path, evidence: &Evidence) -> anyhow::Result<()> {
    let recipe = evidence::read(path)?;
    ensure!(
        recipe["updates"] == evidence.updates && recipe["scope"] == evidence.scope.as_str(),
        "fit run manifest declares other updates or scope than the completed fit"
    );
    Ok(())
}

fn check(run: &Path, out: &Path, accepted: bool) -> anyhow::Result<Checked> {
    ensure!(
        accepted,
        "reviewed release requires --accept-experimental-quality"
    );
    ensure!(!out.try_exists()?, "reviewed release output already exists");
    let run = std::fs::canonicalize(run)?;
    let parent = std::fs::canonicalize(
        out.parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new(".")),
    )?;
    ensure!(
        !parent.starts_with(&run),
        "reviewed release output must be outside the fit's run"
    );
    let fit = run.join("fit");
    let evidence = evidence::verify(&evidence::Paths::fit(&fit))?;
    let config = std::fs::canonicalize(&evidence.config.path)?;
    let cfg = parse_config(&evidence.config.bytes()?, &config)?;
    let inputs = inputs::verify(&fit.join("input_snapshot.json"), &cfg, &evidence)?;
    ensure!(
        std::fs::canonicalize(&inputs.recipe_manifest.path)? == run.join("manifest.json"),
        "fit run manifest is not the fit's recorded recipe manifest"
    );
    recipe_manifest(&inputs.recipe_manifest.path, &evidence)?;
    Ok(Checked {
        fit,
        config,
        cfg,
        evidence,
        inputs,
    })
}

/// Staged path and original path of every copied file.
fn pairs(fit: &Path, config: &Path, updates: usize) -> Vec<(PathBuf, PathBuf)> {
    let mut pairs = vec![
        (PathBuf::from("config.toml"), config.to_path_buf()),
        (
            "best.mpk".into(),
            fit.join(evidence::checkpoint_file(updates)),
        ),
        (
            "input_snapshot.json".into(),
            fit.join("input_snapshot.json"),
        ),
    ];
    for file in evidence::FILES
        .iter()
        .map(|file| file.to_string())
        .chain([evidence::score_file(updates)])
    {
        pairs.push((Path::new("source/fit").join(&file), fit.join(&file)));
    }
    pairs
}

fn copy_closure(checked: &Checked, out: &Path) -> anyhow::Result<Closure> {
    let mut staged = BTreeMap::new();
    let mut originals = BTreeMap::new();
    for (relative, original) in pairs(&checked.fit, &checked.config, checked.evidence.updates) {
        copy_file(&original, &relative, out, &mut staged, &mut originals)?;
    }
    for pin in [
        checked.fit.join("best.mpk"),
        checked.inputs.recipe_manifest.path.clone(),
    ] {
        let sha = hash(&pin)?;
        originals.insert(std::fs::canonicalize(pin)?, sha);
    }
    crate::diagnostic_operator::bind_deployable(out, &checked.cfg)?;
    Ok((staged, originals))
}

fn approve(
    checked: &Checked,
    out: &Path,
    (mut staged, originals): Closure,
    quantization: Quantization,
) -> anyhow::Result<()> {
    for file in GENERATED {
        staged.insert(PathBuf::from(file), hash(&out.join(file))?);
    }
    let evidence = &checked.evidence;
    let approval = Approval {
        schema: SCHEMA.into(),
        scope: SCOPE.into(),
        accepted_experimental_quality: true,
        learning_gate_passed: true,
        general_accuracy_claim: false,
        fit_export_allowed: false,
        dev_target_met: evidence.dev_target_met,
        fresh_unseen_evaluation_pending: true,
        evaluation_scope: EVALUATION_SCOPE.into(),
        fit: checked.fit.clone(),
        fit_scope: evidence.scope.clone(),
        input_policy: checked.inputs.policy,
        config: checked.config.clone(),
        recipe_manifest: std::fs::canonicalize(&checked.inputs.recipe_manifest.path)?,
        optimizer_updates_executed: evidence.updates,
        checkpoint_sha256: evidence.checkpoint_sha256.clone(),
        parameter_sha256: evidence.parameter_sha256.clone(),
        completion_sha256: evidence.completion_sha256.clone(),
        learning_proof_sha256: evidence.learning_proof_sha256.clone(),
        manifest_sha256: evidence.manifest_sha256.clone(),
        input_snapshot_sha256: evidence.input_snapshot_sha256.clone(),
        verified_input_files: checked.inputs.verified_files,
        training_code_identity_sha256: checked.inputs.code_identity_sha256.clone(),
        training_binary_sha256: checked.inputs.binary_sha256.clone(),
        quantization,
        staged_files: staged,
        original_files: originals,
    };
    let mut bytes = serde_json::to_vec_pretty(&approval)?;
    bytes.push(b'\n');
    std::fs::write(out.join("release.json"), bytes)?;
    verified_config(out).map(drop)
}

/// Leave no usable approval or passing gate behind a failed stage. A gate that cannot be read
/// is removed too.
fn withdraw(out: &Path) -> anyhow::Result<()> {
    let release = out.join("release.json");
    if release.try_exists()? {
        std::fs::remove_file(release)?;
    }
    let gate = out.join("quantize.json");
    let failed = std::fs::read(&gate)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok())
        .is_some_and(|gate| gate["passed"] == false);
    if !failed {
        crate::quantize::invalidate_gate(out)?;
    }
    Ok(())
}

/// `trainer reviewed-release`: verify the completed fit in `run/fit` whose learning gate passed,
/// quantize its fixed final checkpoint against its own DEV documents, and stage it for
/// `trainer export`.
pub(crate) fn stage(run: &Path, out: &Path, accepted: bool) -> anyhow::Result<()> {
    let checked = check(run, out, accepted)?;
    for (name, value) in [("RAYON_NUM_THREADS", "2"), ("MATMUL_NUM_THREADS", "1")] {
        ensure!(
            std::env::var(name).ok().as_deref() == Some(value),
            "set {name}={value}, the fit's scoring bounds, before staging"
        );
    }
    let loaded = inputs::dev(&checked.config, &checked.inputs)?;
    let scores = evidence::read(
        &checked
            .fit
            .join(evidence::score_file(checked.evidence.updates)),
    )?;
    std::fs::create_dir(out)?;
    let result = copy_closure(&checked, out).and_then(|closure| {
        let quantization = quantized::run(
            out,
            &quantized::Scoring {
                cfg: &checked.cfg,
                dev: &loaded.cohorts().dev,
                policy: checked.inputs.policy,
                postprocess: checked.evidence.postprocess,
                recorded: &scores["separate_frozen_dev_observation"],
                parameter_sha256: &checked.evidence.parameter_sha256,
            },
        )?;
        approve(&checked, out, closure, quantization)
    });
    if let Err(error) = result {
        return Err(match withdraw(out) {
            Ok(()) => error,
            Err(cleanup) => error.context(format!(
                "and withdrawing the failed stage also failed: {cleanup:#}"
            )),
        });
    }
    println!("reviewed release staged: {}", out.display());
    Ok(())
}

fn closure(approval: &Approval) -> anyhow::Result<()> {
    let pairs = pairs(
        &approval.fit,
        &approval.config,
        approval.optimizer_updates_executed,
    );
    let staged: BTreeSet<PathBuf> = pairs
        .iter()
        .map(|(relative, _)| relative.clone())
        .chain(GENERATED.iter().map(PathBuf::from))
        .collect();
    let best = approval.fit.join("best.mpk");
    let originals: BTreeSet<PathBuf> = pairs
        .iter()
        .map(|(_, original)| original.clone())
        .chain([best.clone(), approval.recipe_manifest.clone()])
        .collect();
    ensure!(
        staged == approval.staged_files.keys().cloned().collect()
            && originals == approval.original_files.keys().cloned().collect(),
        "reviewed release closure differs from the required fit artifacts"
    );
    for (relative, original) in &pairs {
        ensure!(
            approval.staged_files.get(relative) == approval.original_files.get(original),
            "staged file is not bound to its named original: {}",
            relative.display()
        );
    }
    ensure!(
        approval.original_files.get(&best) == Some(&approval.checkpoint_sha256),
        "fit best.mpk differs from the approved checkpoint"
    );
    Ok(())
}

fn verified(run: &Path) -> anyhow::Result<(Approval, Evidence)> {
    let approval: Approval = serde_json::from_slice(&std::fs::read(run.join("release.json"))?)?;
    ensure!(
        approval.schema == SCHEMA
            && approval.scope == SCOPE
            && approval.accepted_experimental_quality
            && approval.learning_gate_passed
            && !approval.general_accuracy_claim
            && !approval.fit_export_allowed
            && approval.fresh_unseen_evaluation_pending
            && approval.evaluation_scope == EVALUATION_SCOPE
            && approval.fit.is_absolute()
            && approval.config.is_absolute()
            && approval.recipe_manifest.is_absolute(),
        "invalid reviewed-fit release approval"
    );
    for marker in [
        "diagnostic.json",
        crate::diagnostic_operator::FILE,
        "selection.json",
        "summary.json",
    ] {
        ensure!(
            !run.join(marker).try_exists()?,
            "a reviewed release stage cannot carry diagnostic or training markers"
        );
    }
    verify_hashes(Some(run), &approval.staged_files)?;
    verify_hashes(None, &approval.original_files)?;
    closure(&approval)?;
    let evidence = evidence::verify(&evidence::Paths::staged(run))?;
    ensure!(
        evidence.scope == approval.fit_scope
            && evidence.updates == approval.optimizer_updates_executed
            && evidence.checkpoint_sha256 == approval.checkpoint_sha256
            && evidence.parameter_sha256 == approval.parameter_sha256
            && evidence.dev_target_met == approval.dev_target_met
            && evidence.completion_sha256 == approval.completion_sha256
            && evidence.learning_proof_sha256 == approval.learning_proof_sha256
            && evidence.manifest_sha256 == approval.manifest_sha256
            && evidence.input_snapshot_sha256 == approval.input_snapshot_sha256
            && std::fs::canonicalize(&evidence.config.path)? == approval.config
            && hash(&run.join("config.toml"))? == evidence.config.sha256,
        "reviewed release approval differs from its preserved fit evidence"
    );
    ensure!(
        std::fs::read(run.join(crate::diagnostic_operator::CONTEXT_FILE))?
            == crate::diagnostic_operator::CONTEXT_CANONICAL,
        "reviewed release context operator binding differs"
    );
    recipe_manifest(&approval.recipe_manifest, &evidence)?;
    approval
        .quantization
        .verify(&evidence::read(&run.join("quantize.json"))?)?;
    Ok((approval, evidence))
}

/// Revalidate a reviewed-fit stage: approval flags, staged and original hashes, the exact file
/// closure, the fit's completion and learning evidence, and the passing DEV quantization record.
pub(crate) fn verify(run: &Path) -> anyhow::Result<Value> {
    Ok(serde_json::to_value(verified(run)?.0)?)
}

/// The reviewed stage's replacement for the legacy shard snapshot check and config loader:
/// verify the approval, reverify every input the fit's own snapshot binds, and only then return
/// the reviewed config, which the legacy loader refuses.
pub(crate) fn verified_config(run: &Path) -> anyhow::Result<Config> {
    let (approval, evidence) = verified(run)?;
    let cfg = load_config(run)?;
    let inputs = inputs::verify(&run.join("input_snapshot.json"), &cfg, &evidence)?;
    ensure!(
        inputs.policy == approval.input_policy
            && inputs.verified_files == approval.verified_input_files
            && inputs.code_identity_sha256 == approval.training_code_identity_sha256
            && inputs.binary_sha256 == approval.training_binary_sha256
            && std::fs::canonicalize(&inputs.recipe_manifest.path)? == approval.recipe_manifest,
        "reviewed release inputs differ from the approved fit snapshot"
    );
    Ok(cfg)
}

/// The detector input policy a verified reviewed-fit approval was trained and scored with.
pub(crate) fn input_policy(approval: &Value) -> anyhow::Result<DetectorInputPolicy> {
    Ok(serde_json::from_value(approval["input_policy"].clone())?)
}

/// Bundle provenance of a verified reviewed-fit approval, in place of the legacy experimental
/// fields.
pub(crate) fn bind_metadata(
    meta: &mut BTreeMap<String, String>,
    approval: &Value,
    run: &Path,
) -> anyhow::Result<()> {
    let approval: Approval = serde_json::from_value(approval.clone())?;
    for (key, value) in [
        (
            "detector_training_updates",
            approval.optimizer_updates_executed.to_string(),
        ),
        ("model_version", MODEL_VERSION.into()),
        ("detector_training_complete", "true".into()),
        (
            "detector_input_policy",
            serde_json::to_value(approval.input_policy)?
                .as_str()
                .context("input policy name absent")?
                .into(),
        ),
        ("detector_parameter_sha256", approval.parameter_sha256),
        (
            "experimental_quality_accepted",
            approval.accepted_experimental_quality.to_string(),
        ),
        (
            "learning_gate_passed",
            approval.learning_gate_passed.to_string(),
        ),
        ("dev_target_met", approval.dev_target_met.to_string()),
        (
            "fresh_unseen_evaluation_pending",
            approval.fresh_unseen_evaluation_pending.to_string(),
        ),
        (
            "general_accuracy_claim",
            approval.general_accuracy_claim.to_string(),
        ),
        ("evaluation_scope", approval.evaluation_scope),
        ("release_approval_scope", approval.scope),
        ("release_approval_sha256", hash(&run.join("release.json"))?),
    ] {
        meta.insert(key.into(), value);
    }
    Ok(())
}
