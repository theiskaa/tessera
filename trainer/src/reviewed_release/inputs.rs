//! The fit's own input snapshot, reverified file by file in place of the legacy shard snapshot,
//! and its DEV cohort re-encoded under the code that quantizes it.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, ensure};
use serde::Deserialize;
use serde_json::Value;

use super::evidence::Evidence;
use crate::config::{Config, DetectorPostprocess};
use crate::detector::DetectorInputPolicy;
use crate::reviewed_data::{Receipt, digest};
use crate::reviewed_train::{Loaded, Prepared};

const NATIVE_SCOPE: &str = "reviewed-native-full-input-snapshot-v1";
const MIXED_SCOPE: &str = "reviewed-native-authored-full-input-snapshot-v1";

#[derive(Deserialize)]
struct Snapshot {
    scope: String,
    identity: Identity,
    files: BTreeMap<PathBuf, String>,
    tokenizer_contract: String,
    decoder_contract: String,
    #[serde(default)]
    postprocess: Option<DetectorPostprocess>,
}

#[derive(Deserialize)]
struct Identity {
    manifest: Receipt,
    native: Native,
}

#[derive(Deserialize)]
struct Native {
    validation: Validation,
}

#[derive(Deserialize)]
struct Validation {
    config_sha256: String,
    code_files: BTreeMap<String, String>,
    binary_sha256: String,
    input_policy: DetectorInputPolicy,
    dev_documents: usize,
    canonical_inputs: Value,
}

/// What a reverified snapshot binds.
pub(super) struct Inputs {
    pub(super) recipe_manifest: Receipt,
    pub(super) verified_files: usize,
    pub(super) code_identity_sha256: String,
    pub(super) binary_sha256: String,
    pub(super) policy: DetectorInputPolicy,
    dev_documents: usize,
    canonical_inputs: Value,
}

/// Whether a fit-time code identity name is a source or workspace manifest file: `*.rs` under
/// `trainer/src` or `tessera/src`, or a workspace `Cargo.toml`/`Cargo.lock`.
fn is_source(name: &str) -> bool {
    let path = Path::new(name);
    let normal = path
        .components()
        .all(|part| matches!(part, std::path::Component::Normal(_)));
    let rust = path.extension().is_some_and(|ext| ext == "rs")
        && (path.starts_with("trainer/src") || path.starts_with("tessera/src"));
    let manifest = matches!(
        name,
        "Cargo.toml" | "Cargo.lock" | "trainer/Cargo.toml" | "tessera/Cargo.toml"
    );
    normal && (rust || manifest)
}

/// Reverify every snapshot file except the fit-time code identity: every `code_files` entry that
/// names a trainer or library `*.rs` source or a workspace `Cargo.toml`/`Cargo.lock` and that
/// the snapshot lists with the same hash. Those are the code the fit ran, not its inputs, and a
/// release runs current export code, as legacy export does; they may since have changed or
/// been deleted. Any other `code_files` name, any entry whose hash differs from the snapshot's,
/// and every other listed file, including the fit's binary, data, reviews, recipe and source
/// checkpoints, must still match.
pub(super) fn verify(snapshot: &Path, cfg: &Config, evidence: &Evidence) -> anyhow::Result<Inputs> {
    let bytes =
        std::fs::read(snapshot).with_context(|| format!("reading {}", snapshot.display()))?;
    let saved: Snapshot = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing {}", snapshot.display()))?;
    let validation = &saved.identity.native.validation;
    let manifest = &saved.identity.manifest;
    let config = std::fs::canonicalize(&evidence.config.path)?;
    let scope = match evidence.postprocess {
        Some(_) => MIXED_SCOPE,
        None => NATIVE_SCOPE,
    };
    ensure!(
        saved.scope == scope
            && saved.postprocess == evidence.postprocess
            && cfg.net.detector_postprocess == evidence.postprocess
            && saved.tokenizer_contract == tessera::internal::TOKENIZER_CONTRACT
            && saved.decoder_contract == cfg.decoder_contract()
            && validation.config_sha256 == evidence.config.sha256
            && validation.input_policy == DetectorInputPolicy::KnownUs
            && saved.files.get(&config) == Some(&evidence.config.sha256)
            && saved.files.get(&manifest.path) == Some(&manifest.sha256),
        "fit input snapshot differs from its config, contracts or postprocess"
    );
    let root = std::fs::canonicalize(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .context("code root missing")?,
    )?;
    let sources: BTreeSet<PathBuf> = validation
        .code_files
        .iter()
        .filter(|(name, _)| is_source(name))
        .map(|(name, sha256)| (root.join(name), sha256))
        .filter(|(path, sha256)| saved.files.get(path) == Some(sha256))
        .map(|(path, _)| path)
        .collect();
    let mut verified_files = 0;
    for (path, sha256) in &saved.files {
        if sources.contains(path) {
            continue;
        }
        Receipt {
            path: path.clone(),
            sha256: sha256.clone(),
        }
        .bytes()?;
        verified_files += 1;
    }
    ensure!(verified_files > 0, "fit input snapshot binds no inputs");
    Ok(Inputs {
        recipe_manifest: manifest.clone(),
        verified_files,
        code_identity_sha256: digest(&serde_json::to_vec(&validation.code_files)?),
        binary_sha256: validation.binary_sha256.clone(),
        policy: validation.input_policy,
        dev_documents: validation.dev_documents,
        canonical_inputs: validation.canonical_inputs.clone(),
    })
}

/// Re-encode the fit's reviewed cohorts and require exactly the canonical features it trained
/// and scored on, so the DEV gate below measures the fit's own DEV documents.
pub(super) fn dev(config: &Path, inputs: &Inputs) -> anyhow::Result<Loaded> {
    let (prepared, loaded) = Prepared::reencode(config)?;
    ensure!(
        prepared.input_policy() == inputs.policy
            && loaded.cohorts().dev.docs.len() == inputs.dev_documents
            && *loaded.canonical_identity() == inputs.canonical_inputs,
        "re-encoded reviewed cohorts differ from the fit's canonical features"
    );
    Ok(loaded)
}
