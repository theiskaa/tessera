//! Compose complete reviewed cohorts while preserving every segment's source proof.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, ensure};
use serde::Serialize;
use serde_json::json;

use super::{Composition, digest};

const SCOPE: &str = "reviewed-native-dataset-preflight-v1";
use crate::config::{Config, Task};

fn destination(out: &Path, checkpoint: &Path) -> anyhow::Result<PathBuf> {
    let parent = out
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let target = parent
        .canonicalize()?
        .join(out.file_name().context("data output name missing")?);
    let source = checkpoint
        .parent()
        .context("checkpoint directory missing")?
        .canonicalize()?;
    ensure!(
        !target.starts_with(source),
        "data preflight output must be outside the immutable source checkpoint directory"
    );
    ensure!(
        matches!(std::fs::symlink_metadata(out), Err(error) if error.kind() == std::io::ErrorKind::NotFound),
        "data preflight output already exists or cannot be inspected"
    );
    Ok(target)
}

fn write_json(path: &Path, value: &impl Serialize) -> anyhow::Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

pub(super) fn run(path: &Path, out: &Path) -> anyhow::Result<()> {
    let bytes = std::fs::read(path)?;
    let manifest: Composition = serde_json::from_slice(&bytes)?;
    manifest.validate()?;
    manifest.verify_receipts()?;
    let config: Config = toml::from_str(std::str::from_utf8(&manifest.config.bytes()?)?)?;
    config.validate()?;
    ensure!(
        config.task == Task::Detector,
        "reviewed entity preflight requires detector features"
    );
    let source_code = crate::cohort_fit::code_identity()?;
    let executable_sha256 = digest(&std::fs::read(std::env::current_exe()?)?);
    let cohorts = manifest.load(&config)?;
    let identity = cohorts.identity()?;
    manifest.verify_receipts()?;
    ensure!(
        std::fs::read(path)? == bytes,
        "reviewed data manifest changed during preflight"
    );
    ensure!(
        crate::cohort_fit::code_identity()? == source_code,
        "code changed during reviewed data preflight"
    );
    ensure!(
        digest(&std::fs::read(std::env::current_exe()?)?) == executable_sha256,
        "preflight executable changed"
    );
    let target = destination(out, &manifest.checkpoint.path)?;
    std::fs::create_dir(&target)?;
    write_json(&target.join("manifest.json"), &manifest)?;
    write_json(
        &target.join("diagnostic.json"),
        &json!({"diagnostic_scope":SCOPE,"scope":"data validation only; never export","export_allowed":false,"fit_allowed":false,"release_quality_claim":false}),
    )?;
    write_json(
        &target.join("preflight.json"),
        &json!({
            "scope":SCOPE,"structural_checks_passed":true,"model_initialized":false,"optimizer_initialized":false,
            "parameter_identity_verified":false,"fit_allowed":false,"export_allowed":false,"release_quality_claim":false,
            "manifest_sha256":digest(&bytes),"config":manifest.config,"checkpoint":manifest.checkpoint,"policy":manifest.policy,
            "code_files":source_code,"binary_sha256":executable_sha256,"train_documents":manifest.train_documents,
            "dev_documents":manifest.dev_documents,"segments":manifest.segments,"identity":identity,
        }),
    )?;
    eprintln!("reviewed data preflight complete; no model, optimizer or training initialized");
    Ok(())
}
