//! Bounded warm-start diagnostics on separately reviewed whole-document cohorts.

mod contracts;
mod data;
mod fit;
mod identity;
mod readiness;
mod scoring;

pub(crate) use fit::verify_gradients;
pub(crate) use identity::{code_identity, source_identity};
pub(crate) use scoring::{
    score_authored_with_postprocess, score_with_input_policy, score_with_postprocess,
};

use std::path::Path;

use anyhow::{Context, ensure};
use serde_json::json;

const SCOPE: &str = "reviewed-whole-cohort-context-rms-v2";

/// Check a preregistered cohort without model initialization, or execute its bounded fit.
pub(crate) fn run(manifest: &Path, out: &Path, preflight: bool) -> anyhow::Result<()> {
    let prepared = contracts::prepare(manifest)?;
    let cohorts = data::load(&prepared.manifest, &prepared.config)?;
    prepared.verify_inputs()?;
    let identity = prepared.identity(&cohorts)?;
    if !preflight {
        prepared.verify_preflight(&identity)?;
        contracts::verify_threads(prepared.manifest.threads)?;
    }
    let destination = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .canonicalize()?
        .join(out.file_name().context("cohort output name missing")?);
    ensure!(
        !destination.starts_with(prepared.manifest.source_run.canonicalize()?),
        "cohort output must be outside the immutable original source run"
    );
    ensure!(
        matches!(std::fs::symlink_metadata(out), Err(e) if e.kind() == std::io::ErrorKind::NotFound),
        "cohort output already exists or cannot be inspected"
    );
    std::fs::create_dir(out).context("creating new cohort diagnostic directory")?;
    write_json(
        &out.join("diagnostic.json"),
        &json!({
            "diagnostic_scope": SCOPE,
            "scope": "bounded reviewed whole-cohort fit; never export",
            "release_quality_claim": false,
            "export_allowed": false,
            "general_accuracy_claim": false,
            "preflight_only": preflight,
            "identity": identity,
        }),
    )?;
    write_json(&out.join("manifest.json"), &prepared.manifest)?;
    if preflight {
        write_json(
            &out.join("preflight.json"),
            &json!({
                "scope": SCOPE,
                "structural_checks_passed": true,
                "model_initialized": false,
                "optimizer_initialized": false,
                "parameter_identity_verified": false,
                "identity": identity,
            }),
        )?;
        eprintln!("cohort preflight complete; no model, optimizer or forward initialized");
        return Ok(());
    }
    fit::run(&prepared, &cohorts, out, &identity)
}

fn write_json(path: &Path, value: &impl serde::Serialize) -> anyhow::Result<()> {
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_scope_is_refused_by_the_existing_export_gate() {
        let dir = tempfile::tempdir().unwrap();
        write_json(&dir.path().join("diagnostic.json"), &json!({"diagnostic_scope":SCOPE,
            "scope":"bounded reviewed whole-cohort fit; never export", "release_quality_claim":false})).unwrap();
        assert!(
            crate::quantize::verify_gate(dir.path())
                .unwrap_err()
                .to_string()
                .contains("diagnostics")
        );
        assert!(write_json(&dir.path().join("diagnostic.json"), &json!({})).is_err());
    }
}
