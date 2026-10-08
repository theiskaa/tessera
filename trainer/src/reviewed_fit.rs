//! Isolated bounded reviewed-native candidate fitting; never quantize, publish or claim release quality.

mod contract;
mod fit;
mod mixed;
mod objective;
pub(crate) use mixed::Postprocess;
mod observations;
mod probe;
mod sampling;
mod snapshot;
mod supervision;

use crate::reviewed_data::Receipt;
use anyhow::{Context, ensure};
use serde_json::json;
use std::io::Write;
use std::path::Path;

/// Scope of a native-only fixed-final fit.
pub(crate) const SCOPE: &str = "reviewed-native-fixed-fit-v1";
/// Scope of a native fixed-final fit mixed with authored TRAIN rows.
pub(crate) const MIXED_SCOPE: &str = "reviewed-native-authored-fixed-fit-v1";

/// Validate all metadata without weights, or execute only the separately pinned fixed recipe.
pub(crate) fn run(
    manifest: &Path,
    out: &Path,
    preflight: bool,
    execution_preflight: Option<Receipt>,
) -> anyhow::Result<()> {
    let mut prepared = contract::Fit::prepare(manifest)?;
    let metadata_snapshot = snapshot::capture(&prepared)?;
    if preflight {
        ensure!(
            execution_preflight.is_none(),
            "metadata mode does not consume an execution receipt"
        );
    } else {
        let receipt = execution_preflight
            .context("explicit separate execution preflight receipt required")?;
        ensure!(
            serde_json::from_slice::<serde_json::Value>(&receipt.bytes()?)?
                == prepared.preflight(&metadata_snapshot),
            "execution preflight differs from fixed recipe, canonical inputs or full source closure"
        );
        prepared.execution_preflight = Some(receipt);
        for (name, value) in [("RAYON_NUM_THREADS", "2"), ("MATMUL_NUM_THREADS", "1")] {
            ensure!(
                std::env::var(name).ok().as_deref() == Some(value),
                "set {name}={value} before fitting"
            );
        }
    }
    if !preflight {
        prepared.require_execution_ready()?;
    }
    let target = prepared.destination(out)?;
    std::fs::create_dir(&target)?;
    write_json(
        &target.join("diagnostic.json"),
        &json!({"diagnostic_scope":prepared.scope(),
        "scope":"bounded reviewed native fixed-final candidate fit; never export",
        "preflight_only":preflight,"export_allowed":false,"general_accuracy_claim":false,"release_quality_claim":false}),
    )?;
    write_json(&target.join("identity.json"), &prepared.identity)?;
    write_json(&target.join("manifest.json"), &prepared.manifest)?;
    let input_snapshot = snapshot::capture(&prepared)?;
    write_json(&target.join("input_snapshot.json"), &input_snapshot)?;
    write_json(&target.join("sampling.json"), &prepared.plan.identity)?;
    write_json(&target.join("batch-plan.json"), &prepared.plan.batches)?;
    write_json(
        &target.join("learning-probe.json"),
        &prepared.probe.identity,
    )?;
    if let Some(mixed) = &prepared.mixed {
        write_json(&target.join("mixed-canonical-inputs.json"), &mixed.record)?;
        write_json(
            &target.join("token-accounting.json"),
            &prepared.identity["mixed_token_accounting"],
        )?;
    }
    if preflight {
        snapshot::verify(&prepared, &target)?;
        write_json(
            &target.join("preflight.json"),
            &prepared.preflight(&metadata_snapshot),
        )?;
        eprintln!("reviewed fit metadata preflight complete; no model, optimizer or forward");
        return Ok(());
    }
    fit::run(&prepared, &target)
}

/// Commit a new private JSON artifact without replacing existing bytes.
pub(super) fn write_json(path: &Path, value: &impl serde::Serialize) -> anyhow::Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)?;
    {
        let mut writer = std::io::BufWriter::new(&mut file);
        serde_json::to_writer_pretty(&mut writer, value)?;
        writer.write_all(b"\n")?;
        writer.flush()?;
    }
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fixed_fit_scope_still_refuses_export_and_output_reuse() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("diagnostic.json");
        write_json(
            &path,
            &json!({"diagnostic_scope":SCOPE,"scope":"never export"}),
        )
        .unwrap();
        assert!(crate::quantize::verify_gate(dir.path()).is_err());
        assert!(write_json(&path, &json!({})).is_err());
    }
}
