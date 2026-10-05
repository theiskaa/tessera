//! Metadata-only canonical recipe validation; no model or optimizer initialization.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, ensure};
use serde::Serialize;
use serde_json::json;

use super::Prepared;

fn destination(out: &Path, source: &Path) -> anyhow::Result<PathBuf> {
    let parent = out
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let target = parent
        .canonicalize()?
        .join(out.file_name().context("recipe output name missing")?);
    ensure!(
        !target.starts_with(source.canonicalize()?),
        "recipe check output must be outside the original checkpoint directory"
    );
    ensure!(
        matches!(std::fs::symlink_metadata(&target), Err(error) if error.kind() == std::io::ErrorKind::NotFound),
        "recipe check output already exists or cannot be inspected"
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

pub(super) fn run(config: &Path, out: &Path) -> anyhow::Result<()> {
    let prepared = Prepared::prepare(config)?;
    let target = destination(out, &prepared.source.run)?;
    let loaded = prepared.encode_documents()?;
    let record = prepared.validation_record(&loaded)?;
    prepared.verify_inputs()?;
    std::fs::create_dir(&target)?;
    write_json(
        &target.join("diagnostic.json"),
        &json!({
            "scope":"data validation only; never export","diagnostic_scope":"reviewed-native-recipe-data-preflight-v2",
            "fit_allowed":false,"export_allowed":false,"release_quality_claim":false
        }),
    )?;
    write_json(&target.join("canonical-inputs.json"), &loaded.identity)?;
    prepared.verify_inputs()?;
    write_json(&target.join("canonical-preflight.json"), &record)?;
    eprintln!("canonical reviewed recipe checked; no model, optimizer or fitting initialized");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_reuse_and_source_output_are_rejected_without_replacement() {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source");
        std::fs::create_dir(&source).unwrap();
        assert!(destination(&source.join("new"), &source).is_err());
        let out = dir.path().join("output");
        destination(&out, &source).unwrap();
        std::fs::create_dir(&out).unwrap();
        let sentinel = out.join("keep.json");
        std::fs::write(&sentinel, b"original\n").unwrap();
        assert!(destination(&out, &source).is_err());
        assert_eq!(std::fs::read(&sentinel).unwrap(), b"original\n");
    }
}
