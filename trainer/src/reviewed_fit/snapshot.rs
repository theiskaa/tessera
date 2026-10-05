//! Exact typed native dependency snapshots, distinct from legacy shard input dispatch.

use super::contract::Fit;
use crate::reviewed_data::{Receipt, digest};
use anyhow::{Context, ensure};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::path::Path;

/// Hash all known typed source, native data and proof dependencies.
pub(super) fn capture(fit: &Fit) -> anyhow::Result<Value> {
    let mut files = BTreeMap::new();
    for receipt in fit.receipts()? {
        receipt.bytes()?;
        let path = receipt.path.canonicalize()?.to_string_lossy().into_owned();
        if let Some(previous) = files.insert(path, receipt.sha256.clone()) {
            ensure!(
                previous == receipt.sha256,
                "native snapshot binds conflicting source bytes"
            );
        }
    }
    let mut record = json!({"scope":"reviewed-native-full-input-snapshot-v1","identity":fit.identity,
        "files":files,"tokenizer_contract":tessera::internal::TOKENIZER_CONTRACT,
        "decoder_contract":fit.native.config().decoder_contract(),
        "source_scope":"all typed native/review/separation receipts, detector inventory, parser snapshot inputs, effective recipe/code/binary; historical scope disclosures preserved, not claimed reconstructed"});
    if let Some(settings) = &fit.manifest.mixed_inputs {
        record["scope"] = json!("reviewed-native-authored-full-input-snapshot-v1");
        record["postprocess"] = json!(settings.postprocess);
        record["source_scope"] = json!(
            "complete native authority and separately typed authored TRAIN/review/parent/feature receipts; authored is repeated TRAIN identity, not heldout or native evidence"
        );
    }
    Ok(record)
}

/// Reject changed dependencies or saved declarations before further updates.
pub(super) fn verify(fit: &Fit, out: &Path) -> anyhow::Result<()> {
    let bytes = std::fs::read(out.join("input_snapshot.json"))?;
    let saved: Value = serde_json::from_slice(&bytes)?;
    ensure!(
        saved == capture(fit)?,
        "native input snapshot closure changed"
    );
    let files = saved["files"]
        .as_object()
        .context("native input files missing")?;
    for (path, sha) in files {
        Receipt {
            path: path.into(),
            sha256: sha.as_str().context("input hash missing")?.to_owned(),
        }
        .bytes()?;
    }
    verify_json(&out.join("identity.json"), &fit.identity)?;
    verify_json(&out.join("manifest.json"), &fit.manifest)?;
    verify_json(&out.join("sampling.json"), &fit.plan.identity)?;
    verify_json(&out.join("batch-plan.json"), &fit.plan.batches)?;
    verify_json(&out.join("learning-probe.json"), &fit.probe.identity)?;
    if let Some(mixed) = &fit.mixed {
        verify_json(&out.join("mixed-canonical-inputs.json"), &mixed.record)?;
        verify_json(
            &out.join("token-accounting.json"),
            &fit.identity["mixed_token_accounting"],
        )?;
    }
    Ok(())
}

fn verify_json(path: &Path, expected: &impl serde::Serialize) -> anyhow::Result<()> {
    let mut bytes = serde_json::to_vec_pretty(expected)?;
    bytes.push(b'\n');
    ensure!(
        digest(&std::fs::read(path)?) == digest(&bytes),
        "saved fit declaration changed: {}",
        path.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saved_declarations_reject_changed_values_or_partial_json() {
        let dir = tempfile::tempdir().unwrap();
        let file = dir.path().join("sampling.json");
        let value = json!({"draws":[0,1,1]});
        super::super::write_json(&file, &value).unwrap();
        verify_json(&file, &value).unwrap();
        std::fs::write(&file, b"{\"draws\":[0,1,2]}\n").unwrap();
        assert!(verify_json(&file, &value).is_err());
        std::fs::write(&file, b"{\"draws\":[0").unwrap();
        assert!(verify_json(&file, &value).is_err());
    }
}
