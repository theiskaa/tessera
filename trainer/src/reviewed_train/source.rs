//! Original detector checkpoint and parser-provenance bindings without parameter loading.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::{Config, Task};
use crate::diagnostic_operator::ForwardOperator;
use crate::reviewed_data::{Receipt, valid_sha};

/// Original full f32 detector source; parser receipts record provenance, never overwrite it.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Source {
    pub(super) run: PathBuf,
    pub(super) config: Receipt,
    pub(super) checkpoint: Receipt,
    pub(super) bundle: Receipt,
    pub(super) parameter_sha256: String,
    pub(super) inventory: Receipt,
    pub(super) ngram_config: Receipt,
    pub(super) ngram_input_snapshot: Receipt,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Inventory {
    source_run: PathBuf,
    files: BTreeMap<String, String>,
}

impl Source {
    /// Enumerate original detector inventory and the separately verified parser provenance inputs.
    pub(super) fn dependency_receipts(&self) -> anyhow::Result<Vec<Receipt>> {
        let mut receipts = [
            &self.config,
            &self.checkpoint,
            &self.bundle,
            &self.inventory,
            &self.ngram_config,
            &self.ngram_input_snapshot,
        ]
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
        let inventory: Inventory = serde_json::from_slice(&self.inventory.bytes()?)?;
        for (name, sha256) in inventory.files {
            receipts.push(Receipt {
                path: self.run.join(name),
                sha256,
            });
        }
        let parser_run = self
            .ngram_config
            .path
            .parent()
            .context("parser source missing")?;
        for input in crate::train::load_input_snapshot(parser_run)?
            .inputs
            .into_values()
        {
            receipts.push(parser_input_receipt(input)?);
        }
        Ok(receipts)
    }

    /// Verify full original source closure and target compatibility without loading weights.
    pub(super) fn verify(&self, target: &Config) -> anyhow::Result<Config> {
        ensure!(
            self.run.is_absolute(),
            "reviewed warm-start run must be absolute"
        );
        ensure!(
            self.config.path.canonicalize()? == self.run.join("config.toml").canonicalize()?
                && self.checkpoint.path.canonicalize()?
                    == self.run.join("best.mpk").canonicalize()?,
            "reviewed warm-start requires original config.toml and best.mpk"
        );
        let source: Config = toml::from_str(std::str::from_utf8(&self.config.bytes()?)?)?;
        source.validate()?;
        self.checkpoint.bytes()?;
        let inventory: Inventory = serde_json::from_slice(&self.inventory.bytes()?)?;
        ensure!(
            inventory.source_run.canonicalize()? == self.run.canonicalize()?
                && !inventory.files.is_empty()
                && inventory.files.values().all(|sha| valid_sha(sha))
                && crate::cohort_fit::source_identity(&self.run)? == inventory.files,
            "original detector source inventory differs"
        );
        let ngram_source: Config =
            toml::from_str(std::str::from_utf8(&self.ngram_config.bytes()?)?)?;
        ngram_source.validate()?;
        self.ngram_input_snapshot.bytes()?;
        let from = Path::new(
            source
                .net
                .ngram_from
                .as_deref()
                .context("source ngram provenance missing")?,
        );
        let run = self
            .ngram_config
            .path
            .parent()
            .context("parser source directory missing")?;
        ensure!(
            run.ends_with(from)
                && self.ngram_config.path.canonicalize()?
                    == run.join("config.toml").canonicalize()?
                && self.ngram_input_snapshot.path.canonicalize()?
                    == run.join("input_snapshot.json").canonicalize()?
                && ngram_source.task == Task::Parser
                && ngram_source.features == source.features,
            "original parser ngram provenance differs"
        );
        crate::train::verify_input_snapshot(run)?;
        ensure!(
            source.task == Task::Detector
                && source.context96_rms()
                && source.reviewed_native.is_none()
                && source.detector_feature_contract()
                    == tessera::internal::DetectorFeatureContract::Legacy23
                && target.task == source.task
                && target.seed == source.seed
                && target.features == source.features
                && target.net.architecture == source.net.architecture
                && target.net.hidden == source.net.hidden
                && target.net.kernel == source.net.kernel
                && target.net.dilations == source.net.dilations
                && target.net.dropout == source.net.dropout
                && source.net.dropout > 0.0
                && target.net.ngram_from == source.net.ngram_from
                && target.net.finetune_ngram
                && source.net.finetune_ngram
                && target.train.gradient_clip_norm == source.train.gradient_clip_norm
                && target
                    .detector
                    .as_ref()
                    .context("target detector settings missing")?
                    .class_weights
                    == source
                        .detector
                        .as_ref()
                        .context("source detector settings missing")?
                        .class_weights,
            "reviewed recipe changes original graph/features/dropout/ngram provenance or native CE"
        );
        crate::diagnostic_operator::verify_binding(
            &self.run,
            &source,
            ForwardOperator::ContextRmsV2,
        )?;
        let bundle: Value = serde_json::from_slice(&self.bundle.bytes()?)?;
        let metadata = &bundle["metadata"];
        let graph: Value = serde_json::from_str(
            metadata["detector_architecture"]
                .as_str()
                .context("published detector graph missing")?,
        )?;
        let features: Value = serde_json::from_str(
            metadata["feature_config"]
                .as_str()
                .context("published detector features missing")?,
        )?;
        ensure!(
            graph["name"] == tessera::internal::CONTEXT96_RMS_NAME
                && graph["dilations"] == serde_json::to_value(&source.net.dilations)?
                && graph["channels"] == source.net.hidden
                && graph["kernel"] == source.net.kernel
                && graph["operator"] == "post-residual-channel-rms-v2"
                && graph["epsilon"] == 0.00001
                && graph["learned_parameters"] == 0
                && graph["decoder_contract"] == source.decoder_contract()
                && graph["window_tokens"] == 2048
                && graph["overlap_tokens"] == 448
                && graph["margin_tokens"] == 96
                && graph["max_entity_tokens"] == 256
                && features["ngram_sizes"] == serde_json::to_value(&source.features.ngram_sizes)?
                && features["hash_buckets"] == source.features.hash_buckets
                && features["hash_seed"] == source.features.hash_seed
                && features["flag_bits"] == crate::dataset::FLAG_BITS,
            "published detector graph or input contract differs from original f32 source"
        );
        ensure!(
            valid_sha(&self.parameter_sha256)
                && metadata["detector_parameter_sha256"] == self.parameter_sha256
                && metadata["detector_best_sha256"] == self.checkpoint.sha256
                && metadata["decoder_contract"] == source.decoder_contract()
                && metadata["tokenizer_contract"] == tessera::internal::TOKENIZER_CONTRACT
                && metadata["detector_input_snapshot_sha256"]
                    == *inventory
                        .files
                        .get("input_snapshot.json")
                        .context("source snapshot absent from inventory")?,
            "published original detector parameter/checkpoint/input/decoder binding differs"
        );
        ensure!(
            crate::fullmix_rms::checkpoint_provenance(&self.run)?["parameter_sha256"]
                == self.parameter_sha256,
            "original checkpoint event parameter identity differs"
        );
        Ok(source)
    }
}

fn parser_input_receipt(input: crate::train::InputFile) -> anyhow::Result<Receipt> {
    // Legacy parser records resolve relative paths from the process working directory.
    let path = Path::new(&input.path)
        .canonicalize()
        .with_context(|| format!("resolving original parser input {}", input.path))?;
    Ok(Receipt {
        path,
        sha256: input.sha256,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legacy_relative_parser_receipt_keeps_original_hash_and_file() {
        let cwd = std::env::current_dir().unwrap();
        let dir = tempfile::tempdir_in(&cwd).unwrap();
        let path = dir.path().join("input.parquet");
        std::fs::write(&path, b"original parser bytes").unwrap();
        let sha256 = crate::reviewed_data::digest(b"original parser bytes");
        for name in [
            path.strip_prefix(&cwd)
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            path.to_string_lossy().into_owned(),
        ] {
            let receipt = parser_input_receipt(crate::train::InputFile {
                path: name,
                sha256: sha256.clone(),
            })
            .unwrap();
            assert!(receipt.path.is_absolute());
            assert_eq!(receipt.path, path.canonicalize().unwrap());
            assert_eq!(receipt.sha256, sha256);
            assert_eq!(receipt.bytes().unwrap(), b"original parser bytes");
            std::fs::write(&path, b"changed parser bytes").unwrap();
            assert!(receipt.bytes().is_err());
            std::fs::write(&path, b"original parser bytes").unwrap();
        }
    }
}
