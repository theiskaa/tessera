//! Pinned inputs, review evidence, and preflight identities for the cohort diagnostic.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::config::{Config, Task};
use crate::diagnostic_operator::ForwardOperator;

pub(super) use crate::reviewed_data::{Receipt, digest, valid_sha};

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Manifest {
    pub(super) scope: String,
    pub(super) source_run: PathBuf,
    pub(super) config: Receipt,
    pub(super) checkpoint: Receipt,
    pub(super) bundle: Receipt,
    pub(super) expected_parameter_sha256: String,
    pub(super) policy: Receipt,
    pub(super) train: Receipt,
    pub(super) train_review: Receipt,
    pub(super) dev: Receipt,
    pub(super) dev_review: Receipt,
    pub(super) separation_audit: Receipt,
    pub(super) train_documents: usize,
    pub(super) dev_documents: usize,
    pub(super) max_steps: usize,
    pub(super) snapshots: Vec<usize>,
    pub(super) threads: usize,
    pub(super) preflight_receipt: Option<Receipt>,
}

pub(super) struct Prepared {
    pub(super) manifest: Manifest,
    pub(super) config: Config,
    manifest_path: PathBuf,
    manifest_sha256: String,
    bindings: BTreeMap<String, String>,
}

fn hash(path: &Path) -> anyhow::Result<String> {
    Ok(digest(&std::fs::read(path)?))
}

pub(super) fn validate_schedule(
    steps: usize,
    snapshots: &[usize],
    warmup: usize,
) -> anyhow::Result<()> {
    ensure!(
        steps <= 2000
            && steps > warmup
            && snapshots.len() >= 2
            && snapshots.len() <= 5
            && snapshots.first() == Some(&0)
            && snapshots.last() == Some(&steps)
            && snapshots.windows(2).all(|p| p[0] < p[1]),
        "cohort schedule requires a fixed budget above warmup and at most2000, with2–5 ordered snapshots including0/final"
    );
    Ok(())
}

pub(super) fn verify_threads(threads: usize) -> anyhow::Result<()> {
    ensure!(threads == 2, "cohort Rayon thread count must be2");
    for (name, expected) in [("RAYON_NUM_THREADS", 2), ("MATMUL_NUM_THREADS", 1)] {
        ensure!(
            std::env::var(name)
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                == Some(expected),
            "set {name}={expected} before launching the cohort diagnostic"
        );
    }
    Ok(())
}

pub(super) fn prepare(path: &Path) -> anyhow::Result<Prepared> {
    let bytes = std::fs::read(path)?;
    let manifest: Manifest = serde_json::from_slice(&bytes)?;
    ensure!(manifest.scope == super::SCOPE, "unknown cohort fit scope");
    ensure!(
        manifest.source_run.is_absolute() && manifest.threads == 2,
        "cohort source path must be absolute and Rayon threads fixed to2"
    );
    ensure!(
        (100..=200).contains(&manifest.train_documents) && manifest.dev_documents == 100,
        "cohort requires100–200 adjudicated whole TRAIN documents and100 separate DEV documents"
    );
    ensure!(
        manifest.config.path.canonicalize()?
            == manifest.source_run.join("config.toml").canonicalize()?
            && manifest.checkpoint.path.canonicalize()?
                == manifest.source_run.join("best.mpk").canonicalize()?,
        "warm-start must use the original run config and f32 best checkpoint"
    );
    let cfg_bytes = manifest.config.bytes()?;
    manifest.checkpoint.bytes()?;
    manifest.policy.bytes()?;
    let config: Config = toml::from_str(std::str::from_utf8(&cfg_bytes)?)?;
    config.validate()?;
    ensure!(
        config.task == Task::Detector
            && config.context96_rms()
            && config.net.finetune_ngram
            && config.net.dropout > 0.0
            && config.net.dropout < 1.0
            && config.train.batch_size == 32
            && config.train.learning_rate == 1e-4
            && config.train.warmup_steps == 500
            && config.train.diagnostic_schedule_steps == Some(7000)
            && config.train.gradient_clip_norm == Some(1.0)
            && config.seed == 42
            && config
                .detector
                .as_ref()
                .context("missing detector config")?
                .class_weights
                == [1.0, 3.0, 2.0, 3.0, 2.0, 2.0, 1.5],
        "cohort fit requires the current graph's native weighted CE, dropout and pinned schedule"
    );
    validate_schedule(
        manifest.max_steps,
        &manifest.snapshots,
        config.train.warmup_steps,
    )?;
    crate::diagnostic_operator::verify_binding(
        &manifest.source_run,
        &config,
        ForwardOperator::ContextRmsV2,
    )?;
    let bundle: Value = serde_json::from_slice(&manifest.bundle.bytes()?)?;
    let metadata = &bundle["metadata"];
    ensure!(
        valid_sha(&manifest.expected_parameter_sha256)
            && metadata["detector_parameter_sha256"] == manifest.expected_parameter_sha256
            && metadata["detector_best_sha256"] == manifest.checkpoint.sha256
            && metadata["decoder_contract"] == config.decoder_contract()
            && metadata["tokenizer_contract"] == tessera::internal::TOKENIZER_CONTRACT,
        "original f32 checkpoint/parameter/tokenizer/decoder identity differs from the pinned bundle"
    );
    let graph: Value = serde_json::from_str(
        metadata["detector_architecture"]
            .as_str()
            .context("bundle graph missing")?,
    )?;
    ensure!(
        graph["name"] == tessera::internal::CONTEXT96_RMS_NAME
            && graph["dilations"] == json!(config.net.dilations)
            && graph["channels"] == 96
            && graph["kernel"] == 3
            && graph["operator"] == "post-residual-channel-rms-v2"
            && graph["epsilon"] == 1e-5
            && graph["learned_parameters"] == 0
            && graph["window_tokens"] == 2048
            && graph["overlap_tokens"] == 448
            && graph["margin_tokens"] == 96
            && graph["max_entity_tokens"] == 256
            && graph["decoder_contract"] == config.decoder_contract(),
        "bundle current graph or native whole-document context contract differs"
    );
    let features: Value = serde_json::from_str(
        metadata["feature_config"]
            .as_str()
            .context("bundle features missing")?,
    )?;
    ensure!(
        features["ngram_sizes"] == json!(config.features.ngram_sizes)
            && features["hash_buckets"] == config.features.hash_buckets
            && features["hash_seed"] == config.features.hash_seed
            && features["flag_bits"] == crate::dataset::FLAG_BITS,
        "bundle input feature contract differs"
    );
    let bindings = super::identity::source_identity(&manifest.source_run)?;
    ensure!(
        metadata["detector_input_snapshot_sha256"]
            == *bindings
                .get("input_snapshot.json")
                .context("source input snapshot missing")?,
        "bundle original input snapshot identity differs"
    );
    let proof = crate::fullmix_rms::checkpoint_provenance(&manifest.source_run)?;
    ensure!(
        proof["parameter_sha256"] == manifest.expected_parameter_sha256,
        "source checkpoint event parameter identity differs from the released bundle"
    );
    ensure!(
        manifest.train.sha256 != manifest.dev.sha256
            && manifest.train_review.sha256 != manifest.dev_review.sha256,
        "TRAIN and DEV require distinct datasets and review evidence"
    );
    Ok(Prepared {
        manifest,
        config,
        manifest_path: path.canonicalize()?,
        manifest_sha256: digest(&bytes),
        bindings,
    })
}

pub(super) fn acceptance_criteria() -> Value {
    json!({"metric":"filtered exact-span F1", "train_each_neural_kind_minimum":0.99,
        "dev_org_minimum_absolute_gain_from_step0":0.05,
        "dev_org_target_alternative":{"precision_minimum":0.95,"recall_minimum":0.95,
            "requires_f1_no_decline_from_step0":true},
        "dev_person_address_minimum_change_from_step0":0.0,
        "floating_point_gain_comparison_tolerance":1e-12,
        "neural_kinds":["person","org","address"],
        "controls":"current original f32 checkpoint on the same separate DEV cohort at step0",
        "release_quality_claim":false})
}

impl Prepared {
    pub(super) fn identity(&self, cohorts: &super::data::Cohorts) -> anyhow::Result<Value> {
        let mut recipe = serde_json::to_value(&self.manifest)?;
        recipe
            .as_object_mut()
            .context("invalid cohort recipe")?
            .remove("preflight_receipt");
        Ok(json!({
            "recipe_sha256": digest(&serde_json::to_vec(&recipe)?),
            "code_files": super::identity::code_identity()?,
            "binary_sha256": hash(&std::env::current_exe()?)?,
            "source_binding_files": self.bindings,
            "cohorts": cohorts.identity()?,
            "operator": "post-residual-channel-rms-v2",
            "decoder_contract": self.config.decoder_contract(),
            "native_ce_rule_tokens": "O targets included; only padding masked",
            "document_scope": "complete existing native dataset rows; source rows may be excerpts, not complete original webpages",
            "optimizer": "fresh AdamW; weight_decay0.01; no restored optimizer state",
            "checkpoint_selection": "none; fixed observations only",
            "sampling": "seeded shuffle without replacement each epoch; final partial epoch recorded",
            "thread_bounds": {"RAYON_NUM_THREADS":2, "MATMUL_NUM_THREADS":1},
            "maximum_retained_tokens_per_whole_document": 900,
            "acceptance_criteria": acceptance_criteria(),
            "source_parameter_identity": "declared; verified only after f32 checkpoint loading in fit mode",
        }))
    }

    pub(super) fn verify_preflight(&self, identity: &Value) -> anyhow::Result<()> {
        let receipt = self
            .manifest
            .preflight_receipt
            .as_ref()
            .context("fit requires a pinned successful preflight receipt")?;
        let value: Value = serde_json::from_slice(&receipt.bytes()?)?;
        ensure!(
            value["scope"] == super::SCOPE
                && value["structural_checks_passed"] == true
                && value["model_initialized"] == false
                && value["optimizer_initialized"] == false
                && value["parameter_identity_verified"] == false
                && value["identity"] == *identity,
            "cohort preflight recipe, code, binary, bindings or data changed"
        );
        Ok(())
    }

    pub(super) fn verify_inputs(&self) -> anyhow::Result<()> {
        ensure!(
            hash(&self.manifest_path)? == self.manifest_sha256,
            "cohort manifest changed during execution"
        );
        for receipt in [
            &self.manifest.config,
            &self.manifest.checkpoint,
            &self.manifest.bundle,
            &self.manifest.policy,
            &self.manifest.train,
            &self.manifest.train_review,
            &self.manifest.dev,
            &self.manifest.dev_review,
            &self.manifest.separation_audit,
        ] {
            receipt.bytes()?;
        }
        super::readiness::verify_nested_receipts(&self.manifest)?;
        ensure!(
            super::identity::source_identity(&self.manifest.source_run)? == self.bindings,
            "immutable original source artifacts changed during cohort execution"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshots_are_preregistered_bounded_and_include_initial_and_final() {
        validate_schedule(2000, &[0, 500, 1000, 1500, 2000], 500).unwrap();
        for (cap, snapshots) in [
            (2001, vec![0, 2001]),
            (500, vec![0, 500]),
            (1000, vec![500, 1000]),
            (1000, vec![0, 500]),
            (1000, vec![0, 500, 500, 1000]),
            (1000, vec![0, 800, 500, 1000]),
        ] {
            assert!(validate_schedule(cap, &snapshots, 500).is_err());
        }
    }

    #[test]
    fn receipt_rejects_changed_bytes_before_any_model_loader() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("checkpoint.mpk");
        std::fs::write(&path, b"original").unwrap();
        let receipt = Receipt {
            path: path.clone(),
            sha256: digest(b"original"),
        };
        assert!(receipt.bytes().is_ok());
        std::fs::write(path, b"changed").unwrap();
        assert!(receipt.bytes().is_err());
    }
}
