//! Typed recipe binding and complete reviewed native TRAIN/DEV loading.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::source::Source;
use crate::config::{Config, Task};
use crate::detector::DetectorInputPolicy;
use crate::reviewed_data::{Cohorts, Composition, Receipt, digest, valid_sha};

/// Opt-in recipe reference; its presence forbids legacy fitting until dependent gates exist.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Settings {
    /// Pin source/data authority independently from the effective training config bytes.
    pub(crate) recipe: Receipt,
}

impl Settings {
    /// Reject legacy source mixing without choosing any training budget or sampler.
    pub(crate) fn validate(&self, cfg: &Config) -> anyhow::Result<()> {
        let detector = cfg
            .detector
            .as_ref()
            .context("reviewed native mode needs detector settings")?;
        ensure!(
            self.recipe.path.is_absolute()
                && valid_sha(&self.recipe.sha256)
                && cfg.task == Task::Detector
                && cfg.context96_rms()
                && cfg.net.finetune_ngram
                && cfg.net.ngram_from.is_some()
                && cfg.names.is_none()
                && cfg.generate.is_none()
                && cfg.augment.copies == 0
                && cfg.data.source.is_empty()
                && cfg.data.processed.is_empty()
                && cfg.data.manifests.is_empty()
                && cfg.data.coverage_exceptions.is_empty()
                && cfg.data.train_per_country == 0
                && cfg.data.valid_per_country == 0
                && cfg.data.test_per_country == 0
                && detector.silver.is_empty()
                && detector.silver_repeats.is_empty()
                && detector.silver_repeat == 1
                && detector.synthetic_per_epoch.is_none()
                && detector.typed_synthetic.is_none()
                && detector.learning_check.is_none()
                && cfg.train.diagnostic_schedule_steps.is_none()
                && cfg.train.gradient_clip_norm.is_some(),
            "reviewed native mode requires only declared reviewed rows; no shards, silver, generation, augmentation, legacy probe or diagnostic schedule"
        );
        Ok(())
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Recipe {
    scope: String,
    input_policy: DetectorInputPolicy,
    composition: Receipt,
    data_preflight: Option<Receipt>,
    source: Source,
}

fn parse_recipe(bytes: &[u8]) -> anyhow::Result<Recipe> {
    let value: Value = serde_json::from_slice(bytes)?;
    ensure!(
        value.get("data_preflight").is_some(),
        "declare structural preflight or explicit null"
    );
    Ok(serde_json::from_value(value)?)
}

/// Verified effective config and immutable recipe, without any model or optimizer.
pub(crate) struct Prepared {
    pub(super) config: Config,
    pub(super) source_config: Config,
    pub(super) source: Source,
    composition: Composition,
    config_path: PathBuf,
    config_sha256: String,
    recipe: Receipt,
    composition_receipt: Receipt,
    structural_preflight: Option<Receipt>,
    canonical_preflight: Option<Receipt>,
    input_policy: DetectorInputPolicy,
    code: BTreeMap<String, String>,
    binary_sha256: String,
}

/// Complete ordered declared TRAIN/DEV and full canonical feature identities.
pub(crate) struct Loaded {
    pub(super) cohorts: Cohorts,
    pub(super) identity: Value,
}

impl Loaded {
    /// Borrow validated encodings without exposing mutable native membership or features.
    pub(crate) fn cohorts(&self) -> &Cohorts {
        &self.cohorts
    }

    /// Canonical feature identity of the encoded cohorts, as a fit's preflight records it.
    pub(crate) fn canonical_identity(&self) -> &Value {
        &self.identity
    }
}

impl Prepared {
    /// Require a separately saved current-code canonical receipt before initialization.
    pub(crate) fn load(path: &Path, canonical_preflight: Receipt) -> anyhow::Result<Self> {
        let mut prepared = Self::prepare(path)?;
        canonical_preflight.bytes()?;
        prepared.canonical_preflight = Some(canonical_preflight);
        prepared.load_documents()?;
        Ok(prepared)
    }

    /// Re-encode the declared cohorts under current code for a release check, without the
    /// fit-time execution identity; the caller must compare the returned canonical identity.
    pub(crate) fn reencode(path: &Path) -> anyhow::Result<(Self, Loaded)> {
        let prepared = Self::prepare(path)?;
        let loaded = prepared.encode_documents()?;
        Ok((prepared, loaded))
    }

    /// Bind metadata for a new canonical preflight without requiring a circular receipt.
    pub(super) fn prepare(path: &Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let config: Config = toml::from_str(std::str::from_utf8(&bytes)?)?;
        config.validate()?;
        let settings = config
            .reviewed_native
            .as_ref()
            .context("explicit reviewed native recipe missing")?;
        let recipe_bytes = settings.recipe.bytes()?;
        let recipe = parse_recipe(&recipe_bytes)?;
        ensure!(
            recipe.scope == "reviewed-native-training-foundation-v2",
            "unknown reviewed native training scope"
        );
        let composition: Composition = serde_json::from_slice(&recipe.composition.bytes()?)?;
        composition.validate()?;
        ensure!(
            composition.config.sha256 == recipe.source.config.sha256
                && composition.config.path.canonicalize()?
                    == recipe.source.config.path.canonicalize()?
                && composition.checkpoint.sha256 == recipe.source.checkpoint.sha256
                && composition.checkpoint.path.canonicalize()?
                    == recipe.source.checkpoint.path.canonicalize()?,
            "reviewed native composition must bind the verified original source"
        );
        let source_config = recipe.source.verify(&config)?;
        composition.verify_receipts()?;
        let prepared = Self {
            code: crate::cohort_fit::code_identity()?,
            binary_sha256: digest(&std::fs::read(std::env::current_exe()?)?),
            config_path: path.canonicalize()?,
            config_sha256: digest(&bytes),
            recipe: Receipt {
                path: settings.recipe.path.clone(),
                sha256: settings.recipe.sha256.clone(),
            },
            config,
            source_config,
            source: recipe.source,
            composition,
            composition_receipt: recipe.composition,
            structural_preflight: recipe.data_preflight,
            canonical_preflight: None,
            input_policy: recipe.input_policy,
        };
        prepared.verify_inputs()?;
        Ok(prepared)
    }

    /// Read the verified effective config without permitting in-memory recipe changes.
    pub(crate) fn config(&self) -> &Config {
        &self.config
    }

    /// The explicit rule selector used for both encoded features and rule observations.
    pub(crate) fn input_policy(&self) -> DetectorInputPolicy {
        self.input_policy
    }

    /// Exact complete native TRAIN input authorities used by the separate authored loader.
    pub(crate) fn native_train_inputs(&self) -> anyhow::Result<Vec<Receipt>> {
        self.verify_inputs()?;
        Ok(self
            .composition
            .segments
            .iter()
            .map(|segment| segment.train.clone())
            .collect())
    }

    /// Shared unchanged annotation policy authority.
    pub(crate) fn annotation_policy(&self) -> &Receipt {
        &self.composition.policy
    }

    /// Original immutable warm-start directory, also excluded from output destinations.
    pub(crate) fn source_run(&self) -> &Path {
        &self.source.run
    }

    /// Full typed native/source/proof closure plus effective code and executable identities.
    pub(crate) fn dependency_receipts(&self) -> anyhow::Result<Vec<Receipt>> {
        self.verify_inputs()?;
        let mut receipts = vec![
            Receipt {
                path: self.config_path.clone(),
                sha256: self.config_sha256.clone(),
            },
            self.recipe.clone(),
            self.composition_receipt.clone(),
        ];
        receipts.extend(
            [&self.structural_preflight, &self.canonical_preflight]
                .into_iter()
                .flatten()
                .cloned(),
        );
        receipts.extend(self.composition.dependency_receipts()?);
        receipts.extend(self.source.dependency_receipts()?);
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .context("code root missing")?;
        for (name, sha256) in &self.code {
            receipts.push(Receipt {
                path: root.join(name),
                sha256: sha256.clone(),
            });
        }
        receipts.push(Receipt {
            path: std::env::current_exe()?,
            sha256: self.binary_sha256.clone(),
        });
        Ok(receipts)
    }

    /// Recheck all originals, proof chains, source bytes and effective execution identities.
    pub(crate) fn verify_inputs(&self) -> anyhow::Result<()> {
        let bytes = std::fs::read(&self.config_path)?;
        let config: Config = toml::from_str(std::str::from_utf8(&bytes)?)?;
        ensure!(
            digest(&bytes) == self.config_sha256
                && serde_json::to_value(&config)? == serde_json::to_value(&self.config)?,
            "effective reviewed config changed"
        );
        self.recipe.bytes()?;
        self.composition_receipt.bytes()?;
        for receipt in [&self.structural_preflight, &self.canonical_preflight]
            .into_iter()
            .flatten()
        {
            receipt.bytes()?;
        }
        self.composition.verify_receipts()?;
        self.source.verify(&self.config)?;
        ensure!(
            crate::cohort_fit::code_identity()? == self.code
                && digest(&std::fs::read(std::env::current_exe()?)?) == self.binary_sha256,
            "reviewed training code or executable changed"
        );
        Ok(())
    }

    fn verify_structural_preflight(&self, cohorts: &Cohorts) -> anyhow::Result<()> {
        let Some(receipt) = &self.structural_preflight else {
            return Ok(());
        };
        let preflight: Value = serde_json::from_slice(&receipt.bytes()?)?;
        ensure!(
            preflight["scope"] == "reviewed-native-dataset-preflight-v1"
                && preflight["structural_checks_passed"] == true
                && preflight["model_initialized"] == false
                && preflight["optimizer_initialized"] == false
                && preflight["parameter_identity_verified"] == false
                && preflight["fit_allowed"] == false
                && preflight["export_allowed"] == false
                && preflight["release_quality_claim"] == false
                && preflight["manifest_sha256"] == self.composition_receipt.sha256
                && preflight["train_documents"] == self.composition.train_documents
                && preflight["dev_documents"] == self.composition.dev_documents
                && preflight["config"] == serde_json::to_value(&self.composition.config)?
                && preflight["checkpoint"] == serde_json::to_value(&self.composition.checkpoint)?
                && preflight["policy"] == serde_json::to_value(&self.composition.policy)?
                && preflight["segments"] == serde_json::to_value(&self.composition.segments)?
                && preflight["identity"] == cohorts.identity()?,
            "historical structural preflight differs from the complete declared composition"
        );
        Ok(())
    }

    /// Revalidate complete native sources and encode the explicitly selected Text policy.
    pub(super) fn encode_documents(&self) -> anyhow::Result<Loaded> {
        self.verify_inputs()?;
        let (cohorts, identity) = self
            .composition
            .load_with_input_policy(&self.config, self.input_policy)?;
        self.verify_structural_preflight(&cohorts)?;
        self.verify_inputs()?;
        Ok(Loaded { cohorts, identity })
    }

    /// Bind all canonical inputs without embedding the receipt being produced.
    pub(super) fn validation_record(&self, loaded: &Loaded) -> anyhow::Result<Value> {
        ensure!(
            loaded.cohorts.identity()? == loaded.identity["cohorts"],
            "loaded reviewed rows changed"
        );
        Ok(json!({"scope":"reviewed-native-recipe-data-preflight-v2",
            "structural_checks_passed":true,"canonical_features_checked":true,
            "config_sha256":self.config_sha256,"recipe":self.recipe,
            "composition":self.composition_receipt,"data_preflight":self.structural_preflight,
            "source":self.source,"code_files":self.code,"binary_sha256":self.binary_sha256,
            "train_documents":self.composition.train_documents,"dev_documents":self.composition.dev_documents,
            "segments":self.composition.segments,"input_policy":self.input_policy,
            "canonical_inputs":loaded.identity,"format":"Text",
            "model_initialized":false,"optimizer_initialized":false,"parameter_identity_verified":false,
            "fit_integration_complete":false,"fit_allowed":false,
            "export_allowed":false,"release_quality_claim":false}))
    }

    /// Recompute canonical native inputs and require their exact independent preflight identity.
    pub(crate) fn load_documents(&self) -> anyhow::Result<Loaded> {
        let receipt = self
            .canonical_preflight
            .as_ref()
            .context("separate canonical recipe preflight required")?;
        let loaded = self.encode_documents()?;
        let actual: Value = serde_json::from_slice(&receipt.bytes()?)?;
        ensure!(
            actual == self.validation_record(&loaded)?,
            "canonical recipe preflight differs in policy, full features, sources or execution identity"
        );
        self.verify_inputs()?;
        Ok(loaded)
    }

    /// Record reviewed canonical bindings without declaring fitting or export readiness.
    pub(crate) fn identity(&self, loaded: &Loaded) -> anyhow::Result<Value> {
        let receipt = self
            .canonical_preflight
            .as_ref()
            .context("separate canonical recipe preflight required")?;
        let record = self.validation_record(loaded)?;
        ensure!(
            serde_json::from_slice::<Value>(&receipt.bytes()?)? == record,
            "canonical recipe inputs changed after preflight"
        );
        Ok(json!({"validation":record,"canonical_preflight":receipt,
            "training_integration_complete":false,"fit_allowed":false,"export_allowed":false}))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        let mut cfg = crate::config::learning_test_config();
        cfg.net.architecture = Some(tessera::internal::CONTEXT96_RMS_NAME.into());
        cfg.net.hidden = 96;
        cfg.net.dilations = tessera::internal::CONTEXT96_RMS_DILATIONS.to_vec();
        cfg.names = None;
        cfg.generate = None;
        cfg.augment.copies = 0;
        cfg.data.source.clear();
        cfg.data.processed.clear();
        cfg.data.manifests.clear();
        cfg.data.coverage_exceptions.clear();
        cfg.data.train_per_country = 0;
        cfg.data.valid_per_country = 0;
        cfg.data.test_per_country = 0;
        cfg.train.diagnostic_schedule_steps = None;
        cfg.train.gradient_clip_norm = Some(1.0);
        let detector = cfg.detector.as_mut().unwrap();
        detector.silver.clear();
        detector.silver_repeats.clear();
        detector.silver_repeat = 1;
        detector.synthetic_per_epoch = None;
        detector.typed_synthetic = None;
        detector.learning_check = None;
        cfg.reviewed_native = Some(Settings {
            recipe: Receipt {
                path: "/reviewed-recipe.json".into(),
                sha256: "a".repeat(64),
            },
        });
        cfg
    }

    #[test]
    fn explicit_native_mode_cannot_fit_or_mix_legacy_sources() {
        let mut cfg = config();
        cfg.validate().unwrap();
        assert!(super::super::refuse_incomplete(&cfg).is_err());
        cfg.detector
            .as_mut()
            .unwrap()
            .silver
            .push("legacy.jsonl".into());
        assert!(cfg.validate().is_err());
        cfg.detector.as_mut().unwrap().silver.clear();
        cfg.augment.copies = 1;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn default_mode_and_strict_recipe_schema_remain_distinct() {
        let cfg = crate::config::learning_test_config();
        assert!(cfg.reviewed_native.is_none());
        super::super::refuse_incomplete(&cfg).unwrap();
        assert!(
            serde_json::from_value::<Settings>(json!({
                "recipe":{"path":"/r","sha256":"a".repeat(64)},"allow_legacy":true
            }))
            .is_err()
        );
        assert!(
            serde_json::from_value::<DetectorInputPolicy>(json!("native-empty-hint-v1")).is_err()
        );
    }
    #[test]
    fn recipe_requires_canonical_policy_and_explicit_structural_receipt_declaration() {
        let receipt = json!({"path":"/receipt","sha256":"a".repeat(64)});
        let mut value = json!({"scope":"reviewed-native-training-foundation-v2",
            "input_policy":"known_us","composition":receipt,"data_preflight":null,
            "source":{"run":"/original","config":receipt,"checkpoint":receipt,
                "bundle":receipt,"parameter_sha256":"a".repeat(64),"inventory":receipt,
                "ngram_config":receipt,"ngram_input_snapshot":receipt}});
        for policy in ["known_us", "auto_text"] {
            value["input_policy"] = json!(policy);
            parse_recipe(&serde_json::to_vec(&value).unwrap()).unwrap();
        }
        for policy in [json!("native-empty-hint-v1"), Value::Null] {
            value["input_policy"] = policy;
            assert!(parse_recipe(&serde_json::to_vec(&value).unwrap()).is_err());
        }
        value["input_policy"] = json!("known_us");
        value.as_object_mut().unwrap().remove("data_preflight");
        assert!(parse_recipe(&serde_json::to_vec(&value).unwrap()).is_err());
    }
}
