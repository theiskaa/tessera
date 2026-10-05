//! Run configuration, read from `configs/*.toml`. Unknown keys are errors.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub name: String,
    pub task: Task,
    pub seed: u64,
    pub data: DataConfig,
    pub features: FeaturesConfig,
    pub net: NetConfig,
    pub train: TrainConfig,
    #[serde(default)]
    pub augment: AugmentConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub names: Option<NamesConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generate: Option<GenerateConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detector: Option<DetectorConfig>,
    /// Explicit reviewed-only source authority, separate from legacy detector inputs.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) reviewed_native: Option<crate::reviewed_train::Settings>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Task {
    Parser,
    Detector,
}

impl Task {
    /// The network's name: its tensor prefix in the bundle and its golden-vector directory.
    pub fn name(self) -> &'static str {
        match self {
            Task::Parser => "parser",
            Task::Detector => "detector",
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DataConfig {
    /// Name of the source manifest under `manifests`, without `.json`.
    #[serde(default)]
    pub source: String,
    pub countries: Vec<String>,
    pub train_per_country: usize,
    pub valid_per_country: usize,
    pub test_per_country: usize,
    /// Reviewed minimums for a country whose source cannot supply the requested quota.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub coverage_exceptions: BTreeMap<String, CoverageException>,
    pub manifests: String,
    pub processed: String,
    /// Where `prepare` records what it sampled. A learning-curve config points this outside
    /// `manifests` so it does not replace the shipped model's sample manifest.
    #[serde(default = "default_sample_manifest")]
    pub sample_manifest: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CoverageException {
    pub train: usize,
    pub valid: usize,
    pub test: usize,
    pub reason: String,
}

fn default_sample_manifest() -> String {
    "data/manifests/parser-sample.json".to_string()
}

#[derive(Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct FeaturesConfig {
    pub ngram_sizes: Vec<u8>,
    pub hash_buckets: u32,
    pub hash_seed: u64,
    pub ngram_dim: usize,
    pub shape_dim: usize,
}

/// Versioned detector features, independent of country rule selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectorFeatures {
    /// Original detector flag layout.
    Legacy23,
    /// Explicit TAB adjacency in bits 23 and 24.
    TabCells25,
}
impl DetectorFeatures {
    /// Shared runtime feature contract.
    pub fn contract(self) -> tessera::internal::DetectorFeatureContract {
        match self {
            Self::Legacy23 => tessera::internal::DetectorFeatureContract::Legacy23,
            Self::TabCells25 => tessera::internal::DetectorFeatureContract::TabCells25,
        }
    }
}

/// Explicit bundle/runtime ADDRESS refinement; absence retains historical serialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectorPostprocess {
    /// Preserve the graph's original ADDRESS continuation decoder.
    AddressContinuationV1,
    /// Use the separately versioned strict labeled-postal-field refinement.
    AddressLabeledFieldsV1,
}
impl DetectorPostprocess {
    /// Shared runtime contract used by export, never inferred from model weights or labels.
    pub fn contract(self) -> tessera::internal::DetectorPostprocessContract {
        match self {
            Self::AddressContinuationV1 => {
                tessera::internal::DetectorPostprocessContract::AddressContinuationV1
            }
            Self::AddressLabeledFieldsV1 => {
                tessera::internal::DetectorPostprocessContract::AddressLabeledFieldsV1
            }
        }
    }
}
fn present_detector_postprocess<'de, D: serde::Deserializer<'de>>(
    d: D,
) -> Result<Option<DetectorPostprocess>, D::Error> {
    DetectorPostprocess::deserialize(d).map(Some)
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NetConfig {
    /// Opt-in runtime refinement; absent is the previous architecture-specific decoder.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "present_detector_postprocess"
    )]
    pub detector_postprocess: Option<DetectorPostprocess>,
    /// Detector-only versioned flags; absence preserves the original 23-bit graph.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detector_features: Option<DetectorFeatures>,
    /// Explicit versioned deployment graph; absence retains the legacy task architecture.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub architecture: Option<String>,
    pub hidden: usize,
    pub kernel: usize,
    pub dilations: Vec<usize>,
    pub dropout: f64,
    /// Budget for parameters outside the embedding tables.
    pub max_params: usize,
    /// Budget for the n-gram table, in int8 bytes.
    #[serde(default = "default_max_embedding_bytes")]
    pub max_embedding_bytes: usize,
    /// A parser run whose n-gram table initializes this network. The run's feature settings
    /// must equal this config's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ngram_from: Option<String>,
    /// Update the parser-initialized n-gram table during detector training. This makes the
    /// detector keep a separate table in the exported bundle.
    #[serde(default)]
    pub finetune_ngram: bool,
}

fn default_max_embedding_bytes() -> usize {
    4_000_000
}

/// Detector training settings.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DetectorConfig {
    /// Loss weight per label, in `O, B-PERSON, I-PERSON, B-ORG, I-ORG, B-ADDRESS, I-ADDRESS`
    /// order; most positions are `O`, and `B-` labels fix boundaries.
    pub class_weights: Vec<f32>,
    /// Silver-labelled real documents added to the train split, as JSONL with `text`,
    /// `country`, and byte-offset `entities` (see `bench/silver/agent_label.py`).
    #[serde(default)]
    pub silver: Vec<String>,
    /// How many times each silver document is repeated in an epoch.
    #[serde(default = "default_silver_repeat")]
    pub silver_repeat: usize,
    /// Optional repeat count for each silver source, in `silver` order.
    #[serde(default)]
    pub silver_repeats: Vec<usize>,
    /// Number of distinct synthetic documents sampled each epoch from the full train shard.
    #[serde(default)]
    pub synthetic_per_epoch: Option<usize>,
    /// Reviewed authored synthetic documents, accepted only by zero-update data checks.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub typed_synthetic: Option<crate::typed_synthetic::Settings>,
    /// Fixed training-example probe used to stop a run that fails to learn a class.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub learning_check: Option<LearningCheckConfig>,
}

/// A diagnostic check on training examples; its thresholds are not release targets.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct LearningCheckConfig {
    /// Maximum whole training documents in the fixed probe.
    pub documents: usize,
    /// Optimizer updates between probe evaluations.
    pub every_steps: usize,
    /// First update at which every learned kind must meet the recall threshold.
    pub start_step: usize,
    /// Minimum exact-span recall for each of person, organization, and address.
    pub minimum_recall: f64,
}

fn default_silver_repeat() -> usize {
    1
}

/// Name sampling for `trainer names`.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NamesConfig {
    /// ISO 3166-1 alpha-2 codes; the Wikidata QID and label languages are looked up in `names::COUNTRIES`.
    pub countries: Vec<String>,
    /// People kept per country, shared equally between its label languages.
    pub people_per_country: usize,
    /// Organizations kept per country: GLEIF first, then Wikidata for any shortfall.
    pub orgs_per_country: usize,
    /// Cache directory for query results and downloads.
    pub raw: String,
    /// Where derived lookup files such as the legal-form abbreviations are written.
    pub interim: String,
    /// Where `people.parquet` and `orgs.parquet` are written.
    pub out: String,
    /// Seeds the per-country sampling shuffles; the split itself is a hash of the name.
    pub sample_seed: u64,
    /// Inclusive range of birth years queried per country and language.
    pub birth_years: (u32, u32),
}

/// Synthetic detector documents for `trainer generate`. The seed, countries, and output
/// directory are the config's top-level `seed`, `[data] countries`, and `[data] processed`;
/// names come from `[names] out`.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GenerateConfig {
    /// Training documents, from the six training families' seen templates.
    pub train: usize,
    /// Validation documents, from the same templates as training.
    pub valid: usize,
    /// Test documents from seen templates with unseen entities.
    pub test_seen: usize,
    /// Test documents from the held-out templates of the six training families.
    pub test_heldout_templates: usize,
    /// Test documents from the two held-out families.
    pub test_heldout_families: usize,
    /// Directory of the parser shards the addresses come from.
    pub addresses: String,
    /// Reviewed evaluation cases whose entity surfaces are excluded from generated documents.
    pub exclude_gold: String,
    /// Include reviewed US unit-parent pairs in training-only documents.
    #[serde(default)]
    pub source_backed_hierarchy: bool,
    /// Longest document kept, in non-whitespace tokens, so training never needs chunking.
    pub max_tokens: usize,
}

/// Augmented copies per training row.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AugmentConfig {
    pub copies: usize,
}

impl Default for AugmentConfig {
    fn default() -> Self {
        AugmentConfig { copies: 2 }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    /// Fixed learning-rate decay horizon for a bounded detector comparison.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub diagnostic_schedule_steps: Option<usize>,
    /// Maximum per-parameter-tensor gradient norm before each optimizer step.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gradient_clip_norm: Option<f32>,
    #[serde(default = "default_warmup_steps")]
    pub warmup_steps: usize,
    /// Epochs without a better validation F1 before stopping.
    #[serde(default = "default_patience")]
    pub patience: usize,
}

fn default_warmup_steps() -> usize {
    200
}

fn default_patience() -> usize {
    3
}

impl FeaturesConfig {
    /// The library's feature configuration these settings describe.
    pub fn to_tessera(&self) -> tessera::internal::FeatureConfig {
        tessera::internal::FeatureConfig {
            ngram_sizes: self.ngram_sizes.clone(),
            hash_buckets: self.hash_buckets,
            hash_seed: self.hash_seed,
        }
    }
}

impl Config {
    /// Reject settings that would panic during feature extraction or produce a model the
    /// library cannot load. This runs when a config is read, before prepare or training.
    pub fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            !self.data.countries.is_empty(),
            "data.countries must not be empty"
        );
        let mut countries = std::collections::HashSet::new();
        for country in &self.data.countries {
            ensure!(
                country.len() == 2 && country.bytes().all(|b| b.is_ascii_uppercase()),
                "country {country:?} must be a two-letter uppercase code"
            );
            ensure!(countries.insert(country), "duplicate country {country}");
        }
        ensure!(
            self.data.countries == ["US"],
            "this project currently supports only US training data"
        );
        if let Some(names) = &self.names {
            ensure!(
                names.countries == ["US"],
                "name sampling currently supports only US data"
            );
        }
        let sizes = &self.features.ngram_sizes;
        ensure!(
            !sizes.is_empty()
                && sizes.len() <= 8
                && sizes.iter().all(|&n| (1..=8).contains(&n))
                && sizes
                    .iter()
                    .enumerate()
                    .all(|(i, n)| !sizes[..i].contains(n)),
            "features.ngram_sizes must contain distinct lengths from 1 through 8"
        );
        ensure!(
            self.features.hash_buckets > 0,
            "features.hash_buckets must be positive"
        );
        ensure!(
            self.features.ngram_dim == 48 && self.features.shape_dim == 16,
            "the runtime requires features.ngram_dim=48 and shape_dim=16"
        );
        ensure!(
            self.task == Task::Detector || self.net.detector_features.is_none(),
            "parser must not declare detector feature contracts"
        );
        ensure!(
            self.detector_feature_contract()
                == tessera::internal::DetectorFeatureContract::Legacy23
                || self.reviewed_native.is_some(),
            "TabCells25 requires the reviewed canonical native route; legacy training is unsupported"
        );
        ensure!(
            self.net.detector_postprocess.is_none()
                || (self.task == Task::Detector && self.context96_rms()),
            "explicit detector postprocessing requires the Context96 detector graph"
        );
        let dilations: &[usize] = match self.task {
            Task::Parser => &[1, 2, 4, 8],
            Task::Detector if self.context96_rms() => &tessera::internal::CONTEXT96_RMS_DILATIONS,
            Task::Detector => &[1, 2, 4, 8, 16, 1],
        };
        ensure!(
            self.net.architecture.is_none()
                || (self.task == Task::Detector && self.context96_rms() && self.net.hidden == 96),
            "unknown or incompatible named network architecture"
        );
        ensure!(
            self.net.kernel == 3
                && self.net.dilations == dilations
                && (1..=1024).contains(&self.net.hidden),
            "the runtime requires kernel=3, dilations={dilations:?}, and hidden in 1..=1024"
        );
        ensure!(
            self.net.dropout.is_finite() && (0.0..1.0).contains(&self.net.dropout),
            "net.dropout must be finite and in [0, 1)"
        );
        ensure!(
            !self.net.finetune_ngram
                || (self.task == Task::Detector && self.net.ngram_from.is_some()),
            "net.finetune_ngram requires a detector with net.ngram_from"
        );
        ensure!(
            self.train.epochs > 0
                && self.train.batch_size > 0
                && self.train.learning_rate.is_finite()
                && self.train.learning_rate > 0.0,
            "training epochs, batch_size, and learning_rate must be positive"
        );
        ensure!(
            self.train
                .diagnostic_schedule_steps
                .is_none_or(|steps| self.task == Task::Detector && steps > self.train.warmup_steps),
            "train.diagnostic_schedule_steps requires a detector and must exceed warmup"
        );
        ensure!(
            self.train
                .gradient_clip_norm
                .is_none_or(|norm| norm.is_finite() && norm > 0.0),
            "train.gradient_clip_norm must be finite and positive"
        );
        if self.task == Task::Detector {
            let detector = self
                .detector
                .as_ref()
                .context("detector task needs [detector]")?;
            ensure!(
                detector.class_weights.len() == crate::detector::DETECTOR_LABELS
                    && detector
                        .class_weights
                        .iter()
                        .all(|w| w.is_finite() && *w > 0.0),
                "detector.class_weights must have seven finite positive values"
            );
            ensure!(
                detector.silver_repeat > 0,
                "detector.silver_repeat must be positive"
            );
            ensure!(
                detector.silver_repeats.is_empty()
                    || (detector.silver_repeats.len() == detector.silver.len()
                        && detector.silver_repeats.iter().all(|&repeat| repeat > 0)),
                "detector.silver_repeats must give one positive count per silver source"
            );
            ensure!(
                detector.synthetic_per_epoch.is_none_or(|count| count > 0),
                "detector.synthetic_per_epoch must be positive"
            );
            if let Some(typed) = &detector.typed_synthetic {
                typed.validate()?;
            }
            if let Some(check) = &detector.learning_check {
                ensure!(
                    check.documents >= 6
                        && check.every_steps > 0
                        && check.start_step > 0
                        && check.start_step.is_multiple_of(check.every_steps)
                        && check.minimum_recall.is_finite()
                        && (0.0..=1.0).contains(&check.minimum_recall)
                        && check.minimum_recall > 0.0,
                    "detector.learning_check requires at least six documents, positive steps, an aligned start, and minimum_recall in (0, 1]"
                );
            }
        }
        if let Some(reviewed) = &self.reviewed_native {
            reviewed.validate(self)?;
        }
        Ok(())
    }

    /// The parser network these settings describe; script and shape each get half of `shape_dim`.
    pub fn parser_net_config(&self) -> crate::net::TaggerNetConfig {
        self.net_config(crate::dataset::PARSER_LABELS)
    }

    /// Whether this config selects the explicitly deployable seven-block RMS graph.
    pub(crate) fn context96_rms(&self) -> bool {
        self.net.architecture.as_deref() == Some(tessera::internal::CONTEXT96_RMS_NAME)
    }

    /// Decoder contract selected by the versioned architecture.
    pub(crate) fn decoder_contract(&self) -> &'static str {
        if self.context96_rms() {
            tessera::internal::CONTEXT96_DECODER_CONTRACT
        } else {
            tessera::internal::DECODER_CONTRACT
        }
    }

    /// The network this run trains, by its task.
    pub fn tagger_net_config(&self) -> crate::net::TaggerNetConfig {
        match self.task {
            Task::Parser => self.parser_net_config(),
            Task::Detector => self.detector_net_config(),
        }
    }

    /// The tensor-name prefix of this run's network in the bundle.
    pub fn net_name(&self) -> &'static str {
        self.task.name()
    }

    /// The detector network these settings describe.
    pub fn detector_net_config(&self) -> crate::net::TaggerNetConfig {
        self.net_config(crate::detector::DETECTOR_LABELS)
            .with_flag_bits(self.detector_feature_contract().flag_bits())
    }

    /// Contract explicitly selected by the detector config, defaulting to Legacy23.
    pub fn detector_feature_contract(&self) -> tessera::internal::DetectorFeatureContract {
        self.net
            .detector_features
            .map(DetectorFeatures::contract)
            .unwrap_or_default()
    }

    fn net_config(&self, labels: usize) -> crate::net::TaggerNetConfig {
        crate::net::TaggerNetConfig::new(
            self.features.hash_buckets as usize,
            self.net.dilations.clone(),
            labels,
        )
        .with_ngram_dim(self.features.ngram_dim)
        .with_script_dim(self.features.shape_dim / 2)
        .with_shape_dim(self.features.shape_dim / 2)
        .with_hidden(self.net.hidden)
        .with_kernel(self.net.kernel)
        .with_dropout(self.net.dropout)
    }
}

/// Reads and validates a run config; unknown keys are errors.
pub fn load(path: &Path) -> anyhow::Result<Config> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let config: Config =
        toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    ensure!(
        config.detector_feature_contract() == tessera::internal::DetectorFeatureContract::Legacy23,
        "TabCells25 is supported only by the reviewed canonical recipe route; legacy fit/eval/export/quantize/inspect config loading is unsupported"
    );
    ensure!(
        config.net.detector_postprocess != Some(DetectorPostprocess::AddressLabeledFieldsV1),
        "labeled-field postprocessing requires the explicitly paired reviewed route; legacy fit/eval/export/quantize/inspect config loading is unsupported"
    );
    config
        .validate()
        .with_context(|| format!("validating {}", path.display()))?;
    Ok(config)
}

#[cfg(test)]
pub(crate) fn learning_test_config() -> Config {
    let mut cfg: Config =
        toml::from_str(include_str!("../../configs/detector-shared-v5.toml")).unwrap();
    cfg.name = "detector-learning-test".into();
    cfg.net.finetune_ngram = true;
    cfg.train.epochs = 25;
    cfg.train.warmup_steps = 500;
    cfg.train.patience = 5;
    cfg.detector.as_mut().unwrap().learning_check = Some(LearningCheckConfig {
        documents: 192,
        every_steps: 500,
        start_step: 2000,
        minimum_recall: 0.5,
    });
    cfg.validate().unwrap();
    cfg
}

#[cfg(test)]
mod tests {
    #[test]
    fn legacy_commands_reject_postprocessing_they_do_not_score() {
        let mut cfg = learning_test_config();
        cfg.net.architecture = Some(tessera::internal::CONTEXT96_RMS_NAME.into());
        cfg.net.hidden = 96;
        cfg.net.dilations = tessera::internal::CONTEXT96_RMS_DILATIONS.to_vec();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        for selected in [None, Some(DetectorPostprocess::AddressContinuationV1)] {
            cfg.net.detector_postprocess = selected;
            cfg.validate().unwrap();
            std::fs::write(&path, toml::to_string(&cfg).unwrap()).unwrap();
            assert_eq!(load(&path).unwrap().net.detector_postprocess, selected);
        }
        cfg.net.detector_postprocess = Some(DetectorPostprocess::AddressLabeledFieldsV1);
        cfg.validate().unwrap();
        std::fs::write(&path, toml::to_string(&cfg).unwrap()).unwrap();
        assert!(
            load(&path)
                .unwrap_err()
                .to_string()
                .contains("explicitly paired reviewed route")
        );
    }

    #[test]
    fn tab_contract_defaults_legacy_and_requires_reviewed_detector_route() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut cfg = load(&root.join("configs/detector-small.toml")).unwrap();
        assert_eq!(cfg.net.detector_features, None);
        assert_eq!(cfg.detector_net_config().flag_bits, 23);
        cfg.net.detector_features = Some(DetectorFeatures::TabCells25);
        assert_eq!(cfg.detector_net_config().flag_bits, 25);
        assert_eq!(cfg.parser_net_config().flag_bits, 23);
        assert!(cfg.validate().is_err());
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("tab.toml");
        std::fs::write(&path, toml::to_string(&cfg).unwrap()).unwrap();
        assert!(
            load(&path)
                .unwrap_err()
                .to_string()
                .contains("legacy fit/eval/export/quantize/inspect")
        );
        let invalid: Result<DetectorFeatures, _> = serde_json::from_str("\"anything25\"");
        assert!(invalid.is_err());
        cfg = load(&root.join("configs/parser-small.toml")).unwrap();
        cfg.net.detector_features = Some(DetectorFeatures::Legacy23);
        assert!(cfg.validate().is_err());
    }

    use super::*;

    #[test]
    fn shipped_configs_parse() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let p = load(&root.join("configs/parser-small.toml")).unwrap();
        assert_eq!(p.task, Task::Parser);
        assert_eq!(p.data.countries, ["US"]);
        assert_eq!(p.net.dilations, vec![1, 2, 4, 8]);
        let d = load(&root.join("configs/detector-small.toml")).unwrap();
        assert_eq!(d.task, Task::Detector);
        assert_eq!(d.data.countries, ["US"]);
        assert_eq!(d.net.dilations, vec![1, 2, 4, 8, 16, 1]);
        assert_eq!(d.names.unwrap().birth_years, (1930, 2005));
        assert_eq!(d.generate.unwrap().max_tokens, 900);
        assert!(p.names.is_none());
        assert!(p.generate.is_none());
    }

    #[test]
    fn a_written_config_reads_back_the_same() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut c = load(&root.join("configs/parser-small.toml")).unwrap();
        c.train.epochs = 3;
        let written = toml::to_string(&c).unwrap();
        let back: Config = toml::from_str(&written).unwrap();
        assert_eq!(back.train.epochs, 3);
        assert_eq!(toml::to_string(&back).unwrap(), written);
    }

    #[test]
    fn ngram_finetuning_requires_a_detector_source() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut detector = load(&root.join("configs/detector-small.toml")).unwrap();
        detector.net.finetune_ngram = true;
        detector.net.ngram_from = None;
        assert!(detector.validate().is_err());
        detector.net.ngram_from = Some("runs/parser-us-v1".into());
        assert!(detector.validate().is_ok());

        let mut parser = load(&root.join("configs/parser-small.toml")).unwrap();
        parser.net.finetune_ngram = true;
        parser.net.ngram_from = Some("runs/parser-us-v1".into());
        assert!(parser.validate().is_err());
    }

    #[test]
    fn invalid_feature_and_runtime_settings_fail_before_training() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut c = load(&root.join("configs/detector-small.toml")).unwrap();
        c.features.hash_buckets = 0;
        assert!(
            c.validate()
                .unwrap_err()
                .to_string()
                .contains("hash_buckets")
        );
        c.features.hash_buckets = 32_768;
        c.features.ngram_sizes = vec![2, 2];
        assert!(
            c.validate()
                .unwrap_err()
                .to_string()
                .contains("ngram_sizes")
        );
        c.features.ngram_sizes = vec![2, 3, 4];
        c.features.shape_dim = 15;
        assert!(c.validate().unwrap_err().to_string().contains("shape_dim"));
        c.features.shape_dim = 16;
        c.net.kernel = 4;
        assert!(c.validate().unwrap_err().to_string().contains("kernel"));
        c.net.kernel = 3;
        c.data.countries.push("US".into());
        assert!(
            c.validate()
                .unwrap_err()
                .to_string()
                .contains("duplicate country")
        );
    }

    #[test]
    fn non_us_training_inputs_fail_before_training() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut c = load(&root.join("configs/detector-small.toml")).unwrap();
        c.data.countries.push("GB".into());
        assert!(c.validate().unwrap_err().to_string().contains("only US"));
        c.data.countries.pop();
        c.names.as_mut().unwrap().countries.push("GB".into());
        assert!(c.validate().unwrap_err().to_string().contains("only US"));
    }
    #[test]
    fn context96_config_is_explicit_and_does_not_relax_legacy_graphs() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/detector-shared.toml");
        let mut cfg = load(&path).unwrap();
        let legacy = toml::to_string(&cfg).unwrap();
        assert!(!legacy.contains("architecture"));
        cfg.net.dilations = tessera::internal::CONTEXT96_RMS_DILATIONS.to_vec();
        assert!(cfg.validate().is_err());
        cfg.net.architecture = Some(tessera::internal::CONTEXT96_RMS_NAME.into());
        cfg.validate().unwrap();
        assert_eq!(
            cfg.detector_net_config().dilations,
            vec![1, 2, 4, 8, 16, 1, 64]
        );
        assert_eq!(
            cfg.decoder_contract(),
            tessera::internal::CONTEXT96_DECODER_CONTRACT
        );
        cfg.net.hidden = 64;
        assert!(cfg.validate().is_err());
        cfg.net.hidden = 96;
        cfg.net.dilations = vec![1, 2, 4, 8, 16, 1];
        assert!(cfg.validate().is_err());
        cfg.net.dilations = tessera::internal::CONTEXT96_RMS_DILATIONS.to_vec();
        cfg.net.architecture = Some("unknown-rms-graph".into());
        assert!(cfg.validate().is_err());
        cfg.net.architecture = Some(tessera::internal::CONTEXT96_RMS_NAME.into());
        cfg.task = Task::Parser;
        assert!(cfg.validate().is_err());
    }
    #[test]
    fn postprocess_config_is_opt_in_closed_and_graph_checked() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../configs/detector-shared.toml");
        let mut cfg = load(&path).unwrap();
        let old = toml::to_string(&cfg).unwrap();
        assert!(cfg.net.detector_postprocess.is_none());
        assert!(!old.contains("detector_postprocess"));
        let mut json = serde_json::to_value(&cfg).unwrap();
        json["net"]["detector_postprocess"] = serde_json::Value::Null;
        assert!(serde_json::from_value::<Config>(json).is_err());
        assert!(serde_json::from_str::<DetectorPostprocess>("\"future\"").is_err());
        cfg.net.detector_postprocess = Some(DetectorPostprocess::AddressLabeledFieldsV1);
        assert!(cfg.validate().is_err());
        cfg.net.architecture = Some(tessera::internal::CONTEXT96_RMS_NAME.into());
        cfg.net.dilations = tessera::internal::CONTEXT96_RMS_DILATIONS.to_vec();
        cfg.net.hidden = 96;
        cfg.validate().unwrap();
        let encoded = toml::to_string(&cfg).unwrap();
        assert!(encoded.contains("address_labeled_fields_v1"));
        cfg.task = Task::Parser;
        assert!(cfg.validate().is_err());
    }
}
