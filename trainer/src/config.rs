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

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct NetConfig {
    pub hidden: usize,
    pub kernel: usize,
    pub dilations: Vec<usize>,
    pub dropout: f64,
    /// Budget for parameters outside the embedding tables.
    pub max_params: usize,
    /// Budget for the n-gram table, in int8 bytes.
    #[serde(default = "default_max_embedding_bytes")]
    pub max_embedding_bytes: usize,
    /// A parser run whose n-gram table this network uses, frozen, so the bundle carries the
    /// table once. The run's feature settings must equal this config's.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ngram_from: Option<String>,
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
        let dilations: &[usize] = match self.task {
            Task::Parser => &[1, 2, 4, 8],
            Task::Detector => &[1, 2, 4, 8, 16, 1],
        };
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
            self.train.epochs > 0
                && self.train.batch_size > 0
                && self.train.learning_rate.is_finite()
                && self.train.learning_rate > 0.0,
            "training epochs, batch_size, and learning_rate must be positive"
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
        }
        Ok(())
    }

    /// The parser network these settings describe; script and shape each get half of `shape_dim`.
    pub fn parser_net_config(&self) -> crate::net::TaggerNetConfig {
        self.net_config(crate::dataset::PARSER_LABELS)
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
    config
        .validate()
        .with_context(|| format!("validating {}", path.display()))?;
    Ok(config)
}

#[cfg(test)]
mod tests {
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
}
