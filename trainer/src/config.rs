//! Run configuration, read from `configs/*.toml`. Unknown keys are errors.

use std::path::Path;

use anyhow::Context;
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
    pub manifests: String,
    pub processed: String,
    /// Where `prepare` records what it sampled. A learning-curve config points this outside
    /// `manifests` so it does not replace the shipped model's sample manifest.
    #[serde(default = "default_sample_manifest")]
    pub sample_manifest: String,
}

fn default_sample_manifest() -> String {
    "data/manifests/parser-sample.json".to_string()
}

#[derive(Debug, Deserialize, Serialize)]
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
    /// The parser network these settings describe; script and shape each get half of `shape_dim`.
    pub fn parser_net_config(&self) -> crate::net::TaggerNetConfig {
        self.net_config(crate::dataset::PARSER_LABELS)
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
    toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipped_configs_parse() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let p = load(&root.join("configs/parser-small.toml")).unwrap();
        assert_eq!(p.task, Task::Parser);
        assert_eq!(p.net.dilations, vec![1, 2, 4, 8]);
        let d = load(&root.join("configs/detector-small.toml")).unwrap();
        assert_eq!(d.task, Task::Detector);
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
}
