//! Run configuration, read from `configs/*.toml`. Unknown keys are errors.
//!
//! The fields are parsed and validated here; the code that reads them arrives with data
//! preparation and training in Milestone 2, hence the module-wide `dead_code` allowance.

#![allow(dead_code)]

use std::path::Path;

use anyhow::Context;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub name: String,
    pub task: Task,
    pub seed: u64,
    pub data: DataConfig,
    pub features: FeaturesConfig,
    pub net: NetConfig,
    pub train: TrainConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Task {
    Parser,
    Detector,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataConfig {
    pub countries: Vec<String>,
    pub train_per_country: usize,
    pub valid_per_country: usize,
    pub test_per_country: usize,
    pub manifests: String,
    pub processed: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeaturesConfig {
    pub ngram_sizes: Vec<u8>,
    pub hash_buckets: u32,
    pub hash_seed: u64,
    pub ngram_dim: usize,
    pub shape_dim: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NetConfig {
    pub hidden: usize,
    pub kernel: usize,
    pub dilations: Vec<usize>,
    pub dropout: f64,
    pub max_params: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
}

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
    }
}
