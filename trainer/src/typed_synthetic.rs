//! Reviewed train-only authored synthetic inputs, never a real-silver or fit route.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, ensure};
use polars::prelude::{ParquetReader, SerReader};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tessera::internal::FeatureConfig;

use crate::config::Config;
use crate::detector::{AnnotationDedup, DetectorDoc};

/// One exact reviewed artifact.
#[derive(Debug, Deserialize, Serialize, Clone)]
#[serde(deny_unknown_fields)]
pub(crate) struct Pin {
    pub(crate) path: String,
    pub(crate) sha256: String,
}

impl Pin {
    fn verify(&self) -> anyhow::Result<()> {
        ensure!(
            Path::new(&self.path).is_absolute() && valid_hash(&self.sha256),
            "typed artifact needs an absolute path and lowercase SHA256"
        );
        ensure!(
            hash(Path::new(&self.path))? == self.sha256,
            "typed artifact changed: {}",
            self.path
        );
        Ok(())
    }
}

/// Optional reviewed train-only supplement; absence preserves historical serialization.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Settings {
    pub(crate) manifest: Pin,
    /// Reviewed entities-only real-source replacements, accepted for data checks only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) corrected_real_sources: Option<Pin>,
}

impl Settings {
    /// Validates configuration without permitting any model operation.
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            Path::new(&self.manifest.path).is_absolute() && valid_hash(&self.manifest.sha256),
            "typed manifest needs an absolute path and lowercase SHA256"
        );
        if let Some(pin) = &self.corrected_real_sources {
            ensure!(
                Path::new(&pin.path).is_absolute() && valid_hash(&pin.sha256),
                "corrected real sources need an absolute path and lowercase SHA256"
            );
        }
        Ok(())
    }
}

/// An ordered block of distinct authored documents, not raw duplicate rows.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Family {
    name: String,
    documents: usize,
    repeat: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    origin: String,
    split: String,
    parquet: Pin,
    packet: Pin,
    mapping: Pin,
    native_dedup: Pin,
    native_proof: Pin,
    label_reviews: Vec<Pin>,
    assembly_reviews: Vec<Pin>,
    real_amendments: Pin,
    original_real_sources: Vec<Pin>,
    real_corrections: Pin,
    real_amendment_review: Pin,
    families: Vec<Family>,
}

/// Typed inputs and their native family blocks.
#[derive(Default)]
pub(crate) struct Loaded {
    pub(crate) docs: Vec<DetectorDoc>,
    pub(crate) pieces: Vec<usize>,
    pub(crate) repeats: Vec<usize>,
    pub(crate) metadata: Vec<Value>,
}

fn valid_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn hash(path: &Path) -> anyhow::Result<String> {
    Ok(crate::export::sha256_hex(
        &std::fs::read(path).with_context(|| format!("reading {}", path.display()))?,
    ))
}

fn read(pin: &Pin) -> anyhow::Result<Value> {
    pin.verify()?;
    Ok(serde_json::from_slice(&std::fs::read(&pin.path)?)?)
}

fn file_pin(value: &Value) -> anyhow::Result<Pin> {
    Ok(Pin {
        path: value["path"]
            .as_str()
            .context("artifact path absent")?
            .to_owned(),
        sha256: value["sha256"]
            .as_str()
            .context("artifact hash absent")?
            .to_owned(),
    })
}

#[path = "typed_synthetic_amendment.rs"]
mod amendment;
#[path = "typed_synthetic_archival.rs"]
mod archival;
#[path = "typed_synthetic_label_delta.rs"]
mod label_delta;
#[path = "typed_synthetic_loader.rs"]
mod loader;
#[path = "typed_synthetic_manifest.rs"]
mod manifest;
#[path = "typed_synthetic_preview.rs"]
mod preview;
#[path = "typed_synthetic_source_delta.rs"]
mod source_delta;
#[path = "typed_synthetic_successor.rs"]
mod successor;
#[cfg(test)]
#[path = "typed_synthetic_tests.rs"]
mod tests;

pub(crate) use loader::load;
pub(crate) use manifest::{generation_sources, input_paths};
pub(crate) use preview::preview;

/// Refuses authored data before any run directory or model initialization.
pub(crate) fn refuse_fit(cfg: &Config) -> anyhow::Result<()> {
    ensure!(
        cfg.detector
            .as_ref()
            .is_none_or(|d| d.typed_synthetic.is_none()),
        "authored synthetic augmentation is zero-update data-check only; fit/probe/RMS integration requires separate review"
    );
    if let Some(detector) = &cfg.detector {
        crate::detector::validate_real_silver_sources(&detector.silver)?;
    }
    Ok(())
}
