//! Exact file identities shared by reviewed data and bounded fitting contracts.

use std::path::PathBuf;

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
/// Immutable absolute file reference with a SHA-256 binding.
pub(crate) struct Receipt {
    pub(crate) path: PathBuf,
    pub(crate) sha256: String,
}

impl Receipt {
    /// Read bytes only if the absolute reference still matches its declared digest.
    pub(crate) fn bytes(&self) -> anyhow::Result<Vec<u8>> {
        ensure!(
            self.path.is_absolute(),
            "cohort receipt paths must be absolute"
        );
        let bytes = std::fs::read(&self.path)
            .with_context(|| format!("reading {}", self.path.display()))?;
        ensure!(
            valid_sha(&self.sha256) && digest(&bytes) == self.sha256,
            "cohort receipt changed: {}",
            self.path.display()
        );
        Ok(bytes)
    }
}

/// SHA-256 of the exact bytes, without normalization.
pub(crate) fn digest(bytes: &[u8]) -> String {
    crate::export::sha256_hex(bytes)
}

/// Accept only a complete lowercase hexadecimal SHA-256 digest.
pub(crate) fn valid_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
