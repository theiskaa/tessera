//! Adapter from the fixed fitting protocol to shared reviewed data validation.

use super::contracts::Manifest;
use crate::config::Config;

pub(super) use crate::reviewed_data::{Cohort, Cohorts};

pub(super) fn inputs(manifest: &Manifest) -> crate::reviewed_data::Inputs<'_> {
    crate::reviewed_data::Inputs {
        policy: &manifest.policy,
        checkpoint: &manifest.checkpoint,
        train: &manifest.train,
        train_review: &manifest.train_review,
        dev: &manifest.dev,
        dev_review: &manifest.dev_review,
        separation_audit: &manifest.separation_audit,
        train_documents: manifest.train_documents,
        dev_documents: manifest.dev_documents,
    }
}

/// Fully supervised cohort fits refuse rows with declared reviewed loss exclusions.
pub(super) fn load(manifest: &Manifest, cfg: &Config) -> anyhow::Result<Cohorts> {
    let cohorts = crate::reviewed_data::load(&inputs(manifest), cfg)?;
    cohorts.train.refuse_declared_exclusions("cohort fit")?;
    Ok(cohorts)
}
