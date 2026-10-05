//! Shared composition of independently reviewed complete native datasets.

use std::collections::BTreeSet;

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};

use super::{Inputs, Receipt};
use crate::config::{Config, Task};

const SCOPE: &str = "reviewed-native-dataset-preflight-v1";

/// An immutable complete TRAIN segment and its original review/separation receipts.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Segment {
    pub(crate) train: Receipt,
    pub(crate) train_review: Receipt,
    pub(crate) separation_audit: Receipt,
    pub(crate) documents: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn composition(counts: &[usize], total: usize) -> Composition {
        let receipt = json!({"path":"/receipt","sha256":"a".repeat(64)});
        serde_json::from_value(json!({"scope":SCOPE,"config":receipt,"checkpoint":receipt,
            "policy":receipt,"dev":receipt,"dev_review":receipt,"dev_documents":100,
            "train_documents":total,"segments":counts.iter().map(|count| json!({
                "train":receipt,"train_review":receipt,"separation_audit":receipt,"documents":count
            })).collect::<Vec<_>>()}))
        .unwrap()
    }

    #[test]
    fn composition_rejects_count_overflow_and_unreviewed_empty_segments() {
        assert!(
            composition(&[usize::MAX, 1], usize::MAX)
                .validate()
                .is_err()
        );
        assert!(composition(&[178, 119], 298).validate().is_err());
        assert!(composition(&[178, 0], 178).validate().is_err());
        composition(&[178, 119], 297).validate().unwrap();
    }
}

/// A data-only composition with one shared, gold-independent development set.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Composition {
    pub(crate) scope: String,
    pub(crate) config: Receipt,
    pub(crate) checkpoint: Receipt,
    pub(crate) policy: Receipt,
    pub(crate) dev: Receipt,
    pub(crate) dev_review: Receipt,
    pub(crate) dev_documents: usize,
    pub(crate) train_documents: usize,
    pub(crate) segments: Vec<Segment>,
}

impl Composition {
    fn inputs<'a>(&'a self, segment: &'a Segment) -> Inputs<'a> {
        Inputs {
            policy: &self.policy,
            checkpoint: &self.checkpoint,
            train: &segment.train,
            train_review: &segment.train_review,
            dev: &self.dev,
            dev_review: &self.dev_review,
            separation_audit: &segment.separation_audit,
            train_documents: segment.documents,
            dev_documents: self.dev_documents,
        }
    }

    /// Enumerate original proof/native inputs without interpreting historical scope pin maps.
    pub(crate) fn dependency_receipts(&self) -> anyhow::Result<Vec<Receipt>> {
        self.verify_receipts()?;
        let mut receipts = vec![self.config.clone()];
        for segment in &self.segments {
            receipts.extend(super::readiness::dependency_receipts(
                &self.inputs(segment),
            )?);
        }
        Ok(receipts)
    }

    /// Recheck common receipts and the complete nested proof for every segment.
    pub(crate) fn verify_receipts(&self) -> anyhow::Result<()> {
        for receipt in [
            &self.config,
            &self.checkpoint,
            &self.policy,
            &self.dev,
            &self.dev_review,
        ] {
            receipt.bytes()?;
        }
        for segment in &self.segments {
            segment.train.bytes()?;
            segment.train_review.bytes()?;
            segment.separation_audit.bytes()?;
            super::verify_nested_receipts(&self.inputs(segment))?;
        }
        Ok(())
    }

    /// Reject unknown scopes, empty segments, overflow and inconsistent declared counts.
    pub(crate) fn validate(&self) -> anyhow::Result<()> {
        ensure!(self.scope == SCOPE, "unknown reviewed data preflight scope");
        ensure!(
            !self.segments.is_empty() && self.train_documents > 0 && self.dev_documents > 0,
            "data preflight requires nonempty declared TRAIN segments and DEV"
        );
        let count = self.segments.iter().try_fold(0usize, |total, segment| {
            ensure!(
                segment.documents > 0,
                "reviewed data segment cannot be empty"
            );
            total
                .checked_add(segment.documents)
                .context("reviewed data count overflow")
        })?;
        ensure!(
            count == self.train_documents,
            "data segment counts differ from the declared union"
        );
        Ok(())
    }

    /// Load only complete declared native rows, retaining every segment's split evidence.
    pub(crate) fn load(&self, config: &Config) -> anyhow::Result<super::Cohorts> {
        self.load_checked(config, true)
    }

    /// Validate every segment, require union kind support and bind canonical Text features.
    pub(crate) fn load_with_input_policy(
        &self,
        config: &Config,
        policy: crate::detector::DetectorInputPolicy,
    ) -> anyhow::Result<(super::Cohorts, serde_json::Value)> {
        let cohorts = self.load_checked(config, false)?;
        super::data::require_neural_support(&cohorts.train.docs, "composed TRAIN")?;
        super::data::require_neural_support(&cohorts.dev.docs, "DEV")?;
        super::reencode_with_input_policy(cohorts, config, policy)
    }

    fn load_checked(
        &self,
        config: &Config,
        require_segment_support: bool,
    ) -> anyhow::Result<super::Cohorts> {
        self.validate()?;
        self.verify_receipts()?;
        config.validate()?;
        let declared: Config = toml::from_str(std::str::from_utf8(&self.config.bytes()?)?)?;
        declared.validate()?;
        ensure!(
            config.task == Task::Detector
                && declared.task == Task::Detector
                && config.features == declared.features,
            "reviewed composition must retain its declared detector feature settings"
        );
        let mut names = BTreeSet::new();
        let mut texts = BTreeSet::new();
        let mut native_rows = BTreeSet::new();
        let mut development_identity = None;
        let mut training: Option<super::Cohort> = None;
        let mut development = None;
        let mut train_identity = Vec::new();
        for segment in &self.segments {
            let inputs = self.inputs(segment);
            let cohorts = if require_segment_support {
                super::load(&inputs, config)?
            } else {
                super::data::load_composed(&inputs, config)?
            };
            let identity = cohorts.identity()?;
            if let Some(previous) = &development_identity {
                ensure!(
                    previous == &identity["dev"],
                    "DEV identity differs between reviewed segments"
                );
            } else {
                development_identity = Some(identity["dev"].clone());
            }
            for (name, doc) in cohorts.train.names.iter().zip(&cohorts.train.docs) {
                let normalized = doc
                    .text
                    .split_whitespace()
                    .collect::<Vec<_>>()
                    .join(" ")
                    .to_lowercase();
                ensure!(
                    names.insert(name.clone()) && texts.insert(normalized),
                    "reviewed segments duplicate a document name or normalized complete text"
                );
            }
            for row in identity["train"]
                .as_array()
                .context("TRAIN identity array missing")?
            {
                let source = row["origin"]["dataset"]["sha256"]
                    .as_str()
                    .context("native dataset hash missing")?;
                let index = row["origin"]["row_index"]
                    .as_u64()
                    .context("native row index missing")?;
                ensure!(
                    native_rows.insert((source.to_owned(), index)),
                    "reviewed segments repeat the same original native row"
                );
                train_identity.push(row.clone());
            }
            match &mut training {
                Some(train) => train.append(cohorts.train),
                None => training = Some(cohorts.train),
            }
            if development.is_none() {
                development = Some(cohorts.dev);
            }
        }
        ensure!(
            train_identity.len() == self.train_documents && names.len() == self.train_documents,
            "reviewed union identity count differs"
        );
        self.verify_receipts()?;
        Ok(super::Cohorts {
            train: training.context("reviewed TRAIN composition missing")?,
            dev: development.context("reviewed DEV composition missing")?,
        })
    }
}
