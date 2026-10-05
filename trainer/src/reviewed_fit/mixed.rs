//! Separate authored TRAIN owners, complete canonical identities and deterministic mixed draws.

use super::{objective, sampling, supervision};
use crate::dataset::Encoded;
use crate::detector::DetectorDoc;
use crate::reviewed_authored::LoadedAuthored;
use crate::reviewed_data::{Cohort, Receipt, digest};
use crate::reviewed_train::{Loaded as NativeLoaded, Prepared};
use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeSet;

#[path = "mixed_draws.rs"]
mod draws;
#[path = "mixed_accounting.rs"]
mod token_accounting;
pub(super) use draws::{Loss, plan};
pub(super) use token_accounting::accounting;

const DATA_SCOPE: &str = "reviewed-native-authored-canonical-inputs-v1";

/// Shared config/runtime postprocessor for reviewed mixed scoring.
pub(crate) use crate::config::DetectorPostprocess as Postprocess;

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Packet {
    manifest: Receipt,
    documents: usize,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
/// A separate mixed authority, never a replacement native composition.
pub(super) struct Settings {
    scope: String,
    repository_root: std::path::PathBuf,
    authored: Vec<Packet>,
    native_documents: usize,
    authored_documents: usize,
    dev_documents: usize,
    pub(super) postprocess: Postprocess,
    pub(super) canonical_preflight: Option<Receipt>,
    pub(super) loss_weights: Option<Vec<f32>>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(tag = "origin", rename_all = "snake_case", deny_unknown_fields)]
/// Flat draw resolution preserves source kind and owning segment.
enum Index {
    Native { row: usize },
    AuthoredTrainOnly { packet: usize, row: usize },
}

impl Settings {
    pub(super) fn execution_ready(&self) -> anyhow::Result<()> {
        ensure!(
            self.canonical_preflight.is_some() && self.loss_weights.is_some(),
            "mixed fit needs actual complete mixed canonical authority and explicit reviewed loss weights"
        );
        weights(
            self.loss_weights
                .as_deref()
                .context("mixed loss weights missing")?,
        )
    }
}

/// Authored encodings are held independently; native Cohorts and their Origins are untouched.
pub(super) struct LoadedMixed {
    pub(super) authored: Vec<LoadedAuthored>,
    indices: Vec<Index>,
    pub(super) record: Value,
    pub(super) dependencies: Vec<Receipt>,
}

fn feature_identity(doc: &DetectorDoc) -> Value {
    json!({"text_sha256":digest(doc.text.as_bytes()),"token_spans":doc.enc.token_spans,
        "ngram_ids_plus_one":doc.enc.ngram_ids,"script":doc.enc.script,"shape":doc.enc.shape,
        "flags":doc.enc.flags,"breaks":doc.breaks,"labels":doc.enc.labels})
}

/// Per-document counts; `reviewed` adds sidecar-excluded O tokens only for that objective.
fn token_counts(
    doc: &DetectorDoc,
    weights: &[f32],
    reviewed: Option<&[bool]>,
) -> anyhow::Result<Value> {
    ensure!(
        weights.len() == 7 && weights.iter().all(|w| w.is_finite() && *w > 0.0),
        "finite positive loss weights required"
    );
    ensure!(
        doc.enc.flags.len() == doc.enc.labels.len() && !doc.enc.labels.is_empty(),
        "canonical token alignment missing"
    );
    ensure!(
        reviewed.is_none_or(|r| r.len() == doc.enc.labels.len()),
        "reviewed ambiguity row differs from canonical tokens"
    );
    let mut labels = [0usize; 7];
    let mut active = [0usize; 7];
    let mut rule_o = 0usize;
    let mut ambiguity_o = 0usize;
    for (token, (&flags, &label)) in doc.enc.flags.iter().zip(&doc.enc.labels).enumerate() {
        ensure!(label < 7, "neural label outside detector classes");
        labels[usize::from(label)] += 1;
        if reviewed.is_some_and(|r| r[token]) {
            ensure!(
                label == 0 && flags & tessera::internal::flag::IN_RULE_SPAN == 0,
                "reviewed ambiguity token is supervised or rule-controlled"
            );
            ambiguity_o += 1;
        } else if flags & tessera::internal::flag::IN_RULE_SPAN != 0 {
            ensure!(
                label == 0,
                "rule-controlled token has positive neural label"
            );
            rule_o += 1;
        } else {
            active[usize::from(label)] += 1;
        }
    }
    let mass: f64 = active
        .iter()
        .zip(weights)
        .map(|(n, w)| *n as f64 * f64::from(*w))
        .sum();
    ensure!(mass.is_finite(), "nonfinite token mass");
    let mut counts = json!({"retained_tokens":doc.enc.labels.len(),"all_label_counts":labels,
        "excluded_rule_o_tokens":rule_o,"active_label_counts":active,"active_weighted_mass":mass,
        "weighted_mass_is_not_gradient_share":true});
    if reviewed.is_some() {
        counts["excluded_reviewed_ambiguity_o_tokens"] = json!(ambiguity_o);
    }
    Ok(counts)
}

fn validate_native_counts(settings: &Settings, native: usize, dev: usize) -> anyhow::Result<()> {
    ensure!(
        settings.native_documents == native && settings.dev_documents == dev,
        "mixed native/DEV counts differ"
    );
    Ok(())
}

fn validate_authored_owner(
    name: &str,
    text: &str,
    identity: &Value,
    native_train_names: &[String],
    dev_names: &[String],
    dev_texts: &[&str],
) -> anyhow::Result<()> {
    ensure!(
        !dev_names.iter().any(|n| n == name) && !dev_texts.contains(&text),
        "authored equals frozen DEV"
    );
    ensure!(
        identity["origin_type"] == "authored_train_only",
        "authored origin missing"
    );
    let base = identity["base_variant_group"]["native_name"]
        .as_str()
        .context("authored parent name missing")?;
    ensure!(
        native_train_names.iter().any(|n| n == base),
        "authored parent is not native TRAIN"
    );
    Ok(())
}

fn verify_canonical(receipt: &Receipt, record: &Value) -> anyhow::Result<()> {
    ensure!(
        serde_json::from_slice::<Value>(&receipt.bytes()?)? == *record,
        "mixed canonical closure differs"
    );
    Ok(())
}

fn weights(weights: &[f32]) -> anyhow::Result<()> {
    ensure!(
        weights.len() == 7 && weights.iter().all(|v| v.is_finite() && *v > 0.0),
        "finite positive seven-class weights required"
    );
    Ok(())
}

impl LoadedMixed {
    /// Resolve an existing whole encoding; no synthetic-to-native conversion exists.
    pub(super) fn doc<'a>(
        &'a self,
        native: &'a Cohort,
        index: usize,
    ) -> anyhow::Result<&'a DetectorDoc> {
        match self
            .indices
            .get(index)
            .context("mixed draw index missing")?
        {
            Index::Native { row } => native.docs.get(*row).context("native draw missing"),
            Index::AuthoredTrainOnly { packet, row } => self
                .authored
                .get(*packet)
                .and_then(|p| p.docs.get(*row))
                .context("authored draw missing"),
        }
    }
    pub(super) fn len(&self) -> usize {
        self.indices.len()
    }
    /// Native TRAIN row behind a flat draw index; authored rows own no native row.
    pub(super) fn native_owner(&self, index: usize) -> anyhow::Result<Option<usize>> {
        Ok(
            match self
                .indices
                .get(index)
                .context("mixed draw index missing")?
            {
                Index::Native { row } => Some(*row),
                Index::AuthoredTrainOnly { .. } => None,
            },
        )
    }
    pub(super) fn native_owners(&self, indices: &[usize]) -> anyhow::Result<Vec<Option<usize>>> {
        indices.iter().map(|&i| self.native_owner(i)).collect()
    }
    /// Native TRAIN parent name of every authored row.
    pub(super) fn authored_parents(&self) -> anyhow::Result<Vec<&str>> {
        self.authored
            .iter()
            .flat_map(|packet| &packet.identities)
            .map(|identity| {
                identity["base_variant_group"]["native_name"]
                    .as_str()
                    .context("authored parent name missing")
            })
            .collect()
    }
    /// Flat rows owned by the given native rows (`Some`) or by authored rows (`None`).
    #[cfg(test)]
    pub(super) fn with_owners(owners: &[Option<usize>]) -> Self {
        Self {
            authored: Vec::new(),
            indices: owners
                .iter()
                .enumerate()
                .map(|(i, owner)| match owner {
                    Some(row) => Index::Native { row: *row },
                    None => Index::AuthoredTrainOnly { packet: 0, row: i },
                })
                .collect(),
            record: Value::Null,
            dependencies: Vec::new(),
        }
    }
    pub(super) fn items(&self, native: &Cohort, indices: &[usize]) -> anyhow::Result<Vec<Encoded>> {
        indices
            .iter()
            .map(|&i| Ok(self.doc(native, i)?.enc.clone()))
            .collect()
    }
    pub(super) fn verify(&self) -> anyhow::Result<()> {
        for r in &self.dependencies {
            r.bytes()?;
        }
        Ok(())
    }
}

/// Load actual whole authored rows and emit complete features without model initialization.
pub(super) fn load(
    settings: &Settings,
    prepared: &Prepared,
    native: &NativeLoaded,
) -> anyhow::Result<LoadedMixed> {
    let cfg = prepared.config();
    let contract = cfg.detector_feature_contract();
    ensure!(
        settings.scope == "reviewed-native-authored-train-only-v1"
            && !settings.authored.is_empty()
            && contract == tessera::internal::DetectorFeatureContract::TabCells25
            && prepared.input_policy() == crate::detector::DetectorInputPolicy::KnownUs,
        "mixed route requires explicit KnownUs/TabCells25 authored TRAIN contract"
    );
    validate_native_counts(
        settings,
        native.cohorts().train.docs.len(),
        native.cohorts().dev.docs.len(),
    )?;
    let reference = cfg
        .detector
        .as_ref()
        .context("detector weights missing")?
        .class_weights
        .as_slice();
    if let Some(w) = &settings.loss_weights {
        weights(w)?;
    }
    let root = settings.repository_root.as_path();
    ensure!(
        root.is_absolute() && root.is_dir(),
        "authored source repository root must be explicit and absolute"
    );
    let native_inputs = prepared.native_train_inputs()?;
    let native_identity = native.cohorts().identity()?;
    let native_rows = native_identity["train"]
        .as_array()
        .context("native identity missing")?;
    let mut authored = Vec::new();
    let mut dependencies = Vec::new();
    let mut indices = Vec::new();
    let mut rows = Vec::new();
    let mut names = BTreeSet::new();
    let mut texts = BTreeSet::new();
    for (i, doc) in native.cohorts().train.docs.iter().enumerate() {
        ensure!(
            names.insert(native.cohorts().train.names[i].clone()) && texts.insert(doc.text.clone()),
            "native name/text duplicate"
        );
        indices.push(Index::Native { row: i });
        rows.push(json!({"index":indices.len()-1,"name":native.cohorts().train.names[i],
            "origin":indices.last(),"binding":native_rows[i],"features":feature_identity(doc),"token_accounting_reference":token_counts(doc,reference,None)?}));
    }
    let mut manifests = BTreeSet::new();
    for (packet, declaration) in settings.authored.iter().enumerate() {
        ensure!(
            declaration.documents > 0 && manifests.insert(declaration.manifest.path.clone()),
            "authored packet empty/repeated"
        );
        let manifest: Value = serde_json::from_slice(&declaration.manifest.bytes()?)?;
        ensure!(
            manifest["policy"] == serde_json::to_value(prepared.annotation_policy())?,
            "authored/native policy differs"
        );
        let data = crate::reviewed_authored::load(
            &declaration.manifest,
            root,
            &native_inputs,
            &cfg.features.to_tessera(),
            contract,
        )?;
        ensure!(
            data.docs.len() == declaration.documents,
            "authored packet count differs"
        );
        dependencies.push(declaration.manifest.clone());
        dependencies.extend(data.dependency_receipts.clone());
        for (row, doc) in data.docs.iter().enumerate() {
            ensure!(
                names.insert(data.names[row].clone()) && texts.insert(doc.text.clone()),
                "mixed name or exact text duplicate"
            );
            let dev_texts: Vec<_> = native
                .cohorts()
                .dev
                .docs
                .iter()
                .map(|d| d.text.as_str())
                .collect();
            validate_authored_owner(
                &data.names[row],
                &doc.text,
                &data.identities[row],
                &native.cohorts().train.names,
                &native.cohorts().dev.names,
                &dev_texts,
            )?;
            indices.push(Index::AuthoredTrainOnly { packet, row });
            rows.push(json!({"index":indices.len()-1,"name":data.names[row],
                "origin":indices.last(),"binding":data.identities[row],"features":feature_identity(doc),"expected":data.expected[row],
                "token_accounting_reference":token_counts(doc,reference,None)?}));
        }
        authored.push(data);
    }
    let authored_total = authored.iter().try_fold(0usize, |n, a| {
        n.checked_add(a.docs.len())
            .context("authored document count overflow")
    })?;
    ensure!(
        authored_total == settings.authored_documents,
        "authored total differs"
    );
    let record = json!({"scope":DATA_SCOPE,"source_repository_root":root,"native_authority":prepared.identity(native)?,"authored_manifests":settings.authored,
        "input_policy":prepared.input_policy(),"feature_contract":contract.name(),"flag_bits":contract.flag_bits(),
        "forward_operator":"context_rms_v2","base_decoder":"address_continuation_v1","postprocess":settings.postprocess,
        "native_documents":settings.native_documents,"authored_documents":settings.authored_documents,"total_train_documents":indices.len(),
        "dev_documents":settings.dev_documents,"train_inputs":rows,"dev_inputs":"unchanged native authority canonical gold-independent DEV",
        "token_accounting_reference_weights":reference,"model_initialized":false,"optimizer_initialized":false,"fit_allowed":false,
        "authored_is_independent_source_evidence":false});
    let loaded = LoadedMixed {
        authored,
        indices,
        record,
        dependencies,
    };
    loaded.verify()?;
    if let Some(receipt) = &settings.canonical_preflight {
        verify_canonical(receipt, &loaded.record)?;
    }
    Ok(loaded)
}

#[cfg(test)]
#[path = "mixed_tests.rs"]
mod tests;
