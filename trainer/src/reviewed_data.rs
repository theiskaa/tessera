//! Shared whole-document validation for reviewed detector datasets.

mod composition;
mod data;
mod preflight;
mod readiness;
mod receipt;

use crate::config::Config;

pub(crate) use composition::Composition;
pub(crate) use data::{Cohort, Cohorts};
pub(crate) use readiness::DeclaredExclusion;
pub(crate) use receipt::{Receipt, digest, valid_sha};

/// Data bindings shared by a fitting protocol and a data-only preflight.
/// The checkpoint binds exposure evidence; this module never loads its parameters.
pub(crate) struct Inputs<'a> {
    pub(crate) policy: &'a Receipt,
    pub(crate) checkpoint: &'a Receipt,
    pub(crate) train: &'a Receipt,
    pub(crate) train_review: &'a Receipt,
    pub(crate) dev: &'a Receipt,
    pub(crate) dev_review: &'a Receipt,
    pub(crate) separation_audit: &'a Receipt,
    pub(crate) train_documents: usize,
    pub(crate) dev_documents: usize,
}

/// Encode complete reviewed TRAIN rows and gold-independent DEV features.
pub(crate) fn load(inputs: &Inputs<'_>, config: &Config) -> anyhow::Result<Cohorts> {
    data::load(inputs, config)
}

/// Recheck nested review, native-source and exposure receipts without encoding.
pub(crate) fn verify_nested_receipts(inputs: &Inputs<'_>) -> anyhow::Result<()> {
    readiness::verify_nested_receipts(inputs)
}

/// Validate a declared dataset composition without initializing a model or optimizer.
pub(crate) fn preflight(manifest: &std::path::Path, out: &std::path::Path) -> anyhow::Result<()> {
    preflight::run(manifest, out)
}

/// Re-encode strictly reviewed whole Text cohorts for a separately bound future input policy.
/// TRAIN receives supervised targets; DEV labels stay empty and gold never chooses features.
#[allow(
    dead_code,
    reason = "future reviewed route stays unwired until its policy manifest is reviewed"
)]
pub(crate) fn load_with_input_policy(
    inputs: &Inputs<'_>,
    config: &Config,
    policy: crate::detector::DetectorInputPolicy,
) -> anyhow::Result<(Cohorts, serde_json::Value)> {
    let cohorts = data::load(inputs, config)?;
    cohorts
        .train
        .refuse_declared_exclusions("unwired input-policy route")?;
    reencode_with_input_policy(cohorts, config, policy)
}

/// Re-encode an already source-reviewed native composition without changing its gold.
/// The composition loader must validate every origin, review and split before this call.
pub(crate) fn reencode_with_input_policy(
    mut cohorts: Cohorts,
    config: &Config,
    policy: crate::detector::DetectorInputPolicy,
) -> anyhow::Result<(Cohorts, serde_json::Value)> {
    use crate::detector::{self, DetectorInputPolicy};
    use anyhow::{Context, ensure};
    use serde_json::json;
    use tessera::internal::flag;
    let fc = config.features.to_tessera();
    let mut records = Vec::new();
    for (split, cohort) in [("train", &mut cohorts.train), ("dev", &mut cohorts.dev)] {
        for (name, doc) in cohort.names.iter().zip(&mut cohort.docs) {
            let supervised = split == "train";
            let gold = if supervised { doc.gold.as_slice() } else { &[] };
            let mut enc = detector::encode_document_with_feature_contract(
                &doc.text,
                gold,
                &fc,
                policy,
                config.detector_feature_contract(),
            )
            .with_context(|| format!("{split} {name} is unreachable under {policy:?}"))?;
            enc.country = doc.enc.country.clone();
            let breaks = detector::breaks_of(&doc.text);
            let n = enc.token_spans.len();
            ensure!(
                n > 0
                    && enc.ngram_ids.len() == n
                    && enc.script.len() == n
                    && enc.shape.len() == n
                    && enc.flags.len() == n
                    && enc.labels.len() == n
                    && breaks.len() == n,
                "canonical retained input arrays differ in length"
            );
            ensure!(
                enc.flags.iter().all(|f| f & flag::MASKED == 0),
                "canonical Text encoding cannot carry structured-input MASKED flags"
            );
            ensure!(
                supervised || enc.labels.iter().all(|&label| label == 0),
                "DEV encoder consumed supervision"
            );
            let masked: Vec<_> = enc
                .flags
                .iter()
                .map(|f| f & (flag::IN_RULE_SPAN | flag::MASKED) != 0)
                .collect();
            let mut features = json!({
                "contract":"whole-text-retained-detector-input-v1",
                "text_sha256":digest(doc.text.as_bytes()),
                "feature_config":{"ngram_sizes":fc.ngram_sizes,
                    "hash_buckets":fc.hash_buckets,"hash_seed":fc.hash_seed},
                "token_spans":enc.token_spans,"ngram_ids_plus_one":enc.ngram_ids,
                "script":enc.script,"shape":enc.shape,"flags":enc.flags,
                "decoder_masked":masked,"paragraph_breaks":breaks,
            });
            if config.detector_feature_contract()
                != tessera::internal::DetectorFeatureContract::Legacy23
            {
                features["contract"] = json!("whole-text-retained-detector-input-v2");
                features["detector_feature_contract"] =
                    json!(config.detector_feature_contract().name());
                features["flag_bits"] = json!(config.detector_feature_contract().flag_bits());
            }
            records.push(json!({"split":split,"name":name,"input_policy":policy,
                "encoded_feature_sha256":digest(&serde_json::to_vec(&features)?),
                "supervised":supervised,"retained_tokens":n}));
            doc.enc = enc;
            doc.breaks = breaks;
        }
    }
    let selector = match policy {
        DetectorInputPolicy::KnownUs => "scan_text_rules(text, US)",
        DetectorInputPolicy::AutoText => "scan_text_rules(text, empty) with real region inference",
    };
    let mut identity = json!({"input_policy":policy,"format":"Text","featurize_country":null,
        "structured_input_mask":null,"rule_selector":selector,
        "encoded_feature_hash_contract":"whole-text-retained-detector-input-v1; serde_json fixed field order, no gold/labels",
        "cohorts":cohorts.identity()?,"encoded_features":records});
    if config.detector_feature_contract() != tessera::internal::DetectorFeatureContract::Legacy23 {
        identity["detector_feature_contract"] = json!(config.detector_feature_contract().name());
        identity["flag_bits"] = json!(config.detector_feature_contract().flag_bits());
        identity["encoded_feature_hash_contract"] = json!(
            "whole-text-retained-detector-input-v2; serde_json fixed field order, no gold/labels"
        );
    }
    Ok((cohorts, identity))
}
