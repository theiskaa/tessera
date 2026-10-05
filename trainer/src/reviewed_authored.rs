//! Reviewed whole-source authored TRAIN data, separate from native cohorts and legacy synthesis.

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tessera::internal::FeatureConfig;

use crate::detector::DetectorDoc;
use crate::reviewed_data::Receipt;

#[path = "reviewed_authored_adjudication.rs"]
mod adjudication;
#[path = "reviewed_authored_loader.rs"]
mod loader;
#[path = "reviewed_authored_source.rs"]
mod source;
#[path = "reviewed_authored_successor.rs"]
mod successor;
#[cfg(test)]
#[path = "reviewed_authored_tests.rs"]
mod tests;

pub(crate) use tessera::internal::DetectorFeatureContract as InputContract;

/// Exact literal occurrence in the authored text; contact supervision stays separate.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Span {
    pub(crate) kind: String,
    pub(crate) text: String,
    pub(crate) start: u32,
    pub(crate) end: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct AuthoredSource {
    kind: String,
    native_dataset: Receipt,
    native_row_index: usize,
    native_name: String,
    native_text_sha256: String,
    format: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Row {
    name: String,
    country: String,
    input: String,
    expected: Vec<Span>,
    status: String,
    uncertainties: Vec<Value>,
    unresolved: Vec<Value>,
    annotation_policy_sha256: String,
    authored_source: AuthoredSource,
    real_source_text: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    origin: String,
    split: String,
    documents: usize,
    counts: BTreeMap<String, usize>,
    prepared: Receipt,
    policy: Receipt,
    receipt_pins: BTreeMap<std::path::PathBuf, String>,
    finalizer_sha256: String,
    /// Present only for a packet rebuilt on a native label successor; omitted selects the
    /// original route unchanged.
    #[serde(default, deserialize_with = "successor::present")]
    native_label_successor: Option<successor::NativeLabelSuccessor>,
    #[serde(default)]
    all_source_semantics_and_438_notes_reviewed: bool,
    #[serde(default)]
    all_source_bytes_and388_notes_reviewed: bool,
    label_passes: String,
    semantic_review: String,
    native_train_documents_changed: bool,
    fit_route_ready: bool,
    fit_allowed: bool,
    remaining: Vec<String>,
    limits: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Proposal {
    name: String,
    country: String,
    input: String,
    expected: Vec<Span>,
    uncertainties: Vec<Value>,
    #[serde(default, deserialize_with = "present_policy")]
    annotation_policy_sha256: Option<String>,
}

fn present_policy<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<String>, D::Error> {
    String::deserialize(deserializer).map(Some)
}

/// Loaded authored documents with exact source identities and all dependency bindings.
/// No fit, optimizer, sampler, model or native-cohort conversion is implemented here.
pub(crate) struct LoadedAuthored {
    pub(crate) names: Vec<String>,
    pub(crate) docs: Vec<DetectorDoc>,
    pub(crate) expected: Vec<Vec<Span>>,
    pub(crate) rules_expected: Vec<Vec<Span>>,
    pub(crate) identities: Vec<Value>,
    pub(crate) dependency_receipts: Vec<Receipt>,
}

/// Validate and encode a finalized whole-source authored packet, with no fitting side effects.
/// `native_inputs` must name the complete native composition against which text duplicates
/// are excluded; the source prepared dataset must be one of those exact pinned inputs, or,
/// for a native label successor packet, the successor segment the pinned amendment
/// application derives from it.
pub(crate) fn load(
    manifest: &Receipt,
    repository_root: &Path,
    native_inputs: &[Receipt],
    features: &FeatureConfig,
    input_contract: InputContract,
) -> anyhow::Result<LoadedAuthored> {
    loader::load(
        manifest,
        repository_root,
        native_inputs,
        features,
        input_contract,
    )
}
