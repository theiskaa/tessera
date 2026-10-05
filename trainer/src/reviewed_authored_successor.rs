//! Opt-in transfer of reviewed native TRAIN label amendments onto authored variants.
//!
//! The original authored packet stays the semantic authority: every original check runs
//! against it unchanged. A successor packet may differ only by native label changes that a
//! pinned amendment application made to the parent rows, each moved through the original
//! literal copy map to the exact authored occurrence. Case rulings those changes contradict
//! are recorded as superseded instead of being silently dropped.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::loader::{Dependencies, jsonl, validate_spans, verify_manifest};
use super::{Manifest, Span};
use crate::reviewed_data::{Receipt, digest};

#[cfg(test)]
#[path = "reviewed_authored_successor_tests.rs"]
mod tests;

const TRANSFER_SCHEMA: &str = "reviewed-authored-native-label-transfer-v1";
const APPLICATION_VERSION: &str = "native-label-amendments-v2";

/// Manifest pins for an authored packet rebuilt on a native label successor.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct NativeLabelSuccessor {
    amendment_application: Receipt,
    transfer: Receipt,
}

/// An explicit `null` is rejected so that only an omitted field selects the original route.
pub(super) fn present<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Option<NativeLabelSuccessor>, D::Error> {
    NativeLabelSuccessor::deserialize(deserializer).map(Some)
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Application {
    version: String,
    status: String,
    manifest: Receipt,
    adjudication: Receipt,
    native_documents: usize,
    segments: Vec<Segment>,
    changes: Vec<Change>,
    before_counts: BTreeMap<String, i64>,
    after_counts: BTreeMap<String, i64>,
    canonical_features_regenerated: bool,
    independent_successor_validation_complete: bool,
    training_ready: bool,
    training_started: bool,
    limits: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Segment {
    original: Receipt,
    corrected: Receipt,
    documents: usize,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Change {
    name: String,
    input_sha256: String,
    added: Vec<Span>,
    removed: Vec<Span>,
    reviewed_kinds: Vec<String>,
    adjudication_ids: Vec<String>,
}

#[derive(Deserialize)]
struct NativeDecision {
    id: String,
    name: String,
    decision: String,
    span: Span,
    input_sha256: String,
    basis: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Copy {
    source_start: usize,
    source_end: usize,
    output_start: usize,
    output_end: usize,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Transfer {
    schema: String,
    original_manifest: Receipt,
    original_prepared: Receipt,
    amendment_application: Receipt,
    native_adjudication: Receipt,
    source_segment: SegmentPins,
    parents: Vec<ParentTransfer>,
    rows: Vec<RowTransfer>,
    superseded_case_decisions: Vec<Superseded>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SegmentPins {
    original: Receipt,
    corrected: Receipt,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ParentTransfer {
    name: String,
    native_row_index: usize,
    input_sha256: String,
    original_line_sha256: String,
    successor_line_sha256: String,
    added: Vec<Span>,
    removed: Vec<Span>,
    adjudication_ids: Vec<String>,
}

struct NativeChange {
    action: String,
    adjudication_id: String,
    basis: String,
    span: Span,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SpanTransfer {
    action: String,
    adjudication_id: String,
    native: Span,
    authored: Span,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RowTransfer {
    name: String,
    native_name: String,
    spans: Vec<SpanTransfer>,
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Superseded {
    authored_name: String,
    span: Span,
    superseded_decision: String,
    superseded_by: String,
    superseding_basis: String,
}

/// Opened successor pins, checked against the original manifest before any row is read.
pub(super) struct Successor {
    transfer_pin: Receipt,
    application_record: Value,
    application: Application,
    native_decisions: Vec<NativeDecision>,
    transfer: Transfer,
    original_counts: BTreeMap<String, usize>,
}

/// Byte authorities a successor transfer is checked against.
pub(super) struct Evidence<'a> {
    pub(super) source_bytes: &'a [u8],
    pub(super) corrected_bytes: &'a [u8],
    pub(super) reviewed_bytes: &'a [u8],
    pub(super) prepared_bytes: &'a [u8],
    pub(super) maps: &'a [Value],
    pub(super) case_decisions: &'a [Value],
}

fn same(a: &Receipt, b: &Receipt) -> bool {
    a.path == b.path && a.sha256 == b.sha256
}

/// Open the successor pins of `manifest`, or `None` for an original packet.
pub(super) fn open(
    manifest: &Manifest,
    deps: &mut Dependencies,
) -> anyhow::Result<Option<Successor>> {
    let Some(declared) = &manifest.native_label_successor else {
        return Ok(None);
    };
    ensure!(
        manifest.all_source_bytes_and388_notes_reviewed,
        "native label successor requires the source-adjudicated authored route"
    );
    let transfer: Transfer = serde_json::from_slice(&deps.read(&declared.transfer)?)?;
    let application: Application =
        serde_json::from_slice(&deps.read(&declared.amendment_application)?)?;
    ensure!(
        transfer.schema == TRANSFER_SCHEMA
            && application.version == APPLICATION_VERSION
            && same(
                &transfer.amendment_application,
                &declared.amendment_application
            )
            && same(&transfer.native_adjudication, &application.adjudication),
        "native label transfer schema or amendment application binding differs"
    );
    deps.read(&application.manifest)?;
    verify_application(&application)?;
    let native_decisions = jsonl::<NativeDecision>(&deps.read(&application.adjudication)?)?;
    let mut ids = BTreeSet::new();
    ensure!(
        native_decisions
            .iter()
            .all(|decision| ids.insert(decision.id.as_str())),
        "native adjudication repeats a decision id"
    );
    let original: Manifest = serde_json::from_slice(&deps.read(&transfer.original_manifest)?)?;
    verify_original_manifest(manifest, &original, &transfer.original_prepared)?;
    deps.read(&Receipt {
        path: original
            .prepared
            .path
            .parent()
            .context("original prepared parent absent")?
            .join("finalize.py"),
        sha256: original.finalizer_sha256.clone(),
    })?;
    let application_record = json!({"receipt":declared.amendment_application,"manifest":application.manifest,
        "status":application.status,"native_documents":application.native_documents,
        "independent_successor_validation_complete":application.independent_successor_validation_complete,
        "canonical_features_regenerated":application.canonical_features_regenerated,
        "training_ready":application.training_ready,"training_started":application.training_started,
        "limits":application.limits});
    Ok(Some(Successor {
        transfer_pin: declared.transfer.clone(),
        application_record,
        application,
        native_decisions,
        transfer,
        original_counts: original.counts,
    }))
}

/// Unique changes whose net label effect matches the application's own count summary.
fn verify_application(application: &Application) -> anyhow::Result<()> {
    let mut names = BTreeSet::new();
    ensure!(
        application
            .changes
            .iter()
            .all(|change| names.insert(change.name.as_str())),
        "amendment application repeats a native document"
    );
    ensure!(
        application
            .segments
            .iter()
            .map(|segment| segment.documents)
            .sum::<usize>()
            == application.native_documents,
        "amendment application segment membership differs"
    );
    let mut net = application.before_counts.clone();
    for change in &application.changes {
        for (spans, sign) in [(&change.added, 1), (&change.removed, -1)] {
            for span in spans {
                *net.entry(span.kind.clone()).or_default() += sign;
            }
        }
    }
    ensure!(
        net == application.after_counts,
        "amendment application counts differ from its changes"
    );
    Ok(())
}

/// The successor must restate the original packet except for its rows, counts and finalizer.
fn verify_original_manifest(
    successor: &Manifest,
    original: &Manifest,
    original_prepared: &Receipt,
) -> anyhow::Result<()> {
    verify_manifest(original)?;
    ensure!(
        original.native_label_successor.is_none(),
        "native label successor cannot chain another successor"
    );
    ensure!(
        same(&original.prepared, original_prepared)
            && original.prepared.path != successor.prepared.path
            && same(&original.policy, &successor.policy)
            && original.receipt_pins == successor.receipt_pins
            && original.schema == successor.schema
            && original.origin == successor.origin
            && original.split == successor.split
            && original.documents == successor.documents
            && original.all_source_semantics_and_438_notes_reviewed
                == successor.all_source_semantics_and_438_notes_reviewed
            && original.all_source_bytes_and388_notes_reviewed
                == successor.all_source_bytes_and388_notes_reviewed
            && original.label_passes == successor.label_passes
            && original.semantic_review == successor.semantic_review
            && original.native_train_documents_changed == successor.native_train_documents_changed
            && original.fit_route_ready == successor.fit_route_ready
            && original.fit_allowed == successor.fit_allowed
            && original.remaining == successor.remaining
            && successor.limits.starts_with(&original.limits),
        "successor manifest changes original authored authority beyond its rows"
    );
    Ok(())
}

impl Successor {
    /// The original authored rows the successor rows are derived from.
    pub(super) fn original_prepared(&self) -> &Receipt {
        &self.transfer.original_prepared
    }

    /// Return the successor segment the application derives from `source`, which must replace
    /// the original in the pinned native composition.
    pub(super) fn native_source(
        &self,
        source: &Receipt,
        native_inputs: &[Receipt],
    ) -> anyhow::Result<Receipt> {
        let segments: Vec<_> = self
            .application
            .segments
            .iter()
            .filter(|segment| same(&segment.original, source))
            .collect();
        ensure!(
            segments.len() == 1,
            "amendment application does not map the original bundle authority to one successor segment"
        );
        let corrected = &segments[0].corrected;
        ensure!(
            same(&self.transfer.source_segment.original, source)
                && same(&self.transfer.source_segment.corrected, corrected),
            "native label transfer segment mapping differs from the amendment application"
        );
        ensure!(
            !native_inputs.iter().any(|pin| same(pin, source)),
            "native composition retains the superseded original segment"
        );
        Ok(corrected.clone())
    }

    /// Check every successor authored row against the original adjudicated row plus exactly
    /// the transferred native changes, and return one identity record per row.
    pub(super) fn verify(&self, evidence: &Evidence) -> anyhow::Result<Vec<Value>> {
        let source_lines = lines(evidence.source_bytes)?;
        let corrected_lines = lines(evidence.corrected_bytes)?;
        let reviewed_lines = lines(evidence.reviewed_bytes)?;
        let prepared_lines = lines(evidence.prepared_bytes)?;
        ensure!(
            source_lines.len() == corrected_lines.len()
                && reviewed_lines.len() == prepared_lines.len()
                && reviewed_lines.len() == evidence.maps.len(),
            "successor segment or authored row membership differs"
        );
        let changes: BTreeMap<_, _> = self
            .application
            .changes
            .iter()
            .map(|change| (change.name.as_str(), change))
            .collect();
        let mut parents = Vec::new();
        let mut native_transfers = BTreeMap::new();
        let mut rows = Vec::new();
        let mut superseded = Vec::new();
        let mut identities = Vec::new();
        let mut original_counts = BTreeMap::<String, usize>::new();
        for (index, (reviewed_line, prepared_line)) in
            reviewed_lines.iter().zip(&prepared_lines).enumerate()
        {
            let reviewed: Value = serde_json::from_str(reviewed_line)?;
            let prepared: Value = serde_json::from_str(prepared_line)?;
            let native_name = reviewed["authored_source"]["native_name"]
                .as_str()
                .context("authored native parent name absent")?;
            let row_index = usize::try_from(
                reviewed["authored_source"]["native_row_index"]
                    .as_u64()
                    .context("authored native parent index absent")?,
            )?;
            if !native_transfers.contains_key(native_name) {
                let (parent, transfers) = self.parent(
                    native_name,
                    row_index,
                    source_lines.get(row_index).copied(),
                    corrected_lines.get(row_index).copied(),
                    changes.get(native_name).copied(),
                )?;
                parents.push(parent);
                native_transfers.insert(native_name.to_owned(), (row_index, transfers));
            }
            let (parent_index, native) = &native_transfers[native_name];
            ensure!(
                *parent_index == row_index,
                "authored rows disagree on their native parent row"
            );
            let map = &evidence.maps[index];
            let name = reviewed["name"].as_str().context("authored name absent")?;
            ensure!(
                map["name"] == name && map["native_source_document"] == native_name,
                "literal copy map differs from authored row"
            );
            let copies: Vec<Copy> = serde_json::from_value(map["literal_copy_map"].clone())?;
            let input = reviewed["input"]
                .as_str()
                .context("authored input absent")?;
            let spans = native
                .iter()
                .map(|change| {
                    Ok(SpanTransfer {
                        action: change.action.clone(),
                        adjudication_id: change.adjudication_id.clone(),
                        native: change.span.clone(),
                        authored: project(&copies, input, &change.span)?,
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            let original: Vec<Span> = serde_json::from_value(reviewed["expected"].clone())?;
            for span in &original {
                *original_counts.entry(span.kind.clone()).or_default() += 1;
            }
            let expected = apply(
                &original,
                spans.iter().map(|s| (s.action.as_str(), &s.authored)),
            )?;
            let mut actual: Vec<Span> = serde_json::from_value(prepared["expected"].clone())?;
            actual.sort();
            ensure!(
                actual == expected && same_except_expected(&reviewed, &prepared)?,
                "successor authored row differs from original labels plus transferred native changes"
            );
            validate_spans(input, &actual)?;
            ensure!(
                !spans.is_empty() || reviewed_line == prepared_line,
                "authored row without a transferred change is not byte identical"
            );
            let row_superseded = supersessions(name, native, &spans, evidence.case_decisions)?;
            identities.push(json!({"amendment_application":self.application_record,"transfer":self.transfer_pin,
                "source_segment":self.transfer.source_segment,"original_row_sha256":digest(reviewed_line.as_bytes()),
                "original_adjudicated_expected":original,"transferred_native_changes":spans,
                "superseded_case_decisions":row_superseded}));
            superseded.extend(row_superseded);
            if !spans.is_empty() {
                rows.push(RowTransfer {
                    name: name.to_owned(),
                    native_name: native_name.to_owned(),
                    spans,
                });
            }
        }
        ensure!(
            original_counts == self.original_counts,
            "original authored counts differ from the original manifest"
        );
        ensure!(
            serde_json::to_value(&parents)? == serde_json::to_value(&self.transfer.parents)?
                && serde_json::to_value(&rows)? == serde_json::to_value(&self.transfer.rows)?
                && serde_json::to_value(&superseded)?
                    == serde_json::to_value(&self.transfer.superseded_case_decisions)?,
            "declared native label transfer record differs from the recomputed transfer"
        );
        Ok(identities)
    }

    /// Bind one native parent's successor line to its original line and application change.
    fn parent(
        &self,
        name: &str,
        index: usize,
        original_line: Option<&str>,
        successor_line: Option<&str>,
        change: Option<&Change>,
    ) -> anyhow::Result<(ParentTransfer, Vec<NativeChange>)> {
        let original_line = original_line.context("original native parent row absent")?;
        let successor_line = successor_line.context("successor native parent row absent")?;
        let original: Value = serde_json::from_str(original_line)?;
        let successor: Value = serde_json::from_str(successor_line)?;
        let input = original["input"]
            .as_str()
            .context("native parent input absent")?;
        ensure!(original["name"] == name, "native parent row name differs");
        let mut parent = ParentTransfer {
            name: name.to_owned(),
            native_row_index: index,
            input_sha256: digest(input.as_bytes()),
            original_line_sha256: digest(original_line.as_bytes()),
            successor_line_sha256: digest(successor_line.as_bytes()),
            added: vec![],
            removed: vec![],
            adjudication_ids: vec![],
        };
        let Some(change) = change else {
            ensure!(
                original_line == successor_line,
                "unchanged native parent row is not byte identical in the successor"
            );
            return Ok((parent, vec![]));
        };
        ensure!(
            change.input_sha256 == parent.input_sha256
                && !(change.added.is_empty() && change.removed.is_empty())
                && same_except_expected(&original, &successor)?,
            "successor native parent changes more than its amended labels"
        );
        let mut transfers = Vec::new();
        let mut used = BTreeSet::new();
        for (action, spans) in [("remove", &change.removed), ("add", &change.added)] {
            for span in spans {
                ensure!(
                    change.reviewed_kinds.contains(&span.kind),
                    "native change outside its reviewed kinds"
                );
                let decisions: Vec<_> = self
                    .native_decisions
                    .iter()
                    .filter(|d| {
                        d.name == name
                            && d.decision == action
                            && &d.span == span
                            && d.input_sha256 == parent.input_sha256
                            && change.adjudication_ids.contains(&d.id)
                    })
                    .collect();
                ensure!(
                    decisions.len() == 1
                        && used.insert(decisions[0].id.clone())
                        && !decisions[0].basis.trim().is_empty(),
                    "native change lacks exactly one adjudicated decision"
                );
                transfers.push(NativeChange {
                    action: action.to_owned(),
                    adjudication_id: decisions[0].id.clone(),
                    basis: decisions[0].basis.clone(),
                    span: span.clone(),
                });
            }
        }
        ensure!(
            used.len() == change.adjudication_ids.len(),
            "amendment change cites adjudications it does not apply"
        );
        let original_expected: Vec<Span> = serde_json::from_value(original["expected"].clone())?;
        validate_spans(input, &original_expected)?;
        let mut successor_expected: Vec<Span> =
            serde_json::from_value(successor["expected"].clone())?;
        successor_expected.sort();
        validate_spans(input, &successor_expected)?;
        ensure!(
            apply(
                &original_expected,
                transfers.iter().map(|c| (c.action.as_str(), &c.span))
            )? == successor_expected,
            "successor native labels differ from original plus the application change"
        );
        parent.added = change.added.clone();
        parent.removed = change.removed.clone();
        parent.adjudication_ids = change.adjudication_ids.clone();
        Ok((parent, transfers))
    }
}

fn lines(bytes: &[u8]) -> anyhow::Result<Vec<&str>> {
    Ok(std::str::from_utf8(bytes)?.lines().collect())
}

fn same_except_expected(a: &Value, b: &Value) -> anyhow::Result<bool> {
    let mut a = a.as_object().context("row is not an object")?.clone();
    let mut b = b.as_object().context("row is not an object")?.clone();
    Ok(a.remove("expected").is_some() && b.remove("expected").is_some() && a == b)
}

/// Move a native span to the one literal copy that contains it; the bytes must be identical.
fn project(copies: &[Copy], input: &str, span: &Span) -> anyhow::Result<Span> {
    let (start, end) = (span.start as usize, span.end as usize);
    let copy = copies
        .iter()
        .find(|c| c.source_start <= start && end <= c.source_end)
        .context("native label change is not inside one literal copy")?;
    let width = end
        .checked_sub(start)
        .filter(|&n| n > 0)
        .context("empty native label change")?;
    let source_width = copy.source_end.checked_sub(copy.source_start);
    ensure!(
        source_width.is_some() && copy.output_end.checked_sub(copy.output_start) == source_width,
        "literal copy changes length"
    );
    let a = start
        .checked_sub(copy.source_start)
        .and_then(|offset| copy.output_start.checked_add(offset))
        .context("transferred offset out of range")?;
    let b = a
        .checked_add(width)
        .context("transferred offset out of range")?;
    ensure!(
        input.get(a..b) == Some(span.text.as_str()),
        "transferred label text differs at its authored occurrence"
    );
    Ok(Span {
        kind: span.kind.clone(),
        text: span.text.clone(),
        start: u32::try_from(a)?,
        end: u32::try_from(b)?,
    })
}

fn apply<'a>(
    original: &[Span],
    changes: impl IntoIterator<Item = (&'a str, &'a Span)>,
) -> anyhow::Result<Vec<Span>> {
    let mut out = original.to_vec();
    for (action, changed) in changes {
        match (action, out.iter().position(|span| span == changed)) {
            ("remove", Some(at)) => {
                out.remove(at);
            }
            ("add", None) => out.push(changed.clone()),
            _ => anyhow::bail!("native label change does not apply to its original labels"),
        }
    }
    out.sort();
    Ok(out)
}

/// Case rulings a transferred change contradicts; each must be recorded, never dropped.
fn supersessions(
    name: &str,
    native: &[NativeChange],
    transfers: &[SpanTransfer],
    case_decisions: &[Value],
) -> anyhow::Result<Vec<Superseded>> {
    let mut out = Vec::new();
    for (change, transfer) in native.iter().zip(transfers) {
        for case in case_decisions {
            if case["authored_name"] != name
                || serde_json::from_value::<Span>(case["span"].clone())? != transfer.authored
            {
                continue;
            }
            let decision = case["decision"].as_str().context("case decision absent")?;
            let contradicts = match transfer.action.as_str() {
                "add" => decision.starts_with("exclude_"),
                _ => decision.starts_with("retain_"),
            };
            ensure!(
                contradicts,
                "transferred change repeats an existing case ruling"
            );
            out.push(Superseded {
                authored_name: name.to_owned(),
                span: transfer.authored.clone(),
                superseded_decision: decision.to_owned(),
                superseded_by: transfer.adjudication_id.clone(),
                superseding_basis: change.basis.clone(),
            });
        }
    }
    Ok(out)
}
