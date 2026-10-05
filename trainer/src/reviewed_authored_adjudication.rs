//! Explicit original-pass differences and source-case adjudication for contact reflows.

use super::loader::{Dependencies, jsonl, pin, referenced_pins, validate_spans};
use super::{Manifest, Proposal, Row, Span};
use crate::reviewed_data::{Receipt, digest};
use anyhow::{Context, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Deserialize)]
#[serde(deny_unknown_fields)]
struct Delta {
    name: String,
    reviewer: String,
    action: String,
    span: Span,
    source_case_ruling: String,
    reason: String,
}

/// Every key is required by the strict schema; the ones not read are pinned for review.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct Adjudication {
    scope: String,
    status: String,
    documents: usize,
    case_decisions: Vec<Value>,
    pass_specific_deltas: Vec<Delta>,
    original_notes: BTreeMap<String, Vec<Notes>>,
    note_counts: BTreeMap<String, usize>,
    all388_note_dispositions: Vec<Value>,
    source_reviews: BTreeMap<std::path::PathBuf, String>,
    original_labels_and_native340_mutated: bool,
    annotation_policy_mutated: bool,
    policy_limits: Vec<String>,
    fit_allowed: bool,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Notes {
    name: String,
    uncertainties: Vec<Value>,
}

pub(super) fn construction_source(
    construction: &Value,
    manifest: &Manifest,
    root: &Path,
    dependencies: &mut Dependencies,
) -> anyhow::Result<Receipt> {
    ensure!(
        construction["all_original_source_bytes_retained_once_in_order"] == true
            && construction["development_used"] == false
            && construction["model_loaded"] == false,
        "authored reflow construction coverage/no-DEV/no-model absent"
    );
    referenced_pins(root, &construction["input_pins"], dependencies)?;
    let inputs = construction["input_pins"]
        .as_object()
        .context("construction input pins absent")?;
    ensure!(
        inputs
            .get(
                manifest
                    .policy
                    .path
                    .to_str()
                    .context("policy path not UTF8")?
            )
            .and_then(Value::as_str)
            == Some(manifest.policy.sha256.as_str()),
        "construction policy differs"
    );
    let found: Vec<_> = inputs
        .iter()
        .filter(|(path, _)| {
            Path::new(path)
                .file_name()
                .is_some_and(|name| name == "train-authoring-eight-source-bundles-c.jsonl")
        })
        .collect();
    ensure!(found.len() == 1, "full native bundle pin absent/ambiguous");
    let bundle_pin = Receipt {
        path: found[0].0.into(),
        sha256: found[0]
            .1
            .as_str()
            .context("bundle hash absent")?
            .to_owned(),
    };
    let bundles = jsonl::<Value>(&dependencies.read(&bundle_pin)?)?;
    let selected: Vec<_> = bundles
        .iter()
        .filter(|r| {
            r["name"]
                .as_str()
                .is_some_and(|name| name.starts_with("seen-real-"))
        })
        .collect();
    ensure!(
        selected.len() == 6 && manifest.documents == 24,
        "unsupported reflow source/document membership"
    );
    let names: Vec<_> = selected.iter().map(|r| r["name"].clone()).collect();
    ensure!(
        construction["source_documents"] == json!(names),
        "construction source membership differs from bundle"
    );
    let source: Receipt = serde_json::from_value(selected[0]["source_dataset"].clone())?;
    let bytes = dependencies.read(&source)?;
    let lines: Vec<_> = std::str::from_utf8(&bytes)?.lines().collect();
    for bundle in selected {
        ensure!(
            bundle["source_dataset"] == serde_json::to_value(&source)?,
            "reflow source bundles do not share declared prepared authority"
        );
        let index = usize::try_from(
            bundle["prepared_row_index"]
                .as_u64()
                .context("bundle source index absent")?,
        )?;
        let line = lines
            .get(index)
            .context("bundle source physical row absent")?;
        ensure!(
            bundle["prepared_raw_json_line_sha256"] == digest(line.as_bytes())
                && bundle["full_original_prepared_row"] == serde_json::from_str::<Value>(line)?
                && bundle["name"] == bundle["full_original_prepared_row"]["name"],
            "bundle full prepared/native row binding differs"
        );
    }
    ensure!(
        inputs
            .get(source.path.to_str().context("source path not UTF8")?)
            .and_then(Value::as_str)
            == Some(source.sha256.as_str()),
        "construction source authority absent"
    );
    Ok(source)
}

pub(super) fn verify_label_scopes(
    manifest: &Manifest,
    root: &Path,
    a: &Receipt,
    b: &Receipt,
    deps: &mut Dependencies,
) -> anyhow::Result<()> {
    let packet = pin(manifest, "blind-packet.jsonl")?;
    let scope: Value =
        serde_json::from_slice(&deps.read(&pin(manifest, "blind-labels-a-scope.json")?)?)?;
    let outputs = &scope["outputs"];
    let relative_a = a
        .path
        .strip_prefix(root)?
        .to_str()
        .context("A relative path not UTF8")?;
    ensure!(
        outputs[relative_a] == a.sha256
            && scope["complete_documents_read"].as_u64() == Some(manifest.documents as u64)
            && scope["complete_documents_labeled"].as_u64() == Some(manifest.documents as u64)
            && scope["independence"]["construction_maps_build_candidates_peer_proposals_not_read"]
                == true
            && scope["validation"]["all_input_name_country_fulltext_equal"] == true
            && scope["validation"]["all_UTF8_entity_and_note_quotes_valid"] == true,
        "original adjudicated A pass/output scope differs"
    );
    let references: Vec<Receipt> =
        serde_json::from_value(scope["consulted_for_this_pass"].clone())?;
    ensure!(
        references.len() == 2
            && references
                .iter()
                .any(|r| root.join(&r.path) == packet.path && r.sha256 == packet.sha256)
            && references
                .iter()
                .any(|r| root.join(&r.path) == manifest.policy.path
                    && r.sha256 == manifest.policy.sha256),
        "A packet/policy references differ"
    );
    let scope: Value = serde_json::from_slice(&deps.read(&pin(manifest, "blind-scope-b.json")?)?)?;
    ensure!(
        scope["status"] == "frozen_original_pass"
            && scope["packet_sha256"] == packet.sha256
            && scope["annotation_policy_sha256"] == manifest.policy.sha256
            && scope["proposal_sha256"] == b.sha256
            && scope["completed_full_text_reviews"].as_u64() == Some(manifest.documents as u64)
            && scope["independence"]["only_packet_policy_read_for_this_initial_pass"] == true
            && scope["independence"]["peer_root_labels_predictions_gold_maps_or_candidates_read_this_pass"]
                == false
            && scope["checks"]["exact_source_identity"] == true
            && scope["checks"]["utf8_all_slices"] == true
            && scope["checks"]["unique_spans"] == true
            && scope["checks"]["nonoverlapping_expected"] == true,
        "original adjudicated B pass/output scope differs"
    );
    let read_paths: Vec<std::path::PathBuf> = serde_json::from_value(scope["read_paths"].clone())?;
    ensure!(
        read_paths.len() == 2
            && read_paths.contains(&packet.path)
            && read_paths.contains(&manifest.policy.path),
        "B packet/policy paths differ"
    );
    Ok(())
}

fn proposal_identity(row: &Row, proposal: &Proposal) -> anyhow::Result<()> {
    ensure!(
        proposal
            .annotation_policy_sha256
            .as_ref()
            .is_none_or(|policy| policy == &row.annotation_policy_sha256),
        "original adjudicated proposal policy metadata differs from final row"
    );
    ensure!(
        row.name == proposal.name && row.country == proposal.country && row.input == proposal.input,
        "original pass ordered whole text/name differs"
    );
    validate_spans(&proposal.input, &proposal.expected)
}

fn original_notes(
    adjudication: &Adjudication,
    a: &[Proposal],
    b: &[Proposal],
) -> anyhow::Result<()> {
    ensure!(
        adjudication.original_notes.len() == 2 && adjudication.note_counts.len() == 2,
        "original note reviewer membership differs"
    );
    let mut expected = BTreeMap::new();
    for (reviewer, proposals) in [("a", a), ("b", b)] {
        let notes = adjudication
            .original_notes
            .get(reviewer)
            .context("original reviewer notes absent")?;
        ensure!(
            notes.len() == proposals.len(),
            "original note document membership differs"
        );
        let mut count = 0;
        for (row, notes) in proposals.iter().zip(notes) {
            ensure!(
                notes.name == row.name && notes.uncertainties == row.uncertainties,
                "original note payload/order differs"
            );
            for (index, note) in row.uncertainties.iter().enumerate() {
                let start = usize::try_from(
                    note["start"]
                        .as_u64()
                        .context("original note start absent")?,
                )?;
                let end =
                    usize::try_from(note["end"].as_u64().context("original note end absent")?)?;
                ensure!(
                    start < end && row.input.get(start..end) == note["text"].as_str(),
                    "original note literal UTF8 quote differs"
                );
                expected.insert((reviewer.to_owned(), row.name.clone(), index), note.clone());
                count += 1;
            }
        }
        ensure!(
            adjudication.note_counts.get(reviewer) == Some(&count),
            "original note count differs"
        );
    }
    let mut observed = BTreeMap::new();
    for disposition in &adjudication.all388_note_dispositions {
        let reviewer = disposition["reviewer"]
            .as_str()
            .context("note reviewer absent")?;
        let name = disposition["name"]
            .as_str()
            .context("note case name absent")?;
        let index = usize::try_from(
            disposition["original_note_index"]
                .as_u64()
                .context("note original index absent")?,
        )?;
        let key = (reviewer.to_owned(), name.to_owned(), index);
        ensure!(
            disposition["disposition"]
                .as_str()
                .is_some_and(|s| !s.trim().is_empty())
                && expected.get(&key) == Some(&disposition["original_note"])
                && observed
                    .insert(key, disposition["original_note"].clone())
                    .is_none(),
            "missing/mutated/duplicate note disposition"
        );
    }
    ensure!(
        expected == observed && expected.len() == 388,
        "adjudication does not preserve and close all388 original notes"
    );
    Ok(())
}

pub(super) fn verify_final_deltas(
    adjudication_value: &Value,
    rows: &[Row],
    a: &[Proposal],
    b: &[Proposal],
) -> anyhow::Result<()> {
    let adjudication: Adjudication = serde_json::from_value(adjudication_value.clone())?;
    ensure!(
        adjudication.status == "resolved_scope_limited"
            && adjudication.documents == rows.len()
            && !adjudication.original_labels_and_native340_mutated
            && !adjudication.annotation_policy_mutated,
        "adjudication unresolved or mutates original authority"
    );
    ensure!(
        rows.len() == a.len() && rows.len() == b.len(),
        "adjudicated pass membership differs"
    );
    original_notes(&adjudication, a, b)?;
    let mut needed = BTreeSet::new();
    for ((row, left), right) in rows.iter().zip(a).zip(b) {
        validate_spans(&row.input, &row.expected)?;
        let final_spans: BTreeSet<_> = row.expected.iter().cloned().collect();
        for (reviewer, proposed) in [("a", left), ("b", right)] {
            proposal_identity(row, proposed)?;
            let spans: BTreeSet<_> = proposed.expected.iter().cloned().collect();
            for (action, differences) in [
                (
                    "exclude",
                    spans.difference(&final_spans).cloned().collect::<Vec<_>>(),
                ),
                (
                    "include",
                    final_spans.difference(&spans).cloned().collect::<Vec<_>>(),
                ),
            ] {
                for span in differences {
                    needed.insert((
                        row.name.clone(),
                        reviewer.to_owned(),
                        action.to_owned(),
                        span,
                    ));
                }
            }
        }
    }
    let mut decisions = BTreeMap::new();
    for case in &adjudication.case_decisions {
        let name = case["authored_name"]
            .as_str()
            .context("source ruling case absent")?;
        let span: Span = serde_json::from_value(case["span"].clone())?;
        let row = rows
            .iter()
            .find(|r| r.name == name)
            .context("source ruling unknown case")?;
        validate_spans(&row.input, std::slice::from_ref(&span))?;
        let decision = case["decision"]
            .as_str()
            .context("source decision absent")?;
        let included = match decision {
            "exclude_generic_anaphoric_reference" | "exclude_generic_job_title_reference" => false,
            "retain_established_distinctive_proper_short_name" => true,
            _ => anyhow::bail!("unsupported/unresolved source case ruling"),
        };
        ensure!(
            row.expected.contains(&span) == included
                && case["native_source_document"] == row.authored_source.native_name
                && case["policy_reason"]
                    .as_str()
                    .is_some_and(|reason| !reason.trim().is_empty()),
            "source case ruling/final native identity differs"
        );
        let index = rows
            .iter()
            .position(|r| r.name == name)
            .context("source case index absent")?;
        ensure!(
            case["included_in_A"] == a[index].expected.contains(&span)
                && case["included_in_B"] == b[index].expected.contains(&span),
            "source ruling original pass inclusion differs"
        );
        ensure!(
            decisions.insert((name.to_owned(), span), case).is_none(),
            "duplicate source case ruling"
        );
    }
    let mut actual = BTreeSet::new();
    for delta in &adjudication.pass_specific_deltas {
        let ruling = decisions
            .get(&(delta.name.clone(), delta.span.clone()))
            .context("pass delta lacks exact source case ruling")?;
        ensure!(
            ruling["decision"] == delta.source_case_ruling
                && ruling["policy_reason"] == delta.reason
                && actual.insert((
                    delta.name.clone(),
                    delta.reviewer.clone(),
                    delta.action.clone(),
                    delta.span.clone()
                )),
            "duplicate/different pass delta or reason"
        );
    }
    ensure!(
        actual == needed
            && actual.len() == 20
            && decisions.len() == 16
            && decisions
                .keys()
                .all(|(name, span)| needed.iter().any(|(n, _, _, s)| n == name && s == span)),
        "final20 pass differences are not closed exactly by16 actual source case rulings"
    );
    Ok(())
}

pub(super) fn verify(
    manifest: &Manifest,
    root: &Path,
    rows: &[Row],
    a: &[Proposal],
    b: &[Proposal],
    deps: &mut Dependencies,
) -> anyhow::Result<Value> {
    let adjudication_pin = pin(manifest, "final-adjudication-root.json")?;
    let adjudication_value: Value = serde_json::from_slice(&deps.read(&adjudication_pin)?)?;
    verify_final_deltas(&adjudication_value, rows, a, b)?;
    let a_pin = pin(manifest, "independent-source-discrepancy-review-a.json")?;
    let b_pin = pin(
        manifest,
        "independent-source-semantic-transfer-review-b.json",
    )?;
    let review_a: Value = serde_json::from_slice(&deps.read(&a_pin)?)?;
    let review_b: Value = serde_json::from_slice(&deps.read(&b_pin)?)?;
    let review_pins = BTreeMap::from([
        (a_pin.path.clone(), a_pin.sha256.clone()),
        (b_pin.path.clone(), b_pin.sha256.clone()),
    ]);
    ensure!(
        adjudication_value["source_reviews"] == serde_json::to_value(review_pins)?,
        "final adjudication does not bind both actual source reviews"
    );
    ensure!(
        review_a["status"] == "source_case_rulings_recommended"
            && review_b["status"]
                == "passed_source_semantic_transfer_with_explicit_case_local_rulings"
            && review_a["blockers"].as_array().is_some_and(Vec::is_empty)
            && review_b["blockers"].as_array().is_some_and(Vec::is_empty)
            && review_a["complete_original_six_bundles_verified"] == true
            && review_a["all24_full_source_byte_copy_maps_verified"] == true
            && review_b["complete_original_six_source_texts_read"] == true
            && review_b["inputs_unchanged"] == true
            && review_b["native_or_original_proposals_mutated"] == false,
        "passed independent source-review closure absent"
    );
    ensure!(
        review_a["discrepancies"] == adjudication_value["case_decisions"]
            && review_b["all_original_notes"] == adjudication_value["all388_note_dispositions"],
        "adjudication case/note rulings differ from actual independent source reviews"
    );
    referenced_pins(root, &review_a["consulted_pins"], deps)?;
    referenced_pins(root, &review_b["input_pins"], deps)?;
    let assembly = review_b["assembly_checks"]
        .as_array()
        .context("source semantic assembly rows absent")?;
    ensure!(
        assembly.len() == rows.len(),
        "source semantic assembly membership differs"
    );
    for (row, check) in rows.iter().zip(assembly) {
        let mut supported: Vec<Span> =
            serde_json::from_value(check["independently_supported_expected"].clone())?;
        supported.sort();
        let mut final_expected = row.expected.clone();
        final_expected.sort();
        ensure!(
            check["name"] == row.name
                && check["source_name"] == row.authored_source.native_name
                && check["source_prepared_row_index"].as_u64()
                    == Some(row.authored_source.native_row_index as u64)
                && check["source_text_sha256"] == row.authored_source.native_text_sha256
                && check["input_sha256"] == digest(row.input.as_bytes())
                && check["source_all_bytes_once_in_order"] == true
                && check["all_inserted_bytes_reconstructed"] == true
                && check["projected_spans_source_bound"] == true
                && supported == final_expected,
            "final row differs from independent complete-source semantic review"
        );
    }
    let maps = jsonl::<Value>(&deps.read(&pin(manifest, "literal-copy-map.jsonl")?)?)?;
    for case in adjudication_value["case_decisions"]
        .as_array()
        .context("source cases absent")?
    {
        let name = case["authored_name"]
            .as_str()
            .context("source case name absent")?;
        let row = rows
            .iter()
            .find(|row| row.name == name)
            .context("source case row absent")?;
        let matching: Vec<_> = maps.iter().filter(|map| map["name"] == name).collect();
        ensure!(matching.len() == 1, "source case map absent/ambiguous");
        let source_bytes = deps.read(&row.authored_source.native_dataset)?;
        let source_rows = jsonl::<Value>(&source_bytes)?;
        let source = source_rows
            .get(row.authored_source.native_row_index)
            .context("case source row absent")?["input"]
            .as_str()
            .context("case source text absent")?;
        let start = usize::try_from(
            case["native_quote_start"]
                .as_u64()
                .context("native case quote start absent")?,
        )?;
        let end = usize::try_from(
            case["native_quote_end"]
                .as_u64()
                .context("native case quote end absent")?,
        )?;
        let native_quote = source
            .get(start..end)
            .context("native source case quote not UTF8 aligned")?;
        ensure!(
            start < end
                && case["native_context"].as_str().is_some_and(
                    |context| source.contains(context) && context.contains(native_quote)
                ),
            "source case quoted context differs from complete native text"
        );
        let copies = matching[0]["literal_copy_map"]
            .as_array()
            .context("case source copies absent")?;
        let mapped = |at: usize| -> anyhow::Result<usize> {
            for copy in copies {
                let a = usize::try_from(
                    copy["source_start"]
                        .as_u64()
                        .context("copy source start absent")?,
                )?;
                let b = usize::try_from(
                    copy["source_end"]
                        .as_u64()
                        .context("copy source end absent")?,
                )?;
                if a <= at && at < b {
                    return Ok(usize::try_from(
                        copy["output_start"]
                            .as_u64()
                            .context("copy output start absent")?,
                    )? + at
                        - a);
                }
            }
            anyhow::bail!("source case quote not copied")
        };
        let span: Span = serde_json::from_value(case["span"].clone())?;
        ensure!(
            mapped(start)? == span.start as usize && mapped(end - 1)? + 1 == span.end as usize,
            "source case quote does not map to exact adjudicated occurrence"
        );
    }
    let notes_a: Value =
        serde_json::from_slice(&deps.read(&pin(manifest, "blind-notes-a.json")?)?)?;
    let notes_b: Value =
        serde_json::from_slice(&deps.read(&pin(manifest, "blind-notes-b.json")?)?)?;
    ensure!(
        notes_a["notes"] == adjudication_value["original_notes"]["a"]
            && notes_b["notes"] == adjudication_value["original_notes"]["b"],
        "final adjudication changes original notes files"
    );
    let mut needed = BTreeSet::new();
    for delta in adjudication_value["pass_specific_deltas"]
        .as_array()
        .context("deltas absent")?
    {
        let span: Span = serde_json::from_value(delta["span"].clone())?;
        needed.insert((
            delta["name"]
                .as_str()
                .context("delta name absent")?
                .to_owned(),
            delta["reviewer"]
                .as_str()
                .context("delta reviewer absent")?
                .to_owned(),
            delta["action"]
                .as_str()
                .context("delta action absent")?
                .to_owned(),
            span,
        ));
    }
    let mut observed = BTreeSet::new();
    for delta in review_b["proposal_to_supported_final_deltas"]
        .as_array()
        .context("independent B deltas absent")?
    {
        ensure!(
            delta["multiplicity"] == 1,
            "independent delta multiplicity differs"
        );
        let span: Span = serde_json::from_value(delta["span"].clone())?;
        ensure!(
            observed.insert((
                delta["name"]
                    .as_str()
                    .context("B delta name absent")?
                    .to_owned(),
                delta["reviewer"]
                    .as_str()
                    .context("B delta reviewer absent")?
                    .to_owned(),
                delta["operation"]
                    .as_str()
                    .context("B delta operation absent")?
                    .to_owned(),
                span
            )),
            "duplicate independent B delta"
        );
    }
    ensure!(
        needed == observed,
        "final adjudication changes independently supported pass-specific deltas"
    );
    Ok(
        json!({"receipt":adjudication_pin,"original_pass_deltas":20,"source_case_decisions":16,"all_original_notes":388,
        "blind_agreement_claimed":false,"semantic_authority":"explicit root adjudication bound to two independent post-blind source-case reviews"}),
    )
}

pub(super) fn verify_bundle_maps(
    construction: &Value,
    maps: &[Value],
    deps: &mut Dependencies,
) -> anyhow::Result<()> {
    let inputs = construction["input_pins"]
        .as_object()
        .context("construction inputs absent")?;
    let (path, sha) = inputs
        .iter()
        .find(|(path, _)| {
            Path::new(path)
                .file_name()
                .is_some_and(|name| name == "train-authoring-eight-source-bundles-c.jsonl")
        })
        .context("bundle pin absent")?;
    let bundles = jsonl::<Value>(&deps.read(&Receipt {
        path: path.into(),
        sha256: sha.as_str().context("bundle hash absent")?.to_owned(),
    })?)?;
    for map in maps {
        let matching: Vec<_> = bundles
            .iter()
            .filter(|b| b["name"] == map["native_source_document"])
            .collect();
        ensure!(
            matching.len() == 1,
            "reflow map native bundle absent/ambiguous"
        );
        let bundle = matching[0];
        ensure!(
            map["source_dataset"] == bundle["source_dataset"]
                && map["source_row_index"] == bundle["prepared_row_index"]
                && map["source_row_sha256"] == bundle["prepared_raw_json_line_sha256"]
                && map["source_context_evidence"] == bundle["source_context_evidence"]
                && map["forbidden_inferences"] == bundle["forbidden_inferences"],
            "reflow map changes original source bundle identity/context/limits"
        );
    }
    Ok(())
}
