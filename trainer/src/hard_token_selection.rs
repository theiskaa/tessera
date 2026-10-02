//! Test-only resolution of frozen byte selectors on complete native training rows.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tessera::internal::{FeatureConfig, flag};

use crate::detector::{DetectorDoc, annotated_document};

const GROUP_NAMES: [&str; 3] = ["person", "org_contrast", "address"];
const CONFLICT_POLICY: &str = "deduplicate within a group; omit optional boundary O shared by groups; reject overlapping core targets";

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Pin {
    path: PathBuf,
    sha256: String,
}

impl Pin {
    fn bytes(&self) -> anyhow::Result<Vec<u8>> {
        ensure!(
            self.path.is_absolute(),
            "selector evidence path must be absolute"
        );
        let bytes = std::fs::read(&self.path)?;
        ensure!(
            crate::export::sha256_hex(&bytes) == self.sha256,
            "selector evidence changed"
        );
        Ok(bytes)
    }
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Target {
    group: String,
    start: usize,
    end: usize,
    text: String,
    expected: String,
    reason: String,
    adjacent_o_tokens_requested: usize,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Record {
    role: String,
    name: String,
    text_sha256: String,
    expected_sha256: String,
    native_index_reference_only: usize,
    full_original_parent_required: bool,
    targets: Vec<Target>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Draft {
    scope: String,
    groups: Vec<String>,
    lambda_each_proposal: f64,
    records: Vec<Record>,
    evidence: Vec<Pin>,
    model_operations: usize,
    native_token_resolution_passed: bool,
    training_ready: bool,
    boundary_o_conflict_policy: String,
    held_cases: Vec<String>,
    must_preserve: Vec<String>,
}

impl Draft {
    fn validate(&self) -> anyhow::Result<()> {
        ensure!(
            self.scope
                == "static train-only byte selectors, not a native mask or fit authorization"
                && self.groups == GROUP_NAMES
                && self.boundary_o_conflict_policy == CONFLICT_POLICY,
            "selector draft policy differs"
        );
        ensure!(
            self.lambda_each_proposal.is_finite()
                && (0.0..=1.0 / 16.0).contains(&self.lambda_each_proposal),
            "invalid selector lambda"
        );
        ensure!(
            self.model_operations == 0
                && !self.native_token_resolution_passed
                && !self.training_ready,
            "selector draft cannot authorize fitting"
        );
        ensure!(
            !self.held_cases.is_empty() && !self.must_preserve.is_empty(),
            "selector limitations absent"
        );
        let mut paths = BTreeSet::new();
        for pin in &self.evidence {
            ensure!(
                paths.insert(pin.path.clone()),
                "duplicate selector evidence pin"
            );
            pin.bytes()?;
        }
        let mut identities = BTreeSet::new();
        let mut indices = BTreeSet::new();
        for record in &self.records {
            ensure!(
                identities.insert(record.name.clone())
                    && indices.insert(record.native_index_reference_only),
                "selector row identity collision"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceRow {
    name: String,
    country: String,
    input: String,
    expected: Vec<Value>,
}

fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(map) => {
            let sorted: BTreeMap<_, _> = map
                .iter()
                .map(|(key, value)| (key.clone(), canonical(value)))
                .collect();
            Value::Object(sorted.into_iter().collect())
        }
        Value::Array(values) => Value::Array(values.iter().map(canonical).collect()),
        value => value.clone(),
    }
}

fn expected_hash(expected: &[Value]) -> anyhow::Result<String> {
    Ok(crate::export::sha256_hex(&serde_json::to_vec(&canonical(
        &json!(expected),
    ))?))
}

fn load_rows(pin: &Pin) -> anyhow::Result<BTreeMap<String, SourceRow>> {
    let bytes = pin.bytes()?;
    let mut rows = BTreeMap::new();
    for line in std::str::from_utf8(&bytes)?
        .lines()
        .filter(|line| !line.trim().is_empty())
    {
        let row: SourceRow = serde_json::from_str(line)?;
        ensure!(
            rows.insert(row.name.clone(), row).is_none(),
            "duplicate selector source name"
        );
    }
    Ok(rows)
}

#[derive(Serialize)]
struct Resolved {
    name: String,
    native_index_reference_only: usize,
    content_tokens: usize,
    native_label_counts: [usize; 7],
    selected_indices: [Vec<usize>; 3],
    group_denominators: [f64; 3],
    binary_masks_sha256: String,
    omitted_shared_optional_o: Vec<usize>,
    text_sha256: String,
    expected_sha256: String,
    native_encoding_sha256: String,
    native_token_spans: Vec<(u32, u32)>,
    selected_tokens: Vec<Value>,
}

fn resolve(
    record: &Record,
    row: &SourceRow,
    reference_index: usize,
    fc: &FeatureConfig,
    weights: &[f32; 7],
) -> anyhow::Result<(Resolved, [Vec<f32>; 3], DetectorDoc)> {
    ensure!(
        record.full_original_parent_required
            && record.name == row.name
            && record.native_index_reference_only == reference_index,
        "selector parent identity differs"
    );
    let prefix = match record.role.as_str() {
        "real" => "seen-real-",
        "authored" => "seen-authored-",
        _ => anyhow::bail!("unknown selector source role"),
    };
    ensure!(
        row.name.starts_with(prefix) && row.country == "US",
        "selector source role or country differs"
    );
    ensure!(
        crate::export::sha256_hex(row.input.as_bytes()) == record.text_sha256
            && expected_hash(&row.expected)? == record.expected_sha256,
        "selector text or complete gold changed"
    );
    ensure!(
        weights
            .iter()
            .all(|&weight| weight.is_finite() && weight > 0.0),
        "invalid selector class weights"
    );
    ensure!(!record.targets.is_empty(), "selector has no targets");
    for entity in &row.expected {
        let start = entity["start"].as_u64().context("gold start absent")? as usize;
        let end = entity["end"].as_u64().context("gold end absent")? as usize;
        let surface = row
            .input
            .get(start..end)
            .filter(|_| start < end)
            .context("invalid complete gold span")?;
        ensure!(entity["text"] == surface, "complete gold surface differs");
    }
    let doc = annotated_document(
        &row.input,
        &row.country,
        &serde_json::to_string(&row.expected)?,
        fc,
    )?;
    ensure!(
        !doc.enc.labels.is_empty() && doc.enc.labels.len() <= 900,
        "full original row exceeds native training limit"
    );
    let length = doc.enc.labels.len();
    let mut core: [BTreeSet<usize>; 3] = std::array::from_fn(|_| BTreeSet::new());
    let mut optional: [BTreeSet<usize>; 3] = std::array::from_fn(|_| BTreeSet::new());
    for target in &record.targets {
        let group = GROUP_NAMES
            .iter()
            .position(|&name| name == target.group)
            .context("unknown selector group")?;
        ensure!(
            target.adjacent_o_tokens_requested <= 1 && !target.reason.is_empty(),
            "invalid selector boundary request"
        );
        let surface = row
            .input
            .get(target.start..target.end)
            .filter(|_| target.start < target.end)
            .context("selector span is outside text or UTF-8 boundaries")?;
        ensure!(surface == target.text, "selector surface differs");
        let first = doc
            .enc
            .token_spans
            .iter()
            .position(|&(start, _)| start as usize == target.start)
            .context("selector start is not a native token boundary")?;
        let last = doc
            .enc
            .token_spans
            .iter()
            .rposition(|&(_, end)| end as usize == target.end)
            .context("selector end is not a native token boundary")?;
        ensure!(first <= last, "selector token range reversed");
        let expected_kind = match target.expected.as_str() {
            "O" => None,
            "person" if group == 0 => Some(0),
            "org" if group == 1 => Some(1),
            "address" if group == 2 => Some(2),
            _ => anyhow::bail!("selector kind differs from group"),
        };
        if let Some(kind) = expected_kind {
            ensure!(
                doc.gold.iter().any(|span| span.kind == kind
                    && span.start as usize == target.start
                    && span.end as usize == target.end),
                "positive selector is not a complete native gold entity"
            );
            for index in first..=last {
                let expected = 1 + 2 * kind + usize::from(index != first);
                ensure!(
                    usize::from(doc.enc.labels[index]) == expected,
                    "native selector BIO label differs"
                );
            }
        } else {
            for entity in &row.expected {
                let start = entity["start"].as_u64().context("gold start absent")? as usize;
                let end = entity["end"].as_u64().context("gold end absent")? as usize;
                ensure!(
                    target.end <= start || end <= target.start,
                    "O selector overlaps complete gold"
                );
            }
            ensure!(
                target.adjacent_o_tokens_requested == 0,
                "negative selector cannot request boundary tokens"
            );
            ensure!(
                doc.enc.labels[first..=last].iter().all(|&label| label == 0),
                "O selector has nonzero native label"
            );
        }
        for index in first..=last {
            ensure!(
                doc.enc.flags[index] & flag::IN_RULE_SPAN == 0,
                "selector overlaps a deterministic rule span"
            );
            core[group].insert(index);
        }
        if target.adjacent_o_tokens_requested == 1 {
            for index in first
                .checked_sub(1)
                .into_iter()
                .chain((last + 1 < length).then_some(last + 1))
            {
                let token = doc.enc.token_spans[index];
                let overlaps_declared = row.expected.iter().any(|entity| {
                    entity["start"]
                        .as_u64()
                        .zip(entity["end"].as_u64())
                        .is_some_and(|(start, end)| {
                            u64::from(token.0) < end && start < u64::from(token.1)
                        })
                });
                if doc.enc.labels[index] == 0
                    && doc.enc.flags[index] & flag::IN_RULE_SPAN == 0
                    && !overlaps_declared
                {
                    optional[group].insert(index);
                }
            }
        }
    }
    let mut omitted = Vec::new();
    let mut masks: [Vec<f32>; 3] = std::array::from_fn(|_| vec![0.; length]);
    for (index, _) in doc.enc.labels.iter().enumerate() {
        let owners: Vec<_> = (0..3)
            .filter(|&group| core[group].contains(&index))
            .collect();
        ensure!(owners.len() <= 1, "overlapping core selector groups");
        let boundaries: Vec<_> = (0..3)
            .filter(|&group| optional[group].contains(&index))
            .collect();
        if let Some(&owner) = owners.first() {
            ensure!(
                boundaries.iter().all(|&group| group == owner),
                "core selector collides with another group boundary"
            );
            masks[owner][index] = 1.;
        } else if boundaries.len() == 1 {
            masks[boundaries[0]][index] = 1.;
        } else if boundaries.len() > 1 {
            omitted.push(index);
        }
    }
    crate::hard_token_masks::HardMasks::new(
        [1, length],
        &vec![1.; length],
        masks.clone(),
        [1. / 16.; 3],
    )?;
    let selected_indices: [Vec<usize>; 3] = std::array::from_fn(|group| {
        (0..length)
            .filter(|&index| masks[group][index] == 1.)
            .collect()
    });
    let group_denominators = std::array::from_fn(|group| {
        (0..length)
            .filter(|&index| masks[group][index] == 1.)
            .map(|index| f64::from(weights[doc.enc.labels[index] as usize]))
            .sum()
    });
    let mut counts = [0; 7];
    for &label in &doc.enc.labels {
        counts[label as usize] += 1;
    }
    let encoding = json!({"spans":doc.enc.token_spans,"labels":doc.enc.labels,"ngrams":doc.enc.ngram_ids,
        "script":doc.enc.script,"shape":doc.enc.shape,"flags":doc.enc.flags,"breaks":doc.breaks,"country":doc.enc.country});
    let selected_tokens = (0..3).flat_map(|group| selected_indices[group].iter().map(move |&index| (group, index)))
        .map(|(group,index)| json!({"index":index,"start":doc.enc.token_spans[index].0,"end":doc.enc.token_spans[index].1,
            "native_label":doc.enc.labels[index],"group":GROUP_NAMES[group],"core":core[group].contains(&index),
            "optional_boundary":!core[group].contains(&index)})).collect();
    let proof = Resolved {
        name: row.name.clone(),
        native_index_reference_only: reference_index,
        content_tokens: length,
        native_label_counts: counts,
        selected_indices,
        group_denominators,
        binary_masks_sha256: crate::export::sha256_hex(&serde_json::to_vec(&masks)?),
        omitted_shared_optional_o: omitted,
        text_sha256: record.text_sha256.clone(),
        expected_sha256: record.expected_sha256.clone(),
        native_encoding_sha256: crate::export::sha256_hex(&serde_json::to_vec(&encoding)?),
        native_token_spans: doc.enc.token_spans.clone(),
        selected_tokens,
    };
    Ok((proof, masks, doc))
}

fn reference_indices(
    preview: &Value,
    real: &BTreeMap<String, SourceRow>,
) -> anyhow::Result<BTreeMap<String, usize>> {
    let boundary = preview["full_synthetic_boundary"]
        .as_u64()
        .context("native boundary absent")? as usize;
    ensure!(boundary == 170569, "historical native boundary differs");
    let mut references = BTreeMap::new();
    for row in preview["typed_synthetic"]["authored_documents"]
        .as_array()
        .context("authored native map absent")?
    {
        let identity = &row["identity"];
        ensure!(
            identity["origin"] == "synthetic_authored" && identity["split"] == "train",
            "native map is not train-only"
        );
        let name = format!(
            "seen-authored-{}",
            identity["id"]
                .as_str()
                .context("authored native id absent")?
        );
        let index = identity["native_index"]
            .as_u64()
            .context("authored native index absent")? as usize;
        ensure!(
            references.insert(name, index).is_none(),
            "duplicate native authored reference"
        );
    }
    let pieces: Vec<_> = preview["silver"]["source_pieces"]
        .as_array()
        .context("source prefixes absent")?
        .iter()
        .map(|count| {
            count
                .as_u64()
                .map(|n| n as usize)
                .context("invalid source count")
        })
        .collect::<Result<_, _>>()?;
    for name in real.keys() {
        let parts: Vec<_> = name.split('-').collect();
        ensure!(
            parts.len() == 4 && parts[0] == "seen" && parts[1] == "real",
            "real native reference name invalid"
        );
        let source: usize = parts[2].parse()?;
        let row: usize = parts[3].parse()?;
        ensure!(
            source < pieces.len() && row < pieces[source],
            "real native reference outside source"
        );
        let prefix = pieces[..source].iter().try_fold(boundary, |sum, &count| {
            sum.checked_add(count).context("native prefix overflow")
        })?;
        ensure!(
            references
                .insert(
                    name.clone(),
                    prefix.checked_add(row).context("native index overflow")?
                )
                .is_none(),
            "duplicate native real reference"
        );
    }
    Ok(references)
}

#[cfg(test)]
fn verify_unchanged_inputs(prior: &DetectorDoc, current: &DetectorDoc) -> anyhow::Result<()> {
    ensure!(
        prior.text == current.text && prior.breaks == current.breaks,
        "successor changed full original text or native breaks"
    );
    ensure!(
        prior.enc.labels.len() == prior.enc.token_spans.len()
            && current.enc.labels.len() == current.enc.token_spans.len(),
        "native label dimensions differ"
    );
    let mut original_inputs = prior.enc.clone();
    original_inputs.labels = current.enc.labels.clone();
    ensure!(
        original_inputs == current.enc,
        "successor changed native inputs beyond corrected labels"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const WEIGHTS: [f32; 7] = [1., 3., 2., 3., 2., 2., 1.5];

    fn fixture() -> (Record, SourceRow) {
        let row = SourceRow {
            name: "seen-real-00-0000".into(),
            country: "US".into(),
            input: "Émilie visits Harbor Bureau today".into(),
            expected: vec![
                json!({"kind":"person","start":0,"end":7,"text":"Émilie"}),
                json!({"kind":"org","start":15,"end":28,"text":"Harbor Bureau"}),
            ],
        };
        let record = Record {
            role: "real".into(),
            name: row.name.clone(),
            text_sha256: crate::export::sha256_hex(row.input.as_bytes()),
            expected_sha256: expected_hash(&row.expected).unwrap(),
            native_index_reference_only: 170569,
            full_original_parent_required: true,
            targets: vec![
                Target {
                    group: "person".into(),
                    start: 0,
                    end: 7,
                    text: "Émilie".into(),
                    expected: "person".into(),
                    reason: "fixture".into(),
                    adjacent_o_tokens_requested: 1,
                },
                Target {
                    group: "org_contrast".into(),
                    start: 15,
                    end: 28,
                    text: "Harbor Bureau".into(),
                    expected: "org".into(),
                    reason: "fixture".into(),
                    adjacent_o_tokens_requested: 1,
                },
            ],
        };
        (record, row)
    }

    fn run(
        record: &Record,
        row: &SourceRow,
    ) -> anyhow::Result<(Resolved, [Vec<f32>; 3], DetectorDoc)> {
        resolve(record, row, 170569, &FeatureConfig::default(), &WEIGHTS)
    }

    #[test]
    fn complete_utf8_rows_preserve_native_inputs_deduplicate_and_omit_shared_boundary() {
        let (mut record, row) = fixture();
        let original = annotated_document(
            &row.input,
            "US",
            &serde_json::to_string(&row.expected).unwrap(),
            &FeatureConfig::default(),
        )
        .unwrap();
        let (proof, masks, doc) = run(&record, &row).unwrap();
        assert_eq!(doc.enc, original.enc);
        assert_eq!(doc.text, row.input);
        assert_eq!(proof.omitted_shared_optional_o, vec![1]);
        assert_eq!(proof.selected_indices, [vec![0], vec![2, 3, 4], vec![]]);
        assert_eq!(proof.group_denominators, [3., 6., 0.]);
        record.targets.push(record.targets[0].clone());
        assert_eq!(run(&record, &row).unwrap().1, masks);
    }

    #[test]
    fn negative_selectors_require_o_gold_and_exact_token_alignment() {
        let (mut record, row) = fixture();
        record.targets = vec![Target {
            group: "address".into(),
            start: 29,
            end: 34,
            text: "today".into(),
            expected: "O".into(),
            reason: "negative".into(),
            adjacent_o_tokens_requested: 0,
        }];
        assert_eq!(run(&record, &row).unwrap().0.selected_indices[2], vec![4]);
        record.targets[0].start = 30;
        record.targets[0].text = "oday".into();
        assert!(run(&record, &row).is_err());
        record.targets[0].start = 0;
        record.targets[0].end = 7;
        record.targets[0].text = "Émilie".into();
        assert!(run(&record, &row).is_err());
    }

    #[test]
    fn optional_boundaries_exclude_declared_phone_and_email_missed_by_rules() {
        for kind in ["phone", "email"] {
            let (mut record, mut row) = fixture();
            record.targets.truncate(1);
            let original = run(&record, &row).unwrap();
            assert_eq!(original.0.selected_indices[0], vec![0, 1]);
            assert_eq!(original.2.enc.labels[1], 0);
            assert_eq!(original.2.enc.flags[1] & flag::IN_RULE_SPAN, 0);
            row.expected
                .push(json!({"kind":kind,"start":8,"end":14,"text":"visits"}));
            record.expected_sha256 = expected_hash(&row.expected).unwrap();
            assert_eq!(run(&record, &row).unwrap().0.selected_indices[0], vec![0]);
        }
    }

    #[test]
    fn altered_parent_spans_kinds_groups_and_core_collisions_refuse() {
        let (record, row) = fixture();
        for field in [
            "role", "name", "text", "gold", "index", "full", "group", "kind", "span", "surface",
            "adjacent",
        ] {
            let mut bad = record.clone();
            match field {
                "role" => bad.role = "test".into(),
                "name" => bad.name.push('x'),
                "text" => bad.text_sha256.clear(),
                "gold" => bad.expected_sha256.clear(),
                "index" => bad.native_index_reference_only += 1,
                "full" => bad.full_original_parent_required = false,
                "group" => bad.targets[0].group = "unknown".into(),
                "kind" => bad.targets[0].expected = "org".into(),
                "span" => bad.targets[0].start = 1,
                "surface" => bad.targets[0].text.clear(),
                _ => bad.targets[0].adjacent_o_tokens_requested = 2,
            }
            assert!(run(&bad, &row).is_err(), "{field}");
        }
        let mut bad = record;
        let o = Target {
            group: "person".into(),
            start: 8,
            end: 14,
            text: "visits".into(),
            expected: "O".into(),
            reason: "fixture".into(),
            adjacent_o_tokens_requested: 0,
        };
        bad.targets.push(o.clone());
        let mut other = o;
        other.group = "address".into();
        bad.targets.push(other);
        assert!(run(&bad, &row).is_err());
    }

    #[test]
    fn pinned_sources_unknown_draft_fields_and_invalid_lambdas_refuse() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("source.jsonl");
        std::fs::write(&path, b"original").unwrap();
        let pin = Pin {
            path,
            sha256: crate::export::sha256_hex(b"original"),
        };
        assert!(pin.bytes().is_ok());
        std::fs::write(&pin.path, b"changed").unwrap();
        assert!(pin.bytes().is_err());
        assert!(serde_json::from_value::<Target>(json!({"group":"person","start":0,"end":1,"text":"x","expected":"person","reason":"x","adjacent_o_tokens_requested":0,"unknown":true})).is_err());
        let mut draft = Draft {
            scope: "static train-only byte selectors, not a native mask or fit authorization"
                .into(),
            groups: GROUP_NAMES.map(str::to_owned).to_vec(),
            lambda_each_proposal: 1. / 16.,
            records: vec![],
            evidence: vec![],
            model_operations: 0,
            native_token_resolution_passed: false,
            training_ready: false,
            boundary_o_conflict_policy: CONFLICT_POLICY.into(),
            held_cases: vec!["held".into()],
            must_preserve: vec!["context".into()],
        };
        for lambda in [f64::NAN, f64::INFINITY, -0.001, 0.063] {
            draft.lambda_each_proposal = lambda;
            assert!(draft.validate().is_err());
        }
        draft.lambda_each_proposal = 1. / 16.;
        let (record, _) = fixture();
        draft.records = vec![record.clone(), record];
        assert!(draft.validate().is_err());
    }

    #[test]
    fn successor_comparison_allows_only_native_label_corrections() {
        let (_, row) = fixture();
        let fc = FeatureConfig::default();
        let prior = annotated_document(
            &row.input,
            "US",
            &serde_json::to_string(&row.expected).unwrap(),
            &fc,
        )
        .unwrap();
        let mut corrected = annotated_document(&row.input, "US", "[]", &fc).unwrap();
        assert_ne!(prior.enc.labels, corrected.enc.labels);
        verify_unchanged_inputs(&prior, &corrected).unwrap();
        corrected.enc.flags[0] ^= 1;
        assert!(verify_unchanged_inputs(&prior, &corrected).is_err());
        corrected.enc.flags[0] ^= 1;
        corrected.text.push(' ');
        assert!(verify_unchanged_inputs(&prior, &corrected).is_err());
    }
}

/// A freshly resolved complete row for binding to an actual native assembly.
pub(crate) struct PreviewRow {
    pub(crate) name: String,
    pub(crate) role: String,
    pub(crate) doc: DetectorDoc,
    pub(crate) masks: [Vec<f32>; 3],
    pub(crate) proof: Value,
}

/// Reuse the strict selector resolver and compare every row with an explicit native proof.
pub(crate) fn resolve_preview(
    draft_path: &Path,
    proof: &Value,
    cfg: &crate::config::Config,
) -> anyhow::Result<Vec<PreviewRow>> {
    let bytes = std::fs::read(draft_path)?;
    let draft: Draft = serde_json::from_slice(&bytes)?;
    draft.validate()?;
    ensure!(
        proof["passed"] == true
            && proof["training_ready"] == false
            && proof["model_initialized"] == false
            && proof["model_forwards"] == 0
            && proof["optimizer_updates"] == 0
            && proof["draft"]["sha256"] == crate::export::sha256_hex(&bytes)
            && proof["draft"]["path"] == json!(draft_path)
            && proof["compiled_selector_sha256"]
                == crate::export::sha256_hex(include_bytes!("hard_token_selection.rs")),
        "native proof or compiled selector binding differs"
    );
    let weights: [f32; 7] = cfg
        .detector
        .as_ref()
        .context("preview detector absent")?
        .class_weights
        .clone()
        .try_into()
        .map_err(|_| anyhow::anyhow!("preview needs seven class weights"))?;
    ensure!(
        weights == [1., 3., 2., 3., 2., 2., 1.5]
            && draft.lambda_each_proposal == 1. / 16.
            && proof["lambda_each_proposal"] == 1. / 16.
            && proof["group_order"] == json!(GROUP_NAMES)
            && proof["feature_config_sha256"]
                == crate::export::sha256_hex(&serde_json::to_vec(&cfg.features)?)
            && proof["class_weights_sha256"]
                == crate::export::sha256_hex(&serde_json::to_vec(&weights)?),
        "preview native feature or objective binding differs"
    );
    ensure!(draft.evidence.len() >= 4, "selector evidence absent");
    let evidence: Vec<_> = draft
        .evidence
        .iter()
        .map(|pin| json!({"path":pin.path,"sha256":pin.sha256}))
        .collect();
    ensure!(
        proof["evidence"] == json!(evidence),
        "proof evidence differs"
    );
    let authored = load_rows(&draft.evidence[0])?;
    let real = load_rows(&draft.evidence[1])?;
    ensure!(
        (authored.len(), real.len()) == (569, 1799),
        "preview complete source membership differs"
    );
    let historical: Value = serde_json::from_slice(&draft.evidence[3].bytes()?)?;
    let references = reference_indices(&historical, &real)?;
    let mut result = Vec::new();
    let mut targets = [0usize; 3];
    for record in &draft.records {
        let rows = match record.role.as_str() {
            "authored" => &authored,
            "real" => &real,
            _ => anyhow::bail!("unknown preview selector role"),
        };
        let row = rows.get(&record.name).context("selector source absent")?;
        let reference = *references
            .get(&record.name)
            .context("selector historical reference absent")?;
        let (resolved, masks, doc) =
            resolve(record, row, reference, &cfg.features.to_tessera(), &weights)?;
        for target in &record.targets {
            targets[GROUP_NAMES
                .iter()
                .position(|&group| group == target.group)
                .context("selector target group absent")?] += 1;
        }
        result.push(PreviewRow {
            name: record.name.clone(),
            role: record.role.clone(),
            doc,
            masks,
            proof: serde_json::to_value(resolved)?,
        });
    }
    ensure!(
        !result.is_empty()
            && proof["full_parent_rows"] == result.len()
            && proof["target_spans_by_group"] == json!(targets)
            && proof["rows"] == json!(result.iter().map(|row| &row.proof).collect::<Vec<_>>()),
        "native proof selectors, masks, labels or complete encoding differ"
    );
    Ok(result)
}

#[cfg(test)]
mod preview_api_tests {
    use super::*;

    #[test]
    fn wrapper_reuses_native_resolution_and_rejects_changed_proof_and_evidence() {
        let dir = tempfile::tempdir().unwrap();
        let cfg: crate::config::Config =
            toml::from_str(include_str!("../../configs/detector-shared-v5.toml")).unwrap();
        let authored = dir.path().join("authored.jsonl");
        let real = dir.path().join("real.jsonl");
        let note = dir.path().join("note.json");
        let historical = dir.path().join("historical.json");
        let source = SourceRow {
            name: "seen-real-00-0000".into(),
            country: "US".into(),
            input: "Émilie visits Harbor Bureau today".into(),
            expected: vec![
                json!({"kind":"person","start":0,"end":7,"text":"Émilie"}),
                json!({"kind":"org","start":15,"end":28,"text":"Harbor Bureau"}),
            ],
        };
        let record = Record {
            role: "real".into(),
            name: source.name.clone(),
            text_sha256: crate::export::sha256_hex(source.input.as_bytes()),
            expected_sha256: expected_hash(&source.expected).unwrap(),
            native_index_reference_only: 170569,
            full_original_parent_required: true,
            targets: vec![Target {
                group: "person".into(),
                start: 0,
                end: 7,
                text: "Émilie".into(),
                expected: "person".into(),
                reason: "fixture".into(),
                adjacent_o_tokens_requested: 0,
            }],
        };
        let source_line = |name: &str, input: &str, expected: &[Value]| {
            serde_json::to_string(
                &json!({"name":name,"country":"US","input":input,"expected":expected}),
            )
            .unwrap()
        };
        std::fs::write(
            &authored,
            (0..569)
                .map(|i| source_line(&format!("seen-authored-unused-{i}"), "Unused", &[]))
                .collect::<Vec<_>>()
                .join("\n"),
        )
        .unwrap();
        std::fs::write(
            &real,
            (0..1799)
                .map(|i| {
                    if i == 0 {
                        source_line(&source.name, &source.input, &source.expected)
                    } else {
                        source_line(&format!("seen-real-00-{i:04}"), "Unused", &[])
                    }
                })
                .collect::<Vec<_>>()
                .join("\n"),
        )
        .unwrap();
        std::fs::write(&note, b"{}").unwrap();
        std::fs::write(&historical,serde_json::to_vec(&json!({"full_synthetic_boundary":170569,"typed_synthetic":{"authored_documents":[]},"silver":{"source_pieces":[1799]}})).unwrap()).unwrap();
        let evidence:Vec<_>=[&authored,&real,&note,&historical].into_iter().map(|path|json!({"path":path,"sha256":crate::export::sha256_hex(&std::fs::read(path).unwrap())})).collect();
        let draft = json!({"scope":"static train-only byte selectors, not a native mask or fit authorization","groups":GROUP_NAMES,"lambda_each_proposal":1./16.,
            "records":[{"role":record.role,"name":record.name,"text_sha256":record.text_sha256,"expected_sha256":record.expected_sha256,
                "native_index_reference_only":170569,"full_original_parent_required":true,"targets":[{"group":"person","start":0,"end":7,"text":"Émilie","expected":"person","reason":"fixture","adjacent_o_tokens_requested":0}]}],
            "evidence":evidence,"model_operations":0,"native_token_resolution_passed":false,"training_ready":false,"boundary_o_conflict_policy":CONFLICT_POLICY,"held_cases":["fixture"],"must_preserve":["fullparent"]});
        let path = dir.path().join("draft.json");
        std::fs::write(&path, serde_json::to_vec(&draft).unwrap()).unwrap();
        let weights = [1., 3., 2., 3., 2., 2., 1.5];
        let (resolved, _, _) = resolve(
            &record,
            &source,
            170569,
            &cfg.features.to_tessera(),
            &weights,
        )
        .unwrap();
        let proof = json!({"passed":true,"training_ready":false,"model_initialized":false,"model_forwards":0,"optimizer_updates":0,
            "draft":{"path":path,"sha256":crate::export::sha256_hex(&std::fs::read(&path).unwrap())},
            "compiled_selector_sha256":crate::export::sha256_hex(include_bytes!("hard_token_selection.rs")),"lambda_each_proposal":1./16.,"group_order":GROUP_NAMES,
            "feature_config_sha256":crate::export::sha256_hex(&serde_json::to_vec(&cfg.features).unwrap()),
            "class_weights_sha256":crate::export::sha256_hex(&serde_json::to_vec(&weights).unwrap()),"evidence":evidence,
            "full_parent_rows":1,"target_spans_by_group":[1,0,0],"rows":[resolved]});
        assert_eq!(resolve_preview(&path, &proof, &cfg).unwrap().len(), 1);
        for field in [
            "native_encoding_sha256",
            "binary_masks_sha256",
            "native_token_spans",
            "selected_indices",
            "selected_tokens",
            "group_denominators",
        ] {
            let mut changed = proof.clone();
            changed["rows"][0][field] = json!("changed");
            assert!(resolve_preview(&path, &changed, &cfg).is_err());
        }
        for field in [
            "feature_config_sha256",
            "class_weights_sha256",
            "compiled_selector_sha256",
        ] {
            let mut changed = proof.clone();
            changed[field] = json!("changed");
            assert!(resolve_preview(&path, &changed, &cfg).is_err());
        }
        let mut changed = proof.clone();
        changed["draft"]["sha256"] = json!("changed");
        assert!(resolve_preview(&path, &changed, &cfg).is_err());
        std::fs::write(&note, b"changed").unwrap();
        assert!(resolve_preview(&path, &proof, &cfg).is_err());
    }
}
