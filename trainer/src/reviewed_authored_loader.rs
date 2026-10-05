//! Receipt-bound authored packet loading and canonical detector encoding.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, ensure};
use serde::de::DeserializeOwned;
use serde_json::{Value, json};
use tessera::internal::{
    FeatureConfig, decode_detector_with_text, flag, normalized_us_address_end,
};

use super::{InputContract, LoadedAuthored, Manifest, Proposal, Row, Span};
use crate::detector::{self, DetectorDoc, DetectorInputPolicy, KindSpan};
use crate::reviewed_data::{Receipt, digest};

#[derive(Default)]
pub(super) struct Dependencies(BTreeMap<PathBuf, (String, Vec<u8>)>);

impl Dependencies {
    pub(super) fn read(&mut self, pin: &Receipt) -> anyhow::Result<Vec<u8>> {
        if let Some((sha, bytes)) = self.0.get(&pin.path) {
            ensure!(sha == &pin.sha256, "one dependency has inconsistent hashes");
            return Ok(bytes.clone());
        }
        let bytes = pin.bytes()?;
        self.0
            .insert(pin.path.clone(), (pin.sha256.clone(), bytes.clone()));
        Ok(bytes)
    }

    fn finish(self) -> anyhow::Result<Vec<Receipt>> {
        self.0
            .into_iter()
            .map(|(path, (sha256, _))| {
                let pin = Receipt { path, sha256 };
                pin.bytes()?;
                Ok(pin)
            })
            .collect()
    }
}

pub(super) fn jsonl<T: DeserializeOwned>(bytes: &[u8]) -> anyhow::Result<Vec<T>> {
    std::str::from_utf8(bytes)?
        .lines()
        .enumerate()
        .map(|(i, line)| {
            ensure!(!line.trim().is_empty(), "blank physical JSONL row{}", i + 1);
            serde_json::from_str(line).with_context(|| format!("JSONL row{}", i + 1))
        })
        .collect()
}

pub(super) fn pin(manifest: &Manifest, name: &str) -> anyhow::Result<Receipt> {
    let found: Vec<_> = manifest
        .receipt_pins
        .iter()
        .filter(|(path, _)| path.file_name().is_some_and(|file| file == name))
        .collect();
    ensure!(
        found.len() == 1,
        "missing/ambiguous named authored receipt {name}"
    );
    Ok(Receipt {
        path: found[0].0.clone(),
        sha256: found[0].1.clone(),
    })
}

fn resolve(root: &Path, value: &str) -> anyhow::Result<PathBuf> {
    let path = Path::new(value);
    if path.is_absolute() {
        return Ok(path.to_owned());
    }
    ensure!(
        path.components().all(|part| matches!(
            part,
            std::path::Component::Normal(_) | std::path::Component::CurDir
        )),
        "unsafe relative review reference"
    );
    Ok(root.join(path))
}

pub(super) fn referenced_pins(
    root: &Path,
    value: &Value,
    dependencies: &mut Dependencies,
) -> anyhow::Result<()> {
    for (path, sha) in value.as_object().context("review pin map missing")? {
        dependencies.read(&Receipt {
            path: resolve(root, path)?,
            sha256: sha.as_str().context("review hash absent")?.to_owned(),
        })?;
    }
    Ok(())
}

pub(super) fn normalized(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

pub(super) fn validate_spans(text: &str, spans: &[Span]) -> anyhow::Result<()> {
    ensure!(
        text.len() <= u32::MAX as usize,
        "authored text exceeds byte offset width"
    );
    let mut ordered: Vec<_> = spans.iter().collect();
    ordered.sort_by_key(|span| (span.start, span.end));
    for span in &ordered {
        ensure!(
            ["person", "org", "address", "email", "phone"].contains(&span.kind.as_str())
                && span.start < span.end
                && span.text.trim() == span.text
                && text.get(span.start as usize..span.end as usize) == Some(span.text.as_str()),
            "unknown kind, edge whitespace or invalid exact UTF8 span"
        );
    }
    ensure!(
        ordered.windows(2).all(|pair| pair[0].end <= pair[1].start),
        "duplicate/overlapping authored spans"
    );
    Ok(())
}

pub(super) fn verify_row(row: &Row, policy: &str) -> anyhow::Result<()> {
    ensure!(
        !row.name.trim().is_empty()
            && !row.input.trim().is_empty()
            && row.country == "US"
            && row.annotation_policy_sha256 == policy
            && row.status == "resolved"
            && row.uncertainties.is_empty()
            && row.unresolved.is_empty()
            && !row.real_source_text
            && row.authored_source.kind == "whole_source_format_variant",
        "unresolved/non-authored row or wrong policy"
    );
    validate_spans(&row.input, &row.expected)
}

pub(super) fn verify_identity(
    row: &Row,
    native_names: &BTreeSet<String>,
    native_texts: &BTreeSet<String>,
    names: &mut BTreeSet<String>,
    texts: &mut BTreeSet<String>,
) -> anyhow::Result<()> {
    ensure!(
        names.insert(row.name.clone())
            && texts.insert(row.input.clone())
            && !native_names.contains(&row.name)
            && !native_texts.contains(&row.input),
        "duplicate/exact-native-equal authored name or UTF8 whole text"
    );
    Ok(())
}

pub(super) fn verify_proposal(row: &Row, labels: &Proposal) -> anyhow::Result<()> {
    ensure!(
        labels
            .annotation_policy_sha256
            .as_ref()
            .is_none_or(|policy| policy == &row.annotation_policy_sha256),
        "original proposal policy metadata differs from final row"
    );
    let mut expected = labels.expected.clone();
    expected.sort();
    let mut final_expected = row.expected.clone();
    final_expected.sort();
    validate_spans(&labels.input, &labels.expected)?;
    ensure!(
        labels.name == row.name
            && labels.country == row.country
            && labels.input == row.input
            && expected == final_expected,
        "final authored gold is not independently bound by both original passes; disagreement route not implemented"
    );
    Ok(())
}

fn sorted_json(value: &Value) -> Value {
    match value {
        Value::Object(object) => {
            let keys: BTreeMap<_, _> = object.iter().collect();
            Value::Object(
                keys.into_iter()
                    .map(|(key, value)| (key.clone(), sorted_json(value)))
                    .collect(),
            )
        }
        Value::Array(values) => Value::Array(values.iter().map(sorted_json).collect()),
        _ => value.clone(),
    }
}

fn note_key(reviewer: &str, name: &str, note: &Value) -> anyhow::Result<(String, String, String)> {
    Ok((
        reviewer.to_owned(),
        name.to_owned(),
        digest(&serde_json::to_vec(&sorted_json(note))?),
    ))
}

pub(super) fn verify_notes(a: &[Proposal], b: &[Proposal], review: &Value) -> anyhow::Result<()> {
    let mut expected = BTreeMap::<_, usize>::new();
    for (reviewer, rows) in [("a", a), ("b", b)] {
        for row in rows {
            for note in &row.uncertainties {
                let start = note["start"].as_u64().context("note start absent")?;
                let end = note["end"].as_u64().context("note end absent")?;
                ensure!(
                    start < end
                        && row
                            .input
                            .get(usize::try_from(start)?..usize::try_from(end)?)
                            == note["text"].as_str(),
                    "original review note quote differs"
                );
                *expected
                    .entry(note_key(reviewer, &row.name, note)?)
                    .or_default() += 1;
            }
        }
    }
    let mut observed = BTreeMap::<_, usize>::new();
    for disposition in review["note_dispositions"]
        .as_array()
        .context("semantic note dispositions absent")?
    {
        let reviewer = disposition["reviewer"]
            .as_str()
            .context("note reviewer absent")?;
        let name = disposition["name"].as_str().context("note name absent")?;
        let hash = disposition["canonical_note_sha256"]
            .as_str()
            .context("note hash absent")?;
        ensure!(
            ["a", "b"].contains(&reviewer)
                && disposition["action"]
                    .as_str()
                    .is_some_and(|s| !s.trim().is_empty()),
            "unresolved note disposition"
        );
        if let Some(index) = disposition.get("note_index").and_then(Value::as_u64) {
            let rows = if reviewer == "a" { a } else { b };
            let row = rows
                .iter()
                .find(|row| row.name == name)
                .context("disposition names unknown proposal")?;
            let note = row
                .uncertainties
                .get(usize::try_from(index)?)
                .context("disposition note index absent")?;
            ensure!(
                note_key(reviewer, name, note)?.2 == hash,
                "disposition index/hash differs"
            );
        }
        *observed
            .entry((reviewer.to_owned(), name.to_owned(), hash.to_owned()))
            .or_default() += 1;
    }
    ensure!(
        expected == observed
            && review["counts"]["all_notes"].as_u64()
                == Some(expected.values().sum::<usize>() as u64),
        "semantic review does not close every original note exactly once"
    );
    Ok(())
}

fn same_reference(value: &Value, expected: &Receipt, root: &Path) -> anyhow::Result<bool> {
    Ok(resolve(
        root,
        value["path"]
            .as_str()
            .context("scope reference path absent")?,
    )? == expected.path
        && value["sha256"].as_str() == Some(expected.sha256.as_str()))
}

fn verify_label_scopes(
    manifest: &Manifest,
    root: &Path,
    a: &Receipt,
    b: &Receipt,
    dependencies: &mut Dependencies,
) -> anyhow::Result<()> {
    let packet = pin(manifest, "blind-packet.jsonl")?;
    let scope_a: Value =
        serde_json::from_slice(&dependencies.read(&pin(manifest, "blind-labels-a-scope.json")?)?)?;
    ensure!(
        scope_a["reviewer"] == "agent A"
            && same_reference(&scope_a["output"], a, root)?
            && scope_a["coverage"]["complete_documents_read"].as_u64()
                == Some(manifest.documents as u64)
            && scope_a["coverage"]["complete_documents_labeled"].as_u64()
                == Some(manifest.documents as u64)
            && scope_a["independence"]["consulted_only_blind_packet_and_annotation_policy_for_this_label_pass"]
                == true
            && scope_a["independence"]["construction_maps_peer_proposals_saved_predictions_not_consulted_for_this_pass"]
                == true,
        "original A pass scope/output binding absent"
    );
    let inputs = scope_a["inputs"]
        .as_array()
        .context("A scope input references absent")?;
    ensure!(
        inputs.len() == 2
            && inputs
                .iter()
                .any(|p| same_reference(p, &packet, root).unwrap_or(false))
            && inputs
                .iter()
                .any(|p| same_reference(p, &manifest.policy, root).unwrap_or(false)),
        "A pass packet/policy binding differs"
    );
    let scope_b: Value =
        serde_json::from_slice(&dependencies.read(&pin(manifest, "blind-scope-b.json")?)?)?;
    ensure!(
        same_reference(&scope_b["input"], &packet, root)?
            && same_reference(&scope_b["policy"], &manifest.policy, root)?
            && same_reference(&scope_b["proposal"], b, root)?
            && scope_b["complete_documents_read_and_labeled"].as_u64()
                == Some(manifest.documents as u64)
            && scope_b["span_slices_unique_nonoverlapping_validated"] == true,
        "original B pass scope/output binding absent"
    );
    Ok(())
}

pub(super) fn verify_semantic_status(review: &Value, draft_pin: &Receipt) -> anyhow::Result<()> {
    ensure!(
        review["status"] == "passed"
            && review["blockers"].as_array().is_some_and(Vec::is_empty)
            && review["draft_sha256"] == draft_pin.sha256,
        "passed independent source semantic review absent"
    );
    for required in [
        "every_selected_complete_raw_table_caption_header_body_cell_exact",
        "every_nonwhitespace_source_byte_once_in_copy_maps",
        "all32_independent_text_reconstructions_exact",
        "every_original_expected_span_mapped_once",
        "triplicate_408_spans_equal",
        "all_utf8_spans_notes_valid",
        "all438_original_notes_retained_exact",
        "a_subject_rulings_explicitly_match",
        "b_status_rulings_all_supported",
        "no_new_proper_entity_fields_inserted",
        "builder_reads_only_reviewed_TRAIN_source_policy_not_DEV",
        "input_source_hashes_unchanged",
    ] {
        ensure!(
            review["validation"][required] == true,
            "source review coverage {required} absent"
        );
    }
    Ok(())
}

pub(super) fn encode(
    row: &Row,
    fc: &FeatureConfig,
    contract: InputContract,
) -> anyhow::Result<(DetectorDoc, Value)> {
    let gold: Vec<_> = row
        .expected
        .iter()
        .filter_map(|span| {
            ["person", "org", "address"]
                .iter()
                .position(|&kind| kind == span.kind.as_str())
                .map(|kind| KindSpan {
                    kind,
                    start: span.start,
                    end: span.end,
                })
        })
        .collect();
    let blank = detector::encode_document_with_feature_contract(
        &row.input,
        &[],
        fc,
        DetectorInputPolicy::KnownUs,
        contract,
    )?;
    let mut enc = detector::encode_document_with_feature_contract(
        &row.input,
        &gold,
        fc,
        DetectorInputPolicy::KnownUs,
        contract,
    )
    .with_context(|| format!("unreachable authored targets {}", row.name))?;
    let mut without_labels = enc.clone();
    without_labels.labels.fill(0);
    ensure!(
        without_labels == blank,
        "gold changed canonical runtime input arrays"
    );
    let breaks = detector::breaks_of(&row.input);
    let n = enc.token_spans.len();
    ensure!(
        n > 0
            && enc.labels.len() == n
            && breaks.len() == n
            && enc
                .flags
                .iter()
                .all(|flags| flags >> contract.flag_bits() == 0),
        "invalid retained feature arrays/width"
    );
    let masked: Vec<_> = enc
        .flags
        .iter()
        .map(|f| f & (flag::IN_RULE_SPAN | flag::MASKED) != 0)
        .collect();
    let bounds: Vec<_> = enc
        .token_spans
        .iter()
        .map(|&(a, b)| (a as usize, b as usize))
        .collect();
    let mut oracle = vec![0.0; n * detector::DETECTOR_LABELS];
    for (i, &label) in enc.labels.iter().enumerate() {
        oracle[i * detector::DETECTOR_LABELS + label as usize] = 1.0;
    }
    let mut represented: Vec<_> =
        decode_detector_with_text(&row.input, &bounds, &oracle, &masked, &breaks)
            .into_iter()
            .map(|span| {
                let start = enc.token_spans[span.first].0;
                let raw_end = enc.token_spans[span.last].1;
                let end = if span.kind == tessera::Kind::Address {
                    normalized_us_address_end(&row.input, start as usize, raw_end as usize) as u32
                } else {
                    raw_end
                };
                (span.kind.as_str().to_owned(), start, end)
            })
            .collect();
    represented.sort();
    let mut intended: Vec<_> = gold
        .iter()
        .map(|span| {
            (
                ["person", "org", "address"][span.kind].to_owned(),
                span.start,
                span.end,
            )
        })
        .collect();
    intended.sort();
    ensure!(
        represented == intended,
        "neural gold not exactly reachable through continuation/postal-end"
    );
    let features = json!({"contract":"reviewed-authored-runtime-input-v1", "input_policy":"known_us", "feature_contract":contract.name(),
        "flag_bits":contract.flag_bits(), "text_sha256":digest(row.input.as_bytes()),
        "feature_config":{"ngram_sizes":fc.ngram_sizes, "hash_buckets":fc.hash_buckets, "hash_seed":fc.hash_seed},
        "token_spans":blank.token_spans, "ngram_ids_plus_one":blank.ngram_ids, "script":blank.script, "shape":blank.shape,
        "flags":blank.flags, "decoder_masked":masked, "paragraph_breaks":breaks});
    enc.country = "US".to_owned();
    Ok((
        DetectorDoc {
            text: row.input.clone(),
            enc,
            gold,
            breaks,
        },
        json!({
        "encoded_feature_sha256":digest(&serde_json::to_vec(&features)?), "retained_tokens":n,
        "input_policy":"known_us", "feature_contract":contract.name(), "flag_bits":contract.flag_bits(),
        "gold_independent_all_input_arrays":true, "all_neural_gold_reachable":true,
        "reachability_method":"synthetic one-hot BIO through AddressContinuationV1 and unchanged postal-end; no weights/forward",
        "rules_only_gold_not_in_neural_targets":true}),
    ))
}

fn differences<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(left, right)| left != right)
        .count()
        + a.len().abs_diff(b.len())
}

pub(super) fn compare_native_features(
    row: &Row,
    native_text: &str,
    fc: &FeatureConfig,
    contract: InputContract,
) -> anyhow::Result<Value> {
    let base = detector::encode_document_with_feature_contract(
        native_text,
        &[],
        fc,
        DetectorInputPolicy::KnownUs,
        contract,
    )?;
    let authored = detector::encode_document_with_feature_contract(
        &row.input,
        &[],
        fc,
        DetectorInputPolicy::KnownUs,
        contract,
    )?;
    let base_breaks = detector::breaks_of(native_text);
    let authored_breaks = detector::breaks_of(&row.input);
    let masks = |flags: &[u32]| {
        flags
            .iter()
            .map(|f| f & (flag::IN_RULE_SPAN | flag::MASKED) != 0)
            .collect::<Vec<_>>()
    };
    let counts = BTreeMap::from([
        (
            "ngram_ids_plus_one",
            differences(&base.ngram_ids, &authored.ngram_ids),
        ),
        ("script", differences(&base.script, &authored.script)),
        ("shape", differences(&base.shape, &authored.shape)),
        ("flags", differences(&base.flags, &authored.flags)),
        (
            "decoder_masked",
            differences(&masks(&base.flags), &masks(&authored.flags)),
        ),
        (
            "paragraph_breaks",
            differences(&base_breaks, &authored_breaks),
        ),
    ]);
    ensure!(
        counts.values().any(|&n| n > 0),
        "inert authored variant under selected contract: only text/byte offsets differ"
    );
    Ok(
        json!({"feature_contract":contract.name(),"input_policy":"known_us","comparison":"two empty-gold canonical encodings",
        "native_tokens":base.token_spans.len(),"authored_tokens":authored.token_spans.len(),"difference_counts":counts,
        "token_span_difference_count":differences(&base.token_spans,&authored.token_spans),
        "token_bounds_alone_do_not_qualify_as_feature_change":true,"actual_feature_or_mask_or_break_change":true,
        "no_content_or_new_identity_claim":true}),
    )
}

pub(super) fn verify_manifest(manifest: &Manifest) -> anyhow::Result<()> {
    ensure!(
        manifest.schema == "reviewed-whole-source-authored-data-v1"
            && manifest.origin == "authored_train_only"
            && manifest.split == "train"
            && manifest.documents > 0
            && !manifest.native_train_documents_changed,
        "wrong authored schema/origin/split or mutated native TRAIN"
    );
    ensure!(
        manifest.all_source_semantics_and_438_notes_reviewed
            != manifest.all_source_bytes_and388_notes_reviewed,
        "exactly one supported authored review route required"
    );
    Ok(())
}

pub(super) fn load(
    manifest_pin: &Receipt,
    root: &Path,
    native_inputs: &[Receipt],
    fc: &FeatureConfig,
    contract: InputContract,
) -> anyhow::Result<LoadedAuthored> {
    ensure!(
        root.is_absolute()
            && !native_inputs.is_empty()
            && fc.hash_buckets > 0
            && !fc.ngram_sizes.is_empty()
            && fc.ngram_sizes.iter().all(|&n| n > 0),
        "explicit root/native composition/feature configuration required"
    );
    let mut dependencies = Dependencies::default();
    let manifest: Manifest = serde_json::from_slice(&dependencies.read(manifest_pin)?)?;
    verify_manifest(&manifest)?;
    let adjudicated = manifest.all_source_bytes_and388_notes_reviewed;
    dependencies.read(&manifest.policy)?;
    for (path, sha256) in &manifest.receipt_pins {
        dependencies.read(&Receipt {
            path: path.clone(),
            sha256: sha256.clone(),
        })?;
    }
    let finalizer = Receipt {
        path: manifest
            .prepared
            .path
            .parent()
            .context("prepared parent absent")?
            .join("finalize.py"),
        sha256: manifest.finalizer_sha256.clone(),
    };
    dependencies.read(&finalizer)?;
    let successor = super::successor::open(&manifest, &mut dependencies)?;
    let construction: Value =
        serde_json::from_slice(&dependencies.read(&pin(&manifest, "construction.json")?)?)?;
    let source_pin: Receipt = if adjudicated {
        super::adjudication::construction_source(&construction, &manifest, root, &mut dependencies)?
    } else {
        serde_json::from_value(construction["source"].clone())?
    };
    let count_field = if adjudicated {
        "documents"
    } else {
        "candidate_documents"
    };
    ensure!(
        construction[count_field].as_u64() == Some(manifest.documents as u64),
        "construction/prepared document counts differ"
    );
    let native_source = match &successor {
        Some(successor) => successor.native_source(&source_pin, native_inputs)?,
        None => source_pin.clone(),
    };
    ensure!(
        native_inputs
            .iter()
            .any(|pin| pin.path == native_source.path && pin.sha256 == native_source.sha256),
        "source prepared dataset absent from pinned native composition"
    );
    referenced_pins(
        root,
        &construction[if adjudicated {
            "output_pins"
        } else {
            "outputs"
        }],
        &mut dependencies,
    )?;
    let builder = Receipt {
        path: pin(&manifest, "construction.json")?
            .path
            .parent()
            .context("construction parent absent")?
            .join("build.py"),
        sha256: construction["build_script_sha256"]
            .as_str()
            .context("builder hash absent")?
            .to_owned(),
    };
    dependencies.read(&builder)?;
    if !adjudicated {
        let construction_policy: Receipt = serde_json::from_value(construction["policy"].clone())?;
        ensure!(
            construction_policy.path == manifest.policy.path
                && construction_policy.sha256 == manifest.policy.sha256,
            "construction policy differs"
        );
    }
    let mut native_names = BTreeSet::new();
    let mut native_texts = BTreeSet::new();
    let mut native_pins = BTreeSet::new();
    let mut native_normalized_groups = BTreeMap::<String, Vec<String>>::new();
    for receipt in native_inputs {
        ensure!(
            native_pins.insert((receipt.path.clone(), receipt.sha256.clone())),
            "duplicate native exclusion input"
        );
        for row in jsonl::<Value>(&dependencies.read(receipt)?)? {
            native_names.insert(
                row["name"]
                    .as_str()
                    .context("native name absent")?
                    .to_owned(),
            );
            let text = row["input"].as_str().context("native whole text absent")?;
            native_texts.insert(text.to_owned());
            native_normalized_groups
                .entry(normalized(text))
                .or_default()
                .push(
                    row["name"]
                        .as_str()
                        .context("native name absent")?
                        .to_owned(),
                );
        }
    }
    let source_rows = jsonl::<Value>(&dependencies.read(&source_pin)?)?;
    let prepared_bytes = dependencies.read(&manifest.prepared)?;
    let rows = jsonl::<Row>(&prepared_bytes)?;
    let reviewed_bytes = match &successor {
        Some(successor) => dependencies.read(successor.original_prepared())?,
        None => prepared_bytes.clone(),
    };
    let reviewed = jsonl::<Row>(&reviewed_bytes)?;
    let a_pin = pin(&manifest, "blind-labels-a.jsonl")?;
    let b_pin = pin(&manifest, "blind-labels-b.jsonl")?;
    ensure!(
        a_pin.path != b_pin.path && a_pin.sha256 != b_pin.sha256,
        "two separate original label passes required"
    );
    if adjudicated {
        super::adjudication::verify_label_scopes(
            &manifest,
            root,
            &a_pin,
            &b_pin,
            &mut dependencies,
        )?;
    } else {
        verify_label_scopes(&manifest, root, &a_pin, &b_pin, &mut dependencies)?;
    }
    let a = jsonl::<Proposal>(&dependencies.read(&a_pin)?)?;
    let b = jsonl::<Proposal>(&dependencies.read(&b_pin)?)?;
    let candidates = jsonl::<Value>(&dependencies.read(&pin(&manifest, "candidates.jsonl")?)?)?;
    let packet = jsonl::<Value>(&dependencies.read(&pin(&manifest, "blind-packet.jsonl")?)?)?;
    let map_bytes = dependencies.read(&pin(&manifest, "literal-copy-map.jsonl")?)?;
    let maps = jsonl::<super::source::CopyMap>(&map_bytes)?;
    if adjudicated {
        super::adjudication::verify_bundle_maps(
            &construction,
            &jsonl::<Value>(&map_bytes)?,
            &mut dependencies,
        )?;
    }
    ensure!(
        [
            rows.len(),
            reviewed.len(),
            a.len(),
            b.len(),
            candidates.len(),
            packet.len(),
            maps.len()
        ]
        .iter()
        .all(|&n| n == manifest.documents),
        "authored/review/map counts differ"
    );
    let adjudication = if adjudicated {
        Some(super::adjudication::verify(
            &manifest,
            root,
            &reviewed,
            &a,
            &b,
            &mut dependencies,
        )?)
    } else {
        let review: Value = serde_json::from_slice(&dependencies.read(&pin(
            &manifest,
            "independent-semantic-transfer-review-a.json",
        )?)?)?;
        ensure!(
            review["counts"]["authored_documents"].as_u64() == Some(manifest.documents as u64),
            "semantic/prepared document counts differ"
        );
        let draft_pin = pin(&manifest, "root-adjudication-transfer-draft.json")?;
        verify_semantic_status(&review, &draft_pin)?;
        referenced_pins(root, &review["consulted_pins"], &mut dependencies)?;
        let draft: Value = serde_json::from_slice(&dependencies.read(&draft_pin)?)?;
        referenced_pins(root, &draft["input_pins"], &mut dependencies)?;
        verify_notes(&a, &b, &review)?;
        None
    };
    let transferred = match &successor {
        Some(successor) => {
            let adjudication: Value = serde_json::from_slice(
                &dependencies.read(&pin(&manifest, "final-adjudication-root.json")?)?,
            )?;
            successor.verify(&super::successor::Evidence {
                source_bytes: &dependencies.read(&source_pin)?,
                corrected_bytes: &dependencies.read(&native_source)?,
                reviewed_bytes: &reviewed_bytes,
                prepared_bytes: &prepared_bytes,
                maps: &jsonl::<Value>(&map_bytes)?,
                case_decisions: adjudication["case_decisions"]
                    .as_array()
                    .context("source case decisions absent")?,
            })?
        }
        None => Vec::new(),
    };
    let mut loaded = LoadedAuthored {
        names: Vec::new(),
        docs: Vec::new(),
        expected: Vec::new(),
        rules_expected: Vec::new(),
        identities: Vec::new(),
        dependency_receipts: Vec::new(),
    };
    let mut names = BTreeSet::new();
    let mut texts = BTreeSet::new();
    let mut counts = BTreeMap::<_, usize>::new();
    for (index, (adjudicated_row, row)) in reviewed.iter().zip(&rows).enumerate() {
        verify_row(adjudicated_row, &manifest.policy.sha256)?;
        verify_row(row, &manifest.policy.sha256)?;
        verify_identity(row, &native_names, &native_texts, &mut names, &mut texts)?;
        ensure!(
            [adjudicated_row, row].iter().all(|r| {
                r.authored_source.native_dataset.path == source_pin.path
                    && r.authored_source.native_dataset.sha256 == source_pin.sha256
            }),
            "authored row changes its native source prepared dataset"
        );
        if !adjudicated {
            for labels in [&a[index], &b[index]] {
                verify_proposal(adjudicated_row, labels)?;
            }
        }
        for evidence in [&candidates[index], &packet[index]] {
            ensure!(
                evidence["name"] == adjudicated_row.name
                    && evidence["country"] == adjudicated_row.country
                    && evidence["input"] == adjudicated_row.input,
                "candidate/blind packet whole text differs"
            );
        }
        let mut projected: Vec<Span> =
            serde_json::from_value(candidates[index]["expected"].clone())?;
        projected.sort();
        let mut final_expected = adjudicated_row.expected.clone();
        final_expected.sort();
        ensure!(
            projected == final_expected
                && packet[index]["annotation_policy_sha256"] == manifest.policy.sha256,
            "source-projected spans or blind packet policy differ from adjudicated/independently reviewed final labels"
        );
        let native = source_rows
            .get(row.authored_source.native_row_index)
            .context("native prepared row index absent")?;
        let source = super::source::verify(adjudicated_row, &maps[index], native, &mut |pin| {
            dependencies.read(pin)
        })?;
        let (doc, encoding) = encode(row, fc, contract)?;
        let feature_changes = compare_native_features(
            row,
            native["input"]
                .as_str()
                .context("native base text absent")?,
            fc,
            contract,
        )
        .with_context(|| format!("authored selected-contract base comparison {}", row.name))?;
        let normalized_input = normalized(&row.input);
        let native_collision_names = native_normalized_groups
            .get(&normalized_input)
            .cloned()
            .unwrap_or_default();
        let authored_collision_names: Vec<_> = rows
            .iter()
            .filter(|other| normalized(&other.input) == normalized_input)
            .map(|other| other.name.clone())
            .collect();
        for span in &row.expected {
            *counts.entry(span.kind.clone()).or_default() += 1;
        }
        loaded.names.push(row.name.clone());
        loaded.rules_expected.push(
            row.expected
                .iter()
                .filter(|span| ["phone", "email"].contains(&span.kind.as_str()))
                .cloned()
                .collect(),
        );
        loaded.expected.push(row.expected.clone());
        let mut identity = json!({"name":row.name, "text_sha256":digest(row.input.as_bytes()),
            "expected_sha256":digest(&serde_json::to_vec(&row.expected)?), "origin_type":"authored_train_only",
            "review_route":if adjudicated { "explicit_source_adjudicated_deltas" } else { "exact_two_pass_agreement" },
            "adjudication":adjudication, "source":source, "encoding":encoding, "canonical_base_comparison":feature_changes,
            "identity_contract":"exact_utf8_authored_train_variant_v2",
            "base_variant_group":{"native_prepared":row.authored_source.native_dataset,"native_name":row.authored_source.native_name,
                "native_text_sha256":row.authored_source.native_text_sha256},
            "whitespace_normalized_collision":{"native_names":native_collision_names,"authored_names_including_self":authored_collision_names,
                "normalized_sha256":digest(normalized_input.as_bytes())},
            "content_novelty_or_new_entity_or_source_family_claimed":false});
        if let Some(transfer) = transferred.get(index) {
            identity["native_label_successor"] = transfer.clone();
        }
        loaded.identities.push(identity);
        loaded.docs.push(doc);
    }
    ensure!(
        counts == manifest.counts,
        "actual authored five-kind counts differ"
    );
    loaded.dependency_receipts = dependencies.finish()?;
    Ok(loaded)
}
