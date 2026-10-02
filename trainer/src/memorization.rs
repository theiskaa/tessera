//! Original real training documents for a bounded memorization diagnostic.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

#[path = "memorization_exposure.rs"]
pub(crate) mod exposure;
#[path = "memorization_frozen.rs"]
mod frozen;

pub(crate) use frozen::{SCOPE as FROZEN_SCOPE, prepare_frozen, verify_frozen_selection};

use crate::config::Config;
use crate::detector::{self, DetectorDoc, KindSpan};

const MIN_DOCUMENTS_PER_GROUP: usize = 4;
// Match ordinary silver loading; the diagnostic token limit only filters whole rows.
const SILVER_MAX_TOKENS: usize = 900;

/// Unmodified native training documents and their selection provenance.
pub(crate) struct Prepared {
    pub(crate) documents: Vec<DetectorDoc>,
    pub(crate) manifest: Value,
}

#[derive(Deserialize)]
struct OriginalRow {
    text: String,
    entities: Vec<OriginalSpan>,
    #[serde(flatten)]
    provenance: serde_json::Map<String, Value>,
}

#[derive(Deserialize)]
struct OriginalSpan {
    kind: String,
    start: u32,
    end: u32,
}

struct Candidate {
    index: usize,
    source: usize,
    line: usize,
    provenance: Value,
    groups: [bool; 4],
    form: String,
    rank: [u8; 32],
}

fn support(doc: &DetectorDoc) -> anyhow::Result<[bool; 4]> {
    let mut groups = [false; 4];
    for span in &doc.gold {
        ensure!(
            span.kind < 3,
            "memorization gold contains an unsupported kind"
        );
        groups[span.kind] = true;
    }
    groups[3] = doc.gold.is_empty();
    Ok(groups)
}

fn form(doc: &DetectorDoc, groups: [bool; 4]) -> anyhow::Result<String> {
    let mut acronym = false;
    let mut single_person = false;
    for span in &doc.gold {
        let text = doc
            .text
            .get(span.start as usize..span.end as usize)
            .context("memorization gold is outside original text")?;
        if span.kind == 1 {
            let letters: Vec<_> = text.chars().filter(|ch| ch.is_alphabetic()).collect();
            acronym |= !letters.is_empty() && letters.iter().all(|ch| ch.is_uppercase());
        }
        single_person |= span.kind == 0 && text.split_whitespace().count() == 1;
    }
    Ok(format!(
        "{:?}:multiline={}:acronym={acronym}:single_person={single_person}:over_32_tokens={}",
        groups,
        doc.text.contains('\n'),
        doc.enc.token_spans.len() > 32,
    ))
}

fn original_gold(row: &OriginalRow) -> Vec<KindSpan> {
    let mut gold: Vec<_> = row
        .entities
        .iter()
        .filter_map(|span| {
            let kind = match span.kind.as_str() {
                "person" => 0,
                "org" => 1,
                "address" => 2,
                _ => return None,
            };
            Some(KindSpan {
                kind,
                start: span.start,
                end: span.end,
            })
        })
        .collect();
    gold.sort_by_key(|span| (span.start, span.end, span.kind));
    gold
}

fn select(candidates: &[Candidate], max_documents: usize) -> anyhow::Result<Vec<usize>> {
    ensure!(
        max_documents >= 4 * MIN_DOCUMENTS_PER_GROUP,
        "memorization selection needs a budget of at least 16 documents"
    );
    let mut available = [0usize; 4];
    for candidate in candidates {
        for (count, present) in available.iter_mut().zip(candidate.groups) {
            *count += usize::from(present);
        }
    }
    for (name, count) in ["person", "org", "address", "fully negative"]
        .into_iter()
        .zip(available)
    {
        ensure!(
            count >= MIN_DOCUMENTS_PER_GROUP,
            "memorization selection has only {count} eligible {name} documents; needs four"
        );
    }
    let mut selected = Vec::new();
    let mut seen = BTreeSet::new();
    let mut counts = [0usize; 4];
    let mut sources = BTreeMap::<usize, usize>::new();
    let mut forms = BTreeMap::<&str, usize>::new();
    while selected.len() < max_documents.min(candidates.len()) {
        let next = candidates
            .iter()
            .enumerate()
            .filter(|(index, _)| !seen.contains(index))
            .min_by_key(|(_, candidate)| {
                let needed = candidate
                    .groups
                    .iter()
                    .zip(counts)
                    .filter(|(present, count)| **present && *count < MIN_DOCUMENTS_PER_GROUP)
                    .count();
                (
                    std::cmp::Reverse(needed),
                    sources.get(&candidate.source).copied().unwrap_or(0),
                    forms.get(candidate.form.as_str()).copied().unwrap_or(0),
                    candidate.rank,
                    candidate.index,
                )
            })
            .map(|(index, _)| index)
            .context("memorization selection exhausted eligible documents")?;
        let candidate = &candidates[next];
        seen.insert(next);
        selected.push(next);
        for (count, present) in counts.iter_mut().zip(candidate.groups) {
            *count += usize::from(present);
        }
        *sources.entry(candidate.source).or_default() += 1;
        *forms.entry(&candidate.form).or_default() += 1;
    }
    ensure!(
        counts.iter().all(|&count| count >= MIN_DOCUMENTS_PER_GROUP),
        "memorization selection did not retain four documents in every support group"
    );
    Ok(selected)
}

/// Select whole, distinct original silver rows by training labels and source/form diversity.
///
/// The caller must validate the approved training sources and their held-data exclusions.
/// No development predictions or held gold influence selection, and no text is shortened.
pub(crate) fn prepare(
    cfg: &Config,
    max_documents: usize,
    max_tokens: usize,
) -> anyhow::Result<Prepared> {
    ensure!(
        max_documents >= 4 * MIN_DOCUMENTS_PER_GROUP,
        "memorization selection needs a budget of at least 16 documents"
    );
    ensure!(max_tokens > 0, "memorization token limit must be positive");
    let detector = cfg
        .detector
        .as_ref()
        .context("memorization needs detector data")?;
    let (documents, loaded) = detector::load_silver(
        &detector.silver,
        &cfg.features.to_tessera(),
        SILVER_MAX_TOKENS,
    )?;
    let mut source_manifest = Vec::new();
    let mut candidates = Vec::new();
    let mut distinct = BTreeSet::new();
    let mut start = 0usize;
    let mut omitted_pieces = 0;
    for (source, (path, &pieces)) in detector
        .silver
        .iter()
        .zip(&loaded.source_pieces)
        .enumerate()
    {
        let bytes = std::fs::read(path).with_context(|| format!("reading {path}"))?;
        let text = std::str::from_utf8(&bytes).with_context(|| format!("decoding {path}"))?;
        let mut originals = BTreeMap::new();
        for (line, content) in text
            .lines()
            .enumerate()
            .filter(|(_, row)| !row.trim().is_empty())
        {
            let row: OriginalRow = serde_json::from_str(content)
                .with_context(|| format!("{path}:{} invalid original row", line + 1))?;
            originals.entry(row.text.clone()).or_insert((line + 1, row));
        }
        source_manifest.push(json!({
            "path": path,
            "sha256": format!("{:x}", Sha256::digest(&bytes)),
            "loaded_pieces": pieces,
        }));
        let end = start
            .checked_add(pieces)
            .context("memorization source range overflow")?;
        let source_docs = documents
            .get(start..end)
            .context("memorization source range mismatch")?;
        for (offset, doc) in source_docs.iter().enumerate() {
            let groups = support(doc)?;
            if doc.enc.token_spans.is_empty() || doc.enc.token_spans.len() > max_tokens {
                continue;
            }
            let Some((line, row)) = originals.get(&doc.text) else {
                omitted_pieces += 1;
                continue;
            };
            let mut gold = doc.gold.clone();
            gold.sort_by_key(|span| (span.start, span.end, span.kind));
            if gold != original_gold(row) {
                omitted_pieces += 1;
                continue;
            }
            if !distinct.insert(doc.text.as_str()) {
                continue;
            }
            let mut digest = Sha256::new();
            digest.update(cfg.seed.to_le_bytes());
            digest.update(doc.text.as_bytes());
            candidates.push(Candidate {
                index: start + offset,
                source,
                line: *line,
                provenance: Value::Object(row.provenance.clone()),
                groups,
                form: form(doc, groups)?,
                rank: digest.finalize().into(),
            });
        }
        start = end;
    }
    ensure!(
        start == documents.len(),
        "memorization source counts differ from native documents"
    );
    let selection = select(&candidates, max_documents)?;
    let mut chosen = Vec::new();
    let mut selected_manifest = Vec::new();
    let mut support_documents = [0usize; 4];
    let mut gold_spans = [0usize; 3];
    let mut content_tokens = 0;
    for candidate in selection.into_iter().map(|index| &candidates[index]) {
        let doc = &documents[candidate.index];
        for (count, present) in support_documents.iter_mut().zip(candidate.groups) {
            *count += usize::from(present);
        }
        for span in &doc.gold {
            gold_spans[span.kind] += 1;
        }
        content_tokens += doc.enc.token_spans.len();
        selected_manifest.push(json!({
            "source_index": candidate.source,
            "source_path": detector.silver[candidate.source],
            "source_line": candidate.line,
            "source_provenance": candidate.provenance,
            "original_source_row": true,
            "derived_piece": false,
            "text_sha256": format!("{:x}", Sha256::digest(doc.text.as_bytes())),
            "content_tokens": doc.enc.token_spans.len(),
            "form": candidate.form,
            "gold": doc.gold.iter().map(|span| json!({
                "kind": detector::KINDS[span.kind].as_str(),
                "start": span.start,
                "end": span.end,
            })).collect::<Vec<_>>(),
        }));
        chosen.push(doc.clone());
    }
    Ok(Prepared {
        documents: chosen,
        manifest: json!({
            "scope": "memorization of selected original real training rows; not independent accuracy or release quality",
            "selection": "training labels only; four documents per learned kind and four fully negative; prefer source/form diversity; seeded text hash breaks ties",
            "seed": cfg.seed,
            "max_documents": max_documents,
            "max_content_tokens": max_tokens,
            "ordinary_silver_max_tokens": SILVER_MAX_TOKENS,
            "original_text_and_gold_preserved": true,
            "text_scope": "complete existing silver JSONL row; may already be an upstream source excerpt",
            "caller_must_verify_held_data_separation": true,
            "eligible_distinct_original_rows": candidates.len(),
            "omitted_non_original_pieces_or_gold_mismatches": omitted_pieces,
            "selected_documents": selected_manifest.len(),
            "content_tokens": content_tokens,
            "group_order": ["person", "org", "address", "fully_negative"],
            "support_documents": support_documents,
            "kind_order": ["person", "org", "address"],
            "gold_spans": gold_spans,
            "sources": source_manifest,
            "documents": selected_manifest,
        }),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::{breaks_of, encode_document};

    fn doc(text: &str, gold: Vec<KindSpan>) -> DetectorDoc {
        DetectorDoc {
            text: text.into(),
            enc: encode_document(text, &gold, &tessera::internal::FeatureConfig::default())
                .unwrap(),
            breaks: breaks_of(text),
            gold,
        }
    }

    fn candidates() -> (Vec<DetectorDoc>, Vec<Candidate>) {
        let mut docs = Vec::new();
        let mut candidates = Vec::new();
        for group in 0..4 {
            for number in 0..6 {
                let text = format!("Sample {group} item {number}");
                let gold = if group < 3 {
                    vec![KindSpan {
                        kind: group,
                        start: 0,
                        end: 6,
                    }]
                } else {
                    Vec::new()
                };
                let document = doc(&text, gold);
                let groups = support(&document).unwrap();
                candidates.push(Candidate {
                    index: docs.len(),
                    source: number % 3,
                    line: number + 1,
                    provenance: json!({}),
                    groups,
                    form: form(&document, groups).unwrap(),
                    rank: Sha256::digest(text.as_bytes()).into(),
                });
                docs.push(document);
            }
        }
        (docs, candidates)
    }

    #[test]
    fn deterministic_selection_preserves_native_documents_and_balances_support() {
        let (docs, candidates) = candidates();
        let before: Vec<_> = docs
            .iter()
            .map(|doc| {
                (
                    doc.text.clone(),
                    doc.enc.clone(),
                    doc.gold.clone(),
                    doc.breaks.clone(),
                )
            })
            .collect();
        let selected = select(&candidates, 16).unwrap();
        assert_eq!(selected, select(&candidates, 16).unwrap());
        assert_eq!(selected.len(), 16);
        assert_eq!(selected.iter().collect::<BTreeSet<_>>().len(), 16);
        for group in 0..4 {
            assert_eq!(
                selected
                    .iter()
                    .filter(|&&index| candidates[index].groups[group])
                    .count(),
                4
            );
        }
        assert_eq!(
            selected
                .iter()
                .map(|&index| candidates[index].source)
                .collect::<BTreeSet<_>>()
                .len(),
            3
        );
        for (index, document) in docs.iter().enumerate() {
            assert_eq!(
                (
                    &document.text,
                    &document.enc,
                    &document.gold,
                    &document.breaks
                ),
                (
                    &before[index].0,
                    &before[index].1,
                    &before[index].2,
                    &before[index].3
                )
            );
        }
    }

    #[test]
    fn missing_kind_negative_support_and_small_budgets_fail() {
        let (_, candidates) = candidates();
        assert!(select(&candidates, 15).is_err());
        for group in 0..4 {
            let incomplete: Vec<_> = candidates
                .iter()
                .filter(|candidate| !candidate.groups[group])
                .collect();
            let copied: Vec<_> = incomplete
                .into_iter()
                .map(|candidate| Candidate {
                    index: candidate.index,
                    source: candidate.source,
                    line: candidate.line,
                    provenance: candidate.provenance.clone(),
                    groups: candidate.groups,
                    form: candidate.form.clone(),
                    rank: candidate.rank,
                })
                .collect();
            assert!(select(&copied, 32).is_err());
        }
    }

    #[test]
    fn unsupported_gold_kind_fails_without_indexing_support() {
        let mut document = doc("Sample", Vec::new());
        document.gold.push(KindSpan {
            kind: 3,
            start: 0,
            end: 6,
        });
        assert!(support(&document).is_err());
    }

    #[test]
    fn negative_support_means_no_model_gold_even_when_rules_exist() {
        let document = doc("Contact help@example.com", Vec::new());
        assert_eq!(support(&document).unwrap(), [false, false, false, true]);
        assert!(
            document
                .enc
                .flags
                .iter()
                .any(|flags| flags & tessera::internal::flag::IN_RULE_SPAN != 0)
        );
    }

    #[test]
    fn fills_remaining_budget_without_duplicate_indices_or_losing_quotas() {
        let (_, candidates) = candidates();
        let selected = select(&candidates, 20).unwrap();
        assert_eq!(selected.len(), 20);
        assert_eq!(selected.iter().collect::<BTreeSet<_>>().len(), 20);
        assert_eq!(select(&candidates, 32).unwrap().len(), candidates.len());
    }

    #[test]
    fn prepare_keeps_original_rows_and_native_arrays_with_limits_and_duplicate_sources() {
        let (docs, _) = candidates();
        let directory = tempfile::tempdir().unwrap();
        let mut cfg = crate::config::load(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../configs/detector-shared-v6.toml"),
        )
        .unwrap();
        let paths: Vec<_> = (0..3)
            .map(|source| directory.path().join(format!("source-{source}.jsonl")))
            .collect();
        for (source, path) in paths.iter().enumerate() {
            let mut rows = Vec::new();
            for (index, document) in docs
                .iter()
                .enumerate()
                .filter(|(index, _)| index % 3 == source)
            {
                rows.push(json!({
                    "id": format!("original-{index}"),
                    "source_document_key": format!("source-document-{index}"),
                    "text": document.text,
                    "country": "US",
                    "entities": document.gold.iter().map(|span| json!({
                        "kind": detector::KINDS[span.kind].as_str(),
                        "start": span.start,
                        "end": span.end,
                    })).collect::<Vec<_>>(),
                }));
            }
            if source == 0 {
                rows.push(json!({"text": "long ".repeat(900) + "\n\nshort tail", "country": "US", "entities": []}));
            }
            if source == 1 {
                let duplicate = &docs[0];
                rows.push(json!({"text": duplicate.text, "country": "US", "entities": [{"kind": "person", "start": 0, "end": 6}]}));
            }
            std::fs::write(
                path,
                rows.iter()
                    .map(Value::to_string)
                    .collect::<Vec<_>>()
                    .join("\n")
                    + "\n",
            )
            .unwrap();
        }
        cfg.detector.as_mut().unwrap().silver = paths
            .iter()
            .map(|path| path.display().to_string())
            .collect();
        let prepared = prepare(&cfg, 20, 8).unwrap();
        let second = prepare(&cfg, 20, 8).unwrap();
        assert_eq!(prepared.manifest, second.manifest);
        assert_eq!(prepared.documents.len(), 20);
        assert_eq!(prepared.manifest["eligible_distinct_original_rows"], 24);
        assert_eq!(
            prepared.manifest["omitted_non_original_pieces_or_gold_mismatches"],
            1
        );
        assert_eq!(
            prepared
                .documents
                .iter()
                .map(|document| &document.text)
                .collect::<BTreeSet<_>>()
                .len(),
            20
        );
        for document in &prepared.documents {
            let original = docs
                .iter()
                .find(|original| original.text == document.text)
                .unwrap();
            let mut encoded =
                encode_document(&original.text, &original.gold, &cfg.features.to_tessera())
                    .unwrap();
            encoded.country = "US".into();
            assert_eq!(document.enc, encoded);
            assert_eq!(document.gold, original.gold);
            assert_eq!(document.breaks, original.breaks);
            assert!(document.enc.token_spans.len() <= 8);
        }
        let manifest_rows = prepared.manifest["documents"].as_array().unwrap();
        assert!(
            manifest_rows
                .iter()
                .all(|row| row["source_provenance"]["id"].as_str().is_some())
        );
        assert!(
            manifest_rows
                .iter()
                .all(|row| row["original_source_row"] == true && row["derived_piece"] == false)
        );
        assert!(prepare(&cfg, 20, 0).is_err());
        assert!(prepare(&cfg, 20, 1).is_err());
    }
}
