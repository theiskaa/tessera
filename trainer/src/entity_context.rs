//! Read-only feasibility of training views that preserve entity context and original labels.

use std::collections::BTreeSet;

use anyhow::{Context, ensure};
use serde::Serialize;

use crate::detector::{DETECTOR_LABELS, DetectorDoc};

const MIN_PARENT_TOKENS: usize = 128;
const MAX_POSITIVE_FRACTION: f64 = 0.20;

#[derive(Default, Serialize)]
struct Counts {
    parent_documents: usize,
    eligible_parents: usize,
    unique_candidate_windows: usize,
    unique_useful_windows: usize,
    full_parent_rejections: usize,
    no_density_gain_rejections: usize,
    parent_content_tokens: usize,
    parent_positive_tokens: usize,
    eligible_parent_content_tokens: usize,
    eligible_parent_positive_tokens: usize,
    useful_window_content_tokens: usize,
    useful_window_positive_tokens: usize,
    min_useful_window_tokens: Option<usize>,
    max_useful_window_tokens: Option<usize>,
    covered_gold_spans: [usize; 3],
    window_gold_span_presentations: [usize; 3],
}

fn fraction(positive: usize, total: usize) -> Option<f64> {
    (total > 0).then(|| positive as f64 / total as f64)
}

impl Counts {
    fn json(&self) -> anyhow::Result<serde_json::Value> {
        let mut value = serde_json::to_value(self)?;
        value["positive_token_fraction"] = serde_json::json!({
            "all_parents": fraction(self.parent_positive_tokens, self.parent_content_tokens),
            "eligible_parents": fraction(self.eligible_parent_positive_tokens, self.eligible_parent_content_tokens),
            "useful_windows": fraction(self.useful_window_positive_tokens, self.useful_window_content_tokens),
        });
        Ok(value)
    }
}

/// Validate original encodings and return each gold span's content-token range and kind.
pub(crate) fn aligned_gold(
    doc: &DetectorDoc,
    index: usize,
) -> anyhow::Result<Vec<(usize, usize, usize)>> {
    let length = doc.enc.token_spans.len();
    ensure!(
        length > 0,
        "context audit parent {index} has no content tokens"
    );
    ensure!(
        [
            doc.enc.ngram_ids.len(),
            doc.enc.script.len(),
            doc.enc.shape.len(),
            doc.enc.flags.len(),
            doc.enc.labels.len(),
            doc.breaks.len()
        ]
        .iter()
        .all(|&count| count == length),
        "context audit parent {index} has inconsistent token dimensions"
    );
    let mut previous_end = 0;
    for &(start, end) in &doc.enc.token_spans {
        ensure!(
            start < end
                && start >= previous_end
                && doc.text.get(start as usize..end as usize).is_some(),
            "context audit parent {index} has invalid token byte ranges"
        );
        previous_end = end;
    }
    ensure!(
        doc.enc
            .labels
            .iter()
            .all(|&label| usize::from(label) < DETECTOR_LABELS),
        "context audit parent {index} has an invalid BIO label"
    );
    let mut labels = vec![0; length];
    let mut occupied = vec![false; length];
    let mut spans = Vec::with_capacity(doc.gold.len());
    for gold in &doc.gold {
        ensure!(
            gold.kind < 3 && gold.start < gold.end,
            "context audit parent {index} has invalid gold kind or range"
        );
        let first = doc
            .enc
            .token_spans
            .iter()
            .position(|token| token.0 == gold.start)
            .with_context(|| {
                format!("context audit parent {index} gold start is not token aligned")
            })?;
        let last = doc
            .enc
            .token_spans
            .iter()
            .position(|token| token.1 == gold.end)
            .with_context(|| {
                format!("context audit parent {index} gold end is not token aligned")
            })?;
        ensure!(
            first <= last,
            "context audit parent {index} has reversed gold tokens"
        );
        for position in first..=last {
            ensure!(
                !occupied[position],
                "context audit parent {index} has overlapping gold"
            );
            occupied[position] = true;
            labels[position] = (2 * gold.kind + if position == first { 1 } else { 2 }) as u8;
        }
        spans.push((first, last + 1, gold.kind));
    }
    ensure!(
        labels == doc.enc.labels,
        "context audit parent {index} BIO labels differ from original gold"
    );
    Ok(spans)
}

/// Expand an anchor until every intersected gold also retains the requested context.
pub(crate) fn closure(
    anchor: (usize, usize, usize),
    gold: &[(usize, usize, usize)],
    length: usize,
    radius: usize,
) -> (usize, usize) {
    let mut range = (
        anchor.0.saturating_sub(radius),
        anchor.1.saturating_add(radius).min(length),
    );
    loop {
        let previous = range;
        for &(start, end, _) in gold {
            if start < range.1 && range.0 < end {
                range.0 = range.0.min(start.saturating_sub(radius));
                range.1 = range.1.max(end.saturating_add(radius).min(length));
            }
        }
        if range == previous {
            return range;
        }
    }
}

/// Audit exploratory entity-context windows without modifying or re-encoding any example.
///
/// Eligibility is a hypothesis, not a readiness gate. Every intersected gold span must keep
/// the requested context on both sides, except at the true parent boundary. Gold coverage
/// counts distinct parent spans; presentations also count their appearances in overlapping
/// useful windows. Input parents must carry training BIO labels matching their original gold.
pub(crate) fn audit(
    docs: &[DetectorDoc],
    source_pieces: &[usize],
    radius: usize,
) -> anyhow::Result<serde_json::Value> {
    ensure!(radius > 0, "context audit radius must be positive");
    ensure!(!docs.is_empty(), "context audit needs training parents");
    ensure!(
        source_pieces.iter().all(|&count| count > 0),
        "context audit source ranges must be nonempty"
    );
    let count = source_pieces
        .iter()
        .try_fold(0usize, |total, &count| total.checked_add(count))
        .context("context audit source ranges overflow")?;
    ensure!(
        count == docs.len(),
        "context audit source ranges differ from parent count"
    );
    let mut total = Counts::default();
    let mut sources = Vec::with_capacity(source_pieces.len());
    let mut offset = 0;
    for (source_index, &count) in source_pieces.iter().enumerate() {
        let mut source = Counts::default();
        for (index, doc) in docs.iter().enumerate().skip(offset).take(count) {
            let gold = aligned_gold(doc, index)?;
            let length = doc.enc.labels.len();
            let positive = doc.enc.labels.iter().filter(|&&label| label != 0).count();
            let eligible = length >= MIN_PARENT_TOKENS
                && positive > 0
                && positive as f64 / length as f64 <= MAX_POSITIVE_FRACTION;
            for counts in [&mut total, &mut source] {
                counts.parent_documents += 1;
                counts.parent_content_tokens += length;
                counts.parent_positive_tokens += positive;
                if eligible {
                    counts.eligible_parents += 1;
                    counts.eligible_parent_content_tokens += length;
                    counts.eligible_parent_positive_tokens += positive;
                }
            }
            if !eligible {
                continue;
            }
            let ranges: BTreeSet<_> = gold
                .iter()
                .copied()
                .map(|anchor| closure(anchor, &gold, length, radius))
                .collect();
            let mut covered = vec![false; gold.len()];
            for (start, end) in ranges {
                for counts in [&mut total, &mut source] {
                    counts.unique_candidate_windows += 1;
                }
                if start == 0 && end == length {
                    for counts in [&mut total, &mut source] {
                        counts.full_parent_rejections += 1;
                    }
                    continue;
                }
                let window_tokens = end - start;
                let window_positive = doc.enc.labels[start..end]
                    .iter()
                    .filter(|&&label| label != 0)
                    .count();
                if (window_positive as u128) * (length as u128)
                    <= (positive as u128) * (window_tokens as u128)
                {
                    for counts in [&mut total, &mut source] {
                        counts.no_density_gain_rejections += 1;
                    }
                    continue;
                }
                for counts in [&mut total, &mut source] {
                    counts.unique_useful_windows += 1;
                    counts.useful_window_content_tokens += window_tokens;
                    counts.useful_window_positive_tokens += window_positive;
                    counts.min_useful_window_tokens = Some(
                        counts
                            .min_useful_window_tokens
                            .map_or(window_tokens, |previous| previous.min(window_tokens)),
                    );
                    counts.max_useful_window_tokens = Some(
                        counts
                            .max_useful_window_tokens
                            .map_or(window_tokens, |previous| previous.max(window_tokens)),
                    );
                }
                for (gold_index, &(first, last, kind)) in gold.iter().enumerate() {
                    if start <= first && last <= end {
                        for counts in [&mut total, &mut source] {
                            counts.window_gold_span_presentations[kind] += 1;
                            if !covered[gold_index] {
                                counts.covered_gold_spans[kind] += 1;
                            }
                        }
                        covered[gold_index] = true;
                    }
                }
            }
        }
        sources.push(serde_json::json!({"source_index": source_index, "counts": source.json()?}));
        offset += count;
    }
    Ok(serde_json::json!({
        "scope": "read-only exploratory training-window feasibility; thresholds are hypotheses, not readiness gates or evidence of learning",
        "eligibility": {"minimum_parent_content_tokens": MIN_PARENT_TOKENS,
            "positive_token_fraction": "fraction of all non-O BIO tokens, across all three kinds",
            "minimum_positive_token_fraction_exclusive": 0.0,
            "maximum_positive_token_fraction_inclusive": MAX_POSITIVE_FRACTION},
        "context": {"radius_content_tokens": radius,
            "boundary_policy": "recursive closure for every intersected gold; clipping only at true parent ends",
            "window_token_limit": "parent token count; no smaller cap imposed",
            "deduplication": "token ranges within each parent only",
            "whole_parent_views": "rejected",
            "density_gain": "strictly greater positive-token fraction than the parent"},
        "kind_order": ["person", "org", "address"],
        "coverage_scope": "distinct gold spans are keyed by parent and gold index; presentations include overlap between useful windows",
        "total": total.json()?, "sources": sources,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Encoded;
    use crate::detector::KindSpan;

    fn doc(length: usize, spans: &[(usize, usize, usize)]) -> DetectorDoc {
        let words: Vec<_> = (0..length).map(|index| format!("é名{index}")).collect();
        let text = words.join(" ");
        let mut offset = 0u32;
        let token_spans: Vec<_> = words
            .iter()
            .map(|word| {
                let start = offset;
                offset += word.len() as u32 + 1;
                (start, offset - 1)
            })
            .collect();
        let mut labels = vec![0; length];
        let gold = spans
            .iter()
            .map(|&(first, end, kind)| {
                labels[first] = (2 * kind + 1) as u8;
                labels[first + 1..end].fill((2 * kind + 2) as u8);
                KindSpan {
                    kind,
                    start: token_spans[first].0,
                    end: token_spans[end - 1].1,
                }
            })
            .collect();
        DetectorDoc {
            text,
            gold,
            breaks: (0..length).map(|index| index == 0).collect(),
            enc: Encoded {
                token_spans,
                labels,
                ngram_ids: vec![vec![1]; length],
                script: vec![1; length],
                shape: vec![1; length],
                flags: vec![0; length],
                country: "US".into(),
            },
        }
    }

    #[test]
    fn closure_cascades_and_duplicate_anchor_views_include_all_kinds() {
        let original = doc(128, &[(40, 41, 0), (43, 44, 1), (46, 47, 2)]);
        let before = original.enc.clone();
        let spans = aligned_gold(&original, 0).unwrap();
        assert_eq!(closure(spans[0], &spans, 128, 4), (36, 51));
        let result = audit(std::slice::from_ref(&original), &[1], 4).unwrap();
        assert_eq!(result["total"]["unique_candidate_windows"], 1);
        assert_eq!(result["total"]["unique_useful_windows"], 1);
        assert_eq!(
            result["total"]["covered_gold_spans"],
            serde_json::json!([1, 1, 1])
        );
        assert_eq!(original.enc, before);
        assert_eq!(
            &original.text[original.gold[0].start as usize..original.gold[0].end as usize],
            "é名40"
        );
    }

    #[test]
    fn context_clips_only_at_actual_parent_ends() {
        let original = doc(128, &[(0, 1, 0), (127, 128, 2)]);
        let spans = aligned_gold(&original, 0).unwrap();
        assert_eq!(closure(spans[0], &spans, 128, 4), (0, 5));
        assert_eq!(closure(spans[1], &spans, 128, 4), (123, 128));
        let result = audit(&[original], &[1], 4).unwrap();
        assert_eq!(result["total"]["unique_useful_windows"], 2);
        assert_eq!(result["total"]["min_useful_window_tokens"], 5);
    }

    #[test]
    fn full_parent_and_no_density_gain_views_are_rejected() {
        let full = doc(128, &[(0, 1, 0), (127, 128, 1)]);
        let result = audit(&[full], &[1], 128).unwrap();
        assert_eq!(result["total"]["full_parent_rejections"], 1);
        let mut spans = vec![(40, 41, 0)];
        spans.extend((100..124).map(|position| (position, position + 1, 1)));
        let result = audit(&[doc(128, &spans)], &[1], 4).unwrap();
        assert_eq!(result["total"]["no_density_gain_rejections"], 1);
        assert_eq!(result["total"]["unique_useful_windows"], 1);
    }

    #[test]
    fn eligibility_excludes_short_negative_and_dense_parents() {
        let docs = [
            doc(127, &[(40, 41, 0)]),
            doc(128, &[]),
            doc(128, &[(40, 66, 0)]),
        ];
        let result = audit(&docs, &[1, 2], 32).unwrap();
        assert_eq!(result["total"]["eligible_parents"], 0);
        assert_eq!(result["sources"][0]["counts"]["parent_documents"], 1);
        assert_eq!(result["sources"][1]["counts"]["parent_documents"], 2);
        assert_eq!(
            result["total"]["positive_token_fraction"]["useful_windows"],
            serde_json::Value::Null
        );
    }

    #[test]
    fn rejects_zero_ranges_alignment_dimensions_and_label_disagreement() {
        let original = doc(128, &[(40, 41, 0)]);
        assert!(audit(std::slice::from_ref(&original), &[1], 0).is_err());
        assert!(audit(&[], &[], 32).is_err());
        assert!(audit(std::slice::from_ref(&original), &[0, 1], 32).is_err());
        assert!(audit(std::slice::from_ref(&original), &[2], 32).is_err());
        assert!(audit(std::slice::from_ref(&original), &[usize::MAX, 1], 32).is_err());
        let mut bad = original.clone();
        bad.gold[0].start += 1;
        assert!(audit(&[bad], &[1], 32).is_err());
        let mut bad = original.clone();
        bad.gold[0].kind = 3;
        assert!(audit(&[bad], &[1], 32).is_err());
        let mut bad = original.clone();
        bad.enc.labels[40] = 0;
        assert!(audit(&[bad], &[1], 32).is_err());
        let mut bad = original;
        bad.enc.flags.pop();
        assert!(audit(&[bad], &[1], 32).is_err());
    }
}
