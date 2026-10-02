//! Verified contiguous views of sparse real training examples, with address parents protected.

use std::collections::BTreeSet;

use anyhow::{Context, ensure};
use tessera::internal::{FeatureConfig, decode_detector, flag, scan_rules};

use crate::detector::{DETECTOR_LABELS, DetectorDoc, KINDS, KindSpan, breaks_of, encode_document};
use crate::entity_context::{aligned_gold, closure};

/// A derived document and its exact byte range in an unchanged training parent.
pub(crate) struct View {
    pub(crate) parent: usize,
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) doc: DetectorDoc,
}

fn byte_range(doc: &DetectorDoc, start: usize, end: usize) -> (usize, usize) {
    let left = if start == 0 {
        0
    } else {
        doc.enc.token_spans[start - 1].1 as usize
    };
    let right = if end == doc.enc.token_spans.len() {
        doc.text.len()
    } else {
        doc.enc.token_spans[end].0 as usize
    };
    (left, right)
}

fn close_range(
    doc: &DetectorDoc,
    gold: &[(usize, usize, usize)],
    rules: &[(usize, usize)],
    mut range: (usize, usize),
    halo: usize,
) -> (usize, usize) {
    let length = doc.enc.token_spans.len();
    loop {
        let previous = range;
        for &span in gold {
            if span.0 < range.1 && range.0 < span.1 {
                let expanded = closure(span, gold, length, halo);
                range.0 = range.0.min(expanded.0);
                range.1 = range.1.max(expanded.1);
            }
        }
        let (left, right) = byte_range(doc, range.0, range.1);
        for &(start, end) in rules {
            if start < right && left < end && (start < left || end > right) {
                for (index, &(token_start, token_end)) in doc.enc.token_spans.iter().enumerate() {
                    if (token_start as usize) < end && start < token_end as usize {
                        range.0 = range.0.min(index);
                        range.1 = range.1.max(index + 1);
                    }
                }
            }
        }
        if range == previous {
            return range;
        }
    }
}

fn protected_inputs_match(
    parent: &DetectorDoc,
    view: &DetectorDoc,
    range: (usize, usize),
    byte_start: usize,
    gold: &[(usize, usize, usize)],
    radius: usize,
) -> bool {
    let original = &parent.enc;
    let derived = &view.enc;
    if derived.token_spans.len() != range.1 - range.0
        || derived.labels != original.labels[range.0..range.1]
    {
        return false;
    }
    for (offset, &(start, end)) in derived.token_spans.iter().enumerate() {
        let expected = original.token_spans[range.0 + offset];
        if start as usize + byte_start != expected.0 as usize
            || end as usize + byte_start != expected.1 as usize
        {
            return false;
        }
    }
    for &(first, last, _) in gold {
        if first < range.0 || last > range.1 {
            continue;
        }
        let left = first.saturating_sub(radius);
        let right = last.saturating_add(radius).min(original.labels.len());
        if left < range.0 || right > range.1 {
            return false;
        }
        for position in left..right {
            let offset = position - range.0;
            if original.ngram_ids[position] != derived.ngram_ids[offset]
                || original.script[position] != derived.script[offset]
                || original.shape[position] != derived.shape[offset]
                || original.flags[position] != derived.flags[offset]
                || parent.breaks[position] != view.breaks[offset]
            {
                return false;
            }
        }
    }
    true
}

fn verify_gold_round_trip(doc: &DetectorDoc) -> anyhow::Result<()> {
    let count = doc
        .enc
        .labels
        .len()
        .checked_mul(DETECTOR_LABELS)
        .context("context-view probability dimensions overflow")?;
    let mut probabilities = vec![0.0; count];
    for (index, &label) in doc.enc.labels.iter().enumerate() {
        ensure!(
            usize::from(label) < DETECTOR_LABELS,
            "context view has an invalid BIO label"
        );
        probabilities[index * DETECTOR_LABELS + usize::from(label)] = 1.0;
    }
    let masked: Vec<_> = doc
        .enc
        .flags
        .iter()
        .map(|bits| bits & flag::IN_RULE_SPAN != 0)
        .collect();
    let mut decoded = Vec::new();
    for span in decode_detector(&probabilities, &masked, &doc.breaks) {
        let kind = KINDS
            .iter()
            .position(|&kind| kind == span.kind)
            .context("context view decoded an unsupported entity kind")?;
        let first = doc
            .enc
            .token_spans
            .get(span.first)
            .context("context view decoded an invalid first token")?;
        let last = doc
            .enc
            .token_spans
            .get(span.last)
            .context("context view decoded an invalid last token")?;
        decoded.push(KindSpan {
            kind,
            start: first.0,
            end: last.1,
        });
    }
    decoded.sort_by_key(|span| (span.start, span.end, span.kind));
    let mut expected = doc.gold.clone();
    expected.sort_by_key(|span| (span.start, span.end, span.kind));
    ensure!(
        decoded == expected,
        "context-view native gold round-trip differs from complete original gold"
    );
    Ok(())
}

fn materialize(
    parent: &DetectorDoc,
    parent_index: usize,
    gold: &[(usize, usize, usize)],
    rules: &[(usize, usize)],
    initial: (usize, usize),
    fc: &FeatureConfig,
    radius: usize,
) -> anyhow::Result<Option<View>> {
    let length = parent.enc.labels.len();
    let parent_positive = parent
        .enc
        .labels
        .iter()
        .filter(|&&label| label != 0)
        .count();
    let halo = radius
        .checked_add(1)
        .context("context-view halo overflow")?;
    let mut range = initial;
    loop {
        range = close_range(parent, gold, rules, range, halo);
        if range == (0, length) {
            return Ok(None);
        }
        let (start, end) = byte_range(parent, range.0, range.1);
        let text = parent
            .text
            .get(start..end)
            .context("context-view range is not on UTF-8 boundaries")?;
        let mut derived_gold = Vec::new();
        for span in &parent.gold {
            if (span.start as usize) < end && start < span.end as usize {
                ensure!(
                    start <= span.start as usize && span.end as usize <= end,
                    "context view cuts an original gold span"
                );
                derived_gold.push(KindSpan {
                    kind: span.kind,
                    start: span.start - start as u32,
                    end: span.end - start as u32,
                });
            }
        }
        if let Ok(mut enc) = encode_document(text, &derived_gold, fc) {
            enc.country = parent.enc.country.clone();
            let derived = DetectorDoc {
                text: text.to_owned(),
                enc,
                gold: derived_gold,
                breaks: breaks_of(text),
            };
            if protected_inputs_match(parent, &derived, range, start, gold, radius) {
                let positive = derived
                    .enc
                    .labels
                    .iter()
                    .filter(|&&label| label != 0)
                    .count();
                if (positive as u128) * (length as u128)
                    <= (parent_positive as u128) * (derived.enc.labels.len() as u128)
                {
                    return Ok(None);
                }
                verify_gold_round_trip(&derived)?;
                return Ok(Some(View {
                    parent: parent_index,
                    start,
                    end,
                    doc: derived,
                }));
            }
        }
        range.0 = range.0.saturating_sub(1);
        range.1 = range.1.saturating_add(1).min(length);
    }
}

/// Re-encode denser views, preserving every included entity's original neural input context.
///
/// Address-bearing parents remain whole. Other parents need at least 128 content tokens
/// and at most 20% non-O tokens. These are candidate-selection assumptions, not quality
/// thresholds. Differences outside protected contexts and Viterbi decoding can still
/// change predictions; no learning benefit follows from these preparation checks.
pub(crate) fn prepare(
    parents: &[DetectorDoc],
    fc: &FeatureConfig,
    radius: usize,
) -> anyhow::Result<Vec<View>> {
    ensure!(radius > 0, "context-view radius must be positive");
    let halo = radius
        .checked_add(1)
        .context("context-view halo overflow")?;
    let mut views = Vec::new();
    for (index, parent) in parents.iter().enumerate() {
        let gold = aligned_gold(parent, index)?;
        let length = parent.enc.labels.len();
        let positive = parent
            .enc
            .labels
            .iter()
            .filter(|&&label| label != 0)
            .count();
        if length < 128
            || positive == 0
            || positive as f64 / length as f64 > 0.20
            || parent.gold.iter().any(|span| span.kind == 2)
        {
            continue;
        }
        let rules: Vec<_> = scan_rules(&parent.text, &[])
            .iter()
            .map(|span| (span.start, span.end))
            .collect();
        let ranges: BTreeSet<_> = gold
            .iter()
            .copied()
            .map(|anchor| closure(anchor, &gold, length, halo))
            .collect();
        let mut emitted = BTreeSet::new();
        for range in ranges {
            if let Some(view) = materialize(parent, index, &gold, &rules, range, fc, radius)?
                && emitted.insert((view.start, view.end))
            {
                views.push(view);
            }
        }
    }
    Ok(views)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn native_parent(text: String, surfaces: &[(&str, usize)]) -> DetectorDoc {
        let gold = surfaces
            .iter()
            .map(|&(surface, kind)| {
                let start = text.find(surface).unwrap() as u32;
                KindSpan {
                    kind,
                    start,
                    end: start + surface.len() as u32,
                }
            })
            .collect::<Vec<_>>();
        let mut enc = encode_document(&text, &gold, &FeatureConfig::default()).unwrap();
        enc.country = "US".into();
        DetectorDoc {
            breaks: breaks_of(&text),
            text,
            gold,
            enc,
        }
    }

    fn parent(length: usize, inserts: &[(usize, &str, usize)]) -> DetectorDoc {
        let mut words: Vec<_> = (0..length).map(|index| format!("word{index}")).collect();
        for &(index, value, _) in inserts {
            words[index] = value.to_owned();
        }
        let text = words.join(" ");
        let gold: Vec<_> = inserts
            .iter()
            .map(|&(_, value, kind)| {
                let start = text.find(value).unwrap() as u32;
                KindSpan {
                    kind,
                    start,
                    end: start + value.len() as u32,
                }
            })
            .collect();
        let mut enc = encode_document(&text, &gold, &FeatureConfig::default()).unwrap();
        enc.country = "US".into();
        DetectorDoc {
            breaks: breaks_of(&text),
            text,
            gold,
            enc,
        }
    }

    #[test]
    fn midline_unicode_views_reencode_with_exact_protected_inputs() {
        let doc = parent(200, &[(80, "Émilie", 0)]);
        let original = doc.enc.clone();
        let views = prepare(std::slice::from_ref(&doc), &FeatureConfig::default(), 32).unwrap();
        assert_eq!(views.len(), 1);
        let view = &views[0];
        assert_eq!(view.doc.enc.labels.len(), 67);
        assert_eq!(&doc.text[view.start..view.end], view.doc.text);
        assert_eq!(
            &view.doc.text[view.doc.gold[0].start as usize..view.doc.gold[0].end as usize],
            "Émilie"
        );
        assert_eq!(doc.enc, original);
        let gold = aligned_gold(&doc, 0).unwrap();
        assert!(protected_inputs_match(
            &doc,
            &view.doc,
            (47, 114),
            view.start,
            &gold,
            32
        ));
    }

    #[test]
    fn intersected_gold_is_complete_and_address_parents_remain_whole() {
        let no_address = parent(200, &[(80, "Émilie", 0), (100, "Harbor Bureau", 1)]);
        let views = prepare(
            std::slice::from_ref(&no_address),
            &FeatureConfig::default(),
            32,
        )
        .unwrap();
        assert_eq!(views.len(), 1);
        assert_eq!(views[0].doc.gold.len(), 2);
        let address = parent(200, &[(80, "Émilie", 0), (100, "10 Elm Street", 2)]);
        assert!(
            prepare(&[address], &FeatureConfig::default(), 32)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn rejects_invalid_parent_encodings_and_zero_radius() {
        let mut doc = parent(200, &[(80, "Émilie", 0)]);
        assert!(prepare(std::slice::from_ref(&doc), &FeatureConfig::default(), 0).is_err());
        doc.enc.labels[80] = 0;
        assert!(prepare(&[doc], &FeatureConfig::default(), 32).is_err());
    }

    #[test]
    fn short_negative_and_full_parent_context_do_not_create_views() {
        let short = parent(127, &[(60, "Émilie", 0)]);
        let negative = parent(200, &[]);
        let full = parent(128, &[(60, "Émilie", 0)]);
        assert!(
            prepare(&[short, negative], &FeatureConfig::default(), 32)
                .unwrap()
                .is_empty()
        );
        assert!(
            prepare(&[full], &FeatureConfig::default(), 128)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn crops_never_cut_original_rule_spans() {
        let mut doc = parent(240, &[(80, "Émilie", 0)]);
        doc.text = doc.text.replace("word47", "contact@example.invalid");
        let start = doc.text.find("Émilie").unwrap() as u32;
        doc.gold[0].start = start;
        doc.gold[0].end = start + "Émilie".len() as u32;
        doc.enc = encode_document(&doc.text, &doc.gold, &FeatureConfig::default()).unwrap();
        doc.enc.country = "US".into();
        doc.breaks = breaks_of(&doc.text);
        let views = prepare(std::slice::from_ref(&doc), &FeatureConfig::default(), 32).unwrap();
        assert!(!views.is_empty());
        for view in views {
            for rule in scan_rules(&doc.text, &[]) {
                if rule.start < view.end && view.start < rule.end {
                    assert!(view.start <= rule.start && rule.end <= view.end);
                }
            }
        }
    }

    #[test]
    fn short_line_flag_drift_expands_beyond_the_initial_guard() {
        let mut words = (0..180)
            .map(|index| format!("word{index}"))
            .collect::<Vec<_>>();
        words[65] = "Émilie".into();
        let text = format!(
            "{}\n{}\n{}",
            words[..37].join(" "),
            words[37..39].join(" "),
            words[39..].join(" ")
        );
        let doc = native_parent(text, &[("Émilie", 0)]);
        let gold = aligned_gold(&doc, 0).unwrap();
        let initial = closure(gold[0], &gold, doc.enc.labels.len(), 33);
        assert_eq!(initial, (32, 99));
        let (start, end) = byte_range(&doc, initial.0, initial.1);
        let trial = native_parent(doc.text[start..end].to_owned(), &[("Émilie", 0)]);
        assert_ne!(
            doc.enc.flags[37] & flag::IN_SHORT_LINE_BLOCK,
            trial.enc.flags[5] & flag::IN_SHORT_LINE_BLOCK
        );
        assert!(!protected_inputs_match(
            &doc, &trial, initial, start, &gold, 32
        ));
        let views = prepare(std::slice::from_ref(&doc), &FeatureConfig::default(), 32).unwrap();
        assert_eq!(views.len(), 1);
        let range = (28, 103);
        assert_eq!(views[0].doc.enc.labels.len(), range.1 - range.0);
        assert!(protected_inputs_match(
            &doc,
            &views[0].doc,
            range,
            views[0].start,
            &gold,
            32
        ));
    }

    #[test]
    fn drift_expansion_recloses_newly_intersected_gold() {
        let mut words = (0..200)
            .map(|index| format!("word{index}"))
            .collect::<Vec<_>>();
        words[64] = "Harbor".into();
        words[65] = "Bureau".into();
        words[100] = "Émilie".into();
        let text = format!(
            "{}\n{}\n{}",
            words[..72].join(" "),
            words[72..74].join(" "),
            words[74..].join(" ")
        );
        let doc = native_parent(text, &[("Émilie", 0), ("Harbor Bureau", 1)]);
        let gold = aligned_gold(&doc, 0).unwrap();
        let initial = closure(gold[0], &gold, doc.enc.labels.len(), 33);
        assert_eq!(initial, (67, 134));
        let view = materialize(&doc, 0, &gold, &[], initial, &FeatureConfig::default(), 32)
            .unwrap()
            .unwrap();
        assert_eq!(view.doc.gold.len(), 2);
        for span in &view.doc.gold {
            assert!(doc.gold.iter().any(|original| {
                original.kind == span.kind
                    && original.start as usize == span.start as usize + view.start
                    && original.end as usize == span.end as usize + view.start
            }));
        }
        let first = doc
            .enc
            .token_spans
            .iter()
            .position(|span| span.0 as usize >= view.start)
            .unwrap();
        let range = (first, first + view.doc.enc.labels.len());
        assert!(range.0 <= gold[1].0 - 32);
        assert!(protected_inputs_match(
            &doc, &view.doc, range, view.start, &gold, 32
        ));
    }

    #[test]
    fn protected_radius_endpoints_and_all_labels_fail_closed() {
        let doc = parent(200, &[(80, "Émilie", 0)]);
        let views = prepare(std::slice::from_ref(&doc), &FeatureConfig::default(), 32).unwrap();
        let view = &views[0];
        let gold = aligned_gold(&doc, 0).unwrap();
        let range = (47, 114);
        for offset in [1, 65] {
            let mut drift = view.doc.clone();
            drift.enc.flags[offset] ^= flag::LINE_START;
            assert!(!protected_inputs_match(
                &doc, &drift, range, view.start, &gold, 32
            ));
            let mut drift = view.doc.clone();
            drift.breaks[offset] = !drift.breaks[offset];
            assert!(!protected_inputs_match(
                &doc, &drift, range, view.start, &gold, 32
            ));
        }
        let mut drift = view.doc.clone();
        drift.enc.labels[0] = 3;
        assert!(!protected_inputs_match(
            &doc, &drift, range, view.start, &gold, 32
        ));
        let mut edge_only = view.doc.clone();
        edge_only.enc.flags[0] ^= flag::LINE_START;
        assert!(protected_inputs_match(
            &doc, &edge_only, range, view.start, &gold, 32
        ));
    }

    #[test]
    fn native_gold_round_trip_preserves_unicode_spans_and_rule_masks() {
        let doc = parent(240, &[(80, "Émilie", 0), (100, "Dépôt Harbor", 1)]);
        let mut doc = native_parent(
            doc.text.replace("word47", "contact@example.invalid"),
            &[("Émilie", 0), ("Dépôt Harbor", 1)],
        );
        doc.enc.country = "US".into();
        let views = prepare(std::slice::from_ref(&doc), &FeatureConfig::default(), 32).unwrap();
        assert!(!views.is_empty());
        for view in views {
            let mut probabilities = vec![0.0; view.doc.enc.labels.len() * DETECTOR_LABELS];
            for (index, &label) in view.doc.enc.labels.iter().enumerate() {
                probabilities[index * DETECTOR_LABELS + usize::from(label)] = 1.0;
            }
            let mask = view
                .doc
                .enc
                .flags
                .iter()
                .map(|bits| bits & flag::IN_RULE_SPAN != 0)
                .collect::<Vec<_>>();
            let decoded = decode_detector(&probabilities, &mask, &view.doc.breaks);
            let mut spans = decoded
                .iter()
                .map(|span| KindSpan {
                    kind: crate::detector::KINDS
                        .iter()
                        .position(|&kind| kind == span.kind)
                        .unwrap(),
                    start: view.doc.enc.token_spans[span.first].0,
                    end: view.doc.enc.token_spans[span.last].1,
                })
                .collect::<Vec<_>>();
            spans.sort_by_key(|span| span.start);
            let mut expected = view.doc.gold.clone();
            expected.sort_by_key(|span| span.start);
            assert_eq!(spans, expected);
        }
    }

    #[test]
    fn native_round_trip_rejects_missing_gold_and_extra_positive_labels() {
        let doc = parent(200, &[(80, "Émilie", 0)]);
        verify_gold_round_trip(&doc).unwrap();
        let mut missing = doc.clone();
        missing.gold.clear();
        assert!(verify_gold_round_trip(&missing).is_err());
        let mut extra = doc;
        extra.enc.labels[90] = 3;
        assert!(verify_gold_round_trip(&extra).is_err());
    }

    #[test]
    fn context_clipping_preserves_true_parent_edge_whitespace() {
        for position in [0, 199] {
            let original = parent(200, &[(position, "Émilie", 0)]);
            let doc = native_parent(
                format!("\n \t\n\u{a0}{}\n\t", original.text),
                &[("Émilie", 0)],
            );
            let gold = aligned_gold(&doc, 0).unwrap();
            let views = prepare(std::slice::from_ref(&doc), &FeatureConfig::default(), 32).unwrap();
            assert_eq!(views.len(), 1);
            let view = &views[0];
            let range = if position == 0 {
                assert_eq!(view.start, 0);
                assert!(view.doc.text.starts_with("\n \t\n\u{a0}"));
                (0, 34)
            } else {
                assert_eq!(view.end, doc.text.len());
                assert!(view.doc.text.ends_with("\n\t"));
                (166, 200)
            };
            assert!(protected_inputs_match(
                &doc, &view.doc, range, view.start, &gold, 32
            ));
            verify_gold_round_trip(&view.doc).unwrap();
        }
    }
}
