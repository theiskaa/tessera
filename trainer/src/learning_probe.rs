//! Select a fixed training probe with configured-file coverage, not source-document diversity.

use std::collections::HashSet;
use std::ops::Deref;

use anyhow::{Context, ensure};

use crate::detector::DetectorDoc;

/// Whole training documents with the real/synthetic boundary retained for separate checks.
#[derive(Debug)]
pub(crate) struct SelectedProbe {
    /// Real documents first, followed by synthetic documents, without relabeling.
    pub(crate) documents: Vec<DetectorDoc>,
    /// Number of real documents at the beginning of `documents`.
    pub(crate) real_count: usize,
}

impl Deref for SelectedProbe {
    type Target = [DetectorDoc];

    fn deref(&self) -> &Self::Target {
        &self.documents
    }
}

/// Record probe support and exact text identities without copying source text to the log.
pub(crate) fn manifest(probe: &SelectedProbe) -> serde_json::Value {
    let mut gold = [0usize; 3];
    let mut group_gold = [[0usize; 3]; 2];
    for (index, doc) in probe.iter().enumerate() {
        let group = usize::from(index >= probe.real_count);
        for span in &doc.gold {
            gold[span.kind] += 1;
            group_gold[group][span.kind] += 1;
        }
    }
    serde_json::json!({
        "selection": "whole first-epoch training documents; shortest and longest per silver file, then sampled synthetic; not independent evaluation",
        "kind_order": ["person", "org", "address"],
        "gold_spans": gold,
        "group_counts": {"real": probe.real_count, "synthetic": probe.len() - probe.real_count},
        "group_gold_spans": {"real": group_gold[0], "synthetic": group_gold[1]},
        "documents": probe.iter().enumerate().map(|(index, doc)| serde_json::json!({
            "group": if index < probe.real_count { "real" } else { "synthetic" },
            "text_sha256": crate::export::sha256_hex(doc.text.as_bytes()),
            "content_tokens": doc.enc.token_spans.len(),
            "gold": doc.gold.iter().map(|span| serde_json::json!({
                "kind": crate::detector::KINDS[span.kind].as_str(),
                "start": span.start,
                "end": span.end,
            })).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    })
}

/// Require original gold support for every learned kind in both probe groups.
pub(crate) fn validate_groups(probe: &SelectedProbe) -> anyhow::Result<()> {
    ensure!(
        probe.real_count <= probe.len(),
        "training probe real boundary is invalid"
    );
    for (group, docs) in [
        ("real", &probe[..probe.real_count]),
        ("synthetic", &probe[probe.real_count..]),
    ] {
        let mut support = [0usize; 3];
        for doc in docs {
            for span in &doc.gold {
                ensure!(
                    span.kind < support.len(),
                    "training probe gold kind is invalid"
                );
                support[span.kind] += 1;
            }
        }
        for (kind, count) in ["person", "org", "address"].into_iter().zip(support) {
            ensure!(count > 0, "{group} training probe has no {kind} gold spans");
        }
    }
    Ok(())
}

/// Clone whole training documents, preserving their encodings, gold and paragraph breaks.
///
/// Each silver file contributes its shortest and longest available unique texts, or one
/// when it has only one. Negative documents remain eligible. Synthetic examples come only
/// from the first epoch's draws. File coverage does not imply distinct source documents.
pub(crate) fn select(
    synthetic: &[DetectorDoc],
    silver: &[DetectorDoc],
    source_pieces: &[usize],
    first_epoch_draws: &[usize],
    max_documents: usize,
) -> anyhow::Result<SelectedProbe> {
    ensure!(
        max_documents > 0,
        "training probe document cap must be positive"
    );
    let silver_count = source_pieces
        .iter()
        .try_fold(0usize, |total, count| total.checked_add(*count))
        .context("training probe silver ranges overflow")?;
    ensure!(
        silver_count == silver.len(),
        "training probe silver ranges do not match encoded documents"
    );
    let total = synthetic
        .len()
        .checked_add(silver.len())
        .context("training probe document count overflow")?;
    ensure!(
        first_epoch_draws.iter().all(|&index| index < total),
        "training probe draw index is outside encoded documents"
    );
    let exposed: HashSet<usize> = first_epoch_draws.iter().copied().collect();
    let mut seen = HashSet::new();
    let mut selected = Vec::new();
    let mut start = 0;
    for (file, &count) in source_pieces.iter().enumerate() {
        ensure!(
            count > 0,
            "training probe silver file {file} has no encoded pieces"
        );
        let end = start + count;
        let mut candidates: Vec<usize> = (start..end)
            .filter(|&index| !seen.contains(silver[index].text.as_str()))
            .collect();
        candidates.sort_by_key(|&index| (silver[index].enc.token_spans.len(), index));
        let shortest = candidates.first().copied().with_context(|| {
            format!("training probe cannot cover silver file {file} with unique text")
        })?;
        let longest = candidates
            .iter()
            .rev()
            .copied()
            .find(|&index| silver[index].text != silver[shortest].text);
        for index in std::iter::once(shortest).chain(longest) {
            ensure!(
                exposed.contains(&(synthetic.len() + index)),
                "training probe silver file {file} includes a document absent from first-epoch draws"
            );
            seen.insert(silver[index].text.as_str());
            selected.push(silver[index].clone());
        }
        start = end;
    }
    let mut synthetic_indices = Vec::new();
    for &index in first_epoch_draws {
        if index < synthetic.len() && seen.insert(synthetic[index].text.as_str()) {
            synthetic_indices.push(index);
        }
    }
    let reserved_synthetic = (max_documents / 2).min(synthetic_indices.len());
    ensure!(
        selected.len() <= max_documents - reserved_synthetic,
        "training probe cap cannot cover every silver file and reserve {reserved_synthetic} synthetic documents"
    );
    let real_count = selected.len();
    let remaining = max_documents - real_count;
    selected.extend(
        synthetic_indices
            .into_iter()
            .take(remaining)
            .map(|index| synthetic[index].clone()),
    );
    let mut support = [0usize; 3];
    for doc in &selected {
        for span in &doc.gold {
            ensure!(
                span.kind < support.len(),
                "training probe has an unknown gold kind"
            );
            support[span.kind] += 1;
        }
    }
    for (kind, count) in ["person", "org", "address"].into_iter().zip(support) {
        ensure!(count > 0, "training probe has no {kind} gold spans");
    }
    Ok(SelectedProbe {
        documents: selected,
        real_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Encoded;
    use crate::detector::KindSpan;

    #[test]
    fn aggregate_support_cannot_hide_an_unrepresented_source_group() {
        let probe = SelectedProbe {
            documents: vec![doc("real", 3, &[0]), doc("synthetic", 3, &[0, 1, 2])],
            real_count: 1,
        };
        assert!(
            validate_groups(&probe)
                .unwrap_err()
                .to_string()
                .contains("real training probe has no org")
        );
        let complete = SelectedProbe {
            documents: vec![doc("real", 3, &[0, 1, 2]), doc("synthetic", 3, &[0, 1, 2])],
            real_count: 1,
        };
        validate_groups(&complete).unwrap();
    }

    fn doc(name: &str, length: usize, kinds: &[usize]) -> DetectorDoc {
        let words: Vec<_> = (0..length).map(|index| format!("{name}{index}")).collect();
        let text = words.join(" ");
        let mut start = 0u32;
        let token_spans: Vec<_> = words
            .iter()
            .map(|word| {
                let end = start + word.len() as u32;
                let span = (start, end);
                start = end + 1;
                span
            })
            .collect();
        let gold: Vec<_> = kinds
            .iter()
            .enumerate()
            .map(|(index, &kind)| KindSpan {
                kind,
                start: token_spans[index].0,
                end: token_spans[index].1,
            })
            .collect();
        let mut labels = vec![0; length];
        for (index, &kind) in kinds.iter().enumerate() {
            labels[index] = (kind * 2 + 1) as u8;
        }
        DetectorDoc {
            text,
            enc: Encoded {
                token_spans,
                ngram_ids: vec![vec![1]; length],
                script: vec![1; length],
                shape: vec![1; length],
                flags: vec![0; length],
                labels,
                country: "US".into(),
            },
            gold,
            breaks: (0..length).map(|index| index == 0).collect(),
        }
    }

    #[test]
    fn whole_documents_preserve_negatives_and_file_extremes() {
        let silver = vec![
            doc("middle", 5, &[]),
            doc("short", 3, &[0, 1, 2]),
            doc("long", 9, &[]),
            doc("single", 4, &[0]),
        ];
        let probe = select(&[], &silver, &[3, 1], &[0, 1, 2, 3], 3).unwrap();
        for (selected, index) in probe.iter().zip([1, 2, 3]) {
            assert_eq!(selected.text, silver[index].text);
            assert_eq!(selected.enc, silver[index].enc);
            assert_eq!(selected.gold, silver[index].gold);
            assert_eq!(selected.breaks, silver[index].breaks);
        }
        assert!(probe[1].gold.is_empty());
    }

    #[test]
    fn synthetic_fill_uses_distinct_first_epoch_texts_in_draw_order() {
        let synthetic = vec![
            doc("notdrawn", 3, &[0, 1, 2]),
            doc("drawn", 3, &[0, 1, 2]),
            doc("other", 3, &[0]),
            doc("drawn", 3, &[0, 1, 2]),
        ];
        let probe = select(&synthetic, &[], &[], &[2, 1, 1, 3], 4).unwrap();
        assert_eq!(probe.len(), 2);
        assert_eq!(probe[0].text, synthetic[2].text);
        assert_eq!(probe[1].text, synthetic[1].text);
    }

    #[test]
    fn cap_must_reserve_synthetic_without_dropping_real_files() {
        let synthetic: Vec<_> = (0..5)
            .map(|i| doc(&format!("s{i}"), 3, &[0, 1, 2]))
            .collect();
        let silver = vec![doc("r1", 3, &[0]), doc("r2", 3, &[1]), doc("r3", 3, &[2])];
        let draws: Vec<_> = (0..8).collect();
        assert!(select(&synthetic, &silver, &[1, 1, 1], &draws, 4).is_err());
        assert_eq!(
            select(&synthetic, &silver, &[1, 1, 1], &draws, 5)
                .unwrap()
                .len(),
            5
        );
        let probe = select(&synthetic, &silver, &[1, 1, 1], &draws, 6).unwrap();
        assert_eq!(probe.len(), 6);
        assert_eq!(probe[2].text, silver[2].text);
        assert_eq!(probe[3].text, synthetic[0].text);
    }

    #[test]
    fn duplicate_real_text_is_kept_once_and_cannot_hide_a_missing_file() {
        let repeated = doc("real", 3, &[0, 1, 2]);
        let silver = vec![repeated.clone(), repeated];
        assert_eq!(select(&[], &silver, &[2], &[0, 1], 2).unwrap().len(), 1);
        assert!(select(&[], &silver, &[1, 1], &[0, 1], 2).is_err());
    }

    #[test]
    fn ranges_indices_and_actual_exposure_are_checked() {
        let silver = vec![doc("real", 3, &[0, 1, 2])];
        assert!(select(&[], &silver, &[], &[0], 2).is_err());
        assert!(select(&[], &silver, &[0, 1], &[0], 2).is_err());
        assert!(select(&[], &silver, &[1], &[1], 2).is_err());
        assert!(select(&[], &silver, &[1], &[], 2).is_err());
        assert!(select(&[], &silver, &[1], &[0], 0).is_err());
    }

    #[test]
    fn all_three_kinds_need_original_gold_support() {
        let synthetic = vec![doc("two", 3, &[0, 1])];
        assert!(select(&synthetic, &[], &[], &[0], 1).is_err());
        assert!(select(&[], &[], &[], &[], 1).is_err());
    }

    #[test]
    fn group_boundary_and_manifest_keep_gold_support_separate() {
        let silver = vec![doc("real", 3, &[0, 1, 2]), doc("negative", 4, &[])];
        let synthetic = vec![doc("synthetic", 3, &[0, 0]), doc("other", 3, &[1, 2])];
        let probe = select(&synthetic, &silver, &[2], &[0, 1, 2, 3], 4).unwrap();
        assert_eq!(probe.real_count, 2);
        assert_eq!(probe[probe.real_count - 1].enc, silver[1].enc);
        assert_eq!(probe[probe.real_count].enc, synthetic[0].enc);
        assert_eq!(probe[probe.real_count].gold, synthetic[0].gold);
        let manifest = manifest(&probe);
        assert_eq!(manifest["group_counts"]["real"], 2);
        assert_eq!(manifest["group_counts"]["synthetic"], 2);
        assert_eq!(
            manifest["group_gold_spans"]["real"],
            serde_json::json!([1, 1, 1])
        );
        assert_eq!(
            manifest["group_gold_spans"]["synthetic"],
            serde_json::json!([2, 1, 1])
        );
        assert_eq!(manifest["documents"][1]["group"], "real");
        assert_eq!(manifest["documents"][2]["group"], "synthetic");
        assert_eq!(manifest["gold_spans"], serde_json::json!([3, 2, 2]));
    }
}
