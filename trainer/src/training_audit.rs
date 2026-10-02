//! Count actual detector draws and their class-weighted contributions to batch loss.
//! These are loss coefficients, not measured losses or gradient magnitudes.

use anyhow::{Context, ensure};
use serde::Serialize;

use crate::dataset::Encoded;
use crate::detector::DETECTOR_LABELS;

const LABEL_ORDER: [&str; DETECTOR_LABELS] = [
    "O",
    "B-PERSON",
    "I-PERSON",
    "B-ORG",
    "I-ORG",
    "B-ADDRESS",
    "I-ADDRESS",
];

/// Exposure of one source group, with shares averaged over every batch in the epoch.
#[derive(Debug, Default, Serialize)]
pub(crate) struct Exposure {
    /// Encoded examples drawn, including repeated indices.
    pub draws: u64,
    /// Retained content tokens; padding does not contribute to the loss.
    pub content_tokens: u64,
    /// Token counts in the enclosing audit's BIO label order.
    pub label_token_counts: [u64; DETECTOR_LABELS],
    /// Class weight times token count, summed across all draws.
    pub weighted_token_mass: f64,
    /// Weighted token mass for each BIO label.
    pub label_weighted_token_mass: [f64; DETECTOR_LABELS],
    /// Mean source-group share of the actual batch loss denominator.
    pub mean_batch_weight_share: f64,
    /// Mean BIO-label shares of the actual batch loss denominator.
    pub mean_batch_label_weight_share: [f64; DETECTOR_LABELS],
    /// Batches containing a B or I token of each learned kind, in kind_order.
    pub positive_batches: [u64; 3],
}

/// Exact draw counts and normalized loss coefficients for a supplied epoch's batches.
#[derive(Debug, Serialize)]
pub(crate) struct EpochExposure {
    /// Order of every seven-element label array.
    pub label_order: [&'static str; DETECTOR_LABELS],
    /// Order of the positive-batch counts.
    pub kind_order: [&'static str; 3],
    /// Number of actual optimizer batches, including the final partial batch.
    pub batches: usize,
    /// Exposure from both source groups combined.
    pub total: Exposure,
    /// Exposure from indices below synthetic_count.
    pub synthetic: Exposure,
    /// Exposure from the remaining indices.
    pub real: Exposure,
}

fn add_count(target: &mut u64, value: u64) -> anyhow::Result<()> {
    *target = target
        .checked_add(value)
        .context("training exposure count overflow")?;
    Ok(())
}

fn token_counts(item: &Encoded, index: usize) -> anyhow::Result<[u64; DETECTOR_LABELS]> {
    let length = item.token_spans.len();
    ensure!(
        [
            item.labels.len(),
            item.ngram_ids.len(),
            item.script.len(),
            item.shape.len(),
            item.flags.len()
        ]
        .iter()
        .all(|&count| count == length),
        "encoded item {index} has inconsistent token dimensions"
    );
    let mut counts = [0u64; DETECTOR_LABELS];
    for &label in &item.labels {
        let count = counts
            .get_mut(usize::from(label))
            .with_context(|| format!("encoded item {index} has invalid BIO label {label}"))?;
        add_count(count, 1)?;
    }
    Ok(counts)
}

impl Exposure {
    fn record_batch(
        &mut self,
        draws: u64,
        counts: &[u64; DETECTOR_LABELS],
        weights: &[f32],
        denominator: f64,
    ) -> anyhow::Result<()> {
        add_count(&mut self.draws, draws)?;
        for (label, &count) in counts.iter().enumerate() {
            add_count(&mut self.label_token_counts[label], count)?;
            add_count(&mut self.content_tokens, count)?;
            self.mean_batch_label_weight_share[label] +=
                count as f64 * f64::from(weights[label]) / denominator;
        }
        for kind in 0..3 {
            if counts[1 + 2 * kind] > 0 || counts[2 + 2 * kind] > 0 {
                add_count(&mut self.positive_batches[kind], 1)?;
            }
        }
        Ok(())
    }

    fn finish(&mut self, batches: usize, weights: &[f32]) {
        for (label, &weight) in weights.iter().enumerate() {
            self.label_weighted_token_mass[label] =
                self.label_token_counts[label] as f64 * f64::from(weight);
            self.mean_batch_label_weight_share[label] /= batches as f64;
        }
        self.weighted_token_mass = self.label_weighted_token_mass.iter().sum();
        self.mean_batch_weight_share = self.mean_batch_label_weight_share.iter().sum();
    }
}

/// Audit actual epoch batches without initializing a model or changing any inputs.
///
/// Each share uses the combined real/synthetic batch denominator and averages over all
/// batches, including those without that source or label. The denominator is clamped to
/// 1.0 exactly as in masked_loss; all-empty encoded batches therefore contribute zero.
pub(crate) fn audit_epoch(
    items: &[Encoded],
    batches: &[Vec<usize>],
    synthetic_count: usize,
    weights: &[f32],
) -> anyhow::Result<EpochExposure> {
    ensure!(
        synthetic_count <= items.len(),
        "synthetic count exceeds encoded items"
    );
    ensure!(
        !batches.is_empty(),
        "training audit needs at least one batch"
    );
    ensure!(
        weights.len() == DETECTOR_LABELS
            && weights
                .iter()
                .all(|weight| weight.is_finite() && *weight > 0.0),
        "training audit needs seven finite positive class weights"
    );
    let counts = items
        .iter()
        .enumerate()
        .map(|(index, item)| token_counts(item, index))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut result = EpochExposure {
        label_order: LABEL_ORDER,
        kind_order: ["person", "org", "address"],
        batches: batches.len(),
        total: Exposure::default(),
        synthetic: Exposure::default(),
        real: Exposure::default(),
    };
    for (batch_index, batch) in batches.iter().enumerate() {
        ensure!(
            !batch.is_empty(),
            "training audit batch {batch_index} is empty"
        );
        let mut grouped = [[0u64; DETECTOR_LABELS]; 2];
        let mut draws = [0u64; 2];
        for &index in batch {
            let item = counts.get(index).with_context(|| {
                format!("training audit batch {batch_index} has invalid index {index}")
            })?;
            let group = usize::from(index >= synthetic_count);
            add_count(&mut draws[group], 1)?;
            for (target, &count) in grouped[group].iter_mut().zip(item) {
                add_count(target, count)?;
            }
        }
        let mut combined = grouped[0];
        for (target, &count) in combined.iter_mut().zip(&grouped[1]) {
            add_count(target, count)?;
        }
        let mass: f64 = combined
            .iter()
            .zip(weights)
            .map(|(&count, &weight)| count as f64 * f64::from(weight))
            .sum();
        ensure!(mass.is_finite(), "training audit batch weight overflow");
        let denominator = mass.max(1.0);
        let mut total_draws = draws[0];
        add_count(&mut total_draws, draws[1])?;
        result
            .total
            .record_batch(total_draws, &combined, weights, denominator)?;
        result
            .synthetic
            .record_batch(draws[0], &grouped[0], weights, denominator)?;
        result
            .real
            .record_batch(draws[1], &grouped[1], weights, denominator)?;
    }
    result.total.finish(batches.len(), weights);
    result.synthetic.finish(batches.len(), weights);
    result.real.finish(batches.len(), weights);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encoded(labels: &[u8]) -> Encoded {
        let length = labels.len();
        Encoded {
            token_spans: (0..length)
                .map(|index| (index as u32, index as u32 + 1))
                .collect(),
            ngram_ids: vec![vec![1]; length],
            script: vec![0; length],
            shape: vec![0; length],
            flags: vec![0; length],
            labels: labels.to_vec(),
            country: "US".into(),
        }
    }

    fn close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
    }

    #[test]
    fn averages_actual_batches_instead_of_global_token_mass() {
        let items = [encoded(&[0; 100]), encoded(&[1])];
        let result = audit_epoch(
            &items,
            &[vec![0], vec![1]],
            1,
            &[1.0, 3.0, 2.0, 3.0, 2.0, 2.0, 1.5],
        )
        .unwrap();
        assert_eq!(result.total.content_tokens, 101);
        assert_eq!(result.total.label_token_counts, [100, 1, 0, 0, 0, 0, 0]);
        close(result.total.weighted_token_mass, 103.0);
        close(result.real.weighted_token_mass, 3.0);
        close(result.real.mean_batch_weight_share, 0.5);
        close(result.synthetic.mean_batch_weight_share, 0.5);
        close(result.total.mean_batch_label_weight_share[1], 0.5);
        assert_eq!(result.real.positive_batches, [1, 0, 0]);
        let value = serde_json::to_value(result).unwrap();
        assert_eq!(value["label_order"][1], "B-PERSON");
        assert_eq!(value["kind_order"][2], "address");
    }

    #[test]
    fn counts_repeated_draws_and_source_contributions_in_mixed_batches() {
        let items = [encoded(&[0]), encoded(&[5, 6]), encoded(&[1, 2, 3])];
        let result = audit_epoch(
            &items,
            &[vec![0, 0, 2], vec![1]],
            2,
            &[1.0, 3.0, 2.0, 3.0, 2.0, 2.0, 1.5],
        )
        .unwrap();
        assert_eq!(result.total.draws, 4);
        assert_eq!(result.synthetic.draws, 3);
        assert_eq!(result.real.draws, 1);
        assert_eq!(result.total.positive_batches, [1, 1, 1]);
        assert_eq!(result.real.positive_batches, [1, 1, 0]);
        assert_eq!(result.synthetic.positive_batches, [0, 0, 1]);
        close(result.real.mean_batch_weight_share, 0.4);
        close(result.synthetic.mean_batch_weight_share, 0.6);
        close(result.total.mean_batch_label_weight_share[0], 0.1);
        close(result.total.mean_batch_label_weight_share[1], 0.15);
        close(result.total.mean_batch_label_weight_share.iter().sum(), 1.0);
        close(result.total.weighted_token_mass, 13.5);
    }

    #[test]
    fn matches_loss_denominator_clamp_and_empty_encoded_examples() {
        let result = audit_epoch(
            &[encoded(&[1]), encoded(&[])],
            &[vec![0], vec![1]],
            0,
            &[0.25; DETECTOR_LABELS],
        )
        .unwrap();
        close(result.real.mean_batch_label_weight_share[1], 0.125);
        assert_eq!(result.total.draws, 2);
        assert_eq!(result.total.content_tokens, 1);
    }

    #[test]
    fn rejects_invalid_dimensions_indices_labels_and_weights() {
        let item = encoded(&[1]);
        let weights = [1.0; DETECTOR_LABELS];
        assert!(audit_epoch(std::slice::from_ref(&item), &[], 1, &weights).is_err());
        assert!(audit_epoch(std::slice::from_ref(&item), &[vec![]], 1, &weights).is_err());
        assert!(audit_epoch(std::slice::from_ref(&item), &[vec![1]], 1, &weights).is_err());
        assert!(audit_epoch(std::slice::from_ref(&item), &[vec![0]], 2, &weights).is_err());
        assert!(audit_epoch(std::slice::from_ref(&item), &[vec![0]], 1, &weights[..6]).is_err());
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            let mut invalid = weights;
            invalid[0] = bad;
            assert!(audit_epoch(std::slice::from_ref(&item), &[vec![0]], 1, &invalid).is_err());
        }
        let mut invalid = item.clone();
        invalid.labels[0] = 7;
        assert!(audit_epoch(&[invalid], &[vec![0]], 1, &weights).is_err());
        let mut invalid = item;
        invalid.flags.clear();
        assert!(audit_epoch(&[invalid], &[vec![0]], 1, &weights).is_err());
    }
}
