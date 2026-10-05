//! Explicit loss-only rule and reviewed-ambiguity exclusion; forward/context masks remain the
//! batcher's padding mask.

use crate::dataset::Encoded;
use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use tessera::internal::flag;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize, Serialize)]
/// Versioned supervision policy; absent historical fields retain the original objective.
pub(super) enum Objective {
    #[default]
    #[serde(rename = "native_padding_weighted_ce_v1")]
    LegacyPaddingV1,
    #[serde(rename = "canonical_rule_excluded_weighted_ce_v1")]
    CanonicalRuleExcludedV1,
    #[serde(rename = "canonical_rule_and_reviewed_ambiguity_excluded_weighted_ce_v1")]
    CanonicalRuleAndReviewedAmbiguityExcludedV1,
}

impl Objective {
    /// Omit the default field so old manifest and identity representations remain unchanged.
    pub(super) fn is_legacy(&self) -> bool {
        *self == Self::LegacyPaddingV1
    }

    /// Only this objective consumes, and must consume, a reviewed ambiguity sidecar.
    pub(super) fn excludes_reviewed_ambiguity(&self) -> bool {
        *self == Self::CanonicalRuleAndReviewedAmbiguityExcludedV1
    }
}

#[derive(Debug, Clone, Serialize)]
/// Pooled token counts and weighted denominator inputs, not measured gradient shares.
pub(super) struct Report {
    pub(super) retained_tokens: usize,
    pub(super) excluded_rule_o_tokens: usize,
    /// Present only for the reviewed-ambiguity objective, so older reports stay byte-identical.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) excluded_reviewed_ambiguity_o_tokens: Option<usize>,
    pub(super) active_label_counts: [usize; 7],
    pub(super) active_weighted_mass: f64,
}

#[derive(Debug)]
/// Independent loss mask in the batcher's exact row-major padded token geometry.
pub(super) struct LossMask {
    pub(super) values: Vec<f32>,
    pub(super) dimensions: [usize; 2],
    pub(super) report: Report,
}

fn active_from_flags(flags: u32) -> bool {
    flags & flag::IN_RULE_SPAN == 0
}

/// Build a whole-batch loss mask, optionally also excluding reviewed-ambiguity tokens.
pub(super) fn batch_loss_mask(
    items: &[Encoded],
    ambiguity: Option<&[Vec<bool>]>,
    weights: &[f32],
) -> anyhow::Result<LossMask> {
    let width = items
        .iter()
        .map(|e| e.token_spans.len())
        .max()
        .context("empty loss batch")?;
    loss_mask(items, ambiguity, [items.len(), width], weights)
}

/// Canonical flags and, when given, per-row reviewed sidecar exclusions select activity.
/// Labels only check conflicts and describe the denominator; they never select activity, and
/// excluded reviewed-ambiguity tokens must be canonical O and outside every rule span.
fn loss_mask(
    items: &[Encoded],
    ambiguity: Option<&[Vec<bool>]>,
    dimensions: [usize; 2],
    weights: &[f32],
) -> anyhow::Result<LossMask> {
    ensure!(
        weights.len() == 7 && weights.iter().all(|w| w.is_finite() && *w > 0.0),
        "loss requires seven finite positive unchanged class weights"
    );
    let width = items
        .iter()
        .map(|e| e.token_spans.len())
        .max()
        .context("empty loss batch")?;
    ensure!(
        dimensions == [items.len(), width] && width > 0,
        "loss geometry differs from whole canonical batch"
    );
    ensure!(
        ambiguity.is_none_or(|rows| rows.len() == items.len()),
        "reviewed ambiguity rows differ from batch rows"
    );
    let size = dimensions[0]
        .checked_mul(dimensions[1])
        .context("loss geometry overflow")?;
    let mut values = vec![0.0; size];
    let mut report = Report {
        retained_tokens: 0,
        excluded_rule_o_tokens: 0,
        excluded_reviewed_ambiguity_o_tokens: ambiguity.map(|_| 0),
        active_label_counts: [0; 7],
        active_weighted_mass: 0.0,
    };
    for (row, item) in items.iter().enumerate() {
        let n = item.token_spans.len();
        ensure!(
            n > 0
                && item.flags.len() == n
                && item.labels.len() == n
                && item.ngram_ids.len() == n
                && item.script.len() == n
                && item.shape.len() == n,
            "canonical loss arrays differ in length"
        );
        let reviewed = ambiguity.map(|rows| rows[row].as_slice());
        ensure!(
            reviewed.is_none_or(|r| r.len() == n),
            "reviewed ambiguity row differs from canonical tokens"
        );
        for (token, (&flags, &label)) in item.flags.iter().zip(&item.labels).enumerate() {
            ensure!(
                label < 7 && flags & flag::MASKED == 0,
                "invalid neural label or structured mask in canonical whole Text loss"
            );
            report.retained_tokens += 1;
            if reviewed.is_some_and(|r| r[token]) {
                ensure!(
                    label == 0 && active_from_flags(flags),
                    "reviewed ambiguity token is supervised or rule-controlled"
                );
                if let Some(count) = &mut report.excluded_reviewed_ambiguity_o_tokens {
                    *count += 1;
                }
            } else if active_from_flags(flags) {
                values[row * width + token] = 1.0;
                report.active_label_counts[usize::from(label)] += 1;
                report.active_weighted_mass += f64::from(weights[usize::from(label)]);
            } else {
                ensure!(
                    label == 0,
                    "rule-controlled token has positive neural supervision"
                );
                report.excluded_rule_o_tokens += 1;
            }
        }
    }
    ensure!(
        report.active_label_counts.iter().sum::<usize>() > 0
            && report.active_weighted_mass.is_finite()
            && report.active_weighted_mass > 0.0,
        "empty or nonfinite active neural loss denominator"
    );
    Ok(LossMask {
        values,
        dimensions,
        report,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rule_excluded_mask(
        items: &[Encoded],
        dimensions: [usize; 2],
        weights: &[f32],
    ) -> anyhow::Result<LossMask> {
        loss_mask(items, None, dimensions, weights)
    }
    fn item(flags: Vec<u32>, labels: Vec<u8>) -> Encoded {
        let n = flags.len();
        Encoded {
            token_spans: (0..n).map(|i| (i as u32, i as u32 + 1)).collect(),
            ngram_ids: vec![vec![1]; n],
            script: vec![0; n],
            shape: vec![0; n],
            flags,
            labels,
            country: "US".into(),
        }
    }
    #[test]
    fn flags_alone_choose_activity_and_padding_is_excluded() {
        let items = vec![
            item(vec![flag::IN_RULE_SPAN, 0], vec![0, 1]),
            item(vec![0], vec![0]),
        ];
        let mask = rule_excluded_mask(&items, [2, 2], &[1.0; 7]).unwrap();
        assert_eq!(mask.values, vec![0.0, 1.0, 1.0, 0.0]);
        assert_eq!(mask.report.active_label_counts, [1, 1, 0, 0, 0, 0, 0]);
        let mut changed = items.clone();
        changed[0].labels[1] = 6;
        assert_eq!(
            rule_excluded_mask(&changed, [2, 2], &[1.0; 7])
                .unwrap()
                .values,
            mask.values
        );
        assert_eq!(items[0].flags, vec![flag::IN_RULE_SPAN, 0]);
    }
    #[test]
    fn bad_shapes_conflicts_weights_and_empty_active_fail() {
        let source = vec![item(vec![flag::IN_RULE_SPAN, 0], vec![0, 1])];
        assert!(rule_excluded_mask(&source, [1, 3], &[1.0; 7]).is_err());
        let mut wrong = source.clone();
        wrong[0].labels.pop();
        assert!(rule_excluded_mask(&wrong, [1, 2], &[1.0; 7]).is_err());
        wrong = source.clone();
        wrong[0].labels[0] = 1;
        assert!(rule_excluded_mask(&wrong, [1, 2], &[1.0; 7]).is_err());
        assert!(
            rule_excluded_mask(
                &[item(vec![flag::IN_RULE_SPAN], vec![0])],
                [1, 1],
                &[1.0; 7]
            )
            .is_err()
        );
        let mut weights = [1.0; 7];
        weights[2] = f32::NAN;
        assert!(rule_excluded_mask(&source, [1, 2], &weights).is_err());
        weights[2] = 0.0;
        assert!(rule_excluded_mask(&source, [1, 2], &weights).is_err());
    }
    #[test]
    fn actual_ad_ce_has_zero_rule_and_padding_logit_gradient_but_keeps_neural_gradient() {
        use burn::backend::{Autodiff, NdArray};
        use burn::prelude::*;
        type AD = Autodiff<NdArray>;
        let device = Default::default();
        let items = vec![
            item(vec![flag::IN_RULE_SPAN, 0], vec![0, 1]),
            item(vec![0], vec![0]),
        ];
        let loss_mask = rule_excluded_mask(&items, [2, 2], &[1.0; 7]).unwrap();
        let logits = Tensor::<AD, 3>::zeros([2, 2, 7], &device).require_grad();
        let labels =
            Tensor::<AD, 2, Int>::from_data(TensorData::new(vec![0i64, 1, 0, 0], [2, 2]), &device);
        let forward_mask = Tensor::<AD, 2>::from_data(
            TensorData::new(vec![1.0f32, 1.0, 1.0, 0.0], [2, 2]),
            &device,
        );
        let loss_mask = Tensor::<AD, 2>::from_data(
            TensorData::new(loss_mask.values, loss_mask.dimensions),
            &device,
        );
        let weights = Tensor::<AD, 1>::ones([7], &device);
        let loss =
            crate::train::masked_loss(logits.clone(), labels.clone(), loss_mask, weights.clone());
        let scalar = loss.clone().into_scalar();
        assert!((scalar - 7.0f32.ln()).abs() < 1e-6);
        let gradient = logits
            .grad(&loss.backward())
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(gradient.iter().all(|v| v.is_finite()));
        assert!(
            gradient[0..7]
                .iter()
                .chain(&gradient[21..28])
                .all(|v| *v == 0.0)
        );
        assert!(gradient[8] < 0.0 && gradient[14] < 0.0);
        assert!(
            gradient[7..14]
                .iter()
                .chain(&gradient[14..21])
                .any(|v| *v > 0.0)
        );
        assert_eq!(
            forward_mask.clone().into_data().to_vec::<f32>().unwrap(),
            vec![1.0, 1.0, 1.0, 0.0]
        );
        let legacy_logits = Tensor::<AD, 3>::zeros([2, 2, 7], &device).require_grad();
        let legacy =
            crate::train::masked_loss(legacy_logits.clone(), labels, forward_mask, weights);
        let legacy_grad = legacy_logits
            .grad(&legacy.backward())
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        assert!(legacy_grad[0] < 0.0);
    }
    #[test]
    fn reviewed_ambiguity_is_loss_only_and_reported_only_for_its_objective() {
        let items = vec![
            item(vec![0, 0, flag::IN_RULE_SPAN], vec![0, 1, 0]),
            item(vec![0], vec![0]),
        ];
        let rows = vec![vec![true, false, false], vec![false]];
        let mask = batch_loss_mask(&items, Some(&rows), &[1.0; 7]).unwrap();
        assert_eq!(mask.values, vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(mask.report.excluded_reviewed_ambiguity_o_tokens, Some(1));
        assert_eq!(mask.report.excluded_rule_o_tokens, 1);
        assert_eq!(mask.report.active_label_counts, [1, 1, 0, 0, 0, 0, 0]);
        assert_eq!(mask.report.active_weighted_mass, 2.0);
        let rule_only = batch_loss_mask(&items, None, &[1.0; 7]).unwrap();
        assert_eq!(rule_only.values, vec![1.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
        let legacy = serde_json::to_value(&rule_only.report).unwrap();
        assert!(legacy.get("excluded_reviewed_ambiguity_o_tokens").is_none());
        assert_eq!(
            legacy,
            serde_json::json!({"retained_tokens":4,"excluded_rule_o_tokens":1,
                "active_label_counts":[2,1,0,0,0,0,0],"active_weighted_mass":3.0})
        );
        let none =
            batch_loss_mask(&items, Some(&[vec![false; 3], vec![false]]), &[1.0; 7]).unwrap();
        assert_eq!(none.values, rule_only.values);
        assert_eq!(
            serde_json::to_value(&none.report).unwrap()["excluded_reviewed_ambiguity_o_tokens"],
            0
        );
        for bad in [
            vec![vec![false, true, false], vec![false]],
            vec![vec![false, false, true], vec![false]],
            vec![vec![true, false], vec![false]],
            vec![vec![true, false, false]],
        ] {
            assert!(batch_loss_mask(&items, Some(&bad), &[1.0; 7]).is_err());
        }
        assert!(
            batch_loss_mask(
                &items,
                Some(&[vec![true, true, false], vec![true]]),
                &[1.0; 7]
            )
            .is_err()
        );
    }
    #[test]
    fn actual_ad_ce_zeroes_reviewed_ambiguity_gradient_and_keeps_forward_inputs() {
        use burn::backend::{Autodiff, NdArray};
        use burn::data::dataloader::batcher::Batcher;
        use burn::prelude::*;
        type AD = Autodiff<NdArray>;
        let device = Default::default();
        let items = vec![
            item(vec![0, 0, flag::IN_RULE_SPAN], vec![0, 1, 0]),
            item(vec![0, 0], vec![0, 0]),
        ];
        let before = items.clone();
        let rows = vec![vec![true, false, false], vec![false, true]];
        let reviewed = batch_loss_mask(&items, Some(&rows), &[1.0; 7]).unwrap();
        let rule_only = batch_loss_mask(&items, None, &[1.0; 7]).unwrap();
        assert_eq!(items, before);
        assert_eq!(reviewed.values, vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);
        let batcher = crate::dataset::FeatureBatcher::detector(
            tessera::internal::DetectorFeatureContract::TabCells25,
        );
        let forward = |batch: crate::dataset::ParserBatch<AD>| {
            (
                batch.ngram_ids.into_data().to_vec::<i64>().unwrap(),
                batch.script.into_data().to_vec::<i64>().unwrap(),
                batch.shape.into_data().to_vec::<i64>().unwrap(),
                batch.flags.into_data().to_vec::<f32>().unwrap(),
                batch.mask.into_data().to_vec::<f32>().unwrap(),
            )
        };
        let batch: crate::dataset::ParserBatch<AD> = batcher.batch(items.clone(), &device);
        let labels = batch.labels.clone();
        let inputs = forward(batch);
        assert_eq!(inputs, forward(batcher.batch(before, &device)));
        assert_eq!(inputs.4, vec![1.0, 1.0, 1.0, 1.0, 1.0, 0.0]);
        assert_eq!(
            labels.clone().into_data().to_vec::<i64>().unwrap(),
            vec![0, 1, 0, 0, 0, 0]
        );
        let weights = Tensor::<AD, 1>::ones([7], &device);
        let gradient = |mask: LossMask| {
            let logits = Tensor::<AD, 3>::zeros([2, 3, 7], &device).require_grad();
            let mask =
                Tensor::<AD, 2>::from_data(TensorData::new(mask.values, mask.dimensions), &device);
            let loss =
                crate::train::masked_loss(logits.clone(), labels.clone(), mask, weights.clone());
            let scalar = loss.clone().into_scalar();
            assert!((scalar - 7.0f32.ln()).abs() < 1e-6);
            logits
                .grad(&loss.backward())
                .unwrap()
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        };
        let excluded = gradient(reviewed);
        let token = |t: usize| &excluded[t * 7..(t + 1) * 7];
        assert!(excluded.iter().all(|v| v.is_finite()));
        for t in [0, 2, 4, 5] {
            assert!(token(t).iter().all(|v| *v == 0.0), "token {t} has gradient");
        }
        for t in [1, 3] {
            assert!(
                token(t).iter().any(|v| *v != 0.0),
                "token {t} lost gradient"
            );
        }
        assert!(token(1)[1] < 0.0 && token(3)[0] < 0.0);
        let kept = gradient(rule_only);
        assert!(kept[0] < 0.0 && kept[28] < 0.0);
    }
    #[test]
    fn version_is_closed_and_absent_field_defaults_legacy() {
        assert_eq!(Objective::default(), Objective::LegacyPaddingV1);
        assert!(serde_json::from_str::<Objective>("\"future\"").is_err());
        let reviewed: Objective = serde_json::from_str(
            "\"canonical_rule_and_reviewed_ambiguity_excluded_weighted_ce_v1\"",
        )
        .unwrap();
        assert!(reviewed.excludes_reviewed_ambiguity() && !reviewed.is_legacy());
        assert!(!Objective::CanonicalRuleExcludedV1.excludes_reviewed_ambiguity());
        assert_eq!(
            serde_json::to_value(Objective::CanonicalRuleExcludedV1).unwrap(),
            "canonical_rule_excluded_weighted_ce_v1"
        );
    }
}
