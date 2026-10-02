//! Shared hard-token batch preparation, objective, and exposure accounting.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, ensure};
use burn::prelude::*;
use burn::tensor::activation::log_softmax;
use serde::Serialize;
use serde_json::json;

const WEIGHTS: [f32; 7] = [1., 3., 2., 3., 2., 2., 1.5];
const GROUPS: usize = 3;

/// One complete native row with validated disjoint person, org_contrast, address masks.
pub(crate) struct NativeRow {
    index: usize,
    labels: Vec<u8>,
    masks: [Vec<f32>; GROUPS],
    identity: String,
}

impl NativeRow {
    /// Validate host labels and masks before preparing any loss tensors.
    pub(crate) fn new(
        index: usize,
        labels: Vec<u8>,
        masks: Option<[Vec<f32>; GROUPS]>,
    ) -> anyhow::Result<Self> {
        ensure!(
            !labels.is_empty() && labels.len() <= 900,
            "invalid complete native row length"
        );
        ensure!(
            labels.iter().all(|&label| label < 7),
            "native label outside seven classes"
        );
        let masks = masks.unwrap_or_else(|| std::array::from_fn(|_| vec![0.; labels.len()]));
        crate::hard_token_masks::HardMasks::new(
            [1, labels.len()],
            &vec![1.; labels.len()],
            masks.clone(),
            [1. / 16.; GROUPS],
        )?;
        let identity = crate::export::sha256_hex(&serde_json::to_vec(&json!({
            "native_index":index,"labels":labels,"masks":masks,
        }))?);
        Ok(Self {
            index,
            labels,
            masks,
            identity,
        })
    }
}

/// Immutable native-index lookup; duplicate presentations belong in ordered chunks.
pub(crate) struct Rows(BTreeMap<usize, NativeRow>);

impl Rows {
    /// Refuse duplicate native identities instead of silently replacing a row.
    pub(crate) fn new(rows: Vec<NativeRow>) -> anyhow::Result<Self> {
        let mut map = BTreeMap::new();
        for row in rows {
            ensure!(
                map.insert(row.index, row).is_none(),
                "duplicate native row identity"
            );
        }
        Ok(Self(map))
    }
}

/// Pure normalized coefficients, with auxiliary groups kept separate from base CE.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub(crate) struct Coefficients {
    base: f64,
    auxiliary: [f64; GROUPS],
    total: f64,
    lr_base: f64,
    lr_auxiliary: [f64; GROUPS],
    lr_total: f64,
}

impl Coefficients {
    fn add(&mut self, other: &Self) -> anyhow::Result<()> {
        self.base += other.base;
        self.total += other.total;
        self.lr_base += other.lr_base;
        self.lr_total += other.lr_total;
        for group in 0..GROUPS {
            self.auxiliary[group] += other.auxiliary[group];
            self.lr_auxiliary[group] += other.lr_auxiliary[group];
        }
        ensure!(
            self.base.is_finite()
                && self.total.is_finite()
                && self.lr_base.is_finite()
                && self.lr_total.is_finite()
                && self.auxiliary.iter().all(|n| n.is_finite())
                && self.lr_auxiliary.iter().all(|n| n.is_finite()),
            "cumulative coefficient overflow"
        );
        Ok(())
    }
}

/// Compact exact batch identity and native denominators for planned and actual replay.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct BatchRecord {
    update_index: usize,
    ordered_native_indices: Vec<usize>,
    lengths: Vec<usize>,
    shape: [usize; 2],
    label_mask_identity_sha256: String,
    row_identities: Vec<String>,
    class_weights: [f32; 7],
    coefficients: Option<[f32; GROUPS]>,
    base_denominator: f64,
    group_denominators: [f64; GROUPS],
    scheduled_learning_rate: f64,
}

/// Validated ordered full-row batch; only this host preparation supplies loss targets.
pub(crate) struct PreparedBatch {
    labels: Vec<i64>,
    real: Vec<f32>,
    masks: [Vec<f32>; GROUPS],
    record: BatchRecord,
}

impl PreparedBatch {
    /// Collate unchanged labels and masks in exact chunk order, preserving duplicates.
    pub(crate) fn collate(
        rows: &Rows,
        chunk: &[usize],
        class_weights: &[f32],
        coefficients: Option<[f32; GROUPS]>,
        update_index: usize,
        scheduled_learning_rate: f64,
    ) -> anyhow::Result<Self> {
        ensure!(
            !chunk.is_empty() && chunk.len() <= 32,
            "invalid native batch size"
        );
        ensure!(
            class_weights == WEIGHTS,
            "batch requires the exact seven native class weights"
        );
        ensure!(
            scheduled_learning_rate.is_finite() && scheduled_learning_rate > 0.,
            "invalid scheduled LR"
        );
        if let Some(coefficients) = coefficients {
            ensure!(
                coefficients
                    .iter()
                    .all(|&coefficient| coefficient.is_finite()
                        && (0.0..=1. / 16.).contains(&coefficient)),
                "invalid hard coefficient"
            );
        }
        let selected: Vec<_> = chunk
            .iter()
            .map(|index| rows.0.get(index).context("ordered chunk index absent"))
            .collect::<Result<_, _>>()?;
        let length = selected
            .iter()
            .map(|row| row.labels.len())
            .max()
            .context("batch has no rows")?;
        let count = chunk
            .len()
            .checked_mul(length)
            .context("batch dimensions overflow")?;
        let mut labels = vec![0i64; count];
        let mut real = vec![0.; count];
        let mut masks: [Vec<f32>; GROUPS] = std::array::from_fn(|_| vec![0.; count]);
        let mut base_denominator = 0.;
        let mut group_denominators = [0.; GROUPS];
        for (slot, row) in selected.iter().enumerate() {
            for (token, &label) in row.labels.iter().enumerate() {
                let flat = slot * length + token;
                labels[flat] = i64::from(label);
                real[flat] = 1.;
                let weight = f64::from(WEIGHTS[label as usize]);
                base_denominator += weight;
                for group in 0..GROUPS {
                    masks[group][flat] = row.masks[group][token];
                    if masks[group][flat] == 1. {
                        group_denominators[group] += weight;
                    }
                }
            }
        }
        // Integer and half-integer weights sum exactly in f32 within the native batch bound.
        ensure!(
            base_denominator <= 86400. && f64::from(base_denominator as f32) == base_denominator,
            "native denominator is not proven f32 exact"
        );
        let identity = crate::export::sha256_hex(&serde_json::to_vec(&json!({
            "shape":[chunk.len(),length],"indices":chunk,"labels":labels,"real_mask":real,"hard_masks":masks,
        }))?);
        let record = BatchRecord {
            update_index,
            ordered_native_indices: chunk.to_vec(),
            lengths: selected.iter().map(|row| row.labels.len()).collect(),
            shape: [chunk.len(), length],
            label_mask_identity_sha256: identity,
            row_identities: selected.iter().map(|row| row.identity.clone()).collect(),
            class_weights: WEIGHTS,
            coefficients,
            base_denominator,
            group_denominators,
            scheduled_learning_rate,
        };
        Ok(Self {
            labels,
            real,
            masks,
            record,
        })
    }

    /// The same immutable record is compared before an update and recorded afterward.
    pub(crate) fn record(&self) -> &BatchRecord {
        &self.record
    }

    /// Exact mathematical loss coefficients at one padded batch position.
    pub(crate) fn token_coefficients(
        &self,
        slot: usize,
        token: usize,
    ) -> anyhow::Result<Coefficients> {
        let [batch, length] = self.record.shape;
        ensure!(
            slot < batch && token < length,
            "coefficient position outside batch"
        );
        let flat = slot * length + token;
        if self.real[flat] == 0. {
            return Ok(Coefficients::default());
        }
        let weight = f64::from(WEIGHTS[self.labels[flat] as usize]);
        let base = weight / self.record.base_denominator.max(1.);
        let auxiliary = std::array::from_fn(|group| {
            let denominator = self.record.group_denominators[group];
            if self.masks[group][flat] == 1. && denominator > 0. {
                self.record
                    .coefficients
                    .map_or(0., |coefficients| f64::from(coefficients[group]))
                    * weight
                    / denominator
            } else {
                0.
            }
        });
        let total = base + auxiliary.iter().sum::<f64>();
        let lr = self.record.scheduled_learning_rate;
        Ok(Coefficients {
            base,
            auxiliary,
            total,
            lr_base: lr * base,
            lr_auxiliary: auxiliary.map(|value| lr * value),
            lr_total: lr * total,
        })
    }

    fn active_groups(&self) -> [bool; GROUPS] {
        std::array::from_fn(|group| {
            self.record.group_denominators[group] > 0.
                && self
                    .record
                    .coefficients
                    .is_some_and(|coefficients| coefficients[group] > 0.)
        })
    }
}

/// Efficient CE on unchanged logits, with no host reads and one shared log-softmax.
pub(crate) fn loss<B: Backend>(
    logits: Tensor<B, 3>,
    batch: &PreparedBatch,
) -> anyhow::Result<Tensor<B, 1>> {
    let [rows, length] = batch.record.shape;
    ensure!(
        logits.dims() == [rows, length, 7],
        "logit shape differs from validated host batch"
    );
    let device = logits.device();
    let labels = Tensor::<B, 2, Int>::from_data(
        TensorData::new(batch.labels.clone(), [rows, length]),
        &device,
    );
    let real =
        Tensor::<B, 2>::from_data(TensorData::new(batch.real.clone(), [rows, length]), &device);
    let weights = Tensor::<B, 1>::from_data(WEIGHTS, &device);
    let active = batch.active_groups();
    if !active.contains(&true) {
        return Ok(crate::train::masked_loss(logits, labels, real, weights));
    }
    let count = rows * length;
    let logp = log_softmax(logits.reshape([count, 7]), 1);
    let target = labels.reshape([count, 1]);
    let nll = logp.gather(1, target.clone()).neg().reshape([count]);
    let w = weights.gather(0, target.reshape([count]));
    let mask = real.reshape([count]);
    let numerator = (nll.clone() * w.clone() * mask.clone()).sum();
    let denominator = (w.clone() * mask).sum().clamp_min(1.);
    let mut total = numerator / denominator;
    let coefficients = batch
        .record
        .coefficients
        .context("active hard group lacks coefficients")?;
    for group in 0..GROUPS {
        if !active[group] {
            continue;
        }
        let hard = Tensor::<B, 1>::from_data(
            TensorData::new(batch.masks[group].clone(), [count]),
            &device,
        );
        let weighted = w.clone() * hard;
        total =
            total + (nll.clone() * weighted.clone()).sum() / weighted.sum() * coefficients[group];
    }
    Ok(total)
}

/// Planned-versus-actual pure replay; callers record only after an optimizer update.
pub(crate) struct Accounting {
    expected: Vec<BatchRecord>,
    next: usize,
    totals: BTreeMap<(usize, usize), Coefficients>,
    presentations: BTreeMap<usize, usize>,
    selected: Option<BTreeSet<(usize, usize)>>,
}

impl Accounting {
    /// Bind an ordered prefix produced by the shared batch calculator.
    pub(crate) fn new(expected: Vec<BatchRecord>) -> anyhow::Result<Self> {
        ensure!(!expected.is_empty(), "empty accounting plan");
        for (index, record) in expected.iter().enumerate() {
            ensure!(
                record.update_index == index,
                "accounting update sequence differs"
            );
        }
        Ok(Self {
            expected,
            next: 0,
            totals: BTreeMap::new(),
            presentations: BTreeMap::new(),
            selected: None,
        })
    }

    /// Restrict coefficient accumulation while preserving exact row presentation accounting.
    pub(crate) fn selected_only(
        expected: Vec<BatchRecord>,
        selected: BTreeSet<(usize, usize)>,
    ) -> anyhow::Result<Self> {
        ensure!(!selected.is_empty(), "empty selected accounting set");
        let mut accounting = Self::new(expected)?;
        accounting.selected = Some(selected);
        Ok(accounting)
    }

    /// Verify identities, labels, masks, coefficients, denominators and LR before backward.
    pub(crate) fn verify_next(&self, batch: &PreparedBatch) -> anyhow::Result<()> {
        let expected = self
            .expected
            .get(self.next)
            .context("accounting prefix exhausted")?;
        ensure!(
            expected == batch.record(),
            "actual batch differs from planned identity or objective"
        );
        Ok(())
    }

    /// Read cumulative mathematical coefficients without recomputing the objective.
    pub(crate) fn cumulative_coefficients(
        &self,
    ) -> impl Iterator<Item = (usize, usize, &Coefficients)> {
        self.totals
            .iter()
            .map(|(&(index, token), coefficients)| (index, token, coefficients))
    }

    /// Read actual recorded row multiplicities, including duplicates within each batch.
    pub(crate) fn presentations(&self) -> &BTreeMap<usize, usize> {
        &self.presentations
    }

    /// Accumulate the shared coefficients only for a verified completed update.
    pub(crate) fn record(&mut self, batch: &PreparedBatch) -> anyhow::Result<()> {
        self.verify_next(batch)?;
        let mut totals: BTreeMap<_, Coefficients> = BTreeMap::new();
        let mut presentations = BTreeMap::new();
        for (slot, &index) in batch.record.ordered_native_indices.iter().enumerate() {
            let count = presentations
                .entry(index)
                .or_insert_with(|| self.presentations.get(&index).copied().unwrap_or(0usize));
            *count = count
                .checked_add(1)
                .context("presentation count overflow")?;
            if self.selected.as_ref().is_some_and(|selected| {
                selected
                    .range((index, 0)..=(index, usize::MAX))
                    .next()
                    .is_none()
            }) {
                continue;
            }
            for token in 0..batch.record.lengths[slot] {
                if self
                    .selected
                    .as_ref()
                    .is_some_and(|selected| !selected.contains(&(index, token)))
                {
                    continue;
                }
                totals
                    .entry((index, token))
                    .or_insert_with(|| {
                        self.totals
                            .get(&(index, token))
                            .cloned()
                            .unwrap_or_default()
                    })
                    .add(&batch.token_coefficients(slot, token)?)?;
            }
        }
        self.totals.extend(totals);
        self.presentations.extend(presentations);
        self.next += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};

    fn rows() -> Rows {
        Rows::new(vec![
            NativeRow::new(
                10,
                vec![1, 2, 0],
                Some([vec![1., 1., 0.], vec![0., 0., 1.], vec![0.; 3]]),
            )
            .unwrap(),
            NativeRow::new(
                20,
                vec![5, 6],
                Some([vec![0.; 2], vec![0.; 2], vec![1.; 2]]),
            )
            .unwrap(),
        ])
        .unwrap()
    }

    fn batch(coefficients: Option<[f32; GROUPS]>, update: usize, lr: f64) -> PreparedBatch {
        PreparedBatch::collate(&rows(), &[10, 20, 10], &WEIGHTS, coefficients, update, lr).unwrap()
    }

    fn scores() -> Vec<f32> {
        (0..63)
            .map(|index| ((index * 17 % 23) as f32 - 11.) * 0.25)
            .collect()
    }

    fn evaluate(batch: &PreparedBatch) -> (f32, Vec<f32>) {
        type B = Autodiff<NdArray>;
        let device = Default::default();
        let logits =
            Tensor::<B, 3>::from_data(TensorData::new(scores(), [3, 3, 7]), &device).require_grad();
        let result = loss(logits.clone(), batch).unwrap();
        let value = result.clone().into_scalar();
        let gradient = result.backward();
        (
            value,
            logits
                .grad(&gradient)
                .unwrap()
                .into_data()
                .to_vec::<f32>()
                .unwrap(),
        )
    }

    #[test]
    fn ordered_duplicate_rows_padding_and_group_denominators_are_exact() {
        let batch = batch(Some([1. / 16.; 3]), 0, 0.001);
        assert_eq!(batch.record.shape, [3, 3]);
        assert_eq!(batch.record.lengths, vec![3, 2, 3]);
        assert_eq!(batch.record.base_denominator, 15.5);
        assert_eq!(batch.record.group_denominators, [10., 2., 3.5]);
        assert_eq!(batch.real, vec![1., 1., 1., 1., 1., 0., 1., 1., 1.]);
        assert_eq!(batch.labels, vec![1, 2, 0, 5, 6, 0, 1, 2, 0]);
        assert_eq!(
            batch.token_coefficients(1, 2).unwrap(),
            Coefficients::default()
        );
        assert_eq!(
            batch.token_coefficients(0, 0).unwrap(),
            batch.token_coefficients(2, 0).unwrap()
        );
    }

    #[test]
    fn efficient_loss_and_gradients_match_shared_pure_token_coefficients() {
        let batch = batch(Some([1. / 16.; 3]), 0, 0.001);
        let (value, gradient) = evaluate(&batch);
        let mut expected_loss = 0.;
        for (token, scores) in scores().chunks_exact(7).enumerate() {
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let shifted: Vec<f64> = scores.iter().map(|&score| f64::from(score - max)).collect();
            let sum: f64 = shifted.iter().map(|score| score.exp()).sum();
            let coefficient = batch.token_coefficients(token / 3, token % 3).unwrap();
            expected_loss += (sum.ln() - shifted[batch.labels[token] as usize]) * coefficient.total;
            assert_eq!(coefficient.lr_total, coefficient.total * 0.001);
            for (label, score) in shifted.iter().enumerate() {
                let target = if label == batch.labels[token] as usize {
                    1.
                } else {
                    0.
                };
                let expected = (score.exp() / sum - target) * coefficient.total;
                assert!((f64::from(gradient[token * 7 + label]) - expected).abs() < 2e-5);
            }
        }
        assert!((f64::from(value) - expected_loss).abs() < 2e-5);
    }

    #[test]
    fn absent_disabled_and_empty_groups_delegate_exact_native_base() {
        let absent = batch(None, 0, 0.001);
        let disabled = batch(Some([0.; 3]), 0, 0.001);
        assert_eq!(evaluate(&absent), evaluate(&disabled));
        let empty_rows = Rows::new(vec![
            NativeRow::new(10, vec![1, 2, 0], None).unwrap(),
            NativeRow::new(20, vec![5, 6], None).unwrap(),
        ])
        .unwrap();
        let empty = PreparedBatch::collate(
            &empty_rows,
            &[10, 20, 10],
            &WEIGHTS,
            Some([1. / 16.; 3]),
            0,
            0.001,
        )
        .unwrap();
        assert_eq!(evaluate(&absent), evaluate(&empty));
        let device = Default::default();
        let logits = Tensor::<NdArray, 3>::from_data(TensorData::new(scores(), [3, 3, 7]), &device);
        let labels = Tensor::from_data(TensorData::new(absent.labels.clone(), [3, 3]), &device);
        let real = Tensor::from_data(TensorData::new(absent.real.clone(), [3, 3]), &device);
        let weights = Tensor::from_data(WEIGHTS, &device);
        assert_eq!(
            loss(logits.clone(), &absent).unwrap().into_scalar(),
            crate::train::masked_loss(logits, labels, real, weights).into_scalar()
        );
    }

    #[test]
    fn unselected_tokens_keep_base_coefficients_and_gradients() {
        let rows = Rows::new(vec![
            NativeRow::new(
                1,
                vec![1, 2, 0],
                Some([vec![1., 0., 0.], vec![0.; 3], vec![0.; 3]]),
            )
            .unwrap(),
        ])
        .unwrap();
        let hard =
            PreparedBatch::collate(&rows, &[1], &WEIGHTS, Some([1. / 16.; 3]), 0, 0.001).unwrap();
        let base = PreparedBatch::collate(&rows, &[1], &WEIGHTS, None, 0, 0.001).unwrap();
        type B = Autodiff<NdArray>;
        let device = Default::default();
        let mut outputs = Vec::new();
        for batch in [&hard, &base] {
            let logits = Tensor::<B, 3>::from_data(
                TensorData::new(scores()[..21].to_vec(), [1, 3, 7]),
                &device,
            )
            .require_grad();
            let gradients = loss(logits.clone(), batch).unwrap().backward();
            outputs.push(
                logits
                    .grad(&gradients)
                    .unwrap()
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap(),
            );
        }
        assert_eq!(&outputs[0][7..], &outputs[1][7..]);
        for token in 1..3 {
            assert_eq!(
                hard.token_coefficients(0, token).unwrap(),
                base.token_coefficients(0, token).unwrap()
            );
        }
    }

    #[test]
    fn accounting_replay_counts_duplicates_and_lr_without_advancing_on_verification() {
        let first = batch(Some([1. / 16.; 3]), 0, 0.001);
        let second = batch(Some([1. / 16.; 3]), 1, 0.0005);
        let mut replay =
            Accounting::new(vec![first.record().clone(), second.record().clone()]).unwrap();
        replay.verify_next(&first).unwrap();
        replay.verify_next(&first).unwrap();
        assert_eq!(replay.next, 0);
        replay.record(&first).unwrap();
        replay.record(&second).unwrap();
        assert_eq!(replay.presentations()[&10], 4);
        assert_eq!(replay.presentations()[&20], 2);
        let mut expected = Coefficients::default();
        for batch in [&first, &second] {
            expected
                .add(&batch.token_coefficients(0, 0).unwrap())
                .unwrap();
            expected
                .add(&batch.token_coefficients(2, 0).unwrap())
                .unwrap();
        }
        assert_eq!(
            replay
                .cumulative_coefficients()
                .find(|&(index, token, _)| index == 10 && token == 0)
                .unwrap()
                .2,
            &expected
        );
        assert!(replay.verify_next(&second).is_err());
        assert!(Accounting::new(vec![]).is_err());
        assert!(Accounting::new(vec![second.record().clone()]).is_err());
    }

    #[test]
    fn label_mask_order_lr_and_coefficient_mutations_reject_before_recording() {
        let expected = batch(Some([1. / 16.; 3]), 0, 0.001);
        let mut replay = Accounting::new(vec![expected.record().clone()]).unwrap();
        let changed_label = Rows::new(vec![
            NativeRow::new(
                10,
                vec![1, 1, 0],
                Some([vec![1., 1., 0.], vec![0., 0., 1.], vec![0.; 3]]),
            )
            .unwrap(),
            NativeRow::new(
                20,
                vec![5, 6],
                Some([vec![0.; 2], vec![0.; 2], vec![1.; 2]]),
            )
            .unwrap(),
        ])
        .unwrap();
        let changed_mask = Rows::new(vec![
            NativeRow::new(
                10,
                vec![1, 2, 0],
                Some([vec![1., 0., 0.], vec![0., 0., 1.], vec![0.; 3]]),
            )
            .unwrap(),
            NativeRow::new(
                20,
                vec![5, 6],
                Some([vec![0.; 2], vec![0.; 2], vec![1.; 2]]),
            )
            .unwrap(),
        ])
        .unwrap();
        let changed = [
            PreparedBatch::collate(
                &changed_label,
                &[10, 20, 10],
                &WEIGHTS,
                Some([1. / 16.; 3]),
                0,
                0.001,
            )
            .unwrap(),
            PreparedBatch::collate(
                &changed_mask,
                &[10, 20, 10],
                &WEIGHTS,
                Some([1. / 16.; 3]),
                0,
                0.001,
            )
            .unwrap(),
            PreparedBatch::collate(
                &rows(),
                &[20, 10, 10],
                &WEIGHTS,
                Some([1. / 16.; 3]),
                0,
                0.001,
            )
            .unwrap(),
            batch(Some([1. / 16.; 3]), 0, 0.002),
            batch(Some([0.; 3]), 0, 0.001),
        ];
        for batch in changed {
            assert!(replay.record(&batch).is_err());
            assert_eq!(replay.next, 0);
            assert!(replay.totals.is_empty());
        }
    }

    #[test]
    fn invalid_host_rows_coefficients_weights_indices_shapes_and_lr_refuse() {
        assert!(NativeRow::new(0, vec![7], None).is_err());
        assert!(NativeRow::new(0, vec![], None).is_err());
        assert!(NativeRow::new(0, vec![0; 901], None).is_err());
        assert!(NativeRow::new(0, vec![0], Some([vec![1.], vec![1.], vec![0.]])).is_err());
        assert!(NativeRow::new(0, vec![0], Some([vec![f32::NAN], vec![0.], vec![0.]])).is_err());
        assert!(
            Rows::new(vec![
                NativeRow::new(0, vec![0], None).unwrap(),
                NativeRow::new(0, vec![0], None).unwrap()
            ])
            .is_err()
        );
        for lr in [f64::NAN, f64::INFINITY, 0., -0.001] {
            assert!(PreparedBatch::collate(&rows(), &[10], &WEIGHTS, None, 0, lr).is_err());
        }
        for coefficient in [f32::NAN, f32::INFINITY, -0.01, 0.063] {
            assert!(
                PreparedBatch::collate(&rows(), &[10], &WEIGHTS, Some([coefficient; 3]), 0, 0.001)
                    .is_err()
            );
        }
        assert!(PreparedBatch::collate(&rows(), &[], &WEIGHTS, None, 0, 0.001).is_err());
        assert!(PreparedBatch::collate(&rows(), &[10; 33], &WEIGHTS, None, 0, 0.001).is_err());
        assert!(PreparedBatch::collate(&rows(), &[99], &WEIGHTS, None, 0, 0.001).is_err());
        assert!(PreparedBatch::collate(&rows(), &[10], &[1.; 7], None, 0, 0.001).is_err());
        let device = Default::default();
        let batch = batch(None, 0, 0.001);
        assert!(loss(Tensor::<NdArray, 3>::zeros([3, 3, 6], &device), &batch).is_err());
    }
}
