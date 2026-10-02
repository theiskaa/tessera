//! Actual presentation and native loss coefficients for the explicit whole-original cohort.

use std::path::Path;

use anyhow::{Context, ensure};
use serde_json::{Value, json};

use crate::config::Config;
use crate::dataset::Encoded;

/// Exact CPU initialization shared with the original bounded diagnostic.
pub(crate) const INITIAL_SHA: &str =
    "7271f208ee44530d419a7aa5160d1a4c4dae5e6c8ecd9881eba78468507efca9";
/// Fixed completed-epoch observations plus the initial and primary final checkpoints.
pub(crate) const OBSERVATIONS: [usize; 7] = [0, 510, 765, 1080, 1500, 1995, 2000];

#[derive(Default)]
struct RowExposure {
    presentations: usize,
    inverse_mass_sum: f64,
    lr_inverse_mass_sum: f64,
}

/// Diagnostic-only accounting; it never runs a model or changes the objective.
pub(crate) struct Exposure {
    rows: Vec<RowExposure>,
    selection: Value,
    batches: Vec<Value>,
    lr_sum: f64,
}

impl Exposure {
    /// Activate only for the explicitly marked fifteen-row native objective.
    pub(crate) fn new(
        cfg: &Config,
        run: &Path,
        items: &[Encoded],
        order: &[usize],
        cap: Option<usize>,
        ordinary_detector_epoch: bool,
    ) -> anyhow::Result<Option<Self>> {
        let marker_path = run.join("diagnostic.json");
        if !marker_path.exists() {
            return Ok(None);
        }
        let marker: Value = serde_json::from_slice(&std::fs::read(marker_path)?)?;
        if marker.get("frozen_cohort_scope").is_none() {
            return Ok(None);
        }
        let selection: Value = serde_json::from_slice(&std::fs::read(run.join("selection.json"))?)?;
        crate::memorization::verify_frozen_selection(cfg, &marker, &selection)?;
        ensure!(
            !ordinary_detector_epoch
                && cap == Some(2000)
                && cfg.train.epochs == 134
                && cfg
                    .detector
                    .as_ref()
                    .is_some_and(|d| d.learning_check.is_none()),
            "frozen PERSON diagnostic schedule or objective changed"
        );
        let path = Path::new(
            marker["frozen_cohort_path"]
                .as_str()
                .context("missing cohort path")?,
        );
        let prepared = crate::memorization::prepare_frozen(cfg, path)?;
        ensure!(
            items.len() == 15
                && items
                    .iter()
                    .zip(&prepared.documents)
                    .all(|(item, doc)| *item == doc.enc),
            "frozen training arrays changed"
        );
        ensure!(
            order.len() == 480
                && (0..15).all(|index| order.iter().filter(|&&draw| draw == index).count() == 32),
            "frozen PERSON diagnostic requires32 copies per original per epoch"
        );
        let initial: Value =
            serde_json::from_slice(&std::fs::read(run.join("diagnostic_initial.json"))?)?;
        ensure!(
            initial["parameter_sha256"] == INITIAL_SHA
                && initial["learning_rate_horizon_steps"] == 7000,
            "frozen initial parameters or horizon changed"
        );
        Ok(Some(Self {
            rows: (0..items.len()).map(|_| RowExposure::default()).collect(),
            selection,
            batches: Vec::new(),
            lr_sum: 0.0,
        }))
    }

    /// Record successful actual duplicate draws and their native weighted-token denominator.
    pub(crate) fn record(
        &mut self,
        chunk: &[usize],
        items: &[Encoded],
        weights: &[f32],
        lr: f64,
    ) -> anyhow::Result<()> {
        ensure!(
            chunk.len() == 32 && lr.is_finite() && lr > 0.0,
            "frozen exposure requires physical batch32 and a positive finite rate"
        );
        let mut mass = 0.0;
        for &index in chunk {
            let item = items
                .get(index)
                .context("exposure draw is outside original rows")?;
            ensure!(
                !item.labels.is_empty() && item.labels.len() <= 900,
                "exposure row is empty or exceeds900 tokens"
            );
            for &label in &item.labels {
                let weight = f64::from(
                    *weights
                        .get(usize::from(label))
                        .context("invalid exposure label")?,
                );
                ensure!(
                    weight > 0.0 && weight <= 3.0 && (weight * 2.0).fract() == 0.0,
                    "exposure requires exact native integer or half-integer weights"
                );
                mass += weight;
            }
        }
        ensure!(
            mass.is_finite() && mass > 0.0,
            "invalid native batch denominator"
        );
        // Half-integer sums below86400 are represented exactly by the native f32 reduction.
        ensure!(
            f64::from(mass as f32) == mass,
            "exposure denominator is not native-f32 exact"
        );
        for &index in chunk {
            let row = self
                .rows
                .get_mut(index)
                .context("exposure state row missing")?;
            row.presentations += 1;
            row.inverse_mass_sum += 1.0 / mass;
            row.lr_inverse_mass_sum += lr / mass;
        }
        self.batches
            .push(json!({"completed_update": self.batches.len() + 1,
            "original_indices": chunk, "native_weighted_token_denominator": mass,
            "learning_rate": lr}));
        self.lr_sum += lr;
        Ok(())
    }

    /// Serialize per-token and per-gold coefficient sums at every fixed observation.
    pub(crate) fn write(
        &self,
        run: &Path,
        items: &[Encoded],
        weights: &[f32],
    ) -> anyhow::Result<()> {
        let selected = self.selection["documents"]
            .as_array()
            .context("missing selected rows")?;
        ensure!(
            items.len() == self.rows.len() && selected.len() == items.len(),
            "exposure output row alignment changed"
        );
        let mut rows = Vec::new();
        for ((state, item), doc) in self.rows.iter().zip(items).zip(selected) {
            let gold = doc["gold"].as_array().context("missing native gold")?;
            let mut spans = Vec::new();
            for span in gold {
                let start = u32::try_from(span["start"].as_u64().context("invalid gold start")?)?;
                let end = u32::try_from(span["end"].as_u64().context("invalid gold end")?)?;
                let mut tokens = Vec::new();
                for (index, &(token_start, token_end)) in item.token_spans.iter().enumerate() {
                    if start <= token_start && token_end <= end {
                        let label = *item
                            .labels
                            .get(index)
                            .context("exposure token label missing")?;
                        let weight = f64::from(
                            *weights
                                .get(usize::from(label))
                                .context("exposure weight missing")?,
                        );
                        tokens.push(
                            json!({"index": index, "start": token_start, "end": token_end,
                            "native_label": label, "class_weight": weight,
                            "coefficient_sum": weight * state.inverse_mass_sum,
                            "lr_weighted_coefficient_sum": weight * state.lr_inverse_mass_sum}),
                        );
                    }
                }
                ensure!(!tokens.is_empty(), "exposure gold has no native tokens");
                spans.push(json!({"gold": span, "presentations": state.presentations,
                                  "native_tokens": tokens}));
            }
            rows.push(
                json!({"name": doc["name"], "source_path": doc["source_path"],
                "source_line": doc["source_line"], "presentations": state.presentations,
                "inverse_native_denominator_sum": state.inverse_mass_sum,
                "lr_inverse_native_denominator_sum": state.lr_inverse_mass_sum, "gold": spans}),
            );
        }
        let updates = self.batches.len();
        std::fs::write(
            run.join(format!("exposure-step-{updates}.json")),
            serde_json::to_vec_pretty(
                &json!({"updates": updates, "learning_rate_sum": self.lr_sum,
                "draws": self.rows.iter().map(|r| r.presentations).sum::<usize>(),
                "scope": "coefficient sums, not gradient magnitudes; actual partial-epoch counts",
                "batches": self.batches, "documents": rows}),
            )?,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::seq::SliceRandom;
    use rand_chacha::ChaCha8Rng;

    fn fixture(count: usize) -> (Vec<Encoded>, Exposure) {
        let items = (0..count)
            .map(|_| {
                crate::detector::encode_document(
                    "Ada Lane",
                    &[crate::detector::KindSpan {
                        kind: 0,
                        start: 0,
                        end: 8,
                    }],
                    &tessera::internal::FeatureConfig::default(),
                )
                .unwrap()
            })
            .collect();
        let exposure = Exposure {
            rows: (0..count).map(|_| RowExposure::default()).collect(),
            selection: json!({}),
            batches: vec![],
            lr_sum: 0.0,
        };
        (items, exposure)
    }

    #[test]
    fn duplicate_draw_coefficients_use_actual_denominator_and_rate() {
        let (items, mut exposure) = fixture(2);
        let weights = [1.0, 3.0, 2.0, 3.0, 2.0, 2.0, 1.5];
        let mut chunk = vec![0; 10];
        chunk.extend(vec![1; 22]);
        exposure.record(&chunk, &items, &weights, 0.0001).unwrap();
        assert_eq!(exposure.rows[0].presentations, 10);
        assert_eq!(exposure.rows[1].presentations, 22);
        assert_eq!(
            exposure.batches[0]["native_weighted_token_denominator"],
            160.0
        );
        assert!((exposure.rows[0].inverse_mass_sum - 10.0 / 160.0).abs() < 1e-15);
        assert!((exposure.rows[1].lr_inverse_mass_sum - 22.0 * 0.0001 / 160.0).abs() < 1e-15);
        assert!(exposure.record(&[0; 31], &items, &weights, 0.0001).is_err());
        assert!(exposure.record(&[2; 32], &items, &weights, 0.0001).is_err());
    }

    #[test]
    fn fixed2000_uses133_full_epochs_plus_actual160_draws() {
        let (items, mut exposure) = fixture(15);
        let mut order: Vec<_> = (0..32).flat_map(|_| 0..15).collect();
        let mut update = 0;
        let mut final_counts = [0; 15];
        for epoch in 1..=134 {
            order.shuffle(&mut ChaCha8Rng::seed_from_u64(42 ^ epoch));
            for chunk in order.chunks(32) {
                if update == 2000 {
                    break;
                }
                if epoch == 134 {
                    for &i in chunk {
                        final_counts[i] += 1;
                    }
                }
                let lr = crate::train::lr_at(update, 7000, 0.0001, 500);
                exposure
                    .record(chunk, &items, &[1., 3., 2., 3., 2., 2., 1.5], lr)
                    .unwrap();
                update += 1;
            }
        }
        assert_eq!(exposure.batches.len(), 2000);
        assert_eq!(final_counts.iter().sum::<usize>(), 160);
        assert_eq!(
            exposure.rows.iter().map(|r| r.presentations).sum::<usize>(),
            64000
        );
        for (row, last) in exposure.rows.iter().zip(final_counts) {
            assert_eq!(row.presentations, 4256 + last);
        }
        assert!(final_counts.iter().any(|count| *count != final_counts[0]));
    }
}
