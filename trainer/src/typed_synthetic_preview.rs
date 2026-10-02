//! Zero-update native finalized batch and objective accounting.

use super::*;

/// Exact zero-update finalized prefix with typed membership and objective coefficients.
pub(crate) fn preview(
    cfg: &Config,
    items: &[crate::dataset::Encoded],
    batches: &[Vec<usize>],
    uniform: usize,
    boundary: usize,
    metadata: &[Value],
    horizon: usize,
) -> anyhow::Result<Value> {
    ensure!(
        boundary == uniform + metadata.len() && items.len() >= boundary && !batches.is_empty(),
        "typed preview needs actual native finalized batches and boundaries"
    );
    let weights = &cfg
        .detector
        .as_ref()
        .context("missing detector")?
        .class_weights;
    let mut presentations = vec![0usize; items.len()];
    let mut inverse = vec![0.0; items.len()];
    let mut lr_inverse = vec![0.0; items.len()];
    let mut records = Vec::new();
    for (step, indices) in batches.iter().enumerate() {
        let denominator = crate::fullmix_exposure::denominator(items, indices, weights)?;
        let lr = crate::train::lr_at(
            step,
            horizon,
            cfg.train.learning_rate,
            cfg.train.warmup_steps,
        );
        for &index in indices {
            presentations[index] += 1;
            inverse[index] += 1.0 / denominator;
            lr_inverse[index] += lr / denominator;
        }
        records.push(json!({"planned_update":step+1,"optimizer_step_index":step,"indices":indices,"native_weighted_token_denominator":denominator,"scheduled_learning_rate":lr}));
    }
    let mut authored = Vec::new();
    for (offset, meta) in metadata.iter().enumerate() {
        let index = uniform + offset;
        let mut counts = [0usize; crate::detector::DETECTOR_LABELS];
        for &label in &items[index].labels {
            counts[label as usize] += 1;
        }
        authored.push(json!({"identity":meta,"content_tokens":items[index].token_spans.len(),"label_token_counts":counts,"presentations":presentations[index],
            "inverse_native_denominator_sum":inverse[index],"lr_inverse_native_denominator_sum":lr_inverse[index],
            "label_normalized_coefficients":counts.iter().zip(weights).map(|(count,weight)| *count as f64 * *weight as f64 * inverse[index]).collect::<Vec<_>>(),
            "label_lr_coefficients":counts.iter().zip(weights).map(|(count,weight)| *count as f64 * *weight as f64 * lr_inverse[index]).collect::<Vec<_>>() }));
    }
    let result = json!({"scope":"native encoded typed train-only zero-update finalized prefix; no model/fit/probe integration approval","model_initialized":false,"model_forwards":0,"optimizer_updates_executed":0,
        "uniform_synthetic_pool":uniform,"full_synthetic_boundary":boundary,"real_documents":items.len()-boundary,"planned_updates":batches.len(),
        "batch_sequence_sha256":crate::export::sha256_hex(&serde_json::to_vec(&records)?),"batches":records,"authored_documents":authored,
        "aggregate_exposure":crate::training_audit::audit_epoch(items,batches,boundary,weights)?});
    Ok(result)
}
