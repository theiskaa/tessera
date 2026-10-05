//! Fixed native weighted-CE updates; no DEV early stopping or checkpoint ranking.

use super::{contract::Fit, observations, snapshot, write_json};
use crate::dataset::ParserBatch;
use crate::diagnostic_operator::ForwardOperator;
use crate::reviewed_data::digest;
use anyhow::{Context, ensure};
use burn::backend::{Autodiff, NdArray};
use burn::data::dataloader::batcher::Batcher;
use burn::optim::{GradientsParams, Optimizer};
use burn::prelude::*;
use serde_json::{Value, json};
use std::io::Write;
use std::path::Path;

type TrainBackend = Autodiff<NdArray>;

/// Execute exactly the preregistered TRAIN batches and retain fixed-final weights.
pub(super) fn run(fit: &Fit, out: &Path) -> anyhow::Result<()> {
    snapshot::verify(fit, out)?;
    let device = Default::default();
    let (mut model, mut optimizer, initialization) =
        crate::reviewed_train::initialize::initialize(&fit.native, &fit.loaded, &device)?;
    write_json(&out.join("initialization.json"), &initialization)?;
    let cfg = fit.native.config();
    let class_weights = fit.class_weights()?;
    let weights = Tensor::<TrainBackend, 1>::from_data(
        TensorData::new(class_weights.clone(), [crate::detector::DETECTOR_LABELS]),
        &device,
    );
    let mut observer = observations::Observer::new(fit, out)?;
    let mut draws = vec![0usize; fit.train_len()];
    observer.capture(0, &model, &draws)?;
    let mut progress = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(out.join("completed-batches.jsonl"))?;
    let started = std::time::Instant::now();
    let mut previous_batch = None::<String>;
    for (index, indices) in fit.plan.batches.iter().enumerate() {
        let items = fit.train_items(indices)?;
        let loss_mask = fit.loss_mask(indices, &items)?;
        if let Some(mask) = &loss_mask {
            fit.require_planned_activity(index, &mask.report)?;
        }
        let batch: ParserBatch<TrainBackend> =
            crate::dataset::FeatureBatcher::detector(cfg.detector_feature_contract())
                .batch(items, &device);
        let logits = ForwardOperator::ContextRmsV2.forward(
            &model,
            batch.ngram_ids,
            batch.script,
            batch.shape,
            batch.flags,
            batch.mask.clone(),
        )?;
        let loss_report = loss_mask.as_ref().map(|mask| mask.report.clone());
        let mask = match loss_mask {
            None => batch.mask,
            Some(mask) => {
                ensure!(
                    batch.labels.dims() == mask.dimensions
                        && logits.dims() == [mask.dimensions[0], mask.dimensions[1], 7],
                    "loss-excluded CE tensor geometry differs from canonical batch"
                );
                Tensor::<TrainBackend, 2>::from_data(
                    TensorData::new(mask.values, mask.dimensions),
                    &device,
                )
            }
        };
        let loss = crate::train::masked_loss(logits, batch.labels, mask, weights.clone());
        let value: f64 = loss.clone().into_scalar().elem();
        ensure!(
            value.is_finite() && value < 100.0,
            "nonfinite or runaway native weighted CE"
        );
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        crate::reviewed_train::initialize::verify_gradients(&model, &gradients)?;
        let lr = crate::train::lr_at(
            index,
            fit.manifest.updates,
            cfg.train.learning_rate,
            cfg.train.warmup_steps,
        );
        model = optimizer.step(lr, model, gradients);
        for &i in indices {
            draws[i] = draws[i].checked_add(1).context("draw count overflow")?;
        }
        let step = index + 1;
        let mut record = json!({"scope":fit.scope(),"step":step,"document_indices":indices,
            "learning_rate":lr,"native_weighted_ce":value,"elapsed_seconds":started.elapsed().as_secs_f64(),
            "previous_batch_sha256":previous_batch});
        if let Some(report) = loss_report {
            record["loss_objective"] = json!(fit.manifest.loss_objective);
            record["loss_activity"] = serde_json::to_value(report)?;
        }
        let mut line = serde_json::to_vec(&record)?;
        line.push(b'\n');
        progress.write_all(&line)?;
        progress.flush()?;
        previous_batch = Some(digest(&line));
        if step.is_multiple_of(25) {
            eprintln!(
                "reviewed native step{step}/{}, CE{value:.5}",
                fit.manifest.updates
            );
        }
        if fit.manifest.snapshots.contains(&step) {
            observer.capture(step, &model, &draws)?;
        }
    }
    progress.sync_all()?;
    snapshot::verify(fit, out)?;
    let final_record = observer
        .final_record
        .as_ref()
        .context("fixed final observation missing")?;
    let ledger_sha =
        observations::verify_ledger(fit, out, &draws, &final_record["final_checkpoint"])?;
    let batch_sha = verify_batches(fit, out, &previous_batch)?;
    let checkpoint = out.join(format!("checkpoints/step-{}.mpk", fit.manifest.updates));
    let bytes = std::fs::read(&checkpoint)?;
    ensure!(
        digest(&bytes) == final_record["final_checkpoint"]["checkpoint_sha256"],
        "final checkpoint changed"
    );
    let mut best = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(out.join("best.mpk"))?;
    best.write_all(&bytes)?;
    best.sync_all()?;
    ensure!(
        digest(&std::fs::read(out.join("best.mpk"))?) == digest(&bytes),
        "fixed final copy changed"
    );
    write_json(
        &out.join("learning-proof.json"),
        &final_record["learning_sanity_proof"],
    )?;
    snapshot::verify(fit, out)?;
    ensure!(
        observations::verify_ledger(fit, out, &draws, &final_record["final_checkpoint"])?
            == ledger_sha,
        "checkpoint ledger changed before completion"
    );
    ensure!(
        verify_batches(fit, out, &previous_batch)? == batch_sha,
        "completed batch ledger changed before completion"
    );
    ensure!(
        digest(&std::fs::read(out.join("best.mpk"))?) == digest(&bytes),
        "final selected weights changed before completion"
    );
    ensure!(
        serde_json::from_slice::<Value>(&std::fs::read(out.join("learning-proof.json"))?)?
            == final_record["learning_sanity_proof"],
        "learning proof changed before completion"
    );
    write_json(
        &out.join("completion.json"),
        &json!({"scope":fit.scope(),
        "optimizer_updates_executed":fit.plan.batches.len(),"fixed_updates":fit.manifest.updates,
        "document_draw_counts":draws,"checkpoint_selection":"fixed final; DEV never ranks checkpoints",
        "final":final_record,"dev_target_met":final_record["dev_95_per_kind_precision_recall_objective"]["target_met"],
        "learning_sanity_passed":final_record["learning_sanity_proof"]["passed"],"checkpoint_ledger_sha256":ledger_sha,
        "completed_batches_sha256":batch_sha,
        "input_snapshot_sha256":digest(&std::fs::read(out.join("input_snapshot.json"))?),
        "best_sha256":digest(&std::fs::read(out.join("best.mpk"))?),
        "learning_proof_sha256":digest(&std::fs::read(out.join("learning-proof.json"))?),
        "source_inputs_unchanged":true,"export_allowed":false,"general_accuracy_claim":false,"release_quality_claim":false}),
    )?;
    ensure!(
        final_record["learning_sanity_proof"]["passed"] == true,
        "fixed updates completed but real TRAIN learning sanity failed; inspect retained completion and proof"
    );
    Ok(())
}

fn verify_batches(
    fit: &Fit,
    out: &Path,
    expected_final: &Option<String>,
) -> anyhow::Result<String> {
    let text = std::fs::read_to_string(out.join("completed-batches.jsonl"))?;
    let lines: Vec<_> = text.split_inclusive('\n').collect();
    ensure!(
        lines.len() == fit.manifest.updates,
        "completed optimizer count differs"
    );
    let mut previous = None::<String>;
    for (index, (line, expected)) in lines.iter().zip(&fit.plan.batches).enumerate() {
        let row: Value = serde_json::from_str(line)?;
        ensure!(
            line.ends_with('\n')
                && row["scope"] == fit.scope()
                && row["step"] == index + 1
                && row["previous_batch_sha256"] == serde_json::to_value(&previous)?
                && row["document_indices"] == serde_json::to_value(expected)?
                && row["learning_rate"]
                    == crate::train::lr_at(
                        index,
                        fit.manifest.updates,
                        fit.native.config().train.learning_rate,
                        fit.native.config().train.warmup_steps
                    )
                && row["native_weighted_ce"]
                    .as_f64()
                    .is_some_and(|v| v.is_finite() && v < 100.0),
            "completed batch ledger changed"
        );
        if !fit.manifest.loss_objective.is_legacy() {
            let items = fit.train_items(expected)?;
            let mask = fit
                .loss_mask(expected, &items)?
                .context("non-legacy objective replayed without a loss mask")?;
            fit.require_planned_activity(index, &mask.report)?;
            ensure!(
                row["loss_objective"] == serde_json::to_value(fit.manifest.loss_objective)?
                    && row["loss_activity"] == serde_json::to_value(mask.report)?,
                "completed loss objective/activity changed"
            );
        } else {
            ensure!(
                row.get("loss_objective").is_none() && row.get("loss_activity").is_none(),
                "legacy batch ledger contains a different objective"
            );
        }
        previous = Some(digest(line.as_bytes()));
    }
    ensure!(
        previous == *expected_final,
        "completed batch chain differs from actual writes"
    );
    Ok(digest(text.as_bytes()))
}

#[cfg(test)]
mod tests {
    #[test]
    fn scheduled_learning_rates_survive_exact_json_roundtrip() {
        for step in 0..200 {
            let rate = crate::train::lr_at(step, 200, 3e-5, 20);
            let encoded = serde_json::to_vec(&serde_json::json!({"learning_rate": rate})).unwrap();
            let decoded: serde_json::Value = serde_json::from_slice(&encoded).unwrap();
            assert_eq!(
                decoded["learning_rate"].as_f64().unwrap().to_bits(),
                rate.to_bits()
            );
        }
    }
}
