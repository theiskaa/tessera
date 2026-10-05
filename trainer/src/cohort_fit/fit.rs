//! Isolated native CE fitting with fixed observations and no development selection.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, ensure};
use burn::backend::{Autodiff, NdArray};
use burn::data::dataloader::batcher::Batcher;
use burn::module::{AutodiffModule, Module, ModuleVisitor, Param};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer, grad_clipping::GradientClippingConfig};
use burn::prelude::*;
#[cfg(test)]
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use serde_json::{Value, json};

use super::contracts::{Prepared, digest};
use super::data::Cohorts;
use crate::dataset::{ParserBatch, ParserBatcher};
use crate::detector;
use crate::diagnostic_operator::ForwardOperator;
use crate::net::TaggerNet;

type TrainBackend = Autodiff<NdArray>;
type EvalBackend = NdArray<f32>;

struct Observation<'a> {
    prepared: &'a Prepared,
    cohorts: &'a Cohorts,
    out: &'a Path,
    identity: &'a Value,
    previous_event_sha256: Option<String>,
    baseline_dev: Option<Value>,
    acceptance: Option<Value>,
    events: std::fs::File,
}

fn finite_parameter_sha256(tensors: &[crate::quantize::F32Tensor]) -> anyhow::Result<String> {
    crate::native_checkpoint::parameter_sha256(tensors)
}

fn verify_saved_observations(out: &Path, snapshots: &[usize]) -> anyhow::Result<()> {
    let ledger = std::fs::read_to_string(out.join("checkpoint-events.jsonl"))?;
    let mut previous: Option<String> = None;
    let lines: Vec<_> = ledger.split_inclusive('\n').collect();
    ensure!(
        lines.len() == snapshots.len(),
        "cohort snapshot ledger count differs"
    );
    for (line, step) in lines.iter().zip(snapshots) {
        let event: Value = serde_json::from_str(line)?;
        let checkpoint = format!("checkpoints/step-{step}.mpk");
        let score = format!("scores-step-{step}.json");
        ensure!(
            line.ends_with('\n')
                && event["scope"] == super::SCOPE
                && event["optimizer_updates_executed"] == *step
                && event["checkpoint_file"] == checkpoint
                && event["score_file"] == score
                && event["native_roundtrip_verified"] == true
                && event["all_parameters_finite"] == true
                && event["previous_event_sha256"] == serde_json::to_value(&previous)?
                && event["checkpoint_sha256"] == digest(&std::fs::read(out.join(checkpoint))?)
                && event["score_sha256"] == digest(&std::fs::read(out.join(score))?),
            "cohort saved observation or ledger changed: step{step}"
        );
        previous = Some(digest(line.as_bytes()));
    }
    Ok(())
}

/// Reject missing or nonfinite trainable gradients before an optimizer step.
pub(crate) fn verify_gradients(
    model: &TaggerNet<TrainBackend>,
    gradients: &GradientsParams,
) -> anyhow::Result<()> {
    struct Visitor<'a> {
        gradients: &'a GradientsParams,
        found: usize,
        missing: usize,
        non_finite: usize,
    }
    impl ModuleVisitor<TrainBackend> for Visitor<'_> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<TrainBackend, D>>) {
            if let Some(gradient) = self.gradients.get::<EvalBackend, D>(param.id) {
                self.found += 1;
                if !gradient
                    .into_data()
                    .to_vec::<f32>()
                    .is_ok_and(|v| v.iter().all(|x| x.is_finite()))
                {
                    self.non_finite += 1;
                }
            } else if param.val().is_require_grad() {
                self.missing += 1;
            }
        }
    }
    let mut visitor = Visitor {
        gradients,
        found: 0,
        missing: 0,
        non_finite: 0,
    };
    model.visit(&mut visitor);
    ensure!(
        visitor.found > 0 && visitor.missing == 0 && visitor.non_finite == 0,
        "cohort gradients invalid before optimizer mutation: {} present, {} missing, {} non-finite",
        visitor.found,
        visitor.missing,
        visitor.non_finite
    );
    Ok(())
}

impl Observation<'_> {
    fn capture(
        &mut self,
        step: usize,
        model: &TaggerNet<TrainBackend>,
        draws: &[usize],
    ) -> anyhow::Result<()> {
        ensure!(
            self.prepared.manifest.snapshots.contains(&step),
            "unregistered cohort snapshot"
        );
        self.prepared.verify_inputs()?;
        ensure!(
            self.prepared.identity(self.cohorts)? == *self.identity,
            "cohort code, binary, recipe or cohort identity changed during fit"
        );
        let device = Default::default();
        let valid = model.valid();
        let relative = format!("checkpoints/step-{step}.mpk");
        let base = self.out.join("checkpoints").join(format!("step-{step}"));
        let saved = crate::native_checkpoint::save(valid, &base, &device)?;
        let parameters = saved.parameter_sha256;
        let checkpoint_sha256 = saved.checkpoint_sha256;
        let restored = saved.model;
        let train = super::scoring::score(&restored, &self.cohorts.train, &device)?;
        let dev = super::scoring::score(&restored, &self.cohorts.dev, &device)?;
        if step == 0 {
            self.baseline_dev = Some(dev.clone());
        }
        if step == self.prepared.manifest.max_steps {
            self.acceptance = Some(super::scoring::acceptance(
                &train,
                &dev,
                self.baseline_dev
                    .as_ref()
                    .context("step0 baseline missing")?,
            )?);
        }
        self.prepared.verify_inputs()?;
        ensure!(
            digest(&std::fs::read(base.with_extension("mpk"))?) == checkpoint_sha256,
            "cohort saved checkpoint changed during observation scoring"
        );
        let scores = json!({"step":step,"train_seen_fit":train,"separate_dev_observation":dev,
            "development_used_for_selection":false, "parameter_sha256":parameters});
        let score_file = format!("scores-step-{step}.json");
        super::write_json(&self.out.join(&score_file), &scores)?;
        let event = json!({"scope":super::SCOPE,"optimizer_updates_executed":step,
            "checkpoint_file":relative, "checkpoint_sha256":checkpoint_sha256,
            "parameter_sha256":parameters, "native_roundtrip_verified":true,
            "all_parameters_finite":true,"score_file":score_file,
            "score_sha256":digest(&std::fs::read(self.out.join(&score_file))?),
            "document_draw_counts":draws,"selection":"preregistered snapshot; no best selection",
            "previous_event_sha256":self.previous_event_sha256,"release_quality_claim":false});
        let mut bytes = serde_json::to_vec(&event)?;
        bytes.push(b'\n');
        self.events.write_all(&bytes)?;
        self.events.sync_all()?;
        self.previous_event_sha256 = Some(digest(&bytes));
        Ok(())
    }
}

pub(super) fn run(
    prepared: &Prepared,
    cohorts: &Cohorts,
    out: &Path,
    identity: &Value,
) -> anyhow::Result<()> {
    prepared.verify_inputs()?;
    let cfg = &prepared.config;
    let device = Default::default();
    // Loading on the inference backend loses the parameter gradient flags in Burn 0.21.
    let mut model = crate::quantize::load_best_for_operator::<TrainBackend>(
        &prepared.manifest.source_run,
        cfg,
        &device,
        ForwardOperator::ContextRmsV2,
    )?;
    let actual_initial_sha256 =
        finite_parameter_sha256(&crate::quantize::extract(&model.valid(), "detector"))?;
    ensure!(
        actual_initial_sha256 == prepared.manifest.expected_parameter_sha256,
        "warm-start f32 parameter identity differs; refusing optimizer initialization"
    );
    let proof = crate::fullmix_rms::verify_checkpoint_parameters(
        &prepared.manifest.source_run,
        &actual_initial_sha256,
    )?;
    prepared.verify_inputs()?;
    super::write_json(
        &out.join("initialization.json"),
        &json!({"actual_initial_sha256":actual_initial_sha256,
        "original_checkpoint_proof":proof,"optimizer_state":"fresh; source optimizer not loaded",
        "ngram_weights":"preserved from original detector checkpoint; no parser overwrite",
        "training_dropout_active":true,"observation_dropout_active":false}),
    )?;
    ensure!(
        TrainBackend::ad_enabled(&device) && !EvalBackend::ad_enabled(&device),
        "cohort training/observation backends must enable/disable dropout respectively"
    );
    crate::net::assert_size(&model, cfg.net.max_params, cfg.net.max_embedding_bytes)?;
    TrainBackend::seed(&device, cfg.seed);
    let mut optim = AdamWConfig::new()
        .with_weight_decay(0.01)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init::<TrainBackend, TaggerNet<TrainBackend>>();
    let weights = cfg
        .detector
        .as_ref()
        .context("missing detector class weights")?
        .class_weights
        .clone();
    let class_weights = Tensor::<TrainBackend, 1>::from_data(
        TensorData::new(weights, [detector::DETECTOR_LABELS]),
        &device,
    );
    std::fs::create_dir(out.join("checkpoints"))?;
    let events = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(out.join("checkpoint-events.jsonl"))?;
    let mut observation = Observation {
        prepared,
        cohorts,
        out,
        identity,
        previous_event_sha256: None,
        baseline_dev: None,
        acceptance: None,
        events,
    };
    let mut draws = vec![0usize; cohorts.train.docs.len()];
    observation.capture(0, &model, &draws)?;
    let mut progress = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(out.join("completed-batches.jsonl"))?;
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);
    let mut step = 0;
    let mut epoch = 0;
    let started = std::time::Instant::now();
    while step < prepared.manifest.max_steps {
        epoch += 1;
        let mut order: Vec<_> = (0..cohorts.train.docs.len()).collect();
        order.shuffle(&mut rng);
        for indices in order.chunks(cfg.train.batch_size) {
            if step == prepared.manifest.max_steps {
                break;
            }
            let items = indices
                .iter()
                .map(|&i| cohorts.train.docs[i].enc.clone())
                .collect();
            let batch: ParserBatch<TrainBackend> = ParserBatcher.batch(items, &device);
            let logits = ForwardOperator::ContextRmsV2.forward(
                &model,
                batch.ngram_ids,
                batch.script,
                batch.shape,
                batch.flags,
                batch.mask.clone(),
            )?;
            let loss =
                crate::train::masked_loss(logits, batch.labels, batch.mask, class_weights.clone());
            let value: f64 = loss.clone().into_scalar().elem();
            ensure!(
                value.is_finite() && value < 100.0,
                "cohort step{step}: non-finite or runaway native CE {value}"
            );
            let gradients = GradientsParams::from_grads(loss.backward(), &model);
            verify_gradients(&model, &gradients)?;
            let lr =
                crate::train::lr_at(step, 7000, cfg.train.learning_rate, cfg.train.warmup_steps);
            model = optim.step(lr, model, gradients);
            for &i in indices {
                draws[i] += 1;
            }
            step += 1;
            writeln!(
                progress,
                "{}",
                json!({"step":step,"epoch":epoch,"document_indices":indices,
                "learning_rate":lr,"native_weighted_ce":value,"elapsed_seconds":started.elapsed().as_secs_f64()})
            )?;
            progress.flush()?;
            if step.is_multiple_of(25) {
                eprintln!(
                    "cohort step{step}/{}, native CE{value:.5}",
                    prepared.manifest.max_steps
                );
            }
            if prepared.manifest.snapshots.contains(&step) {
                observation.capture(step, &model, &draws)?;
            }
        }
    }
    progress.sync_all()?;
    prepared.verify_inputs()?;
    verify_saved_observations(out, &prepared.manifest.snapshots)?;
    ensure!(
        prepared.identity(cohorts)? == *identity,
        "cohort provenance changed before completion"
    );
    let acceptance = observation
        .acceptance
        .as_ref()
        .context("final diagnostic acceptance report missing")?;
    super::write_json(
        &out.join("completion.json"),
        &json!({"scope":super::SCOPE,
        "optimizer_updates_executed":step,"fixed_budget":prepared.manifest.max_steps,
        "document_draw_counts":draws,"epochs_started":epoch,"seconds":started.elapsed().as_secs_f64(),
        "checkpoint_event_sha256":observation.previous_event_sha256,
        "checkpoint_ledger_sha256":digest(&std::fs::read(out.join("checkpoint-events.jsonl"))?),
        "completed_batches_sha256":digest(&std::fs::read(out.join("completed-batches.jsonl"))?),
        "checkpoint_selection":"none","general_accuracy_claim":false,"export_allowed":false,
        "acceptance":acceptance,
        "release_quality_claim":false,"source_inputs_unchanged":true}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn warm_start_gradients() -> (TaggerNet<TrainBackend>, GradientsParams) {
        let device = Default::default();
        let cfg = crate::net::TaggerNetConfig::new(4, vec![1], 7)
            .with_ngram_dim(2)
            .with_script_dim(2)
            .with_shape_dim(2)
            .with_hidden(4);
        let original = cfg.init::<EvalBackend>(&device);
        let initial =
            finite_parameter_sha256(&crate::quantize::extract(&original, "detector")).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("original");
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        original.save_file(&base, &recorder).unwrap();
        let inference = cfg
            .init::<EvalBackend>(&device)
            .load_file(&base, &recorder, &device)
            .unwrap();
        let disabled: TaggerNet<TrainBackend> = inference.train();
        assert!(!disabled.ngram.weight.val().is_require_grad());
        assert!(!disabled.head.weight.val().is_require_grad());
        let model = cfg
            .init::<TrainBackend>(&device)
            .load_file(&base, &recorder, &device)
            .unwrap();
        assert_eq!(
            finite_parameter_sha256(&crate::quantize::extract(&model.valid(), "detector")).unwrap(),
            initial
        );
        assert!(model.ngram.weight.val().is_require_grad());
        assert!(model.head.weight.val().is_require_grad());
        let mask = Tensor::from_data([[1., 1., 1.]], &device);
        let logits = model.forward(
            Tensor::from_data([[[1], [2], [3]]], &device),
            Tensor::from_data([[0, 1, 0]], &device),
            Tensor::from_data([[0, 1, 0]], &device),
            Tensor::zeros([1, 3, crate::dataset::FLAG_BITS], &device),
            mask.clone(),
        );
        let loss = crate::train::masked_loss(
            logits,
            Tensor::from_data([[0, 1, 2]], &device),
            mask,
            Tensor::ones([7], &device),
        );
        let gradients = GradientsParams::from_grads(loss.backward(), &model);
        (model, gradients)
    }

    #[test]
    fn native_checkpoint_warm_start_preserves_weights_and_updates_all_parameters() {
        let (model, gradients) = warm_start_gradients();
        verify_gradients(&model, &gradients).unwrap();
        assert_eq!(gradients.len(), 9);
        let initial =
            finite_parameter_sha256(&crate::quantize::extract(&model.valid(), "detector")).unwrap();
        let mut optimizer = AdamWConfig::new().init::<TrainBackend, TaggerNet<TrainBackend>>();
        let updated = optimizer.step(1e-4, model, gradients);
        assert_ne!(
            finite_parameter_sha256(&crate::quantize::extract(&updated.valid(), "detector"))
                .unwrap(),
            initial
        );
    }

    #[test]
    fn missing_and_nonfinite_gradients_are_refused_before_updates() {
        let (model, mut gradients) = warm_start_gradients();
        let head_id = model.head.weight.id;
        gradients.remove::<EvalBackend, 2>(head_id).unwrap();
        assert!(verify_gradients(&model, &gradients).is_err());
        gradients.register(
            head_id,
            Tensor::<EvalBackend, 2>::full([4, 7], f32::NAN, &Default::default()),
        );
        assert!(verify_gradients(&model, &gradients).is_err());
        assert!(verify_gradients(&model, &GradientsParams::new()).is_err());
    }

    #[test]
    fn autodiff_backend_selects_dropout_without_model_or_forward_initialization() {
        let device = Default::default();
        assert!(TrainBackend::ad_enabled(&device));
        assert!(!EvalBackend::ad_enabled(&device));
    }

    #[test]
    fn checkpoint_parameters_must_be_finite_before_a_success_identity() {
        let mut tensors = vec![crate::quantize::F32Tensor {
            name: "detector.weight".into(),
            shape: vec![1],
            data: vec![0.5],
        }];
        assert!(finite_parameter_sha256(&tensors).is_ok());
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            tensors[0].data[0] = value;
            assert!(finite_parameter_sha256(&tensors).is_err());
        }
    }

    #[test]
    fn native_snapshot_roundtrip_preserves_f32_parameters_without_a_forward() {
        let device = Default::default();
        let model = crate::net::TaggerNetConfig::new(4, vec![1], 7)
            .with_ngram_dim(2)
            .with_script_dim(2)
            .with_shape_dim(2)
            .with_hidden(4)
            .init::<EvalBackend>(&device);
        let initial =
            finite_parameter_sha256(&crate::quantize::extract(&model, "detector")).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let base = dir.path().join("step-0");
        model
            .clone()
            .save_file(&base, &NamedMpkFileRecorder::<FullPrecisionSettings>::new())
            .unwrap();
        let restored = model
            .load_file(
                &base,
                &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
                &device,
            )
            .unwrap();
        assert_eq!(
            finite_parameter_sha256(&crate::quantize::extract(&restored, "detector")).unwrap(),
            initial
        );
    }
}
