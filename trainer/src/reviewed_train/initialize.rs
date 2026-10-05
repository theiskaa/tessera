//! Full detector f32 warm start on the training AD backend, with fresh optimizer state.

use anyhow::{Context, ensure};
use burn::backend::{Autodiff, NdArray};
use burn::module::{AutodiffModule, Module, ModuleVisitor, Param};
use burn::optim::{AdamWConfig, Optimizer, grad_clipping::GradientClippingConfig};
use burn::prelude::*;
use serde_json::{Value, json};

use super::{Loaded, Prepared};
use crate::diagnostic_operator::ForwardOperator;
use crate::net::TaggerNet;

type TrainBackend = Autodiff<NdArray>;

fn trainable(model: &TaggerNet<TrainBackend>) -> anyhow::Result<()> {
    struct Visitor {
        parameters: usize,
        frozen: usize,
    }
    impl ModuleVisitor<TrainBackend> for Visitor {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<TrainBackend, D>>) {
            self.parameters += 1;
            self.frozen += usize::from(!param.val().is_require_grad());
        }
    }
    let mut visitor = Visitor {
        parameters: 0,
        frozen: 0,
    };
    model.visit(&mut visitor);
    ensure!(
        visitor.parameters > 0 && visitor.frozen == 0,
        "original detector parameters must all retain AD gradient flags"
    );
    Ok(())
}

/// Initialize only from verified original detector f32 weights; no forward or optimizer step.
pub(crate) fn initialize(
    prepared: &Prepared,
    loaded: &Loaded,
    device: &burn::backend::ndarray::NdArrayDevice,
) -> anyhow::Result<(
    TaggerNet<TrainBackend>,
    impl Optimizer<TaggerNet<TrainBackend>, TrainBackend>,
    Value,
)> {
    prepared.verify_inputs()?;
    let bindings = prepared.identity(loaded)?;
    for (name, value) in [("RAYON_NUM_THREADS", "2"), ("MATMUL_NUM_THREADS", "1")] {
        ensure!(
            std::env::var(name).ok().as_deref() == Some(value),
            "set {name}={value} before reviewed initialization"
        );
    }
    let model = crate::quantize::load_best_for_operator::<TrainBackend>(
        &prepared.source.run,
        &prepared.source_config,
        device,
        ForwardOperator::ContextRmsV2,
    )?;
    let parameters = crate::quantize::extract(&model.valid(), "detector");
    ensure!(
        !parameters.is_empty()
            && parameters
                .iter()
                .all(|p| p.data.iter().all(|v| v.is_finite())),
        "original detector contains missing or nonfinite f32 parameters"
    );
    let actual = crate::training_diagnostic::parameter_sha256(&parameters);
    ensure!(
        actual == prepared.source.parameter_sha256,
        "original detector f32 parameter identity differs"
    );
    let proof = crate::fullmix_rms::verify_checkpoint_parameters(&prepared.source.run, &actual)?;
    let (model, migration) = match prepared.config.detector_feature_contract() {
        tessera::internal::DetectorFeatureContract::Legacy23 => (model, Value::Null),
        tessera::internal::DetectorFeatureContract::TabCells25 => {
            let (model, migration) = crate::tab_cell_migration::extend_model(model, device)?;
            ensure!(
                migration["original_parameter_sha256"] == actual,
                "migration is not bound to the original source proof"
            );
            (model, migration)
        }
    };
    let initialized_parameter_sha256 = crate::training_diagnostic::parameter_sha256(
        &crate::quantize::extract(&model.valid(), "detector"),
    );
    ensure!(
        model.flag_bits() == prepared.config.detector_feature_contract().flag_bits(),
        "initialized flags differ from canonical recipe"
    );
    trainable(&model)?;
    ensure!(
        TrainBackend::ad_enabled(device),
        "reviewed training initialization requires AD and training dropout"
    );
    crate::net::assert_size(
        &model,
        prepared.config.net.max_params,
        prepared.config.net.max_embedding_bytes,
    )?;
    prepared.verify_inputs()?;
    let clip = prepared
        .config
        .train
        .gradient_clip_norm
        .context("reviewed training requires explicit gradient clipping")?;
    TrainBackend::seed(device, prepared.config.seed);
    let optimizer = AdamWConfig::new()
        .with_weight_decay(0.01)
        .with_grad_clipping(Some(GradientClippingConfig::Norm(clip)))
        .init::<TrainBackend, TaggerNet<TrainBackend>>();
    Ok((
        model,
        optimizer,
        json!({"actual_parameter_sha256":initialized_parameter_sha256,"original_parameter_sha256":actual,"original_checkpoint_proof":proof,"feature_migration":migration,
        "optimizer_state":"fresh AdamW; source optimizer not loaded","ngram_from":prepared.config.net.ngram_from,
        "ngram_weights":"preserved from original detector; parser weights not copied",
        "training_dropout_active":true,"optimizer_updates_executed":0,"forward_executed":false,
        "thread_bounds":{"RAYON_NUM_THREADS":2,"MATMUL_NUM_THREADS":1},
        "recipe_identity":bindings,"fit_integration_complete":false,"fit_allowed":false,
        "runtime_feature_parity_verified":false,"export_allowed":false}),
    ))
}

/// Reject missing or nonfinite gradients before any future optimizer mutation.
pub(crate) fn verify_gradients(
    model: &TaggerNet<TrainBackend>,
    gradients: &burn::optim::GradientsParams,
) -> anyhow::Result<()> {
    crate::cohort_fit::verify_gradients(model, gradients)
}
