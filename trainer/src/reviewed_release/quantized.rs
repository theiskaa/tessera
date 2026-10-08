//! Int8 weights for the fit's final checkpoint, gated on the fit's own DEV documents through its
//! deployable decode path: Context96 forward, configured postprocess and confidence policy.

use std::path::Path;

use anyhow::{Context, ensure};
use burn::backend::NdArray;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::config::{Config, DetectorPostprocess};
use crate::detector::{DetectorInputPolicy, KINDS};
use crate::net::TaggerNet;
use crate::quantize::{self, MAX_F1_DROP};
use crate::release_candidate::hash;
use crate::reviewed_data::Cohort;

/// The fit's DEV documents and the decode path and observation it recorded for them.
pub(super) struct Scoring<'a> {
    pub(super) cfg: &'a Config,
    pub(super) dev: &'a Cohort,
    pub(super) policy: DetectorInputPolicy,
    pub(super) postprocess: Option<DetectorPostprocess>,
    pub(super) recorded: &'a Value,
    pub(super) parameter_sha256: &'a str,
}

/// DEV F1 of the f32 and int8 weights, recorded in the gate and the approval.
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Quantization {
    pub(super) dev_documents: usize,
    pub(super) f32_macro_exact_f1: f64,
    pub(super) int8_macro_exact_f1: f64,
    pub(super) f32_micro_f1: f64,
    pub(super) int8_micro_f1: f64,
    pub(super) max_f1_drop: f64,
}

impl Quantization {
    /// The existing quantization limit, on the same macro exact F1 over neural kinds.
    pub(super) fn passed(&self) -> bool {
        self.f32_macro_exact_f1.is_finite()
            && self.int8_macro_exact_f1.is_finite()
            && self.max_f1_drop == MAX_F1_DROP
            && self.f32_macro_exact_f1 - self.int8_macro_exact_f1 <= MAX_F1_DROP
    }

    /// Gate fields; the quantization gate adds its decision and the artifact hashes.
    pub(super) fn summary(&self) -> Map<String, Value> {
        let mut summary = Map::new();
        summary.insert(
            "evaluation".into(),
            "reviewed fit DEV documents through the fit's deployable decode path".into(),
        );
        summary.insert("dev_documents".into(), self.dev_documents.into());
        summary.insert(
            "dev_f32_macro_exact_f1".into(),
            self.f32_macro_exact_f1.into(),
        );
        summary.insert(
            "dev_int8_macro_exact_f1".into(),
            self.int8_macro_exact_f1.into(),
        );
        summary.insert("dev_f32_micro_f1".into(), self.f32_micro_f1.into());
        summary.insert("dev_int8_micro_f1".into(), self.int8_micro_f1.into());
        summary.insert("max_f1_drop".into(), self.max_f1_drop.into());
        summary
    }

    /// Require a passing record that the published gate repeats exactly.
    pub(super) fn verify(&self, gate: &Value) -> anyhow::Result<()> {
        ensure!(
            self.passed()
                && gate["passed"] == true
                && self
                    .summary()
                    .iter()
                    .all(|(key, value)| gate.get(key) == Some(value)),
            "reviewed release quantization record differs from its passing gate"
        );
        Ok(())
    }
}

fn score(
    model: &TaggerNet<NdArray>,
    scoring: &Scoring,
    device: &burn::backend::ndarray::NdArrayDevice,
) -> anyhow::Result<Value> {
    match scoring.postprocess {
        Some(postprocess) => crate::cohort_fit::score_with_postprocess(
            model,
            scoring.dev,
            device,
            scoring.policy,
            postprocess,
        ),
        None => {
            crate::cohort_fit::score_with_input_policy(model, scoring.dev, device, scoring.policy)
        }
    }
}

fn macro_f1(score: &Value) -> anyhow::Result<f64> {
    let mut total = 0.0;
    for kind in KINDS {
        total += score["filtered"]["per_kind"][kind.as_str()]["f1"]
            .as_f64()
            .context("DEV per-kind F1 absent")?;
    }
    Ok(total / KINDS.len() as f64)
}

fn micro_f1(score: &Value) -> anyhow::Result<f64> {
    score["filtered"]["micro"]["f1"]
        .as_f64()
        .context("DEV micro F1 absent")
}

/// Require the staged final checkpoint's f32 DEV score to reproduce the fit's own observation,
/// then quantize it; the shared quantization gate publishes the weights only if int8 DEV macro
/// exact F1 stays within `MAX_F1_DROP`.
pub(super) fn run(out: &Path, scoring: &Scoring) -> anyhow::Result<Quantization> {
    let initial = json!({
        "config_sha256": hash(&out.join("config.toml"))?,
        "best_sha256": hash(&out.join("best.mpk"))?,
        "input_snapshot_sha256": hash(&out.join("input_snapshot.json"))?,
    });
    let device = Default::default();
    let net = scoring.cfg.detector_net_config();
    let (model, parameters) =
        crate::span_scores::load_checkpoint(&net, &out.join("best.mpk"), &device)?;
    ensure!(
        parameters == scoring.parameter_sha256
            && model.flag_bits() == scoring.cfg.detector_feature_contract().flag_bits(),
        "staged checkpoint parameters differ from the fit's fixed final checkpoint"
    );
    let f32_score = score(&model, scoring, &device)?;
    ensure!(
        f32_score == *scoring.recorded,
        "f32 DEV rescoring differs from the fit's recorded fixed-final DEV observation"
    );
    let tensors = quantize::extract(&model, "detector");
    let (q, biases, dequantized) = quantize::quantize_all(&tensors);
    let pending = out.join("quantized.pending.safetensors");
    quantize::write_safetensors(&pending, &q, &biases, None)?;
    let int8_model = quantize::inject::<NdArray>(&net, &dequantized, "detector", &device)?;
    let int8_score = score(&int8_model, scoring, &device)?;
    let quantization = Quantization {
        dev_documents: scoring.dev.docs.len(),
        f32_macro_exact_f1: macro_f1(&f32_score)?,
        int8_macro_exact_f1: macro_f1(&int8_score)?,
        f32_micro_f1: micro_f1(&f32_score)?,
        int8_micro_f1: micro_f1(&int8_score)?,
        max_f1_drop: MAX_F1_DROP,
    };
    let max_err = tensors
        .iter()
        .zip(&dequantized)
        .flat_map(|(a, b)| a.data.iter().zip(&b.data).map(|(x, y)| (x - y).abs()))
        .fold(0f32, f32::max);
    let file_bytes = std::fs::metadata(&pending)?.len();
    let mut summary = quantization.summary();
    summary.insert(
        "dev_f32_per_kind".into(),
        f32_score["filtered"]["per_kind"].clone(),
    );
    summary.insert(
        "dev_int8_per_kind".into(),
        int8_score["filtered"]["per_kind"].clone(),
    );
    summary.insert("max_abs_weight_error".into(), max_err.into());
    summary.insert("file_bytes".into(), file_bytes.into());
    println!(
        "detector DEV macro exact F1: f32 {:.4}, int8 {:.4}; micro F1: f32 {:.4}, int8 {:.4}; max weight error {max_err:.2e}; {file_bytes} bytes",
        quantization.f32_macro_exact_f1,
        quantization.int8_macro_exact_f1,
        quantization.f32_micro_f1,
        quantization.int8_micro_f1,
    );
    quantize::publish_scored(
        out,
        &pending,
        &initial,
        (
            quantization.f32_macro_exact_f1,
            quantization.int8_macro_exact_f1,
        ),
        &mut summary,
    )?;
    Ok(quantization)
}
