//! `trainer quantize`: per-output-channel symmetric int8 weights with f32 scales, biases kept
//! in f32, written as `runs/<name>/quantized.safetensors` under the shipping tensor names
//! (`parser.*` or `detector.*`), then re-scored on the validation split; the run fails if its
//! F1 (component F1 for the parser, macro exact span F1 for the detector) drops by more than
//! `MAX_F1_DROP`.
//!
//! For a weight whose first dimension indexes output channels:
//! `scale[c] = max(|W[c, ..]|) / 127`, `q = round_half_to_even(W / scale[c])` clamped to
//! `[-127, 127]`, and an all-zero channel gets `scale = 1`. Embedding tables are quantized per
//! column instead, since the embedding dimension is what a lookup outputs. The library's
//! kernels reproduce exactly this.

use std::collections::{BTreeMap, HashMap};
use std::io::ErrorKind;
use std::path::Path;

use anyhow::{Context, bail};
use burn::module::Param;
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::TensorData;

use crate::data::{LabelledExample, Split, read_shard};
use crate::detector;
use crate::model_eval::{encode_examples, quick_score};
use crate::net::{TaggerNet, TaggerNetConfig};

/// A parameter in f32, in its shipping layout.
#[derive(Debug, Clone, PartialEq)]
pub struct F32Tensor {
    /// Shipping tensor name, such as `parser.proj.weight`.
    pub name: String,
    /// Dimensions, outermost first.
    pub shape: Vec<usize>,
    /// Values in row-major order.
    pub data: Vec<f32>,
}

/// An int8 parameter with one scale per channel.
#[derive(Debug, Clone, PartialEq)]
pub struct QTensor {
    /// Shipping tensor name; its scales ship as `<name>.scale`.
    pub name: String,
    /// Dimensions, outermost first.
    pub shape: Vec<usize>,
    /// Values in row-major order; `value * scale` recovers the weight.
    pub data: Vec<i8>,
    /// One scale per channel along `channel_axis`.
    pub scales: Vec<f32>,
}

/// Which axis carries the channels: 0 for weights, 1 for embedding tables.
pub fn channel_axis(name: &str) -> usize {
    if name.contains(".embed.") { 1 } else { 0 }
}

fn to_vec<B: Backend, const D: usize>(t: Tensor<B, D>) -> (Vec<usize>, Vec<f32>) {
    let shape = t.dims().to_vec();
    (
        shape,
        t.into_data()
            .convert::<f32>()
            .to_vec()
            .expect("converted to f32"),
    )
}

/// Every parameter of a tagger under its shipping name, `<net>.*`. Linear weights are
/// transposed from Burn's `[in, out]` to `[out, in]`; conv weights keep `[out, in, kernel]`.
pub fn extract<B: Backend>(model: &TaggerNet<B>, net: &str) -> Vec<F32Tensor> {
    let mut out = Vec::new();
    let mut push = |name: String, (shape, data): (Vec<usize>, Vec<f32>)| {
        out.push(F32Tensor { name, shape, data })
    };
    push(
        format!("{net}.embed.ngram"),
        to_vec(model.ngram.weight.val()),
    );
    push(
        format!("{net}.embed.script"),
        to_vec(model.script.weight.val()),
    );
    push(
        format!("{net}.embed.shape"),
        to_vec(model.shape.weight.val()),
    );
    push(
        format!("{net}.proj.weight"),
        to_vec(model.proj.weight.val().transpose()),
    );
    if let Some(b) = &model.proj.bias {
        push(format!("{net}.proj.bias"), to_vec(b.val()));
    }
    for (i, block) in model.blocks.iter().enumerate() {
        push(
            format!("{net}.block{i}.conv.weight"),
            to_vec(block.conv.weight.val()),
        );
        if let Some(b) = &block.conv.bias {
            push(format!("{net}.block{i}.conv.bias"), to_vec(b.val()));
        }
    }
    push(
        format!("{net}.head.weight"),
        to_vec(model.head.weight.val().transpose()),
    );
    if let Some(b) = &model.head.bias {
        push(format!("{net}.head.bias"), to_vec(b.val()));
    }
    out
}

fn tensor<B: Backend, const D: usize>(
    t: &F32Tensor,
    device: &B::Device,
) -> anyhow::Result<Tensor<B, D>> {
    let shape: [usize; D] = t
        .shape
        .clone()
        .try_into()
        .map_err(|_| anyhow::anyhow!("{} has {} dims, expected {D}", t.name, t.shape.len()))?;
    Ok(Tensor::from_data(
        TensorData::new(t.data.clone(), shape),
        device,
    ))
}

/// A model whose parameters are `tensors`, given in the shipping names `<net>.*` and layouts.
pub fn inject<B: Backend>(
    cfg: &TaggerNetConfig,
    tensors: &[F32Tensor],
    net: &str,
    device: &B::Device,
) -> anyhow::Result<TaggerNet<B>> {
    let by_name: HashMap<&str, &F32Tensor> = tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let get = |name: &str| {
        by_name
            .get(name)
            .copied()
            .with_context(|| format!("missing tensor `{name}`"))
    };
    let mut model = cfg.init::<B>(device);
    model.ngram.weight = Param::from_tensor(tensor(get(&format!("{net}.embed.ngram"))?, device)?);
    model.script.weight = Param::from_tensor(tensor(get(&format!("{net}.embed.script"))?, device)?);
    model.shape.weight = Param::from_tensor(tensor(get(&format!("{net}.embed.shape"))?, device)?);
    model.proj.weight = Param::from_tensor(
        tensor::<B, 2>(get(&format!("{net}.proj.weight"))?, device)?.transpose(),
    );
    model.proj.bias = Some(Param::from_tensor(tensor(
        get(&format!("{net}.proj.bias"))?,
        device,
    )?));
    for (i, block) in model.blocks.iter_mut().enumerate() {
        block.conv.weight = Param::from_tensor(tensor(
            get(&format!("{net}.block{i}.conv.weight"))?,
            device,
        )?);
        block.conv.bias = Some(Param::from_tensor(tensor(
            get(&format!("{net}.block{i}.conv.bias"))?,
            device,
        )?));
    }
    model.head.weight = Param::from_tensor(
        tensor::<B, 2>(get(&format!("{net}.head.weight"))?, device)?.transpose(),
    );
    model.head.bias = Some(Param::from_tensor(tensor(
        get(&format!("{net}.head.bias"))?,
        device,
    )?));
    Ok(model)
}

/// Whether a tensor is quantized: weights and embeddings are, biases stay f32.
pub fn is_quantized(name: &str) -> bool {
    !name.ends_with(".bias")
}

fn channel_of(shape: &[usize], axis: usize, flat: usize) -> usize {
    let channels = shape[axis];
    if axis == 0 {
        flat / (shape.iter().product::<usize>() / channels.max(1)).max(1)
    } else {
        flat % channels
    }
}

/// Per-channel symmetric int8 along `axis`: scale is the channel's largest magnitude over
/// 127, values rounded half to even.
pub fn quantize_tensor(t: &F32Tensor, axis: usize) -> QTensor {
    let channels = t.shape[axis];
    let mut scales = vec![0f32; channels];
    for (i, &v) in t.data.iter().enumerate() {
        let c = channel_of(&t.shape, axis, i);
        scales[c] = scales[c].max(v.abs());
    }
    for s in &mut scales {
        *s = if *s == 0.0 { 1.0 } else { *s / 127.0 };
    }
    let data = t
        .data
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            (v / scales[channel_of(&t.shape, axis, i)])
                .round_ties_even()
                .clamp(-127.0, 127.0) as i8
        })
        .collect();
    QTensor {
        name: t.name.clone(),
        shape: t.shape.clone(),
        data,
        scales,
    }
}

/// The f32 tensor an int8 tensor stands for, as the library computes it.
pub fn dequantize(q: &QTensor, axis: usize) -> F32Tensor {
    let data = q
        .data
        .iter()
        .enumerate()
        .map(|(i, &v)| f32::from(v) * q.scales[channel_of(&q.shape, axis, i)])
        .collect();
    F32Tensor {
        name: q.name.clone(),
        shape: q.shape.clone(),
        data,
    }
}

/// The run's weights quantized, and the same weights dequantized back to f32.
pub fn quantize_all(tensors: &[F32Tensor]) -> (Vec<QTensor>, Vec<F32Tensor>, Vec<F32Tensor>) {
    let mut q = Vec::new();
    let mut biases = Vec::new();
    let mut dequantized = Vec::new();
    for t in tensors {
        if is_quantized(&t.name) {
            let axis = channel_axis(&t.name);
            let qt = quantize_tensor(t, axis);
            dequantized.push(dequantize(&qt, axis));
            q.push(qt);
        } else {
            biases.push(t.clone());
            dequantized.push(t.clone());
        }
    }
    (q, biases, dequantized)
}

/// Writes int8 tensors with their `.scale` companions and f32 tensors as safetensors. The
/// file is written here rather than through the `safetensors` crate because the crate takes
/// metadata as a `HashMap` and serializes it in iteration order, which changes per process;
/// with sorted keys and name-ordered tensors the same run always gives the same bytes and
/// checksum.
pub fn write_safetensors(
    path: &Path,
    q: &[QTensor],
    f32s: &[F32Tensor],
    metadata: Option<BTreeMap<String, String>>,
) -> anyhow::Result<()> {
    let mut owned: Vec<(String, &str, Vec<usize>, Vec<u8>)> = Vec::new();
    for t in q {
        owned.push((
            t.name.clone(),
            "I8",
            t.shape.clone(),
            t.data.iter().map(|&x| x as u8).collect(),
        ));
        owned.push((
            format!("{}.scale", t.name),
            "F32",
            vec![t.scales.len()],
            t.scales.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ));
    }
    for t in f32s {
        owned.push((
            t.name.clone(),
            "F32",
            t.shape.clone(),
            t.data.iter().flat_map(|x| x.to_le_bytes()).collect(),
        ));
    }
    owned.sort_by(|a, b| a.0.cmp(&b.0));
    let mut header = serde_json::Map::new();
    if let Some(m) = metadata {
        header.insert("__metadata__".into(), serde_json::to_value(m)?);
    }
    let mut at = 0usize;
    for (name, dtype, shape, bytes) in &owned {
        header.insert(
            name.clone(),
            serde_json::json!({ "dtype": dtype, "shape": shape, "data_offsets": [at, at + bytes.len()] }),
        );
        at += bytes.len();
    }
    let mut header = serde_json::to_string(&header)?;
    // Padding to a multiple of 8 keeps the data section aligned, as the format recommends.
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let mut out = Vec::with_capacity(8 + header.len() + at);
    out.extend_from_slice(&(header.len() as u64).to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for (_, _, _, bytes) in &owned {
        out.extend_from_slice(bytes);
    }
    std::fs::write(path, out).with_context(|| format!("writing {}", path.display()))
}

/// Loads a run's best checkpoint, a parser or a detector as its config's task says.
pub fn load_best<B: Backend>(
    run_dir: &Path,
    cfg: &crate::config::Config,
    device: &B::Device,
) -> anyhow::Result<TaggerNet<B>> {
    let operator = crate::diagnostic_operator::require_deployable(run_dir, cfg)?;
    load_best_for_operator(run_dir, cfg, device, operator)
}

/// Load weights only after the caller has verified their diagnostic forward operator.
pub(crate) fn load_best_for_operator<B: Backend>(
    run_dir: &Path,
    cfg: &crate::config::Config,
    device: &B::Device,
    operator: crate::diagnostic_operator::ForwardOperator,
) -> anyhow::Result<TaggerNet<B>> {
    crate::diagnostic_operator::verify_binding(run_dir, cfg, operator)?;
    cfg.tagger_net_config()
        .init::<B>(device)
        .load_file(
            run_dir.join("best"),
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
            device,
        )
        .with_context(|| format!("loading {}", run_dir.join("best.mpk").display()))
}

/// F1 may drop by at most this much when the weights go to int8.
const MAX_F1_DROP: f64 = 0.002;
const GATE_VERSION: u64 = 2;

/// Remove approval before any operation that may change the run's effective artifacts.
pub(crate) fn invalidate_gate(run_dir: &Path) -> anyhow::Result<()> {
    let path = run_dir.join("quantize.json");
    match std::fs::remove_file(&path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e).with_context(|| format!("removing {}", path.display())),
    }
}

fn file_hash(path: &Path) -> anyhow::Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(crate::export::sha256_hex(&bytes))
}

fn artifact_hashes(run_dir: &Path, quantized: &Path) -> anyhow::Result<serde_json::Value> {
    Ok(serde_json::json!({
        "config_sha256": file_hash(&run_dir.join("config.toml"))?,
        "best_sha256": file_hash(&run_dir.join("best.mpk"))?,
        "input_snapshot_sha256": file_hash(&run_dir.join("input_snapshot.json"))?,
        "quantized_sha256": file_hash(quantized)?,
    }))
}

/// Fail closed on legacy gates and on any changed run artifact. Used for both network tasks.
pub(crate) fn verify_gate(run_dir: &Path) -> anyhow::Result<()> {
    reject_diagnostic_run(run_dir)?;
    let path = run_dir.join("quantize.json");
    let gate: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).with_context(|| {
            format!("reading {}; run `trainer quantize` first", path.display())
        })?)
        .with_context(|| format!("parsing {}; requantize this run", path.display()))?;
    anyhow::ensure!(
        gate["gate_version"].as_u64() == Some(GATE_VERSION),
        "{} has a legacy or unsupported quantization gate; requantize this run",
        run_dir.display()
    );
    anyhow::ensure!(
        gate["passed"] == true,
        "{} did not pass the quantization gate; run `trainer quantize` first",
        run_dir.display()
    );
    let current = artifact_hashes(run_dir, &run_dir.join("quantized.safetensors"))?;
    for key in [
        "config_sha256",
        "best_sha256",
        "input_snapshot_sha256",
        "quantized_sha256",
    ] {
        anyhow::ensure!(
            gate[key].as_str() == current[key].as_str(),
            "{} changed since quantization ({key}); requantize this run",
            run_dir.display()
        );
    }
    Ok(())
}

fn write_gate(
    run_dir: &Path,
    summary: &serde_json::Map<String, serde_json::Value>,
) -> anyhow::Result<()> {
    let pending = run_dir.join("quantize.pending.json");
    std::fs::write(&pending, serde_json::to_string_pretty(summary)? + "\n")
        .with_context(|| format!("writing {}", pending.display()))?;
    std::fs::rename(&pending, run_dir.join("quantize.json"))
        .with_context(|| format!("publishing {}", run_dir.join("quantize.json").display()))
}

fn publish_validated(
    run_dir: &Path,
    pending: &Path,
    initial_hashes: &serde_json::Value,
    summary: &mut serde_json::Map<String, serde_json::Value>,
) -> anyhow::Result<()> {
    invalidate_gate(run_dir)?;
    let hashes = artifact_hashes(run_dir, pending)?;
    anyhow::ensure!(
        hashes["config_sha256"] == initial_hashes["config_sha256"]
            && hashes["best_sha256"] == initial_hashes["best_sha256"]
            && hashes["input_snapshot_sha256"] == initial_hashes["input_snapshot_sha256"],
        "config, checkpoint, or input snapshot changed during quantization; requantize this run"
    );
    for key in [
        "config_sha256",
        "best_sha256",
        "input_snapshot_sha256",
        "quantized_sha256",
    ] {
        summary.insert(key.into(), hashes[key].clone());
    }
    std::fs::rename(pending, run_dir.join("quantized.safetensors"))?;
    write_gate(run_dir, summary)
}

fn reject_quantization(
    run_dir: &Path,
    pending: &Path,
    summary: &serde_json::Map<String, serde_json::Value>,
) -> anyhow::Result<()> {
    invalidate_gate(run_dir)?;
    std::fs::rename(pending, run_dir.join("quantized.rejected.safetensors"))?;
    write_gate(run_dir, summary)
}

/// Validation F1 of the f32 and the int8 model, with the summary keys to record them under:
/// component F1 and exact parses for the parser, macro exact span F1 for the detector.
fn validation_scores<B: Backend>(
    cfg: &crate::config::Config,
    model: &TaggerNet<B>,
    int8_model: &TaggerNet<B>,
    device: &B::Device,
) -> anyhow::Result<(f64, f64, serde_json::Map<String, serde_json::Value>)> {
    let fc = cfg.features.to_tessera();
    let processed = std::path::PathBuf::from(&cfg.data.processed);
    let mut fields = serde_json::Map::new();
    match cfg.task {
        crate::config::Task::Parser => {
            let valid: Vec<LabelledExample> =
                read_shard(&processed.join("valid.parquet"), Split::Valid)?
                    .into_iter()
                    .filter(|e| !e.augmented)
                    .collect();
            let (kept, items) = encode_examples(&valid, &fc);
            anyhow::ensure!(
                !valid.is_empty() && items.len() == valid.len(),
                "parser quantization validation has {} encoded rows out of {}; repair the validation labels",
                items.len(),
                valid.len()
            );
            let f32_score = quick_score(model, &kept, &items, cfg.train.batch_size, device);
            let int8_score = quick_score(int8_model, &kept, &items, cfg.train.batch_size, device);
            fields.insert(
                "valid_f32_component_f1".into(),
                f32_score.component_f1.into(),
            );
            fields.insert(
                "valid_int8_component_f1".into(),
                int8_score.component_f1.into(),
            );
            fields.insert("valid_f32_exact".into(), f32_score.exact.into());
            fields.insert("valid_int8_exact".into(), int8_score.exact.into());
            Ok((f32_score.component_f1, int8_score.component_f1, fields))
        }
        crate::config::Task::Detector => {
            let (docs, _) = detector::load_split(&processed, Split::Valid, &fc)?;
            anyhow::ensure!(
                !docs.is_empty(),
                "detector quantization validation has no encoded documents"
            );
            let gold: Vec<_> = docs.iter().map(|d| d.gold.clone()).collect();
            let score = |m: &TaggerNet<B>| -> anyhow::Result<_> {
                let raw = detector::predict_scored_with_operator(
                    m,
                    &docs,
                    cfg.train.batch_size,
                    device,
                    crate::diagnostic_operator::deployable(cfg),
                )?;
                Ok(detector::score(
                    &gold,
                    &detector::apply_confidence_policy(&raw),
                ))
            };
            let (f32_score, int8_score) = (score(model)?, score(int8_model)?);
            fields.insert(
                "valid_f32_macro_exact_f1".into(),
                f32_score.macro_exact_f1.into(),
            );
            fields.insert(
                "valid_int8_macro_exact_f1".into(),
                int8_score.macro_exact_f1.into(),
            );
            fields.insert("valid_int8".into(), serde_json::to_value(&int8_score)?);
            Ok((f32_score.macro_exact_f1, int8_score.macro_exact_f1, fields))
        }
    }
}

/// `trainer quantize`: quantizes the best checkpoint and fails if validation F1 drops by more
/// than the allowed margin.
pub fn run<B: Backend>(run_dir: &Path, device: &B::Device) -> anyhow::Result<()> {
    invalidate_gate(run_dir)?;
    reject_diagnostic_run(run_dir)?;
    crate::train::verify_input_snapshot(run_dir)?;
    let initial_hashes = serde_json::json!({
        "config_sha256": file_hash(&run_dir.join("config.toml"))?,
        "best_sha256": file_hash(&run_dir.join("best.mpk"))?,
        "input_snapshot_sha256": file_hash(&run_dir.join("input_snapshot.json"))?,
    });
    let cfg = crate::config::load(&run_dir.join("config.toml"))?;
    verify_learning_result(run_dir, &cfg)?;
    let net = cfg.net_name();
    let model = load_best::<B>(run_dir, &cfg, device)?;
    let tensors = extract(&model, net);
    let (q, biases, dequantized) = quantize_all(&tensors);
    let pending = run_dir.join("quantized.pending.safetensors");
    write_safetensors(&pending, &q, &biases, None)?;

    let max_err = tensors
        .iter()
        .zip(&dequantized)
        .flat_map(|(a, b)| a.data.iter().zip(&b.data).map(|(x, y)| (x - y).abs()))
        .fold(0f32, f32::max);
    let int8_model = inject::<B>(&cfg.tagger_net_config(), &dequantized, net, device)?;
    let (f32_f1, int8_f1, fields) = validation_scores(&cfg, &model, &int8_model, device)?;
    anyhow::ensure!(
        f32_f1.is_finite() && int8_f1.is_finite(),
        "quantization validation produced a non-finite score"
    );
    let file_bytes = std::fs::metadata(&pending)?.len();
    let drop = f32_f1 - int8_f1;
    let passed = drop <= MAX_F1_DROP;
    let mut summary = serde_json::Map::new();
    summary.insert("gate_version".into(), GATE_VERSION.into());
    summary.insert("passed".into(), passed.into());
    summary.extend(fields);
    summary.insert("max_abs_weight_error".into(), max_err.into());
    summary.insert("file_bytes".into(), file_bytes.into());
    println!(
        "{net} validation F1: f32 {f32_f1:.4}, int8 {int8_f1:.4}; max weight error {max_err:.2e}; {file_bytes} bytes"
    );
    if !passed {
        let rejected = run_dir.join("quantized.rejected.safetensors");
        reject_quantization(run_dir, &pending, &summary)?;
        bail!(
            "int8 weights lose {drop:.4} F1, over the {MAX_F1_DROP} limit; {} kept for inspection",
            rejected.display()
        );
    }
    crate::train::verify_input_snapshot(run_dir)?;
    publish_validated(run_dir, &pending, &initial_hashes, &mut summary)?;
    Ok(())
}

fn reject_diagnostic_run(run_dir: &Path) -> anyhow::Result<()> {
    anyhow::ensure!(
        !run_dir.join("diagnostic.json").try_exists()?
            && !run_dir
                .join(crate::diagnostic_operator::FILE)
                .try_exists()?,
        "bounded learning diagnostics cannot be quantized or exported"
    );
    let path = run_dir.join("summary.json");
    if path.try_exists()? {
        let summary: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
        anyhow::ensure!(
            summary["training_complete"] != false,
            "incomplete training cannot be quantized or exported"
        );
    }
    Ok(())
}

pub(crate) fn verify_learning_result(
    run_dir: &Path,
    cfg: &crate::config::Config,
) -> anyhow::Result<()> {
    if cfg
        .detector
        .as_ref()
        .is_some_and(|d| d.learning_check.is_some())
    {
        let summary: serde_json::Value = serde_json::from_slice(
            &std::fs::read(run_dir.join("summary.json"))
                .context("learning-checked training requires a completed summary")?,
        )?;
        anyhow::ensure!(
            summary["training_complete"] == true && summary["learning_check_passed"] == true,
            "training did not complete and pass its enforced learning check"
        );
        let proof: serde_json::Value = serde_json::from_slice(
            &std::fs::read(run_dir.join("best_learning_check.json"))
                .context("selected checkpoint has no learning check proof")?,
        )?;
        anyhow::ensure!(
            proof["passed"] == true
                && proof["checkpoint_sha256"].as_str()
                    == Some(file_hash(&run_dir.join("best.mpk"))?.as_str())
                && proof["config_sha256"].as_str()
                    == Some(file_hash(&run_dir.join("config.toml"))?.as_str()),
            "selected checkpoint or config differs from its passing learning check"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;

    use super::*;

    fn gate_run() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("config.toml"), b"config").unwrap();
        std::fs::write(dir.path().join("best.mpk"), b"checkpoint").unwrap();
        std::fs::write(dir.path().join("input_snapshot.json"), b"inputs").unwrap();
        std::fs::write(
            dir.path().join("quantized.safetensors"),
            b"previous weights",
        )
        .unwrap();
        dir
    }

    fn passing_summary() -> serde_json::Map<String, serde_json::Value> {
        serde_json::json!({"gate_version": GATE_VERSION, "passed": true, "file_bytes": 11})
            .as_object()
            .unwrap()
            .clone()
    }

    #[test]
    fn bounded_diagnostic_cannot_reuse_a_passing_quantization_gate() {
        let dir = gate_run();
        let pending = dir.path().join("quantized.pending.safetensors");
        std::fs::write(&pending, b"new weights").unwrap();
        let hashes = artifact_hashes(dir.path(), &pending).unwrap();
        publish_validated(dir.path(), &pending, &hashes, &mut passing_summary()).unwrap();
        verify_gate(dir.path()).unwrap();
        std::fs::write(dir.path().join("diagnostic.json"), r#"{"max_steps":2000}"#).unwrap();
        assert!(
            verify_gate(dir.path())
                .unwrap_err()
                .to_string()
                .contains("diagnostics")
        );
        assert!(
            run::<NdArray>(dir.path(), &Default::default())
                .unwrap_err()
                .to_string()
                .contains("diagnostics")
        );
        assert!(!dir.path().join("quantize.json").exists());
    }

    #[test]
    fn incomplete_and_failed_learning_results_cannot_ship() {
        let dir = gate_run();
        std::fs::write(
            dir.path().join("summary.json"),
            r#"{"training_complete":false}"#,
        )
        .unwrap();
        assert!(reject_diagnostic_run(dir.path()).is_err());
        let cfg = crate::config::learning_test_config();
        for value in [
            serde_json::json!({"training_complete":true,"learning_check_passed":false}),
            serde_json::json!({"training_complete":true}),
            serde_json::json!({"training_complete":false,"learning_check_passed":true}),
        ] {
            std::fs::write(dir.path().join("summary.json"), value.to_string()).unwrap();
            assert!(verify_learning_result(dir.path(), &cfg).is_err());
        }
        std::fs::write(
            dir.path().join("summary.json"),
            r#"{"training_complete":true,"learning_check_passed":true}"#,
        )
        .unwrap();
        assert!(verify_learning_result(dir.path(), &cfg).is_err());
        let proof = serde_json::json!({
            "passed": true,
            "checkpoint_sha256": file_hash(&dir.path().join("best.mpk")).unwrap(),
            "config_sha256": file_hash(&dir.path().join("config.toml")).unwrap(),
        });
        std::fs::write(
            dir.path().join("best_learning_check.json"),
            proof.to_string(),
        )
        .unwrap();
        verify_learning_result(dir.path(), &cfg).unwrap();
        std::fs::write(dir.path().join("best.mpk"), b"changed checkpoint").unwrap();
        assert!(verify_learning_result(dir.path(), &cfg).is_err());
    }

    #[test]
    fn validated_weights_publish_with_bound_gate() {
        let dir = gate_run();
        let pending = dir.path().join("quantized.pending.safetensors");
        std::fs::write(&pending, b"new weights").unwrap();
        let hashes = artifact_hashes(dir.path(), &pending).unwrap();
        let mut summary = passing_summary();
        publish_validated(dir.path(), &pending, &hashes, &mut summary).unwrap();
        assert!(!pending.exists());
        assert_eq!(
            std::fs::read(dir.path().join("quantized.safetensors")).unwrap(),
            b"new weights"
        );
        verify_gate(dir.path()).unwrap();
    }

    #[test]
    fn rejected_weights_cannot_keep_a_previous_passing_gate() {
        let dir = gate_run();
        std::fs::write(dir.path().join("quantize.json"), r#"{"passed":true}"#).unwrap();
        let pending = dir.path().join("quantized.pending.safetensors");
        std::fs::write(&pending, b"bad weights").unwrap();
        let mut summary = passing_summary();
        summary.insert("passed".into(), false.into());
        reject_quantization(dir.path(), &pending, &summary).unwrap();
        assert_eq!(
            std::fs::read(dir.path().join("quantized.rejected.safetensors")).unwrap(),
            b"bad weights"
        );
        assert!(
            verify_gate(dir.path())
                .unwrap_err()
                .to_string()
                .contains("did not pass")
        );
    }

    #[test]
    fn failure_after_weight_publish_leaves_no_gate() {
        let dir = gate_run();
        let pending = dir.path().join("quantized.pending.safetensors");
        std::fs::write(&pending, b"new weights").unwrap();
        let hashes = artifact_hashes(dir.path(), &pending).unwrap();
        std::fs::write(dir.path().join("quantize.json"), r#"{"passed":true}"#).unwrap();
        // Force the final gate write to fail after the weights have been renamed.
        std::fs::create_dir(dir.path().join("quantize.pending.json")).unwrap();
        let err =
            publish_validated(dir.path(), &pending, &hashes, &mut passing_summary()).unwrap_err();
        assert!(err.to_string().contains("writing"));
        assert!(!dir.path().join("quantize.json").exists());
        assert!(!pending.exists());
        assert!(verify_gate(dir.path()).is_err());
    }

    #[test]
    fn checkpoint_changed_during_quantization_cannot_publish() {
        let dir = gate_run();
        let pending = dir.path().join("quantized.pending.safetensors");
        std::fs::write(&pending, b"new weights").unwrap();
        let hashes = artifact_hashes(dir.path(), &pending).unwrap();
        std::fs::write(dir.path().join("best.mpk"), b"changed checkpoint").unwrap();
        assert!(publish_validated(dir.path(), &pending, &hashes, &mut passing_summary()).is_err());
        assert!(!dir.path().join("quantize.json").exists());
        assert!(pending.exists());
    }

    #[test]
    fn input_snapshot_changed_during_quantization_cannot_publish() {
        let dir = gate_run();
        let pending = dir.path().join("quantized.pending.safetensors");
        std::fs::write(&pending, b"new weights").unwrap();
        let hashes = artifact_hashes(dir.path(), &pending).unwrap();
        std::fs::write(dir.path().join("input_snapshot.json"), b"changed inputs").unwrap();
        let error =
            publish_validated(dir.path(), &pending, &hashes, &mut passing_summary()).unwrap_err();
        assert!(error.to_string().contains("input snapshot"), "{error:#}");
        assert!(!dir.path().join("quantize.json").exists());
    }

    fn t(shape: Vec<usize>, data: Vec<f32>) -> F32Tensor {
        F32Tensor {
            name: "w".into(),
            shape,
            data,
        }
    }

    #[test]
    fn per_channel_scales_and_rounding() {
        let q = quantize_tensor(&t(vec![2, 3], vec![1.0, -2.0, 4.0, 0.5, 0.25, 0.0]), 0);
        assert_eq!(q.scales, vec![4.0 / 127.0, 0.5 / 127.0]);
        assert_eq!(q.data, vec![32, -64, 127, 127, 64, 0]);
    }

    #[test]
    fn zero_channel_gets_unit_scale() {
        let q = quantize_tensor(&t(vec![2, 2], vec![0.0, 0.0, 1.0, -1.0]), 0);
        assert_eq!(q.scales[0], 1.0);
        assert_eq!(&q.data[..2], &[0, 0]);
    }

    #[test]
    fn round_trip_error_is_at_most_half_a_step() {
        let data: Vec<f32> = (0..60)
            .map(|i| ((i * 37 % 23) as f32 - 11.0) / 7.0)
            .collect();
        let orig = t(vec![4, 15], data);
        let q = quantize_tensor(&orig, 0);
        let back = dequantize(&q, 0);
        for (i, (a, b)) in orig.data.iter().zip(&back.data).enumerate() {
            assert!((a - b).abs() <= q.scales[i / 15] / 2.0 + 1e-7);
        }
    }

    #[test]
    fn embeddings_are_quantized_per_column() {
        let q = quantize_tensor(&t(vec![3, 2], vec![1.0, 10.0, -2.0, 5.0, 0.5, -20.0]), 1);
        assert_eq!(q.scales, vec![2.0 / 127.0, 20.0 / 127.0]);
    }

    #[test]
    fn extract_and_inject_round_trip() {
        let device = Default::default();
        let cfg = TaggerNetConfig::new(64, vec![1, 2, 4, 8], crate::dataset::PARSER_LABELS);
        let model = cfg.init::<NdArray>(&device);
        let tensors = extract(&model, "parser");
        assert_eq!(tensors.len(), 15);
        assert_eq!(
            tensors
                .iter()
                .find(|t| t.name == "parser.proj.weight")
                .unwrap()
                .shape,
            vec![96, 87]
        );
        assert_eq!(
            tensors
                .iter()
                .find(|t| t.name == "parser.head.weight")
                .unwrap()
                .shape,
            vec![23, 96]
        );
        let back = inject::<NdArray>(&cfg, &tensors, "parser", &device).unwrap();
        assert_eq!(extract(&back, "parser"), tensors);
    }
}
