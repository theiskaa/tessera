//! `trainer quantize`: per-output-channel symmetric int8 weights with f32 scales, biases kept
//! in f32, written as `runs/<name>/quantized.safetensors` under the shipping tensor names,
//! then re-scored on the validation split; the run fails if component F1 drops by more than
//! `MAX_F1_DROP`.
//!
//! For a weight whose first dimension indexes output channels:
//! `scale[c] = max(|W[c, ..]|) / 127`, `q = round_half_to_even(W / scale[c])` clamped to
//! `[-127, 127]`, and an all-zero channel gets `scale = 1`. Embedding tables are quantized per
//! column instead, since the embedding dimension is what a lookup outputs. The library's
//! kernels reproduce exactly this.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::{Context, bail};
use burn::module::Param;
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::TensorData;

use crate::data::{LabelledExample, Split, read_shard};
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

/// Every parameter of the parser under its shipping name. Linear weights are transposed from
/// Burn's `[in, out]` to `[out, in]`; conv weights keep `[out, in, kernel]`.
pub fn extract<B: Backend>(model: &TaggerNet<B>) -> Vec<F32Tensor> {
    let mut out = Vec::new();
    let mut push = |name: String, (shape, data): (Vec<usize>, Vec<f32>)| {
        out.push(F32Tensor { name, shape, data })
    };
    push(
        "parser.embed.ngram".into(),
        to_vec(model.ngram.weight.val()),
    );
    push(
        "parser.embed.script".into(),
        to_vec(model.script.weight.val()),
    );
    push(
        "parser.embed.shape".into(),
        to_vec(model.shape.weight.val()),
    );
    push(
        "parser.proj.weight".into(),
        to_vec(model.proj.weight.val().transpose()),
    );
    if let Some(b) = &model.proj.bias {
        push("parser.proj.bias".into(), to_vec(b.val()));
    }
    for (i, block) in model.blocks.iter().enumerate() {
        push(
            format!("parser.block{i}.conv.weight"),
            to_vec(block.conv.weight.val()),
        );
        if let Some(b) = &block.conv.bias {
            push(format!("parser.block{i}.conv.bias"), to_vec(b.val()));
        }
    }
    push(
        "parser.head.weight".into(),
        to_vec(model.head.weight.val().transpose()),
    );
    if let Some(b) = &model.head.bias {
        push("parser.head.bias".into(), to_vec(b.val()));
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

/// A model whose parameters are `tensors`, given in the shipping names and layouts.
pub fn inject<B: Backend>(
    cfg: &TaggerNetConfig,
    tensors: &[F32Tensor],
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
    model.ngram.weight = Param::from_tensor(tensor(get("parser.embed.ngram")?, device)?);
    model.script.weight = Param::from_tensor(tensor(get("parser.embed.script")?, device)?);
    model.shape.weight = Param::from_tensor(tensor(get("parser.embed.shape")?, device)?);
    model.proj.weight =
        Param::from_tensor(tensor::<B, 2>(get("parser.proj.weight")?, device)?.transpose());
    model.proj.bias = Some(Param::from_tensor(tensor(
        get("parser.proj.bias")?,
        device,
    )?));
    for (i, block) in model.blocks.iter_mut().enumerate() {
        block.conv.weight = Param::from_tensor(tensor(
            get(&format!("parser.block{i}.conv.weight"))?,
            device,
        )?);
        block.conv.bias = Some(Param::from_tensor(tensor(
            get(&format!("parser.block{i}.conv.bias"))?,
            device,
        )?));
    }
    model.head.weight =
        Param::from_tensor(tensor::<B, 2>(get("parser.head.weight")?, device)?.transpose());
    model.head.bias = Some(Param::from_tensor(tensor(
        get("parser.head.bias")?,
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

/// Loads a run's best checkpoint.
pub fn load_best<B: Backend>(
    run_dir: &Path,
    cfg: &crate::config::Config,
    device: &B::Device,
) -> anyhow::Result<TaggerNet<B>> {
    cfg.parser_net_config()
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

/// `trainer quantize`: quantizes the best checkpoint and fails if validation F1 drops by more
/// than the allowed margin.
pub fn run<B: Backend>(run_dir: &Path, device: &B::Device) -> anyhow::Result<()> {
    let cfg = crate::config::load(&run_dir.join("config.toml"))?;
    let model = load_best::<B>(run_dir, &cfg, device)?;
    let tensors = extract(&model);
    let (q, biases, dequantized) = quantize_all(&tensors);
    let path = run_dir.join("quantized.safetensors");
    write_safetensors(&path, &q, &biases, None)?;

    let max_err = tensors
        .iter()
        .zip(&dequantized)
        .flat_map(|(a, b)| a.data.iter().zip(&b.data).map(|(x, y)| (x - y).abs()))
        .fold(0f32, f32::max);
    let int8_model = inject::<B>(&cfg.parser_net_config(), &dequantized, device)?;
    let fc = cfg.features.to_tessera();
    let valid: Vec<LabelledExample> = read_shard(
        &std::path::PathBuf::from(&cfg.data.processed).join("valid.parquet"),
        Split::Valid,
    )?
    .into_iter()
    .filter(|e| !e.augmented)
    .collect();
    let (kept, items) = encode_examples(&valid, &fc);
    let f32_score = quick_score(&model, &kept, &items, cfg.train.batch_size, device);
    let int8_score = quick_score(&int8_model, &kept, &items, cfg.train.batch_size, device);
    let file_bytes = std::fs::metadata(&path)?.len();
    let drop = f32_score.component_f1 - int8_score.component_f1;
    let passed = drop <= MAX_F1_DROP;
    let summary = serde_json::json!({
        "passed": passed,
        "valid_f32_component_f1": f32_score.component_f1,
        "valid_int8_component_f1": int8_score.component_f1,
        "valid_f32_exact": f32_score.exact,
        "valid_int8_exact": int8_score.exact,
        "max_abs_weight_error": max_err,
        "file_bytes": file_bytes,
    });
    std::fs::write(
        run_dir.join("quantize.json"),
        serde_json::to_string_pretty(&summary)? + "\n",
    )?;
    println!(
        "valid component F1: f32 {:.4}, int8 {:.4}; max weight error {max_err:.2e}; {} bytes",
        f32_score.component_f1, int8_score.component_f1, file_bytes
    );
    if !passed {
        // Renamed so that `export`, which reads `quantized.safetensors`, cannot ship it.
        let rejected = run_dir.join("quantized.rejected.safetensors");
        std::fs::rename(&path, &rejected)?;
        bail!(
            "int8 weights lose {drop:.4} component F1, over the {MAX_F1_DROP} limit; {} kept for inspection",
            rejected.display()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;

    use super::*;

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
        let tensors = extract(&model);
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
        let back = inject::<NdArray>(&cfg, &tensors, &device).unwrap();
        assert_eq!(extract(&back), tensors);
    }
}
