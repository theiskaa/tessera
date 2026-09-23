//! `trainer train`: a manual training loop for the parser with masked, class-weighted
//! cross-entropy, AdamW, warmup then cosine decay, early stopping on validation component F1,
//! checkpoints, and a `metrics.jsonl` log.
//!
//! A manual loop rather than Burn's `Learner`: the loop is short, it validates on decoded
//! spans instead of loss, and the detector in Milestone 4 reuses it unchanged.

use std::fs::File;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::Context;
use burn::backend::{Autodiff, NdArray, Wgpu};
use burn::data::dataloader::batcher::Batcher;
use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::TensorData;
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::AutodiffBackend;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

use crate::data::{LabelledExample, Split, read_shard};
use crate::dataset::{Encoded, PARSER_LABELS, ParserBatch, ParserBatcher, ParserDataset};
use crate::model_eval::{encode_examples, quick_score};
use crate::net;

/// Which Burn backend trains the model.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum BackendKind {
    /// The GPU through wgpu (Metal on macOS).
    Wgpu,
    /// The CPU.
    Ndarray,
}

/// Masked, class-weighted cross-entropy: `sum(w[y] * -log p[y] * mask) / sum(w[y] * mask)`.
pub fn parser_loss<B: Backend>(
    logits: Tensor<B, 3>,
    labels: Tensor<B, 2, Int>,
    mask: Tensor<B, 2>,
    class_weights: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let [b, l, c] = logits.dims();
    let logp = log_softmax(logits.reshape([b * l, c]), 1);
    let idx = labels.reshape([b * l, 1]);
    let nll = logp.gather(1, idx.clone()).neg().reshape([b * l]);
    let w = class_weights.gather(0, idx.reshape([b * l]));
    let m = mask.reshape([b * l]);
    let num = (nll * w.clone() * m.clone()).sum();
    let den = (w * m).sum().clamp_min(1.0);
    num / den
}

/// `min(5, sqrt(count(O) / count(label)))`, so rare labels are not drowned out by `O`.
pub fn class_weights(items: &[Encoded]) -> [f32; PARSER_LABELS] {
    let mut counts = [0f64; PARSER_LABELS];
    for e in items {
        for &l in &e.labels {
            counts[l as usize] += 1.0;
        }
    }
    let o = counts[0].max(1.0);
    let mut w = [1.0f32; PARSER_LABELS];
    for (i, weight) in w.iter_mut().enumerate().skip(1) {
        *weight = ((o / counts[i].max(1.0)).sqrt() as f32).min(5.0);
    }
    w
}

/// Linear warmup to `base`, then cosine decay to `base / 100` at `total`.
pub fn lr_at(step: usize, total: usize, base: f64, warmup: usize) -> f64 {
    if step < warmup {
        return base * (step as f64 + 1.0) / warmup as f64;
    }
    let t = (step - warmup) as f64 / (total.saturating_sub(warmup).max(1)) as f64;
    let floor = base / 100.0;
    floor + (base - floor) * 0.5 * (1.0 + (std::f64::consts::PI * t.min(1.0)).cos())
}

/// Overrides from the command line.
pub struct TrainArgs<'a> {
    /// The run's config file.
    pub config: &'a Path,
    /// Run name overriding the config's.
    pub name: Option<&'a str>,
    /// Keep the first N original rows per country, for learning curves.
    pub train_per_country: Option<usize>,
    /// Maximum epochs overriding the config's.
    pub epochs: Option<usize>,
    /// Where to train.
    pub backend: BackendKind,
}

/// `trainer train`: trains the parser from a config on the chosen backend.
pub fn run(args: TrainArgs<'_>) -> anyhow::Result<()> {
    match args.backend {
        BackendKind::Wgpu => train::<Autodiff<Wgpu>>(&args, &Default::default()),
        BackendKind::Ndarray => train::<Autodiff<NdArray>>(&args, &Default::default()),
    }
}

fn train<B: AutodiffBackend>(args: &TrainArgs<'_>, device: &B::Device) -> anyhow::Result<()> {
    let mut cfg = crate::config::load(args.config)?;
    anyhow::ensure!(
        cfg.task == crate::config::Task::Parser,
        "{} is a {:?} config; only the parser trains before milestone 4",
        args.config.display(),
        cfg.task
    );
    if let Some(name) = args.name {
        cfg.name = name.to_string();
    }
    if let Some(e) = args.epochs {
        cfg.train.epochs = e;
    }
    let run_dir = PathBuf::from("runs").join(&cfg.name);
    std::fs::create_dir_all(run_dir.join("checkpoints"))?;
    // The effective config, overrides applied, so every later step reads what was trained.
    std::fs::write(run_dir.join("config.toml"), toml::to_string(&cfg)?)
        .context("writing the run config")?;
    B::seed(device, cfg.seed);

    let sample: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&cfg.data.sample_manifest)
            .with_context(|| format!("reading {}", cfg.data.sample_manifest))?,
    )?;
    anyhow::ensure!(
        sample["filters"]["countries"] == serde_json::json!(cfg.data.countries),
        "{} was prepared for {} but the config trains {:?}; run prepare again",
        cfg.data.sample_manifest,
        sample["filters"]["countries"],
        cfg.data.countries
    );
    anyhow::ensure!(
        sample["checks"]["passed"].as_bool() == Some(true),
        "{} has no passing data checks; run prepare again",
        cfg.data.sample_manifest
    );
    anyhow::ensure!(
        sample["augment_copies"].as_u64() == Some(cfg.augment.copies as u64),
        "{} was prepared with augment_copies {}, but the config says {}; copies would map to \
         the wrong originals",
        cfg.data.sample_manifest,
        sample["augment_copies"],
        cfg.augment.copies
    );
    let fc = cfg.features.to_tessera();
    let processed = PathBuf::from(&cfg.data.processed);
    let train_rows = read_shard(&processed.join("train.parquet"), Split::Train)?;
    let (train_ds, train_stats) =
        ParserDataset::from_rows(&train_rows, &fc, args.train_per_country, cfg.augment.copies);
    let valid_rows: Vec<LabelledExample> =
        read_shard(&processed.join("valid.parquet"), Split::Valid)?
            .into_iter()
            .filter(|e| !e.augmented)
            .collect();
    let (valid_kept, valid_items) = encode_examples(&valid_rows, &fc);
    eprintln!(
        "train rows {} ({} dropped: a span cuts a token), valid rows {}",
        train_stats.encoded,
        train_stats.cuts_token,
        valid_items.len()
    );

    let weights = class_weights(&train_ds.items);
    serde_json::to_writer_pretty(
        File::create(run_dir.join("class_weights.json"))?,
        &weights.to_vec(),
    )?;
    let class_weights =
        Tensor::<B, 1>::from_data(TensorData::new(weights.to_vec(), [PARSER_LABELS]), device);

    let mut model = cfg.parser_net_config().init::<B>(device);
    let (dense, embedding) =
        net::assert_size(&model, cfg.net.max_params, cfg.net.max_embedding_bytes)?;
    eprintln!("parameters: {dense} dense, {embedding} in the n-gram table");
    let mut optim = AdamWConfig::new()
        .with_weight_decay(0.01)
        .init::<B, net::ParserNet<B>>();
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();

    let batch_size = cfg.train.batch_size.max(1);
    let steps_per_epoch = train_ds.items.len().div_ceil(batch_size);
    let total_steps = cfg.train.epochs * steps_per_epoch;
    let mut metrics = File::create(run_dir.join("metrics.jsonl"))?;
    let (mut best_f1, mut best_exact, mut best_epoch, mut since_best, mut step) =
        (f64::MIN, 0f64, 0usize, 0usize, 0usize);
    let started = std::time::Instant::now();
    let mut order: Vec<usize> = (0..train_ds.items.len()).collect();

    for epoch in 1..=cfg.train.epochs {
        order.shuffle(&mut ChaCha8Rng::seed_from_u64(cfg.seed ^ epoch as u64));
        let (mut epoch_loss, mut batches) = (0f64, 0usize);
        for chunk in order.chunks(batch_size) {
            let items: Vec<Encoded> = chunk.iter().map(|&i| train_ds.items[i].clone()).collect();
            let batch: ParserBatch<B> = ParserBatcher.batch(items, device);
            let logits = model.forward(
                batch.ngram_ids,
                batch.script,
                batch.shape,
                batch.flags,
                batch.mask.clone(),
            );
            let loss = parser_loss(logits, batch.labels, batch.mask, class_weights.clone());
            let value: f64 = loss.clone().into_scalar().elem();
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            let lr = lr_at(
                step,
                total_steps,
                cfg.train.learning_rate,
                cfg.train.warmup_steps,
            );
            model = optim.step(lr, model, grads);
            epoch_loss += value;
            batches += 1;
            step += 1;
            if step % 200 == 0 {
                eprintln!(
                    "epoch {epoch} step {step}/{total_steps} loss {:.4} lr {lr:.2e}",
                    epoch_loss / batches as f64
                );
            }
        }
        let valid = quick_score(
            &model.valid(),
            &valid_kept,
            &valid_items,
            batch_size,
            device,
        );
        writeln!(
            metrics,
            r#"{{"epoch":{epoch},"train_loss":{:.6},"valid_component_f1":{:.5},"valid_exact":{:.5},"lr":{:.3e}}}"#,
            epoch_loss / batches.max(1) as f64,
            valid.component_f1,
            valid.exact,
            lr_at(
                step.saturating_sub(1),
                total_steps,
                cfg.train.learning_rate,
                cfg.train.warmup_steps
            )
        )?;
        eprintln!(
            "epoch {epoch}: loss {:.4}, valid component F1 {:.4}, exact {:.4}",
            epoch_loss / batches.max(1) as f64,
            valid.component_f1,
            valid.exact
        );
        model.clone().save_file(
            run_dir.join(format!("checkpoints/epoch-{epoch}")),
            &recorder,
        )?;
        if valid.component_f1 > best_f1 {
            best_f1 = valid.component_f1;
            best_exact = valid.exact;
            best_epoch = epoch;
            since_best = 0;
            model.clone().save_file(run_dir.join("best"), &recorder)?;
        } else {
            since_best += 1;
            if since_best >= cfg.train.patience {
                eprintln!("early stop at epoch {epoch}; best epoch {best_epoch}, F1 {best_f1:.4}");
                break;
            }
        }
    }
    let summary = serde_json::json!({
        "best_epoch": best_epoch,
        "best_valid_component_f1": best_f1,
        "best_valid_exact": best_exact,
        "wall_clock_seconds": started.elapsed().as_secs(),
        "train_rows": train_stats.encoded,
        "train_per_country": args.train_per_country,
        "originals_per_country": train_stats.originals,
        "epochs": cfg.train.epochs,
        "steps": step,
        "dense_parameters": dense,
        "embedding_parameters": embedding,
    });
    std::fs::write(
        run_dir.join("summary.json"),
        serde_json::to_string_pretty(&summary)? + "\n",
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_warms_up_then_decays() {
        let lrs: Vec<f64> = (0..1000).map(|s| lr_at(s, 1000, 1e-3, 200)).collect();
        assert!(lrs[..200].windows(2).all(|w| w[1] >= w[0]));
        assert!(lrs[200..].windows(2).all(|w| w[1] <= w[0] + 1e-15));
        assert!((lrs[199] - 1e-3).abs() < 1e-12);
    }

    #[test]
    fn loss_is_zero_when_certain_and_ln_c_when_uniform() {
        type B = NdArray;
        let device = Default::default();
        let labels =
            Tensor::<B, 2, Int>::from_data(TensorData::new(vec![1i64, 3], [1, 2]), &device);
        let mask = Tensor::<B, 2>::ones([1, 2], &device);
        let weights = Tensor::<B, 1>::ones([PARSER_LABELS], &device);
        let mut certain = vec![-50f32; 2 * PARSER_LABELS];
        certain[1] = 50.0;
        certain[PARSER_LABELS + 3] = 50.0;
        let certain =
            Tensor::<B, 3>::from_data(TensorData::new(certain, [1, 2, PARSER_LABELS]), &device);
        let loss: f32 = parser_loss(certain, labels.clone(), mask.clone(), weights.clone())
            .into_scalar()
            .elem();
        assert!(loss < 1e-3, "{loss}");
        let uniform = Tensor::<B, 3>::zeros([1, 2, PARSER_LABELS], &device);
        let loss: f32 = parser_loss(uniform, labels, mask, weights)
            .into_scalar()
            .elem();
        assert!((loss - (PARSER_LABELS as f32).ln()).abs() < 1e-4, "{loss}");
    }

    #[test]
    fn class_weights_cap_and_leave_o_alone() {
        let e = Encoded {
            token_spans: vec![(0, 1); 101],
            ngram_ids: vec![Vec::new(); 101],
            script: vec![0; 101],
            shape: vec![0; 101],
            flags: vec![0; 101],
            labels: std::iter::once(3)
                .chain(std::iter::repeat_n(0, 100))
                .collect(),
            country: "GB".into(),
        };
        let w = class_weights(&[e]);
        assert_eq!(w[0], 1.0);
        assert_eq!(w[3], 5.0);
    }
}
