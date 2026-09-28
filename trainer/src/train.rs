//! `trainer train`: one manual training loop for the parser and the detector, with masked,
//! class-weighted cross-entropy, AdamW, warmup then cosine decay, early stopping on a
//! validation score (component F1 for the parser, macro exact span F1 for the detector),
//! checkpoints, and a `metrics.jsonl` log.
//!
//! A manual loop rather than Burn's `Learner`: the loop is short and it validates on decoded
//! spans instead of loss.

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
use serde::{Deserialize, Serialize};
use tessera::internal::{DECODER_CONTRACT, TOKENIZER_CONTRACT};

use crate::config::Config;
use crate::data::{LabelledExample, Split, read_shard};
use crate::dataset::{Encoded, PARSER_LABELS, ParserBatch, ParserBatcher, ParserDataset};
use crate::detector::{self, DETECTOR_LABELS, KindSpan};
use crate::model_eval::{encode_examples, quick_score};
use crate::net::{self, TaggerNet};

/// Which Burn backend trains the model.
#[derive(Debug, Clone, Copy, clap::ValueEnum)]
pub enum BackendKind {
    /// The GPU through wgpu (Metal on macOS).
    Wgpu,
    /// The CPU.
    Ndarray,
}

/// Masked, class-weighted cross-entropy over any label count:
/// `sum(w[y] * -log p[y] * mask) / sum(w[y] * mask)`.
pub fn masked_loss<B: Backend>(
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

/// `trainer train`: trains the parser or the detector, as the config's `task` says.
pub fn run(args: TrainArgs<'_>) -> anyhow::Result<()> {
    match args.backend {
        BackendKind::Wgpu => train::<Autodiff<Wgpu>>(&args, &Default::default()),
        BackendKind::Ndarray => train::<Autodiff<NdArray>>(&args, &Default::default()),
    }
}

fn train<B: AutodiffBackend>(args: &TrainArgs<'_>, device: &B::Device) -> anyhow::Result<()> {
    let mut cfg = crate::config::load(args.config)?;
    if let Some(name) = args.name {
        cfg.name = name.to_string();
    }
    if let Some(e) = args.epochs {
        cfg.train.epochs = e;
    }
    validate_ngram_source(&cfg)?;
    validate_training_scope(&cfg)?;
    let run_dir = PathBuf::from("runs").join(&cfg.name);
    initialize_run(&run_dir, &cfg)?;
    capture_input_snapshot(&cfg, &run_dir)?;
    B::seed(device, cfg.seed);
    let result = match cfg.task {
        crate::config::Task::Parser => train_parser::<B>(args, &cfg, &run_dir, device),
        crate::config::Task::Detector => train_detector::<B>(&cfg, &run_dir, device),
    };
    result?;
    verify_input_snapshot(&run_dir)
}

fn validate_training_scope(cfg: &Config) -> anyhow::Result<()> {
    use polars::prelude::{ParquetReader, SerReader};

    anyhow::ensure!(
        cfg.data.countries.len() == 1 && cfg.data.countries[0] == "US",
        "training config must declare only US data"
    );
    for split in ["train", "valid", "test"] {
        let path = Path::new(&cfg.data.processed).join(format!("{split}.parquet"));
        let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
        let frame = ParquetReader::new(file)
            .with_columns(Some(vec!["country".to_string()]))
            .finish()
            .with_context(|| format!("reading countries from {}", path.display()))?;
        let countries = frame.column("country")?.str()?;
        anyhow::ensure!(!countries.is_empty(), "{} has no rows", path.display());
        for (index, country) in countries.iter().enumerate() {
            anyhow::ensure!(
                country == Some("US"),
                "{} row {} has country {:?}; prepare US-only data before training",
                path.display(),
                index + 1,
                country
            );
        }
    }
    Ok(())
}

const INPUT_SNAPSHOT_VERSION: u64 = 2;

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct InputFile {
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct InputSnapshot {
    pub version: u64,
    pub config_sha256: String,
    #[serde(default)]
    pub tokenizer_contract: String,
    #[serde(default)]
    pub decoder_contract: String,
    pub inputs: std::collections::BTreeMap<String, InputFile>,
}

fn hash_file(path: &Path) -> anyhow::Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(crate::export::sha256_hex(&bytes))
}

fn validate_ngram_source(cfg: &Config) -> anyhow::Result<()> {
    if !matches!(cfg.task, crate::config::Task::Detector) {
        return Ok(());
    }
    let Some(from) = &cfg.net.ngram_from else {
        return Ok(());
    };
    let run = Path::new(from);
    let source = crate::config::load(&run.join("config.toml"))?;
    anyhow::ensure!(
        matches!(source.task, crate::config::Task::Parser),
        "{} is not a parser n-gram source",
        run.display()
    );
    anyhow::ensure!(
        source.features == cfg.features,
        "{} has other feature settings than this config",
        run.display()
    );
    verify_input_snapshot(run)
        .with_context(|| format!("checking parser n-gram source {}", run.display()))
}

pub(crate) fn input_paths(cfg: &Config) -> Vec<(String, PathBuf)> {
    let processed = Path::new(&cfg.data.processed);
    let mut paths = vec![
        ("train_shard".to_string(), processed.join("train.parquet")),
        ("valid_shard".to_string(), processed.join("valid.parquet")),
    ];
    match cfg.task {
        crate::config::Task::Parser => paths.push((
            "parser_sample_manifest".to_string(),
            PathBuf::from(&cfg.data.sample_manifest),
        )),
        crate::config::Task::Detector => {
            if let Some(detector) = &cfg.detector {
                paths.extend(
                    detector
                        .silver
                        .iter()
                        .enumerate()
                        .map(|(i, path)| (format!("silver_{i:04}"), PathBuf::from(path))),
                );
            }
            if let Some(from) = &cfg.net.ngram_from {
                paths.push((
                    "ngram_source_config".to_string(),
                    Path::new(from).join("config.toml"),
                ));
                paths.push((
                    "ngram_source_best".to_string(),
                    Path::new(from).join("best.mpk"),
                ));
                paths.push((
                    "ngram_source_input_snapshot".to_string(),
                    Path::new(from).join("input_snapshot.json"),
                ));
            }
        }
    }
    paths
}

pub(crate) fn capture_input_snapshot(cfg: &Config, run_dir: &Path) -> anyhow::Result<()> {
    let inputs = input_paths(cfg)
        .into_iter()
        .map(|(role, path)| {
            Ok((
                role,
                InputFile {
                    path: path.display().to_string(),
                    sha256: hash_file(&path)?,
                },
            ))
        })
        .collect::<anyhow::Result<std::collections::BTreeMap<_, _>>>()?;
    let snapshot = InputSnapshot {
        version: INPUT_SNAPSHOT_VERSION,
        config_sha256: hash_file(&run_dir.join("config.toml"))?,
        tokenizer_contract: TOKENIZER_CONTRACT.to_string(),
        decoder_contract: DECODER_CONTRACT.to_string(),
        inputs,
    };
    std::fs::write(
        run_dir.join("input_snapshot.json"),
        serde_json::to_string_pretty(&snapshot)? + "\n",
    )?;
    Ok(())
}

pub(crate) fn load_input_snapshot(run_dir: &Path) -> anyhow::Result<InputSnapshot> {
    let path = run_dir.join("input_snapshot.json");
    let snapshot: InputSnapshot =
        serde_json::from_slice(&std::fs::read(&path).with_context(|| {
            format!(
                "{} has no run-bound input snapshot; retrain before export",
                run_dir.display()
            )
        })?)
        .with_context(|| format!("parsing {}", path.display()))?;
    anyhow::ensure!(
        snapshot.version == INPUT_SNAPSHOT_VERSION && !snapshot.inputs.is_empty(),
        "{} has an unsupported or empty input snapshot; retrain before export",
        run_dir.display()
    );
    anyhow::ensure!(
        snapshot.tokenizer_contract == TOKENIZER_CONTRACT
            && snapshot.decoder_contract == DECODER_CONTRACT,
        "{} tokenizer or decoder contract differs from the training snapshot; retrain before export",
        run_dir.display()
    );
    anyhow::ensure!(
        snapshot.config_sha256 == hash_file(&run_dir.join("config.toml"))?,
        "{} config changed after input snapshot; retrain before export",
        run_dir.display()
    );
    let cfg = crate::config::load(&run_dir.join("config.toml"))?;
    let expected: std::collections::BTreeMap<_, _> = input_paths(&cfg)
        .into_iter()
        .map(|(role, path)| (role, path.display().to_string()))
        .collect();
    let actual: std::collections::BTreeMap<_, _> = snapshot
        .inputs
        .iter()
        .map(|(role, input)| (role.clone(), input.path.clone()))
        .collect();
    anyhow::ensure!(
        actual == expected,
        "{} input snapshot omits or changes a configured training source; retrain before export",
        run_dir.display()
    );
    validate_ngram_source(&cfg)?;
    Ok(snapshot)
}

pub(crate) fn verify_input_snapshot(run_dir: &Path) -> anyhow::Result<()> {
    let snapshot = load_input_snapshot(run_dir)?;
    for (role, input) in &snapshot.inputs {
        anyhow::ensure!(
            input.sha256 == hash_file(Path::new(&input.path))?,
            "{role} ({}) changed during training; discard this run",
            input.path
        );
    }
    Ok(())
}

fn initialize_run(run_dir: &Path, cfg: &Config) -> anyhow::Result<()> {
    let best = run_dir.join("best.mpk");
    let has_best = match std::fs::symlink_metadata(&best) {
        Ok(_) => true,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => false,
        Err(e) => return Err(e).with_context(|| format!("checking {}", best.display())),
    };
    let checkpoints = run_dir.join("checkpoints");
    let has_checkpoint = if checkpoints.exists() {
        let mut found = false;
        for entry in std::fs::read_dir(&checkpoints)
            .with_context(|| format!("reading {}", checkpoints.display()))?
        {
            if entry?.path().extension().is_some_and(|ext| ext == "mpk") {
                found = true;
                break;
            }
        }
        found
    } else {
        false
    };
    anyhow::ensure!(
        !has_best && !has_checkpoint,
        "{} already contains a checkpoint; choose a new run name",
        run_dir.display()
    );
    std::fs::create_dir_all(run_dir.join("checkpoints"))?;
    // A reused run must not retain approval for its previous config and checkpoint.
    crate::quantize::invalidate_gate(run_dir)?;
    // The effective config, overrides applied, so every later step reads what was trained.
    std::fs::write(run_dir.join("config.toml"), toml::to_string(&cfg)?)
        .context("writing the run config")?;
    Ok(())
}

fn train_parser<B: AutodiffBackend>(
    args: &TrainArgs<'_>,
    cfg: &Config,
    run_dir: &Path,
    device: &B::Device,
) -> anyhow::Result<()> {
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
    crate::data::check_parser_manifest(cfg, &sample)?;
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
    anyhow::ensure!(
        train_stats.unencodable == 0,
        "parser training has {} unencodable rows out of {}; repair the shard before training",
        train_stats.unencodable,
        train_stats.encoded + train_stats.unencodable
    );
    let valid_rows: Vec<LabelledExample> =
        read_shard(&processed.join("valid.parquet"), Split::Valid)?
            .into_iter()
            .filter(|e| !e.augmented)
            .collect();
    let (valid_kept, valid_items) = encode_examples(&valid_rows, &fc);
    anyhow::ensure!(
        !valid_rows.is_empty() && valid_items.len() == valid_rows.len(),
        "parser validation has {} encoded rows out of {}; repair the invalid or missing validation labels before training",
        valid_items.len(),
        valid_rows.len()
    );
    eprintln!(
        "train rows {} ({} unencodable), valid rows {}",
        train_stats.encoded,
        train_stats.unencodable,
        valid_items.len()
    );
    let weights = class_weights(&train_ds.items).to_vec();
    serde_json::to_writer_pretty(File::create(run_dir.join("class_weights.json"))?, &weights)?;
    let model = cfg.parser_net_config().init::<B>(device);
    let batch_size = cfg.train.batch_size.max(1);
    let fit = fit::<B>(
        cfg,
        run_dir,
        model,
        TrainingData {
            items: &train_ds.items,
            draws: (0..train_ds.items.len()).collect(),
        },
        &weights,
        device,
        |model| {
            let valid = quick_score(model, &valid_kept, &valid_items, batch_size, device);
            (
                valid.component_f1,
                serde_json::json!({
                    "valid_component_f1": valid.component_f1,
                    "valid_exact": valid.exact,
                }),
            )
        },
    )?;
    let summary = serde_json::json!({
        "best_epoch": fit.best_epoch,
        "best_valid_component_f1": fit.best_score,
        "best_valid_exact": fit.best_metrics["valid_exact"],
        "wall_clock_seconds": fit.seconds,
        "train_rows": train_stats.encoded,
        "train_per_country": args.train_per_country,
        "originals_per_country": train_stats.originals,
        "epochs": cfg.train.epochs,
        "steps": fit.steps,
        "dense_parameters": fit.dense,
        "embedding_parameters": fit.embedding,
    });
    std::fs::write(
        run_dir.join("summary.json"),
        serde_json::to_string_pretty(&summary)? + "\n",
    )?;
    Ok(())
}

/// The longest silver piece in content tokens: the synthetic documents' limit.
const SILVER_MAX_TOKENS: usize = 900;

/// Validate and count the complete detector silver input without creating a run or GPU state.
pub fn check_silver(config: &Path) -> anyhow::Result<()> {
    let cfg = crate::config::load(config)?;
    anyhow::ensure!(
        matches!(cfg.task, crate::config::Task::Detector),
        "check-silver requires a detector config"
    );
    let detector_cfg = cfg
        .detector
        .as_ref()
        .context("a detector config needs a [detector] section")?;
    let (_docs, counts) = detector::load_silver(
        &detector_cfg.silver,
        &cfg.features.to_tessera(),
        SILVER_MAX_TOKENS,
    )?;
    println!("{}", serde_json::to_string_pretty(&counts)?);
    Ok(())
}

/// The old materialized order: synthetic entries once, then each silver entry per repeat.
fn detector_draw_indices(
    synthetic_count: usize,
    silver_count: usize,
    silver_repeat: usize,
) -> anyhow::Result<Vec<usize>> {
    let unique_count = synthetic_count
        .checked_add(silver_count)
        .context("detector example count overflow")?;
    let draw_count = silver_count
        .checked_mul(silver_repeat)
        .and_then(|n| synthetic_count.checked_add(n))
        .context("detector draw count overflow")?;
    let mut draws = Vec::with_capacity(draw_count);
    draws.extend(0..synthetic_count);
    for _ in 0..silver_repeat {
        draws.extend(synthetic_count..unique_count);
    }
    Ok(draws)
}

/// Trains the detector on the synthetic corpus of phase 4.2 with the configured class
/// weights, validating on exact span F1 macro-averaged over person, org, and address.
fn train_detector<B: AutodiffBackend>(
    cfg: &Config,
    run_dir: &Path,
    device: &B::Device,
) -> anyhow::Result<()> {
    let detector_cfg = cfg
        .detector
        .as_ref()
        .context("a detector config needs a [detector] section")?;
    anyhow::ensure!(
        detector_cfg.class_weights.len() == DETECTOR_LABELS,
        "[detector] class_weights needs {DETECTOR_LABELS} values"
    );
    let fc = cfg.features.to_tessera();
    let dir = PathBuf::from(&cfg.data.processed);
    let (train_docs, train_counts) = detector::load_split(&dir, Split::Train, &fc)?;
    let (valid_docs, valid_counts) = detector::load_split(&dir, Split::Valid, &fc)?;
    anyhow::ensure!(
        !valid_docs.is_empty(),
        "detector validation has no encoded documents"
    );
    eprintln!("train {train_counts:?}, valid {valid_counts:?}");
    let mut train_items: Vec<Encoded> = train_docs.into_iter().map(|d| d.enc).collect();
    let (silver_docs, silver_counts) =
        detector::load_silver(&detector_cfg.silver, &fc, SILVER_MAX_TOKENS)?;
    let draws = detector_draw_indices(
        train_items.len(),
        silver_docs.len(),
        detector_cfg.silver_repeat,
    )?;
    if !silver_docs.is_empty() {
        eprintln!(
            "silver {silver_counts:?}, each piece repeated {} times",
            detector_cfg.silver_repeat
        );
        if detector_cfg.silver_repeat > 0 {
            train_items.extend(silver_docs.into_iter().map(|d| d.enc));
        }
    }
    let valid_gold: Vec<Vec<KindSpan>> = valid_docs.iter().map(|d| d.gold.clone()).collect();
    let mut model = cfg.detector_net_config().init::<B>(device);
    if let Some(run) = &cfg.net.ngram_from {
        model.ngram = frozen_ngram::<B>(cfg, Path::new(run), device)?;
    }
    let batch_size = cfg.train.batch_size.max(1);
    let fit = fit::<B>(
        cfg,
        run_dir,
        model,
        TrainingData {
            items: &train_items,
            draws,
        },
        &detector_cfg.class_weights,
        device,
        |model| {
            let pred = detector::predict(model, &valid_docs, batch_size, device);
            let scores = detector::score(&valid_gold, &pred);
            let f1 = |k: &str| scores.per_kind.get(k).map_or(0.0, |s| s.exact.f1);
            eprintln!(
                "valid exact F1: person {:.4}, org {:.4}, address {:.4}",
                f1("person"),
                f1("org"),
                f1("address")
            );
            (
                scores.macro_exact_f1,
                serde_json::json!({ "valid": scores }),
            )
        },
    )?;
    let summary = serde_json::json!({
        "best_epoch": fit.best_epoch,
        "best_valid_macro_exact_f1": fit.best_score,
        "best_valid": fit.best_metrics["valid"],
        "wall_clock_seconds": fit.seconds,
        "train": train_counts,
        "valid": valid_counts,
        "silver": silver_counts,
        "silver_repeat": detector_cfg.silver_repeat,
        "epochs": cfg.train.epochs,
        "steps": fit.steps,
        "dense_parameters": fit.dense,
        "embedding_parameters": fit.embedding,
        "class_weights": detector_cfg.class_weights,
    });
    std::fs::write(
        run_dir.join("summary.json"),
        serde_json::to_string_pretty(&summary)? + "\n",
    )?;
    Ok(())
}

/// The n-gram table of the parser run in `run`, with gradients off so training leaves it as
/// it is. Its features must be this config's, or the same ids would mean different n-grams.
fn frozen_ngram<B: AutodiffBackend>(
    cfg: &Config,
    run: &Path,
    device: &B::Device,
) -> anyhow::Result<burn::nn::Embedding<B>> {
    let parser_cfg = crate::config::load(&run.join("config.toml"))?;
    anyhow::ensure!(
        parser_cfg.features == cfg.features,
        "{} has other feature settings than this config",
        run.display()
    );
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let parser = parser_cfg
        .parser_net_config()
        .init::<B>(device)
        .load_file(run.join("best"), &recorder, device)
        .with_context(|| format!("loading {}", run.join("best.mpk").display()))?;
    eprintln!("n-gram table from {}, frozen", run.display());
    Ok(parser.ngram.no_grad())
}

/// What a finished `fit` reports.
struct FitSummary {
    best_epoch: usize,
    best_score: f64,
    best_metrics: serde_json::Value,
    steps: usize,
    seconds: u64,
    dense: usize,
    embedding: usize,
}

struct TrainingData<'a> {
    items: &'a [Encoded],
    draws: Vec<usize>,
}

/// The shared training loop: AdamW with warmup then cosine decay, masked class-weighted
/// cross-entropy, one validation per epoch through `validate` (a score to maximize and the
/// metrics to log), a checkpoint per epoch, `best` for the highest score, and early stopping
/// after `patience` epochs without improvement.
fn fit<B: AutodiffBackend>(
    cfg: &Config,
    run_dir: &Path,
    mut model: TaggerNet<B>,
    data: TrainingData<'_>,
    weights: &[f32],
    device: &B::Device,
    mut validate: impl FnMut(&TaggerNet<B::InnerBackend>) -> (f64, serde_json::Value),
) -> anyhow::Result<FitSummary> {
    let TrainingData {
        items,
        draws: mut order,
    } = data;
    anyhow::ensure!(cfg.train.epochs > 0, "training requires at least one epoch");
    anyhow::ensure!(!order.is_empty(), "training has no examples");
    anyhow::ensure!(
        order.iter().all(|&index| index < items.len()),
        "training draw index is outside the encoded examples"
    );
    anyhow::ensure!(
        weights.iter().all(|w| w.is_finite() && *w > 0.0),
        "class weights must be finite and positive"
    );
    let (dense, embedding) =
        net::assert_size(&model, cfg.net.max_params, cfg.net.max_embedding_bytes)?;
    eprintln!("parameters: {dense} dense, {embedding} in the n-gram table");
    let class_weights =
        Tensor::<B, 1>::from_data(TensorData::new(weights.to_vec(), [weights.len()]), device);
    let mut optim = AdamWConfig::new()
        .with_weight_decay(0.01)
        .init::<B, TaggerNet<B>>();
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let batch_size = cfg.train.batch_size.max(1);
    let total_steps = cfg.train.epochs * order.len().div_ceil(batch_size);
    let mut metrics = File::create(run_dir.join("metrics.jsonl"))?;
    let mut best = (f64::MIN, 0usize, serde_json::Value::Null);
    let (mut since_best, mut step) = (0usize, 0usize);
    let started = std::time::Instant::now();

    for epoch in 1..=cfg.train.epochs {
        order.shuffle(&mut ChaCha8Rng::seed_from_u64(cfg.seed ^ epoch as u64));
        let (mut epoch_loss, mut batches) = (0f64, 0usize);
        for chunk in order.chunks(batch_size) {
            let batch_items: Vec<Encoded> = chunk.iter().map(|&i| items[i].clone()).collect();
            let batch: ParserBatch<B> = ParserBatcher.batch(batch_items, device);
            let logits = model.forward(
                batch.ngram_ids,
                batch.script,
                batch.shape,
                batch.flags,
                batch.mask.clone(),
            );
            let loss = masked_loss(logits, batch.labels, batch.mask, class_weights.clone());
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
        let (score, logged) = validate(&model.valid());
        anyhow::ensure!(
            score.is_finite(),
            "epoch {epoch} produced a non-finite validation score; refusing to select a checkpoint"
        );
        let train_loss = epoch_loss / batches.max(1) as f64;
        let mut line = serde_json::json!({
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": lr_at(
                step.saturating_sub(1),
                total_steps,
                cfg.train.learning_rate,
                cfg.train.warmup_steps
            ),
        });
        if let (Some(line), serde_json::Value::Object(fields)) = (line.as_object_mut(), &logged) {
            line.extend(fields.clone());
        }
        writeln!(metrics, "{line}")?;
        eprintln!("epoch {epoch}: loss {train_loss:.4}, validation score {score:.4}");
        model.clone().save_file(
            run_dir.join(format!("checkpoints/epoch-{epoch}")),
            &recorder,
        )?;
        if score > best.0 {
            best = (score, epoch, logged);
            since_best = 0;
            model.clone().save_file(run_dir.join("best"), &recorder)?;
        } else {
            since_best += 1;
            if since_best >= cfg.train.patience {
                eprintln!(
                    "early stop at epoch {epoch}; best epoch {}, score {:.4}",
                    best.1, best.0
                );
                break;
            }
        }
    }
    Ok(FitSummary {
        best_epoch: best.1,
        best_score: best.0,
        best_metrics: best.2,
        steps: step,
        seconds: started.elapsed().as_secs(),
        dense,
        embedding,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn training_scope_rejects_a_foreign_test_row_before_run_setup() {
        use polars::prelude::{Column, DataFrame, ParquetWriter};

        let dir = tempfile::tempdir().unwrap();
        let mut cfg = crate::config::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/parser-small.toml"),
        )
        .unwrap();
        cfg.data.processed = dir.path().display().to_string();
        for (split, country) in [("train", "US"), ("valid", "US"), ("test", "GB")] {
            let path = dir.path().join(format!("{split}.parquet"));
            let mut frame =
                DataFrame::new_infer_height(vec![Column::new("country".into(), vec![country])])
                    .unwrap();
            ParquetWriter::new(File::create(path).unwrap())
                .finish(&mut frame)
                .unwrap();
        }
        let error = validate_training_scope(&cfg).unwrap_err().to_string();
        assert!(error.contains("test.parquet row 1"), "{error}");
        assert!(error.contains("country Some(\"GB\")"), "{error}");
    }

    #[test]
    fn changed_training_file_after_snapshot_invalidates_run() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = crate::config::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/parser-small.toml"),
        )
        .unwrap();
        cfg.data.processed = dir.path().join("processed").display().to_string();
        cfg.data.sample_manifest = dir.path().join("sample.json").display().to_string();
        std::fs::create_dir_all(&cfg.data.processed).unwrap();
        for path in [
            Path::new(&cfg.data.processed).join("train.parquet"),
            Path::new(&cfg.data.processed).join("valid.parquet"),
            PathBuf::from(&cfg.data.sample_manifest),
        ] {
            std::fs::write(path, b"original data").unwrap();
        }
        std::fs::write(
            dir.path().join("config.toml"),
            toml::to_string(&cfg).unwrap(),
        )
        .unwrap();
        capture_input_snapshot(&cfg, dir.path()).unwrap();
        let snapshot = load_input_snapshot(dir.path()).unwrap();
        assert_eq!(snapshot.version, INPUT_SNAPSHOT_VERSION);
        assert_eq!(snapshot.tokenizer_contract, TOKENIZER_CONTRACT);
        assert_eq!(snapshot.decoder_contract, DECODER_CONTRACT);
        verify_input_snapshot(dir.path()).unwrap();
        std::fs::write(
            Path::new(&cfg.data.processed).join("train.parquet"),
            b"changed data",
        )
        .unwrap();
        let error = verify_input_snapshot(dir.path()).unwrap_err();
        assert!(error.to_string().contains("train_shard"), "{error:#}");
    }

    #[test]
    fn detector_requires_current_parser_source_snapshot_before_run_setup() {
        let dir = tempfile::tempdir().unwrap();
        let parser_run = dir.path().join("parser");
        std::fs::create_dir(&parser_run).unwrap();
        let mut parser = crate::config::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/parser-small.toml"),
        )
        .unwrap();
        parser.data.processed = dir.path().join("parser_processed").display().to_string();
        parser.data.sample_manifest = dir.path().join("parser_sample.json").display().to_string();
        std::fs::write(
            parser_run.join("config.toml"),
            toml::to_string(&parser).unwrap(),
        )
        .unwrap();
        for (_, path) in input_paths(&parser) {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"parser input").unwrap();
        }
        capture_input_snapshot(&parser, &parser_run).unwrap();
        std::fs::write(parser_run.join("best.mpk"), b"parser checkpoint").unwrap();

        let mut detector = crate::config::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/detector-shared.toml"),
        )
        .unwrap();
        detector.net.ngram_from = Some(parser_run.display().to_string());
        detector.data.processed = dir.path().join("detector_processed").display().to_string();
        detector.detector.as_mut().unwrap().silver =
            vec![dir.path().join("silver.jsonl").display().to_string()];
        assert_eq!(detector.features, parser.features);
        assert!(input_paths(&detector).iter().any(|(role, path)| {
            role == "ngram_source_input_snapshot" && path == &parser_run.join("input_snapshot.json")
        }));
        validate_ngram_source(&detector).unwrap();

        let source_snapshot = parser_run.join("input_snapshot.json");
        let original = std::fs::read(&source_snapshot).unwrap();
        std::fs::remove_file(&source_snapshot).unwrap();
        let error = validate_ngram_source(&detector).unwrap_err();
        assert!(format!("{error:#}").contains("no run-bound input snapshot"));
        std::fs::write(&source_snapshot, &original).unwrap();

        let mut old: serde_json::Value = serde_json::from_slice(&original).unwrap();
        old["version"] = 1.into();
        old.as_object_mut().unwrap().remove("tokenizer_contract");
        old.as_object_mut().unwrap().remove("decoder_contract");
        std::fs::write(&source_snapshot, serde_json::to_vec(&old).unwrap()).unwrap();
        let error = validate_ngram_source(&detector).unwrap_err();
        assert!(format!("{error:#}").contains("unsupported or empty input snapshot"));

        let mut wrong: serde_json::Value = serde_json::from_slice(&original).unwrap();
        wrong["tokenizer_contract"] = "future-tokenizer".into();
        std::fs::write(&source_snapshot, serde_json::to_vec(&wrong).unwrap()).unwrap();
        let error = validate_ngram_source(&detector).unwrap_err();
        assert!(format!("{error:#}").contains("contract differs"));

        std::fs::write(&source_snapshot, &original).unwrap();
        let detector_run = dir.path().join("detector");
        std::fs::create_dir(&detector_run).unwrap();
        std::fs::write(
            detector_run.join("config.toml"),
            toml::to_string(&detector).unwrap(),
        )
        .unwrap();
        for (role, path) in input_paths(&detector) {
            if role.starts_with("ngram_source_") {
                continue;
            }
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"detector input").unwrap();
        }
        capture_input_snapshot(&detector, &detector_run).unwrap();
        verify_input_snapshot(&detector_run).unwrap();
        std::fs::write(&source_snapshot, serde_json::to_vec(&wrong).unwrap()).unwrap();
        let error = verify_input_snapshot(&detector_run).unwrap_err();
        assert!(format!("{error:#}").contains("contract differs"));
    }

    #[test]
    fn input_snapshot_cannot_omit_configured_silver_sources() {
        let dir = tempfile::tempdir().unwrap();
        let mut config = crate::config::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/detector-shared.toml"),
        )
        .unwrap();
        config.detector.as_mut().unwrap().silver = vec!["silver.jsonl".into()];
        std::fs::write(
            dir.path().join("config.toml"),
            toml::to_string(&config).unwrap(),
        )
        .unwrap();
        let snapshot = InputSnapshot {
            version: INPUT_SNAPSHOT_VERSION,
            config_sha256: hash_file(&dir.path().join("config.toml")).unwrap(),
            tokenizer_contract: TOKENIZER_CONTRACT.to_string(),
            decoder_contract: DECODER_CONTRACT.to_string(),
            inputs: std::collections::BTreeMap::from([(
                "train_shard".to_string(),
                InputFile {
                    path: "train.parquet".into(),
                    sha256: "irrelevant".into(),
                },
            )]),
        };
        std::fs::write(
            dir.path().join("input_snapshot.json"),
            serde_json::to_vec(&snapshot).unwrap(),
        )
        .unwrap();
        let error = load_input_snapshot(dir.path()).unwrap_err();
        assert!(error.to_string().contains("omits or changes"), "{error:#}");
    }

    #[test]
    fn repeated_silver_draws_keep_expanded_example_order_and_steps() {
        let synthetic = [10usize, 11];
        let silver = [20usize, 21];
        let unique = [synthetic.as_slice(), silver.as_slice()].concat();
        let seed = 42u64;
        for repeat in [0, 1, 3] {
            let expanded = synthetic
                .iter()
                .copied()
                .chain((0..repeat).flat_map(|_| silver.iter().copied()))
                .collect::<Vec<_>>();
            let draws = detector_draw_indices(synthetic.len(), silver.len(), repeat).unwrap();
            assert_eq!(
                draws.iter().map(|&index| unique[index]).collect::<Vec<_>>(),
                expanded
            );
            for batch_size in [1, 2, 3] {
                assert_eq!(
                    draws.len().div_ceil(batch_size),
                    expanded.len().div_ceil(batch_size)
                );
            }
            let mut old_order = (0..expanded.len()).collect::<Vec<_>>();
            let mut new_order = draws;
            for epoch in 1..=2 {
                let epoch_seed = seed ^ epoch;
                old_order.shuffle(&mut ChaCha8Rng::seed_from_u64(epoch_seed));
                new_order.shuffle(&mut ChaCha8Rng::seed_from_u64(epoch_seed));
                assert_eq!(
                    new_order
                        .iter()
                        .map(|&index| unique[index])
                        .collect::<Vec<_>>(),
                    old_order
                        .iter()
                        .map(|&index| expanded[index])
                        .collect::<Vec<_>>(),
                    "repeat={repeat}, epoch={epoch}"
                );
            }
        }
    }

    #[test]
    fn repeated_silver_draw_count_overflow_is_an_error() {
        assert!(detector_draw_indices(1, usize::MAX, 2).is_err());
        assert!(detector_draw_indices(usize::MAX, 1, 0).is_err());
    }

    #[test]
    fn reused_run_without_checkpoint_loses_quantization_approval_before_training() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = crate::config::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/parser-small.toml"),
        )
        .unwrap();
        std::fs::write(dir.path().join("quantize.json"), r#"{"passed":true}"#).unwrap();
        std::fs::write(dir.path().join("quantized.safetensors"), b"old weights").unwrap();
        std::fs::create_dir(dir.path().join("checkpoints")).unwrap();
        std::fs::write(
            dir.path().join("checkpoints/.DS_Store"),
            b"not a checkpoint",
        )
        .unwrap();
        initialize_run(dir.path(), &cfg).unwrap();
        assert!(!dir.path().join("quantize.json").exists());
        assert_eq!(
            std::fs::read(dir.path().join("quantized.safetensors")).unwrap(),
            b"old weights"
        );
        assert!(dir.path().join("config.toml").exists());
        assert!(dir.path().join("checkpoints").exists());
        assert_eq!(
            std::fs::read(dir.path().join("checkpoints/.DS_Store")).unwrap(),
            b"not a checkpoint"
        );
    }

    #[test]
    fn reused_run_with_best_is_rejected_before_mutation() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = crate::config::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/parser-small.toml"),
        )
        .unwrap();
        std::fs::write(dir.path().join("best.mpk"), b"old checkpoint").unwrap();
        std::fs::write(dir.path().join("config.toml"), b"old config").unwrap();
        std::fs::write(dir.path().join("quantize.json"), b"old gate").unwrap();
        std::fs::write(dir.path().join("quantized.safetensors"), b"old weights").unwrap();
        let error = initialize_run(dir.path(), &cfg).unwrap_err().to_string();
        assert!(error.contains("choose a new run name"), "{error}");
        for (path, expected) in [
            ("best.mpk", b"old checkpoint".as_slice()),
            ("config.toml", b"old config".as_slice()),
            ("quantize.json", b"old gate".as_slice()),
            ("quantized.safetensors", b"old weights".as_slice()),
        ] {
            assert_eq!(std::fs::read(dir.path().join(path)).unwrap(), expected);
        }
        assert!(!dir.path().join("checkpoints").exists());
    }

    #[test]
    fn reused_run_with_epoch_checkpoint_is_rejected_before_mutation() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = crate::config::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/parser-small.toml"),
        )
        .unwrap();
        std::fs::create_dir(dir.path().join("checkpoints")).unwrap();
        std::fs::write(
            dir.path().join("checkpoints/epoch-1.mpk"),
            b"old checkpoint",
        )
        .unwrap();
        std::fs::write(dir.path().join("config.toml"), b"old config").unwrap();
        std::fs::write(dir.path().join("quantize.json"), b"old gate").unwrap();
        let error = initialize_run(dir.path(), &cfg).unwrap_err().to_string();
        assert!(error.contains("choose a new run name"), "{error}");
        assert_eq!(
            std::fs::read(dir.path().join("config.toml")).unwrap(),
            b"old config"
        );
        assert_eq!(
            std::fs::read(dir.path().join("quantize.json")).unwrap(),
            b"old gate"
        );
        assert_eq!(
            std::fs::read(dir.path().join("checkpoints/epoch-1.mpk")).unwrap(),
            b"old checkpoint"
        );
    }

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
        let loss: f32 = masked_loss(certain, labels.clone(), mask.clone(), weights.clone())
            .into_scalar()
            .elem();
        assert!(loss < 1e-3, "{loss}");
        let uniform = Tensor::<B, 3>::zeros([1, 2, PARSER_LABELS], &device);
        let loss: f32 = masked_loss(uniform, labels, mask, weights)
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
