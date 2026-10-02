//! `trainer train`: one manual training loop for the parser and the detector, with masked,
//! class-weighted cross-entropy, AdamW, warmup then cosine decay, early stopping on a
//! validation score (component F1 for the parser, minimum real-development exact F1 for the detector),
//! checkpoints, and a `metrics.jsonl` log.
//!
//! A manual loop rather than Burn's `Learner`: the loop is short and it validates on decoded
//! spans instead of loss.

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::Context;
use burn::backend::{Autodiff, NdArray, Wgpu};
use burn::data::dataloader::batcher::Batcher;
use burn::module::AutodiffModule;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer, grad_clipping::GradientClippingConfig};
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::TensorData;
use burn::tensor::activation::log_softmax;
use burn::tensor::backend::AutodiffBackend;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
#[cfg(test)]
use tessera::internal::DECODER_CONTRACT;
use tessera::internal::TOKENIZER_CONTRACT;

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
    /// A bounded diagnostic stops after this many updates and cannot be exported.
    pub max_steps: Option<usize>,
    /// Where to train.
    pub backend: BackendKind,
}

/// `trainer train`: trains the parser or the detector, as the config's `task` says.
pub fn run(args: TrainArgs<'_>) -> anyhow::Result<()> {
    crate::typed_synthetic::refuse_fit(&crate::config::load(args.config)?)?;
    match args.backend {
        BackendKind::Wgpu => train::<Autodiff<Wgpu>>(&args, &Default::default()),
        BackendKind::Ndarray => train::<Autodiff<NdArray>>(&args, &Default::default()),
    }
}

/// Initialize or fit exactly one reviewed CPU fullmix RMS diagnostic.
pub(crate) fn diagnose_fullmix_rms(
    manifest: &Path,
    out: &Path,
    preflight: bool,
) -> anyhow::Result<()> {
    let (request, cfg) = crate::fullmix_rms::Request::load(manifest, preflight)?;
    request.validate_authored_route(&cfg)?;
    anyhow::ensure!(
        matches!(std::fs::symlink_metadata(out), Err(error) if error.kind() == std::io::ErrorKind::NotFound),
        "fullmix output must be a new directory"
    );
    validate_ngram_source(&cfg)?;
    validate_training_scope(&cfg)?;
    verify_real_development_gold(&cfg)?;
    verify_generation_manifest(&cfg)?;
    initialize_run(out, &cfg)?;
    capture_input_snapshot(&cfg, out)?;
    request.bind(&cfg, out)?;
    let device = Default::default();
    <Autodiff<NdArray> as Backend>::seed(&device, cfg.seed);
    let result = train_detector::<Autodiff<NdArray>>(
        &cfg,
        out,
        Some(crate::fullmix_rms::STEPS),
        Some(&request),
        &device,
    );
    verify_input_snapshot(out)?;
    result
}

/// Explicit cohort and optional forward intervention for the bounded diagnostic.
pub(crate) struct MemorizeScope<'a> {
    pub(crate) cohort: Option<&'a Path>,
    pub(crate) residual_rms: bool,
}

struct VerifiedMemorizeScope<'a> {
    cohort: Option<&'a Path>,
    forward_operator: crate::diagnostic_operator::ForwardOperator,
}

/// Run a bounded memorization check using original training documents only.
pub(crate) fn memorize(
    config: &Path,
    out: &Path,
    steps: usize,
    backend: BackendKind,
    logit_penalty: f64,
    expected_initial_sha256: Option<&str>,
    scope: MemorizeScope<'_>,
) -> anyhow::Result<()> {
    let cfg = crate::config::load(config)?;
    crate::typed_synthetic::refuse_fit(&cfg)?;
    anyhow::ensure!(
        !cfg.context96_rms(),
        "context RMS uses its separately bound fit route, not legacy memorization"
    );
    let MemorizeScope {
        cohort,
        residual_rms,
    } = scope;
    anyhow::ensure!(
        cohort.is_none() || matches!(backend, BackendKind::Ndarray),
        "whole-original PERSON diagnostic requires the materialized CPU initialization"
    );
    let forward_operator = crate::diagnostic_operator::validate_request(
        residual_rms,
        cohort.is_some(),
        matches!(backend, BackendKind::Ndarray),
        steps,
        logit_penalty,
        expected_initial_sha256,
    )?;
    crate::loss_stability::validate(logit_penalty, expected_initial_sha256)?;
    match backend {
        BackendKind::Wgpu => memorize_with::<Autodiff<Wgpu>>(
            config,
            out,
            steps,
            logit_penalty,
            expected_initial_sha256,
            VerifiedMemorizeScope {
                cohort,
                forward_operator,
            },
            &Default::default(),
        ),
        BackendKind::Ndarray => memorize_with::<Autodiff<NdArray>>(
            config,
            out,
            steps,
            logit_penalty,
            expected_initial_sha256,
            VerifiedMemorizeScope {
                cohort,
                forward_operator,
            },
            &Default::default(),
        ),
    }
}

fn memorize_with<B: AutodiffBackend>(
    config: &Path,
    out: &Path,
    steps: usize,
    logit_penalty: f64,
    expected_initial_sha256: Option<&str>,
    scope: VerifiedMemorizeScope<'_>,
    device: &B::Device,
) -> anyhow::Result<()> {
    let VerifiedMemorizeScope {
        cohort,
        forward_operator,
    } = scope;
    anyhow::ensure!(
        matches!(std::fs::symlink_metadata(out), Err(e) if e.kind() == std::io::ErrorKind::NotFound),
        "memorization output already exists or cannot be inspected"
    );
    let mut cfg = crate::config::load(config)?;
    crate::typed_synthetic::refuse_fit(&cfg)?;
    anyhow::ensure!(
        cfg.task == crate::config::Task::Detector
            && steps > cfg.train.warmup_steps
            && (1000..=2000).contains(&steps),
        "memorization needs a detector and 1000–2000 updates, above its warmup"
    );
    validate_ngram_source(&cfg)?;
    validate_training_scope(&cfg)?;
    verify_real_development_gold(&cfg)?;
    verify_generation_manifest(&cfg)?;
    let frozen = cohort.is_some();
    anyhow::ensure!(
        !frozen
            || (steps == 2000
                && logit_penalty == 0.0
                && expected_initial_sha256 == Some(crate::memorization::exposure::INITIAL_SHA)),
        "whole-original PERSON diagnostic requires fixed2000 native CE and exact initialization"
    );
    let mut prepared = match cohort {
        Some(path) => crate::memorization::prepare_frozen(&cfg, path)?,
        None => crate::memorization::prepare(&cfg, 32, 128)?,
    };
    anyhow::ensure!(
        prepared.documents.len() == if frozen { 15 } else { 32 },
        "memorization original training document count differs from its explicit scope"
    );
    let detector = cfg.detector.as_mut().context("missing detector settings")?;
    let weights = detector.class_weights.clone();
    detector.learning_check = None;
    let items: Vec<_> = prepared
        .documents
        .iter()
        .map(|doc| doc.enc.clone())
        .collect();
    let repeats = if frozen { 32 } else { 64 };
    let draws: Vec<_> = (0..repeats).flat_map(|_| 0..items.len()).collect();
    let batches_per_epoch = draws.len().div_ceil(cfg.train.batch_size);
    cfg.train.epochs = steps.div_ceil(batches_per_epoch);
    cfg.train.diagnostic_schedule_steps = Some(
        cfg.train
            .diagnostic_schedule_steps
            .unwrap_or(steps)
            .max(steps),
    );
    std::fs::create_dir(out)?;
    initialize_run(out, &cfg)?;
    std::fs::write(
        out.join("diagnostic.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "max_steps": steps,
            "scope": "fixed training-only memorization diagnostic; never export",
            "source_config": config,
            "source_config_sha256": hash_file(config)?,
            "changes": ["32 original training documents repeated 64 times per epoch",
                        "no development evaluation or checkpoint selection",
                        "learning gate replaced by diagnostic scoring of every selected document"],
            "release_quality_claim": false,
            "logit_penalty": logit_penalty,
            "expected_initial_sha256": expected_initial_sha256,
        }))?,
    )?;
    if let Some(path) = cohort {
        let marker_path = out.join("diagnostic.json");
        let mut marker: serde_json::Value = serde_json::from_slice(&std::fs::read(&marker_path)?)?;
        marker["frozen_cohort_scope"] = serde_json::json!(crate::memorization::FROZEN_SCOPE);
        marker["frozen_cohort_path"] = serde_json::json!(path);
        marker["frozen_cohort_sha256"] = serde_json::json!(hash_file(path)?);
        marker["changes"] = serde_json::json!([
            "15 reviewed whole original rows repeated32 times per complete epoch",
            "fixed2000 updates; no in-loop scoring or best checkpoint selection",
            "fixed preregistered checkpoints; all scoring occurs after training",
        ]);
        marker["observation_steps"] =
            serde_json::json!(crate::memorization::exposure::OBSERVATIONS);
        std::fs::write(marker_path, serde_json::to_vec_pretty(&marker)?)?;
    }
    if forward_operator == crate::diagnostic_operator::ForwardOperator::ResidualRmsV1 {
        let marker_path = out.join("diagnostic.json");
        let mut marker: serde_json::Value = serde_json::from_slice(&std::fs::read(&marker_path)?)?;
        crate::diagnostic_operator::attach(out, &mut marker, &mut prepared.manifest)?;
        std::fs::write(marker_path, serde_json::to_vec_pretty(&marker)?)?;
    }
    std::fs::write(
        out.join("selection.json"),
        serde_json::to_vec_pretty(&prepared.manifest)?,
    )?;
    let mut cases = File::create(out.join("cases.jsonl"))?;
    for (index, doc) in prepared.documents.iter().enumerate() {
        let expected = doc
            .gold
            .iter()
            .map(|span| {
                Ok(serde_json::json!({
                    "kind": detector::KINDS[span.kind].as_str(),
                    "start": span.start,
                    "end": span.end,
                    "text": doc.text.get(span.start as usize..span.end as usize)
                        .context("memorization gold cuts source text")?,
                }))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        writeln!(
            cases,
            "{}",
            serde_json::json!({
                "name": if frozen { prepared.manifest["documents"][index]["name"].as_str()
                    .context("frozen document name is missing")?.to_owned() }
                    else { format!("memorization-{index}") }, "country": "US",
                "doc_type": "training_memorization", "input": doc.text, "expected": expected,
            })
        )?;
    }
    capture_input_snapshot(&cfg, out)?;
    B::seed(device, cfg.seed);
    let mut model = cfg.detector_net_config().init::<B>(device);
    if let Some(parser) = &cfg.net.ngram_from {
        model.ngram = pretrained_ngram::<B>(&cfg, Path::new(parser), device)?;
    }
    let initial_sha256 = crate::training_diagnostic::parameter_sha256(&crate::quantize::extract(
        &model.valid(),
        cfg.net_name(),
    ));
    crate::loss_stability::verify_initial(&initial_sha256, expected_initial_sha256)?;
    let diagnostic_path = out.join("diagnostic.json");
    let mut diagnostic: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&diagnostic_path)?)?;
    diagnostic["actual_initial_sha256"] = serde_json::json!(initial_sha256);
    std::fs::write(diagnostic_path, serde_json::to_vec_pretty(&diagnostic)?)?;
    let gold: Vec<_> = prepared
        .documents
        .iter()
        .map(|doc| doc.gold.clone())
        .collect();
    let fit = fit::<B>(
        &cfg,
        out,
        model,
        TrainingData {
            items: &items,
            draws,
            detector_epoch: None,
            learning_probe: None,
            max_steps: Some(steps),
            diagnostic_logit_penalty: (logit_penalty > 0.0).then_some(logit_penalty),
            forward_operator,
            completed_exposure: None,
        },
        &weights,
        device,
        |model| {
            if frozen {
                return Ok((
                    CheckpointScore::new(0., 0.),
                    serde_json::json!({"scoring": "after-fit only"}),
                ));
            }
            let candidates =
                detector::predict_scored(model, &prepared.documents, cfg.train.batch_size, device);
            let raw: Vec<_> = candidates
                .iter()
                .map(|spans| spans.iter().map(|s| s.span).collect())
                .collect();
            let kept = detector::apply_confidence_policy(&candidates);
            let filtered = detector::score(&gold, &kept);
            let unfiltered = detector::score(&gold, &raw);
            Ok((
                CheckpointScore::new(minimum_model_exact_f1(&filtered), filtered.macro_exact_f1),
                serde_json::json!({"memorization_filtered": filtered, "memorization_unfiltered": unfiltered}),
            ))
        },
    )?;
    verify_input_snapshot(out)?;
    std::fs::write(
        out.join("memorization-result.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "steps": fit.steps,
            "seconds": fit.seconds,
            "checkpoint": format!("checkpoints/diagnostic-step-{steps}.mpk"),
            "checkpoint_sha256": hash_file(&out.join(format!("checkpoints/diagnostic-step-{steps}.mpk")))?,
            "selection_sha256": hash_file(&out.join("selection.json"))?,
            "cases_sha256": hash_file(&out.join("cases.jsonl"))?,
            "scope": "seen-example learnability only; not independent accuracy or release approval",
        }))?,
    )?;
    Ok(())
}

fn train<B: AutodiffBackend>(args: &TrainArgs<'_>, device: &B::Device) -> anyhow::Result<()> {
    let mut cfg = crate::config::load(args.config)?;
    crate::typed_synthetic::refuse_fit(&cfg)?;
    if let Some(name) = args.name {
        cfg.name = name.to_string();
    }
    if let Some(e) = args.epochs {
        cfg.train.epochs = e;
    }
    crate::training_diagnostic::validate_launch(&cfg, args.max_steps)?;
    validate_ngram_source(&cfg)?;
    validate_training_scope(&cfg)?;
    anyhow::ensure!(
        args.max_steps.is_none_or(|steps| steps > 0)
            && (args.max_steps.is_none() || matches!(cfg.task, crate::config::Task::Detector)),
        "max-steps must be positive and is only supported for detector diagnostics"
    );
    if matches!(cfg.task, crate::config::Task::Detector) {
        verify_real_development_gold(&cfg)?;
        verify_generation_manifest(&cfg)?;
    }
    let run_dir = PathBuf::from("runs").join(&cfg.name);
    initialize_run(&run_dir, &cfg)?;
    if let Some(max_steps) = args.max_steps {
        std::fs::write(
            run_dir.join("diagnostic.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "max_steps": max_steps,
                "scope": "bounded learning diagnostic; not exportable",
            }))? + "\n",
        )?;
    }
    capture_input_snapshot(&cfg, &run_dir)?;
    B::seed(device, cfg.seed);
    let result = match cfg.task {
        crate::config::Task::Parser => train_parser::<B>(args, &cfg, &run_dir, device),
        crate::config::Task::Detector => {
            train_detector::<B>(&cfg, &run_dir, args.max_steps, None, device)
        }
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
const REAL_DEV_GOLD: &str = "data/interim/review/us-development-gold-v2.jsonl";
const REAL_DEV_GOLD_SHA256: &str =
    "d7fd7166a5a40cc1d199eed1cfda234639f9737a16bf92fa9130ba06fe65c2f5";
const REAL_DEV_MANIFEST: &str = "data/interim/review/us-development-gold-v2.manifest.json";
const REAL_DEV_ERRATA: &str = "bench/us/fixtures/us-development-gold-errata-v2.json";
const REAL_DEV_EXCLUSIONS: &str = "data/interim/review/us-eval-exclusions-v1.jsonl";
const REAL_DEV_GOLD_V3: &str = "data/interim/review/us-development-gold-v3.jsonl";
const REAL_DEV_GOLD_V3_SHA256: &str =
    "65006863254b6e3b5520b5063e4e08153e7ecd269c05155203b3de10b0833c74";
const REAL_DEV_MANIFEST_V3: &str = "data/interim/review/us-development-gold-v3.manifest.json";
const REAL_DEV_ERRATA_V3: &str = "bench/us/fixtures/us-development-gold-errata-v3.json";
const REAL_DEV_EXCLUSIONS_V4: &str = "data/interim/review/us-eval-exclusions-v4.jsonl";
const REAL_DEV_EXCLUSIONS_V5: &str = "data/interim/review/us-eval-exclusions-v5.jsonl";
const REAL_DEV_EXCLUSIONS_V5_SHA256: &str =
    "788fba979eff4362b74c503a61307001af07a1141fe6d4dabcfe73050e806c14";
const REAL_DEV_EXCLUSIONS_V5_MANIFEST: &str =
    "data/interim/review/us-eval-exclusions-v5.manifest.json";
const REAL_DEV_EXCLUSIONS_V5_BUILDER: &str = "bench/us/build_us_next_eval_exclusions.py";
const REVIEW_POLICY: &str = "internal/bench/review/GUIDELINES.md";
const REAL_DEV_EXCLUSIONS_V5_SOURCES: [&str; 3] = [
    REAL_DEV_EXCLUSIONS_V4,
    "data/interim/review/us-long-org-eval-gold-v4.jsonl",
    "data/interim/review/us-doe-org-overviews-strict-gold-v1.jsonl",
];
const REAL_DEV_SOURCES: [&str; 4] = [
    "data/interim/review/us-dev-v1.jsonl",
    "data/interim/review/us-office-eval-v1.jsonl",
    "data/interim/review/us-staff-challenge-v1.jsonl",
    "data/interim/review/us-park-address-challenge-v1.jsonl",
];

fn repo_path(relative: &str) -> PathBuf {
    if let Some(root) = crate::diagnostic_decode::compiled_input_root() {
        return Path::new(root).join(relative);
    }
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("trainer lives inside the repository")
        .join(relative)
}

#[derive(Deserialize)]
struct RealDevManifest {
    kind: String,
    cases: usize,
    sha256: String,
    base_sha256: String,
    errata_sha256: String,
    sources_sha256: BTreeMap<String, String>,
}

fn uses_v4_gold(cfg: &Config) -> bool {
    cfg.name == "detector-us-v4"
        || cfg.data.processed == "data/processed/detector-us-v4"
        || cfg.data.manifests == "data/manifests/v4"
        || cfg
            .generate
            .as_ref()
            .is_some_and(|generate| generate.exclude_gold == REAL_DEV_EXCLUSIONS_V4)
}

fn uses_v5_gold(cfg: &Config) -> bool {
    cfg.name == "detector-us-v5"
        || cfg.data.processed == "data/processed/detector-us-v5"
        || cfg.data.manifests == "data/manifests/v5"
        || cfg
            .generate
            .as_ref()
            .is_some_and(|generate| generate.exclude_gold == REAL_DEV_EXCLUSIONS_V5)
}

fn real_dev_gold(cfg: &Config) -> (&'static str, &'static str) {
    if uses_v4_gold(cfg) || uses_v5_gold(cfg) {
        (REAL_DEV_GOLD_V3, REAL_DEV_GOLD_V3_SHA256)
    } else {
        (REAL_DEV_GOLD, REAL_DEV_GOLD_SHA256)
    }
}

fn verify_real_development_gold(cfg: &Config) -> anyhow::Result<()> {
    let manifest: RealDevManifest =
        serde_json::from_slice(&std::fs::read(repo_path(REAL_DEV_MANIFEST))?)?;
    anyhow::ensure!(
        manifest.kind == "us_development_gold_corrected"
            && manifest.cases == 196
            && manifest.sha256 == REAL_DEV_GOLD_SHA256
            && hash_file(&repo_path(REAL_DEV_GOLD))? == REAL_DEV_GOLD_SHA256,
        "pinned real development gold changed"
    );
    anyhow::ensure!(
        hash_file(&repo_path(REAL_DEV_ERRATA))? == manifest.errata_sha256,
        "reviewed real development errata changed"
    );
    let mut base = Vec::new();
    let mut source_names = BTreeSet::new();
    for source in REAL_DEV_SOURCES {
        let source_path = repo_path(source);
        let path = source_path.as_path();
        let name = path
            .file_name()
            .context("development source has no file name")?
            .to_string_lossy()
            .into_owned();
        anyhow::ensure!(
            manifest.sources_sha256.get(&name) == Some(&hash_file(path)?),
            "real development source {name} changed"
        );
        source_names.insert(name);
        base.extend(std::fs::read(path)?);
    }
    anyhow::ensure!(
        manifest
            .sources_sha256
            .keys()
            .cloned()
            .collect::<BTreeSet<_>>()
            == source_names
            && crate::export::sha256_hex(&base) == manifest.base_sha256,
        "real development source set changed"
    );
    let exclusions = std::fs::read_to_string(repo_path(REAL_DEV_EXCLUSIONS))?;
    let excluded: BTreeMap<String, String> = exclusions
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let row: serde_json::Value = serde_json::from_str(line)?;
            Ok((
                row["name"]
                    .as_str()
                    .context("evaluation exclusion has no name")?
                    .to_string(),
                row["input"]
                    .as_str()
                    .context("evaluation exclusion has no input")?
                    .to_string(),
            ))
        })
        .collect::<anyhow::Result<_>>()?;
    let gold = std::fs::read_to_string(repo_path(REAL_DEV_GOLD))?;
    let mut count = 0;
    for line in gold.lines().filter(|line| !line.trim().is_empty()) {
        let row: serde_json::Value = serde_json::from_str(line)?;
        let name = row["name"]
            .as_str()
            .context("real development case has no name")?;
        let input = row["input"]
            .as_str()
            .context("real development case has no input")?;
        anyhow::ensure!(
            excluded.get(name).is_some_and(|original| original == input),
            "real development case {name} is absent from training exclusions"
        );
        count += 1;
    }
    anyhow::ensure!(
        count == manifest.cases,
        "real development case count changed"
    );
    if uses_v4_gold(cfg) || uses_v5_gold(cfg) {
        verify_v4_real_development_gold()?;
    }
    if uses_v5_gold(cfg) {
        verify_v5_evaluation_exclusions()?;
    }
    Ok(())
}

fn verify_v4_real_development_gold() -> anyhow::Result<()> {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(repo_path(REAL_DEV_MANIFEST_V3))?)?;
    anyhow::ensure!(
        manifest["kind"] == "us_development_gold_corrected_v3"
            && manifest["cases"] == 196
            && manifest["added_org_spans"] == 5
            && manifest["sha256"] == REAL_DEV_GOLD_V3_SHA256
            && manifest["base_sha256"] == REAL_DEV_GOLD_SHA256
            && manifest["errata_sha256"] == hash_file(&repo_path(REAL_DEV_ERRATA_V3))?
            && hash_file(&repo_path(REAL_DEV_GOLD_V3))? == REAL_DEV_GOLD_V3_SHA256,
        "pinned V4 real development gold changed"
    );
    let exclusions = std::fs::read_to_string(repo_path(REAL_DEV_EXCLUSIONS_V4))?;
    let excluded: BTreeMap<String, serde_json::Value> = exclusions
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let row: serde_json::Value = serde_json::from_str(line)?;
            let name = row["name"]
                .as_str()
                .context("V4 evaluation exclusion has no name")?
                .to_string();
            Ok((name, row))
        })
        .collect::<anyhow::Result<_>>()?;
    anyhow::ensure!(excluded.len() == 343, "V4 evaluation exclusions changed");
    let exclusion_manifest: serde_json::Value = serde_json::from_slice(&std::fs::read(
        repo_path("data/interim/review/us-eval-exclusions-v4.manifest.json"),
    )?)?;
    anyhow::ensure!(
        exclusion_manifest["kind"] == "us_v4_eval_exclusions_v1"
            && exclusion_manifest["cases"] == 343
            && exclusion_manifest["sha256"] == hash_file(&repo_path(REAL_DEV_EXCLUSIONS_V4))?
            && exclusion_manifest["source_sha256"][REAL_DEV_GOLD_V3] == REAL_DEV_GOLD_V3_SHA256
            && exclusion_manifest["source_sha256"]["data/interim/review/us-eval-exclusions-v3.jsonl"]
                == hash_file(&repo_path(
                    "data/interim/review/us-eval-exclusions-v3.jsonl"
                ))?,
        "V4 evaluation exclusion manifest changed"
    );
    let gold = std::fs::read_to_string(repo_path(REAL_DEV_GOLD_V3))?;
    let mut count = 0;
    for line in gold.lines().filter(|line| !line.trim().is_empty()) {
        let row: serde_json::Value = serde_json::from_str(line)?;
        let name = row["name"]
            .as_str()
            .context("V4 real development case has no name")?;
        anyhow::ensure!(
            excluded.get(name) == Some(&row),
            "V4 real development case {name} differs from training exclusions"
        );
        count += 1;
    }
    anyhow::ensure!(count == 196, "V4 real development case count changed");
    Ok(())
}

fn verify_v5_evaluation_exclusions() -> anyhow::Result<()> {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(repo_path(REAL_DEV_EXCLUSIONS_V5_MANIFEST))?)?;
    let bytes = std::fs::read(repo_path(REAL_DEV_EXCLUSIONS_V5))?;
    anyhow::ensure!(
        manifest["kind"] == "us_next_eval_exclusions_v5"
            && manifest["cases"] == 371
            && manifest["training_eligible"] == false
            && manifest["sha256"] == REAL_DEV_EXCLUSIONS_V5_SHA256
            && crate::export::sha256_hex(&bytes) == REAL_DEV_EXCLUSIONS_V5_SHA256,
        "V5 evaluation exclusion manifest changed"
    );
    let mut combined = Vec::new();
    let mut sources = BTreeMap::new();
    let mut inputs = BTreeMap::new();
    for (source, cases) in REAL_DEV_EXCLUSIONS_V5_SOURCES
        .into_iter()
        .zip([343, 15, 13])
    {
        let path = repo_path(source);
        let source_bytes = std::fs::read(&path)?;
        let hash = crate::export::sha256_hex(&source_bytes);
        let source_manifest_path = path.with_extension("manifest.json");
        let source_manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&source_manifest_path)?)?;
        anyhow::ensure!(
            source_manifest["sha256"] == hash && source_manifest["cases"] == cases,
            "V5 evaluation source manifest changed: {source}"
        );
        sources.insert(source.to_string(), hash);
        inputs.insert(
            Path::new(source)
                .with_extension("manifest.json")
                .to_string_lossy()
                .into_owned(),
            hash_file(&source_manifest_path)?,
        );
        combined.extend(source_bytes);
    }
    for source in [REAL_DEV_EXCLUSIONS_V5_BUILDER, REVIEW_POLICY] {
        inputs.insert(source.to_string(), hash_file(&repo_path(source))?);
    }
    anyhow::ensure!(
        bytes == combined
            && manifest["source_sha256"] == serde_json::to_value(sources)?
            && manifest["input_sha256"] == serde_json::to_value(inputs)?,
        "V5 evaluation exclusions differ from their complete pinned sources"
    );
    let mut excluded = BTreeMap::new();
    for line in std::str::from_utf8(&bytes)?
        .lines()
        .filter(|line| !line.trim().is_empty())
    {
        let row: serde_json::Value = serde_json::from_str(line)?;
        let name = row["name"]
            .as_str()
            .context("V5 evaluation case has no name")?
            .to_string();
        anyhow::ensure!(
            row["country"] == "US",
            "V5 evaluation case is not US: {name}"
        );
        anyhow::ensure!(
            excluded.insert(name.clone(), row).is_none(),
            "duplicate V5 evaluation case: {name}"
        );
    }
    anyhow::ensure!(
        excluded.len() == 371,
        "V5 evaluation exclusion membership changed"
    );
    for line in std::fs::read_to_string(repo_path(REAL_DEV_GOLD_V3))?
        .lines()
        .filter(|line| !line.trim().is_empty())
    {
        let row: serde_json::Value = serde_json::from_str(line)?;
        let name = row["name"]
            .as_str()
            .context("V5 development case has no name")?;
        anyhow::ensure!(
            excluded.get(name) == Some(&row),
            "V5 development case {name} differs from training exclusions"
        );
    }
    Ok(())
}

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

fn verify_generation_manifest(cfg: &Config) -> anyhow::Result<()> {
    let amended_sources = crate::typed_synthetic::generation_sources(cfg)?;
    if !uses_v4_gold(cfg) && !uses_v5_gold(cfg) {
        anyhow::ensure!(
            amended_sources.is_none(),
            "typed authored data needs the reviewed V4/V5 generation chain"
        );
        return Ok(());
    }
    let (version, exclusions) = if uses_v5_gold(cfg) {
        ("V5", REAL_DEV_EXCLUSIONS_V5)
    } else {
        ("V4", REAL_DEV_EXCLUSIONS_V4)
    };
    let generate = cfg
        .generate
        .as_ref()
        .with_context(|| format!("{version} detector needs [generate]"))?;
    anyhow::ensure!(
        generate.exclude_gold == exclusions,
        "{version} detector must exclude corrected {version} evaluation gold"
    );
    if uses_v5_gold(cfg) {
        anyhow::ensure!(
            cfg.name != "detector-us-v4"
                && cfg.data.processed != "data/processed/detector-us-v4"
                && cfg.data.manifests != "data/manifests/v4",
            "V5 detector must use separate V5 output paths and run name"
        );
    }
    let detector = cfg
        .detector
        .as_ref()
        .with_context(|| format!("{version} detector needs [detector]"))?;
    if uses_v5_gold(cfg) {
        anyhow::ensure!(
            detector.silver.len() == 34
                && detector.silver.iter().collect::<BTreeSet<_>>().len() == 34,
            "V5 detector must use 34 distinct reviewed silver sources"
        );
    }
    let path = repo_path(&cfg.data.manifests).join("detector-synthetic.json");
    let manifest: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?,
    )?;
    anyhow::ensure!(
        manifest["source"] == "tessera-generator"
            && manifest["version"] == 3
            && manifest["template_policy"] == "source_backed_us_unit_parent_v1"
            && manifest["seed"] == cfg.seed
            && manifest["max_tokens"] == generate.max_tokens
            && manifest["names"].as_str() == cfg.names.as_ref().map(|names| names.out.as_str())
            && manifest["addresses"] == generate.addresses
            && manifest["exclude_gold"] == generate.exclude_gold
            && manifest["exclude_gold_sha256"] == hash_file(&repo_path(&generate.exclude_gold))?,
        "{version} generator manifest does not match configured exclusions"
    );
    let shard_hashes = manifest["synthetic_parquet_sha256"]
        .as_object()
        .with_context(|| format!("{version} generator manifest has no shard hashes"))?;
    anyhow::ensure!(
        shard_hashes.len() == 3,
        "{version} generator shard set changed"
    );
    for split in ["train", "valid", "test"] {
        let shard = repo_path(&cfg.data.processed).join(format!("{split}.parquet"));
        anyhow::ensure!(
            shard_hashes.get(split).and_then(serde_json::Value::as_str)
                == Some(hash_file(&shard)?.as_str()),
            "{version} generated {split} shard differs from its manifest"
        );
    }
    let silver_hashes = manifest["real_silver_sha256"]
        .as_object()
        .with_context(|| format!("{version} generator manifest has no silver hashes"))?;
    anyhow::ensure!(
        silver_hashes.len() == detector.silver.len(),
        "{version} generator silver source set changed"
    );
    let generation_sources = amended_sources.as_ref().unwrap_or(&detector.silver);
    for source in generation_sources {
        anyhow::ensure!(
            silver_hashes
                .get(source)
                .and_then(serde_json::Value::as_str)
                == Some(hash_file(&repo_path(source))?.as_str()),
            "{version} silver source {source} differs from its generator manifest"
        );
    }
    Ok(())
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
            let (gold, _) = real_dev_gold(cfg);
            let (manifest, errata, exclusions) = if uses_v5_gold(cfg) {
                (
                    REAL_DEV_MANIFEST_V3,
                    REAL_DEV_ERRATA_V3,
                    REAL_DEV_EXCLUSIONS_V5,
                )
            } else if uses_v4_gold(cfg) {
                (
                    REAL_DEV_MANIFEST_V3,
                    REAL_DEV_ERRATA_V3,
                    REAL_DEV_EXCLUSIONS_V4,
                )
            } else {
                (REAL_DEV_MANIFEST, REAL_DEV_ERRATA, REAL_DEV_EXCLUSIONS)
            };
            paths.extend([
                ("real_dev_gold".to_string(), repo_path(gold)),
                ("real_dev_manifest".to_string(), repo_path(manifest)),
                ("real_dev_errata".to_string(), repo_path(errata)),
                ("real_dev_exclusions".to_string(), repo_path(exclusions)),
            ]);
            if uses_v4_gold(cfg) || uses_v5_gold(cfg) {
                paths.extend([
                    ("real_dev_base".to_string(), repo_path(REAL_DEV_GOLD)),
                    (
                        "real_dev_base_manifest".to_string(),
                        repo_path(REAL_DEV_MANIFEST),
                    ),
                    (
                        "real_dev_base_errata".to_string(),
                        repo_path(REAL_DEV_ERRATA),
                    ),
                    (
                        "eval_exclusions_base".to_string(),
                        repo_path("data/interim/review/us-eval-exclusions-v3.jsonl"),
                    ),
                    (
                        "eval_exclusions_manifest".to_string(),
                        repo_path("data/interim/review/us-eval-exclusions-v4.manifest.json"),
                    ),
                    (
                        "generator_manifest".to_string(),
                        Path::new(&cfg.data.manifests).join("detector-synthetic.json"),
                    ),
                ]);
                paths.extend(
                    REAL_DEV_SOURCES
                        .iter()
                        .enumerate()
                        .map(|(i, source)| (format!("real_dev_source_{i}"), repo_path(source))),
                );
            } else {
                paths.extend(
                    REAL_DEV_SOURCES
                        .iter()
                        .enumerate()
                        .map(|(i, source)| (format!("real_dev_source_{i}"), repo_path(source))),
                );
            }
            if uses_v5_gold(cfg) {
                paths.extend([
                    ("test_shard".to_string(), processed.join("test.parquet")),
                    (
                        "v5_exclusions_manifest".to_string(),
                        repo_path(REAL_DEV_EXCLUSIONS_V5_MANIFEST),
                    ),
                    (
                        "v5_exclusions_builder".to_string(),
                        repo_path(REAL_DEV_EXCLUSIONS_V5_BUILDER),
                    ),
                    ("review_policy".to_string(), repo_path(REVIEW_POLICY)),
                ]);
                for (index, source) in REAL_DEV_EXCLUSIONS_V5_SOURCES.iter().enumerate() {
                    paths.push((format!("v5_exclusions_source_{index}"), repo_path(source)));
                    paths.push((
                        format!("v5_exclusions_source_manifest_{index}"),
                        repo_path(source).with_extension("manifest.json"),
                    ));
                }
            }
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

/// Includes all configured sources and reviewed typed-input dependencies.
pub(crate) fn checked_input_paths(cfg: &Config) -> anyhow::Result<Vec<(String, PathBuf)>> {
    let mut paths = input_paths(cfg);
    for (index, path) in crate::typed_synthetic::input_paths(cfg)?
        .into_iter()
        .enumerate()
    {
        paths.push((format!("typed_synthetic_input_{index:04}"), path));
    }
    Ok(paths)
}

pub(crate) fn capture_input_snapshot(cfg: &Config, run_dir: &Path) -> anyhow::Result<()> {
    let inputs = checked_input_paths(cfg)?
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
        decoder_contract: cfg.decoder_contract().to_string(),
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
    let cfg = crate::config::load(&run_dir.join("config.toml"))?;
    anyhow::ensure!(
        snapshot.tokenizer_contract == TOKENIZER_CONTRACT
            && snapshot.decoder_contract == cfg.decoder_contract(),
        "{} tokenizer or decoder contract differs from the training snapshot; retrain before export",
        run_dir.display()
    );
    anyhow::ensure!(
        snapshot.config_sha256 == hash_file(&run_dir.join("config.toml"))?,
        "{} config changed after input snapshot; retrain before export",
        run_dir.display()
    );
    let expected: std::collections::BTreeMap<_, _> = checked_input_paths(&cfg)?
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
    crate::diagnostic_operator::bind_deployable(run_dir, cfg)?;
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
            detector_epoch: None,
            learning_probe: None,
            max_steps: None,
            diagnostic_logit_penalty: None,
            forward_operator: crate::diagnostic_operator::ForwardOperator::Standard,
            completed_exposure: None,
        },
        &weights,
        device,
        |model| {
            let valid = quick_score(model, &valid_kept, &valid_items, batch_size, device);
            Ok((
                CheckpointScore::new(valid.component_f1, 0.0),
                serde_json::json!({
                    "valid_component_f1": valid.component_f1,
                    "valid_exact": valid.exact,
                }),
            ))
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
pub fn check_silver(config: &Path, context_windows: bool) -> anyhow::Result<()> {
    let cfg = crate::config::load(config)?;
    anyhow::ensure!(
        matches!(cfg.task, crate::config::Task::Detector),
        "check-silver requires a detector config"
    );
    let detector_cfg = cfg
        .detector
        .as_ref()
        .context("a detector config needs a [detector] section")?;
    let (docs, counts) = detector::load_silver(
        &detector_cfg.silver,
        &cfg.features.to_tessera(),
        SILVER_MAX_TOKENS,
    )?;
    if context_windows {
        let radius = cfg
            .net
            .dilations
            .iter()
            .try_fold(0usize, |radius, &dilation| {
                dilation
                    .checked_mul((cfg.net.kernel - 1) / 2)
                    .and_then(|padding| radius.checked_add(padding))
                    .context("detector context radius overflow")
            })?;
        let audit = crate::entity_context::audit(&docs, &counts.source_pieces, radius)?;
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "silver": counts,
                "source_paths": detector_cfg.silver,
                "context_windows": audit,
            }))?
        );
    } else {
        println!("{}", serde_json::to_string_pretty(&counts)?);
    }
    Ok(())
}

/// Validate the exact detector training inputs without creating a run or GPU state.
pub fn check_detector_data(
    config: &Path,
    context_views: bool,
    exposure: Option<(&Path, &Path)>,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        exposure.is_none() || !context_views,
        "planned exposure cannot substitute context views"
    );
    let cfg = crate::config::load(config)?;
    let exposure_request = exposure
        .map(|(targets, out)| crate::fullmix_exposure::Request::load(config, &cfg, targets, out))
        .transpose()?;
    anyhow::ensure!(
        matches!(cfg.task, crate::config::Task::Detector),
        "check-detector-data requires a detector config"
    );
    validate_ngram_source(&cfg)?;
    validate_training_scope(&cfg)?;
    verify_real_development_gold(&cfg)?;
    verify_generation_manifest(&cfg)?;
    let fc = cfg.features.to_tessera();
    let mut synthetic = BTreeMap::new();
    let mut training_lengths = Vec::new();
    let mut synthetic_train_docs = Vec::new();
    for split in Split::ALL {
        let (docs, counts) = detector::load_split(Path::new(&cfg.data.processed), split, &fc)?;
        if split == Split::Train {
            training_lengths.extend(docs.iter().map(|doc| doc.enc.token_spans.len()));
            synthetic_train_docs = docs;
        }
        synthetic.insert(
            split.name(),
            serde_json::json!({
                "documents": counts.encoded,
                "encoding": counts,
            }),
        );
    }
    let detector_cfg = cfg
        .detector
        .as_ref()
        .context("a detector config needs a [detector] section")?;
    let (silver_docs, silver) =
        detector::load_silver(&detector_cfg.silver, &fc, SILVER_MAX_TOKENS)?;
    let uniform_synthetic_count = synthetic_train_docs.len();
    let typed = crate::typed_synthetic::load(&cfg, &fc, &synthetic_train_docs, &silver_docs)?;
    anyhow::ensure!(
        typed.docs.is_empty() || (exposure_request.is_none() && !context_views),
        "typed authored inputs use the native data-check preview; frozen exposure/context routes are unchanged"
    );
    training_lengths.extend(typed.docs.iter().map(|doc| doc.enc.token_spans.len()));
    let typed_metadata = typed.metadata;
    let typed_pieces = typed.pieces;
    let typed_repeats = typed.repeats;
    synthetic_train_docs.extend(typed.docs);
    let exposure_plan = exposure_request
        .map(|request| {
            request.resolve(
                &cfg,
                &silver_docs,
                &silver.source_pieces,
                synthetic_train_docs.len(),
            )
        })
        .transpose()?;
    training_lengths.extend(silver_docs.iter().map(|doc| doc.enc.token_spans.len()));
    let synthetic_count = synthetic_train_docs.len();
    let source_repeats = if detector_cfg.silver_repeats.is_empty() {
        vec![detector_cfg.silver_repeat; detector_cfg.silver.len()]
    } else {
        detector_cfg.silver_repeats.clone()
    };
    let synthetic_per_epoch = detector_cfg
        .synthetic_per_epoch
        .unwrap_or(uniform_synthetic_count);
    let mut fixed_pieces = typed_pieces.clone();
    fixed_pieces.extend(&silver.source_pieces);
    let mut fixed_repeats = typed_repeats.clone();
    fixed_repeats.extend(source_repeats);
    let plan = DetectorEpochDraws {
        synthetic_count: uniform_synthetic_count,
        synthetic_per_epoch,
        source_pieces: fixed_pieces,
        source_repeats: fixed_repeats,
        seed: cfg.seed,
    };
    let draws_per_epoch = plan.len()?;
    anyhow::ensure!(
        training_lengths.len()
            == uniform_synthetic_count + plan.source_pieces.iter().sum::<usize>(),
        "detector token counts do not match encoded pieces"
    );
    let batches_per_epoch = draws_per_epoch.div_ceil(cfg.train.batch_size);
    let scheduled_steps = cfg
        .train
        .epochs
        .checked_mul(batches_per_epoch)
        .context("optimizer step count overflow")?;
    let lr_horizon = crate::training_diagnostic::horizon(&cfg, scheduled_steps);
    validate_learning_budget(&cfg, draws_per_epoch, None)?;
    let probe_manifest = if let Some(check) = &detector_cfg.learning_check {
        let probe = crate::learning_probe::select(
            &synthetic_train_docs,
            &silver_docs,
            &silver.source_pieces,
            &plan.indices(1)?,
            check.documents,
        )?;
        crate::learning_probe::validate_groups(&probe)?;
        Some(crate::learning_probe::manifest(&probe))
    } else {
        None
    };
    let views = if context_views {
        let radius = cfg
            .net
            .dilations
            .iter()
            .try_fold(0usize, |radius, &dilation| {
                dilation
                    .checked_mul((cfg.net.kernel - 1) / 2)
                    .and_then(|padding| radius.checked_add(padding))
                    .context("detector context radius overflow")
            })?;
        crate::context_views::prepare(&silver_docs, &fc, radius)?
    } else {
        Vec::new()
    };
    let original_count = synthetic_train_docs.len() + silver_docs.len();
    let mut replacements = vec![Vec::new(); silver_docs.len()];
    let view_descriptions: Vec<_> = views.iter().enumerate().map(|(index, view)| {
        replacements[view.parent].push(original_count + index);
        serde_json::json!({
            "parent_index": view.parent,
            "parent_text_sha256": crate::export::sha256_hex(silver_docs[view.parent].text.as_bytes()),
            "parent_byte_range": [view.start, view.end],
            "text": view.doc.text,
            "gold": view.doc.gold.iter().map(|span| serde_json::json!({
                "kind": detector::KINDS[span.kind].as_str(),
                "start": span.start,
                "end": span.end,
            })).collect::<Vec<_>>(),
            "content_tokens": view.doc.enc.labels.len(),
        })
    }).collect();
    let mut training_items: Vec<Encoded> = synthetic_train_docs
        .into_iter()
        .chain(silver_docs)
        .map(|doc| doc.enc)
        .collect();
    training_items.extend(views.into_iter().map(|view| view.doc.enc));
    let mut epoch_tokens = Vec::new();
    let mut epoch_exposure = Vec::new();
    let mut learning_batches = Vec::new();
    let mut context_epoch_exposure = Vec::new();
    let mut context_learning_batches = Vec::new();
    let learning_steps = detector_cfg
        .learning_check
        .as_ref()
        .map_or(0, |check| check.start_step);
    let uses_epoch_plan = !typed_metadata.is_empty()
        || detector_cfg.synthetic_per_epoch.is_some()
        || !detector_cfg.silver_repeats.is_empty();
    let mut legacy_order = if uses_epoch_plan {
        Vec::new()
    } else {
        detector_draw_indices(synthetic_count, silver.pieces, detector_cfg.silver_repeat)?
    };
    for epoch in 1..=cfg.train.epochs {
        let draws = if uses_epoch_plan {
            plan.indices(epoch)?
        } else {
            legacy_order.clone()
        };
        let mut synthetic_tokens = 0usize;
        let mut real_tokens = 0usize;
        for &index in &draws {
            let tokens = training_lengths[index];
            if index < synthetic_count {
                synthetic_tokens += tokens;
            } else {
                real_tokens += tokens;
            }
        }
        epoch_tokens.push(serde_json::json!({
            "epoch": epoch,
            "synthetic_content_tokens": synthetic_tokens,
            "real_content_tokens": real_tokens,
        }));
        let mut order = draws;
        order.shuffle(&mut ChaCha8Rng::seed_from_u64(cfg.seed ^ epoch as u64));
        let batches = if uses_epoch_plan {
            length_bucket_batches(
                &order,
                &training_lengths,
                cfg.train.batch_size,
                cfg.seed ^ epoch as u64,
            )
        } else {
            legacy_order = order.clone();
            order
                .chunks(cfg.train.batch_size)
                .map(<[usize]>::to_vec)
                .collect()
        };
        epoch_exposure.push(serde_json::json!({
            "epoch": epoch,
            "exposure": crate::training_audit::audit_epoch(
                &training_items, &batches, synthetic_count, &detector_cfg.class_weights)?,
        }));
        if context_views {
            let substituted = crate::context_sampling::substitute_batches(
                &batches,
                synthetic_count,
                original_count,
                &replacements,
                epoch - 1,
            )?;
            context_epoch_exposure.push(serde_json::json!({
                "epoch": epoch,
                "exposure": crate::training_audit::audit_epoch(
                    &training_items, &substituted, synthetic_count, &detector_cfg.class_weights)?,
            }));
            context_learning_batches.extend(
                substituted
                    .into_iter()
                    .take(learning_steps.saturating_sub(context_learning_batches.len())),
            );
        }
        learning_batches.extend(
            batches
                .into_iter()
                .take(learning_steps.saturating_sub(learning_batches.len())),
        );
    }
    let learning_exposure = if learning_steps > 0 {
        let mut presentations = BTreeMap::<usize, usize>::new();
        for &index in learning_batches.iter().flatten() {
            *presentations.entry(index).or_default() += 1;
        }
        Some(serde_json::json!({
            "optimizer_steps": learning_batches.len(),
            "exposure": crate::training_audit::audit_epoch(
                &training_items, &learning_batches, synthetic_count, &detector_cfg.class_weights)?,
            "max_real_piece_presentations": presentations.iter().filter(|(index, _)| **index >= synthetic_count)
                .map(|(_, count)| *count).max().unwrap_or(0),
            "max_synthetic_piece_presentations": presentations.iter().filter(|(index, _)| **index < synthetic_count)
                .map(|(_, count)| *count).max().unwrap_or(0),
        }))
    } else {
        None
    };
    let context_simulation = if context_views {
        Some(serde_json::json!({
            "training_enabled": false,
            "scope": "candidate views and hypothetical loss coefficients only; not predictions, gradients, or accuracy",
            "substitution": "alternate whole/view per eligible parent, within existing parent budgets and finalized original batches",
            "address_policy": "every parent containing ADDRESS gold remains whole",
            "probe_policy": "unchanged original whole-document probe selected before any views",
            "candidate_views": view_descriptions,
            "epoch_training_exposure": context_epoch_exposure,
            "learning_check_exposure": if learning_steps > 0 {
                Some(crate::training_audit::audit_epoch(
                    &training_items, &context_learning_batches, synthetic_count, &detector_cfg.class_weights)?)
            } else { None },
        }))
    } else {
        None
    };
    if let Some(exposure_plan) = exposure_plan {
        exposure_plan.write(
            &cfg,
            &training_items,
            &learning_batches,
            synthetic_count,
            lr_horizon,
            learning_exposure
                .as_ref()
                .context("planned exposure needs the existing learning audit")?,
        )?;
    }
    let typed_preview = if typed_metadata.is_empty() {
        None
    } else {
        Some(crate::typed_synthetic::preview(
            &cfg,
            &training_items,
            &learning_batches,
            uniform_synthetic_count,
            synthetic_count,
            &typed_metadata,
            lr_horizon,
        )?)
    };
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "synthetic": synthetic,
            "silver": silver,
            "typed_synthetic": typed_preview,
            "uniform_synthetic_pool": uniform_synthetic_count,
            "full_synthetic_boundary": synthetic_count,
            "input_snapshot_paths": crate::typed_synthetic::input_paths(&cfg)?,
            "draws_per_epoch": draws_per_epoch,
            "synthetic_draws_per_epoch": synthetic_per_epoch + typed_pieces.iter().zip(&typed_repeats).map(|(n,r)| n*r).sum::<usize>(),
            "real_draws_per_epoch": silver.source_pieces.iter().zip(&plan.source_repeats[typed_pieces.len()..]).map(|(n,r)| n*r).sum::<usize>(),
            "batches_per_epoch": batches_per_epoch,
            "scheduled_optimizer_steps": scheduled_steps,
            "learning_rate_horizon_steps": lr_horizon,
            "warmup_steps": cfg.train.warmup_steps,
            "epoch_content_tokens": epoch_tokens,
            "epoch_training_exposure": epoch_exposure,
            "detector_initialization": "fresh detector; optional parser n-gram transfer only",
            "ngram_trainable": cfg.net.ngram_from.is_none() || cfg.net.finetune_ngram,
            "scheduled_learning_rate_sum_scope": "hypothetical configured epochs; not the bounded diagnostic budget",
            "scheduled_learning_rate_sum": (0..scheduled_steps)
                .map(|step| lr_at(step, lr_horizon,
                    cfg.train.learning_rate, cfg.train.warmup_steps)).sum::<f64>(),
            "learning_check": detector_cfg.learning_check,
            "learning_probe": probe_manifest,
            "learning_check_exposure": learning_exposure,
            "learning_check_learning_rate_sum": (0..learning_steps)
                .map(|step| lr_at(step, lr_horizon, cfg.train.learning_rate,
                    cfg.train.warmup_steps)).sum::<f64>(),
            "config_sha256": hash_file(config)?,
            "context_view_simulation": context_simulation,
            "scope": "input validity and planned supervision only; passing does not prove learning or 95% quality",
        }))?
    );
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

struct DetectorEpochDraws {
    synthetic_count: usize,
    synthetic_per_epoch: usize,
    source_pieces: Vec<usize>,
    source_repeats: Vec<usize>,
    seed: u64,
}

impl DetectorEpochDraws {
    fn len(&self) -> anyhow::Result<usize> {
        anyhow::ensure!(
            self.synthetic_per_epoch > 0 && self.synthetic_per_epoch <= self.synthetic_count,
            "detector synthetic_per_epoch exceeds the prepared train shard"
        );
        anyhow::ensure!(
            self.source_pieces.len() == self.source_repeats.len(),
            "detector silver source counts and repeats differ"
        );
        self.source_pieces
            .iter()
            .zip(&self.source_repeats)
            .try_fold(self.synthetic_per_epoch, |total, (&pieces, &repeat)| {
                pieces
                    .checked_mul(repeat)
                    .and_then(|count| total.checked_add(count))
                    .context("detector draw count overflow")
            })
    }

    fn indices(&self, epoch: usize) -> anyhow::Result<Vec<usize>> {
        let mut draws = Vec::with_capacity(self.len()?);
        let mut synthetic: Vec<usize> = (0..self.synthetic_count).collect();
        if self.synthetic_per_epoch < self.synthetic_count {
            synthetic.shuffle(&mut ChaCha8Rng::seed_from_u64(
                self.seed ^ epoch as u64 ^ 0x9e37_79b9_7f4a_7c15,
            ));
            synthetic.truncate(self.synthetic_per_epoch);
        }
        draws.extend(synthetic);
        let mut offset = self.synthetic_count;
        for (&pieces, &repeat) in self.source_pieces.iter().zip(&self.source_repeats) {
            for _ in 0..repeat {
                draws.extend(offset..offset + pieces);
            }
            offset += pieces;
        }
        Ok(draws)
    }
}

/// Trains the detector on synthetic and reviewed real documents. The lowest exact F1 among
/// person, org, and address on pinned real development cases selects the checkpoint. Tied
/// minimum scores use real development macro F1 so an early zero does not stall selection.
fn train_detector<B: AutodiffBackend>(
    cfg: &Config,
    run_dir: &Path,
    max_steps: Option<usize>,
    fullmix: Option<&crate::fullmix_rms::Request>,
    device: &B::Device,
) -> anyhow::Result<()> {
    match fullmix {
        Some(request) => request.validate_authored_route(cfg)?,
        None => crate::typed_synthetic::refuse_fit(cfg)?,
    }
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
    let (mut train_docs, train_counts) = detector::load_split(&dir, Split::Train, &fc)?;
    let (valid_docs, valid_counts) = detector::load_split(&dir, Split::Valid, &fc)?;
    let (gold_path, gold_sha256) = real_dev_gold(cfg);
    let real_dev_docs = detector::load_development_gold(&repo_path(gold_path), &fc)?;
    anyhow::ensure!(
        !valid_docs.is_empty(),
        "detector validation has no encoded documents"
    );
    anyhow::ensure!(
        real_dev_docs.len() == 196,
        "pinned real development set must have 196 cases"
    );
    eprintln!("train {train_counts:?}, valid {valid_counts:?}");
    let (silver_docs, silver_counts) =
        detector::load_silver(&detector_cfg.silver, &fc, SILVER_MAX_TOKENS)?;
    let uniform_synthetic_count = train_docs.len();
    let typed = crate::typed_synthetic::load(cfg, &fc, &train_docs, &silver_docs)?;
    let typed_metadata = typed.metadata;
    let mut fixed_pieces = typed.pieces;
    let mut fixed_repeats = typed.repeats;
    train_docs.extend(typed.docs);
    let synthetic_count = train_docs.len();
    let source_repeats = if detector_cfg.silver_repeats.is_empty() {
        vec![detector_cfg.silver_repeat; detector_cfg.silver.len()]
    } else {
        detector_cfg.silver_repeats.clone()
    };
    fixed_pieces.extend(&silver_counts.source_pieces);
    fixed_repeats.extend(&source_repeats);
    let synthetic_per_epoch = detector_cfg
        .synthetic_per_epoch
        .unwrap_or(uniform_synthetic_count);
    let epoch_plan = if detector_cfg.synthetic_per_epoch.is_some()
        || !detector_cfg.silver_repeats.is_empty()
    {
        let plan = DetectorEpochDraws {
            synthetic_count: uniform_synthetic_count,
            synthetic_per_epoch,
            source_pieces: fixed_pieces,
            source_repeats: fixed_repeats,
            seed: cfg.seed,
        };
        anyhow::ensure!(
            plan.source_pieces.iter().sum::<usize>() == typed_metadata.len() + silver_docs.len(),
            "detector silver source pieces do not match encoded pieces"
        );
        plan.len()?;
        Some(plan)
    } else {
        None
    };
    let draws = if epoch_plan.is_some() {
        Vec::new()
    } else {
        detector_draw_indices(
            synthetic_count,
            silver_docs.len(),
            detector_cfg.silver_repeat,
        )?
    };
    let learning_probe = if let Some(check) = &detector_cfg.learning_check {
        let first_draws = match &epoch_plan {
            Some(plan) => plan.indices(1)?,
            None => draws.clone(),
        };
        validate_learning_budget(cfg, first_draws.len(), max_steps)?;
        Some(crate::learning_probe::select(
            &train_docs,
            &silver_docs,
            &silver_counts.source_pieces,
            &first_draws,
            check.documents,
        )?)
    } else {
        None
    };
    if let Some(probe) = &learning_probe {
        crate::learning_probe::validate_groups(probe)?;
        std::fs::write(
            run_dir.join("learning_probe.json"),
            serde_json::to_string_pretty(&crate::learning_probe::manifest(probe))? + "\n",
        )?;
    }
    if let Some(request) = fullmix {
        request.verify_training_membership(
            cfg,
            &train_docs[uniform_synthetic_count..],
            &silver_docs,
            &typed_metadata,
        )?;
        request.verify_probe_against_typed_preview(run_dir)?;
    }
    let actual_probe = fullmix
        .map(|request| request.verify_probe(run_dir))
        .transpose()?;
    let completed_exposure = fullmix
        .filter(|request| !request.preflight || request.hard_objective().is_some())
        .map(|request| {
            request.accounting(
                cfg,
                run_dir,
                &silver_docs,
                &train_docs,
                &silver_counts.source_pieces,
                synthetic_count,
                &typed_metadata,
            )
        })
        .transpose()?;
    let forward_operator = if fullmix.is_some() {
        fullmix
            .context("missing bounded RMS request")?
            .forward_operator()?
    } else {
        crate::diagnostic_operator::deployable(cfg)
    };
    let mut train_items: Vec<Encoded> = train_docs.into_iter().map(|d| d.enc).collect();
    if !silver_docs.is_empty() {
        eprintln!(
            "silver {silver_counts:?}, source repeats {source_repeats:?}, synthetic per epoch {synthetic_per_epoch}"
        );
        train_items.extend(silver_docs.into_iter().map(|d| d.enc));
    }
    let valid_gold: Vec<Vec<KindSpan>> = valid_docs.iter().map(|d| d.gold.clone()).collect();
    let real_dev_gold: Vec<Vec<KindSpan>> = real_dev_docs.iter().map(|d| d.gold.clone()).collect();
    let mut model = cfg.detector_net_config().init::<B>(device);
    if let Some(run) = &cfg.net.ngram_from {
        model.ngram = pretrained_ngram::<B>(cfg, Path::new(run), device)?;
    }
    if let Some(request) = fullmix {
        let tensors = crate::quantize::extract(&model.valid(), cfg.net_name());
        let initial = crate::training_diagnostic::parameter_sha256(&tensors);
        let probe = actual_probe
            .as_deref()
            .context("fullmix probe not initialized")?;
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        model
            .clone()
            .save_file(run_dir.join("checkpoints/diagnostic-step-0"), &recorder)?;
        model.clone().save_file(run_dir.join("best"), &recorder)?;
        request.initialized(run_dir, &initial, probe)?;
        if request.preflight {
            let restored = model
                .clone()
                .load_file(run_dir.join("best"), &recorder, device)?;
            let restored_tensors = crate::quantize::extract(&restored.valid(), cfg.net_name());
            anyhow::ensure!(
                crate::training_diagnostic::parameter_sha256(&restored_tensors) == initial,
                "native preflight checkpoint round-trip changes initialization"
            );
            crate::diagnostic_operator::load(run_dir, cfg)?;
        }
        crate::fullmix_rms::checkpoint(
            &model.valid(),
            cfg,
            run_dir,
            &run_dir.join("checkpoints/diagnostic-step-0.mpk"),
            0,
            crate::fullmix_rms::CheckpointKind::Snapshot,
        )?;
        crate::fullmix_rms::checkpoint(
            &model.valid(),
            cfg,
            run_dir,
            &run_dir.join("best.mpk"),
            0,
            crate::fullmix_rms::CheckpointKind::DevelopmentSelected,
        )?;
        if request.preflight {
            request.write_preflight(cfg, run_dir, &initial, probe)?;
            return Ok(());
        }
    }
    let batch_size = cfg.train.batch_size.max(1);
    let fit = fit::<B>(
        cfg,
        run_dir,
        model,
        TrainingData {
            items: &train_items,
            draws,
            detector_epoch: epoch_plan,
            learning_probe: learning_probe.as_ref(),
            max_steps,
            diagnostic_logit_penalty: None,
            forward_operator,
            completed_exposure,
        },
        &detector_cfg.class_weights,
        device,
        |model| {
            let valid_pred =
                diagnostic_predictions(model, &valid_docs, batch_size, device, forward_operator)?;
            let valid_scores = detector::score(&valid_gold, &valid_pred);
            let real_dev_pred = diagnostic_predictions(
                model,
                &real_dev_docs,
                batch_size,
                device,
                forward_operator,
            )?;
            let real_dev_scores = detector::score(&real_dev_gold, &real_dev_pred);
            let min_f1 = minimum_model_exact_f1(&real_dev_scores);
            eprintln!(
                "real dev exact F1: person {:.4}, org {:.4}, address {:.4}; minimum {:.4}; synthetic valid macro {:.4}",
                real_dev_scores.per_kind["person"].exact.f1,
                real_dev_scores.per_kind["org"].exact.f1,
                real_dev_scores.per_kind["address"].exact.f1,
                min_f1,
                valid_scores.macro_exact_f1,
            );
            Ok((
                CheckpointScore::new(min_f1, real_dev_scores.macro_exact_f1),
                serde_json::json!({
                    "real_dev": real_dev_scores,
                    "real_dev_min_exact_f1": min_f1,
                    "valid": valid_scores,
                }),
            ))
        },
    )?;
    let selected_learning_check = if max_steps.is_none()
        && let (Some(check), Some(probe)) = (&detector_cfg.learning_check, &learning_probe)
    {
        let selected = crate::quantize::load_best::<B::InnerBackend>(run_dir, cfg, device)?;
        let predicted =
            diagnostic_predictions(&selected, probe, batch_size, device, forward_operator)?;
        let gold: Vec<_> = probe.iter().map(|doc| doc.gold.clone()).collect();
        let mut groups = BTreeMap::new();
        for (group, range) in [
            ("real", 0..probe.real_count),
            ("synthetic", probe.real_count..probe.len()),
        ] {
            let scores = detector::score(&gold[range.clone()], &predicted[range]);
            groups.insert(
                group,
                crate::learning_gate::evaluate(
                    &scores,
                    check.minimum_recall,
                    fit.steps,
                    check.start_step,
                )?,
            );
        }
        let passed = groups.values().all(|check| check.enforced && check.passed);
        std::fs::write(
            run_dir.join("best_learning_check.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "checkpoint": "best.mpk",
                "checkpoint_sha256": hash_file(&run_dir.join("best.mpk"))?,
                "config_sha256": hash_file(&run_dir.join("config.toml"))?,
                "passed": passed,
                "source_groups": groups,
                "scope": "selected checkpoint learning on seen training examples; not release quality",
            }))? + "\n",
        )?;
        anyhow::ensure!(
            passed,
            "selected best checkpoint failed the training learning check"
        );
        Some(passed)
    } else {
        None
    };
    let summary = serde_json::json!({
        "best_epoch": fit.best_epoch,
        "best_real_dev_min_exact_f1": fit.best_score,
        "best_real_dev": fit.best_metrics["real_dev"],
        "best_valid": fit.best_metrics["valid"],
        "real_dev_cases": real_dev_docs.len(),
        "real_dev_gold_sha256": gold_sha256,
        "wall_clock_seconds": fit.seconds,
        "train": train_counts,
        "valid": valid_counts,
        "silver": silver_counts,
        "silver_repeat": detector_cfg.silver_repeat,
        "silver_repeats": source_repeats,
        "synthetic_per_epoch": synthetic_per_epoch,
        "epochs": cfg.train.epochs,
        "steps": fit.steps,
        "training_complete": max_steps.is_none(),
        "learning_check_passed": if max_steps.is_some() { fit.learning_check_passed } else { selected_learning_check },
        "learning_check_scope": if max_steps.is_some() { "final bounded checkpoint" } else { "selected best checkpoint" },
        "last_training_learning_check_passed": fit.learning_check_passed,
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

fn minimum_model_exact_f1(scores: &detector::SpanScores) -> f64 {
    ["person", "org", "address"]
        .iter()
        .map(|kind| scores.per_kind[*kind].exact.f1)
        .fold(f64::INFINITY, f64::min)
}

/// Initialize the detector from the parser's n-gram table. Its features must match, or the
/// same ids would mean different n-grams. The default freezes it to allow bundle sharing.
fn diagnostic_predictions<B: Backend>(
    model: &TaggerNet<B>,
    docs: &[detector::DetectorDoc],
    batch_size: usize,
    device: &B::Device,
    operator: crate::diagnostic_operator::ForwardOperator,
) -> anyhow::Result<Vec<Vec<KindSpan>>> {
    if operator == crate::diagnostic_operator::ForwardOperator::Standard {
        Ok(detector::predict(model, docs, batch_size, device))
    } else {
        let scored =
            detector::predict_scored_with_operator(model, docs, batch_size, device, operator)?;
        Ok(detector::apply_confidence_policy(&scored))
    }
}

fn pretrained_ngram<B: AutodiffBackend>(
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
    if cfg.net.finetune_ngram {
        eprintln!("n-gram table from {}, trainable", run.display());
        Ok(parser.ngram)
    } else {
        eprintln!("n-gram table from {}, frozen", run.display());
        Ok(parser.ngram.no_grad())
    }
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
    learning_check_passed: Option<bool>,
}

fn validate_learning_budget(
    cfg: &Config,
    draws_per_epoch: usize,
    max_steps: Option<usize>,
) -> anyhow::Result<()> {
    if let Some(check) = cfg
        .detector
        .as_ref()
        .and_then(|d| d.learning_check.as_ref())
    {
        let batches = draws_per_epoch.div_ceil(cfg.train.batch_size);
        let scheduled = cfg
            .train
            .epochs
            .checked_mul(batches)
            .context("optimizer step count overflow")?;
        anyhow::ensure!(
            check.start_step > batches,
            "learning check must start after every first-epoch probe document has been trained on"
        );
        anyhow::ensure!(
            max_steps.unwrap_or(scheduled).min(scheduled) >= check.start_step,
            "training schedule ends before the enforced learning check"
        );
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
struct CheckpointScore {
    primary: f64,
    secondary: f64,
}

impl CheckpointScore {
    fn new(primary: f64, secondary: f64) -> Self {
        Self { primary, secondary }
    }

    fn is_finite(self) -> bool {
        self.primary.is_finite() && self.secondary.is_finite()
    }
}

struct TrainingData<'a> {
    items: &'a [Encoded],
    draws: Vec<usize>,
    detector_epoch: Option<DetectorEpochDraws>,
    learning_probe: Option<&'a crate::learning_probe::SelectedProbe>,
    max_steps: Option<usize>,
    diagnostic_logit_penalty: Option<f64>,
    forward_operator: crate::diagnostic_operator::ForwardOperator,
    completed_exposure: Option<crate::fullmix_rms::Accounting>,
}

fn length_bucket_batches(
    order: &[usize],
    lengths: &[usize],
    batch_size: usize,
    seed: u64,
) -> Vec<Vec<usize>> {
    let pool_size = batch_size.saturating_mul(64).max(batch_size);
    let mut batches = Vec::with_capacity(order.len().div_ceil(batch_size));
    for pool in order.chunks(pool_size) {
        let mut sorted = pool.to_vec();
        sorted.sort_by_key(|&index| lengths[index]);
        batches.extend(sorted.chunks(batch_size).map(<[usize]>::to_vec));
    }
    batches.shuffle(&mut ChaCha8Rng::seed_from_u64(seed));
    batches
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
    mut validate: impl FnMut(
        &TaggerNet<B::InnerBackend>,
    ) -> anyhow::Result<(CheckpointScore, serde_json::Value)>,
) -> anyhow::Result<FitSummary> {
    let TrainingData {
        items,
        draws: mut order,
        detector_epoch,
        learning_probe,
        max_steps,
        diagnostic_logit_penalty,
        forward_operator,
        mut completed_exposure,
    } = data;
    if matches!(
        forward_operator,
        crate::diagnostic_operator::ForwardOperator::ResidualRmsV1
            | crate::diagnostic_operator::ForwardOperator::ContextRmsV2
    ) && completed_exposure.is_some()
        || forward_operator == crate::diagnostic_operator::ForwardOperator::ResidualRmsV1
    {
        let whole = max_steps == Some(2000)
            && diagnostic_logit_penalty.is_none()
            && detector_epoch.is_none()
            && learning_probe.is_none()
            && items.len() == 15
            && completed_exposure.is_none();
        let mixed = max_steps == Some(crate::fullmix_rms::STEPS)
            && diagnostic_logit_penalty.is_none()
            && detector_epoch.is_some()
            && learning_probe.is_some()
            && completed_exposure.is_some();
        anyhow::ensure!(
            whole || mixed,
            "residual RMS is outside its explicit bounded diagnostic loop"
        );
        anyhow::ensure!(
            crate::diagnostic_operator::load(run_dir, cfg)? == forward_operator,
            "training forward operator metadata differs"
        );
    } else {
        anyhow::ensure!(
            completed_exposure.is_none(),
            "standard fit refuses mixed RMS accounting"
        );
        anyhow::ensure!(
            crate::diagnostic_operator::load(run_dir, cfg)? == forward_operator,
            "standard training refuses a candidate forward operator"
        );
    }
    if let Some(coefficient) = diagnostic_logit_penalty {
        crate::loss_stability::validate(coefficient, None)?;
    }
    if let Some(coefficient) = diagnostic_logit_penalty.filter(|value| *value > 0.0) {
        let marker: serde_json::Value =
            serde_json::from_slice(&std::fs::read(run_dir.join("diagnostic.json"))?)?;
        anyhow::ensure!(
            max_steps.is_some()
                && marker["scope"] == "fixed training-only memorization diagnostic; never export"
                && marker["logit_penalty"].as_f64() == Some(coefficient),
            "logit penalty requires a bounded training-only memorization diagnostic"
        );
    }
    anyhow::ensure!(cfg.train.epochs > 0, "training requires at least one epoch");
    let epoch_draw_count = match &detector_epoch {
        Some(plan) => plan.len()?,
        None => order.len(),
    };
    anyhow::ensure!(epoch_draw_count > 0, "training has no examples");
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
    let mut optim_config = AdamWConfig::new().with_weight_decay(0.01);
    if let Some(norm) = cfg.train.gradient_clip_norm {
        optim_config = optim_config.with_grad_clipping(Some(GradientClippingConfig::Norm(norm)));
    }
    let mut optim = optim_config.init::<B, TaggerNet<B>>();
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    let batch_size = cfg.train.batch_size.max(1);
    let scheduled_steps = cfg
        .train
        .epochs
        .checked_mul(epoch_draw_count.div_ceil(batch_size))
        .context("optimizer step count overflow")?;
    anyhow::ensure!(
        max_steps.is_none_or(|cap| cap <= scheduled_steps),
        "training epochs cannot reach the diagnostic update limit"
    );
    crate::training_diagnostic::validate_launch(cfg, max_steps)?;
    let total_steps = crate::training_diagnostic::horizon(cfg, scheduled_steps);
    if max_steps.is_some() {
        let tensors = crate::quantize::extract(&model.valid(), cfg.net_name());
        std::fs::write(
            run_dir.join("diagnostic_initial.json"),
            serde_json::to_string_pretty(&serde_json::json!({
                "parameter_sha256": crate::training_diagnostic::parameter_sha256(&tensors),
                "learning_rate_horizon_steps": total_steps,
                "scheduled_optimizer_steps": scheduled_steps,
                "max_steps": max_steps,
            }))? + "\n",
        )?;
    }
    let mut exposure = crate::memorization::exposure::Exposure::new(
        cfg,
        run_dir,
        items,
        &order,
        max_steps,
        detector_epoch.is_some(),
    )?;
    if let Some(accounting) = &exposure {
        model
            .clone()
            .save_file(run_dir.join("checkpoints/diagnostic-step-0"), &recorder)?;
        accounting.write(run_dir, items, weights)?;
    }
    let mut metrics = File::create(run_dir.join("metrics.jsonl"))?;
    let mut fullmix_progress = completed_exposure
        .as_ref()
        .map(|_| File::create(run_dir.join("fullmix-progress.jsonl")))
        .transpose()?;
    if let Some(accounting) = &completed_exposure {
        accounting.write(weights)?;
    }
    let mut probe_metrics = learning_probe
        .map(|_| File::create(run_dir.join("learning_metrics.jsonl")))
        .transpose()?;
    let probe_gold =
        learning_probe.map(|docs| docs.iter().map(|doc| doc.gold.clone()).collect::<Vec<_>>());
    let learning_check = cfg
        .detector
        .as_ref()
        .and_then(|d| d.learning_check.as_ref());
    anyhow::ensure!(
        learning_check.is_some() == learning_probe.is_some(),
        "learning check and training probe must be configured together"
    );
    let mut last_learning_check = None;
    let mut best = (
        CheckpointScore::new(f64::MIN, f64::MIN),
        0usize,
        serde_json::Value::Null,
    );
    let (mut since_best, mut step) = (0usize, 0usize);
    let started = std::time::Instant::now();
    let detector_lengths = detector_epoch.as_ref().map(|_| {
        items
            .iter()
            .map(|item| item.token_spans.len())
            .collect::<Vec<_>>()
    });

    for epoch in 1..=cfg.train.epochs {
        if let Some(plan) = &detector_epoch {
            order = plan.indices(epoch)?;
            anyhow::ensure!(
                order.iter().all(|&index| index < items.len()),
                "detector epoch draw index is outside the encoded examples"
            );
        }
        order.shuffle(&mut ChaCha8Rng::seed_from_u64(cfg.seed ^ epoch as u64));
        let epoch_batches = match &detector_lengths {
            Some(lengths) => {
                length_bucket_batches(&order, lengths, batch_size, cfg.seed ^ epoch as u64)
            }
            None => order.chunks(batch_size).map(<[usize]>::to_vec).collect(),
        };
        let (mut epoch_loss, mut batches) = (0f64, 0usize);
        let (mut epoch_cross_entropy, mut epoch_penalty) = (0f64, 0f64);
        let mut reached_limit = false;
        for chunk in epoch_batches {
            let scheduled_lr = lr_at(
                step,
                total_steps,
                cfg.train.learning_rate,
                cfg.train.warmup_steps,
            );
            let hard_batch = completed_exposure
                .as_ref()
                .map(|accounting| accounting.prepare_hard(&chunk, weights, scheduled_lr))
                .transpose()?
                .flatten();
            if let Some(accounting) = &completed_exposure
                && hard_batch.is_none()
            {
                accounting.verify_next(cfg, items, &chunk, weights, scheduled_lr)?;
            }
            let batch_items: Vec<Encoded> = chunk.iter().map(|&i| items[i].clone()).collect();
            let batch: ParserBatch<B> = ParserBatcher.batch(batch_items, device);
            let logits = forward_operator.forward(
                &model,
                batch.ngram_ids,
                batch.script,
                batch.shape,
                batch.flags,
                batch.mask.clone(),
            )?;
            let (loss, components) = if let Some(prepared) = &hard_batch {
                anyhow::ensure!(
                    diagnostic_logit_penalty.is_none(),
                    "hard objective refuses a second penalty"
                );
                (crate::hard_token_batch::loss(logits, prepared)?, None)
            } else if let Some(coefficient) = diagnostic_logit_penalty.filter(|value| *value > 0.0)
            {
                let cross_entropy = masked_loss(
                    logits.clone(),
                    batch.labels.clone(),
                    batch.mask.clone(),
                    class_weights.clone(),
                );
                let penalty = crate::loss_stability::centered_penalty(
                    logits,
                    batch.labels,
                    batch.mask,
                    class_weights.clone(),
                );
                let ce_value: f64 = cross_entropy.clone().into_scalar().elem();
                let penalty_value: f64 = penalty.clone().into_scalar().elem();
                anyhow::ensure!(
                    ce_value.is_finite() && penalty_value.is_finite(),
                    "epoch {epoch} step {step}: non-finite diagnostic loss component"
                );
                (
                    cross_entropy + penalty * coefficient,
                    Some((ce_value, penalty_value)),
                )
            } else {
                (
                    masked_loss(logits, batch.labels, batch.mask, class_weights.clone()),
                    None,
                )
            };
            let value: f64 = loss.clone().into_scalar().elem();
            anyhow::ensure!(
                value.is_finite() && value < 100.0,
                "epoch {epoch} step {step}: loss {value} is non-finite or runaway"
            );
            let grads = GradientsParams::from_grads(loss.backward(), &model);
            let lr = lr_at(
                step,
                total_steps,
                cfg.train.learning_rate,
                cfg.train.warmup_steps,
            );
            model = optim.step(lr, model, grads);
            if let Some(accounting) = &mut completed_exposure {
                if let Some(prepared) = &hard_batch {
                    accounting.record_hard(prepared)?;
                } else {
                    accounting.record(cfg, items, &chunk, weights, lr)?;
                }
            }
            if let Some(accounting) = &mut exposure {
                accounting.record(&chunk, items, weights, lr)?;
            }
            epoch_loss += value;
            if let Some((cross_entropy, penalty)) = components {
                epoch_cross_entropy += cross_entropy;
                epoch_penalty += penalty;
            }
            batches += 1;
            step += 1;
            if let Some(progress) = &mut fullmix_progress {
                writeln!(
                    progress,
                    "{}",
                    serde_json::to_string(&{
                        let mut record = serde_json::json!({"optimizer_updates_executed":step,
                            "epoch":epoch,"scheduled_learning_rate":lr,"elapsed_seconds":started.elapsed().as_secs_f64()});
                        record[if hard_batch.is_some() {
                            "native_batch_objective"
                        } else {
                            "native_batch_cross_entropy"
                        }] = serde_json::json!(value);
                        record
                    })?
                )?;
                progress.flush()?;
                if step == 1 || step.is_multiple_of(25) {
                    eprintln!(
                        "fullmix RMS step {step}/4000, native CE {value:.5}, elapsed {:.0}s",
                        started.elapsed().as_secs_f64()
                    );
                }
            }
            if max_steps == Some(step)
                || (completed_exposure.is_some() && crate::fullmix_rms::SNAPSHOTS.contains(&step))
                || (exposure.is_some()
                    && crate::memorization::exposure::OBSERVATIONS.contains(&step))
            {
                model.clone().save_file(
                    run_dir.join(format!("checkpoints/diagnostic-step-{step}")),
                    &recorder,
                )?;
                if let Some(accounting) = &exposure {
                    accounting.write(run_dir, items, weights)?;
                }
                if let Some(accounting) = &completed_exposure {
                    if hard_batch.is_some() {
                        accounting.write(weights)?;
                    }
                    crate::fullmix_rms::checkpoint(
                        &model.valid(),
                        cfg,
                        run_dir,
                        &run_dir.join(format!("checkpoints/diagnostic-step-{step}.mpk")),
                        step,
                        crate::fullmix_rms::CheckpointKind::Snapshot,
                    )?;
                }
            }
            if let (Some(check), Some(probe), Some(gold), Some(log)) = (
                learning_check,
                learning_probe,
                &probe_gold,
                &mut probe_metrics,
            ) && (step.is_multiple_of(check.every_steps)
                || max_steps == Some(step)
                || step == total_steps)
            {
                let candidates = detector::predict_scored_with_operator(
                    &model.valid(),
                    probe,
                    batch_size,
                    device,
                    forward_operator,
                )?;
                let predicted = detector::apply_confidence_policy(&candidates);
                let unfiltered: Vec<Vec<KindSpan>> = candidates
                    .iter()
                    .map(|document| document.iter().map(|candidate| candidate.span).collect())
                    .collect();
                let mut group_results = BTreeMap::new();
                let mut candidate_results = BTreeMap::new();
                let mut filtered_results = BTreeMap::new();
                for (group, range) in [
                    ("real", 0..probe.real_count),
                    ("synthetic", probe.real_count..probe.len()),
                ] {
                    let scores = detector::score(&gold[range.clone()], &predicted[range.clone()]);
                    let candidate_scores =
                        detector::score(&gold[range.clone()], &unfiltered[range.clone()]);
                    let diagnostic = crate::learning_gate::evaluate(
                        &scores,
                        check.minimum_recall,
                        step,
                        check.start_step,
                    )?;
                    eprintln!(
                        "training candidates {group} step {step}: person precision {:.3} recall {:.3}, org precision {:.3} recall {:.3}, address precision {:.3} recall {:.3}; before confidence filtering, not the stopping criterion",
                        candidate_scores.per_kind["person"].exact.precision,
                        candidate_scores.per_kind["person"].exact.recall,
                        candidate_scores.per_kind["org"].exact.precision,
                        candidate_scores.per_kind["org"].exact.recall,
                        candidate_scores.per_kind["address"].exact.precision,
                        candidate_scores.per_kind["address"].exact.recall,
                    );
                    eprintln!(
                        "training probe {group} step {step}: person recall {:.3}, org {:.3}, address {:.3}; enforced {}, passed {}",
                        scores.per_kind["person"].exact.recall,
                        scores.per_kind["org"].exact.recall,
                        scores.per_kind["address"].exact.recall,
                        diagnostic.enforced,
                        diagnostic.passed
                    );
                    group_results.insert(group, diagnostic);
                    candidate_results.insert(group, candidate_scores);
                    if completed_exposure.is_some() {
                        filtered_results.insert(group, scores);
                    }
                }
                let passed = group_results.values().all(|diagnostic| diagnostic.passed);
                let enforced = step >= check.start_step;
                let mut probe_record = serde_json::json!({
                    "step": step, "source_groups": group_results,
                    "candidate_source_groups": candidate_results,
                    "candidate_scope": "unfiltered decoded exact-span precision/recall/F1; diagnostic only, no change to confidence policy or learning enforcement",
                    "passed": passed, "enforced": enforced,
                });
                if completed_exposure.is_some() {
                    probe_record["filtered_source_groups"] =
                        serde_json::to_value(filtered_results)?;
                    let marker: serde_json::Value =
                        serde_json::from_slice(&std::fs::read(run_dir.join("diagnostic.json"))?)?;
                    probe_record["diagnostic_scope"] =
                        serde_json::json!(crate::fullmix_rms::diagnostic_scope(&marker)?);
                    probe_record["seen_quality_goal_exact_f1"] = serde_json::json!(0.95);
                }
                writeln!(log, "{}", serde_json::to_string(&probe_record)?)?;
                log.flush()?;
                if !passed {
                    model.clone().save_file(
                        run_dir.join(format!("checkpoints/learning-failed-step-{step}")),
                        &recorder,
                    )?;
                    if completed_exposure.is_some() {
                        crate::fullmix_rms::checkpoint(
                            &model.valid(),
                            cfg,
                            run_dir,
                            &run_dir.join(format!("checkpoints/learning-failed-step-{step}.mpk")),
                            step,
                            crate::fullmix_rms::CheckpointKind::LearningFailure,
                        )?;
                    }
                    anyhow::bail!(
                        "training probe failed at step {step}; diagnostic checkpoint retained, review learning_metrics.jsonl before another run"
                    );
                }
                last_learning_check = Some(enforced && passed);
            }
            if step % 200 == 0 {
                eprintln!(
                    "epoch {epoch} step {step}/{total_steps} loss {:.4} lr {lr:.2e}",
                    epoch_loss / batches as f64
                );
            }
            if max_steps.is_some_and(|limit| step >= limit) {
                reached_limit = true;
                break;
            }
        }
        let (score, logged) = if exposure.is_some() {
            (
                CheckpointScore::new(0.0, 0.0),
                serde_json::json!({
                    "scope": "whole-original PERSON diagnostic; scoring only after fixed snapshots",
                }),
            )
        } else {
            validate(&model.valid())?
        };
        anyhow::ensure!(
            score.is_finite(),
            "epoch {epoch} produced a non-finite validation score; refusing to select a checkpoint"
        );
        let train_loss = epoch_loss / batches.max(1) as f64;
        let mut line = serde_json::json!({
            "epoch": epoch,
            "optimizer_steps": step,
            "partial_epoch": reached_limit,
            "train_loss": train_loss,
            "lr": lr_at(
                step.saturating_sub(1),
                total_steps,
                cfg.train.learning_rate,
                cfg.train.warmup_steps
            ),
        });
        if let Some(coefficient) = diagnostic_logit_penalty.filter(|value| *value > 0.0) {
            line["train_cross_entropy"] = (epoch_cross_entropy / batches.max(1) as f64).into();
            line["train_logit_penalty_unscaled"] = (epoch_penalty / batches.max(1) as f64).into();
            line["train_objective"] = train_loss.into();
            line["diagnostic_logit_penalty"] = coefficient.into();
        }
        if let (Some(line), serde_json::Value::Object(fields)) = (line.as_object_mut(), &logged) {
            line.extend(fields.clone());
        }
        writeln!(metrics, "{line}")?;
        metrics.flush()?;
        eprintln!(
            "epoch {epoch}: loss {train_loss:.4}, validation score {:.4}, tie break {:.4}",
            score.primary, score.secondary
        );
        if exposure.is_none() {
            model.clone().save_file(
                run_dir.join(format!("checkpoints/epoch-{epoch}")),
                &recorder,
            )?;
            if completed_exposure.is_some() {
                crate::fullmix_rms::checkpoint(
                    &model.valid(),
                    cfg,
                    run_dir,
                    &run_dir.join(format!("checkpoints/epoch-{epoch}.mpk")),
                    step,
                    crate::fullmix_rms::CheckpointKind::Epoch,
                )?;
            }
            if score > best.0 {
                best = (score, epoch, logged);
                since_best = 0;
                model.clone().save_file(run_dir.join("best"), &recorder)?;
                if completed_exposure.is_some() {
                    crate::fullmix_rms::checkpoint(
                        &model.valid(),
                        cfg,
                        run_dir,
                        &run_dir.join("best.mpk"),
                        step,
                        crate::fullmix_rms::CheckpointKind::DevelopmentSelected,
                    )?;
                }
            } else {
                since_best += 1;
                if since_best >= cfg.train.patience
                    && max_steps.is_none()
                    && learning_check.is_none_or(|check| step >= check.start_step)
                {
                    eprintln!(
                        "early stop at epoch {epoch}; best epoch {}, score {:.4}",
                        best.1, best.0.primary
                    );
                    break;
                }
            }
        }
        if reached_limit {
            break;
        }
    }
    if let Some(accounting) = &exposure {
        anyhow::ensure!(
            step == 2000,
            "whole-original PERSON diagnostic did not reach its fixed cap"
        );
        accounting.write(run_dir, items, weights)?;
    }
    if let Some(accounting) = &completed_exposure {
        anyhow::ensure!(
            step == crate::fullmix_rms::STEPS,
            "fullmix did not reach its fixed cap"
        );
        accounting.write(weights)?;
    }
    let learning_check_passed = last_learning_check;
    anyhow::ensure!(
        learning_check.is_none() || learning_check_passed == Some(true),
        "run ended without passing the enforced training learning check"
    );
    Ok(FitSummary {
        best_epoch: best.1,
        best_score: best.0.primary,
        best_metrics: best.2,
        steps: step,
        seconds: started.elapsed().as_secs(),
        dense,
        embedding,
        learning_check_passed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_validation_route_cannot_fall_back_to_standard_forward() {
        let mut cfg = crate::config::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/detector-shared-v6.toml"),
        )
        .unwrap();
        cfg.features.hash_buckets = 64;
        cfg.features.ngram_dim = 4;
        cfg.features.shape_dim = 4;
        cfg.net.hidden = 4;
        cfg.net.dilations = vec![1];
        let gold = vec![KindSpan {
            kind: 0,
            start: 0,
            end: 4,
        }];
        let enc = detector::encode_document("Anna", &gold, &cfg.features.to_tessera()).unwrap();
        let docs = vec![detector::DetectorDoc {
            text: "Anna".into(),
            enc,
            gold,
            breaks: vec![true],
        }];
        let device = Default::default();
        let mut model = cfg.detector_net_config().init::<NdArray>(&device);
        assert!(
            diagnostic_predictions(
                &model,
                &docs,
                32,
                &device,
                crate::diagnostic_operator::ForwardOperator::Standard
            )
            .is_ok()
        );
        model.blocks[0].conv.bias = Some(burn::module::Param::from_tensor(Tensor::from_data(
            [1e20_f32; 4],
            &device,
        )));
        assert!(
            diagnostic_predictions(
                &model,
                &docs,
                32,
                &device,
                crate::diagnostic_operator::ForwardOperator::ResidualRmsV1
            )
            .is_err()
        );
    }

    #[test]
    fn learning_failure_retains_evidence_without_completing_the_run() {
        use burn::backend::{Autodiff, NdArray};
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut cfg = crate::config::load(&root.join("configs/detector-shared-v6.toml")).unwrap();
        cfg.net.hidden = 4;
        cfg.net.dropout = 0.0;
        cfg.train.epochs = 3;
        cfg.train.batch_size = 6;
        cfg.train.warmup_steps = 0;
        let check = cfg
            .detector
            .as_mut()
            .unwrap()
            .learning_check
            .as_mut()
            .unwrap();
        check.documents = 6;
        check.every_steps = 1;
        check.start_step = 2;
        check.minimum_recall = 1.0;
        // Identical input features with three gold kinds cannot satisfy every recall gate.
        let docs: Vec<_> = (0..6)
            .map(|index| {
                let text = format!("entity{index}");
                let kind = index % 3;
                let end = text.len() as u32;
                detector::DetectorDoc {
                    text,
                    enc: Encoded {
                        token_spans: vec![(0, end)],
                        ngram_ids: vec![vec![1]],
                        script: vec![1],
                        shape: vec![1],
                        flags: vec![0],
                        labels: vec![(kind * 2 + 1) as u8],
                        country: "US".into(),
                    },
                    gold: vec![detector::KindSpan {
                        kind,
                        start: 0,
                        end,
                    }],
                    breaks: vec![true],
                }
            })
            .collect();
        let items: Vec<_> = docs.iter().map(|doc| doc.enc.clone()).collect();
        let probe = crate::learning_probe::SelectedProbe {
            documents: docs,
            real_count: 3,
        };
        let run = tempfile::tempdir().unwrap();
        initialize_run(run.path(), &cfg).unwrap();
        let device = Default::default();
        let model = cfg.detector_net_config().init::<Autodiff<NdArray>>(&device);
        let result = fit(
            &cfg,
            run.path(),
            model,
            TrainingData {
                items: &items,
                draws: Vec::new(),
                detector_epoch: Some(DetectorEpochDraws {
                    synthetic_count: 3,
                    synthetic_per_epoch: 3,
                    source_pieces: vec![3],
                    source_repeats: vec![1],
                    seed: 42,
                }),
                learning_probe: Some(&probe),
                max_steps: Some(2),
                diagnostic_logit_penalty: None,
                forward_operator: crate::diagnostic_operator::ForwardOperator::Standard,
                completed_exposure: None,
            },
            &[1.0; 7],
            &device,
            |_| Ok((CheckpointScore::new(1.0, 1.0), serde_json::json!({}))),
        );
        assert!(
            result
                .err()
                .unwrap()
                .to_string()
                .contains("training probe failed at step 2")
        );
        assert!(run.path().join("best.mpk").exists());
        assert!(run.path().join("diagnostic_initial.json").exists());
        assert!(
            run.path()
                .join("checkpoints/diagnostic-step-2.mpk")
                .exists()
        );
        assert!(
            !run.path()
                .join("checkpoints/diagnostic-step-3.mpk")
                .exists()
        );
        assert!(
            run.path()
                .join("checkpoints/learning-failed-step-2.mpk")
                .exists()
        );
        assert!(!run.path().join("summary.json").exists());
        let rows: Vec<serde_json::Value> =
            std::fs::read_to_string(run.path().join("learning_metrics.jsonl"))
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1]["step"], 2);
        assert_eq!(rows[1]["enforced"], true);
        assert_eq!(rows[1]["passed"], false);
        for row in &rows {
            for group in ["real", "synthetic"] {
                for kind in ["person", "org", "address"] {
                    let kept = &row["source_groups"][group]["per_kind"][kind];
                    let raw = &row["candidate_source_groups"][group]["per_kind"][kind];
                    assert_eq!(raw["gold"], kept["gold"]);
                    assert!(
                        raw["exact"]["recall"].as_f64().unwrap()
                            >= kept["exact"]["recall"].as_f64().unwrap()
                    );
                    assert!((0.0..=1.0).contains(&raw["exact"]["precision"].as_f64().unwrap()));
                }
            }
        }
    }

    #[test]
    fn diagnostic_logit_penalty_is_scoped_and_zero_preserves_updates() {
        type B = Autodiff<NdArray>;
        let mut cfg = crate::config::load(&repo_path("configs/detector-shared-v6.toml")).unwrap();
        cfg.features.hash_buckets = 16;
        cfg.net.hidden = 4;
        cfg.net.ngram_from = None;
        cfg.net.finetune_ngram = false;
        cfg.net.dropout = 0.0;
        cfg.train.epochs = 2;
        cfg.train.batch_size = 1;
        cfg.train.warmup_steps = 0;
        cfg.train.diagnostic_schedule_steps = Some(2);
        cfg.detector.as_mut().unwrap().learning_check = None;
        let items = vec![Encoded {
            token_spans: vec![(0, 4)],
            ngram_ids: vec![vec![1]],
            script: vec![1],
            shape: vec![1],
            flags: vec![0],
            labels: vec![1],
            country: "US".into(),
        }];
        let device = Default::default();
        B::seed(&device, 42);
        let model = cfg.detector_net_config().init::<B>(&device);
        // Burn clones unmaterialized parameters with independent lazy initialization.
        let _ = crate::quantize::extract(&model.valid(), cfg.net_name());
        let refused = tempfile::tempdir().unwrap();
        initialize_run(refused.path(), &cfg).unwrap();
        let data = |coefficient| TrainingData {
            items: &items,
            draws: vec![0],
            detector_epoch: None,
            learning_probe: None,
            max_steps: Some(2),
            diagnostic_logit_penalty: coefficient,
            forward_operator: crate::diagnostic_operator::ForwardOperator::Standard,
            completed_exposure: None,
        };
        assert!(
            fit(
                &cfg,
                refused.path(),
                model.clone(),
                data(Some(1e-4)),
                &[1.; 7],
                &device,
                |_| Ok((CheckpointScore::new(1., 1.), serde_json::json!({})))
            )
            .is_err()
        );
        assert!(!refused.path().join("diagnostic_initial.json").exists());
        assert!(!refused.path().join("metrics.jsonl").exists());
        let mut controls = Vec::new();
        for coefficient in [None, Some(0.0), Some(1e-4)] {
            let run = tempfile::tempdir().unwrap();
            initialize_run(run.path(), &cfg).unwrap();
            if let Some(value) = coefficient {
                std::fs::write(
                    run.path().join("diagnostic.json"),
                    serde_json::to_vec(&serde_json::json!({
                        "scope": "fixed training-only memorization diagnostic; never export",
                        "logit_penalty": value,
                    }))
                    .unwrap(),
                )
                .unwrap();
            }
            fit(
                &cfg,
                run.path(),
                model.clone(),
                data(coefficient),
                &[1.; 7],
                &device,
                |_| Ok((CheckpointScore::new(1., 1.), serde_json::json!({}))),
            )
            .unwrap();
            let metrics = std::fs::read_to_string(run.path().join("metrics.jsonl")).unwrap();
            if coefficient == Some(1e-4) {
                for line in metrics.lines() {
                    let row: serde_json::Value = serde_json::from_str(line).unwrap();
                    let objective = row["train_objective"].as_f64().unwrap();
                    let ce = row["train_cross_entropy"].as_f64().unwrap();
                    let penalty = row["train_logit_penalty_unscaled"].as_f64().unwrap();
                    assert!(penalty >= 0.0);
                    assert!((objective - ce - 1e-4 * penalty).abs() < 1e-6);
                }
            } else {
                let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
                let final_model = model
                    .clone()
                    .load_file(
                        run.path().join("checkpoints/diagnostic-step-2"),
                        &recorder,
                        &device,
                    )
                    .unwrap();
                controls.push((
                    metrics,
                    crate::training_diagnostic::parameter_sha256(&crate::quantize::extract(
                        &final_model.valid(),
                        cfg.net_name(),
                    )),
                ));
            }
        }
        assert_eq!(controls[0], controls[1]);
    }

    #[test]
    fn bounded_diagnostic_reaches_its_cap_despite_epoch_patience() {
        let mut cfg = crate::config::load(&repo_path("configs/detector-shared-v6.toml")).unwrap();
        cfg.features.hash_buckets = 16;
        cfg.net.hidden = 4;
        cfg.net.ngram_from = None;
        cfg.net.finetune_ngram = false;
        cfg.net.dropout = 0.0;
        cfg.train.epochs = 5;
        cfg.train.batch_size = 1;
        cfg.train.warmup_steps = 0;
        cfg.train.patience = 1;
        cfg.train.diagnostic_schedule_steps = Some(10);
        cfg.detector.as_mut().unwrap().learning_check = None;
        let items = vec![Encoded {
            token_spans: vec![(0, 4)],
            ngram_ids: vec![vec![1]],
            script: vec![1],
            shape: vec![1],
            flags: vec![0],
            labels: vec![1],
            country: "US".into(),
        }];
        let run = tempfile::tempdir().unwrap();
        initialize_run(run.path(), &cfg).unwrap();
        let device = Default::default();
        let model = cfg.detector_net_config().init::<Autodiff<NdArray>>(&device);
        let result = fit(
            &cfg,
            run.path(),
            model,
            TrainingData {
                items: &items,
                draws: vec![0],
                detector_epoch: None,
                learning_probe: None,
                max_steps: Some(3),
                diagnostic_logit_penalty: None,
                forward_operator: crate::diagnostic_operator::ForwardOperator::Standard,
                completed_exposure: None,
            },
            &[1.0; 7],
            &device,
            |_| Ok((CheckpointScore::new(1.0, 1.0), serde_json::json!({}))),
        )
        .unwrap();
        assert_eq!(result.steps, 3);
        assert_eq!(result.best_epoch, 1);
        assert!(
            run.path()
                .join("checkpoints/diagnostic-step-3.mpk")
                .exists()
        );
        assert!(!run.path().join("checkpoints/epoch-4.mpk").exists());
        let initial: serde_json::Value = serde_json::from_slice(
            &std::fs::read(run.path().join("diagnostic_initial.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(initial["learning_rate_horizon_steps"], 10);
        assert_eq!(initial["scheduled_optimizer_steps"], 5);
    }

    #[test]
    fn learning_check_requires_exposure_and_an_enforced_check_before_stopping() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let mut cfg = crate::config::load(&root.join("configs/detector-shared-v6.toml")).unwrap();
        validate_learning_budget(&cfg, 8856, Some(2000)).unwrap();
        assert!(validate_learning_budget(&cfg, 8856, Some(1999)).is_err());
        cfg.train.epochs = 3;
        assert!(validate_learning_budget(&cfg, 8856, None).is_err());
        cfg.train.epochs = 25;
        cfg.detector
            .as_mut()
            .unwrap()
            .learning_check
            .as_mut()
            .unwrap()
            .start_step = 277;
        assert!(validate_learning_budget(&cfg, 8856, None).is_err());
    }

    #[test]
    #[ignore = "requires the locally prepared US training corpus"]
    fn pinned_real_development_set_is_source_backed_and_encodable() {
        let cfg = crate::config::load(&repo_path("configs/detector-shared.toml")).unwrap();
        verify_real_development_gold(&cfg).unwrap();
        let docs =
            detector::load_development_gold(&repo_path(REAL_DEV_GOLD), &cfg.features.to_tessera())
                .unwrap();
        assert_eq!(docs.len(), 196);
        for kind in 0..3 {
            assert!(
                docs.iter()
                    .any(|doc| doc.gold.iter().any(|span| span.kind == kind))
            );
        }
    }

    #[test]
    #[ignore = "requires the locally prepared US training corpus"]
    fn v4_training_uses_corrected_development_gold_and_exclusions() {
        let cfg = crate::config::load(&repo_path("configs/detector-shared-v4.toml")).unwrap();
        verify_real_development_gold(&cfg).unwrap();
        assert_eq!(real_dev_gold(&cfg).0, REAL_DEV_GOLD_V3);
        let inputs: BTreeMap<_, _> = input_paths(&cfg).into_iter().collect();
        assert_eq!(inputs["real_dev_gold"], repo_path(REAL_DEV_GOLD_V3));
        assert_eq!(
            inputs["real_dev_exclusions"],
            repo_path(REAL_DEV_EXCLUSIONS_V4)
        );
        assert_eq!(
            inputs["generator_manifest"],
            Path::new(&cfg.data.manifests).join("detector-synthetic.json")
        );
        let mut changed =
            crate::config::load(&repo_path("configs/detector-shared-v4.toml")).unwrap();
        changed.generate.as_mut().unwrap().exclude_gold =
            "data/interim/review/us-eval-exclusions-v3.jsonl".into();
        assert_eq!(real_dev_gold(&changed).0, REAL_DEV_GOLD_V3);
        let error = verify_generation_manifest(&changed).unwrap_err();
        assert!(error.to_string().contains("corrected V4 evaluation gold"));
    }

    #[test]
    #[ignore = "requires the locally prepared US training corpus"]
    fn v5_training_pins_all_exclusions_and_corrected_development_gold() {
        let cfg = crate::config::load(&repo_path("configs/detector-shared-v5.toml")).unwrap();
        verify_real_development_gold(&cfg).unwrap();
        assert_eq!(real_dev_gold(&cfg).0, REAL_DEV_GOLD_V3);
        let inputs: BTreeMap<_, _> = input_paths(&cfg).into_iter().collect();
        for path in [
            REAL_DEV_GOLD_V3,
            REAL_DEV_MANIFEST_V3,
            REAL_DEV_ERRATA_V3,
            REAL_DEV_GOLD,
            REAL_DEV_MANIFEST,
            REAL_DEV_ERRATA,
            REAL_DEV_EXCLUSIONS_V5,
            REAL_DEV_EXCLUSIONS_V5_MANIFEST,
            REAL_DEV_EXCLUSIONS_V5_BUILDER,
            REVIEW_POLICY,
        ] {
            assert!(
                inputs.values().any(|value| value == &repo_path(path)),
                "{path}"
            );
        }
        for source in REAL_DEV_EXCLUSIONS_V5_SOURCES {
            assert!(inputs.values().any(|value| value == &repo_path(source)));
            assert!(
                inputs
                    .values()
                    .any(|value| value == &repo_path(source).with_extension("manifest.json"))
            );
        }
        assert_eq!(
            inputs["test_shard"],
            Path::new(&cfg.data.processed).join("test.parquet")
        );
        assert_eq!(
            inputs["generator_manifest"],
            Path::new(&cfg.data.manifests).join("detector-synthetic.json")
        );
    }

    #[test]
    fn v5_training_rejects_downgraded_exclusions_without_falling_back() {
        for hint in 0..4 {
            let mut cfg =
                crate::config::load(&repo_path("configs/detector-shared-v5.toml")).unwrap();
            cfg.name = "custom-detector".into();
            cfg.data.processed = "data/processed/custom-detector".into();
            cfg.data.manifests = "data/manifests/custom-detector".into();
            cfg.generate.as_mut().unwrap().exclude_gold = REAL_DEV_EXCLUSIONS_V4.into();
            match hint {
                0 => cfg.name = "detector-us-v5".into(),
                1 => cfg.data.processed = "data/processed/detector-us-v5".into(),
                2 => cfg.data.manifests = "data/manifests/v5".into(),
                _ => cfg.generate.as_mut().unwrap().exclude_gold = REAL_DEV_EXCLUSIONS_V5.into(),
            }
            assert!(uses_v5_gold(&cfg));
            assert_eq!(real_dev_gold(&cfg).0, REAL_DEV_GOLD_V3);
            if hint < 3 {
                let error = verify_generation_manifest(&cfg).unwrap_err();
                assert!(error.to_string().contains("corrected V5 evaluation gold"));
            }
        }
    }

    #[test]
    #[ignore = "requires the locally prepared US training corpus"]
    fn v5_generation_manifest_rejects_altered_shards_and_silver_hashes() {
        let dir = tempfile::tempdir().unwrap();
        let mut cfg = crate::config::load(&repo_path("configs/detector-shared-v5.toml")).unwrap();
        cfg.data.processed = dir.path().join("processed").to_string_lossy().into_owned();
        cfg.data.manifests = dir.path().join("manifests").to_string_lossy().into_owned();
        std::fs::create_dir_all(&cfg.data.processed).unwrap();
        std::fs::create_dir_all(&cfg.data.manifests).unwrap();
        let mut shards = BTreeMap::new();
        for split in ["train", "valid", "test"] {
            let path = Path::new(&cfg.data.processed).join(format!("{split}.parquet"));
            std::fs::write(&path, split).unwrap();
            shards.insert(split, hash_file(&path).unwrap());
        }
        let generate = cfg.generate.as_ref().unwrap();
        let silver = &cfg.detector.as_ref().unwrap().silver;
        let silver_hashes: BTreeMap<_, _> = silver
            .iter()
            .map(|path| (path, hash_file(&repo_path(path)).unwrap()))
            .collect();
        let manifest = serde_json::json!({
            "source": "tessera-generator",
            "version": 3,
            "template_policy": "source_backed_us_unit_parent_v1",
            "seed": cfg.seed,
            "max_tokens": generate.max_tokens,
            "names": cfg.names.as_ref().unwrap().out,
            "addresses": generate.addresses,
            "exclude_gold": generate.exclude_gold,
            "exclude_gold_sha256": hash_file(&repo_path(&generate.exclude_gold)).unwrap(),
            "synthetic_parquet_sha256": shards,
            "real_silver_sha256": silver_hashes,
        });
        let path = Path::new(&cfg.data.manifests).join("detector-synthetic.json");
        std::fs::write(&path, serde_json::to_vec(&manifest).unwrap()).unwrap();
        verify_generation_manifest(&cfg).unwrap();
        let test_shard = Path::new(&cfg.data.processed).join("test.parquet");
        std::fs::write(&test_shard, "changed").unwrap();
        assert!(
            verify_generation_manifest(&cfg)
                .unwrap_err()
                .to_string()
                .contains("generated test shard")
        );
        std::fs::write(&test_shard, "test").unwrap();
        let mut changed = manifest.clone();
        changed["real_silver_sha256"][&silver[0]] = serde_json::json!("changed");
        std::fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
        assert!(
            verify_generation_manifest(&cfg)
                .unwrap_err()
                .to_string()
                .contains("differs from its generator manifest")
        );
        let mut changed = manifest;
        changed["exclude_gold_sha256"] = serde_json::json!("changed");
        std::fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
        assert!(
            verify_generation_manifest(&cfg)
                .unwrap_err()
                .to_string()
                .contains("configured exclusions")
        );
    }

    #[test]
    fn detector_checkpoint_score_follows_weakest_real_field() {
        let scores = |person: f64, org: f64, address: f64| detector::SpanScores {
            per_kind: BTreeMap::from([
                (
                    "person",
                    detector::KindScores {
                        exact: detector::Prf {
                            f1: person,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                ),
                (
                    "org",
                    detector::KindScores {
                        exact: detector::Prf {
                            f1: org,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                ),
                (
                    "address",
                    detector::KindScores {
                        exact: detector::Prf {
                            f1: address,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                ),
            ]),
            macro_exact_f1: (person + org + address) / 3.0,
        };
        let high_macro_weak_org = scores(0.95, 0.25, 0.95);
        let balanced = scores(0.70, 0.70, 0.70);
        assert!(high_macro_weak_org.macro_exact_f1 > balanced.macro_exact_f1);
        assert!(minimum_model_exact_f1(&balanced) > minimum_model_exact_f1(&high_macro_weak_org));
        assert!(
            CheckpointScore::new(minimum_model_exact_f1(&balanced), balanced.macro_exact_f1)
                > CheckpointScore::new(
                    minimum_model_exact_f1(&high_macro_weak_org),
                    high_macro_weak_org.macro_exact_f1,
                )
        );
        assert!(CheckpointScore::new(0.0, 0.6) > CheckpointScore::new(0.0, 0.5));
    }

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
    #[ignore = "requires the locally prepared US training corpus"]
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
        detector.detector.as_mut().unwrap().silver_repeats.clear();
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
            if role.starts_with("ngram_source_") || role.starts_with("real_dev_") {
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
        config.detector.as_mut().unwrap().silver_repeats.clear();
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
    fn detector_epoch_draws_rotate_synthetic_rows_and_weight_sources() {
        let plan = DetectorEpochDraws {
            synthetic_count: 10,
            synthetic_per_epoch: 4,
            source_pieces: vec![2, 1],
            source_repeats: vec![3, 2],
            seed: 42,
        };
        assert_eq!(plan.len().unwrap(), 12);
        let first = plan.indices(1).unwrap();
        let second = plan.indices(2).unwrap();
        assert_eq!(first.len(), 12);
        assert_eq!(
            first[..4]
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>()
                .len(),
            4
        );
        assert_ne!(first[..4], second[..4]);
        assert!(first[..4].iter().all(|&index| index < 10));
        assert_eq!(first[4..].iter().filter(|&&index| index == 10).count(), 3);
        assert_eq!(first[4..].iter().filter(|&&index| index == 11).count(), 3);
        assert_eq!(first[4..].iter().filter(|&&index| index == 12).count(), 2);
    }

    #[test]
    fn detector_epoch_draws_reject_invalid_counts() {
        let mut plan = DetectorEpochDraws {
            synthetic_count: 10,
            synthetic_per_epoch: 11,
            source_pieces: vec![1],
            source_repeats: vec![2],
            seed: 42,
        };
        assert!(plan.len().is_err());
        plan.synthetic_per_epoch = 4;
        plan.source_repeats.clear();
        assert!(plan.len().is_err());
        plan.source_repeats = vec![usize::MAX];
        assert!(plan.len().is_err());
    }

    #[test]
    fn length_bucketing_preserves_draws_and_reduces_padding() {
        let lengths: Vec<usize> = (0..128)
            .map(|index| if index % 8 == 0 { 300 } else { 20 })
            .collect();
        let order: Vec<usize> = (0..128).collect();
        let batches = length_bucket_batches(&order, &lengths, 4, 42);
        let mut actual: Vec<usize> = batches.iter().flatten().copied().collect();
        actual.sort_unstable();
        assert_eq!(actual, order);
        let padded = |batches: &[Vec<usize>]| -> usize {
            batches
                .iter()
                .map(|batch| batch.iter().map(|&index| lengths[index]).max().unwrap() * batch.len())
                .sum()
        };
        let unbucketed: Vec<Vec<usize>> = order.chunks(4).map(<[usize]>::to_vec).collect();
        assert!(padded(&batches) < padded(&unbucketed) / 2);
        assert_eq!(batches, length_bucket_batches(&order, &lengths, 4, 42));
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
    fn weighted_loss_gradients_match_cross_entropy_and_ignore_padding() {
        type B = Autodiff<NdArray>;
        let device = Default::default();
        let values = vec![
            0.0f32, 1.0, 2.0, 1000.0, 1001.0, 998.0, -1000.0, -999.0, -1002.0, 4.0, 0.0, -3.0, 8.0,
            -2.0, 1.0, 1000.0, -1000.0, 0.0,
        ];
        let targets = [2usize, 1, 0, 0, 2, 1];
        let active = [1.0f32, 1.0, 1.0, 0.0, 1.0, 0.0];
        let weights = [1.0f32, 3.0, 5.0];
        let denominator: f64 = targets
            .iter()
            .zip(active)
            .map(|(&target, mask)| f64::from(weights[target] * mask))
            .sum();
        let logits = Tensor::<B, 3>::from_data(TensorData::new(values.clone(), [2, 3, 3]), &device)
            .require_grad();
        let loss = masked_loss(
            logits.clone(),
            Tensor::from_data(
                TensorData::new(targets.map(|target| target as i64).to_vec(), [2, 3]),
                &device,
            ),
            Tensor::from_data(TensorData::new(active.to_vec(), [2, 3]), &device),
            Tensor::from_data(TensorData::new(weights.to_vec(), [3]), &device),
        );
        let actual_loss: f32 = loss.clone().into_scalar();
        let gradients = loss.backward();
        let actual = logits
            .grad(&gradients)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let mut expected_loss = 0.0;
        for (token, scores) in values.chunks_exact(3).enumerate() {
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let shifted: Vec<f64> = scores
                .iter()
                .map(|&score| f64::from(score - maximum))
                .collect();
            let sum: f64 = shifted.iter().map(|score| score.exp()).sum();
            let factor = f64::from(weights[targets[token]] * active[token]) / denominator;
            expected_loss += (sum.ln() - shifted[targets[token]]) * factor;
            for (label, score) in shifted.iter().enumerate() {
                let target = if label == targets[token] { 1.0 } else { 0.0 };
                let expected = (score.exp() / sum - target) * factor;
                assert!(
                    (f64::from(actual[token * 3 + label]) - expected).abs() < 2e-5,
                    "token {token}, label {label}: expected {expected}, got {}",
                    actual[token * 3 + label]
                );
            }
        }
        assert!((f64::from(actual_loss) - expected_loss).abs() < 2e-5);
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
