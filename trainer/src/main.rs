//! Data preparation, training, evaluation, quantization, and export.
//! Depends on `tessera` for the tokenizer and features. Never ships.

mod baselines;
mod bench;
mod bodies;
mod check;
mod config;
mod data;
mod dataset;
mod detect_eval;
mod detector;
mod eval;
mod export;
mod filler;
mod fixtures;
mod generate;
mod group_eval;
mod inflect;
mod local_templates;
mod model_eval;
mod names;
mod negatives;
mod net;
mod pool_filter;
mod quantize;
mod report;
mod templates;
mod train;

use std::path::PathBuf;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "trainer", about = "tessera training and evaluation")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Stream source corpora into grouped, deduplicated, split shards.
    Prepare {
        #[arg(long)]
        config: PathBuf,
        /// Date recorded in the sample manifest, for example `$(date +%F)`.
        #[arg(long, default_value = "unknown")]
        date: String,
    },
    /// Sample person and organization names from Wikidata and GLEIF.
    Names {
        #[arg(long)]
        config: PathBuf,
        /// Re-download cached query results and files.
        #[arg(long)]
        refresh: bool,
        /// Date recorded in the manifests, for example `$(date +%F)`.
        #[arg(long, default_value = "unknown")]
        date: String,
    },
    /// Generate synthetic detector documents with safe contact details.
    Generate {
        #[arg(long)]
        config: PathBuf,
        /// Print this many bracketed documents instead of writing the corpus.
        #[arg(long)]
        sample: Option<usize>,
        /// With `--sample`: only this family, such as `signature`.
        #[arg(long)]
        family: Option<String>,
    },
    /// Train the parser or the detector and write a run directory.
    Train {
        #[arg(long)]
        config: PathBuf,
        /// Run name, overriding the config's; the run is written to `runs/<name>/`.
        #[arg(long)]
        name: Option<String>,
        /// Keep the first N original rows per country, for learning curves.
        #[arg(long)]
        train_per_country: Option<usize>,
        /// Maximum epochs, overriding the config's.
        #[arg(long)]
        epochs: Option<usize>,
        #[arg(long, value_enum, default_value = "wgpu")]
        backend: train::BackendKind,
    },
    /// Score a run, the deterministic baselines, or external predictions.
    Eval(EvalArgs),
    /// Time each pipeline stage natively on the profiling fixtures and append to the report.
    Bench {
        #[arg(long, default_value = "models/tessera-v1.safetensors")]
        model: PathBuf,
        #[arg(long, default_value_t = 200)]
        iterations: u32,
        #[arg(long, default_value = "internal/reports/m7-profile.md")]
        out: PathBuf,
    },
    /// Per-channel int8 quantization of a trained run.
    Quantize {
        #[arg(long)]
        run: PathBuf,
        #[arg(long, value_enum, default_value = "ndarray")]
        backend: train::BackendKind,
    },
    /// Render the milestone report tables from run and eval JSON files.
    Report {
        /// Learning-curve run directories, in order.
        #[arg(long, num_args = 1..)]
        curve: Vec<PathBuf>,
        /// The main run; its `eval/test.json`, `eval/test-shipped.json`, `quantize.json`, and
        /// `config.toml` are read.
        #[arg(long)]
        run: PathBuf,
        #[arg(long, default_value = "data/manifests/parser-sample.json")]
        manifest: PathBuf,
        #[arg(long, default_value = "models/tessera-v1.safetensors")]
        bundle: PathBuf,
        #[arg(long, default_value = "internal/reports/m2-parser.md")]
        out: PathBuf,
    },
    /// Write the two-network safetensors bundle and the golden vectors of both networks.
    Export {
        /// The parser run.
        #[arg(long)]
        parser_run: PathBuf,
        /// The detector run.
        #[arg(long)]
        detector_run: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Date recorded in the bundle manifest, for example `$(date +%F)`.
        #[arg(long, default_value = "unknown")]
        date: String,
    },
}

#[derive(clap::Args)]
struct EvalArgs {
    /// Run directory of a trained model.
    #[arg(long, conflicts_with = "baseline")]
    run: Option<PathBuf>,
    /// Score the deterministic baselines instead of a run.
    #[arg(long)]
    baseline: bool,
    /// Directory holding the fixture files.
    #[arg(long, default_value = "fixtures")]
    fixtures: PathBuf,
    /// Split name for prepared datasets; fixtures ignore it.
    #[arg(long, default_value = "test")]
    split: String,
    /// External prediction files to score (repeatable).
    #[arg(long)]
    external: Vec<PathBuf>,
    /// Output directory for reports.
    #[arg(long, default_value = "runs/baseline")]
    out: PathBuf,
    /// With `--run`: write this many worst examples to `internal/reports/m2-parser-errors.md`.
    #[arg(long)]
    errors: Option<usize>,
    /// With `--run`: the backend to run the model on.
    #[arg(long, value_enum, default_value = "wgpu")]
    backend: train::BackendKind,
    /// Score the shipped `detect` on reviewed documents in this JSONL file, one case per line,
    /// with `--baseline` and each `--predictions` file beside it.
    #[arg(long)]
    gold: Option<PathBuf>,
    /// With `--gold`: an external system's predictions by case name (repeatable).
    #[arg(long)]
    predictions: Vec<PathBuf>,
    /// The bundle `detect` loads for `--gold` and for a detector run's split.
    #[arg(long, default_value = "models/tessera-v1.safetensors")]
    bundle: PathBuf,
    /// With `--gold`, a detector run, or `--grouper`: write the report as Markdown here.
    #[arg(long)]
    report: Option<PathBuf>,
    /// Score the contact grouper on the fixtures in this directory, on gold entities and end
    /// to end with `--bundle`.
    #[arg(long)]
    grouper: Option<PathBuf>,
    /// With `--grouper`: a directory of known-hard grouper cases, scored but never gating.
    #[arg(long)]
    grouper_hard: Option<PathBuf>,
    /// Score the `--bundle`'s address parser, or a parser `--run`, on reviewed real addresses:
    /// parser fixture files in this directory. `--report` receives the addresses it parses
    /// wrong, as JSON lines.
    #[arg(long)]
    addresses: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Prepare { config, date } => data::prepare(&config, &date),
        Command::Names {
            config,
            refresh,
            date,
        } => names::run(&config, refresh, &date),
        Command::Generate {
            config,
            sample,
            family,
        } => generate::run(&config, sample, family.as_deref()),
        Command::Train {
            config,
            name,
            train_per_country,
            epochs,
            backend,
        } => train::run(train::TrainArgs {
            config: &config,
            name: name.as_deref(),
            train_per_country,
            epochs,
            backend,
        }),
        Command::Eval(args) => eval::run(args),
        Command::Bench {
            model,
            iterations,
            out,
        } => bench::run(&model, iterations, &out),
        Command::Quantize { run, backend } => match backend {
            train::BackendKind::Wgpu => {
                quantize::run::<burn::backend::Wgpu>(&run, &Default::default())
            }
            train::BackendKind::Ndarray => {
                quantize::run::<burn::backend::NdArray>(&run, &Default::default())
            }
        },
        Command::Export {
            parser_run,
            detector_run,
            out,
            date,
        } => export::run(&parser_run, &detector_run, &out, &date),
        Command::Report {
            curve,
            run,
            manifest,
            bundle,
            out,
        } => report::run(report::ReportArgs {
            curve: &curve,
            run: &run,
            manifest: &manifest,
            bundle: &bundle,
            out: &out,
        }),
    }
}
