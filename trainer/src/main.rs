//! Data preparation, training, evaluation, quantization, and export.
//! Depends on `tessera` for the tokenizer and features. Never ships.

mod address_prefix;
mod baselines;
mod bench;
mod bodies;
mod check;
mod config;
mod context_sampling;
mod context_views;
mod data;
mod dataset;
mod detect_eval;
mod detector;
mod diagnostic_decode;
mod diagnostic_operator;
mod entity_context;
mod eval;
mod export;
mod filler;
mod fixtures;
mod fullmix_exposure;
mod fullmix_hard;
mod fullmix_rms;
mod generate;
mod group_eval;
mod hard_token_batch;
mod hard_token_masks;
mod hard_token_selection;
mod inflect;
mod learning_gate;
mod learning_probe;
mod local_templates;
mod loss_diagnostic;
mod loss_stability;
mod memorization;
mod model_eval;
mod names;
mod negatives;
mod net;
mod pool_filter;
mod quantize;
mod residual_rms;
mod templates;
mod train;
mod training_audit;
mod training_diagnostic;
mod typed_synthetic;

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
        /// Stop a diagnostic run after this many updates; incomplete runs cannot be exported.
        #[arg(long)]
        max_steps: Option<usize>,
        #[arg(long, value_enum, default_value = "wgpu")]
        backend: train::BackendKind,
    },
    /// Run one pinned CPU mixed-data RMS diagnostic, or initialize it without updates.
    DiagnoseFullmixRms {
        #[arg(long)]
        manifest: PathBuf,
        #[arg(long)]
        out: PathBuf,
        /// Discover native initialization and probe identities without any forward or fit.
        #[arg(long)]
        preflight: bool,
    },
    /// Test learning on a small fixed training subset without scoring development data.
    Memorize {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 1000)]
        steps: usize,
        /// Diagnostic centered-logit penalty; zero preserves the original objective.
        #[arg(long, default_value_t = 0.0)]
        logit_penalty: f64,
        /// Require the control's exact initial parameter identity before optimization.
        #[arg(long)]
        expected_initial_sha256: Option<String>,
        /// Exact reviewed fifteen-row whole-original PERSON diagnostic manifest.
        #[arg(long)]
        cohort: Option<PathBuf>,
        /// Parameter-free post-residual RMS; only for the frozen whole15 diagnostic.
        #[arg(long)]
        residual_rms: bool,
        #[arg(long, value_enum, default_value = "ndarray")]
        backend: train::BackendKind,
    },
    /// Compare frozen checkpoint loss with evaluation and seeded dropout-active forwards.
    InspectLoss {
        #[arg(long)]
        run: PathBuf,
        #[arg(long)]
        gold: PathBuf,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, default_value_t = 8)]
        samples: usize,
    },
    /// Validate detector silver labels and report distinct supervision before training.
    CheckSilver {
        #[arg(long)]
        config: PathBuf,
        /// Inspect possible training windows without changing the data or sampler.
        #[arg(long)]
        context_windows: bool,
    },
    /// Validate all prepared detector inputs without starting training.
    CheckDetectorData {
        #[arg(long)]
        config: PathBuf,
        /// Simulate verified context views without changing training or its configuration.
        #[arg(long)]
        context_views: bool,
        /// Pin the five source biographies and a reference planned schedule audit.
        #[arg(long, requires = "exposure_out", conflicts_with = "context_views")]
        exposure_targets: Option<PathBuf>,
        /// Write planned coefficients to a new JSON receipt; never starts a model.
        #[arg(long, requires = "exposure_targets", conflicts_with = "context_views")]
        exposure_out: Option<PathBuf>,
    },
    /// Score a run, the deterministic baselines, or external predictions.
    Eval(Box<EvalArgs>),
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
    /// Score reviewed JSONL: the shipped detector by default, or a run's f32 model with --run.
    #[arg(long)]
    gold: Option<PathBuf>,
    /// With `--gold`: use the gold country as a phone hint or infer it from the document.
    #[arg(long, value_enum, default_value = "known")]
    country_hint_mode: detect_eval::CountryHintMode,
    /// With `--gold`: an external system's predictions by case name (repeatable).
    #[arg(long)]
    predictions: Vec<PathBuf>,
    /// With `--gold`: save the selected detector's spans for error analysis.
    #[arg(long)]
    dump_predictions: Option<PathBuf>,
    /// With --run --gold: save all decoded candidates before confidence filtering.
    #[arg(long, requires_all = ["run", "gold"], conflicts_with_all = ["grouper", "addresses"])]
    dump_confidence: Option<PathBuf>,
    /// Opt into the reviewed ADDRESS continuation diagnostic on final training-seen gold.
    #[arg(
        long, value_enum, requires_all = ["run", "gold"],
        conflicts_with_all = ["baseline", "grouper", "addresses", "report", "errors"]
    )]
    diagnostic_decoder: Option<diagnostic_decode::Decoder>,
    /// The bundle `detect` loads for `--gold` and for a detector run's split.
    #[arg(long, default_value = "models/tessera-v1.safetensors")]
    bundle: PathBuf,
    /// With shipped `--gold`, a detector run's split, or `--grouper`: write Markdown here.
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
    let command = Cli::parse().command;
    diagnostic_decode::validate_build_context(&command)?;
    match command {
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
            max_steps,
            backend,
        } => train::run(train::TrainArgs {
            config: &config,
            name: name.as_deref(),
            train_per_country,
            epochs,
            max_steps,
            backend,
        }),
        Command::Memorize {
            config,
            out,
            steps,
            logit_penalty,
            expected_initial_sha256,
            cohort,
            residual_rms,
            backend,
        } => train::memorize(
            &config,
            &out,
            steps,
            backend,
            logit_penalty,
            expected_initial_sha256.as_deref(),
            train::MemorizeScope {
                cohort: cohort.as_deref(),
                residual_rms,
            },
        ),
        Command::InspectLoss {
            run,
            gold,
            out,
            samples,
        } => loss_diagnostic::run(&run, &gold, &out, samples),
        Command::CheckSilver {
            config,
            context_windows,
        } => train::check_silver(&config, context_windows),
        Command::CheckDetectorData {
            config,
            context_views,
            exposure_targets,
            exposure_out,
        } => train::check_detector_data(
            &config,
            context_views,
            exposure_targets.as_deref().zip(exposure_out.as_deref()),
        ),
        Command::DiagnoseFullmixRms {
            manifest,
            out,
            preflight,
        } => train::diagnose_fullmix_rms(&manifest, &out, preflight),
        Command::Eval(args) => eval::run(*args),
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
    }
}

#[cfg(test)]
mod exposure_cli_tests {
    use super::*;

    #[test]
    fn planned_exposure_flags_are_paired_and_exclude_context_views() {
        let prefix = ["trainer", "check-detector-data", "--config", "config.toml"];
        assert!(Cli::try_parse_from(prefix).is_ok());
        let mut targets_only = prefix.to_vec();
        targets_only.extend(["--exposure-targets", "targets.json"]);
        assert!(Cli::try_parse_from(targets_only).is_err());
        let mut output_only = prefix.to_vec();
        output_only.extend(["--exposure-out", "receipt.json"]);
        assert!(Cli::try_parse_from(output_only).is_err());
        let mut paired = prefix.to_vec();
        paired.extend([
            "--exposure-targets",
            "targets.json",
            "--exposure-out",
            "receipt.json",
        ]);
        assert!(Cli::try_parse_from(&paired).is_ok());
        paired.push("--context-views");
        assert!(Cli::try_parse_from(paired).is_err());
    }
}

#[cfg(test)]
mod fullmix_cli_tests {
    use super::*;

    #[test]
    fn fullmix_has_no_backend_budget_or_ordinary_train_overrides() {
        let args = [
            "trainer",
            "diagnose-fullmix-rms",
            "--manifest",
            "plan.json",
            "--out",
            "new-run",
        ];
        assert!(Cli::try_parse_from(args).is_ok());
        for override_arg in [
            "--backend",
            "--max-steps",
            "--epochs",
            "--config",
            "--train-per-country",
        ] {
            let mut wrong = args.to_vec();
            wrong.extend([override_arg, "1"]);
            assert!(Cli::try_parse_from(wrong).is_err());
        }
        let mut preflight = args.to_vec();
        preflight.push("--preflight");
        assert!(Cli::try_parse_from(preflight).is_ok());
    }
}
