//! Data preparation, training, evaluation, quantization, and export.
//! Depends on `tessera` for the tokenizer and features. Never ships.

mod baselines;
mod config;
mod eval;
mod fixtures;

use std::path::PathBuf;

use anyhow::bail;
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
    },
    /// Sample person and organization names from Wikidata and GLEIF.
    Names {
        #[arg(long)]
        config: PathBuf,
    },
    /// Generate synthetic detector documents with safe contact details.
    Generate {
        #[arg(long)]
        config: PathBuf,
    },
    /// Train a detector or parser and write a run directory.
    Train {
        #[arg(long)]
        config: PathBuf,
    },
    /// Score a run, the deterministic baselines, or external predictions.
    Eval(EvalArgs),
    /// Per-channel int8 quantization of a trained run.
    Quantize {
        #[arg(long)]
        run: PathBuf,
    },
    /// Write the safetensors bundle and golden vectors.
    Export {
        #[arg(long)]
        run: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
}

#[derive(clap::Args)]
struct EvalArgs {
    /// Run directory of a trained model.
    #[arg(long)]
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
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Prepare { config } => {
            let cfg = config::load(&config)?;
            bail!("`prepare` for `{}` arrives in milestone 2", cfg.name)
        }
        Command::Names { .. } | Command::Generate { .. } => bail!("arrives in milestone 4"),
        Command::Train { config } => {
            let cfg = config::load(&config)?;
            bail!("`train` for `{}` arrives in milestone 2", cfg.name)
        }
        Command::Eval(args) => eval::run(args),
        Command::Quantize { .. } | Command::Export { .. } => bail!("arrives in milestone 2"),
    }
}
