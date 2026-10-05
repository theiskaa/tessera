//! Explicit reviewed-native training foundation; fitting and export integration remain gated.

pub(crate) mod initialize;
mod preflight;
mod recipe;
mod source;

pub(crate) use recipe::{Loaded, Prepared, Settings};

/// Prevent any existing fitting route from substituting legacy or synthetic inputs.
pub(crate) fn refuse_incomplete(config: &crate::config::Config) -> anyhow::Result<()> {
    anyhow::ensure!(
        config.reviewed_native.is_none(),
        "reviewed-native fitting is incomplete: canonical recipe receipt, reviewed-real probe, fixed-final selection, input snapshot and validation/export source dispatch are required"
    );
    Ok(())
}

/// Check canonical reviewed native inputs and source receipts without loading weights.
pub(crate) fn preflight(config: &std::path::Path, out: &std::path::Path) -> anyhow::Result<()> {
    preflight::run(config, out)
}
