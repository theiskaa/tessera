//! Finite native f32 snapshot round-trips without randomized observation initialization.

use crate::net::TaggerNet;
use crate::reviewed_data::digest;
use anyhow::ensure;
use burn::backend::NdArray;
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use std::path::Path;

/// Saved native weights and the verified observation model loaded from identical bytes.
pub(crate) struct Snapshot {
    /// Dropout-disabled observation model; cloning/loading never initializes random weights.
    pub(crate) model: TaggerNet<NdArray>,
    /// Exact serialized native checkpoint digest.
    pub(crate) checkpoint_sha256: String,
    /// Every finite f32 parameter in native name/shape order.
    pub(crate) parameter_sha256: String,
}

/// Reject absent/nonfinite tensors before reporting a checkpoint parameter identity.
pub(crate) fn parameter_sha256(tensors: &[crate::quantize::F32Tensor]) -> anyhow::Result<String> {
    ensure!(
        !tensors.is_empty() && tensors.iter().all(|t| t.data.iter().all(|v| v.is_finite())),
        "native checkpoint contains missing or nonfinite f32 parameters"
    );
    Ok(crate::training_diagnostic::parameter_sha256(tensors))
}

/// Save/reload a new native checkpoint and verify finite full-f32 parameter equality.
pub(crate) fn save(
    valid: TaggerNet<NdArray>,
    base: &Path,
    device: &burn::backend::ndarray::NdArrayDevice,
) -> anyhow::Result<Snapshot> {
    ensure!(
        !base.with_extension("mpk").try_exists()?,
        "native snapshot already exists"
    );
    let parameters = parameter_sha256(&crate::quantize::extract(&valid, "detector"))?;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
    valid.clone().save_file(base, &recorder)?;
    let checkpoint_sha256 = digest(&std::fs::read(base.with_extension("mpk"))?);
    let restored = valid.clone().load_file(base, &recorder, device)?;
    ensure!(
        digest(&std::fs::read(base.with_extension("mpk"))?) == checkpoint_sha256,
        "saved native checkpoint changed during reload"
    );
    ensure!(
        parameter_sha256(&crate::quantize::extract(&restored, "detector"))? == parameters,
        "native round-trip parameter identity differs"
    );
    Ok(Snapshot {
        model: restored,
        checkpoint_sha256,
        parameter_sha256: parameters,
    })
}
