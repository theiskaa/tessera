//! Closed opt-in detector postprocessing, independent of input feature width.

use crate::{Error, model};

/// The postprocessing phase after the architecture's existing ADDRESS continuation decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectorPostprocessContract {
    /// Existing continuation and confidence calculation without labeled-field refinement.
    AddressContinuationV1,
    /// Strict complete-postal labeled-field refinement; no label, weight or threshold change.
    AddressLabeledFieldsV1,
}

impl DetectorPostprocessContract {
    /// Exact safetensors metadata and reviewed scorer protocol identifier.
    pub const fn name(self) -> &'static str {
        match self {
            Self::AddressContinuationV1 => "address_continuation_v1",
            Self::AddressLabeledFieldsV1 => "address_labeled_fields_v1",
        }
    }

    /// Resolve a supported contract; callers must reject every unrecognized declaration.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "address_continuation_v1" => Some(Self::AddressContinuationV1),
            "address_labeled_fields_v1" => Some(Self::AddressLabeledFieldsV1),
            _ => None,
        }
    }
}

pub(crate) fn decode(
    uses_continuation: bool,
    declared: Option<DetectorPostprocessContract>,
    text: &str,
    bounds: &[(usize, usize)],
    probs: &[f32],
    masked: &[bool],
    breaks: &[bool],
) -> Result<Vec<model::bio::DetectedSpan>, Error> {
    if !uses_continuation {
        if declared.is_some() {
            return Err(Error::UnsupportedVersion);
        }
        return Ok(model::bio::decode_detector(probs, masked, breaks));
    }
    Ok(
        match declared.unwrap_or(DetectorPostprocessContract::AddressContinuationV1) {
            DetectorPostprocessContract::AddressContinuationV1 => {
                model::bio::decode_detector_with_text(text, bounds, probs, masked, breaks)
            }
            DetectorPostprocessContract::AddressLabeledFieldsV1 => {
                model::bio::decode_detector_with_labeled_fields(text, bounds, probs, masked, breaks)
            }
        },
    )
}

#[cfg(test)]
#[path = "detector_postprocess_tests.rs"]
mod tests;
