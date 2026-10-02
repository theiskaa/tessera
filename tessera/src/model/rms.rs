//! Checked post-residual channel RMS for the explicitly bound context detector.

use super::context::CONTEXT96_RMS_EPSILON;
use crate::Error;

pub(crate) fn normalize(values: &mut [f32], channels: usize) -> Result<(), Error> {
    for token in values.chunks_exact_mut(channels) {
        let mean_square = token.iter().map(|x| x * x).sum::<f32>() / channels as f32;
        let denominator = (mean_square + CONTEXT96_RMS_EPSILON).sqrt();
        if !mean_square.is_finite() || !denominator.is_finite() {
            return Err(Error::Inference { stage: "detector" });
        }
        for value in token {
            *value /= denominator;
            if !value.is_finite() {
                return Err(Error::Inference { stage: "detector" });
            }
        }
    }
    Ok(())
}
