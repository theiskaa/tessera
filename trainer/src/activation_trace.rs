//! Compact activation moments from actual diagnostic forwards, excluding padded tokens.

use burn::prelude::*;

/// Real-token activation moments for one document or a complete batch.
#[derive(Debug, serde::Serialize)]
pub(crate) struct ActivationMoments {
    /// Number of real token positions included in the moments.
    pub(super) real_tokens: usize,
    /// Number of real scalar activations included in the moments.
    pub(super) elements: usize,
    /// Arithmetic mean over real scalar activations.
    pub(super) mean: f64,
    /// Root mean square over real scalar activations.
    pub(super) rms: f64,
    /// Maximum absolute real scalar activation.
    pub(super) max_abs: f64,
}

/// Compact statistics from a single actual forward activation.
#[derive(Debug, serde::Serialize)]
pub(crate) struct LayerActivation {
    /// Projection, residual block operation, or label head being measured.
    pub(super) layer: String,
    /// Channels or labels at each token position.
    pub(super) channels: usize,
    /// Batch moments, excluding every padded token position.
    pub(super) aggregate: ActivationMoments,
    /// Document moments in the same order as the input batch.
    pub(super) documents: Vec<ActivationMoments>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) normalization: Option<crate::residual_rms::NormalizationTrace>,
}

fn activation_moments(
    tokens: usize,
    channels: usize,
    sum: f64,
    squares: f64,
    max_abs: f64,
) -> ActivationMoments {
    let elements = tokens * channels;
    let denominator = elements.max(1) as f64;
    ActivationMoments {
        real_tokens: tokens,
        elements,
        mean: sum / denominator,
        rms: (squares / denominator).sqrt(),
        max_abs,
    }
}

pub(super) fn layer_activation<B: Backend>(
    layer: &str,
    activation: Tensor<B, 3>,
    mask: &[bool],
) -> anyhow::Result<LayerActivation> {
    let [batch, length, channels] = activation.dims();
    anyhow::ensure!(
        mask.len() == batch * length,
        "layer trace mask dimensions differ"
    );
    let values = activation
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()
        .map_err(|error| anyhow::anyhow!("reading {layer} activations: {error:?}"))?;
    let (mut total_tokens, mut total_sum, mut total_squares, mut total_max) = (0, 0., 0., 0f64);
    let mut documents = Vec::with_capacity(batch);
    for document in 0..batch {
        let (mut tokens, mut sum, mut squares, mut max_abs) = (0, 0., 0., 0f64);
        for position in 0..length {
            if !mask[document * length + position] {
                continue;
            }
            tokens += 1;
            let start = (document * length + position) * channels;
            for &value in &values[start..start + channels] {
                anyhow::ensure!(value.is_finite(), "non-finite real activation in {layer}");
                let value = f64::from(value);
                sum += value;
                squares += value * value;
                max_abs = max_abs.max(value.abs());
            }
        }
        documents.push(activation_moments(tokens, channels, sum, squares, max_abs));
        total_tokens += tokens;
        total_sum += sum;
        total_squares += squares;
        total_max = total_max.max(max_abs);
    }
    Ok(LayerActivation {
        layer: layer.into(),
        channels,
        aggregate: activation_moments(total_tokens, channels, total_sum, total_squares, total_max),
        documents,
        normalization: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn layer_moments_exclude_padding_and_match_independent_scalar_statistics() {
        let device = Default::default();
        let activation = Tensor::<NdArray, 3>::from_data(
            [
                [[1., -3.], [2., 4.], [999., -999.]],
                [[-2., 6.], [1000., -1000.], [888., 999.]],
            ],
            &device,
        );
        let measured = layer_activation(
            "fixture",
            activation.clone(),
            &[true, true, false, true, false, false],
        )
        .unwrap();
        assert_eq!(measured.aggregate.elements, 6);
        assert!((measured.aggregate.mean - 8. / 6.).abs() < 1e-12);
        assert!((measured.aggregate.rms - (70f64 / 6.).sqrt()).abs() < 1e-12);
        assert_eq!(measured.aggregate.max_abs, 6.);
        assert_eq!(measured.documents[0].mean, 1.);
        assert_eq!(measured.documents[0].rms, 7.5f64.sqrt());
        assert_eq!(measured.documents[0].max_abs, 4.);
        assert_eq!(measured.documents[1].mean, 2.);
        assert_eq!(measured.documents[1].rms, 20f64.sqrt());
        let empty = layer_activation("empty", activation, &[false; 6]).unwrap();
        assert_eq!(empty.aggregate.rms, 0.);
        assert_eq!(empty.aggregate.mean, 0.);
        assert_eq!(empty.aggregate.max_abs, 0.);
    }
}
