//! Parameter-free channel RMS for one non-exportable forward diagnostic.

use burn::prelude::*;

const EPSILON: f64 = 1e-5;

#[derive(Debug, serde::Serialize)]
pub(crate) struct NormalizationTrace {
    pub(crate) real_tokens: usize,
    pub(crate) max_token_channel_rms: f64,
    pub(crate) native_mean_square_finite: bool,
    pub(crate) native_denominator_finite: bool,
}

fn normalize_inner<B: Backend>(
    residual: Tensor<B, 3>,
    keep: Tensor<B, 3>,
) -> anyhow::Result<Tensor<B, 3>> {
    let [batch, channels, length] = residual.dims();
    anyhow::ensure!(
        channels > 0 && keep.dims() == [batch, 1, length],
        "RMS mask dimensions differ"
    );
    let residual = residual.mask_fill(keep.clone().equal_elem(0.), 0.);
    let mean_square = residual.clone().square().mean_dim(1);
    // A finite residual can overflow its squared reduction; division by infinity would hide it.
    anyhow::ensure!(
        mean_square
            .clone()
            .is_finite()
            .all()
            .into_scalar()
            .elem::<bool>(),
        "non-finite residual RMS mean-square"
    );
    let denominator = (mean_square + EPSILON).sqrt();
    anyhow::ensure!(
        denominator
            .clone()
            .is_finite()
            .all()
            .into_scalar()
            .elem::<bool>(),
        "non-finite residual RMS denominator"
    );
    Ok((residual / denominator) * keep)
}

pub(crate) fn normalize<B: Backend>(
    residual: Tensor<B, 3>,
    keep: Tensor<B, 3>,
) -> anyhow::Result<Tensor<B, 3>> {
    normalize_inner(residual, keep)
}

pub(crate) fn normalize_traced<B: Backend>(
    residual: Tensor<B, 3>,
    keep: Tensor<B, 3>,
) -> anyhow::Result<(Tensor<B, 3>, NormalizationTrace)> {
    let real_count: f64 = keep.clone().sum().into_scalar().elem();
    let real_tokens = real_count as usize;
    let normalized = normalize_inner(residual, keep.clone())?;
    let token_rms = normalized.clone().square().mean_dim(1).sqrt() * keep;
    let maximum: f64 = token_rms.max().into_scalar().elem();
    anyhow::ensure!(maximum.is_finite(), "non-finite normalized token RMS");
    Ok((
        normalized,
        NormalizationTrace {
            real_tokens,
            max_token_channel_rms: maximum,
            native_mean_square_finite: true,
            native_denominator_finite: true,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};

    #[test]
    fn native_gradient_matches_channel_rms_jacobian_and_padding_is_zero() {
        type B = Autodiff<NdArray>;
        let device = Default::default();
        let values: [[[f32; 4]; 3]; 1] = [[
            [0., 1e-4, 2., 9.],
            [0., -2e-4, -3., -8.],
            [0., 3e-4, 5., 7.],
        ]];
        let x = Tensor::<B, 3>::from_data(values, &device).require_grad();
        let keep = Tensor::<B, 3>::from_data([[[1., 1., 1., 0.]]], &device);
        let upstream: [[[f32; 4]; 3]; 1] =
            [[[1., 2., 3., 4.], [2., -1., 4., 5.], [-3., 1., -2., 6.]]];
        let y = normalize(x.clone(), keep).unwrap();
        let data = y.clone().into_data().to_vec::<f32>().unwrap();
        let gradients = (y * Tensor::from_data(upstream, &device)).sum().backward();
        let actual = x
            .grad(&gradients)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        for token in 0..4 {
            let square = (0..3)
                .map(|channel| f64::from(values[0][channel][token]).powi(2))
                .sum::<f64>()
                / 3.;
            let scale = (square + EPSILON).sqrt();
            let dot = (0..3)
                .map(|channel| {
                    f64::from(values[0][channel][token]) * f64::from(upstream[0][channel][token])
                })
                .sum::<f64>();
            let mut rms = 0.;
            for channel in 0..3 {
                let index = channel * 4 + token;
                let expected = if token == 3 {
                    0.
                } else {
                    f64::from(upstream[0][channel][token]) / scale
                        - f64::from(values[0][channel][token]) * dot / (3. * scale.powi(3))
                };
                assert!((f64::from(actual[index]) - expected).abs() < 1e-3 + expected.abs() * 2e-5);
                rms += f64::from(data[index]).powi(2) / 3.;
                if token == 3 {
                    assert_eq!(data[index], 0.);
                    assert_eq!(actual[index], 0.);
                }
            }
            assert!(rms.sqrt() <= 1. + 2e-6);
        }
    }

    #[test]
    fn zero_padding_stays_zero_and_overflow_is_refused() {
        let device = Default::default();
        let output = normalize(
            Tensor::<NdArray, 3>::zeros([2, 96, 3], &device),
            Tensor::zeros([2, 1, 3], &device),
        )
        .unwrap();
        assert!(
            output
                .into_data()
                .to_vec::<f32>()
                .unwrap()
                .iter()
                .all(|x| *x == 0.)
        );
        let huge = Tensor::<NdArray, 3>::from_data([[[1e30f32], [1e30]]], &device);
        assert!(normalize(huge, Tensor::ones([1, 1, 1], &device)).is_err());
    }

    #[test]
    fn irrelevant_overflowing_or_nonfinite_padding_is_cleared_before_reduction() {
        let device = Default::default();
        let residual = Tensor::<NdArray, 3>::from_data(
            [[
                [2., f32::INFINITY, f32::NAN, 1e30],
                [3., f32::NEG_INFINITY, f32::NAN, -1e30],
            ]],
            &device,
        );
        let keep = Tensor::<NdArray, 3>::from_data([[[1., 0., 0., 0.]]], &device);
        let masked = normalize(residual, keep)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let alone = normalize(
            Tensor::<NdArray, 3>::from_data([[[2.], [3.]]], &device),
            Tensor::ones([1, 1, 1], &device),
        )
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
        assert_eq!(masked[0], alone[0]);
        assert_eq!(masked[4], alone[1]);
        for i in [1, 2, 3, 5, 6, 7] {
            assert_eq!(masked[i], 0.);
        }
    }
}
