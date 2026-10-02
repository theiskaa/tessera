//! Centered-logit regularization for bounded training memorization diagnostics only.

use anyhow::ensure;
use burn::prelude::*;

/// Validate the diagnostic coefficient and an optional initial parameter identity.
pub(crate) fn validate(coefficient: f64, expected_initial: Option<&str>) -> anyhow::Result<()> {
    ensure!(
        coefficient.is_finite() && (0.0..=1e-2).contains(&coefficient),
        "diagnostic logit penalty must be finite and in [0, 0.01]"
    );
    ensure!(
        expected_initial.is_none_or(
            |value| value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
        ),
        "expected initial parameter SHA must contain exactly 64 hexadecimal characters"
    );
    Ok(())
}

/// Refuse a different initialization before any optimization begins.
pub(crate) fn verify_initial(actual: &str, expected: Option<&str>) -> anyhow::Result<()> {
    ensure!(
        expected.is_none_or(|value| value.eq_ignore_ascii_case(actual)),
        "memorization initial parameter SHA does not match the required control"
    );
    Ok(())
}

/// Class-weighted mean squared centered logits over real tokens, excluding padding.
pub(crate) fn centered_penalty<B: Backend>(
    logits: Tensor<B, 3>,
    labels: Tensor<B, 2, Int>,
    mask: Tensor<B, 2>,
    weights: Tensor<B, 1>,
) -> Tensor<B, 1> {
    let [batch, length, _] = logits.dims();
    let centered = logits.clone() - logits.mean_dim(2);
    let padding = mask.clone().equal_elem(0.0).unsqueeze_dim::<3>(2);
    let variance = centered
        .mask_fill(padding, 0.0)
        .powf_scalar(2.0)
        .mean_dim(2)
        .squeeze_dim::<2>(2);
    let weighted_mask = weights
        .gather(0, labels.reshape([batch * length]))
        .reshape([batch, length])
        * mask;
    (variance * weighted_mask.clone()).sum() / weighted_mask.sum().clamp_min(1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    use burn::tensor::TensorData;

    #[test]
    fn analytic_centered_gradient_uses_class_weights_and_excludes_padding() {
        type B = Autodiff<NdArray>;
        let device = Default::default();
        let logits = Tensor::<B, 3>::from_data(
            TensorData::new(vec![1., 2., 3., 0., 3., 6., 100., -100., 99.], [1, 3, 3]),
            &device,
        )
        .require_grad();
        let labels =
            Tensor::<B, 2, Int>::from_data(TensorData::new(vec![0, 1, 0], [1, 3]), &device);
        let mask = Tensor::<B, 2>::from_data([[1., 1., 0.]], &device);
        let weights = Tensor::<B, 1>::from_data([1., 3., 1.], &device);
        let penalty = centered_penalty(logits.clone(), labels, mask, weights);
        let value: f32 = penalty.clone().into_scalar();
        assert!((value - 14. / 3.).abs() < 1e-5);
        let gradients = penalty.backward();
        let actual = logits
            .grad(&gradients)
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let expected = [-1. / 6., 0., 1. / 6., -1.5, 0., 1.5, 0., 0., 0.];
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn per_token_common_shifts_and_masked_values_do_not_change_penalty() {
        let device = Default::default();
        let labels = Tensor::<NdArray, 2, Int>::from_data([[0, 1, 0]], &device);
        let mask = Tensor::<NdArray, 2>::from_data([[1., 1., 0.]], &device);
        let weights = Tensor::<NdArray, 1>::from_data([1., 3., 1.], &device);
        let a =
            Tensor::<NdArray, 3>::from_data([[[1., 2., 3.], [0., 3., 6.], [7., 8., 9.]]], &device);
        let b = Tensor::<NdArray, 3>::from_data(
            [[[11., 12., 13.], [-10., -7., -4.], [900., -900., 0.]]],
            &device,
        );
        assert_eq!(
            centered_penalty(a, labels.clone(), mask.clone(), weights.clone()).into_scalar(),
            centered_penalty(b.clone(), labels.clone(), mask, weights.clone()).into_scalar()
        );
        let zero = Tensor::<NdArray, 2>::zeros([1, 3], &device);
        assert_eq!(centered_penalty(b, labels, zero, weights).into_scalar(), 0.);
    }

    #[test]
    fn rejects_invalid_coefficients_and_initial_identities() {
        for coefficient in [f64::NAN, f64::INFINITY, -1e-4, 0.010001] {
            assert!(validate(coefficient, None).is_err());
        }
        assert!(validate(0., None).is_ok());
        assert!(validate(1e-4, Some(&"a".repeat(64))).is_ok());
        assert!(validate(1e-4, Some("abc")).is_err());
        assert!(validate(1e-4, Some(&"g".repeat(64))).is_err());
        assert!(verify_initial(&"a".repeat(64), Some(&"A".repeat(64))).is_ok());
        assert!(verify_initial(&"a".repeat(64), Some(&"b".repeat(64))).is_err());
    }
}
