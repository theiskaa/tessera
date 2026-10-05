//! Exact zero-extension of the detector projection; no initialization or optimizer state.

use crate::net::TaggerNet;
use crate::quantize::F32Tensor;
use anyhow::{Context, ensure};
use burn::prelude::*;
use serde_json::{Value, json};

/// Append two zero input columns in shipping [out,in] layout; preserve all original bits.
pub(crate) fn extend_tensors(original: &[F32Tensor]) -> anyhow::Result<Vec<F32Tensor>> {
    ensure!(!original.is_empty(), "missing migration parameters");
    let mut names = std::collections::BTreeSet::new();
    for tensor in original {
        ensure!(
            names.insert(&tensor.name)
                && tensor
                    .shape
                    .iter()
                    .try_fold(1usize, |n, &d| n.checked_mul(d))
                    == Some(tensor.data.len())
                && tensor.data.iter().all(|v| v.is_finite()),
            "duplicate, malformed or nonfinite original tensor"
        );
    }
    let mut extended = original.to_vec();
    let proj = extended
        .iter_mut()
        .find(|t| t.name == "detector.proj.weight")
        .context("missing detector projection")?;
    ensure!(
        proj.shape == [96, 87],
        "migration requires original Context96 detector projection [96,87]"
    );
    proj.data = proj
        .data
        .chunks_exact(87)
        .flat_map(|row| row.iter().copied().chain([0.0, 0.0]))
        .collect();
    proj.shape[1] = 89;
    Ok(extended)
}

/// Compare the complete parameter set against the only permitted zero-column delta.
pub(crate) fn verify_tensors(original: &[F32Tensor], extended: &[F32Tensor]) -> anyhow::Result<()> {
    let expected = extend_tensors(original)?;
    ensure!(
        expected.len() == extended.len(),
        "migration parameter set changed"
    );
    for (a, b) in expected.iter().zip(extended) {
        ensure!(
            a.name == b.name
                && a.shape == b.shape
                && a.data.len() == b.data.len()
                && a.data
                    .iter()
                    .zip(&b.data)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
            "migration changed original bits, parameter order or new zeros"
        );
    }
    Ok(())
}

fn extend_projection<B: Backend>(
    value: Tensor<B, 2>,
    channels: usize,
    device: &B::Device,
) -> (Tensor<B, 2>, bool) {
    let original_require_grad = value.is_require_grad();
    let concatenated = Tensor::cat(vec![value, Tensor::zeros([2, channels], device)], 0);
    let intermediate_require_grad = concatenated.is_require_grad();
    // The wider projection replaces a parameter leaf, not an operation linked to old-width weights.
    (
        concatenated
            .detach()
            .set_require_grad(original_require_grad),
        intermediate_require_grad,
    )
}

/// Extend an already source-proved, directly loaded AD detector without RNG or graph reinit.
pub(crate) fn extend_model<B: Backend>(
    mut model: TaggerNet<B>,
    device: &B::Device,
) -> anyhow::Result<(TaggerNet<B>, Value)> {
    let original = crate::quantize::extract(&model, "detector");
    let expected = extend_tensors(&original)?;
    let [rows, channels] = model.proj.weight.dims();
    ensure!(
        rows == 87
            && channels == 96
            && model.flag_bits() == 23
            && model.blocks.len() == 7
            && model.head.weight.dims()[1] == 7,
        "migration source graph differs"
    );
    let projection_id = model.proj.weight.id;
    let original_projection_require_grad = model.proj.weight.val().is_require_grad();
    let mut concatenated_projection_require_grad = false;
    model.proj.weight = model.proj.weight.map(|value| {
        let (leaf, intermediate) = extend_projection(value, channels, device);
        concatenated_projection_require_grad = intermediate;
        leaf
    });
    let migrated_projection_require_grad = model.proj.weight.val().is_require_grad();
    ensure!(
        model.proj.weight.id == projection_id
            && original_projection_require_grad == migrated_projection_require_grad,
        "projection parameter ID or gradient flag changed during leaf migration"
    );
    let actual = crate::quantize::extract(&model, "detector");
    verify_tensors(&original, &actual)?;
    verify_tensors(&original, &expected)?;
    Ok((
        model,
        json!({"contract":"detector-legacy23-to-tab-cells25-zero-projection-v2",
        "original_parameter_sha256":crate::training_diagnostic::parameter_sha256(&original),
        "migrated_parameter_sha256":crate::training_diagnostic::parameter_sha256(&actual),
        "original_projection_shape":[96,87],"migrated_projection_shape":[96,89],
        "added_parameters":192,"added_f32_bytes":768,"new_columns":"positive f32 zeros",
        "original_parameter_bits_preserved":true,"all_parameters_finite":true,
        "model_initializations":0,"rng_calls":0,"parameter_ids":"preserved by Param::map","projection_parameter_id":projection_id.serialize(),"original_projection_require_grad":original_projection_require_grad,"concatenated_projection_require_grad":concatenated_projection_require_grad,"migrated_projection_require_grad":migrated_projection_require_grad,"projection_leaf_conversion":"detach concatenation then restore original require-grad flag; no old-width ancestor graph","optimizer_state":"not touched",
        "new_features":"AFTER_TAB bit23; BEFORE_TAB bit24","forward_equality_executed":false}),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn concatenated_ad_projection_becomes_full_trainable_leaf_without_old_ancestor() {
        use burn::backend::{Autodiff, NdArray};
        use burn::module::Param;
        type AD = Autodiff<NdArray>;
        let device = Default::default();
        let source =
            Tensor::<AD, 2>::from_floats([[-0.0, 1.0], [2.0, -3.0]], &device).require_grad();
        let param = Param::from_tensor(source.clone());
        let id = param.id;
        let mut intermediate_flag = true;
        let param = param.map(|value| {
            let (leaf, intermediate) = extend_projection(value, 2, &device);
            intermediate_flag = intermediate;
            leaf
        });
        assert!(!intermediate_flag);
        assert_eq!(param.id, id);
        let leaf = param.val();
        assert!(leaf.is_require_grad());
        assert_eq!(leaf.dims(), [4, 2]);
        let bits = leaf.clone().into_data().to_vec::<f32>().unwrap();
        assert_eq!(bits[0].to_bits(), (-0.0f32).to_bits());
        assert_eq!(&bits[1..4], &[1.0, 2.0, -3.0]);
        assert!(bits[4..].iter().all(|v| v.to_bits() == 0));
        let gradients = leaf.clone().sum().backward();
        let gradient = leaf.grad(&gradients).unwrap();
        assert_eq!(gradient.dims(), [4, 2]);
        assert_eq!(gradient.into_data().to_vec::<f32>().unwrap(), vec![1.0; 8]);
        assert!(source.grad(&gradients).is_none());
        let plain = Tensor::<NdArray, 2>::zeros([2, 2], &device);
        let (plain, intermediate) = extend_projection(plain, 2, &device);
        assert!(!intermediate && !plain.is_require_grad());
    }
    fn original() -> Vec<F32Tensor> {
        vec![
            F32Tensor {
                name: "detector.proj.weight".into(),
                shape: vec![96, 87],
                data: (0..96 * 87)
                    .map(|i| {
                        if i % 2 == 0 {
                            i as f32 / 8.0
                        } else {
                            -(i as f32) / 8.0
                        }
                    })
                    .collect(),
            },
            F32Tensor {
                name: "detector.head.bias".into(),
                shape: vec![2],
                data: vec![-0.0, 1.0],
            },
        ]
    }
    #[test]
    fn exact_zero_columns_preserve_every_original_f32_bit() {
        let source = original();
        let migrated = extend_tensors(&source).unwrap();
        verify_tensors(&source, &migrated).unwrap();
        assert_eq!(migrated[0].shape, [96, 89]);
        assert_eq!(migrated[0].data.len() - source[0].data.len(), 192);
        for (a, b) in source[0]
            .data
            .chunks_exact(87)
            .zip(migrated[0].data.chunks_exact(89))
        {
            assert_eq!(a, &b[..87]);
            assert_eq!(b[87].to_bits(), 0);
            assert_eq!(b[88].to_bits(), 0);
        }
        assert_eq!(migrated[1].data[0].to_bits(), (-0.0f32).to_bits());
        let mut changed = migrated.clone();
        changed[1].data[0] = 0.0;
        assert!(verify_tensors(&source, &changed).is_err());
        changed = migrated.clone();
        changed[0].data[87] = 1.0;
        assert!(verify_tensors(&source, &changed).is_err());
    }
    #[test]
    fn wrong_shape_duplicate_and_nonfinite_sources_are_rejected() {
        let mut s = original();
        s[0].shape = vec![96, 89];
        assert!(extend_tensors(&s).is_err());
        s = original();
        s[0].data[1] = f32::NAN;
        assert!(extend_tensors(&s).is_err());
        s = original();
        s.push(s[1].clone());
        assert!(extend_tensors(&s).is_err());
    }
}
