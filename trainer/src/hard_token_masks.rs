//! Validate disjoint hard-token masks before native batch and selector preparation.

use anyhow::{Context, ensure};

const GROUPS: usize = 3;
const MAX_COEFFICIENT: f32 = 1.0 / 16.0;

/// Validated masks in person, org_contrast, address order, bound to the real-token mask.
pub(crate) struct HardMasks;

fn binary(values: &[f32]) -> bool {
    values.iter().all(|&value| value == 0.0 || value == 1.0)
}

impl HardMasks {
    /// Validate all raw masks before any auxiliary tensor is constructed.
    pub(crate) fn new(
        shape: [usize; 2],
        real: &[f32],
        groups: [Vec<f32>; GROUPS],
        coefficients: [f32; GROUPS],
    ) -> anyhow::Result<Self> {
        let count = shape[0]
            .checked_mul(shape[1])
            .context("hard mask dimensions overflow")?;
        ensure!(shape.iter().all(|&size| size > 0), "empty hard mask shape");
        ensure!(
            real.len() == count && binary(real),
            "invalid real-token mask"
        );
        ensure!(
            coefficients
                .iter()
                .all(|&value| value.is_finite() && (0.0..=MAX_COEFFICIENT).contains(&value)),
            "hard coefficients must be finite and in [0, 1/16]"
        );
        for group in &groups {
            ensure!(
                group.len() == count && binary(group),
                "invalid hard group mask"
            );
        }
        for index in 0..count {
            let selected = groups.iter().filter(|group| group[index] == 1.0).count();
            ensure!(selected <= 1, "hard groups overlap at token {index}");
            ensure!(
                real[index] == 1.0 || selected == 0,
                "hard mask selects padding"
            );
        }
        Ok(Self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn real() -> Vec<f32> {
        vec![1., 1., 1., 0., 1., 0.]
    }

    fn groups() -> [Vec<f32>; GROUPS] {
        [
            vec![1., 0., 0., 0., 0., 0.],
            vec![0., 1., 0., 0., 0., 0.],
            vec![0., 0., 0., 0., 1., 0.],
        ]
    }

    #[test]
    fn raw_masks_reject_bad_shape_padding_overlap_and_coefficients() {
        for bad in [f32::NAN, f32::INFINITY, -1., 0.5, 2.] {
            let mut masks = groups();
            masks[0][0] = bad;
            assert!(HardMasks::new([2, 3], &real(), masks, [MAX_COEFFICIENT; GROUPS]).is_err());
        }
        for coefficient in [f32::NAN, f32::INFINITY, -0.01, MAX_COEFFICIENT + 0.001] {
            assert!(HardMasks::new([2, 3], &real(), groups(), [coefficient; GROUPS]).is_err());
        }
        let mut masks = groups();
        masks[1][0] = 1.;
        assert!(HardMasks::new([2, 3], &real(), masks, [0.; GROUPS]).is_err());
        let mut masks = groups();
        masks[0][3] = 1.;
        assert!(HardMasks::new([2, 3], &real(), masks, [MAX_COEFFICIENT; GROUPS]).is_err());
        let mut masks = groups();
        masks[0].pop();
        assert!(HardMasks::new([2, 3], &real(), masks, [MAX_COEFFICIENT; GROUPS]).is_err());
        assert!(HardMasks::new([0, 3], &[], groups(), [MAX_COEFFICIENT; GROUPS]).is_err());
        assert!(HardMasks::new([usize::MAX, 2], &[], groups(), [MAX_COEFFICIENT; GROUPS]).is_err());
        assert!(HardMasks::new([2, 3], &[1.; 5], groups(), [MAX_COEFFICIENT; GROUPS]).is_err());
        let mut bad_real = real();
        bad_real[0] = f32::NAN;
        assert!(HardMasks::new([2, 3], &bad_real, groups(), [MAX_COEFFICIENT; GROUPS]).is_err());
    }
}
