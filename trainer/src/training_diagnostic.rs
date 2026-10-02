//! Fixed schedules and parameter identities for bounded training comparisons.

use anyhow::ensure;
use sha2::{Digest, Sha256};

use crate::config::{Config, Task};
use crate::quantize::F32Tensor;

/// Reject a diagnostic schedule when no bounded detector run was requested.
pub(crate) fn validate_launch(cfg: &Config, max_steps: Option<usize>) -> anyhow::Result<()> {
    if let Some(horizon) = cfg.train.diagnostic_schedule_steps {
        ensure!(
            cfg.task == Task::Detector,
            "diagnostic schedules require a detector"
        );
        let cap = max_steps.ok_or_else(|| {
            anyhow::anyhow!("train.diagnostic_schedule_steps requires --max-steps")
        })?;
        ensure!(
            cfg.train.warmup_steps < cap && cap <= horizon,
            "diagnostic budget must exceed warmup and fit within its learning-rate horizon"
        );
    }
    Ok(())
}

/// The decay horizon, independent of epoch size only for an explicit diagnostic schedule.
pub(crate) fn horizon(cfg: &Config, scheduled_steps: usize) -> usize {
    cfg.train
        .diagnostic_schedule_steps
        .unwrap_or(scheduled_steps)
}

/// Hash parameter names, shapes and exact f32 values, excluding random recorder IDs.
pub(crate) fn parameter_sha256(tensors: &[F32Tensor]) -> String {
    let mut ordered: Vec<_> = tensors.iter().collect();
    ordered.sort_by(|a, b| a.name.cmp(&b.name));
    let mut digest = Sha256::new();
    digest.update((ordered.len() as u64).to_le_bytes());
    for tensor in ordered {
        digest.update((tensor.name.len() as u64).to_le_bytes());
        digest.update(tensor.name.as_bytes());
        digest.update((tensor.shape.len() as u64).to_le_bytes());
        for &dimension in &tensor.shape {
            digest.update((dimension as u64).to_le_bytes());
        }
        digest.update((tensor.data.len() as u64).to_le_bytes());
        for &value in &tensor.data {
            digest.update(value.to_bits().to_le_bytes());
        }
    }
    format!("{:x}", digest.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn config() -> Config {
        crate::config::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/detector-shared-v6.toml"),
        )
        .unwrap()
    }

    #[test]
    fn explicit_schedule_keeps_the_same_rate_for_different_epoch_sizes() {
        let mut cfg = config();
        assert_eq!(horizon(&cfg, 6925), 6925);
        assert_eq!(horizon(&cfg, 13950), 13950);
        cfg.train.diagnostic_schedule_steps = Some(6925);
        validate_launch(&cfg, Some(2000)).unwrap();
        for step in 0..2000 {
            assert_eq!(
                crate::train::lr_at(step, horizon(&cfg, 6925), 1e-4, 500),
                crate::train::lr_at(step, horizon(&cfg, 13950), 1e-4, 500),
            );
        }
    }

    #[test]
    fn diagnostic_schedule_cannot_launch_a_full_run_or_an_invalid_budget() {
        let mut cfg = config();
        cfg.train.diagnostic_schedule_steps = Some(6925);
        assert!(validate_launch(&cfg, None).is_err());
        assert!(validate_launch(&cfg, Some(500)).is_err());
        assert!(validate_launch(&cfg, Some(6926)).is_err());
        cfg.task = Task::Parser;
        assert!(validate_launch(&cfg, Some(2000)).is_err());
    }

    #[test]
    fn parameter_identity_ignores_order_but_detects_values_names_and_shapes() {
        let a = F32Tensor {
            name: "a".into(),
            shape: vec![2],
            data: vec![1.0, 2.0],
        };
        let b = F32Tensor {
            name: "b".into(),
            shape: vec![1],
            data: vec![3.0],
        };
        let expected = parameter_sha256(&[a.clone(), b.clone()]);
        assert_eq!(expected, parameter_sha256(&[b.clone(), a.clone()]));
        let mut changed = a.clone();
        changed.data[0] = 4.0;
        assert_ne!(expected, parameter_sha256(&[changed, b.clone()]));
        let mut changed = a.clone();
        changed.name = "c".into();
        assert_ne!(expected, parameter_sha256(&[changed, b.clone()]));
        let mut changed = a;
        changed.shape = vec![1, 2];
        assert_ne!(expected, parameter_sha256(&[changed, b]));
    }
}
