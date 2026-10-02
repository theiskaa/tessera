//! Optional zero-update audit of the existing planned fullmix biography schedule.

#[path = "fullmix_exposure_targets.rs"]
mod targets;

#[path = "fullmix_completed.rs"]
mod completed;
pub(crate) use completed::Completed;

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, ensure};
use serde_json::{Value, json};

use crate::config::Config;
use crate::dataset::Encoded;
use crate::detector::DetectorDoc;
use targets::{Manifest, Resolved, hash};

pub(crate) struct Request {
    manifest: Manifest,
    reference: Value,
    pins: BTreeMap<String, String>,
    out: PathBuf,
}

pub(crate) struct Plan {
    targets: Vec<Resolved>,
    reference: Value,
    pins: BTreeMap<String, String>,
    out: PathBuf,
}

impl Request {
    pub(crate) fn load(
        config: &Path,
        cfg: &Config,
        target_path: &Path,
        out: &Path,
    ) -> anyhow::Result<Self> {
        ensure!(!out.exists(), "exposure output already exists");
        let manifest: Manifest = serde_json::from_slice(&std::fs::read(target_path)?)?;
        targets::validate_manifest(&manifest)?;
        let config_hash = hash(config)?;
        ensure!(
            config_hash == manifest.config_sha256,
            "exposure configuration changed"
        );
        let detector = cfg
            .detector
            .as_ref()
            .context("missing detector configuration")?;
        ensure!(
            cfg.train.batch_size == 32
                && detector
                    .learning_check
                    .as_ref()
                    .is_some_and(|check| check.start_step == 4000),
            "exposure requires the existing physical batch32, step4000 learning schedule"
        );
        let reference_path = Path::new(&manifest.reference_audit.path);
        ensure!(
            hash(reference_path)? == manifest.reference_audit.sha256,
            "reference schedule audit changed"
        );
        let reference: Value = serde_json::from_slice(&std::fs::read(reference_path)?)?;
        ensure!(
            reference["config_sha256"] == config_hash
                && reference["learning_check_exposure"]["optimizer_steps"] == 4000,
            "reference schedule has a different config or bounded prefix"
        );
        let mut pins = BTreeMap::new();
        for path in [
            config.to_path_buf(),
            target_path.to_path_buf(),
            reference_path.to_path_buf(),
        ] {
            pins.insert(path.to_string_lossy().into_owned(), hash(&path)?);
        }
        let mut dataset_pins = BTreeMap::new();
        let generator = Path::new(&cfg.data.manifests).join("detector-synthetic.json");
        dataset_pins.insert(generator.to_string_lossy().into_owned(), hash(&generator)?);
        for split in crate::data::Split::ALL {
            let path = Path::new(&cfg.data.processed).join(format!("{}.parquet", split.name()));
            dataset_pins.insert(path.to_string_lossy().into_owned(), hash(&path)?);
        }
        for path in &detector.silver {
            dataset_pins.insert(path.clone(), hash(Path::new(path))?);
        }
        ensure!(
            dataset_pins == manifest.input_sha256,
            "fullmix dataset pins changed or are incomplete"
        );
        pins.extend(dataset_pins);
        Ok(Self {
            manifest,
            reference,
            pins,
            out: out.to_path_buf(),
        })
    }

    pub(crate) fn resolve(
        mut self,
        cfg: &Config,
        docs: &[DetectorDoc],
        source_pieces: &[usize],
        synthetic_count: usize,
    ) -> anyhow::Result<Plan> {
        let targets = targets::resolve(
            self.manifest,
            cfg,
            docs,
            source_pieces,
            synthetic_count,
            &mut self.pins,
        )?;
        Ok(Plan {
            targets,
            reference: self.reference,
            pins: self.pins,
            out: self.out,
        })
    }
}

pub(crate) fn denominator(
    items: &[Encoded],
    indices: &[usize],
    weights: &[f32],
) -> anyhow::Result<f64> {
    ensure!(
        !indices.is_empty() && indices.len() <= 32,
        "invalid planned batch size"
    );
    ensure!(
        weights.len() == 7
            && weights.iter().all(|&weight| weight.is_finite()
                && weight > 0.0
                && weight <= 3.0
                && (weight * 2.0).fract() == 0.0),
        "audit needs positive native integer or half-integer weights at most3"
    );
    let mut mass = 0.0;
    for &index in indices {
        let item = items
            .get(index)
            .context("planned draw is outside native training items")?;
        ensure!(
            item.labels.len() <= 900 && item.labels.len() == item.token_spans.len(),
            "planned item has invalid native token dimensions"
        );
        for &label in &item.labels {
            mass += f64::from(
                *weights
                    .get(usize::from(label))
                    .context("invalid native BIO label")?,
            );
        }
    }
    // Nonnegative half-integer partial sums below86400 are exact under any f32 reduction order.
    ensure!(
        mass.is_finite() && mass <= 86400.0 && f64::from(mass as f32) == mass,
        "denominator is not proven native-f32 exact"
    );
    Ok(mass.max(1.0))
}

#[derive(Default)]
struct Accumulated {
    presentations: usize,
    inverse_denominator_sum: f64,
    lr_inverse_denominator_sum: f64,
}

impl Accumulated {
    fn add(&mut self, multiplicity: usize, denominator: f64, lr: f64) -> anyhow::Result<()> {
        ensure!(
            denominator.is_finite() && denominator >= 1.0 && lr.is_finite() && lr > 0.0,
            "invalid coefficient denominator or learning rate"
        );
        self.presentations = self
            .presentations
            .checked_add(multiplicity)
            .context("target presentation overflow")?;
        for _ in 0..multiplicity {
            self.inverse_denominator_sum += 1.0 / denominator;
            self.lr_inverse_denominator_sum += lr / denominator;
        }
        ensure!(
            self.inverse_denominator_sum.is_finite() && self.lr_inverse_denominator_sum.is_finite(),
            "coefficient sum overflow"
        );
        Ok(())
    }
}

/// Normalizes native numeric JSON through the same parser as pinned references.
pub(crate) fn reference_representation(value: &Value) -> anyhow::Result<Value> {
    // serdeJSON's default decimal parser need not recover the original native f64 bits.
    Ok(serde_json::from_slice(&serde_json::to_vec(value)?)?)
}

fn verify_reference(
    reference: &Value,
    exposure: &Value,
    cfg: &Config,
    horizon: usize,
) -> anyhow::Result<()> {
    ensure!(
        reference["learning_rate_horizon_steps"] == horizon
            && reference["warmup_steps"] == cfg.train.warmup_steps,
        "reference learning-rate schedule differs"
    );
    ensure!(
        reference["learning_check_exposure"]["exposure"] == reference_representation(exposure)?,
        "replayed fullmix aggregate differs from the reference schedule audit"
    );
    Ok(())
}

fn verify_replay(existing: &Value, reference: &Value, replay: &Value) -> anyhow::Result<()> {
    ensure!(
        existing["exposure"] == *replay && existing["optimizer_steps"] == 4000,
        "receipt differs from the existing finalized native schedule"
    );
    ensure!(
        reference["learning_check_exposure"] == reference_representation(existing)?,
        "replayed prefix counts or coefficients differ from reference"
    );
    Ok(())
}

impl Plan {
    pub(crate) fn write(
        self,
        cfg: &Config,
        items: &[Encoded],
        batches: &[Vec<usize>],
        synthetic_count: usize,
        horizon: usize,
        existing: &Value,
    ) -> anyhow::Result<()> {
        ensure!(
            batches.len() == 4000,
            "planned audit prefix must contain exactly4000 batches"
        );
        let weights = &cfg
            .detector
            .as_ref()
            .context("missing detector configuration")?
            .class_weights;
        let replay = serde_json::to_value(crate::training_audit::audit_epoch(
            items,
            batches,
            synthetic_count,
            weights,
        )?)?;
        verify_reference(&self.reference, &replay, cfg, horizon)?;
        verify_replay(existing, &self.reference, &replay)?;
        let mut totals: Vec<_> = self
            .targets
            .iter()
            .map(|_| Accumulated::default())
            .collect();
        let mut records = Vec::with_capacity(batches.len());
        for (step, indices) in batches.iter().enumerate() {
            let denominator = denominator(items, indices, weights)?;
            let lr = crate::train::lr_at(
                step,
                horizon,
                cfg.train.learning_rate,
                cfg.train.warmup_steps,
            );
            let mut multiplicities = Vec::with_capacity(self.targets.len());
            for (target, total) in self.targets.iter().zip(&mut totals) {
                let multiplicity = indices
                    .iter()
                    .filter(|&&index| index == target.original_index)
                    .count();
                total.add(multiplicity, denominator, lr)?;
                multiplicities.push(multiplicity);
            }
            records.push(json!({"planned_update": step + 1, "optimizer_step_index": step, "original_indices": indices, "native_weighted_token_denominator": denominator, "scheduled_learning_rate": lr, "target_multiplicities": multiplicities}));
        }
        let batch_sequence_sha256 = crate::export::sha256_hex(&serde_json::to_vec(&records)?);
        let documents: Vec<_> = self.targets.into_iter().zip(totals).map(|(target, total)| {
            let tokens: Vec<_> = target.tokens.into_iter().map(|mut token| -> anyhow::Result<Value> {
                let label = token["bio_label"].as_u64().context("target BIO identity absent")? as usize;
                let weight = f64::from(*weights.get(label).context("target BIO label out of range")?);
                token["native_class_weight"] = json!(weight);
                token["presentations"] = json!(total.presentations);
                token["cumulative_normalized_coefficient"] = json!(weight * total.inverse_denominator_sum);
                token["lr_cumulative_normalized_coefficient"] = json!(weight * total.lr_inverse_denominator_sum);
                Ok(token)
            }).collect::<anyhow::Result<_>>()?;
            Ok(json!({"identity": target.identity, "original_encoded_index": target.original_index, "native_encoded_sha256": target.encoding_sha256, "native_label_token_counts": target.label_token_counts, "presentations": total.presentations, "inverse_native_denominator_sum": total.inverse_denominator_sum, "lr_inverse_native_denominator_sum": total.lr_inverse_denominator_sum, "gold_tokens": tokens}))
        }).collect::<anyhow::Result<_>>()?;
        for (path, expected) in &self.pins {
            ensure!(
                hash(Path::new(path))? == *expected,
                "exposure input changed during native audit: {path}"
            );
        }
        let implementation_sha256: BTreeMap<_, _> = [
            (
                "trainer/src/train.rs",
                include_bytes!("train.rs").as_slice(),
            ),
            (
                "trainer/src/training_audit.rs",
                include_bytes!("training_audit.rs").as_slice(),
            ),
            (
                "trainer/src/fullmix_exposure.rs",
                include_bytes!("fullmix_exposure.rs").as_slice(),
            ),
            (
                "trainer/src/fullmix_exposure_targets.rs",
                include_bytes!("fullmix_exposure_targets.rs").as_slice(),
            ),
            (
                "trainer/src/detector.rs",
                include_bytes!("detector.rs").as_slice(),
            ),
            (
                "trainer/src/dataset.rs",
                include_bytes!("dataset.rs").as_slice(),
            ),
            ("trainer/src/main.rs", include_bytes!("main.rs").as_slice()),
            (
                "trainer/src/config.rs",
                include_bytes!("config.rs").as_slice(),
            ),
            ("trainer/src/data.rs", include_bytes!("data.rs").as_slice()),
            (
                "tessera/src/token.rs",
                include_bytes!("../../tessera/src/token.rs").as_slice(),
            ),
            (
                "tessera/src/features.rs",
                include_bytes!("../../tessera/src/features.rs").as_slice(),
            ),
            (
                "tessera/src/rules/mod.rs",
                include_bytes!("../../tessera/src/rules/mod.rs").as_slice(),
            ),
            (
                "tessera/src/rules/email.rs",
                include_bytes!("../../tessera/src/rules/email.rs").as_slice(),
            ),
            (
                "tessera/src/rules/phone.rs",
                include_bytes!("../../tessera/src/rules/phone.rs").as_slice(),
            ),
            (
                "tessera/src/rules/phone_tables.rs",
                include_bytes!("../../tessera/src/rules/phone_tables.rs").as_slice(),
            ),
            (
                "tessera/src/rules/region.rs",
                include_bytes!("../../tessera/src/rules/region.rs").as_slice(),
            ),
            ("Cargo.lock", include_bytes!("../../Cargo.lock").as_slice()),
        ]
        .into_iter()
        .map(|(path, bytes)| (path, crate::export::sha256_hex(bytes)))
        .collect();
        let binary_sha256 = hash(&std::env::current_exe()?)?;
        let receipt = json!({"scope": "planned-fullmix-biography-exposure-v1", "planned_updates": batches.len(), "optimizer_updates_executed": 0, "model_initialized": false, "scope_limits": "planned native objective coefficients; not actual completed optimizer exposure, losses, gradients, fitting or accuracy", "lr_weighting": "scheduled scalar learning rate only; not effective Adam update magnitude", "coefficient_formula": "class_weight * sum(1/native_batch_denominator) over every duplicate planned presentation", "lr_coefficient_formula": "class_weight * sum(scheduled_lr/native_batch_denominator)", "batch_sequence_sha256": batch_sequence_sha256, "input_sha256": self.pins, "compiled_implementation_sha256": implementation_sha256, "binary_sha256": binary_sha256, "native_replay_matches_pinned_reference": true, "aggregate_exposure": replay, "documents": documents, "batches": records});
        let bytes = serde_json::to_vec_pretty(&receipt)?;
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&self.out)?;
        file.write_all(&bytes)?;
        file.write_all(b"\n")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(labels: Vec<u8>) -> Encoded {
        let count = labels.len();
        Encoded {
            token_spans: (0..count)
                .map(|index| (index as u32, index as u32 + 1))
                .collect(),
            ngram_ids: vec![vec![1]; count],
            script: vec![0; count],
            shape: vec![0; count],
            flags: vec![0; count],
            labels,
            country: "US".into(),
        }
    }

    #[test]
    fn duplicated_target_uses_the_combined_native_batch_denominator() {
        let weights = [1.0, 3.0, 2.0, 3.0, 2.0, 2.0, 1.5];
        let items = [item(vec![1, 2]), item(vec![0; 6])];
        let mass = denominator(&items, &[0, 0, 1], &weights).unwrap();
        assert_eq!(mass, 16.0);
        let mut total = Accumulated::default();
        total.add(2, mass, 0.1).unwrap();
        assert_eq!(total.presentations, 2);
        assert_eq!(3.0 * total.inverse_denominator_sum, 0.375);
        assert!((3.0 * total.lr_inverse_denominator_sum - 0.0375).abs() < 1e-15);
        let aggregate =
            crate::training_audit::audit_epoch(&items, &[vec![0, 0, 1]], 0, &weights).unwrap();
        assert_eq!(aggregate.total.mean_batch_label_weight_share[1], 0.375);
        total.add(0, 1.0, 0.1).unwrap();
        assert_eq!(total.presentations, 2);
    }

    #[test]
    fn half_integer_denominators_are_native_exact_and_padding_is_absent() {
        let weights = [1.0, 3.0, 2.0, 3.0, 2.0, 2.0, 1.5];
        assert_eq!(
            denominator(&[item(vec![6]), item(vec![])], &[0, 1], &weights).unwrap(),
            1.5
        );
        assert_eq!(denominator(&[item(vec![])], &[0], &weights).unwrap(), 1.0);
        let mut unsupported = weights;
        unsupported[1] = 1.1;
        assert!(denominator(&[item(vec![1])], &[0], &unsupported).is_err());
        assert!(denominator(&[item(vec![9])], &[0], &weights).is_err());
        assert!(denominator(&[item(vec![0; 901])], &[0], &weights).is_err());
    }

    #[test]
    fn changed_finalized_batch_membership_changes_replay() {
        let weights = [1.0, 3.0, 2.0, 3.0, 2.0, 2.0, 1.5];
        let items = [item(vec![1]), item(vec![0])];
        let first = serde_json::to_value(
            crate::training_audit::audit_epoch(&items, &[vec![0, 0], vec![1]], 1, &weights)
                .unwrap(),
        )
        .unwrap();
        let changed = serde_json::to_value(
            crate::training_audit::audit_epoch(&items, &[vec![0, 1], vec![0]], 1, &weights)
                .unwrap(),
        )
        .unwrap();
        assert_ne!(first, changed);
        let existing =
            json!({"optimizer_steps":4000,"exposure":first,"max_real_piece_presentations":2});
        let reference = json!({"learning_check_exposure":existing});
        verify_replay(&existing, &reference, &first).unwrap();
        assert!(verify_replay(&existing, &reference, &changed).is_err());
        let mut wrong_counts = existing.clone();
        wrong_counts["max_real_piece_presentations"] = json!(3);
        assert!(verify_replay(&wrong_counts, &reference, &first).is_err());
        assert_ne!(
            crate::export::sha256_hex(&serde_json::to_vec(&vec![vec![0, 0], vec![1]]).unwrap()),
            crate::export::sha256_hex(&serde_json::to_vec(&vec![vec![0, 1], vec![0]]).unwrap())
        );
    }
    #[test]
    fn parsed_reference_scalars_use_the_same_serialized_representation() {
        for (bits, token) in [
            (0x3fa7556f476066e2_u64, "0.045573689902267636"),
            (0x3f9425981b8debd9_u64, "0.019674660379205598"),
        ] {
            let fresh = json!(f64::from_bits(bits));
            let parsed: Value = serde_json::from_str(token).unwrap();
            assert_ne!(fresh, parsed);
            assert_eq!(
                fresh
                    .as_f64()
                    .unwrap()
                    .to_bits()
                    .abs_diff(parsed.as_f64().unwrap().to_bits()),
                1
            );
            assert_eq!(serde_json::to_string(&fresh).unwrap(), token);
            assert_eq!(reference_representation(&fresh).unwrap(), parsed);
        }
    }

    #[test]
    fn parsed_reference_normalization_does_not_relax_fresh_replay_or_counts() {
        let fresh = json!({"total":{"draws":127874,"mean_batch_label_weight_share":[0.0, f64::from_bits(0x3fa7556f476066e2)]},"real":{"draws":41699}});
        let existing =
            json!({"optimizer_steps":4000,"exposure":fresh,"max_real_piece_presentations":240});
        let reference: Value = serde_json::from_slice(
            &serde_json::to_vec(&json!({"learning_check_exposure":existing})).unwrap(),
        )
        .unwrap();
        verify_replay(&existing, &reference, &fresh).unwrap();
        let mut changed_native = fresh.clone();
        changed_native["total"]["mean_batch_label_weight_share"][1] =
            json!(f64::from_bits(0x3fa7556f476066e1));
        assert!(verify_replay(&existing, &reference, &changed_native).is_err());
        let mut changed_counts = reference.clone();
        changed_counts["learning_check_exposure"]["exposure"]["real"]["draws"] = json!(41700);
        assert!(verify_replay(&existing, &changed_counts, &fresh).is_err());
        let mut changed_reference = reference.clone();
        changed_reference["learning_check_exposure"]["exposure"]["total"]["mean_batch_label_weight_share"]
            [1] = json!(0.046);
        assert!(verify_replay(&existing, &changed_reference, &fresh).is_err());
    }
}
