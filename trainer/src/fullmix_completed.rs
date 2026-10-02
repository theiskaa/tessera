//! Completed fullmix accounting against a reviewed immutable planned batch sequence.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, ensure};
use serde_json::{Value, json};

use super::targets::{Resolved, hash};
use super::{Accumulated, Plan, denominator, reference_representation};
use crate::config::Config;
use crate::dataset::Encoded;

/// Successful optimizer updates compared with a pinned planned schedule, without RNG draws.
pub(crate) struct Completed {
    targets: Vec<Resolved>,
    expected: Value,
    totals: Vec<Accumulated>,
    records: Vec<Value>,
    log: std::fs::File,
    run: PathBuf,
    pins: BTreeMap<String, String>,
}

impl Plan {
    pub(crate) fn completed(mut self, reference: &Path, run: &Path) -> anyhow::Result<Completed> {
        let bytes = std::fs::read(reference)?;
        let expected: Value = serde_json::from_slice(&bytes)?;
        ensure!(
            expected["scope"] == "planned-fullmix-biography-exposure-v1"
                && expected["planned_updates"] == 4000
                && expected["native_replay_matches_pinned_reference"] == true
                && expected["batches"]
                    .as_array()
                    .is_some_and(|batches| batches.len() == 4000),
            "completed exposure requires the reviewed fullmix prefix"
        );
        let documents = expected["documents"]
            .as_array()
            .context("planned target identities missing")?;
        ensure!(
            documents.len() == self.targets.len(),
            "planned target count differs"
        );
        for (target, document) in self.targets.iter().zip(documents) {
            ensure!(
                document["identity"] == serde_json::to_value(&target.identity)?
                    && document["original_encoded_index"] == target.original_index
                    && document["native_encoded_sha256"] == target.encoding_sha256
                    && document["native_label_token_counts"] == json!(target.label_token_counts),
                "planned target native identity differs"
            );
            let tokens = document["gold_tokens"]
                .as_array()
                .context("planned gold tokens missing")?;
            ensure!(
                tokens.len() == target.tokens.len(),
                "planned gold token count differs"
            );
            for (actual, planned) in target.tokens.iter().zip(tokens) {
                for (key, value) in actual.as_object().context("invalid resolved gold token")? {
                    ensure!(
                        planned.get(key) == Some(value),
                        "planned target token identity differs"
                    );
                }
            }
        }
        self.pins.insert(
            reference.to_string_lossy().into_owned(),
            crate::export::sha256_hex(&bytes),
        );
        let log = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(run.join("completed-batches.jsonl"))?;
        let totals = self
            .targets
            .iter()
            .map(|_| Accumulated::default())
            .collect();
        Ok(Completed {
            targets: self.targets,
            expected,
            totals,
            records: Vec::new(),
            log,
            run: run.to_path_buf(),
            pins: self.pins,
        })
    }
}

impl Completed {
    fn next(
        &self,
        cfg: &Config,
        items: &[Encoded],
        indices: &[usize],
        weights: &[f32],
        lr: f64,
    ) -> anyhow::Result<Value> {
        let step = self.records.len();
        ensure!(step < 4000, "completed exposure exceeds the fixed prefix");
        let horizon = cfg
            .train
            .diagnostic_schedule_steps
            .context("fullmix horizon missing")?;
        let native_lr = crate::train::lr_at(
            step,
            horizon,
            cfg.train.learning_rate,
            cfg.train.warmup_steps,
        );
        ensure!(
            native_lr.to_bits() == lr.to_bits(),
            "completed optimizer learning rate changed"
        );
        let mass = denominator(items, indices, weights)?;
        let multiplicities: Vec<_> = self
            .targets
            .iter()
            .map(|target| {
                indices
                    .iter()
                    .filter(|&&index| index == target.original_index)
                    .count()
            })
            .collect();
        let record = json!({"planned_update":step+1,"optimizer_step_index":step,"original_indices":indices,
            "native_weighted_token_denominator":mass,"scheduled_learning_rate":lr,"target_multiplicities":multiplicities});
        ensure!(
            reference_representation(&record)? == self.expected["batches"][step],
            "completed batch differs from the pinned planned update{}",
            step + 1
        );
        Ok(record)
    }

    pub(crate) fn verify_next(
        &self,
        cfg: &Config,
        items: &[Encoded],
        indices: &[usize],
        weights: &[f32],
        lr: f64,
    ) -> anyhow::Result<()> {
        self.next(cfg, items, indices, weights, lr)?;
        Ok(())
    }

    pub(crate) fn record(
        &mut self,
        cfg: &Config,
        items: &[Encoded],
        indices: &[usize],
        weights: &[f32],
        lr: f64,
    ) -> anyhow::Result<()> {
        let record = self.next(cfg, items, indices, weights, lr)?;
        let mass = denominator(items, indices, weights)?;
        for (target, total) in self.targets.iter().zip(&mut self.totals) {
            total.add(
                indices
                    .iter()
                    .filter(|&&index| index == target.original_index)
                    .count(),
                mass,
                lr,
            )?;
        }
        writeln!(self.log, "{}", serde_json::to_string(&record)?)?;
        self.log.flush()?;
        self.records.push(record);
        self.write(weights)
    }

    fn documents(&self, weights: &[f32]) -> anyhow::Result<Vec<Value>> {
        self.targets.iter().zip(&self.totals).map(|(target,total)| {
            let tokens: Vec<_> = target.tokens.iter().cloned().map(|mut token| -> anyhow::Result<Value> {
                let label = token["bio_label"].as_u64().context("target label missing")? as usize;
                let weight = f64::from(*weights.get(label).context("target label out of range")?);
                token["native_class_weight"] = json!(weight);
                token["presentations"] = json!(total.presentations);
                token["cumulative_normalized_coefficient"] = json!(weight * total.inverse_denominator_sum);
                token["lr_cumulative_normalized_coefficient"] = json!(weight * total.lr_inverse_denominator_sum);
                Ok(token)
            }).collect::<anyhow::Result<_>>()?;
            Ok(json!({"identity":target.identity,"original_encoded_index":target.original_index,
                "native_encoded_sha256":target.encoding_sha256,"native_label_token_counts":target.label_token_counts,
                "presentations":total.presentations,"inverse_native_denominator_sum":total.inverse_denominator_sum,
                "lr_inverse_native_denominator_sum":total.lr_inverse_denominator_sum,"gold_tokens":tokens}))
        }).collect::<anyhow::Result<_>>()
    }

    pub(crate) fn write(&self, weights: &[f32]) -> anyhow::Result<()> {
        if self.records.is_empty() || self.records.len() == 4000 {
            for (path, expected) in &self.pins {
                ensure!(
                    hash(Path::new(path))? == *expected,
                    "completed exposure input changed: {path}"
                );
            }
        }
        let documents = self.documents(weights)?;
        let sequence = crate::export::sha256_hex(&serde_json::to_vec(&self.records)?);
        let complete = self.records.len() == 4000;
        if complete {
            ensure!(
                self.expected["batch_sequence_sha256"] == sequence,
                "completed native batch bytes differ from the reviewed plan"
            );
            ensure!(
                reference_representation(&json!(documents))? == self.expected["documents"],
                "completed target coefficients differ from the reviewed plan"
            );
        }
        let receipt = json!({"scope":"completed-fullmix-biography-exposure-v1","optimizer_updates_executed":self.records.len(),
            "planned_updates":4000,"complete":complete,"completed_batch_sequence_sha256":sequence,
            "planned_batch_sequence_sha256":self.expected["batch_sequence_sha256"],"input_sha256":self.pins,
            "documents":documents,"scope_limits":"completed native objective coefficients; not gradients, effective Adam updates, accuracy, or sufficient training budget"});
        std::fs::write(
            self.run.join("completed-exposure.json"),
            serde_json::to_vec_pretty(&receipt)?,
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::super::targets::Target;
    use super::*;

    fn config() -> Config {
        let mut cfg = crate::config::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/detector-shared-v6.toml"),
        )
        .unwrap();
        cfg.train.diagnostic_schedule_steps = Some(7000);
        cfg.train.learning_rate = 0.0001;
        cfg.train.warmup_steps = 500;
        cfg
    }

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

    fn fixture(run: &Path) -> (Completed, Config, Vec<Encoded>, Vec<f32>) {
        let cfg = config();
        let items = vec![item(vec![1, 2]), item(vec![0; 6])];
        let weights = vec![1.0, 3.0, 2.0, 3.0, 2.0, 2.0, 1.5];
        let target = Resolved {
            identity: Target {
                name: "source-row".into(),
                source_path: "unused".into(),
                source_line: 1,
                source_sha256: "a".repeat(64),
                row_sha256: "b".repeat(64),
                text_sha256: "c".repeat(64),
                model_gold_sha256: "d".repeat(64),
                content_tokens: 2,
            },
            original_index: 0,
            tokens: vec![
                json!({"bio_label":1,"token_index":0}),
                json!({"bio_label":2,"token_index":1}),
            ],
            encoding_sha256: "e".repeat(64),
            label_token_counts: [0, 1, 1, 0, 0, 0, 0],
        };
        let records: Vec<_> = (0..4000).map(|step| json!({"planned_update":step+1,"optimizer_step_index":step,"original_indices":[0,0,1],"native_weighted_token_denominator":16.0,"scheduled_learning_rate":crate::train::lr_at(step,7000,cfg.train.learning_rate,cfg.train.warmup_steps),"target_multiplicities":[2]})).collect();
        let expected = reference_representation(&json!({"batches":records})).unwrap();
        let log = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(run.join("completed-batches.jsonl"))
            .unwrap();
        (
            Completed {
                targets: vec![target],
                expected,
                totals: vec![Accumulated::default()],
                records: Vec::new(),
                log,
                run: run.into(),
                pins: BTreeMap::new(),
            },
            cfg,
            items,
            weights,
        )
    }

    #[test]
    fn verification_is_not_a_completed_optimizer_update_and_partials_are_exact() {
        let run = tempfile::tempdir().unwrap();
        let (mut actual, cfg, items, weights) = fixture(run.path());
        let lr = crate::train::lr_at(0, 7000, cfg.train.learning_rate, cfg.train.warmup_steps);
        actual
            .verify_next(&cfg, &items, &[0, 0, 1], &weights, lr)
            .unwrap();
        assert!(actual.records.is_empty());
        assert_eq!(actual.totals[0].presentations, 0);
        actual.write(&weights).unwrap();
        let zero: Value = serde_json::from_slice(
            &std::fs::read(run.path().join("completed-exposure.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(zero["optimizer_updates_executed"], 0);
        actual
            .record(&cfg, &items, &[0, 0, 1], &weights, lr)
            .unwrap();
        let partial: Value = serde_json::from_slice(
            &std::fs::read(run.path().join("completed-exposure.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(partial["optimizer_updates_executed"], 1);
        assert_eq!(partial["complete"], false);
        assert_eq!(partial["documents"][0]["presentations"], 2);
        assert_eq!(
            partial["documents"][0]["gold_tokens"][0]["cumulative_normalized_coefficient"],
            0.375
        );
        assert_eq!(
            std::fs::read_to_string(run.path().join("completed-batches.jsonl"))
                .unwrap()
                .lines()
                .count(),
            1
        );
    }

    #[test]
    fn learning_rate_membership_order_and_denominator_changes_are_refused() {
        let run = tempfile::tempdir().unwrap();
        let (mut actual, cfg, mut items, weights) = fixture(run.path());
        let lr = crate::train::lr_at(0, 7000, cfg.train.learning_rate, cfg.train.warmup_steps);
        assert!(
            actual
                .verify_next(
                    &cfg,
                    &items,
                    &[0, 0, 1],
                    &weights,
                    f64::from_bits(lr.to_bits() + 1)
                )
                .is_err()
        );
        assert!(
            actual
                .verify_next(&cfg, &items, &[0, 1, 0], &weights, lr)
                .is_err()
        );
        assert!(
            actual
                .verify_next(&cfg, &items, &[0, 1, 1], &weights, lr)
                .is_err()
        );
        items[1] = item(vec![0; 7]);
        assert!(
            actual
                .verify_next(&cfg, &items, &[0, 0, 1], &weights, lr)
                .is_err()
        );
        items[1] = item(vec![0; 6]);
        actual.expected["batches"][0]["scheduled_learning_rate"] = json!(lr * 2.0);
        assert!(
            actual
                .verify_next(&cfg, &items, &[0, 0, 1], &weights, lr)
                .is_err()
        );
        assert!(actual.records.is_empty());
    }

    #[test]
    fn completed_sequence_hash_and_final_gold_coefficients_cannot_be_substituted() {
        let run = tempfile::tempdir().unwrap();
        let (mut actual, cfg, items, weights) = fixture(run.path());
        for step in 0..4000 {
            let lr =
                crate::train::lr_at(step, 7000, cfg.train.learning_rate, cfg.train.warmup_steps);
            let record = actual.next(&cfg, &items, &[0, 0, 1], &weights, lr).unwrap();
            actual.totals[0].add(2, 16.0, lr).unwrap();
            actual.records.push(record);
        }
        actual.expected["batch_sequence_sha256"] = json!(crate::export::sha256_hex(
            &serde_json::to_vec(&actual.records).unwrap()
        ));
        actual.expected["documents"] =
            reference_representation(&json!(actual.documents(&weights).unwrap())).unwrap();
        actual.write(&weights).unwrap();
        let complete: Value = serde_json::from_slice(
            &std::fs::read(run.path().join("completed-exposure.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(complete["complete"], true);
        assert_eq!(complete["optimizer_updates_executed"], 4000);
        assert_eq!(complete["documents"][0]["presentations"], 8000);
        actual.expected["documents"][0]["gold_tokens"][0]["cumulative_normalized_coefficient"] =
            json!(1499.0);
        assert!(actual.write(&weights).is_err());
        actual.expected["documents"] = complete["documents"].clone();
        actual.expected["batch_sequence_sha256"] = json!("0".repeat(64));
        assert!(actual.write(&weights).is_err());
    }
}
