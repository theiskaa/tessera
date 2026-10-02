//! Completed typed diagnostic batches reuse the native zero-update objective calculator.

use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, ensure};
use serde_json::{Value, json};

use crate::config::Config;
use crate::dataset::Encoded;

/// Accounting for the existing diagnostic or its explicitly bound typed supplement.
pub(crate) enum Accounting {
    Legacy(crate::fullmix_exposure::Completed),
    Typed(TypedCompleted),
    Hard(crate::fullmix_hard::Completed),
}

/// Native update records for one reviewed typed finalized prefix.
pub(crate) struct TypedCompleted {
    expected: Value,
    metadata: Vec<Value>,
    records: Vec<Value>,
    indices: Vec<Vec<usize>>,
    log: std::fs::File,
    run: PathBuf,
    actual: Option<Value>,
    real_targets: Value,
    real_totals: Vec<(usize, f64, f64)>,
    exposure_sha256: String,
    real_targets_sha256: String,
    simulation: bool,
}

impl TypedCompleted {
    pub(super) fn new(
        run: &Path,
        expected: Value,
        metadata: &[Value],
        real_targets: Value,
        real_docs: &[crate::detector::DetectorDoc],
        exposure_sha256: &str,
        real_targets_sha256: &str,
    ) -> anyhow::Result<Self> {
        ensure!(
            expected["planned_updates"] == super::STEPS
                && expected["uniform_synthetic_pool"] == 170000
                && expected["full_synthetic_boundary"] == 170569
                && expected["real_documents"] == 1799
                && expected["batches"]
                    .as_array()
                    .is_some_and(|v| v.len() == super::STEPS)
                && metadata.len() == 569,
            "typed completed accounting needs the reviewed finalized4000 prefix"
        );
        let targets = expected["authored_documents"]
            .as_array()
            .context("typed objective targets absent")?;
        ensure!(
            targets.len() == metadata.len(),
            "typed objective target count differs"
        );
        for (target, actual) in targets.iter().zip(metadata) {
            ensure!(
                target["identity"] == *actual,
                "typed objective native identity differs"
            );
        }
        let real_rows = real_targets["documents"]
            .as_array()
            .context("typed real targets absent")?;
        ensure!(
            real_rows.len() == 14
                && real_targets["full_synthetic_boundary"] == 170569
                && real_targets["uniform_synthetic_pool"] == 170000
                && real_targets["model_execution"] == false
                && real_targets["training_ready"] == false,
            "typed real target scope differs"
        );
        for row in real_rows {
            let index = row["native_index"]
                .as_u64()
                .context("real native target index absent")? as usize;
            let doc = real_docs
                .get(
                    index
                        .checked_sub(170569)
                        .context("real target precedes real boundary")?,
                )
                .context("real target beyond real boundary")?;
            let mut counts = [0usize; 7];
            for &label in &doc.enc.labels {
                counts[label as usize] += 1;
            }
            ensure!(
                row["origin"] == "real_silver"
                    && row["family"] == "biography_real"
                    && row["text_sha256"] == crate::export::sha256_hex(doc.text.as_bytes())
                    && row["label_token_counts"] == json!(counts)
                    && row["native_content_tokens"] == doc.enc.token_spans.len(),
                "typed real target actual native membership differs"
            );
        }
        let log = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(run.join("completed-batches.jsonl"))?;
        Ok(Self {
            expected,
            metadata: metadata.to_vec(),
            records: Vec::new(),
            indices: Vec::new(),
            log,
            run: run.to_path_buf(),
            actual: None,
            real_totals: vec![(0, 0.0, 0.0); 14],
            real_targets,
            exposure_sha256: exposure_sha256.to_owned(),
            real_targets_sha256: real_targets_sha256.to_owned(),
            simulation: false,
        })
    }

    fn next(
        &self,
        cfg: &Config,
        items: &[Encoded],
        indices: &[usize],
        weights: &[f32],
        lr: f64,
    ) -> anyhow::Result<Value> {
        let step = self.records.len();
        ensure!(
            step < super::STEPS,
            "typed diagnostic exceeds fixed4000 prefix"
        );
        let horizon = cfg
            .train
            .diagnostic_schedule_steps
            .context("typed diagnostic LR horizon absent")?;
        ensure!(
            crate::train::lr_at(
                step,
                horizon,
                cfg.train.learning_rate,
                cfg.train.warmup_steps
            )
            .to_bits()
                == lr.to_bits(),
            "typed completed LR differs"
        );
        let mass = crate::fullmix_exposure::denominator(items, indices, weights)?;
        let record = json!({"planned_update":step+1,"optimizer_step_index":step,"indices":indices,"native_weighted_token_denominator":mass,"scheduled_learning_rate":lr});
        ensure!(
            crate::fullmix_exposure::reference_representation(&record)?
                == self.expected["batches"][step],
            "typed completed batch differs from reviewed plan at update{}",
            step + 1
        );
        Ok(record)
    }

    fn record(
        &mut self,
        cfg: &Config,
        items: &[Encoded],
        indices: &[usize],
        weights: &[f32],
        lr: f64,
    ) -> anyhow::Result<()> {
        let record = self.next(cfg, items, indices, weights, lr)?;
        writeln!(self.log, "{}", serde_json::to_string(&record)?)?;
        self.log.flush()?;
        self.records.push(record);
        self.indices.push(indices.to_vec());
        let mass = crate::fullmix_exposure::denominator(items, indices, weights)?;
        for (row, total) in self.real_targets["documents"]
            .as_array()
            .context("real target rows absent")?
            .iter()
            .zip(&mut self.real_totals)
        {
            let index = row["native_index"]
                .as_u64()
                .context("real native index absent")? as usize;
            for _ in indices.iter().filter(|&&i| i == index) {
                total.0 += 1;
                total.1 += 1.0 / mass;
                total.2 += lr / mass;
            }
        }
        if self.records.len() == super::STEPS {
            let horizon = cfg
                .train
                .diagnostic_schedule_steps
                .context("typed diagnostic horizon absent")?;
            let mut actual = crate::typed_synthetic::preview(
                cfg,
                items,
                &self.indices,
                170000,
                170569,
                &self.metadata,
                horizon,
            )?;
            for key in [
                "batches",
                "batch_sequence_sha256",
                "authored_documents",
                "aggregate_exposure",
            ] {
                ensure!(
                    crate::fullmix_exposure::reference_representation(&actual[key])?
                        == self.expected[key],
                    "completed typed native objective differs: {key}"
                );
            }
            let fields = actual
                .as_object_mut()
                .context("typed objective must be an object")?;
            for key in [
                "model_initialized",
                "model_forwards",
                "optimizer_updates_executed",
            ] {
                fields.remove(key);
            }
            actual["scope"] = json!(
                "native objective reconstruction from completed4000 updates; no additional model forwards"
            );
            let mut real_actual = Vec::new();
            for (row, total) in self.real_targets["documents"]
                .as_array()
                .context("real target rows absent")?
                .iter()
                .zip(&self.real_totals)
            {
                let counts: Vec<usize> = serde_json::from_value(row["label_token_counts"].clone())?;
                let computed = json!({"presentations":total.0,"inverse_native_denominator_sum":total.1,"lr_inverse_native_denominator_sum":total.2,"label_normalized_coefficients":counts.iter().zip(weights).map(|(n,w)| *n as f64 * *w as f64 * total.1).collect::<Vec<_>>(),"label_lr_coefficients":counts.iter().zip(weights).map(|(n,w)| *n as f64 * *w as f64 * total.2).collect::<Vec<_>>()});
                let normalized = crate::fullmix_exposure::reference_representation(&computed)?;
                for (key, value) in normalized
                    .as_object()
                    .context("real computed objective absent")?
                {
                    ensure!(
                        row[key] == *value,
                        "completed real target objective differs: {key}"
                    );
                }
                let mut target = computed;
                target["native_index"] = row["native_index"].clone();
                target["id"] = row["id"].clone();
                target["text_sha256"] = row["text_sha256"].clone();
                real_actual.push(target);
            }
            actual["real_target_documents"] = json!(real_actual);
            self.actual = Some(actual);
        }
        self.write()
    }

    fn write(&self) -> anyhow::Result<()> {
        let receipt = json!({"scope":if self.simulation {"SIMULATION typed objective accounting; no model or optimizer execution"} else {"completed-native-typed-fullmix-exposure-v1"},"optimizer_updates_executed":if self.simulation {0} else {self.records.len()},"simulated_records":if self.simulation {self.records.len()} else {0},"planned_exposure_sha256":self.exposure_sha256,"real_targets_sha256":self.real_targets_sha256,"planned_updates":super::STEPS,"complete":self.records.len()==super::STEPS,"completed_batch_sequence_sha256":crate::export::sha256_hex(&serde_json::to_vec(&self.records)?),"planned_batch_sequence_sha256":self.expected["batch_sequence_sha256"],"objective":self.actual,"scope_limits":"native objective coefficients; not gradients, Adam updates, accuracy or a sufficient training budget"});
        std::fs::write(
            self.run.join("completed-exposure.json"),
            serde_json::to_vec_pretty(&receipt)?,
        )?;
        Ok(())
    }
}

impl Accounting {
    /// Prepare the guarded objective only for the explicit hard lane.
    pub(crate) fn prepare_hard(
        &self,
        indices: &[usize],
        weights: &[f32],
        lr: f64,
    ) -> anyhow::Result<Option<crate::hard_token_batch::PreparedBatch>> {
        match self {
            Self::Hard(a) => Ok(Some(a.prepare(indices, weights, lr)?)),
            _ => Ok(None),
        }
    }

    /// Record the exact prepared objective after a successful optimizer step.
    pub(crate) fn record_hard(
        &mut self,
        batch: &crate::hard_token_batch::PreparedBatch,
    ) -> anyhow::Result<()> {
        match self {
            Self::Hard(a) => a.record(batch),
            _ => anyhow::bail!("hard batch outside hard lane"),
        }
    }

    /// Verifies the exact next native batch before the optimizer update.
    pub(crate) fn verify_next(
        &self,
        cfg: &Config,
        items: &[Encoded],
        indices: &[usize],
        weights: &[f32],
        lr: f64,
    ) -> anyhow::Result<()> {
        match self {
            Self::Hard(_) => {
                anyhow::bail!("hard objective must verify the same immutable prepared batch")
            }
            Self::Legacy(a) => a.verify_next(cfg, items, indices, weights, lr),
            Self::Typed(a) => {
                a.next(cfg, items, indices, weights, lr)?;
                Ok(())
            }
        }
    }

    /// Records a batch only after the native optimizer update succeeds.
    pub(crate) fn record(
        &mut self,
        cfg: &Config,
        items: &[Encoded],
        indices: &[usize],
        weights: &[f32],
        lr: f64,
    ) -> anyhow::Result<()> {
        match self {
            Self::Hard(_) => {
                anyhow::bail!("hard objective must record the same immutable prepared batch")
            }
            Self::Legacy(a) => a.record(cfg, items, indices, weights, lr),
            Self::Typed(a) => a.record(cfg, items, indices, weights, lr),
        }
    }

    /// Writes the current completed-update evidence.
    pub(crate) fn write(&self, weights: &[f32]) -> anyhow::Result<()> {
        match self {
            Self::Hard(a) => a.write(),
            Self::Legacy(a) => a.write(weights),
            Self::Typed(a) => a.write(),
        }
    }
}
