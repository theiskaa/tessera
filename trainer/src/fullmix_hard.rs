//! Guarded hard-token objective and actual native update accounting.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::config::Config;
use crate::detector::DetectorDoc;
use crate::fullmix_rms::Receipt;
use crate::hard_token_batch::{Accounting, NativeRow, PreparedBatch, Rows};

pub(crate) const COMPLETED_SCOPE: &str = "completed-native-hard-typed-fullmix-exposure-v1";
const GROUPS: [&str; 3] = ["person", "org_contrast", "address"];

/// Canonical mathematical objective identity, independent of dataset and run paths.
pub(crate) fn objective_identity() -> anyhow::Result<String> {
    Ok(crate::export::sha256_hex(&serde_json::to_vec(&json!({
        "schema":"native_hard_objective_v1", "base":"native weighted masked CE, denominator clamped at one",
        "auxiliary":"weighted selected CE divided by each nonempty group's actual weighted mass",
        "groups":GROUPS,"coefficients":[0.0625,0.0625,0.0625],"class_weights":[1.,3.,2.,3.,2.,2.,1.5],
        "context":"unchanged complete native rows","empty_groups":"exact native base branch",
        "overlap":"disjoint binary masks, padding zero"
    }))?))
}

/// Exact reviewed bindings required before the new lane can initialize a model.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Contract {
    pub(crate) schema: String,
    pub(crate) objective_identity_sha256: String,
    pub(crate) selector_draft: Receipt,
    pub(crate) native_selector_proof: Receipt,
    pub(crate) hard_dose_preview: Receipt,
    pub(crate) corrected_sources: Receipt,
    pub(crate) corrected_gold: Receipt,
    pub(crate) source_reviews: Vec<Receipt>,
    pub(crate) measured_batch_source: Receipt,
    pub(crate) native_dose_reviews: Vec<Receipt>,
    pub(crate) feature_sha256: String,
    pub(crate) class_weights_sha256: String,
    pub(crate) groups: Vec<String>,
    pub(crate) coefficients: [f32; 3],
}

impl Contract {
    /// Refuse alternate formulas, groups or coefficients.
    pub(crate) fn validate_identity(&self) -> anyhow::Result<()> {
        ensure!(
            self.schema == "native_hard_objective_v1"
                && self.objective_identity_sha256 == objective_identity()?
                && self.groups == GROUPS
                && self.coefficients == [0.0625; 3],
            "hard objective identity differs"
        );
        Ok(())
    }

    /// Include all objective evidence in the recipe input closure.
    pub(crate) fn input_paths(&self) -> anyhow::Result<Vec<PathBuf>> {
        self.validate_identity()?;
        let mut paths = vec![
            self.selector_draft.path.clone(),
            self.native_selector_proof.path.clone(),
            self.hard_dose_preview.path.clone(),
            self.corrected_sources.path.clone(),
            self.corrected_gold.path.clone(),
            self.measured_batch_source.path.clone(),
        ];
        paths.extend(
            self.source_reviews
                .iter()
                .chain(&self.native_dose_reviews)
                .map(|p| p.path.clone()),
        );
        Ok(paths)
    }

    /// Validate reviewed source, selector, zero-model dose, and effective configuration bindings.
    pub(crate) fn validate(
        &self,
        cfg: &Config,
        config: &Receipt,
        base: &Receipt,
        _targets: &Receipt,
        exposure: &Value,
    ) -> anyhow::Result<()> {
        self.validate_identity()?;
        ensure!(
            self.feature_sha256 == crate::export::sha256_hex(&serde_json::to_vec(&cfg.features)?)
                && self.class_weights_sha256
                    == crate::export::sha256_hex(&serde_json::to_vec(
                        &cfg.detector
                            .as_ref()
                            .context("hard detector absent")?
                            .class_weights
                    )?),
            "hard feature or class-weight binding differs"
        );
        let settings = cfg
            .detector
            .as_ref()
            .and_then(|d| d.typed_synthetic.as_ref())
            .context("hard typed settings absent")?;
        let corrected = settings
            .corrected_real_sources
            .as_ref()
            .context("hard corrected source receipt absent")?;
        ensure!(
            serde_json::to_value(corrected)? == serde_json::to_value(&self.corrected_sources)?,
            "hard source successor differs"
        );
        self.corrected_sources.bytes()?;
        self.corrected_gold.bytes()?;
        let proof: Value = serde_json::from_slice(&self.native_selector_proof.bytes()?)?;
        let rows =
            crate::hard_token_selection::resolve_preview(&self.selector_draft.path, &proof, cfg)?;
        ensure!(rows.len() == 541, "hard selector requires exact541 parents");
        let dose: Value = serde_json::from_slice(&self.hard_dose_preview.bytes()?)?;
        ensure!(
            dose["passed"] == true
                && dose["model_initialized"] == false
                && dose["model_forwards"] == 0
                && dose["optimizer_updates"] == 0
                && dose["planned_updates"] == 4000
                && dose["native_pool_documents"] == 172368
                && dose["selected_full_parent_rows"] == 541
                && dose["compiled_batch_sha256"] == self.measured_batch_source.sha256
                && dose["base_batch_sequence_sha256"]
                    == exposure["typed_synthetic"]["batch_sequence_sha256"],
            "hard dose native binding differs"
        );
        self.measured_batch_source.bytes()?;
        for pin in [config, &self.selector_draft, &self.native_selector_proof] {
            ensure!(
                dose["inputs"].as_array().is_some_and(|pins| pins
                    .iter()
                    .any(|p| p == &serde_json::to_value(pin).unwrap_or(Value::Null))),
                "hard dose input binding absent"
            );
        }
        for summary in dose["summaries"]
            .as_array()
            .context("hard summaries absent")?
        {
            for kind in ["positive", "O"] {
                ensure!(
                    summary[kind]["zero_base_dose_tokens"] == 0
                        && summary[kind]["zero_auxiliary_dose_tokens"] == 0,
                    "hard selected token has zero planned dose"
                );
            }
        }
        for (review_kind, reviews) in [&self.source_reviews, &self.native_dose_reviews]
            .into_iter()
            .enumerate()
        {
            ensure!(
                reviews.len() == 2
                    && reviews[0].sha256 != reviews[1].sha256
                    && std::fs::canonicalize(&reviews[0].path)?
                        != std::fs::canonicalize(&reviews[1].path)?,
                "hard needs two independent reviews"
            );
            for pin in reviews {
                let review: Value = serde_json::from_slice(&pin.bytes()?)?;
                ensure!(
                    review["passed"] == true
                        && review["blockers"].as_array().is_some_and(Vec::is_empty),
                    "hard review failed"
                );
                if review_kind == 0 {
                    for (file, bytes) in [
                        (
                            "hard_token_batch.rs",
                            include_bytes!("hard_token_batch.rs").as_slice(),
                        ),
                        (
                            "hard_token_selection.rs",
                            include_bytes!("hard_token_selection.rs").as_slice(),
                        ),
                        (
                            "hard_token_masks.rs",
                            include_bytes!("hard_token_masks.rs").as_slice(),
                        ),
                        (
                            "fullmix_hard.rs",
                            include_bytes!("fullmix_hard.rs").as_slice(),
                        ),
                        ("train.rs", include_bytes!("train.rs").as_slice()),
                        ("main.rs", include_bytes!("main.rs").as_slice()),
                        (
                            "diagnostic_operator.rs",
                            include_bytes!("diagnostic_operator.rs").as_slice(),
                        ),
                        (
                            "fullmix_typed_completed.rs",
                            include_bytes!("fullmix_typed_completed.rs").as_slice(),
                        ),
                        (
                            "fullmix_rms.rs",
                            include_bytes!("fullmix_rms.rs").as_slice(),
                        ),
                        (
                            "fullmix_rms_binding.rs",
                            include_bytes!("fullmix_rms_binding.rs").as_slice(),
                        ),
                        (
                            "fullmix_typed.rs",
                            include_bytes!("fullmix_typed.rs").as_slice(),
                        ),
                        (
                            "fullmix_checkpoint.rs",
                            include_bytes!("fullmix_checkpoint.rs").as_slice(),
                        ),
                        (
                            "fullmix_training_seen/mod.rs",
                            include_bytes!("fullmix_training_seen/mod.rs").as_slice(),
                        ),
                        (
                            "fullmix_training_seen/hard_bindings.rs",
                            include_bytes!("fullmix_training_seen/hard_bindings.rs").as_slice(),
                        ),
                    ] {
                        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("src").join(file);
                        let key = path.to_string_lossy();
                        let sha = crate::export::sha256_hex(bytes);
                        ensure!(
                            review["reviewed_files"][key.as_ref()] == sha
                                || review["reviewed_files"].as_array().is_some_and(|pins| pins
                                    .iter()
                                    .any(|p| p["path"] == key.as_ref() && p["sha256"] == sha)),
                            "hard runtime source review binding absent"
                        );
                    }
                    let key = self.measured_batch_source.path.to_string_lossy();
                    ensure!(
                        review["reviewed_files"][key.as_ref()] == self.measured_batch_source.sha256
                            || review["reviewed_files"]
                                .as_array()
                                .is_some_and(|pins| pins.iter().any(|p| p["path"] == key.as_ref()
                                    && p["sha256"] == self.measured_batch_source.sha256)),
                        "hard measured predecessor source review absent"
                    );
                }
                for required in [
                    config,
                    base,
                    &self.selector_draft,
                    &self.native_selector_proof,
                    &self.hard_dose_preview,
                ] {
                    let path = required.path.to_string_lossy();
                    ensure!(
                        review["reviewed_files"][path.as_ref()] == required.sha256
                            || review["reviewed_files"].as_array().is_some_and(|pins| pins
                                .iter()
                                .any(|p| p["path"] == path.as_ref()
                                    && p["sha256"] == required.sha256)),
                        "hard review required binding absent"
                    );
                }
            }
        }
        Ok(())
    }
}

fn index(
    row: &crate::hard_token_selection::PreviewRow,
    boundary: usize,
    pieces: &[usize],
    metadata: &[Value],
) -> anyhow::Result<usize> {
    if row.role == "authored" {
        let matches: Vec<_> = metadata
            .iter()
            .enumerate()
            .filter(|(_, m)| {
                m["id"]
                    .as_str()
                    .is_some_and(|id| row.name == format!("seen-authored-{id}"))
            })
            .collect();
        ensure!(matches.len() == 1, "hard authored membership collision");
        let (offset, identity) = matches[0];
        let index = 170000 + offset;
        ensure!(
            identity["origin"] == "synthetic_authored"
                && identity["split"] == "train"
                && identity["native_index"] == index
                && identity["text_sha256"] == crate::export::sha256_hex(row.doc.text.as_bytes()),
            "hard authored identity differs"
        );
        Ok(index)
    } else {
        ensure!(row.role == "real", "hard row role invalid");
        let parts: Vec<_> = row.name.split('-').collect();
        ensure!(
            parts.len() == 4 && parts[0] == "seen" && parts[1] == "real",
            "hard real name invalid"
        );
        let source: usize = parts[2].parse()?;
        let offset: usize = parts[3].parse()?;
        ensure!(
            source < pieces.len() && offset < pieces[source],
            "hard actual source parent absent"
        );
        Ok(boundary + pieces[..source].iter().sum::<usize>() + offset)
    }
}

/// The actual objective uses the same immutable host batch as pre-step verification and post-step accounting.
pub(crate) struct Completed {
    rows: Rows,
    accounting: Accounting,
    records: Vec<Value>,
    selected: Vec<Value>,
    expected: Value,
    bindings: Value,
    run: PathBuf,
    log: std::fs::File,
    simulation: bool,
}

impl Completed {
    /// Bind actual complete parents and calculate the exact reviewed prefix before initialization.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        contract: &Contract,
        cfg: &Config,
        run: &Path,
        synthetic: &[DetectorDoc],
        real: &[DetectorDoc],
        pieces: &[usize],
        boundary: usize,
        metadata: &[Value],
        config: &Receipt,
        base: &Receipt,
    ) -> anyhow::Result<Self> {
        ensure!(
            synthetic.len() == 170569
                && boundary == synthetic.len()
                && metadata.len() == 569
                && real.len() == 1799
                && pieces.iter().sum::<usize>() == real.len(),
            "hard native pool differs"
        );
        let proof = serde_json::from_slice(&contract.native_selector_proof.bytes()?)?;
        let resolved = crate::hard_token_selection::resolve_preview(
            &contract.selector_draft.path,
            &proof,
            cfg,
        )?;
        let expected: Value = serde_json::from_slice(&contract.hard_dose_preview.bytes()?)?;
        let mut masks = BTreeMap::new();
        let mut selected = Vec::new();
        for row in &resolved {
            let i = index(row, boundary, pieces, metadata)?;
            let actual = if i < boundary {
                synthetic.get(i)
            } else {
                real.get(i - boundary)
            }
            .context("hard actual parent absent")?;
            ensure!(
                actual.text == row.doc.text
                    && actual.enc == row.doc.enc
                    && actual.breaks == row.doc.breaks
                    && actual.gold == row.doc.gold
                    && masks.insert(i, row.masks.clone()).is_none(),
                "hard actual full parent differs"
            );
            let planned = expected["rows"]
                .as_array()
                .context("hard dose rows absent")?
                .iter()
                .find(|p| p["name"] == row.name)
                .context("hard planned parent absent")?;
            ensure!(
                planned["actual_native_index"] == i
                    && planned["native_encoding_sha256"] == row.proof["native_encoding_sha256"]
                    && planned["binary_masks_sha256"] == row.proof["binary_masks_sha256"],
                "hard actual selector identity differs"
            );
            for (group, group_name) in GROUPS.iter().enumerate() {
                for (token, &mask) in row.masks[group].iter().enumerate() {
                    if mask == 1. {
                        selected.push(json!({"name":row.name,"native_index":i,"token_index":token,"group":group_name,
                    "native_label":actual.enc.labels[token],"span":actual.enc.token_spans[token]}));
                    }
                }
            }
        }
        let rows = Rows::new(
            synthetic
                .iter()
                .chain(real)
                .enumerate()
                .map(|(i, d)| NativeRow::new(i, d.enc.labels.clone(), masks.remove(&i)))
                .collect::<Result<Vec<_>, _>>()?,
        )?;
        let weights = &cfg
            .detector
            .as_ref()
            .context("hard detector absent")?
            .class_weights;
        let horizon = cfg
            .train
            .diagnostic_schedule_steps
            .context("hard LR horizon absent")?;
        let mut records = Vec::new();
        for (step, record) in expected["batches"]
            .as_array()
            .context("hard batches absent")?
            .iter()
            .enumerate()
        {
            let chunk: Vec<usize> =
                serde_json::from_value(record["ordered_native_indices"].clone())?;
            let lr = crate::train::lr_at(
                step,
                horizon,
                cfg.train.learning_rate,
                cfg.train.warmup_steps,
            );
            let prepared = PreparedBatch::collate(
                &rows,
                &chunk,
                weights,
                Some(contract.coefficients),
                step,
                lr,
            )?;
            ensure!(
                crate::fullmix_exposure::reference_representation(&serde_json::to_value(
                    prepared.record()
                )?)? == *record,
                "hard freshly prepared plan differs"
            );
            records.push(prepared.record().clone());
        }
        ensure!(
            records.len() == 4000,
            "hard prefix must contain4000 updates"
        );
        let bindings = json!({"objective_identity_sha256":contract.objective_identity_sha256,
            "config":config,"selector_draft":contract.selector_draft,"native_selector_proof":contract.native_selector_proof,
            "hard_dose_preview":contract.hard_dose_preview,"corrected_sources":contract.corrected_sources,
            "corrected_gold":contract.corrected_gold,"base_native_preview":base});
        let log = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(run.join("completed-batches.jsonl"))?;
        let selected_set = selected
            .iter()
            .map(|t| {
                Ok((
                    t["native_index"].as_u64().context("index absent")? as usize,
                    t["token_index"].as_u64().context("token absent")? as usize,
                ))
            })
            .collect::<anyhow::Result<_>>()?;
        Ok(Self {
            rows,
            accounting: Accounting::selected_only(records, selected_set)?,
            records: Vec::new(),
            selected,
            expected,
            bindings,
            run: run.to_path_buf(),
            log,
            simulation: false,
        })
    }

    /// Prepare and verify one unchanged native chunk before backward.
    pub(crate) fn prepare(
        &self,
        indices: &[usize],
        weights: &[f32],
        lr: f64,
    ) -> anyhow::Result<PreparedBatch> {
        let batch = PreparedBatch::collate(
            &self.rows,
            indices,
            weights,
            Some([0.0625; 3]),
            self.records.len(),
            lr,
        )?;
        self.accounting.verify_next(&batch)?;
        Ok(batch)
    }

    /// Commit only the same prepared batch after the optimizer has returned successfully.
    pub(crate) fn record(&mut self, batch: &PreparedBatch) -> anyhow::Result<()> {
        self.accounting.record(batch)?;
        let record = serde_json::to_value(batch.record())?;
        writeln!(self.log, "{}", serde_json::to_string(&record)?)?;
        self.log.flush()?;
        self.records.push(record);
        Ok(())
    }

    /// Persist completed objective evidence; only the exact full prefix is complete.
    pub(crate) fn write(&self) -> anyhow::Result<()> {
        let totals: BTreeMap<_, _> = self
            .accounting
            .cumulative_coefficients()
            .map(|(i, t, c)| ((i, t), c))
            .collect();
        let tokens: Vec<_> = self
            .selected
            .iter()
            .map(|token| -> anyhow::Result<Value> {
                let mut token = token.clone();
                let key = (
                    token["native_index"].as_u64().context("index absent")? as usize,
                    token["token_index"].as_u64().context("token absent")? as usize,
                );
                token["dose"] =
                    serde_json::to_value(totals.get(&key).copied().cloned().unwrap_or_default())?;
                Ok(token)
            })
            .collect::<Result<_, _>>()?;
        let complete = !self.simulation && self.records.len() == 4000;
        if complete {
            for token in &tokens {
                let planned = self.expected["rows"]
                    .as_array()
                    .context("rows absent")?
                    .iter()
                    .find(|p| p["name"] == token["name"])
                    .context("planned row absent")?["selected_tokens"]
                    .as_array()
                    .context("planned tokens absent")?
                    .iter()
                    .find(|p| p["index"] == token["token_index"] && p["group"] == token["group"])
                    .context("planned token absent")?;
                ensure!(
                    crate::fullmix_exposure::reference_representation(&token["dose"])?
                        == planned["dose"],
                    "completed selected dose differs"
                );
            }
        }
        let receipt = json!({"scope":if self.simulation { "SIMULATION hard objective accounting; no optimizer execution" } else { COMPLETED_SCOPE },"complete":complete,"optimizer_updates_executed":if self.simulation {0} else {self.records.len()},
            "simulation":self.simulation,"execution":if self.simulation {"simulation"} else {"native optimizer success"},"simulated_records":if self.simulation {self.records.len()} else {0},"planned_updates":4000,"bindings":self.bindings,"selected_full_parent_rows":self.expected["rows"].as_array().map_or(0,Vec::len),
            "selected_tokens":tokens,"presentations":self.accounting.presentations(),
            "completed_batch_sequence_sha256":crate::export::sha256_hex(&serde_json::to_vec(
                &crate::fullmix_exposure::reference_representation(&json!(self.records))?)?),
            "planned_batch_sequence_sha256":crate::export::sha256_hex(&serde_json::to_vec(&self.expected["batches"])?),
            "batches":self.records,"scope_limits":"Actual mathematical objective coefficients; not accuracy or optimizer displacement."});
        std::fs::write(
            self.run.join("completed-exposure.json"),
            serde_json::to_vec_pretty(&receipt)?,
        )?;
        Ok(())
    }
}

/// Require exact actual hard-objective completion for final checkpoint and evaluation readers.
pub(crate) fn validate_completed(completed: &Value, manifest: &Value) -> anyhow::Result<()> {
    let contract = &manifest["typed_diagnostic"]["hard_objective"];
    ensure!(
        manifest["scope"] == crate::fullmix_rms::HARD_SCOPE
            && completed["scope"] == COMPLETED_SCOPE
            && completed["complete"] == true
            && completed["optimizer_updates_executed"] == 4000
            && completed["planned_updates"] == 4000
            && completed["simulation"] == false
            && completed["execution"] == "native optimizer success"
            && completed["simulated_records"] == 0
            && completed["batches"]
                .as_array()
                .is_some_and(|v| v.len() == 4000)
            && completed["bindings"]["objective_identity_sha256"] == objective_identity()?
            && completed["bindings"]["objective_identity_sha256"]
                == contract["objective_identity_sha256"]
            && completed["bindings"]["config"] == manifest["config"]
            && completed["bindings"]["base_native_preview"] == manifest["exposure"],
        "hard completed objective is incomplete or foreign"
    );
    for field in [
        "selector_draft",
        "native_selector_proof",
        "hard_dose_preview",
        "corrected_sources",
        "corrected_gold",
    ] {
        ensure!(
            completed["bindings"][field] == contract[field],
            "hard completion evidence binding differs"
        );
    }
    ensure!(
        completed["completed_batch_sequence_sha256"]
            == crate::export::sha256_hex(&serde_json::to_vec(&completed["batches"])?),
        "hard completed batch sequence changed"
    );
    let pin: Receipt = serde_json::from_value(contract["hard_dose_preview"].clone())?;
    let expected: Value = serde_json::from_slice(&pin.bytes()?)?;
    ensure!(
        completed["batches"] == expected["batches"]
            && completed["planned_batch_sequence_sha256"]
                == crate::export::sha256_hex(&serde_json::to_vec(&expected["batches"])?),
        "hard completed ordered batches differ from pinned plan"
    );
    ensure!(
        completed["selected_full_parent_rows"]
            == expected["rows"]
                .as_array()
                .context("planned rows absent")?
                .len(),
        "hard completed selected row count differs"
    );
    let actual_tokens = completed["selected_tokens"]
        .as_array()
        .context("completed tokens absent")?;
    let mut expected_tokens = Vec::new();
    for row in expected["rows"].as_array().context("planned rows absent")? {
        let index = row["actual_native_index"]
            .as_u64()
            .context("planned index absent")?;
        ensure!(
            completed["presentations"][index.to_string()] == row["presentations"],
            "hard completed row presentation count differs"
        );
        for token in row["selected_tokens"]
            .as_array()
            .context("planned selected tokens absent")?
        {
            expected_tokens.push(json!({"name":row["name"],"native_index":index,"token_index":token["index"],
                "group":token["group"],"native_label":token["native_label"],"span":token["span"],"dose":token["dose"]}));
        }
    }
    let canonical = |tokens: &[Value]| -> anyhow::Result<BTreeMap<String, Value>> {
        let mut result = BTreeMap::new();
        for token in tokens {
            let key = format!(
                "{}:{}:{}",
                token["native_index"], token["token_index"], token["group"]
            );
            ensure!(
                result.insert(key, token.clone()).is_none(),
                "duplicate completed hard token"
            );
        }
        Ok(result)
    };
    ensure!(
        canonical(actual_tokens)? == canonical(&expected_tokens)?,
        "hard completed token identity or dose differs"
    );
    Ok(())
}

#[cfg(test)]
#[path = "fullmix_hard_tests.rs"]
mod tests;
