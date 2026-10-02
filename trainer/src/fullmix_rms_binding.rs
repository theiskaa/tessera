//! Run bindings, source-backed inspection, and checkpoint identities for the fixed fullmix scope.

use std::path::Path;

use anyhow::{Context, ensure};
use serde_json::{Value, json};

use super::{
    CONTEXT_SCOPE, FILE, HARD_SCOPE, Request, SCOPE, SNAPSHOTS, STEPS, hash,
    marker_scope_for_scope, recipe_sha256, valid_sha,
};
use crate::config::Config;

fn validate_inspection_contract(
    scope: &str,
    candidate: &Config,
    original: &Config,
) -> anyhow::Result<()> {
    ensure!(
        serde_json::to_value(&candidate.features)? == serde_json::to_value(&original.features)?,
        "inspection native feature contract differs"
    );
    let reference = serde_json::to_value(&original.net)?;
    let actual = serde_json::to_value(&candidate.net)?;
    let expected = match scope {
        SCOPE => reference,
        CONTEXT_SCOPE | HARD_SCOPE => {
            ensure!(
                original.net.architecture.is_none()
                    && original.net.dilations == [1, 2, 4, 8, 16, 1]
                    && candidate.context96_rms(),
                "inspection reference or candidate graph is not the exact named context96 graph"
            );
            let mut derived = reference;
            derived["architecture"] = json!("detector-context96-rms-v2");
            derived["dilations"] = json!([1, 2, 4, 8, 16, 1, 64]);
            derived
        }
        _ => anyhow::bail!("unknown inspection scope"),
    };
    ensure!(
        actual == expected,
        "inspection network differs beyond the explicit named graph transition"
    );
    Ok(())
}

impl Request {
    pub(crate) fn bind(&self, cfg: &Config, run: &Path) -> anyhow::Result<()> {
        std::fs::write(run.join(FILE), &self.bytes)?;
        let mut selection = self.selection(cfg)?;
        let mut marker = json!({"scope":marker_scope_for_scope(&self.manifest.scope)?,"diagnostic_scope":self.manifest.scope,"max_steps":STEPS,
            "release_quality_claim":false,"logit_penalty":0.0,"preflight":self.preflight,
            "fullmix_manifest_sha256":crate::export::sha256_hex(&self.bytes),
            "source_config_sha256":self.manifest.config.sha256,"run_config_sha256":hash(&run.join("config.toml"))?,
            "expected_initial_sha256":self.manifest.expected_initial_sha256,
            "expected_probe_sha256":self.manifest.expected_probe_sha256,"snapshot_steps":SNAPSHOTS});
        if let Some(hard) = self.hard_objective() {
            marker["objective_identity_sha256"] = json!(hard.objective_identity_sha256);
        }
        crate::diagnostic_operator::attach_for_operator(
            run,
            &mut marker,
            &mut selection,
            self.forward_operator()?,
        )?;
        std::fs::write(
            run.join("diagnostic.json"),
            serde_json::to_vec_pretty(&marker)?,
        )?;
        std::fs::write(
            run.join("selection.json"),
            serde_json::to_vec_pretty(&selection)?,
        )?;
        std::fs::write(run.join("cases.jsonl"), self.manifest.cases.bytes()?)?;
        Ok(())
    }

    fn selection(&self, cfg: &Config) -> anyhow::Result<Value> {
        let cohort: Value = serde_json::from_slice(&self.manifest.cohort.bytes()?)?;
        let original_config = Path::new(
            cohort["source_config"]
                .as_str()
                .context("cohort source config missing")?,
        );
        let original = crate::config::load(original_config)?;
        validate_inspection_contract(&self.manifest.scope, cfg, &original)?;
        let prepared = crate::memorization::prepare_frozen(&original, &self.manifest.cohort.path)?;
        let detector = cfg
            .detector
            .as_ref()
            .context("missing candidate detector")?;
        let (candidate_docs, counts) =
            crate::detector::load_silver(&detector.silver, &cfg.features.to_tessera(), 900)?;
        let mut membership = Vec::new();
        for (doc, selected) in prepared.documents.iter().zip(
            prepared.manifest["documents"]
                .as_array()
                .context("missing reference documents")?,
        ) {
            let mut offset = 0;
            let mut matches = Vec::new();
            for (path, count) in detector.silver.iter().zip(&counts.source_pieces) {
                for (index, candidate) in candidate_docs[offset..offset + count].iter().enumerate()
                {
                    if candidate.text == doc.text
                        && candidate.gold == doc.gold
                        && candidate.enc == doc.enc
                    {
                        matches.push(json!({"candidate_source_path":path,"candidate_source_piece":index,"candidate_source_sha256":hash(Path::new(path))?}));
                    }
                }
                offset += count;
            }
            ensure!(
                !matches.is_empty(),
                "original inspection row is absent from the current native silver data"
            );
            membership.push(json!({"name":selected["name"],"candidate_native_matches":matches}));
        }
        let mut selection = prepared.manifest;
        selection["candidate_native_membership"] = json!(membership);
        let reference_initial = selection
            .as_object_mut()
            .context("invalid inspection selection")?
            .remove("expected_initial_sha256");
        selection["scope"] = json!(self.manifest.scope);
        selection["reference_cohort_initial_sha256"] = json!(reference_initial);
        selection["fullmix_manifest_sha256"] = json!(crate::export::sha256_hex(&self.bytes));
        selection["forward_operator"] = json!(match self.forward_operator()? {
            crate::diagnostic_operator::ForwardOperator::ResidualRmsV1 =>
                crate::diagnostic_operator::NAME,
            crate::diagnostic_operator::ForwardOperator::ContextRmsV2 =>
                crate::diagnostic_operator::CONTEXT_NAME,
            _ => anyhow::bail!("inspection requires a bounded RMS operator"),
        });
        if matches!(self.manifest.scope.as_str(), CONTEXT_SCOPE | HARD_SCOPE) {
            selection["reference_network_lineage_only"] = json!(true);
            selection["candidate_network"] = serde_json::to_value(&cfg.net)?;
        }
        if let Some(hard) = self.hard_objective() {
            selection["objective_identity_sha256"] = json!(hard.objective_identity_sha256);
        }
        let (names, _) = crate::loss_diagnostic::load_cases_with_limits(
            &self.manifest.cases.path,
            cfg,
            &selection,
            15,
            900,
        )?;
        ensure!(
            names
                .iter()
                .zip(
                    selection["documents"]
                        .as_array()
                        .context("missing inspection documents")?
                )
                .all(|(name, doc)| doc["name"] == *name),
            "inspection source names differ"
        );
        Ok(selection)
    }

    pub(super) fn selection_identity(&self, cfg: &Config) -> anyhow::Result<String> {
        let mut selection = self.selection(cfg)?;
        selection
            .as_object_mut()
            .context("invalid fullmix selection")?
            .remove("fullmix_manifest_sha256");
        Ok(crate::export::sha256_hex(&serde_json::to_vec(&selection)?))
    }

    pub(crate) fn verify_probe(&self, run: &Path) -> anyhow::Result<String> {
        let actual = hash(&run.join("learning_probe.json"))?;
        ensure!(
            self.manifest
                .expected_probe_sha256
                .as_ref()
                .is_none_or(|expected| *expected == actual),
            "native fullmix probe identity differs"
        );
        Ok(actual)
    }

    pub(crate) fn initialized(&self, run: &Path, initial: &str, probe: &str) -> anyhow::Result<()> {
        ensure!(
            valid_sha(initial)
                && self
                    .manifest
                    .expected_initial_sha256
                    .as_ref()
                    .is_none_or(|expected| expected == initial),
            "native fullmix initialization differs"
        );
        let path = run.join("diagnostic.json");
        let mut marker: Value = serde_json::from_slice(&std::fs::read(&path)?)?;
        marker["actual_initial_sha256"] = json!(initial);
        marker["actual_probe_sha256"] = json!(probe);
        std::fs::write(path, serde_json::to_vec_pretty(&marker)?)?;
        Ok(())
    }

    pub(crate) fn write_preflight(
        &self,
        cfg: &Config,
        run: &Path,
        initial: &str,
        probe: &str,
    ) -> anyhow::Result<()> {
        ensure!(
            self.preflight,
            "preflight proof cannot describe an optimizer run"
        );
        let mut proof = json!({"scope":self.manifest.scope,"preflight":true,"model_initialized":true,"optimizer_updates_executed":0,
                "actual_initial_sha256":initial,"actual_probe_sha256":probe,"source_config_sha256":self.manifest.config.sha256,
                "run_config_sha256":hash(&run.join("config.toml"))?,"input_sha256":self.manifest.input_sha256,
                "operator_sha256":self.manifest.operator_sha256,"binary_sha256":hash(&std::env::current_exe()?)?,
                "checkpoint_sha256":hash(&run.join("best.mpk"))?,"checkpoint_roundtrip_verified":true,
                "initial_checkpoint_event":super::checkpoint_provenance(run)?,
                "checkpoint_events":{"path":run.join(super::CHECKPOINT_LEDGER),"sha256":hash(&run.join(super::CHECKPOINT_LEDGER))?},
                "checkpoint_sidecar":{"path":run.join("checkpoint.json"),"sha256":hash(&run.join("checkpoint.json"))?},
                "recipe_sha256":recipe_sha256(&self.manifest)?,"selection_identity_sha256":self.selection_identity(cfg)?,
                "cohort_sha256":self.manifest.cohort.sha256,"cases_sha256":self.manifest.cases.sha256,
                "exposure_sha256":self.manifest.exposure.sha256,"exposure_targets_sha256":self.manifest.exposure_targets.sha256,"scope_limits":"native initialization and source-backed selection; no forward or optimization"});
        if let Some(hard) = self.hard_objective() {
            proof["objective_identity_sha256"] = json!(hard.objective_identity_sha256);
        }
        std::fs::write(
            run.join("preflight.json"),
            serde_json::to_vec_pretty(&proof)?,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn accounting(
        &self,
        cfg: &Config,
        run: &Path,
        silver: &[crate::detector::DetectorDoc],
        synthetic: &[crate::detector::DetectorDoc],
        pieces: &[usize],
        synthetic_count: usize,
        typed_metadata: &[Value],
    ) -> anyhow::Result<super::Accounting> {
        if let Some(contract) = self.hard_objective() {
            return Ok(super::Accounting::Hard(
                crate::fullmix_hard::Completed::new(
                    contract,
                    cfg,
                    run,
                    synthetic,
                    silver,
                    pieces,
                    synthetic_count,
                    typed_metadata,
                    &self.manifest.config,
                    &self.manifest.exposure,
                )?,
            ));
        }
        if let Some(expected) = self.typed_exposure()? {
            return Ok(super::Accounting::Typed(
                super::typed_completed::TypedCompleted::new(
                    run,
                    expected,
                    typed_metadata,
                    serde_json::from_slice(&self.manifest.exposure_targets.bytes()?)?,
                    silver,
                    &self.manifest.exposure.sha256,
                    &self.manifest.exposure_targets.sha256,
                )?,
            ));
        }
        let request = crate::fullmix_exposure::Request::load(
            &self.manifest.config.path,
            cfg,
            &self.manifest.exposure_targets.path,
            &run.join("unused-planned-output.json"),
        )?;
        Ok(super::Accounting::Legacy(
            request
                .resolve(cfg, silver, pieces, synthetic_count)?
                .completed(&self.manifest.exposure.path, run)?,
        ))
    }

    pub(crate) fn verify_run(
        &self,
        cfg: &Config,
        run: &Path,
        marker: &Value,
        selection: &Value,
    ) -> anyhow::Result<()> {
        let source = crate::config::load(&self.manifest.config.path)?;
        ensure!(
            serde_json::to_value(cfg)? == serde_json::to_value(source)?,
            "serialized run config differs semantically from pinned source"
        );
        ensure!(
            marker["scope"] == marker_scope_for_scope(&self.manifest.scope)?
                && marker["diagnostic_scope"] == self.manifest.scope
                && marker["max_steps"] == STEPS
                && marker["release_quality_claim"] == false
                && marker["logit_penalty"] == 0.0
                && marker["preflight"] == self.preflight
                && marker["fullmix_manifest_sha256"] == crate::export::sha256_hex(&self.bytes)
                && marker["source_config_sha256"] == self.manifest.config.sha256
                && marker["run_config_sha256"] == hash(&run.join("config.toml"))?
                && marker["expected_initial_sha256"]
                    == json!(self.manifest.expected_initial_sha256)
                && marker["expected_probe_sha256"] == json!(self.manifest.expected_probe_sha256)
                && marker["snapshot_steps"] == json!(SNAPSHOTS)
                && self
                    .hard_objective()
                    .is_none_or(|hard| marker["objective_identity_sha256"]
                        == hard.objective_identity_sha256)
                && self
                    .manifest
                    .expected_initial_sha256
                    .as_ref()
                    .is_none_or(|sha| marker["actual_initial_sha256"] == *sha),
            "fullmix diagnostic marker changed"
        );
        ensure!(
            self.selection(cfg)? == *selection,
            "fullmix source-backed inspection selection changed"
        );
        let probe = self.verify_probe(run)?;
        ensure!(
            marker["actual_probe_sha256"] == probe,
            "fullmix actual probe marker differs"
        );
        crate::train::verify_input_snapshot(run)
    }
}

pub(crate) fn verify(
    run: &Path,
    cfg: &Config,
    marker: &Value,
    selection: &Value,
) -> anyhow::Result<()> {
    let preflight = marker["preflight"]
        .as_bool()
        .context("fullmix preflight identity missing")?;
    let (request, _) = Request::load(&run.join(FILE), preflight)?;
    request.verify_run(cfg, run, marker, selection)
}

pub(crate) fn load_inspection(
    run: &Path,
    cfg: &Config,
    gold: &Path,
    selection: &Value,
) -> anyhow::Result<(Vec<String>, Vec<crate::detector::DetectorDoc>)> {
    let marker: Value = serde_json::from_slice(&std::fs::read(run.join("diagnostic.json"))?)?;
    let (request, _) = Request::load(
        &run.join(FILE),
        marker["preflight"]
            .as_bool()
            .context("missing fullmix mode")?,
    )?;
    request.verify_run(cfg, run, &marker, selection)?;
    ensure!(
        hash(gold)? == request.manifest.cases.sha256,
        "fullmix inspection must use the pinned original cases"
    );
    crate::loss_diagnostic::load_cases_with_limits(gold, cfg, selection, 15, 900)
}

pub(crate) fn scoring_docs(
    run: &Path,
    cfg: &Config,
    gold: &Path,
    selection: &Value,
) -> anyhow::Result<Vec<crate::detector::DetectorDoc>> {
    let marker: Value = serde_json::from_slice(&std::fs::read(run.join("diagnostic.json"))?)?;
    let (request, _) = Request::load(
        &run.join(FILE),
        marker["preflight"]
            .as_bool()
            .context("missing fullmix mode")?,
    )?;
    request.verify_run(cfg, run, &marker, selection)?;
    if hash(gold)? == request.manifest.cases.sha256 {
        return Ok(load_inspection(run, cfg, gold, selection)?.1);
    }
    if let Some(docs) = request.seen_docs(run, cfg, gold)? {
        return Ok(docs);
    }
    let snapshot = crate::train::load_input_snapshot(run)?;
    let development = snapshot
        .inputs
        .get("real_dev_gold")
        .context("missing development binding")?;
    ensure!(
        hash(gold)? == development.sha256,
        "fullmix evaluation allows only pinned seen inspection or development gold"
    );
    crate::detector::load_development_gold(gold, &cfg.features.to_tessera())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context96_inspection_accepts_only_the_explicit_network_transition() {
        let original = crate::config::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/detector-shared-v6.toml"),
        )
        .unwrap();
        let mut candidate: Config =
            serde_json::from_value(serde_json::to_value(&original).unwrap()).unwrap();
        validate_inspection_contract(SCOPE, &candidate, &original).unwrap();
        candidate.net.architecture = Some("detector-context96-rms-v2".into());
        candidate.net.dilations.push(64);
        validate_inspection_contract(CONTEXT_SCOPE, &candidate, &original).unwrap();
        validate_inspection_contract(HARD_SCOPE, &candidate, &original).unwrap();
        assert!(validate_inspection_contract(SCOPE, &candidate, &original).is_err());
        candidate.net.dropout = 0.2;
        assert!(validate_inspection_contract(CONTEXT_SCOPE, &candidate, &original).is_err());
        assert!(validate_inspection_contract(HARD_SCOPE, &candidate, &original).is_err());
        candidate.net.dropout = original.net.dropout;
        candidate.features.hash_seed += 1;
        assert!(validate_inspection_contract(CONTEXT_SCOPE, &candidate, &original).is_err());
    }
}
