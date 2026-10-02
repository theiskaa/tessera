//! One explicit typed diagnostic recipe preserves the legacy fullmix branch.

use std::path::{Path, PathBuf};

use crate::detector::KindSpan;
use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::{Receipt, Request, STEPS};
use crate::config::Config;

/// Exact reviewed measurement plan for the typed diagnostic route.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct TypedDiagnostic {
    pub(super) seen_plan: Receipt,
    pub(super) seen_plan_review: Receipt,
    pub(super) native_preview_reviews: Vec<Receipt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) hard_objective: Option<crate::fullmix_hard::Contract>,
}

impl TypedDiagnostic {
    pub(super) fn seen(&self) -> anyhow::Result<super::training_seen::Seen> {
        super::training_seen::Seen::load(&self.seen_plan, &self.seen_plan_review)
    }

    pub(super) fn input_paths(&self) -> anyhow::Result<Vec<PathBuf>> {
        let mut paths = vec![
            self.seen_plan.path.clone(),
            self.seen_plan_review.path.clone(),
        ];
        paths.extend(self.native_preview_reviews.iter().map(|p| p.path.clone()));
        if let Some(hard) = &self.hard_objective {
            paths.extend(hard.input_paths()?);
        }
        paths.extend(self.seen()?.input_paths()?);
        Ok(paths)
    }

    pub(super) fn validate(
        &self,
        cfg: &Config,
        config: &Receipt,
        exposure_pin: &Receipt,
        targets_pin: &Receipt,
        exposure: &Value,
    ) -> anyhow::Result<()> {
        ensure!(
            cfg.detector
                .as_ref()
                .is_some_and(|d| d.typed_synthetic.is_some()),
            "typed diagnostic needs explicit typed inputs"
        );
        let seen = self.seen()?;
        ensure!(
            self.hard_objective.is_some()
                == (seen.plan["schema"] == "training_seen_evaluation_v5_1912_hard_v1"),
            "hard objective and training-seen plan versions differ"
        );
        ensure!(
            self.native_preview_reviews.len() == 2
                && self.native_preview_reviews[0].sha256 != self.native_preview_reviews[1].sha256
                && std::fs::canonicalize(&self.native_preview_reviews[0].path)?
                    != std::fs::canonicalize(&self.native_preview_reviews[1].path)?,
            "typed preview needs two distinct exact source/data-check reviews"
        );
        for pin in &self.native_preview_reviews {
            let review: Value = serde_json::from_slice(&pin.bytes()?)?;
            ensure!(
                review["passed"] == true
                    && review["blockers"].as_array().is_some_and(Vec::is_empty),
                "typed native preview review failed"
            );
            for required in [config, exposure_pin] {
                let key = required.path.to_string_lossy();
                ensure!(
                    review["reviewed_files"][key.as_ref()] == required.sha256
                        || review["reviewed_files"].as_array().is_some_and(|rows| rows
                            .iter()
                            .any(|p| p["path"] == key.as_ref() && p["sha256"] == required.sha256)),
                    "typed review does not bind required native preview/config"
                );
            }
        }
        if let Some(hard) = &self.hard_objective {
            hard.validate(cfg, config, exposure_pin, targets_pin, exposure)?;
        } else {
            self.validate_legacy_targets(config, targets_pin)?;
        }
        self.validate_native_exposure(config, exposure)
    }

    fn validate_legacy_targets(
        &self,
        config: &Receipt,
        targets_pin: &Receipt,
    ) -> anyhow::Result<()> {
        let targets: Value = serde_json::from_slice(&targets_pin.bytes()?)?;
        ensure!(
            targets["scope"]
                == "exact fourteen reviewed biography-real targets for typed4000 native objective accounting"
                && targets["config"] == serde_json::to_value(config)?
                && targets["model_execution"] == false
                && targets["training_ready"] == false
                && targets["legacy_config_binding_reused"] == false
                && targets["full_synthetic_boundary"] == 170569
                && targets["documents"]
                    .as_array()
                    .is_some_and(|v| v.len() == 14),
            "typed real target reference differs"
        );
        for field in [
            "reviewed_document_dose",
            "reviewed_dose_phase",
            "legacy_targets_lineage_only",
        ] {
            let pin: Receipt = serde_json::from_value(targets[field].clone())?;
            pin.bytes()?;
        }
        for row in targets["documents"]
            .as_array()
            .context("typed targets absent")?
        {
            let pin: Receipt = serde_json::from_value(row["selected_source"].clone())?;
            pin.bytes()?;
        }
        Ok(())
    }

    fn validate_native_exposure(&self, config: &Receipt, exposure: &Value) -> anyhow::Result<()> {
        let actual = &exposure["typed_synthetic"];
        ensure!(
            exposure["config_sha256"] == config.sha256,
            "typed exposure configuration binding differs"
        );
        ensure!(
            actual["scope"]
                == "native encoded typed train-only zero-update finalized prefix; no model/fit/probe integration approval"
                && actual["model_initialized"] == false
                && actual["model_forwards"] == 0
                && actual["optimizer_updates_executed"] == 0
                && actual["planned_updates"] == STEPS
                && actual["uniform_synthetic_pool"] == 170000
                && actual["full_synthetic_boundary"] == 170569
                && actual["real_documents"] == 1799
                && actual["batches"]
                    .as_array()
                    .is_some_and(|v| v.len() == STEPS)
                && actual["authored_documents"]
                    .as_array()
                    .is_some_and(|v| v.len() == 569),
            "typed diagnostic zero-update native exposure contract differs"
        );
        Ok(())
    }
}

impl Request {
    /// Allows authored inputs only through this validated fixed typed diagnostic request.
    pub(crate) fn validate_authored_route(&self, cfg: &Config) -> anyhow::Result<()> {
        let corrected = cfg
            .detector
            .as_ref()
            .and_then(|d| d.typed_synthetic.as_ref())
            .is_some_and(|s| s.corrected_real_sources.is_some());
        if corrected {
            ensure!(
                self.manifest.scope == super::HARD_SCOPE && self.hard_objective().is_some(),
                "corrected real source successors are data-check only; a new reviewed fullmix recipe is required before model initialization"
            );
            super::validate_mode(&self.manifest, self.preflight)?;
            super::validate_config(cfg, &self.manifest.scope)?;
            let exposure: Value = serde_json::from_slice(&self.manifest.exposure.bytes()?)?;
            self.manifest
                .typed_diagnostic
                .as_ref()
                .context("hard typed recipe absent")?
                .validate(
                    cfg,
                    &self.manifest.config,
                    &self.manifest.exposure,
                    &self.manifest.exposure_targets,
                    &exposure,
                )?;
            self.validate_context_inputs(cfg)?;
            crate::typed_synthetic::input_paths(cfg)?;
        } else {
            ensure!(
                self.manifest.scope != super::HARD_SCOPE && self.hard_objective().is_none(),
                "hard recipe requires reviewed corrected real sources"
            );
        }
        if cfg
            .detector
            .as_ref()
            .is_some_and(|d| d.typed_synthetic.is_some())
        {
            ensure!(
                self.manifest.typed_diagnostic.is_some(),
                "authored inputs require the explicitly bound typed fullmix recipe"
            );
            let pinned = crate::config::load(&self.manifest.config.path)?;
            ensure!(
                serde_json::to_value(cfg)? == serde_json::to_value(pinned)?,
                "typed diagnostic effective configuration differs"
            );
            crate::detector::validate_real_silver_sources(
                &cfg.detector.as_ref().context("detector absent")?.silver,
            )?;
        } else {
            ensure!(
                self.manifest.typed_diagnostic.is_none(),
                "typed measurement plan lacks typed inputs"
            );
            crate::typed_synthetic::refuse_fit(cfg)?;
        }
        Ok(())
    }

    pub(crate) fn verify_training_membership(
        &self,
        cfg: &Config,
        authored: &[crate::detector::DetectorDoc],
        real: &[crate::detector::DetectorDoc],
        metadata: &[Value],
    ) -> anyhow::Result<()> {
        if let Some(typed) = &self.manifest.typed_diagnostic {
            typed
                .seen()?
                .validate_membership(cfg, authored, real, metadata)?;
        }
        Ok(())
    }

    pub(super) fn typed_exposure(&self) -> anyhow::Result<Option<Value>> {
        self.manifest
            .typed_diagnostic
            .as_ref()
            .map(|_| -> anyhow::Result<Value> {
                let exposure: Value = serde_json::from_slice(&self.manifest.exposure.bytes()?)?;
                Ok(exposure["typed_synthetic"].clone())
            })
            .transpose()
    }

    pub(crate) fn verify_probe_against_typed_preview(&self, run: &Path) -> anyhow::Result<()> {
        if self.manifest.typed_diagnostic.is_some() {
            let exposure: Value = serde_json::from_slice(&self.manifest.exposure.bytes()?)?;
            let actual: Value =
                serde_json::from_slice(&std::fs::read(run.join("learning_probe.json"))?)?;
            ensure!(
                actual == exposure["learning_probe"],
                "typed training assembly changed the original192 probe"
            );
        }
        Ok(())
    }

    pub(super) fn seen_group(&self, gold: &Path) -> anyhow::Result<Option<String>> {
        self.manifest
            .typed_diagnostic
            .as_ref()
            .map(|t| -> anyhow::Result<Option<String>> {
                Ok(t.seen()?.group(gold)?.map(str::to_owned))
            })
            .transpose()
            .map(Option::flatten)
    }

    pub(super) fn seen_docs(
        &self,
        run: &Path,
        cfg: &Config,
        gold: &Path,
    ) -> anyhow::Result<Option<Vec<crate::detector::DetectorDoc>>> {
        let Some(typed) = &self.manifest.typed_diagnostic else {
            return Ok(None);
        };
        let seen = typed.seen()?;
        let Some(group) = seen.group(gold)? else {
            return Ok(None);
        };
        require_final_snapshot(run)?;
        Ok(Some(seen.docs(group, cfg)?))
    }
}

fn require_final_snapshot(run: &Path) -> anyhow::Result<()> {
    let proof = super::checkpoint_provenance(run)?;
    ensure!(
        proof["optimizer_updates_executed"] == STEPS
            && proof["selection"] == "snapshot"
            && proof["checkpoint_file"] == format!("checkpoints/diagnostic-step-{STEPS}.mpk"),
        "TRAINING-SEEN scoring requires the exact native final4000 snapshot"
    );
    let completed_path = run.join("completed-exposure.json");
    ensure!(
        proof["completed_exposure_sha256"] == super::hash(&completed_path)?,
        "TRAINING-SEEN completed objective differs from native checkpoint event"
    );
    let completed: Value = serde_json::from_slice(&std::fs::read(completed_path)?)?;
    let manifest: Value = serde_json::from_slice(&std::fs::read(run.join(super::FILE))?)?;
    if manifest["scope"] == super::HARD_SCOPE {
        ensure!(
            proof["objective_identity_sha256"]
                == manifest["typed_diagnostic"]["hard_objective"]["objective_identity_sha256"],
            "TRAINING-SEEN checkpoint hard objective identity differs"
        );
        return crate::fullmix_hard::validate_completed(&completed, &manifest);
    }
    ensure!(
        completed["scope"] == "completed-native-typed-fullmix-exposure-v1"
            && completed["complete"] == true
            && completed["optimizer_updates_executed"] == STEPS
            && completed["simulated_records"] == 0
            && completed["planned_exposure_sha256"] == manifest["exposure"]["sha256"]
            && completed["real_targets_sha256"] == manifest["exposure_targets"]["sha256"],
        "TRAINING-SEEN final native objective scope differs"
    );
    Ok(())
}

/// Returns a plan-bound role before accepting the training-seen gold schema.
pub(crate) fn training_seen_role(
    run: &Path,
    cfg: &Config,
    gold: &Path,
) -> anyhow::Result<Option<String>> {
    let marker: Value = serde_json::from_slice(&std::fs::read(run.join("diagnostic.json"))?)?;
    if !super::is_fullmix(&marker) {
        return Ok(None);
    }
    let (request, _) = Request::load(
        &run.join(super::FILE),
        marker["preflight"]
            .as_bool()
            .context("fullmix mode absent")?,
    )?;
    request.validate_authored_route(cfg)?;
    let group = request.seen_group(gold)?;
    if group.is_some() {
        require_final_snapshot(run)?;
    }
    Ok(group)
}

/// Exact family and negative-control observations independent of optional confidence dumps.
pub(crate) fn training_seen_metrics(
    run: &Path,
    cfg: &Config,
    gold_path: &Path,
    gold: &[Vec<KindSpan>],
    raw: &[Vec<KindSpan>],
    filtered: &[Vec<KindSpan>],
) -> anyhow::Result<Option<Value>> {
    let marker: Value = serde_json::from_slice(&std::fs::read(run.join("diagnostic.json"))?)?;
    if !super::is_fullmix(&marker) {
        return Ok(None);
    }
    let (request, _) = Request::load(
        &run.join(super::FILE),
        marker["preflight"]
            .as_bool()
            .context("fullmix mode absent")?,
    )?;
    request.validate_authored_route(cfg)?;
    let Some(typed) = &request.manifest.typed_diagnostic else {
        return Ok(None);
    };
    let seen = typed.seen()?;
    let Some(group) = seen.group(gold_path)? else {
        return Ok(None);
    };
    require_final_snapshot(run)?;
    let mut result = seen.metrics(group, gold, raw, filtered)?;
    result["seen_plan"] = json!({"path":typed.seen_plan.path,"sha256":typed.seen_plan.sha256});
    Ok(Some(result))
}

#[cfg(test)]
mod successor_tests {
    use super::*;

    #[test]
    fn absent_hard_contract_preserves_legacy_typed_serialization() {
        let pin = json!({"path":"/unread.json","sha256":"a".repeat(64)});
        let legacy = json!({"seen_plan":pin,"seen_plan_review":pin,
            "native_preview_reviews":[pin,pin]});
        let typed: TypedDiagnostic = serde_json::from_value(legacy.clone()).unwrap();
        assert!(typed.hard_objective.is_none());
        let prior_bytes = format!(
            "{{\"seen_plan\":{pin},\"seen_plan_review\":{pin},\"native_preview_reviews\":[{pin},{pin}]}}"
        );
        assert_eq!(serde_json::to_vec(&typed).unwrap(), prior_bytes.as_bytes());
        let mut explicit_none = legacy.clone();
        explicit_none["hard_objective"] = Value::Null;
        assert_eq!(
            serde_json::to_value(serde_json::from_value::<TypedDiagnostic>(explicit_none).unwrap())
                .unwrap(),
            legacy
        );
        let mut unknown = legacy;
        unknown["unreviewed_hard_switch"] = json!(true);
        assert!(serde_json::from_value::<TypedDiagnostic>(unknown).is_err());
    }

    #[test]
    fn corrected_real_sources_refuse_fullmix_before_any_recipe_or_model_read() {
        let pin = json!({"path":"/not/read.json","sha256":"a".repeat(64)});
        let mut request = Request {
            manifest: serde_json::from_value(json!({
                "scope":"not-validated", "config":pin, "input_sha256":{},
                "operator_sha256":"not-validated", "expected_initial_sha256":null,
                "expected_probe_sha256":null, "preflight_receipt":null,
                "exposure":pin, "exposure_targets":pin, "cohort":pin, "cases":pin
            }))
            .unwrap(),
            bytes: Vec::new(),
            preflight: false,
        };
        let config = Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/detector-shared.toml");
        let mut cfg = crate::config::load(&config).unwrap();
        cfg.detector.as_mut().unwrap().typed_synthetic = Some(
            serde_json::from_value(json!({
                "manifest":pin, "corrected_real_sources":pin
            }))
            .unwrap(),
        );
        for scope in [
            "not-validated",
            super::super::SCOPE,
            super::super::CONTEXT_SCOPE,
            super::super::HARD_SCOPE,
        ] {
            request.manifest.scope = scope.to_owned();
            let error = request.validate_authored_route(&cfg).unwrap_err();
            assert!(error.to_string().contains("data-check only"));
        }
        assert!(crate::typed_synthetic::refuse_fit(&cfg).is_err());
    }
}
