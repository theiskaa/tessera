//! One pinned CPU mixed-data RMS diagnostic, separate from frozen-cohort memorization.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[path = "fullmix_rms_binding.rs"]
mod binding;
pub(crate) use binding::{load_inspection, scoring_docs, verify};
#[path = "fullmix_checkpoint.rs"]
mod checkpoints;
pub(crate) use checkpoints::{
    CheckpointKind, LEDGER as CHECKPOINT_LEDGER, checkpoint, checkpoint_provenance,
    verify_checkpoint_parameters,
};

#[path = "fullmix_typed.rs"]
mod typed;
pub(crate) use typed::{TypedDiagnostic, training_seen_metrics, training_seen_role};
#[path = "fullmix_training_seen/mod.rs"]
mod training_seen;
#[path = "fullmix_typed_completed.rs"]
mod typed_completed;
pub(crate) use typed_completed::Accounting;

use crate::config::{Config, Task};

pub(crate) const SCOPE: &str = "fullmix-rms-stability-v1";
pub(crate) const CONTEXT_SCOPE: &str = "context96-rms-fit-v1";
/// Separate recipe authority for corrected data and the reviewed hard-token objective.
pub(crate) const HARD_SCOPE: &str = "context96-rms-hard-fit-v1";
/// Training-only marker that cannot reuse the base-objective scope.
pub(crate) const HARD_MARKER_SCOPE: &str =
    "fixed context96 RMS hard-objective training-only diagnostic; never export";
pub(crate) const CONTEXT_MARKER_SCOPE: &str =
    "fixed context96 RMS training-only diagnostic; never export";
pub(crate) const MARKER_SCOPE: &str = "fixed fullmix training-only diagnostic; never export";
pub(crate) const FILE: &str = "fullmix.json";
pub(crate) const STEPS: usize = 4000;
pub(crate) const SNAPSHOTS: [usize; 5] = [0, 1000, 2000, 3000, 4000];

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Receipt {
    pub(crate) path: PathBuf,
    pub(crate) sha256: String,
}

impl Receipt {
    /// Read exact pinned bytes before accepting an artifact as recipe evidence.
    pub(crate) fn bytes(&self) -> anyhow::Result<Vec<u8>> {
        let bytes = std::fs::read(&self.path)?;
        ensure!(
            crate::export::sha256_hex(&bytes) == self.sha256,
            "fullmix receipt changed: {}",
            self.path.display()
        );
        Ok(bytes)
    }
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    scope: String,
    config: Receipt,
    input_sha256: BTreeMap<String, String>,
    operator_sha256: String,
    expected_initial_sha256: Option<String>,
    expected_probe_sha256: Option<String>,
    preflight_receipt: Option<Receipt>,
    exposure: Receipt,
    exposure_targets: Receipt,
    cohort: Receipt,
    cases: Receipt,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    typed_diagnostic: Option<TypedDiagnostic>,
}

pub(crate) struct Request {
    manifest: Manifest,
    bytes: Vec<u8>,
    pub(crate) preflight: bool,
}

fn hash(path: &Path) -> anyhow::Result<String> {
    Ok(crate::export::sha256_hex(&std::fs::read(path)?))
}

fn valid_sha(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn recipe_sha256(manifest: &Manifest) -> anyhow::Result<String> {
    let mut recipe = serde_json::to_value(manifest)?;
    let fields = recipe.as_object_mut().context("invalid fullmix recipe")?;
    for key in [
        "expected_initial_sha256",
        "expected_probe_sha256",
        "preflight_receipt",
    ] {
        fields.remove(key);
    }
    Ok(crate::export::sha256_hex(&serde_json::to_vec(&recipe)?))
}

fn forward_operator_for_scope(
    scope: &str,
) -> anyhow::Result<crate::diagnostic_operator::ForwardOperator> {
    match scope {
        SCOPE => Ok(crate::diagnostic_operator::ForwardOperator::ResidualRmsV1),
        CONTEXT_SCOPE | HARD_SCOPE => Ok(crate::diagnostic_operator::ForwardOperator::ContextRmsV2),
        _ => anyhow::bail!("unknown fullmix scope"),
    }
}

fn marker_scope_for_scope(scope: &str) -> anyhow::Result<&'static str> {
    match scope {
        SCOPE => Ok(MARKER_SCOPE),
        CONTEXT_SCOPE => Ok(CONTEXT_MARKER_SCOPE),
        HARD_SCOPE => Ok(HARD_MARKER_SCOPE),
        _ => anyhow::bail!("unknown fullmix marker scope"),
    }
}

/// Resolve only the explicit bounded diagnostic scopes.
pub(crate) fn diagnostic_scope(marker: &Value) -> anyhow::Result<&'static str> {
    match marker["diagnostic_scope"].as_str() {
        Some(SCOPE) => Ok(SCOPE),
        Some(CONTEXT_SCOPE) => Ok(CONTEXT_SCOPE),
        Some(HARD_SCOPE) => Ok(HARD_SCOPE),
        _ => anyhow::bail!("unknown fullmix diagnostic scope"),
    }
}

/// Require the marker to name the matching scope and fixed optimizer budget.
pub(crate) fn validate_marker_scope(marker: &Value) -> anyhow::Result<()> {
    ensure!(
        marker["scope"] == marker_scope_for_scope(diagnostic_scope(marker)?)?
            && marker["max_steps"] == STEPS
            && marker["release_quality_claim"] == false,
        "fullmix marker scope differs"
    );
    Ok(())
}

fn validate_mode(manifest: &Manifest, preflight: bool) -> anyhow::Result<()> {
    let canonical = match forward_operator_for_scope(&manifest.scope)? {
        crate::diagnostic_operator::ForwardOperator::ResidualRmsV1 => {
            crate::diagnostic_operator::CANONICAL
        }
        crate::diagnostic_operator::ForwardOperator::ContextRmsV2 => {
            crate::diagnostic_operator::CONTEXT_CANONICAL
        }
        _ => anyhow::bail!("fullmix requires an explicit bounded RMS operator"),
    };
    ensure!(
        manifest.operator_sha256 == crate::export::sha256_hex(canonical)
            && (!matches!(manifest.scope.as_str(), CONTEXT_SCOPE | HARD_SCOPE)
                || manifest.typed_diagnostic.is_some()),
        "unknown fullmix operator or context scope lacks corrected typed inputs"
    );
    ensure!(
        (manifest.scope == HARD_SCOPE)
            == manifest
                .typed_diagnostic
                .as_ref()
                .is_some_and(|typed| typed.hard_objective.is_some()),
        "hard objective requires its explicit bounded recipe scope"
    );
    if let Some(hard) = manifest
        .typed_diagnostic
        .as_ref()
        .and_then(|typed| typed.hard_objective.as_ref())
    {
        hard.validate_identity()?;
    }
    for sha in [
        &manifest.expected_initial_sha256,
        &manifest.expected_probe_sha256,
    ]
    .into_iter()
    .flatten()
    {
        ensure!(valid_sha(sha), "invalid expected fullmix identity");
    }
    ensure!(
        preflight
            || (manifest.expected_initial_sha256.is_some()
                && manifest.expected_probe_sha256.is_some()
                && manifest.preflight_receipt.is_some()),
        "fitting requires pinned native initialization and probe from preflight"
    );
    Ok(())
}

fn validate_config(cfg: &Config, scope: &str) -> anyhow::Result<()> {
    forward_operator_for_scope(scope)?;
    let network_matches = match scope {
        SCOPE => cfg.net.architecture.is_none() && cfg.net.dilations == [1, 2, 4, 8, 16, 1],
        CONTEXT_SCOPE | HARD_SCOPE => {
            cfg.context96_rms() && cfg.net.dilations == tessera::internal::CONTEXT96_RMS_DILATIONS
        }
        _ => false,
    };
    let detector = cfg.detector.as_ref().context("fullmix needs a detector")?;
    ensure!(
        cfg.task == Task::Detector
            && cfg.data.countries == ["US"]
            && cfg.seed == 42
            && cfg.train.batch_size == 32
            && cfg.train.epochs == 25
            && cfg.train.learning_rate == 0.0001
            && cfg.train.warmup_steps == 500
            && cfg.train.gradient_clip_norm == Some(1.0)
            && cfg.train.diagnostic_schedule_steps == Some(7000)
            && cfg.net.hidden == 96
            && cfg.net.kernel == 3
            && network_matches
            && cfg.net.dropout == 0.1
            && cfg.net.ngram_from.is_some()
            && cfg.net.finetune_ngram
            && detector.class_weights == [1.0, 3.0, 2.0, 3.0, 2.0, 2.0, 1.5]
            && detector.synthetic_per_epoch == Some(12000)
            && !detector.silver_repeats.is_empty(),
        "fullmix objective, initialization, network, or sampler contract differs"
    );
    let check = detector
        .learning_check
        .as_ref()
        .context("fullmix needs the unchanged learning probe")?;
    ensure!(
        check.documents == 192
            && check.every_steps == 500
            && check.start_step == STEPS
            && check.minimum_recall == 0.5,
        "fullmix learning cadence differs"
    );
    crate::training_diagnostic::validate_launch(cfg, Some(STEPS))
}

fn validate_context_plan_bindings(
    plan: &Value,
    config: &Receipt,
    exposure: &Receipt,
    typed_manifest: &Value,
) -> anyhow::Result<()> {
    let binding = &plan["current_input_binding"];
    ensure!(
        plan["schema"] == "training_seen_evaluation_v4_1904_v1"
            && plan["model_gold_support"]["real"]["org"] == 1904
            && binding["config"] == serde_json::to_value(config)?
            && binding["native_preview"] == serde_json::to_value(exposure)?
            && binding["typed_manifest"] == *typed_manifest,
        "context scope requires a separately reviewed new-graph config and preview in its current1904 plan"
    );
    Ok(())
}

impl Request {
    /// Select the operator from the validated recipe scope, never from a legacy toggle.
    pub(crate) fn forward_operator(
        &self,
    ) -> anyhow::Result<crate::diagnostic_operator::ForwardOperator> {
        forward_operator_for_scope(&self.manifest.scope)
    }

    /// Returns the objective only for its separately validated bounded recipe.
    pub(crate) fn hard_objective(&self) -> Option<&crate::fullmix_hard::Contract> {
        self.manifest
            .typed_diagnostic
            .as_ref()?
            .hard_objective
            .as_ref()
    }

    fn validate_context_inputs(&self, cfg: &Config) -> anyhow::Result<()> {
        if !matches!(self.manifest.scope.as_str(), CONTEXT_SCOPE | HARD_SCOPE) {
            return Ok(());
        }
        let typed = self
            .manifest
            .typed_diagnostic
            .as_ref()
            .context("context scope lacks typed recipe")?;
        let seen = typed.seen()?;
        let settings = cfg
            .detector
            .as_ref()
            .and_then(|d| d.typed_synthetic.as_ref())
            .context("context scope lacks corrected typed input manifest")?;
        let binding = &seen.plan["current_input_binding"];
        if self.manifest.scope == HARD_SCOPE {
            let hard = self.hard_objective().context("hard objective absent")?;
            let objective_pin: Receipt = serde_json::from_value(binding["hard_objective"].clone())?;
            let objective: Value = serde_json::from_slice(&objective_pin.bytes()?)?;
            ensure!(
                seen.plan["schema"] == "training_seen_evaluation_v5_1912_hard_v1"
                    && seen.plan["model_gold_support"]["real"]["org"] == 1912
                    && binding["config"] == serde_json::to_value(&self.manifest.config)?
                    && binding["base_native_preview"]
                        == serde_json::to_value(&self.manifest.exposure)?
                    && binding["typed_manifest"] == serde_json::to_value(&settings.manifest)?
                    && binding["source_successor_receipt"]
                        == serde_json::to_value(&settings.corrected_real_sources)?
                    && binding["selector_draft"] == serde_json::to_value(&hard.selector_draft)?
                    && binding["native_selector_proof"]
                        == serde_json::to_value(&hard.native_selector_proof)?
                    && binding["hard_dose_preview"]
                        == serde_json::to_value(&hard.hard_dose_preview)?
                    && objective == serde_json::to_value(hard)?,
                "hard scope requires its exact reviewed1912 config, sources, selectors and dose preview"
            );
            return Ok(());
        }
        validate_context_plan_bindings(
            &seen.plan,
            &self.manifest.config,
            &self.manifest.exposure,
            &serde_json::to_value(&settings.manifest)?,
        )?;
        let manifest_pin: Receipt = serde_json::from_value(binding["typed_manifest"].clone())?;
        let manifest: Value = serde_json::from_slice(&manifest_pin.bytes()?)?;
        ensure!(
            manifest["real_amendments"] == binding["real_amendments"],
            "context amendment binding differs"
        );
        let amendment_pin: Receipt = serde_json::from_value(binding["real_amendments"].clone())?;
        let amendment: Value = serde_json::from_slice(&amendment_pin.bytes()?)?;
        ensure!(
            amendment["schema"] == "reviewed_training_amendments_v4",
            "context requires corrected v4 real inputs"
        );
        Ok(())
    }

    pub(crate) fn load(path: &Path, preflight: bool) -> anyhow::Result<(Self, Config)> {
        let producer = crate::diagnostic_decode::producer_binary()?;
        let bytes = std::fs::read(path)?;
        let manifest: Manifest = serde_json::from_slice(&bytes)?;
        validate_mode(&manifest, preflight)?;
        manifest.config.bytes()?;
        let cfg = crate::config::load(&manifest.config.path)?;
        validate_config(&cfg, &manifest.scope)?;
        ensure!(
            !matches!(manifest.scope.as_str(), CONTEXT_SCOPE | HARD_SCOPE) || producer.is_none(),
            "context fit cannot reuse a compiled historical producer identity"
        );
        let mut input_paths = crate::train::checked_input_paths(&cfg)?;
        if let Some(typed) = &manifest.typed_diagnostic {
            input_paths.extend(
                typed
                    .input_paths()?
                    .into_iter()
                    .enumerate()
                    .map(|(i, path)| (format!("training_seen_input_{i:04}"), path)),
            );
        }
        let actual: BTreeMap<_, _> = input_paths
            .into_iter()
            .map(|(_, path)| Ok((path.to_string_lossy().into_owned(), hash(&path)?)))
            .collect::<anyhow::Result<_>>()?;
        ensure!(
            actual == manifest.input_sha256,
            "fullmix input closure differs or omits native dependencies"
        );
        let exposure: Value = serde_json::from_slice(&manifest.exposure.bytes()?)?;
        if let Some(typed) = &manifest.typed_diagnostic {
            typed.validate(
                &cfg,
                &manifest.config,
                &manifest.exposure,
                &manifest.exposure_targets,
                &exposure,
            )?;
        } else {
            ensure!(
                exposure["scope"] == "planned-fullmix-biography-exposure-v1"
                    && exposure["planned_updates"] == STEPS
                    && exposure["model_initialized"] == false
                    && exposure["optimizer_updates_executed"] == 0
                    && exposure["native_replay_matches_pinned_reference"] == true,
                "fullmix needs the reviewed zero-update schedule audit"
            );
            for (path, expected) in exposure["input_sha256"]
                .as_object()
                .context("missing exposure input closure")?
            {
                ensure!(
                    expected.as_str() == Some(hash(Path::new(path))?.as_str()),
                    "planned exposure input changed: {path}"
                );
            }
        }
        manifest.exposure_targets.bytes()?;
        manifest.cohort.bytes()?;
        manifest.cases.bytes()?;
        let request = Self {
            manifest,
            bytes,
            preflight,
        };
        request.validate_context_inputs(&cfg)?;
        request.validate_authored_route(&cfg)?;
        let manifest = &request.manifest;
        if let Some(receipt) = &manifest.preflight_receipt {
            let proof: Value = serde_json::from_slice(&receipt.bytes()?)?;
            let ledger: Receipt = serde_json::from_value(proof["checkpoint_events"].clone())?;
            let sidecar: Receipt = serde_json::from_value(proof["checkpoint_sidecar"].clone())?;
            let event = checkpoints::verify_event_bytes(&sidecar.bytes()?, &ledger.bytes()?)?;
            ensure!(
                proof["scope"] == manifest.scope
                    && event["scope"] == manifest.scope
                    && proof["preflight"] == true
                    && proof["model_initialized"] == true
                    && proof["optimizer_updates_executed"] == 0
                    && proof["checkpoint_roundtrip_verified"] == true
                    && proof["initial_checkpoint_event"] == event
                    && event["optimizer_updates_executed"] == 0
                    && event["parameter_sha256"] == proof["actual_initial_sha256"]
                    && event["checkpoint_sha256"] == proof["checkpoint_sha256"]
                    && proof["recipe_sha256"] == recipe_sha256(manifest)?
                    && proof["selection_identity_sha256"] == request.selection_identity(&cfg)?
                    && proof["cohort_sha256"] == manifest.cohort.sha256
                    && proof["cases_sha256"] == manifest.cases.sha256
                    && proof["exposure_sha256"] == manifest.exposure.sha256
                    && proof["exposure_targets_sha256"] == manifest.exposure_targets.sha256
                    && proof["source_config_sha256"] == manifest.config.sha256
                    && proof["input_sha256"] == json!(manifest.input_sha256)
                    && proof["operator_sha256"] == manifest.operator_sha256
                    && request.hard_objective().is_none_or(
                        |hard| proof["objective_identity_sha256"] == hard.objective_identity_sha256
                            && event["objective_identity_sha256"] == hard.objective_identity_sha256
                    )
                    && crate::diagnostic_decode::validate_preflight_binary(
                        producer.as_ref(),
                        &proof["binary_sha256"],
                    )?
                    && manifest
                        .expected_initial_sha256
                        .as_ref()
                        .is_none_or(|sha| proof["actual_initial_sha256"] == *sha)
                    && manifest
                        .expected_probe_sha256
                        .as_ref()
                        .is_none_or(|sha| proof["actual_probe_sha256"] == *sha),
                "fullmix preflight identity differs"
            );
        }
        Ok((request, cfg))
    }
}

pub(crate) fn is_fullmix(marker: &Value) -> bool {
    matches!(
        marker["diagnostic_scope"].as_str(),
        Some(SCOPE | CONTEXT_SCOPE | HARD_SCOPE)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> Manifest {
        let receipt = || Receipt {
            path: "unread".into(),
            sha256: "a".repeat(64),
        };
        Manifest {
            scope: SCOPE.into(),
            config: receipt(),
            input_sha256: BTreeMap::new(),
            operator_sha256: crate::export::sha256_hex(crate::diagnostic_operator::CANONICAL),
            expected_initial_sha256: None,
            expected_probe_sha256: None,
            preflight_receipt: None,
            exposure: receipt(),
            exposure_targets: receipt(),
            cohort: receipt(),
            cases: receipt(),
            typed_diagnostic: None,
        }
    }

    #[test]
    fn missing_native_identities_are_allowed_only_for_explicit_preflight() {
        let mut plan = manifest();
        validate_mode(&plan, true).unwrap();
        assert!(validate_mode(&plan, false).is_err());
        plan.expected_initial_sha256 = Some("a".repeat(64));
        assert!(validate_mode(&plan, false).is_err());
        plan.expected_probe_sha256 = Some("b".repeat(64));
        assert!(validate_mode(&plan, false).is_err());
        plan.preflight_receipt = Some(Receipt {
            path: "proof".into(),
            sha256: "c".repeat(64),
        });
        validate_mode(&plan, false).unwrap();
        plan.scope = crate::memorization::FROZEN_SCOPE.into();
        assert!(validate_mode(&plan, true).is_err());
    }

    #[test]
    fn recipe_identity_ignores_only_discovered_fields_and_detects_changed_investigation() {
        let mut plan = manifest();
        let recipe = recipe_sha256(&plan).unwrap();
        plan.expected_initial_sha256 = Some("a".repeat(64));
        plan.expected_probe_sha256 = Some("b".repeat(64));
        plan.preflight_receipt = Some(Receipt {
            path: "proof.json".into(),
            sha256: "c".repeat(64),
        });
        assert_eq!(recipe_sha256(&plan).unwrap(), recipe);
        plan.cases.sha256 = "d".repeat(64);
        assert_ne!(recipe_sha256(&plan).unwrap(), recipe);
        plan.cases.sha256 = "a".repeat(64);
        plan.exposure.sha256 = "e".repeat(64);
        assert_ne!(recipe_sha256(&plan).unwrap(), recipe);
    }

    #[test]
    fn changed_native_initialization_is_refused_before_marker_mutation() {
        let mut plan = manifest();
        plan.expected_initial_sha256 = Some("a".repeat(64));
        let request = Request {
            manifest: plan,
            bytes: Vec::new(),
            preflight: false,
        };
        let run = tempfile::tempdir().unwrap();
        let marker = b"{\"scope\":\"unchanged\"}";
        std::fs::write(run.path().join("diagnostic.json"), marker).unwrap();
        assert!(
            request
                .initialized(run.path(), &"b".repeat(64), &"c".repeat(64))
                .is_err()
        );
        assert_eq!(
            std::fs::read(run.path().join("diagnostic.json")).unwrap(),
            marker
        );
        request
            .initialized(run.path(), &"a".repeat(64), &"c".repeat(64))
            .unwrap();
        let written: Value =
            serde_json::from_slice(&std::fs::read(run.path().join("diagnostic.json")).unwrap())
                .unwrap();
        assert_eq!(written["actual_initial_sha256"], "a".repeat(64));
        assert!(!run.path().join("preflight.json").exists());
    }

    #[test]
    fn fitting_refuses_operator_and_initial_identity_substitution() {
        let mut plan = manifest();
        plan.operator_sha256 = "0".repeat(64);
        assert!(validate_mode(&plan, true).is_err());
        plan.operator_sha256 = crate::export::sha256_hex(crate::diagnostic_operator::CANONICAL);
        plan.expected_initial_sha256 = Some("not-a-hash".into());
        assert!(validate_mode(&plan, true).is_err());
    }
    #[test]
    fn context96_scope_operator_and_mode_do_not_reuse_legacy_authorization() {
        let mut legacy = manifest();
        validate_mode(&legacy, true).unwrap();
        assert_eq!(
            forward_operator_for_scope(SCOPE).unwrap(),
            crate::diagnostic_operator::ForwardOperator::ResidualRmsV1
        );
        legacy.scope = CONTEXT_SCOPE.into();
        assert!(validate_mode(&legacy, true).is_err());
        legacy.operator_sha256 =
            crate::export::sha256_hex(crate::diagnostic_operator::CONTEXT_CANONICAL);
        assert!(validate_mode(&legacy, true).is_err());
        let receipt = || Receipt {
            path: "unused".into(),
            sha256: "a".repeat(64),
        };
        legacy.typed_diagnostic = Some(TypedDiagnostic {
            seen_plan: receipt(),
            seen_plan_review: receipt(),
            native_preview_reviews: vec![receipt(), receipt()],
            hard_objective: None,
        });
        validate_mode(&legacy, true).unwrap();
        assert!(validate_mode(&legacy, false).is_err());
        assert_eq!(
            forward_operator_for_scope(CONTEXT_SCOPE).unwrap(),
            crate::diagnostic_operator::ForwardOperator::ContextRmsV2
        );
        assert!(forward_operator_for_scope("context96-rms-fit-v2").is_err());
        let context_hash = legacy.operator_sha256.clone();
        legacy.scope = SCOPE.into();
        assert!(validate_mode(&legacy, true).is_err());
        legacy.scope = CONTEXT_SCOPE.into();
        legacy.operator_sha256 = context_hash;
        legacy.expected_initial_sha256 = Some("a".repeat(64));
        legacy.expected_probe_sha256 = Some("b".repeat(64));
        legacy.preflight_receipt = Some(receipt());
        validate_mode(&legacy, false).unwrap();
    }

    #[test]
    fn context96_scope_config_accepts_only_the_named_seven_block_graph() {
        let mut cfg = crate::config::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/detector-shared-v6.toml"),
        )
        .unwrap();
        cfg.train.diagnostic_schedule_steps = Some(7000);
        let detector = cfg.detector.as_mut().unwrap();
        detector.synthetic_per_epoch = Some(12000);
        detector.learning_check.as_mut().unwrap().start_step = STEPS;
        validate_config(&cfg, SCOPE).unwrap();
        assert!(validate_config(&cfg, CONTEXT_SCOPE).is_err());
        cfg.net.architecture = Some("detector-context96-rms-v2".into());
        cfg.net.dilations.push(64);
        validate_config(&cfg, CONTEXT_SCOPE).unwrap();
        validate_config(&cfg, HARD_SCOPE).unwrap();
        assert!(validate_config(&cfg, SCOPE).is_err());
        cfg.net.dilations[6] = 32;
        assert!(validate_config(&cfg, CONTEXT_SCOPE).is_err());
        assert!(validate_config(&cfg, HARD_SCOPE).is_err());
        cfg.net.dilations[6] = 64;
        cfg.net.architecture = None;
        assert!(validate_config(&cfg, CONTEXT_SCOPE).is_err());
    }

    #[test]
    fn context96_plan_requires_its_new_config_preview_and_corrected_manifest() {
        let config = Receipt {
            path: "/new/config.toml".into(),
            sha256: "a".repeat(64),
        };
        let exposure = Receipt {
            path: "/new/preview.json".into(),
            sha256: "b".repeat(64),
        };
        let typed = json!({"path":"/new/typed.json","sha256":"c".repeat(64)});
        let plan = json!({"schema":"training_seen_evaluation_v4_1904_v1",
            "model_gold_support":{"real":{"org":1904}},
            "current_input_binding":{"config":config,"native_preview":exposure,"typed_manifest":typed}});
        validate_context_plan_bindings(&plan, &config, &exposure, &typed).unwrap();
        for field in ["config", "native_preview", "typed_manifest"] {
            let mut changed = plan.clone();
            changed["current_input_binding"][field]["sha256"] = json!("d".repeat(64));
            assert!(validate_context_plan_bindings(&changed, &config, &exposure, &typed).is_err());
        }
        let mut changed = plan.clone();
        changed["model_gold_support"]["real"]["org"] = json!(1900);
        assert!(validate_context_plan_bindings(&changed, &config, &exposure, &typed).is_err());
        changed = plan.clone();
        changed["schema"] = Value::Null;
        assert!(validate_context_plan_bindings(&changed, &config, &exposure, &typed).is_err());
    }

    #[test]
    fn context96_markers_cannot_relabel_legacy_scope_or_budget() {
        let marker = json!({"scope":CONTEXT_MARKER_SCOPE,"diagnostic_scope":CONTEXT_SCOPE,
            "max_steps":STEPS,"release_quality_claim":false});
        validate_marker_scope(&marker).unwrap();
        for (field, value) in [
            ("scope", json!(MARKER_SCOPE)),
            ("diagnostic_scope", json!(SCOPE)),
            ("max_steps", json!(6000)),
            ("release_quality_claim", json!(true)),
        ] {
            let mut changed = marker.clone();
            changed[field] = value;
            assert!(validate_marker_scope(&changed).is_err());
        }
        assert!(diagnostic_scope(&json!({"diagnostic_scope":"unknown"})).is_err());
    }

    #[test]
    fn hard_scope_requires_its_contract_and_preserves_the_context_operator() {
        let mut plan = manifest();
        plan.scope = HARD_SCOPE.into();
        plan.operator_sha256 =
            crate::export::sha256_hex(crate::diagnostic_operator::CONTEXT_CANONICAL);
        assert!(validate_mode(&plan, true).is_err());
        let pin = json!({"path":"/unread.json","sha256":"a".repeat(64)});
        plan.typed_diagnostic = Some(
            serde_json::from_value(json!({
                "seen_plan":pin, "seen_plan_review":pin, "native_preview_reviews":[pin,pin]
            }))
            .unwrap(),
        );
        assert!(validate_mode(&plan, true).is_err());
        let hard = json!({
            "schema":"native_hard_objective_v1",
            "objective_identity_sha256":crate::fullmix_hard::objective_identity().unwrap(),
            "selector_draft":pin,"native_selector_proof":pin,"hard_dose_preview":pin,
            "corrected_sources":pin,"corrected_gold":pin,"source_reviews":[pin,pin],
            "native_dose_reviews":[pin,pin],"measured_batch_source":pin,"feature_sha256":"a".repeat(64),
            "class_weights_sha256":"b".repeat(64),"groups":["person","org_contrast","address"],
            "coefficients":[0.0625,0.0625,0.0625]
        });
        plan.typed_diagnostic.as_mut().unwrap().hard_objective =
            Some(serde_json::from_value(hard).unwrap());
        validate_mode(&plan, true).unwrap();
        assert!(validate_mode(&plan, false).is_err());
        let recipe = recipe_sha256(&plan).unwrap();
        plan.scope = CONTEXT_SCOPE.into();
        assert!(validate_mode(&plan, true).is_err());
        plan.scope = SCOPE.into();
        plan.operator_sha256 = crate::export::sha256_hex(crate::diagnostic_operator::CANONICAL);
        assert!(validate_mode(&plan, true).is_err());
        plan.scope = HARD_SCOPE.into();
        plan.operator_sha256 =
            crate::export::sha256_hex(crate::diagnostic_operator::CONTEXT_CANONICAL);
        let contract = plan
            .typed_diagnostic
            .as_mut()
            .unwrap()
            .hard_objective
            .as_mut()
            .unwrap();
        contract.coefficients[0] = 0.125;
        assert!(validate_mode(&plan, true).is_err());
        assert_ne!(recipe_sha256(&plan).unwrap(), recipe);
        assert_eq!(
            forward_operator_for_scope(HARD_SCOPE).unwrap(),
            forward_operator_for_scope(CONTEXT_SCOPE).unwrap()
        );
        let marker = json!({"scope":HARD_MARKER_SCOPE,"diagnostic_scope":HARD_SCOPE,
            "max_steps":STEPS,"release_quality_claim":false});
        validate_marker_scope(&marker).unwrap();
        assert!(is_fullmix(&marker));
        for (field, value) in [
            ("scope", json!(CONTEXT_MARKER_SCOPE)),
            ("diagnostic_scope", json!(CONTEXT_SCOPE)),
            ("max_steps", json!(3999)),
        ] {
            let mut changed = marker.clone();
            changed[field] = value;
            assert!(validate_marker_scope(&changed).is_err());
        }
    }
}
