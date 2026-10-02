//! Native checkpoint events for the fixed fullmix diagnostic and copied scoring views.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, ensure};
use burn::tensor::backend::Backend;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

#[cfg(test)]
use super::SCOPE;
use super::{SNAPSHOTS, STEPS, hash, valid_sha};
use crate::config::Config;
use crate::net::TaggerNet;

pub(crate) const LEDGER: &str = "checkpoint-events.jsonl";

/// Which existing native save path produced a checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum CheckpointKind {
    Snapshot,
    DevelopmentSelected,
    Epoch,
    LearningFailure,
}

impl CheckpointKind {
    fn validate(self, file: &str, steps: usize) -> anyhow::Result<()> {
        let valid = match self {
            Self::Snapshot => {
                SNAPSHOTS.contains(&steps)
                    && file == format!("checkpoints/diagnostic-step-{steps}.mpk")
            }
            Self::DevelopmentSelected => file == "best.mpk",
            Self::Epoch => {
                steps > 0
                    && file
                        .strip_prefix("checkpoints/epoch-")
                        .and_then(|name| name.strip_suffix(".mpk"))
                        .is_some_and(|epoch| epoch.parse::<usize>().is_ok_and(|epoch| epoch > 0))
            }
            Self::LearningFailure => {
                steps > 0 && file == format!("checkpoints/learning-failed-step-{steps}.mpk")
            }
        };
        ensure!(
            steps <= STEPS && valid,
            "fullmix checkpoint save path or step differs"
        );
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Sidecar {
    scope: String,
    event_index: usize,
    checkpoint_file: String,
    optimizer_updates_executed: usize,
    checkpoint_sha256: String,
    parameter_sha256: String,
    selection: CheckpointKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    completed_exposure_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    objective_identity_sha256: Option<String>,
    release_quality_claim: bool,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Event {
    scope: String,
    event_index: usize,
    checkpoint_file: String,
    optimizer_updates_executed: usize,
    checkpoint_sha256: String,
    parameter_sha256: String,
    sidecar_sha256: String,
    selection: CheckpointKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    completed_exposure_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    objective_identity_sha256: Option<String>,
    previous_event_sha256: Option<String>,
}

fn read_events_for_scope(bytes: &[u8], scope: &str) -> anyhow::Result<Vec<(Event, String)>> {
    super::forward_operator_for_scope(scope)?;
    ensure!(
        bytes.is_empty() || bytes.ends_with(b"\n"),
        "checkpoint event ledger is truncated"
    );
    let mut events: Vec<(Event, String)> = Vec::new();
    if bytes.is_empty() {
        return Ok(events);
    }
    for line in bytes[..bytes.len() - 1].split(|byte| *byte == b'\n') {
        let event: Event = serde_json::from_slice(line)?;
        ensure!(
            serde_json::to_vec(&event)? == line,
            "checkpoint event bytes are noncanonical"
        );
        let previous = events.last();
        ensure!(
            event.scope == scope
                && (if scope == super::HARD_SCOPE {
                    event.objective_identity_sha256.as_deref()
                        == Some(crate::fullmix_hard::objective_identity()?.as_str())
                } else {
                    event.objective_identity_sha256.is_none()
                })
                && (!matches!(scope, super::CONTEXT_SCOPE | super::HARD_SCOPE)
                    || event.optimizer_updates_executed != STEPS
                    || event.completed_exposure_sha256.is_some())
                && event.event_index == events.len() + 1
                && event.previous_event_sha256.as_ref() == previous.map(|(_, sha)| sha)
                && previous.is_none_or(|(prior, _)| event.optimizer_updates_executed
                    >= prior.optimizer_updates_executed)
                && event
                    .completed_exposure_sha256
                    .as_ref()
                    .is_none_or(|sha| valid_sha(sha))
                && [
                    &event.checkpoint_sha256,
                    &event.parameter_sha256,
                    &event.sidecar_sha256
                ]
                .into_iter()
                .all(|sha| valid_sha(sha)),
            "checkpoint event sequence differs"
        );
        event
            .selection
            .validate(&event.checkpoint_file, event.optimizer_updates_executed)?;
        events.push((event, crate::export::sha256_hex(line)));
    }
    Ok(events)
}

#[cfg(test)]
fn read_events(bytes: &[u8]) -> anyhow::Result<Vec<(Event, String)>> {
    read_events_for_scope(bytes, SCOPE)
}

/// Save evidence for a checkpoint that the shared loop has already written successfully.
pub(crate) fn checkpoint<B: Backend>(
    model: &TaggerNet<B>,
    cfg: &Config,
    run: &Path,
    path: &Path,
    steps: usize,
    kind: CheckpointKind,
) -> anyhow::Result<()> {
    let marker: Value = serde_json::from_slice(&std::fs::read(run.join("diagnostic.json"))?)?;
    super::validate_marker_scope(&marker)?;
    let scope = super::diagnostic_scope(&marker)?;
    ensure!(
        !matches!(scope, super::CONTEXT_SCOPE | super::HARD_SCOPE)
            || (cfg.context96_rms()
                && cfg.net.hidden == 96
                && cfg.net.kernel == 3
                && cfg.net.dilations == tessera::internal::CONTEXT96_RMS_DILATIONS),
        "context checkpoint requires its exact named seven-block graph"
    );
    let file = path
        .strip_prefix(run)
        .context("checkpoint is outside its diagnostic run")?
        .to_string_lossy()
        .into_owned();
    kind.validate(&file, steps)?;
    let ledger_path = run.join(LEDGER);
    let previous_bytes = match std::fs::read(&ledger_path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(error.into()),
    };
    let previous = read_events_for_scope(&previous_bytes, scope)?;
    let parameter_sha256 = crate::training_diagnostic::parameter_sha256(&crate::quantize::extract(
        model,
        cfg.net_name(),
    ));
    let completed_exposure_sha256 = if steps == STEPS
        && cfg
            .detector
            .as_ref()
            .is_some_and(|d| d.typed_synthetic.is_some())
    {
        let completed_path = run.join("completed-exposure.json");
        let completed: Value = serde_json::from_slice(&std::fs::read(&completed_path)?)?;
        let manifest: Value = serde_json::from_slice(&std::fs::read(run.join(super::FILE))?)?;
        if scope == super::HARD_SCOPE {
            crate::fullmix_hard::validate_completed(&completed, &manifest)?;
        } else {
            ensure!(
                completed["scope"] == "completed-native-typed-fullmix-exposure-v1"
                    && completed["complete"] == true
                    && completed["optimizer_updates_executed"] == STEPS
                    && completed["planned_exposure_sha256"] == manifest["exposure"]["sha256"]
                    && completed["real_targets_sha256"] == manifest["exposure_targets"]["sha256"],
                "typed final checkpoint lacks native completed objective evidence"
            );
        }
        Some(hash(&completed_path)?)
    } else {
        None
    };
    let sidecar = Sidecar {
        scope: scope.into(),
        event_index: previous.len() + 1,
        checkpoint_file: file.clone(),
        optimizer_updates_executed: steps,
        checkpoint_sha256: hash(path)?,
        parameter_sha256: parameter_sha256.clone(),
        selection: kind,
        completed_exposure_sha256: completed_exposure_sha256.clone(),
        objective_identity_sha256: if scope == super::HARD_SCOPE {
            Some(crate::fullmix_hard::objective_identity()?)
        } else {
            None
        },
        release_quality_claim: false,
    };
    let sidecar_bytes = serde_json::to_vec_pretty(&sidecar)?;
    let event = Event {
        scope: scope.into(),
        event_index: sidecar.event_index,
        checkpoint_file: file,
        optimizer_updates_executed: steps,
        checkpoint_sha256: sidecar.checkpoint_sha256.clone(),
        parameter_sha256,
        sidecar_sha256: crate::export::sha256_hex(&sidecar_bytes),
        selection: kind,
        completed_exposure_sha256,
        objective_identity_sha256: sidecar.objective_identity_sha256.clone(),
        previous_event_sha256: previous.last().map(|(_, sha)| sha.clone()),
    };
    ensure!(
        previous
            .last()
            .is_none_or(|(prior, _)| steps >= prior.optimizer_updates_executed),
        "checkpoint updates move backwards"
    );
    std::fs::write(path.with_extension("json"), &sidecar_bytes)?;
    let mut ledger = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(ledger_path)?;
    ledger.write_all(&serde_json::to_vec(&event)?)?;
    ledger.write_all(b"\n")?;
    ledger.flush()?;
    if kind == CheckpointKind::DevelopmentSelected {
        std::fs::write(run.join("checkpoint.json"), &sidecar_bytes)?;
    }
    Ok(())
}

pub(super) fn verify_event_bytes(
    sidecar_bytes: &[u8],
    ledger_bytes: &[u8],
) -> anyhow::Result<Value> {
    let sidecar: Sidecar = serde_json::from_slice(sidecar_bytes)?;
    ensure!(
        super::forward_operator_for_scope(&sidecar.scope).is_ok()
            && !sidecar.release_quality_claim
            && sidecar.event_index > 0
            && (sidecar.scope != super::CONTEXT_SCOPE
                || sidecar.optimizer_updates_executed != STEPS
                || sidecar
                    .completed_exposure_sha256
                    .as_ref()
                    .is_some_and(|sha| valid_sha(sha)))
            && valid_sha(&sidecar.checkpoint_sha256)
            && valid_sha(&sidecar.parameter_sha256),
        "invalid fullmix checkpoint sidecar"
    );
    sidecar
        .selection
        .validate(&sidecar.checkpoint_file, sidecar.optimizer_updates_executed)?;
    let events = read_events_for_scope(ledger_bytes, &sidecar.scope)?;
    let (event, event_sha) = events
        .get(sidecar.event_index - 1)
        .context("checkpoint event is absent")?;
    ensure!(
        event.sidecar_sha256 == crate::export::sha256_hex(sidecar_bytes)
            && event.checkpoint_sha256 == sidecar.checkpoint_sha256
            && event.parameter_sha256 == sidecar.parameter_sha256
            && event.optimizer_updates_executed == sidecar.optimizer_updates_executed
            && event.checkpoint_file == sidecar.checkpoint_file
            && event.selection == sidecar.selection
            && event.completed_exposure_sha256 == sidecar.completed_exposure_sha256
            && event.objective_identity_sha256 == sidecar.objective_identity_sha256,
        "sidecar differs from its native checkpoint event"
    );
    let mut result = serde_json::to_value(sidecar)?;
    result["checkpoint_event_sha256"] = json!(event_sha);
    result["checkpoint_ledger_sha256"] = json!(crate::export::sha256_hex(ledger_bytes));
    Ok(result)
}

/// Validate a scoring view's exact sidecar, preserved native event and checkpoint bytes.
pub(crate) fn checkpoint_provenance(run: &Path) -> anyhow::Result<Value> {
    let result = verify_event_bytes(
        &std::fs::read(run.join("checkpoint.json"))?,
        &std::fs::read(run.join(LEDGER))?,
    )?;
    let marker: Value = serde_json::from_slice(&std::fs::read(run.join("diagnostic.json"))?)?;
    super::validate_marker_scope(&marker)?;
    ensure!(
        result["scope"] == super::diagnostic_scope(&marker)?
            && (result["scope"] != super::HARD_SCOPE
                || result["objective_identity_sha256"] == marker["objective_identity_sha256"]),
        "checkpoint scope differs from its diagnostic marker"
    );
    let initial = marker["actual_initial_sha256"]
        .as_str()
        .context("checkpoint requires actual native initialization")?;
    ensure!(
        valid_sha(initial)
            && result["checkpoint_sha256"] == hash(&run.join("best.mpk"))?
            && (marker["preflight"] != true || result["optimizer_updates_executed"] == 0)
            && ((result["optimizer_updates_executed"] == 0)
                == (result["parameter_sha256"] == initial)),
        "fullmix checkpoint bytes or initial/update identity differs"
    );
    Ok(result)
}

/// Compare actual loaded parameter values before any scoring or inspection forward.
pub(crate) fn verify_checkpoint_parameters(run: &Path, actual: &str) -> anyhow::Result<Value> {
    let proof = checkpoint_provenance(run)?;
    ensure!(
        proof["parameter_sha256"] == actual,
        "loaded native parameters differ from the checkpoint event"
    );
    Ok(proof)
}
#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::module::{Module, Param};
    use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};

    #[test]
    fn legacy_checkpoint_metadata_omits_new_optional_binding() {
        let sidecar_bytes = r#"{"scope":"fullmix-rms-stability-v1","event_index":20,"checkpoint_file":"best.mpk","optimizer_updates_executed":4000,"checkpoint_sha256":"9b3b7ec3d5c717517858ed51a896d6f620e7d8c2778e8d377260a04c69bab1dd","parameter_sha256":"df71db04a5cd12d0016336f40963f914fdfecd616ccddd094b2a2ff94c5e68db","selection":"development-selected","release_quality_claim":false}"#;
        let event_bytes = r#"{"scope":"fullmix-rms-stability-v1","event_index":1,"checkpoint_file":"checkpoints/diagnostic-step-0.mpk","optimizer_updates_executed":0,"checkpoint_sha256":"9a08d402ce84677b0b0ba0fd3e968ab8bf62d6f974749bab23e002a2c4fe4528","parameter_sha256":"7271f208ee44530d419a7aa5160d1a4c4dae5e6c8ecd9881eba78468507efca9","sidecar_sha256":"1dcfa3af0c6d6ba58f101cf421b42a2f65dd57e0fe29f1abd74dcde7e91649d1","selection":"snapshot","previous_event_sha256":null}"#;
        let sidecar: Sidecar = serde_json::from_str(sidecar_bytes).unwrap();
        let event: Event = serde_json::from_str(event_bytes).unwrap();
        assert!(sidecar.completed_exposure_sha256.is_none());
        assert!(event.completed_exposure_sha256.is_none());
        assert_eq!(serde_json::to_string(&sidecar).unwrap(), sidecar_bytes);
        assert_eq!(serde_json::to_string(&event).unwrap(), event_bytes);
    }

    fn metadata_event(scope: &str, steps: usize, completed: Option<String>) -> (Vec<u8>, Vec<u8>) {
        let sidecar = Sidecar {
            scope: scope.into(),
            event_index: 1,
            checkpoint_file: "best.mpk".into(),
            optimizer_updates_executed: steps,
            checkpoint_sha256: "a".repeat(64),
            parameter_sha256: "b".repeat(64),
            selection: CheckpointKind::DevelopmentSelected,
            completed_exposure_sha256: completed.clone(),
            objective_identity_sha256: if scope == super::super::HARD_SCOPE {
                Some(crate::fullmix_hard::objective_identity().unwrap())
            } else {
                None
            },
            release_quality_claim: false,
        };
        let sidecar_bytes = serde_json::to_vec_pretty(&sidecar).unwrap();
        let event = Event {
            scope: scope.into(),
            event_index: 1,
            checkpoint_file: sidecar.checkpoint_file,
            optimizer_updates_executed: steps,
            checkpoint_sha256: sidecar.checkpoint_sha256,
            parameter_sha256: sidecar.parameter_sha256,
            sidecar_sha256: crate::export::sha256_hex(&sidecar_bytes),
            selection: sidecar.selection,
            completed_exposure_sha256: completed,
            objective_identity_sha256: sidecar.objective_identity_sha256.clone(),
            previous_event_sha256: None,
        };
        let mut ledger = serde_json::to_vec(&event).unwrap();
        ledger.push(b'\n');
        (sidecar_bytes, ledger)
    }

    #[test]
    fn context96_checkpoint_events_require_their_scope_and_final_accounting() {
        let scope = super::super::CONTEXT_SCOPE;
        for (steps, completed) in [(0, None), (4000, Some("c".repeat(64)))] {
            let (sidecar, ledger) = metadata_event(scope, steps, completed);
            let proof = verify_event_bytes(&sidecar, &ledger).unwrap();
            assert_eq!(proof["scope"], scope);
            assert!(read_events_for_scope(&ledger, SCOPE).is_err());
            let (_, other_ledger) = metadata_event(SCOPE, steps, None);
            assert!(verify_event_bytes(&sidecar, &other_ledger).is_err());
            assert!(verify_event_bytes(&sidecar, &ledger[..ledger.len() - 1]).is_err());
        }
        let (sidecar, ledger) = metadata_event(scope, 4000, None);
        assert!(verify_event_bytes(&sidecar, &ledger).is_err());
        let (sidecar, ledger) = metadata_event("unknown", 0, None);
        assert!(verify_event_bytes(&sidecar, &ledger).is_err());
        let (sidecar, ledger) = metadata_event(SCOPE, 4000, None);
        assert!(verify_event_bytes(&sidecar, &ledger).is_ok());
    }

    #[test]
    fn hard_checkpoint_events_require_objective_identity_and_final_accounting() {
        let scope = super::super::HARD_SCOPE;
        let (sidecar, ledger) = metadata_event(scope, 4000, Some("c".repeat(64)));
        assert!(verify_event_bytes(&sidecar, &ledger).is_ok());
        let (_, missing) = metadata_event(scope, 4000, None);
        assert!(read_events_for_scope(&missing, scope).is_err());
        let mut event: Value = serde_json::from_slice(&ledger[..ledger.len() - 1]).unwrap();
        for identity in [Value::Null, json!("a".repeat(64))] {
            event["objective_identity_sha256"] = identity;
            let mut changed = serde_json::to_vec(&event).unwrap();
            changed.push(b'\n');
            assert!(read_events_for_scope(&changed, scope).is_err());
        }
        let (legacy, ledger) = metadata_event(SCOPE, 0, None);
        let proof = verify_event_bytes(&legacy, &ledger).unwrap();
        assert!(proof.get("objective_identity_sha256").is_none());
    }

    fn config() -> Config {
        let mut cfg = crate::config::learning_test_config();
        cfg.features.hash_buckets = 64;
        cfg.features.ngram_dim = 4;
        cfg.features.shape_dim = 4;
        cfg.net.hidden = 4;
        cfg.net.dilations = vec![1];
        cfg
    }

    fn parameter_sha(model: &TaggerNet<NdArray>, cfg: &Config) -> String {
        crate::training_diagnostic::parameter_sha256(&crate::quantize::extract(
            model,
            cfg.net_name(),
        ))
    }

    fn marker(run: &Path, initial: &str, preflight: bool) {
        std::fs::write(
            run.join("diagnostic.json"),
            serde_json::to_vec(
                &json!({"actual_initial_sha256":initial,"preflight":preflight,
                "diagnostic_scope":SCOPE,"scope":super::super::MARKER_SCOPE,
                "max_steps":STEPS,"release_quality_claim":false}),
            )
            .unwrap(),
        )
        .unwrap();
    }

    fn changed(mut model: TaggerNet<NdArray>) -> TaggerNet<NdArray> {
        let bias = model.proj.bias.take().unwrap().val() + 0.25;
        model.proj.bias = Some(Param::from_tensor(bias));
        model
    }

    #[test]
    fn native_initial_roundtrip_and_loaded_parameters_bind_the_zero_update_event() {
        let cfg = config();
        let device = Default::default();
        let model = cfg.detector_net_config().init::<NdArray>(&device);
        let initial = parameter_sha(&model, &cfg);
        let run = tempfile::tempdir().unwrap();
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        model
            .clone()
            .save_file(run.path().join("best"), &recorder)
            .unwrap();
        let restored = model
            .clone()
            .load_file(run.path().join("best"), &recorder, &device)
            .unwrap();
        assert_eq!(parameter_sha(&restored, &cfg), initial);
        marker(run.path(), &initial, true);
        checkpoint(
            &restored,
            &cfg,
            run.path(),
            &run.path().join("best.mpk"),
            0,
            CheckpointKind::DevelopmentSelected,
        )
        .unwrap();
        let proof =
            verify_checkpoint_parameters(run.path(), &parameter_sha(&restored, &cfg)).unwrap();
        assert_eq!(proof["optimizer_updates_executed"], 0);
        assert_eq!(proof["parameter_sha256"], initial);
        assert!(
            verify_checkpoint_parameters(run.path(), &parameter_sha(&changed(restored), &cfg))
                .is_err()
        );
        let exact = std::fs::read(run.path().join("checkpoint.json")).unwrap();
        let mut wrong: Value = serde_json::from_slice(&exact).unwrap();
        wrong["parameter_sha256"] = json!("f".repeat(64));
        std::fs::write(
            run.path().join("checkpoint.json"),
            serde_json::to_vec_pretty(&wrong).unwrap(),
        )
        .unwrap();
        assert!(checkpoint_provenance(run.path()).is_err());
        wrong.as_object_mut().unwrap().remove("parameter_sha256");
        std::fs::write(
            run.path().join("checkpoint.json"),
            serde_json::to_vec_pretty(&wrong).unwrap(),
        )
        .unwrap();
        assert!(checkpoint_provenance(run.path()).is_err());
        let mut wrong: Value = serde_json::from_slice(&exact).unwrap();
        marker(run.path(), &initial, false);
        wrong["optimizer_updates_executed"] = json!(4000);
        std::fs::write(
            run.path().join("checkpoint.json"),
            serde_json::to_vec_pretty(&wrong).unwrap(),
        )
        .unwrap();
        assert!(checkpoint_provenance(run.path()).is_err());
        std::fs::write(run.path().join("checkpoint.json"), &exact).unwrap();
        std::fs::write(run.path().join("best.mpk"), b"different bytes").unwrap();
        assert!(checkpoint_provenance(run.path()).is_err());
    }

    #[test]
    fn initial_parameter_identity_cannot_claim_updates_even_with_a_changed_event() {
        let cfg = config();
        let device = Default::default();
        let model = cfg.detector_net_config().init::<NdArray>(&device);
        let initial = parameter_sha(&model, &cfg);
        let run = tempfile::tempdir().unwrap();
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        model
            .clone()
            .save_file(run.path().join("best"), &recorder)
            .unwrap();
        marker(run.path(), &initial, false);
        checkpoint(
            &model,
            &cfg,
            run.path(),
            &run.path().join("best.mpk"),
            0,
            CheckpointKind::DevelopmentSelected,
        )
        .unwrap();
        let mut sidecar: Sidecar =
            serde_json::from_slice(&std::fs::read(run.path().join("checkpoint.json")).unwrap())
                .unwrap();
        sidecar.optimizer_updates_executed = 4000;
        let bytes = serde_json::to_vec_pretty(&sidecar).unwrap();
        let mut events = read_events(&std::fs::read(run.path().join(LEDGER)).unwrap()).unwrap();
        events[0].0.optimizer_updates_executed = 4000;
        events[0].0.sidecar_sha256 = crate::export::sha256_hex(&bytes);
        std::fs::write(run.path().join("checkpoint.json"), bytes).unwrap();
        let mut ledger = serde_json::to_vec(&events[0].0).unwrap();
        ledger.push(b'\n');
        std::fs::write(run.path().join(LEDGER), ledger).unwrap();
        assert!(checkpoint_provenance(run.path()).is_err());
    }

    #[test]
    fn every_native_save_kind_has_an_event_and_copied_views_refuse_wrong_receipts() {
        let cfg = config();
        let device = Default::default();
        let mut model = cfg.detector_net_config().init::<NdArray>(&device);
        let initial = parameter_sha(&model, &cfg);
        let run = tempfile::tempdir().unwrap();
        std::fs::create_dir(run.path().join("checkpoints")).unwrap();
        marker(run.path(), &initial, false);
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::new();
        for (kind, file, steps) in [
            (CheckpointKind::Snapshot, "checkpoints/diagnostic-step-0", 0),
            (CheckpointKind::Epoch, "checkpoints/epoch-1", 557),
            (
                CheckpointKind::Snapshot,
                "checkpoints/diagnostic-step-1000",
                1000,
            ),
            (
                CheckpointKind::LearningFailure,
                "checkpoints/learning-failed-step-4000",
                4000,
            ),
        ] {
            if steps > 0 {
                model = changed(model);
            }
            model
                .clone()
                .save_file(run.path().join(file), &recorder)
                .unwrap();
            checkpoint(
                &model,
                &cfg,
                run.path(),
                &run.path().join(format!("{file}.mpk")),
                steps,
                kind,
            )
            .unwrap();
            assert!(run.path().join(format!("{file}.json")).exists());
        }
        let bytes = std::fs::read(run.path().join(LEDGER)).unwrap();
        let events = read_events(&bytes).unwrap();
        assert_eq!(events.len(), 4);
        assert_eq!(events[1].0.optimizer_updates_executed, 557);
        assert_eq!(events[3].0.selection, CheckpointKind::LearningFailure);
        let view = tempfile::tempdir().unwrap();
        marker(view.path(), &initial, false);
        std::fs::copy(
            run.path().join("checkpoints/learning-failed-step-4000.mpk"),
            view.path().join("best.mpk"),
        )
        .unwrap();
        std::fs::copy(
            run.path()
                .join("checkpoints/learning-failed-step-4000.json"),
            view.path().join("checkpoint.json"),
        )
        .unwrap();
        std::fs::copy(run.path().join(LEDGER), view.path().join(LEDGER)).unwrap();
        assert_eq!(
            verify_checkpoint_parameters(view.path(), &parameter_sha(&model, &cfg)).unwrap()["optimizer_updates_executed"],
            4000
        );
        std::fs::copy(
            run.path().join("checkpoints/epoch-1.json"),
            view.path().join("checkpoint.json"),
        )
        .unwrap();
        assert!(checkpoint_provenance(view.path()).is_err());
        std::fs::copy(
            run.path()
                .join("checkpoints/learning-failed-step-4000.json"),
            view.path().join("checkpoint.json"),
        )
        .unwrap();
        std::fs::write(view.path().join(LEDGER), &bytes[..bytes.len() - 1]).unwrap();
        assert!(checkpoint_provenance(view.path()).is_err());
        std::fs::remove_file(view.path().join(LEDGER)).unwrap();
        assert!(checkpoint_provenance(view.path()).is_err());
    }
}
