//! Explicit, hashed forward semantics for a bounded whole-original diagnostic.

use std::path::Path;

use anyhow::{Context, ensure};
use burn::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::config::Config;
use crate::net::TaggerNet;

pub(crate) const FILE: &str = "operator.json";
pub(crate) const NAME: &str = "residual-rms-v1";
pub(crate) const CANONICAL: &[u8] = b"{\"name\":\"residual-rms-v1\",\"epsilon\":0.00001,\"axis\":1,\"blocks\":6,\"channels\":96,\"placement\":\"post-residual-add\",\"learned_parameters\":0}\n";

/// Immutable forward selection, kept outside the model's parameter record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ForwardOperator {
    Standard,
    ResidualRmsV1,
    ContextRmsV2,
}

impl ForwardOperator {
    /// The named context graph always uses its bound decoder; legacy graphs preserve explicit opt-in behavior.
    pub(crate) fn decoder(
        self,
        explicit: Option<crate::diagnostic_decode::Decoder>,
    ) -> Option<crate::diagnostic_decode::Decoder> {
        if self == Self::ContextRmsV2 {
            Some(crate::diagnostic_decode::Decoder::AddressContinuationV1)
        } else {
            explicit
        }
    }

    pub(crate) fn forward<B: Backend>(
        self,
        model: &TaggerNet<B>,
        ids: Tensor<B, 3, Int>,
        script: Tensor<B, 2, Int>,
        shape: Tensor<B, 2, Int>,
        flags: Tensor<B, 3>,
        mask: Tensor<B, 2>,
    ) -> anyhow::Result<Tensor<B, 3>> {
        match self {
            Self::Standard => {
                ensure!(
                    model.blocks.len() != 7,
                    "standard forward refuses the named context RMS graph"
                );
                Ok(model.forward(ids, script, shape, flags, mask))
            }
            Self::ResidualRmsV1 => {
                ensure!(
                    model.blocks.len() != 7,
                    "legacy diagnostic RMS refuses the named context graph"
                );
                model.residual_rms_forward(ids, script, shape, flags, mask)
            }
            Self::ContextRmsV2 => model.context96_rms_forward(ids, script, shape, flags, mask),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Descriptor {
    name: String,
    file: String,
    sha256: String,
}

pub(crate) fn validate_request(
    enabled: bool,
    frozen: bool,
    ndarray: bool,
    steps: usize,
    penalty: f64,
    initial: Option<&str>,
) -> anyhow::Result<ForwardOperator> {
    ensure!(
        !enabled
            || (frozen
                && ndarray
                && steps == 2000
                && penalty == 0.
                && initial == Some(crate::memorization::exposure::INITIAL_SHA)),
        "residual RMS requires the fixed whole15 CPU native-CE diagnostic"
    );
    Ok(if enabled {
        ForwardOperator::ResidualRmsV1
    } else {
        ForwardOperator::Standard
    })
}

pub(crate) fn attach(run: &Path, marker: &mut Value, selection: &mut Value) -> anyhow::Result<()> {
    std::fs::write(run.join(FILE), CANONICAL)?;
    marker["diagnostic_variant"] = json!(NAME);
    marker["forward_operator"] =
        json!({"name": NAME, "file": FILE, "sha256": crate::export::sha256_hex(CANONICAL)});
    selection["forward_operator"] = json!(NAME);
    Ok(())
}

fn read_binding(run: &Path) -> anyhow::Result<ForwardOperator> {
    let marker_path = run.join("diagnostic.json");
    let selection_path = run.join("selection.json");
    let operator_path = run.join(FILE);
    if !marker_path.try_exists()? {
        ensure!(
            !operator_path.try_exists()?,
            "operator file requires its diagnostic marker"
        );
        if selection_path.try_exists()? {
            let selection: Value = serde_json::from_slice(&std::fs::read(selection_path)?)?;
            ensure!(
                selection.get("forward_operator").is_none(),
                "operator selection requires its diagnostic marker"
            );
        }
        return Ok(ForwardOperator::Standard);
    }
    let marker: Value = serde_json::from_slice(&std::fs::read(marker_path)?)?;
    let selection: Option<Value> = if selection_path.try_exists()? {
        Some(serde_json::from_slice(&std::fs::read(selection_path)?)?)
    } else {
        None
    };
    let variant = marker.get("diagnostic_variant");
    if variant == Some(&json!(CONTEXT_NAME)) {
        let descriptor: Descriptor = serde_json::from_value(marker["forward_operator"].clone())?;
        ensure!(
            descriptor.name == CONTEXT_NAME
                && descriptor.file == CONTEXT_FILE
                && descriptor.sha256 == crate::export::sha256_hex(CONTEXT_CANONICAL),
            "context diagnostic operator descriptor differs"
        );
        ensure!(
            std::fs::read(run.join(CONTEXT_FILE))? == CONTEXT_CANONICAL,
            "noncanonical context diagnostic operator"
        );
        ensure!(
            selection
                .as_ref()
                .context("context diagnostic requires selection")?["forward_operator"]
                == CONTEXT_NAME,
            "context diagnostic selection operator differs"
        );
        let scope = crate::fullmix_rms::diagnostic_scope(&marker)?;
        crate::fullmix_rms::validate_marker_scope(&marker)?;
        ensure!(
            matches!(
                scope,
                crate::fullmix_rms::CONTEXT_SCOPE | crate::fullmix_rms::HARD_SCOPE
            ) && marker["logit_penalty"] == 0.0
                && (scope != crate::fullmix_rms::HARD_SCOPE
                    || marker["objective_identity_sha256"]
                        == crate::fullmix_hard::objective_identity()?),
            "context operator is outside its bounded fit scope"
        );
        ensure!(
            !operator_path.try_exists()?,
            "context operator cannot reuse legacy diagnostic operator file"
        );
        return Ok(ForwardOperator::ContextRmsV2);
    }
    ensure!(
        variant.is_none_or(|value| value == NAME),
        "unknown diagnostic variant"
    );
    let Some(descriptor) = marker.get("forward_operator") else {
        ensure!(
            variant.is_none() && marker.get("diagnostic_scope").is_none(),
            "RMS variant is missing its operator descriptor"
        );
        ensure!(
            !operator_path.try_exists()?
                && selection
                    .as_ref()
                    .is_none_or(|value| value.get("forward_operator").is_none()),
            "missing diagnostic forward operator descriptor"
        );
        return Ok(ForwardOperator::Standard);
    };
    ensure!(
        variant == Some(&json!(NAME)),
        "operator requires its persistent diagnostic variant"
    );
    let descriptor: Descriptor = serde_json::from_value(descriptor.clone())?;
    ensure!(
        descriptor.name == NAME
            && descriptor.file == FILE
            && descriptor.sha256 == crate::export::sha256_hex(CANONICAL),
        "unknown or mismatched diagnostic forward operator"
    );
    ensure!(
        std::fs::read(operator_path)? == CANONICAL,
        "noncanonical diagnostic forward operator file"
    );
    ensure!(
        selection.context("operator requires frozen selection")?["forward_operator"] == NAME,
        "operator selection differs"
    );
    ensure!(
        marker
            .get("diagnostic_scope")
            .is_none_or(|scope| scope == crate::fullmix_rms::SCOPE),
        "unknown diagnostic operator scope"
    );
    let fullmix = crate::fullmix_rms::is_fullmix(&marker);
    ensure!(
        (fullmix
            && marker["scope"] == crate::fullmix_rms::MARKER_SCOPE
            && marker["max_steps"] == crate::fullmix_rms::STEPS
            && marker["logit_penalty"] == 0.0
            && marker["release_quality_claim"] == false)
            || (!fullmix
                && marker["frozen_cohort_scope"] == crate::memorization::FROZEN_SCOPE
                && marker["max_steps"] == 2000
                && marker["logit_penalty"] == 0.0
                && marker["scope"] == "fixed training-only memorization diagnostic; never export"
                && marker["release_quality_claim"] == false),
        "operator marker is outside a verified diagnostic scope"
    );
    Ok(ForwardOperator::ResidualRmsV1)
}

pub(crate) const CONTEXT_NAME: &str = tessera::internal::CONTEXT96_RMS_NAME;
pub(crate) const CONTEXT_FILE: &str = "context-operator.json";
pub(crate) const CONTEXT_CANONICAL: &[u8] = tessera::internal::CONTEXT96_RMS_CONTRACT.as_bytes();

/// The deployment operator selected by the already validated named configuration.
pub(crate) fn deployable(cfg: &Config) -> ForwardOperator {
    if cfg.context96_rms() {
        ForwardOperator::ContextRmsV2
    } else {
        ForwardOperator::Standard
    }
}

/// Records only the explicit deployment contract, independently of diagnostic markers.
pub(crate) fn bind_deployable(run: &Path, cfg: &Config) -> anyhow::Result<()> {
    let path = run.join(CONTEXT_FILE);
    if cfg.context96_rms() {
        std::fs::write(path, tessera::internal::CONTEXT96_RMS_CONTRACT)?;
    } else {
        ensure!(
            !path.try_exists()?,
            "legacy config cannot inherit a context operator binding"
        );
    }
    Ok(())
}

fn verify_deployable(
    run: &Path,
    cfg: &Config,
    old: ForwardOperator,
) -> anyhow::Result<ForwardOperator> {
    let path = run.join(CONTEXT_FILE);
    if cfg.context96_rms() {
        ensure!(
            matches!(
                old,
                ForwardOperator::Standard | ForwardOperator::ContextRmsV2
            ),
            "context graph cannot reuse a frozen diagnostic operator"
        );
        ensure!(
            std::fs::read(path)? == tessera::internal::CONTEXT96_RMS_CONTRACT.as_bytes(),
            "context operator binding differs"
        );
        Ok(ForwardOperator::ContextRmsV2)
    } else {
        ensure!(
            !path.try_exists()?,
            "context operator requires its named graph config"
        );
        Ok(old)
    }
}

/// Attaches the exact operator requested by one of the separately versioned bounded scopes.
pub(crate) fn attach_for_operator(
    run: &Path,
    marker: &mut Value,
    selection: &mut Value,
    operator: ForwardOperator,
) -> anyhow::Result<()> {
    match operator {
        ForwardOperator::ResidualRmsV1 => attach(run, marker, selection),
        ForwardOperator::ContextRmsV2 => {
            std::fs::write(run.join(CONTEXT_FILE), CONTEXT_CANONICAL)?;
            marker["diagnostic_variant"] = json!(CONTEXT_NAME);
            marker["forward_operator"] = json!({"name":CONTEXT_NAME,"file":CONTEXT_FILE,"sha256":crate::export::sha256_hex(CONTEXT_CANONICAL)});
            selection["forward_operator"] = json!(CONTEXT_NAME);
            Ok(())
        }
        ForwardOperator::Standard => {
            anyhow::bail!("bounded RMS scope requires its explicit operator")
        }
    }
}

/// Verify the binding and native source selection once before choosing a forward route.
pub(crate) fn load(run: &Path, cfg: &Config) -> anyhow::Result<ForwardOperator> {
    let operator = read_binding(run)?;
    if matches!(
        operator,
        ForwardOperator::ResidualRmsV1 | ForwardOperator::ContextRmsV2
    ) && run.join("diagnostic.json").try_exists()?
    {
        let marker = serde_json::from_slice(&std::fs::read(run.join("diagnostic.json"))?)?;
        let selection = serde_json::from_slice(&std::fs::read(run.join("selection.json"))?)?;
        if crate::fullmix_rms::is_fullmix(&marker) {
            crate::fullmix_rms::verify(run, cfg, &marker, &selection)?;
        } else {
            crate::memorization::verify_frozen_selection(cfg, &marker, &selection)?;
        }
    }
    verify_deployable(run, cfg, operator)
}

/// Ensure a checkpoint loader cannot silently discard the selected forward semantics.
pub(crate) fn verify_binding(
    run: &Path,
    cfg: &Config,
    expected: ForwardOperator,
) -> anyhow::Result<()> {
    ensure!(
        verify_deployable(run, cfg, read_binding(run)?)? == expected,
        "checkpoint loader operator differs"
    );
    Ok(())
}

/// Ordinary checkpoint entry points must not silently score a different forward function.
pub(crate) fn require_standard(run: &Path, cfg: &Config) -> anyhow::Result<()> {
    ensure!(
        load(run, cfg)? == ForwardOperator::Standard,
        "ordinary checkpoint path refuses a residual RMS diagnostic"
    );
    Ok(())
}

/// Shipping paths accept only the explicitly bound legacy or context deployment operator.
pub(crate) fn require_deployable(run: &Path, cfg: &Config) -> anyhow::Result<ForwardOperator> {
    let operator = load(run, cfg)?;
    ensure!(
        operator == deployable(cfg),
        "deployment path refuses a residual RMS diagnostic"
    );
    Ok(operator)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(run: &Path) {
        let mut marker = json!({"scope":"fixed training-only memorization diagnostic; never export", "release_quality_claim":false,"frozen_cohort_scope":crate::memorization::FROZEN_SCOPE,"max_steps":2000,"logit_penalty":0.0});
        let mut selection = json!({});
        attach(run, &mut marker, &mut selection).unwrap();
        std::fs::write(
            run.join("diagnostic.json"),
            serde_json::to_vec(&marker).unwrap(),
        )
        .unwrap();
        std::fs::write(
            run.join("selection.json"),
            serde_json::to_vec(&selection).unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn fullmix_scope_is_separate_and_missing_descriptors_cannot_select_standard() {
        let dir = tempfile::tempdir().unwrap();
        let mut marker = json!({"scope":crate::fullmix_rms::MARKER_SCOPE,"diagnostic_scope":crate::fullmix_rms::SCOPE,
            "max_steps":4000,"logit_penalty":0.0,"release_quality_claim":false,"preflight":true});
        let mut selection = json!({});
        attach(dir.path(), &mut marker, &mut selection).unwrap();
        let save = |marker: &Value| {
            std::fs::write(
                dir.path().join("diagnostic.json"),
                serde_json::to_vec(marker).unwrap(),
            )
            .unwrap()
        };
        save(&marker);
        std::fs::write(
            dir.path().join("selection.json"),
            serde_json::to_vec(&selection).unwrap(),
        )
        .unwrap();
        assert_eq!(
            read_binding(dir.path()).unwrap(),
            ForwardOperator::ResidualRmsV1
        );
        marker["max_steps"] = json!(2000);
        save(&marker);
        assert!(read_binding(dir.path()).is_err());
        marker["max_steps"] = json!(4000);
        marker["diagnostic_scope"] = json!("unknown-fullmix");
        save(&marker);
        assert!(read_binding(dir.path()).is_err());
        marker["diagnostic_scope"] = json!(crate::fullmix_rms::SCOPE);
        marker.as_object_mut().unwrap().remove("diagnostic_variant");
        marker.as_object_mut().unwrap().remove("forward_operator");
        save(&marker);
        std::fs::remove_file(dir.path().join(FILE)).unwrap();
        selection
            .as_object_mut()
            .unwrap()
            .remove("forward_operator");
        std::fs::write(
            dir.path().join("selection.json"),
            serde_json::to_vec(&selection).unwrap(),
        )
        .unwrap();
        assert!(read_binding(dir.path()).is_err());
    }

    #[test]
    fn missing_unknown_or_noncanonical_operator_cannot_select_standard_forward() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read_binding(dir.path()).unwrap(), ForwardOperator::Standard);
        binding(dir.path());
        assert_eq!(
            read_binding(dir.path()).unwrap(),
            ForwardOperator::ResidualRmsV1
        );
        for bad in [
            CANONICAL
                .to_vec()
                .into_iter()
                .filter(|b| *b != b'\n')
                .collect::<Vec<_>>(),
            b"{\"epsilon\":0.001}".to_vec(),
        ] {
            std::fs::write(dir.path().join(FILE), bad).unwrap();
            assert!(read_binding(dir.path()).is_err());
        }
        binding(dir.path());
        std::fs::remove_file(dir.path().join(FILE)).unwrap();
        assert!(read_binding(dir.path()).is_err());
        for field in ["name", "file", "sha256"] {
            binding(dir.path());
            let path = dir.path().join("diagnostic.json");
            let mut marker: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
            marker["forward_operator"][field] = json!("wrong");
            std::fs::write(path, serde_json::to_vec(&marker).unwrap()).unwrap();
            assert!(read_binding(dir.path()).is_err());
        }
        binding(dir.path());
        std::fs::write(dir.path().join("diagnostic.json"), b"{}").unwrap();
        assert!(read_binding(dir.path()).is_err());
        binding(dir.path());
        std::fs::remove_file(dir.path().join("diagnostic.json")).unwrap();
        assert!(read_binding(dir.path()).is_err());
    }

    #[test]
    fn candidate_request_refuses_default_scope_wrong_backend_budget_loss_or_initialization() {
        let initial = Some(crate::memorization::exposure::INITIAL_SHA);
        assert_eq!(
            validate_request(true, true, true, 2000, 0., initial).unwrap(),
            ForwardOperator::ResidualRmsV1
        );
        for (frozen, ndarray, steps, penalty, sha) in [
            (false, true, 2000, 0., initial),
            (true, false, 2000, 0., initial),
            (true, true, 1995, 0., initial),
            (true, true, 2000, 1e-4, initial),
            (true, true, 2000, 0., None),
        ] {
            assert!(validate_request(true, frozen, ndarray, steps, penalty, sha).is_err());
        }
    }

    #[test]
    fn candidate_record_roundtrip_preserves_weights_and_verified_forward_identity() {
        use burn::backend::NdArray;
        use burn::module::Module;
        use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
        let dir = tempfile::tempdir().unwrap();
        binding(dir.path());
        let operator = read_binding(dir.path()).unwrap();
        let device = Default::default();
        let cfg = crate::net::TaggerNetConfig::new(16, vec![1, 2], 7)
            .with_ngram_dim(3)
            .with_script_dim(2)
            .with_shape_dim(2)
            .with_hidden(4)
            .with_dropout(0.);
        let model = cfg.init::<NdArray>(&device);
        let tensors = crate::quantize::extract(&model, "detector");
        let count = model.num_params();
        let before = crate::training_diagnostic::parameter_sha256(&tensors);
        model
            .clone()
            .save_file(
                dir.path().join("best"),
                &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
            )
            .unwrap();
        let loaded = cfg
            .init::<NdArray>(&device)
            .load_file(
                dir.path().join("best"),
                &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
                &device,
            )
            .unwrap();
        assert_eq!(count, loaded.num_params());
        assert_eq!(
            before,
            crate::training_diagnostic::parameter_sha256(&crate::quantize::extract(
                &loaded, "detector"
            ))
        );
        let ids = Tensor::<NdArray, 3, Int>::ones([1, 3, 2], &device);
        let script = Tensor::<NdArray, 2, Int>::ones([1, 3], &device);
        let flags = Tensor::<NdArray, 3>::ones([1, 3, crate::dataset::FLAG_BITS], &device);
        let mask = Tensor::<NdArray, 2>::ones([1, 3], &device);
        let original = operator
            .forward(
                &model,
                ids.clone(),
                script.clone(),
                script.clone(),
                flags.clone(),
                mask.clone(),
            )
            .unwrap();
        let reloaded_operator = read_binding(dir.path()).unwrap();
        let restored = reloaded_operator
            .forward(
                &loaded,
                ids.clone(),
                script.clone(),
                script.clone(),
                flags.clone(),
                mask.clone(),
            )
            .unwrap();
        assert_eq!(original.clone().into_data(), restored.into_data());
        let ordinary = ForwardOperator::Standard
            .forward(&loaded, ids, script.clone(), script, flags, mask)
            .unwrap();
        assert!((original - ordinary).abs().max().into_scalar() > 1e-5);
        std::fs::remove_file(dir.path().join(FILE)).unwrap();
        assert!(read_binding(dir.path()).is_err());
    }

    #[test]
    fn persistent_variant_refuses_incomplete_operator_copy() {
        let dir = tempfile::tempdir().unwrap();
        binding(dir.path());
        let marker_path = dir.path().join("diagnostic.json");
        let selection_path = dir.path().join("selection.json");
        let mut marker: Value =
            serde_json::from_slice(&std::fs::read(&marker_path).unwrap()).unwrap();
        marker.as_object_mut().unwrap().remove("forward_operator");
        std::fs::write(&marker_path, serde_json::to_vec(&marker).unwrap()).unwrap();
        std::fs::write(&selection_path, b"{}").unwrap();
        std::fs::remove_file(dir.path().join(FILE)).unwrap();
        assert!(read_binding(dir.path()).is_err());
        binding(dir.path());
        let mut marker: Value =
            serde_json::from_slice(&std::fs::read(&marker_path).unwrap()).unwrap();
        marker["diagnostic_variant"] = json!("unknown");
        std::fs::write(&marker_path, serde_json::to_vec(&marker).unwrap()).unwrap();
        assert!(read_binding(dir.path()).is_err());
    }
    #[test]
    fn context96_operator_binding_is_separate_and_rejects_relabeling() {
        let mut cfg = crate::config::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/detector-shared.toml"),
        )
        .unwrap();
        let run = tempfile::tempdir().unwrap();
        assert_eq!(load(run.path(), &cfg).unwrap(), ForwardOperator::Standard);
        cfg.net.architecture = Some(CONTEXT_NAME.into());
        cfg.net.dilations = tessera::internal::CONTEXT96_RMS_DILATIONS.to_vec();
        cfg.validate().unwrap();
        assert!(load(run.path(), &cfg).is_err());
        bind_deployable(run.path(), &cfg).unwrap();
        assert_eq!(
            require_deployable(run.path(), &cfg).unwrap(),
            ForwardOperator::ContextRmsV2
        );
        assert!(require_standard(run.path(), &cfg).is_err());
        assert!(verify_binding(run.path(), &cfg, ForwardOperator::Standard).is_err());
        verify_binding(run.path(), &cfg, ForwardOperator::ContextRmsV2).unwrap();
        std::fs::write(run.path().join(CONTEXT_FILE), CANONICAL).unwrap();
        assert!(load(run.path(), &cfg).is_err());
        bind_deployable(run.path(), &cfg).unwrap();
        assert!(verify_deployable(run.path(), &cfg, ForwardOperator::ResidualRmsV1).is_err());
        cfg.net.architecture = None;
        cfg.net.dilations = vec![1, 2, 4, 8, 16, 1];
        assert!(load(run.path(), &cfg).is_err());
        assert_eq!(NAME, "residual-rms-v1");
        assert!(
            std::str::from_utf8(CANONICAL)
                .unwrap()
                .contains("\"blocks\":6")
        );
    }
    #[test]
    fn context96_decoder_selection_is_automatic_only_for_its_named_operator() {
        for legacy in [ForwardOperator::Standard, ForwardOperator::ResidualRmsV1] {
            assert_eq!(legacy.decoder(None), None);
            assert_eq!(
                legacy.decoder(Some(
                    crate::diagnostic_decode::Decoder::AddressContinuationV1
                )),
                Some(crate::diagnostic_decode::Decoder::AddressContinuationV1)
            );
        }
        assert_eq!(
            ForwardOperator::ContextRmsV2.decoder(None),
            Some(crate::diagnostic_decode::Decoder::AddressContinuationV1)
        );
    }
    fn context_binding(run: &Path, scope: &str, marker_scope: &str) -> Value {
        let mut marker = json!({"scope":marker_scope,"diagnostic_scope":scope,
            "max_steps":4000,"logit_penalty":0.0,"release_quality_claim":false,"preflight":true});
        if scope == crate::fullmix_rms::HARD_SCOPE {
            marker["objective_identity_sha256"] =
                json!(crate::fullmix_hard::objective_identity().unwrap());
        }
        let mut selection = json!({});
        attach_for_operator(
            run,
            &mut marker,
            &mut selection,
            ForwardOperator::ContextRmsV2,
        )
        .unwrap();
        std::fs::write(
            run.join("diagnostic.json"),
            serde_json::to_vec(&marker).unwrap(),
        )
        .unwrap();
        std::fs::write(
            run.join("selection.json"),
            serde_json::to_vec(&selection).unwrap(),
        )
        .unwrap();
        marker
    }

    #[test]
    fn bounded_context_operator_accepts_only_exact_old_and_hard_scope_pairs() {
        let mut cfg = crate::config::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/detector-shared.toml"),
        )
        .unwrap();
        cfg.net.architecture = Some(CONTEXT_NAME.into());
        cfg.net.dilations = tessera::internal::CONTEXT96_RMS_DILATIONS.to_vec();
        cfg.validate().unwrap();
        for (scope, marker_scope) in [
            (
                crate::fullmix_rms::CONTEXT_SCOPE,
                crate::fullmix_rms::CONTEXT_MARKER_SCOPE,
            ),
            (
                crate::fullmix_rms::HARD_SCOPE,
                crate::fullmix_rms::HARD_MARKER_SCOPE,
            ),
        ] {
            let run = tempfile::tempdir().unwrap();
            let marker = context_binding(run.path(), scope, marker_scope);
            assert_eq!(
                read_binding(run.path()).unwrap(),
                ForwardOperator::ContextRmsV2
            );
            verify_binding(run.path(), &cfg, ForwardOperator::ContextRmsV2).unwrap();
            assert_eq!(
                std::fs::read(run.path().join(CONTEXT_FILE)).unwrap(),
                CONTEXT_CANONICAL
            );
            for (key, value) in [
                (
                    "scope",
                    json!("fixed training-only memorization diagnostic; never export"),
                ),
                (
                    "scope",
                    json!(if scope == crate::fullmix_rms::HARD_SCOPE {
                        crate::fullmix_rms::CONTEXT_MARKER_SCOPE
                    } else {
                        crate::fullmix_rms::HARD_MARKER_SCOPE
                    }),
                ),
                ("diagnostic_scope", json!(crate::fullmix_rms::SCOPE)),
                ("diagnostic_scope", json!("unknown")),
                ("max_steps", json!(3999)),
                ("logit_penalty", json!(0.001)),
                ("release_quality_claim", json!(true)),
            ] {
                let mut changed = marker.clone();
                changed[key] = value;
                std::fs::write(
                    run.path().join("diagnostic.json"),
                    serde_json::to_vec(&changed).unwrap(),
                )
                .unwrap();
                assert!(read_binding(run.path()).is_err(), "{scope} {key}");
            }
            std::fs::write(
                run.path().join("diagnostic.json"),
                serde_json::to_vec(&marker).unwrap(),
            )
            .unwrap();
            if scope == crate::fullmix_rms::HARD_SCOPE {
                for identity in [Value::Null, json!("a".repeat(64))] {
                    let mut changed = marker.clone();
                    changed["objective_identity_sha256"] = identity;
                    std::fs::write(
                        run.path().join("diagnostic.json"),
                        serde_json::to_vec(&changed).unwrap(),
                    )
                    .unwrap();
                    assert!(read_binding(run.path()).is_err());
                }
            }
            context_binding(run.path(), scope, marker_scope);
            std::fs::write(run.path().join(FILE), CANONICAL).unwrap();
            assert!(read_binding(run.path()).is_err());
        }
    }
}
