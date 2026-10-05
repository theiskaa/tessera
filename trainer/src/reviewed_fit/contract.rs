//! Fixed reviewed-native recipe and metadata-only readiness for an isolated candidate fit.

use super::{mixed, objective, probe, sampling, supervision};
use crate::reviewed_data::{Receipt, digest};
use crate::reviewed_train::{Loaded, Prepared};
use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::path::{Path, PathBuf};

/// No DEV selector, legacy sources, source overrides or optimizer resume fields exist.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Manifest {
    pub(super) scope: String,
    pub(super) config: Receipt,
    pub(super) canonical_preflight: Receipt,
    pub(super) updates: usize,
    pub(super) snapshots: Vec<usize>,
    pub(super) sampling: sampling::Sampling,
    pub(super) members: Vec<sampling::Member>,
    pub(super) probe: probe::Request,
    #[serde(default, skip_serializing_if = "objective::Objective::is_legacy")]
    pub(super) loss_objective: objective::Objective,
    /// Pinned reviewed source-ambiguity sidecar, paired only with its own objective.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) reviewed_ambiguity_exclusions: Option<Receipt>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) mixed_inputs: Option<mixed::Settings>,
}

impl Manifest {
    fn validate(&self, batch_size: usize, warmup: usize) -> anyhow::Result<()> {
        ensure!(
            self.scope
                == (if self.mixed_inputs.is_some() {
                    super::MIXED_SCOPE
                } else {
                    super::SCOPE
                })
                && self.updates > 0
                && batch_size > 0
                && batch_size <= 32
                && warmup < self.updates,
            "declare finite updates, bounded batch size and warmup shorter than the run"
        );
        ensure!(
            self.snapshots.first() == Some(&0)
                && self.snapshots.last() == Some(&self.updates)
                && self.snapshots.windows(2).all(|p| p[0] < p[1]),
            "snapshots must be ordered and include baseline and fixed final update"
        );
        ensure!(
            self.loss_objective.excludes_reviewed_ambiguity()
                == self.reviewed_ambiguity_exclusions.is_some(),
            "reviewed ambiguity sidecar and its loss objective must be declared together"
        );
        self.updates
            .checked_mul(batch_size)
            .context("fixed batch budget overflow")?;
        Ok(())
    }
}

fn validate_postprocess(
    cfg: &crate::config::Config,
    mixed: Option<&mixed::Settings>,
) -> anyhow::Result<()> {
    match mixed {
        Some(settings) => ensure!(
            cfg.net.detector_postprocess == Some(settings.postprocess),
            "mixed config must explicitly declare the manifest postprocessor"
        ),
        None => ensure!(
            cfg.net.detector_postprocess
                != Some(crate::config::DetectorPostprocess::AddressLabeledFieldsV1),
            "native-only reviewed fit does not activate labeled-field postprocessing; use the explicitly paired mixed route"
        ),
    }
    Ok(())
}

/// Validated canonical data, fixed draws and TRAIN probe prepared without a model.
pub(super) struct Fit {
    pub(super) manifest: Manifest,
    pub(super) native: Prepared,
    pub(super) loaded: Loaded,
    pub(super) plan: sampling::Plan,
    pub(super) probe: probe::Probe,
    pub(super) identity: Value,
    pub(super) manifest_receipt: Receipt,
    pub(super) execution_preflight: Option<Receipt>,
    pub(super) mixed: Option<mixed::LoadedMixed>,
    pub(super) exclusions: Option<supervision::Resolved>,
}

/// Resolve the paired sidecar and require it to equal the TRAIN review's declared set exactly;
/// without the reviewed-ambiguity objective, any declared exclusion refuses the fit.
fn resolve_exclusions(
    manifest: &Manifest,
    train: &crate::reviewed_data::Cohort,
) -> anyhow::Result<Option<supervision::Resolved>> {
    let Some(receipt) = &manifest.reviewed_ambiguity_exclusions else {
        train
            .refuse_declared_exclusions("reviewed fit without the reviewed-ambiguity objective")?;
        return Ok(None);
    };
    let resolved = supervision::load(receipt)?.resolve(receipt, &supervision::cases(train)?)?;
    resolved.require_declared(train.declared_exclusions())?;
    Ok(Some(resolved))
}

/// Authored variants of an excluded case would supervise the same unresolved text as O.
pub(super) fn require_unexcluded_parents(
    parents: &[&str],
    resolved: &supervision::Resolved,
) -> anyhow::Result<()> {
    for parent in parents {
        ensure!(
            !resolved.excludes_case(parent),
            "authored TRAIN variant of {parent} would supervise its reviewed-ambiguity spans"
        );
    }
    Ok(())
}

/// Native TRAIN owner of each drawn row; authored rows have none.
pub(super) fn owners(
    mixed: Option<&mixed::LoadedMixed>,
    native_rows: usize,
    indices: &[usize],
) -> anyhow::Result<Vec<Option<usize>>> {
    match mixed {
        Some(data) => data.native_owners(indices),
        None => indices
            .iter()
            .map(|&i| {
                ensure!(i < native_rows, "native draw missing");
                Ok(Some(i))
            })
            .collect(),
    }
}

/// The objective's loss mask for drawn TRAIN rows; planned, executed and replayed batches share it.
pub(super) fn batch_mask(
    objective: objective::Objective,
    exclusions: Option<&supervision::Resolved>,
    owners: &[Option<usize>],
    items: &[crate::dataset::Encoded],
    weights: &[f32],
) -> anyhow::Result<Option<objective::LossMask>> {
    if objective.is_legacy() {
        return Ok(None);
    }
    ensure!(
        objective.excludes_reviewed_ambiguity() == exclusions.is_some(),
        "reviewed ambiguity exclusions differ from the declared objective"
    );
    let ambiguity = exclusions
        .map(|resolved| resolved.batch(owners, items))
        .transpose()?;
    objective::batch_loss_mask(items, ambiguity.as_deref(), weights).map(Some)
}

impl Fit {
    /// Resolve all native closures before permitting model initialization.
    pub(super) fn prepare(path: &Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)?;
        let manifest: Manifest = serde_json::from_slice(&bytes)?;
        manifest.config.bytes()?;
        let native = Prepared::load(&manifest.config.path, manifest.canonical_preflight.clone())?;
        ensure!(
            native.input_policy() == crate::detector::DetectorInputPolicy::KnownUs,
            "this US fixed-fit scope requires canonical KnownUs inputs matching the declared product default"
        );
        let cfg = native.config();
        validate_postprocess(cfg, manifest.mixed_inputs.as_ref())?;
        manifest.validate(cfg.train.batch_size, cfg.train.warmup_steps)?;
        let loaded = native.load_documents()?;
        let native_identity = loaded.cohorts().identity()?;
        let exclusions = resolve_exclusions(&manifest, &loaded.cohorts().train)?;
        let mixed = manifest
            .mixed_inputs
            .as_ref()
            .map(|settings| {
                ensure!(
                    !manifest.loss_objective.is_legacy(),
                    "mixed route requires explicit rule-excluded loss"
                );
                mixed::load(settings, &native, &loaded)
            })
            .transpose()?;
        if let (Some(data), Some(resolved)) = (&mixed, &exclusions) {
            require_unexcluded_parents(&data.authored_parents()?, resolved)?;
        }
        let plan = match (&mixed, &manifest.mixed_inputs) {
            (Some(data), Some(settings)) => {
                ensure!(
                    matches!(
                        &manifest.sampling,
                        sampling::Sampling::CompletePassStratified
                    ),
                    "mixed route requires complete-pass draws"
                );
                mixed::plan(
                    data,
                    &loaded.cohorts().train,
                    settings,
                    &manifest.members,
                    &native_identity,
                    mixed::Loss {
                        reference_weights: &cfg
                            .detector
                            .as_ref()
                            .context("weights missing")?
                            .class_weights,
                        exclusions: exclusions.as_ref(),
                    },
                    sampling::Budget {
                        seed: cfg.seed,
                        updates: manifest.updates,
                        size: cfg.train.batch_size,
                    },
                )?
            }
            _ => sampling::prepare(
                &manifest.sampling,
                &manifest.members,
                &loaded.cohorts().train,
                &native_identity,
                &manifest.canonical_preflight.sha256,
                sampling::Budget {
                    seed: cfg.seed,
                    updates: manifest.updates,
                    size: cfg.train.batch_size,
                },
            )?,
        };
        let probe = probe::prepare(&manifest.probe, &loaded.cohorts().train, &native_identity)?;
        let manifest_receipt = Receipt {
            path: path.canonicalize()?,
            sha256: digest(&bytes),
        };
        let mut identity = json!({"scope":manifest.scope,"manifest":manifest_receipt,
            "native":native.identity(&loaded)?,"sampling":plan.identity,"probe":probe.identity,
            "fixed_updates":manifest.updates,"snapshots":manifest.snapshots,
            "selection":"fixed final optimizer update count; no DEV checkpoint choice or early stopping",
            "thread_bounds":{"RAYON_NUM_THREADS":2,"MATMUL_NUM_THREADS":1},
            "schedule":"lr_at(step,fixed_updates,effective learning_rate,effective warmup_steps)",
            "loss":"unchanged native class-weighted pooled-token CE; only padding masked",
            "development_objective":"each of PERSON/ORG/ADDRESS/EMAIL/PHONE exact filtered precision >= .95 and recall >= .95; diagnostic objective only, never release approval",
            "export_allowed":false,"general_accuracy_claim":false,"release_quality_claim":false});
        if let Some(data) = &mixed {
            identity["mixed_training"] = data.record.clone();
            let settings = manifest
                .mixed_inputs
                .as_ref()
                .context("mixed settings missing")?;
            identity["explicit_loss_weights"] = serde_json::to_value(&settings.loss_weights)?;
            identity["mixed_token_accounting"] = mixed::accounting(
                data,
                &loaded.cohorts().train,
                &plan,
                settings.loss_weights.as_deref().unwrap_or(
                    &cfg.detector
                        .as_ref()
                        .context("weights missing")?
                        .class_weights,
                ),
                settings.loss_weights.is_some(),
                exclusions.as_ref(),
            )?;
        }
        if !manifest.loss_objective.is_legacy() {
            let weights = manifest
                .mixed_inputs
                .as_ref()
                .and_then(|m| m.loss_weights.as_deref())
                .unwrap_or(
                    &cfg.detector
                        .as_ref()
                        .context("class weights missing")?
                        .class_weights,
                );
            let train = &loaded.cohorts().train;
            let mut batches = Vec::new();
            for indices in &plan.batches {
                let items: Vec<_> = match &mixed {
                    Some(data) => data.items(train, indices)?,
                    None => indices
                        .iter()
                        .map(|&i| train.docs.get(i).map(|d| d.enc.clone()))
                        .collect::<Option<_>>()
                        .context("native draw missing")?,
                };
                let mask = batch_mask(
                    manifest.loss_objective,
                    exclusions.as_ref(),
                    &owners(mixed.as_ref(), train.docs.len(), indices)?,
                    &items,
                    weights,
                )?
                .context("non-legacy objective produced no loss mask")?;
                batches.push(mask.report);
            }
            match &exclusions {
                None => {
                    identity["loss"] = json!(
                        "native weighted CE with canonical IN_RULE_SPAN tokens excluded only from loss; unchanged forward padding/context mask"
                    );
                    identity["loss_objective"] = json!({"version":manifest.loss_objective,
                        "selection":"canonical flags alone; positive neural label on rule position rejected",
                        "class_weights":weights,"planned_batches":batches,
                        "weighted_denominator_is_not_gradient_share":true,
                        "native_normalization":"unchanged masked_loss denominator clamp_min(1.0)"});
                }
                Some(resolved) => {
                    identity["loss"] = json!(
                        "native weighted CE with canonical IN_RULE_SPAN tokens and reviewed source-ambiguity O tokens excluded only from loss; unchanged forward inputs and padding/context mask"
                    );
                    identity["loss_objective"] = json!({"version":manifest.loss_objective,
                        "selection":"canonical flags plus pinned reviewed sidecar resolved on native TRAIN; never model scores or DEV",
                        "reviewed_ambiguity_exclusions":resolved.identity,
                        "class_weights":weights,"planned_batches":batches,
                        "weighted_denominator_is_not_gradient_share":true,
                        "native_normalization":"unchanged masked_loss denominator clamp_min(1.0)"});
                }
            }
        }
        let fit = Self {
            manifest,
            native,
            loaded,
            plan,
            probe,
            identity,
            manifest_receipt,
            execution_preflight: None,
            mixed,
            exclusions,
        };
        fit.verify()?;
        Ok(fit)
    }

    /// Recheck immutable authorities and the prepared canonical identity.
    pub(super) fn verify(&self) -> anyhow::Result<()> {
        self.manifest_receipt.bytes()?;
        self.manifest.config.bytes()?;
        self.manifest.canonical_preflight.bytes()?;
        if let Some(receipt) = &self.plan.receipt {
            receipt.bytes()?;
        }
        if let Some(receipt) = &self.execution_preflight {
            receipt.bytes()?;
        }
        self.native.verify_inputs()?;
        match (
            resolve_exclusions(&self.manifest, &self.loaded.cohorts().train)?,
            &self.exclusions,
        ) {
            (None, None) => {}
            (Some(again), Some(resolved)) => {
                resolved.verify()?;
                ensure!(
                    again.identity == resolved.identity
                        && self.identity["loss_objective"]["reviewed_ambiguity_exclusions"]
                            == again.identity,
                    "reviewed ambiguity resolution changed"
                );
            }
            _ => anyhow::bail!("reviewed ambiguity resolution changed"),
        }
        if let Some(mixed) = &self.mixed {
            mixed.verify()?;
            ensure!(
                self.identity["mixed_training"] == mixed.record,
                "mixed identity changed"
            );
            if let Some(receipt) = self
                .manifest
                .mixed_inputs
                .as_ref()
                .and_then(|m| m.canonical_preflight.as_ref())
            {
                ensure!(
                    serde_json::from_slice::<Value>(&receipt.bytes()?)? == mixed.record,
                    "mixed canonical receipt changed"
                );
            }
        }
        ensure!(
            self.identity["native"] == self.native.identity(&self.loaded)?,
            "native fit identity changed"
        );
        Ok(())
    }

    /// Collect typed input closure receipts without interpreting arbitrary JSON hashes.
    pub(super) fn receipts(&self) -> anyhow::Result<Vec<Receipt>> {
        self.verify()?;
        let mut receipts = self.native.dependency_receipts()?;
        receipts.extend([
            self.manifest_receipt.clone(),
            self.manifest.config.clone(),
            self.manifest.canonical_preflight.clone(),
        ]);
        receipts.extend(self.plan.receipt.iter().cloned());
        receipts.extend(self.execution_preflight.iter().cloned());
        if let Some(resolved) = &self.exclusions {
            receipts.extend(resolved.dependencies.iter().cloned());
        }
        if let Some(mixed) = &self.mixed {
            receipts.extend(mixed.dependencies.clone());
            receipts.extend(
                self.manifest
                    .mixed_inputs
                    .as_ref()
                    .and_then(|m| m.canonical_preflight.as_ref())
                    .cloned(),
            );
        }
        Ok(receipts)
    }

    pub(super) fn scope(&self) -> &str {
        &self.manifest.scope
    }
    pub(super) fn train_len(&self) -> usize {
        self.mixed.as_ref().map_or(
            self.loaded.cohorts().train.docs.len(),
            mixed::LoadedMixed::len,
        )
    }
    pub(super) fn train_items(
        &self,
        indices: &[usize],
    ) -> anyhow::Result<Vec<crate::dataset::Encoded>> {
        match &self.mixed {
            Some(mixed) => mixed.items(&self.loaded.cohorts().train, indices),
            None => indices
                .iter()
                .map(|&i| {
                    self.loaded
                        .cohorts()
                        .train
                        .docs
                        .get(i)
                        .map(|d| d.enc.clone())
                        .context("native draw missing")
                })
                .collect(),
        }
    }
    /// Loss mask of drawn rows under the declared objective; `None` for the legacy objective.
    pub(super) fn loss_mask(
        &self,
        indices: &[usize],
        items: &[crate::dataset::Encoded],
    ) -> anyhow::Result<Option<objective::LossMask>> {
        let train = &self.loaded.cohorts().train;
        batch_mask(
            self.manifest.loss_objective,
            self.exclusions.as_ref(),
            &owners(self.mixed.as_ref(), train.docs.len(), indices)?,
            items,
            &self.class_weights()?,
        )
    }
    /// Require an executed or replayed batch's loss activity to equal its saved plan.
    pub(super) fn require_planned_activity(
        &self,
        index: usize,
        report: &objective::Report,
    ) -> anyhow::Result<()> {
        let actual = serde_json::to_value(report)?;
        ensure!(
            self.identity["loss_objective"]["planned_batches"].get(index) == Some(&actual)
                && (self.mixed.is_none()
                    || self.plan.identity["token_batches"].get(index) == Some(&actual)),
            "batch {} loss activity differs from its planned record",
            index + 1
        );
        Ok(())
    }
    pub(super) fn class_weights(&self) -> anyhow::Result<Vec<f32>> {
        Ok(self
            .manifest
            .mixed_inputs
            .as_ref()
            .and_then(|m| m.loss_weights.clone())
            .unwrap_or(
                self.native
                    .config()
                    .detector
                    .as_ref()
                    .context("weights missing")?
                    .class_weights
                    .clone(),
            ))
    }
    pub(super) fn require_execution_ready(&self) -> anyhow::Result<()> {
        if let Some(settings) = &self.manifest.mixed_inputs {
            settings.execution_ready()?;
        }
        self.verify()
    }

    /// Reject existing destinations and writes inside the original checkpoint source.
    pub(super) fn destination(&self, out: &Path) -> anyhow::Result<PathBuf> {
        let parent = out
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let target = parent
            .canonicalize()?
            .join(out.file_name().context("fit output name missing")?);
        ensure!(
            !target.starts_with(self.native.source_run().canonicalize()?),
            "fit output inside immutable source"
        );
        ensure!(
            matches!(std::fs::symlink_metadata(&target),Err(e) if e.kind()==std::io::ErrorKind::NotFound),
            "fit output already exists or cannot be inspected"
        );
        Ok(target)
    }

    /// Bind metadata checks independently of any later execution receipt.
    pub(super) fn preflight(&self, snapshot: &Value) -> Value {
        let mut record = json!({"scope":self.scope(),"structural_checks_passed":true,
            "identity":self.identity,"input_snapshot":snapshot,
            "model_initialized":false,"optimizer_initialized":false,"parameter_identity_verified":false,
            "fit_executed":false,"export_allowed":false,"release_quality_claim":false});
        if let Some(settings) = &self.manifest.mixed_inputs {
            record["mixed_execution_inputs_present"] = json!(settings.execution_ready().is_ok());
            record["authored_is_native_or_heldout"] = json!(false);
        }
        record
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn postprocessor_requires_explicit_matching_mixed_authority() {
        use crate::config::DetectorPostprocess::{AddressContinuationV1, AddressLabeledFieldsV1};

        let mut cfg = crate::config::learning_test_config();
        let settings: mixed::Settings = serde_json::from_value(json!({
            "scope":"reviewed-native-authored-train-only-v1",
            "repository_root":"/source",
            "authored":[],
            "native_documents":340,
            "authored_documents":56,
            "dev_documents":100,
            "postprocess":"address_labeled_fields_v1",
            "canonical_preflight":null,
            "loss_weights":null
        }))
        .unwrap();
        assert!(validate_postprocess(&cfg, Some(&settings)).is_err());
        cfg.net.detector_postprocess = Some(AddressContinuationV1);
        assert!(validate_postprocess(&cfg, Some(&settings)).is_err());
        cfg.net.detector_postprocess = Some(AddressLabeledFieldsV1);
        validate_postprocess(&cfg, Some(&settings)).unwrap();
        assert!(validate_postprocess(&cfg, None).is_err());
        cfg.net.detector_postprocess = Some(AddressContinuationV1);
        validate_postprocess(&cfg, None).unwrap();
        cfg.net.detector_postprocess = None;
        validate_postprocess(&cfg, None).unwrap();

        assert_eq!(
            serde_json::to_value(AddressContinuationV1).unwrap(),
            json!("address_continuation_v1")
        );
        for unsupported in ["\"continuation_only_v1\"", "\"unknown\"", "null"] {
            assert!(
                serde_json::from_str::<crate::config::DetectorPostprocess>(unsupported).is_err()
            );
        }
    }

    #[test]
    fn reviewed_ambiguity_sidecar_and_objective_are_strictly_paired() {
        let receipt = json!({"path":"/receipt","sha256":"a".repeat(64)});
        let base = json!({"scope":super::super::SCOPE,"config":receipt,"canonical_preflight":receipt,
            "updates":10,"snapshots":[0,10],"sampling":{"mode":"complete_pass_stratified"},"members":[],
            "probe":{"train_names":["case"],"minimum_recall":0.5}});
        let reviewed = "canonical_rule_and_reviewed_ambiguity_excluded_weighted_ce_v1";
        for (objective, sidecar, valid) in [
            (Some(reviewed), true, true),
            (Some(reviewed), false, false),
            (Some("canonical_rule_excluded_weighted_ce_v1"), true, false),
            (None, true, false),
        ] {
            let mut value = base.clone();
            if let Some(objective) = objective {
                value["loss_objective"] = json!(objective);
            }
            if sidecar {
                value["reviewed_ambiguity_exclusions"] = receipt.clone();
            }
            let manifest: Manifest = serde_json::from_value(value.clone()).unwrap();
            assert_eq!(manifest.validate(32, 0).is_ok(), valid, "{value}");
            assert_eq!(serde_json::to_value(&manifest).unwrap(), value);
        }
        let mut unknown = base;
        unknown["reviewed_ambiguity"] = receipt;
        assert!(serde_json::from_value::<Manifest>(unknown).is_err());
    }

    #[test]
    fn planned_and_replayed_masks_follow_native_owners_in_batch_order() {
        use crate::dataset::Encoded;
        use std::collections::BTreeMap;
        use tessera::internal::flag;
        let item = |flags: Vec<u32>| Encoded {
            token_spans: (0..flags.len() as u32).map(|i| (i, i + 1)).collect(),
            ngram_ids: vec![vec![1]; flags.len()],
            script: vec![0; flags.len()],
            shape: vec![0; flags.len()],
            labels: vec![0; flags.len()],
            flags,
            country: "US".into(),
        };
        let native = [
            item(vec![0, 0]),
            item(vec![0]),
            item(vec![flag::IN_RULE_SPAN, 0, 0]),
        ];
        let resolved = supervision::Resolved::from_rows(BTreeMap::from([(0, vec![true, false])]));
        let reviewed = objective::Objective::CanonicalRuleAndReviewedAmbiguityExcludedV1;
        let indices = [2, 0, 1];
        let items: Vec<_> = indices.iter().map(|&i| native[i].clone()).collect();
        let drawn = owners(None, native.len(), &indices).unwrap();
        assert_eq!(drawn, vec![Some(2), Some(0), Some(1)]);
        let mask = batch_mask(reviewed, Some(&resolved), &drawn, &items, &[1.0; 7])
            .unwrap()
            .unwrap();
        assert_eq!(
            mask.values,
            vec![0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        );
        assert_eq!(mask.report.excluded_reviewed_ambiguity_o_tokens, Some(1));
        let replayed = batch_mask(reviewed, Some(&resolved), &drawn, &items, &[1.0; 7])
            .unwrap()
            .unwrap();
        assert_eq!(replayed.values, mask.values);
        assert_eq!(
            serde_json::to_value(replayed.report).unwrap(),
            serde_json::to_value(mask.report).unwrap()
        );
        let rule = objective::Objective::CanonicalRuleExcludedV1;
        let rule_mask = batch_mask(rule, None, &drawn, &items, &[1.0; 7])
            .unwrap()
            .unwrap();
        assert_eq!(rule_mask.values[3], 1.0);
        assert!(
            rule_mask
                .report
                .excluded_reviewed_ambiguity_o_tokens
                .is_none()
        );
        assert!(
            batch_mask(
                objective::Objective::LegacyPaddingV1,
                None,
                &drawn,
                &items,
                &[1.0; 7]
            )
            .unwrap()
            .is_none()
        );
        assert!(batch_mask(rule, Some(&resolved), &drawn, &items, &[1.0; 7]).is_err());
        assert!(batch_mask(reviewed, None, &drawn, &items, &[1.0; 7]).is_err());
        assert!(owners(None, 2, &indices).is_err());
    }

    #[test]
    fn fixed_budget_and_final_selection_are_required() {
        let receipt = json!({"path":"/receipt","sha256":"a".repeat(64)});
        let mut value = json!({"scope":super::super::SCOPE,"config":receipt,"canonical_preflight":receipt,
            "updates":10,"snapshots":[0,10],"sampling":{"mode":"complete_pass_stratified"},"members":[],
            "probe":{"train_names":["case"],"minimum_recall":0.5}});
        let manifest: Manifest = serde_json::from_value(value.clone()).unwrap();
        manifest.validate(32, 0).unwrap();
        assert!(manifest.loss_objective.is_legacy());
        assert!(manifest.mixed_inputs.is_none());
        assert!(
            serde_json::to_value(&manifest)
                .unwrap()
                .get("mixed_inputs")
                .is_none()
        );
        assert!(
            serde_json::to_value(&manifest)
                .unwrap()
                .get("loss_objective")
                .is_none()
        );
        assert_eq!(serde_json::to_value(&manifest).unwrap(), value);
        let mut opted = value.clone();
        opted["loss_objective"] = json!("canonical_rule_excluded_weighted_ce_v1");
        assert!(
            !serde_json::from_value::<Manifest>(opted)
                .unwrap()
                .loss_objective
                .is_legacy()
        );
        value["snapshots"] = json!([0, 9]);
        assert!(
            serde_json::from_value::<Manifest>(value.clone())
                .unwrap()
                .validate(32, 0)
                .is_err()
        );
        value["updates"] = json!(0);
        assert!(
            serde_json::from_value::<Manifest>(value)
                .unwrap()
                .validate(32, 0)
                .is_err()
        );
        assert!(manifest.validate(33, 0).is_err());
        assert!(manifest.validate(32, 10).is_err());
    }
}
