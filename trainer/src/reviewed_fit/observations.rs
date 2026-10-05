//! Committed native checkpoints and exact observations; DEV never chooses updates or weights.

use super::{contract::Fit, probe, snapshot, write_json};
use crate::net::TaggerNet;
use crate::reviewed_data::digest;
use anyhow::{Context, ensure};
use burn::backend::{Autodiff, NdArray};
use burn::module::AutodiffModule;
use serde_json::{Value, json};
use std::io::Write;
use std::path::Path;

/// Save preregistered checkpoints and observations without checkpoint ranking.
pub(super) struct Observer<'a> {
    fit: &'a Fit,
    out: &'a Path,
    events: std::fs::File,
    previous: Option<String>,
    baseline_probe: Option<Value>,
    baseline_dev: Option<Value>,
    baseline_parameter: Option<String>,
    /// Completed fixed-final observation, present only after the last declared update.
    pub(super) final_record: Option<Value>,
}

impl<'a> Observer<'a> {
    /// Create a fresh, append-only native checkpoint ledger.
    pub(super) fn new(fit: &'a Fit, out: &'a Path) -> anyhow::Result<Self> {
        std::fs::create_dir(out.join("checkpoints"))?;
        let events = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(out.join("checkpoint-events.jsonl"))?;
        Ok(Self {
            fit,
            out,
            events,
            previous: None,
            baseline_probe: None,
            baseline_dev: None,
            baseline_parameter: None,
            final_record: None,
        })
    }

    /// Verify finite round-trip weights and score with the exact canonical policy.
    pub(super) fn capture(
        &mut self,
        step: usize,
        model: &TaggerNet<Autodiff<NdArray>>,
        draws: &[usize],
    ) -> anyhow::Result<()> {
        ensure!(
            self.fit.manifest.snapshots.contains(&step),
            "unregistered native observation"
        );
        snapshot::verify(self.fit, self.out)?;
        let device = Default::default();
        let base = self.out.join("checkpoints").join(format!("step-{step}"));
        let native = crate::native_checkpoint::save(model.valid(), &base, &device)?;
        let score = |cohort| {
            if let Some(settings) = &self.fit.manifest.mixed_inputs {
                return crate::cohort_fit::score_with_postprocess(
                    &native.model,
                    cohort,
                    &device,
                    self.fit.native.input_policy(),
                    settings.postprocess,
                );
            }
            crate::cohort_fit::score_with_input_policy(
                &native.model,
                cohort,
                &device,
                self.fit.native.input_policy(),
            )
        };
        let cohorts = self.fit.loaded.cohorts();
        let train = score(&cohorts.train)?;
        let dev = score(&cohorts.dev)?;
        let probe_score = score(&self.fit.probe.cohort)?;
        if step == 0 {
            self.baseline_probe = Some(probe_score.clone());
            self.baseline_dev = Some(dev.clone());
            self.baseline_parameter = Some(native.parameter_sha256.clone());
        }
        snapshot::verify(self.fit, self.out)?;
        ensure!(
            digest(&std::fs::read(base.with_extension("mpk"))?) == native.checkpoint_sha256,
            "snapshot changed during scoring"
        );
        let score_file = format!("scores-step-{step}.json");
        let mut scores = json!({"scope":self.fit.scope(),"step":step,"parameter_sha256":native.parameter_sha256,
            "input_policy":self.fit.native.input_policy(),"train_seen_fit":train,
            "separate_frozen_dev_observation":dev,"real_train_probe":probe_score,
            "development_used_for_selection":false,"general_accuracy_claim":false});
        if let Some(mixed) = &self.fit.mixed {
            let postprocess = self
                .fit
                .manifest
                .mixed_inputs
                .as_ref()
                .context("mixed settings missing")?
                .postprocess;
            let mut authored_scores = Vec::new();
            for packet in &mixed.authored {
                let gold: Vec<Vec<_>> = packet
                    .expected
                    .iter()
                    .map(|spans| {
                        spans
                            .iter()
                            .map(|s| (s.kind.clone(), s.start, s.end))
                            .collect()
                    })
                    .collect();
                authored_scores.push(crate::cohort_fit::score_authored_with_postprocess(
                    &native.model,
                    &packet.names,
                    &packet.docs,
                    &gold,
                    &device,
                    postprocess,
                )?);
            }
            scores["authored_train_seen_fit"] = json!({"packets":authored_scores,"source_kind":"authored_train_only","independent_accuracy_claim":false,"learning_sanity_uses_authored":false});
            scores["postprocess"] = json!(postprocess);
            snapshot::verify(self.fit, self.out)?;
            ensure!(
                digest(&std::fs::read(base.with_extension("mpk"))?) == native.checkpoint_sha256,
                "snapshot changed during authored scoring"
            );
        }
        write_json(&self.out.join(&score_file), &scores)?;
        let event = json!({"scope":self.fit.scope(),"optimizer_updates_executed":step,
            "checkpoint_file":format!("checkpoints/step-{step}.mpk"),"checkpoint_sha256":native.checkpoint_sha256,
            "parameter_sha256":native.parameter_sha256,"native_roundtrip_verified":true,"all_parameters_finite":true,
            "score_file":score_file,"score_sha256":digest(&std::fs::read(self.out.join(&score_file))?),
            "document_draw_counts":draws,"previous_event_sha256":self.previous,
            "selection":"preregistered observation; final selection depends only on fixed update count"});
        let mut line = serde_json::to_vec(&event)?;
        line.push(b'\n');
        self.events.write_all(&line)?;
        self.events.sync_all()?;
        self.previous = Some(digest(&line));
        if step == self.fit.manifest.updates {
            let proof = probe::proof(
                &self.fit.manifest.probe,
                &self.fit.probe.identity,
                self.baseline_probe
                    .as_ref()
                    .context("baseline TRAIN probe missing")?,
                &scores["real_train_probe"],
                &native.checkpoint_sha256,
                &native.parameter_sha256,
                self.baseline_parameter
                    .as_deref()
                    .context("baseline parameter identity missing")?,
            )?;
            let baseline = self.baseline_dev.as_ref().context("baseline DEV missing")?;
            let mut changes = Vec::new();
            for kind in ["person", "org", "address", "email", "phone"] {
                let before = &baseline["filtered"]["per_kind"][kind];
                let after =
                    &scores["separate_frozen_dev_observation"]["filtered"]["per_kind"][kind];
                changes.push(json!({"kind":kind,"baseline":before,"final":after,
                    "f1_delta":after["f1"].as_f64().context("final F1 missing")?-before["f1"].as_f64().context("baseline F1 missing")?}));
            }
            self.final_record = Some(
                json!({"final_checkpoint":event,"learning_sanity_proof":proof,
                "dev_change_observation":changes,
                "dev_95_per_kind_precision_recall_objective":development_objective(&scores["separate_frozen_dev_observation"])? ,
                "comparison_scope":"same canonical KnownUs policy at initial and fixed-final steps; original empty-mask pilot is not this baseline",
                "dev_used_for_checkpoint_selection":false,
                "heldout_or_release_acceptance_proven":false,"scope":"candidate fit and development observation; no independent general accuracy claim"}),
            );
        }
        Ok(())
    }
}

/// Revalidate committed bytes, hash chaining, step bindings and actual draws.
pub(super) fn verify_ledger(
    fit: &Fit,
    out: &Path,
    draws: &[usize],
    expected_final: &Value,
) -> anyhow::Result<String> {
    let bytes = std::fs::read(out.join("checkpoint-events.jsonl"))?;
    let text = std::str::from_utf8(&bytes)?;
    let lines: Vec<_> = text.split_inclusive('\n').collect();
    ensure!(
        lines.len() == fit.manifest.snapshots.len(),
        "native checkpoint ledger count differs"
    );
    ensure!(
        lines
            .last()
            .is_some_and(|line| serde_json::from_str::<Value>(line)
                .is_ok_and(|event| event == *expected_final)),
        "final native event differs from the in-memory committed observation"
    );
    let mut previous = None::<String>;
    for (line, &step) in lines.iter().zip(&fit.manifest.snapshots) {
        let event: Value = serde_json::from_str(line)?;
        let checkpoint = format!("checkpoints/step-{step}.mpk");
        let score = format!("scores-step-{step}.json");
        let score_bytes = std::fs::read(out.join(&score))?;
        let scored: Value = serde_json::from_slice(&score_bytes)?;
        let expected_draws = draw_counts(&fit.plan.batches[..step], draws.len())?;
        ensure!(
            line.ends_with('\n')
                && event["scope"] == fit.scope()
                && event["optimizer_updates_executed"] == step
                && event["checkpoint_file"] == checkpoint
                && event["score_file"] == score
                && event["native_roundtrip_verified"] == true
                && event["all_parameters_finite"] == true
                && event["previous_event_sha256"] == serde_json::to_value(&previous)?
                && event["checkpoint_sha256"] == digest(&std::fs::read(out.join(checkpoint))?)
                && event["score_sha256"] == digest(&score_bytes)
                && event["document_draw_counts"] == serde_json::to_value(&expected_draws)?
                && scored["scope"] == fit.scope()
                && scored["step"] == step
                && scored["parameter_sha256"] == event["parameter_sha256"]
                && scored["input_policy"] == serde_json::to_value(fit.native.input_policy())?,
            "saved native observation changed"
        );
        if let Some(settings) = &fit.manifest.mixed_inputs {
            ensure!(
                scored["postprocess"] == serde_json::to_value(settings.postprocess)?
                    && scored["authored_train_seen_fit"]["source_kind"] == "authored_train_only"
                    && scored["authored_train_seen_fit"]["learning_sanity_uses_authored"] == false
                    && scored["authored_train_seen_fit"]["packets"]
                        .as_array()
                        .is_some_and(
                            |p| p.len() == fit.mixed.as_ref().map_or(0, |m| m.authored.len())
                        ),
                "mixed observation scope/postprocess changed"
            );
        } else {
            ensure!(
                scored.get("postprocess").is_none()
                    && scored.get("authored_train_seen_fit").is_none(),
                "native observation cannot contain authored diagnostics"
            );
        }
        previous = Some(digest(line.as_bytes()));
    }
    ensure!(
        draw_counts(&fit.plan.batches, draws.len())? == draws,
        "final native draw counts differ"
    );
    Ok(digest(&bytes))
}

/// Count each actual document occurrence, retaining across-batch repetition.
pub(super) fn draw_counts(batches: &[Vec<usize>], documents: usize) -> anyhow::Result<Vec<usize>> {
    let mut counts = vec![0usize; documents];
    for batch in batches {
        for &index in batch {
            let count = counts
                .get_mut(index)
                .context("completed draw index missing")?;
            *count = count
                .checked_add(1)
                .context("completed draw count overflow")?;
        }
    }
    Ok(counts)
}

fn development_objective(score: &Value) -> anyhow::Result<Value> {
    let mut checks = Vec::new();
    for kind in ["person", "org", "address", "email", "phone"] {
        let counts = &score["filtered"]["per_kind"][kind]["counts"];
        let tp = counts["tp"].as_u64().context("DEV TP missing")?;
        let predicted = counts["predicted"]
            .as_u64()
            .context("DEV predictions missing")?;
        let gold = counts["gold"].as_u64().context("DEV gold missing")?;
        ensure!(
            tp <= predicted && tp <= gold,
            "DEV exact counts inconsistent"
        );
        let precision_passed = predicted > 0 && u128::from(tp) * 100 >= u128::from(predicted) * 95;
        let recall_passed = gold > 0 && u128::from(tp) * 100 >= u128::from(gold) * 95;
        checks.push(
            json!({"kind":kind,"counts":counts,"precision_minimum":0.95,"recall_minimum":0.95,
            "precision_passed":precision_passed,"recall_passed":recall_passed,
            "passed":precision_passed && recall_passed}),
        );
    }
    Ok(
        json!({"checks":checks,"target_met":checks.iter().all(|c|c["passed"]==true),
        "scope":"exact filtered frozen DEV diagnostic objective; no independent release quality proof",
        "checkpoint_selection_uses_objective":false,"release_eligible":false}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_95_objective_requires_both_metrics_and_all_five_kinds() {
        let mut kinds = serde_json::Map::new();
        for kind in ["person", "org", "address", "email", "phone"] {
            kinds.insert(
                kind.into(),
                json!({"counts":{"tp":19,"predicted":20,"gold":20}}),
            );
        }
        let mut scores = json!({"filtered":{"per_kind":kinds}});
        assert_eq!(development_objective(&scores).unwrap()["target_met"], true);
        scores["filtered"]["per_kind"]["phone"]["counts"]["gold"] = json!(21);
        assert_eq!(development_objective(&scores).unwrap()["target_met"], false);
        scores["filtered"]["per_kind"]["phone"]["counts"] = json!({"tp":0,"predicted":0,"gold":0});
        assert_eq!(development_objective(&scores).unwrap()["target_met"], false);
        scores["filtered"]["per_kind"]["phone"]["counts"]["tp"] = json!(1);
        assert!(development_objective(&scores).is_err());
    }

    #[test]
    fn actual_draw_counter_retains_repetition_and_rejects_unknown_indices() {
        assert_eq!(draw_counts(&[vec![0, 1], vec![1]], 2).unwrap(), [1, 2]);
        assert!(draw_counts(&[vec![2]], 2).is_err());
    }
}
