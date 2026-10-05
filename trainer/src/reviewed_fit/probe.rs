//! A fixed real native TRAIN probe; no DEV features or fictitious synthetic group.

use crate::reviewed_data::Cohort;
use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeSet;

/// Explicit complete TRAIN names and a preregistered exact neural recall floor.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Request {
    pub(super) train_names: Vec<String>,
    pub(super) minimum_recall: f64,
}

/// Complete selected TRAIN documents and their native identity.
pub(super) struct Probe {
    pub(super) cohort: Cohort,
    pub(super) identity: Value,
}

/// Require declared TRAIN membership and actual support for all neural kinds.
pub(super) fn prepare(request: &Request, train: &Cohort, native: &Value) -> anyhow::Result<Probe> {
    let indices = indices(request, &train.names)?;
    let cohort = train.subset(&indices)?;
    let mut support = [0usize; 3];
    for span in cohort.docs.iter().flat_map(|doc| &doc.gold) {
        ensure!(span.kind < 3, "probe gold kind invalid");
        support[span.kind] += 1;
    }
    ensure!(
        support.iter().all(|&n| n > 0),
        "real TRAIN probe requires PERSON, ORG and ADDRESS"
    );
    let rows = native["train"]
        .as_array()
        .context("probe native identity missing")?;
    let identity = json!({"scope":"real native TRAIN-only learning sanity probe",
        "train_indices":indices,"request":request,"support":support,
        "native_rows":indices.iter().map(|&index|rows.get(index).context("probe native row missing")).collect::<anyhow::Result<Vec<_>>>()?,
        "independent_evaluation":false,"synthetic_rows":0});
    Ok(Probe { cohort, identity })
}

fn indices(request: &Request, names: &[String]) -> anyhow::Result<Vec<usize>> {
    ensure!(
        !request.train_names.is_empty()
            && request.minimum_recall.is_finite()
            && request.minimum_recall > 0.0
            && request.minimum_recall <= 1.0,
        "declare supported TRAIN probe and finite recall floor"
    );
    let mut seen = BTreeSet::new();
    request
        .train_names
        .iter()
        .map(|name| {
            ensure!(seen.insert(name), "probe repeats a TRAIN document");
            names
                .iter()
                .position(|n| n == name)
                .context("probe name is not native TRAIN")
        })
        .collect::<anyhow::Result<Vec<_>>>()
}

/// Report supported real TRAIN recall floors and observed parameter updates.
pub(super) fn proof(
    request: &Request,
    identity: &Value,
    baseline: &Value,
    final_score: &Value,
    checkpoint_sha256: &str,
    parameter_sha256: &str,
    baseline_parameter_sha256: &str,
) -> anyhow::Result<Value> {
    let mut checks = Vec::new();
    for kind in ["person", "org", "address"] {
        let per_kind = &final_score["filtered"]["per_kind"][kind];
        let recall = per_kind["recall"]
            .as_f64()
            .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
            .context("probe recall invalid")?;
        ensure!(
            per_kind["counts"]["gold"].as_u64().is_some_and(|n| n > 0),
            "probe kind lacks real support"
        );
        checks.push(json!({"kind":kind,"recall":recall,"minimum":request.minimum_recall,
            "baseline":baseline["filtered"]["per_kind"][kind],"final":per_kind,"passed":recall >= request.minimum_recall}));
    }
    Ok(
        json!({"scope":"reviewed-real TRAIN sanity only; not independent heldout quality",
        "identity":identity,"checks":checks,
        "parameters_changed":parameter_sha256 != baseline_parameter_sha256,
        "baseline_parameter_sha256":baseline_parameter_sha256,
        "passed":parameter_sha256 != baseline_parameter_sha256 && checks.iter().all(|c|c["passed"]==true),
        "checkpoint_sha256":checkpoint_sha256,"parameter_sha256":parameter_sha256,
        "selection":"fixed final optimizer count regardless probe or DEV score",
        "proof_semantics":"updated finite parameters and supported TRAIN recall floors; not a proof of positive learning gain or heldout gain",
        "export_allowed":false,"release_quality_claim":false}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_membership_rejects_missing_dev_duplicate_and_empty_names() {
        let names = vec!["train".into()];
        let request = |chosen: Vec<&str>| Request {
            train_names: chosen.into_iter().map(str::to_owned).collect(),
            minimum_recall: 0.8,
        };
        assert_eq!(indices(&request(vec!["train"]), &names).unwrap(), [0]);
        assert!(indices(&request(vec!["dev"]), &names).is_err());
        assert!(indices(&request(vec!["train", "train"]), &names).is_err());
        assert!(indices(&request(vec![]), &names).is_err());
    }

    #[test]
    fn learning_sanity_requires_supported_recall_and_parameter_updates() {
        let request = Request {
            train_names: vec!["train".into()],
            minimum_recall: 0.8,
        };
        let mut score = json!({"filtered":{"per_kind":{}}});
        for kind in ["person", "org", "address"] {
            score["filtered"]["per_kind"][kind] = json!({"recall":0.8,"counts":{"gold":1}});
        }
        let check = |s: &Value, final_sha: &str| {
            proof(
                &request,
                &json!({}),
                &score,
                s,
                "checkpoint",
                final_sha,
                "initial",
            )
        };
        assert_eq!(check(&score, "final").unwrap()["passed"], true);
        assert_eq!(check(&score, "initial").unwrap()["passed"], false);
        let mut missing = score.clone();
        missing["filtered"]["per_kind"]["address"]["counts"]["gold"] = json!(0);
        assert!(check(&missing, "final").is_err());
        missing = score.clone();
        missing["filtered"]["per_kind"]["org"]["recall"] = json!(0.79);
        assert_eq!(check(&missing, "final").unwrap()["passed"], false);
    }
}
