//! Exact diagnostic metrics and preregistered acceptance without checkpoint selection.

use super::data::Cohort;
use crate::detector::{self, KindSpan};
use crate::diagnostic_operator::ForwardOperator;
use crate::net::TaggerNet;
use anyhow::{Context, ensure};
use burn::backend::NdArray;
use serde_json::{Value, json};
use std::collections::BTreeMap;
type Entity = (String, u32, u32);

fn entities(spans: &[KindSpan]) -> Vec<Entity> {
    spans
        .iter()
        .map(|s| (detector::KINDS[s.kind].as_str().to_owned(), s.start, s.end))
        .collect()
}

fn f1(scores: &Value, kind: &str) -> anyhow::Result<f64> {
    metric(scores, kind, "f1")
}

fn metric(scores: &Value, kind: &str, name: &str) -> anyhow::Result<f64> {
    scores["filtered"]["per_kind"][kind][name]
        .as_f64()
        .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
        .with_context(|| format!("cohort acceptance requires exact-span {name} in [0,1]"))
}

pub(super) fn acceptance(
    train: &Value,
    dev: &Value,
    baseline_dev: &Value,
) -> anyhow::Result<Value> {
    let mut train_checks = BTreeMap::new();
    let mut control_checks = BTreeMap::new();
    for kind in ["person", "org", "address"] {
        let value = f1(train, kind)?;
        train_checks.insert(
            kind,
            json!({"f1":value,"minimum":0.99,"passed":value >= 0.99}),
        );
    }
    for kind in ["person", "address"] {
        let value = f1(dev, kind)?;
        let baseline = f1(baseline_dev, kind)?;
        control_checks.insert(
            kind,
            json!({"f1":value,"step0_f1":baseline,"passed":value >= baseline}),
        );
    }
    let org = f1(dev, "org")?;
    let initial_org = f1(baseline_dev, "org")?;
    let precision = metric(dev, "org", "precision")?;
    let recall = metric(dev, "org", "recall")?;
    let initial_precision = metric(baseline_dev, "org", "precision")?;
    let initial_recall = metric(baseline_dev, "org", "recall")?;
    let gain_passed = org - initial_org >= 0.05 - 1e-12;
    let target_preserved_passed = precision >= 0.95 && recall >= 0.95 && org >= initial_org;
    let org_passed = gain_passed || target_preserved_passed;
    let branch = match (gain_passed, target_preserved_passed) {
        (true, true) => "gain_and_target_preserved",
        (true, false) => "gain",
        (false, true) => "target_preserved",
        (false, false) => "neither",
    };
    let passed = org_passed
        && train_checks.values().all(|v| v["passed"] == true)
        && control_checks.values().all(|v| v["passed"] == true);
    Ok(
        json!({"criteria":super::contracts::acceptance_criteria(),"passed":passed,
        "train_each_neural_kind":train_checks,"dev_controls":control_checks,
        "dev_org":{"f1":org,"step0_f1":initial_org,"absolute_gain":org-initial_org,
            "precision":precision,"recall":recall,"counts":dev["filtered"]["per_kind"]["org"]["counts"],
            "step0_precision":initial_precision,"step0_recall":initial_recall,
            "step0_counts":baseline_dev["filtered"]["per_kind"]["org"]["counts"],
            "minimum_gain":0.05,"gain_passed":gain_passed,
            "target_preserved_passed":target_preserved_passed,"accepted_branch":branch,"passed":org_passed},
        "scope":"bounded diagnostic outcome only; no independent general accuracy or release approval",
        "release_quality_claim":false}),
    )
}

fn metrics(gold: &[Vec<Entity>], predicted: &[Vec<Entity>]) -> Value {
    let mut counts = BTreeMap::new();
    let mut exact_documents = 0;
    for (gold, predicted) in gold.iter().zip(predicted) {
        let matched = crate::exact_metrics::matches(gold, predicted);
        if matched.iter().all(|&yes| yes) && predicted.len() == gold.len() {
            exact_documents += 1;
        }
        for kind in ["person", "org", "address", "email", "phone"] {
            let value = crate::exact_metrics::Counts {
                tp: predicted
                    .iter()
                    .zip(&matched)
                    .filter(|(s, yes)| s.0 == kind && **yes)
                    .count(),
                predicted: predicted.iter().filter(|s| s.0 == kind).count(),
                gold: gold.iter().filter(|s| s.0 == kind).count(),
            };
            counts
                .entry(kind)
                .or_insert_with(crate::exact_metrics::Counts::default)
                .add(&value);
        }
    }
    let mut per_kind = BTreeMap::new();
    let mut micro = crate::exact_metrics::Counts::default();
    for (kind, count) in counts {
        let (precision, recall, f1) = count.prf();
        per_kind.insert(
            kind,
            json!({"counts":count, "precision":precision, "recall":recall, "f1":f1}),
        );
        micro.add(&count);
    }
    let (precision, recall, f1) = micro.prf();
    json!({"per_kind":per_kind, "micro":{"counts":micro,"precision":precision,"recall":recall,"f1":f1},
        "exact_documents":exact_documents, "documents":gold.len(),
        "exact_document_fraction":exact_documents as f64 / gold.len().max(1) as f64})
}

pub(super) fn score(
    model: &TaggerNet<NdArray>,
    cohort: &Cohort,
    device: &burn::backend::ndarray::NdArrayDevice,
) -> anyhow::Result<Value> {
    score_selected(model, cohort, device, None)
}

/// Native raw/filtered scoring with the same explicit Text selector as canonical features.
pub(crate) fn score_with_input_policy(
    model: &TaggerNet<NdArray>,
    cohort: &Cohort,
    device: &burn::backend::ndarray::NdArrayDevice,
    policy: detector::DetectorInputPolicy,
) -> anyhow::Result<Value> {
    score_selected(model, cohort, device, Some(policy))
}

fn score_selected(
    model: &TaggerNet<NdArray>,
    cohort: &Cohort,
    device: &burn::backend::ndarray::NdArrayDevice,
    policy: Option<detector::DetectorInputPolicy>,
) -> anyhow::Result<Value> {
    let gold: Vec<Vec<Entity>> = cohort
        .expected
        .iter()
        .map(|spans| {
            spans
                .iter()
                .map(|s| (s.kind.clone(), s.start, s.end))
                .collect()
        })
        .collect();
    score_rows(
        model,
        &cohort.names,
        &cohort.docs,
        &gold,
        device,
        policy,
        None,
    )
}

/// Score typed native rows with an explicit separately bound postprocessor.
pub(crate) fn score_with_postprocess(
    model: &TaggerNet<NdArray>,
    cohort: &Cohort,
    device: &burn::backend::ndarray::NdArrayDevice,
    policy: detector::DetectorInputPolicy,
    postprocess: crate::reviewed_fit::Postprocess,
) -> anyhow::Result<Value> {
    let gold = cohort
        .expected
        .iter()
        .map(|spans| {
            spans
                .iter()
                .map(|s| (s.kind.clone(), s.start, s.end))
                .collect()
        })
        .collect::<Vec<_>>();
    score_rows(
        model,
        &cohort.names,
        &cohort.docs,
        &gold,
        device,
        Some(policy),
        Some(postprocess),
    )
}

/// Borrow separate authored owners; no native Cohort or Origin is manufactured.
pub(crate) fn score_authored_with_postprocess(
    model: &TaggerNet<NdArray>,
    names: &[String],
    docs: &[detector::DetectorDoc],
    gold: &[Vec<(String, u32, u32)>],
    device: &burn::backend::ndarray::NdArrayDevice,
    postprocess: crate::reviewed_fit::Postprocess,
) -> anyhow::Result<Value> {
    score_rows(
        model,
        names,
        docs,
        gold,
        device,
        Some(detector::DetectorInputPolicy::KnownUs),
        Some(postprocess),
    )
}

fn score_rows(
    model: &TaggerNet<NdArray>,
    names: &[String],
    docs: &[detector::DetectorDoc],
    gold: &[Vec<Entity>],
    device: &burn::backend::ndarray::NdArrayDevice,
    policy: Option<detector::DetectorInputPolicy>,
    postprocess: Option<crate::reviewed_fit::Postprocess>,
) -> anyhow::Result<Value> {
    ensure!(
        names.len() == docs.len() && docs.len() == gold.len() && !docs.is_empty(),
        "scoring whole-row alignment differs"
    );
    let candidates = match postprocess {
        None => detector::predict_scored_with_operator_and_decoder(
            model,
            docs,
            32,
            device,
            ForwardOperator::ContextRmsV2,
            None,
        )?,
        Some(version) => {
            detector::predict_scored_with_postprocess(model, docs, 32, device, version)?
        }
    };
    ensure!(
        candidates.len() == docs.len(),
        "scoring document alignment differs"
    );
    let raw: Vec<Vec<KindSpan>> = candidates
        .iter()
        .map(|spans| spans.iter().map(|s| s.span).collect())
        .collect();
    let filtered = detector::apply_confidence_policy(&candidates);
    let mut raw_entities: Vec<_> = raw.iter().map(|s| entities(s)).collect();
    let mut filtered_entities: Vec<_> = filtered.iter().map(|s| entities(s)).collect();
    for (index, doc) in docs.iter().enumerate() {
        let selected_rules = match policy {
            None => tessera::internal::scan_rules(&doc.text, &["US"]),
            Some(detector::DetectorInputPolicy::KnownUs) => {
                tessera::internal::scan_text_rules(&doc.text, &["US"])
            }
            Some(detector::DetectorInputPolicy::AutoText) => {
                tessera::internal::scan_text_rules(&doc.text, &[])
            }
        };
        let rules: Vec<_> = selected_rules
            .into_iter()
            .map(|s| (s.kind.as_str().to_owned(), s.start as u32, s.end as u32))
            .collect();
        raw_entities[index].extend(rules.clone());
        filtered_entities[index].extend(rules);
        raw_entities[index].sort();
        filtered_entities[index].sort();
    }
    let mut result = json!({"raw":metrics(gold, &raw_entities), "filtered":metrics(gold, &filtered_entities),
    "predictions":names.iter().zip(&raw_entities).zip(&filtered_entities)
        .map(|((name, raw), filtered)| json!({"name":name,"raw":raw,"filtered":filtered})).collect::<Vec<_>>(),
    "semantics":match policy {
        None => "native f32 current graph and decoder; PHONE/EMAIL with explicit US hint; encoder rule masks retain their original empty-country-hint rules; US hint applies only to diagnostic rule output; this is not public detect API parity; gold does not affect features",
        Some(detector::DetectorInputPolicy::KnownUs) => "native f32 raw and threshold-filtered neural spans from canonical known-US feature masks; rule outputs use the same explicit US selector; no packaged parser/public API quality claim",
        Some(detector::DetectorInputPolicy::AutoText) => "native f32 raw and threshold-filtered neural spans from canonical AUTO Text inferred-rule masks; rule outputs use the same document inference selector; no packaged parser/public API quality claim",
    }});
    if let Some(version) = postprocess {
        result["pipeline"] = json!({"forward_operator":"context_rms_v2","base_decoder":"address_continuation_v1","postprocess":version,"input_policy":policy});
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_scoring_counts_duplicate_predictions_and_wrong_boundaries() {
        let entity = ("person".to_owned(), 0, 8);
        let result = metrics(&[vec![entity.clone()]], &[vec![entity.clone(), entity]]);
        assert_eq!(result["per_kind"]["person"]["counts"]["tp"], 1);
        assert_eq!(result["per_kind"]["person"]["counts"]["predicted"], 2);
        assert_eq!(result["exact_documents"], 0);
        let wrong = metrics(
            &[vec![("phone".into(), 2, 15)]],
            &[vec![("phone".into(), 2, 14)]],
        );
        assert_eq!(wrong["per_kind"]["phone"]["counts"]["tp"], 0);
    }

    #[test]
    fn acceptance_requires_org_gain_and_preserved_neural_controls() {
        fn score(person: f64, org: f64, address: f64) -> Value {
            json!({"filtered":{"per_kind":{"person":{"f1":person},"org":{"f1":org,"precision":org,"recall":org},"address":{"f1":address}}}})
        }
        let train = score(0.99, 0.99, 0.99);
        let baseline = score(0.7, 0.5, 0.8);
        assert_eq!(
            acceptance(&train, &score(0.7, 0.85, 0.8), &score(0.7, 0.8, 0.8)).unwrap()["passed"],
            true
        );
        assert_eq!(
            acceptance(&train, &score(0.7, 0.56, 0.8), &baseline).unwrap()["passed"],
            true
        );
        assert_eq!(
            acceptance(&train, &score(0.69, 0.56, 0.8), &baseline).unwrap()["passed"],
            false
        );
        assert_eq!(
            acceptance(&train, &score(0.7, 0.54, 0.8), &baseline).unwrap()["passed"],
            false
        );
        assert_eq!(
            acceptance(&score(0.99, 0.98, 0.99), &score(0.7, 0.56, 0.8), &baseline).unwrap()["passed"],
            false
        );
    }

    #[test]
    fn acceptance_can_preserve_org_target_above_the_gain_ceiling() {
        fn score(org: f64) -> Value {
            json!({"filtered":{"per_kind":{"person":{"f1":0.99},
                "org":{"f1":org,"precision":org,"recall":org},"address":{"f1":0.99}}}})
        }
        let result = acceptance(&score(0.99), &score(0.96), &score(0.96)).unwrap();
        assert_eq!(result["passed"], true);
        assert_eq!(result["dev_org"]["accepted_branch"], "target_preserved");
        assert_eq!(result["dev_org"]["gain_passed"], false);
        let decline = acceptance(&score(0.99), &score(0.96), &score(0.97)).unwrap();
        assert_eq!(decline["passed"], false);
    }

    #[test]
    fn high_org_f1_does_not_replace_both_precision_and_recall_targets() {
        let scores = |precision: f64, recall: f64| {
            json!({"filtered":{"per_kind":{"person":{"f1":0.99},
                "org":{"f1":2.0*precision*recall/(precision+recall),"precision":precision,"recall":recall},
                "address":{"f1":0.99}}}})
        };
        let baseline = scores(0.96, 0.96);
        for dev in [scores(0.94, 1.0), scores(1.0, 0.94)] {
            let result = acceptance(&scores(0.99, 0.99), &dev, &baseline).unwrap();
            assert!(result["dev_org"]["f1"].as_f64().unwrap() > 0.95);
            assert_eq!(result["passed"], false);
        }
        let mut missing = scores(0.96, 0.96);
        missing["filtered"]["per_kind"]["org"]["precision"] = Value::Null;
        assert!(acceptance(&scores(0.99, 0.99), &missing, &baseline).is_err());
    }
}
