//! Detect catastrophic failure to learn on a fixed training probe, not model quality.

use std::collections::BTreeMap;

use anyhow::{Context, ensure};
use serde::Serialize;

use crate::detector::{Prf, SpanScores};

const KINDS: [&str; 3] = ["person", "org", "address"];

/// Exact-span metrics and support for one learned kind in the training probe.
#[derive(Debug, Serialize)]
pub(crate) struct KindDiagnostic {
    /// Precision, recall and F1 measured against the original training labels.
    pub(crate) exact: Prf,
    /// Number of original gold spans supporting these scores.
    pub(crate) gold: usize,
}

/// Serializable outcome of the catastrophic learning check.
#[derive(Debug, Serialize)]
pub(crate) struct Diagnostic {
    /// Completed optimizer updates when the fixed probe was evaluated.
    pub(crate) step: usize,
    /// First update at which low recall must stop the experiment.
    pub(crate) start_step: usize,
    /// Inclusive minimum exact recall for each learned kind.
    pub(crate) min_recall: f64,
    /// Whether the configured enforcement start has been reached.
    pub(crate) enforced: bool,
    /// True before enforcement, or when every kind meets the minimum recall.
    pub(crate) passed: bool,
    /// Kinds below the minimum, including observations before enforcement starts.
    pub(crate) failed_kinds: Vec<&'static str>,
    /// All three kinds and their original gold support.
    pub(crate) per_kind: BTreeMap<&'static str, KindDiagnostic>,
    /// Limits of what passing this check establishes.
    pub(crate) scope: &'static str,
}

/// Validate probe metrics and diagnose low recall without performing training or I/O.
///
/// Invalid metrics or absent gold support always fail validation. Low recall becomes a
/// failing diagnostic at `start_step`, allowing callers to record it before stopping.
pub(crate) fn evaluate(
    scores: &SpanScores,
    min_recall: f64,
    step: usize,
    start_step: usize,
) -> anyhow::Result<Diagnostic> {
    ensure!(
        min_recall.is_finite() && min_recall > 0.0 && min_recall <= 1.0,
        "training probe minimum recall must be finite and in (0, 1]"
    );
    let mut per_kind = BTreeMap::new();
    let mut failed_kinds = Vec::new();
    for kind in KINDS {
        let scores = scores
            .per_kind
            .get(kind)
            .with_context(|| format!("training probe is missing {kind} scores"))?;
        ensure!(scores.gold > 0, "training probe has no {kind} gold spans");
        for (metric, value) in [
            ("precision", scores.exact.precision),
            ("recall", scores.exact.recall),
            ("F1", scores.exact.f1),
        ] {
            ensure!(
                value.is_finite() && (0.0..=1.0).contains(&value),
                "training probe {kind} exact {metric} must be finite and in [0, 1]"
            );
        }
        if scores.exact.recall < min_recall {
            failed_kinds.push(kind);
        }
        per_kind.insert(
            kind,
            KindDiagnostic {
                exact: scores.exact,
                gold: scores.gold,
            },
        );
    }
    let enforced = step >= start_step;
    Ok(Diagnostic {
        step,
        start_step,
        min_recall,
        enforced,
        passed: !enforced || failed_kinds.is_empty(),
        failed_kinds,
        per_kind,
        scope: "catastrophic learning check on fixed training examples; not a quality or 95% accuracy gate",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::KindScores;

    fn scores(person_recall: f64) -> SpanScores {
        let per_kind = KINDS
            .into_iter()
            .map(|kind| {
                let recall = if kind == "person" { person_recall } else { 1.0 };
                (
                    kind,
                    KindScores {
                        exact: Prf {
                            precision: recall,
                            recall,
                            f1: recall,
                        },
                        gold: 10,
                        ..Default::default()
                    },
                )
            })
            .collect();
        SpanScores {
            per_kind,
            macro_exact_f1: (person_recall + 2.0) / 3.0,
        }
    }

    #[test]
    fn zero_person_recall_fails_despite_other_classes_scoring_perfectly() {
        let result = evaluate(&scores(0.0), 0.05, 100, 100).unwrap();
        assert!(result.enforced);
        assert!(!result.passed);
        assert_eq!(result.failed_kinds, ["person"]);
        let json = serde_json::to_value(result).unwrap();
        assert_eq!(json["per_kind"]["org"]["exact"]["f1"], 1.0);
        assert_eq!(json["per_kind"]["person"]["gold"], 10);
        assert_eq!(json["passed"], false);
    }

    #[test]
    fn low_recall_is_observed_but_not_enforced_before_start() {
        let result = evaluate(&scores(0.0), 0.05, 99, 100).unwrap();
        assert!(!result.enforced);
        assert!(result.passed);
        assert_eq!(result.failed_kinds, ["person"]);
    }

    #[test]
    fn threshold_boundary_is_inclusive() {
        assert!(evaluate(&scores(0.05), 0.05, 100, 100).unwrap().passed);
        assert!(!evaluate(&scores(0.049), 0.05, 100, 100).unwrap().passed);
        assert!(evaluate(&scores(1.0), 1.0, 100, 100).unwrap().passed);
    }

    #[test]
    fn missing_kind_or_zero_support_is_invalid_even_before_start() {
        let mut missing = scores(1.0);
        missing.per_kind.remove("org");
        assert!(evaluate(&missing, 0.05, 0, 100).is_err());
        let mut empty = scores(1.0);
        empty.per_kind.get_mut("address").unwrap().gold = 0;
        assert!(evaluate(&empty, 0.05, 0, 100).is_err());
    }

    #[test]
    fn nonfinite_or_out_of_range_exact_metrics_are_invalid() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -0.01, 1.01] {
            for metric in 0..3 {
                let mut scores = scores(1.0);
                let exact = &mut scores.per_kind.get_mut("person").unwrap().exact;
                match metric {
                    0 => exact.precision = value,
                    1 => exact.recall = value,
                    _ => exact.f1 = value,
                }
                assert!(evaluate(&scores, 0.05, 0, 100).is_err());
            }
        }
    }

    #[test]
    fn invalid_minimum_recall_is_rejected() {
        for minimum in [0.0, -0.01, 1.01, f64::NAN, f64::INFINITY] {
            assert!(evaluate(&scores(1.0), minimum, 100, 100).is_err());
        }
    }
}
