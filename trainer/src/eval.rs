//! Scoring for the deterministic baselines and for external predictions, over the
//! fixture files. The metric definitions are fixed here, before any model exists, so the
//! comparison in Milestone 2 is against numbers nobody chose after seeing them.

use std::collections::BTreeMap;

use anyhow::{Context, bail};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::EvalArgs;
use crate::baselines::{self, Span};
use crate::fixtures::{self, DetectorFixture, GrouperFixture, ParserFixture};

/// Precision, recall, and F1 from match counts. A zero denominator scores zero.
pub fn prf(tp: usize, pred: usize, gold: usize) -> (f64, f64, f64) {
    let p = if pred == 0 {
        0.0
    } else {
        tp as f64 / pred as f64
    };
    let r = if gold == 0 {
        0.0
    } else {
        tp as f64 / gold as f64
    };
    let f = if p + r == 0.0 {
        0.0
    } else {
        2.0 * p * r / (p + r)
    };
    (p, r, f)
}

fn round4(x: f64) -> f64 {
    (x * 10_000.0).round() / 10_000.0
}

#[derive(Debug, Default, Clone)]
struct Counts {
    tp: usize,
    pred: usize,
    gold: usize,
    boundary_hits: usize,
    boundary_candidates: usize,
}

impl Counts {
    fn json(&self) -> Value {
        let (p, r, f) = prf(self.tp, self.pred, self.gold);
        json!({
            "precision": round4(p),
            "recall": round4(r),
            "f1": round4(f),
            "tp": self.tp,
            "predicted": self.pred,
            "gold": self.gold,
        })
    }
}

#[derive(Debug, Default)]
struct Group {
    overall: Counts,
    by_key: BTreeMap<String, Counts>,
    by_country: BTreeMap<String, Counts>,
}

impl Group {
    fn add(&mut self, key: &str, country: &str, f: impl Fn(&mut Counts)) {
        f(&mut self.overall);
        f(self.by_key.entry(key.to_string()).or_default());
        f(self.by_country.entry(country.to_string()).or_default());
    }
}

fn map_json(m: &BTreeMap<String, Counts>) -> Value {
    Value::Object(m.iter().map(|(k, v)| (k.clone(), v.json())).collect())
}

/// A predicted span a fixture says must never be produced.
#[derive(Debug)]
struct MustNotViolation {
    fixture: String,
    case: String,
    kind: String,
    text: String,
}

/// One prediction file produced outside the trainer, such as Presidio's.
#[derive(Debug, Deserialize)]
struct External {
    system: String,
    predictions: Vec<ExternalPrediction>,
}

#[derive(Debug, Deserialize)]
struct ExternalPrediction {
    fixture: String,
    case: String,
    entities: Vec<ExternalEntity>,
}

#[derive(Debug, Deserialize)]
struct ExternalEntity {
    kind: String,
    start: usize,
    end: usize,
}

/// `trainer eval`: scores a run, the deterministic baselines, or external predictions.
pub fn run(args: EvalArgs) -> anyhow::Result<()> {
    if let Some(run) = &args.run
        && !args.baseline
    {
        let split = match args.split.as_str() {
            "train" => crate::data::Split::Train,
            "valid" => crate::data::Split::Valid,
            "test" => crate::data::Split::Test,
            other => bail!("unknown split `{other}`"),
        };
        return match args.backend {
            crate::train::BackendKind::Wgpu => crate::model_eval::run::<burn::backend::Wgpu>(
                run,
                split,
                args.errors,
                &Default::default(),
            ),
            crate::train::BackendKind::Ndarray => crate::model_eval::run::<burn::backend::NdArray>(
                run,
                split,
                args.errors,
                &Default::default(),
            ),
        };
    }
    if !args.baseline && args.external.is_empty() {
        bail!("pass --baseline, --external, or both");
    }

    let parser_fixtures: Vec<(String, ParserFixture)> =
        fixtures::load_dir(&args.fixtures.join("parser"))?;
    let detector_fixtures: Vec<(String, DetectorFixture)> =
        fixtures::load_dir(&args.fixtures.join("detector"))?;
    let grouper_fixtures: Vec<(String, GrouperFixture)> =
        fixtures::load_dir(&args.fixtures.join("grouper"))?;

    fixtures::check_offsets(&parser_fixtures, &detector_fixtures, &grouper_fixtures)?;

    let mut systems: Vec<Value> = Vec::new();
    let mut tables: Vec<(String, Value)> = Vec::new();

    if args.baseline {
        let value = score_baseline(&parser_fixtures, &detector_fixtures, &grouper_fixtures)?;
        tables.push(("baseline".to_string(), value.clone()));
        systems.push(value);
    }
    for path in &args.external {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let external: External =
            serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        let value = score_external(&external, &detector_fixtures)?;
        tables.push((external.system.clone(), value.clone()));
        systems.push(value);
    }

    let out_dir = args.out.join("eval");
    std::fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let json_path = out_dir.join("fixtures.json");
    std::fs::write(
        &json_path,
        serde_json::to_string_pretty(&json!({ "systems": systems }))? + "\n",
    )?;
    let markdown = render_markdown(&tables);
    std::fs::write(out_dir.join("fixtures.md"), &markdown)?;
    print!("{markdown}");
    println!("written: {}", json_path.display());
    Ok(())
}

fn score_baseline(
    parser_fixtures: &[(String, ParserFixture)],
    detector_fixtures: &[(String, DetectorFixture)],
    grouper_fixtures: &[(String, GrouperFixture)],
) -> anyhow::Result<Value> {
    let mut detector = Group::default();
    let mut violations: Vec<MustNotViolation> = Vec::new();
    for (file, fixture) in detector_fixtures {
        for case in &fixture.cases {
            let country = case.country.clone().unwrap_or_else(|| "??".into());
            let predicted = baselines::detect(&case.input, case.country.as_deref());
            let gold: Vec<Span> = case
                .expected
                .iter()
                .map(|e| Span {
                    kind: e.kind.clone(),
                    start: e.start,
                    end: e.end,
                })
                .collect();
            score_spans(&mut detector, &predicted, &gold, &country);
            for m in &case.must_not {
                if predicted
                    .iter()
                    .any(|p| p.kind == m.kind && case.input[p.start..p.end] == m.text)
                {
                    violations.push(MustNotViolation {
                        fixture: file.clone(),
                        case: case.name.clone(),
                        kind: m.kind.clone(),
                        text: m.text.clone(),
                    });
                }
            }
        }
    }

    let mut parser = Group::default();
    let mut exact = 0usize;
    let mut inexact: Vec<String> = Vec::new();
    let mut parser_cases = 0usize;
    for (_, fixture) in parser_fixtures {
        for case in &fixture.cases {
            parser_cases += 1;
            let predicted = baselines::parse_address(&case.input, Some(&case.country));
            let gold: Vec<(String, usize, usize)> = case
                .components
                .iter()
                .map(|c| (c.label.clone(), c.start, c.end))
                .collect();
            let got: Vec<(String, usize, usize)> = predicted
                .iter()
                .map(|c| (c.label.clone(), c.start, c.end))
                .collect();
            if got == gold {
                exact += 1;
            } else {
                inexact.push(case.name.clone());
            }
            for g in &gold {
                parser.add(&g.0, &case.country, |c| c.gold += 1);
            }
            for p in &got {
                parser.add(&p.0, &case.country, |c| c.pred += 1);
                if gold.contains(p) {
                    parser.add(&p.0, &case.country, |c| c.tp += 1);
                }
            }
        }
    }

    let (mut correct, mut wrong, mut missed, mut total) = (0usize, 0usize, 0usize, 0usize);
    for (_, fixture) in grouper_fixtures {
        for case in &fixture.cases {
            let entities: Vec<Span> = case
                .entities
                .iter()
                .map(|e| Span {
                    kind: e.kind.clone(),
                    start: e.start,
                    end: e.end,
                })
                .collect();
            let predicted = baselines::group(&case.input, &entities);
            for (i, entity) in entities.iter().enumerate() {
                if matches!(entity.kind.as_str(), "person" | "org") {
                    continue;
                }
                total += 1;
                let gold_anchor = if case.unassigned.contains(&i) {
                    None
                } else {
                    case.contacts.iter().find_map(|c| {
                        let holds = c.addresses.contains(&i)
                            || c.emails.contains(&i)
                            || c.phones.contains(&i);
                        holds.then(|| c.person.or(c.org)).flatten()
                    })
                };
                let got_anchor = predicted.contacts.iter().find_map(|c| {
                    let holds =
                        c.addresses.contains(&i) || c.emails.contains(&i) || c.phones.contains(&i);
                    holds.then(|| c.person.or(c.org)).flatten()
                });
                match (gold_anchor, got_anchor) {
                    (a, b) if a == b => correct += 1,
                    (Some(_), None) => missed += 1,
                    _ => wrong += 1,
                }
            }
        }
    }
    let rate = |n: usize| {
        if total == 0 {
            0.0
        } else {
            round4(n as f64 / total as f64)
        }
    };

    Ok(json!({
        "system": "baseline",
        "detector": {
            "overall": detector.overall.json(),
            "boundary_accuracy": round4(boundary(&detector.overall)),
            "must_not_violations": violations.len(),
            "violations": violations.iter().map(|v| json!({
                "fixture": v.fixture, "case": v.case, "kind": v.kind, "text": v.text
            })).collect::<Vec<_>>(),
            "by_kind": map_json(&detector.by_key),
            "by_country": map_json(&detector.by_country),
        },
        "parser": {
            "overall": parser.overall.json(),
            "exact_parse": round4(if parser_cases == 0 { 0.0 } else { exact as f64 / parser_cases as f64 }),
            "cases": parser_cases,
            "not_exact": inexact,
            "by_label": map_json(&parser.by_key),
            "by_country": map_json(&parser.by_country),
        },
        "grouper": {
            "assignment_accuracy": rate(correct),
            "wrong_rate": rate(wrong),
            "unassigned_rate": rate(missed),
            "cases": total,
        },
    }))
}

/// Scores an external system over every detector case. A case the file leaves out counts
/// as predicting nothing, so skipped cases lower recall instead of disappearing from it.
fn score_external(
    external: &External,
    detector_fixtures: &[(String, DetectorFixture)],
) -> anyhow::Result<Value> {
    for prediction in &external.predictions {
        let known = detector_fixtures.iter().any(|(f, fixture)| {
            *f == prediction.fixture && fixture.cases.iter().any(|c| c.name == prediction.case)
        });
        if !known {
            bail!(
                "external predictions name unknown case `{}/{}`",
                prediction.fixture,
                prediction.case
            );
        }
    }

    let mut detector = Group::default();
    let mut missing = 0usize;
    for (file, fixture) in detector_fixtures {
        for case in &fixture.cases {
            let prediction = external
                .predictions
                .iter()
                .find(|p| p.fixture == *file && p.case == case.name);
            if prediction.is_none() {
                missing += 1;
            }
            let predicted: Vec<Span> = prediction
                .into_iter()
                .flat_map(|p| &p.entities)
                .map(|e| Span {
                    kind: e.kind.clone(),
                    start: e.start,
                    end: e.end,
                })
                .collect();
            let gold: Vec<Span> = case
                .expected
                .iter()
                .map(|e| Span {
                    kind: e.kind.clone(),
                    start: e.start,
                    end: e.end,
                })
                .collect();
            let country = case.country.clone().unwrap_or_else(|| "??".into());
            score_spans(&mut detector, &predicted, &gold, &country);
        }
    }
    Ok(json!({
        "system": external.system,
        "detector": {
            "overall": detector.overall.json(),
            "boundary_accuracy": round4(boundary(&detector.overall)),
            "cases_missing_from_predictions": missing,
            "by_kind": map_json(&detector.by_key),
            "by_country": map_json(&detector.by_country),
        },
    }))
}

/// Exact `(kind, start, end)` matches, plus boundary accuracy over same-kind overlaps.
fn score_spans(group: &mut Group, predicted: &[Span], gold: &[Span], country: &str) {
    for g in gold {
        group.add(&g.kind, country, |c| c.gold += 1);
    }
    for p in predicted {
        group.add(&p.kind, country, |c| c.pred += 1);
        if gold.contains(p) {
            group.add(&p.kind, country, |c| c.tp += 1);
        }
        let overlaps = gold
            .iter()
            .any(|g| g.kind == p.kind && p.start < g.end && g.start < p.end);
        if overlaps {
            group.add(&p.kind, country, |c| c.boundary_candidates += 1);
            if gold.contains(p) {
                group.add(&p.kind, country, |c| c.boundary_hits += 1);
            }
        }
    }
}

fn boundary(c: &Counts) -> f64 {
    if c.boundary_candidates == 0 {
        0.0
    } else {
        c.boundary_hits as f64 / c.boundary_candidates as f64
    }
}

fn pct(v: &Value, key: &str) -> String {
    v.get(key)
        .and_then(Value::as_f64)
        .map_or("-".into(), |x| format!("{:.1}%", x * 100.0))
}

fn render_markdown(tables: &[(String, Value)]) -> String {
    let mut out = String::from("# Fixture evaluation\n");
    for (system, value) in tables {
        out.push_str(&format!("\n## {system}\n"));
        for (section, key) in [("detector", "by_kind"), ("parser", "by_label")] {
            let Some(part) = value.get(section) else {
                continue;
            };
            let overall = part.get("overall").cloned().unwrap_or(Value::Null);
            out.push_str(&format!(
                "\n### {section}\n\noverall P {} R {} F1 {}",
                pct(&overall, "precision"),
                pct(&overall, "recall"),
                pct(&overall, "f1"),
            ));
            if let Some(extra) = part.get("exact_parse") {
                out.push_str(&format!(
                    ", exact parse {:.1}%",
                    extra.as_f64().unwrap_or(0.0) * 100.0
                ));
            }
            if let Some(extra) = part.get("boundary_accuracy") {
                out.push_str(&format!(
                    ", boundary {:.1}%",
                    extra.as_f64().unwrap_or(0.0) * 100.0
                ));
            }
            if let Some(extra) = part.get("must_not_violations") {
                out.push_str(&format!(", must-not violations {extra}"));
            }
            out.push('\n');
            for group in [key, "by_country"] {
                let Some(Value::Object(map)) = part.get(group) else {
                    continue;
                };
                out.push_str(&format!("\n| {group} | P | R | F1 | gold | predicted |\n| --- | ---: | ---: | ---: | ---: | ---: |\n"));
                for (name, counts) in map {
                    out.push_str(&format!(
                        "| {name} | {} | {} | {} | {} | {} |\n",
                        pct(counts, "precision"),
                        pct(counts, "recall"),
                        pct(counts, "f1"),
                        counts.get("gold").unwrap_or(&Value::Null),
                        counts.get("predicted").unwrap_or(&Value::Null),
                    ));
                }
            }
        }
        if let Some(g) = value.get("grouper") {
            out.push_str(&format!(
                "\n### grouper\n\nassignment accuracy {}, wrong {}, unassigned {}, entities {}\n",
                pct(g, "assignment_accuracy"),
                pct(g, "wrong_rate"),
                pct(g, "unassigned_rate"),
                g.get("cases").unwrap_or(&Value::Null),
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The external reader must line predictions up with the fixture case they name.
    #[test]
    fn external_predictions_score_against_the_fixture() {
        let fixtures: Vec<(String, DetectorFixture)> = vec![(
            "signatures".to_string(),
            serde_json::from_value(json!({
                "cases": [{
                    "name": "one",
                    "country": "GB",
                    "input": "Jane O'Brien\nAcme Ltd\n",
                    "expected": [
                        { "kind": "person", "text": "Jane O'Brien", "start": 0, "end": 12 },
                        { "kind": "org", "text": "Acme Ltd", "start": 13, "end": 21 }
                    ]
                }]
            }))
            .unwrap(),
        )];
        let external: External = serde_json::from_value(json!({
            "system": "stub",
            "predictions": [{
                "fixture": "signatures",
                "case": "one",
                "entities": [
                    { "kind": "person", "start": 0, "end": 12 },
                    { "kind": "org", "start": 13, "end": 20 }
                ]
            }]
        }))
        .unwrap();
        let scored = score_external(&external, &fixtures).unwrap();
        let overall = &scored["detector"]["overall"];
        assert_eq!(overall["tp"], 1);
        assert_eq!(overall["predicted"], 2);
        assert_eq!(overall["gold"], 2);
        assert_eq!(overall["precision"], 0.5);
    }

    /// Leaving a case out of the predictions must cost recall, not hide the case.
    #[test]
    fn missing_cases_count_against_recall() {
        let fixtures: Vec<(String, DetectorFixture)> = vec![(
            "signatures".to_string(),
            serde_json::from_value(json!({
                "cases": [
                    { "name": "one", "input": "Jane Smith", "expected": [ { "kind": "person", "text": "Jane Smith", "start": 0, "end": 10 } ] },
                    { "name": "two", "input": "John Doe", "expected": [ { "kind": "person", "text": "John Doe", "start": 0, "end": 8 } ] }
                ]
            }))
            .unwrap(),
        )];
        let external: External = serde_json::from_value(json!({
            "system": "stub",
            "predictions": [ { "fixture": "signatures", "case": "one", "entities": [ { "kind": "person", "start": 0, "end": 10 } ] } ]
        }))
        .unwrap();
        let scored = score_external(&external, &fixtures).unwrap();
        assert_eq!(scored["detector"]["overall"]["gold"], 2);
        assert_eq!(scored["detector"]["overall"]["recall"], 0.5);
        assert_eq!(scored["detector"]["cases_missing_from_predictions"], 1);

        let unknown: External = serde_json::from_value(json!({
            "system": "stub",
            "predictions": [ { "fixture": "signatures", "case": "nope", "entities": [] } ]
        }))
        .unwrap();
        assert!(score_external(&unknown, &fixtures).is_err());
    }

    #[test]
    fn metric_arithmetic() {
        let (p, r, f) = prf(2, 4, 3);
        assert_eq!((round4(p), round4(r), round4(f)), (0.5, 0.6667, 0.5714));
        assert_eq!(prf(0, 0, 0), (0.0, 0.0, 0.0));
        assert_eq!(prf(0, 3, 0), (0.0, 0.0, 0.0));
        assert_eq!(prf(3, 3, 3), (1.0, 1.0, 1.0));
    }
}
