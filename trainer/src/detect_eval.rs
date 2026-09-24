//! Scores the shipped detection path, `tessera::Tessera::detect` over a bundle (not the
//! trainer's f32 model), beside the deterministic baseline and external systems' predictions,
//! on a reviewed gold file or a generated split: exact and lenient (same-kind overlap) span
//! scores per kind, per slice of the cases, and per script of the spans, documents with a
//! false positive, latency, and the Markdown tables of the detector report.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;
use std::time::Instant;

use anyhow::{Context, bail};
use polars::prelude::*;
use serde::Deserialize;
use serde_json::{Value, json};
use tessera::internal::{Script, is_content, tokenize};
use tessera::{Config, Kind, Query, Tessera};

use crate::EvalArgs;
use crate::baselines::{self, Span};
use crate::eval::{ExternalEntity, prf, round4};

/// The kinds in report order.
const KINDS: [&str; 5] = ["person", "org", "address", "email", "phone"];

/// Retained-token buckets for latency, by lower bound.
const LATENCY_BUCKETS: [(usize, &str); 4] = [
    (0, "<256"),
    (256, "256–1023"),
    (1024, "1024–4095"),
    (4096, "≥4096"),
];

/// One scored document: its text, gold spans, and the slices it belongs to.
#[derive(Debug, Clone)]
pub struct Case {
    pub name: String,
    pub input: String,
    pub expected: Vec<Span>,
    pub country: String,
    /// Slice name to value, such as `doc_type` → `notice` or `family` → `signature`.
    pub slices: BTreeMap<&'static str, String>,
}

#[derive(Deserialize)]
struct GoldLine {
    name: String,
    input: String,
    expected: Vec<GoldSpan>,
    country: String,
    doc_type: String,
}

#[derive(Deserialize)]
struct GoldSpan {
    kind: String,
    start: usize,
    end: usize,
    #[serde(default)]
    text: Option<String>,
}

/// Reads `gold.jsonl`, one case per line, checking that every span's `text` is its slice.
pub fn load_gold(path: &Path) -> anyhow::Result<Vec<Case>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let mut cases = Vec::new();
    for (i, line) in text
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
    {
        let g: GoldLine = serde_json::from_str(line)
            .with_context(|| format!("{}:{}: not a gold case", path.display(), i + 1))?;
        for s in &g.expected {
            let slice = g.input.get(s.start..s.end);
            if slice.is_none() || s.text.as_deref().is_some_and(|t| Some(t) != slice) {
                bail!(
                    "{}: `{}` span {}..{} does not slice to its text",
                    path.display(),
                    g.name,
                    s.start,
                    s.end
                );
            }
        }
        cases.push(Case {
            expected: g
                .expected
                .into_iter()
                .map(|s| Span {
                    kind: s.kind,
                    start: s.start,
                    end: s.end,
                })
                .collect(),
            slices: BTreeMap::from([("country", g.country.clone()), ("doc_type", g.doc_type)]),
            name: g.name,
            input: g.input,
            country: g.country,
        });
    }
    Ok(cases)
}

#[derive(Deserialize)]
struct SyntheticGold {
    kind: String,
    start: usize,
    end: usize,
}

/// Reads a generated detector split with its subset, family, country, and category.
pub fn load_synthetic(path: &Path) -> anyhow::Result<Vec<Case>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let df = ParquetReader::new(file).finish()?;
    let col = |name: &str| -> anyhow::Result<StringChunked> {
        Ok(df
            .column(name)
            .ok()
            .with_context(|| format!("{} has no {name} column", path.display()))?
            .str()?
            .clone())
    };
    let (text, entities) = (col("text")?, col("entities_json")?);
    let (subset, family, country, category) = (
        col("subset")?,
        col("family")?,
        col("country")?,
        col("category")?,
    );
    let at = |c: &StringChunked, i: usize| c.get(i).unwrap_or_default().to_string();
    let mut cases = Vec::with_capacity(df.height());
    for i in 0..df.height() {
        let gold: Vec<SyntheticGold> =
            serde_json::from_str(entities.get(i).context("null entities")?)?;
        cases.push(Case {
            name: format!("{i}"),
            input: at(&text, i),
            expected: gold
                .into_iter()
                .map(|g| Span {
                    kind: g.kind,
                    start: g.start,
                    end: g.end,
                })
                .collect(),
            country: at(&country, i),
            slices: BTreeMap::from([
                ("subset", at(&subset, i)),
                ("family", at(&family, i)),
                ("country", at(&country, i)),
                ("category", at(&category, i)),
            ]),
        });
    }
    Ok(cases)
}

/// What one system predicted for every case, in case order.
pub struct Predictions {
    pub system: String,
    pub spans: Vec<Vec<Span>>,
}

/// The library's `detect` over every case, with the case's country as the phone hint, and the
/// time each call took by retained-token count.
pub fn predict_library(
    bundle: &[u8],
    cases: &[Case],
) -> anyhow::Result<(Predictions, Vec<(usize, f64)>)> {
    let tessera = Tessera::load(
        bundle,
        Config {
            kinds: Kind::all(),
            expected_checksum: None,
        },
    )
    .map_err(|e| anyhow::anyhow!("loading the bundle: {e}"))?;
    let mut spans = Vec::with_capacity(cases.len());
    let mut timings = Vec::with_capacity(cases.len());
    for case in cases {
        let hints: Vec<&str> = hint(&case.country).into_iter().collect();
        let query = Query {
            country_hint: &hints,
            ..Query::default()
        };
        let started = Instant::now();
        let found = tessera
            .detect(&case.input, &query)
            .map_err(|e| anyhow::anyhow!("{}: {e}", case.name))?;
        let ms = started.elapsed().as_secs_f64() * 1000.0;
        let tokens = tokenize(&case.input)
            .iter()
            .filter(|t| is_content(t))
            .count();
        timings.push((tokens, ms));
        spans.push(
            found
                .iter()
                .map(|e| Span {
                    kind: e.kind.as_str().to_string(),
                    start: e.start,
                    end: e.end,
                })
                .collect(),
        );
    }
    Ok((
        Predictions {
            system: "tessera".into(),
            spans,
        },
        timings,
    ))
}

/// A country code usable as a phone hint: two uppercase letters, not `EU`.
fn hint(country: &str) -> Option<&str> {
    (country.len() == 2 && country.chars().all(|c| c.is_ascii_uppercase()) && country != "EU")
        .then_some(country)
}

/// Milestone 1's deterministic detector over every case.
pub fn predict_baseline(cases: &[Case]) -> Predictions {
    Predictions {
        system: "baseline".into(),
        spans: cases
            .iter()
            .map(|c| baselines::detect(&c.input, hint(&c.country)))
            .collect(),
    }
}

#[derive(Deserialize)]
struct ExternalFile {
    system: String,
    predictions: Vec<ExternalCase>,
}

#[derive(Deserialize)]
struct ExternalCase {
    name: String,
    entities: Vec<ExternalEntity>,
}

/// An external system's predictions by case name; a case the file leaves out predicts nothing,
/// and a name that is not a case is an error.
pub fn load_predictions(path: &Path, cases: &[Case]) -> anyhow::Result<Predictions> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let file: ExternalFile =
        serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
    let mut by_name: BTreeMap<&str, Vec<Span>> = BTreeMap::new();
    for p in &file.predictions {
        if !cases.iter().any(|c| c.name == p.name) {
            bail!("{}: unknown case `{}`", path.display(), p.name);
        }
        by_name.insert(
            &p.name,
            p.entities
                .iter()
                .map(|e| Span {
                    kind: e.kind.clone(),
                    start: e.start,
                    end: e.end,
                })
                .collect(),
        );
    }
    Ok(Predictions {
        system: file.system,
        spans: cases
            .iter()
            .map(|c| by_name.remove(c.name.as_str()).unwrap_or_default())
            .collect(),
    })
}

/// Exact and lenient match counts. A lenient match is a same-kind overlap; boundary accuracy
/// is the share of predictions with a lenient match that also match exactly.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct Tally {
    pub gold: usize,
    pub predicted: usize,
    pub exact: usize,
    /// Predictions that overlap a gold span of their kind.
    pub lenient_predicted: usize,
    /// Gold spans overlapped by a prediction of their kind.
    pub lenient_gold: usize,
}

impl Tally {
    fn add(&mut self, other: &Tally) {
        self.gold += other.gold;
        self.predicted += other.predicted;
        self.exact += other.exact;
        self.lenient_predicted += other.lenient_predicted;
        self.lenient_gold += other.lenient_gold;
    }

    pub fn exact_prf(&self) -> (f64, f64, f64) {
        prf(self.exact, self.predicted, self.gold)
    }

    /// Lenient precision and recall count different sides, so F1 is built from the two.
    pub fn lenient_prf(&self) -> (f64, f64, f64) {
        let (p, _, _) = prf(self.lenient_predicted, self.predicted, self.gold);
        let (_, r, _) = prf(self.lenient_gold, self.predicted, self.gold);
        let f = if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        };
        (p, r, f)
    }

    pub fn boundary_accuracy(&self) -> f64 {
        if self.lenient_predicted == 0 {
            0.0
        } else {
            self.exact as f64 / self.lenient_predicted as f64
        }
    }

    fn json(&self) -> Value {
        let (p, r, f) = self.exact_prf();
        let (lp, lr, lf) = self.lenient_prf();
        json!({
            "exact": { "precision": round4(p), "recall": round4(r), "f1": round4(f) },
            "lenient": { "precision": round4(lp), "recall": round4(lr), "f1": round4(lf) },
            "boundary_accuracy": round4(self.boundary_accuracy()),
            "gold": self.gold,
            "predicted": self.predicted,
        })
    }
}

fn overlaps(a: &Span, b: &Span) -> bool {
    a.kind == b.kind && a.start < b.end && b.start < a.end
}

/// Per-kind tallies of one document, keyed by the script of each span as well.
fn score_doc(text: &str, gold: &[Span], predicted: &[Span]) -> Vec<(String, String, Tally)> {
    let mut out: Vec<(String, String, Tally)> = Vec::new();
    let mut bump = |kind: &str, script: String, f: &dyn Fn(&mut Tally)| match out
        .iter_mut()
        .find(|(k, s, _)| k == kind && *s == script)
    {
        Some((_, _, t)) => f(t),
        None => {
            let mut t = Tally::default();
            f(&mut t);
            out.push((kind.to_string(), script, t));
        }
    };
    for g in gold {
        let script = span_script(text.get(g.start..g.end).unwrap_or("")).to_string();
        bump(&g.kind, script.clone(), &|t| t.gold += 1);
        if predicted.iter().any(|p| overlaps(p, g)) {
            bump(&g.kind, script, &|t| t.lenient_gold += 1);
        }
    }
    for p in predicted {
        let script = span_script(text.get(p.start..p.end).unwrap_or("")).to_string();
        bump(&p.kind, script.clone(), &|t| t.predicted += 1);
        if gold.contains(p) {
            bump(&p.kind, script.clone(), &|t| t.exact += 1);
        }
        if gold.iter().any(|g| overlaps(p, g)) {
            bump(&p.kind, script, &|t| t.lenient_predicted += 1);
        }
    }
    out
}

/// The script of most of a span's letter-bearing tokens: `latin`, `georgian`, `japanese`
/// (kanji and kana), `cyrillic`, or `other`; `none` for a span with no letters.
pub fn span_script(text: &str) -> &'static str {
    let mut counts: BTreeMap<&'static str, usize> = BTreeMap::new();
    for t in tokenize(text) {
        let slice = &text[t.start..t.end];
        if !slice.chars().any(char::is_alphabetic) {
            continue;
        }
        let name = match t.script {
            Script::Latin => "latin",
            Script::Georgian => "georgian",
            Script::Han | Script::Hiragana | Script::Katakana => "japanese",
            Script::Cyrillic => "cyrillic",
            _ => "other",
        };
        *counts.entry(name).or_default() += 1;
    }
    counts
        .into_iter()
        .max_by_key(|&(name, n)| (n, std::cmp::Reverse(name)))
        .map_or("none", |(name, _)| name)
}

/// One system's scores over all cases: per kind, per kind within each slice value, per kind
/// within each script, and the share of documents with at least one false positive per kind.
pub fn score(cases: &[Case], predictions: &Predictions) -> Value {
    let mut by_kind: BTreeMap<String, Tally> = BTreeMap::new();
    let mut overall = Tally::default();
    let mut by_slice: BTreeMap<&'static str, BTreeMap<String, BTreeMap<String, Tally>>> =
        BTreeMap::new();
    let mut by_script: BTreeMap<String, BTreeMap<String, Tally>> = BTreeMap::new();
    let mut fp_docs: BTreeMap<String, usize> = BTreeMap::new();
    for (case, predicted) in cases.iter().zip(&predictions.spans) {
        for (kind, script, t) in score_doc(&case.input, &case.expected, predicted) {
            overall.add(&t);
            by_kind.entry(kind.clone()).or_default().add(&t);
            by_script
                .entry(script)
                .or_default()
                .entry(kind.clone())
                .or_default()
                .add(&t);
            for (slice, value) in &case.slices {
                by_slice
                    .entry(slice)
                    .or_default()
                    .entry(value.clone())
                    .or_default()
                    .entry(kind.clone())
                    .or_default()
                    .add(&t);
            }
        }
        for kind in KINDS {
            let false_positive = predicted
                .iter()
                .any(|p| p.kind == kind && !case.expected.iter().any(|g| overlaps(p, g)));
            if false_positive {
                *fp_docs.entry(kind.to_string()).or_default() += 1;
            }
        }
    }
    let kinds_json = |m: &BTreeMap<String, Tally>| {
        Value::Object(m.iter().map(|(k, t)| (k.clone(), t.json())).collect())
    };
    let docs = cases.len().max(1) as f64;
    json!({
        "system": predictions.system,
        "cases": cases.len(),
        "overall": overall.json(),
        "by_kind": kinds_json(&by_kind),
        "by_slice": Value::Object(by_slice.iter().map(|(slice, values)| {
            (slice.to_string(), Value::Object(values.iter().map(|(v, kinds)| (v.clone(), kinds_json(kinds))).collect()))
        }).collect()),
        "by_script": Value::Object(by_script.iter().map(|(s, kinds)| (s.clone(), kinds_json(kinds))).collect()),
        "doc_false_positive_rate": Value::Object(KINDS.iter().map(|k| {
            (k.to_string(), json!(round4(*fp_docs.get(*k).unwrap_or(&0) as f64 / docs)))
        }).collect()),
    })
}

/// The median time per retained-token bucket, in milliseconds, with the bucket's case count.
pub fn latency(timings: &[(usize, f64)]) -> Value {
    let mut out = serde_json::Map::new();
    for (i, (low, name)) in LATENCY_BUCKETS.iter().enumerate() {
        let high = LATENCY_BUCKETS.get(i + 1).map_or(usize::MAX, |b| b.0);
        let mut ms: Vec<f64> = timings
            .iter()
            .filter(|(n, _)| (*low..high).contains(n))
            .map(|(_, ms)| *ms)
            .collect();
        ms.sort_by(f64::total_cmp);
        let median = (!ms.is_empty()).then(|| round4(ms[ms.len() / 2]));
        out.insert(
            name.to_string(),
            json!({ "cases": ms.len(), "median_ms": median }),
        );
    }
    Value::Object(out)
}

fn pct(x: f64) -> String {
    format!("{:.1}", x * 100.0)
}

fn cell(scores: &Value, path: &[&str], metric: &str, side: &str) -> String {
    let mut v = scores;
    for key in path {
        v = &v[*key];
    }
    match &v[side][metric] {
        Value::Number(n) => pct(n.as_f64().unwrap_or(0.0)),
        _ => "–".into(),
    }
}

/// A table of exact and lenient P/R/F1 per kind for each system, restricted to `path`
/// (`["by_kind"]`, or `["by_slice", "country", "US"]` and so on).
pub fn kind_table(systems: &[Value], path: &[&str]) -> String {
    let mut out = String::from(
        "| kind | system | gold | predicted | exact P | exact R | exact F1 | lenient P | lenient R | lenient F1 | boundary |\n| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n",
    );
    for kind in KINDS {
        for s in systems {
            let mut v = s;
            for key in path {
                v = &v[*key];
            }
            let t = &v[kind];
            if t.is_null() {
                continue;
            }
            let full: Vec<&str> = path.iter().copied().chain([kind]).collect();
            out.push_str(&format!(
                "| {kind} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
                s["system"].as_str().unwrap_or("?"),
                t["gold"],
                t["predicted"],
                cell(s, &full, "precision", "exact"),
                cell(s, &full, "recall", "exact"),
                cell(s, &full, "f1", "exact"),
                cell(s, &full, "precision", "lenient"),
                cell(s, &full, "recall", "lenient"),
                cell(s, &full, "f1", "lenient"),
                t["boundary_accuracy"].as_f64().map_or("–".into(), pct),
            ));
        }
    }
    out
}

/// One `## title` section per value of `slice`, each with its kind table.
pub fn slice_sections(systems: &[Value], group: &str, slice: &str) -> String {
    let mut values: Vec<String> = systems
        .iter()
        .filter_map(|s| s[group][slice].as_object())
        .flat_map(|m| m.keys().cloned())
        .collect();
    values.sort();
    values.dedup();
    let mut out = String::new();
    for v in values {
        out.push_str(&format!("\n#### {slice} = {v}\n\n"));
        out.push_str(&kind_table(systems, &[group, slice, &v]));
    }
    out
}

/// Scripts are not a slice of the case but of each span, so they are keyed on their own.
pub fn script_sections(systems: &[Value]) -> String {
    let mut scripts: Vec<String> = systems
        .iter()
        .filter_map(|s| s["by_script"].as_object())
        .flat_map(|m| m.keys().cloned())
        .collect();
    scripts.sort();
    scripts.dedup();
    let mut out = String::new();
    for s in scripts {
        out.push_str(&format!("\n#### script = {s}\n\n"));
        out.push_str(&kind_table(systems, &["by_script", &s]));
    }
    out
}

/// The document-level false-positive table, one row per system.
pub fn fp_table(systems: &[Value]) -> String {
    let mut out = format!(
        "| system | {} |\n| --- |{}\n",
        KINDS.join(" | "),
        " ---: |".repeat(KINDS.len())
    );
    for s in systems {
        let row: Vec<String> = KINDS
            .iter()
            .map(|k| {
                s["doc_false_positive_rate"][*k]
                    .as_f64()
                    .map_or("–".into(), pct)
            })
            .collect();
        out.push_str(&format!(
            "| {} | {} |\n",
            s["system"].as_str().unwrap_or("?"),
            row.join(" | ")
        ));
    }
    out
}

/// The latency table from [`latency`]'s output.
pub fn latency_table(latency: &Value) -> String {
    let mut out = String::from("| retained tokens | cases | median ms |\n| --- | ---: | ---: |\n");
    for (_, name) in LATENCY_BUCKETS {
        let b = &latency[name];
        out.push_str(&format!(
            "| {name} | {} | {} |\n",
            b["cases"],
            b["median_ms"]
                .as_f64()
                .map_or("–".into(), |m| format!("{m:.2}"))
        ));
    }
    out
}

/// `trainer eval --gold`: the shipped `detect`, and any baseline or external predictions, on
/// the reviewed documents, written as JSON under `<out>/eval/review.json` and, with `--report`,
/// as the report's review-set tables.
pub fn run_gold(args: &EvalArgs) -> anyhow::Result<()> {
    let gold_path = args.gold.as_deref().context("--gold is required")?;
    let cases = load_gold(gold_path)?;
    let bundle = std::fs::read(&args.bundle)
        .with_context(|| format!("reading {}", args.bundle.display()))?;
    let (tessera, timings) = predict_library(&bundle, &cases)?;
    let mut systems = vec![score(&cases, &tessera)];
    if args.baseline {
        systems.push(score(&cases, &predict_baseline(&cases)));
    }
    for path in &args.predictions {
        systems.push(score(&cases, &load_predictions(path, &cases)?));
    }
    let latency = latency(&timings);
    write_json(
        &args.out.join("eval").join("review.json"),
        &json!({ "gold": gold_path, "systems": systems, "latency": latency }),
    )?;
    let tables = [
        (
            "Review set by kind".to_string(),
            kind_table(&systems, &["by_kind"]),
        ),
        (
            "Review set by country".to_string(),
            slice_sections(&systems, "by_slice", "country"),
        ),
        (
            "Review set by script".to_string(),
            script_sections(&systems),
        ),
        (
            "Review set by document type".to_string(),
            slice_sections(&systems, "by_slice", "doc_type"),
        ),
        (
            "Documents with a false positive, per kind".to_string(),
            fp_table(&systems),
        ),
        (
            "Latency of `detect`, native release build".to_string(),
            latency_table(&latency),
        ),
    ];
    finish(args.report.as_deref(), &tables)
}

/// `trainer eval --run <detector run> --split test`: the shipped `detect` on a generated split,
/// by subset, family, country, and category.
pub fn run_split(args: &EvalArgs, run: &Path) -> anyhow::Result<()> {
    let cfg = crate::config::load(&run.join("config.toml"))?;
    let path = Path::new(&cfg.data.processed).join(format!("{}.parquet", args.split));
    let cases = load_synthetic(&path)?;
    let bundle = std::fs::read(&args.bundle)
        .with_context(|| format!("reading {}", args.bundle.display()))?;
    let (tessera, _) = predict_library(&bundle, &cases)?;
    let systems = vec![score(&cases, &tessera)];
    write_json(
        &run.join("eval")
            .join(format!("{}-shipped.json", args.split)),
        &json!({ "split": args.split, "systems": systems }),
    )?;
    let tables = [
        (
            format!("Synthetic {} by kind", args.split),
            kind_table(&systems, &["by_kind"]),
        ),
        (
            format!("Synthetic {} by subset", args.split),
            slice_sections(&systems, "by_slice", "subset"),
        ),
        (
            format!("Synthetic {} by family", args.split),
            slice_sections(&systems, "by_slice", "family"),
        ),
        (
            format!("Synthetic {} by country", args.split),
            slice_sections(&systems, "by_slice", "country"),
        ),
        (
            format!("Synthetic {} by category", args.split),
            slice_sections(&systems, "by_slice", "category"),
        ),
    ];
    finish(args.report.as_deref(), &tables)
}

fn write_json(path: &Path, value: &Value) -> anyhow::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    std::fs::write(path, serde_json::to_string_pretty(value)? + "\n")?;
    eprintln!("wrote {}", path.display());
    Ok(())
}

/// Prints the tables, and writes them to `report` when given.
fn finish(report: Option<&Path>, tables: &[(String, String)]) -> anyhow::Result<()> {
    let markdown: String = tables
        .iter()
        .map(|(title, body)| format!("### {title}\n\n{body}\n"))
        .collect();
    print!("{markdown}");
    if let Some(path) = report {
        std::fs::write(path, &markdown).with_context(|| format!("writing {}", path.display()))?;
        eprintln!("wrote {}", path.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(kind: &str, start: usize, end: usize) -> Span {
        Span {
            kind: kind.into(),
            start,
            end,
        }
    }

    fn total(rows: Vec<(String, String, Tally)>) -> Tally {
        let mut t = Tally::default();
        for (_, _, r) in rows {
            t.add(&r);
        }
        t
    }

    #[test]
    fn exact_lenient_and_boundary_on_three_gold_spans() {
        let text = "Anna Schmidt, Acme GmbH, 1 Main Street";
        let gold = [
            span("person", 0, 12),
            span("org", 14, 23),
            span("address", 25, 38),
        ];
        let predicted = [span("person", 0, 12), span("org", 14, 18)];
        let t = total(score_doc(text, &gold, &predicted));
        let round = |(p, r, f): (f64, f64, f64)| (round4(p), round4(r), round4(f));
        assert_eq!(round(t.exact_prf()).0, 0.5);
        assert_eq!(round(t.exact_prf()).1, 0.3333);
        assert_eq!(round(t.lenient_prf()).0, 1.0);
        assert_eq!(round(t.lenient_prf()).1, 0.6667);
        assert_eq!(t.boundary_accuracy(), 0.5);
    }

    #[test]
    fn the_script_of_a_span_is_its_majority_script() {
        assert_eq!(span_script("ნინო ბერიძე"), "georgian");
        assert_eq!(span_script("Nino Beridze"), "latin");
        assert_eq!(span_script("株式会社サクラ物流"), "japanese");
        assert_eq!(span_script("+44 20 7946 0321"), "none");
    }

    #[test]
    fn false_positive_documents_and_slices_are_counted_per_kind() {
        let cases = vec![
            Case {
                name: "a".into(),
                input: "Anna Schmidt".into(),
                expected: vec![span("person", 0, 12)],
                country: "DE".into(),
                slices: BTreeMap::from([("doc_type", "notice".to_string())]),
            },
            Case {
                name: "b".into(),
                input: "Seite 1 von 2".into(),
                expected: vec![],
                country: "DE".into(),
                slices: BTreeMap::from([("doc_type", "directory".to_string())]),
            },
        ];
        let predictions = Predictions {
            system: "stub".into(),
            spans: vec![vec![span("person", 0, 12)], vec![span("person", 0, 5)]],
        };
        let s = score(&cases, &predictions);
        assert_eq!(s["doc_false_positive_rate"]["person"], 0.5);
        assert_eq!(
            s["by_slice"]["doc_type"]["notice"]["person"]["exact"]["f1"],
            1.0
        );
        assert_eq!(
            s["by_slice"]["doc_type"]["directory"]["person"]["predicted"],
            1
        );
        assert_eq!(s["by_kind"]["person"]["exact"]["precision"], 0.5);
    }

    #[test]
    fn latency_takes_the_median_per_bucket() {
        let l = latency(&[(10, 1.0), (20, 3.0), (30, 2.0), (300, 7.0)]);
        assert_eq!(l["<256"]["median_ms"], 2.0);
        assert_eq!(l["256–1023"]["cases"], 1);
        assert!(l["≥4096"]["median_ms"].is_null());
    }

    #[test]
    fn hints_are_region_codes_only() {
        assert_eq!(hint("GB"), Some("GB"));
        assert_eq!(hint("EU"), None);
        assert_eq!(hint("unknown"), None);
    }
}
