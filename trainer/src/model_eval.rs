//! Scoring the learned parser and the deterministic baseline on prepared shards with the
//! same code: component precision, recall, and F1 by label and by country, exact-parse rate,
//! character accuracy, confidence calibration, and the worst errors for review.

use crate::eval::round4;
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::Context;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use burn::tensor::activation::softmax;
use serde_json::{Value, json};
use tessera::AddressLabel;

use crate::data::{LabelledExample, Span, Split, read_shard};
use crate::dataset::{Encoded, ParserBatcher, encode};
use crate::net::TaggerNet;

/// Match counts for one group of components.
#[derive(Debug, Default, Clone, Copy)]
pub struct Counts {
    /// Predicted components that match a gold one exactly.
    pub tp: usize,
    /// Predicted components.
    pub pred: usize,
    /// Gold components.
    pub gold: usize,
}

impl Counts {
    /// Component F1 from the counts.
    pub fn f1(&self) -> f64 {
        crate::eval::prf(self.tp, self.pred, self.gold).2
    }

    fn json(&self) -> Value {
        let (p, r, f) = crate::eval::prf(self.tp, self.pred, self.gold);
        json!({ "precision": round4(p), "recall": round4(r), "f1": round4(f), "tp": self.tp, "predicted": self.pred, "gold": self.gold })
    }
}

/// Component-level scores over a set of examples.
#[derive(Debug, Default, Clone)]
pub struct ParserScores {
    pub overall: Counts,
    pub per_label: BTreeMap<String, Counts>,
    pub per_country: BTreeMap<String, Counts>,
    /// `(exact, rows)` per country.
    pub exact_per_country: BTreeMap<String, (usize, usize)>,
    pub exact: usize,
    pub rows: usize,
    /// Non-whitespace characters whose predicted label equals the gold label, `O` included.
    pub char_correct: usize,
    pub char_total: usize,
}

impl ParserScores {
    /// Share of examples whose predicted components all match the gold ones.
    pub fn exact_rate(&self) -> f64 {
        self.exact as f64 / self.rows.max(1) as f64
    }

    /// The scores as the eval report JSON.
    pub fn json(&self) -> Value {
        let map = |m: &BTreeMap<String, Counts>| {
            Value::Object(m.iter().map(|(k, v)| (k.clone(), v.json())).collect())
        };
        let exact_by_country: serde_json::Map<String, Value> = self
            .exact_per_country
            .iter()
            .map(|(c, (e, n))| (c.clone(), json!(round4(*e as f64 / (*n).max(1) as f64))))
            .collect();
        let rows_by_country: serde_json::Map<String, Value> = self
            .exact_per_country
            .iter()
            .map(|(c, (_, n))| (c.clone(), json!(n)))
            .collect();
        json!({
            "overall": self.overall.json(),
            "exact_parse": round4(self.exact_rate()),
            "char_accuracy": round4(self.char_correct as f64 / self.char_total.max(1) as f64),
            "rows": self.rows,
            "per_label": map(&self.per_label),
            "per_country": map(&self.per_country),
            "exact_per_country": exact_by_country,
            "rows_per_country": rows_by_country,
        })
    }
}

fn label_at(spans: &[Span], byte: u32) -> Option<AddressLabel> {
    spans
        .iter()
        .find(|s| s.start <= byte && byte < s.end)
        .map(|s| s.label)
}

/// Scores predictions against the gold spans of `examples`, aligned by index.
pub fn score(examples: &[&LabelledExample], preds: &[Vec<Span>]) -> ParserScores {
    let mut out = ParserScores::default();
    for (ex, pred) in examples.iter().zip(preds) {
        out.rows += 1;
        let exact = *pred == ex.spans;
        if exact {
            out.exact += 1;
        }
        let e = out.exact_per_country.entry(ex.country.clone()).or_default();
        e.1 += 1;
        if exact {
            e.0 += 1;
        }
        let mut bump = |label: AddressLabel, f: &dyn Fn(&mut Counts)| {
            f(&mut out.overall);
            f(out.per_label.entry(label.as_str().to_string()).or_default());
            f(out.per_country.entry(ex.country.clone()).or_default());
        };
        for g in &ex.spans {
            bump(g.label, &|c| c.gold += 1);
        }
        for p in pred {
            bump(p.label, &|c| c.pred += 1);
            if ex.spans.contains(p) {
                bump(p.label, &|c| c.tp += 1);
            }
        }
        for (b, ch) in ex.text.char_indices() {
            if ch.is_whitespace() {
                continue;
            }
            out.char_total += 1;
            if label_at(&ex.spans, b as u32) == label_at(pred, b as u32) {
                out.char_correct += 1;
            }
        }
    }
    out
}

/// Greedy decoding with transition masking: at each position the most probable label among
/// those legal after the previous choice, where `I-x` may only follow `B-x` or `I-x`.
/// The library's `bio::decode_probs` is the same procedure, NaN handling included.
pub fn decode_probs(probs: &[Vec<f32>]) -> Vec<(u8, f32)> {
    let mut out = Vec::with_capacity(probs.len());
    let mut prev: u8 = 0;
    for row in probs {
        let mut best = (0u8, f32::NEG_INFINITY);
        for (id, &p) in row.iter().enumerate() {
            let id = id as u8;
            let p = if p.is_nan() { 0.0 } else { p };
            let is_inside = id != 0 && (id - 1) % 2 == 1;
            let legal = !is_inside || (prev != 0 && (prev - 1) / 2 == (id - 1) / 2);
            if legal && p > best.1 {
                best = (id, p);
            }
        }
        out.push(best);
        prev = best.0;
    }
    out
}

/// Spans from decoded labels, each with the mean probability of its tokens as confidence.
pub fn decode_labels_conf(token_spans: &[(u32, u32)], decoded: &[(u8, f32)]) -> Vec<(Span, f32)> {
    let ids: Vec<u8> = decoded.iter().map(|d| d.0).collect();
    let spans = crate::dataset::decode_labels(token_spans, &ids);
    spans
        .into_iter()
        .map(|s| {
            let probs: Vec<f32> = token_spans
                .iter()
                .zip(decoded)
                .filter(|((a, b), _)| *a >= s.start && *b <= s.end)
                .map(|(_, d)| d.1)
                .collect();
            let conf = probs.iter().sum::<f32>() / probs.len().max(1) as f32;
            (s, conf)
        })
        .collect()
}

/// Ten equal-width bins of component confidence: count, mean confidence, and accuracy, plus
/// the count-weighted expected calibration error.
#[derive(Debug, Clone)]
pub struct Calibration {
    pub bins: [(usize, f32, f32); 10],
    pub ece: f32,
}

fn calibration_of(items: impl Iterator<Item = (f32, bool)>) -> Calibration {
    let mut sums = [(0usize, 0f64, 0usize); 10];
    let mut total = 0usize;
    for (conf, correct) in items {
        let bin = ((conf * 10.0) as usize).min(9);
        sums[bin].0 += 1;
        sums[bin].1 += f64::from(conf);
        sums[bin].2 += usize::from(correct);
        total += 1;
    }
    let mut bins = [(0usize, 0f32, 0f32); 10];
    let mut ece = 0f64;
    for (i, (n, conf, correct)) in sums.iter().enumerate() {
        if *n == 0 {
            continue;
        }
        let mean = conf / *n as f64;
        let acc = *correct as f64 / *n as f64;
        bins[i] = (*n, mean as f32, acc as f32);
        ece += (*n as f64 / total.max(1) as f64) * (mean - acc).abs();
    }
    Calibration {
        bins,
        ece: ece as f32,
    }
}

/// Encodes examples, keeping only those that encode, so predictions stay aligned with them.
pub fn encode_examples<'a>(
    examples: &'a [LabelledExample],
    fc: &tessera::internal::FeatureConfig,
) -> (Vec<&'a LabelledExample>, Vec<Encoded>) {
    let mut kept = Vec::new();
    let mut encoded = Vec::new();
    for ex in examples {
        if let Ok(mut e) = encode(&ex.text, &ex.spans, fc) {
            e.country = ex.country.clone();
            kept.push(ex);
            encoded.push(e);
        }
    }
    (kept, encoded)
}

/// Runs the model over encoded items in order and decodes spans with confidences.
pub fn predict<B: Backend>(
    model: &TaggerNet<B>,
    items: &[Encoded],
    batch_size: usize,
    device: &B::Device,
) -> Vec<Vec<(Span, f32)>> {
    let mut out = Vec::with_capacity(items.len());
    for chunk in items.chunks(batch_size.max(1)) {
        let batch: crate::dataset::ParserBatch<B> = ParserBatcher.batch(chunk.to_vec(), device);
        let lengths = batch.lengths.clone();
        let logits = model.forward(
            batch.ngram_ids,
            batch.script,
            batch.shape,
            batch.flags,
            batch.mask.clone(),
        );
        let [_, l, c] = logits.dims();
        let probs: Vec<f32> = softmax(logits, 2)
            .into_data()
            .to_vec()
            .expect("probabilities are f32");
        for (i, item) in chunk.iter().enumerate() {
            let rows: Vec<Vec<f32>> = (0..lengths[i])
                .map(|t| probs[(i * l + t) * c..(i * l + t + 1) * c].to_vec())
                .collect();
            out.push(decode_labels_conf(&item.token_spans, &decode_probs(&rows)));
        }
    }
    out
}

/// Validation scores used for early stopping during training.
pub struct QuickScores {
    /// Component F1 over all rows.
    pub component_f1: f64,
    /// Share of rows parsed exactly.
    pub exact: f64,
}

/// Component F1 and exact-parse rate on encoded validation rows, for early stopping.
pub fn quick_score<B: Backend>(
    model: &TaggerNet<B>,
    kept: &[&LabelledExample],
    items: &[Encoded],
    batch_size: usize,
    device: &B::Device,
) -> QuickScores {
    let preds: Vec<Vec<Span>> = predict(model, items, batch_size, device)
        .into_iter()
        .map(|p| p.into_iter().map(|(s, _)| s).collect())
        .collect();
    let s = score(kept, &preds);
    QuickScores {
        component_f1: s.overall.f1(),
        exact: s.exact_rate(),
    }
}

/// The baseline's components for one example, in the same span type.
fn baseline_spans(ex: &LabelledExample) -> Vec<Span> {
    crate::baselines::parse_address(&ex.text, Some(&ex.country))
        .into_iter()
        .filter_map(|c| {
            Some(Span {
                label: AddressLabel::from_str_label(&c.label)?,
                start: c.start as u32,
                end: c.end as u32,
            })
        })
        .collect()
}

/// Scores the exported bundle as users get it: every original row of `examples` through the
/// library's `parse_address`, confidence policy included. Components the policy leaves
/// `unknown` count as not predicted.
pub fn score_shipped(
    bundle: &[u8],
    checksum: Option<&str>,
    examples: &[LabelledExample],
) -> anyhow::Result<ParserScores> {
    let originals: Vec<&LabelledExample> = examples.iter().filter(|e| !e.augmented).collect();
    let preds = shipped_spans(bundle, checksum, &originals)?;
    Ok(score(&originals, &preds))
}

fn shipped_spans(
    bundle: &[u8],
    checksum: Option<&str>,
    examples: &[&LabelledExample],
) -> anyhow::Result<Vec<Vec<Span>>> {
    let tessera = tessera::Tessera::load(
        bundle,
        tessera::Config {
            kinds: tessera::Kind::Address.into(),
            expected_checksum: checksum,
        },
    )?;
    let query = tessera::Query {
        country_hint: &[],
        ..tessera::Query::default()
    };
    examples
        .iter()
        .map(|ex| {
            let entity = tessera.parse_address(&ex.text, &query)?;
            Ok(entity
                .components
                .iter()
                .filter(|c| c.label != AddressLabel::Unknown)
                .map(|c| Span {
                    label: c.label,
                    start: c.start as u32,
                    end: c.end as u32,
                })
                .collect())
        })
        .collect()
}

/// Reviewed real addresses in parser fixture files in `dir`, with their case names.
fn address_examples(dir: &Path) -> anyhow::Result<(Vec<String>, Vec<LabelledExample>)> {
    let fixtures: Vec<(String, crate::fixtures::ParserFixture)> = crate::fixtures::load_dir(dir)?;
    let cases: Vec<&crate::fixtures::ParserCase> =
        fixtures.iter().flat_map(|(_, f)| &f.cases).collect();
    let examples = cases
        .iter()
        .enumerate()
        .map(|(i, case)| {
            let spans = case
                .components
                .iter()
                .map(|c| {
                    Ok(Span {
                        label: AddressLabel::from_str_label(&c.label)
                            .with_context(|| format!("{}: label {}", case.name, c.label))?,
                        start: u32::try_from(c.start)?,
                        end: u32::try_from(c.end)?,
                    })
                })
                .collect::<anyhow::Result<Vec<Span>>>()?;
            Ok(LabelledExample {
                id: i as u64,
                group_id: i as u64,
                country: case.country.clone(),
                language: String::new(),
                text: case.input.clone(),
                spans,
                split: Split::Test,
                augmented: false,
            })
        })
        .collect::<anyhow::Result<Vec<LabelledExample>>>()?;
    Ok((cases.iter().map(|c| c.name.clone()).collect(), examples))
}

/// `trainer eval --addresses`: the bundle's `parse_address` on reviewed real addresses, parser
/// fixture files in `dir`, scored per country and label. With `report`, every address whose
/// parse is not exact is written there as JSON lines for error analysis.
pub fn run_addresses(dir: &Path, bundle: &Path, report: Option<&Path>) -> anyhow::Result<()> {
    let bytes = std::fs::read(bundle).with_context(|| format!("reading {}", bundle.display()))?;
    let (names, examples) = address_examples(dir)?;
    let refs: Vec<&LabelledExample> = examples.iter().collect();
    let preds = shipped_spans(&bytes, None, &refs)?;
    report_addresses(&names, &examples, &preds, report)
}

/// `trainer eval --addresses --run`: a parser run's best checkpoint on the same addresses,
/// without the library's confidence policy, so a parser can be compared before a detector is
/// trained on its n-gram table. An address the run cannot encode counts as parsed wrong.
pub fn run_addresses_model<B: Backend>(
    dir: &Path,
    run_dir: &Path,
    report: Option<&Path>,
    device: &B::Device,
) -> anyhow::Result<()> {
    let cfg = crate::config::load(&run_dir.join("config.toml"))?;
    let fc = cfg.features.to_tessera();
    let (names, examples) = address_examples(dir)?;
    let (kept, items) = encode_examples(&examples, &fc);
    let model: TaggerNet<B> = cfg
        .parser_net_config()
        .init::<B>(device)
        .load_file(
            run_dir.join("best"),
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
            device,
        )
        .with_context(|| format!("loading {}", run_dir.join("best.mpk").display()))?;
    let predicted = predict(&model, &items, cfg.train.batch_size, device);
    let mut by_id: BTreeMap<u64, Vec<Span>> = kept
        .iter()
        .zip(predicted)
        .map(|(ex, p)| (ex.id, p.into_iter().map(|(s, _)| s).collect()))
        .collect();
    let preds: Vec<Vec<Span>> = examples
        .iter()
        .map(|ex| by_id.remove(&ex.id).unwrap_or_default())
        .collect();
    report_addresses(&names, &examples, &preds, report)
}

fn report_addresses(
    names: &[String],
    examples: &[LabelledExample],
    preds: &[Vec<Span>],
    report: Option<&Path>,
) -> anyhow::Result<()> {
    let refs: Vec<&LabelledExample> = examples.iter().collect();
    let scores = score(&refs, preds);

    println!(
        "{:<8} {:>6} {:>8} {:>8}",
        "country", "rows", "comp-F1", "exact"
    );
    for (country, counts) in &scores.per_country {
        let (exact, rows) = scores
            .exact_per_country
            .get(country)
            .copied()
            .unwrap_or_default();
        println!(
            "{country:<8} {rows:>6} {:>8.4} {:>8.4}",
            counts.f1(),
            exact as f64 / rows.max(1) as f64
        );
    }
    println!(
        "{:<8} {:>6} {:>8.4} {:>8.4}",
        "all",
        scores.rows,
        scores.overall.f1(),
        scores.exact_rate()
    );
    for (label, counts) in &scores.per_label {
        println!("  {label:<13} F1 {:.4}  gold {}", counts.f1(), counts.gold);
    }

    if let Some(path) = report {
        let mut out = String::new();
        for ((ex, pred), name) in examples.iter().zip(preds).zip(names) {
            if *pred == ex.spans {
                continue;
            }
            let parts = |spans: &[Span]| -> Vec<Value> {
                spans
                    .iter()
                    .map(|s| json!([s.label.as_str(), &ex.text[s.start as usize..s.end as usize]]))
                    .collect()
            };
            let line = json!({
                "name": name,
                "country": ex.country,
                "input": ex.text,
                "gold": parts(&ex.spans),
                "pred": parts(pred),
            });
            writeln!(out, "{line}")?;
        }
        std::fs::write(path, out).with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}

/// `trainer eval --run`: scores the run's best checkpoint and the baseline on one split,
/// writes `runs/<name>/eval/<split>.json`, and optionally the worst errors.
pub fn run<B: Backend>(
    run_dir: &Path,
    split: Split,
    errors: Option<usize>,
    device: &B::Device,
) -> anyhow::Result<()> {
    let cfg = crate::config::load(&run_dir.join("config.toml"))?;
    let fc = cfg.features.to_tessera();
    let shard = PathBuf::from(&cfg.data.processed).join(format!("{}.parquet", split.name()));
    let examples: Vec<LabelledExample> = read_shard(&shard, split)?
        .into_iter()
        .filter(|e| !e.augmented)
        .collect();
    let (kept, items) = encode_examples(&examples, &fc);
    let unencoded = examples.len() - kept.len();
    if unencoded > 0 {
        eprintln!("{unencoded} rows could not be encoded and are not scored");
    }

    let model: TaggerNet<B> = cfg
        .parser_net_config()
        .init::<B>(device)
        .load_file(
            run_dir.join("best"),
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
            device,
        )
        .with_context(|| format!("loading {}", run_dir.join("best.mpk").display()))?;
    let predictions = predict(&model, &items, cfg.train.batch_size, device);
    let model_spans: Vec<Vec<Span>> = predictions
        .iter()
        .map(|p| p.iter().map(|(s, _)| *s).collect())
        .collect();
    let model_scores = score(&kept, &model_spans);
    let calib = calibration_of(
        predictions
            .iter()
            .zip(&kept)
            .flat_map(|(p, ex)| p.iter().map(|(s, c)| (*c, ex.spans.contains(s)))),
    );
    let baseline: Vec<Vec<Span>> = kept.iter().map(|ex| baseline_spans(ex)).collect();
    let baseline_scores = score(&kept, &baseline);

    let out_dir = run_dir.join("eval");
    std::fs::create_dir_all(&out_dir)?;
    let report = json!({
        "unencoded_rows": unencoded,
        "run": run_dir.display().to_string(),
        "split": split.name(),
        "model": model_scores.json(),
        "baseline": baseline_scores.json(),
        "calibration": {
            "bins": calib.bins.iter().map(|(n, c, a)| json!({ "count": n, "confidence": round4(f64::from(*c)), "accuracy": round4(f64::from(*a)) })).collect::<Vec<_>>(),
            "ece": round4(f64::from(calib.ece)),
        },
    });
    std::fs::write(
        out_dir.join(format!("{}.json", split.name())),
        serde_json::to_string_pretty(&report)? + "\n",
    )?;

    println!(
        "{:<8} {:>6} {:>14} {:>17} {:>12} {:>15}",
        "country", "rows", "model comp-F1", "baseline comp-F1", "model exact", "baseline exact"
    );
    for (country, counts) in &model_scores.per_country {
        let base = baseline_scores
            .per_country
            .get(country)
            .copied()
            .unwrap_or_default();
        let rows = model_scores
            .exact_per_country
            .get(country)
            .map_or(0, |e| e.1);
        let rate = |m: &BTreeMap<String, (usize, usize)>| {
            m.get(country)
                .map_or(0.0, |(e, n)| *e as f64 / (*n).max(1) as f64)
        };
        println!(
            "{:<8} {:>6} {:>14.4} {:>17.4} {:>12.4} {:>15.4}",
            country,
            rows,
            counts.f1(),
            base.f1(),
            rate(&model_scores.exact_per_country),
            rate(&baseline_scores.exact_per_country)
        );
    }
    println!(
        "{:<8} {:>6} {:>14.4} {:>17.4} {:>12.4} {:>15.4}",
        "all",
        model_scores.rows,
        model_scores.overall.f1(),
        baseline_scores.overall.f1(),
        model_scores.exact_rate(),
        baseline_scores.exact_rate()
    );
    println!("calibration ECE {:.4}", calib.ece);

    if let Some(n) = errors {
        let path = PathBuf::from("internal/reports/m2-parser-errors.md");
        write_errors(&path, run_dir, split, &kept, &predictions, n)?;
        println!("written: {}", path.display());
    }
    Ok(())
}

/// Characters whose label differs between gold and prediction.
fn wrong_chars(ex: &LabelledExample, pred: &[Span]) -> usize {
    ex.text
        .char_indices()
        .filter(|(_, c)| !c.is_whitespace())
        .filter(|(b, _)| label_at(&ex.spans, *b as u32) != label_at(pred, *b as u32))
        .count()
}

fn write_errors(
    path: &Path,
    run_dir: &Path,
    split: Split,
    kept: &[&LabelledExample],
    predictions: &[Vec<(Span, f32)>],
    n: usize,
) -> anyhow::Result<()> {
    let mut ranked: Vec<(usize, f32, usize)> = kept
        .iter()
        .zip(predictions)
        .enumerate()
        .map(|(i, (ex, p))| {
            let spans: Vec<Span> = p.iter().map(|(s, _)| *s).collect();
            (
                wrong_chars(ex, &spans),
                p.iter().map(|(_, c)| c).sum::<f32>(),
                i,
            )
        })
        .filter(|(w, _, _)| *w > 0)
        .collect();
    let total_wrong = ranked.len();
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.total_cmp(&b.1)));
    let mut md = String::new();
    writeln!(md, "# Parser errors\n")?;
    writeln!(
        md,
        "Run `{}`, split `{}`: {} of {} examples have at least one wrong character. The {} worst follow, most wrong characters first.\n",
        run_dir.display(),
        split.name(),
        total_wrong,
        kept.len(),
        n.min(total_wrong)
    )?;
    writeln!(
        md,
        "## Review\n\nTo be written after reading the examples below.\n"
    )?;
    for (rank, (wrong, _, i)) in ranked.into_iter().take(n).enumerate() {
        let ex = kept[i];
        writeln!(
            md,
            "## {}. {} ({} wrong characters)\n",
            rank + 1,
            ex.country,
            wrong
        )?;
        writeln!(md, "```text\n{}\n```\n", ex.text)?;
        writeln!(
            md,
            "| gold | text | | predicted | text | confidence |\n| --- | --- | --- | --- | --- | ---: |"
        )?;
        let rows = ex.spans.len().max(predictions[i].len());
        for r in 0..rows {
            let g = ex.spans.get(r).map_or((String::new(), String::new()), |s| {
                (
                    s.label.as_str().to_string(),
                    ex.text[s.start as usize..s.end as usize].replace('\n', " "),
                )
            });
            let p = predictions[i].get(r).map_or(
                (String::new(), String::new(), String::new()),
                |(s, c)| {
                    (
                        s.label.as_str().to_string(),
                        ex.text[s.start as usize..s.end as usize].replace('\n', " "),
                        format!("{c:.2}"),
                    )
                },
            );
            writeln!(md, "| {} | {} | | {} | {} | {} |", g.0, g.1, p.0, p.1, p.2)?;
        }
        writeln!(md)?;
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, md)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::PARSER_LABELS;

    #[test]
    fn decode_prefers_the_best_legal_label() {
        let row = |pairs: &[(usize, f32)]| {
            let mut r = vec![0.0; PARSER_LABELS];
            for &(i, p) in pairs {
                r[i] = p;
            }
            r
        };
        // Row 2's mass is on I-road (4), but row 1 chose O, so I-road is illegal there.
        let probs = vec![
            row(&[(0, 0.9), (3, 0.1)]),
            row(&[(4, 0.6), (3, 0.3), (0, 0.1)]),
            row(&[(4, 0.9)]),
        ];
        let got: Vec<u8> = decode_probs(&probs).into_iter().map(|d| d.0).collect();
        assert_eq!(got, vec![0, 3, 4]);
    }

    #[test]
    fn calibration_bins_and_ece() {
        let c =
            calibration_of([(0.95, true), (0.95, true), (0.55, true), (0.55, false)].into_iter());
        assert_eq!(c.bins[9].0, 2);
        assert_eq!(c.bins[5].0, 2);
        assert!((c.bins[9].2 - 1.0).abs() < 1e-6 && (c.bins[5].2 - 0.5).abs() < 1e-6);
        // 0.5 * |0.95 - 1.0| + 0.5 * |0.55 - 0.5|
        assert!((c.ece - 0.05).abs() < 1e-6, "{}", c.ece);
    }

    #[test]
    fn scores_match_exact_spans() {
        let ex = LabelledExample {
            id: 0,
            group_id: 0,
            country: "GB".into(),
            language: "en".into(),
            text: "10 Downing Street".into(),
            spans: vec![
                Span {
                    label: AddressLabel::HouseNumber,
                    start: 0,
                    end: 2,
                },
                Span {
                    label: AddressLabel::Road,
                    start: 3,
                    end: 17,
                },
            ],
            split: Split::Test,
            augmented: false,
        };
        let s = score(
            &[&ex],
            &[vec![
                Span {
                    label: AddressLabel::HouseNumber,
                    start: 0,
                    end: 2,
                },
                Span {
                    label: AddressLabel::Road,
                    start: 3,
                    end: 10,
                },
            ]],
        );
        assert_eq!(
            (s.overall.tp, s.overall.pred, s.overall.gold, s.exact),
            (1, 2, 2, 0)
        );
        assert_eq!(s.char_total, 15);
        assert_eq!(s.char_correct, 9);
    }
}
