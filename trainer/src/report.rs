//! Milestone 2 report: renders the learning-curve, test-split, per-label, calibration, and
//! quantization tables from the run JSON files, so numbers are copied by code rather than by
//! hand. Prose sections are left as headings with a `<!-- write -->` marker.

use std::path::{Path, PathBuf};

use anyhow::Context;
use serde_json::Value;

const WRITE: &str = "<!-- write -->";

fn read(path: &Path) -> anyhow::Result<Value> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(serde_json::from_str(&text)?)
}

/// A JSON value as table text; a missing one is `n/a`.
fn field(v: &Value) -> String {
    match v {
        Value::Null => "n/a".into(),
        Value::String(s) => s.clone(),
        v => v.to_string(),
    }
}

/// NaN when the key is missing, which the formatters below print as `n/a`.
fn num(v: &Value, path: &[&str]) -> f64 {
    path.iter()
        .fold(v, |v, k| &v[*k])
        .as_f64()
        .unwrap_or(f64::NAN)
}

fn fixed(x: f64, places: usize) -> String {
    if x.is_nan() {
        "n/a".into()
    } else {
        format!("{x:.places$}")
    }
}

fn pct(x: f64) -> String {
    if x.is_nan() {
        "n/a".into()
    } else {
        format!("{:.1}", x * 100.0)
    }
}

fn delta(a: f64, b: f64) -> String {
    if a.is_nan() || b.is_nan() {
        "n/a".into()
    } else {
        format!("{:+.1}", (a - b) * 100.0)
    }
}

fn data_section(manifest: &Value) -> String {
    let mut out = String::from(
        "## Data\n\n| country | train | valid | test | augmented train |\n|---|---:|---:|---:|---:|\n",
    );
    if let Some(counts) = manifest["counts"].as_object() {
        for (c, v) in counts {
            out += &format!(
                "| {c} | {} | {} | {} | {} |\n",
                field(&v["train"]),
                field(&v["valid"]),
                field(&v["test"]),
                field(&v["augmented"])
            );
        }
    }
    out += &format!("\nLines read: {}\n", field(&manifest["lines_read"]));
    for (key, title) in [
        ("dropped_rows", "Dropped rows by reason"),
        ("outside_tags", "Tags kept as unlabelled text"),
    ] {
        if let Some(m) = manifest[key].as_object() {
            let list: Vec<String> = m.iter().map(|(k, v)| format!("{k} {v}")).collect();
            out += &format!("\n{title}: {}\n", list.join(", "));
        }
    }
    out
}

fn curve_section(runs: &[PathBuf]) -> anyhow::Result<String> {
    let mut out = String::from(
        "## Learning curves\n\n| run | rows per country | train rows | best valid comp-F1 | valid exact | best epoch | wall-clock |\n|---|---:|---:|---:|---:|---:|---:|\n",
    );
    for run in runs {
        let s = read(&run.join("summary.json"))?;
        let name = run
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let per = s["originals_per_country"]
            .as_object()
            .map(|m| {
                m.iter()
                    .map(|(c, n)| format!("{c} {n}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_else(|| "n/a".into());
        let clock = s["wall_clock_seconds"]
            .as_u64()
            .map_or("n/a".into(), |secs| {
                format!("{}m{:02}s", secs / 60, secs % 60)
            });
        out += &format!(
            "| {name} | {per} | {} | {} | {} | {} | {clock} |\n",
            field(&s["train_rows"]),
            pct(num(&s, &["best_valid_component_f1"])),
            pct(num(&s, &["best_valid_exact"])),
            field(&s["best_epoch"]),
        );
    }
    Ok(out)
}

fn test_section(eval: &Value) -> String {
    let mut out = String::from(
        "## Test split: model versus baseline\n\n| country | rows | model comp-F1 | baseline comp-F1 | delta | model exact | baseline exact | delta |\n|---|---:|---:|---:|---:|---:|---:|---:|\n",
    );
    let (m, b) = (&eval["model"], &eval["baseline"]);
    let mut row = |name: &str, rows: &Value, mf: f64, bf: f64, me: f64, be: f64| {
        out += &format!(
            "| {name} | {} | {} | {} | {} | {} | {} | {} |\n",
            field(rows),
            pct(mf),
            pct(bf),
            delta(mf, bf),
            pct(me),
            pct(be),
            delta(me, be)
        );
    };
    if let Some(countries) = m["per_country"].as_object() {
        for c in countries.keys() {
            row(
                c,
                &m["rows_per_country"][c],
                num(m, &["per_country", c, "f1"]),
                num(b, &["per_country", c, "f1"]),
                num(m, &["exact_per_country", c]),
                num(b, &["exact_per_country", c]),
            );
        }
    }
    row(
        "all",
        &m["rows"],
        num(m, &["overall", "f1"]),
        num(b, &["overall", "f1"]),
        num(m, &["exact_parse"]),
        num(b, &["exact_parse"]),
    );
    out
}

/// The exported int8 bundle through the library's `parse_address`, confidence policy included,
/// beside the f32 checkpoint's raw decoding: what a caller gets against what the network says.
fn shipped_section(eval: &Value, shipped: &Value) -> String {
    let mut out = String::from(
        "## Test split: shipped bundle\n\nThe int8 bundle through `parse_address`, where components under 0.5 confidence and repeated single labels become `unknown` and count as not predicted, beside the f32 checkpoint's raw decoding above.\n\n| country | shipped comp-F1 | raw comp-F1 | shipped exact | raw exact |\n|---|---:|---:|---:|---:|\n",
    );
    let m = &eval["model"];
    let mut row = |name: &str, sf: f64, mf: f64, se: f64, me: f64| {
        out += &format!(
            "| {name} | {} | {} | {} | {} |\n",
            pct(sf),
            pct(mf),
            pct(se),
            pct(me)
        );
    };
    if let Some(countries) = shipped["per_country"].as_object() {
        for c in countries.keys() {
            row(
                c,
                num(shipped, &["per_country", c, "f1"]),
                num(m, &["per_country", c, "f1"]),
                num(shipped, &["exact_per_country", c]),
                num(m, &["exact_per_country", c]),
            );
        }
    }
    row(
        "all",
        num(shipped, &["overall", "f1"]),
        num(m, &["overall", "f1"]),
        num(shipped, &["exact_parse"]),
        num(m, &["exact_parse"]),
    );
    out
}

fn label_section(eval: &Value) -> String {
    let mut out = String::from(
        "## Per-label scores\n\n| label | precision | recall | F1 | support |\n|---|---:|---:|---:|---:|\n",
    );
    if let Some(labels) = eval["model"]["per_label"].as_object() {
        for (l, v) in labels {
            out += &format!(
                "| {l} | {} | {} | {} | {} |\n",
                pct(num(v, &["precision"])),
                pct(num(v, &["recall"])),
                pct(num(v, &["f1"])),
                field(&v["gold"])
            );
        }
    }
    out
}

fn calibration_section(eval: &Value) -> String {
    let mut out = String::from(
        "## Calibration\n\n| bin | components | mean confidence | accuracy |\n|---|---:|---:|---:|\n",
    );
    if let Some(bins) = eval["calibration"]["bins"].as_array() {
        for (i, b) in bins.iter().enumerate() {
            out += &format!(
                "| {:.1}-{:.1} | {} | {} | {} |\n",
                i as f64 / 10.0,
                (i + 1) as f64 / 10.0,
                field(&b["count"]),
                // An empty bin has no mean confidence or accuracy.
                if b["count"].as_u64() == Some(0) {
                    "n/a".into()
                } else {
                    fixed(num(b, &["confidence"]), 3)
                },
                if b["count"].as_u64() == Some(0) {
                    "n/a".into()
                } else {
                    fixed(num(b, &["accuracy"]), 3)
                }
            );
        }
    }
    out += &format!(
        "\nExpected calibration error: {}\n",
        fixed(num(eval, &["calibration", "ece"]), 4)
    );
    out
}

fn quantization_section(run: &Path, bundle: &Path) -> anyhow::Result<String> {
    let q = read(&run.join("quantize.json"))?;
    let bytes = std::fs::read(bundle).with_context(|| format!("reading {}", bundle.display()))?;
    let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::best());
    std::io::Write::write_all(&mut gz, &bytes)?;
    let gzip = gz.finish()?.len();
    let cfg = read_toml(&run.join("config.toml"))?;
    Ok(format!(
        "## Quantization\n\n| measure | value |\n|---|---:|\n\
         | valid comp-F1, f32 | {} |\n| valid comp-F1, int8 | {} |\n| drop | {} |\n\
         | valid exact, f32 | {} |\n| valid exact, int8 | {} |\n\
         | max absolute weight error | {} |\n| bundle bytes | {} |\n| bundle bytes, gzip level 9 (flate2) | {gzip} |\n\
         | hash_buckets | {} |\n",
        pct(num(&q, &["valid_f32_component_f1"])),
        pct(num(&q, &["valid_int8_component_f1"])),
        fixed(
            num(&q, &["valid_f32_component_f1"]) - num(&q, &["valid_int8_component_f1"]),
            4
        ),
        pct(num(&q, &["valid_f32_exact"])),
        pct(num(&q, &["valid_int8_exact"])),
        fixed(num(&q, &["max_abs_weight_error"]), 5),
        bytes.len(),
        cfg.get("features")
            .and_then(|f| f.get("hash_buckets"))
            .map_or("?".to_string(), |v| v.to_string()),
    ))
}

fn read_toml(path: &Path) -> anyhow::Result<toml::Value> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(toml::from_str(&text)?)
}

/// Arguments of `trainer report`.
pub struct ReportArgs<'a> {
    pub curve: &'a [PathBuf],
    pub run: &'a Path,
    pub manifest: &'a Path,
    pub bundle: &'a Path,
    pub out: &'a Path,
}

/// `trainer report`: writes the milestone report tables from the run and eval files.
pub fn run(args: ReportArgs<'_>) -> anyhow::Result<()> {
    let eval = read(&args.run.join("eval/test.json"))?;
    let shipped = read(&args.run.join("eval/test-shipped.json"))?;
    let manifest = read(args.manifest)?;
    let sections = [
        "# Milestone 2: learned address parser\n".to_string(),
        format!("## Summary\n\n{WRITE}\n"),
        data_section(&manifest),
        curve_section(args.curve)?,
        test_section(&eval),
        shipped_section(&eval, &shipped),
        label_section(&eval),
        calibration_section(&eval),
        quantization_section(args.run, args.bundle)?,
        format!("## Golden gate\n\n{WRITE}\n"),
        format!("## Fixtures\n\n{WRITE}\n"),
        format!("## Error review\n\n{WRITE}\n"),
        format!("## Decisions made in this milestone\n\n{WRITE}\n"),
        format!("## Decision\n\n{WRITE}\n"),
        format!("## Next\n\n{WRITE}\n"),
    ];
    if let Some(dir) = args.out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(args.out, sections.join("\n"))?;
    eprintln!("wrote {}", args.out.display());
    Ok(())
}
