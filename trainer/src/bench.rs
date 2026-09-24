//! Native per-stage timing on the profiling fixtures: `trainer bench`. Appends a section to the
//! profile report that the SIMD decision reads.

use std::fs;
use std::path::Path;
use std::time::Instant;

use anyhow::Context;
use tessera::profile::{StageTimings, extract_timed, median, p95_total, parse_timed};
use tessera::{Config, Kind, Query, Tessera};

const WARMUP: u32 = 10;

pub fn run(model: &Path, iterations: u32, out: &Path) -> anyhow::Result<()> {
    let bytes = fs::read(model).with_context(|| format!("reading {}", model.display()))?;
    let tessera = Tessera::load(
        &bytes,
        Config {
            kinds: Kind::all(),
            expected_checksum: None,
        },
    )
    .map_err(|e| anyhow::anyhow!("loading {}: {e}", model.display()))?;
    let address = fs::read_to_string("fixtures/profile/address-gb.txt")
        .context("reading fixtures/profile/address-gb.txt")?;
    let address = address.trim_end();
    let document = fs::read_to_string("fixtures/profile/document-10k.txt")
        .context("reading fixtures/profile/document-10k.txt")?;

    let origin = Instant::now();
    let now = move || origin.elapsed().as_secs_f64() * 1000.0;
    let query = Query::default();
    let address_runs = collect(iterations, || {
        parse_timed(&tessera, address, &query, now).map(|(t, _)| t)
    })?;
    let document_runs = collect(iterations, || {
        extract_timed(&tessera, &document, &query, now).map(|(t, _)| t)
    })?;

    let section = render(
        &host_line(),
        iterations,
        document.chars().count(),
        (median(&address_runs), p95_total(&address_runs)),
        (median(&document_runs), p95_total(&document_runs)),
    );
    let mut existing = fs::read_to_string(out).unwrap_or_default();
    if !existing.is_empty() && !existing.ends_with('\n') {
        existing.push('\n');
    }
    existing.push_str(&section);
    fs::write(out, existing).with_context(|| format!("writing {}", out.display()))?;
    print!("{section}");
    Ok(())
}

/// `iterations` timed runs after `WARMUP` discarded ones.
fn collect(
    iterations: u32,
    mut one: impl FnMut() -> Result<StageTimings, tessera::Error>,
) -> anyhow::Result<Vec<StageTimings>> {
    let mut runs = Vec::with_capacity(iterations as usize);
    for i in 0..(iterations + WARMUP) {
        let t = one().map_err(|e| anyhow::anyhow!("timed run: {e}"))?;
        if i >= WARMUP {
            runs.push(t);
        }
    }
    Ok(runs)
}

fn host_line() -> String {
    let rustc = std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();
    format!(
        "{} {} · {}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        rustc.trim()
    )
}

fn render(
    host: &str,
    iterations: u32,
    document_chars: usize,
    address: (StageTimings, f64),
    document: (StageTimings, f64),
) -> String {
    let row = |name: &str, t: &StageTimings, p95: f64| {
        format!(
            "| {name} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {p95:.3} |\n",
            t.tokenize_ms,
            t.featurize_ms,
            t.rules_ms,
            t.detect_ms,
            t.parse_ms,
            t.group_ms,
            t.total_ms
        )
    };
    let mut s = format!(
        "\n## Native ({host})\n\n{iterations} iterations after {WARMUP} warmups, median per stage, milliseconds. Document: {document_chars} chars.\n\n"
    );
    s.push_str(
        "| input | tokenize | featurize | rules | detect | parse | group | total | p95 total |\n",
    );
    s.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
    s.push_str(&row("address-gb", &address.0, address.1));
    s.push_str(&row("document-10k", &document.0, document.1));
    s
}
