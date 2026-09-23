//! The measured sections under the demo: accuracy, calibration, the pipeline on one address,
//! the training data, and download sizes. Every number comes from [`crate::stats`].

use leptos::prelude::*;

use super::format::{bytes, percent, thousands, words};
use crate::stats::{self, Row};

/// One bar of a chart.
struct Bar {
    label: String,
    value: String,
    /// Share of the track the bar fills, from 0 to 100.
    width: f64,
    /// Blue bar and dark label for tessera and kept rows; grey otherwise.
    accent: bool,
    /// Set apart from the row above.
    separated: bool,
}

/// Tick labels over a chart's track: four evenly spaced marks and the end.
struct Axis {
    ticks: [&'static str; 4],
    end: &'static str,
}

const PERCENT_AXIS: Axis = Axis {
    ticks: ["0%", "25%", "50%", "75%"],
    end: "100%",
};

fn bar_style(width: f64) -> String {
    format!("width: {:.2}%", width.clamp(0.0, 100.0))
}

fn axis_row(axis: Axis) -> impl IntoView {
    view! {
        <div class="chart-row axis" aria-hidden="true">
            <span></span>
            <span class="ticks">{axis.ticks.map(|t| view! { <span>{t}</span> })}</span>
            <span class="value">{axis.end}</span>
        </div>
    }
}

fn chart(class: &'static str, axis: Option<Axis>, bars: Vec<Bar>) -> impl IntoView {
    view! {
        <div class=format!("chart {class}")>
            {axis.map(axis_row)}
            {bars
                .into_iter()
                .map(|b| {
                    view! {
                        <div class="chart-row" class:accent=b.accent class:separated=b.separated>
                            <span class="label">{b.label}</span>
                            <span class="track">
                                <span class="bar" style=bar_style(b.width)></span>
                            </span>
                            <span class="value">{b.value}</span>
                        </div>
                    }
                })
                .collect_view()}
        </div>
    }
}

#[component]
fn Figure(
    title: &'static str,
    #[prop(into)] subtitle: String,
    label: &'static str,
    children: Children,
) -> impl IntoView {
    view! {
        <section class="figure" aria-label=label>
            <div>
                <div class="figure-title">{title}</div>
                <div class="figure-sub">{subtitle}</div>
            </div>
            {children()}
        </section>
    }
}

/// "Great Britain, Germany, Georgia, the United States and Japan".
fn country_list() -> String {
    let names: Vec<String> = stats::BY_COUNTRY
        .iter()
        .map(|c| match c.code {
            "US" => format!("the {}", c.name),
            _ => c.name.to_string(),
        })
        .collect();
    match names.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} and {last}", rest.join(", ")),
        _ => names.concat(),
    }
}

/// "3,000 addresses each, 2,918 for Japan".
fn rows_per_country() -> String {
    let mut counts: Vec<u32> = stats::BY_COUNTRY.iter().map(|c| c.rows).collect();
    counts.sort_unstable();
    let common = counts.get(counts.len() / 2).copied().unwrap_or(0);
    let exceptions: Vec<String> = stats::BY_COUNTRY
        .iter()
        .filter(|c| c.rows != common)
        .map(|c| format!("{} for {}", thousands(c.rows.into()), c.name))
        .collect();
    let mut out = format!("{} addresses each", thousands(common.into()));
    for e in exceptions {
        out.push_str(", ");
        out.push_str(&e);
    }
    out
}

/// The headline and every measured section.
#[component]
pub(crate) fn Figures() -> impl IntoView {
    view! {
        <Headline />
        <ByCountry />
        <ByLabel />
        <Calibration />
        <Pipeline />
        <TrainingData />
        <Sizes />
    }
}

#[component]
fn Headline() -> impl IntoView {
    view! {
        <section class="headline">
            <p>
                <span class="big">{percent(stats::EXACT_TESSERA)}</span>
                {format!(
                    " of held-out addresses from {} have every component labelled correctly. The rules baseline gets {}.",
                    country_list(),
                    percent(stats::EXACT_RULES),
                )}
            </p>
        </section>
    }
}

#[component]
fn ByCountry() -> impl IntoView {
    let bars = stats::BY_COUNTRY
        .iter()
        .flat_map(|c| {
            [
                Bar {
                    label: format!("{} · tessera", c.name),
                    value: percent(c.tessera),
                    width: c.tessera * 100.0,
                    accent: true,
                    separated: false,
                },
                Bar {
                    label: format!("{} · rules", c.name),
                    value: percent(c.rules),
                    width: c.rules * 100.0,
                    accent: false,
                    separated: false,
                },
            ]
        })
        .collect();
    view! {
        <Figure
            title="Exact address parses by country"
            subtitle=format!("held-out test set, {} · higher is better", rows_per_country())
            label="By country"
        >
            {chart("narrow-label by-country", Some(PERCENT_AXIS), bars)}
            <p class="note">
                "Addresses from the libpostal OpenStreetMap training set, 2017-03-04. Test rows share no record with training; variants of one address are grouped before the split. Tessera is the shipped int8 bundle run through the library; rules is the deterministic parser used as the baseline."
            </p>
        </Figure>
    }
}

#[component]
fn ByLabel() -> impl IntoView {
    // The axis starts at 80%, so each point above it is five percent of the track.
    let bars = stats::BY_LABEL
        .iter()
        .map(|l| Bar {
            label: l.name.to_string(),
            value: percent(l.f1),
            width: (l.f1 - 0.8) * 500.0,
            accent: true,
            separated: false,
        })
        .collect();
    view! {
        <Figure
            title="Component F1 by label"
            subtitle=format!("all {} countries, f32 checkpoint · higher is better", words(stats::BY_COUNTRY.len() as u32))
            label="By label"
        >
            {chart(
                "narrow-label",
                Some(Axis {
                    ticks: ["80%", "85%", "90%", "95%"],
                    end: "100%",
                }),
                bars,
            )}
            <p class="note">
                "Suburb and district score lowest: OpenStreetMap tags the same kind of place either way, so part of the error is disagreement in the labels themselves."
            </p>
        </Figure>
    }
}

#[component]
fn Calibration() -> impl IntoView {
    let total: u32 = stats::CALIBRATION.iter().map(|b| b.count).sum();
    let top = stats::CALIBRATION.last();
    let high_share = top.map_or(0.0, |b| f64::from(b.count) / f64::from(total.max(1)));
    let high_accuracy = top.map_or(0.0, |b| b.accuracy);
    let rows = stats::CALIBRATION
        .iter()
        .map(|b| {
            view! {
                <div class="chart-row">
                    <span class="label">{format!("{:.1}–{:.1}", b.low, b.high)}</span>
                    <span class="track">
                        <span class="bar muted-bar" style=bar_style(b.accuracy * 100.0)></span>
                        <span
                            class="said"
                            style=format!("left: calc({:.1}% - 1px)", b.confidence * 100.0)
                        ></span>
                    </span>
                    <span class="value muted">{thousands(b.count.into())}</span>
                </div>
            }
        })
        .collect_view();
    view! {
        <Figure
            title="Stated confidence against accuracy"
            subtitle=format!(
                "{} predicted components, f32 checkpoint · bars that match the stated confidence are calibrated",
                thousands(total.into()),
            )
            label="Confidence"
        >
            <div class="chart narrow-label">
                {axis_row(Axis {
                    ticks: PERCENT_AXIS.ticks,
                    end: "n",
                })}
                {rows}
            </div>
            <p class="note">
                {format!(
                    "Filled bar: share correct. Blue tick: the confidence the model stated. {} of components are stated at 0.9 or more and {} of those are right; between 0.5 and 0.9 the model is overconfident. Components under 0.5 are dropped, or returned as unknown when uncertain results are asked for; an address is as confident as its weakest component and has review_recommended set under 0.85. Expected calibration error {}.",
                    percent(high_share),
                    percent(high_accuracy),
                    stats::ECE,
                )}
            </p>
        </Figure>
    }
}

#[component]
fn Pipeline() -> impl IntoView {
    view! {
        <Figure
            title="What happens to one address"
            subtitle=format!("{} · values from the shipped model", stats::EXAMPLE)
            label="Pipeline"
        >
            <div class="pipeline">
                {stats::PIPELINE
                    .iter()
                    .map(|s| {
                        view! {
                            <div class="step">
                                <span class="step-name">{s.step}</span>
                                <span class="step-what">{s.what}</span>
                            </div>
                        }
                    })
                    .collect_view()}
            </div>
            <p class="note">
                {format!(
                    "The network is trained in Rust with Burn and run by a hand-written int8 forward pass. On {} golden cases the largest logit difference between the two is {:.2e} (limit {:e}), natively and in Chrome, Firefox and Safari. Going from f32 to int8 weights cost {:.4} validation F1.",
                    stats::GOLDEN_CASES,
                    stats::GOLDEN_WORST,
                    stats::GOLDEN_TOLERANCE,
                    stats::INT8_F1_DROP,
                )}
            </p>
        </Figure>
    }
}

#[component]
fn TrainingData() -> impl IntoView {
    let top = (stats::LINES_READ.max(2) as f64).log10();
    let bar = |label: &str, rows: u64, kind: Row| Bar {
        label: label.to_string(),
        value: thousands(rows),
        width: (rows.max(1) as f64).log10() / top * 100.0,
        accent: kind != Row::Dropped,
        separated: kind == Row::Generated,
    };
    let bars = std::iter::once(bar("lines read", stats::LINES_READ, Row::Dropped))
        .chain(stats::FUNNEL.iter().map(|f| bar(f.name, f.rows, f.kind)))
        .collect();
    let codes: Vec<&str> = stats::BY_COUNTRY.iter().map(|c| c.code).collect();
    let codes = match codes.split_last() {
        Some((last, rest)) if !rest.is_empty() => format!("{} and {last}", rest.join(", ")),
        _ => codes.concat(),
    };
    view! {
        <Figure
            title="Training data"
            subtitle=format!(
                "where the {} lines read went; rows from {codes} are kept · log scale",
                thousands(stats::LINES_READ),
            )
            label="Training data"
        >
            {chart("wide-label funnel", None, bars)}
            <p class="note">
                {format!(
                    "Source: libpostal's tagged OpenStreetMap addresses, {:.2} GB, © OpenStreetMap contributors, ODbL 1.0. The rows under lines read add up to it. The generated copies are not source lines: each training row gets up to {} rewritten copies (one line with commas, abbreviated roads, other separators, a floor line); validation and test rows are not rewritten.",
                    stats::SOURCE_BYTES as f64 / 1e9,
                    words(stats::AUGMENT_COPIES),
                )}
            </p>
        </Figure>
    }
}

#[component]
fn Sizes() -> impl IntoView {
    let largest = stats::BUNDLE_GZIP.max(1) as f64;
    let bars = [
        ("address model, int8", stats::BUNDLE_GZIP),
        ("library wasm, rules and parser", stats::WASM_GZIP),
        ("of which phone tables", stats::PHONE_TABLES_GZIP),
    ]
    .into_iter()
    .map(|(label, n)| Bar {
        label: label.to_string(),
        value: bytes(n),
        width: n as f64 / largest * 100.0,
        accent: true,
        separated: false,
    })
    .collect();
    view! {
        <Figure title="What the browser downloads" subtitle="gzip -9 · lower is better" label="Size">
            {chart("wide-label", None, bars)}
            <p class="note">
                {format!(
                    "The phone tables cover {} regions, generated from {}, in place of the phonenumber crate ({} gzip). The library wasm includes the parser's forward pass; its SIMD build is {}. This page's own wasm is larger because it includes the interface.",
                    stats::PHONE_REGIONS,
                    stats::PHONE_SOURCE,
                    bytes(stats::PHONE_METADATA_GZIP),
                    bytes(stats::WASM_SIMD_GZIP),
                )}
            </p>
        </Figure>
    }
}
