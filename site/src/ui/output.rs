//! The output pane: what the library returned, as a list or as JSON, written as the scan
//! reaches each entity, with the status line and the phone-region hint above it.

use leptos::prelude::*;

use super::demo::{DemoState, Failure, HINTS, Shown};
use super::highlight;
use super::json::{self, kind_name};
use crate::protocol::{Found, FoundKind};

/// What the list shows after the range: the phone in E.164 with its region, or the library's
/// `source` for the entity with its confidence.
fn extra(found: &Found) -> String {
    match found.kind {
        FoundKind::Phone => [found.normalized.as_deref(), found.region.as_deref()]
            .into_iter()
            .flatten()
            .collect::<Vec<_>>()
            .join(" "),
        FoundKind::Email => format!("{} {:.2}", found.source, found.confidence),
        FoundKind::Address => format!("{} {:.3}", found.source, found.confidence),
    }
}

/// What the output pane holds.
#[derive(Debug, Clone, PartialEq)]
enum Content {
    Lists,
    Error(String),
    Empty,
    Waiting,
}

/// The output pane.
#[component]
pub(crate) fn OutputPane(state: DemoState) -> impl IntoView {
    // An answer is listed while it is for the text on screen, or while the reader is typing and
    // a newer one is on its way; never for a text too long to analyze.
    let shown = Memo::new(move |_| {
        let shown = state.answer.get().and_then(Result::ok)?;
        let current = *shown.text == *state.text.get();
        let typing = state.editing.get() && !state.too_long.get();
        (current || typing).then_some(shown)
    });

    let status = move || match (state.failure.get(), shown.get(), state.answer.get()) {
        (Some(Failure::Model(_)), _, _) => {
            view! { <span class="failed">"model did not load"</span> }.into_any()
        }
        (Some(Failure::Worker(_)), _, _) => {
            view! { <span class="failed">"worker stopped"</span> }.into_any()
        }
        (None, Some(shown), _) if state.done.get() => {
            let n = shown.found.len();
            let count = match n {
                1 => "1 entity found".to_string(),
                n => format!("{n} entities found"),
            };
            view! { <span class="found-count">{count}</span> }.into_any()
        }
        // Only the count is announced; the scanning label is a visual effect.
        (None, Some(shown), _) => view! {
            <span class="scanning" aria-hidden="true">
                {format!("scanning {} bytes", super::format::thousands(shown.text.len() as u64))}
            </span>
        }
        .into_any(),
        (None, None, _) if state.too_long.get() => {
            view! { <span class="failed">"not analyzed"</span> }.into_any()
        }
        (None, None, None) => view! { <span class="loading">"loading model"</span> }.into_any(),
        (None, None, Some(Ok(_))) => view! { <span class="loading">"analyzing"</span> }.into_any(),
        (None, None, Some(Err(_))) => {
            view! { <span class="failed">"analysis failed"</span> }.into_any()
        }
    };

    // What the pane holds, reduced to what changes its layout, so a new answer or an added
    // address updates the lists in place instead of rebuilding them.
    let content = Memo::new(move |_| match state.failure.get() {
        Some(Failure::Model(message)) => {
            Content::Error(format!("could not load the model: {message}"))
        }
        Some(Failure::Worker(message)) => {
            Content::Error(format!("the inference worker failed: {message}"))
        }
        None if shown.with(Option::is_some) => Content::Lists,
        None => match state.answer.get() {
            Some(Err(message)) => Content::Error(format!("tessera returned an error: {message}")),
            Some(Ok(_)) => Content::Empty,
            None => Content::Waiting,
        },
    });
    let body = move || match content.get() {
        Content::Lists => written(shown, state).into_any(),
        Content::Error(message) => view! { <p class="error">{message}</p> }.into_any(),
        Content::Empty => ().into_any(),
        Content::Waiting => view! {
            <p class="waiting">
                "fetching tessera-v1.safetensors and checking its SHA-256 in a web worker"
            </p>
        }
        .into_any(),
    };

    // Scrolling the output by hand stops it following the newest line; scrolling back to the
    // bottom resumes it.
    let stop_following = move || state.follow.set_value(false);
    let on_scroll = move |_| {
        if let Some(pane) = state.output.get_untracked() {
            let gap = pane.scroll_height() - pane.scroll_top() - pane.client_height();
            if gap <= 4 {
                state.follow.set_value(true);
            }
        }
    };

    view! {
        <div class="out-pane">
            <div class="pane-head">
                <span class="out-title">
                    <span class="out-name">"detect · parse_address · "</span>
                    <select
                        class="hint"
                        aria-label="region for phone numbers without a country code"
                        on:change=move |ev| state.set_hint(event_target_value(&ev))
                    >
                        {HINTS
                            .iter()
                            .map(|&h| {
                                view! {
                                    <option value=h prop:selected=move || state.hint.get() == h>
                                        {h}
                                    </option>
                                }
                            })
                            .collect_view()}
                    </select>
                </span>
                <span class="status" aria-live="polite">{status}</span>
            </div>
            <div
                class="pane-scroll"
                node_ref=state.output
                on:wheel=move |_| stop_following()
                on:touchmove=move |_| stop_following()
                on:keydown=move |_| stop_following()
                on:scroll=on_scroll
            >
                {body}
            </div>
        </div>
    }
}

/// One line of the list, with everything it shows. The key covers the content, so a line is
/// redrawn only when what it says changes.
#[derive(Debug, Clone, PartialEq)]
struct Line {
    key: String,
    /// The entity's first byte: the line is written when the scan reaches it.
    start: usize,
    component: Option<usize>,
    kind: String,
    text: String,
    range: String,
    extra: String,
}

fn lines(text: &str, found: &[Found]) -> Vec<Line> {
    let slice = |start: usize, end: usize| text.get(start..end).unwrap_or("").to_string();
    let mut out = Vec::new();
    for e in found {
        let extra = extra(e);
        out.push(Line {
            key: format!("{}:{}:{}:{extra}", e.start, e.end, kind_name(e.kind)),
            start: e.start,
            component: None,
            kind: kind_name(e.kind).to_string(),
            text: slice(e.start, e.end),
            range: format!("{}..{}", e.start, e.end),
            extra,
        });
        for (k, c) in e.components.iter().enumerate() {
            out.push(Line {
                key: format!(
                    "{}:{}:{k}:{}:{}:{}",
                    e.start, e.end, c.label, c.start, c.end
                ),
                start: e.start,
                component: Some(k),
                kind: c.label.clone(),
                text: slice(c.start, c.end),
                range: format!("{}..{}", c.start, c.end),
                extra: String::new(),
            });
        }
    }
    out
}

/// One JSON object, keyed by its text.
#[derive(Debug, Clone, PartialEq)]
struct Object {
    start: usize,
    body: String,
}

/// Both output views, each holding what the scan has reached so far; after a scan, everything.
/// Lines and objects are keyed, so an address added later is inserted without redrawing the
/// rest. Only one view is shown; both grow, so switching views mid-scan shows the other at the
/// same point.
fn written(shown: Memo<Option<Shown>>, state: DemoState) -> impl IntoView {
    let all_lines = Memo::new(move |_| {
        shown.with(|s| {
            s.as_ref()
                .map(|s| lines(&s.text, &s.found))
                .unwrap_or_default()
        })
    });
    let all_objects = Memo::new(move |_| {
        shown.with(|s| {
            s.as_ref()
                .map(|s| {
                    s.found
                        .iter()
                        .map(|e| Object {
                            start: e.start,
                            body: json::object(&s.text, e),
                        })
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        })
    });
    let visible_lines = move || {
        let reached = state.reached.get();
        all_lines.with(|l| {
            l.iter()
                .filter(|l| l.start < reached)
                .cloned()
                .collect::<Vec<_>>()
        })
    };
    let visible_objects = move || {
        let reached = state.reached.get();
        all_objects.with(|o| {
            o.iter()
                .filter(|o| o.start < reached)
                .cloned()
                .collect::<Vec<_>>()
        })
    };
    let first = Memo::new(move |_| {
        let reached = state.reached.get();
        all_objects.with(|o| o.iter().find(|o| o.start < reached).map(|o| o.body.clone()))
    });
    view! {
        <div class="rows" hidden=move || state.json.get()>
            <For each=visible_lines key=|l| l.key.clone() children=move |l| list_row(l, state) />
        </div>
        <pre class="json" hidden=move || !state.json.get()>
            {json::OPEN}
            <For
                each=visible_objects
                key=|o| o.body.clone()
                children=move |o| {
                    let body = o.body.clone();
                    let separator = move || {
                        if first.with(|f| f.as_deref() == Some(body.as_str())) {
                            json::FIRST
                        } else {
                            json::NEXT
                        }
                    };
                    view! {
                        <span class="piece" data-anim="">
                            <span class="t-punct">{separator}</span>
                            {highlight::render(highlight::json(&o.body))}
                        </span>
                    }
                }
            />
            {move || state.done.get().then_some(json::CLOSE)}
        </pre>
    }
}

/// One row of the list: an entity with its range and details, or one of its components, which
/// lands a moment after the entity.
fn list_row(line: Line, state: DemoState) -> impl IntoView {
    let start = line.start;
    let selected = move || state.selected.get() == Some(start);
    let delay = line.component.map_or(String::new(), |k| {
        format!("animation-delay: {}ms", (k + 1) * 25)
    });
    match line.component {
        Some(_) => view! {
            <button
                type="button"
                class="row sub"
                class:selected=selected
                data-anim=""
                style=delay
                on:click=move |_| state.toggle(start)
            >
                <span class="row-kind">{line.kind}</span>
                <span class="row-text">{line.text}</span>
                <span class="row-range">{line.range}</span>
                <span class="row-extra"></span>
            </button>
        }
        .into_any(),
        None => view! {
            <button
                type="button"
                class="row"
                class:selected=selected
                aria-pressed=move || selected().to_string()
                data-anim=""
                on:click=move |_| state.toggle(start)
            >
                <span class=format!("row-kind c-{}", line.kind)>{line.kind.clone()}</span>
                <span class="row-text">{line.text}</span>
                <span class="row-range">{line.range}</span>
                <span class="row-extra">{line.extra}</span>
            </button>
        }
        .into_any(),
    }
}
