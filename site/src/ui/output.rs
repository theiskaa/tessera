//! The output pane: what the library returned, as JSON, a list, or contact cards, written as the
//! scan reaches each entity, with the status line and the phone-region hint above it.

use leptos::prelude::*;

use super::demo::{DemoState, Failure, HINTS, Shown, View};
use super::format::thousands;
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
        FoundKind::Person | FoundKind::Org | FoundKind::Address => {
            format!("{} {:.3}", found.source, found.confidence)
        }
    }
}

/// The bytes `start..end` of `text` as an owned string; empty when they do not slice it.
fn covered(text: &str, start: usize, end: usize) -> String {
    text.get(start..end).unwrap_or("").to_string()
}

/// `items` the scan has reached: those whose entity starts before byte `reached`.
fn written_so_far<T: Clone>(items: &[T], first: impl Fn(&T) -> usize, reached: usize) -> Vec<T> {
    items
        .iter()
        .filter(|item| first(item) < reached)
        .cloned()
        .collect()
}

/// `n` of something, as `1 contact` or `3 contacts`.
fn count(n: usize, one: &str, many: &str) -> String {
    if n == 1 {
        format!("1 {one}")
    } else {
        format!("{n} {many}")
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
            let counted = if state.view.get() == View::Cards {
                count(shown.contacts.len(), "contact", "contacts")
            } else {
                count(shown.found.len(), "entity found", "entities found")
            };
            let scope = if shown.section.is_some() {
                " in the selection"
            } else {
                ""
            };
            view! { <span class="found-count">{counted}{scope}</span> }.into_any()
        }
        // Only the count is announced; the scanning label is a visual effect.
        (None, Some(shown), _) => view! {
            <span class="scanning" aria-hidden="true">
                {format!("scanning {} bytes", thousands(shown.span_len() as u64))}
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

    // What the pane holds, reduced to what changes its layout, so a new answer updates the lists
    // in place instead of rebuilding them.
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

    view! {
        <div class="out-pane">
            <div class="pane-head">
                <span class="out-title">
                    <span class="out-name">
                        {move || {
                            if state.view.get() == View::Cards {
                                "extract_contacts · "
                            } else {
                                "detect · "
                            }
                        }}
                    </span>
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
    let mut out = Vec::new();
    for e in found {
        let extra = extra(e);
        out.push(Line {
            key: format!("{}:{}:{}:{extra}", e.start, e.end, kind_name(e.kind)),
            start: e.start,
            component: None,
            kind: kind_name(e.kind).to_string(),
            text: covered(text, e.start, e.end),
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
                text: covered(text, c.start, c.end),
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
    let visible_lines =
        move || all_lines.with(|l| written_so_far(l, |l: &Line| l.start, state.reached.get()));
    let visible_objects =
        move || all_objects.with(|o| written_so_far(o, |o: &Object| o.start, state.reached.get()));
    let first = Memo::new(move |_| {
        let reached = state.reached.get();
        all_objects.with(|o| o.iter().find(|o| o.start < reached).map(|o| o.body.clone()))
    });
    view! {
        <div class="rows" hidden=move || state.view.get() != View::List>
            <For each=visible_lines key=|l| l.key.clone() children=move |l| list_row(l, state) />
        </div>
        <pre class="json" hidden=move || state.view.get() != View::Json>
            {json::OPEN}
            <For
                each=visible_objects
                key=|o| o.body.clone()
                children=move |o| {
                    let start = o.start;
                    let (enter, leave) = state.hover(start);
                    let body = o.body.clone();
                    let separator = move || {
                        if first.with(|f| f.as_deref() == Some(body.as_str())) {
                            json::FIRST
                        } else {
                            json::NEXT
                        }
                    };
                    view! {
                        <span
                            class="piece"
                            class:selected=move || state.is_selected(start)
                            role="button"
                            tabindex="0"
                            data-anim=""
                            data-out=start.to_string()
                            on:mouseenter=enter
                            on:mouseleave=leave
                            on:click=move |_| state.reveal_in_document(start)
                            on:keydown=move |ev| {
                                if ev.key() == "Enter" || ev.key() == " " {
                                    ev.prevent_default();
                                    state.reveal_in_document(start);
                                }
                            }
                        >
                            <span class="t-punct">{separator}</span>
                            {highlight::render(highlight::json(&o.body))}
                        </span>
                    }
                }
            />
            {move || state.done.get().then_some(json::CLOSE)}
        </pre>
        <div class="cards" hidden=move || state.view.get() != View::Cards>
            {cards(shown, state)}
        </div>
    }
}

/// One row of the list: an entity with its range and details, or one of its components, which
/// lands a moment after the entity.
fn list_row(line: Line, state: DemoState) -> impl IntoView {
    let start = line.start;
    let (enter, leave) = state.hover(start);
    let (class, kind_class, delay) = match line.component {
        Some(k) => (
            "row sub",
            "row-kind".to_string(),
            format!("animation-delay: {}ms", (k + 1) * 25),
        ),
        None => ("row", format!("row-kind c-{}", line.kind), String::new()),
    };
    view! {
        <button
            type="button"
            class=class
            class:selected=move || state.is_selected(start)
            data-out=start.to_string()
            data-anim=""
            style=delay
            on:mouseenter=enter
            on:mouseleave=leave
            on:click=move |_| state.reveal_in_document(start)
        >
            <span class=kind_class>{line.kind}</span>
            <span class="row-text">{line.text}</span>
            <span class="row-range">{line.range}</span>
            <span class="row-extra">{line.extra}</span>
        </button>
    }
}

/// One contact card as drawn: its title, confidence, flag, and rows, keyed by content so a new
/// answer redraws only the cards that changed.
#[derive(Debug, Clone, PartialEq)]
struct CardView {
    key: String,
    /// The card appears when the scan reaches its first entity.
    first: usize,
    title: String,
    percent: String,
    review: bool,
    rows: Vec<CardRow>,
}

/// One member of a contact.
#[derive(Debug, Clone, PartialEq)]
struct CardRow {
    kind: FoundKind,
    text: String,
    start: usize,
    end: usize,
}

fn card_views(s: &Shown) -> Vec<CardView> {
    s.contacts
        .iter()
        .map(|c| {
            let members: Vec<&Found> = c.members.iter().filter_map(|&i| s.found.get(i)).collect();
            let rows: Vec<CardRow> = members
                .iter()
                .map(|e| CardRow {
                    kind: e.kind,
                    text: covered(&s.text, e.start, e.end),
                    start: e.start,
                    end: e.end,
                })
                .collect();
            CardView {
                key: format!("{rows:?}{}{}", c.confidence, c.review_recommended),
                first: members.iter().map(|e| e.start).min().unwrap_or(0),
                title: rows.first().map(|r| r.text.clone()).unwrap_or_default(),
                percent: format!("{:.0}%", c.confidence * 100.0),
                review: c.review_recommended,
                rows,
            }
        })
        .collect()
}

/// The contacts the scan has reached, one card each, updated in place; once the scan is done,
/// the entities no contact took, as chips.
fn cards(shown: Memo<Option<Shown>>, state: DemoState) -> impl IntoView {
    let all = Memo::new(move |_| shown.with(|s| s.as_ref().map(card_views).unwrap_or_default()));
    let visible =
        move || all.with(|c| written_so_far(c, |c: &CardView| c.first, state.reached.get()));
    let chips = move || {
        if !state.done.get() {
            return None;
        }
        shown.with(|s| {
            let s = s.as_ref()?;
            let loose: Vec<&Found> = s
                .unassigned
                .iter()
                .filter_map(|&i| s.found.get(i))
                .collect();
            let none = s
                .contacts
                .is_empty()
                .then(|| view! { <p class="waiting">"no contacts"</p> });
            let list = (!loose.is_empty()).then(|| {
                view! {
                    <div class="unassigned">
                        <span class="control-label">"unassigned"</span>
                        <div class="chips">
                            {loose
                                .into_iter()
                                .map(|e| chip(covered(&s.text, e.start, e.end), e.kind, e.start, state))
                                .collect_view()}
                        </div>
                    </div>
                }
            });
            Some(view! { {none} {list} })
        })
    };
    view! {
        <For each=visible key=|c| c.key.clone() children=move |c| card(c, state) />
        {chips}
    }
}

fn card(c: CardView, state: DemoState) -> impl IntoView {
    let rows = c
        .rows
        .into_iter()
        .map(|row| {
            let start = row.start;
            let (enter, leave) = state.hover(start);
            view! {
                <button
                    type="button"
                    class="row card-row"
                    class:selected=move || state.is_selected(start)
                    data-out=start.to_string()
                    on:mouseenter=enter
                    on:mouseleave=leave
                    on:click=move |_| state.reveal_in_document(start)
                >
                    <span class=format!("row-kind c-{}", kind_name(row.kind))>
                        {kind_name(row.kind)}
                    </span>
                    <span class="row-text">{row.text}</span>
                    <span class="row-range">{format!("{start}..{}", row.end)}</span>
                </button>
            }
        })
        .collect_view();
    view! {
        <article class="card" data-anim="">
            <header class="card-head">
                <span class="card-title">{c.title}</span>
                <span class="card-meta">
                    {c.percent}
                    {c.review.then(|| view! { <span class="badge">"review"</span> })}
                </span>
            </header>
            {rows}
        </article>
    }
}

fn chip(text: String, kind: FoundKind, start: usize, state: DemoState) -> impl IntoView {
    let (enter, leave) = state.hover(start);
    view! {
        <button
            type="button"
            class=format!("chip k-{}", kind_name(kind))
            class:selected=move || state.is_selected(start)
            data-out=start.to_string()
            on:mouseenter=enter
            on:mouseleave=leave
            on:click=move |_| state.reveal_in_document(start)
        >
            {text}
        </button>
    }
}
