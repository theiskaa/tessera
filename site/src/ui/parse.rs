//! "parse an address": one address in, its parts out, from `parse_address` in the worker. It
//! splits an address it is given; it does not look for addresses in other text.

use std::time::Duration;

use leptos::prelude::*;

use super::demo::{DemoState, Failure};
use super::highlight;
use super::json;
use crate::protocol::{Found, Request};

/// Quiet time after the last keystroke before the address is parsed.
const DEBOUNCE: Duration = Duration::from_millis(250);

/// One example per country the parser was trained on. Where a real place is named, it is a
/// public landmark; the rest are the stock example addresses of their countries.
const EXAMPLES: [(&str, &str); 5] = [
    ("GB", "Flat 4, 221B Baker Street, London NW1 6XE"),
    ("DE", "Musterstraße 12, 10115 Berlin"),
    ("US", "742 Evergreen Terrace, Springfield, OR 97477, USA"),
    ("GE", "14 Rustaveli Avenue, Tbilisi 0108, Georgia"),
    ("JP", "〒530-0001 大阪府大阪市北区梅田3丁目1-1"),
];

/// The parser's answer for one input, kept with that input.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Parsed {
    text: String,
    result: Result<Found, String>,
}

/// The box's state; its answers come through the demo's worker.
#[derive(Clone, Copy)]
pub(crate) struct ParseState {
    input: RwSignal<String>,
    parsed: RwSignal<Option<Parsed>>,
    /// Id and text of the newest request; answers to older ones are dropped.
    latest: StoredValue<(u64, String)>,
    debounce: StoredValue<Option<TimeoutHandle>, LocalStorage>,
}

impl ParseState {
    pub(crate) fn new() -> ParseState {
        ParseState {
            input: RwSignal::new(EXAMPLES[0].1.to_string()),
            parsed: RwSignal::new(None),
            latest: StoredValue::new((0, String::new())),
            debounce: StoredValue::new_local(None),
        }
    }

    /// Handles an answer, dropping it when a newer input has been sent since.
    pub(crate) fn receive(self, id: u64, result: Result<Found, String>) {
        let (latest, text) = self.latest.get_value();
        if id == latest {
            self.parsed.set(Some(Parsed { text, result }));
        }
    }

    /// Sends the input now; an empty input is not sent.
    fn send(self, demo: DemoState) {
        if let Some(handle) = self.debounce.try_update_value(Option::take).flatten() {
            handle.clear();
        }
        let text = self.input.get_untracked();
        let (id, _) = self.latest.get_value();
        let id = id.wrapping_add(1);
        self.latest.set_value((id, text.clone()));
        if text.trim().is_empty() {
            self.parsed.set(None);
            return;
        }
        demo.send(Request::ParseAddress { id, text });
    }

    fn type_in(self, demo: DemoState, value: String) {
        self.input.set(value);
        if let Some(handle) = self.debounce.try_update_value(Option::take).flatten() {
            handle.clear();
        }
        let handle = set_timeout_with_handle(move || self.send(demo), DEBOUNCE);
        self.debounce.set_value(handle.ok());
    }
}

/// The input with each component lit, labelled on hover.
fn highlighted(parsed: &Parsed, found: &Found) -> impl IntoView + use<> {
    let text = &parsed.text;
    let mut runs = Vec::new();
    let mut cursor = 0;
    for c in &found.components {
        if c.start < cursor {
            continue;
        }
        if let Some(before) = text.get(cursor..c.start).filter(|b| !b.is_empty()) {
            runs.push(view! { <span class="plain">{before.to_string()}</span> }.into_any());
        }
        if let Some(part) = text.get(c.start..c.end) {
            let unknown = c.label == "unknown";
            runs.push(
                view! {
                    <span class="part" class:unknown=unknown title=c.label.clone()>
                        {part.to_string()}
                    </span>
                }
                .into_any(),
            );
            cursor = c.end;
        }
    }
    if let Some(rest) = text.get(cursor..).filter(|r| !r.is_empty()) {
        runs.push(view! { <span class="plain">{rest.to_string()}</span> }.into_any());
    }
    runs.collect_view()
}

/// The answer in one line: the error, or the address's confidence.
fn summary(parsed: &Parsed) -> String {
    match &parsed.result {
        Err(error) if error.starts_with("InputTooLarge") => {
            format!("too long for one address: at most 256 tokens and 8 KiB ({error})")
        }
        Err(error) => format!("could not parse it: {error}"),
        Ok(found) if found.components.is_empty() => "no parts found".to_string(),
        Ok(found) if found.review_recommended => format!(
            "confidence {:.3} · low confidence: check the parts",
            found.confidence
        ),
        Ok(found) => format!("confidence {:.3}", found.confidence),
    }
}

fn result(parsed: Parsed, json_view: bool) -> AnyView {
    let found = match &parsed.result {
        Err(_) => return view! { <p class="error">{summary(&parsed)}</p> }.into_any(),
        Ok(found) if found.components.is_empty() => {
            return view! { <p class="waiting">{summary(&parsed)}</p> }.into_any();
        }
        Ok(found) => found.clone(),
    };
    if json_view {
        let body = format!(
            "{}{}{}{}",
            json::OPEN,
            json::FIRST,
            json::object(&parsed.text, &found),
            json::CLOSE
        );
        return view! { <pre class="json">{highlight::render(highlight::json(&body))}</pre> }
            .into_any();
    }
    let rows = found
        .components
        .iter()
        .map(|c| {
            let unknown = c.label == "unknown";
            view! {
                <tr class:unknown=unknown>
                    <td class="part-label">{c.label.clone()}</td>
                    <td>{parsed.text.get(c.start..c.end).unwrap_or("").to_string()}</td>
                    <td class="muted">{format!("{}..{}", c.start, c.end)}</td>
                    <td class="muted num">{format!("{:.3}", c.confidence)}</td>
                </tr>
            }
        })
        .collect_view();
    view! {
        <div class="parsed-text">{highlighted(&parsed, &found)}</div>
        <table class="parts">
            <thead>
                <tr>
                    <th scope="col">"label"</th>
                    <th scope="col">"text"</th>
                    <th scope="col">"bytes"</th>
                    <th scope="col" class="num">"confidence"</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
        <p class="parse-summary">{summary(&parsed)}</p>
    }
    .into_any()
}

/// The "parse an address" section.
#[component]
pub(crate) fn ParseBox(state: DemoState) -> impl IntoView {
    let parse = state.parse;
    let json_view = RwSignal::new(true);
    // The first parse goes out once the worker exists; the worker queues it until the bundle
    // has loaded.
    Effect::new(move |_| parse.send(state));
    // Once the model or the worker has failed no answer will come, so that is shown instead of
    // waiting, in the words the demo's output pane uses.
    let body = move || {
        let empty = parse.input.with(|t| t.trim().is_empty());
        match (state.failure.get(), parse.parsed.get()) {
            (Some(Failure::Model(message)), _) => {
                let message = format!("could not load the model: {message}");
                view! { <p class="error">{message}</p> }.into_any()
            }
            (Some(Failure::Worker(message)), _) => {
                let message = format!("the inference worker failed: {message}");
                view! { <p class="error">{message}</p> }.into_any()
            }
            _ if empty => view! { <p class="waiting">"type or paste one address"</p> }.into_any(),
            (None, None) => view! { <p class="waiting">"parsing"</p> }.into_any(),
            (None, Some(parsed)) => result(parsed, json_view.get()),
        }
    };

    view! {
        <section class="parse" aria-labelledby="parse-title">
            <div class="controls">
                <div class="control-group">
                    <span id="parse-title" class="control-label">"parse an address"</span>
                    {EXAMPLES
                        .iter()
                        .map(|&(code, address)| {
                            view! {
                                <button
                                    type="button"
                                    class="toggle"
                                    class:on=move || parse.input.with(|t| t == address)
                                    aria-pressed=move || {
                                        parse.input.with(|t| t == address).to_string()
                                    }
                                    aria-label=format!("example address, {code}")
                                    on:click=move |_| {
                                        parse.input.set(address.to_string());
                                        parse.send(state);
                                    }
                                >
                                    {code}
                                </button>
                            }
                        })
                        .collect_view()}
                </div>
                <div class="control-group">
                    <span class="control-label">"output"</span>
                    <button
                        type="button"
                        class="toggle"
                        class:on=move || json_view.get()
                        aria-pressed=move || json_view.get().to_string()
                        on:click=move |_| json_view.set(true)
                    >
                        "json"
                    </button>
                    <button
                        type="button"
                        class="toggle"
                        class:on=move || !json_view.get()
                        aria-pressed=move || (!json_view.get()).to_string()
                        on:click=move |_| json_view.set(false)
                    >
                        "table"
                    </button>
                </div>
            </div>
            <p class="note">
                "Splits one address into its parts with "<code>"parse_address"</code>
                "; it does not find addresses in other text."
            </p>
            <div class="parse-box">
                <label for="parse-input" class="visually-hidden">"address"</label>
                <textarea
                    id="parse-input"
                    class="parse-input"
                    rows="2"
                    spellcheck="false"
                    prop:value=move || parse.input.get()
                    on:input=move |ev| parse.type_in(state, event_target_value(&ev))
                ></textarea>
                <div class="parse-result">{body}</div>
                // The result is rebuilt on every answer, and a live region created along with its
                // text is not reliably announced, so only the summary is announced, from here.
                <p class="visually-hidden" aria-live="polite">
                    {move || parse.parsed.with(|p| p.as_ref().map(summary))}
                </p>
            </div>
        </section>
    }
}
