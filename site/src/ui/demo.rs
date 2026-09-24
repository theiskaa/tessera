//! The live demo: a document whose people, organizations, addresses, emails, and phone numbers
//! light up as a scan line passes over it, while the list or JSON of what the library returned
//! is written beside it on the same clock. The reader can edit the document and change the phone
//! region. Offsets are UTF-8 bytes into the document.

use std::sync::Arc;
use std::time::Duration;

use gloo_worker::WorkerBridge;
use leptos::html::Div;
use leptos::prelude::*;
use web_sys::{
    HtmlDivElement, ScrollBehavior, ScrollIntoViewOptions, ScrollLogicalPosition, ScrollToOptions,
};

use super::document::DocumentPane;
use super::json::kind_name;
use super::output::OutputPane;
use super::parse::ParseState;
use super::stream::{self, Step};
use crate::inference::Inference;
use crate::protocol::{Found, FoundKind, Request, Response};
use crate::samples::SAMPLES;

/// Longest document the demo analyzes, in bytes.
pub(crate) const MAX_TEXT: usize = 20 * 1024;
/// Quiet time after the last keystroke before an edited document is analyzed.
const DEBOUNCE: Duration = Duration::from_millis(300);

/// The regions the phone tables cover, offered as the country hint.
pub(crate) const HINTS: [&str; 11] = [
    "GB", "DE", "US", "GE", "JP", "AT", "BE", "CH", "IE", "NL", "CA",
];

/// What the library returned for one text, kept with that text so the output never mixes a
/// result with a newer edit.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Shown {
    pub(crate) text: Arc<str>,
    pub(crate) found: Arc<Vec<Found>>,
}

/// Why nothing can be analyzed.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Failure {
    /// The worker runs but could not fetch, verify, or load the bundle.
    Model(String),
    /// The worker could not start, or stopped with an error.
    Worker(String),
}

/// Everything the demo shows, as signals the page and the worker callback share.
#[derive(Clone, Copy)]
pub(crate) struct DemoState {
    pub(crate) sample: RwSignal<usize>,
    /// The document as it is now, edited or not.
    pub(crate) text: RwSignal<Arc<str>>,
    pub(crate) hint: RwSignal<String>,
    pub(crate) editing: RwSignal<bool>,
    /// The newest answer to a request still wanted.
    pub(crate) answer: RwSignal<Option<Result<Shown, String>>>,
    pub(crate) failure: RwSignal<Option<Failure>>,
    /// Whether the document is over [`MAX_TEXT`] and so was not sent.
    pub(crate) too_long: RwSignal<bool>,
    /// Bumped to restart the scan; its parity picks one of two identical keyframe sets.
    pub(crate) run: RwSignal<u32>,
    /// Seconds of the current scan.
    pub(crate) scan: RwSignal<f64>,
    /// Whether a scan is running; highlights animate only then.
    pub(crate) scanning: RwSignal<bool>,
    /// Entities starting before this byte are written to the output.
    pub(crate) reached: RwSignal<usize>,
    pub(crate) done: RwSignal<bool>,
    /// The entity whose row was clicked, by its first byte.
    pub(crate) selected: RwSignal<Option<usize>>,
    pub(crate) json: RwSignal<bool>,
    /// Whether the output pane keeps its newest line in view; the reader scrolling turns it off.
    pub(crate) follow: StoredValue<bool>,
    /// The scrolling boxes of the document and output panes.
    pub(crate) document: NodeRef<Div>,
    pub(crate) output: NodeRef<Div>,
    /// The "parse an address" box, whose answers come through the same worker.
    pub(crate) parse: ParseState,
    /// Id and text of the newest analysis; answers to older ones are dropped.
    latest: StoredValue<(u64, Arc<str>)>,
    /// Seconds of the scan that plays when the newest analysis is answered.
    next_scan: StoredValue<f64>,
    timers: StoredValue<Vec<TimeoutHandle>, LocalStorage>,
    debounce: StoredValue<Option<TimeoutHandle>, LocalStorage>,
    bridge: StoredValue<Option<WorkerBridge<Inference>>, LocalStorage>,
}

impl DemoState {
    pub(crate) fn new() -> DemoState {
        DemoState {
            sample: RwSignal::new(0),
            text: RwSignal::new(Arc::from(SAMPLES[0].text)),
            hint: RwSignal::new(SAMPLES[0].country_hint.to_string()),
            editing: RwSignal::new(false),
            answer: RwSignal::new(None),
            failure: RwSignal::new(None),
            too_long: RwSignal::new(false),
            run: RwSignal::new(0),
            scan: RwSignal::new(stream::SCAN),
            scanning: RwSignal::new(false),
            reached: RwSignal::new(0),
            done: RwSignal::new(false),
            selected: RwSignal::new(None),
            json: RwSignal::new(true),
            follow: StoredValue::new(true),
            document: NodeRef::new(),
            output: NodeRef::new(),
            parse: ParseState::new(),
            latest: StoredValue::new((0, Arc::from(""))),
            next_scan: StoredValue::new(stream::SCAN),
            timers: StoredValue::new_local(Vec::new()),
            debounce: StoredValue::new_local(None),
            bridge: StoredValue::new_local(None),
        }
    }

    /// Hands the state the worker to send requests to, and analyzes the first sample.
    pub(crate) fn connect(self, bridge: WorkerBridge<Inference>) {
        self.bridge.set_value(Some(bridge));
        self.pick(0);
    }

    pub(crate) fn send(self, request: Request) {
        self.bridge.with_value(|bridge| {
            if let Some(bridge) = bridge {
                bridge.send(request);
            }
        });
    }

    /// Handles a worker answer, dropping it when a newer request has been sent since.
    pub(crate) fn receive(self, response: Response) {
        match response {
            Response::LoadFailed(message) => self.fail(Failure::Model(message)),
            Response::Detected { id, result } => {
                let (latest, text) = self.latest.get_value();
                if id != latest {
                    return;
                }
                let shown = result.map(|found| Shown {
                    text,
                    found: Arc::new(found),
                });
                self.answer.set(Some(shown));
                self.replay_with(self.next_scan.get_value());
            }
            Response::Parsed { id, result } => self.parse.receive(id, result),
        }
    }

    /// Records why no more answers will come. The first failure is kept.
    pub(crate) fn fail(self, failure: Failure) {
        if self.failure.get_untracked().is_none() {
            self.failure.set(Some(failure));
        }
    }

    /// Shows sample `index` as written, and scans it.
    pub(crate) fn pick(self, index: usize) {
        let Some(sample) = SAMPLES.get(index) else {
            return;
        };
        self.selected.set(None);
        self.sample.set(index);
        self.editing.set(false);
        self.too_long.set(false);
        self.text.set(Arc::from(sample.text));
        self.hint.set(sample.country_hint.to_string());
        self.answer.set(None);
        self.analyze(stream::SCAN);
    }

    /// Restores the current sample's text and hint.
    pub(crate) fn reset(self) {
        self.pick(self.sample.get_untracked());
    }

    /// Whether the document differs from the sample as written.
    pub(crate) fn edited(self) -> bool {
        SAMPLES
            .get(self.sample.get())
            .is_some_and(|s| *self.text.get() != *s.text)
    }

    /// Switches between reading and editing. Leaving edit mode analyzes any pending edit and
    /// scans the result in full.
    pub(crate) fn toggle_editing(self) {
        let editing = !self.editing.get_untracked();
        self.selected.set(None);
        self.editing.set(editing);
        if editing {
            return;
        }
        let text = self.text.get_untracked();
        let pending = self.debounce.with_value(Option::is_some);
        let (_, sent) = self.latest.get_value();
        let answered = matches!(self.answer.get_untracked(), Some(Ok(ref s)) if s.text == text);
        if pending || *sent != *text {
            self.analyze(stream::SCAN);
        } else if answered {
            self.replay();
        } else {
            // The answer for this text is on its way; it plays the full scan when it lands.
            self.next_scan.set_value(stream::SCAN);
        }
    }

    /// Takes an edit, and analyzes it once typing pauses.
    pub(crate) fn input(self, value: String) {
        self.text.set(Arc::from(value));
        self.cancel_debounce();
        let handle = set_timeout_with_handle(
            move || {
                self.debounce.set_value(None);
                self.analyze(stream::QUICK_SCAN);
            },
            DEBOUNCE,
        );
        self.debounce.set_value(handle.ok());
    }

    /// Sets the region for phone numbers without a country code, and analyzes again.
    pub(crate) fn set_hint(self, hint: String) {
        self.hint.set(hint);
        self.analyze(stream::QUICK_SCAN);
    }

    fn cancel_debounce(self) {
        if let Some(handle) = self.debounce.try_update_value(Option::take).flatten() {
            handle.clear();
        }
    }

    /// Sends the document as it is now. A text over the size limit is not sent, and any answer
    /// still coming for an older text is dropped either way.
    fn analyze(self, scan: f64) {
        self.cancel_debounce();
        let text = self.text.get_untracked();
        let (id, _) = self.latest.get_value();
        let id = id.wrapping_add(1);
        self.latest.set_value((id, text.clone()));
        let too_long = text.len() > MAX_TEXT;
        if self.too_long.get_untracked() != too_long {
            self.too_long.set(too_long);
        }
        if too_long {
            return;
        }
        self.next_scan.set_value(scan);
        self.send(Request::Detect {
            id,
            text: text.to_string(),
            country_hint: vec![self.hint.get_untracked()],
        });
    }

    /// Restarts the full scan from the top of both panes with nothing written.
    pub(crate) fn replay(self) {
        self.replay_with(stream::SCAN);
    }

    /// Restarts a `scan`-second scan. The jump to the top is instant: a smooth scroll would
    /// still be running when the new output replaces the old one.
    fn replay_with(self, scan: f64) {
        self.timers.update_value(|timers| {
            for t in timers.drain(..) {
                t.clear();
            }
        });
        self.scan.set(scan);
        self.run.update(|r| *r = r.wrapping_add(1));
        self.selected.set(None);
        self.follow.set_value(true);
        let top = ScrollToOptions::new();
        top.set_top(0.0);
        top.set_behavior(ScrollBehavior::Instant);
        for pane in [self.document, self.output] {
            if let Some(pane) = pane.get_untracked() {
                pane.scroll_to_with_scroll_to_options(&top);
            }
        }
        self.reached.set(0);
        self.done.set(false);
        let Some(Ok(shown)) = self.answer.get_untracked() else {
            self.scanning.set(false);
            return;
        };
        if reduced_motion() {
            self.scanning.set(false);
            self.follow.set_value(false);
            self.write(Step::Done);
            return;
        }
        self.scanning.set(true);
        let mut handles: Vec<TimeoutHandle> =
            stream::schedule(&shown.found, shown.text.len(), scan)
                .into_iter()
                .filter_map(|(at, step)| {
                    let wait = Duration::from_secs_f64(at.max(0.0));
                    set_timeout_with_handle(move || self.write(step), wait).ok()
                })
                .collect();
        let settled = Duration::from_secs_f64(scan + stream::SETTLE);
        handles.extend(set_timeout_with_handle(move || self.scanning.set(false), settled).ok());
        self.timers.set_value(handles);
    }

    fn write(self, step: Step) {
        match step {
            Step::Reach(at) => self.reached.set(at),
            Step::Done => {
                self.reached.set(usize::MAX);
                self.done.set(true);
            }
        }
        if self.follow.get_value() {
            let output = self.output;
            request_animation_frame(move || {
                if let Some(pane) = output.get_untracked() {
                    let bottom = ScrollToOptions::new();
                    bottom.set_top(f64::from(pane.scroll_height()));
                    pane.scroll_to_with_scroll_to_options(&bottom);
                }
            });
        }
    }

    /// Selects the entity starting at byte `start`, or clears the selection when it is already
    /// selected, and brings a newly selected entity into view in the document pane.
    pub(crate) fn toggle(self, start: usize) {
        let selecting = self.selected.get_untracked() != Some(start);
        self.selected.set(selecting.then_some(start));
        if selecting && let Some(pane) = self.document.get_untracked() {
            reveal(&pane, start);
        }
    }
}

fn reduced_motion() -> bool {
    window()
        .match_media("(prefers-reduced-motion: reduce)")
        .ok()
        .flatten()
        .is_some_and(|query| query.matches())
}

/// Scrolls the document pane so the entity starting at byte `start` sits in its middle, then
/// brings the pane itself into view if it is off screen, as on a phone where the panes stack.
/// Whether either scroll is smooth is left to CSS, which turns it off for reduced motion.
fn reveal(pane: &HtmlDivElement, start: usize) {
    let Ok(Some(span)) = pane.query_selector(&format!("[data-entity=\"{start}\"]")) else {
        return;
    };
    let frame = pane.get_bounding_client_rect();
    let found = span.get_bounding_client_rect();
    let offset =
        found.top() - frame.top() - (f64::from(pane.client_height()) - found.height()) / 2.0;
    let by = ScrollToOptions::new();
    by.set_top(offset);
    pane.scroll_by_with_scroll_to_options(&by);
    let into = ScrollIntoViewOptions::new();
    into.set_block(ScrollLogicalPosition::Nearest);
    pane.scroll_into_view_with_scroll_into_view_options(&into);
}

/// Legend order.
pub(crate) const KINDS: [FoundKind; 5] = [
    FoundKind::Person,
    FoundKind::Org,
    FoundKind::Address,
    FoundKind::Phone,
    FoundKind::Email,
];

/// The demo section.
#[component]
pub(crate) fn Demo(state: DemoState) -> impl IntoView {
    let phase = move || {
        let phase = match (state.failure.get(), state.answer.get()) {
            (_, Some(Ok(_))) if state.run.get().is_multiple_of(2) => "panes run-a",
            (_, Some(Ok(_))) => "panes run-b",
            (Some(_), _) | (None, Some(Err(_))) => "panes failed",
            (None, None) => "panes idle",
        };
        if state.scanning.get() {
            format!("{phase} scanning")
        } else {
            phase.to_string()
        }
    };
    view! {
        <section id="demo" class="demo" aria-label="Live demo">
            <div class="controls">
                <div class="control-group">
                    <span class="control-label">"document"</span>
                    {SAMPLES
                        .iter()
                        .enumerate()
                        .map(|(i, s)| {
                            view! {
                                <button
                                    type="button"
                                    class="toggle"
                                    class:on=move || state.sample.get() == i
                                    aria-pressed=move || (state.sample.get() == i).to_string()
                                    on:click=move |_| state.pick(i)
                                >
                                    {s.name}
                                </button>
                            }
                        })
                        .collect_view()}
                    <button type="button" class="replay" on:click=move |_| state.replay()>
                        "[ replay ]"
                    </button>
                </div>
                <div class="control-group">
                    <span class="control-label">"output"</span>
                    <button
                        type="button"
                        class="toggle"
                        class:on=move || state.json.get()
                        aria-pressed=move || state.json.get().to_string()
                        on:click=move |_| state.json.set(true)
                    >
                        "json"
                    </button>
                    <button
                        type="button"
                        class="toggle"
                        class:on=move || !state.json.get()
                        aria-pressed=move || (!state.json.get()).to_string()
                        on:click=move |_| state.json.set(false)
                    >
                        "list"
                    </button>
                </div>
            </div>
            <div class=phase style=move || format!("--scan: {:.2}s", state.scan.get())>
                <DocumentPane state=state />
                <OutputPane state=state />
            </div>
            {move || {
                state
                    .too_long
                    .get()
                    .then(|| {
                        view! {
                            <p class="note-line error" aria-live="polite">
                                {format!(
                                    "The text is over {} KB and is not analyzed; shorten it to see results.",
                                    MAX_TEXT / 1024,
                                )}
                            </p>
                        }
                    })
            }}
            <div class="legend-line">
                <span class="legend">
                    {KINDS
                        .iter()
                        .map(|&k| {
                            view! {
                                <span class="legend-item">
                                    <span class=format!("swatch k-{}", kind_name(k))></span>
                                    {kind_name(k)}
                                </span>
                            }
                        })
                        .collect_view()}
                </span>
                <span>"click a row to find it in the document · offsets are UTF-8 bytes"</span>
            </div>
            <p class="note">
                "Everything highlighted is "<code>"detect"</code>
                "'s output for the whole document, run in a web worker in this browser: emails and phones from validating rules, people, organizations and addresses from an int8 network, and each address split into parts by "
                <code>"parse_address"</code>"."
            </p>
        </section>
    }
}
