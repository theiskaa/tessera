//! The live demo: a document whose people, organizations, addresses, emails, and phone numbers
//! light up as a scan line passes over it, while what the library returned is written beside it
//! on the same clock, as JSON, a list, or contact cards. The reader can edit the document,
//! select part of it to analyze that part alone, and change the phone region. Offsets are UTF-8
//! bytes into the document.

use std::sync::Arc;
use std::time::Duration;

use gloo_worker::WorkerBridge;
use leptos::ev;
use leptos::html::Div;
use leptos::prelude::*;
use web_sys::{
    HtmlDivElement, MouseEvent, ScrollBehavior, ScrollIntoViewOptions, ScrollLogicalPosition,
    ScrollToOptions,
};

use super::document::{self, DocumentPane};
use super::json::kind_name;
use super::output::OutputPane;
use super::parse::ParseState;
use super::selection;
use super::stream::{self, Step};
use crate::inference::Inference;
use crate::protocol::{Card, Found, FoundKind, Request, Response};
use crate::samples::SAMPLES;

/// Longest document the demo analyzes, in bytes.
pub(crate) const MAX_TEXT: usize = 20 * 1024;
/// Quiet time after the last keystroke before an edited document is analyzed.
const DEBOUNCE: Duration = Duration::from_millis(300);

/// The hint that sends none, so the library reads the region from the document itself.
pub(crate) const AUTO: &str = "auto";

/// `AUTO`, then the regions the phone tables cover, offered as the country hint.
pub(crate) const HINTS: [&str; 12] = [
    AUTO, "GB", "DE", "US", "GE", "JP", "AT", "BE", "CH", "IE", "NL", "CA",
];

/// What the library returned for one text, kept with that text so the output never mixes a
/// result with a newer edit.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Shown {
    pub(crate) text: Arc<str>,
    /// The bytes of `text` that were analyzed, when the reader selected a section.
    pub(crate) section: Option<(usize, usize)>,
    pub(crate) found: Arc<Vec<Found>>,
    pub(crate) contacts: Arc<Vec<Card>>,
    pub(crate) unassigned: Arc<Vec<usize>>,
}

impl Shown {
    /// The bytes the scan crosses: the section, or the whole text.
    pub(crate) fn span(&self) -> (usize, usize) {
        self.section.unwrap_or((0, self.text.len()))
    }

    /// How many bytes the scan crosses.
    pub(crate) fn span_len(&self) -> usize {
        let (start, end) = self.span();
        end - start
    }
}

/// An analysis request: its id, the document it was for, and the section of it that was sent.
#[derive(Debug, Clone)]
struct Sent {
    id: u64,
    text: Arc<str>,
    section: Option<(usize, usize)>,
}

/// How the output pane shows the answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum View {
    /// `detect`'s entities as the CLI prints them.
    Json,
    /// `detect`'s entities one per row, address components under their address.
    List,
    /// `extract_contacts`'s contacts, and what it left unassigned.
    Cards,
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
    /// The selected bytes the next analysis covers; `None` analyzes the whole document.
    pub(crate) section: RwSignal<Option<(usize, usize)>>,
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
    /// The entity lit in both panes, by its first byte: the one under the pointer, else the
    /// one last clicked.
    pub(crate) selected: RwSignal<Option<usize>>,
    /// The entity last clicked, which the selection returns to when the pointer leaves another.
    pinned: StoredValue<Option<usize>>,
    /// Whether the pointer went down in the document, so letting go anywhere takes the text it
    /// dragged over as a section.
    dragging: StoredValue<bool>,
    pub(crate) view: RwSignal<View>,
    /// The scrolling boxes of the document and output panes.
    pub(crate) document: NodeRef<Div>,
    pub(crate) output: NodeRef<Div>,
    /// The "parse an address" box, whose answers come through the same worker.
    pub(crate) parse: ParseState,
    /// The newest analysis; answers to older ones are dropped.
    latest: StoredValue<Sent>,
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
            hint: RwSignal::new(AUTO.to_string()),
            section: RwSignal::new(None),
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
            pinned: StoredValue::new(None),
            dragging: StoredValue::new(false),
            view: RwSignal::new(View::Json),
            document: NodeRef::new(),
            output: NodeRef::new(),
            parse: ParseState::new(),
            latest: StoredValue::new(Sent {
                id: 0,
                text: Arc::from(""),
                section: None,
            }),
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
            Response::Analyzed { id, result } => {
                let Sent {
                    id: latest,
                    text,
                    section,
                } = self.latest.get_value();
                if id != latest {
                    return;
                }
                let base = section.map_or(0, |(start, _)| start);
                let shown = result.map(|analysis| Shown {
                    text,
                    section,
                    found: Arc::new(
                        analysis
                            .found
                            .into_iter()
                            .map(|f| f.shifted(base))
                            .collect(),
                    ),
                    contacts: Arc::new(analysis.contacts),
                    unassigned: Arc::new(analysis.unassigned),
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
        self.clear_selection();
        self.sample.set(index);
        self.editing.set(false);
        self.too_long.set(false);
        self.text.set(Arc::from(sample.text));
        self.hint.set(AUTO.to_string());
        self.section.set(None);
        self.answer.set(None);
        self.analyze(stream::SCAN);
    }

    /// Restores the current sample's text, with the region read from it.
    pub(crate) fn reset(self) {
        self.pick(self.sample.get_untracked());
    }

    /// Whether the document differs from the sample as written.
    pub(crate) fn edited(self) -> bool {
        SAMPLES
            .get(self.sample.get())
            .is_some_and(|s| *self.text.get() != *s.text)
    }

    /// Switches between reading and editing. Editing drops any selected section; leaving edit
    /// mode analyzes any pending edit, or the whole document when a section was analyzed last,
    /// and scans the result in full.
    pub(crate) fn toggle_editing(self) {
        let editing = !self.editing.get_untracked();
        self.clear_selection();
        self.editing.set(editing);
        if editing {
            self.section.set(None);
            return;
        }
        let text = self.text.get_untracked();
        let pending = self.debounce.with_value(Option::is_some);
        let sent = self.latest.get_value();
        let answered = matches!(self.answer.get_untracked(), Some(Ok(ref s)) if s.text == text);
        if pending || *sent.text != *text || sent.section.is_some() {
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
        self.section.set(None);
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

    /// The pointer went down in the document. A double or triple click selects a word or a
    /// line to read, not a section to analyze, so only a single press starts a drag.
    pub(crate) fn press(self, ev: &MouseEvent) {
        let single = ev.detail() <= 1 && ev.button() == 0;
        self.dragging
            .set_value(single && !self.editing.get_untracked());
    }

    /// The pointer came up, anywhere on the page: after a drag in the document, the text it
    /// selected is analyzed alone. Its highlights and output replace the whole document's, and
    /// its scan covers only its lines.
    fn release(self, ev: &MouseEvent) {
        if !self
            .dragging
            .try_update_value(std::mem::take)
            .unwrap_or(false)
            || ev.detail() > 1
        {
            return;
        }
        let Some(doc) = self
            .document
            .get_untracked()
            .and_then(|pane| pane.query_selector(".doc").ok().flatten())
        else {
            return;
        };
        let text = self.text.get_untracked();
        let Some((start, end)) = selection::take(&doc, &text) else {
            return;
        };
        let Some(section) = selection::section(&text, start, end) else {
            return;
        };
        self.section.set(section);
        self.clear_selection();
        self.analyze(stream::SCAN);
    }

    /// Goes back from a section to the whole document.
    pub(crate) fn whole_document(self) {
        self.section.set(None);
        self.clear_selection();
        self.analyze(stream::SCAN);
    }

    /// Sends the document, or its selected section, as it is now. A text over the size limit
    /// is not sent, and any answer still coming for an older text is dropped either way.
    fn analyze(self, scan: f64) {
        self.cancel_debounce();
        let text = self.text.get_untracked();
        let section = self
            .section
            .get_untracked()
            .filter(|&(start, end)| text.get(start..end).is_some());
        let sent = section.map_or(&*text, |(start, end)| &text[start..end]);
        let id = self.latest.with_value(|s| s.id).wrapping_add(1);
        self.latest.set_value(Sent {
            id,
            text: text.clone(),
            section,
        });
        let too_long = sent.len() > MAX_TEXT;
        if self.too_long.get_untracked() != too_long {
            self.too_long.set(too_long);
        }
        if too_long {
            return;
        }
        self.next_scan.set_value(scan);
        let hint = self.hint.get_untracked();
        self.send(Request::Analyze {
            id,
            text: sent.to_string(),
            country_hint: if hint == AUTO { Vec::new() } else { vec![hint] },
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
        self.clear_selection();
        self.reached.set(0);
        self.done.set(false);
        let Some(Ok(shown)) = self.answer.get_untracked() else {
            self.scanning.set(false);
            return;
        };
        let top = ScrollToOptions::new();
        top.set_top(0.0);
        top.set_behavior(ScrollBehavior::Instant);
        // A section is scanned where the reader selected it, so the document stays put.
        let panes = if shown.section.is_some() {
            vec![self.output]
        } else {
            vec![self.document, self.output]
        };
        for pane in panes {
            if let Some(pane) = pane.get_untracked() {
                pane.scroll_to_with_scroll_to_options(&top);
            }
        }
        let pane = self.document;
        request_animation_frame(move || {
            if let Some(pane) = pane.get_untracked() {
                document::fit_scan_line(&pane);
            }
        });
        if reduced_motion() {
            self.scanning.set(false);
            self.write(Step::Done);
            return;
        }
        self.scanning.set(true);
        let mut handles: Vec<TimeoutHandle> = stream::schedule(&shown.found, shown.span(), scan)
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
    }

    /// Clears what is lit and what was clicked.
    fn clear_selection(self) {
        self.pinned.set_value(None);
        self.selected.set(None);
    }

    /// Whether the entity starting at byte `start` is lit; tracks the selection.
    pub(crate) fn is_selected(self, start: usize) -> bool {
        self.selected.get() == Some(start)
    }

    /// Handlers that light the entity starting at byte `start` while the pointer is over its
    /// highlight, line, object, card row, or chip, and on leaving return to the entity last
    /// clicked. An entity the scan has not reached yet is still unlit, so hovering it does
    /// nothing.
    pub(crate) fn hover(
        self,
        start: usize,
    ) -> (impl Fn(MouseEvent) + Copy, impl Fn(MouseEvent) + Copy) {
        (
            move |_| {
                if start < self.reached.get_untracked() {
                    self.selected.set(Some(start));
                }
            },
            move |_| {
                if self.selected.get_untracked() == Some(start) {
                    self.selected.set(self.pinned.get_value());
                }
            },
        )
    }

    /// Lights the entity starting at byte `start` until another is clicked, and brings its
    /// highlight into view in the document pane: what clicking its result does.
    pub(crate) fn reveal_in_document(self, start: usize) {
        self.pin(start);
        if let Some(pane) = self.document.get_untracked() {
            reveal(&pane, &format!("[data-entity=\"{start}\"]"));
        }
    }

    /// The same from the document's side: brings the entity's line, object, or card row into
    /// view in the output pane, once the scan has written it there.
    pub(crate) fn reveal_in_output(self, start: usize) {
        if start >= self.reached.get_untracked() {
            return;
        }
        self.pin(start);
        if let Some(pane) = self.output.get_untracked() {
            // Every view stays in the page; only the shown one has boxes to scroll to.
            let shown = [".rows", ".json", ".cards"]
                .map(|view| format!("{view}:not([hidden]) [data-out=\"{start}\"]"))
                .join(", ");
            reveal(&pane, &shown);
        }
    }

    fn pin(self, start: usize) {
        self.pinned.set_value(Some(start));
        self.selected.set(Some(start));
    }
}

fn reduced_motion() -> bool {
    window()
        .match_media("(prefers-reduced-motion: reduce)")
        .ok()
        .flatten()
        .is_some_and(|query| query.matches())
}

/// Scrolls `pane` so the first element matching `selector` sits in its middle, then brings
/// the pane itself into view if it is off screen, as on a phone where the panes stack.
/// Whether either scroll is smooth is left to CSS, which turns it off for reduced motion.
fn reveal(pane: &HtmlDivElement, selector: &str) {
    let Ok(Some(span)) = pane.query_selector(selector) else {
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
const KINDS: [FoundKind; 5] = [
    FoundKind::Person,
    FoundKind::Org,
    FoundKind::Address,
    FoundKind::Phone,
    FoundKind::Email,
];

/// The demo section.
#[component]
pub(crate) fn Demo(state: DemoState) -> impl IntoView {
    // A drag that starts in the document may end anywhere on the page.
    let release = window_event_listener(ev::mouseup, move |ev| state.release(&ev));
    on_cleanup(move || release.remove());
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
                    {[(View::Json, "json"), (View::List, "list"), (View::Cards, "contacts")]
                        .into_iter()
                        .map(|(view, name)| {
                            view! {
                                <button
                                    type="button"
                                    class="toggle"
                                    class:on=move || state.view.get() == view
                                    aria-pressed=move || (state.view.get() == view).to_string()
                                    on:click=move |_| state.view.set(view)
                                >
                                    {name}
                                </button>
                            }
                        })
                        .collect_view()}
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
                <span>
                    "select text to analyze just that part · click a result or a highlight to find its match"
                </span>
            </div>
            <p class="note">
                "Everything highlighted is what "<code>"extract_contacts"</code>
                " returned for the document, or for the part of it you selected, run in a web worker in this browser: emails and phones from validating rules, people, organizations and addresses from an int8 network, and each address split into its parts by a second one. Offsets are UTF-8 bytes."
            </p>
        </section>
    }
}
