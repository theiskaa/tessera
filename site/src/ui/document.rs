//! The document pane: the text with what the library found lit up, or a text box while the
//! reader edits it. Every run of the text carries its first byte in `data-start`, which is how
//! a selection maps back to bytes, and the runs of a selected section carry `data-in`, which is
//! how the scan line finds that section's lines.

use leptos::html::Textarea;
use leptos::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{Element, HtmlDivElement, HtmlElement};

use super::demo::{DemoState, MAX_TEXT};
use super::format::thousands;
use super::json::kind_name;
use super::stream;
use crate::protocol::{Found, FoundKind};
use crate::samples::SAMPLES;

/// A run of the document, plain or covering entity `entity`.
struct Segment {
    text: String,
    start: usize,
    entity: Option<usize>,
}

/// The document as runs, plain or covering one entity, with plain runs also cut where the
/// analyzed `span` begins and ends, so each run lies wholly inside or outside it.
fn segments(text: &str, found: &[Found], span: (usize, usize)) -> Vec<Segment> {
    let mut out = Vec::new();
    let plain = |from: usize, to: usize, out: &mut Vec<Segment>| {
        let mut cuts = vec![from, to];
        cuts.extend([span.0, span.1].into_iter().filter(|&c| from < c && c < to));
        cuts.sort_unstable();
        for pair in cuts.windows(2) {
            if let Some(run) = text.get(pair[0]..pair[1]).filter(|r| !r.is_empty()) {
                out.push(Segment {
                    text: run.to_string(),
                    start: pair[0],
                    entity: None,
                });
            }
        }
    };
    let mut cursor = 0;
    for (i, e) in found.iter().enumerate() {
        // An entity overlapping one already drawn stays in the list but is drawn once.
        if e.start < cursor || text.get(cursor..e.start).is_none() {
            continue;
        }
        let Some(covered) = text.get(e.start..e.end) else {
            continue;
        };
        plain(cursor, e.start, &mut out);
        out.push(Segment {
            text: covered.to_string(),
            start: e.start,
            entity: Some(i),
        });
        cursor = e.end;
    }
    plain(cursor, text.len(), &mut out);
    out
}

/// Where a run starts, as a share of the scanned span: its highlight waits that share of the
/// scan.
fn place(start: usize, span: (usize, usize)) -> String {
    format!("--f: {:.5}", stream::share(start, span))
}

/// Longest text the edit box takes, in UTF-16 units: well past the analysis limit, so a long
/// paste is shown as too long rather than cut, but bounded.
const MAX_EDIT_UNITS: usize = 4 * MAX_TEXT;

/// What identifies a drawn run: a new mark, or a new section, redraws only the runs it changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct RunKey {
    start: usize,
    end: usize,
    kind: Option<FoundKind>,
    place: Place,
}

/// Where a run lies against the section the reader selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Place {
    /// No section is selected: the whole document is analyzed and scanned.
    Whole,
    /// Inside the selected section, whose lines the scan line spans.
    Inside,
    /// Outside it: neither analyzed nor scanned.
    Outside,
}

/// One drawn run of the document.
#[derive(Debug, Clone, PartialEq)]
struct Drawn {
    key: RunKey,
    text: String,
}

/// The document pane.
#[component]
pub(crate) fn DocumentPane(state: DemoState) -> impl IntoView {
    let area: NodeRef<Textarea> = NodeRef::new();
    let file = move || SAMPLES.get(state.sample.get()).map_or("", |s| s.file);
    // The answer is drawn only while it is for the text on screen.
    let drawn = Memo::new(move |_| match state.answer.get() {
        Some(Ok(shown)) if *shown.text == *state.text.get() => {
            let span = shown.span();
            Some(
                segments(&shown.text, &shown.found, span)
                    .into_iter()
                    .map(|seg| {
                        let place = match shown.section {
                            None => Place::Whole,
                            Some(_) if seg.start < span.0 || seg.start >= span.1 => Place::Outside,
                            Some(_) => Place::Inside,
                        };
                        Drawn {
                            key: RunKey {
                                start: seg.start,
                                end: seg.start + seg.text.len(),
                                kind: seg.entity.and_then(|i| shown.found.get(i)).map(|e| e.kind),
                                place,
                            },
                            text: seg.text,
                        }
                    })
                    .collect::<Vec<_>>(),
            )
        }
        _ => None,
    });
    let answered = Memo::new(move |_| drawn.with(Option::is_some));
    let span = move || match state.answer.get() {
        Some(Ok(shown)) => shown.span(),
        _ => (0, state.text.with(|t| t.len())),
    };

    let run = move |d: Drawn| {
        let start = d.key.start;
        let style = move || place(start, span());
        let inside = (d.key.place == Place::Inside).then_some("");
        match d.key.kind {
            None if d.key.place == Place::Outside => view! {
                <span class="plain outside" data-start=start.to_string()>
                    {d.text}
                </span>
            }
            .into_any(),
            None => view! {
                <span
                    class="plain"
                    data-anim=""
                    data-start=start.to_string()
                    data-in=inside
                    style=style
                >
                    {d.text}
                </span>
            }
            .into_any(),
            Some(kind) => {
                // Drawn outside a scan, a mark is new: it fades in instead of being scanned,
                // until the next scan takes it over.
                let born = (!state.scanning.get_untracked()).then(|| state.run.get_untracked());
                let fresh = move || born.is_some_and(|run| run == state.run.get());
                let (enter, leave) = state.hover(start);
                view! {
                    <span
                        class=format!("hit k-{}", kind_name(kind))
                        class:fresh=fresh
                        class:selected=move || state.is_selected(start)
                        role="button"
                        tabindex="0"
                        on:mouseenter=enter
                        on:mouseleave=leave
                        on:click=move |_| state.reveal_in_output(start)
                        on:keydown=move |ev| {
                            if ev.key() == "Enter" || ev.key() == " " {
                                ev.prevent_default();
                                state.reveal_in_output(start);
                            }
                        }
                        data-entity=start.to_string()
                        data-start=start.to_string()
                        data-in=inside
                        data-anim=""
                        style=style
                    >
                        {d.text}
                    </span>
                }
                .into_any()
            }
        }
    };

    let body = move || {
        if state.editing.get() {
            let text = state.text.get_untracked().to_string();
            request_animation_frame(move || {
                if let Some(area) = area.get_untracked() {
                    let _ = area.focus();
                }
            });
            return view! {
                <textarea
                    class="doc-edit"
                    aria-label="document text"
                    spellcheck="false"
                    maxlength=MAX_EDIT_UNITS.to_string()
                    node_ref=area
                    prop:value=text
                    on:input=move |ev| state.input(event_target_value(&ev))
                ></textarea>
            }
            .into_any();
        }
        let content = move || {
            if answered.get() {
                view! {
                    <For
                        each=move || drawn.get().unwrap_or_default()
                        key=|d| d.key
                        children=run
                    />
                }
                .into_any()
            } else {
                view! {
                    <span class="plain" data-start="0">
                        {move || state.text.get().to_string()}
                    </span>
                }
                .into_any()
            }
        };
        view! {
            <div class="doc-wrap">
                <div class="doc">{content}</div>
                <span class="scan" data-anim="" aria-hidden="true"></span>
            </div>
        }
        .into_any()
    };

    view! {
        <div class="doc-pane">
            <div class="pane-head">
                <span>{file}</span>
                <span class="doc-tools">
                    <span>{move || format!("{} bytes", thousands(state.text.get().len() as u64))}</span>
                    <button
                        type="button"
                        class="link"
                        aria-pressed=move || state.editing.get().to_string()
                        on:click=move |_| state.toggle_editing()
                    >
                        {move || if state.editing.get() { "[ done ]" } else { "[ edit ]" }}
                    </button>
                    <button
                        type="button"
                        class="link"
                        hidden=move || state.section.get().is_none()
                        on:click=move |_| state.whole_document()
                    >
                        "[ whole document ]"
                    </button>
                    <button
                        type="button"
                        class="link"
                        hidden=move || !state.edited()
                        on:click=move |_| state.reset()
                    >
                        "[ reset ]"
                    </button>
                </span>
            </div>
            <div
                class="pane-scroll"
                class:editing=move || state.editing.get()
                node_ref=state.document
                on:mousedown=move |ev| state.press(&ev)
            >
                {body}
            </div>
        </div>
    }
}

/// Confines the scan line to the lines of the analyzed section, the runs marked `data-in`, or
/// lets it cross the whole document when there are none.
pub(crate) fn fit_scan_line(pane: &HtmlDivElement) {
    let Some(wrap) = pane
        .query_selector(".doc-wrap")
        .ok()
        .flatten()
        .and_then(|w| w.dyn_into::<HtmlElement>().ok())
    else {
        return;
    };
    let style = wrap.style();
    let inside = pane.query_selector_all("[data-in]").ok();
    let at = |i: Option<u32>| -> Option<Element> {
        inside.as_ref()?.item(i?)?.dyn_into::<Element>().ok()
    };
    let count = inside.as_ref().map_or(0, |l| l.length());
    let (Some(first), Some(last)) = (at(Some(0)), at(count.checked_sub(1))) else {
        let _ = style.remove_property("--scan-from");
        let _ = style.remove_property("--scan-to");
        return;
    };
    let origin = wrap.get_bounding_client_rect().top();
    let from = first.get_bounding_client_rect().top() - origin;
    let to = last.get_bounding_client_rect().bottom() - origin;
    let _ = style.set_property("--scan-from", &format!("{from:.1}px"));
    let _ = style.set_property("--scan-to", &format!("{to:.1}px"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::FoundKind;

    fn found(start: usize, end: usize) -> Found {
        Found {
            kind: FoundKind::Email,
            start,
            end,
            confidence: 0.99,
            review_recommended: false,
            source: "rules".to_string(),
            normalized: None,
            region: None,
            components: Vec::new(),
        }
    }

    #[test]
    fn plain_runs_are_cut_at_the_section() {
        let text = "call +44 20 7946 0321 or a@b.example";
        let segs = segments(text, &[found(5, 21)], (2, 24));
        let runs: Vec<(usize, &str)> = segs.iter().map(|s| (s.start, s.text.as_str())).collect();
        assert_eq!(
            runs,
            [
                (0, "ca"),
                (2, "ll "),
                (5, "+44 20 7946 0321"),
                (21, " or"),
                (24, " a@b.example")
            ]
        );
    }

    #[test]
    fn segments_cover_the_text_once_and_skip_overlaps() {
        let text = "call +44 20 7946 0321 or a@b.example";
        let segs = segments(
            text,
            &[found(5, 21), found(9, 12), found(25, 36)],
            (0, text.len()),
        );
        let joined: String = segs.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(joined, text);
        let drawn: Vec<Option<usize>> = segs.iter().map(|s| s.entity).collect();
        assert_eq!(drawn, vec![None, Some(0), None, Some(2)]);
    }
}
