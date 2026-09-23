//! The document pane: the text with what the library found lit up, or a text box while the
//! reader edits it.

use leptos::html::Textarea;
use leptos::prelude::*;

use super::demo::{DemoState, MAX_TEXT};
use super::format::thousands;
use super::json::kind_name;
use crate::protocol::{Found, FoundKind};
use crate::samples::SAMPLES;

/// A run of the document, plain or covering entity `entity`.
struct Segment {
    text: String,
    start: usize,
    entity: Option<usize>,
}

fn segments(text: &str, found: &[Found]) -> Vec<Segment> {
    let mut out = Vec::new();
    let mut cursor = 0;
    for (i, e) in found.iter().enumerate() {
        // An entity overlapping one already drawn stays in the list but is drawn once.
        if e.start < cursor {
            continue;
        }
        let (Some(before), Some(span)) = (text.get(cursor..e.start), text.get(e.start..e.end))
        else {
            continue;
        };
        if !before.is_empty() {
            out.push(Segment {
                text: before.to_string(),
                start: cursor,
                entity: None,
            });
        }
        out.push(Segment {
            text: span.to_string(),
            start: e.start,
            entity: Some(i),
        });
        cursor = e.end;
    }
    if let Some(rest) = text.get(cursor..).filter(|r| !r.is_empty()) {
        out.push(Segment {
            text: rest.to_string(),
            start: cursor,
            entity: None,
        });
    }
    out
}

/// Where a run starts, as a share of the document: its highlight waits that share of the scan.
fn place(start: usize, len: usize) -> String {
    format!("--f: {:.5}", start as f64 / len.max(1) as f64)
}

/// Longest text the edit box takes, in UTF-16 units: well past the analysis limit, so a long
/// paste is shown as too long rather than cut, but bounded.
const MAX_EDIT_UNITS: usize = 4 * MAX_TEXT;

/// One drawn run of the document, keyed so a new mark redraws only the text it splits.
#[derive(Debug, Clone, PartialEq)]
struct Drawn {
    key: (usize, usize, &'static str),
    text: String,
    kind: Option<FoundKind>,
}

/// The document pane.
#[component]
pub(crate) fn DocumentPane(state: DemoState) -> impl IntoView {
    let area: NodeRef<Textarea> = NodeRef::new();
    let file = move || SAMPLES.get(state.sample.get()).map_or("", |s| s.file);
    // The answer is drawn only while it is for the text on screen.
    let drawn = Memo::new(move |_| match state.answer.get() {
        Some(Ok(shown)) if *shown.text == *state.text.get() => Some(
            segments(&shown.text, &shown.found)
                .into_iter()
                .map(|seg| {
                    let kind = seg.entity.and_then(|i| shown.found.get(i)).map(|e| e.kind);
                    let end = seg.start + seg.text.len();
                    Drawn {
                        key: (seg.start, end, kind.map_or("plain", kind_name)),
                        text: seg.text,
                        kind,
                    }
                })
                .collect::<Vec<_>>(),
        ),
        _ => None,
    });
    let answered = Memo::new(move |_| drawn.with(Option::is_some));
    let len = move || state.text.with(|t| t.len());

    let run = move |d: Drawn| {
        let start = d.key.0;
        let style = move || place(start, len());
        match d.kind {
            None => {
                view! { <span class="plain" data-anim="" style=style>{d.text}</span> }.into_any()
            }
            Some(kind) => {
                // Drawn outside a scan, a mark is new: it fades in instead of being scanned,
                // until the next scan takes it over.
                let born = (!state.scanning.get_untracked()).then(|| state.run.get_untracked());
                let fresh = move || born.is_some_and(|run| run == state.run.get());
                view! {
                    <span
                        class=format!("hit k-{}", kind_name(kind))
                        class:fresh=fresh
                        class:selected=move || state.selected.get() == Some(start)
                        data-entity=start.to_string()
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
                view! { <span class="plain">{move || state.text.get().to_string()}</span> }
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
            >
                {body}
            </div>
        </div>
    }
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
    fn segments_cover_the_text_once_and_skip_overlaps() {
        let text = "call +44 20 7946 0321 or a@b.example";
        let segs = segments(text, &[found(5, 21), found(9, 12), found(25, 36)]);
        let joined: String = segs.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(joined, text);
        let drawn: Vec<Option<usize>> = segs.iter().map(|s| s.entity).collect();
        assert_eq!(drawn, vec![None, Some(0), None, Some(2)]);
    }
}
