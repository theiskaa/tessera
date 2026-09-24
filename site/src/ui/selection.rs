//! The reader's selection in the document, as a byte range of the document: the section the
//! demo analyzes alone. The document's runs are elements carrying their first byte in
//! `data-start`; the DOM counts offsets in UTF-16 code units, the library in UTF-8 bytes.

use leptos::prelude::{document, window};
use wasm_bindgen::JsCast;
use web_sys::{Element, Node};

/// The bytes of `text` selected in `doc`, the element holding the document's runs, and clears
/// the selection it reads. A boundary before `doc` counts as its start and one after it as its
/// end, so a drag released below the last line still selects to the end. `None` without a
/// selection, or when it lies wholly outside `doc`.
pub(crate) fn take(doc: &Element, text: &str) -> Option<(usize, usize)> {
    let selection = window().get_selection().ok()??;
    if selection.is_collapsed() || selection.range_count() == 0 {
        return None;
    }
    let range = selection.get_range_at(0).ok()?;
    let within = document().create_range().ok()?;
    within.select_node_contents(doc).ok()?;
    let byte = |node: Node, offset: u32| -> Option<usize> {
        match within.compare_point(&node, offset).ok()? {
            before if before < 0 => Some(0),
            0 => byte_at(text, &node, offset),
            _ => Some(text.len()),
        }
    };
    let start = byte(range.start_container().ok()?, range.start_offset().ok()?)?;
    let end = byte(range.end_container().ok()?, range.end_offset().ok()?)?;
    if start == end {
        return None;
    }
    let _ = selection.remove_all_ranges();
    Some((start.min(end), start.max(end)))
}

/// The section a selection of `start..end` stands for: trimmed of whitespace at either end,
/// `Some(None)` when that covers all of the document's text, `None` when nothing is left.
pub(crate) fn section(text: &str, start: usize, end: usize) -> Option<Option<(usize, usize)>> {
    let slice = text.get(start..end)?;
    let start = start + (slice.len() - slice.trim_start().len());
    let end = end - (slice.len() - slice.trim_end().len());
    if start >= end {
        return None;
    }
    let lead = text.len() - text.trim_start().len();
    let whole = (start, end) == (lead, text.trim_end().len());
    Some((!whole).then_some((start, end)))
}

/// The byte of `text` at a selection boundary inside the document: `node` and `offset` as a
/// `Range` reports them.
fn byte_at(text: &str, node: &Node, offset: u32) -> Option<usize> {
    if node.node_type() == Node::TEXT_NODE {
        let start = run_start(&node.parent_element()?)?;
        return Some(start + units_to_bytes(text.get(start..)?, offset as usize));
    }
    let element = node.dyn_ref::<Element>()?;
    if let Some(start) = run_start(element) {
        // A run holds one text node: the boundary is before it or after it.
        let len = if offset == 0 {
            0
        } else {
            element.text_content()?.len()
        };
        return Some(start + len);
    }
    // A container: `offset` counts its children, and the boundary is where the next run starts,
    // or where the one before it ends. Framework comment nodes sit between runs.
    let children = element.child_nodes();
    let run = |i: u32| {
        let child = children.item(i)?.dyn_into::<Element>().ok()?;
        Some((run_start(&child)?, child))
    };
    (offset..children.length())
        .find_map(|i| run(i).map(|(start, _)| start))
        .or_else(|| {
            (0..offset.min(children.length())).rev().find_map(|i| {
                run(i).map(|(start, child)| start + child.text_content().map_or(0, |t| t.len()))
            })
        })
}

fn run_start(run: &Element) -> Option<usize> {
    run.get_attribute("data-start")?.parse().ok()
}

/// The bytes of `s` its first `units` UTF-16 code units cover. A count that ends inside a
/// character, between the halves of a surrogate pair, takes the whole character.
fn units_to_bytes(s: &str, units: usize) -> usize {
    let mut seen = 0;
    for (i, c) in s.char_indices() {
        if seen >= units {
            return i;
        }
        seen += c.len_utf16();
    }
    s.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn units_count_as_bytes_across_scripts() {
        let s = "aé日😀b";
        assert_eq!(units_to_bytes(s, 0), 0);
        assert_eq!(units_to_bytes(s, 2), 3);
        assert_eq!(units_to_bytes(s, 3), 6);
        assert_eq!(units_to_bytes(s, 5), 10);
        assert_eq!(
            units_to_bytes(s, 4),
            10,
            "half a surrogate pair takes the character"
        );
        assert_eq!(units_to_bytes(s, 99), s.len());
    }

    #[test]
    fn a_section_is_trimmed_and_the_whole_text_is_no_section() {
        let text = "  Hi Ana,\n\nCall 020 7946 0321.\n";
        assert_eq!(section(text, 0, 9), Some(Some((2, 9))));
        assert_eq!(section(text, 8, 11), Some(Some((8, 9))));
        assert_eq!(section(text, 9, 11), None);
        assert_eq!(section(text, 0, text.len()), Some(None));
        assert_eq!(section(text, 2, text.trim_end().len()), Some(None));
        assert_eq!(section(text, 0, 99), None);
    }
}
