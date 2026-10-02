//! Conservative insertion boundaries derived from native address component spans.

use tessera::AddressLabel;

use crate::data::LabelledExample;

fn unique_start(text: &str, needle: &str) -> Option<usize> {
    if needle.is_empty() {
        return None;
    }
    let mut matches = text.match_indices(needle);
    let (start, _) = matches.next()?;
    matches.next().is_none().then_some(start)
}

/// Locates a distinct native street or PO-box line before mapped city and postcode spans.
pub(crate) fn street_boundary(example: &LabelledExample, filtered: &str) -> Option<usize> {
    if example.country != "US" || example.augmented || !example.text.contains('\n') {
        return None;
    }
    if example.spans.iter().any(|span| {
        span.start >= span.end
            || example
                .text
                .get(span.start as usize..span.end as usize)
                .is_none()
    }) {
        return None;
    }
    let mut candidates = Vec::new();
    let mut offset = 0;
    for source_line in example.text.split_inclusive('\n') {
        let line = source_line.trim();
        let start = offset + source_line.len() - source_line.trim_start().len();
        let end = start + line.len();
        offset += source_line.len();
        let spans: Vec<_> = example
            .spans
            .iter()
            .filter(|span| (span.start as usize) < end && start < span.end as usize)
            .collect();
        if !spans
            .iter()
            .any(|span| matches!(span.label, AddressLabel::Road | AddressLabel::PoBox))
        {
            continue;
        }
        if spans.iter().any(|span| {
            (span.start as usize) < start
                || span.end as usize > end
                || !matches!(
                    span.label,
                    AddressLabel::HouseNumber
                        | AddressLabel::Road
                        | AddressLabel::PoBox
                        | AddressLabel::Unit
                        | AddressLabel::Level
                )
        }) || line.char_indices().any(|(at, ch)| {
            ch.is_alphanumeric()
                && !spans.iter().any(|span| {
                    span.start as usize <= start + at
                        && start + at + ch.len_utf8() <= span.end as usize
                })
        }) {
            return None;
        }
        candidates.push((line, end));
    }
    let [(line, source_end)] = candidates.as_slice() else {
        return None;
    };
    let mapped_start = unique_start(filtered, line)?;
    let boundary = mapped_start + line.len();
    if !filtered[..mapped_start].ends_with('\n') && mapped_start != 0 {
        return None;
    }
    if !filtered[boundary..].starts_with(['\n', '\r']) {
        return None;
    }
    for label in [AddressLabel::City, AddressLabel::Postcode] {
        let mut spans = example.spans.iter().filter(|span| span.label == label);
        let span = spans.next()?;
        if spans.next().is_some() || (span.start as usize) <= *source_end {
            return None;
        }
        let component = example.text.get(span.start as usize..span.end as usize)?;
        if unique_start(filtered, component)? <= boundary {
            return None;
        }
    }
    Some(boundary)
}

/// Inserts generated delivery text while retaining every byte of the rendered base address.
pub(super) fn insert(address: &str, boundary: usize, detail: &str) -> Option<String> {
    let before = address.get(..boundary)?;
    let after = address.get(boundary..)?;
    let separator = if after.starts_with("\r\n") {
        "\r\n"
    } else if after.starts_with('\n') {
        "\n"
    } else if after.starts_with(", ") {
        ", "
    } else {
        return None;
    };
    Some(format!("{before}{separator}{detail}{after}"))
}
