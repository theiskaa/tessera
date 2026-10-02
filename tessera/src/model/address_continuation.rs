//! Conservative continuation of adjacent decoded ADDRESS fragments in one postal block.

use super::{DETECTOR_LABELS, bio::DetectedSpan};
use crate::{Kind, chunk::MAX_ENTITY_TOKENS};

pub(super) fn join(
    text: &str,
    tokens: &[(usize, usize)],
    probs: &[f32],
    labels: &[usize],
    masked: &[bool],
    breaks: &[bool],
    spans: Vec<DetectedSpan>,
) -> Vec<DetectedSpan> {
    let mut output = Vec::with_capacity(spans.len());
    let mut i = 0;
    while let Some(a) = spans.get(i).copied() {
        if let Some(b) = spans.get(i + 1).copied()
            && eligible(a, b, labels, masked, breaks)
            && let (Some((start, left_end)), Some((right_start, end))) = (
                tokens
                    .get(a.first)
                    .zip(tokens.get(a.last))
                    .map(|(f, l)| (f.0, l.1)),
                tokens
                    .get(b.first)
                    .zip(tokens.get(b.last))
                    .map(|(f, l)| (f.0, l.1)),
            )
            && accepts(text, start, left_end, right_start, end)
        {
            let mut sum = chosen(probs, labels, a.first);
            for index in a.first + 1..=b.last {
                sum += chosen(probs, labels, index);
            }
            output.push(DetectedSpan {
                kind: Kind::Address,
                first: a.first,
                last: b.last,
                confidence: sum / (b.last + 1 - a.first) as f32,
            });
            i += 2;
            continue;
        }
        output.push(a);
        i += 1;
    }
    output
}

fn chosen(probs: &[f32], labels: &[usize], index: usize) -> f32 {
    labels
        .get(index)
        .and_then(|&label| probs.get(index * DETECTOR_LABELS + label))
        .copied()
        .filter(|p| !p.is_nan())
        .unwrap_or(0.0)
}

fn eligible(
    a: DetectedSpan,
    b: DetectedSpan,
    labels: &[usize],
    masked: &[bool],
    breaks: &[bool],
) -> bool {
    a.kind == Kind::Address
        && b.kind == Kind::Address
        && a.last.checked_add(1) == Some(b.first)
        && b.last >= a.first
        && b.last + 1 - a.first <= MAX_ENTITY_TOKENS
        && labels
            .get(a.first..=b.last)
            .is_some_and(|s| s.iter().all(|&l| l == 5 || l == 6))
        && masked
            .get(a.first..=b.last)
            .is_some_and(|s| s.iter().all(|&m| !m))
        && breaks
            .get(a.first + 1..=b.last)
            .is_some_and(|s| s.iter().all(|&b| !b))
}

fn lines(text: &str) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

fn accepts(text: &str, start: usize, left_end: usize, right_start: usize, raw_end: usize) -> bool {
    let end = crate::detect::normalized_us_address_end(text, right_start, raw_end);
    let (Some(left), Some(gap), Some(right), Some(before), Some(block)) = (
        text.get(start..left_end),
        text.get(left_end..right_start),
        text.get(right_start..end),
        text.get(..start),
        text.get(start..end),
    ) else {
        return false;
    };
    if block
        .chars()
        .any(|c| matches!(c, '\u{85}' | '\u{2028}' | '\u{2029}'))
    {
        return false;
    }
    let (left, gap, right, before, block) = (
        lines(left),
        lines(gap),
        lines(right),
        lines(before),
        lines(block),
    );
    if gap.is_empty()
        || !gap.chars().all(|c| matches!(c, ' ' | '\t' | '\n'))
        || gap.bytes().filter(|&c| c == b'\n').count() > 1
        || block.split('\n').any(|line| line.trim().is_empty())
    {
        return false;
    }
    let mut previous = before.rsplit('\n');
    if previous.next().is_none_or(|tail| !tail.trim().is_empty()) {
        return false;
    }
    let heading = previous.next().unwrap_or("");
    let postal_heading = heading.split_ascii_whitespace().any(|word| {
        matches!(
            lower_word(word).as_str(),
            "delivery" | "mailing" | "correspondence" | "location" | "address"
        )
    });
    let Some(has_unit) = postal_tail(&right) else {
        return false;
    };
    building_prefix(&left)
        || unit_field(&left)
        || (room_code(&left) && postal_heading)
        || (bare_name(&left) && has_unit && !gap.contains('\n') && postal_heading)
}

fn lower_word(word: &str) -> String {
    word.trim_matches(|c: char| !c.is_ascii_alphanumeric())
        .to_ascii_lowercase()
}

fn postal_tail(text: &str) -> Option<bool> {
    let fields: Vec<_> = text.split([',', '\n']).map(str::trim).collect();
    if fields.len() < 3 || fields.iter().any(|f| f.is_empty()) || text.contains(';') {
        return None;
    }
    let last = fields.len() - 1;
    if !state_zip(fields[last]) || !city(fields[last - 1]) {
        return None;
    }
    if fields[..last].iter().any(|f| {
        f.split_ascii_whitespace()
            .any(|w| matches!(lower_word(w).as_str(), "and" | "or"))
    }) {
        return None;
    }
    let has_unit = unit_field(fields[0]);
    let street = usize::from(has_unit);
    if street >= last - 1 || !physical_field(fields[street]) {
        return None;
    }
    if !fields[street + 1..last - 1]
        .iter()
        .all(|field| unit_field(field))
    {
        return None;
    }
    Some(has_unit)
}

fn state_zip(field: &str) -> bool {
    let words: Vec<_> = field.split_ascii_whitespace().collect();
    let [state, zip] = words.as_slice() else {
        return false;
    };
    const STATES: &[&str] = &[
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN",
        "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
        "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT",
        "VT", "VA", "WA", "WV", "WI", "WY",
    ];
    if !STATES.iter().any(|s| state.eq_ignore_ascii_case(s)) {
        return false;
    }
    let bytes = zip.as_bytes();
    (bytes.len() == 5 && bytes.iter().all(u8::is_ascii_digit))
        || (bytes.len() == 10
            && bytes[5] == b'-'
            && bytes[..5].iter().all(u8::is_ascii_digit)
            && bytes[6..].iter().all(u8::is_ascii_digit))
}

fn city(field: &str) -> bool {
    let words: Vec<_> = field.split_ascii_whitespace().collect();
    !words.is_empty()
        && words.len() <= 5
        && field.len() <= 64
        && words.iter().all(|word| {
            word.as_bytes().first().is_some_and(u8::is_ascii_uppercase)
                && word
                    .bytes()
                    .all(|c| c.is_ascii_alphabetic() || matches!(c, b'\'' | b'-' | b'.'))
        })
}

fn unit_field(field: &str) -> bool {
    let words: Vec<_> = field.split_ascii_whitespace().collect();
    let identifier = match words.as_slice() {
        [kind, id]
            if matches!(
                lower_word(kind).as_str(),
                "room" | "rm" | "suite" | "ste" | "floor" | "building" | "bldg" | "stop"
            ) =>
        {
            Some(*id)
        }
        [mail, kind, id]
            if mail.eq_ignore_ascii_case("mail")
                && (kind.eq_ignore_ascii_case("stop") || kind.eq_ignore_ascii_case("code")) =>
        {
            Some(*id)
        }
        _ => None,
    };
    identifier.is_some_and(|id| {
        !id.is_empty()
            && id.len() <= 20
            && id.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-')
    })
}

fn road(word: &str) -> bool {
    matches!(
        lower_word(word).as_str(),
        "street"
            | "st"
            | "avenue"
            | "ave"
            | "road"
            | "rd"
            | "way"
            | "drive"
            | "dr"
            | "boulevard"
            | "blvd"
            | "lane"
            | "ln"
            | "parkway"
            | "pkwy"
            | "court"
            | "ct"
            | "place"
            | "pl"
            | "highway"
            | "hwy"
            | "circle"
            | "cir"
    )
}

fn house_number(word: &str) -> bool {
    let bytes = word.as_bytes();
    let digits = bytes.iter().take_while(|c| c.is_ascii_digit()).count();
    digits > 0
        && (digits == bytes.len()
            || (digits + 1 == bytes.len() && bytes[digits].is_ascii_alphabetic()))
}

fn physical_field(field: &str) -> bool {
    let words: Vec<_> = field.split_ascii_whitespace().collect();
    if let [po, bx, number] = words.as_slice()
        && po.replace('.', "").eq_ignore_ascii_case("po")
        && bx.eq_ignore_ascii_case("box")
    {
        return number.bytes().all(|c| c.is_ascii_digit()) && !number.is_empty();
    }
    if let [post, office, bx, number] = words.as_slice()
        && post.eq_ignore_ascii_case("post")
        && office.eq_ignore_ascii_case("office")
        && bx.eq_ignore_ascii_case("box")
    {
        return number.bytes().all(|c| c.is_ascii_digit()) && !number.is_empty();
    }
    if words.len() < 3 || !house_number(words[0]) {
        return false;
    }
    let roads: Vec<_> = words
        .iter()
        .enumerate()
        .filter(|(_, word)| road(word))
        .map(|(i, _)| i)
        .collect();
    let [at] = roads.as_slice() else {
        return false;
    };
    if *at < 2
        || !words[1..*at].iter().all(|word| {
            word.bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'-' | b'\'' | b'.'))
        })
    {
        return false;
    }
    let mut tail = &words[*at + 1..];
    if tail.first().is_some_and(|w| {
        matches!(
            w.to_ascii_uppercase().as_str(),
            "N" | "S" | "E" | "W" | "NE" | "NW" | "SE" | "SW"
        )
    }) {
        tail = &tail[1..];
    }
    tail.is_empty() || unit_field(&tail.join(" "))
}

fn building_prefix(field: &str) -> bool {
    let words: Vec<_> = field.split_ascii_whitespace().collect();
    words.len() >= 2
        && field.len() <= 96
        && words
            .first()
            .is_some_and(|w| w.as_bytes().first().is_some_and(u8::is_ascii_alphabetic))
        && words
            .last()
            .is_some_and(|w| matches!(lower_word(w).as_str(), "building" | "bldg"))
        && words
            .iter()
            .all(|w| !road(w) && !matches!(lower_word(w).as_str(), "box" | "and" | "or"))
        && field.bytes().all(|c| {
            c.is_ascii_alphanumeric()
                || matches!(c, b' ' | b'\t' | b'.' | b'&' | b'\'' | b'-' | b'/')
        })
}

fn room_code(field: &str) -> bool {
    let bytes = field.as_bytes();
    let letters = bytes.iter().take_while(|c| c.is_ascii_uppercase()).count();
    if !(1..=2).contains(&letters) {
        return false;
    }
    let digits = bytes[letters..]
        .iter()
        .take_while(|c| c.is_ascii_digit())
        .count();
    (2..=5).contains(&digits)
        && (letters + digits == bytes.len()
            || (letters + digits + 1 == bytes.len()
                && bytes[letters + digits].is_ascii_uppercase()))
}

fn bare_name(field: &str) -> bool {
    let bytes = field.as_bytes();
    bytes.len() >= 2
        && bytes[0].is_ascii_uppercase()
        && bytes[1].is_ascii_lowercase()
        && bytes
            .iter()
            .all(|c| c.is_ascii_alphabetic() || matches!(c, b'\'' | b'-'))
}

#[cfg(test)]
#[path = "address_continuation_tests.rs"]
mod tests;
