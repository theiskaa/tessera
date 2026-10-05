//! Conservative separation of two complete postal values at a literal line field heading.

use super::{DETECTOR_LABELS, bio::DetectedSpan};
use crate::{Kind, chunk::MAX_ENTITY_TOKENS};

pub(super) fn separate(
    text: &str,
    tokens: &[(usize, usize)],
    probs: &[f32],
    labels: &[usize],
    masked: &[bool],
    breaks: &[bool],
    spans: Vec<DetectedSpan>,
) -> Vec<DetectedSpan> {
    if !valid_inputs(text, tokens, probs, labels, masked, breaks, &spans) {
        return spans;
    }
    let mut output = Vec::with_capacity(spans.len());
    for (index, span) in spans.iter().copied().enumerate() {
        let mut replacement = None;
        if eligible(span, labels, masked, breaks) {
            let heads: Vec<_> = (span.first + 1..span.last)
                .filter_map(|at| heading(text, tokens, at).map(|after| (at, after)))
                .collect();
            if let [(head, after)] = heads.as_slice() {
                let left = piece(span, span.first, head - 1, probs, labels);
                if complete(text, tokens, left) {
                    if *after <= span.last {
                        let right = piece(span, *after, span.last, probs, labels);
                        if separator(text, tokens[*head].0, tokens[*after].0)
                            && complete(text, tokens, right)
                        {
                            replacement = Some(vec![left, right]);
                        }
                    } else if let Some(next) = spans.get(index + 1).copied()
                        && next.first == *after
                        && eligible(next, labels, masked, breaks)
                        && !breaks[next.first]
                        && separator(text, tokens[*head].0, tokens[next.first].0)
                        && complete(text, tokens, next)
                    {
                        replacement = Some(vec![left]);
                    }
                }
            }
        }
        if let Some(pieces) = replacement {
            output.extend(pieces);
        } else {
            output.push(span);
        }
    }
    output
}

fn valid_inputs(
    text: &str,
    tokens: &[(usize, usize)],
    probs: &[f32],
    labels: &[usize],
    masked: &[bool],
    breaks: &[bool],
    spans: &[DetectedSpan],
) -> bool {
    let n = tokens.len();
    probs.len() == n.saturating_mul(DETECTOR_LABELS)
        && probs
            .iter()
            .all(|p| p.is_finite() && (0.0..=1.0).contains(p))
        && labels.len() == n
        && masked.len() == n
        && breaks.len() == n
        && tokens.iter().all(|&(a, b)| {
            a < b && b <= text.len() && text.is_char_boundary(a) && text.is_char_boundary(b)
        })
        && tokens.windows(2).all(|w| w[0].1 <= w[1].0)
        && spans.iter().all(|s| s.first <= s.last && s.last < n)
        && spans.windows(2).all(|w| w[0].last < w[1].first)
}

fn eligible(span: DetectedSpan, labels: &[usize], masked: &[bool], breaks: &[bool]) -> bool {
    span.kind == Kind::Address
        && span.last + 1 - span.first <= MAX_ENTITY_TOKENS
        && labels[span.first..=span.last]
            .iter()
            .all(|&l| l == 5 || l == 6)
        && masked[span.first..=span.last].iter().all(|&m| !m)
        && breaks[span.first + 1..=span.last].iter().all(|&b| !b)
}

fn piece(
    original: DetectedSpan,
    first: usize,
    last: usize,
    probs: &[f32],
    labels: &[usize],
) -> DetectedSpan {
    let mut sum = 0.0;
    for index in first..=last {
        let p = probs[index * DETECTOR_LABELS + labels[index]];
        sum += if p.is_nan() { 0.0 } else { p };
    }
    DetectedSpan {
        first,
        last,
        confidence: sum / (last + 1 - first) as f32,
        ..original
    }
}

fn heading(text: &str, tokens: &[(usize, usize)], at: usize) -> Option<usize> {
    let &(a, b) = tokens.get(at)?;
    let &(c, d) = tokens.get(at + 1)?;
    let first = text.get(a..b)?;
    if !(first.eq_ignore_ascii_case("Physical") || first.eq_ignore_ascii_case("Mailing"))
        || !text.get(c..d)?.eq_ignore_ascii_case("Address")
        || !horizontal(text.get(b..c)?)
    {
        return None;
    }
    let before = text.get(..a)?;
    if !before.contains('\n')
        || !before
            .rsplit('\n')
            .next()?
            .chars()
            .all(|c| matches!(c, ' ' | '\t'))
    {
        return None;
    }
    let previous = tokens.get(at.checked_sub(1)?)?.1;
    let gap = text.get(previous..a)?.replace("\r\n", "\n");
    if gap.bytes().filter(|&c| c == b'\n').count() != 1
        || !gap.chars().all(|c| matches!(c, ' ' | '\t' | '\n'))
    {
        return None;
    }
    let mut next = at + 2;
    if let Some(&(a, b)) = tokens.get(next)
        && text.get(a..b) == Some(":")
    {
        next += 1;
    }
    Some(next)
}

fn horizontal(text: &str) -> bool {
    !text.is_empty() && text.chars().all(|c| matches!(c, ' ' | '\t'))
}

fn separator(text: &str, start: usize, end: usize) -> bool {
    let Some(field) = text.get(start..end) else {
        return false;
    };
    let normalized = field.replace("\r\n", "\n");
    if normalized
        .chars()
        .any(|c| matches!(c, '\r' | '\u{85}' | '\u{2028}' | '\u{2029}'))
        || normalized.bytes().filter(|&c| c == b'\n').count() > 1
    {
        return false;
    }
    let words: Vec<_> = normalized.split_whitespace().collect();
    matches!(words.as_slice(), [first, second]
        if (first.eq_ignore_ascii_case("Physical") || first.eq_ignore_ascii_case("Mailing"))
            && (second.eq_ignore_ascii_case("Address") || second.eq_ignore_ascii_case("Address:")))
        || matches!(words.as_slice(), [first, second, colon]
            if (first.eq_ignore_ascii_case("Physical") || first.eq_ignore_ascii_case("Mailing"))
                && second.eq_ignore_ascii_case("Address") && *colon == ":")
}

fn complete(text: &str, tokens: &[(usize, usize)], span: DetectedSpan) -> bool {
    let Some(raw) = text.get(tokens[span.first].0..tokens[span.last].1) else {
        return false;
    };
    postal_block(raw)
}

fn postal_block(raw: &str) -> bool {
    if raw
        .chars()
        .any(|c| matches!(c, '\u{85}' | '\u{2028}' | '\u{2029}' | ';'))
    {
        return false;
    }
    let normalized = raw.replace("\r\n", "\n");
    if normalized.contains('\r') || normalized.split('\n').any(|line| line.trim().is_empty()) {
        return false;
    }
    let fields: Vec<_> = normalized.split([',', '\n']).map(str::trim).collect();
    if fields.len() < 2 || fields.iter().any(|f| f.is_empty()) {
        return false;
    }
    let last = fields.len() - 1;
    let tail: Vec<_> = fields[last].split_whitespace().collect();
    if tail.len() < 2 || !state(tail[tail.len() - 2]) || !zip(tail[tail.len() - 1]) {
        return false;
    }
    let street_end = if tail.len() > 2 {
        if !city(&tail[..tail.len() - 2]) {
            return false;
        }
        last
    } else {
        if last < 2 || !city(&fields[last - 1].split_whitespace().collect::<Vec<_>>()) {
            return false;
        }
        last - 1
    };
    street_end > 0 && physical(fields[0]) && fields[1..street_end].iter().all(|field| unit(field))
}

fn state(word: &str) -> bool {
    const STATES: &[&str] = &[
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN",
        "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH",
        "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT",
        "VT", "VA", "WA", "WV", "WI", "WY",
    ];
    STATES.iter().any(|state| word.eq_ignore_ascii_case(state))
}

fn zip(word: &str) -> bool {
    let b = word.as_bytes();
    (b.len() == 5 && b.iter().all(u8::is_ascii_digit))
        || (b.len() == 10
            && b[5] == b'-'
            && b[..5].iter().all(u8::is_ascii_digit)
            && b[6..].iter().all(u8::is_ascii_digit))
}

fn city(words: &[&str]) -> bool {
    !words.is_empty()
        && words.len() <= 5
        && words.iter().map(|w| w.len()).sum::<usize>() <= 64
        && words.iter().all(|w| {
            w.as_bytes().first().is_some_and(u8::is_ascii_uppercase)
                && w.bytes()
                    .all(|c| c.is_ascii_alphabetic() || matches!(c, b'\'' | b'-' | b'.'))
        })
}

fn lower(word: &str) -> String {
    word.trim_end_matches('.').to_ascii_lowercase()
}

fn unit(field: &str) -> bool {
    let words: Vec<_> = field.split_whitespace().collect();
    match words.as_slice() {
        [kind, id]
            if matches!(
                lower(kind).as_str(),
                "room" | "rm" | "suite" | "ste" | "floor" | "building" | "bldg"
            ) =>
        {
            !id.is_empty()
                && id.len() <= 20
                && id.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-')
        }
        _ => false,
    }
}

fn physical(field: &str) -> bool {
    let w: Vec<_> = field.split_whitespace().collect();
    if let [po, bx, number] = w.as_slice()
        && po.replace('.', "").eq_ignore_ascii_case("PO")
        && bx.eq_ignore_ascii_case("Box")
    {
        return !number.is_empty() && number.bytes().all(|c| c.is_ascii_digit());
    }
    if let [post, office, bx, number] = w.as_slice()
        && post.eq_ignore_ascii_case("Post")
        && office.eq_ignore_ascii_case("Office")
        && bx.eq_ignore_ascii_case("Box")
    {
        return !number.is_empty() && number.bytes().all(|c| c.is_ascii_digit());
    }
    if w.len() < 3 || !house(w[0]) {
        return false;
    }
    let roads: Vec<_> = w
        .iter()
        .enumerate()
        .filter(|(_, word)| road(word))
        .map(|(i, _)| i)
        .collect();
    let [at] = roads.as_slice() else {
        return false;
    };
    if *at < 2
        || !w[1..*at].iter().all(|word| {
            word.bytes()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, b'\'' | b'-' | b'.'))
                && !matches!(lower(word).as_str(), "and" | "or")
        })
    {
        return false;
    }
    let mut tail = &w[*at + 1..];
    if tail.first().is_some_and(|word| {
        matches!(
            word.trim_end_matches('.').to_ascii_uppercase().as_str(),
            "N" | "S" | "E" | "W" | "NE" | "NW" | "SE" | "SW"
        )
    }) {
        tail = &tail[1..];
    }
    tail.is_empty() || unit(&tail.join(" "))
}

fn house(word: &str) -> bool {
    let b = word.as_bytes();
    let n = b.iter().take_while(|c| c.is_ascii_digit()).count();
    n > 0 && (n == b.len() || (n + 1 == b.len() && b[n].is_ascii_alphabetic()))
}

fn road(word: &str) -> bool {
    matches!(
        lower(word).as_str(),
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

#[cfg(test)]
#[path = "address_field_boundaries_tests.rs"]
mod tests;
