//! Windowing for long documents and merging of predictions from overlapping windows.
//!
//! Windows are ranges over the retained (non-whitespace) token sequence. Features are
//! computed over the whole document once, so a position's byte offsets are already document
//! offsets and nothing is remapped. Consecutive windows overlap by `OVERLAP_TOKENS`, and a
//! window trusts only predictions at least `CONTEXT_MARGIN_TOKENS` from an inner edge, so every
//! entity of at most `MAX_ENTITY_TOKENS` positions is trusted in at least one window.
//!
//! Worked example: 2,500 retained positions with paragraph breaks before 900, 1,700, and 2,400
//! and line breaks before the other multiples of 40, and no whitespace between any other
//! two positions (as in a run of punctuation). Window 1 targets 1,024; the search over
//! 768..=1,024 finds the paragraph break at 900, so it is `[0, 900)`. The next start is
//! `900 - 384 = 516`, snapped back to the line break at 480. Window 2 targets 1,504 and cuts
//! at the line break 1,480: `[480, 1480)`, next start 1,096 snapped to 1,080. Window 3
//! targets 2,104 and cuts at 2,080: `[1080, 2080)`, next start 1,696 snapped to 1,680.
//! Window 4 reaches the end: `[1680, 2500)`. Consecutive overlaps are 420, 400, and 400.

use std::collections::BTreeMap;

use crate::token::{Token, TokenClass};
use crate::{Entity, Error, Source};

/// Longest entity, in retained positions, that windowing guarantees to see whole.
pub const MAX_ENTITY_TOKENS: usize = 256;
/// Positions at a window's inner edges whose predictions the window does not trust.
pub(crate) const CONTEXT_MARGIN_TOKENS: usize = 64;
/// Most retained positions in one window.
pub(crate) const WINDOW_TOKENS: usize = 1024;
/// Least overlap between consecutive windows: a longest entity plus a margin on each side.
pub(crate) const OVERLAP_TOKENS: usize = MAX_ENTITY_TOKENS + 2 * CONTEXT_MARGIN_TOKENS;
/// A longer run of retained tokens with no whitespace anywhere inside is `InputTooLarge`.
pub(crate) const MAX_UNBROKEN_TOKENS: usize = 4096;
/// How far back from a window's target end a natural boundary is searched for.
const CUT_SEARCH: usize = 256;

/// Byte ranges entities may lie in, for input where only some bytes are content, such as
/// Markdown. Plain text has no mask.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mask {
    ranges: Vec<(usize, usize)>,
}

/// A masked run of at least this many retained positions is cut out of windowing altogether;
/// shorter ones (inline code, a link destination) stay as context inside a window.
pub(crate) const SKIP_SPLIT_TOKENS: usize = 32;

impl Mask {
    /// `scan` minus `skip`: both sorted, `scan` disjoint. The result is sorted, disjoint, and
    /// holds no empty range.
    pub fn new(scan: &[(usize, usize)], skip: &[(usize, usize)]) -> Mask {
        let mut ranges = Vec::with_capacity(scan.len());
        let mut skips = skip.iter().copied().peekable();
        for &(scan_start, scan_end) in scan {
            let mut cursor = scan_start;
            while let Some(&(skip_start, skip_end)) = skips.peek() {
                if skip_end <= cursor {
                    skips.next();
                    continue;
                }
                if skip_start >= scan_end {
                    break;
                }
                if skip_start > cursor {
                    ranges.push((cursor, skip_start));
                }
                cursor = cursor.max(skip_end);
                if skip_end <= scan_end {
                    skips.next();
                } else {
                    break;
                }
            }
            if cursor < scan_end {
                ranges.push((cursor, scan_end));
            }
        }
        Mask { ranges }
    }

    /// Whether one range holds all of `start..end`.
    pub fn contains(&self, start: usize, end: usize) -> bool {
        let i = self.ranges.partition_point(|&(_, e)| e <= start);
        matches!(self.ranges.get(i), Some(&(s, e)) if s <= start && end <= e)
    }

    /// Whether the token lies entirely inside the mask.
    pub fn covers(&self, token: &Token) -> bool {
        self.contains(token.start, token.end)
    }

    /// The ranges, sorted and disjoint.
    pub fn ranges(&self) -> &[(usize, usize)] {
        &self.ranges
    }
}

/// A window over the retained tokens `tok_start..tok_end`, whose text is the byte range
/// `start..end` of the document.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Window {
    pub start: usize,
    pub end: usize,
    pub tok_start: usize,
    pub tok_end: usize,
}

/// What separates a retained position from the previous one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Break {
    None,
    Space,
    Line,
    Paragraph,
}

/// Per retained position, the whitespace between it and the previous retained token.
/// Position 0 counts as a paragraph break.
fn breaks_before(tokens: &[Token], retained: &[usize]) -> Vec<Break> {
    let mut out = Vec::with_capacity(retained.len());
    for (i, &r) in retained.iter().enumerate() {
        if i == 0 {
            out.push(Break::Paragraph);
            continue;
        }
        let between = &tokens[retained[i - 1] + 1..r];
        let newlines = between
            .iter()
            .filter(|t| t.class == TokenClass::Newline)
            .count();
        out.push(match (between.is_empty(), newlines) {
            (true, _) => Break::None,
            (false, 0) => Break::Space,
            (false, 1) => Break::Line,
            _ => Break::Paragraph,
        });
    }
    out
}

/// Per retained position, whether a blank line separates it from the previous one. Position 0
/// counts as one, like the start of any document.
pub fn paragraph_breaks(tokens: &[Token], retained: &[usize]) -> Vec<bool> {
    breaks_before(tokens, retained)
        .into_iter()
        .map(|b| b == Break::Paragraph)
        .collect()
}

/// Covers the retained positions with windows of at most `window` positions, overlapping by
/// at least `overlap`, each ending at the strongest natural boundary within `CUT_SEARCH`
/// positions of its target end. With `masked` (per retained position, whether it is outside
/// the mask), a masked run of at least `SKIP_SPLIT_TOKENS` positions belongs to no window and
/// the positions on either side are windowed separately, and no window ends inside a shorter
/// masked run.
pub(crate) fn windows(
    tokens: &[Token],
    retained: &[usize],
    window: usize,
    overlap: usize,
    masked: Option<&[bool]>,
) -> Result<Vec<Window>, Error> {
    let n = retained.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let breaks = breaks_before(tokens, retained);
    check_unbroken(&breaks)?;
    let mut out = Vec::new();
    for (from, to) in pieces(masked, n) {
        window_piece(
            tokens,
            retained,
            &breaks,
            masked,
            (from, to),
            window,
            overlap,
            &mut out,
        );
    }
    Ok(out)
}

/// The ranges of retained positions left once every masked run of at least
/// `SKIP_SPLIT_TOKENS` positions is removed; the whole sequence without a mask.
fn pieces(masked: Option<&[bool]>, n: usize) -> Vec<(usize, usize)> {
    let Some(masked) = masked else {
        return vec![(0, n)];
    };
    let mut out = Vec::new();
    let (mut from, mut i) = (0, 0);
    while i < n {
        if !masked[i] {
            i += 1;
            continue;
        }
        let run = i;
        while i < n && masked[i] {
            i += 1;
        }
        if i - run >= SKIP_SPLIT_TOKENS {
            if from < run {
                out.push((from, run));
            }
            from = i;
        }
    }
    if from < n {
        out.push((from, n));
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn window_piece(
    tokens: &[Token],
    retained: &[usize],
    breaks: &[Break],
    masked: Option<&[bool]>,
    (from, to): (usize, usize),
    window: usize,
    overlap: usize,
    out: &mut Vec<Window>,
) {
    let mut start = from;
    loop {
        let target = start + window;
        if target >= to {
            out.push(make(tokens, retained, start, to));
            return;
        }
        let cut = outside_masked_run(masked, start, find_cut(breaks, start, target));
        out.push(make(tokens, retained, start, cut));
        let mut next = cut.saturating_sub(overlap).max(start + 1);
        // Snap back to a whitespace boundary so a window never opens mid-word; moving back
        // only lengthens the overlap.
        let floor = next.saturating_sub(CONTEXT_MARGIN_TOKENS).max(start + 1);
        while next > floor && breaks[next] == Break::None {
            next -= 1;
        }
        start = next;
    }
}

/// `cut` moved back to the first position of the masked run it falls inside, or past the run's
/// end when that would leave the window empty.
fn outside_masked_run(masked: Option<&[bool]>, start: usize, cut: usize) -> usize {
    let Some(masked) = masked else {
        return cut;
    };
    let inside = |i: usize| i > 0 && i < masked.len() && masked[i] && masked[i - 1];
    if !inside(cut) {
        return cut;
    }
    let mut back = cut;
    while back > start && inside(back) {
        back -= 1;
    }
    if back > start {
        return back;
    }
    let mut forward = cut;
    while forward < masked.len() && masked[forward] {
        forward += 1;
    }
    forward
}

/// The retained position the window ends before: the last paragraph break in the search
/// range, else the last line break, else the last space, else the target itself.
fn find_cut(breaks: &[Break], start: usize, target: usize) -> usize {
    let floor = target.saturating_sub(CUT_SEARCH).max(start + 1);
    for wanted in [Break::Paragraph, Break::Line, Break::Space] {
        if let Some(i) = (floor..=target).rev().find(|&i| breaks[i] == wanted) {
            return i;
        }
    }
    target
}

fn check_unbroken(breaks: &[Break]) -> Result<(), Error> {
    let mut run = 0;
    for b in breaks {
        run = if *b == Break::None { run + 1 } else { 0 };
        if run > MAX_UNBROKEN_TOKENS {
            return Err(Error::InputTooLarge);
        }
    }
    Ok(())
}

fn make(tokens: &[Token], retained: &[usize], a: usize, b: usize) -> Window {
    Window {
        start: tokens[retained[a]].start,
        end: tokens[retained[b - 1]].end,
        tok_start: a,
        tok_end: b,
    }
}

/// Whether window `w` trusts a prediction over retained positions `first..=last` of a
/// document with `n` retained positions: it must sit at least `CONTEXT_MARGIN_TOKENS` from
/// every edge of `w` that is not a document edge.
///
/// Coverage: a span of `L <= MAX_ENTITY_TOKENS` positions untrusted in window A (its end
/// within the margin of A's end) and in the next window B (its start within the margin of B's
/// start) needs `L > OVERLAP_TOKENS - 2 * CONTEXT_MARGIN_TOKENS = MAX_ENTITY_TOKENS`, a
/// contradiction; so every such span is trusted in at least one window.
pub(crate) fn trusted(w: &Window, n: usize, first: usize, last: usize) -> bool {
    (w.tok_start == 0 || first >= w.tok_start + CONTEXT_MARGIN_TOKENS)
        && (w.tok_end == n || last + CONTEXT_MARGIN_TOKENS < w.tok_end)
}

/// Deduplicates predictions from overlapping windows and resolves overlaps: exact duplicates
/// collapse; spans are then taken best first (a rules span before a model span, then the
/// longer, then the more confident, then the earlier) and each is kept unless it overlaps one
/// already kept, so a span is lost only to a better span it overlaps. Rules spans never
/// overlap each other, so none is dropped for a model span. With a mask, spans it does not fully
/// contain are dropped first. Output is sorted by start and non-overlapping.
pub(crate) fn merge(mut spans: Vec<Entity>, mask: Option<&Mask>) -> Vec<Entity> {
    if let Some(mask) = mask {
        spans.retain(|e| mask.contains(e.start, e.end));
    }
    spans.sort_by(|a, b| a.start.cmp(&b.start).then(b.end.cmp(&a.end)));
    spans.dedup_by(|a, b| a.start == b.start && a.end == b.end && a.kind == b.kind);
    let rank = |e: &Entity| {
        (
            e.source == Source::Rules,
            e.end - e.start,
            e.confidence.to_bits(),
            std::cmp::Reverse(e.start),
        )
    };
    spans.sort_by_key(|e| std::cmp::Reverse(rank(e)));
    let mut kept: BTreeMap<usize, Entity> = BTreeMap::new();
    for s in spans {
        let before = kept.range(..s.end).next_back();
        if before.is_none_or(|(_, k)| k.end <= s.start) {
            kept.insert(s.start, s);
        }
    }
    kept.into_values().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Kind;
    use crate::token::tokenize;

    /// Plain-text windowing, the path every caller without a mask takes.
    fn windows(
        tokens: &[Token],
        retained: &[usize],
        window: usize,
        overlap: usize,
    ) -> Result<Vec<Window>, Error> {
        super::windows(tokens, retained, window, overlap, None)
    }

    /// Plain-text merging.
    fn merge(spans: Vec<Entity>) -> Vec<Entity> {
        super::merge(spans, None)
    }

    /// A text of `n` one-character tokens, `sep(i)` before token `i`. The tokens are `.` so
    /// that an empty separator still leaves them separate tokens.
    fn text_with(n: usize, sep: impl Fn(usize) -> &'static str) -> String {
        let mut s = String::from(".");
        for i in 1..n {
            s.push_str(sep(i));
            s.push('.');
        }
        s
    }

    fn retained(tokens: &[Token]) -> Vec<usize> {
        tokens
            .iter()
            .enumerate()
            .filter(|(_, t)| !matches!(t.class, TokenClass::Space | TokenClass::Newline))
            .map(|(i, _)| i)
            .collect()
    }

    fn cover(text: &str) -> (Vec<Window>, usize) {
        let tokens = tokenize(text);
        let r = retained(&tokens);
        let n = r.len();
        (
            windows(&tokens, &r, WINDOW_TOKENS, OVERLAP_TOKENS).unwrap(),
            n,
        )
    }

    fn example() -> String {
        text_with(2500, |i| match i {
            900 | 1700 | 2400 => "\n\n",
            i if i % 40 == 0 => "\n",
            _ => "",
        })
    }

    #[test]
    fn a_short_document_is_one_window() {
        let (w, n) = cover(&text_with(50, |_| " "));
        assert_eq!(n, 50);
        assert_eq!(w.len(), 1);
        assert_eq!((w[0].tok_start, w[0].tok_end), (0, 50));
    }

    #[test]
    fn the_worked_example_cuts_where_the_module_doc_says() {
        let (w, _) = cover(&example());
        let got: Vec<(usize, usize)> = w.iter().map(|w| (w.tok_start, w.tok_end)).collect();
        assert_eq!(got, vec![(0, 900), (480, 1480), (1080, 2080), (1680, 2500)]);
    }

    #[test]
    fn windows_are_bounded_overlapping_and_cover_the_document() {
        let (w, n) = cover(&example());
        assert_eq!(w[0].tok_start, 0);
        assert_eq!(w.last().unwrap().tok_end, n);
        for pair in w.windows(2) {
            assert!(pair[1].tok_start > pair[0].tok_start);
            assert!(pair[0].tok_end - pair[1].tok_start >= OVERLAP_TOKENS);
        }
        assert!(w.iter().all(|w| w.tok_end - w.tok_start <= WINDOW_TOKENS));
    }

    #[test]
    fn byte_ranges_match_the_token_ranges() {
        let text = example();
        let tokens = tokenize(&text);
        let r = retained(&tokens);
        let w = windows(&tokens, &r, WINDOW_TOKENS, OVERLAP_TOKENS).unwrap();
        for w in &w {
            assert_eq!(w.start, tokens[r[w.tok_start]].start);
            assert_eq!(w.end, tokens[r[w.tok_end - 1]].end);
        }
    }

    #[test]
    fn a_span_across_a_naive_cut_is_trusted_somewhere() {
        let (w, n) = cover(&text_with(2500, |i| if i % 40 == 0 { "\n" } else { " " }));
        assert!(w.iter().any(|w| trusted(w, n, 1000, 1039)));
    }

    #[test]
    fn every_entity_up_to_the_limit_is_trusted_in_some_window() {
        let (w, n) = cover(&example());
        for len in 1..=MAX_ENTITY_TOKENS {
            for first in (0..n - len).step_by(7) {
                let last = first + len - 1;
                assert!(
                    w.iter().any(|w| trusted(w, n, first, last)),
                    "span {first}..={last}"
                );
            }
        }
    }

    #[test]
    fn an_unbroken_run_is_too_large_and_a_broken_one_is_not() {
        let unbroken = ".".repeat(5000);
        let tokens = tokenize(&unbroken);
        let r = retained(&tokens);
        assert_eq!(r.len(), 5000);
        assert_eq!(
            windows(&tokens, &r, WINDOW_TOKENS, OVERLAP_TOKENS),
            Err(Error::InputTooLarge)
        );
        let mut broken = ".".repeat(4000);
        broken.push(' ');
        broken.push_str(&".".repeat(1000));
        let tokens = tokenize(&broken);
        let r = retained(&tokens);
        assert!(windows(&tokens, &r, WINDOW_TOKENS, OVERLAP_TOKENS).is_ok());
    }

    #[test]
    fn an_empty_document_has_no_windows() {
        assert_eq!(
            windows(&[], &[], WINDOW_TOKENS, OVERLAP_TOKENS),
            Ok(Vec::new())
        );
    }

    fn entity(kind: Kind, source: Source, start: usize, end: usize, confidence: f32) -> Entity {
        Entity {
            kind,
            start,
            end,
            confidence,
            review_recommended: false,
            source,
            components: Vec::new(),
            normalized: None,
            region: None,
        }
    }

    #[test]
    fn merge_resolves_duplicates_nesting_and_ties() {
        let m = |k, s, e, c| entity(k, Source::Model, s, e, c);
        assert_eq!(
            merge(vec![
                m(Kind::Person, 0, 10, 0.9),
                m(Kind::Person, 0, 10, 0.9)
            ])
            .len(),
            1
        );
        let nested = merge(vec![
            m(Kind::Person, 10, 22, 0.99),
            m(Kind::Address, 0, 60, 0.6),
        ]);
        assert_eq!(nested.len(), 1);
        assert_eq!(nested[0].kind, Kind::Address);
        let tie = merge(vec![m(Kind::Person, 0, 10, 0.7), m(Kind::Org, 5, 15, 0.9)]);
        assert_eq!(tie.len(), 1);
        assert_eq!(tie[0].kind, Kind::Org);
    }

    #[test]
    fn merge_keeps_a_rules_span_over_a_longer_model_span() {
        let out = merge(vec![
            entity(Kind::Org, Source::Model, 15, 40, 0.95),
            entity(Kind::Email, Source::Rules, 20, 40, 0.99),
        ]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].kind, Kind::Email);
    }

    /// A span is lost only to a better span it overlaps: `A` overlaps the longer `B`, `B`
    /// overlaps the still longer `C`, and `A` and `C` do not overlap, so `A` survives.
    #[test]
    fn merge_keeps_a_span_whose_rival_lost_to_a_third() {
        let m = |k, s, e| entity(k, Source::Model, s, e, 0.9);
        let out = merge(vec![
            m(Kind::Person, 0, 5),
            m(Kind::Org, 3, 15),
            m(Kind::Address, 14, 40),
        ]);
        let got: Vec<(Kind, usize, usize)> = out.iter().map(|e| (e.kind, e.start, e.end)).collect();
        assert_eq!(got, vec![(Kind::Person, 0, 5), (Kind::Address, 14, 40)]);
    }

    #[test]
    fn merge_output_is_sorted_and_empty_stays_empty() {
        assert!(merge(Vec::new()).is_empty());
        let m = |s, e| entity(Kind::Person, Source::Model, s, e, 0.9);
        let out = merge(vec![m(50, 60), m(0, 10), m(20, 30)]);
        let starts: Vec<usize> = out.iter().map(|e| e.start).collect();
        assert_eq!(starts, vec![0, 20, 50]);
    }

    #[test]
    fn mask_subtracts_skips_from_scans() {
        let m = Mask::new(
            &[(0, 100), (200, 300)],
            &[(10, 20), (90, 110), (250, 260), (400, 410)],
        );
        assert_eq!(m.ranges(), &[(0, 10), (20, 90), (200, 250), (260, 300)]);
        assert!(m.contains(20, 90));
        assert!(!m.contains(15, 25));
        assert!(!m.contains(85, 95));
        assert!(m.contains(260, 300));
        assert!(!m.contains(300, 301));
    }

    #[test]
    fn mask_with_skip_covering_whole_scan_is_empty_there() {
        let m = Mask::new(&[(0, 10), (20, 30)], &[(0, 10)]);
        assert_eq!(m.ranges(), &[(20, 30)]);
    }

    #[test]
    fn a_long_masked_run_belongs_to_no_window() {
        let text = format!("x y z ```\n{}```\n p q", "code ".repeat(40));
        let tokens = tokenize(&text);
        let r = retained(&tokens);
        let code_start = text.find("```").unwrap_or(0);
        let code_end = text.rfind("```").map_or(0, |i| i + 3);
        let mask = Mask::new(&[(0, text.len())], &[(code_start, code_end)]);
        let masked: Vec<bool> = r.iter().map(|&i| !mask.covers(&tokens[i])).collect();
        let w = super::windows(&tokens, &r, WINDOW_TOKENS, OVERLAP_TOKENS, Some(&masked)).unwrap();
        assert_eq!(w.len(), 2);
        for win in &w {
            assert!((win.tok_start..win.tok_end).all(|i| !masked[i]));
        }
    }

    #[test]
    fn a_window_never_ends_inside_a_short_masked_run() {
        let masked = [false, false, true, true, true, false, false];
        assert_eq!(outside_masked_run(Some(&masked), 0, 3), 2);
        assert_eq!(outside_masked_run(Some(&masked), 2, 3), 5);
        assert_eq!(outside_masked_run(Some(&masked), 0, 5), 5);
        assert_eq!(outside_masked_run(None, 0, 3), 3);
    }

    #[test]
    fn merge_drops_spans_the_mask_does_not_contain() {
        let m = |s, e| entity(Kind::Email, Source::Rules, s, e, 0.99);
        let mask = Mask::new(&[(0, 20)], &[]);
        let out = super::merge(vec![m(2, 10), m(15, 25)], Some(&mask));
        assert_eq!(out.len(), 1);
        assert_eq!((out[0].start, out[0].end), (2, 10));
    }
}
