//! BIO label ids and decoding, mirroring the trainer's `dataset::decode_labels`,
//! `model_eval::decode_probs`, and `detector::decode`. Id 0 is `O`; semantic label `i` has
//! `B` at `1 + 2i` and `I` at `2 + 2i`, in `AddressLabel::ALL` order for the parser and in
//! `DETECTOR_KINDS` order for the detector.

use super::{DETECTOR_LABELS, PARSER_LABELS};
use crate::chunk::MAX_ENTITY_TOKENS;
use crate::{AddressLabel, Component, Kind};

/// The detector's kinds in label order.
pub const DETECTOR_KINDS: [Kind; 3] = [Kind::Person, Kind::Org, Kind::Address];

/// Detector label strings in id order, as the bundle manifest lists them.
pub fn detector_label_strings() -> Vec<String> {
    let mut out = Vec::with_capacity(DETECTOR_LABELS);
    out.push("O".to_string());
    for kind in DETECTOR_KINDS {
        let name = kind.as_str().to_uppercase();
        out.push(format!("B-{name}"));
        out.push(format!("I-{name}"));
    }
    out
}

/// A decoded detector span over retained positions `first..=last`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectedSpan {
    pub kind: Kind,
    pub first: usize,
    pub last: usize,
    /// Mean probability of the chosen label over the span.
    pub confidence: f32,
}

/// Decoding of `[len, DETECTOR_LABELS]` probabilities: the most probable label sequence in
/// which every `I-X` continues an `X` span (Viterbi over log probabilities), `O` at masked
/// positions (inside an email or phone), and no span crossing a paragraph break. Spans longer
/// than `MAX_ENTITY_TOKENS` are dropped. Per-position argmax cuts spans where one label dips
/// below its neighbours (`[農林]水産省`); the sequence keeps them whole. On the real evaluation
/// sets a stray `I-X` is nearly always the model's error, so it never opens a span: letting it
/// open one at a cost gave back most of the gain.
pub fn decode_detector(
    probs: &[f32],
    masked: &[bool],
    paragraph_break: &[bool],
) -> Vec<DetectedSpan> {
    let labels = viterbi(probs, masked, paragraph_break);
    spans_of(probs, &labels, paragraph_break)
}

/// The cost of label `to` after label `from` (`None` at the start or after a paragraph
/// break): an `I-X` that does not continue an `X` span is impossible.
fn transition(from: Option<usize>, to: usize) -> f32 {
    if to == 0 || !to.is_multiple_of(2) {
        return 0.0;
    }
    let continues = from.is_some_and(|f| f != 0 && (f - 1) / 2 == (to - 1) / 2);
    if !continues { f32::NEG_INFINITY } else { 0.0 }
}

/// The best label sequence. Masked positions are `O`; after a paragraph break a position
/// cannot continue a span. A probability of zero or NaN (damaged weights, or a softmax that
/// saturated) counts as the smallest positive f32, and the scores are shifted so their best is
/// zero after every position: a huge floor would otherwise swallow every later difference in
/// f32 rounding and decide the rest of the window by ties.
fn viterbi(probs: &[f32], masked: &[bool], paragraph_break: &[bool]) -> Vec<usize> {
    const L: usize = DETECTOR_LABELS;
    let n = probs.len() / L;
    let floor = f32::MIN_POSITIVE.ln();
    let log = |p: f32| {
        if p.is_nan() || p <= 0.0 {
            floor
        } else {
            p.ln().max(floor)
        }
    };
    let mut score = [f32::NEG_INFINITY; L];
    let mut back: Vec<[usize; L]> = Vec::with_capacity(n);
    for t in 0..n {
        let row = &probs[t * L..(t + 1) * L];
        let forced_o = masked.get(t).copied().unwrap_or(false);
        let fresh = t == 0 || paragraph_break.get(t).copied().unwrap_or(false);
        let mut next = [f32::NEG_INFINITY; L];
        let mut from = [0usize; L];
        for (to, slot) in next.iter_mut().enumerate() {
            if forced_o && to != 0 {
                continue;
            }
            let emit = if forced_o { 0.0 } else { log(row[to]) };
            if t == 0 {
                *slot = emit + transition(None, to);
                continue;
            }
            for (prev, &s) in score.iter().enumerate() {
                let cost = transition(if fresh { None } else { Some(prev) }, to);
                if s + emit + cost > *slot {
                    *slot = s + emit + cost;
                    from[to] = prev;
                }
            }
        }
        let best = next.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        if best.is_finite() {
            next.iter_mut().for_each(|s| *s -= best);
        }
        score = next;
        back.push(from);
    }
    let mut labels = vec![0; n];
    if n == 0 {
        return labels;
    }
    let mut best = argmax(&score);
    for t in (0..n).rev() {
        labels[t] = best;
        best = back[t][best];
    }
    labels
}

/// The spans of a label sequence, with the mean probability of their labels. An `I-X` opens a
/// span unless it continues an `X` span, and every span closes before a paragraph break.
fn spans_of(probs: &[f32], labels: &[usize], paragraph_break: &[bool]) -> Vec<DetectedSpan> {
    let mut out = Vec::new();
    let mut open: Option<(usize, usize, f32, usize)> = None;
    let mut close = |open: &mut Option<(usize, usize, f32, usize)>, last: usize| {
        if let Some((kind, first, sum, n)) = open.take()
            && last + 1 - first <= MAX_ENTITY_TOKENS
        {
            out.push(DetectedSpan {
                kind: DETECTOR_KINDS[kind],
                first,
                last,
                confidence: sum / n as f32,
            });
        }
    };
    let mut prev: Option<usize> = None;
    for (t, row) in probs.chunks(DETECTOR_LABELS).enumerate() {
        if t > 0 && paragraph_break.get(t).copied().unwrap_or(false) {
            close(&mut open, t - 1);
            prev = None;
        }
        let label = labels.get(t).copied().unwrap_or(0);
        let kind = (label != 0).then(|| (label - 1) / 2);
        let begin = label % 2 == 1;
        let p = row
            .get(label)
            .copied()
            .filter(|p| !p.is_nan())
            .unwrap_or(0.0);
        match (kind, open.as_mut()) {
            (Some(k), Some(o)) if !begin && prev == Some(k) => {
                o.2 += p;
                o.3 += 1;
            }
            _ => {
                if t > 0 {
                    close(&mut open, t - 1);
                }
                if let Some(k) = kind {
                    open = Some((k, t, p, 1));
                }
            }
        }
        prev = kind;
    }
    if let Some(last) = (probs.len() / DETECTOR_LABELS).checked_sub(1) {
        close(&mut open, last);
    }
    out
}

/// The first index of the largest value; NaN never wins.
pub fn argmax(row: &[f32]) -> usize {
    let mut best = (0, f32::NEG_INFINITY);
    for (i, &p) in row.iter().enumerate() {
        if p > best.1 {
            best = (i, p);
        }
    }
    best.0
}

/// Label strings in id order, as the bundle manifest lists them.
pub fn parser_label_strings() -> Vec<String> {
    let mut out = Vec::with_capacity(PARSER_LABELS);
    out.push("O".to_string());
    for label in &AddressLabel::ALL[..(PARSER_LABELS - 1) / 2] {
        out.push(format!("B-{}", label.as_str()));
        out.push(format!("I-{}", label.as_str()));
    }
    out
}

fn is_inside(id: u8) -> bool {
    id != 0 && (id - 1) % 2 == 1
}

fn semantic(id: u8) -> usize {
    ((id - 1) / 2) as usize
}

/// The most probable legal label per row of `[len, classes]` probabilities, with its
/// probability. `I-x` is legal only after `B-x` or `I-x`.
pub(crate) fn decode_probs(probs: &[f32], classes: usize) -> Vec<(u8, f32)> {
    let mut out = Vec::with_capacity(probs.len() / classes.max(1));
    let mut prev: u8 = 0;
    for row in probs.chunks(classes) {
        let mut best = (0u8, f32::NEG_INFINITY);
        for (id, &p) in row.iter().enumerate() {
            let id = id as u8;
            // Damaged weights can produce NaN; it must not win, nor be reported as a probability.
            let p = if p.is_nan() { 0.0 } else { p };
            let legal = !is_inside(id) || (prev != 0 && semantic(prev) == semantic(id));
            if legal && p > best.1 {
                best = (id, p);
            }
        }
        out.push(best);
        prev = best.0;
    }
    out
}

/// Decoded labels merged into components over the retained tokens' byte spans. A component's
/// confidence is the mean probability of its tokens' labels.
pub(crate) fn components(token_spans: &[(usize, usize)], decoded: &[(u8, f32)]) -> Vec<Component> {
    let mut out: Vec<Component> = Vec::new();
    let mut sums: Vec<(f32, usize)> = Vec::new();
    let mut prev: u8 = 0;
    for (&(start, end), &(l, p)) in token_spans.iter().zip(decoded) {
        if l == 0 {
            prev = 0;
            continue;
        }
        let continues = is_inside(l) && prev != 0 && semantic(prev) == semantic(l);
        match (continues, out.last_mut(), sums.last_mut()) {
            (true, Some(c), Some(s)) => {
                c.end = end;
                s.0 += p;
                s.1 += 1;
            }
            _ => {
                out.push(Component {
                    label: AddressLabel::ALL[semantic(l)],
                    start,
                    end,
                    confidence: 0.0,
                });
                sums.push((p, 1));
            }
        }
        prev = l;
    }
    for (c, (s, n)) in out.iter_mut().zip(sums) {
        c.confidence = s / n as f32;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(labels: &[usize]) -> Vec<f32> {
        labels
            .iter()
            .flat_map(|&l| (0..DETECTOR_LABELS).map(move |k| if k == l { 0.9 } else { 0.1 / 6.0 }))
            .collect()
    }

    fn spans(found: &[DetectedSpan]) -> Vec<(Kind, usize, usize)> {
        found.iter().map(|s| (s.kind, s.first, s.last)).collect()
    }

    #[test]
    fn detector_labels_name_the_kinds_in_order() {
        assert_eq!(
            detector_label_strings(),
            [
                "O",
                "B-PERSON",
                "I-PERSON",
                "B-ORG",
                "I-ORG",
                "B-ADDRESS",
                "I-ADDRESS"
            ]
        );
    }

    #[test]
    fn stray_inside_labels_never_open_spans_and_masks_force_o() {
        // O, B-PERSON, I-PERSON, I-ORG, O, I-ADDRESS, I-ADDRESS
        let p = rows(&[0, 1, 2, 4, 0, 6, 6]);
        let none = [false; 7];
        let found = spans(&decode_detector(&p, &none, &none));
        assert_eq!(found[0], (Kind::Person, 1, 2));
        assert!(
            !found
                .iter()
                .any(|&(k, first, _)| k == Kind::Org && first == 3)
        );
        let mut masked = [false; 7];
        masked[1] = true;
        let found = decode_detector(&p, &masked, &none);
        assert!(!found.iter().any(|s| s.kind == Kind::Person && s.first == 1));
    }

    #[test]
    fn a_masked_token_ends_a_span() {
        // B-PERSON, I-PERSON, I-PERSON with the middle position masked.
        let p = rows(&[1, 2, 2]);
        let found = decode_detector(&p, &[false, true, false], &[false; 3]);
        assert_eq!(spans(&found)[0], (Kind::Person, 0, 0));
        assert!(!spans(&found).iter().any(|&(_, first, _)| first == 1));
    }

    #[test]
    fn empty_or_saturated_rows_do_not_swallow_the_rest_of_the_window() {
        let l = DETECTOR_LABELS;
        let row = |label: usize| {
            let mut r = vec![0.01; l];
            r[label] = 0.94;
            r
        };
        let none = |n: usize| vec![false; n];
        for bad in [vec![f32::NAN; l], vec![0.0; l], {
            let mut r = vec![0.0; l];
            r[2] = 1.0;
            r
        }] {
            let p: Vec<f32> = [bad, row(0), row(3), row(4), row(0)].concat();
            assert_eq!(
                spans(&decode_detector(&p, &none(5), &none(5))),
                [(Kind::Org, 2, 3)]
            );
        }
        let mut only_inside = vec![0.0; l];
        only_inside[2] = 1.0;
        let p: Vec<f32> = [row(1), row(2), only_inside, row(0), row(5), row(6)].concat();
        let mut breaks = none(6);
        breaks[2] = true;
        assert_eq!(
            spans(&decode_detector(&p, &none(6), &breaks)),
            [(Kind::Person, 0, 1), (Kind::Address, 4, 5)]
        );
    }

    #[test]
    fn a_dip_inside_a_span_keeps_it_whole() {
        // B-ORG 0.6, then a position where O (0.55) edges out I-ORG (0.45), then I-ORG 0.9.
        let mut p = vec![0.0; 3 * DETECTOR_LABELS];
        p[3] = 0.6;
        p[0] = 0.4;
        p[DETECTOR_LABELS] = 0.55;
        p[DETECTOR_LABELS + 4] = 0.45;
        p[2 * DETECTOR_LABELS + 4] = 0.9;
        p[2 * DETECTOR_LABELS] = 0.1;
        let found = decode_detector(&p, &[false; 3], &[false; 3]);
        assert_eq!(spans(&found), [(Kind::Org, 0, 2)]);
    }

    #[test]
    fn a_paragraph_break_splits_a_span() {
        let p = rows(&[5, 6, 6, 6]);
        let mut breaks = [false; 4];
        breaks[2] = true;
        assert_eq!(
            spans(&decode_detector(&p, &[false; 4], &breaks)),
            [(Kind::Address, 0, 1), (Kind::Address, 2, 3)]
        );
    }

    #[test]
    fn spans_over_the_entity_limit_are_dropped() {
        let mut labels = vec![5];
        labels.extend(std::iter::repeat_n(6, MAX_ENTITY_TOKENS));
        let n = labels.len();
        assert!(decode_detector(&rows(&labels), &vec![false; n], &vec![false; n]).is_empty());
        labels.pop();
        let n = labels.len();
        assert_eq!(
            decode_detector(&rows(&labels), &vec![false; n], &vec![false; n]).len(),
            1
        );
    }

    #[test]
    fn label_strings_follow_the_id_table() {
        let l = parser_label_strings();
        assert_eq!(l.len(), PARSER_LABELS);
        assert_eq!(l[1], "B-house_number");
        assert_eq!(l[4], "I-road");
        assert_eq!(l[22], "I-po_box");
    }

    #[test]
    fn inside_without_begin_is_masked() {
        let mut probs = vec![0.0f32; 2 * PARSER_LABELS];
        // Row 0: I-road (4) is most probable but illegal at the start; B-unit (5) wins.
        probs[4] = 0.6;
        probs[5] = 0.3;
        probs[1] = 0.1;
        // Row 1: I-unit (6) follows B-unit and is legal.
        probs[PARSER_LABELS + 6] = 0.9;
        probs[PARSER_LABELS] = 0.1;
        let d = decode_probs(&probs, PARSER_LABELS);
        assert_eq!(d, vec![(5, 0.3), (6, 0.9)]);
    }

    #[test]
    fn components_merge_and_average() {
        let spans = [(0, 4), (5, 6), (6, 7), (8, 13)];
        let decoded = [(5, 0.9), (6, 0.8), (0, 0.99), (3, 0.7)];
        let c = components(&spans, &decoded);
        assert_eq!(c.len(), 2);
        assert_eq!(
            (c[0].label, c[0].start, c[0].end),
            (AddressLabel::Unit, 0, 6)
        );
        assert!((c[0].confidence - 0.85).abs() < 1e-6);
        assert_eq!(
            (c[1].label, c[1].start, c[1].end),
            (AddressLabel::Road, 8, 13)
        );
        assert!((c[1].confidence - 0.7).abs() < 1e-6);
    }

    #[test]
    fn a_new_begin_starts_a_new_component() {
        let c = components(&[(0, 1), (2, 3)], &[(3, 0.9), (3, 0.9)]);
        assert_eq!(c.len(), 2);
    }
}
