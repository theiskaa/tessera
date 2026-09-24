//! BIO label ids and greedy decoding, mirroring the trainer's `dataset::decode_labels`,
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

/// Greedy decoding of `[len, DETECTOR_LABELS]` probabilities: the most probable label per
/// position, `O` at masked positions (inside an email or phone), an `I-X` that does not
/// continue an `X` span read as `B-X`, and every span closed before a paragraph break. Spans
/// longer than `MAX_ENTITY_TOKENS` are dropped.
pub fn decode_detector(
    probs: &[f32],
    masked: &[bool],
    paragraph_break: &[bool],
) -> Vec<DetectedSpan> {
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
        let label = if masked.get(t).copied().unwrap_or(false) {
            0
        } else {
            argmax(row)
        };
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
    fn stray_inside_labels_start_spans_and_masks_force_o() {
        // O, B-PERSON, I-PERSON, I-ORG, O, I-ADDRESS, I-ADDRESS
        let p = rows(&[0, 1, 2, 4, 0, 6, 6]);
        let none = [false; 7];
        assert_eq!(
            spans(&decode_detector(&p, &none, &none)),
            [
                (Kind::Person, 1, 2),
                (Kind::Org, 3, 3),
                (Kind::Address, 5, 6)
            ]
        );
        let mut masked = [false; 7];
        masked[1] = true;
        let found = decode_detector(&p, &masked, &none);
        assert!(!found.iter().any(|s| s.kind == Kind::Person && s.first == 1));
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
