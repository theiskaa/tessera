//! BIO label ids and greedy transition-masked decoding, mirroring the trainer's
//! `dataset::decode_labels` and `model_eval::decode_probs`. Id 0 is `O`; semantic label `i`
//! (in `AddressLabel::ALL` order) has `B` at `1 + 2i` and `I` at `2 + 2i`.

use super::PARSER_LABELS;
use crate::{AddressLabel, Component};

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
