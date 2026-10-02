use super::*;
use crate::{
    chunk,
    features::is_content,
    model::bio::{decode_detector, decode_detector_with_text},
    token::tokenize,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Piece {
    kind: String,
    start: usize,
    end: usize,
    confidence: f32,
}
#[derive(Deserialize)]
struct Fixture {
    name: String,
    text: String,
    pieces: Vec<Piece>,
    expected_join: bool,
}

type DecoderInputs = (Vec<(usize, usize)>, Vec<f32>, Vec<bool>, Vec<bool>);

fn input(text: &str, pieces: &[Piece]) -> DecoderInputs {
    let tokens = tokenize(text);
    let retained: Vec<_> = tokens
        .iter()
        .enumerate()
        .filter_map(|(i, t)| is_content(t).then_some(i))
        .collect();
    let bounds: Vec<_> = retained
        .iter()
        .map(|&i| (tokens[i].start, tokens[i].end))
        .collect();
    let mut probs = vec![0.001; bounds.len() * DETECTOR_LABELS];
    for (i, &(s, e)) in bounds.iter().enumerate() {
        let piece = pieces.iter().find(|p| p.start <= s && e <= p.end);
        let (label, confidence) = if let Some(p) = piece {
            let base = match p.kind.as_str() {
                "person" => 1,
                "org" => 3,
                "address" => 5,
                _ => 0,
            };
            (base + usize::from(s != p.start && base != 0), p.confidence)
        } else {
            (0, 0.99)
        };
        probs[i * DETECTOR_LABELS + label] = confidence;
    }
    let masked = vec![false; bounds.len()];
    let breaks = chunk::paragraph_breaks(&tokens, &retained);
    (bounds, probs, masked, breaks)
}

#[test]
fn address_continuation_all_control_and_saved_text_fixtures() {
    let fixtures: Vec<Fixture> =
        serde_json::from_str(include_str!("address_continuation_fixtures.json")).unwrap();
    let mut joined = 0;
    for fixture in &fixtures {
        let (bounds, probs, masked, breaks) = input(&fixture.text, &fixture.pieces);
        let old = decode_detector(&probs, &masked, &breaks);
        let new = decode_detector_with_text(&fixture.text, &bounds, &probs, &masked, &breaks);
        if fixture.expected_join {
            assert_eq!(old.len(), 2, "{}", fixture.name);
            assert_eq!(new.len(), 1, "{}", fixture.name);
            assert_eq!(
                (new[0].first, new[0].last),
                (old[0].first, old[1].last),
                "{}",
                fixture.name
            );
            joined += 1;
        } else {
            assert_eq!(new, old, "{}", fixture.name);
        }
    }
    assert_eq!(joined, 14);
    assert_eq!(fixtures.len(), 73);
}

fn sample() -> Fixture {
    let text =
        "Office delivery location\nNorth Building\n123 Main Street\nDenver, CO 80202".to_string();
    let left = text.find("North Building").unwrap();
    let right = text.find("123 Main Street").unwrap();
    let end = text.len();
    Fixture {
        name: "sample".to_string(),
        text,
        pieces: vec![
            Piece {
                kind: "address".to_string(),
                start: left,
                end: left + 14,
                confidence: 0.61,
            },
            Piece {
                kind: "address".to_string(),
                start: right,
                end,
                confidence: 0.98,
            },
        ],
        expected_join: true,
    }
}

#[test]
fn address_continuation_exact_original_mixed_b_i_f32_confidence() {
    let f = sample();
    let (bounds, mut probs, masked, breaks) = input(&f.text, &f.pieces);
    let old = decode_detector(&probs, &masked, &breaks);
    assert_eq!(old.len(), 2);
    let start = old[0].first;
    let last = old[1].last;
    for i in start..=last {
        let label = if i == start || i == old[1].first {
            5
        } else {
            6
        };
        probs[i * DETECTOR_LABELS + label] = if i % 2 == 0 { 0.9876543 } else { 0.6123457 };
    }
    let labels: Vec<_> = (0..bounds.len())
        .map(|i| {
            if i == start || i == old[1].first {
                5
            } else {
                6
            }
        })
        .collect();
    assert_eq!(
        decode_detector(&probs, &masked, &breaks)
            .iter()
            .map(|s| (s.first, s.last))
            .collect::<Vec<_>>(),
        old.iter().map(|s| (s.first, s.last)).collect::<Vec<_>>()
    );
    let mut sum = probs[start * DETECTOR_LABELS + labels[start]];
    for i in start + 1..=last {
        sum += probs[i * DETECTOR_LABELS + labels[i]];
    }
    let new = decode_detector_with_text(&f.text, &bounds, &probs, &masked, &breaks);
    assert_eq!(new.len(), 1);
    assert_eq!(
        new[0].confidence.to_bits(),
        (sum / (last + 1 - start) as f32).to_bits()
    );
    assert_eq!(labels[old[1].first], 5);
}

#[test]
fn address_continuation_refuses_o_masked_paragraph_and_invalid_bounds() {
    let f = sample();
    let (bounds, probs, masked, breaks) = input(&f.text, &f.pieces);
    let old = decode_detector(&probs, &masked, &breaks);
    let middle = old[1].first;
    let mut bp = breaks.clone();
    bp[middle] = true;
    assert_eq!(
        decode_detector_with_text(&f.text, &bounds, &probs, &masked, &bp),
        decode_detector(&probs, &masked, &bp)
    );
    let mut mask = masked.clone();
    mask[middle] = true;
    assert_eq!(
        decode_detector_with_text(&f.text, &bounds, &probs, &mask, &breaks),
        decode_detector(&probs, &mask, &breaks)
    );
    let mut o = probs.clone();
    o[middle * DETECTOR_LABELS..(middle + 1) * DETECTOR_LABELS].fill(0.001);
    o[middle * DETECTOR_LABELS] = 0.99;
    assert_eq!(
        decode_detector_with_text(&f.text, &bounds, &o, &masked, &breaks),
        decode_detector(&o, &masked, &breaks)
    );
    let mut bad = bounds.clone();
    bad[0].1 = f.text.len() + 1;
    assert_eq!(
        decode_detector_with_text(&f.text, &bad, &probs, &masked, &breaks),
        old
    );
    assert_eq!(
        decode_detector_with_text(
            &f.text,
            &bounds[..bounds.len() - 1],
            &probs,
            &masked,
            &breaks
        ),
        old
    );
}

#[test]
fn address_continuation_max_span_and_window_relative_indices() {
    for (names, expected) in [(248, true), (249, false)] {
        let text = format!(
            "North Building\n123 {}Road\nDenver, CO 80202",
            "Oak ".repeat(names)
        );
        let right = text.find("123").unwrap();
        let pieces = vec![
            Piece {
                kind: "address".to_string(),
                start: 0,
                end: 14,
                confidence: 0.95,
            },
            Piece {
                kind: "address".to_string(),
                start: right,
                end: text.len(),
                confidence: 0.95,
            },
        ];
        let (bounds, p, masked, breaks) = input(&text, &pieces);
        let old = decode_detector(&p, &masked, &breaks);
        let new = decode_detector_with_text(&text, &bounds, &p, &masked, &breaks);
        assert_eq!(old.len(), 2);
        assert_eq!(new.len(), if expected { 1 } else { 2 });
        if expected {
            assert_eq!(new[0].last + 1 - new[0].first, 256);
        } else {
            assert_eq!(new, old);
        }
    }
    let f = sample();
    let prefix = "padding ".repeat(80) + "\n";
    let text = prefix.clone() + &f.text + "\n" + &"after ".repeat(80);
    let pieces: Vec<_> = f
        .pieces
        .iter()
        .map(|p| Piece {
            kind: p.kind.clone(),
            start: p.start + prefix.len(),
            end: p.end + prefix.len(),
            confidence: p.confidence,
        })
        .collect();
    let (bounds, p, masked, breaks) = input(&text, &pieces);
    let full = decode_detector_with_text(&text, &bounds, &p, &masked, &breaks);
    let range = 16..bounds.len() - 16;
    let window = decode_detector_with_text(
        &text,
        &bounds[range.clone()],
        &p[range.start * 7..range.end * 7],
        &masked[range.clone()],
        &breaks[range.clone()],
    );
    assert_eq!(window.len(), 1);
    assert_eq!(
        (window[0].first + range.start, window[0].last + range.start),
        (full[0].first, full[0].last)
    );
    assert_eq!(window[0].confidence.to_bits(), full[0].confidence.to_bits());
    let w = chunk::Window {
        start: bounds[range.start].0,
        end: bounds[range.end - 1].1,
        tok_start: range.start,
        tok_end: range.end,
        piece_start: 0,
        piece_end: bounds.len(),
    };
    assert!(chunk::trusted(&w, full[0].first, full[0].last));
}

#[test]
fn address_continuation_zip4_rule_view_preserves_raw_indices_and_confidence() {
    let text = "Office delivery location\nE231\n1301 North Columbia Road Stop 9037\nGrand Forks, ND 58202-9037";
    let left = text.find("E231").unwrap();
    let right = text.find("1301").unwrap();
    let raw_end = text.rfind("-9037").unwrap();
    let pieces = vec![
        Piece {
            kind: "address".to_string(),
            start: left,
            end: left + 4,
            confidence: 0.65,
        },
        Piece {
            kind: "address".to_string(),
            start: right,
            end: raw_end,
            confidence: 0.97,
        },
    ];
    let (bounds, p, masked, breaks) = input(text, &pieces);
    let old = decode_detector(&p, &masked, &breaks);
    assert_eq!(old.len(), 2);
    let new = decode_detector_with_text(text, &bounds, &p, &masked, &breaks);
    assert_eq!(new.len(), 1);
    assert_eq!(bounds[new[0].last].1, raw_end);
    assert_eq!(new[0].last, old[1].last);
    assert_eq!(
        crate::detect::normalized_us_address_end(
            text,
            bounds[new[0].first].0,
            bounds[new[0].last].1
        ),
        text.len()
    );
    let labels: Vec<_> = (0..bounds.len())
        .map(|i| {
            if i == old[0].first || i == old[1].first {
                5
            } else {
                6
            }
        })
        .collect();
    let mut sum = p[new[0].first * 7 + labels[new[0].first]];
    for i in new[0].first + 1..=new[0].last {
        sum += p[i * 7 + labels[i]];
    }
    assert_eq!(
        new[0].confidence.to_bits(),
        (sum / (new[0].last + 1 - new[0].first) as f32).to_bits()
    );
}
