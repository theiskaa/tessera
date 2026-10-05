use super::*;
use crate::{features::is_content, token::tokenize};
use serde::Deserialize;

#[derive(Deserialize)]
struct Control {
    name: String,
    text: String,
    mode: String,
    expected_address_texts: Vec<String>,
    mutation: Option<serde_json::Value>,
}

#[test]
fn strict_field_boundaries_native_token_controls() {
    let controls: Vec<Control> = serde_json::from_str(include_str!(
        "../../../fixtures/postprocess/address-field-boundaries.json"
    ))
    .unwrap();
    assert_eq!(controls.len(), 37);
    for control in controls {
        let text = &control.text;
        let all = tokenize(text);
        let mut tokens: Vec<_> = all
            .iter()
            .filter(|token| is_content(token))
            .map(|token| (token.start, token.end))
            .collect();
        let n = tokens.len();
        let mut labels = vec![6; n];
        labels[0] = 5;
        let mut probs = vec![0.001; n * DETECTOR_LABELS];
        for (index, &label) in labels.iter().enumerate() {
            probs[index * DETECTOR_LABELS + label] = 0.91;
        }
        let mut masked = vec![false; n];
        let mut breaks = vec![false; n];
        let mutation = control.mutation.unwrap_or(serde_json::Value::Null);
        let mut spans = vec![DetectedSpan {
            kind: Kind::Address,
            first: 0,
            last: n - 1,
            confidence: 0.91,
        }];
        let at_word = |word: &str| {
            tokens
                .iter()
                .position(|&(a, b)| &text[a..b] == word)
                .unwrap()
        };
        if control.mode == "tail" {
            let at = at_word("101");
            let kind = if mutation["next_kind"].as_str() == Some("org") {
                Kind::Org
            } else {
                Kind::Address
            };
            spans[0].last = at - 1;
            spans.push(DetectedSpan {
                kind,
                first: at,
                last: n - 1,
                confidence: 0.91,
            });
            labels[at] = if kind == Kind::Org { 3 } else { 5 };
            if kind == Kind::Org {
                labels[at + 1..].fill(4);
            }
        }
        if let Some(word) = mutation["mask"].as_str() {
            masked[at_word(word)] = true;
        }
        if let Some(word) = mutation["break"].as_str() {
            breaks[at_word(word)] = true;
        }
        if let Some(change) = mutation["label"].as_array() {
            labels[at_word(change[0].as_str().unwrap())] = change[1].as_u64().unwrap() as usize;
        }
        let invalid_bounds = mutation["invalid_bound"].as_bool() == Some(true);
        if invalid_bounds {
            tokens[0].0 = text.find('\u{a0}').unwrap() + 1;
        }
        if mutation["overlap"].as_bool() == Some(true) {
            tokens[1].0 = tokens[0].0;
        }
        if mutation["truncate_probs"].as_bool() == Some(true) {
            probs.truncate(probs.len() - DETECTOR_LABELS);
        }
        if let Some(kind) = mutation["probability"].as_str() {
            let value = match kind {
                "nan" => f32::NAN,
                "pos_inf" => f32::INFINITY,
                "neg_inf" => f32::NEG_INFINITY,
                "negative" => -0.1,
                "above_one" => 1.1,
                _ => panic!("unknown malformed-probability control"),
            };
            let at = if mutation["unchosen"].as_bool() == Some(true) {
                0
            } else {
                labels[0]
            };
            probs[at] = value;
        }
        let output = separate(
            text,
            &tokens,
            &probs,
            &labels,
            &masked,
            &breaks,
            spans.clone(),
        );
        if invalid_bounds {
            assert_eq!(output, spans, "{}", control.name);
            continue;
        }
        let addresses: Vec<_> = output
            .iter()
            .filter(|span| span.kind == Kind::Address)
            .map(|span| text[tokens[span.first].0..tokens[span.last].1].to_string())
            .collect();
        assert_eq!(
            addresses, control.expected_address_texts,
            "{}",
            control.name
        );
        assert_eq!(
            output
                .iter()
                .filter(|span| span.kind != Kind::Address)
                .copied()
                .collect::<Vec<_>>(),
            spans
                .iter()
                .filter(|span| span.kind != Kind::Address)
                .copied()
                .collect::<Vec<_>>(),
            "{}",
            control.name
        );
    }
}

#[test]
fn splitting_recomputes_chosen_labels_without_heading_or_kind_sums() {
    let text = "PO Box 17\nDenver CO 80202\nPhysical Address 123 Main Street\nDenver CO 80202";
    let all = tokenize(text);
    let tokens: Vec<_> = all
        .iter()
        .filter(|token| is_content(token))
        .map(|token| (token.start, token.end))
        .collect();
    let n = tokens.len();
    let labels: Vec<_> = (0..n).map(|i| if i == 0 { 5 } else { 6 }).collect();
    let mut probs = vec![0.001; n * DETECTOR_LABELS];
    for (index, &(a, b)) in tokens.iter().enumerate() {
        probs[index * DETECTOR_LABELS + labels[index]] =
            if matches!(&text[a..b], "Physical" | "Address") {
                0.1
            } else {
                0.9
            };
    }
    let output = separate(
        text,
        &tokens,
        &probs,
        &labels,
        &vec![false; n],
        &vec![false; n],
        vec![DetectedSpan {
            kind: Kind::Address,
            first: 0,
            last: n - 1,
            confidence: 0.5,
        }],
    );
    assert_eq!(output.len(), 2);
    assert!(
        output
            .iter()
            .all(|span| (span.confidence - 0.9).abs() < 1e-6)
    );
}

#[test]
fn explicit_decoder_separates_postal_fields_after_legacy_continuation() {
    let text = "PO Box 17\nDenver CO 80202\nPhysical Address 123 Main Street\nDenver CO 80202";
    let tokens: Vec<_> = tokenize(text)
        .into_iter()
        .filter(is_content)
        .map(|token| (token.start, token.end))
        .collect();
    let mut probs = vec![0.001; tokens.len() * DETECTOR_LABELS];
    for (index, &(a, b)) in tokens.iter().enumerate() {
        let label = if index == 0 { 5 } else { 6 };
        probs[index * DETECTOR_LABELS + label] = if matches!(&text[a..b], "Physical" | "Address") {
            0.1
        } else {
            0.9
        };
    }
    let masked = vec![false; tokens.len()];
    let breaks = vec![false; tokens.len()];
    let legacy =
        super::super::bio::decode_detector_with_text(text, &tokens, &probs, &masked, &breaks);
    assert_eq!(legacy.len(), 1);
    let separated = super::super::bio::decode_detector_with_labeled_fields(
        text, &tokens, &probs, &masked, &breaks,
    );
    let texts: Vec<_> = separated
        .iter()
        .map(|span| &text[tokens[span.first].0..tokens[span.last].1])
        .collect();
    assert_eq!(
        texts,
        [
            "PO Box 17\nDenver CO 80202",
            "123 Main Street\nDenver CO 80202"
        ]
    );
    assert!(
        separated
            .iter()
            .all(|span| (span.confidence - 0.9).abs() < 1e-6)
    );
}
