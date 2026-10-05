use super::*;
use crate::{Kind, features::is_content, token::tokenize};

fn same(a: &[model::bio::DetectedSpan], b: &[model::bio::DetectedSpan]) {
    let tuples = |values: &[model::bio::DetectedSpan]| {
        values
            .iter()
            .map(|s| (s.kind, s.first, s.last, s.confidence.to_bits()))
            .collect::<Vec<_>>()
    };
    assert_eq!(tuples(a), tuples(b));
}

type Fixture = (Vec<(usize, usize)>, Vec<f32>, Vec<bool>, Vec<bool>);

fn fixture(text: &str) -> Fixture {
    let all = tokenize(text);
    let bounds: Vec<_> = all
        .iter()
        .filter(|t| is_content(t))
        .map(|t| (t.start, t.end))
        .collect();
    let mut probs = vec![0.001; bounds.len() * model::DETECTOR_LABELS];
    for (i, row) in probs.chunks_mut(model::DETECTOR_LABELS).enumerate() {
        row[if i == 0 { 5 } else { 6 }] = 0.91;
    }
    let n = bounds.len();
    (bounds, probs, vec![false; n], vec![false; n])
}

#[test]
fn missing_and_explicit_old_contract_preserve_both_existing_decoders_bitwise() {
    let text = "123 Main Street, Denver, CO 80202";
    let (bounds, probs, mask, breaks) = fixture(text);
    let legacy = model::bio::decode_detector(&probs, &mask, &breaks);
    same(
        &decode(false, None, text, &bounds, &probs, &mask, &breaks).unwrap(),
        &legacy,
    );
    let old = model::bio::decode_detector_with_text(text, &bounds, &probs, &mask, &breaks);
    same(
        &decode(true, None, text, &bounds, &probs, &mask, &breaks).unwrap(),
        &old,
    );
    same(
        &decode(
            true,
            Some(DetectorPostprocessContract::AddressContinuationV1),
            text,
            &bounds,
            &probs,
            &mask,
            &breaks,
        )
        .unwrap(),
        &old,
    );
    assert!(
        decode(
            false,
            Some(DetectorPostprocessContract::AddressLabeledFieldsV1),
            text,
            &bounds,
            &probs,
            &mask,
            &breaks
        )
        .is_err()
    );
}

#[test]
fn opt_in_uses_actual_labeled_field_core_with_same_chosen_label_confidence() {
    let text = "PO Box 549\nVinton IA 52349\nPhysical Address\n101 First Street\nVinton IA 52349";
    let (bounds, probs, mask, breaks) = fixture(text);
    let old = decode(true, None, text, &bounds, &probs, &mask, &breaks).unwrap();
    assert_eq!(old.len(), 1);
    let selected = decode(
        true,
        Some(DetectorPostprocessContract::AddressLabeledFieldsV1),
        text,
        &bounds,
        &probs,
        &mask,
        &breaks,
    )
    .unwrap();
    let direct =
        model::bio::decode_detector_with_labeled_fields(text, &bounds, &probs, &mask, &breaks);
    same(&selected, &direct);
    assert_eq!(selected.len(), 2);
    let quotes: Vec<_> = selected
        .iter()
        .map(|s| &text[bounds[s.first].0..bounds[s.last].1])
        .collect();
    assert_eq!(
        quotes,
        [
            "PO Box 549\nVinton IA 52349",
            "101 First Street\nVinton IA 52349"
        ]
    );
    assert!(
        selected
            .iter()
            .all(|s| s.kind == Kind::Address && (s.confidence - 0.91).abs() < 1e-6)
    );
}

#[test]
fn unsupported_heading_and_incomplete_postal_block_preserve_original_ranges() {
    for text in [
        "PO Box 549\nVinton IA 52349\nBilling Address\n101 First Street\nVinton IA 52349",
        "PO Box 549\nVinton IA\nPhysical Address\n101 First Street\nVinton IA 52349",
    ] {
        let (bounds, probs, mask, breaks) = fixture(text);
        let old = decode(true, None, text, &bounds, &probs, &mask, &breaks).unwrap();
        let selected = decode(
            true,
            Some(DetectorPostprocessContract::AddressLabeledFieldsV1),
            text,
            &bounds,
            &probs,
            &mask,
            &breaks,
        )
        .unwrap();
        same(&selected, &old);
    }
}

#[test]
fn proper_people_and_org_spans_preserve_bounds_and_confidence() {
    let text = "Jane Doe River Office";
    let (bounds, mut probs, mask, breaks) = fixture(text);
    assert_eq!(bounds.len(), 4);
    for (row, label) in probs.chunks_mut(model::DETECTOR_LABELS).zip([1, 2, 3, 4]) {
        row.fill(0.001);
        row[label] = 0.88;
    }
    let old = decode(true, None, text, &bounds, &probs, &mask, &breaks).unwrap();
    let selected = decode(
        true,
        Some(DetectorPostprocessContract::AddressLabeledFieldsV1),
        text,
        &bounds,
        &probs,
        &mask,
        &breaks,
    )
    .unwrap();
    same(&selected, &old);
    assert_eq!(selected.len(), 2);
    assert_eq!(selected[0].kind, Kind::Person);
    assert_eq!(selected[1].kind, Kind::Org);
}
