//! Invariants a prepared sample must hold before anything trains on it: no record in two
//! splits, every span valid and on token boundaries, every augmented copy traceable to a
//! training original. `prepare` refuses to write shards that fail, and `train` refuses a
//! sample manifest without a passing record.

use std::collections::HashMap;

use tessera::AddressLabel;
use tessera::internal::FeatureConfig;

use crate::data::{LabelledExample, Span, Split, augment, group_key};

/// Counts of each kind of violation; every one must be zero.
#[derive(Debug, Default, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Checks {
    /// Split keys that occur in more than one split.
    pub keys_in_two_splits: usize,
    /// Spans out of order, outside the text, off a char boundary, padded with whitespace, or
    /// labelled `unknown`.
    pub invalid_spans: usize,
    /// Rows whose spans cut a token, so the model could never predict them.
    pub unencodable_rows: usize,
    /// Augmented rows whose original is missing, in another country, or not in train.
    pub orphan_copies: usize,
    /// Augmented rows outside the train split.
    pub copies_outside_train: usize,
    /// Valid or test originals whose labelled components, as a set, match a train original's:
    /// the same address in two splits, whatever the split key says.
    pub components_in_two_splits: usize,
    /// Originals in valid or test sharing country, postcode, and house number with a train
    /// original: an independent look at leakage, since the split key cannot see it. Reported,
    /// not failed, because two places can share both.
    pub postcode_and_number_shared: usize,
}

impl Checks {
    /// Whether every failing count is zero.
    pub fn passed(&self) -> bool {
        Checks {
            postcode_and_number_shared: 0,
            ..self.clone()
        } == Checks::default()
    }
}

/// Every check over the prepared rows, originals and copies together, with up to `limit`
/// offending rows of each kind for error messages.
pub fn verify(
    rows: &[LabelledExample],
    copies: usize,
    fc: &FeatureConfig,
    limit: usize,
) -> (Checks, Vec<String>) {
    let mut checks = Checks::default();
    let mut examples = Vec::new();
    let mut note = |count: usize, kind: &str, r: &LabelledExample, why: String| {
        if count <= limit {
            examples.push(format!("{kind}: {} {:?}{why}", r.country, r.text));
        }
    };
    let mut split_of: HashMap<(String, String), Split> = HashMap::new();
    let mut originals: HashMap<u64, (&str, Split)> = HashMap::new();
    for r in rows.iter().filter(|r| !r.augmented) {
        originals.insert(r.id, (r.country.as_str(), r.split));
        let key = (r.country.clone(), group_key(&r.country, &r.text, &r.spans));
        match split_of.get(&key) {
            Some(&s) if s != r.split => {
                checks.keys_in_two_splits += 1;
                note(
                    checks.keys_in_two_splits,
                    "key in two splits",
                    r,
                    String::new(),
                );
            }
            Some(_) => {}
            None => {
                split_of.insert(key, r.split);
            }
        }
    }
    let pair = |r: &LabelledExample| {
        let get = |l: AddressLabel| {
            r.spans.iter().find(|s| s.label == l).map(|s| {
                r.text[s.start as usize..s.end as usize]
                    .to_lowercase()
                    .replace(' ', "")
            })
        };
        Some((
            r.country.clone(),
            get(AddressLabel::Postcode)?,
            get(AddressLabel::HouseNumber)?,
        ))
    };
    let components = |r: &LabelledExample| {
        let mut parts: Vec<(&str, String)> = r
            .spans
            .iter()
            .filter_map(|s| {
                let text = r.text.get(s.start as usize..s.end as usize)?;
                Some((s.label.as_str(), text.to_lowercase()))
            })
            .collect();
        parts.sort();
        (r.country.clone(), parts)
    };
    let train_components: std::collections::HashSet<_> = rows
        .iter()
        .filter(|r| !r.augmented && r.split == Split::Train)
        .map(components)
        .collect();
    for r in rows
        .iter()
        .filter(|r| !r.augmented && r.split != Split::Train)
    {
        if train_components.contains(&components(r)) {
            checks.components_in_two_splits += 1;
            note(
                checks.components_in_two_splits,
                "components in two splits",
                r,
                String::new(),
            );
        }
    }
    let train: std::collections::HashSet<_> = rows
        .iter()
        .filter(|r| !r.augmented && r.split == Split::Train)
        .filter_map(pair)
        .collect();
    checks.postcode_and_number_shared = rows
        .iter()
        .filter(|r| !r.augmented && r.split != Split::Train)
        .filter_map(pair)
        .filter(|k| train.contains(k))
        .count();
    for r in rows {
        if !spans_valid(&r.text, &r.spans) {
            checks.invalid_spans += 1;
            note(
                checks.invalid_spans,
                "invalid spans",
                r,
                format!(" {:?}", r.spans),
            );
        }
        if let Err(e) = crate::dataset::encode(&r.text, &r.spans, fc) {
            checks.unencodable_rows += 1;
            note(checks.unencodable_rows, "unencodable", r, format!(": {e}"));
        }
        if r.augmented {
            if r.split != Split::Train {
                checks.copies_outside_train += 1;
                note(
                    checks.copies_outside_train,
                    "copy outside train",
                    r,
                    String::new(),
                );
            }
            let original = originals.get(&augment::original_id(r.id, copies));
            if !original.is_some_and(|&(c, s)| c == r.country && s == Split::Train) {
                checks.orphan_copies += 1;
                note(checks.orphan_copies, "orphan copy", r, String::new());
            }
        }
    }
    (checks, examples)
}

/// Spans in order, non-empty, inside the text on char boundaries, without surrounding
/// whitespace, and never labelled `unknown`.
pub fn spans_valid(text: &str, spans: &[Span]) -> bool {
    let mut last = 0;
    spans.iter().all(|s| {
        let (a, b) = (s.start as usize, s.end as usize);
        let ok = a >= last
            && a < b
            && text.get(a..b).is_some_and(|t| t == t.trim())
            && s.label != AddressLabel::Unknown;
        last = b;
        ok
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(
        id: u64,
        text: &str,
        spans: Vec<Span>,
        split: Split,
        augmented: bool,
    ) -> LabelledExample {
        LabelledExample {
            id,
            group_id: 0,
            country: "GB".into(),
            language: "en".into(),
            text: text.into(),
            spans,
            split,
            augmented,
        }
    }

    fn span(label: AddressLabel, start: u32, end: u32) -> Span {
        Span { label, start, end }
    }

    #[test]
    fn a_clean_sample_passes() {
        let fc = FeatureConfig::default();
        let text = "10 Downing Street";
        let spans = vec![
            span(AddressLabel::HouseNumber, 0, 2),
            span(AddressLabel::Road, 3, 17),
        ];
        let copy_id = u64::MAX - 7 * 2;
        let rows = vec![
            row(7, text, spans.clone(), Split::Train, false),
            row(
                copy_id,
                "10 Downing St",
                vec![
                    span(AddressLabel::HouseNumber, 0, 2),
                    span(AddressLabel::Road, 3, 13),
                ],
                Split::Train,
                true,
            ),
            row(
                8,
                "London",
                vec![span(AddressLabel::City, 0, 6)],
                Split::Test,
                false,
            ),
        ];
        assert!(verify(&rows, 2, &fc, 0).0.passed());
    }

    #[test]
    fn each_violation_is_counted() {
        let fc = FeatureConfig::default();
        let spans = vec![
            span(AddressLabel::HouseNumber, 0, 2),
            span(AddressLabel::Road, 3, 17),
        ];
        let rows = vec![
            row(1, "10 Downing Street", spans.clone(), Split::Train, false),
            row(2, "10 Downing Street", spans, Split::Test, false),
            row(
                3,
                "London ",
                vec![span(AddressLabel::City, 0, 7)],
                Split::Train,
                false,
            ),
            row(
                4,
                "Londonderry",
                vec![span(AddressLabel::City, 0, 6)],
                Split::Train,
                false,
            ),
            row(
                u64::MAX - 999 * 2,
                "Leeds",
                vec![span(AddressLabel::City, 0, 5)],
                Split::Train,
                true,
            ),
            row(
                u64::MAX - 2,
                "10 Downing",
                vec![span(AddressLabel::HouseNumber, 0, 2)],
                Split::Valid,
                true,
            ),
        ];
        let c = verify(&rows, 2, &fc, 0).0;
        assert_eq!(c.keys_in_two_splits, 1);
        assert_eq!(c.invalid_spans, 1);
        assert_eq!(c.unencodable_rows, 1);
        assert_eq!(c.orphan_copies, 1);
        assert_eq!(c.copies_outside_train, 1);
        assert_eq!(c.components_in_two_splits, 1);
        assert!(!c.passed());
        let (_, examples) = verify(&rows, 2, &fc, 1);
        assert_eq!(examples.len(), 6);
    }

    #[test]
    fn a_shared_postcode_and_number_is_reported_not_failed() {
        let fc = FeatureConfig::default();
        let text = "10 Downing Street SW1A 2AA";
        let spans = |road_end: u32| {
            vec![
                span(AddressLabel::HouseNumber, 0, 2),
                span(AddressLabel::Road, 3, road_end),
                span(AddressLabel::Postcode, 18, 26),
            ]
        };
        let rows = vec![
            row(1, text, spans(17), Split::Train, false),
            row(
                2,
                "10 Whitehall Road SW1A 2AA",
                vec![
                    span(AddressLabel::HouseNumber, 0, 2),
                    span(AddressLabel::Road, 3, 17),
                    span(AddressLabel::Postcode, 18, 26),
                ],
                Split::Test,
                false,
            ),
        ];
        let c = verify(&rows, 2, &fc, 0).0;
        assert_eq!(c.postcode_and_number_shared, 1);
        assert!(c.passed());
    }
}
