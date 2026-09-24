//! Confidence bands. What the library is allowed to return is decided here and
//! nowhere else: high passes, medium passes flagged, low is dropped unless the
//! caller asks for uncertain results. `parse_address` is the exception: its caller
//! already knows the text is an address, so the entity is always returned and the
//! bands apply to its components.

use crate::{AddressLabel, Component, Entity, Extraction, Kind};

pub(crate) const HIGH: f32 = 0.85;
pub(crate) const MEDIUM: f32 = 0.50;

/// Lowest mean label probability a detected person keeps. Person and org are precision-first:
/// nothing downstream can reject a wrong name.
pub const DETECT_MIN_PERSON: f32 = 0.80;
/// Lowest mean label probability a detected org keeps.
pub const DETECT_MIN_ORG: f32 = 0.80;
/// Lowest mean label probability a detected address keeps. Address is recall-first: the parser
/// rejects spans it cannot decompose.
pub const DETECT_MIN_ADDRESS: f32 = 0.50;

/// The detection threshold for a model kind; rule kinds are never thresholded here.
pub(crate) fn detect_min(kind: Kind) -> f32 {
    match kind {
        Kind::Person => DETECT_MIN_PERSON,
        Kind::Org => DETECT_MIN_ORG,
        Kind::Address => DETECT_MIN_ADDRESS,
        Kind::Email | Kind::Phone => 0.0,
    }
}

pub(crate) const STAGE_DETECT: &str = "detect";
pub(crate) const STAGE_PARSE: &str = "parse";

/// `c` rounded to four decimals as an `f64`, so serialized output shows 0.99 rather than the
/// widened 0.9900000095367432. Rounding arithmetically instead of formatting and reparsing the
/// float keeps the float printing and parsing code out of the wasm build, about 24 KB gzip.
pub fn display_confidence(c: f32) -> f64 {
    (f64::from(c) * 10_000.0).round() / 10_000.0
}

/// Apply the bands: set `review_recommended`, drop low-confidence entities
/// unless `include_uncertain`. Order is preserved.
pub(crate) fn apply(entities: Vec<Entity>, include_uncertain: bool) -> Vec<Entity> {
    entities
        .into_iter()
        .filter_map(|mut e| {
            if e.confidence >= HIGH {
                e.review_recommended = false;
                Some(e)
            } else if e.confidence >= MEDIUM || include_uncertain {
                e.review_recommended = true;
                Some(e)
            } else {
                None
            }
        })
        .collect()
}

/// Components with any separator punctuation at their edges cut off, so a model that labels
/// the comma in `日本、神奈川県` as part of the region still returns `神奈川県`. A component
/// that is only punctuation is dropped.
pub(crate) fn trim_separators(text: &str, components: Vec<Component>) -> Vec<Component> {
    let separator = |c: char| c.is_whitespace() || ",;、，；。".contains(c);
    components
        .into_iter()
        .filter_map(|mut c| {
            let span = text.get(c.start..c.end)?;
            let head = span.len() - span.trim_start_matches(separator).len();
            let tail = span.len() - span.trim_end_matches(separator).len();
            if head + tail >= span.len() {
                return None;
            }
            c.start += head;
            c.end -= tail;
            Some(c)
        })
        .collect()
}

/// Suburbs that touch, with nothing between them, merged into one: a Japanese town and its
/// block (`栄` then `3丁目`) are one suburb, and the model can split them mid-word. Other
/// labels stay apart, since a subprefecture and a ward (`石狩振興局豊平`) are written the same
/// way. The merged confidence is the lower of the two.
pub(crate) fn merge_touching_suburbs(components: Vec<Component>) -> Vec<Component> {
    let mut out: Vec<Component> = Vec::with_capacity(components.len());
    for c in components {
        match out.last_mut() {
            Some(prev)
                if prev.label == AddressLabel::Suburb
                    && c.label == AddressLabel::Suburb
                    && prev.end == c.start =>
            {
                prev.end = c.end;
                prev.confidence = prev.confidence.min(c.confidence);
            }
            _ => out.push(c),
        }
    }
    out
}

/// Labels a well-formed address carries at most once.
const SINGLE_LABELS: [AddressLabel; 4] = [
    AddressLabel::HouseNumber,
    AddressLabel::Postcode,
    AddressLabel::Country,
    AddressLabel::City,
];

/// The `Unknown` rules for parsed components, in order: a component below `MEDIUM` becomes
/// `Unknown`, and of repeated single-occurrence labels only the most confident keeps its label.
/// Low-confidence `Unknown` components are then dropped unless `include_uncertain`. The span is
/// kept whenever the label is not, so a caller can still show the text without an asserted
/// label.
pub(crate) fn address_components(
    mut components: Vec<Component>,
    include_uncertain: bool,
) -> Vec<Component> {
    for c in &mut components {
        if c.confidence < MEDIUM {
            c.label = AddressLabel::Unknown;
        }
    }
    for label in SINGLE_LABELS {
        let best = components
            .iter()
            .enumerate()
            .filter(|(_, c)| c.label == label)
            .max_by(|a, b| a.1.confidence.total_cmp(&b.1.confidence))
            .map(|(i, _)| i);
        for (i, c) in components.iter_mut().enumerate() {
            if c.label == label && Some(i) != best {
                c.label = AddressLabel::Unknown;
            }
        }
    }
    // Every label that survived the first rule has confidence of at least MEDIUM, so this
    // only drops low-confidence `Unknown` components.
    components
        .into_iter()
        .filter(|c| include_uncertain || c.confidence >= MEDIUM)
        .collect()
}

/// An address is as confident as its weakest labelled component; with none, it is 0.
pub(crate) fn address_confidence(components: &[Component]) -> f32 {
    components
        .iter()
        .filter(|c| c.label != AddressLabel::Unknown)
        .map(|c| c.confidence)
        .reduce(f32::min)
        .unwrap_or(0.0)
}

/// Sets `review_recommended` on contacts and dissolves those below `MEDIUM` into `unassigned`
/// unless `include_uncertain` is set. Entities keep the flags the entity policy gave them.
pub(crate) fn apply_contacts(extraction: Extraction, include_uncertain: bool) -> Extraction {
    let mut unassigned = extraction.unassigned;
    let mut contacts = Vec::with_capacity(extraction.contacts.len());
    for mut contact in extraction.contacts {
        if contact.confidence < MEDIUM && !include_uncertain {
            unassigned.extend(contact.person.take());
            unassigned.extend(contact.org.take());
            unassigned.append(&mut contact.addresses);
            unassigned.append(&mut contact.emails);
            unassigned.append(&mut contact.phones);
            continue;
        }
        contact.review_recommended = contact.confidence < HIGH;
        contacts.push(contact);
    }
    unassigned.sort_by_key(|e| e.start);
    Extraction {
        contacts,
        unassigned,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Kind, Source};

    fn entity(confidence: f32) -> Entity {
        Entity {
            kind: Kind::Email,
            start: 0,
            end: 1,
            confidence,
            review_recommended: false,
            source: Source::Rules,
            components: Vec::new(),
            normalized: None,
            region: None,
        }
    }

    #[test]
    fn touching_suburbs_merge() {
        let out = merge_touching_suburbs(vec![
            component(AddressLabel::Suburb, 0, 3, 0.9),
            component(AddressLabel::Suburb, 3, 10, 0.7),
            component(AddressLabel::District, 10, 13, 0.9),
            component(AddressLabel::District, 13, 16, 0.9),
        ]);
        assert_eq!(
            out.iter()
                .map(|c| (c.label, c.start, c.end, c.confidence))
                .collect::<Vec<_>>(),
            vec![
                (AddressLabel::Suburb, 0, 10, 0.7),
                (AddressLabel::District, 10, 13, 0.9),
                (AddressLabel::District, 13, 16, 0.9),
            ]
        );
    }

    #[test]
    fn separators_are_trimmed_from_component_edges() {
        let text = "日本、神奈川県, 、";
        let out = trim_separators(
            text,
            vec![
                component(AddressLabel::Country, 0, 6, 0.9),
                component(AddressLabel::Region, 6, 22, 0.9),
                component(AddressLabel::Unknown, 22, text.len(), 0.9),
            ],
        );
        let texts: Vec<&str> = out.iter().map(|c| &text[c.start..c.end]).collect();
        assert_eq!(texts, ["日本", "神奈川県"]);
    }

    #[test]
    fn confidence_displays_as_written() {
        assert_eq!(display_confidence(0.99).to_string(), "0.99");
        assert_eq!(display_confidence(0.75).to_string(), "0.75");
        assert_eq!(display_confidence(0.123_456).to_string(), "0.1235");
    }

    #[test]
    fn bands() {
        let out = apply(vec![entity(0.9), entity(0.6), entity(0.2)], false);
        assert_eq!(
            out.iter()
                .map(|e| (e.confidence, e.review_recommended))
                .collect::<Vec<_>>(),
            vec![(0.9, false), (0.6, true)]
        );
        let out = apply(vec![entity(0.2)], true);
        assert_eq!(out.len(), 1);
        assert!(out[0].review_recommended);
        assert!(!apply(vec![entity(0.85)], false)[0].review_recommended);
    }

    fn component(label: AddressLabel, start: usize, end: usize, confidence: f32) -> Component {
        Component {
            label,
            start,
            end,
            confidence,
        }
    }

    #[test]
    fn unknown_rules() {
        let parsed = vec![
            component(AddressLabel::HouseNumber, 0, 2, 0.4),
            component(AddressLabel::Road, 3, 5, 0.9),
            component(AddressLabel::City, 6, 12, 0.7),
            component(AddressLabel::City, 13, 18, 0.95),
        ];
        let labels = |cs: &[Component]| cs.iter().map(|c| (c.label, c.start)).collect::<Vec<_>>();
        let strict = address_components(parsed.clone(), false);
        assert_eq!(
            labels(&strict),
            vec![
                (AddressLabel::Road, 3),
                (AddressLabel::Unknown, 6),
                (AddressLabel::City, 13),
            ]
        );
        let all = address_components(parsed, true);
        assert_eq!(all.len(), 4);
        assert_eq!(all[0].label, AddressLabel::Unknown);
    }

    fn at(kind: Kind, start: usize) -> Entity {
        Entity {
            kind,
            start,
            end: start + 4,
            confidence: 0.9,
            review_recommended: false,
            source: crate::Source::Model,
            components: Vec::new(),
            normalized: None,
            region: None,
        }
    }

    fn contact(confidence: f32) -> crate::Contact {
        crate::Contact {
            start: 0,
            end: 24,
            confidence,
            review_recommended: false,
            person: Some(at(Kind::Person, 0)),
            org: None,
            addresses: Vec::new(),
            emails: vec![at(Kind::Email, 20)],
            phones: Vec::new(),
        }
    }

    #[test]
    fn low_confidence_contact_is_dissolved() {
        let x = apply_contacts(
            Extraction {
                contacts: vec![contact(0.4)],
                unassigned: vec![at(Kind::Phone, 10)],
            },
            false,
        );
        assert!(x.contacts.is_empty());
        let got: Vec<(Kind, usize)> = x.unassigned.iter().map(|e| (e.kind, e.start)).collect();
        assert_eq!(
            got,
            vec![(Kind::Person, 0), (Kind::Phone, 10), (Kind::Email, 20)]
        );
        let kept = apply_contacts(
            Extraction {
                contacts: vec![contact(0.4)],
                unassigned: Vec::new(),
            },
            true,
        );
        assert!(kept.contacts[0].review_recommended);
    }

    #[test]
    fn medium_confidence_contact_is_flagged() {
        let x = apply_contacts(
            Extraction {
                contacts: vec![contact(0.7), contact(0.9)],
                unassigned: Vec::new(),
            },
            false,
        );
        let flags: Vec<bool> = x.contacts.iter().map(|c| c.review_recommended).collect();
        assert_eq!(flags, vec![true, false]);
    }
}
