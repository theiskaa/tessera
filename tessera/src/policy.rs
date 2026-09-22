//! Confidence bands. What the library is allowed to return is decided here and
//! nowhere else: high passes, medium passes flagged, low is dropped unless the
//! caller asks for uncertain results.

use crate::Entity;

pub(crate) const HIGH: f32 = 0.85;
pub(crate) const MEDIUM: f32 = 0.50;

// used from Milestone 4
#[allow(dead_code)]
pub(crate) const STAGE_TOKENIZE: &str = "tokenize";
// used from Milestone 4
#[allow(dead_code)]
pub(crate) const STAGE_RULES: &str = "rules";
pub(crate) const STAGE_DETECT: &str = "detect";
pub(crate) const STAGE_PARSE: &str = "parse";
pub(crate) const STAGE_GROUP: &str = "group";
// used from Milestone 4
#[allow(dead_code)]
pub(crate) const STAGE_CHUNK: &str = "chunk";

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
}
