//! Generated delivery details attached to an existing complete postal address.

use rand::Rng;
use rand_chacha::ChaCha8Rng;

#[path = "address_delivery.rs"]
mod delivery;

pub(crate) use delivery::street_boundary;

const STYLES: usize = 18;
const INFIX_STYLES: usize = 9;

fn letters(rng: &mut ChaCha8Rng) -> String {
    (0..3)
        .map(|_| char::from(rng.random_range(b'A'..=b'Z')))
        .collect()
}

fn ordinal(number: u32) -> String {
    let suffix = match (number % 100, number % 10) {
        (11..=13, _) => "th",
        (_, 1) => "st",
        (_, 2) => "nd",
        (_, 3) => "rd",
        _ => "th",
    };
    format!("{number}{suffix}")
}

fn prefix(style: usize, rng: &mut ChaCha8Rng) -> String {
    let room = rng.random_range(100..1000);
    let code = rng.random_range(1000..10000);
    let floor = rng.random_range(1..20);
    match style {
        0 => format!("Room {room}, "),
        1 => format!("Room {room}\n"),
        2 => format!("Mail Code {code}, "),
        3 => format!("Mail Stop {code}\n"),
        4 => format!("North Office Building, Room {room}, "),
        5 => format!("Administrative Building\nRoom {room}\n"),
        6 => format!("Suite {room}, "),
        7 => format!("MS: {code}, "),
        8 => format!("Floor {floor}, Room {room}, "),
        9 => format!("West Building, Room W12-{room}, "),
        10 => format!("MS: {}, ", letters(rng)),
        11 => format!("MS: {}/{floor}W, ", letters(rng)),
        12 => format!("MS: {} ({}/{floor}W), ", letters(rng), letters(rng)),
        13 => format!("Room C{floor}-{:02}-{:02}, ", room / 10, room % 100),
        14 => format!("Room {floor}C{room}/{floor}C{}, ", room + 1),
        15 => format!("Central Building Ground Floor, Room E{floor}-{room}, "),
        16 => format!("Room {room} of the Administrative Building, "),
        _ => format!(
            "Operations Center, {} Floor Conference Room, ",
            ordinal(floor)
        ),
    }
}

/// Adds generated room, building, floor, or mail-stop details without changing the source address.
pub(crate) fn prepend(address: &str, rng: &mut ChaCha8Rng) -> String {
    let style = rng.random_range(0..STYLES);
    format!("{}{address}", prefix(style, rng))
}

fn infix(style: usize, rng: &mut ChaCha8Rng) -> String {
    let room = rng.random_range(100..1000);
    let code = rng.random_range(1000..10000);
    let floor = rng.random_range(1..20);
    match style {
        0 => format!("Room {room}"),
        1 => format!("Suite {room}"),
        2 => format!("Mail Stop {code}"),
        3 => format!("Mail Code {code}"),
        4 => format!("Room {}-{room}", letters(rng)),
        5 => format!("Suite C{floor}-{room}"),
        6 => format!("(Room {room})"),
        7 => format!("(Mail Stop {}-{code})", letters(rng)),
        _ => format!("MS: {} ({}/{floor}W)", letters(rng), letters(rng)),
    }
}

/// Samples delivery formats, using a native street boundary for after-street variants.
pub(crate) fn augment(address: &str, boundary: Option<usize>, rng: &mut ChaCha8Rng) -> String {
    let Some(boundary) = boundary else {
        return prepend(address, rng);
    };
    let style = rng.random_range(0..STYLES + INFIX_STYLES);
    if style < STYLES {
        return format!("{}{address}", prefix(style, rng));
    }
    let detail = infix(style - STYLES, rng);
    delivery::insert(address, boundary, &detail).unwrap_or_else(|| prepend(address, rng))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::{DETECTOR_LABELS, KindSpan, breaks_of, encode_document};
    use rand::SeedableRng;
    use tessera::internal::{FeatureConfig, decode_detector};

    #[test]
    fn prefixes_cover_coded_delivery_details() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mail = prefix(10, &mut rng);
        assert!(mail[4..7].bytes().all(|byte| byte.is_ascii_uppercase()));
        assert!(prefix(11, &mut rng).contains('/'));
        let parenthesized = prefix(12, &mut rng);
        assert!(parenthesized.contains('(') && parenthesized.contains("W), "));
        assert_eq!(prefix(13, &mut rng).matches('-').count(), 2);
        assert!(prefix(14, &mut rng).contains('/'));
        assert!(prefix(15, &mut rng).contains("Building Ground Floor"));
        assert!(prefix(16, &mut rng).contains("of the Administrative Building"));
        assert!(prefix(17, &mut rng).contains("Floor Conference Room"));
    }

    #[test]
    fn ordinal_floors_handle_teens() {
        for (number, expected) in [
            (1, "1st"),
            (2, "2nd"),
            (3, "3rd"),
            (11, "11th"),
            (12, "12th"),
            (13, "13th"),
            (21, "21st"),
        ] {
            assert_eq!(ordinal(number), expected);
        }
    }

    #[test]
    fn delivery_details_preserve_unicode_and_multiline_source() {
        let source = "10 Élan Street\nNew Haven, CT 06510";
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        for _ in 0..100 {
            let address = prepend(source, &mut rng);
            assert!(address.ends_with(source));
            assert!(address.len() > source.len());
        }
    }

    #[test]
    fn every_delivery_format_encodes_and_decodes_as_one_complete_address() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        for style in 0..STYLES {
            for source in [
                "10 Élan Street, New Haven, CT 06510",
                "10 Élan Street\nNew Haven, CT 06510",
            ] {
                let address = format!("{}{source}", prefix(style, &mut rng));
                let text = format!("Mail to {address}. Questions welcome.");
                let gold = KindSpan {
                    kind: 2,
                    start: 8,
                    end: (8 + address.len()) as u32,
                };
                let enc = encode_document(&text, &[gold], &FeatureConfig::default())
                    .unwrap_or_else(|error| panic!("style {style}: {error}; {text}"));
                let mut probs = vec![0.0; enc.labels.len() * DETECTOR_LABELS];
                for (index, &label) in enc.labels.iter().enumerate() {
                    probs[index * DETECTOR_LABELS + usize::from(label)] = 1.0;
                }
                let decoded =
                    decode_detector(&probs, &vec![false; enc.labels.len()], &breaks_of(&text));
                assert_eq!(decoded.len(), 1, "style {style}: {text}");
                assert_eq!(enc.token_spans[decoded[0].first].0, gold.start);
                assert_eq!(enc.token_spans[decoded[0].last].1, gold.end);
            }
        }
    }

    #[test]
    fn every_after_street_format_retains_and_labels_the_complete_address() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let street = "10 Élan Street";
        for style in 0..INFIX_STYLES {
            for separator in [", ", "\n", "\r\n"] {
                let source = format!("{street}{separator}New Haven, CT 06510");
                let detail = infix(style, &mut rng);
                let address = delivery::insert(&source, street.len(), &detail).unwrap();
                assert_eq!(address.replace(&format!("{separator}{detail}"), ""), source);
                assert!(address.ends_with("CT 06510"));
                let text = format!("Café contact: {address}. Questions welcome.");
                let gold = KindSpan {
                    kind: 2,
                    start: "Café contact: ".len() as u32,
                    end: ("Café contact: ".len() + address.len()) as u32,
                };
                let enc = encode_document(&text, &[gold], &FeatureConfig::default()).unwrap();
                let mut probs = vec![0.0; enc.labels.len() * DETECTOR_LABELS];
                for (index, &label) in enc.labels.iter().enumerate() {
                    probs[index * DETECTOR_LABELS + usize::from(label)] = 1.0;
                }
                let decoded =
                    decode_detector(&probs, &vec![false; enc.labels.len()], &breaks_of(&text));
                assert_eq!(decoded.len(), 1, "style {style}: {text}");
                assert_eq!(enc.token_spans[decoded[0].first].0, gold.start);
                assert_eq!(enc.token_spans[decoded[0].last].1, gold.end);
            }
        }
    }

    #[test]
    fn unavailable_boundary_preserves_the_original_prefix_sampler() {
        let address = "10 Élan Street, New Haven, CT 06510";
        let mut original_rng = ChaCha8Rng::seed_from_u64(81);
        let mut fallback_rng = original_rng.clone();
        for _ in 0..100 {
            assert_eq!(
                augment(address, None, &mut fallback_rng),
                prepend(address, &mut original_rng)
            );
        }
        assert!(delivery::insert(address, 4, "Room 200").is_none());
    }
}
