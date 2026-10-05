//! Explicit numeric equivalents of displayed US vanity telephone numbers.

use super::{Region, labelled_as_another_number, region_id, validate_dial};
use crate::{Entity, Kind, Source};

fn preceded_by_number(text: &str, start: usize) -> bool {
    text[..start]
        .trim_end_matches(super::horizontal_phone_space)
        .ends_with(|c: char| c.is_numeric() || c == '+')
}

fn numeric_continuation(tail: &str) -> bool {
    let tail = tail.trim_start_matches(super::horizontal_phone_space);
    let Some(separator) = tail
        .chars()
        .next()
        .filter(|c| super::DASHES.contains(c) || matches!(c, '/' | '.' | ',' | '\u{2011}'))
    else {
        return false;
    };
    tail[separator.len_utf8()..]
        .trim_start_matches(super::horizontal_phone_space)
        .starts_with(char::is_numeric)
}

fn keypad(letter: u8) -> Option<u8> {
    Some(match letter.to_ascii_uppercase() {
        b'A'..=b'C' => b'2',
        b'D'..=b'F' => b'3',
        b'G'..=b'I' => b'4',
        b'J'..=b'L' => b'5',
        b'M'..=b'O' => b'6',
        b'P'..=b'S' => b'7',
        b'T'..=b'V' => b'8',
        b'W'..=b'Z' => b'9',
        _ => return None,
    })
}

/// Read a displayed vanity number only when its printed numeric equivalent agrees.
pub(super) fn scan(text: &str, chars: &[(usize, char)], hints: &[&'static Region]) -> Vec<Entity> {
    if !hints.iter().any(|r| region_id(r) == Some("US")) {
        return Vec::new();
    }
    let bytes = text.as_bytes();
    let mut found = Vec::new();
    for (i, &(start, ch)) in chars.iter().enumerate() {
        if ch != '(' || labelled_as_another_number(chars, i) {
            continue;
        }
        let Some(display) = bytes.get(start..start + 21) else {
            continue;
        };
        if display[0] != b'('
            || display[4..6] != *b") "
            || display[9] != b'-'
            || display[14..16] != *b" ("
            || display[20] != b')'
            || !display[1..4].iter().all(u8::is_ascii_digit)
            || !display[6..9].iter().all(u8::is_ascii_digit)
            || !display[10..14].iter().all(u8::is_ascii_alphabetic)
            || !display[16..20].iter().all(u8::is_ascii_digit)
            || !display[10..14]
                .iter()
                .zip(&display[16..20])
                .all(|(&letter, &digit)| keypad(letter) == Some(digit))
        {
            continue;
        }
        if text[..start].ends_with(|c: char| c.is_alphanumeric() || c == '+')
            || text[start + 21..].starts_with(|c: char| c.is_alphanumeric())
            || preceded_by_number(text, start)
            || numeric_continuation(&text[start + 21..])
        {
            continue;
        }
        let dial: String = display[1..4]
            .iter()
            .chain(&display[6..9])
            .chain(&display[16..20])
            .map(|&b| char::from(b))
            .collect();
        let Some((confidence, normalized, region)) =
            validate_dial(&dial, false, false, false, hints)
        else {
            continue;
        };
        if region.as_deref() != Some("US") {
            continue;
        }
        let mut entity = Entity {
            kind: Kind::Phone,
            start,
            end: start + 21,
            confidence,
            review_recommended: false,
            source: Source::Rules,
            components: Vec::new(),
            normalized: Some(normalized),
            region,
        };
        super::expand_us_phone_span(text, &mut entity);
        found.push(entity);
    }
    found
}

#[cfg(all(test, feature = "phone-metadata"))]
mod tests {
    use super::super::scan;

    #[test]
    fn complete_displayed_equivalent_keeps_literal_utf8_boundaries() {
        let text = "☎ P: (360) 307-VOTE (8683)\nF: 360-337-5769";
        let found = scan(text, &["US"]);
        let spans: Vec<_> = found.iter().map(|e| &text[e.start..e.end]).collect();
        assert_eq!(spans, ["(360) 307-VOTE (8683)", "360-337-5769"]);
        assert_eq!(found[0].normalized.as_deref(), Some("+13603078683"));
        assert_eq!(found[0].region.as_deref(), Some("US"));
    }

    #[test]
    fn vanity_alias_requires_matching_equivalent_region_and_non_identifier_context() {
        for text in [
            "P: (360) 307-VOTE (8684)",
            "P: (360) 307-VOTE",
            "P: (100) 307-VOTE (8683)",
            "SKU: (360) 307-VOTE (8683)",
            "account: (360) 307-VOTE (8683)",
            "id(360) 307-VOTE (8683)",
            "P: (360) 307-VOTE (8683)99",
            "P: (360) 307-VOTE (8683)-12",
        ] {
            assert!(scan(text, &["US"]).is_empty(), "{text}");
        }
        for hints in [&[][..], &["GB"][..], &["CA"][..]] {
            assert!(scan("P: (360) 307-VOTE (8683)", hints).is_empty());
        }
    }

    #[test]
    fn vanity_extensions_use_the_existing_displayed_extension_boundary() {
        for tail in [" ext. 123", " ext.123", ", ext. 123"] {
            let text = format!("Phone: (202) 555-HELP (4357){tail}");
            let found = scan(&text, &["US"]);
            assert_eq!(found.len(), 1);
            assert_eq!(&text[found[0].start..found[0].end], &text[7..]);
            assert_eq!(found[0].normalized.as_deref(), Some("+12025554357"));
        }
        let text = "Phone: (202) 555-HELP (4357)\nExt. 123";
        let found = scan(text, &["US"]);
        assert_eq!(&text[found[0].start..found[0].end], "(202) 555-HELP (4357)");
        let text = "Phone: (202) 555-HELP (4357)- assistance";
        assert_eq!(scan(text, &["US"]).len(), 1);
    }

    #[test]
    fn country_code_and_unicode_numeric_continuations_cannot_be_discarded() {
        for prefix in ["+44 ", "+1 ", "+44\u{A0}", "+1\t", "44 ", "1 "] {
            let text = format!("Phone: {prefix}(202) 555-HELP (4357)");
            assert!(scan(&text, &["US"]).is_empty(), "{text}");
        }
        for dash in super::super::DASHES
            .iter()
            .copied()
            .chain(['\u{2011}', '/', '.', ','])
        {
            for gap in ["", " ", "\u{A0}", "\t", "\u{2009}", "    "] {
                for digits in ["12", "١٢", "１２"] {
                    let text = format!("Phone: (202) 555-HELP (4357){dash}{gap}{digits}");
                    assert!(scan(&text, &["US"]).is_empty(), "{text}");
                }
            }
        }
    }

    #[test]
    fn matched_other_vanity_word_and_case_are_not_specific_to_vote() {
        let text = "Phone: (202) 555-HELP (4357)";
        let found = scan(text, &["US"]);
        assert_eq!(found.len(), 1);
        assert_eq!(&text[found[0].start..found[0].end], "(202) 555-HELP (4357)");
        let lower = text.replace("HELP", "help");
        assert_eq!(scan(&lower, &["US"]).len(), 1);
    }
}
