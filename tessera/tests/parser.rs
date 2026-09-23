//! Fixture-driven tests for `fixtures/parser/`: hand-written addresses per country that
//! `parse_address` must split exactly. Cases marked `expected_failure` must keep failing, so a
//! fix is noticed and the mark removed.

mod common;

use tessera::{AddressLabel, Error, Query};

#[test]
fn parser_fixtures() {
    let tessera = common::load_tessera();
    let mut failures = Vec::new();
    for (file, fixture) in common::parser_fixtures() {
        for case in fixture.cases {
            let query = Query::default();
            let got = tessera.parse_address(&case.input, &query).unwrap();
            let got: Vec<(&str, usize, usize)> = got
                .components
                .iter()
                .filter(|c| c.label != AddressLabel::Unknown)
                .map(|c| (c.label.as_str(), c.start, c.end))
                .collect();
            let want: Vec<(&str, usize, usize)> = case
                .components
                .iter()
                .map(|c| (c.label.as_str(), c.start, c.end))
                .collect();
            match (got == want, &case.expected_failure) {
                (true, None) | (false, Some(_)) => {}
                (true, Some(reason)) => failures.push(format!(
                    "{file}: `{}` now passes; remove expected_failure ({reason})",
                    case.name
                )),
                (false, None) => {
                    let show = |v: &[(&str, usize, usize)]| {
                        v.iter()
                            .map(|(l, s, e)| format!("{l}={:?}", &case.input[*s..*e]))
                            .collect::<Vec<_>>()
                            .join(" ")
                    };
                    failures.push(format!(
                        "{file}: `{}`\n  want {}\n  got  {}",
                        case.name,
                        show(&want),
                        show(&got)
                    ));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn parse_address_limits() {
    let tessera = common::load_tessera();
    let q = Query::default();
    let empty = tessera.parse_address("", &q).unwrap();
    assert!(empty.components.is_empty() && empty.confidence == 0.0);
    let whitespace = tessera.parse_address("  \n ", &q).unwrap();
    assert!(whitespace.components.is_empty());
    let one_token = "a".repeat(8 * 1024);
    assert!(tessera.parse_address(&one_token, &q).is_ok());
    assert_eq!(
        tessera
            .parse_address(&"a".repeat(8 * 1024 + 1), &q)
            .unwrap_err(),
        Error::InputTooLarge
    );
    assert!(tessera.parse_address(&"1 ".repeat(256), &q).is_ok());
    assert_eq!(
        tessera.parse_address(&"1 ".repeat(257), &q).unwrap_err(),
        Error::InputTooLarge
    );
    let many_long_words = format!("{} ", "a".repeat(39)).repeat(200);
    assert_eq!(many_long_words.len(), 8000);
    assert!(tessera.parse_address(&many_long_words, &q).is_ok());
}

#[test]
fn parse_address_needs_the_parser() {
    let rules_only = tessera::Tessera::load(
        &[],
        tessera::Config {
            kinds: tessera::Kind::Email | tessera::Kind::Phone,
            expected_checksum: None,
        },
    )
    .unwrap();
    assert!(matches!(
        rules_only.parse_address("10 Downing Street", &Query::default()),
        Err(Error::Inference { .. })
    ));
}

#[test]
fn uncertain_components_follow_the_bands() {
    let tessera = common::load_tessera();
    let texts = [
        "10 Downing Street, London SW1A 2AA",
        "Near the svan tower",
        "Kleintier-Ambulanz-Rheingönheim, Ludwigshafen am Rhein",
        "zq vx 17 kk",
    ];
    for text in texts {
        let sure = tessera.parse_address(text, &Query::default()).unwrap();
        let all = tessera
            .parse_address(
                text,
                &Query {
                    include_uncertain: true,
                    ..Query::default()
                },
            )
            .unwrap();
        assert!(
            sure.components.iter().all(|c| c.confidence >= 0.5),
            "{text}"
        );
        assert!(
            sure.components.iter().all(|c| all.components.contains(c)),
            "{text}"
        );
        assert!(
            all.components
                .iter()
                .filter(|c| c.confidence < 0.5)
                .all(|c| c.label == AddressLabel::Unknown),
            "{text}"
        );
        let weakest = sure
            .components
            .iter()
            .filter(|c| c.label != AddressLabel::Unknown)
            .map(|c| c.confidence)
            .reduce(f32::min)
            .unwrap_or(0.0);
        assert_eq!(sure.confidence, weakest, "{text}");
        assert_eq!(sure.confidence, all.confidence, "{text}");
        assert_eq!(sure.review_recommended, sure.confidence < 0.85, "{text}");
    }
    let clean = tessera
        .parse_address("10 Downing Street, London SW1A 2AA", &Query::default())
        .unwrap();
    assert!(!clean.review_recommended && clean.confidence >= 0.85);
}
