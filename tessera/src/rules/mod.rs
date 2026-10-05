//! Stage 1: deterministic scanners for the kinds with a strict grammar.
//! Their spans are later passed to the detector as features and masked from
//! its output, so the model never labels an email or a phone.

pub(crate) mod email;
pub(crate) mod phone;
#[cfg(feature = "phone-metadata")]
mod phone_tables;
pub(crate) mod region;

use crate::Entity;
#[cfg(feature = "markdown")]
use crate::Source;
#[cfg(feature = "markdown")]
use crate::markdown::LinkTarget;

/// Run every scanner over `text` and return their entities sorted by start,
/// non-overlapping. Emails win over phones when spans overlap.
pub fn scan(text: &str, country_hint: &[&str]) -> Vec<Entity> {
    let emails = email::scan(text);
    let phones = phone::scan(text, country_hint);
    let mut out: Vec<Entity> = phones
        .into_iter()
        .filter(|p| !emails.iter().any(|e| p.start < e.end && e.start < p.end))
        .collect();
    out.extend(emails);
    out.sort_by_key(|e| (e.start, e.end));
    out
}

/// Run the rules with the same hint selection as public plain-text detection.
/// An empty hint invokes document region inference; it is not a literal empty-hint scan.
pub fn scan_text_rules(text: &str, country_hint: &[&str]) -> Vec<Entity> {
    scan_text_rules_with_regions(text, country_hint).1
}

pub(crate) fn scan_text_rules_with_regions<'a>(
    text: &str,
    country_hint: &[&'a str],
) -> (Vec<&'a str>, Vec<Entity>) {
    let effective = if country_hint.is_empty() {
        region::infer(text)
    } else {
        country_hint.to_vec()
    };
    let entities = scan(text, &effective);
    (effective, entities)
}

/// Email and phone entities from `mailto:` and `tel:` link destinations: each spans the
/// visible link text, and its `normalized` value and region come from the destination, read by
/// the same scanners as running text. A `mailto:` with several addresses yields the first only,
/// because the others have no span in the document. Percent escapes other than `%40`, `%2B`,
/// and `%20` fail validation.
#[cfg(feature = "markdown")]
pub(crate) fn from_links(links: &[LinkTarget], country_hint: &[&str]) -> Vec<Entity> {
    let mut out = Vec::new();
    for link in links {
        if link.text_start >= link.text_end {
            continue;
        }
        let found = if let Some(rest) = strip_scheme(&link.dest, "mailto:") {
            let address = rest.split(['?', ',']).next().unwrap_or("");
            email::whole(&decode_percent(address))
        } else if let Some(rest) = strip_scheme(&link.dest, "tel:") {
            let number: String = decode_percent(rest.split(';').next().unwrap_or(""))
                .chars()
                .filter(|c| !matches!(c, '-' | '.' | '(' | ')' | ' '))
                .collect();
            phone::whole(&number, country_hint)
        } else {
            None
        };
        out.extend(found.map(|e| Entity {
            start: link.text_start,
            end: link.text_end,
            confidence: e.confidence.min(0.99),
            source: Source::Rules,
            ..e
        }));
    }
    out
}

/// Scanned entities plus link entities, sorted by start and non-overlapping. A link entity wins
/// over any scanned entity it overlaps, since the destination is where a click goes, and of two
/// overlapping link entities the innermost wins.
#[cfg(feature = "markdown")]
pub(crate) fn merge_links(mut scanned: Vec<Entity>, mut linked: Vec<Entity>) -> Vec<Entity> {
    if linked.is_empty() {
        return scanned;
    }
    // pulldown-cmark nests an autolink inside a link's text, so link entities can overlap.
    // The innermost is kept: it is the more specific destination. What remains is disjoint,
    // so sorted by start the ends ascend too, and the only link an entity can overlap first
    // is the first one ending after the entity starts.
    linked.sort_by_key(|l| (l.end - l.start, l.start));
    let mut kept: Vec<Entity> = Vec::with_capacity(linked.len());
    for l in linked {
        if !kept.iter().any(|k| l.start < k.end && k.start < l.end) {
            kept.push(l);
        }
    }
    let mut linked = kept;
    linked.sort_by_key(|l| l.start);
    scanned.retain(|e| {
        let i = linked.partition_point(|l| l.end <= e.start);
        linked.get(i).is_none_or(|l| l.start >= e.end)
    });
    scanned.extend(linked);
    scanned.sort_by_key(|e| (e.start, e.end));
    scanned
}

#[cfg(feature = "markdown")]
fn strip_scheme<'a>(dest: &'a str, scheme: &str) -> Option<&'a str> {
    let head = dest.get(..scheme.len())?;
    head.eq_ignore_ascii_case(scheme)
        .then(|| dest.get(scheme.len()..))
        .flatten()
}

#[cfg(feature = "markdown")]
fn decode_percent(s: &str) -> String {
    s.replace("%40", "@")
        .replace("%2B", "+")
        .replace("%2b", "+")
        .replace("%20", "")
}

/// Drops rule entities the mask does not fully contain; plain text has no mask.
#[cfg(feature = "markdown")]
pub(crate) fn retain_in_mask(entities: &mut Vec<Entity>, mask: Option<&crate::chunk::Mask>) {
    if let Some(mask) = mask {
        entities.retain(|e| mask.contains(e.start, e.end));
    }
}

#[cfg(all(test, feature = "markdown"))]
mod link_tests {
    use super::*;
    use crate::Kind;

    fn link(dest: &str, text_start: usize, text_end: usize) -> LinkTarget {
        LinkTarget {
            dest: dest.to_string(),
            start: text_start.saturating_sub(1),
            end: text_end + 20,
            text_start,
            text_end,
        }
    }

    fn email_at(start: usize, end: usize, normalized: &str) -> Entity {
        Entity {
            kind: Kind::Email,
            start,
            end,
            confidence: 0.99,
            review_recommended: false,
            source: Source::Rules,
            components: Vec::new(),
            normalized: Some(normalized.into()),
            region: None,
        }
    }

    #[test]
    fn mailto_yields_email_over_link_text() {
        let out = from_links(
            &[link(
                "mailto:nino@kavkaz-freight.example?subject=Hi",
                10,
                14,
            )],
            &[],
        );
        assert_eq!(out.len(), 1);
        assert_eq!(
            (out[0].kind, out[0].start, out[0].end),
            (Kind::Email, 10, 14)
        );
        assert_eq!(
            out[0].normalized.as_deref(),
            Some("nino@kavkaz-freight.example")
        );
        assert_eq!(out[0].source, Source::Rules);
    }

    #[cfg(feature = "phone-metadata")]
    #[test]
    fn tel_yields_phone_with_region() {
        let out = from_links(&[link("tel:+44-20-7946-0958;ext=12", 72, 82)], &["GB"]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].kind, Kind::Phone);
        assert_eq!(out[0].normalized.as_deref(), Some("+442079460958"));
        assert_eq!(out[0].region.as_deref(), Some("GB"));
    }

    #[cfg(feature = "phone-metadata")]
    #[test]
    fn tel_without_plus_uses_country_hint() {
        let out = from_links(&[link("tel:020%207946%200958", 0, 3)], &["GB"]);
        assert_eq!(out[0].normalized.as_deref(), Some("+442079460958"));
    }

    #[test]
    fn other_schemes_and_invalid_targets_yield_nothing() {
        let links = [
            link("https://kavkaz-freight.example", 0, 3),
            link("mailto:not-an-email", 0, 3),
            link("tel:12", 0, 3),
            link("MAILTO:", 0, 3),
        ];
        assert!(from_links(&links, &["GB"]).is_empty());
    }

    #[test]
    fn scheme_is_case_insensitive() {
        let out = from_links(&[link("MailTo:ops@kavkaz-freight.example", 0, 3)], &[]);
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn link_entity_wins_over_overlapping_scanned_entity() {
        let scanned = vec![email_at(238, 264, "old@kavkaz-freight.example")];
        let linked = from_links(&[link("mailto:new@kavkaz-freight.example", 238, 264)], &[]);
        let merged = merge_links(scanned, linked);
        assert_eq!(merged.len(), 1);
        assert_eq!(
            merged[0].normalized.as_deref(),
            Some("new@kavkaz-freight.example")
        );
    }

    #[test]
    fn only_scanned_entities_overlapping_a_link_are_dropped() {
        let scanned = vec![
            email_at(0, 5, "a@b.example"),
            email_at(8, 12, "c@d.example"),
            email_at(20, 30, "e@f.example"),
            email_at(40, 45, "g@h.example"),
        ];
        let linked = from_links(
            &[
                link("mailto:x@y.example", 25, 35),
                link("mailto:z@y.example", 5, 9),
            ],
            &[],
        );
        let merged = merge_links(scanned, linked);
        assert_eq!(
            merged.iter().map(|e| (e.start, e.end)).collect::<Vec<_>>(),
            [(0, 5), (5, 9), (25, 35), (40, 45)]
        );
    }

    #[test]
    fn a_link_nested_in_a_link_keeps_the_inner_one() {
        let scanned = vec![
            email_at(11, 22, "x@y.example"),
            email_at(24, 37, "ops@z.example"),
        ];
        let linked = from_links(
            &[
                link("mailto:x@y.example", 4, 22),
                link("mailto:w@z.example", 1, 37),
            ],
            &[],
        );
        let merged = merge_links(scanned, linked);
        assert_eq!(
            merged
                .iter()
                .map(|e| (e.start, e.end, e.normalized.as_deref()))
                .collect::<Vec<_>>(),
            [
                (4, 22, Some("x@y.example")),
                (24, 37, Some("ops@z.example"))
            ]
        );
    }

    #[test]
    fn merged_output_is_sorted() {
        let scanned = vec![email_at(300, 320, "a@b.example")];
        let linked = from_links(&[link("mailto:c@d.example", 10, 14)], &[]);
        let merged = merge_links(scanned, linked);
        assert_eq!(
            merged.iter().map(|e| e.start).collect::<Vec<_>>(),
            [10, 300]
        );
    }
}

#[cfg(all(test, feature = "phone-metadata"))]
mod text_input_policy_tests {
    use super::*;
    use crate::{Config, Format, Kind, KindSet, Query, Tessera};

    type Record = (
        Kind,
        usize,
        usize,
        Option<String>,
        Option<String>,
        u32,
        bool,
    );

    fn records(entities: &[Entity]) -> Vec<Record> {
        entities
            .iter()
            .map(|e| {
                (
                    e.kind,
                    e.start,
                    e.end,
                    e.normalized.clone(),
                    e.region.clone(),
                    e.confidence.to_bits(),
                    e.review_recommended,
                )
            })
            .collect()
    }

    fn instance(kinds: KindSet) -> Tessera {
        let result = Tessera::load(
            &[],
            Config {
                kinds,
                expected_checksum: None,
            },
        )
        .unwrap();
        assert!(result.model_version().is_none());
        result
    }

    #[test]
    fn public_text_explicit_and_auto_rules_keep_unicode_offsets_and_regions() {
        let text = "é ☎ +1 202 555 0199; (202) 555-0199; a@example.test";
        let engine = instance(Kind::Email | Kind::Phone);
        for hints in [&["US"][..], &[][..]] {
            let effective = if hints.is_empty() {
                region::infer(text)
            } else {
                hints.to_vec()
            };
            let old_rules = scan(text, &effective);
            let selected = scan_text_rules(text, hints);
            let actual = engine
                .detect(
                    text,
                    &Query {
                        country_hint: hints,
                        include_uncertain: true,
                        format: Format::Text,
                    },
                )
                .unwrap();
            assert_eq!(records(&selected), records(&old_rules));
            assert_eq!(records(&actual), records(&old_rules));
            assert_eq!(
                actual
                    .iter()
                    .filter(|e| e.kind == Kind::Phone)
                    .map(|e| e.text(text))
                    .collect::<Vec<_>>(),
                ["+1 202 555 0199", "(202) 555-0199"]
            );
            for entity in actual {
                assert!(text.get(entity.start..entity.end).is_some());
                if entity.kind == Kind::Phone {
                    assert_eq!(entity.region.as_deref(), Some("US"));
                }
            }
        }
    }

    #[test]
    fn requested_output_kinds_do_not_remove_rules_from_input_features() {
        use crate::features::{FeatureConfig, featurize, flag};
        let text = "Desk (202) 555-0199 and a@example.test";
        let full = scan_text_rules(text, &["US"]);
        let tokens = crate::token::tokenize(text);
        let spans: Vec<_> = full.iter().map(|e| (e.start, e.end)).collect();
        let fc = FeatureConfig {
            ngram_sizes: vec![2, 3, 4],
            hash_buckets: 32768,
            hash_seed: 0,
        };
        let features = featurize(text, &tokens, &spans, None, &fc, None);
        assert_eq!(full.len(), 2);
        for selected in [Kind::Email, Kind::Phone] {
            let found = instance(selected.into())
                .detect(
                    text,
                    &Query {
                        country_hint: &["US"],
                        include_uncertain: true,
                        format: Format::Text,
                    },
                )
                .unwrap();
            assert_eq!(found.len(), 1);
            assert_eq!(found[0].kind, selected);
            assert_eq!(records(&full), records(&scan_text_rules(text, &["US"])));
            for rule in &full {
                assert!(
                    tokens
                        .iter()
                        .zip(&features)
                        .filter(|(t, _)| rule.start < t.end && t.start < rule.end)
                        .all(|(_, f)| f.flags & flag::IN_RULE_SPAN != 0)
                );
            }
        }
    }

    #[cfg(feature = "markdown")]
    #[test]
    fn markdown_keeps_visible_region_and_link_mask_while_hidden_phone_stays_hidden() {
        let text =
            "Germany\n030 23125 480\n\n```text\n+44 20 7946 0958\n```\n\n[desk](tel:+493023125480)";
        let found = instance(Kind::Email | Kind::Phone)
            .detect(
                text,
                &Query {
                    country_hint: &[],
                    include_uncertain: true,
                    format: Format::Markdown(crate::MarkdownOptions::default()),
                },
            )
            .unwrap();
        assert_eq!(
            found
                .iter()
                .filter(|e| e.kind == Kind::Phone)
                .map(|e| e.text(text))
                .collect::<Vec<_>>(),
            ["030 23125 480", "desk"]
        );
        assert!(found.iter().all(|e| e.region.as_deref() == Some("DE")));
        assert_eq!(
            found.last().unwrap().normalized.as_deref(),
            Some("+493023125480")
        );
        assert!(!found.iter().any(|e| e.text(text).contains("+44")));
    }
}
