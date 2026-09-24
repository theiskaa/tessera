//! Stage 1: deterministic scanners for the kinds with a strict grammar.
//! Their spans are later passed to the detector as features and masked from
//! its output, so the model never labels an email or a phone.

pub(crate) mod email;
pub(crate) mod phone;
#[cfg(feature = "phone-metadata")]
mod phone_tables;

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

/// Email and phone entities from `mailto:` and `tel:` link destinations: each spans the
/// visible link text, and its `normalized` value and region come from the destination, read by
/// the same scanners as running text. A `mailto:` with several addresses yields the first only,
/// because the others have no span in the document. Percent escapes other than `%40`, `%2B`,
/// and `%20` fail validation.
#[cfg(feature = "markdown")]
#[cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "Format::Markdown in phase 6.4 is the first caller"
    )
)]
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

/// Scanned entities plus link entities, sorted by start. A link entity wins over any scanned
/// entity it overlaps: the destination is where a click goes.
pub(crate) fn merge_links(mut scanned: Vec<Entity>, linked: Vec<Entity>) -> Vec<Entity> {
    scanned.retain(|e| !linked.iter().any(|l| e.start < l.end && l.start < e.end));
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
