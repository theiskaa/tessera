//! Pragmatic email scanner. Accepts the addresses people write in signatures
//! and forms, not full RFC 5322: no quoted local parts, comments, IP literals,
//! or non-ASCII. The normalized form lowercases the domain only.

use crate::{Entity, Kind, Source};

const MAX_LOCAL: usize = 64;
const MAX_LABEL: usize = 63;
const MAX_DOMAIN: usize = 253;
const TRAILING: &[u8] = b".,;:)]>'\"-";

fn is_local(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b"!#$%&'*+/=?^_`{|}~-.".contains(&b)
}

/// Characters that are legal in a local part but in running text mark where an address
/// begins: Markdown code and emphasis, braces, and `key=value` pairs. An apostrophe is only
/// stripped when it opens the local part, since `o'brien@` is a real address.
const LOCAL_DELIMITERS: &[u8] = b"`*{}|=";

fn is_domain(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'-' || b == b'.'
}

/// Every email address in `text`, sorted by start, non-overlapping.
pub fn scan(text: &str) -> Vec<Entity> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    // Positions of the last whitespace and the last `://` seen so far. `@` positions only
    // grow, so one forward pass answers "is this inside a URL authority" in linear time.
    let mut cursor = 0;
    let mut last_space: Option<usize> = None;
    let mut last_scheme: Option<usize> = None;
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'@' {
            i += 1;
            continue;
        }
        let at = i;
        // Never scan left into the previous match, so results stay non-overlapping.
        let floor = out.last().map_or(0, |e: &Entity| e.end);
        let mut start = at;
        while start > floor && is_local(bytes[start - 1]) {
            start -= 1;
        }
        if let Some(d) = bytes[start..at]
            .iter()
            .rposition(|b| LOCAL_DELIMITERS.contains(b))
        {
            start += d + 1;
        }
        while start < at && bytes[start] == b'\'' {
            start += 1;
        }
        while cursor < at {
            if bytes[cursor].is_ascii_whitespace() {
                last_space = Some(cursor);
            } else if bytes[cursor..].starts_with(b"://") {
                last_scheme = Some(cursor);
            }
            cursor += 1;
        }
        let mut end = at + 1;
        while end < bytes.len() && is_domain(bytes[end]) {
            end += 1;
        }
        while end > at + 1 && TRAILING.contains(&bytes[end - 1]) {
            end -= 1;
        }
        i = end.max(at + 1);
        let Some(confidence) = validate(&bytes[start..at], &bytes[at + 1..end]) else {
            continue;
        };
        // A slash before or at the start of the local part means a URL path or an escape, not an address.
        if bytes[start] == b'/' || (start > 0 && matches!(bytes[start - 1], b'/' | b'\\')) {
            continue;
        }
        // `user:password@host` inside a URL; reporting it would also leak the password.
        // No whitespace or `://` can sit inside the local part, so the state at `at` holds.
        let in_url_authority = last_scheme.is_some_and(|s| last_space.is_none_or(|w| s > w));
        if in_url_authority {
            continue;
        }
        // Local-part and domain bytes are ASCII, so start, at, and end are char boundaries.
        let local = &text[start..at];
        let domain = text[at + 1..end].to_ascii_lowercase();
        out.push(Entity {
            kind: Kind::Email,
            start,
            end,
            confidence,
            review_recommended: false,
            source: Source::Rules,
            components: Vec::new(),
            normalized: Some(format!("{local}@{domain}")),
            region: None,
        });
    }
    out
}

/// `candidate` as one email address, as the scanner would read it in running text: the entity
/// when the scan finds exactly one and it covers the whole string.
#[cfg(feature = "markdown")]
pub(crate) fn whole(candidate: &str) -> Option<Entity> {
    let mut found = scan(candidate);
    match found.as_slice() {
        [only] if only.start == 0 && only.end == candidate.len() => found.pop(),
        _ => None,
    }
}

/// `Some(confidence)` when local and domain satisfy the accepted grammar.
fn validate(local: &[u8], domain: &[u8]) -> Option<f32> {
    if local.is_empty()
        || local.len() > MAX_LOCAL
        || local[0] == b'.'
        || local[local.len() - 1] == b'.'
    {
        return None;
    }
    if local.windows(2).any(|w| w == b"..") {
        return None;
    }
    if domain.is_empty() || domain.len() > MAX_DOMAIN {
        return None;
    }
    let labels: Vec<&[u8]> = domain.split(|&b| b == b'.').collect();
    if labels.len() < 2 {
        return None;
    }
    for label in &labels {
        if label.is_empty()
            || label.len() > MAX_LABEL
            || label[0] == b'-'
            || label[label.len() - 1] == b'-'
        {
            return None;
        }
    }
    let tld = labels[labels.len() - 1];
    if tld.iter().all(u8::is_ascii_alphabetic) && (2..=24).contains(&tld.len()) {
        Some(0.99)
    } else {
        Some(0.7)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn found(text: &str) -> Vec<(&str, String, f32)> {
        scan(text)
            .iter()
            .map(|e| {
                (
                    &text[e.start..e.end],
                    e.normalized.clone().unwrap(),
                    e.confidence,
                )
            })
            .collect()
    }

    #[test]
    fn basic() {
        assert_eq!(
            found("nino@kavkaz-freight.example"),
            vec![(
                "nino@kavkaz-freight.example",
                "nino@kavkaz-freight.example".into(),
                0.99
            )]
        );
    }

    #[test]
    fn trailing_punctuation_and_case() {
        assert_eq!(
            found("Mail Nino.B@Kavkaz.EXAMPLE."),
            vec![(
                "Nino.B@Kavkaz.EXAMPLE",
                "Nino.B@kavkaz.example".into(),
                0.99
            )]
        );
    }

    #[test]
    fn rejects() {
        assert!(found("a..b@x.example").is_empty());
        assert!(found("user@localhost").is_empty());
        assert!(found("@example.com").is_empty());
        assert!(found(".a@x.example").is_empty());
        assert!(found("a@-x.example").is_empty());
        assert!(found("see /path@x.example").is_empty());
    }

    #[test]
    fn numeric_tld_is_medium() {
        assert_eq!(found("a@10.0.0.1")[0].2, 0.7);
    }

    #[test]
    fn url_userinfo_is_not_an_email() {
        assert!(found("https://user@host.example/x").is_empty());
        assert!(found("https://user:pass@host.example/x").is_empty());
        assert!(found("see ftp://a:b@files.example now").is_empty());
    }

    #[test]
    fn leading_markup_is_not_part_of_the_address() {
        for text in [
            "Contact 'nino@x.example'",
            "**nino@x.example**",
            "`nino@x.example`",
            "{nino@x.example}",
            "email=nino@x.example",
        ] {
            let f = found(text);
            assert_eq!(f.len(), 1, "{text}");
            assert_eq!(f[0].0, "nino@x.example", "{text}");
            assert_eq!(f[0].1, "nino@x.example", "{text}");
        }
        assert_eq!(found("o'brien@x.example")[0].0, "o'brien@x.example");
        assert_eq!(found("Mail 'o'brien@x.example'")[0].0, "o'brien@x.example");
    }

    /// The URL check used to rescan back to the previous whitespace for every address.
    #[test]
    fn many_addresses_without_whitespace_scan_linearly() {
        let text = "a@b.example,".repeat(40_000);
        let started = std::time::Instant::now();
        assert_eq!(scan(&text).len(), 40_000);
        assert!(
            started.elapsed().as_secs_f64() < 1.0,
            "{:?}",
            started.elapsed()
        );
    }

    #[test]
    fn mailto_is_still_an_email() {
        assert_eq!(found("mailto:ops@a.b.example").len(), 1);
    }

    #[test]
    fn adjacent_matches_do_not_overlap() {
        let spans: Vec<_> = scan("x@a.example@b.example")
            .iter()
            .map(|e| (e.start, e.end))
            .collect();
        assert!(spans.windows(2).all(|w| w[0].1 <= w[1].0));
    }

    #[test]
    fn two_in_a_line() {
        let f = found("a@x.example, b@y.example");
        assert_eq!(f.len(), 2);
        assert_eq!(f[1].0, "b@y.example");
    }
}
