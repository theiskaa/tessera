//! The regions a document's national phone numbers most likely belong to, read from what the
//! document says about itself, for callers that give no country hint.
//!
//! A number written without its country code, such as `020 7946 0321`, is valid in more than
//! one region, so the scanner needs a region to read it in. The evidence, strongest first:
//! numbers in the same document written with their country code, email domains, postcodes,
//! the script, and country names. Regions come back ordered by weight; the first one a number is valid in
//! reads it. A document with no evidence gives none, and its national numbers stay unread.

use super::{email, phone};

/// A number written with its country code says the most about the others.
const INTERNATIONAL: u32 = 3;
/// A country-code top-level domain on an email address.
const DOMAIN: u32 = 2;
/// Georgian or Japanese kana text.
const SCRIPT: u32 = 2;
/// A postcode in its country's own shape.
const POSTCODE: u32 = 2;
/// A German postcode before a town name; a looser shape, so it weighs less.
const LOOSE_POSTCODE: u32 = 1;
/// A country named in the text.
const NAME: u32 = 1;
/// Characters of a script before it counts as the document's.
const SCRIPT_CHARS: usize = 3;
/// Mentions of one name that count; a list of addresses should not outvote everything else.
const MAX_NAME_MENTIONS: usize = 3;
/// Longest piece scanned at once, so a long document costs bounded memory; pieces are cut at
/// line breaks, which no number or address crosses.
const PIECE_BYTES: usize = 64 * 1024;
/// Look past an arbitrary piece boundary so a phone, email, or postcode beginning before it
/// can be read whole. Valid emails are at most 320 bytes; phone runs are bounded as well.
const PIECE_LOOKAHEAD_BYTES: usize = 1024;

const US_STATES: [&str; 51] = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "DC", "FL", "GA", "HI", "ID", "IL", "IN", "IA",
    "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM",
    "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA",
    "WV", "WI", "WY",
];

const DOMAINS: [(&str, &str); 11] = [
    ("uk", "GB"),
    ("de", "DE"),
    ("jp", "JP"),
    ("ge", "GE"),
    ("at", "AT"),
    ("ch", "CH"),
    ("nl", "NL"),
    ("be", "BE"),
    ("ie", "IE"),
    ("ca", "CA"),
    ("us", "US"),
];

/// Country names in English and in the country's own language. "Georgia" is left out: it is
/// also a US state.
const NAMES: [(&str, &str); 25] = [
    ("United Kingdom", "GB"),
    ("UK", "GB"),
    ("England", "GB"),
    ("Scotland", "GB"),
    ("Wales", "GB"),
    ("Germany", "DE"),
    ("Deutschland", "DE"),
    ("Japan", "JP"),
    ("日本", "JP"),
    ("საქართველო", "GE"),
    ("United States", "US"),
    ("USA", "US"),
    ("Austria", "AT"),
    ("Österreich", "AT"),
    ("Switzerland", "CH"),
    ("Schweiz", "CH"),
    ("Suisse", "CH"),
    ("Netherlands", "NL"),
    ("Nederland", "NL"),
    ("Belgium", "BE"),
    ("België", "BE"),
    ("Belgique", "BE"),
    ("Ireland", "IE"),
    ("Éire", "IE"),
    ("Canada", "CA"),
];

/// The regions `text` points to, heaviest evidence first, ties in order of first evidence.
pub(crate) fn infer(text: &str) -> Vec<&'static str> {
    let mut evidence = Evidence::default();
    evidence.scan(text, 0);
    evidence.rank()
}

/// Accumulates country clues across visible Markdown ranges without keeping a second copy of
/// the document. `scan` accepts each independent visible range in source order.
#[derive(Default)]
pub(crate) struct Evidence {
    votes: Vec<(&'static str, u32, usize)>,
    name_counts: [usize; NAMES.len()],
    script_counts: [usize; 2],
    script_first: [Option<usize>; 2],
}

impl Evidence {
    fn vote(&mut self, region: &'static str, weight: u32, at: usize) {
        match self.votes.iter_mut().find(|(r, _, _)| *r == region) {
            Some(v) => {
                v.1 += weight;
                v.2 = v.2.min(at);
            }
            None => self.votes.push((region, weight, at)),
        }
    }

    pub(crate) fn scan(&mut self, text: &str, base: usize) {
        for (piece_base, owned) in pieces(text) {
            let limit = (piece_base + owned.len() + PIECE_LOOKAHEAD_BYTES).min(text.len());
            let end = (piece_base + owned.len()..=limit)
                .rev()
                .find(|&at| text.is_char_boundary(at))
                .unwrap_or(piece_base + owned.len());
            let piece = &text[piece_base..end];
            for p in phone::scan(piece, &[]) {
                if p.start >= owned.len() {
                    continue;
                }
                if let Some(region) = p.region.as_deref().and_then(known) {
                    self.vote(region, INTERNATIONAL, base + piece_base + p.start);
                }
            }
            for e in email::scan(piece) {
                if e.start >= owned.len() {
                    continue;
                }
                let tld = e
                    .normalized
                    .as_deref()
                    .and_then(|n| n.rsplit('.').next())
                    .unwrap_or("");
                if let Some(&(_, region)) =
                    DOMAINS.iter().find(|(d, _)| tld.eq_ignore_ascii_case(d))
                {
                    self.vote(region, DOMAIN, base + piece_base + e.start);
                }
            }
            for (region, weight, at) in postcodes(piece) {
                if at < owned.len() {
                    self.vote(region, weight, base + piece_base + at);
                }
            }
        }
        for (at, c) in text.char_indices() {
            for (i, range) in ['\u{10a0}'..='\u{10ff}', '\u{3040}'..='\u{30ff}']
                .into_iter()
                .enumerate()
            {
                if range.contains(&c) {
                    self.script_counts[i] += 1;
                    self.script_first[i].get_or_insert(base + at);
                }
            }
        }
        for (i, &(name, region)) in NAMES.iter().enumerate() {
            for at in mentions(text, name).take(MAX_NAME_MENTIONS - self.name_counts[i]) {
                self.vote(region, NAME, base + at);
                self.name_counts[i] += 1;
            }
        }
    }

    pub(crate) fn rank(mut self) -> Vec<&'static str> {
        for (i, region) in ["GE", "JP"].into_iter().enumerate() {
            if self.script_counts[i] >= SCRIPT_CHARS {
                self.vote(region, SCRIPT, self.script_first[i].unwrap_or(0));
            }
        }
        self.votes.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)));
        self.votes
            .into_iter()
            .map(|(region, _, _)| region)
            .collect()
    }
}

/// `text` in pieces of at most `PIECE_BYTES`, each with its offset, cut after a line break
/// where there is one and at a character boundary otherwise.
fn pieces(text: &str) -> impl Iterator<Item = (usize, &str)> {
    let mut start = 0;
    std::iter::from_fn(move || {
        if start >= text.len() {
            return None;
        }
        let mut end = (start + PIECE_BYTES).min(text.len());
        if end < text.len() {
            end = match text[start..end].rfind('\n') {
                Some(i) => start + i + 1,
                None => (start + 1..=end)
                    .rev()
                    .find(|&i| text.is_char_boundary(i))
                    .unwrap_or(text.len()),
            };
        }
        let piece = (start, &text[start..end]);
        start = end;
        Some(piece)
    })
}

/// Postcodes in `text` in a shape one region uses: a US state and ZIP code (`PA 19107`), a UK
/// postcode (`NW1 6XE`), a Dutch one (`1011 AB`), a Japanese one after `〒`, and, weighing
/// less, five digits before a capitalised town name (`10115 Berlin`), as in Germany.
fn postcodes(text: &str) -> Vec<(&'static str, u32, usize)> {
    let words: Vec<(usize, &str)> = text
        .split_whitespace()
        .map(|w| {
            let at = w.as_ptr() as usize - text.as_ptr() as usize;
            (
                at,
                w.trim_matches(|c: char| matches!(c, ',' | '.' | ';' | ':' | '(' | ')')),
            )
        })
        .collect();
    let mut out = Vec::new();
    for pair in words.windows(2) {
        let ((at, a), (_, b)) = (pair[0], pair[1]);
        if US_STATES.contains(&a) && zip(b) {
            out.push(("US", POSTCODE, at));
        } else if uk_outward(a) && uk_inward(b) {
            out.push(("GB", POSTCODE, at));
        } else if a.len() == 4
            && a.bytes().all(|c| c.is_ascii_digit())
            && b.len() == 2
            && b.bytes().all(|c| c.is_ascii_uppercase())
        {
            out.push(("NL", POSTCODE, at));
        } else if a.len() == 5
            && a.bytes().all(|c| c.is_ascii_digit())
            && b.chars().next().is_some_and(char::is_uppercase)
            && b.chars().skip(1).all(char::is_lowercase)
            && b.chars().count() > 2
        {
            out.push(("DE", LOOSE_POSTCODE, at));
        }
    }
    for (at, _) in text.match_indices('〒') {
        out.push(("JP", POSTCODE, at));
    }
    out
}

/// Five digits, or five and four joined by a hyphen.
fn zip(word: &str) -> bool {
    let (five, four) = word.split_once('-').unwrap_or((word, "0000"));
    five.len() == 5
        && four.len() == 4
        && five.bytes().chain(four.bytes()).all(|c| c.is_ascii_digit())
}

/// The outward half of a UK postcode: one or two letters, a digit, and an optional letter or
/// digit (`NW1`, `M1`, `EC1A`, `B33`).
fn uk_outward(word: &str) -> bool {
    let b = word.as_bytes();
    let letters = b.iter().take_while(|c| c.is_ascii_uppercase()).count();
    (1..=2).contains(&letters)
        && b.get(letters).is_some_and(u8::is_ascii_digit)
        && match &b[letters + 1..] {
            [] => true,
            [c] => c.is_ascii_digit() || c.is_ascii_uppercase(),
            _ => false,
        }
}

/// The inward half of a UK postcode: a digit and two letters (`6XE`).
fn uk_inward(word: &str) -> bool {
    matches!(word.as_bytes(), [d, a, b] if d.is_ascii_digit() && a.is_ascii_uppercase() && b.is_ascii_uppercase())
}

/// `region` as one of this module's own region codes.
fn known(region: &str) -> Option<&'static str> {
    DOMAINS
        .iter()
        .map(|&(_, r)| r)
        .find(|r| r.eq_ignore_ascii_case(region))
}

/// Byte offsets where `name` is mentioned. A Latin-script name must stand as a word of its
/// own, since a letter or digit beside it makes it part of another word; scripts written
/// without spaces have no such boundary.
fn mentions<'a>(text: &'a str, name: &'a str) -> impl Iterator<Item = usize> + 'a {
    let latin = name.chars().all(|c| c < '\u{0250}');
    text.match_indices(name).filter_map(move |(at, _)| {
        let before = text[..at].chars().next_back();
        let after = text[at + name.len()..].chars().next();
        let joined = |c: Option<char>| latin && c.is_some_and(char::is_alphanumeric);
        (!joined(before) && !joined(after)).then_some(at)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_evidence_gives_no_region() {
        assert!(infer("Call 020 7946 0321 after lunch.").is_empty());
    }

    #[cfg(feature = "phone-metadata")]
    #[test]
    fn a_number_with_its_country_code_names_the_region() {
        assert_eq!(
            infer("Office +44 20 7946 0958, mobile 07700 900123."),
            ["GB"]
        );
    }

    #[test]
    fn domains_scripts_and_names_count() {
        assert_eq!(infer("Write to anna@kanzlei-weber.de."), ["DE"]);
        assert_eq!(infer("ნინო ბერიძე, 032 212 3456"), ["GE"]);
        assert_eq!(infer("担当：やまだ 03-1234-5678"), ["JP"]);
        assert_eq!(infer("Musterstraße 12, 10115 Berlin, Deutschland"), ["DE"]);
        assert_eq!(infer("東京都、日本の本社"), ["JP"]);
    }

    #[cfg(feature = "phone-metadata")]
    #[test]
    fn heavier_evidence_comes_first() {
        let text = "Our Berlin office (Germany) forwards to +44 20 7946 0958 and ops@firm.co.uk.";
        assert_eq!(infer(text), ["GB", "DE"]);
    }

    #[test]
    fn postcodes_name_their_region() {
        assert_eq!(infer("1200 Market Street, Philadelphia, PA 19107"), ["US"]);
        assert_eq!(infer("Flat 4, 221B Baker Street, London NW1 6XE"), ["GB"]);
        assert_eq!(infer("Damrak 1, 1012 LG Amsterdam"), ["NL"]);
        assert_eq!(infer("Friedrichstraße 43, 10117 Berlin"), ["DE"]);
        assert_eq!(infer("〒530-0001 大阪市"), ["JP"]);
        assert!(infer("Invoice INV 20458, order 12345 shipped").is_empty());
    }

    #[test]
    fn pieces_cover_the_text_at_line_breaks() {
        let line = "x".repeat(1000) + "\n";
        let text = line.repeat(200) + "ünïcödé";
        let parts: Vec<(usize, &str)> = pieces(&text).collect();
        assert!(parts.len() > 1);
        assert!(parts.iter().all(|(_, p)| p.len() <= PIECE_BYTES));
        assert_eq!(parts.iter().map(|(_, p)| *p).collect::<String>(), text);
        assert!(
            parts[..parts.len() - 1]
                .iter()
                .all(|(_, p)| p.ends_with('\n'))
        );
        let unbroken = "é".repeat(PIECE_BYTES);
        let parts: Vec<(usize, &str)> = pieces(&unbroken).collect();
        assert_eq!(parts.iter().map(|(_, p)| *p).collect::<String>(), unbroken);
    }

    #[test]
    fn email_crossing_a_piece_cut_still_names_the_region_once() {
        let text = format!("{}a@b.de", ":".repeat(PIECE_BYTES - 2));
        assert_eq!(pieces(&text).count(), 2);
        assert_eq!(infer(&text), ["DE"]);
    }

    #[cfg(feature = "phone-metadata")]
    #[test]
    fn international_phone_crossing_a_piece_cut_still_names_the_region() {
        let text = format!("{}+44 20 7946 0958", "!".repeat(PIECE_BYTES - 2));
        assert_eq!(pieces(&text).count(), 2);
        assert_eq!(infer(&text), ["GB"]);
    }

    #[test]
    fn a_name_inside_another_word_does_not_count() {
        assert!(infer("UKULELE lessons, USAGE notes").is_empty());
        assert!(infer("Georgia Avenue, Atlanta").is_empty());
    }
}
