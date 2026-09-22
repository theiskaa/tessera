//! Phone candidate scanner with per-region validation and E.164 normalization.
//!
//! The scanner decides what is phone-shaped; libphonenumber metadata decides
//! what is a phone number. National numbers without a country code need a
//! region from `country_hint`; without one they are dropped rather than
//! guessed. Fullwidth and Arabic-Indic digits are not scanned yet, and a
//! trailing extension is left outside the span.

use crate::{Entity, Kind, Source};

/// A phone-shaped run of text, before validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Candidate {
    pub start: usize,
    pub end: usize,
    /// Digits and `+` only, with a leading `00` already rewritten to `+` and a
    /// parenthesised trunk `(0)` after the country code left out.
    pub dial: String,
    pub international: bool,
    pub groups: Vec<u8>,
    /// The separator run between each pair of consecutive groups.
    pub gaps: Vec<String>,
    /// A bracketed group right after the country code, as in `+49 (030) ...`; it may hold
    /// the national trunk prefix, which validation strips.
    pub bracketed_after_code: bool,
}

const MAX_GROUPS: usize = 8;
const MAX_SEPARATOR_RUN: usize = 2;
const MAX_PAREN_DIGITS: usize = 5;
const MAX_UNSEPARATED_DIGITS: usize = 11;
const DATE_SEPARATORS: [&str; 3] = ["-", "/", "."];
/// Group lengths of the date shapes: 2024-01-15, 15.01.2024, 1/15/2024, 15/1/2024, 1/1/2024, 15-01-24.
const DATES: [[u8; 3]; 6] = [
    [4, 2, 2],
    [2, 2, 4],
    [1, 2, 4],
    [2, 1, 4],
    [1, 1, 4],
    [2, 2, 2],
];

fn is_sep(c: char) -> bool {
    matches!(
        c,
        ' ' | '\u{A0}' | '-' | '\u{2010}' | '\u{2013}' | '.' | '/' | '(' | ')'
    )
}

fn is_currency(c: char) -> bool {
    matches!(c, '$' | '€' | '£' | '¥' | '₾' | '%')
}

/// Every phone-shaped candidate in `text`, sorted, non-overlapping.
pub(crate) fn find_candidates(text: &str) -> Vec<Candidate> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i].1;
        let starts = c == '+' || c == '(' || c.is_ascii_digit();
        let blocked = i > 0 && {
            let p = chars[i - 1].1;
            // After ':' the digits are a time's minutes or seconds, never the start of a number.
            p.is_ascii_alphanumeric() || matches!(p, '@' | '#' | '+' | ':')
        };
        if !starts || blocked {
            i += 1;
            continue;
        }
        match take_candidate(text, &chars, i) {
            Ok((cand, next_i)) => {
                if !is_hard_negative(&chars, &cand, i, next_i) {
                    out.push(cand);
                }
                i = next_i;
            }
            // A rejected run is skipped whole, so the tail of an IBAN or account number
            // cannot qualify on its own once its leading groups are gone.
            Err(resume) => i = resume,
        }
    }
    out
}

/// Reads one candidate starting at char index `i`. On success returns it and the index of
/// the first char after it; on failure returns the index to resume scanning from.
fn take_candidate(
    text: &str,
    chars: &[(usize, char)],
    i: usize,
) -> Result<(Candidate, usize), usize> {
    let at = |j: usize| chars.get(j).map(|&(_, c)| c);
    let byte_at = |j: usize| chars.get(j).map_or(text.len(), |&(b, _)| b);

    let start = chars[i].0;
    let mut international = chars[i].1 == '+';
    let mut j = if international { i + 1 } else { i };
    let mut groups: Vec<u8> = Vec::new();
    let mut gaps: Vec<String> = Vec::new();
    // Separators read since the last group; they become a gap once the next group is read.
    let mut pending = String::new();
    let mut digits = String::new();
    let mut paren_used = false;
    let mut bracketed_after_code = false;
    // Index of the first char after the last accepted group; the span ends here.
    let mut end_i = j;

    loop {
        let paren_allowed = !paren_used
            && ((groups.is_empty() && !international) || (international && groups.len() == 1));
        let group_start = j;
        if at(j) == Some('(') && paren_allowed {
            let mut k = j + 1;
            while at(k).is_some_and(|c| c.is_ascii_digit()) {
                k += 1;
            }
            let n = k - (j + 1);
            if n == 0 || n > MAX_PAREN_DIGITS || at(k) != Some(')') {
                break;
            }
            // "+44 (0)20 ...": the bracketed trunk prefix is dialled only inside the country.
            let trunk = international && n == 1 && at(j + 1) == Some('0');
            if !trunk {
                bracketed_after_code = international && groups.len() == 1;
                digits.extend(chars[j + 1..k].iter().map(|&(_, c)| c));
                if !groups.is_empty() {
                    gaps.push(std::mem::take(&mut pending));
                }
                groups.push(n as u8);
            }
            paren_used = true;
            j = k + 1;
        } else if at(j).is_some_and(|c| c.is_ascii_digit()) {
            while at(j).is_some_and(|c| c.is_ascii_digit()) {
                j += 1;
            }
            digits.extend(chars[group_start..j].iter().map(|&(_, c)| c));
            if !groups.is_empty() {
                gaps.push(std::mem::take(&mut pending));
            }
            groups.push((j - group_start) as u8);
        } else {
            break;
        }
        end_i = j;
        if groups.len() == MAX_GROUPS {
            break;
        }

        let run_start = j;
        while at(j).is_some_and(|c| is_sep(c) && c != '(' && c != ')') {
            j += 1;
        }
        if j - run_start > MAX_SEPARATOR_RUN {
            break;
        }
        let next = at(j);
        let continues = next.is_some_and(|c| c.is_ascii_digit())
            || (next == Some('(') && !paren_used && international && groups.len() == 1);
        if !continues {
            break;
        }
        pending.extend(chars[run_start..j].iter().map(|&(_, c)| c));
    }
    let resume = end_i.max(i + 1);

    let count = digits.len();
    let allowed = if international { 8..=15 } else { 7..=15 };
    if !allowed.contains(&count) {
        return Err(resume);
    }
    let end = byte_at(end_i);
    if text
        .as_bytes()
        .get(end)
        .is_some_and(|b| b.is_ascii_alphanumeric())
    {
        return Err(resume);
    }

    let mut dial = String::with_capacity(count + 1);
    if international {
        dial.push('+');
    }
    dial.push_str(&digits);
    // "0044 20 ..." is international; "00 030..." is a stray group from something else.
    if !international
        && dial.starts_with("00")
        && groups.first().is_some_and(|&g| g > 2)
        && count >= 10
    {
        dial = format!("+{}", &dial[2..]);
        international = true;
    }

    let candidate = Candidate {
        start,
        end,
        dial,
        international,
        groups,
        gaps,
        bracketed_after_code,
    };
    Ok((candidate, end_i))
}

fn is_hard_negative(chars: &[(usize, char)], c: &Candidate, first_i: usize, next_i: usize) -> bool {
    let all_gaps =
        |set: &[&str]| !c.gaps.is_empty() && c.gaps.iter().all(|g| set.contains(&g.as_str()));

    let ip =
        c.groups.len() == 4 && c.groups.iter().all(|&g| (1..=3).contains(&g)) && all_gaps(&["."]);
    if ip {
        return true;
    }

    // A date, alone or followed by the hour of a time ("2024-01-15 10:30"), is never a phone
    // number. More groups than that, as in "06-12-34-56-78", make it a phone again.
    let date_shape = c.groups.len() >= 3
        && c.gaps.len() >= 2
        && c.gaps[0] == c.gaps[1]
        && DATE_SEPARATORS.contains(&c.gaps[0].as_str())
        && DATES.iter().any(|d| d[..] == c.groups[..3]);
    let hour_follows =
        c.groups.len() == 4 && c.gaps[2].chars().all(char::is_whitespace) && c.groups[3] <= 2;
    let date = date_shape && (c.groups.len() == 3 || hour_follows);
    if date {
        return true;
    }

    let at = |j: Option<usize>| j.and_then(|j| chars.get(j)).map(|&(_, ch)| ch);
    let before = at(first_i.checked_sub(1));
    let before = if before == Some(' ') {
        at(first_i.checked_sub(2))
    } else {
        before
    };
    let after = at(Some(next_i));
    let after = if after == Some(' ') {
        at(Some(next_i + 1))
    } else {
        after
    };
    if before.is_some_and(is_currency) || after.is_some_and(is_currency) {
        return true;
    }

    !c.international
        && c.gaps.is_empty()
        && c.groups.len() == 1
        && c.dial.len() > MAX_UNSEPARATED_DIGITS
}

/// Every validated phone number in `text`, sorted by start, non-overlapping.
pub fn scan(text: &str, country_hint: &[&str]) -> Vec<Entity> {
    find_candidates(text)
        .into_iter()
        .filter_map(|c| {
            validate(&c, country_hint).map(|(confidence, normalized, region)| Entity {
                kind: Kind::Phone,
                start: c.start,
                end: c.end,
                confidence,
                review_recommended: false,
                source: Source::Rules,
                components: Vec::new(),
                normalized: Some(normalized),
                region,
            })
        })
        .collect()
}

/// Whether `national` starts with `prefix`, where `x` in the prefix matches any digit.
#[cfg(feature = "phone-metadata")]
fn matches_prefix(national: &str, prefix: &str) -> bool {
    prefix.len() <= national.len()
        && prefix
            .bytes()
            .zip(national.bytes())
            .all(|(p, n)| p == b'x' || p == n)
}

/// Valid when a known prefix matches at a possible length; possible when only an ordinary
/// fixed-line or mobile length fits. A length only special types use needs the prefix.
#[cfg(feature = "phone-metadata")]
fn score(region: &super::phone_tables::Region, national: &str) -> Option<f32> {
    const VALID: f32 = 0.99;
    const POSSIBLE: f32 = 0.75;
    let len = u8::try_from(national.len()).ok()?;
    let prefix = region.prefixes.iter().any(|p| matches_prefix(national, p));
    if prefix && region.lengths.contains(&len) {
        Some(VALID)
    } else if region.core_lengths.contains(&len) {
        Some(POSSIBLE)
    } else {
        None
    }
}

/// The best reading so far: score, E.164, region id, and whether that region is the main
/// country for its calling code.
#[cfg(feature = "phone-metadata")]
type Reading = (f32, String, Option<String>, bool);

/// Keep `region`'s reading of `national` when it scores higher than the best so far. Regions
/// sharing a calling code at the same score resolve to the main country, as libphonenumber does.
#[cfg(feature = "phone-metadata")]
fn consider(best: &mut Option<Reading>, region: &super::phone_tables::Region, national: &str) {
    let Some(s) = score(region, national) else {
        return;
    };
    let better = best
        .as_ref()
        .is_none_or(|b| s > b.0 || (s == b.0 && region.main && !b.3));
    if better {
        *best = Some((
            s,
            format!("+{}{}", region.code, national),
            Some(region.id.to_string()),
            region.main,
        ));
    }
}

#[cfg(feature = "phone-metadata")]
fn validate(c: &Candidate, country_hint: &[&str]) -> Option<(f32, String, Option<String>)> {
    use super::phone_tables::REGIONS;

    let digits = c.dial.trim_start_matches('+');
    let mut best: Option<Reading> = None;
    if c.international {
        for region in REGIONS {
            let code = region.code.to_string();
            let Some(national) = digits.strip_prefix(code.as_str()) else {
                continue;
            };
            // "+49 (030) 2312 5456": inside the brackets is the number as dialled at home.
            let national = match region.national_prefix {
                Some(p) if c.bracketed_after_code => national.strip_prefix(p).unwrap_or(national),
                _ => national,
            };
            consider(&mut best, region, national);
        }
    } else {
        for hint in country_hint {
            let Some(region) = REGIONS.iter().find(|r| r.id.eq_ignore_ascii_case(hint)) else {
                continue;
            };
            let national = match region.national_prefix {
                Some(p) => digits.strip_prefix(p).unwrap_or(digits),
                None => digits,
            };
            consider(&mut best, region, national);
            if best.as_ref().is_some_and(|b| b.0 >= 0.99) {
                break;
            }
        }
    }
    best.map(|(score, e164, region, _)| (score, e164, region))
}

#[cfg(not(feature = "phone-metadata"))]
fn validate(_c: &Candidate, _country_hint: &[&str]) -> Option<(f32, String, Option<String>)> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spans(text: &str) -> Vec<(&str, String)> {
        find_candidates(text)
            .iter()
            .map(|c| (&text[c.start..c.end], c.dial.clone()))
            .collect()
    }

    #[test]
    fn international_and_national_shapes() {
        assert_eq!(
            spans("+44 20 7946 0958"),
            vec![("+44 20 7946 0958", "+442079460958".into())]
        );
        assert_eq!(
            spans("(202) 555-0199."),
            vec![("(202) 555-0199", "2025550199".into())]
        );
        assert_eq!(
            spans("Tel: 0113 496 0123, thanks"),
            vec![("0113 496 0123", "01134960123".into())]
        );
        assert_eq!(
            spans("+1 (202) 555-0143"),
            vec![("+1 (202) 555-0143", "+12025550143".into())]
        );
        assert_eq!(
            spans("0044 20 7946 0958"),
            vec![("0044 20 7946 0958", "+442079460958".into())]
        );
    }

    #[test]
    fn hard_negatives() {
        assert!(spans("10.0.0.1").is_empty());
        assert!(spans("2024-01-15").is_empty());
        assert!(spans("15/01/2024").is_empty());
        assert!(spans("$555-0100").is_empty());
        assert!(spans("555-0100 %").is_empty());
        assert!(spans("order 202555014312").is_empty());
        assert!(spans("AB2025550143").is_empty());
        assert!(spans("2025550143x").is_empty());
        assert!(spans("123-456").is_empty());
    }

    #[test]
    fn date_followed_by_time_is_not_a_phone() {
        assert!(spans("Meeting at 2024-01-15 10:30").is_empty());
        assert!(spans("15.01.2024 10:30").is_empty());
        assert!(spans("2024/01/15 12:00").is_empty());
    }

    #[test]
    fn trunk_prefix_in_brackets_is_not_dialled() {
        assert_eq!(
            spans("Tel +44 (0)20 7946 0958"),
            vec![("+44 (0)20 7946 0958", "+442079460958".into())]
        );
        assert_eq!(
            spans("+49 (0) 30 23125456"),
            vec![("+49 (0) 30 23125456", "+493023125456".into())]
        );
    }

    #[test]
    fn tail_of_a_long_number_is_not_a_phone() {
        assert!(spans("IBAN DE89 3704 0044 0532 0130 00").is_empty());
        assert!(spans("card 4111 1111 1111 1111 exp").is_empty());
    }

    #[cfg(feature = "phone-metadata")]
    #[test]
    fn trunk_prefix_normalizes_to_valid_e164() {
        let got: Vec<_> = scan("+44 (0)20 7946 0958, +49 (0)30 23125456", &[])
            .into_iter()
            .map(|e| e.normalized)
            .collect();
        assert_eq!(
            got,
            vec![Some("+442079460958".into()), Some("+493023125456".into())]
        );
    }

    #[cfg(feature = "phone-metadata")]
    #[test]
    fn no_phone_in_dates_or_account_numbers() {
        assert!(scan("Meeting at 2024-01-15 10:30", &["US"]).is_empty());
        assert!(scan("15.01.2024 10:30", &["DE"]).is_empty());
        assert!(scan("IBAN DE89 3704 0044 0532 0130 00", &[]).is_empty());
    }

    #[test]
    fn grouped_numbers_longer_than_a_date_are_phones() {
        assert_eq!(spans("06-12-34-56-78").len(), 1);
        assert_eq!(spans("02.12.34.56.78").len(), 1);
    }

    #[test]
    fn minutes_of_a_time_do_not_start_a_number() {
        assert_eq!(
            spans("10:30-12:00 0301234567"),
            vec![("0301234567", "0301234567".into())]
        );
    }

    #[cfg(feature = "phone-metadata")]
    #[test]
    fn nanp_ties_go_to_the_main_country() {
        let region = |t: &str| scan(t, &[]).first().and_then(|e| e.region.clone());
        assert_eq!(region("+1 800 555 0199").as_deref(), Some("US"));
        assert_eq!(region("+1 888 555 0199").as_deref(), Some("US"));
        assert!(scan("+1 555 0100", &[]).is_empty());
    }

    #[cfg(feature = "phone-metadata")]
    #[test]
    fn bracketed_trunk_prefix_is_stripped() {
        let got = scan("+49 (030) 23125456", &[]);
        assert_eq!(got[0].normalized.as_deref(), Some("+493023125456"));
    }

    #[cfg(feature = "phone-metadata")]
    #[test]
    fn european_grouped_mobiles_and_times() {
        assert_eq!(scan("06-12-34-56-78", &["NL"]).len(), 1);
        assert_eq!(scan("02.12.34.56.78", &["BE"]).len(), 1);
        let got = scan("10:30-12:00 0301234567", &["DE"]);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].normalized.as_deref(), Some("+49301234567"));
    }

    #[test]
    fn extension_is_left_outside() {
        assert_eq!(
            spans("+44 20 7946 0958 ext. 12"),
            vec![("+44 20 7946 0958", "+442079460958".into())]
        );
    }

    #[cfg(feature = "phone-metadata")]
    #[test]
    fn prefix_matching() {
        assert!(matches_prefix("2025550143", "2xx"));
        assert!(!matches_prefix("3025550143", "31"));
        assert!(matches_prefix("3025550143", ""));
        assert!(!matches_prefix("20", "2xx"));
    }

    #[cfg(feature = "phone-metadata")]
    #[test]
    fn shared_calling_code_picks_the_matching_region() {
        let region = |text: &str| scan(text, &[]).first().and_then(|e| e.region.clone());
        assert_eq!(region("+1 202 555 0143").as_deref(), Some("US"));
        assert_eq!(region("+1 416 555 0143").as_deref(), Some("CA"));
    }

    #[test]
    fn long_separator_run_ends_the_candidate() {
        assert_eq!(
            spans("202   555 0143"),
            vec![("555 0143", "5550143".into())]
        );
    }
}
