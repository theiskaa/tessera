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
    /// Char indices `(first, past_last)` of each group in the scanned text, brackets included.
    pub group_spans: Vec<(usize, usize)>,
    /// The separator run between each pair of consecutive groups.
    pub gaps: Vec<String>,
    /// A phone word such as `Tel` or `Customer service:` comes right before.
    pub phone_label: bool,
    /// The read stopped at `MAX_RUN_GROUPS` or `MAX_RUN_DIGITS`, so the run may go on past
    /// `end`, and its last groups may be the start of a number cut in two.
    pub capped: bool,
}

/// Groups in one number.
const MAX_GROUPS: usize = 8;
const MAX_SEPARATOR_RUN: usize = 2;
const MAX_PAREN_DIGITS: usize = 5;
const MAX_UNSEPARATED_DIGITS: usize = 11;
/// E.164 caps a number at fifteen digits. A run is read on for four times the groups of one
/// number and eight numbers' worth of digits, so that numbers written side by side can be split apart
/// in `split`. The caps bound the work per run: a read stops before the group that would pass
/// one, and `split` hands the groups near that cut on to be read again as a run of their own.
const MAX_E164_DIGITS: usize = 15;
const MAX_RUN_GROUPS: usize = 4 * MAX_GROUPS;
const MAX_RUN_DIGITS: usize = 8 * MAX_E164_DIGITS;
const DATE_SEPARATORS: [&str; 3] = ["-", "/", "."];
/// Words that label the number after them as a phone: `Tel`, `Customer service:`,
/// `Order line`, `電話番号`.
const PHONE_LABELS: &[&str] = &[
    "tel",
    "phone",
    "telephone",
    "fax",
    "mobile",
    "mob",
    "cell",
    "call",
    "hotline",
    "telefon",
    "telefonnummer",
    "handy",
    "handynummer",
    "mobil",
    "mobilnummer",
    "ruf",
    "rufnummer",
    "whatsapp",
    "contact",
    "kontakt",
    "service",
    "services",
    "care",
    "line",
    "support",
    "office",
    "desk",
    "enquiries",
    "inquiries",
    "queries",
    "reception",
    "switchboard",
    "home",
    "work",
    "landline",
    "daytime",
    "evening",
    "direct",
    "festnetz",
    "mobiel",
    "telefoon",
    "gsm",
    "natel",
    "portable",
    "fon",
    "ph",
    "t",
    "m",
    "tél",
    "téléphone",
    "電話",
    "電話番号",
    "携帯",
    "携帯電話",
];
/// Words that label a number as something else: `ISBN 0-306-40615-2`, `routing 021000021`.
const NOT_PHONE_LABELS: &[&str] = &[
    "isbn",
    "issn",
    "ean",
    "upc",
    "gtin",
    "sku",
    "serial",
    "sn",
    "imei",
    "routing",
    "aba",
    "iban",
    "bic",
    "swift",
    "account",
    "acct",
    "konto",
    "kto",
    "blz",
    "sort",
    "stnr",
    "ust",
    "idnr",
    "vat",
    "ein",
    "tin",
    "ssn",
    "company",
    "registration",
    "reg",
    "hrb",
    "hra",
    "case",
    "aktenzeichen",
    "order",
    "bestellung",
    "invoice",
    "rechnung",
    "ticket",
    "tracking",
    "ref",
    "reference",
    "customer",
    "kd",
    "art",
    "part",
    "patient",
    "policy",
    "claim",
    "passport",
    "licence",
    "license",
    "vin",
    "flight",
    "pin",
    "postcode",
    "zip",
    "plz",
    "facture",
    "commande",
    "référence",
    "client",
    "tva",
    "btw",
    "kvk",
    "factuur",
    "klant",
    "referentie",
    "bestelling",
];
/// Words that say "a number follows" without saying which kind: `Order No.`, `Kd.-Nr.`,
/// `Tel. no.`, `Facture n°`. The word before the marker decides.
const NUMBER_MARKERS: &[&str] = &[
    "no",
    "nr",
    "number",
    "num",
    "id",
    "code",
    "nummer",
    "n°",
    "nº",
    "№",
    "numéro",
    "numero",
    "ნომერი",
];
/// Stems that make a `-nummer` or `-nr` compound a phone label (`Faxnummer`, `Telefonnr`,
/// `Servicenummer`) rather than another number (`Kundennummer`, `Bestellnr`).
const PHONE_STEMS: &[&str] = &[
    "telefon", "mobil", "handy", "service", "hotline", "festnetz", "telefoon", "mobiel", "fax",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Label {
    Phone,
    NotPhone,
    Marker,
    Other,
}

/// The kind of a lowercased word; `dotted` is whether a `.` follows it.
fn label_of(word: &str, dotted: bool) -> Label {
    let compound_stem = word
        .strip_suffix("nummer")
        .or_else(|| word.strip_suffix("nr"));
    if PHONE_LABELS.contains(&word) {
        Label::Phone
    } else if NUMBER_MARKERS.contains(&word) || (!word.is_ascii() && word.ends_with("id")) {
        Label::Marker
    } else if let Some(stem) = compound_stem {
        // `tel` and `fon` only at the edges of the stem: `bestell-` and `fonds-` are not phones.
        let phone = PHONE_STEMS.iter().any(|s| stem.contains(s))
            || stem.starts_with("tel")
            || stem.ends_with("fon");
        if phone { Label::Phone } else { Label::NotPhone }
    } else if NOT_PHONE_LABELS.contains(&word)
        || word.ends_with("zeichen")
        || (word == "az" && dotted)
        || (word.ends_with("番号") && !word.contains("電話"))
    {
        Label::NotPhone
    } else {
        Label::Other
    }
}

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
    matches!(c, '$' | '€' | '£' | '¥' | '円' | '₾' | '%')
}

/// A gap of spaces only; the empty gap before a bracket (`03(1234)5678`) is not one.
fn is_spaced(gap: &str) -> bool {
    !gap.is_empty() && gap.chars().all(char::is_whitespace)
}

/// A run read at some char index, before or after the hard negatives.
enum Checked {
    Open(Candidate),
    RuledOut(Candidate),
}

/// Every phone-shaped candidate in `text`, sorted, non-overlapping.
#[cfg(test)]
fn find_candidates(text: &str) -> Vec<Candidate> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while let Some((cand, _, next_i)) = next_candidate(text, &chars, i) {
        out.push(cand);
        i = next_i;
    }
    out
}

/// The next open candidate at or after char index `i`, with the char index it starts at and
/// the one after it.
fn next_candidate(
    text: &str,
    chars: &[(usize, char)],
    mut i: usize,
) -> Option<(Candidate, usize, usize)> {
    while i < chars.len() {
        let c = chars[i].1;
        let starts = c == '+' || c == '(' || c.is_ascii_digit();
        let blocked = i > 0 && {
            let p = chars[i - 1].1;
            let before = (i >= 2).then(|| chars[i - 2].1);
            // After a digit and ':' the digits are a time's minutes; after `Tel:` they are a
            // number. After a word and '/' they are a version (`Chrome/128.0.6613.138`).
            let time = p == ':' && before.is_some_and(|c| c.is_ascii_digit());
            let version = (p == '/' && before.is_some_and(|c| c.is_ascii_alphanumeric()))
                || (p == '.' && before.is_some_and(|c| c.is_ascii_digit()));
            p.is_ascii_alphanumeric() || matches!(p, '@' | '#' | '+') || time || version
        };
        if !starts || blocked {
            i += 1;
            continue;
        }
        match checked_candidate(text, chars, i, chars.len()) {
            Ok((Checked::Open(cand), next_i)) => return Some((cand, i, next_i)),
            // A rejected run is skipped whole, so the tail of an IBAN or account number
            // cannot qualify on its own once its leading groups are gone.
            Ok((Checked::RuledOut(cand), next_i)) => {
                i = labelled_head_end(chars, &cand, i).unwrap_or(next_i);
            }
            Err(resume) => i = resume,
        }
    }
    None
}

/// A candidate read at char index `i` with nothing at or past `limit`, open or ruled out by a
/// hard negative, with the index of the first char after it. On failure returns the index to
/// resume scanning from.
fn checked_candidate(
    text: &str,
    chars: &[(usize, char)],
    i: usize,
    limit: usize,
) -> Result<(Checked, usize), usize> {
    let (mut cand, next_i) = take_candidate(text, chars, i, limit)?;
    if is_hard_negative(chars, &cand, i, next_i) {
        return Ok((Checked::RuledOut(cand), next_i));
    }
    cand.phone_label = labelled_as_phone(chars, i);
    Ok((Checked::Open(cand), next_i))
}

/// `PLZ 01067 0351 1234567`: a label for another number that rules out the run at `first_i`
/// covers its first group only when that group is a whole number of four or more digits and
/// the rest starts afresh with a trunk `0` after a space. Returns where the rest starts, to be
/// scanned on its own. Four digits followed by four are a block of an IBAN or account number,
/// whose tail stays skipped.
fn labelled_head_end(chars: &[(usize, char)], c: &Candidate, first_i: usize) -> Option<usize> {
    let (&head, &next) = (c.groups.first()?, c.groups.get(1)?);
    let &(rest, _) = c.group_spans.get(1)?;
    let peel = head >= 4
        && !(head == 4 && next == 4)
        && c.gaps.first().is_some_and(|g| is_spaced(g))
        && chars.get(rest).is_some_and(|&(_, ch)| ch == '0')
        && labelled_as_another_number(chars, first_i);
    peel.then_some(rest)
}

/// Reads one candidate starting at char index `i`, looking at nothing at or past `limit`. On
/// success returns it and the index of the first char after it; on failure returns the index
/// to resume scanning from.
fn take_candidate(
    text: &str,
    chars: &[(usize, char)],
    i: usize,
    limit: usize,
) -> Result<(Candidate, usize), usize> {
    let at = |j: usize| chars.get(j).filter(|_| j < limit).map(|&(_, c)| c);
    let byte_at = |j: usize| chars.get(j).map_or(text.len(), |&(b, _)| b);

    let start = chars[i].0;
    let mut international = chars[i].1 == '+';
    let mut j = if international { i + 1 } else { i };
    let mut groups: Vec<u8> = Vec::new();
    let mut group_spans: Vec<(usize, usize)> = Vec::new();
    let mut gaps: Vec<String> = Vec::new();
    // Separators read since the last group; they become a gap once the next group is read.
    let mut pending = String::new();
    let mut digits = String::new();
    let mut paren_used = false;
    let mut capped = false;

    // `(+49) 30 1234567`: the country code in brackets.
    if at(i) == Some('(') && at(i + 1) == Some('+') {
        let mut k = i + 2;
        while at(k).is_some_and(|c| c.is_ascii_digit()) {
            k += 1;
        }
        let n = k - (i + 2);
        if !(1..=3).contains(&n) || at(k) != Some(')') {
            return Err(i + 1);
        }
        international = true;
        digits.extend(chars[i + 2..k].iter().map(|&(_, c)| c));
        groups.push(n as u8);
        group_spans.push((i, k + 1));
        paren_used = true;
        j = k + 1;
        let run_start = j;
        while at(j).is_some_and(|c| is_sep(c) && c != '(' && c != ')') {
            j += 1;
        }
        if j - run_start > MAX_SEPARATOR_RUN || !at(j).is_some_and(|c| c.is_ascii_digit()) {
            return Err(i + 1);
        }
        pending.extend(chars[run_start..j].iter().map(|&(_, c)| c));
    }
    // Index of the first char after the last accepted group; the span ends here.
    let mut end_i = j;

    loop {
        let paren_allowed = !paren_used
            && ((groups.is_empty() && !international)
                || paren_may_follow(&groups, &digits, international));
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
            if digits.len() + n > MAX_RUN_DIGITS {
                capped = true;
                break;
            }
            // "+44 (0)20 ...": the bracketed trunk prefix is dialled only inside the country.
            let trunk = international && n == 1 && at(j + 1) == Some('0');
            if !trunk {
                digits.extend(chars[j + 1..k].iter().map(|&(_, c)| c));
                if !groups.is_empty() {
                    gaps.push(std::mem::take(&mut pending));
                }
                groups.push(n as u8);
                group_spans.push((j, k + 1));
            }
            paren_used = true;
            j = k + 1;
        } else if at(j).is_some_and(|c| c.is_ascii_digit()) {
            let mut k = j;
            while at(k).is_some_and(|c| c.is_ascii_digit()) {
                k += 1;
            }
            if digits.len() + (k - group_start) > MAX_RUN_DIGITS {
                capped = true;
                break;
            }
            j = k;
            digits.extend(chars[group_start..j].iter().map(|&(_, c)| c));
            if !groups.is_empty() {
                gaps.push(std::mem::take(&mut pending));
            }
            groups.push((j - group_start) as u8);
            group_spans.push((group_start, j));
        } else {
            break;
        }
        end_i = j;
        if groups.len() == MAX_RUN_GROUPS {
            capped = true;
            break;
        }

        let run_start = j;
        while at(j).is_some_and(|c| is_sep(c) && c != '(' && c != ')') {
            j += 1;
        }
        // `030 / 1234567` and `030 - 1234567`: the DIN 5008 slash between area code and
        // number, and the spaced hyphen used the same way.
        let spaced_mark = [" / ", " - "]
            .iter()
            .any(|m| chars[run_start..j].iter().map(|&(_, c)| c).eq(m.chars()));
        if j - run_start > MAX_SEPARATOR_RUN && !spaced_mark {
            break;
        }
        let next = at(j);
        let continues = next.is_some_and(|c| c.is_ascii_digit())
            || (next == Some('(')
                && !paren_used
                && paren_may_follow(&groups, &digits, international));
        if !continues {
            break;
        }
        pending.extend(chars[run_start..j].iter().map(|&(_, c)| c));
    }
    let resume = end_i.max(i + 1);

    // `212-555-0147x204`: the extension is left outside the span.
    let extension = chars.get(end_i).is_some_and(|&(_, c)| c == 'x' || c == 'X')
        && chars
            .get(end_i + 1)
            .is_some_and(|&(_, c)| c.is_ascii_digit());
    let glued = !extension
        && text
            .as_bytes()
            .get(byte_at(end_i))
            .is_some_and(|b| b.is_ascii_alphanumeric());
    if glued {
        // `030 1234567 2nd floor`: a word glued to the last group after a space takes that
        // group with it, and the groups before still stand.
        if !gaps.last().is_some_and(|g| is_spaced(g)) {
            return Err(resume);
        }
        gaps.pop();
        group_spans.pop();
        let dropped = groups.pop().map_or(0, usize::from);
        digits.truncate(digits.len().saturating_sub(dropped));
        end_i = group_spans.last().map_or(end_i, |&(_, e)| e);
    }

    let count = digits.len();
    let allowed = if international {
        8..=MAX_RUN_DIGITS
    } else {
        7..=MAX_RUN_DIGITS
    };
    if !allowed.contains(&count) {
        return Err(resume);
    }
    let end = byte_at(end_i);

    let (dial, international) = match international_digits(&digits, international, &groups) {
        Some(rest) => (format!("+{rest}"), true),
        None => (digits, false),
    };
    let candidate = Candidate {
        start,
        end,
        dial,
        international,
        groups,
        group_spans,
        gaps,
        phone_label: false,
        capped,
    };
    Ok((candidate, end_i))
}

/// The digits after the international prefix when a run's digits are dialled from abroad, or
/// `None` for a national number. After a `+` that is all of them. A leading `00` is the
/// international call prefix when more digits share its group (`0049 30 ...`) or it stands
/// alone before the rest (`00 49 30 ...`), with at least ten digits in all.
fn international_digits<'d>(digits: &'d str, plus: bool, groups: &[u8]) -> Option<&'d str> {
    let spaced_00 = groups.first() == Some(&2) && groups.len() > 1;
    if plus {
        Some(digits)
    } else if (groups.first().is_some_and(|&g| g > 2) || spaced_00) && digits.len() >= 10 {
        digits.strip_prefix("00")
    } else {
        None
    }
}

/// Whether a bracketed group may come next: after the country code (`+49 (030) ...`), or
/// after a national area code in the Japanese style (`03(1234)5678`).
fn paren_may_follow(groups: &[u8], digits: &str, international: bool) -> bool {
    groups.len() == 1 && (international || digits.starts_with('0'))
}

fn is_hard_negative(chars: &[(usize, char)], c: &Candidate, first_i: usize, next_i: usize) -> bool {
    let all_gaps =
        |set: &[&str]| !c.gaps.is_empty() && c.gaps.iter().all(|g| set.contains(&g.as_str()));

    let ip =
        c.groups.len() == 4 && c.groups.iter().all(|&g| (1..=3).contains(&g)) && all_gaps(&["."]);
    if ip {
        return true;
    }
    // `0.25`, `0.123 4567`: a leading zero before a point is a decimal.
    let decimal = c.groups.first() == Some(&1)
        && c.dial.trim_start_matches('+').starts_with('0')
        && c.gaps.first().is_some_and(|g| g.starts_with('.'));
    if decimal {
        return true;
    }
    if is_time_or_date_range(c) {
        return true;
    }
    // One slash after an area code is a phone (`030/1234567`); two are a file or tax number,
    // unless they list whole numbers side by side.
    if c.gaps.iter().filter(|g| g.contains('/')).count() > 1 && !is_slash_list(c) {
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
    // `Firmware 02.10.2024.01`: a dated build or version number.
    let dated_version = !c.international
        && c.groups[..3.min(c.groups.len())].contains(&4)
        && c.gaps.iter().take(3).all(|g| g == ".");
    let date = date_shape && (c.groups.len() == 3 || hour_follows || dated_version);
    if date {
        return true;
    }

    let at = |j: Option<usize>| j.and_then(|j| chars.get(j)).map(|&(_, ch)| ch);
    if !c.international && part_of_an_identifier(chars, first_i) {
        return true;
    }
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

/// `030 1234567 / 0171 1234567 / 0172 1234567`: every slash is a spaced ` / ` and every stretch
/// between them has the seven or more digits of a whole number, bar the last of a run cut
/// short at a cap.
fn is_slash_list(c: &Candidate) -> bool {
    let mut digits = c.groups.first().map_or(0, |&g| usize::from(g));
    for (gap, &g) in c.gaps.iter().zip(c.groups.iter().skip(1)) {
        if gap.contains('/') {
            if gap != " / " || digits < 7 {
                return false;
            }
            digits = 0;
        }
        digits += usize::from(g);
    }
    digits >= 7 || c.capped
}

/// `08.00-12.00`, `0800–1200`, `01.01.26-31.12.26`, `01.09.-30.09.2026`, `01.01.2026
/// 31.12.2026`: segments split at a dash that are each a time (`HH.MM`, `HHMM`) or a day and
/// month with an optional year, or full dates split at spaces.
fn is_time_or_date_range(c: &Candidate) -> bool {
    let dash = |g: &str| g.contains(['-', '\u{2010}', '\u{2013}']);
    let space = |g: &str| g.chars().all(char::is_whitespace);
    // Every shape below has groups of at most four digits, and all but `HHMM-HHMM` join some
    // of them with a point.
    if c.groups.len() < 2
        || !c.gaps.iter().any(|g| dash(g) || space(g))
        || c.groups.iter().any(|&g| g > 4)
        || (c.groups.len() > 2 && !c.gaps.iter().any(|g| g.starts_with('.')))
    {
        return false;
    }
    // Saturating: a group of up to thirty digits must not overflow, and any value that
    // large fails the day, month, and time bounds below anyway.
    let digits: Vec<u32> = {
        let mut d = c.dial.trim_start_matches('+').chars();
        c.groups
            .iter()
            .map(|&n| {
                (0..n)
                    .filter_map(|_| d.next().and_then(|ch| ch.to_digit(10)))
                    .fold(0u32, |acc, x| acc.saturating_mul(10).saturating_add(x))
            })
            .collect()
    };
    let hhmm = |v: u32| v / 100 <= 24 && v % 100 < 60;
    if let ([4, 4], [a, b]) = (&c.groups[..], &digits[..]) {
        return dash(&c.gaps[0]) && hhmm(*a) && hhmm(*b);
    }
    let mut segments: Vec<Vec<(u8, u32)>> = vec![Vec::new()];
    let mut split_at_space = false;
    for (k, (&g, &v)) in c.groups.iter().zip(&digits).enumerate() {
        if k > 0 {
            let gap = c.gaps[k - 1].as_str();
            if dash(gap) || space(gap) {
                split_at_space |= !dash(gap);
                segments.push(Vec::new());
            } else if !gap.starts_with('.') {
                return false;
            }
        }
        if let Some(seg) = segments.last_mut() {
            seg.push((g, v));
        }
    }
    let time = |s: &[(u8, u32)]| matches!(s, [(1 | 2, h), (2, m)] if *h <= 24 && *m < 60);
    let day_month = |d: &u32, m: &u32| (1..=31).contains(d) && (1..=12).contains(m);
    let full_date = |s: &[(u8, u32)]| match s {
        [(1 | 2, d), (1 | 2, m), (2 | 4, _)] => day_month(d, m),
        _ => false,
    };
    let date = |s: &[(u8, u32)]| match s {
        [(1 | 2, d), (1 | 2, m)] => day_month(d, m),
        _ => full_date(s),
    };
    segments.len() >= 2
        && if split_at_space {
            segments.iter().all(|s| full_date(s))
        } else {
            segments.iter().all(|s| time(s) || date(s))
        }
}

/// Whether the digits at `first_i` continue an identifier rather than start a number: labelled
/// as another kind of number (`Kd.-Nr. ...`), glued to a word by a hyphen (`INV-2026-00871`,
/// `RE-2026-004417`), after an IBAN head (`DE89 3704 ...`, `NL91 ABNA 0417 ...`), or after a
/// word mixing letters and digits that itself follows a group with a digit (`1Z 999 AA1 01
/// 2345 6784`), unless those two groups are a postcode.
fn part_of_an_identifier(chars: &[(usize, char)], first_i: usize) -> bool {
    let at = |j: usize| chars.get(j).map(|&(_, ch)| ch);
    if labelled_as_another_number(chars, first_i) {
        return true;
    }
    if first_i >= 2
        && at(first_i - 1) == Some('-')
        && at(first_i - 2).is_some_and(|c| c.is_alphanumeric())
    {
        return true;
    }
    if first_i < 2 || at(first_i - 1) != Some(' ') {
        return false;
    }
    let word_before = |end: usize| {
        let mut j = end;
        while j > 0 && at(j - 1).is_some_and(char::is_alphanumeric) {
            j -= 1;
        }
        (
            j,
            chars[j..end].iter().map(|&(_, c)| c).collect::<Vec<char>>(),
        )
    };
    let iban_head = |w: &[char]| {
        w.len() == 4
            && w[..2].iter().all(char::is_ascii_uppercase)
            && w[2..].iter().all(char::is_ascii_digit)
    };
    let (w_start, word) = word_before(first_i - 1);
    let space_before_word = w_start >= 2 && at(w_start - 1) == Some(' ');
    // `NL91 ABNA 0417 ...`: a bank code of letters only sits between the head and the digits.
    let bank_code = word.len() == 4 && word.iter().all(char::is_ascii_uppercase);
    if bank_code && space_before_word && iban_head(&word_before(w_start - 1).1) {
        return true;
    }
    let mixed = word.iter().any(|c| c.is_alphabetic()) && word.iter().any(char::is_ascii_digit);
    if !mixed {
        return false;
    }
    if iban_head(&word) {
        return true;
    }
    if !space_before_word {
        return false;
    }
    let (_, prev) = word_before(w_start - 1);
    let id_like = prev.iter().any(char::is_ascii_digit);
    id_like && !is_postcode_pair(&prev, &word)
}

/// A GB (`NW1 6XE`, `L3 4BN`), Canadian (`K1A 0B1`) or Irish (`D02 X285`) postcode written as
/// two groups.
fn is_postcode_pair(outward: &[char], inward: &[char]) -> bool {
    let shape = |s: &[char]| -> String {
        s.iter()
            .map(|c| {
                if c.is_ascii_digit() {
                    '9'
                } else if c.is_ascii_uppercase() {
                    'A'
                } else {
                    '?'
                }
            })
            .collect()
    };
    let (o, i) = (shape(outward), shape(inward));
    let gb = matches!(o.as_str(), "A9" | "A99" | "A9A" | "AA9" | "AA99" | "AA9A") && i == "9AA";
    let ca = o == "A9A" && i == "9A9";
    let ie = matches!(o.as_str(), "A99" | "A9A") && i.len() == 4 && !i.contains('?');
    gb || ca || ie
}

/// The kinds of the two words before `first_i`, nearest first. Words are read back over
/// spaces and `.:#/-` so that `Order No. ` and `Kd.-Nr. ` are both seen. A two-letter code
/// after a label for another number stays with it: `BTW BE 0123.456.789`.
fn labels_before(chars: &[(usize, char)], first_i: usize) -> [Label; 2] {
    let word_char = |c: char| c.is_alphabetic() || matches!(c, '°' | 'º' | '№');
    let mut out = [Label::Other; 2];
    let mut country_code = false;
    let mut j = first_i;
    for k in 0..out.len() {
        while j > 0
            && matches!(
                chars[j - 1].1,
                ' ' | '.' | ':' | '#' | '/' | '-' | '\u{A0}' | '：' | '\u{3000}'
            )
        {
            j -= 1;
        }
        let end = j;
        while j > 0 && word_char(chars[j - 1].1) {
            j -= 1;
        }
        if j == end {
            break;
        }
        let word = &chars[j..end];
        let lower: String = word.iter().flat_map(|&(_, c)| c.to_lowercase()).collect();
        let dotted = chars.get(end).is_some_and(|&(_, c)| c == '.');
        out[k] = label_of(&lower, dotted);
        if k == 0 {
            country_code = out[0] == Label::Other
                && word.len() == 2
                && word.iter().all(|&(_, c)| c.is_ascii_uppercase());
        }
    }
    if country_code && out[1] == Label::NotPhone {
        out = [Label::NotPhone, Label::Other];
    }
    out
}

/// Whether the words before `first_i` label the number as something other than a phone.
fn labelled_as_another_number(chars: &[(usize, char)], first_i: usize) -> bool {
    match labels_before(chars, first_i) {
        [Label::NotPhone, _] => true,
        [Label::Marker, Label::Phone] => false,
        [Label::Marker, _] => true,
        _ => false,
    }
}

/// Whether the word right before `first_i` labels the number as a phone.
fn labelled_as_phone(chars: &[(usize, char)], first_i: usize) -> bool {
    matches!(
        labels_before(chars, first_i),
        [Label::Phone, _] | [Label::Marker, Label::Phone]
    )
}

#[cfg(feature = "phone-metadata")]
use super::phone_tables::Region;

/// Without the metadata no region can be hinted.
#[cfg(not(feature = "phone-metadata"))]
enum Region {}

/// The supported regions among `country_hint`, in order, looked up once per scan.
fn hinted_regions(country_hint: &[&str]) -> Vec<&'static Region> {
    #[cfg(feature = "phone-metadata")]
    let regions = country_hint
        .iter()
        .filter_map(|hint| {
            super::phone_tables::REGIONS
                .iter()
                .find(|r| r.id.eq_ignore_ascii_case(hint))
        })
        .collect();
    #[cfg(not(feature = "phone-metadata"))]
    let regions = {
        let _ = country_hint;
        Vec::new()
    };
    regions
}

/// Every validated phone number in `text`, sorted by start, non-overlapping.
pub fn scan(text: &str, country_hint: &[&str]) -> Vec<Entity> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let hints = hinted_regions(country_hint);
    let mut out = Vec::new();
    let mut i = 0;
    // Where a capped run said to read on, and whether a phone label came before that run.
    let mut carried: Option<(usize, bool)> = None;
    while let Some((mut c, first_i, next_i)) = next_candidate(text, &chars, i) {
        if let Some((at, labelled)) = carried.take()
            && at == first_i
        {
            c.phone_label |= labelled;
        }
        let labelled = c.phone_label;
        let whole = if c.capped { None } else { validate(&c, &hints) };
        let (parts, resume) = match whole {
            Some(v) => (vec![(c, v)], None),
            None => split(text, &chars, &c, &hints),
        };
        for (part, v) in parts {
            let (part, v) = without_short_tail(text, &chars, part, v, &hints);
            out.push(entity(&part, v));
        }
        i = resume.unwrap_or(next_i);
        carried = resume.map(|at| (at, labelled));
    }
    out
}

/// Confidence, E.164 form, and region of a validated candidate.
type Validated = (f32, String, Option<String>);

fn entity(c: &Candidate, (confidence, normalized, region): Validated) -> Entity {
    Entity {
        kind: Kind::Phone,
        start: c.start,
        end: c.end,
        confidence,
        review_recommended: false,
        source: Source::Rules,
        components: Vec::new(),
        normalized: Some(normalized),
        region,
    }
}

/// Groups `first..past_last` of `c`, read and checked on their own, or `None` when that read
/// does not cover exactly those groups. A phone label before the run carries over to every
/// part of it: `Tel 2125550147 2125550148`.
fn reread(
    text: &str,
    chars: &[(usize, char)],
    c: &Candidate,
    first: usize,
    past_last: usize,
) -> Option<Checked> {
    let from = if first == 0 {
        chars.partition_point(|&(b, _)| b < c.start)
    } else {
        c.group_spans.get(first)?.0
    };
    let limit = c.group_spans.get(past_last.checked_sub(1)?)?.1;
    match checked_candidate(text, chars, from, limit) {
        Ok((Checked::Open(mut p), next)) if next == limit => {
            p.phone_label |= c.phone_label;
            Some(Checked::Open(p))
        }
        Ok((ruled_out @ Checked::RuledOut(_), next)) if next == limit => Some(ruled_out),
        _ => None,
    }
}

/// `Tel. 030 1234567 10 Uhr`: one or two digits after a space, following a longer group, and
/// followed in turn by a word (`10 Uhr`, `24 hours`, `10 h`) are the number of that word, so
/// when the run without them validates too, that shorter reading wins. Without the word they
/// are the end of the phone: `030 12345 67`, `089 123456 78`. Numbers grouped in pairs
/// (`0470 12 34 56`) end in a pair after a pair and stay whole.
fn without_short_tail(
    text: &str,
    chars: &[(usize, char)],
    c: Candidate,
    v: Validated,
    hints: &[&'static Region],
) -> (Candidate, Validated) {
    let n = c.groups.len();
    let mut after = text.get(c.end..).unwrap_or_default().chars();
    let word_follows = after.next().is_some_and(|ch| ch == ' ' || ch == '\u{A0}')
        && after
            .find(|&ch| ch != ' ' && ch != '\u{A0}')
            .is_some_and(char::is_alphabetic);
    let short_tail = n >= 2
        && c.groups[n - 1] <= 2
        && c.groups[n - 2] >= 3
        && c.gaps.last().is_some_and(|g| is_spaced(g))
        && word_follows;
    if short_tail
        && let Some(Checked::Open(p)) = reread(text, chars, &c, 0, n - 1)
        && let Some(pv) = validate(&p, hints)
    {
        return (p, pv);
    }
    (c, v)
}

/// How a piece of a run reads on its own.
enum Piece {
    Valid(Candidate, Validated),
    /// Validates by its digits alone and cannot be a hard negative, so it is read in full
    /// only once a cutting keeps it. `anchored` and `unspaced` are what `anchored` and the
    /// space check would find on the read candidate.
    Unread {
        confidence: f32,
        anchored: bool,
        unspaced: bool,
    },
    /// A hard negative, such as the date in `12.03.2024 030 1234567`.
    RuledOut,
    /// A house number, hour or count of up to four digits (`Room 12`, `24 hours`), or a year
    /// range (`1998-2024`).
    Short,
    Nothing,
}

/// Reads groups `first..past_last` of `c` in full as a piece.
fn read_piece(
    text: &str,
    chars: &[(usize, char)],
    c: &Candidate,
    first: usize,
    past_last: usize,
    hints: &[&'static Region],
) -> Piece {
    let count: usize = c
        .groups
        .get(first..past_last)
        .unwrap_or_default()
        .iter()
        .map(|&g| usize::from(g))
        .sum();
    // Fewer than seven digits never read as a candidate at all.
    let read = if count >= 7 {
        reread(text, chars, c, first, past_last)
    } else {
        None
    };
    match read {
        Some(Checked::Open(p)) => {
            if let Some(v) = validate(&p, hints) {
                return Piece::Valid(p, v);
            }
        }
        Some(Checked::RuledOut(_)) => return Piece::RuledOut,
        None => {}
    }
    if is_short(chars, c, first, past_last) {
        Piece::Short
    } else {
        Piece::Nothing
    }
}

/// Whether groups `first..past_last` of `c` could be a hard negative on their own when the
/// run as a whole is not. Inside such a run that needs a gap other than spaces (dates, times,
/// IPs, decimals, file numbers) or one long unseparated group: a label, currency sign or
/// identifier next to a piece is next to the whole run too, and rules it out as well.
fn may_rule_out(c: &Candidate, first: usize, past_last: usize) -> bool {
    let groups = c.groups.get(first..past_last).unwrap_or_default();
    let marked = c
        .gaps
        .get(first..past_last.saturating_sub(1))
        .unwrap_or_default()
        .iter()
        .any(|g| !g.chars().all(char::is_whitespace));
    let long = matches!(groups, [g] if usize::from(*g) > MAX_UNSEPARATED_DIGITS);
    marked || long
}

/// The confidence of a piece of a run with these `digits` in these `groups`, taken from the
/// dial a reread would give, before paying for the reread and its hard negatives, and whether
/// that dial is anchored by a country code or trunk `0`. `plus` is whether the piece starts with
/// the run's `+`, and `labelled` whether a phone label comes before the run.
fn dial_reading(
    digits: &str,
    groups: &[u8],
    plus: bool,
    labelled: bool,
    hints: &[&'static Region],
) -> Option<(f32, bool)> {
    // An E.164 number has at most fifteen digits; written out it may add a `00` prefix and a
    // trunk `0` after the country code.
    if !(7..=MAX_E164_DIGITS + 3).contains(&digits.len()) {
        return None;
    }
    let bare = groups.len() == 1 && !labelled;
    match international_digits(digits, plus, groups) {
        Some(rest) => dial_confidence(rest, true, bare, hints).map(|v| (v, true)),
        None => dial_confidence(digits, false, bare, hints).map(|v| (v, digits.starts_with('0'))),
    }
}

/// Whether groups `first..past_last` of `c` are one group of at most four digits, or two
/// years joined by a dash with the earlier first.
fn is_short(chars: &[(usize, char)], c: &Candidate, first: usize, past_last: usize) -> bool {
    match c.groups.get(first..past_last) {
        Some([g]) => *g <= 4,
        _ => is_year(chars, c, first, past_last),
    }
}

/// Whether groups `first..past_last` of `c` are a year (`2019`) or a range of years joined by
/// a dash with the earlier first (`1998-2024`).
fn is_year(chars: &[(usize, char)], c: &Candidate, first: usize, past_last: usize) -> bool {
    let value = |k: usize| -> Option<u32> {
        let &(s, e) = c.group_spans.get(k)?;
        let digits = chars
            .get(s..e)?
            .iter()
            .filter_map(|&(_, ch)| ch.to_digit(10));
        Some(digits.fold(0, |acc, d| acc * 10 + d))
    };
    let year = |k: usize| value(k).is_some_and(|v| (1900..=2099).contains(&v));
    let dash = |k: usize| {
        c.gaps
            .get(k)
            .is_some_and(|g| matches!(g.as_str(), "-" | "\u{2010}" | "\u{2013}"))
    };
    match c.groups.get(first..past_last) {
        Some([4]) => year(first),
        Some([4, 4]) => {
            dash(first) && year(first) && year(first + 1) && value(first) <= value(first + 1)
        }
        _ => false,
    }
}

/// Whether a piece carries its own sign of being a phone: a country code, a trunk `0`, or a
/// phone label before it or before its run.
fn anchored(p: &Candidate) -> bool {
    p.international || p.dial.starts_with('0') || labelled_in_shape(p.phone_label, &p.groups)
}

/// Whether a phone label anchors a piece with these groups. A label alone vouches only for a
/// grouping some country writes without a prefix: the NANP `212 555 0147` or `2125550147`.
/// `Call 12 345 678 901 234` has a label, but no country groups a number 2-3-3-3.
fn labelled_in_shape(labelled: bool, groups: &[u8]) -> bool {
    labelled && matches!(groups, [3, 3, 4] | [10])
}

/// Numbers side by side read as one run (`030 1234567 030 1234568`, `Tel. 030 1234567 / 0171
/// 1234567 / 0172 1234567`), or a number next to a postcode, house number, hour or date (`NY
/// 10118 212-555-0147`, `Call 020 7946 0321 24 hours`). When the run fails validation it is cut
/// at its space and ` / ` gaps into pieces of at most one number's groups, each checked
/// afresh, and the cutting whose kept pieces validate best is chosen.
///
/// Dropping a piece needs a reason to trust the cuts around it: the piece is a hard negative;
/// or it is short and at an edge of the run, or one of a row of years running to its end, and
/// every kept piece is anchored by a country
/// code, a trunk `0` or a phone label; or every kept piece is grouped without spaces, so a
/// space next to one cannot be one of its own gaps. Otherwise `12 345 678 901 234` would
/// yield a phone.
///
/// A run cut short at a cap (`Candidate::capped`) is not validated whole, and no piece may end
/// at the cap. Past the first cut close enough to the cap to hold the head of a number going
/// on past it, the cutting may stop anywhere; only pieces before that cut which the cutting
/// goes on past are kept. The char index to read on from is returned: the end of the last
/// piece kept, or else that cut.
fn split(
    text: &str,
    chars: &[(usize, char)],
    c: &Candidate,
    hints: &[&'static Region],
) -> (Vec<(Candidate, Validated)>, Option<usize>) {
    // Card numbers and account numbers in blocks of four fail as a whole and must not be
    // mined for a phone-shaped piece; a capped one is skipped to the cap.
    if c.groups.len() >= 4 && c.groups.iter().all(|&g| g == 4) {
        return (Vec::new(), None);
    }
    let borders = c
        .gaps
        .iter()
        .enumerate()
        .filter(|(_, g)| is_spaced(g) || g.as_str() == " / ")
        .map(|(k, _)| k + 1);
    let cuts: Vec<usize> = std::iter::once(0)
        .chain(borders)
        .chain(std::iter::once(c.groups.len()))
        .collect();
    let m = cuts.len();
    let offsets: Vec<usize> = std::iter::once(0)
        .chain(c.groups.iter().scan(0, |at, &g| {
            *at += usize::from(g);
            Some(*at)
        }))
        .collect();
    // The first cut after which a capped run holds no more than one number's groups and
    // digits: a piece from there on may go on past the cap.
    let n = c.groups.len();
    let near_cap = (1..m)
        .find(|&x| {
            n - cuts[x] <= MAX_GROUPS && offsets[n] - offsets[cuts[x]] <= MAX_E164_DIGITS + 3
        })
        .unwrap_or(m - 1);
    let read_on = |x: usize| {
        if !c.capped || x >= m - 1 {
            None
        } else {
            c.group_spans.get(cuts[x]).map(|&(s, _)| s)
        }
    };
    // Every piece `(x, y)` from cut `x` to cut `y`, short of the whole run, which already
    // failed, and of the cap, ordered by where it ends so that each is reached after all that
    // end before it.
    let spans: Vec<(usize, usize)> = (1..m)
        .flat_map(|y| (0..y).map(move |x| (x, y)))
        .filter(|&(x, y)| {
            cuts[y] - cuts[x] <= MAX_GROUPS && (x, y) != (0, m - 1) && !(c.capped && y == m - 1)
        })
        .collect();
    let first_i = chars.partition_point(|&(b, _)| b < c.start);
    let plus = text
        .get(c.start..)
        .is_some_and(|t| t.starts_with('+') || t.starts_with("(+"));
    // A run made international by a `00` prefix skips the identifier checks, which its first
    // pieces, too short for the prefix, do not.
    let head_may_rule_out = c.international && !plus && part_of_an_identifier(chars, first_i);
    // The run's digits as written, a `00` prefix kept, and where each group starts in them.
    let digits: String = c
        .group_spans
        .iter()
        .filter_map(|&(s, e)| chars.get(s..e))
        .flatten()
        .map(|&(_, ch)| ch)
        .filter(char::is_ascii_digit)
        .collect();
    let dialled: Vec<Option<(f32, bool)>> = spans
        .iter()
        .map(|&(x, y)| {
            let (first, past_last) = (cuts[x], cuts[y]);
            let piece_digits = digits.get(offsets[first]..offsets[past_last])?;
            let groups = c.groups.get(first..past_last)?;
            let plus = plus && first == 0;
            dial_reading(piece_digits, groups, plus, c.phone_label, hints)
        })
        .collect();
    if dialled.iter().all(Option::is_none) {
        return (Vec::new(), read_on(near_cap));
    }
    let mut pieces: Vec<Piece> = spans
        .iter()
        .zip(dialled)
        .map(|(&(x, y), reading)| {
            let (first, past_last) = (cuts[x], cuts[y]);
            let ruled_out = may_rule_out(c, first, past_last) || (x == 0 && head_may_rule_out);
            match reading {
                Some((confidence, dial_anchored)) if !ruled_out => Piece::Unread {
                    confidence,
                    anchored: dial_anchored
                        || labelled_in_shape(
                            c.phone_label,
                            c.groups.get(first..past_last).unwrap_or_default(),
                        ),
                    unspaced: !c
                        .gaps
                        .get(first..past_last - 1)
                        .unwrap_or_default()
                        .iter()
                        .any(|g| is_spaced(g)),
                },
                None if !ruled_out => {
                    if is_short(chars, c, first, past_last) {
                        Piece::Short
                    } else {
                        Piece::Nothing
                    }
                }
                _ => read_piece(text, chars, c, first, past_last, hints),
            }
        })
        .collect();

    // A cutting is chosen on the readings so far; the unread pieces it keeps are then read,
    // and should one fail, the cutting is chosen again with what is now known. A capped run
    // is cut up to somewhere past `near_cap`, and keeps only the pieces that end before it
    // and that the cutting goes on past, so that each kept piece was judged with what
    // follows it; the rest is read again as the next run.
    let min_end = if c.capped { near_cap } else { m - 1 };
    // A short piece may be dropped at an edge of the run, and so may each of a row of years
    // that runs to its end: `Tel. 030 1234567 2019 2020 2021`.
    let mut years_to_end = vec![true; m];
    for x in (0..m - 1).rev() {
        years_to_end[x] = years_to_end[x + 1] && is_year(chars, c, cuts[x], cuts[x + 1]);
    }
    let at_edge: Vec<bool> = spans
        .iter()
        .map(|&(x, y)| x == 0 || y == m - 1 || years_to_end[x])
        .collect();
    loop {
        let mut path = best_cutting(&spans, &pieces, &at_edge, m, min_end);
        if c.capped {
            path.pop();
            path.retain(|&k| spans.get(k).is_some_and(|&(_, y)| y <= near_cap));
        }
        let mut settled = true;
        for &k in &path {
            if let (Some(Piece::Unread { .. }), Some(&(x, y))) = (pieces.get(k), spans.get(k)) {
                let read = read_piece(text, chars, c, cuts[x], cuts[y], hints);
                settled &= matches!(read, Piece::Valid(..));
                pieces[k] = read;
            }
        }
        if settled {
            let resume = match path.last().and_then(|&k| spans.get(k)) {
                Some(&(_, y)) => read_on(y),
                None => read_on(near_cap),
            };
            let parts = path
                .iter()
                .filter_map(
                    |&k| match std::mem::replace(pieces.get_mut(k)?, Piece::Nothing) {
                        Piece::Valid(p, v) => Some((p, v)),
                        _ => None,
                    },
                )
                .collect();
            return (parts, resume);
        }
    }
}

/// The indices into `spans` of the pieces, kept and dropped, of the best cutting of a run with
/// `m` cuts, in order, or none when no cutting keeps a piece for a trusted reason. `at_edge`
/// tells for each piece whether a short one may be dropped there. The cutting ends at the last
/// cut, or for a run cut short at a cap, at any cut from `min_end` on: the furthest one of its
/// best score. Between cuttings of the same score, one that drops a short piece wins over one
/// that reads it into the number before (`030 1234567 2019`).
fn best_cutting(
    spans: &[(usize, usize)],
    pieces: &[Piece],
    at_edge: &[bool],
    m: usize,
    min_end: usize,
) -> Vec<usize> {
    let mut best_path: Vec<usize> = Vec::new();
    let mut best_score = 0.0;
    // Stricter reasons come first and keep a tie.
    for (short, nothing) in [(false, false), (true, false), (false, true), (true, true)] {
        let keep = |anchored: bool, unspaced: bool| (!short || anchored) && (!nothing || unspaced);
        // `reach[y]`: the best score of a cutting up to cut `y`, and its last piece.
        let mut reach: Vec<Option<(f32, Option<usize>)>> = vec![None; m];
        reach[0] = Some((0.0, None));
        for (k, ((&(x, y), p), &edge)) in spans.iter().zip(pieces).zip(at_edge).enumerate() {
            let Some((base, _)) = reach[x] else {
                continue;
            };
            let gain = match p {
                Piece::Valid(p, v) => {
                    let unspaced = !p.gaps.iter().any(|g| is_spaced(g));
                    if !keep(anchored(p), unspaced) {
                        continue;
                    }
                    v.0
                }
                Piece::Unread {
                    confidence,
                    anchored,
                    unspaced,
                } => {
                    if !keep(*anchored, *unspaced) {
                        continue;
                    }
                    *confidence
                }
                Piece::RuledOut => 0.0,
                Piece::Short if nothing || (short && edge) => 0.0,
                Piece::Nothing if nothing => 0.0,
                Piece::Short | Piece::Nothing => continue,
            };
            let short_drop = matches!(p, Piece::Short);
            if reach[y].is_none_or(|(s, _)| base + gain > s || (short_drop && base + gain == s)) {
                reach[y] = Some((base + gain, Some(k)));
            }
        }
        let mut end: Option<(f32, usize)> = None;
        for (y, reached) in reach.iter().enumerate().skip(min_end) {
            if let Some((score, _)) = *reached
                && end.is_none_or(|(s, _)| score >= s)
            {
                end = Some((score, y));
            }
        }
        let Some((score, y)) = end else {
            continue;
        };
        if score > best_score {
            best_score = score;
            best_path.clear();
            let mut last = reach[y].and_then(|(_, k)| k);
            while let Some(k) = last {
                best_path.push(k);
                last = spans
                    .get(k)
                    .and_then(|&(x, _)| reach[x])
                    .and_then(|(_, k)| k);
            }
        }
    }
    best_path.reverse();
    best_path
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

/// Digits a generated prefix spans at most.
#[cfg(feature = "phone-metadata")]
const PREFIX_DIGITS: usize = 4;

/// One bit for each start of `PREFIX_DIGITS` digits, set when one of `prefixes` matches it, or
/// `None` when a prefix is longer than that or holds something other than digits and `x`.
#[cfg(feature = "phone-metadata")]
fn prefix_set(prefixes: &[&str]) -> Option<Vec<u64>> {
    let starts = 10usize.pow(PREFIX_DIGITS as u32);
    let mut bits = vec![0u64; starts.div_ceil(64)];
    for p in prefixes {
        if p.len() > PREFIX_DIGITS {
            return None;
        }
        let mut matched = vec![0usize];
        for k in 0..PREFIX_DIGITS {
            let digits = match p.as_bytes().get(k) {
                None | Some(b'x') => 0..=9,
                Some(&d @ b'0'..=b'9') => usize::from(d - b'0')..=usize::from(d - b'0'),
                Some(_) => return None,
            };
            matched = matched
                .iter()
                .flat_map(|&m| digits.clone().map(move |d| m * 10 + d))
                .collect();
        }
        for m in matched {
            if let Some(word) = bits.get_mut(m / 64) {
                *word |= 1 << (m % 64);
            }
        }
    }
    Some(bits)
}

/// Whether `set` from `prefix_set` holds the start of `national`, or `None` when `national` is
/// too short to tell.
#[cfg(feature = "phone-metadata")]
fn in_prefix_set(set: &[u64], national: &str) -> Option<bool> {
    let head = national.get(..PREFIX_DIGITS)?;
    let start = head.bytes().try_fold(0usize, |acc, b| {
        b.is_ascii_digit().then(|| acc * 10 + usize::from(b - b'0'))
    })?;
    Some(set.get(start / 64)? & (1 << (start % 64)) != 0)
}

/// Whether one of `region`'s prefixes matches `national`. The lists run to hundreds of
/// prefixes and are checked for every piece of every run, so each is turned once into a set
/// of the starts it matches.
#[cfg(feature = "phone-metadata")]
fn has_prefix(region: &Region, national: &str) -> bool {
    use super::phone_tables::REGIONS;
    static SETS: std::sync::OnceLock<Vec<Option<Vec<u64>>>> = std::sync::OnceLock::new();

    let sets = SETS.get_or_init(|| REGIONS.iter().map(|r| prefix_set(r.prefixes)).collect());
    REGIONS
        .iter()
        .position(|r| std::ptr::eq(r, region))
        .and_then(|k| sets.get(k)?.as_deref())
        .and_then(|set| in_prefix_set(set, national))
        .unwrap_or_else(|| region.prefixes.iter().any(|p| matches_prefix(national, p)))
}

/// Valid when a known prefix matches at a possible length; possible when only an ordinary
/// fixed-line or mobile length fits. A length only special types use needs the prefix.
#[cfg(feature = "phone-metadata")]
fn score(region: &Region, national: &str) -> Option<f32> {
    const VALID: f32 = 0.99;
    const POSSIBLE: f32 = 0.75;
    let len = u8::try_from(national.len()).ok()?;
    let prefix = || has_prefix(region, national);
    if region.lengths.contains(&len) && prefix() {
        Some(VALID)
    } else if region.core_lengths.contains(&len) {
        Some(POSSIBLE)
    } else {
        None
    }
}

/// The best reading so far: score, region, and national number.
#[cfg(feature = "phone-metadata")]
type Reading<'d> = (f32, &'static Region, &'d str);

/// Digits in a calling code.
#[cfg(feature = "phone-metadata")]
fn code_width(code: u16) -> usize {
    code.checked_ilog10().map_or(1, |w| w as usize + 1)
}

/// Keep `region`'s reading of `national` when it scores higher than the best so far. Regions
/// sharing a calling code at the same score resolve to the main country, as libphonenumber does.
#[cfg(feature = "phone-metadata")]
fn consider<'d>(best: &mut Option<Reading<'d>>, region: &'static Region, national: &'d str) {
    if code_width(region.code) + national.len() > MAX_E164_DIGITS {
        return;
    }
    let Some(s) = score(region, national) else {
        return;
    };
    let better = best
        .as_ref()
        .is_none_or(|b| s > b.0 || (s == b.0 && region.main && !b.1.main));
    if better {
        *best = Some((s, region, national));
    }
}

fn validate(c: &Candidate, hints: &[&'static Region]) -> Option<Validated> {
    let bare = c.gaps.is_empty() && !c.phone_label;
    validate_dial(&c.dial, c.international, bare, hints)
}

/// Validates a dialled number, with or without its `+`; `bare` is a single unseparated group
/// with no phone label before it.
#[cfg(feature = "phone-metadata")]
fn validate_dial(
    dial: &str,
    international: bool,
    bare: bool,
    hints: &[&'static Region],
) -> Option<Validated> {
    let (score, region, national) = best_reading(dial, international, bare, hints)?;
    let e164 = format!("+{}{}", region.code, national);
    Some((score, e164, Some(region.id.to_string())))
}

/// The confidence `validate_dial` would give, without building the E.164 form.
#[cfg(feature = "phone-metadata")]
fn dial_confidence(
    dial: &str,
    international: bool,
    bare: bool,
    hints: &[&'static Region],
) -> Option<f32> {
    best_reading(dial, international, bare, hints).map(|(score, _, _)| score)
}

#[cfg(feature = "phone-metadata")]
fn best_reading<'d>(
    dial: &'d str,
    international: bool,
    bare: bool,
    hints: &[&'static Region],
) -> Option<Reading<'d>> {
    let mut best: Option<Reading<'d>> = None;
    readings(dial, international, bare, hints, |region, national| {
        consider(&mut best, region, national);
        // Among hinted regions the first valid reading wins.
        !international && best.as_ref().is_some_and(|b| b.0 >= 0.99)
    });
    best
}

/// `digits` after `prefix`, compared digit by digit: the prefixes are a digit or two, too
/// short for a call to `memcmp` to pay off.
#[cfg(feature = "phone-metadata")]
fn after_prefix<'d>(digits: &'d str, prefix: &str) -> Option<&'d str> {
    let matches =
        digits.len() >= prefix.len() && digits.bytes().zip(prefix.bytes()).all(|(d, p)| d == p);
    if matches {
        digits.get(prefix.len()..)
    } else {
        None
    }
}

/// Calls `visit` with every region a dialled number may belong to and its national number
/// there, until `visit` returns true: regions by calling code for an international number,
/// the hinted regions in order for a national one.
#[cfg(feature = "phone-metadata")]
fn readings<'d>(
    dial: &'d str,
    international: bool,
    bare: bool,
    hints: &[&'static Region],
    mut visit: impl FnMut(&'static Region, &'d str) -> bool,
) {
    use super::phone_tables::REGIONS;

    let digits = dial.trim_start_matches('+');
    if international {
        for region in REGIONS {
            let width = code_width(region.code);
            let code = digits.get(..width).and_then(|d| d.parse::<u16>().ok());
            let (Some(national), true) = (digits.get(width..), code == Some(region.code)) else {
                continue;
            };
            // `+49 (030) 2312 5456`, `+44 020 7946 0958`: the trunk prefix is dialled only
            // inside the country and is often written anyway. The NANP `1` never follows `+1`.
            let national = match region.national_prefix {
                Some(p) if region.code != 1 => after_prefix(national, p).unwrap_or(national),
                _ => national,
            };
            // After the trunk prefix, no national number starts with `0`.
            if !national.starts_with('0') && visit(region, national) {
                return;
            }
        }
    } else {
        for &region in hints {
            // A number written for a national reader carries its trunk prefix: `030 ...`,
            // `020 ...`. Without it, a hinted digit run is an order or ticket number, not a
            // phone. The NANP trunk `1` is the exception: it is usually left off.
            let national = match (region.code, region.national_prefix) {
                // An unseparated ten-digit run is an ID as often as a US number; it needs a
                // phone word before it (`Tel 2025550143`).
                (1, _) if bare => continue,
                (1, Some(p)) => after_prefix(digits, p).unwrap_or(digits),
                (_, Some(p)) => match after_prefix(digits, p) {
                    Some(n) => n,
                    // Georgian mobiles dropped the trunk `0` in 2011: `599 12 34 56`.
                    None if matches!(digits.as_bytes(), [b'5', ..] | [b'7', b'9', b'0', ..])
                        && region.id == "GE" =>
                    {
                        digits
                    }
                    None => continue,
                },
                (_, None) => digits,
            };
            if !national.starts_with('0') && visit(region, national) {
                return;
            }
        }
    }
}

#[cfg(not(feature = "phone-metadata"))]
fn dial_confidence(
    _dial: &str,
    _international: bool,
    _bare: bool,
    _hints: &[&'static Region],
) -> Option<f32> {
    None
}

#[cfg(not(feature = "phone-metadata"))]
fn validate_dial(
    _dial: &str,
    _international: bool,
    _bare: bool,
    _hints: &[&'static Region],
) -> Option<Validated> {
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
    fn identifiers_are_not_phones() {
        let hints = ["GB", "DE", "GE"];
        for text in [
            "Invoice 2026-0417",
            "(ticket 2026-0917-4471)",
            "invoice INV-2026-00871, total",
            "Nr. RE-2026-004417",
            "Kundennummer: K-10293",
            "tracking 1Z 999 AA1 01 2345 6784 has",
            "IBAN GB29 NWBK 6016 1331 9268 19, reference",
            "IBAN DE89 3704 0044 0532 0130 00 · BIC",
            "USt-IdNr. DE123456789 · HRB 187234 B",
            "order 48213 shipped 29.08.2026",
            "ISBN 0-306-40615-2",
            "serial SN 0451-2233-8891",
            "case no. 2024/1187/09",
            "routing 021000021",
            "Steuernummer 21/815/08150",
            "company no. 01234567",
        ] {
            assert!(
                scan(text, &hints).is_empty(),
                "{text}: {:?}",
                scan(text, &hints)
            );
        }
    }

    /// Number-shaped strings from invoices, forms and records that are not phones, scanned
    /// with every supported region hinted.
    #[test]
    fn negative_corpus_is_silent() {
        let hints = [
            "US", "CA", "GB", "DE", "GE", "JP", "NL", "BE", "AT", "CH", "IE",
        ];
        for text in [
            "Invoice 2026-0417",
            "Order No. 0000123456",
            "order #48213-B",
            "ISBN 978-3-16-148410-0",
            "ISBN 0-306-40615-2",
            "SSN 123-45-6789",
            "card 4111 1111 1111 1111",
            "exp 09/28 cvc 123",
            "coordinates 41.7151, 44.8271",
            "version 1.24.3",
            "v2.10.0-rc.1",
            "pallet 120x80 cm",
            "postcode 10115 Berlin",
            "NW1 6XE",
            "from 2019-2024",
            "pages 123-145",
            "pp. 1203-1245",
            "call at 09:30-11:00",
            "price 1.299,00 EUR",
            "EUR 1 234 567,89",
            "total 3,418.60",
            "USD 12,345,678",
            "population 8 982 000",
            "distance 12 345 km",
            "serial SN 0451-2233-8891",
            "case no. 2024/1187/09",
            "account 0012345678",
            "BLZ 370 400 44",
            "sort code 60-16-13",
            "routing 021000021",
            "EAN 4006381333931",
            "UPC 036000291452",
            "tracking 9400 1000 0000 0000 0000 00",
            "DHL 00340434161234567890",
            "Aktenzeichen 12 O 345/24",
            "Az. 3 U 12/23",
            "flight LH 2034 on 12.10.",
            "room 0123",
            "ext. 4471",
            "PIN 0000",
            "year 2026 week 38",
            "date 2026.09.23",
            "date 23.09.26",
            "date 09/23/2026",
            "2026-09-23T10:30:00",
            "1726000000 unix time",
            "ratio 16:9",
            "+/- 0.05",
            "temperature -12 C",
            "Steuernummer 21/815/08150",
            "HRB 187234 B",
            "USt-IdNr. DE123456789",
            "VAT GB 123 4567 89",
            "company no. 01234567",
            "EIN 12-3456789",
            "passport C01X00T47",
            "IP 192.168.100.200",
            "MAC 00:1A:2B:3C:4D:5E",
            "lat 51.5074 lon -0.1278",
        ] {
            assert!(
                scan(text, &hints).is_empty(),
                "{text}: {:?}",
                scan(text, &hints)
            );
        }
    }

    /// Numbers after labels, colons, postcodes and in national forms that must be found.
    #[cfg(feature = "phone-metadata")]
    #[test]
    fn labelled_and_national_forms_are_found() {
        let cases: &[(&str, &[&str])] = &[
            ("Customer service: 020 7946 0321", &["GB"]),
            ("Customer Services: 0345 600 1234", &["GB"]),
            ("Customer care: (800) 555-0199", &["US"]),
            ("Order line: 020 7946 0321", &["GB"]),
            ("Case enquiries: 020 7946 0321", &["GB"]),
            ("Account manager: 07700 900123", &["GB"]),
            ("Invoice queries 020 7946 0321", &["GB"]),
            ("ticket office: 020 7946 0321", &["GB"]),
            ("Policy enquiries 0800 123 4567", &["GB"]),
            ("Dr. Anna Az 030 1234567", &["DE"]),
            ("TEL:03-1234-5678 FAX:03-1234-5679", &["JP"]),
            ("Tel:+49 30 1234567", &["DE"]),
            ("Tel.:030 1234567", &["DE"]),
            ("T:+44 20 7946 0958", &[]),
            ("tel:+81-3-1234-5678", &[]),
            ("Kontakt:030 1234567", &["DE"]),
            ("599 12 34 56", &["GE"]),
            ("mob: 599 123 456", &["GE"]),
            ("Liverpool L3 4BN 0151 496 0147", &["GB"]),
            ("London NW1 6XE 020 7946 0321", &["GB"]),
            ("Suite 4B 020 7946 0321", &["GB"]),
            ("Tel. no. 020 7946 0321", &["GB"]),
            ("Tel 2025550143", &["US"]),
            ("call (202) 555-0143", &["US"]),
            ("00 49 30 1234567", &["DE"]),
            ("030 / 1234567", &["DE"]),
            ("電話番号 03-1234-5678", &["JP"]),
        ];
        for (text, hints) in cases {
            assert!(!scan(text, hints).is_empty(), "{text} with {hints:?}");
        }
    }

    /// Labelled identifiers, time and date ranges, bare integers and versions that must not be.
    #[test]
    fn labelled_identifiers_and_ranges_are_not_phones() {
        let cases: &[(&str, &[&str])] = &[
            ("Öffnungszeiten: 08.00-12.00 Uhr", &["DE"]),
            ("09.30-12.30 und 14.00-18.00", &["DE"]),
            ("07.30-16.30", &["GB"]),
            ("gültig 01.01.26-31.12.26", &["DE"]),
            ("vom 01.09.-30.09.2026", &["DE"]),
            ("SSN 078-05-1120", &["DE", "US"]),
            ("Vertragsnummer: 030 1234567", &["DE"]),
            ("Kd.-Nr. 0301234567", &["DE"]),
            ("Personalnummer 0301234567", &["DE"]),
            ("Art.-Nr. 030 123 4567", &["DE"]),
            ("Part No. 0301-234-567", &["DE"]),
            ("patient ID 030 1234 5678", &["DE"]),
            ("Kto. 0532013000", &["DE"]),
            ("注文番号 0312345678", &["JP"]),
            ("id 2147483647", &["US"]),
            ("4294967295", &["US"]),
            ("Chrome/128.0.6613.138", &["US", "DE"]),
            ("tracking 1Z 999 AA1 01 2345 6784 has", &["GB"]),
            ("IBAN DE89 3704 0044 0532 0130 00", &["DE"]),
        ];
        for (text, hints) in cases {
            assert!(
                scan(text, hints).is_empty(),
                "{text}: {:?}",
                scan(text, hints)
            );
        }
    }

    #[cfg(feature = "phone-metadata")]
    fn found<'t>(text: &'t str, hints: &[&str]) -> Vec<&'t str> {
        scan(text, hints)
            .iter()
            .map(|e| &text[e.start..e.end])
            .collect()
    }

    /// Adjacent numbers, bracketed country codes, new labels and markers, Japanese brackets,
    /// glued extensions, spaced hyphens and Eircodes, with the exact spans each must give.
    #[cfg(feature = "phone-metadata")]
    #[test]
    fn recall_cases() {
        let cases: &[(&str, &[&str], &[&str])] = &[
            (
                "030 1234567 030 1234568",
                &["DE"],
                &["030 1234567", "030 1234568"],
            ),
            (
                "Tel. 030 1234567 / 0171 1234567",
                &["DE"],
                &["030 1234567", "0171 1234567"],
            ),
            (
                "Tel: 212-555-0147 / 212-555-0148",
                &["US"],
                &["212-555-0147", "212-555-0148"],
            ),
            (
                "phone 212-555-0147 212-555-0148",
                &["US"],
                &["212-555-0147", "212-555-0148"],
            ),
            ("NY 10118 212-555-0147", &["US"], &["212-555-0147"]),
            ("Suite 200 212-555-0147", &["US"], &["212-555-0147"]),
            ("〒100-0001 03-1234-5678", &["JP"], &["03-1234-5678"]),
            ("12.03.2024 030 1234567", &["DE"], &["030 1234567"]),
            ("(+49) 30 1234567", &[], &["(+49) 30 1234567"]),
            ("(+44) 20 7946 0958", &[], &["(+44) 20 7946 0958"]),
            ("(+1) 212 555 0147", &[], &["(+1) 212 555 0147"]),
            ("+44 020 7946 0958", &[], &["+44 020 7946 0958"]),
            ("03(1234)5678", &["JP"], &["03(1234)5678"]),
            ("212-555-0147x204", &["US"], &["212-555-0147"]),
            ("Tel. 030 - 1234567", &["DE"], &["030 - 1234567"]),
            ("D02 X285 01 234 5678", &["IE"], &["01 234 5678"]),
            ("Faxnr. 2125550147", &["US"], &["2125550147"]),
            ("Servicenummer 2125550147", &["US"], &["2125550147"]),
            ("Festnetz-Nr. 030 1234567", &["DE"], &["030 1234567"]),
            ("Home no. 020 7946 0321", &["GB"], &["020 7946 0321"]),
            ("Work No. 020 7946 0321", &["GB"], &["020 7946 0321"]),
            ("Landline 2125550147", &["US"], &["2125550147"]),
            ("Daytime no. 020 7946 0321", &["GB"], &["020 7946 0321"]),
            ("Evening no. 020 7946 0321", &["GB"], &["020 7946 0321"]),
            ("Direct no. 020 7946 0321", &["GB"], &["020 7946 0321"]),
            ("Mobiel nr. 06 12345678", &["NL"], &["06 12345678"]),
            ("Telefoon nr. 020 123 4567", &["NL"], &["020 123 4567"]),
            ("GSM no. 0470 12 34 56", &["BE"], &["0470 12 34 56"]),
            ("Natel Nr. 079 123 45 67", &["CH"], &["079 123 45 67"]),
            ("Portable n° 0470 12 34 56", &["BE"], &["0470 12 34 56"]),
            ("Tél. n° 02 123 45 67", &["BE"], &["02 123 45 67"]),
            ("電話：03-1234-5678", &["JP"], &["03-1234-5678"]),
            ("Dr. Anna Az 030 1234567", &["DE"], &["030 1234567"]),
        ];
        for (text, hints, want) in cases {
            assert_eq!(found(text, hints), *want, "{text} with {hints:?}");
        }
    }

    /// Glued words, short numbers beside a phone, three or more numbers in one run, a trailing
    /// hour, labels carried over a split, and a postcode before a phone: spans and E.164.
    #[cfg(feature = "phone-metadata")]
    #[test]
    fn neighbours_of_a_phone() {
        /// Input, hints, and every phone found as its span and E.164 form.
        type Case<'a> = (&'a str, &'a [&'a str], &'a [(&'a str, &'a str)]);
        let cases: &[Case] = &[
            (
                "Tel 030 1234567 2nd floor",
                &["DE"],
                &[("030 1234567", "+49301234567")],
            ),
            (
                "Tel +44 20 7946 0958 3rd floor",
                &[],
                &[("+44 20 7946 0958", "+442079460958")],
            ),
            (
                "Call 020 7946 0321 24 hours a day",
                &["GB"],
                &[("020 7946 0321", "+442079460321")],
            ),
            (
                "Call 212 555 0147 24 hours",
                &["US"],
                &[("212 555 0147", "+12125550147")],
            ),
            (
                "Room 12 020 7946 0321",
                &["GB"],
                &[("020 7946 0321", "+442079460321")],
            ),
            (
                "Mon-Fri 9 020 7946 0321",
                &["GB"],
                &[("020 7946 0321", "+442079460321")],
            ),
            (
                "1998-2024 030 1234567",
                &["DE"],
                &[("030 1234567", "+49301234567")],
            ),
            (
                "Tel. 030 1234567 / 0171 1234567 / 0172 1234567",
                &["DE"],
                &[
                    ("030 1234567", "+49301234567"),
                    ("0171 1234567", "+491711234567"),
                    ("0172 1234567", "+491721234567"),
                ],
            ),
            (
                "030 1234567 030 1234568 030 1234569",
                &["DE"],
                &[
                    ("030 1234567", "+49301234567"),
                    ("030 1234568", "+49301234568"),
                    ("030 1234569", "+49301234569"),
                ],
            ),
            (
                "Tel. 030 1234567 10 Uhr",
                &["DE"],
                &[("030 1234567", "+49301234567")],
            ),
            (
                "Tel. 030 1234567 10 h",
                &["DE"],
                &[("030 1234567", "+49301234567")],
            ),
            ("030 12345 67", &["DE"], &[("030 12345 67", "+49301234567")]),
            (
                "089 123456 78",
                &["DE"],
                &[("089 123456 78", "+498912345678")],
            ),
            (
                "Tel. 030 12345 67.",
                &["DE"],
                &[("030 12345 67", "+49301234567")],
            ),
            (
                "Tel 2125550147 2125550148",
                &["US"],
                &[
                    ("2125550147", "+12125550147"),
                    ("2125550148", "+12125550148"),
                ],
            ),
            (
                "PLZ 01067 0351 1234567",
                &["DE"],
                &[("0351 1234567", "+493511234567")],
            ),
        ];
        for (text, hints, want) in cases {
            let got: Vec<(&str, String)> = scan(text, hints)
                .into_iter()
                .map(|e| (&text[e.start..e.end], e.normalized.unwrap_or_default()))
                .collect();
            let want: Vec<(&str, String)> = want.iter().map(|&(s, n)| (s, n.into())).collect();
            assert_eq!(got, want, "{text} with {hints:?}");
        }
    }

    /// Runs longer than the caps on groups and digits: every number is found whole, none is
    /// cut at a cap, and none is lost past it.
    #[cfg(feature = "phone-metadata")]
    #[test]
    fn runs_past_the_caps() {
        let years: Vec<String> = (2019..=2033).map(|y| y.to_string()).collect();
        let listed = (0..6)
            .map(|k| {
                if k == 0 {
                    "030 1234567".to_string()
                } else {
                    format!("017{k} 1234567")
                }
            })
            .collect::<Vec<_>>()
            .join(" / ");
        /// Input, hints, and every phone found as its span and E.164 form.
        type Case<'a> = (String, &'a [&'a str], Vec<(String, String)>);
        let cases: Vec<Case> = vec![
            (
                "030 123 4567 ".repeat(6),
                &["DE"],
                vec![("030 123 4567".into(), "+49301234567".into()); 6],
            ),
            (
                "020 7946 0958 ".repeat(7),
                &["GB"],
                vec![("020 7946 0958".into(), "+442079460958".into()); 7],
            ),
            (
                "0301234567 ".repeat(7),
                &["DE"],
                vec![("0301234567".into(), "+49301234567".into()); 7],
            ),
            (
                "07911 123456 ".repeat(6),
                &["GB"],
                vec![("07911 123456".into(), "+447911123456".into()); 6],
            ),
            (
                format!("Tel. {listed}"),
                &["DE"],
                (0..6)
                    .map(|k| {
                        if k == 0 {
                            ("030 1234567".into(), "+49301234567".into())
                        } else {
                            (format!("017{k} 1234567"), format!("+4917{k}1234567"))
                        }
                    })
                    .collect(),
            ),
            (
                format!("Tel. 030 1234567 {}", years.join(" ")),
                &["DE"],
                vec![("030 1234567".into(), "+49301234567".into())],
            ),
        ];
        for (text, hints, want) in cases {
            let got: Vec<(String, String)> = scan(&text, hints)
                .into_iter()
                .map(|e| {
                    (
                        text[e.start..e.end].to_string(),
                        e.normalized.unwrap_or_default(),
                    )
                })
                .collect();
            assert_eq!(got, want, "{text} with {hints:?}");
        }
    }

    /// The same numbers repeated from once to well past both caps, so that a cap falls at
    /// every place inside and between them: each is found once, whole.
    #[cfg(feature = "phone-metadata")]
    #[test]
    fn caps_fall_anywhere() {
        let units: &[(&str, &str, &str, &[&str])] = &[
            ("030 123 4567 ", "030 123 4567", "+49301234567", &["DE"]),
            ("020 7946 0958 ", "020 7946 0958", "+442079460958", &["GB"]),
            ("0301234567 ", "0301234567", "+49301234567", &["DE"]),
            ("07911 123456 ", "07911 123456", "+447911123456", &["GB"]),
            ("0470 12 34 56 ", "0470 12 34 56", "+32470123456", &["BE"]),
            (
                "12.03.2024 030 1234567 ",
                "030 1234567",
                "+49301234567",
                &["DE"],
            ),
            ("030 1234567 / ", "030 1234567", "+49301234567", &["DE"]),
            (
                "+44 20 7946 0958 ",
                "+44 20 7946 0958",
                "+442079460958",
                &[],
            ),
        ];
        for &(unit, span, e164, hints) in units {
            for n in 1..=24 {
                let text = unit.repeat(n);
                let got: Vec<(&str, String)> = scan(&text, hints)
                    .into_iter()
                    .map(|e| (&text[e.start..e.end], e.normalized.unwrap_or_default()))
                    .collect();
                assert_eq!(got, vec![(span, e164.to_string()); n], "{unit:?} x {n}");
            }
        }
        let years: Vec<String> = (1950..=2099).map(|y| y.to_string()).collect();
        let text = format!("Tel. 030 1234567 {}", years.join(" "));
        let got: Vec<_> = scan(&text, &["DE"])
            .into_iter()
            .map(|e| e.normalized.unwrap_or_default())
            .collect();
        assert_eq!(got, ["+49301234567"]);
    }

    /// Short numbers, labels and trailing pairs that must not change what is found.
    #[cfg(feature = "phone-metadata")]
    #[test]
    fn neighbours_keep_precision() {
        let all = [
            "US", "CA", "GB", "DE", "GE", "JP", "NL", "BE", "AT", "CH", "IE",
        ];
        for text in [
            "Room 12 212 555 0147",
            "12 345 678 901 234",
            "PLZ 1067 0351 1234567",
            "IBAN 3704 0532 0130 00",
            "Vertragsnummer: 030 1234567",
            "030 / 1234 / 5678 2nd",
        ] {
            assert!(
                scan(text, &all).is_empty(),
                "{text}: {:?}",
                scan(text, &all)
            );
        }
        assert_eq!(found("0470 12 34 56", &["BE"]), ["0470 12 34 56"]);
        assert_eq!(found("079 123 45 67", &["CH"]), ["079 123 45 67"]);
    }

    /// 200 KB of numbers side by side, phones and not, with every region hinted, is scanned in
    /// a few tens of milliseconds at most.
    #[cfg(feature = "phone-metadata")]
    #[test]
    fn long_runs_stay_fast() {
        let hints = [
            "US", "CA", "GB", "DE", "GE", "JP", "NL", "BE", "AT", "CH", "IE",
        ];
        for unit in [
            "030 1234567 ",
            "0 1 2 3 4 5 6 7 8 9 ",
            "0 0 0 0 0 0 0 1234567 ",
            "030 12 34 56 78 90 ",
            "599 12 34 56 ",
            "0049 30 1234567 00 ",
            "+44 20 7946 0958 1 ",
            "12 345 678 ",
            "030 / ",
            "12.03.2024 030 1234567 ",
            "1998-2024 ",
            "Tel 2125550147 ",
        ] {
            let text = unit.repeat(200_000 / unit.len());
            // The best of three release runs, so that a busy machine does not fail the test.
            let runs = if cfg!(debug_assertions) { 1 } else { 3 };
            let elapsed = (0..runs)
                .map(|_| {
                    let start = std::time::Instant::now();
                    let _ = scan(&text, &hints);
                    start.elapsed()
                })
                .min()
                .unwrap_or_default();
            // Debug builds are an order of magnitude slower; the bound is for release.
            let bound = if cfg!(debug_assertions) { 1_000 } else { 50 };
            assert!(elapsed.as_millis() < bound, "{unit:?} took {elapsed:?}");
        }
    }

    #[cfg(feature = "phone-metadata")]
    #[test]
    fn new_forms_normalize_to_e164() {
        let e164 = |text: &str, hints: &[&str]| -> Vec<String> {
            scan(text, hints)
                .into_iter()
                .filter_map(|e| e.normalized)
                .collect()
        };
        assert_eq!(e164("(+49) 30 1234567", &[]), ["+49301234567"]);
        assert_eq!(e164("+44 020 7946 0958", &[]), ["+442079460958"]);
        assert_eq!(e164("03(1234)5678", &["JP"]), ["+81312345678"]);
        assert_eq!(
            e164("Tel. 030 1234567 / 0171 1234567", &["DE"]),
            ["+49301234567", "+491711234567"]
        );
        assert_eq!(
            e164("+49 30 1234567 030 1234568", &["DE"]),
            ["+49301234567", "+49301234568"]
        );
    }

    #[cfg(feature = "phone-metadata")]
    #[test]
    fn nothing_longer_than_e164_allows() {
        assert!(scan("01.01.202608.00212", &["DE"]).is_empty());
        assert!(scan("+4910120260800212", &[]).is_empty());
    }

    /// Decimals, national numbers that still start with `0`, dash ranges, date lists, IBANs
    /// with letter bank codes, labels for other numbers, prices and dated versions.
    #[test]
    fn precision_cases() {
        let all = [
            "US", "CA", "GB", "DE", "GE", "JP", "NL", "BE", "AT", "CH", "IE",
        ];
        let cases: &[(&str, &[&str])] = &[
            ("0.1234567", &["DE"]),
            ("0.12 3456 7890", &["DE", "GB"]),
            ("(021) 255-5014", &["US"]),
            ("+44 0020 7946 0958", &[]),
            ("08.00–12.00", &["DE"]),
            ("08.00\u{2010}12.00", &["DE"]),
            ("0800-1200", &["DE"]),
            ("0830–1730", &["GB"]),
            ("01.01.2026 31.12.2026", &["DE"]),
            ("gültig 01.01.26 31.12.26", &["DE"]),
            ("IBAN NL91 ABNA 0417 1643 00", &["NL"]),
            ("NL91 ABNA 0417 1643 00", &["NL"]),
            ("GB29 NWBK 6016 1331 9268 19", &["GB"]),
            ("card 4111 1111 1111 1111", &all),
            ("Kundennr. 030 1234567", &["DE"]),
            ("Bestellnummer 030 1234567", &["DE"]),
            ("Fondsnummer 030 1234567", &["DE"]),
            ("Unser Zeichen: 030 1234567", &["DE"]),
            ("Geschäftszeichen 030 1234567", &["DE"]),
            ("Az. 030 1234567", &["DE"]),
            ("Facture 0470 12 34 56", &["BE"]),
            ("Facture n° 0470 12 34 56", &["BE"]),
            ("Commande Nº 02 123 45 67", &["BE"]),
            ("Référence 02 123 45 67", &["BE"]),
            ("Client 02 123 45 67", &["BE"]),
            ("TVA BE 0123.456.789", &["BE"]),
            ("BTW BE 0123.456.789", &["BE"]),
            ("KvK 020 123 4567", &["NL"]),
            ("Factuur 020 123 4567", &["NL"]),
            ("Klant 020 123 4567", &["NL"]),
            ("Referentie 020 123 4567", &["NL"]),
            ("Bestelling 020 123 4567", &["NL"]),
            ("№ 0301234567", &["DE"]),
            ("numéro 02 123 45 67", &["BE"]),
            ("ნომერი 032 212 3456", &["GE"]),
            ("注文番号：03-1234-5678", &["JP"]),
            ("注文番号\u{3000}03-1234-5678", &["JP"]),
            ("顧客ID 03-1234-5678", &["JP"]),
            ("03-1234-5678円", &["JP"]),
            ("Firmware 02.10.2024.01", &all),
            ("population 12 345 678 901 234", &all),
            ("Call 12 345 678 901 234", &all),
            ("EUR 1 234 567 890 123,45", &all),
            ("totals 120 450 0300 1200 1800", &all),
            ("week 38 2026 0012 3456 789", &all),
            ("3782 822463 10005", &all),
            ("IBAN IE29 AIBK 9311 5212 3456 78", &all),
            ("GB82 WEST 1234 5698 7654 32", &all),
        ];
        for (text, hints) in cases {
            assert!(
                scan(text, hints).is_empty(),
                "{text}: {:?}",
                scan(text, hints)
            );
        }
    }

    #[cfg(feature = "phone-metadata")]
    #[test]
    fn national_numbers_need_their_trunk_prefix() {
        assert!(scan("call 20 7946 0321", &["GB"]).is_empty());
        assert_eq!(scan("call 020 7946 0321", &["GB"]).len(), 1);
        assert_eq!(scan("Tel 030 23125 000", &["DE"]).len(), 1);
        assert_eq!(scan("call (202) 555-0143", &["US"]).len(), 1);
    }

    #[cfg(feature = "phone-metadata")]
    #[test]
    fn labels_before_a_phone_still_allow_it() {
        assert_eq!(scan("Tel. 030/23125000", &["DE"]).len(), 1);
        assert_eq!(scan("Order hotline: 020 7946 0321", &["GB"]).len(), 1);
        assert_eq!(scan("Account fax 020 7946 0321", &["GB"]).len(), 1);
        assert_eq!(scan("Hotline: 020 7946 0321", &["GB"]).len(), 1);
        assert_eq!(scan("Tel. no. 020 7946 0321", &["GB"]).len(), 1);
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
        let hints = ["US", "GB", "DE", "NL", "IE", "JP"];
        assert!(scan("card 4111 1111 1111 1111 exp", &hints).is_empty());
        assert!(scan("card 4111 0171 1234 5678", &hints).is_empty());
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
    fn prefix_sets_match_the_lists() {
        for region in super::super::phone_tables::REGIONS {
            assert!(prefix_set(region.prefixes).is_some(), "{}", region.id);
            let brute = |n: &str| region.prefixes.iter().any(|q| matches_prefix(n, q));
            for p in region.prefixes {
                for fill in ["0", "5", "9"] {
                    let n = format!("{}{}", p.replace('x', fill), fill.repeat(8));
                    assert_eq!(has_prefix(region, &n), brute(&n), "{} {n}", region.id);
                    let short = &n[..p.len().saturating_sub(1)];
                    assert_eq!(has_prefix(region, short), brute(short));
                }
            }
            for n in 0..20_000u32 {
                let n = format!("{:07}", n * 499);
                assert_eq!(has_prefix(region, &n), brute(&n), "{} {n}", region.id);
            }
        }
        let set = prefix_set(&["2xx", "31"]).unwrap_or_default();
        assert_eq!(in_prefix_set(&set, "1234"), Some(false));
        assert_eq!(in_prefix_set(&set, "2034"), Some(true));
        assert_eq!(in_prefix_set(&set, "3100"), Some(true));
        assert_eq!(in_prefix_set(&set, "203"), None);
        let all = prefix_set(&["", "31"]).unwrap_or_default();
        assert_eq!(in_prefix_set(&all, "1234"), Some(true));
        assert!(prefix_set(&["12345"]).is_none());
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
