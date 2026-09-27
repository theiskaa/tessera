//! Filters unhelpful names and address rows before they enter US synthetic documents.

use tessera::AddressLabel;
use tessera::internal::fnv1a;

use crate::data::LabelledExample;

/// Lowercase words of `s`: runs of letters and digits.
fn words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Whether `phrase` (space-separated lowercase words) occurs as consecutive words of `words`.
fn has_phrase(words: &[String], phrase: &str) -> bool {
    let wanted: Vec<&str> = phrase.split(' ').collect();
    words
        .windows(wanted.len())
        .any(|w| w.iter().zip(&wanted).all(|(a, b)| a == b))
}

fn any_phrase(words: &[String], phrases: &[&str]) -> bool {
    phrases.iter().any(|p| has_phrase(words, p))
}

/// Words that make a name a private arrangement: a family or will trust, a settlement, an
/// estate, a pension or benefit plan. Real documents rarely name these; the registers that feed
/// the pools are full of them.
const PRIVATE: [&str; 34] = [
    "settlement",
    "settlements",
    "trustee",
    "trustees",
    "revocable",
    "irrevocable",
    "living trust",
    "family trust",
    "will trust",
    "discretionary",
    "intestacy",
    "deceased",
    "decd",
    "estate of",
    "trust dated",
    "life interest",
    "accumulation",
    "gift trust",
    "marital trust",
    "nominee",
    "pension",
    "pensions",
    "superannuation",
    "retirement plan",
    "retirement fund",
    "retirement scheme",
    "benefit plan",
    "benefits plan",
    "employee benefit",
    "savings plan",
    "benefits scheme",
    "benefit scheme",
    "retirement benefits",
    "assurance scheme",
];

/// Trusts that are institutions rather than arrangements: banks, NHS trusts, charities.
const INSTITUTIONAL_TRUST: [&str; 12] = [
    "trust bank",
    "trust company",
    "trust corporation",
    "trust plc",
    "trust ltd",
    "trust limited",
    "trust inc",
    "nhs",
    "foundation trust",
    "national trust",
    "housing trust",
    "wildlife trust",
];

/// A private trust, settlement, estate, or pension scheme.
pub fn private_arrangement(name: &str) -> bool {
    let w = words(name);
    any_phrase(&w, &PRIVATE)
        || ((has_phrase(&w, "trust") || has_phrase(&w, "trusts"))
            && !any_phrase(&w, &INSTITUTIONAL_TRUST))
}

/// A name that embeds a person: an honorific first, or a deceased estate. Never kept, even in
/// the small share of private arrangements that stays.
pub fn names_a_person(name: &str) -> bool {
    let w = words(name);
    let first = w.iter().find(|w| w.as_str() != "the").map(String::as_str);
    matches!(
        first,
        Some("mr" | "mrs" | "miss" | "ms" | "sir" | "lady" | "lord" | "dr" | "rev" | "late")
    ) || any_phrase(&w, &["deceased", "decd", "estate of"])
}

const STREET_WORDS: [&str; 20] = [
    "street",
    "st",
    "avenue",
    "ave",
    "road",
    "rd",
    "boulevard",
    "blvd",
    "drive",
    "dr",
    "lane",
    "ln",
    "way",
    "place",
    "court",
    "parkway",
    "highway",
    "terrace",
    "square",
    "plaza",
];

/// A company named after its street address, such as `2100 POWELL STREET MEZZ, LLC`.
pub fn address_like(name: &str) -> bool {
    let w = words(name);
    let numbered = w
        .first()
        .is_some_and(|f| f.len() <= 6 && f.chars().all(|c| c.is_ascii_digit()));
    numbered
        && w.iter()
            .skip(1)
            .take(6)
            .any(|x| STREET_WORDS.contains(&x.as_str()))
}

/// Words Wikidata adds in parentheses to tell items apart, which no document writes.
const DISAMBIGUATORS: [&str; 16] = [
    "party",
    "publisher",
    "company",
    "georgia",
    "japan",
    "band",
    "magazine",
    "newspaper",
    "organization",
    "organisation",
    "business",
    "brand",
    "record label",
    "企業",
    "出版社",
    "პარტია",
];

/// `name` without a trailing Wikidata disambiguator such as `(Japan)`. Acronyms in parentheses,
/// which organizations do write, stay.
pub fn without_disambiguator(name: &str) -> String {
    let trimmed = name.trim_end();
    let Some(open) = trimmed.rfind(['(', '（']) else {
        return name.to_string();
    };
    let inner = trimmed[open..]
        .trim_start_matches(['(', '（'])
        .trim_end_matches([')', '）'])
        .trim()
        .to_lowercase();
    if DISAMBIGUATORS.contains(&inner.as_str()) {
        trimmed[..open].trim_end().to_string()
    } else {
        name.to_string()
    }
}

/// Whether to keep an item of a kind the pools keep only `per_mille` of, decided by its name
/// so the choice is stable across runs.
pub fn keep_some(name: &str, per_mille: u64) -> bool {
    fnv1a(name.as_bytes(), 0x9e37) % 1000 < per_mille
}

/// Map rows about a fire hydrant or water tank rather than a place anyone writes to.
pub fn hydrant(text: &str) -> bool {
    let lower = text.to_lowercase();
    ["hydrant", "water tank"].iter().any(|w| lower.contains(w))
}

/// The row's text without leading and trailing lines that no labelled component touches: the
/// shop, bank, or landmark a map row was about, which the parser data leaves unlabelled and the
/// generator would otherwise label as address text.
pub fn without_venue_lines(e: &LabelledExample) -> String {
    let mut lines: Vec<(usize, &str)> = Vec::new();
    let mut at = 0;
    for line in e.text.split_inclusive('\n') {
        lines.push((at, line));
        at += line.len();
    }
    let touched = |&(start, line): &(usize, &str)| {
        let end = start + line.len();
        e.spans
            .iter()
            .any(|s| (s.start as usize) < end && start < s.end as usize)
    };
    let first = lines.iter().position(touched);
    let last = lines.iter().rposition(touched);
    match (first, last) {
        (Some(a), Some(b)) => lines[a..=b]
            .iter()
            .map(|(_, l)| *l)
            .collect::<String>()
            .trim()
            .to_string(),
        _ => e.text.trim().to_string(),
    }
}

/// Normalize a US country line to its English name when the source uses another label.
pub fn with_local_country(e: &LabelledExample, text: &str) -> String {
    let Some(country) = e
        .spans
        .iter()
        .rev()
        .find(|span| span.label == AddressLabel::Country)
        .map(|span| e.text[span.start as usize..span.end as usize].trim())
    else {
        return text.to_string();
    };
    if [
        "United States",
        "USA",
        "US",
        "United States of America",
        "U.S.A.",
    ]
    .iter()
    .any(|name| name.eq_ignore_ascii_case(country))
    {
        return text.to_string();
    }
    match text.rfind(country) {
        Some(at) => format!(
            "{}United States{}",
            &text[..at],
            &text[at + country.len()..]
        ),
        None => text.to_string(),
    }
}

/// Whether the row is an address a letter could be sent to, as the review guidelines define
/// one: a house number, a PO box, a unit or floor on a named street, or a street with its
/// town and postcode (`FitzRoy Road, Exeter, EX1 3PB`). A town, a postcode with its town, or
/// a street alone is a place, and labelling it as an address would teach the detector the
/// opposite of what it is measured against.
pub fn postal(e: &LabelledExample) -> bool {
    let has = |label: AddressLabel| e.spans.iter().any(|s| s.label == label);
    has(AddressLabel::HouseNumber)
        || has(AddressLabel::PoBox)
        || ((has(AddressLabel::Unit) || has(AddressLabel::Level)) && has(AddressLabel::Road))
        || (has(AddressLabel::Road) && has(AddressLabel::Postcode) && has(AddressLabel::City))
}

/// `name` without a leading English article, which the guidelines keep outside an org span.
pub fn without_article(name: &str) -> &str {
    match name
        .strip_prefix("The ")
        .or_else(|| name.strip_prefix("THE "))
    {
        Some(rest) if rest.contains(' ') || rest.chars().count() > 3 => rest,
        _ => name,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn private_arrangements_and_their_institutional_look_alikes() {
        for name in [
            "THE ANGELA MAUNDRELL SETTLEMENT 2020",
            "Smith Family Trust",
            "ACME Pension Scheme",
            "Estate of John Doe",
        ] {
            assert!(private_arrangement(name), "{name}");
        }
        for name in ["Northern Trust Company", "Acme Widgets Ltd"] {
            assert!(!private_arrangement(name), "{name}");
        }
        assert!(names_a_person("MRS JANE DOE DISCRETIONARY TRUST"));
        assert!(!names_a_person("The Smith Settlement"));
    }

    #[test]
    fn address_like_companies() {
        assert!(address_like("2100 POWELL STREET MEZZ, LLC"));
        assert!(!address_like("3M Company"));
    }

    #[test]
    fn article_removal_keeps_acronyms() {
        assert_eq!(
            without_article("The Planning Inspectorate"),
            "Planning Inspectorate"
        );
        assert_eq!(without_article("THE AA"), "THE AA");
    }

    #[test]
    fn disambiguators_go_and_acronyms_stay() {
        assert_eq!(without_disambiguator("Kodansha (Japan)"), "Kodansha");
        assert_eq!(without_disambiguator("Labour (party)"), "Labour");
        assert_eq!(
            without_disambiguator("Society of Exploration Geophysicists (SEGJ)"),
            "Society of Exploration Geophysicists (SEGJ)"
        );
    }

    #[test]
    fn keeping_some_is_stable_and_about_the_share() {
        assert_eq!(keep_some("a", 50), keep_some("a", 50));
        let kept = (0..10_000)
            .filter(|i| keep_some(&format!("n{i}"), 50))
            .count();
        assert!((350..650).contains(&kept), "{kept}");
    }
}
