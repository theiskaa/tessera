//! What the generator's pools may hold, from the pool audit (`internal/reports/m7-pool-audit.md`):
//! real documents name companies and public bodies, not private trusts or pension schemes, and
//! write an address as its street and place, not as the shop or hydrant a map row was about.
//! Each rule here is a load-time predicate or rewrite; the sources are never changed.

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
    let lower = name.to_lowercase();
    any_phrase(&w, &PRIVATE)
        || [
            "pensionskasse",
            "pensionsfonds",
            "versorgungswerk",
            "年金",
            "信託契約",
            "受託者",
        ]
        .iter()
        .any(|p| lower.contains(p))
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

/// Institutions that a company legal form would misname.
const INSTITUTION: [&str; 18] = [
    "university",
    "universität",
    "institute",
    "institut",
    "foundation",
    "stiftung",
    "association",
    "agency",
    "ministry",
    "prefecture",
    "school",
    "hospital",
    "party",
    "museum",
    "council",
    "church",
    "society",
    "college",
];

/// A public body or non-profit: never given a company's legal form.
pub fn institution(name: &str) -> bool {
    any_phrase(&words(name), &INSTITUTION)
}

/// Legal forms written after a company name, longest first so `L.L.C.` wins over `LLC`.
const LEGAL_FORMS: [&str; 29] = [
    "gmbh & co. kg",
    "ug (haftungsbeschränkt)",
    "co., ltd.",
    "incorporated",
    "corporation",
    "limited",
    "l.l.c.",
    "company",
    "corp.",
    "gmbh",
    "llc",
    "llp",
    "plc",
    "inc.",
    "ltd.",
    "inc",
    "ltd",
    "co.",
    "k.k.",
    "g.k.",
    "e.k.",
    "gbr",
    "ohg",
    "mbh",
    "ag",
    "kg",
    "ug",
    "se",
    "lp",
];

/// `name` without a trailing legal form and the comma before it; `None` when a second form
/// remains, as in `… MEZZ, LLC, INC`, where the name would still read as a foreign company.
pub fn without_legal_form(name: &str) -> Option<String> {
    let strip = |s: &str| -> Option<String> {
        let lower = s.to_lowercase();
        LEGAL_FORMS.iter().find_map(|form| {
            let cut = lower.strip_suffix(form)?;
            let before = cut.chars().last();
            (before.is_none_or(|c| c == ' ' || c == ','))
                .then(|| s[..cut.len()].trim_end_matches([' ', ',']).to_string())
        })
    };
    let base = strip(name).unwrap_or_else(|| name.to_string());
    (!base.is_empty() && strip(&base).is_none()).then_some(base)
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

/// Places in the occupied territories of Georgia, in Georgian, Russian, and Latin script.
const OCCUPIED: [&str; 24] = [
    "ცხინვალ",
    "სოხუმ",
    "გაგრ",
    "გუდაუთ",
    "ბიჭვინთ",
    "ოჩამჩირ",
    "ტყვარჩელ",
    "ახალგორ",
    "აფხაზ",
    "სამაჩაბლო",
    "цхинвал",
    "сухум",
    "гагр",
    "гудаут",
    "пицунд",
    "очамчир",
    "ахалгори",
    "абхаз",
    "tskhinval",
    "sukhum",
    "gagra",
    "gudauta",
    "akhalgori",
    "abkhaz",
];

/// A Georgian address row the pools leave out: in the occupied territories, written in Russian,
/// or garbled.
pub fn unusable_ge_address(e: &LabelledExample) -> bool {
    let places: String = e
        .spans
        .iter()
        .filter(|s| {
            matches!(
                s.label,
                AddressLabel::City
                    | AddressLabel::Region
                    | AddressLabel::District
                    | AddressLabel::Suburb
            )
        })
        .map(|s| e.text[s.start as usize..s.end as usize].to_lowercase())
        .collect::<Vec<_>>()
        .join(" | ");
    let cyrillic_part = e.spans.iter().any(|s| {
        s.label != AddressLabel::Country
            && e.text[s.start as usize..s.end as usize]
                .chars()
                .any(|c| ('\u{0400}'..='\u{04ff}').contains(&c))
    });
    OCCUPIED.iter().any(|p| places.contains(p)) || cyrillic_part || e.text.contains("??")
}

/// Map rows about a fire hydrant or water tank rather than a place anyone writes to.
pub fn hydrant(text: &str) -> bool {
    let lower = text.to_lowercase();
    ["消火栓", "防火水槽", "shōkasen", "shokasen", "hydrant"]
        .iter()
        .any(|w| lower.contains(w))
}

/// Line words that make a Japanese line part of the address: a building's name.
const BUILDING: [&str; 10] = [
    "ビル",
    "マンション",
    "ハイツ",
    "荘",
    "館",
    "レジデンス",
    "コーポ",
    "タワー",
    "号館",
    "住宅",
];

/// The row's text without leading and trailing lines that no labelled component touches: the
/// shop, bank, or landmark a map row was about, which the parser data leaves unlabelled and the
/// generator would otherwise label as address text. Japanese building-name lines stay.
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
            || BUILDING.iter().any(|b| line.contains(b))
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

/// What each country is called in its own addresses: the local and the English name.
const COUNTRY_NAMES: [(&str, &[&str]); 5] = [
    (
        "US",
        &[
            "United States",
            "USA",
            "US",
            "United States of America",
            "U.S.A.",
        ],
    ),
    (
        "GB",
        &[
            "United Kingdom",
            "UK",
            "England",
            "Scotland",
            "Wales",
            "Northern Ireland",
            "Great Britain",
        ],
    ),
    ("DE", &["Deutschland", "Germany"]),
    ("GE", &["საქართველო", "Georgia"]),
    ("JP", &["日本", "Japan", "日本国", "Nippon"]),
];

/// `text` with a country line in another language, such as `Birleşik Krallık` in a British
/// row, written as the country's own name: map data carries the labels of every language.
pub fn with_local_country(e: &LabelledExample, text: &str) -> String {
    let Some((_, names)) = COUNTRY_NAMES.iter().find(|(c, _)| *c == e.country) else {
        return text.to_string();
    };
    let Some(country) = e
        .spans
        .iter()
        .rev()
        .find(|s| s.label == AddressLabel::Country)
        .map(|s| e.text[s.start as usize..s.end as usize].trim())
    else {
        return text.to_string();
    };
    if names.iter().any(|n| n.eq_ignore_ascii_case(country)) {
        return text.to_string();
    }
    match text.rfind(country) {
        Some(at) => format!("{}{}{}", &text[..at], names[0], &text[at + country.len()..]),
        None => text.to_string(),
    }
}

/// Whether the row is an address a letter could be sent to, as the review guidelines define
/// one: a house number, a PO box, or a unit or floor on a named street. A town, a postcode
/// with its town, or a street without a number is a place, and labelling it as an address
/// would teach the detector the opposite of what it is measured against.
pub fn postal(e: &LabelledExample) -> bool {
    let has = |label: AddressLabel| e.spans.iter().any(|s| s.label == label);
    has(AddressLabel::HouseNumber)
        || has(AddressLabel::PoBox)
        || ((has(AddressLabel::Unit) || has(AddressLabel::Level)) && has(AddressLabel::Road))
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
            "Versorgungswerk der Ärzte",
        ] {
            assert!(private_arrangement(name), "{name}");
        }
        for name in [
            "Northern Trust Company",
            "Leeds NHS Foundation Trust",
            "Acme Widgets Ltd",
        ] {
            assert!(!private_arrangement(name), "{name}");
        }
        assert!(names_a_person("MRS JANE DOE DISCRETIONARY TRUST"));
        assert!(!names_a_person("The Smith Settlement"));
    }

    #[test]
    fn address_like_companies_and_institutions() {
        assert!(address_like("2100 POWELL STREET MEZZ, LLC"));
        assert!(!address_like("3M Company"));
        assert!(institution(
            "Okayama Prefecture Industrial Promotion Foundation"
        ));
        assert!(!institution("Acme Holdings"));
    }

    #[test]
    fn legal_forms_come_off_once() {
        assert_eq!(
            without_legal_form("Acme Widgets LIMITED").as_deref(),
            Some("Acme Widgets")
        );
        assert_eq!(without_legal_form("Acme, L.L.C.").as_deref(), Some("Acme"));
        assert_eq!(without_legal_form("Acme").as_deref(), Some("Acme"));
        assert_eq!(without_legal_form("Mezz, LLC, INC"), None);
        assert_eq!(without_legal_form("LLC"), None);
        assert_eq!(without_legal_form("Bagging").as_deref(), Some("Bagging"));
        assert_eq!(
            without_legal_form("PSG Can UG"),
            Some("PSG Can".to_string())
        );
        assert!(private_arrangement(
            "Peak Oil Products Ltd Retirement Benefits Scheme"
        ));
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
