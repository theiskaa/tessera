//! Person and organization name sampling from Wikidata (CC0) and GLEIF (CC0).
//!
//! People are fetched per (country, language, birth year) so every SPARQL query stays small
//! enough for the public endpoint; a year that reaches the query limit is fetched again month
//! by month. Organizations come from the GLEIF Level 1 golden copy, filtered by jurisdiction
//! and streamed straight out of its zip. Where GLEIF holds fewer usable names than the quota
//! (GE has about 160 active entities, JP about 4,000), Wikidata organizations of a fixed set
//! of classes fill the rest.
//!
//! Every name is split by its normalized, lowercased form, so a name never appears in two
//! splits.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, bail};
use polars::prelude::{Column, DataFrame, ParquetWriter};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use sha2::{Digest, Sha256};
use tessera::internal::{Script, TokenClass, fnv1a, tokenize};
use unicode_normalization::UnicodeNormalization;

use crate::config::{self, NamesConfig};
use crate::data::Split;
use crate::export::hex;

/// One development country: its Wikidata item, label languages, and GLEIF jurisdictions.
struct Country {
    code: &'static str,
    qid: &'static str,
    label_langs: &'static [&'static str],
    /// GLEIF `Entity.LegalJurisdiction` prefixes (US and CA use subnational codes such as `US-DE`).
    jurisdiction_prefixes: &'static [&'static str],
}

/// Every country `[names] countries` may name.
const COUNTRIES: &[Country] = &[
    Country {
        code: "US",
        qid: "Q30",
        label_langs: &["en"],
        jurisdiction_prefixes: &["US"],
    },
    Country {
        code: "CA",
        qid: "Q16",
        label_langs: &["en", "fr"],
        jurisdiction_prefixes: &["CA"],
    },
    Country {
        code: "GB",
        qid: "Q145",
        label_langs: &["en"],
        jurisdiction_prefixes: &["GB"],
    },
    Country {
        code: "DE",
        qid: "Q183",
        label_langs: &["de"],
        jurisdiction_prefixes: &["DE"],
    },
    Country {
        code: "NL",
        qid: "Q55",
        label_langs: &["nl"],
        jurisdiction_prefixes: &["NL"],
    },
    Country {
        code: "GE",
        qid: "Q230",
        label_langs: &["ka", "en"],
        jurisdiction_prefixes: &["GE"],
    },
    Country {
        code: "JP",
        qid: "Q17",
        label_langs: &["ja", "en"],
        jurisdiction_prefixes: &["JP"],
    },
];

const ENDPOINT: &str = "https://query.wikidata.org/sparql";
const USER_AGENT: &str = "tessera-trainer/0.1 (https://github.com/theiskaa/tessera)";
const GLEIF_LATEST: &str =
    "https://goldencopy.gleif.org/api/v2/golden-copies/publishes/lei2/latest";
const ELF_PAGE: &str =
    "https://www.gleif.org/en/about-lei/code-lists/iso-20275-entity-legal-forms-code-list";

/// Direct `P31` classes that count as organizations for the Wikidata top-up: business,
/// company, enterprise, public company, university, government agency, research institute,
/// nonprofit, NGO, bank, political party, publisher. The subclass closure (`P279*`) runs into
/// the endpoint's 60-second timeout even for Georgia.
const ORG_CLASSES: &[&str] = &[
    "Q4830453", "Q783794", "Q6881511", "Q891723", "Q3918", "Q327333", "Q31855", "Q163740",
    "Q79913", "Q22687", "Q7278", "Q2085381",
];

/// Row limit of one people query; a year that reaches it is fetched again per month.
const PEOPLE_QUERY_LIMIT: usize = 5000;
/// Row limit of one organization query.
const ORG_QUERY_LIMIT: usize = 50000;
/// Countries with fewer people than this get a warning in the manifest.
const MIN_PEOPLE: usize = 5000;
/// Longest person name kept, in UTF-8 bytes.
const MAX_PERSON_BYTES: usize = 60;
/// Shortest organization name kept, in UTF-8 bytes.
const MIN_ORG_BYTES: usize = 3;
/// GLEIF entity categories whose legal names are not organization names a document would
/// show: funds and branches repeat their parent's name, and sole proprietors are registered
/// under the owner's personal name, which would teach the detector that people are orgs.
const DROPPED_CATEGORIES: &[&str] = &["FUND", "BRANCH", "SOLE_PROPRIETOR"];
const MAX_ATTEMPTS: u32 = 6;

/// Why a candidate name was dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Rejected {
    PunctuationOrDigit,
    OneToken,
    TooLong,
    WrongScript,
    TooShort,
    ContainsLei,
    Category,
    InvalidUtf8,
    /// Includes repeated result rows for items with several native names or classes.
    Duplicate,
}

impl Rejected {
    fn as_str(self) -> &'static str {
        match self {
            Rejected::PunctuationOrDigit => "punctuation_or_digit",
            Rejected::OneToken => "one_token",
            Rejected::TooLong => "too_long",
            Rejected::WrongScript => "wrong_script",
            Rejected::TooShort => "too_short",
            Rejected::ContainsLei => "contains_lei",
            Rejected::Category => "category",
            Rejected::InvalidUtf8 => "invalid_utf8",
            Rejected::Duplicate => "duplicate",
        }
    }
}

/// Drop counts by reason, as written to the manifests.
#[derive(Debug, Default, serde::Serialize)]
struct Dropped(BTreeMap<&'static str, usize>);

impl Dropped {
    fn add(&mut self, r: Rejected) {
        *self.0.entry(r.as_str()).or_default() += 1;
    }
}

/// Where an organization name came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum OrgSource {
    Gleif,
    Wikidata,
}

impl OrgSource {
    fn name(self) -> &'static str {
        match self {
            OrgSource::Gleif => "gleif",
            OrgSource::Wikidata => "wikidata",
        }
    }
}

struct Person {
    name: String,
    country: &'static str,
    lang: &'static str,
    script: Script,
    qid: String,
    group_id: u64,
    split: Split,
}

struct Org {
    name: String,
    country: &'static str,
    jurisdiction: String,
    lang: String,
    elf_code: String,
    lei: String,
    qid: String,
    source: OrgSource,
    group_id: u64,
    split: Split,
}

/// A file fetched once and kept under `data/raw`, with the URL it came from.
struct Download {
    path: PathBuf,
    url: String,
}

/// The sampled people and what the manifest records about them.
struct PeopleSample {
    people: Vec<Person>,
    files: Vec<PathBuf>,
    dropped: Dropped,
    /// Query files that still reached `PEOPLE_QUERY_LIMIT` after the monthly split.
    saturated: Vec<String>,
}

/// The sampled organizations and what the manifests record about them.
struct OrgSample {
    orgs: Vec<Org>,
    gleif: Download,
    gleif_dropped: Dropped,
    gleif_available: BTreeMap<&'static str, usize>,
    wikidata_files: Vec<PathBuf>,
    wikidata_dropped: Dropped,
    /// Countries the two sources together could not fill, with the number missing.
    shortfall: BTreeMap<&'static str, usize>,
    /// Query files that reached `ORG_QUERY_LIMIT`.
    saturated: Vec<String>,
}

/// The ISO 20275 code list and each code's abbreviations.
struct Elf {
    download: Download,
    abbreviations: HashMap<String, Vec<String>>,
}

/// Samples people and organizations for the config's `[names]` section and writes the
/// Parquet files and manifests.
pub fn run(config_path: &Path, refresh: bool, date: &str) -> anyhow::Result<()> {
    let cfg = config::load(config_path)?;
    let Some(names) = cfg.names else {
        bail!("{} has no [names] section", config_path.display());
    };
    let countries = names
        .countries
        .iter()
        .map(|c| {
            COUNTRIES
                .iter()
                .find(|k| k.code == c)
                .with_context(|| format!("country {c} is not in names::COUNTRIES"))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let raw = PathBuf::from(&names.raw);
    let out = PathBuf::from(&names.out);
    let manifests = PathBuf::from(&cfg.data.manifests);

    let people = sample_people(&names, &countries, &raw, refresh)?;
    let elf = elf(&raw, refresh)?;
    let orgs = sample_orgs(&names, &countries, &raw, refresh)?;
    ensure_disjoint_splits(people.people.iter().map(|p| (p.name.as_str(), p.split)))?;
    ensure_disjoint_splits(orgs.orgs.iter().map(|o| (o.name.as_str(), o.split)))?;

    let interim = PathBuf::from(&names.interim).join("gleif");
    create_dir(&interim)?;
    let sorted: BTreeMap<_, _> = elf.abbreviations.iter().collect();
    write_file(
        &interim.join("elf-abbreviations.json"),
        (serde_json::to_string_pretty(&sorted)? + "\n").as_bytes(),
    )?;
    create_dir(&out)?;
    write_people(&out.join("people.parquet"), &people.people)?;
    write_orgs(&out.join("orgs.parquet"), &orgs.orgs, &elf.abbreviations)?;

    let mut people_warnings = check_people(&countries, &people.people);
    people_warnings.extend(
        people
            .saturated
            .iter()
            .map(|f| format!("{f} reached the {PEOPLE_QUERY_LIMIT}-row query limit")),
    );
    let mut org_warnings: Vec<String> = orgs
        .saturated
        .iter()
        .map(|f| format!("{f} reached the {ORG_QUERY_LIMIT}-row query limit"))
        .collect();
    org_warnings.extend(
        orgs.shortfall
            .iter()
            .map(|(c, n)| format!("{c} has {n} fewer organizations than the quota")),
    );
    for w in people_warnings.iter().chain(&org_warnings) {
        eprintln!("warning: {w}");
    }
    write_people_manifest(
        &manifests,
        &names,
        &countries,
        &people,
        &people_warnings,
        date,
    )?;
    write_gleif_manifest(&manifests, &names, &countries, &orgs, date)?;
    write_wikidata_orgs_manifest(&manifests, &names, &orgs, &org_warnings, date)?;
    write_elf_manifest(&manifests, &elf, date)?;
    print_summary(&countries, &people.people, &orgs.orgs);
    Ok(())
}

/// Birth dates in `[from, to)`, as `YYYY-MM-DD`.
fn people_query(qid: &str, lang: &str, from: &str, to: &str) -> String {
    // Starts from the birth-date index with a range filter: `YEAR(?dob) = year` scans every
    // citizen of the country and passes the endpoint's 60-second limit for the US, while the
    // `rangeSafe` hint lets Blazegraph range-scan the dates and answers in about two seconds.
    format!(
        "SELECT ?item ?label ?native WHERE {{\n\
           ?item wdt:P569 ?dob .\n\
           hint:Prior hint:rangeSafe true .\n\
           FILTER(?dob >= \"{from}T00:00:00Z\"^^xsd:dateTime && ?dob < \"{to}T00:00:00Z\"^^xsd:dateTime)\n\
           ?item wdt:P27 wd:{qid} ;\n\
                 wdt:P31 wd:Q5 .\n\
           ?item rdfs:label ?label .\n\
           FILTER(LANG(?label) = \"{lang}\")\n\
           OPTIONAL {{ ?item wdt:P1559 ?native . }}\n\
         }}\n\
         LIMIT {PEOPLE_QUERY_LIMIT}"
    )
}

/// One class per query: the twelve classes in one `VALUES` block time out for Japan, while
/// each class alone answers in seconds.
fn orgs_query(qid: &str, lang: &str, class: &str) -> String {
    format!(
        "SELECT ?item ?label WHERE {{\n\
           ?item wdt:P31 wd:{class} ;\n\
                 wdt:P17 wd:{qid} .\n\
           ?item rdfs:label ?label .\n\
           FILTER(LANG(?label) = \"{lang}\")\n\
         }}\n\
         LIMIT {ORG_QUERY_LIMIT}"
    )
}

/// The first day of `month` (1 to 12) and of the month after it, as `YYYY-MM-DD`.
fn month_bounds(year: u32, month: u32) -> (String, String) {
    let (next_year, next_month) = if month == 12 {
        (year + 1, 1)
    } else {
        (year, month + 1)
    };
    (
        format!("{year}-{month:02}-01"),
        format!("{next_year}-{next_month:02}-01"),
    )
}

#[derive(Deserialize)]
struct PersonRow {
    item: String,
    label: String,
    #[serde(default)]
    native: String,
}

#[derive(Deserialize)]
struct OrgRow {
    item: String,
    label: String,
}

/// Fetches one year of people for a country and language. A year whose result reaches the
/// query limit is only an arbitrary subset, so it is fetched again month by month; a month
/// that still reaches the limit is reported in `saturated`.
fn fetch_people_year(
    country: &Country,
    lang: &str,
    year: u32,
    raw: &Path,
    refresh: bool,
    sample: &mut PeopleSample,
) -> anyhow::Result<Vec<PersonRow>> {
    let dir = raw.join("wikidata");
    let path = dir.join(format!("{}-{lang}-{year}.csv", country.code));
    let what = format!("people {} {lang} {year}", country.code);
    let query = people_query(
        country.qid,
        lang,
        &format!("{year}-01-01"),
        &format!("{}-01-01", year + 1),
    );
    fetch_sparql(&query, &path, refresh, &what)?;
    let rows: Vec<PersonRow> = read_rows(&path)?;
    if rows.len() < PEOPLE_QUERY_LIMIT {
        sample.files.push(path);
        return Ok(rows);
    }
    let mut all = Vec::new();
    for month in 1..=12 {
        let (from, to) = month_bounds(year, month);
        let path = dir.join(format!("{}-{lang}-{year}-{month:02}.csv", country.code));
        let what = format!("people {} {lang} {year}-{month:02}", country.code);
        fetch_sparql(
            &people_query(country.qid, lang, &from, &to),
            &path,
            refresh,
            &what,
        )?;
        let rows: Vec<PersonRow> = read_rows(&path)?;
        if rows.len() >= PEOPLE_QUERY_LIMIT {
            sample.saturated.push(file_name(&path));
        }
        all.extend(rows);
        sample.files.push(path);
    }
    Ok(all)
}

fn sample_people(
    names: &NamesConfig,
    countries: &[&Country],
    raw: &Path,
    refresh: bool,
) -> anyhow::Result<PeopleSample> {
    let mut sample = PeopleSample {
        people: Vec::new(),
        files: Vec::new(),
        dropped: Dropped::default(),
        saturated: Vec::new(),
    };
    for country in countries {
        let mut seen = HashSet::new();
        let mut per_lang = Vec::new();
        for &lang in country.label_langs {
            let mut kept = Vec::new();
            for year in names.birth_years.0..=names.birth_years.1 {
                for row in fetch_people_year(country, lang, year, raw, refresh, &mut sample)? {
                    let (name, script) = match clean_person(&row.label, &row.native, lang) {
                        Ok(n) => n,
                        Err(r) => {
                            sample.dropped.add(r);
                            continue;
                        }
                    };
                    if !seen.insert(name.to_lowercase()) {
                        sample.dropped.add(Rejected::Duplicate);
                        continue;
                    }
                    let (group_id, split) = split_of(&name);
                    kept.push(Person {
                        name,
                        country: country.code,
                        lang,
                        script,
                        qid: qid_of(&row.item),
                        group_id,
                        split,
                    });
                }
            }
            per_lang.push(kept);
        }
        let mut rng = country_rng(names.sample_seed, country.code);
        sample
            .people
            .extend(take_shares(per_lang, names.people_per_country, &mut rng));
    }
    sample.files.sort();
    Ok(sample)
}

/// One random stream per country, so refreshing or reordering one country's data leaves the
/// others' samples unchanged.
fn country_rng(seed: u64, code: &str) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(seed ^ fnv1a(code.as_bytes(), 0))
}

/// Takes an equal share of `total` from each group after shuffling it, then fills any
/// shortfall from the other groups' leftovers in group order.
fn take_shares<T>(mut groups: Vec<Vec<T>>, total: usize, rng: &mut ChaCha8Rng) -> Vec<T> {
    let share = total / groups.len().max(1);
    let mut taken = Vec::with_capacity(total);
    let mut leftovers = Vec::new();
    for g in &mut groups {
        g.shuffle(rng);
        let rest = g.split_off(share.min(g.len()));
        taken.append(g);
        leftovers.push(rest);
    }
    for rest in leftovers {
        let need = total.saturating_sub(taken.len());
        taken.extend(rest.into_iter().take(need));
    }
    taken
}

fn expected_scripts(lang: &str) -> &'static [Script] {
    match lang {
        "ka" => &[Script::Georgian],
        "ja" => &[Script::Han, Script::Hiragana, Script::Katakana],
        _ => &[Script::Latin],
    }
}

fn script_name(s: Script) -> &'static str {
    match s {
        Script::Latin => "latin",
        Script::Cyrillic => "cyrillic",
        Script::Georgian => "georgian",
        Script::Arabic => "arabic",
        Script::Hebrew => "hebrew",
        Script::Greek => "greek",
        Script::Han => "han",
        Script::Hiragana => "hiragana",
        Script::Katakana => "katakana",
        Script::Hangul => "hangul",
        Script::Thai => "thai",
        Script::Devanagari => "devanagari",
        Script::Other => "other",
    }
}

/// The script of most alphabetic tokens; ties go to the script seen first.
fn dominant_script(name: &str) -> Option<Script> {
    let mut counts: Vec<(Script, usize)> = Vec::new();
    for t in tokenize(name) {
        if t.class != TokenClass::Alpha {
            continue;
        }
        match counts.iter_mut().find(|(s, _)| *s == t.script) {
            Some((_, n)) => *n += 1,
            None => counts.push((t.script, 1)),
        }
    }
    let best = counts.iter().map(|(_, n)| *n).max()?;
    counts.into_iter().find(|(_, n)| *n == best).map(|(s, _)| s)
}

fn in_script(name: &str, lang: &str) -> Option<Script> {
    dominant_script(name).filter(|s| expected_scripts(lang).contains(s))
}

/// NFC with every whitespace run (including no-break spaces, which GLEIF names contain)
/// collapsed to one space, so names that differ only in spacing share a split.
fn normalize(name: &str) -> String {
    let nfc: String = name.nfc().collect();
    nfc.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Prefers the native-language name when it is written in the label language's script,
/// then applies the drop rules. Returns the normalized name and its dominant script.
fn clean_person(label: &str, native: &str, lang: &str) -> Result<(String, Script), Rejected> {
    let native = normalize(native);
    let name = if !native.is_empty() && in_script(&native, lang).is_some() {
        native
    } else {
        normalize(label)
    };
    if name
        .chars()
        .any(|c| matches!(c, '(' | ')' | ',' | '/') || c.is_ascii_digit())
    {
        return Err(Rejected::PunctuationOrDigit);
    }
    if name.len() > MAX_PERSON_BYTES {
        return Err(Rejected::TooLong);
    }
    let script = in_script(&name, lang).ok_or(Rejected::WrongScript)?;
    let alpha = tokenize(&name)
        .iter()
        .filter(|t| t.class == TokenClass::Alpha)
        .count();
    // Japanese labels are often written without a space, so one token is a full name there.
    if alpha < 2 && matches!(script, Script::Latin | Script::Georgian) {
        return Err(Rejected::OneToken);
    }
    Ok((name, script))
}

/// GLEIF's rules for every organization name; `lang` adds the script check used for
/// Wikidata labels, whose language tag GLEIF names do not reliably carry.
fn clean_org(name: &str, lang: Option<&str>) -> Result<String, Rejected> {
    let name = normalize(name);
    if name.len() < MIN_ORG_BYTES {
        return Err(Rejected::TooShort);
    }
    // `LEI` as a whole word marks a name that cites its identifier; as a substring it would
    // also drop LEICESTER, LEISURE, and LEIPZIG.
    if tokenize(&name)
        .iter()
        .any(|t| &name[t.start..t.end] == "LEI")
    {
        return Err(Rejected::ContainsLei);
    }
    if let Some(lang) = lang
        && in_script(&name, lang).is_none()
    {
        return Err(Rejected::WrongScript);
    }
    Ok(name)
}

fn qid_of(uri: &str) -> String {
    uri.rsplit('/').next().unwrap_or(uri).to_string()
}

fn sample_orgs(
    names: &NamesConfig,
    countries: &[&Country],
    raw: &Path,
    refresh: bool,
) -> anyhow::Result<OrgSample> {
    let dir = raw.join("gleif");
    let gleif = cached_download(&dir, "lei2-golden-copy.csv.zip", refresh, || {
        let body = get_with_retries(GLEIF_LATEST, &[], "application/json", "gleif latest")?;
        let latest: serde_json::Value = serde_json::from_str(&body)?;
        match latest["data"]["full_file"]["csv"]["url"].as_str() {
            Some(url) => Ok(url.to_string()),
            None => bail!("no data.full_file.csv.url in the golden copy response:\n{body}"),
        }
    })?;
    let mut gleif_dropped = Dropped::default();
    let mut by_country = read_gleif(&gleif.path, countries, &mut gleif_dropped)?;
    let mut sample = OrgSample {
        orgs: Vec::new(),
        gleif,
        gleif_dropped,
        gleif_available: BTreeMap::new(),
        wikidata_files: Vec::new(),
        wikidata_dropped: Dropped::default(),
        shortfall: BTreeMap::new(),
        saturated: Vec::new(),
    };
    for country in countries {
        let mut rng = country_rng(names.sample_seed, country.code);
        let mut gleif = by_country.remove(country.code).unwrap_or_default();
        sample.gleif_available.insert(country.code, gleif.len());
        gleif.shuffle(&mut rng);
        gleif.truncate(names.orgs_per_country);
        let need = names.orgs_per_country - gleif.len();
        let mut seen: HashSet<String> = gleif.iter().map(|o| o.name.to_lowercase()).collect();
        sample.orgs.extend(gleif);
        if need == 0 {
            continue;
        }
        let mut wikidata = Vec::new();
        for (&lang, &class) in country
            .label_langs
            .iter()
            .flat_map(|l| ORG_CLASSES.iter().map(move |c| (l, c)))
        {
            let path = raw
                .join("wikidata")
                .join(format!("orgs-{}-{lang}-{class}.csv", country.code));
            let what = format!("orgs {} {lang} {class}", country.code);
            fetch_sparql(&orgs_query(country.qid, lang, class), &path, refresh, &what)?;
            let rows = read_rows::<OrgRow>(&path)?;
            if rows.len() >= ORG_QUERY_LIMIT {
                sample.saturated.push(file_name(&path));
            }
            for row in rows {
                let name = match clean_org(&row.label, Some(lang)) {
                    Ok(n) => n,
                    Err(r) => {
                        sample.wikidata_dropped.add(r);
                        continue;
                    }
                };
                if !seen.insert(name.to_lowercase()) {
                    sample.wikidata_dropped.add(Rejected::Duplicate);
                    continue;
                }
                let (group_id, split) = split_of(&name);
                wikidata.push(Org {
                    name,
                    country: country.code,
                    jurisdiction: String::new(),
                    lang: lang.to_string(),
                    elf_code: String::new(),
                    lei: String::new(),
                    qid: qid_of(&row.item),
                    source: OrgSource::Wikidata,
                    group_id,
                    split,
                });
            }
            sample.wikidata_files.push(path);
        }
        wikidata.shuffle(&mut rng);
        wikidata.truncate(need);
        if wikidata.len() < need {
            sample.shortfall.insert(country.code, need - wikidata.len());
        }
        sample.orgs.extend(wikidata);
    }
    sample.wikidata_files.sort();
    Ok(sample)
}

/// Streams the golden copy's single CSV out of its zip and keeps active entities of the
/// configured countries outside `DROPPED_CATEGORIES`.
fn read_gleif(
    zip_path: &Path,
    countries: &[&Country],
    dropped: &mut Dropped,
) -> anyhow::Result<HashMap<&'static str, Vec<Org>>> {
    let file = File::open(zip_path).with_context(|| format!("opening {}", zip_path.display()))?;
    let mut archive = zip::ZipArchive::new(BufReader::new(file))?;
    let entry = archive.by_index(0)?;
    let mut csv = csv::ReaderBuilder::new().from_reader(BufReader::with_capacity(1 << 20, entry));
    let headers = csv.byte_headers()?.clone();
    let col = |name: &str| -> anyhow::Result<usize> {
        headers
            .iter()
            .position(|h| h == name.as_bytes())
            .with_context(|| format!("golden copy has no {name} column"))
    };
    let (lei, legal_name, lang, jurisdiction, elf, status, category) = (
        col("LEI")?,
        col("Entity.LegalName")?,
        col("Entity.LegalName.xmllang")?,
        col("Entity.LegalJurisdiction")?,
        col("Entity.LegalForm.EntityLegalFormCode")?,
        col("Entity.EntityStatus")?,
        col("Entity.EntityCategory")?,
    );
    let mut by_country: HashMap<&'static str, Vec<Org>> = HashMap::new();
    let mut seen = HashSet::new();
    let mut record = csv::ByteRecord::new();
    let mut rows = 0u64;
    while csv.read_byte_record(&mut record)? {
        rows += 1;
        if rows.is_multiple_of(500_000) {
            eprintln!("gleif: {rows} records read");
        }
        // Codes are ASCII, so a field that is not UTF-8 simply matches nothing.
        let field = |i: usize| std::str::from_utf8(record.get(i).unwrap_or_default()).ok();
        if field(status) != Some("ACTIVE") {
            continue;
        }
        let j = field(jurisdiction).unwrap_or_default();
        let Some(country) = countries.iter().find(|c| {
            c.jurisdiction_prefixes
                .iter()
                .any(|p| j == *p || j.strip_prefix(p).is_some_and(|r| r.starts_with('-')))
        }) else {
            continue;
        };
        if field(category).is_some_and(|c| DROPPED_CATEGORIES.contains(&c)) {
            dropped.add(Rejected::Category);
            continue;
        }
        let Some(raw_name) = field(legal_name) else {
            dropped.add(Rejected::InvalidUtf8);
            continue;
        };
        let name = match clean_org(raw_name, None) {
            Ok(n) => n,
            Err(r) => {
                dropped.add(r);
                continue;
            }
        };
        if !seen.insert((country.code, name.to_lowercase())) {
            dropped.add(Rejected::Duplicate);
            continue;
        }
        let (group_id, split) = split_of(&name);
        by_country.entry(country.code).or_default().push(Org {
            name,
            country: country.code,
            jurisdiction: j.to_string(),
            lang: field(lang).unwrap_or_default().to_string(),
            elf_code: field(elf).unwrap_or_default().to_string(),
            lei: field(lei).unwrap_or_default().to_string(),
            qid: String::new(),
            source: OrgSource::Gleif,
            group_id,
            split,
        });
    }
    eprintln!("gleif: {rows} records read");
    Ok(by_country)
}

fn elf(raw: &Path, refresh: bool) -> anyhow::Result<Elf> {
    let download = cached_download(&raw.join("gleif"), "elf-code-list.csv", refresh, || {
        let page = get_with_retries(ELF_PAGE, &[], "text/html", "elf code list page")?;
        elf_link(&page).with_context(|| format!("no elf-code-list csv link on {ELF_PAGE}"))
    })?;
    let file = File::open(&download.path)
        .with_context(|| format!("opening {}", download.path.display()))?;
    Ok(Elf {
        abbreviations: parse_elf(file)?,
        download,
    })
}

fn elf_link(page: &str) -> Option<String> {
    page.split('"')
        .find(|s| s.contains("elf-code-list") && s.ends_with(".csv"))
        .map(|s| {
            if s.starts_with('/') {
                format!("https://www.gleif.org{s}")
            } else {
                s.to_string()
            }
        })
}

/// Maps each ELF code to its abbreviations, local-language ones first. The list has one row
/// per code and language in multilingual jurisdictions, so rows of one code are merged in
/// file order; codes without any abbreviation are left out.
fn parse_elf(reader: impl Read) -> anyhow::Result<HashMap<String, Vec<String>>> {
    let mut csv = csv::ReaderBuilder::new().from_reader(reader);
    let headers = csv.headers()?.clone();
    let col = |name: &str| -> anyhow::Result<usize> {
        headers
            .iter()
            .position(|h| h == name)
            .with_context(|| format!("elf code list has no {name} column"))
    };
    let code = col("ELF Code")?;
    let local = col("Abbreviations Local language")?;
    let translit = col("Abbreviations transliterated")?;
    let mut out: HashMap<String, Vec<String>> = HashMap::new();
    for record in csv.records() {
        let record = record?;
        for column in [local, translit] {
            for a in record.get(column).unwrap_or_default().split(';') {
                let a = a.trim();
                if a.is_empty() {
                    continue;
                }
                let abbrs = out
                    .entry(record.get(code).unwrap_or_default().to_string())
                    .or_default();
                if !abbrs.iter().any(|x| x == a) {
                    abbrs.push(a.to_string());
                }
            }
        }
    }
    Ok(out)
}

/// Returns `dir/name`, downloading it from the URL `resolve` returns unless it is cached.
/// The URL is kept next to the file in `name.url`, written before the file is moved into
/// place, so a cached file always has the URL it came from.
fn cached_download(
    dir: &Path,
    name: &str,
    refresh: bool,
    resolve: impl FnOnce() -> anyhow::Result<String>,
) -> anyhow::Result<Download> {
    let path = dir.join(name);
    let url_file = dir.join(format!("{name}.url"));
    if path.exists() && url_file.exists() && !refresh {
        let url = std::fs::read_to_string(&url_file)
            .with_context(|| format!("reading {}", url_file.display()))?;
        eprintln!("{name}: cached");
        return Ok(Download {
            path,
            url: url.trim().to_string(),
        });
    }
    let url = resolve()?;
    create_dir(dir)?;
    eprintln!("{name}: downloading {url}");
    let part = part_path(&path);
    let mut response = ureq::get(&url)
        .header("User-Agent", USER_AGENT)
        .call()
        .with_context(|| format!("downloading {url}"))?;
    let mut file = File::create(&part).with_context(|| format!("creating {}", part.display()))?;
    std::io::copy(&mut response.body_mut().as_reader(), &mut file)
        .with_context(|| format!("downloading {url}"))?;
    write_file(&url_file, format!("{url}\n").as_bytes())?;
    std::fs::rename(&part, &path).with_context(|| format!("writing {}", path.display()))?;
    Ok(Download { path, url })
}

/// Runs a SPARQL query unless its CSV is cached and complete. The endpoint sometimes
/// answers with status 200 and a truncated body (a stack trace after a timeout, or a stream
/// cut mid-row), and its HTTP cache then serves that same body for the same query text for
/// minutes. Each retry therefore appends a SPARQL comment, which changes the URL without
/// changing the query. A truncated body is never cached.
fn fetch_sparql(query: &str, path: &Path, refresh: bool, what: &str) -> anyhow::Result<()> {
    if path.exists() && !refresh {
        let body =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        if !is_truncated(&body) {
            return Ok(());
        }
        eprintln!("wikidata {what}: cached file is truncated, fetching again");
    }
    if let Some(dir) = path.parent() {
        create_dir(dir)?;
    }
    let mut delay = 5;
    for attempt in 0..MAX_ATTEMPTS {
        let text = if attempt == 0 {
            query.to_string()
        } else {
            format!("{query}\n# attempt {attempt}")
        };
        let body = get_with_retries(ENDPOINT, &[("query", &text)], "text/csv", what)?;
        if !is_truncated(&body) {
            write_file(path, body.as_bytes())?;
            eprintln!("wikidata {what}: fetched");
            std::thread::sleep(Duration::from_secs(1));
            return Ok(());
        }
        eprintln!("wikidata {what}: truncated body, retry {attempt}");
        delay = backoff(delay, None);
    }
    bail!("wikidata query for {what} failed after {MAX_ATTEMPTS} attempts")
}

/// A complete result ends in a newline and parses as CSV with the header's field count; a
/// timed-out one carries a stack trace, and a dropped stream stops mid-row.
fn is_truncated(body: &str) -> bool {
    if body.contains("java.util.concurrent") || body.contains("\n\tat ") || !body.ends_with('\n') {
        return true;
    }
    csv::Reader::from_reader(body.as_bytes())
        .records()
        .any(|r| r.is_err())
}

/// GET with up to `MAX_ATTEMPTS` attempts. Status 429 and 5xx, dropped connections, and
/// bodies cut off mid-stream are retried; any other non-success status is an error.
fn get_with_retries(
    url: &str,
    query: &[(&str, &str)],
    accept: &str,
    what: &str,
) -> anyhow::Result<String> {
    let mut delay = 5;
    for attempt in 0..MAX_ATTEMPTS {
        if attempt > 0 {
            eprintln!("{what}: retry {attempt}");
        }
        let mut request = ureq::get(url)
            .header("Accept", accept)
            .header("User-Agent", USER_AGENT)
            .config()
            .http_status_as_error(false)
            .build();
        for (k, v) in query {
            request = request.query(*k, *v);
        }
        let wait = match request.call() {
            Ok(mut r) => {
                let status = r.status().as_u16();
                if status == 429 || status >= 500 {
                    eprintln!("{what}: status {status}");
                    retry_after(&r)
                } else if status >= 400 {
                    bail!("{what}: status {status}");
                } else {
                    match r.body_mut().with_config().limit(256 << 20).read_to_string() {
                        Ok(body) => return Ok(body),
                        Err(e) => {
                            eprintln!("{what}: body failed ({e})");
                            None
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("{what}: {e}");
                None
            }
        };
        if attempt + 1 < MAX_ATTEMPTS {
            delay = backoff(delay, wait);
        }
    }
    bail!("{what} failed after {MAX_ATTEMPTS} attempts")
}

/// Sleeps for the server's `Retry-After` when it gave one, otherwise for `delay` seconds,
/// and returns the next delay (doubling, capped at a minute).
fn backoff(delay: u64, retry_after: Option<u64>) -> u64 {
    std::thread::sleep(Duration::from_secs(retry_after.unwrap_or(delay)));
    (delay * 2).min(60)
}

/// The `Retry-After` header in seconds, capped at five minutes.
fn retry_after(response: &ureq::http::Response<ureq::Body>) -> Option<u64> {
    let secs = response
        .headers()
        .get("Retry-After")?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()?;
    Some(secs.min(300))
}

fn read_rows<T: DeserializeOwned>(path: &Path) -> anyhow::Result<Vec<T>> {
    let mut csv = csv::ReaderBuilder::new()
        .from_path(path)
        .with_context(|| format!("opening {}", path.display()))?;
    csv.deserialize()
        .collect::<Result<Vec<T>, _>>()
        .with_context(|| format!("reading {}", path.display()))
}

/// Writes through a temporary file and a rename, so an interrupted write never leaves a
/// partial file where a cached one is expected.
fn write_file(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let part = part_path(path);
    std::fs::write(&part, bytes).with_context(|| format!("writing {}", part.display()))?;
    std::fs::rename(&part, path).with_context(|| format!("writing {}", path.display()))
}

/// `path` with `.part` appended to the whole file name. Replacing the extension instead
/// would give `a.csv.url` and `a.csv` the same temporary file.
fn part_path(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_owned();
    name.push(".part");
    PathBuf::from(name)
}

fn create_dir(dir: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))
}

fn file_name(path: &Path) -> String {
    path.file_name()
        .map_or_else(String::new, |n| n.to_string_lossy().into_owned())
}

/// `fnv1a(lowercased name, 0) % 100`: 0 to 89 train, 90 to 94 valid, 95 to 99 test.
fn split_of(name: &str) -> (u64, Split) {
    let group_id = fnv1a(name.to_lowercase().as_bytes(), 0);
    let split = match group_id % 100 {
        0..90 => Split::Train,
        90..95 => Split::Valid,
        _ => Split::Test,
    };
    (group_id, split)
}

const SPLIT_RULE: &str = "fnv1a64(0, lowercase normalized name) % 100: 0-89 train, 90-94 valid, 95-99 test; names are NFC with whitespace runs collapsed to one space";

/// Guards the plan's acceptance rule that no lowercased name appears in two splits.
fn ensure_disjoint_splits<'a>(rows: impl Iterator<Item = (&'a str, Split)>) -> anyhow::Result<()> {
    let mut by_name: HashMap<String, Split> = HashMap::new();
    for (name, split) in rows {
        if let Some(prev) = by_name.insert(name.to_lowercase(), split)
            && prev != split
        {
            bail!("{name} is in both {} and {}", prev.name(), split.name());
        }
    }
    Ok(())
}

fn write_parquet(path: &Path, columns: Vec<Column>) -> anyhow::Result<()> {
    let mut df = DataFrame::new_infer_height(columns)?;
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    ParquetWriter::new(file).finish(&mut df)?;
    Ok(())
}

fn str_column<'a, T>(name: &str, rows: &'a [T], f: impl Fn(&'a T) -> &'a str) -> Column {
    Column::new(name.into(), rows.iter().map(f).collect::<Vec<_>>())
}

fn write_people(path: &Path, people: &[Person]) -> anyhow::Result<()> {
    write_parquet(
        path,
        vec![
            str_column("name", people, |p| &p.name),
            str_column("country", people, |p| p.country),
            str_column("lang", people, |p| p.lang),
            str_column("script", people, |p| script_name(p.script)),
            str_column("qid", people, |p| &p.qid),
            Column::new(
                "group_id".into(),
                people.iter().map(|p| p.group_id).collect::<Vec<_>>(),
            ),
            str_column("split", people, |p| p.split.name()),
        ],
    )
}

fn write_orgs(
    path: &Path,
    orgs: &[Org],
    abbreviations: &HashMap<String, Vec<String>>,
) -> anyhow::Result<()> {
    write_parquet(
        path,
        vec![
            str_column("name", orgs, |o| &o.name),
            str_column("country", orgs, |o| o.country),
            str_column("jurisdiction", orgs, |o| &o.jurisdiction),
            str_column("lang", orgs, |o| &o.lang),
            str_column("elf_code", orgs, |o| &o.elf_code),
            str_column("legal_form_abbr", orgs, |o| {
                abbreviations
                    .get(&o.elf_code)
                    .and_then(|a| a.first())
                    .map_or("", String::as_str)
            }),
            str_column("lei", orgs, |o| &o.lei),
            str_column("qid", orgs, |o| &o.qid),
            str_column("source", orgs, |o| o.source.name()),
            Column::new(
                "group_id".into(),
                orgs.iter().map(|o| o.group_id).collect::<Vec<_>>(),
            ),
            str_column("split", orgs, |o| o.split.name()),
        ],
    )
}

/// The acceptance checks the plan puts on the people sample, as warnings: fewer than
/// `MIN_PEOPLE` in a country, or a native script under 40% of GE or JP rows.
fn check_people(countries: &[&Country], people: &[Person]) -> Vec<String> {
    let mut warnings = Vec::new();
    for c in countries {
        let rows: Vec<&Person> = people.iter().filter(|p| p.country == c.code).collect();
        if rows.len() < MIN_PEOPLE {
            warnings.push(format!(
                "{} has {} people, fewer than {MIN_PEOPLE}",
                c.code,
                rows.len()
            ));
        }
        let native = expected_scripts(c.label_langs[0]);
        if native != [Script::Latin] && !rows.is_empty() {
            let share = rows.iter().filter(|p| native.contains(&p.script)).count() as f64
                / rows.len() as f64;
            if share < 0.4 {
                warnings.push(format!(
                    "{} native-script share is {:.0}%, under 40%",
                    c.code,
                    share * 100.0
                ));
            }
        }
    }
    warnings
}

/// Split counts per key plus a `total`, for the manifests' `per_country` and `per_script`.
type SplitCounts<'a> = BTreeMap<&'a str, BTreeMap<&'static str, usize>>;

fn split_counts<'a>(rows: impl Iterator<Item = (&'a str, Split)>) -> SplitCounts<'a> {
    let mut out = SplitCounts::new();
    for (key, split) in rows {
        let m = out.entry(key).or_default();
        *m.entry(split.name()).or_default() += 1;
        *m.entry("total").or_default() += 1;
    }
    out
}

/// Totals per split plus the named breakdowns.
fn counts(splits: &[Split], per: &[(&str, SplitCounts)]) -> serde_json::Value {
    let mut out = serde_json::Map::new();
    for s in Split::ALL {
        out.insert(
            s.name().into(),
            splits.iter().filter(|x| **x == s).count().into(),
        );
    }
    for (key, value) in per {
        out.insert((*key).into(), serde_json::json!(value));
    }
    out.into()
}

/// SHA-256 over the files concatenated in the given order.
fn sha256_files(paths: &[PathBuf]) -> anyhow::Result<String> {
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    for path in paths {
        let mut f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
        }
    }
    Ok(hex(&hasher.finalize()))
}

/// Merges `fields` into the manifest at `path` and removes `stale` keys, keeping everything
/// else Milestone 0 recorded.
fn update_manifest(path: &Path, fields: serde_json::Value, stale: &[&str]) -> anyhow::Result<()> {
    let mut manifest: serde_json::Map<String, serde_json::Value> = if path.exists() {
        serde_json::from_str(&std::fs::read_to_string(path)?)
            .with_context(|| format!("parsing {}", path.display()))?
    } else {
        serde_json::Map::new()
    };
    let serde_json::Value::Object(fields) = fields else {
        bail!("manifest fields for {} are not an object", path.display());
    };
    manifest.extend(fields);
    for key in stale {
        manifest.remove(*key);
    }
    write_file(
        path,
        (serde_json::to_string_pretty(&manifest)? + "\n").as_bytes(),
    )
}

fn write_people_manifest(
    manifests: &Path,
    names: &NamesConfig,
    countries: &[&Country],
    sample: &PeopleSample,
    warnings: &[String],
    date: &str,
) -> anyhow::Result<()> {
    let people = &sample.people;
    let splits: Vec<Split> = people.iter().map(|p| p.split).collect();
    let langs: BTreeMap<&str, &[&str]> =
        countries.iter().map(|c| (c.code, c.label_langs)).collect();
    update_manifest(
        &manifests.join("wikidata-people.json"),
        serde_json::json!({
            "use": "training",
            "status": "downloaded",
            "format": format!("SPARQL CSV results, one file per country, label language, and birth year; a year that reaches the {PEOPLE_QUERY_LIMIT}-row limit is replaced by one file per month"),
            "sha256": sha256_files(&sample.files)?,
            "sha256_rule": "over the cached files concatenated in sorted file-name order",
            "downloaded": date,
            "filters": {
                "properties": ["P31=Q5", "P27", "P569", "P1559", "rdfs:label"],
                "birth_years": [names.birth_years.0, names.birth_years.1],
                "label_langs": langs,
                "query_limit": PEOPLE_QUERY_LIMIT,
                "rules": format!("native name (P1559) used when in the label language's script; names NFC with whitespace collapsed; dropped: parentheses, commas, slashes, or ASCII digits; over {MAX_PERSON_BYTES} bytes; dominant script not the language's; fewer than two alphabetic tokens in Latin or Georgian; duplicates by (country, lowercase name)"),
                "people_per_country": names.people_per_country,
            },
            "files": sample.files.len(),
            "counts": counts(&splits, &[
                ("per_country", split_counts(people.iter().map(|p| (p.country, p.split)))),
                ("per_script", split_counts(people.iter().map(|p| (script_name(p.script), p.split)))),
            ]),
            "sample_seed": names.sample_seed,
            "split_rule": SPLIT_RULE,
            "dropped": sample.dropped,
            "warnings": warnings,
        }),
        &["dropped_tags", "split_seed"],
    )
}

fn write_gleif_manifest(
    manifests: &Path,
    names: &NamesConfig,
    countries: &[&Country],
    sample: &OrgSample,
    date: &str,
) -> anyhow::Result<()> {
    let rows: Vec<&Org> = sample
        .orgs
        .iter()
        .filter(|o| o.source == OrgSource::Gleif)
        .collect();
    let splits: Vec<Split> = rows.iter().map(|o| o.split).collect();
    let jurisdictions: BTreeMap<&str, &[&str]> = countries
        .iter()
        .map(|c| (c.code, c.jurisdiction_prefixes))
        .collect();
    let mut counts = counts(
        &splits,
        &[(
            "per_country",
            split_counts(rows.iter().map(|o| (o.country, o.split))),
        )],
    );
    counts["available"] = serde_json::json!(sample.gleif_available);
    update_manifest(
        &manifests.join("gleif-lei.json"),
        serde_json::json!({
            "use": "training",
            "status": "downloaded",
            "url": sample.gleif.url,
            "format": "Level 1 golden copy CSV (LEI-CDF 3.1), streamed from its zip; columns used: LEI, Entity.LegalName, Entity.LegalName.xmllang, Entity.LegalJurisdiction, Entity.LegalForm.EntityLegalFormCode, Entity.EntityStatus, Entity.EntityCategory",
            "sha256": sha256_files(std::slice::from_ref(&sample.gleif.path))?,
            "downloaded": date,
            "filters": {
                "status": "ACTIVE",
                "categories_dropped": DROPPED_CATEGORIES,
                "jurisdictions": jurisdictions,
                "rules": format!("names NFC with whitespace collapsed; dropped: shorter than {MIN_ORG_BYTES} bytes, the word LEI, duplicates by (country, lowercase name)"),
                "orgs_per_country": names.orgs_per_country,
            },
            "counts": counts,
            "sample_seed": names.sample_seed,
            "split_rule": SPLIT_RULE,
            "dropped": sample.gleif_dropped,
        }),
        &["dropped_tags", "split_seed"],
    )
}

fn write_wikidata_orgs_manifest(
    manifests: &Path,
    names: &NamesConfig,
    sample: &OrgSample,
    warnings: &[String],
    date: &str,
) -> anyhow::Result<()> {
    let rows: Vec<&Org> = sample
        .orgs
        .iter()
        .filter(|o| o.source == OrgSource::Wikidata)
        .collect();
    let splits: Vec<Split> = rows.iter().map(|o| o.split).collect();
    let mut counts = counts(
        &splits,
        &[(
            "per_country",
            split_counts(rows.iter().map(|o| (o.country, o.split))),
        )],
    );
    counts["shortfall"] = serde_json::json!(sample.shortfall);
    update_manifest(
        &manifests.join("wikidata-organizations.json"),
        serde_json::json!({
            "use": "top-up for countries where GLEIF has fewer usable names than the quota",
            "status": "downloaded",
            "format": "SPARQL CSV results, one file per country, label language, and class",
            "sha256": sha256_files(&sample.wikidata_files)?,
            "sha256_rule": "over the cached files concatenated in sorted file-name order",
            "downloaded": date,
            "filters": {
                "properties": ["P31 in classes", "P17 country", "rdfs:label"],
                "classes": ORG_CLASSES,
                "query_limit": ORG_QUERY_LIMIT,
                "rules": format!("names NFC with whitespace collapsed; the label's dominant script must match its language; dropped: shorter than {MIN_ORG_BYTES} bytes, the word LEI, duplicates of GLEIF names and of each other"),
            },
            "files": sample.wikidata_files.len(),
            "counts": counts,
            "sample_seed": names.sample_seed,
            "split_rule": SPLIT_RULE,
            "dropped": sample.wikidata_dropped,
            "warnings": warnings,
        }),
        &["dropped_tags", "split_seed"],
    )
}

fn write_elf_manifest(manifests: &Path, elf: &Elf, date: &str) -> anyhow::Result<()> {
    update_manifest(
        &manifests.join("gleif-elf.json"),
        serde_json::json!({
            "source": "gleif-elf",
            "kind": "org",
            "use": "maps GLEIF legal form codes to abbreviations for the generator; not redistributed",
            "status": "downloaded",
            "url": elf.download.url,
            "index_url": ELF_PAGE,
            "license": "unstated",
            "attribution": "GLEIF, ISO 20275 Entity Legal Forms code list",
            "constraints": [
                "the code list states no license of its own and the LEI Data Terms of Use do not name it (internal licensing review, 4.4); the list itself is never committed or shipped, and legal-form abbreviations are facts about company law",
            ],
            "sha256": sha256_files(std::slice::from_ref(&elf.download.path))?,
            "downloaded": date,
            "counts": { "codes_with_abbreviations": elf.abbreviations.len() },
        }),
        &[],
    )
}

fn print_summary(countries: &[&Country], people: &[Person], orgs: &[Org]) {
    let mut people_rows: BTreeMap<(&str, &str, &str), BTreeMap<Split, usize>> = BTreeMap::new();
    for p in people {
        let key = (p.country, p.lang, script_name(p.script));
        *people_rows
            .entry(key)
            .or_default()
            .entry(p.split)
            .or_default() += 1;
    }
    let mut org_rows: BTreeMap<(&str, &str, &str), BTreeMap<Split, usize>> = BTreeMap::new();
    for o in orgs {
        let key = (o.country, o.source.name(), "");
        *org_rows.entry(key).or_default().entry(o.split).or_default() += 1;
    }
    for (title, rows) in [("people", people_rows), ("orgs", org_rows)] {
        println!(
            "{title:<8}{:<10}{:<12}{:>8}{:>8}{:>8}{:>8}",
            "", "", "train", "valid", "test", "total"
        );
        for c in countries {
            for ((country, a, b), n) in rows.iter().filter(|(k, _)| k.0 == c.code) {
                let by: Vec<usize> = Split::ALL
                    .iter()
                    .map(|s| n.get(s).copied().unwrap_or(0))
                    .collect();
                println!(
                    "{country:<8}{a:<10}{b:<12}{:>8}{:>8}{:>8}{:>8}",
                    by[0],
                    by[1],
                    by[2],
                    by.iter().sum::<usize>()
                );
            }
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn people_query_names_the_country_dates_and_language() {
        let q = people_query("Q230", "ka", "1975-01-01", "1976-01-01");
        assert!(q.contains("wd:Q230"));
        assert!(q.contains("?dob >= \"1975-01-01T00:00:00Z\"^^xsd:dateTime"));
        assert!(q.contains("?dob < \"1976-01-01T00:00:00Z\"^^xsd:dateTime"));
        assert!(q.contains("LANG(?label) = \"ka\""));
        assert!(q.contains(&format!("LIMIT {PEOPLE_QUERY_LIMIT}")));
    }

    #[test]
    fn month_bounds_roll_over_the_year() {
        assert_eq!(
            month_bounds(1975, 1),
            ("1975-01-01".to_string(), "1975-02-01".to_string())
        );
        assert_eq!(
            month_bounds(1975, 12),
            ("1975-12-01".to_string(), "1976-01-01".to_string())
        );
    }

    #[test]
    fn person_filter_drops_disambiguations_suffixes_and_mononyms() {
        assert_eq!(
            clean_person("Nino Beridze (politician)", "", "en"),
            Err(Rejected::PunctuationOrDigit)
        );
        assert_eq!(
            clean_person("John Smith, Jr.", "", "en"),
            Err(Rejected::PunctuationOrDigit)
        );
        assert_eq!(clean_person("ნინო", "", "ka"), Err(Rejected::OneToken));
        assert_eq!(
            clean_person("Nino Beridze", "", "ka"),
            Err(Rejected::WrongScript)
        );
        assert_eq!(
            clean_person("ნინო ბერიძე", "", "ka"),
            Ok(("ნინო ბერიძე".to_string(), Script::Georgian))
        );
        assert_eq!(
            clean_person("山田太郎", "", "ja"),
            Ok(("山田太郎".to_string(), Script::Han))
        );
    }

    #[test]
    fn person_filter_prefers_a_native_name_in_the_right_script() {
        let (name, _) = clean_person("Taro Yamada", "山田太郎", "ja").unwrap();
        assert_eq!(name, "山田太郎");
        let (name, _) = clean_person("Taro Yamada", "山田太郎", "en").unwrap();
        assert_eq!(name, "Taro Yamada");
    }

    #[test]
    fn names_are_nfc_with_whitespace_collapsed() {
        let (name, _) = clean_person(" Jose\u{301}\u{a0} Garci\u{301}a ", "", "en").unwrap();
        assert_eq!(name, "Jos\u{e9} Garc\u{ed}a");
        assert_eq!(
            split_of(&normalize("ACME  LTD")),
            split_of(&normalize("ACME\u{a0}LTD"))
        );
    }

    #[test]
    fn split_depends_only_on_the_lowercased_name() {
        let a = split_of("Nino Beridze");
        assert_eq!(a, split_of("nino beridze"));
        assert_eq!(a, split_of("NINO BERIDZE"));
        let rows = [
            ("Nino Beridze", a.1),
            ("nino beridze", a.1),
            ("Anna Schmidt", split_of("Anna Schmidt").1),
        ];
        assert!(ensure_disjoint_splits(rows.into_iter()).is_ok());
        let other = if a.1 == Split::Train {
            Split::Test
        } else {
            Split::Train
        };
        let leaked = [("Nino Beridze", a.1), ("nino beridze", other)];
        assert!(ensure_disjoint_splits(leaked.into_iter()).is_err());
    }

    #[test]
    fn org_filter_applies_gleif_rules_and_the_wikidata_script_check() {
        assert_eq!(clean_org("AB", None), Err(Rejected::TooShort));
        assert_eq!(
            clean_org("ACME LEI 5493001KJTIIGC8Y1R12", None),
            Err(Rejected::ContainsLei)
        );
        assert_eq!(
            clean_org("LEICESTER LEISURE LTD", None),
            Ok("LEICESTER LEISURE LTD".to_string())
        );
        assert_eq!(
            clean_org("Acme Widgets GmbH", None),
            Ok("Acme Widgets GmbH".to_string())
        );
        assert_eq!(
            clean_org("Tbilisi State University", Some("ka")),
            Err(Rejected::WrongScript)
        );
        assert!(clean_org("თბილისის სახელმწიფო უნივერსიტეტი", Some("ka")).is_ok());
    }

    #[test]
    fn elf_abbreviations_merge_rows_of_one_code_local_first() {
        let csv = "\"ELF Code\",\"Country of formation\",\"Entity Legal Form name Local name\",\"Abbreviations Local language\",\"Abbreviations transliterated\"\n\
                   \"8888\",\"\",\"\",\"\",\"\"\n\
                   \"2HBR\",\"Germany\",\"Gesellschaft mit beschränkter Haftung\",\"GmbH\",\"\"\n\
                   \"AXSB\",\"Austria\",\"Gesellschaft mit beschränkter Haftung\",\"GmbH;Ges.m.b.H.\",\"GmbH\"\n\
                   \"BVVE\",\"Canada\",\"Corporation\",\"Corp.\",\"\"\n\
                   \"BVVE\",\"Canada\",\"Société par actions\",\"SPA;Corp.\",\"\"\n";
        let map = parse_elf(csv.as_bytes()).unwrap();
        assert_eq!(map.len(), 3);
        assert_eq!(map["2HBR"], vec!["GmbH"]);
        assert_eq!(map["AXSB"], vec!["GmbH", "Ges.m.b.H."]);
        assert_eq!(map["BVVE"], vec!["Corp.", "SPA"]);
    }

    #[test]
    fn elf_link_is_found_in_the_code_list_page() {
        let page = r#"<a href="https://www.gleif.org/lei-data/code-lists/iso-20275-entity-legal-forms-code-list/2026-02-19-elf-code-list-v1.6.csv">CSV</a>"#;
        assert_eq!(
            elf_link(page).as_deref(),
            Some(
                "https://www.gleif.org/lei-data/code-lists/iso-20275-entity-legal-forms-code-list/2026-02-19-elf-code-list-v1.6.csv"
            )
        );
    }

    #[test]
    fn truncated_sparql_bodies_are_detected() {
        assert!(is_truncated(
            "item,label\nhttp://www.wikidata.org/entity/Q1,A\njava.util.concurrent.TimeoutException\n\tat x"
        ));
        assert!(is_truncated(
            "item,label\nhttp://www.wikidata.org/entity/Q1,A\nhttp://www.wikidata.org/e"
        ));
        assert!(is_truncated(
            "item,label\nhttp://www.wikidata.org/entity/Q1,A\nQ2\n"
        ));
        assert!(!is_truncated(
            "item,label\nhttp://www.wikidata.org/entity/Q1,A\n"
        ));
        assert!(!is_truncated("item,label\n"));
    }

    #[test]
    fn temporary_files_keep_the_whole_name() {
        assert_eq!(
            part_path(Path::new("gleif/elf-code-list.csv")),
            Path::new("gleif/elf-code-list.csv.part")
        );
        assert_ne!(
            part_path(Path::new("gleif/elf-code-list.csv.url")),
            part_path(Path::new("gleif/elf-code-list.csv"))
        );
    }

    #[test]
    fn shares_fill_a_shortfall_from_other_groups() {
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let taken = take_shares(vec![vec![1, 2], (10..20).collect()], 8, &mut rng);
        assert_eq!(taken.len(), 8);
        assert_eq!(taken.iter().filter(|x| **x < 10).count(), 2);
    }

    #[test]
    fn country_streams_are_independent() {
        let mut a = country_rng(42, "GE");
        let mut b = country_rng(42, "JP");
        let mut xs: Vec<u32> = (0..20).collect();
        let mut ys = xs.clone();
        xs.shuffle(&mut a);
        ys.shuffle(&mut b);
        assert_ne!(xs, ys);
        let mut again = country_rng(42, "GE");
        let mut zs: Vec<u32> = (0..20).collect();
        zs.shuffle(&mut again);
        assert_eq!(xs, zs);
    }
}
