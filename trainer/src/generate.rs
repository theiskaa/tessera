//! Synthetic detector documents: templates with slots, filled from the name, organization,
//! and address pools, with exact byte offsets recorded as the string is built.
//!
//! Every email uses a reserved domain and every phone number comes from a per-country
//! generator; only the generators marked `fixture_safe` draw from ranges reserved for
//! fiction. Names and organizations are real (Wikidata, GLEIF, and USAGov), so generated text is
//! training data only and never enters a versioned fixture.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use polars::prelude::{Column, DataFrame, ParquetReader, ParquetWriter, SerReader};
use rand::seq::IndexedRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tessera::Kind;
use tessera::internal::{is_content, tokenize};
use unicode_normalization::UnicodeNormalization;
use unicode_normalization::char::is_combining_mark;

use crate::bodies::{self, Bodies};
use crate::config::{self, Config, GenerateConfig};
use crate::data::{LabelledExample, Split, read_shard};
use crate::filler;
use crate::inflect;
use crate::negatives;
use crate::pool_filter;
use crate::templates::{self, SlotRef, Template};

/// Where a template's documents come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    Prose,
    EmailBody,
    Signature,
    Letterhead,
    Invoice,
    Markdown,
    Table,
    Support,
    Technical,
}

impl Family {
    pub const ALL: [Family; 9] = [
        Family::Prose,
        Family::EmailBody,
        Family::Signature,
        Family::Letterhead,
        Family::Invoice,
        Family::Markdown,
        Family::Table,
        Family::Support,
        Family::Technical,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Family::Prose => "prose",
            Family::EmailBody => "email_body",
            Family::Signature => "signature",
            Family::Letterhead => "letterhead",
            Family::Invoice => "invoice",
            Family::Markdown => "markdown",
            Family::Table => "table",
            Family::Support => "support",
            Family::Technical => "technical",
        }
    }

    /// Families that appear only in the test split. Markdown stays out of training until the
    /// generator featurizes it with the Markdown mask, as the library does.
    pub fn held_out(self) -> bool {
        matches!(self, Family::Markdown)
    }
}

/// Which labelled kinds a document contains; quotas over these teach the detector that an
/// absent entity is common.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    Nothing,
    NoPersonWithAddress,
    NoAddressWithPerson,
    Both,
}

impl Category {
    const ALL: [Category; 4] = [
        Category::Nothing,
        Category::NoPersonWithAddress,
        Category::NoAddressWithPerson,
        Category::Both,
    ];

    /// Sixths: nothing, no person, and no address one each; both three.
    fn weight(self) -> u32 {
        match self {
            Category::Both => 3,
            _ => 1,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Category::Nothing => "nothing",
            Category::NoPersonWithAddress => "no_person_with_address",
            Category::NoAddressWithPerson => "no_address_with_person",
            Category::Both => "both",
        }
    }

    fn matches(self, entities: &[Gold]) -> bool {
        let person = entities.iter().any(|span| span.kind == "person");
        let address = entities.iter().any(|span| span.kind == "address");
        match self {
            Category::Nothing => !person && !address,
            Category::NoPersonWithAddress => !person && address,
            Category::NoAddressWithPerson => person && !address,
            Category::Both => person && address,
        }
    }
}

/// A placeholder in a template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Slot {
    Person,
    PersonFirst,
    PersonLast,
    PersonNative,
    /// A directory line's name, surname in capitals, with its role glued on by a hyphen and
    /// outside the span: `Joachim NAGEL-President`.
    PersonDirectory,
    PersonList,
    PersonEponymous,
    Org,
    OrgEponymous,
    OrgSchool,
    OrgDistrict,
    OrgGov,
    OrgAcronym,
    OrgUnit,
    OrgLocalOffice,
    OrgRegistry,
    OrgChain,
    OrgUniv,
    OrgCouncil,
    OrgParty,
    OrgMedia,
    OrgList,
    Address,
    AddressMultiline,
    AddressPrefixed,
    Email,
    Phone,
    PhoneLocal,
    NegPlace,
    NegDistrict,
    NegWordName,
    NegHandle,
    NegUrl,
    NegDate,
    NegPrice,
    NegOrder,
    NegIp,
    NegDigits,
    NegRoadSentence,
    NegPartialLocation,
    NegGeIncompleteStreet,
    NegHeader,
    NegDepartment,
    NegPrompt,
    NegHeading,
    NegCaps,
    NegLaw,
    NegLabel,
    NegService,
    NegHours,
    NegCitation,
    NegLine,
    NegCode,
    NegList,
    NegUnitCode,
    NegRegisterNo,
    Officers,
    Register,
    TitleLocal,
    SentenceLocal,
    PlaceLocal,
    RoleHeading,
    Greeting,
    Closing,
    Sentence,
    Title,
    Date,
    Product,
}

impl Slot {
    /// The gold label this slot produces, or `None` for filler and negatives.
    pub fn kind(self) -> Option<Kind> {
        use Slot::*;
        match self {
            Person | PersonFirst | PersonLast | PersonNative | PersonDirectory | PersonList
            | PersonEponymous => Some(Kind::Person),
            Org | OrgEponymous | OrgSchool | OrgDistrict | OrgGov | OrgAcronym | OrgUnit
            | OrgLocalOffice | OrgRegistry | OrgChain | OrgUniv | OrgCouncil | OrgParty
            | OrgMedia | OrgList => Some(Kind::Org),
            Address | AddressMultiline | AddressPrefixed => Some(Kind::Address),
            Email => Some(Kind::Email),
            Phone | PhoneLocal => Some(Kind::Phone),
            _ => None,
        }
    }

    pub fn parse(name: &str) -> Option<Slot> {
        use Slot::*;
        Some(match name {
            "person" => Person,
            "person_first" => PersonFirst,
            "person_last" => PersonLast,
            "person_native" => PersonNative,
            "person_directory" => PersonDirectory,
            "person_list" => PersonList,
            "person_eponymous" => PersonEponymous,
            "org" => Org,
            "org_eponymous" => OrgEponymous,
            "org_school" => OrgSchool,
            "org_district" => OrgDistrict,
            "org_gov" => OrgGov,
            "org_acr" => OrgAcronym,
            "org_unit" => OrgUnit,
            "org_local_office" => OrgLocalOffice,
            "org_registry" => OrgRegistry,
            "org_chain" => OrgChain,
            "org_univ" => OrgUniv,
            "org_council" => OrgCouncil,
            "org_media" => OrgMedia,
            "org_party" => OrgParty,
            "org_list" => OrgList,
            "address" => Address,
            "address_ml" => AddressMultiline,
            "address_prefixed" => AddressPrefixed,
            "email" => Email,
            "phone" => Phone,
            "phone_local" => PhoneLocal,
            "neg_place" => NegPlace,
            "neg_district" => NegDistrict,
            "neg_word_name" => NegWordName,
            "neg_handle" => NegHandle,
            "neg_url" => NegUrl,
            "neg_date" => NegDate,
            "neg_price" => NegPrice,
            "neg_order" => NegOrder,
            "neg_ip" => NegIp,
            "neg_digits" => NegDigits,
            "neg_road_sentence" => NegRoadSentence,
            "neg_partial_location" => NegPartialLocation,
            "neg_ge_incomplete_street" => NegGeIncompleteStreet,
            "neg_header" => NegHeader,
            "neg_department" => NegDepartment,
            "neg_prompt" => NegPrompt,
            "neg_heading" => NegHeading,
            "neg_caps" => NegCaps,
            "neg_law" => NegLaw,
            "neg_label" => NegLabel,
            "neg_service" => NegService,
            "neg_hours" => NegHours,
            "neg_citation" => NegCitation,
            "neg_line" => NegLine,
            "neg_code" => NegCode,
            "neg_list" => NegList,
            "neg_unit_code" => NegUnitCode,
            "neg_register_no" => NegRegisterNo,
            "officers" => Officers,
            "register" => Register,
            "title_local" => TitleLocal,
            "sentence_local" => SentenceLocal,
            "place_local" => PlaceLocal,
            "role_heading" => RoleHeading,
            "greeting" => Greeting,
            "closing" => Closing,
            "sentence" => Sentence,
            "title" => Title,
            "date" => Date,
            "product" => Product,
            _ => return None,
        })
    }

    /// Slots whose value is reused when the same link group appears again in a document. An
    /// acronym is bound with its body's name instead (see `Ctx::acronym`).
    fn bindable(self) -> bool {
        !matches!(
            self,
            Slot::OrgAcronym | Slot::PersonLast | Slot::PersonList | Slot::OrgList
        ) && matches!(
            self.kind(),
            Some(Kind::Person | Kind::Org | Kind::Address | Kind::Email)
        )
    }
}

/// One labelled span of a generated document, in byte offsets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Gold {
    pub kind: &'static str,
    pub start: usize,
    pub end: usize,
}

/// A labelled slot's link group, so Milestone 5 can evaluate grouping on this corpus.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Link {
    pub kind: &'static str,
    pub index: usize,
    pub group: u8,
}

/// A rendered document before the post-render checks.
#[derive(Debug, Clone)]
pub struct Doc {
    pub text: String,
    pub entities: Vec<Gold>,
    pub links: Vec<Link>,
    pub family: Family,
    pub template_id: u32,
    pub category: Category,
    pub country: &'static str,
    pub phones_fixture_safe: bool,
}

/// What `Ctx::fill` produced for one slot: the value, an optional honorific written before
/// the span and post-nominal or name suffix after it, and whether any phone in it comes from a
/// range reserved for fiction.
struct Filled {
    value: String,
    prefix: Option<&'static str>,
    suffix: Option<&'static str>,
    fixture_safe: bool,
}

impl Filled {
    fn plain(value: impl Into<String>) -> Filled {
        Filled {
            value: value.into(),
            prefix: None,
            suffix: None,
            fixture_safe: true,
        }
    }
}

/// A person from the names sample.
#[derive(Debug, Clone)]
pub struct PoolPerson {
    pub name: String,
    pub script: String,
}

impl PoolPerson {
    fn latin(&self) -> bool {
        self.script == "latin"
    }

    /// Japanese-script names are one unit; others split into words.
    fn cjk(&self) -> bool {
        matches!(self.script.as_str(), "han" | "hiragana" | "katakana")
    }
}

/// An organization and the abbreviation of its legal form, if known.
#[derive(Debug, Clone)]
pub struct PoolOrg {
    pub name: String,
    pub legal_form: String,
}

/// An unaugmented address from the parser shards, with one-line and multi-line renderings.
#[derive(Debug, Clone)]
pub struct PoolAddress {
    pub text: String,
}

impl PoolAddress {
    fn from_example(e: &LabelledExample) -> PoolAddress {
        PoolAddress {
            text: pool_filter::with_local_country(e, &pool_filter::without_venue_lines(e)),
        }
    }

    fn multiline(&self) -> bool {
        self.text.contains('\n')
    }

    fn one_line(&self) -> String {
        self.text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty())
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// The pools of one (split, country).
#[derive(Debug, Default)]
pub struct Pools {
    pub people: Vec<PoolPerson>,
    pub orgs: Vec<PoolOrg>,
    pub addresses: Vec<PoolAddress>,
    multiline_addresses: Vec<usize>,
    complete_multiline_addresses: Vec<usize>,
    pub bodies: Bodies,
}

#[derive(Default)]
struct GoldExclusions {
    texts: HashSet<String>,
    people: HashSet<String>,
    person_keys: HashSet<(String, String)>,
    orgs: HashSet<String>,
    addresses: HashSet<String>,
    emails: HashSet<String>,
    phones: HashSet<String>,
    sha256: String,
}

#[derive(Serialize)]
struct SilverSource {
    path: String,
    sha256: String,
}

#[derive(Default)]
struct SilverExclusions {
    surfaces: GoldExclusions,
    sources: Vec<SilverSource>,
}

#[derive(Deserialize)]
struct SilverExclusionCase {
    country: String,
    text: String,
    entities: Vec<SilverExclusionSpan>,
}

#[derive(Deserialize)]
struct SilverExclusionSpan {
    kind: String,
    start: usize,
    end: usize,
}

#[derive(Deserialize)]
struct GoldExclusionCase {
    country: String,
    input: String,
    expected: Vec<GoldExclusionSpan>,
}

#[derive(Deserialize)]
struct GoldExclusionSpan {
    kind: String,
    text: String,
}

impl GoldExclusions {
    fn load(path: &Path) -> anyhow::Result<Self> {
        let bytes = std::fs::read(path)
            .with_context(|| format!("reading evaluation exclusions {}", path.display()))?;
        let mut exclusions = Self {
            sha256: format!("{:x}", Sha256::digest(&bytes)),
            ..Self::default()
        };
        for (index, line) in String::from_utf8(bytes)?.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let case: GoldExclusionCase = serde_json::from_str(line)
                .with_context(|| format!("{}:{}", path.display(), index + 1))?;
            anyhow::ensure!(
                case.country == "US",
                "{}:{} has non-US evaluation data",
                path.display(),
                index + 1
            );
            exclusions.texts.insert(case.input);
            for span in case.expected {
                exclusions.add(&span.kind, &span.text)?;
            }
        }
        anyhow::ensure!(
            !exclusions.people.is_empty() && !exclusions.addresses.is_empty(),
            "{} has no US person or address exclusions",
            path.display()
        );
        Ok(exclusions)
    }

    fn add(&mut self, kind: &str, text: &str) -> anyhow::Result<()> {
        let surface = normalize_surface(text);
        anyhow::ensure!(!surface.is_empty(), "empty exclusion surface");
        match kind {
            "person" => {
                self.people.insert(surface);
                if let Some(key) = person_key(text) {
                    self.person_keys.insert(key);
                }
            }
            "org" => {
                self.orgs.insert(surface);
            }
            "address" => {
                self.addresses.insert(surface);
            }
            "email" => {
                self.emails.insert(surface);
            }
            "phone" => {
                self.phones.insert(surface);
            }
            _ => bail!("unknown exclusion kind {kind:?}"),
        }
        Ok(())
    }

    fn expand_public_body_aliases(&mut self) -> anyhow::Result<()> {
        for body in bodies::public_bodies()? {
            let name = normalize_surface(&body.name);
            let acronym = body
                .acronym
                .as_deref()
                .context("validated public body lacks acronym")?;
            let acronym = normalize_surface(acronym);
            if self.orgs.contains(&name) || self.orgs.contains(&acronym) {
                self.orgs.insert(name);
                self.orgs.insert(acronym);
            }
        }
        Ok(())
    }

    fn check(&self, doc: &Doc) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.texts.contains(&doc.text),
            "generated document duplicates held-out evaluation text"
        );
        let bytes = doc.text.as_bytes();
        for span in &doc.entities {
            let excluded = match span.kind {
                "person" => &self.people,
                "org" => &self.orgs,
                "address" => &self.addresses,
                "email" => &self.emails,
                "phone" => &self.phones,
                _ => continue,
            };
            let surface =
                std::str::from_utf8(bytes.get(span.start..span.end).with_context(|| {
                    format!("generated {} span has invalid offsets", span.kind)
                })?)?;
            anyhow::ensure!(
                !excluded.contains(&normalize_surface(surface)),
                "generated {} span overlaps excluded data: {surface}",
                span.kind
            );
            if span.kind == "person" {
                anyhow::ensure!(
                    person_key(surface).is_none_or(|key| !self.person_keys.contains(&key)),
                    "generated person span shares an excluded first and last name: {surface}"
                );
            }
        }
        Ok(())
    }
}

impl SilverExclusions {
    fn load(paths: &[String]) -> anyhow::Result<Self> {
        let mut out = Self::default();
        for path in paths {
            let bytes = std::fs::read(path).with_context(|| format!("reading silver {path}"))?;
            let sha256 = format!("{:x}", Sha256::digest(&bytes));
            let content = std::str::from_utf8(&bytes)
                .with_context(|| format!("silver {path} is not UTF-8"))?;
            for (index, line) in content.lines().enumerate() {
                if line.trim().is_empty() {
                    continue;
                }
                let case: SilverExclusionCase = serde_json::from_str(line)
                    .with_context(|| format!("{path}:{} invalid silver JSON", index + 1))?;
                anyhow::ensure!(
                    case.country == "US",
                    "{path}:{} has country {:?}; detector silver must be US-only",
                    index + 1,
                    case.country
                );
                out.surfaces.texts.insert(case.text.clone());
                for span in case.entities {
                    let surface = case
                        .text
                        .get(span.start..span.end)
                        .filter(|s| !s.is_empty())
                        .with_context(|| {
                            format!(
                                "{path}:{} has invalid UTF-8 entity offsets {}..{}",
                                index + 1,
                                span.start,
                                span.end
                            )
                        })?;
                    out.surfaces
                        .add(&span.kind, surface)
                        .with_context(|| format!("{path}:{}", index + 1))?;
                }
            }
            out.sources.push(SilverSource {
                path: path.clone(),
                sha256,
            });
        }
        Ok(out)
    }
}

fn normalize_surface(text: &str) -> String {
    text.nfc()
        .collect::<String>()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn person_key(text: &str) -> Option<(String, String)> {
    let folded: String = text
        .nfkd()
        .filter(|char| !is_combining_mark(*char))
        .collect::<String>()
        .to_lowercase();
    fn words(value: &str) -> Vec<&str> {
        value
            .split(|char: char| !char.is_ascii_lowercase())
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>()
    }
    if let Some((surname, given)) = folded.split_once(',') {
        return Some((
            words(given).first()?.to_string(),
            words(surname).last()?.to_string(),
        ));
    }
    let mut parts = words(&folded);
    while parts
        .last()
        .is_some_and(|part| matches!(*part, "jr" | "sr" | "ii" | "iii" | "iv"))
    {
        parts.pop();
    }
    Some((
        parts.first()?.to_string(),
        parts.get(1..)?.last()?.to_string(),
    ))
}

fn prefixed_address(address: String, rng: &mut ChaCha8Rng) -> String {
    let room = rng.random_range(100..1000);
    let code = rng.random_range(1000..10000);
    let floor = rng.random_range(1..20);
    let prefix = match rng.random_range(0..10) {
        0 => format!("Room {room}, "),
        1 => format!("Room {room}\n"),
        2 => format!("Mail Code {code}, "),
        3 => format!("Mail Stop {code}\n"),
        4 => format!("North Office Building, Room {room}, "),
        5 => format!("Administrative Building\nRoom {room}\n"),
        6 => format!("Suite {room}, "),
        7 => format!("MS: {code}, "),
        8 => format!("Floor {floor}, Room {room}, "),
        _ => format!("West Building, Room W12-{room}, "),
    };
    format!("{prefix}{address}")
}

/// Per-document fill state: the pools of the document's split and country, and the values
/// bound to link groups so far.
pub struct Ctx<'a> {
    pub country: &'static str,
    pools: &'a Pools,
    bound: HashMap<(Kind, u8), String>,
    bound_person: HashMap<u8, PoolPerson>,
    bound_acronym: HashMap<u8, String>,
    /// Link groups the template also writes as an acronym, whose body must have one.
    acronym_groups: HashSet<u8>,
    /// Turns off every optional variation (honorifics, casing, email patterns and digits),
    /// for tests that need exact text.
    plain: bool,
}

impl<'a> Ctx<'a> {
    pub fn new(country: &'static str, pools: &'a Pools) -> Ctx<'a> {
        Ctx {
            country,
            pools,
            bound: HashMap::new(),
            bound_person: HashMap::new(),
            bound_acronym: HashMap::new(),
            acronym_groups: HashSet::new(),
            plain: false,
        }
    }

    fn fill(&mut self, slot: Slot, group: u8, rng: &mut ChaCha8Rng) -> anyhow::Result<Filled> {
        if group > 0
            && slot.bindable()
            && let Some(kind) = slot.kind()
            && let Some(v) = self.bound.get(&(kind, group))
            && !matches!(slot, Slot::PersonFirst | Slot::PersonLast)
        {
            return Ok(Filled::plain(v.clone()));
        }
        let filled = match slot {
            Slot::Person => {
                let p = self.person(rng)?;
                let (prefix, suffix) = if self.plain {
                    (None, None)
                } else {
                    self.honorifics(&p, rng)
                };
                if group > 0 {
                    self.bound_person.insert(group, p.clone());
                }
                let value = if self.plain || !p.latin() {
                    p.name
                } else if self.country == "US" && rng.random_bool(0.06) {
                    self.hyphenated(&p.name, rng)
                } else {
                    capitalized(&p.name, rng)
                };
                Filled {
                    value,
                    prefix,
                    suffix,
                    fixture_safe: true,
                }
            }
            Slot::PersonFirst => {
                let p = match self.bound_person.get(&group) {
                    Some(p) => p.clone(),
                    None => {
                        let p = self.person(rng)?;
                        if group > 0 {
                            self.bound_person.insert(group, p.clone());
                            self.bound.insert((Kind::Person, group), p.name.clone());
                        }
                        p
                    }
                };
                Filled::plain(first_name(&p))
            }
            Slot::PersonLast => {
                let p = match self.bound_person.get(&group) {
                    Some(p) => p.clone(),
                    None => {
                        let p = self.person(rng)?;
                        if group > 0 {
                            self.bound_person.insert(group, p.clone());
                            self.bound.insert((Kind::Person, group), p.name.clone());
                        }
                        p
                    }
                };
                Filled::plain(family_name(&p))
            }
            Slot::PersonDirectory => {
                let p = self.person(rng)?;
                let value = match p.name.split_once(' ') {
                    Some((given, surname)) if p.latin() => {
                        format!("{given} {}", surname.to_uppercase())
                    }
                    _ => p.name.clone(),
                };
                Filled {
                    value,
                    prefix: Some(*pick(&["Mr", "Ms", "Mrs", "Dr"], rng)),
                    suffix: Some(*pick(GLUED_ROLES, rng)),
                    fixture_safe: true,
                }
            }
            Slot::PersonNative => {
                let p = self.native_person(0.7, rng)?;
                if group > 0 {
                    self.bound_person.insert(group, p.clone());
                }
                Filled::plain(p.name)
            }
            Slot::PersonList | Slot::OrgList => {
                bail!("list slots are expanded before rendering")
            }
            Slot::PersonEponymous => {
                let given = self
                    .pools
                    .people
                    .iter()
                    .filter(|p| p.latin())
                    .collect::<Vec<_>>()
                    .choose(rng)
                    .map(|p| first_name(p))
                    .unwrap_or_else(|| "Anna".to_string());
                let surname = pick(templates::EPONYMOUS_SURNAMES, rng);
                Filled::plain(format!("{given} {surname}"))
            }
            Slot::Org => {
                if self.country == "US" && rng.random_bool(0.2) {
                    let body = self.pools.bodies.body(true, rng)?;
                    Filled::plain(body.acronym.context("public body lacks an acronym")?)
                } else {
                    Filled::plain(self.org(rng)?)
                }
            }
            Slot::OrgEponymous => Filled::plain(*pick(templates::EPONYMOUS_COMPANIES, rng)),
            Slot::OrgSchool => Filled::plain(*pick(templates::PERSON_NAMED_INSTITUTIONS, rng)),
            Slot::OrgDistrict => Filled::plain(district_name(rng)),
            Slot::OrgGov => {
                let body = self
                    .pools
                    .bodies
                    .body(self.acronym_groups.contains(&group), rng)?;
                if let Some(acronym) = body.acronym.filter(|_| group > 0) {
                    self.bound_acronym.insert(group, acronym);
                }
                Filled::plain(body.name)
            }
            Slot::OrgAcronym => Filled::plain(self.acronym(group, rng)?),
            Slot::OrgUnit => Filled::plain(self.pools.bodies.unit(rng)?),
            Slot::OrgLocalOffice => Filled::plain(
                self.pools
                    .bodies
                    .local_office(rng)
                    .context("no split-safe Georgian local office heading")?,
            ),
            Slot::OrgRegistry => Filled::plain(self.pools.bodies.registry(rng)?),
            Slot::OrgChain => Filled::plain(self.pools.bodies.chain(rng)?),
            Slot::OrgUniv => Filled::plain(self.pools.bodies.university(rng)),
            Slot::OrgCouncil => Filled::plain(self.pools.bodies.council(rng)),
            Slot::OrgMedia => Filled::plain(self.pools.bodies.media(rng)),
            Slot::OrgParty => Filled::plain(self.pools.bodies.party(rng)),
            Slot::Address => {
                let a = self.address(false, rng)?.one_line();
                let address = self.shaped(a, ", ", rng);
                Filled::plain(if self.country == "US" && rng.random_bool(0.15) {
                    prefixed_address(address, rng)
                } else {
                    address
                })
            }
            Slot::AddressMultiline => {
                let a = self.address(true, rng)?.text.clone();
                let address = self.shaped(a, "\n", rng);
                Filled::plain(if self.country == "US" && rng.random_bool(0.15) {
                    prefixed_address(address, rng)
                } else {
                    address
                })
            }
            Slot::AddressPrefixed => {
                let source = self.address(true, rng)?;
                let address = if rng.random_bool(0.5) {
                    source.text.clone()
                } else {
                    source.one_line()
                };
                Filled::plain(prefixed_address(address, rng))
            }
            Slot::Email => Filled::plain(self.email(group, rng)),
            Slot::Phone => {
                let (value, safe) = phone(self.country, rng.random_bool(0.5), rng)?;
                Filled {
                    value,
                    prefix: None,
                    suffix: None,
                    fixture_safe: safe,
                }
            }
            Slot::PhoneLocal => {
                let (value, safe) = phone(self.country, false, rng)?;
                Filled {
                    value,
                    prefix: None,
                    suffix: None,
                    fixture_safe: safe,
                }
            }
            Slot::NegPlace => Filled::plain(*pick(templates::NEG_PLACES, rng)),
            Slot::NegDistrict => Filled::plain(format!(
                "{} County",
                pick(templates::DISTRICT_PLACE_NAMES, rng)
            )),
            Slot::NegWordName => Filled::plain(*pick(templates::NEG_WORD_NAMES, rng)),
            Slot::NegHandle => Filled::plain(*pick(templates::NEG_HANDLES, rng)),
            Slot::NegUrl => Filled::plain(*pick(templates::NEG_URLS, rng)),
            Slot::NegDate if self.country == "US" || rng.random_bool(0.1) => {
                Filled::plain(*pick(templates::NEG_DATES, rng))
            }
            Slot::NegDate => Filled::plain(negatives::date(self.country, rng)),
            Slot::NegPrice if self.country == "US" && rng.random_bool(0.2) => {
                Filled::plain(*pick(templates::NEG_PRICES, rng))
            }
            Slot::NegPrice => Filled::plain(negatives::money(self.country, false, rng)),
            Slot::NegOrder if self.country == "US" && rng.random_bool(0.2) => {
                Filled::plain(*pick(templates::NEG_ORDERS, rng))
            }
            Slot::NegOrder => Filled::plain(negatives::order(self.country, rng)),
            Slot::NegIp => Filled::plain(format!(
                "{}.{}.{}.{}",
                rng.random_range(10..=223),
                rng.random_range(0..=255),
                rng.random_range(0..=255),
                rng.random_range(1..=254)
            )),
            Slot::NegDigits => Filled::plain(*pick(templates::NEG_DIGITS, rng)),
            Slot::NegRoadSentence => Filled::plain(*pick(templates::NEG_ROAD_SENTENCES, rng)),
            Slot::NegPartialLocation => Filled::plain(*pick(templates::NEG_PARTIAL_LOCATIONS, rng)),
            Slot::NegGeIncompleteStreet => Filled::plain(
                self.pools
                    .bodies
                    .incomplete_street(rng)
                    .context("no split-safe Georgian incomplete street")?,
            ),
            Slot::NegHeader => {
                Filled::plain(localized(templates::NEG_HEADERS, self.country, 0.5, rng))
            }
            Slot::NegDepartment => Filled::plain(localized(
                templates::NEG_DEPARTMENTS,
                self.country,
                0.5,
                rng,
            )),
            Slot::NegPrompt => {
                Filled::plain(localized(templates::NEG_PROMPTS, self.country, 0.5, rng))
            }
            Slot::NegHeading => {
                Filled::plain(localized(templates::NEG_HEADINGS, self.country, 0.5, rng))
            }
            Slot::NegCaps => {
                Filled::plain(localized(negatives::CAPS_HEADINGS, self.country, 0.4, rng))
            }
            Slot::NegLaw => Filled::plain(localized(negatives::LAWS, self.country, 0.4, rng)),
            Slot::NegLabel => Filled::plain(localized(negatives::LABELS, self.country, 0.4, rng)),
            Slot::NegService => {
                Filled::plain(localized(negatives::SERVICES, self.country, 0.6, rng))
            }
            Slot::NegHours => Filled::plain(localized(negatives::HOURS, self.country, 0.5, rng)),
            Slot::NegCitation => Filled::plain(negatives::citation(self.country, rng)),
            Slot::NegLine => Filled::plain(negatives::line(self.country, rng)),
            Slot::NegCode => Filled::plain(negatives::code_block(rng)),
            Slot::NegList => Filled::plain(*pick(negatives::LISTS, rng)),
            Slot::NegUnitCode => Filled::plain(negatives::unit_code(rng)),
            Slot::NegRegisterNo => Filled::plain(negatives::register_no(self.country, rng)),
            Slot::Officers => Filled::plain(localized(templates::OFFICERS, self.country, 0.7, rng)),
            Slot::Register => {
                Filled::plain(localized(templates::REGISTERS, self.country, 0.7, rng))
            }
            Slot::TitleLocal => Filled::plain(localized(templates::TITLES, self.country, 1.0, rng)),
            Slot::SentenceLocal => Filled::plain(filler::sentence_local(self.country, rng)),
            Slot::PlaceLocal => Filled::plain(localized(templates::PLACES, self.country, 1.0, rng)),
            Slot::RoleHeading => {
                Filled::plain(localized(templates::ROLE_HEADINGS, self.country, 1.0, rng))
            }
            Slot::Greeting => {
                Filled::plain(localized(templates::GREETINGS, self.country, 0.5, rng))
            }
            Slot::Closing => Filled::plain(localized(templates::CLOSINGS, self.country, 0.5, rng)),
            Slot::Title => Filled::plain(localized(templates::TITLES, self.country, 0.5, rng)),
            Slot::Sentence => Filled::plain(filler::sentence(self.country, rng)),
            Slot::Date => Filled::plain(*pick(templates::DATES, rng)),
            Slot::Product => Filled::plain(*pick(templates::PRODUCTS, rng)),
        };
        if group > 0
            && slot.bindable()
            && let Some(kind) = slot.kind()
        {
            self.bound
                .entry((kind, group))
                .or_insert_with(|| filled.value.clone());
        }
        Ok(filled)
    }

    /// A person whose name is written in the country's own script, when the pool has any.
    /// A spaced Japanese name shows where the family name ends; `spaced` is the share of
    /// Japanese draws that must have one.
    fn native_person(&self, spaced: f64, rng: &mut ChaCha8Rng) -> anyhow::Result<PoolPerson> {
        let native = |p: &&PoolPerson| match self.country {
            "GE" => p.script == "georgian",
            "JP" => p.cjk(),
            _ => p.latin(),
        };
        let candidates: Vec<&PoolPerson> = self.pools.people.iter().filter(native).collect();
        let with_space: Vec<&PoolPerson> = candidates
            .iter()
            .copied()
            .filter(|p| p.cjk() && p.name.contains([' ', '\u{3000}']))
            .collect();
        let candidates = if with_space.is_empty() || !rng.random_bool(spaced) {
            candidates
        } else {
            with_space
        };
        match candidates.choose(rng) {
            Some(p) => Ok((*p).clone()),
            None => self.person(rng),
        }
    }

    /// `name` with a second given name or surname joined by a hyphen, drawn from another
    /// Latin name of the pool: `Jens-Uwe Walther`, `Florian Zick-Mayer`.
    fn hyphenated(&self, name: &str, rng: &mut ChaCha8Rng) -> String {
        if name.contains('-') {
            return name.to_string();
        }
        // A few draws find an unhyphenated Latin donor without scanning the whole pool.
        let other = (0..8)
            .filter_map(|_| self.pools.people.choose(rng))
            .find(|p| p.latin() && p.name != name && !p.name.contains('-'))
            .and_then(|p| p.name.split_once(' '));
        match (name.split_once(' '), other) {
            (Some((given, rest)), Some((other_given, _))) if rng.random_bool(0.5) => {
                format!("{given}-{other_given} {rest}")
            }
            (Some(_), Some((_, other_surname))) => {
                let last = other_surname
                    .split(' ')
                    .next_back()
                    .unwrap_or(other_surname);
                format!("{name}-{last}")
            }
            _ => name.to_string(),
        }
    }

    fn person(&self, rng: &mut ChaCha8Rng) -> anyhow::Result<PoolPerson> {
        self.pools
            .people
            .choose(rng)
            .cloned()
            .with_context(|| format!("no people for {}", self.country))
    }

    /// An English honorific before a Latin-script name and a US post-nominal after it.
    fn honorifics(
        &self,
        p: &PoolPerson,
        rng: &mut ChaCha8Rng,
    ) -> (Option<&'static str>, Option<&'static str>) {
        if !p.latin() {
            return (None, None);
        }
        let prefix = rng
            .random_bool(0.3)
            .then(|| localized(templates::HONORIFICS, self.country, 0.6, rng));
        let suffix = rng
            .random_bool(0.06)
            .then(|| *pick(templates::POSTNOMINALS, rng));
        (prefix, suffix)
    }

    /// The acronym bound to `group`, or a fresh body's, bound with its name so a later
    /// `{org_gov#n}` names the same body.
    fn acronym(&mut self, group: u8, rng: &mut ChaCha8Rng) -> anyhow::Result<String> {
        if let Some(a) = self.bound_acronym.get(&group) {
            return Ok(a.clone());
        }
        let body = self.pools.bodies.body(true, rng)?;
        let acronym = body.acronym.unwrap_or_else(|| body.name.clone());
        if group > 0 {
            self.bound_acronym.insert(group, acronym.clone());
            self.bound.entry((Kind::Org, group)).or_insert(body.name);
        }
        Ok(acronym)
    }

    /// Keep each parser address tied to its source location.
    fn shaped(&self, address: String, _sep: &str, _rng: &mut ChaCha8Rng) -> String {
        address
    }

    /// A random org, 30% of all-caps names title-cased and 20% with a trailing legal form
    /// dropped, as documents write them both ways.
    fn org(&self, rng: &mut ChaCha8Rng) -> anyhow::Result<String> {
        let org = self
            .pools
            .orgs
            .choose(rng)
            .with_context(|| format!("no organizations for {}", self.country))?;
        let mut name = org.name.clone();
        if self.plain {
            return Ok(name);
        }
        // English documents name charities far more often than registered companies.
        if self.country == "US" && rng.random_bool(0.15) {
            return Ok(self.pools.bodies.charity(rng));
        }
        // Registers store names in capitals; documents mostly do not, but some still print them.
        if rng.random_bool(0.65) && name.chars().any(char::is_alphabetic) && !has_lowercase(&name) {
            name = title_case(&name);
        }
        if rng.random_bool(0.2)
            && let Some(stripped) = strip_legal_form(&name, &org.legal_form)
        {
            name = stripped;
        }
        Ok(name)
    }

    fn address(&self, multiline: bool, rng: &mut ChaCha8Rng) -> anyhow::Result<&'a PoolAddress> {
        let pool = &self.pools.addresses;
        if !multiline {
            return pool
                .choose(rng)
                .with_context(|| format!("no addresses for {}", self.country));
        }
        let indices = if !self.pools.complete_multiline_addresses.is_empty() && rng.random_bool(0.8)
        {
            &self.pools.complete_multiline_addresses
        } else {
            &self.pools.multiline_addresses
        };
        let index = indices
            .choose(rng)
            .with_context(|| format!("no multiline addresses for {}", self.country))?;
        Ok(&pool[*index])
    }

    /// A local part from the linked person when their name is Latin, else a generic one;
    /// the domain from the linked org, else a random word; always a reserved domain.
    fn email(&self, group: u8, rng: &mut ChaCha8Rng) -> String {
        let person = (group > 0)
            .then(|| self.bound_person.get(&group))
            .flatten()
            .filter(|p| p.latin());
        let local = match person.map(|p| ascii_words(&p.name)) {
            Some(words) if !words.is_empty() => {
                let first = &words[0];
                let last = &words[words.len() - 1];
                let pattern = if self.plain {
                    0
                } else {
                    rng.random_range(0..5)
                };
                let mut local = match pattern {
                    0 => format!("{first}.{last}"),
                    1 => format!("{}{last}", &first[..1]),
                    2 => format!("{first}_{last}"),
                    3 => first.clone(),
                    _ => format!("{last}.{first}"),
                };
                if !self.plain && rng.random_bool(0.1) {
                    local.push_str(&format!("{:02}", rng.random_range(0..100)));
                }
                local
            }
            _ => pick(templates::LOCAL_PARTS, rng).to_string(),
        };
        let org_slug = (group > 0)
            .then(|| {
                self.bound_acronym
                    .get(&group)
                    .or_else(|| self.bound.get(&(Kind::Org, group)))
            })
            .flatten()
            .map(|o| slug(o))
            .filter(|s| !s.is_empty());
        let domain = org_slug.unwrap_or_else(|| pick(templates::DOMAIN_WORDS, rng).to_string());
        let suffix = if self.plain {
            templates::EMAIL_SUFFIXES[0]
        } else {
            pick(templates::EMAIL_SUFFIXES, rng)
        };
        format!("{local}@{domain}{suffix}")
    }
}

fn pick<'t, T>(items: &'t [T], rng: &mut ChaCha8Rng) -> &'t T {
    &items[rng.random_range(0..items.len())]
}

fn district_name(rng: &mut ChaCha8Rng) -> String {
    format!(
        "{} {}",
        pick(templates::DISTRICT_PLACE_NAMES, rng),
        pick(templates::DISTRICT_FORMS, rng)
    )
}

/// The country's own list with probability `own` when it has one, otherwise the `"en"` list.
pub(crate) fn localized(
    lists: &[(&str, &[&'static str])],
    country: &str,
    own: f64,
    rng: &mut ChaCha8Rng,
) -> &'static str {
    let own_list = lists.iter().find(|(k, _)| *k == country).map(|(_, l)| *l);
    let english = lists
        .iter()
        .find(|(k, _)| *k == "en")
        .map_or(&[][..], |(_, l)| *l);
    match own_list {
        Some(list) if rng.random_bool(own) => pick(list, rng),
        _ => pick(english, rng),
    }
}

/// `name` as directories and forms print it now and then: surnames in capitals
/// (`Maravillas ABADÍA JOVER`, one in ten) or the whole name in capitals (one in fifty).
fn capitalized(name: &str, rng: &mut ChaCha8Rng) -> String {
    let roll = rng.random_range(0..100);
    if roll < 2 {
        return name.to_uppercase();
    }
    match name.split_once(' ') {
        Some((given, surnames)) if roll < 12 => format!("{given} {}", surnames.to_uppercase()),
        _ => name.to_string(),
    }
}

/// Roles directories glue to a name with a hyphen, as the EU Whoiswho writes them.
const GLUED_ROLES: &[&str] = &[
    "-Member",
    "-Delegate",
    "-Substitute",
    "-President",
    "-Vice-President",
    "-Governor",
    "-Chair",
    "-Head of Unit",
    "-Director",
    "-Adviser",
];

/// The family name alone: the last word of a Latin or Georgian name, the first word of a
/// spaced Japanese one, and the part after `・` of a katakana one. An unspaced kanji name,
/// whose family name may be two or three characters long, stays whole.
fn family_name(p: &PoolPerson) -> String {
    if p.cjk() {
        return match p.name.split_once([' ', '\u{3000}']) {
            Some((family, _)) => family.to_string(),
            None => match p.name.rsplit_once('・') {
                Some((_, family)) => family.to_string(),
                None => p.name.clone(),
            },
        };
    }
    p.name
        .split_whitespace()
        .rev()
        .find(|w| !is_suffix_or_initial(w))
        .unwrap_or(&p.name)
        .trim_end_matches(',')
        .to_string()
}

/// A generational suffix or an initial, which never stands alone for the family name.
fn is_suffix_or_initial(word: &str) -> bool {
    let bare = word.trim_end_matches([',', '.']);
    matches!(bare, "Jr" | "Sr" | "II" | "III" | "IV") || bare.chars().count() == 1
}

/// The first word of a name, or the whole name in Japanese script.
fn first_name(p: &PoolPerson) -> String {
    if p.cjk() {
        return p.name.clone();
    }
    p.name
        .split_whitespace()
        .next()
        .unwrap_or(&p.name)
        .to_string()
}

/// Lowercase ASCII words of a Latin name, accents removed.
fn ascii_words(name: &str) -> Vec<String> {
    let ascii: String = name
        .nfd()
        .filter(char::is_ascii)
        .collect::<String>()
        .to_lowercase();
    ascii
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_string)
        .collect()
}

/// `Kavkaz Freight LLC` becomes `kavkaz-freight-llc`: lowercase ASCII words joined by hyphens,
/// at most three of them.
fn slug(org: &str) -> String {
    ascii_words(org)
        .into_iter()
        .take(3)
        .collect::<Vec<_>>()
        .join("-")
}

fn has_lowercase(s: &str) -> bool {
    s.chars().any(char::is_lowercase)
}

fn title_case(s: &str) -> String {
    s.split(' ')
        .map(|w| {
            let mut chars = w.chars();
            match chars.next() {
                Some(first) => first
                    .to_uppercase()
                    .chain(chars.flat_map(char::to_lowercase))
                    .collect(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// The name without its trailing legal form, when the last word is that form.
fn strip_legal_form(name: &str, legal_form: &str) -> Option<String> {
    if legal_form.is_empty() {
        return None;
    }
    let (rest, last) = name.rsplit_once(' ')?;
    let norm = |s: &str| s.trim_end_matches('.').to_lowercase();
    (norm(last) == norm(legal_form) && !rest.trim().is_empty())
        .then(|| rest.trim_end_matches([',', ' ']).to_string())
}

fn imported_org_collides_with_listed(
    name: &str,
    legal_form: &str,
    reserved: &HashSet<&str>,
) -> bool {
    if reserved.contains(name) {
        return true;
    }
    let titled =
        (!has_lowercase(name) && name.chars().any(char::is_alphabetic)).then(|| title_case(name));
    if titled
        .as_ref()
        .is_some_and(|value| reserved.contains(value.as_str()))
    {
        return true;
    }
    strip_legal_form(name, legal_form).is_some_and(|value| reserved.contains(value.as_str()))
        || titled
            .and_then(|value| strip_legal_form(&value, legal_form))
            .is_some_and(|value| reserved.contains(value.as_str()))
}

/// A US phone number in the reserved 555-01xx fictional range.
fn phone(
    country: &str,
    international: bool,
    rng: &mut ChaCha8Rng,
) -> anyhow::Result<(String, bool)> {
    anyhow::ensure!(country == "US", "no phone generator for {country}");
    const AREA: &[&str] = &["212", "415", "312", "617", "206", "305", "718"];
    let area = *pick(AREA, rng);
    let line = format!("01{:02}", rng.random_range(0..100u8));
    let value = if international {
        if rng.random_bool(0.5) {
            format!("+1 {area} 555 {line}")
        } else {
            format!("1-{area}-555-{line}")
        }
    } else {
        match rng.random_range(0..3) {
            0 => format!("({area}) 555-{line}"),
            1 => format!("{area}-555-{line}"),
            _ => format!("{area}.555.{line}"),
        }
    };
    Ok((value, true))
}

/// `the` before a public body, unit, council, or university named in running English text
/// (`told the Planning Inspectorate`, `said. The Environment Agency`), outside its span, as real
/// documents write about one in five of their organizations. Only after a lower-case word or
/// the end of a sentence, so never in headlines, lists, or tag rows, and never before a name
/// written without one (`County Court`, `City Council`).
fn article(
    text: &str,
    value: &str,
    slot: Slot,
    country: &str,
    rng: &mut ChaCha8Rng,
) -> Option<&'static str> {
    let body = matches!(
        slot,
        Slot::OrgGov | Slot::OrgUnit | Slot::OrgCouncil | Slot::OrgUniv
    );
    if !body || country != "US" || !rng.random_bool(0.4) {
        return None;
    }
    let bare = ["Council", "Court", "House"]
        .iter()
        .any(|suffix| value.ends_with(suffix))
        || value.contains("'s ")
        || value.contains("\u{2019}s ");
    let line = text.rsplit('\n').next().unwrap_or("");
    let last = line.split_whitespace().next_back().unwrap_or("");
    let taken = [
        "the", "a", "an", "this", "that", "its", "their", "our", "your",
    ];
    let sentence_end = last.ends_with(['.', '!', '?']);
    let prose = last.chars().all(|c| c.is_lowercase()) && !taken.contains(&last);
    if bare || !line.ends_with(' ') || !(prose || sentence_end) {
        return None;
    }
    Some(if sentence_end { "The" } else { "the" })
}

/// Renders a template, measuring every labelled value's span on the string being built, so
/// the offsets are exact by construction. Honorifics are written before the span starts.
pub fn render(template: &Template, ctx: &mut Ctx, rng: &mut ChaCha8Rng) -> anyhow::Result<Doc> {
    let mut text = String::new();
    let mut entities = Vec::new();
    let mut links = Vec::new();
    let mut phones_fixture_safe = true;
    let expanded = expand_lists(template.text, ctx.country, rng);
    let refs = templates::parse_slots(&expanded)?;
    ctx.acronym_groups = refs
        .iter()
        .filter(|r| r.slot == Slot::OrgAcronym && r.group > 0)
        .map(|r| r.group)
        .collect();
    let mut rest = expanded.as_str();
    for SlotRef { slot, group, form } in refs {
        let open = rest
            .find('{')
            .context("slot list and template text disagree")?;
        let close = open + rest[open..].find('}').context("unclosed slot")?;
        text.push_str(&rest[..open]);
        let mut filled = ctx.fill(slot, group, rng)?;
        // A case ending or possessive goes on the name itself, never after a post-nominal.
        if form != inflect::Form::Plain {
            filled.suffix = None;
        }
        let (value, outside) =
            inflect::apply(&filled.value, form, slot.kind() == Some(Kind::Person), rng);
        filled.value = value;
        if outside.is_some() {
            filled.suffix = outside;
        }
        phones_fixture_safe &= filled.fixture_safe;
        if filled.prefix.is_none() {
            filled.prefix = article(&text, &filled.value, slot, ctx.country, rng);
        }
        if let Some(prefix) = filled.prefix {
            text.push_str(prefix);
            text.push(' ');
        }
        let start = text.len();
        text.push_str(&filled.value);
        let end = text.len();
        if let Some(suffix) = filled.suffix {
            text.push_str(suffix);
        }
        if let Some(kind) = slot.kind() {
            links.push(Link {
                kind: kind.as_str(),
                index: entities.len(),
                group,
            });
            entities.push(Gold {
                kind: kind.as_str(),
                start,
                end,
            });
        }
        rest = &rest[close + 1..];
    }
    text.push_str(rest);
    Ok(Doc {
        text,
        entities,
        links,
        family: template.family,
        template_id: template.id,
        category: template.category,
        country: ctx.country,
        phones_fixture_safe,
    })
}

/// `text` with each list macro written out as several slots of its kind: `{person_list}`
/// becomes two to twelve `{person}` slots (a few runs are longer) joined by commas, line
/// breaks, or `und`/`and`, as mastheads, boards, and staff pages list people; `{org_list}`
/// does the same for organizations.
fn expand_lists(text: &str, country: &str, rng: &mut ChaCha8Rng) -> String {
    let mut out = text.to_string();
    for (list, slot) in [("{person_list}", "{person}"), ("{org_list}", "{org}")] {
        while let Some(at) = out.find(list) {
            let n = if rng.random_bool(0.15) {
                rng.random_range(12..=30)
            } else {
                rng.random_range(2..=12)
            };
            let sep = *pick(&[", ", "\n", "\n\n", "; ", " · "], rng);
            let mut items = vec![slot; n].join(sep);
            if sep == ", " && rng.random_bool(0.4) {
                let last = items.rfind(", ").unwrap_or(0);
                let and = if country == "DE" { " und " } else { " and " };
                items.replace_range(last..last + 2, and);
            }
            out.replace_range(at..at + list.len(), &items);
        }
    }
    out
}

/// `doc` padded with filler text in its country's language (see `filler`), or `doc` as it was
/// when the padded one would not fit in `max_tokens`.
fn wrapped(doc: Doc, ctx: &Ctx, max_tokens: usize, rng: &mut ChaCha8Rng) -> anyhow::Result<Doc> {
    let author = ctx.person(rng)?.name;
    let mut padding = filler::Wrap::draw(ctx.country, &author, rng);
    if matches!(
        doc.category,
        Category::Nothing | Category::NoPersonWithAddress
    ) && padding
        .quote
        .as_ref()
        .is_some_and(|(_, name)| name.is_some())
    {
        padding.quote = None;
    }
    let padded = padding.apply(doc.clone());
    Ok(if check(&padded, max_tokens) == Err(Drop::TooLong) {
        doc
    } else {
        padded
    })
}

/// Why a rendered document was discarded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Drop {
    BoundaryMisaligned,
    Overlap,
    TooLong,
    EmptyEntity,
    Duplicate,
    CategoryMismatch,
    EvaluationOverlap,
    SilverOverlap,
}

impl Drop {
    fn name(self) -> &'static str {
        match self {
            Drop::BoundaryMisaligned => "boundary_misaligned",
            Drop::Overlap => "overlap",
            Drop::TooLong => "too_long",
            Drop::EmptyEntity => "empty_entity",
            Drop::Duplicate => "duplicate",
            Drop::CategoryMismatch => "category_mismatch",
            Drop::EvaluationOverlap => "evaluation_overlap",
            Drop::SilverOverlap => "silver_overlap",
        }
    }
}

/// The post-render checks: every span starts and ends on a token boundary, spans do not
/// overlap or come out empty, and the document fits the model without chunking.
pub fn check(doc: &Doc, max_tokens: usize) -> Result<(), Drop> {
    if !doc.category.matches(&doc.entities) {
        return Err(Drop::CategoryMismatch);
    }
    let tokens = tokenize(&doc.text);
    if tokens.iter().filter(|t| is_content(t)).count() > max_tokens {
        return Err(Drop::TooLong);
    }
    let mut last_end = 0;
    for e in &doc.entities {
        if e.start == e.end {
            return Err(Drop::EmptyEntity);
        }
        if e.start < last_end {
            return Err(Drop::Overlap);
        }
        last_end = e.end;
        let starts = tokens.iter().any(|t| t.start == e.start);
        let ends = tokens.iter().any(|t| t.end == e.end);
        if !starts || !ends {
            return Err(Drop::BoundaryMisaligned);
        }
    }
    Ok(())
}

/// Which slice of the test split a document belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Subset {
    SeenTemplates,
    HeldoutTemplates,
    HeldoutFamilies,
}

impl Subset {
    fn name(self) -> &'static str {
        match self {
            Subset::SeenTemplates => "seen_templates",
            Subset::HeldoutTemplates => "heldout_templates",
            Subset::HeldoutFamilies => "heldout_families",
        }
    }
}

struct Row {
    id: u64,
    split: Split,
    subset: Subset,
    doc: Doc,
}

fn supported_template(template: &Template, country: &str) -> bool {
    if !template.fits(country) {
        return false;
    }
    let Ok(slots) = template.slots() else {
        return false;
    };
    let local_office = slots.iter().any(|item| item.slot == Slot::OrgLocalOffice);
    let registry = slots.iter().any(|item| item.slot == Slot::OrgRegistry);
    if country != "US" || local_office {
        return false;
    }
    let unit = slots.iter().any(|item| item.slot == Slot::OrgUnit);
    let unrelated_parent = slots
        .iter()
        .any(|item| matches!(item.slot, Slot::Org | Slot::OrgGov | Slot::OrgAcronym));
    if unit && unrelated_parent {
        return false;
    }
    let other_than_registry = slots
        .iter()
        .any(|item| item.slot != Slot::OrgRegistry && item.slot.kind() == Some(Kind::Org));
    !(registry && other_than_registry)
}

/// Generates the detector corpus described by the config's `[generate]` section, or with
/// `sample`, prints that many bracketed documents without writing anything.
pub fn run(config_path: &Path, sample: Option<usize>, family: Option<&str>) -> anyhow::Result<()> {
    let cfg = config::load(config_path)?;
    let gen_cfg = cfg
        .generate
        .as_ref()
        .with_context(|| format!("{} has no [generate] section", config_path.display()))?;
    let names = cfg
        .names
        .as_ref()
        .with_context(|| format!("{} has no [names] section", config_path.display()))?;
    let countries = static_countries(&cfg.data.countries)?;
    let mut exclusions = GoldExclusions::load(Path::new(&gen_cfg.exclude_gold))?;
    exclusions.expand_public_body_aliases()?;
    let mut silver = SilverExclusions::load(
        cfg.detector
            .as_ref()
            .map_or(&[], |detector| detector.silver.as_slice()),
    )?;
    silver.surfaces.expand_public_body_aliases()?;
    let pools = load_pools(
        Path::new(&names.out),
        Path::new(&gen_cfg.addresses),
        &countries,
        &exclusions,
        &silver.surfaces,
    )?;
    let all = templates::all()?;
    let mut rng = ChaCha8Rng::seed_from_u64(cfg.seed);

    if let Some(n) = sample {
        let chosen: Vec<&Template> = match family {
            Some(f) => {
                let f = Family::ALL
                    .into_iter()
                    .find(|x| x.name() == f)
                    .with_context(|| format!("unknown family {f}"))?;
                all.iter().filter(|t| t.family == f).collect()
            }
            None => all.iter().collect(),
        };
        for _ in 0..n {
            let mut attempts = 0;
            let (template, doc) = loop {
                attempts += 1;
                anyhow::ensure!(attempts <= 100, "100 sample documents failed exclusions");
                let country = *countries.choose(&mut rng).context("no countries")?;
                let fitting: Vec<&&Template> = chosen
                    .iter()
                    .filter(|t| supported_template(t, country))
                    .collect();
                let Some(template) = fitting.choose(&mut rng) else {
                    continue;
                };
                let mut ctx = Ctx::new(country, &pools[&(Split::Train, country)]);
                let doc = render(template, &mut ctx, &mut rng)?;
                let doc = wrapped(doc, &ctx, usize::MAX, &mut rng)?;
                if exclusions.check(&doc).is_ok() {
                    break (*template, doc);
                }
            };
            println!(
                "--- {} #{} {}",
                template.family.name(),
                template.id,
                doc.country
            );
            println!("{}\n", bracketed(&doc));
        }
        return Ok(());
    }

    let (rows, dropped) = generate(
        gen_cfg,
        &countries,
        &pools,
        &all,
        &exclusions,
        &silver.surfaces,
        &mut rng,
    )?;
    for row in &rows {
        exclusions.check(&row.doc)?;
        if row.split != Split::Train {
            silver.surfaces.check(&row.doc)?;
        }
    }
    let out = PathBuf::from(&cfg.data.processed);
    std::fs::create_dir_all(&out).with_context(|| format!("creating {}", out.display()))?;
    for split in Split::ALL {
        let subset: Vec<&Row> = rows.iter().filter(|r| r.split == split).collect();
        write_split(&out.join(format!("{}.parquet", split.name())), &subset)?;
    }
    write_manifest(
        &cfg,
        gen_cfg,
        &rows,
        &dropped,
        &pools,
        &all,
        &exclusions,
        &silver,
    )?;
    print_summary(&rows, &dropped);
    Ok(())
}

fn static_countries(codes: &[String]) -> anyhow::Result<Vec<&'static str>> {
    const KNOWN: &[&str] = &["US"];
    codes
        .iter()
        .map(|c| {
            KNOWN
                .iter()
                .find(|k| *k == c)
                .copied()
                .with_context(|| format!("no generator support for country {c}"))
        })
        .collect()
}

/// The documents to generate, in order: (split, subset, count).
fn plan(cfg: &GenerateConfig) -> [(Split, Subset, usize); 5] {
    [
        (Split::Train, Subset::SeenTemplates, cfg.train),
        (Split::Valid, Subset::SeenTemplates, cfg.valid),
        (Split::Test, Subset::SeenTemplates, cfg.test_seen),
        (
            Split::Test,
            Subset::HeldoutTemplates,
            cfg.test_heldout_templates,
        ),
        (
            Split::Test,
            Subset::HeldoutFamilies,
            cfg.test_heldout_families,
        ),
    ]
}

/// The templates a subset draws from. Held-out templates are all `Both`, so that subset is
/// exempt from the category quotas.
fn eligible(all: &[Template], subset: Subset) -> Vec<&Template> {
    all.iter()
        .filter(|t| match subset {
            Subset::SeenTemplates => !t.test_only(),
            Subset::HeldoutTemplates => !t.family.held_out() && t.id % 10 == 9,
            Subset::HeldoutFamilies => t.family.held_out(),
        })
        .collect()
}

/// A category by the quotas, then a template of it; for held-out templates, any of them.
fn choose_template<'t>(
    pool: &[&'t Template],
    subset: Subset,
    rng: &mut ChaCha8Rng,
) -> anyhow::Result<&'t Template> {
    if subset == Subset::HeldoutTemplates {
        return pool.choose(rng).copied().context("no held-out templates");
    }
    let total: u32 = Category::ALL.iter().map(|c| c.weight()).sum();
    let mut roll = rng.random_range(0..total);
    let category = Category::ALL
        .into_iter()
        .find(|c| {
            if roll < c.weight() {
                true
            } else {
                roll -= c.weight();
                false
            }
        })
        .context("category roll out of range")?;
    let choices: Vec<&Template> = pool
        .iter()
        .filter(|t| t.category == category)
        .copied()
        .collect();
    // Real notice development cases contain many acronym mentions and room-prefixed addresses;
    // uniform template sampling made both rare in synthetic training.
    let target = match rng.random_range(0..6) {
        0 => "{org_acr",
        1 => "{address_prefixed",
        _ => "",
    };
    let focused: Vec<_> = choices
        .iter()
        .copied()
        .filter(|t| t.text.contains(target))
        .collect();
    let candidates = if focused.is_empty() {
        &choices
    } else {
        &focused
    };
    candidates
        .choose(rng)
        .copied()
        .with_context(|| format!("no {} template in {}", category.name(), subset.name()))
}

type Dropped = BTreeMap<&'static str, usize>;

/// Renders every planned document; a document that fails a post-render check is counted
/// and replaced by a fresh draw, so the split sizes are exact.
fn generate(
    cfg: &GenerateConfig,
    countries: &[&'static str],
    pools: &HashMap<(Split, &'static str), Pools>,
    all: &[Template],
    exclusions: &GoldExclusions,
    silver: &GoldExclusions,
    rng: &mut ChaCha8Rng,
) -> anyhow::Result<(Vec<Row>, Dropped)> {
    let mut rows = Vec::new();
    let mut dropped = Dropped::new();
    let mut seen_text = HashSet::new();
    for (split, subset, count) in plan(cfg) {
        let pool = eligible(all, subset);
        for _ in 0..count {
            let mut attempts = 0;
            let doc = loop {
                attempts += 1;
                if attempts > 100 {
                    bail!("100 rendered documents in a row failed the checks; see `dropped`");
                }
                let country = *countries.choose(rng).context("no countries")?;
                let fitting: Vec<&Template> = pool
                    .iter()
                    .copied()
                    .filter(|t| supported_template(t, country))
                    .collect();
                let template = choose_template(&fitting, subset, rng)?;
                let mut ctx = Ctx::new(country, &pools[&(split, country)]);
                let doc = wrapped(render(template, &mut ctx, rng)?, &ctx, cfg.max_tokens, rng)?;
                match check(&doc, cfg.max_tokens) {
                    Ok(()) if exclusions.check(&doc).is_err() => {
                        *dropped.entry(Drop::EvaluationOverlap.name()).or_default() += 1;
                    }
                    Ok(()) if split != Split::Train && silver.check(&doc).is_err() => {
                        *dropped.entry(Drop::SilverOverlap.name()).or_default() += 1;
                    }
                    Ok(()) if seen_text.insert(doc.text.clone()) => break doc,
                    Ok(()) => *dropped.entry(Drop::Duplicate.name()).or_default() += 1,
                    Err(d) => *dropped.entry(d.name()).or_default() += 1,
                }
            };
            rows.push(Row {
                id: rows.len() as u64,
                split,
                subset,
                doc,
            });
        }
    }
    Ok((rows, dropped))
}

/// Reads the names and address pools per (split, country).
fn load_pools(
    names: &Path,
    addresses: &Path,
    countries: &[&'static str],
    exclusions: &GoldExclusions,
    silver: &GoldExclusions,
) -> anyhow::Result<HashMap<(Split, &'static str), Pools>> {
    let reserved_listed = crate::bodies::reserved_listed_surfaces()?;
    let mut pools: HashMap<(Split, &'static str), Pools> = HashMap::new();
    for &c in countries {
        for split in Split::ALL {
            pools.insert(
                (split, c),
                Pools {
                    bodies: Bodies::new(c, split),
                    ..Pools::default()
                },
            );
        }
    }
    let people = read_table(
        &names.join("people.parquet"),
        &["name", "country", "script", "split"],
    )?;
    for row in people {
        let not_a_person = [
            "disappearance of",
            "assassination of",
            "murder of",
            "death of",
        ]
        .iter()
        .any(|p| row[0].to_lowercase().starts_with(p))
            || row[0].contains(['@', '$']);
        if not_a_person
            || exclusions.people.contains(&normalize_surface(&row[0]))
            || person_key(&row[0]).is_some_and(|key| exclusions.person_keys.contains(&key))
        {
            continue;
        }
        if let Some(p) = pool_mut(&mut pools, &row[1], &row[3]) {
            let split = Split::ALL.into_iter().find(|s| s.name() == row[3]);
            if split != Some(Split::Train)
                && (silver.people.contains(&normalize_surface(&row[0]))
                    || person_key(&row[0]).is_some_and(|key| silver.person_keys.contains(&key)))
            {
                continue;
            }
            p.people.push(PoolPerson {
                name: row[0].clone(),
                script: row[2].clone(),
            });
        }
    }
    let orgs = read_table(
        &names.join("orgs.parquet"),
        &["name", "country", "legal_form_abbr", "split", "source"],
    )?;
    for row in orgs {
        let name = if row[4] == "wikidata" {
            pool_filter::without_disambiguator(&row[0])
        } else {
            row[0].clone()
        };
        let name = pool_filter::without_article(&name).to_string();
        if exclusions.orgs.contains(&normalize_surface(&name)) {
            continue;
        }
        if imported_org_collides_with_listed(&name, &row[2], &reserved_listed) {
            continue;
        }
        // Private trusts, pension schemes, and street-named property companies fill the
        // registers but rarely documents: a few stay, never one that embeds a person's name.
        let keep = if pool_filter::names_a_person(&name) {
            false
        } else if pool_filter::private_arrangement(&name) {
            pool_filter::keep_some(&name, 50)
        } else if pool_filter::address_like(&name) {
            pool_filter::keep_some(&name, 5)
        } else {
            true
        };
        if !keep {
            continue;
        }
        if let Some(p) = pool_mut(&mut pools, &row[1], &row[3]) {
            let split = Split::ALL.into_iter().find(|s| s.name() == row[3]);
            if split != Some(Split::Train) && silver.orgs.contains(&normalize_surface(&name)) {
                continue;
            }
            p.orgs.push(PoolOrg {
                name,
                legal_form: row[2].clone(),
            });
        }
    }
    for split in Split::ALL {
        let path = addresses.join(format!("{}.parquet", split.name()));
        for e in read_shard(&path, split)? {
            if e.augmented || pool_filter::hydrant(&e.text) {
                continue;
            }
            if !pool_filter::postal(&e) {
                continue;
            }
            let address = PoolAddress::from_example(&e);
            let complete = e
                .spans
                .iter()
                .any(|span| span.label == tessera::AddressLabel::City)
                && e.spans
                    .iter()
                    .any(|span| span.label == tessera::AddressLabel::Postcode);
            if exclusions
                .addresses
                .contains(&normalize_surface(&address.text))
                || exclusions
                    .addresses
                    .contains(&normalize_surface(&address.one_line()))
            {
                continue;
            }
            if let Some(p) = pool_mut(&mut pools, &e.country, split.name()) {
                if split != Split::Train
                    && (silver.addresses.contains(&normalize_surface(&address.text))
                        || silver
                            .addresses
                            .contains(&normalize_surface(&address.one_line())))
                {
                    continue;
                }
                if address.multiline() {
                    p.multiline_addresses.push(p.addresses.len());
                    if complete {
                        p.complete_multiline_addresses.push(p.addresses.len());
                    }
                }
                p.addresses.push(address);
            }
        }
    }
    Ok(pools)
}

fn pool_mut<'p>(
    pools: &'p mut HashMap<(Split, &'static str), Pools>,
    country: &str,
    split: &str,
) -> Option<&'p mut Pools> {
    let split = Split::ALL.into_iter().find(|s| s.name() == split)?;
    pools
        .iter_mut()
        .find(|((s, c), _)| *s == split && *c == country)
        .map(|(_, p)| p)
}

/// The named string columns of a Parquet file, row by row.
fn read_table(path: &Path, columns: &[&str]) -> anyhow::Result<Vec<Vec<String>>> {
    let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let df = ParquetReader::new(file).finish()?;
    let cols = columns
        .iter()
        .map(|c| {
            df.column(c)
                .and_then(|s| s.str().cloned())
                .with_context(|| format!("{} column {c}", path.display()))
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    (0..df.height())
        .map(|i| {
            cols.iter()
                .map(|c| c.get(i).map(str::to_string).context("null value"))
                .collect()
        })
        .collect()
}

/// The document with each entity wrapped as `[kind:text]`, for eyeballing templates.
fn bracketed(doc: &Doc) -> String {
    let mut out = String::new();
    let mut at = 0;
    for e in &doc.entities {
        out.push_str(&doc.text[at..e.start]);
        out.push_str(&format!("[{}:{}]", e.kind, &doc.text[e.start..e.end]));
        at = e.end;
    }
    out.push_str(&doc.text[at..]);
    out
}

fn write_split(path: &Path, rows: &[&Row]) -> anyhow::Result<()> {
    let entities = rows
        .iter()
        .map(|r| serde_json::to_string(&r.doc.entities))
        .collect::<Result<Vec<_>, _>>()?;
    let links = rows
        .iter()
        .map(|r| serde_json::to_string(&r.doc.links))
        .collect::<Result<Vec<_>, _>>()?;
    let str_col = |name: &str, f: &dyn Fn(&Row) -> String| -> Column {
        Column::new(name.into(), rows.iter().map(|r| f(r)).collect::<Vec<_>>())
    };
    let mut df = DataFrame::new_infer_height(vec![
        Column::new(
            "doc_id".into(),
            rows.iter().map(|r| r.id).collect::<Vec<u64>>(),
        ),
        str_col("split", &|r| r.split.name().to_string()),
        str_col("subset", &|r| r.subset.name().to_string()),
        str_col("family", &|r| r.doc.family.name().to_string()),
        Column::new(
            "template_id".into(),
            rows.iter().map(|r| r.doc.template_id).collect::<Vec<u32>>(),
        ),
        str_col("category", &|r| r.doc.category.name().to_string()),
        str_col("country", &|r| r.doc.country.to_string()),
        str_col("text", &|r| r.doc.text.clone()),
        Column::new("entities_json".into(), entities),
        Column::new("links_json".into(), links),
        Column::new(
            "phones_fixture_safe".into(),
            rows.iter()
                .map(|r| r.doc.phones_fixture_safe)
                .collect::<Vec<bool>>(),
        ),
    ])?;
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    ParquetWriter::new(file).finish(&mut df)?;
    Ok(())
}

fn write_manifest(
    cfg: &Config,
    gen_cfg: &GenerateConfig,
    rows: &[Row],
    dropped: &Dropped,
    pools: &HashMap<(Split, &'static str), Pools>,
    all: &[Template],
    exclusions: &GoldExclusions,
    silver: &SilverExclusions,
) -> anyhow::Result<()> {
    let count_by = |f: &dyn Fn(&Row) -> String| -> BTreeMap<String, BTreeMap<&'static str, usize>> {
        let mut out: BTreeMap<String, BTreeMap<&'static str, usize>> = BTreeMap::new();
        for r in rows {
            *out.entry(f(r))
                .or_default()
                .entry(r.split.name())
                .or_default() += 1;
        }
        out
    };
    let train: Vec<&Row> = rows.iter().filter(|r| r.split == Split::Train).collect();
    let category_share: BTreeMap<&str, f64> = Category::ALL
        .iter()
        .map(|c| {
            let n = train.iter().filter(|r| r.doc.category == *c).count();
            (c.name(), n as f64 / train.len().max(1) as f64)
        })
        .collect();
    let mut pool_sizes: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    for ((split, country), p) in pools {
        pool_sizes.insert(
            format!("{country}/{}", split.name()),
            serde_json::json!({
                "people": p.people.len(),
                "orgs": p.orgs.len(),
                "borrowed_orgs": 0,
                "addresses": p.addresses.len(),
                "multiline_addresses": p.multiline_addresses.len(),
                "complete_multiline_addresses": p.complete_multiline_addresses.len(),
            }),
        );
    }
    let safe = rows.iter().filter(|r| r.doc.phones_fixture_safe).count();
    let real_silver_sha256: BTreeMap<&str, &str> = silver
        .sources
        .iter()
        .map(|source| (source.path.as_str(), source.sha256.as_str()))
        .collect();
    let mut synthetic_parquet_sha256 = BTreeMap::new();
    for split in Split::ALL {
        let path = Path::new(&cfg.data.processed).join(format!("{}.parquet", split.name()));
        synthetic_parquet_sha256.insert(
            split.name(),
            format!("{:x}", Sha256::digest(std::fs::read(&path)?)),
        );
    }
    let manifest = serde_json::json!({
        "source": "tessera-generator",
        "version": 2,
        "template_policy": "exclude_unverified_us_unit_parent_v1",
        "synthetic_parquet_sha256": synthetic_parquet_sha256,
        "seed": cfg.seed,
        "templates": all.len(),
        "max_tokens": gen_cfg.max_tokens,
        "names": cfg.names.as_ref().map(|n| n.out.clone()),
        "addresses": gen_cfg.addresses,
        "public_bodies": "trainer/data/us-public-bodies.json",
        "public_bodies_sha256": format!("{:x}", Sha256::digest(include_bytes!("../data/us-public-bodies.json"))),
        "exclude_gold": gen_cfg.exclude_gold,
        "exclude_gold_sha256": exclusions.sha256,
        "real_silver_sha256": real_silver_sha256,
        "use": "detector training data only: documents contain real Wikidata, GLEIF, and USAGov names and never enter versioned fixtures",
        "counts": {
            "split": count_by(&|r| r.split.name().to_string()),
            "subset": count_by(&|r| r.subset.name().to_string()),
            "family": count_by(&|r| r.doc.family.name().to_string()),
            "category": count_by(&|r| r.doc.category.name().to_string()),
            "country": count_by(&|r| r.doc.country.to_string()),
        },
        "train_category_share": category_share,
        "dropped": dropped,
        "pools": pool_sizes,
        "phones_fixture_safe_ratio": safe as f64 / rows.len().max(1) as f64,
    });
    let path = Path::new(&cfg.data.manifests).join("detector-synthetic.json");
    std::fs::write(&path, serde_json::to_string_pretty(&manifest)? + "\n")
        .with_context(|| format!("writing {}", path.display()))
}

fn print_summary(rows: &[Row], dropped: &Dropped) {
    for split in Split::ALL {
        let n = rows.iter().filter(|r| r.split == split).count();
        println!("{:<6} {n}", split.name());
    }
    let train: Vec<&Row> = rows.iter().filter(|r| r.split == Split::Train).collect();
    for c in Category::ALL {
        let n = train.iter().filter(|r| r.doc.category == c).count();
        println!(
            "train {:<24} {:.3}",
            c.name(),
            n as f64 / train.len().max(1) as f64
        );
    }
    println!("dropped {dropped:?}");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrapping_keeps_person_free_categories_person_free() {
        let pools = stub_pools(&[("Maya Johnson", "Latin")]);
        let ctx = Ctx::new("US", &pools);
        for category in [Category::Nothing, Category::NoPersonWithAddress] {
            let (text, entities) = if category == Category::Nothing {
                ("Office Hours".to_string(), Vec::new())
            } else {
                (
                    "Mail to 10 Main St".to_string(),
                    vec![Gold {
                        kind: "address",
                        start: 8,
                        end: 18,
                    }],
                )
            };
            let doc = Doc {
                text,
                entities,
                links: Vec::new(),
                family: Family::Prose,
                template_id: 0,
                category,
                country: "US",
                phones_fixture_safe: true,
            };
            for seed in 0..200 {
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                let wrapped = wrapped(doc.clone(), &ctx, 900, &mut rng).unwrap();
                assert!(category.matches(&wrapped.entities), "seed {seed}");
            }
        }
    }

    #[test]
    fn gold_exclusions_normalize_us_entity_surfaces() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("gold.jsonl");
        std::fs::write(
            &path,
            "{\"country\":\"US\",\"input\":\"source text\",\"expected\":[{\"kind\":\"person\",\"text\":\"Maya  Johnson\"},{\"kind\":\"org\",\"text\":\"Example Corp\"},{\"kind\":\"address\",\"text\":\"10 Main St\"}]}\n",
        )
        .unwrap();
        let exclusions = GoldExclusions::load(&path).unwrap();
        assert!(
            exclusions
                .people
                .contains(&normalize_surface("maya johnson"))
        );
        assert!(exclusions.orgs.contains(&normalize_surface("EXAMPLE CORP")));
        assert!(
            exclusions
                .addresses
                .contains(&normalize_surface("10 Main St"))
        );
        let make_doc = |text: &str, kind: &'static str, end: usize| Doc {
            text: text.to_string(),
            entities: vec![Gold {
                kind,
                start: 0,
                end,
            }],
            links: Vec::new(),
            family: Family::Prose,
            template_id: 0,
            category: Category::Both,
            country: "US",
            phones_fixture_safe: true,
        };
        assert!(
            exclusions
                .check(&make_doc("source text", "person", 6))
                .is_err()
        );
        assert!(
            exclusions
                .check(&make_doc("Maya Johnson", "person", 12))
                .is_err()
        );
        assert!(
            exclusions
                .check(&make_doc("Maya A. Johnson", "person", 15))
                .is_err()
        );
        assert_eq!(person_key("Anthony Lee Jr."), person_key("Anthony T. Lee"));
        assert_eq!(person_key("Johnson, Maya"), person_key("Maya Johnson"));
        assert!(
            exclusions
                .check(&make_doc("other name", "person", 10))
                .is_ok()
        );
    }

    #[test]
    fn public_body_acronyms_exclude_their_full_names() {
        let mut acronym = GoldExclusions::default();
        acronym.add("org", "NCI").unwrap();
        acronym.expand_public_body_aliases().unwrap();
        assert!(
            acronym
                .orgs
                .contains(&normalize_surface("National Cancer Institute"))
        );

        let mut full_name = GoldExclusions::default();
        full_name
            .add(
                "org",
                "Pipeline and Hazardous Materials Safety Administration",
            )
            .unwrap();
        full_name.expand_public_body_aliases().unwrap();
        assert!(full_name.orgs.contains(&normalize_surface("PHMSA")));
    }

    #[test]
    fn silver_exclusions_read_every_source_and_reject_invalid_offsets() {
        let dir = tempfile::tempdir().unwrap();
        let first = dir.path().join("first.jsonl");
        let second = dir.path().join("second.jsonl");
        let make_case = |text: &str, kind: &str, surface: &str| {
            let start = text.find(surface).unwrap();
            serde_json::json!({
                "country": "US",
                "text": text,
                "entities": [{"kind": kind, "start": start, "end": start + surface.len()}]
            })
            .to_string()
                + "\n"
        };
        std::fs::write(
            &first,
            make_case("Contact Maya Johnson", "person", "Maya Johnson"),
        )
        .unwrap();
        std::fs::write(
            &second,
            make_case("Example Office", "org", "Example Office"),
        )
        .unwrap();
        let paths = vec![first.display().to_string(), second.display().to_string()];
        let loaded = SilverExclusions::load(&paths).unwrap();
        assert_eq!(loaded.sources.len(), 2);
        assert_eq!(
            loaded.sources[0].sha256,
            format!("{:x}", Sha256::digest(std::fs::read(&first).unwrap()))
        );
        assert!(
            loaded
                .surfaces
                .person_keys
                .contains(&person_key("Maya A. Johnson").unwrap())
        );
        assert!(
            loaded
                .surfaces
                .orgs
                .contains(&normalize_surface("Example Office"))
        );

        std::fs::write(&second, "{\"country\":\"US\",\"text\":\"José\",\"entities\":[{\"kind\":\"person\",\"start\":4,\"end\":5}]}\n").unwrap();
        assert!(SilverExclusions::load(&paths).is_err());
    }

    fn stub_pools(people: &[(&str, &str)]) -> Pools {
        Pools {
            people: people
                .iter()
                .map(|(n, s)| PoolPerson {
                    name: n.to_string(),
                    script: s.to_string(),
                })
                .collect(),
            orgs: vec![PoolOrg {
                name: "Pacific Cargo LLC".into(),
                legal_form: "LLC".into(),
            }],
            addresses: vec![
                PoolAddress {
                    text: "1200 Market Street\nPhiladelphia, PA 19107".into(),
                },
                PoolAddress {
                    text: "4 Misty Wood Circle, Austin, TX 78701".into(),
                },
            ],
            multiline_addresses: vec![0],
            complete_multiline_addresses: vec![0],
            bodies: Bodies::new("US", Split::Train),
        }
    }

    #[test]
    fn multiline_contact_addresses_prefer_complete_postal_blocks() {
        let mut pools = stub_pools(&[("Maya Johnson", "latin")]);
        pools.addresses.push(PoolAddress {
            text: "1200 Market Street\nSuite 2".into(),
        });
        pools.multiline_addresses.push(2);
        let ctx = Ctx::new("US", &pools);
        let mut rng = ChaCha8Rng::seed_from_u64(17);
        let complete = (0..1000)
            .filter(|_| ctx.address(true, &mut rng).unwrap().text.contains("19107"))
            .count();
        assert!((700..950).contains(&complete), "{complete}");
    }

    fn template(text: &'static str) -> Template {
        Template {
            family: Family::Prose,
            id: 0,
            category: Category::Nothing,
            countries: &[],
            text,
        }
    }

    #[test]
    fn render_records_exact_offsets() {
        let pools = Pools {
            orgs: Vec::new(),
            ..stub_pools(&[("Maya Johnson", "latin")])
        };
        let mut ctx = Ctx::new("US", &pools);
        ctx.plain = true;
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let mut doc = render(
            &template("Hi {person#1}, mail {email#1}."),
            &mut ctx,
            &mut rng,
        )
        .unwrap();
        assert!(doc.text.starts_with("Hi Maya Johnson, mail maya.johnson@"));
        assert!(doc.text.ends_with(".example."));
        assert_eq!(
            doc.entities[0],
            Gold {
                kind: "person",
                start: 3,
                end: 15
            }
        );
        assert_eq!(doc.entities[1].kind, "email");
        assert_eq!(doc.entities[1].start, 22);
        assert_eq!(doc.entities[1].end, doc.text.len() - 1);
        assert_eq!(&doc.text[3..15], "Maya Johnson");
        doc.category = Category::NoAddressWithPerson;
        assert_eq!(check(&doc, 900), Ok(()));
    }

    #[test]
    fn multiline_addresses_keep_their_lines_inside_the_span() {
        let pools = stub_pools(&[("Maya Johnson", "latin")]);
        let mut ctx = Ctx::new("US", &pools);
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let mut doc = render(
            &template("Ship to:\n{address_ml}\nThanks"),
            &mut ctx,
            &mut rng,
        )
        .unwrap();
        let e = &doc.entities[0];
        assert_eq!(
            &doc.text[e.start..e.end],
            "1200 Market Street\nPhiladelphia, PA 19107"
        );
        doc.category = Category::NoPersonWithAddress;
        assert_eq!(check(&doc, 900), Ok(()));
    }

    #[test]
    fn single_line_addresses_join_lines_with_commas() {
        let a = PoolAddress {
            text: "1200 Market Street\nPhiladelphia, PA 19107".into(),
        };
        assert_eq!(a.one_line(), "1200 Market Street, Philadelphia, PA 19107");
    }

    #[test]
    fn former_person_street_templates_use_complete_pool_addresses() {
        let all = templates::all().unwrap();
        let pools = stub_pools(&[("Maya Johnson", "latin")]);
        let allowed: Vec<String> = pools
            .addresses
            .iter()
            .flat_map(|address| [address.text.clone(), address.one_line()])
            .collect();
        for id in [5, 208, 308, 609] {
            let template = all.iter().find(|template| template.id == id).unwrap();
            assert!(template.fits("US"));
            let mut ctx = Ctx::new("US", &pools);
            ctx.plain = true;
            let mut rng = ChaCha8Rng::seed_from_u64(id as u64);
            let doc = render(template, &mut ctx, &mut rng).unwrap();
            let addresses: Vec<&str> = doc
                .entities
                .iter()
                .filter(|entity| entity.kind == "address")
                .map(|entity| &doc.text[entity.start..entity.end])
                .collect();
            assert!(!addresses.is_empty(), "template {id} lost its address");
            assert!(
                addresses
                    .iter()
                    .all(|address| allowed.iter().any(|item| item == address)),
                "template {id} assembled a street and city from separate sources: {addresses:?}"
            );
        }
    }

    #[test]
    fn public_body_templates_are_selected() {
        let all = templates::all().unwrap();
        let supported: Vec<_> = all.iter().filter(|t| supported_template(t, "US")).collect();
        assert!(!supported.is_empty());
        assert!(supported.iter().any(|template| template.id == 231));
        assert!(!supported.iter().any(|template| template.id == 230));
        assert!(supported.iter().any(|template| {
            template
                .slots()
                .unwrap()
                .iter()
                .any(|item| matches!(item.slot, Slot::OrgGov | Slot::OrgAcronym))
        }));
        assert!(supported.iter().all(|template| {
            template
                .slots()
                .unwrap()
                .iter()
                .all(|item| item.slot != Slot::OrgLocalOffice)
        }));
    }

    #[test]
    fn synthetic_units_do_not_claim_unverified_parent_organizations() {
        let all = templates::all().unwrap();
        for id in [72, 77, 622] {
            let template = all.iter().find(|template| template.id == id).unwrap();
            assert!(!supported_template(template, "US"), "template {id}");
        }
        let standalone = all.iter().find(|template| template.id == 621).unwrap();
        assert!(supported_template(standalone, "US"));
        for template in all
            .iter()
            .filter(|template| supported_template(template, "US"))
        {
            let slots = template.slots().unwrap();
            assert!(
                !slots.iter().any(|item| item.slot == Slot::OrgUnit)
                    || !slots.iter().any(|item| {
                        matches!(item.slot, Slot::Org | Slot::OrgGov | Slot::OrgAcronym)
                    }),
                "template {} combines independent unit and parent draws",
                template.id
            );
        }
    }

    #[test]
    fn held_public_body_countries_draw_generic_orgs_without_bodies() {
        for country in ["US"] {
            for split in Split::ALL {
                let mut pools = stub_pools(&[("Maya Johnson", "latin")]);
                pools.bodies = Bodies::new(country, split);
                let ctx = Ctx::new(country, &pools);
                for seed in 0..200 {
                    let mut rng = ChaCha8Rng::seed_from_u64(seed);
                    assert!(ctx.org(&mut rng).is_ok(), "{country} {split:?} seed {seed}");
                }
            }
        }
    }

    #[test]
    fn public_body_hold_preserves_template_category_coverage() {
        let all = templates::all().unwrap();
        for country in ["US"] {
            for subset in [Subset::SeenTemplates, Subset::HeldoutFamilies] {
                for category in Category::ALL {
                    assert!(
                        eligible(&all, subset)
                            .iter()
                            .any(|t| t.category == category && supported_template(t, country)),
                        "{country} {} {}",
                        subset.name(),
                        category.name()
                    );
                }
            }
            assert!(
                eligible(&all, Subset::HeldoutTemplates)
                    .iter()
                    .any(|t| supported_template(t, country)),
                "{country} heldout templates"
            );
        }
    }

    #[test]
    fn every_template_renders_for_every_country() {
        let all = templates::all().unwrap();
        for country in ["US"] {
            for split in [Split::Train, Split::Valid, Split::Test] {
                let mut pools = stub_pools(&[("Maya Johnson", "latin"), ("山田太郎", "han")]);
                pools.bodies = Bodies::new(country, split);
                for (i, t) in all
                    .iter()
                    .enumerate()
                    .filter(|(_, t)| supported_template(t, country))
                {
                    let mut ctx = Ctx::new(country, &pools);
                    let mut rng = ChaCha8Rng::seed_from_u64(i as u64);
                    let doc = render(t, &mut ctx, &mut rng).unwrap();
                    assert_eq!(
                        check(&doc, 900),
                        Ok(()),
                        "template {} in {country}: {:?}",
                        t.id,
                        doc.text
                    );
                    assert_eq!(t.derived_category().unwrap(), t.category);
                }
            }
        }
    }

    #[test]
    fn linked_slots_reuse_their_values() {
        let pools = stub_pools(&[("Maya Johnson", "latin"), ("Anna Schmidt", "latin")]);
        let mut ctx = Ctx::new("US", &pools);
        ctx.plain = true;
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let doc = render(
            &template("{person#1} / {person_first#1} / {email#1} / {email#1}"),
            &mut ctx,
            &mut rng,
        )
        .unwrap();
        let text = |i: usize| &doc.text[doc.entities[i].start..doc.entities[i].end];
        assert!(text(0).starts_with(text(1)));
        assert_eq!(text(2), text(3));
    }

    #[test]
    fn safe_phones_are_found_by_the_rules_with_their_region() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let mut missed = Vec::new();
        for c in ["US"] {
            for _ in 0..300 {
                let (value, safe) = phone(c, rng.random_bool(0.5), &mut rng).unwrap();
                if !safe {
                    continue;
                }
                let text = format!("Call {value} today.");
                let found = tessera::internal::scan_rules(&text, &[c]);
                let hit = found.iter().any(|e| {
                    e.kind == Kind::Phone
                        && text[e.start..e.end] == value
                        && e.region.as_deref() == Some(c)
                });
                if !hit {
                    missed.push(format!("{c} {value}: {found:?}"));
                }
            }
        }
        missed.sort();
        missed.dedup_by(|a, b| a[..12] == b[..12]);
        assert!(missed.is_empty(), "{}", missed.join("\n"));
    }

    #[test]
    fn documents_draw_only_from_their_split() {
        let mut pools = HashMap::new();
        for (split, name) in [
            (Split::Train, "Train Person"),
            (Split::Valid, "Valid Person"),
            (Split::Test, "Test Person"),
        ] {
            let mut pool = stub_pools(&[(name, "latin")]);
            pool.bodies = Bodies::new("US", split);
            pools.insert((split, "US"), pool);
        }
        let cfg = GenerateConfig {
            train: 200,
            valid: 50,
            test_seen: 20,
            test_heldout_templates: 15,
            test_heldout_families: 15,
            addresses: String::new(),
            exclude_gold: String::new(),
            max_tokens: 900,
        };
        let all = templates::all().unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let (rows, _) = generate(
            &cfg,
            &["US"],
            &pools,
            &all,
            &GoldExclusions::default(),
            &GoldExclusions::default(),
            &mut rng,
        )
        .unwrap();
        assert_eq!(rows.len(), 300);
        assert_eq!(
            rows.iter()
                .map(|r| r.doc.text.as_str())
                .collect::<HashSet<_>>()
                .len(),
            rows.len(),
            "generated documents must not repeat across splits"
        );
        for r in rows.iter().filter(|r| r.split != Split::Train) {
            assert!(!r.doc.text.contains("Train Person"), "{}", r.doc.text);
        }
        let test_only: Vec<u32> = all.iter().filter(|t| t.test_only()).map(|t| t.id).collect();
        for r in rows.iter().filter(|r| r.split != Split::Test) {
            assert!(
                !test_only.contains(&r.doc.template_id),
                "{}",
                r.doc.template_id
            );
        }
    }

    #[test]
    fn legal_forms_strip_only_as_the_last_word() {
        assert_eq!(
            strip_legal_form("Pacific Cargo LLC", "LLC").as_deref(),
            Some("Pacific Cargo")
        );
        assert_eq!(
            strip_legal_form("Acme Ltd.", "Ltd").as_deref(),
            Some("Acme")
        );
        assert_eq!(strip_legal_form("LLC", "LLC"), None);
        assert_eq!(strip_legal_form("LLC Partners", "LLC"), None);
    }

    #[test]
    fn capitalized_names_keep_the_given_name_and_their_words() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let forms: HashSet<String> = (0..500)
            .map(|_| capitalized("Maravillas Abadía Jover", &mut rng))
            .collect();
        assert!(forms.contains("Maravillas Abadía Jover"));
        assert!(forms.contains("Maravillas ABADÍA JOVER"));
        assert!(forms.contains("MARAVILLAS ABADÍA JOVER"));
        assert_eq!(forms.len(), 3);
        assert_eq!(capitalized("Cher", &mut rng).to_lowercase(), "cher");
    }

    #[test]
    fn address_shapes_keep_source_locations() {
        let pools = stub_pools(&[("Maya Johnson", "latin")]);
        let ctx = Ctx::new("US", &pools);
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        assert_eq!(
            ctx.shaped("1 Main St, Albany, NY 12207".into(), ", ", &mut rng),
            "1 Main St, Albany, NY 12207"
        );
    }

    #[test]
    fn email_parts_are_ascii() {
        assert_eq!(
            ascii_words("José García-Pérez"),
            vec!["jose", "garcia", "perez"]
        );
        assert_eq!(slug("Pacific Cargo LLC"), "pacific-cargo-llc");
    }
}
