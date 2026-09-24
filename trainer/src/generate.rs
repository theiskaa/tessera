//! Synthetic detector documents: templates with slots, filled from the name, organization,
//! and address pools, with exact byte offsets recorded as the string is built.
//!
//! Every email uses a reserved domain and every phone number comes from a per-country
//! generator; only the generators marked `fixture_safe` draw from ranges reserved for
//! fiction. Names and organizations are real (Wikidata and GLEIF), so generated text is
//! training data only and never enters a versioned fixture.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use polars::prelude::{Column, DataFrame, ParquetReader, ParquetWriter, SerReader};
use rand::seq::{IndexedRandom, SliceRandom};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use tessera::internal::{fnv1a, is_content, tokenize};
use tessera::{AddressLabel, Kind};
use unicode_normalization::UnicodeNormalization;

use crate::bodies::Bodies;
use crate::config::{self, Config, GenerateConfig};
use crate::data::{LabelledExample, Split, read_shard};
use crate::filler;
use crate::negatives;
use crate::pool_filter;
use crate::templates::{self, Template};

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
}

/// A placeholder in a template.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Slot {
    Person,
    PersonFirst,
    PersonEponymous,
    Org,
    OrgEponymous,
    OrgSchool,
    OrgGov,
    OrgAcronym,
    OrgUnit,
    OrgRegistry,
    Address,
    AddressMultiline,
    AddressPersonStreet,
    Email,
    Phone,
    PhoneLocal,
    NegPlace,
    NegWordName,
    NegHandle,
    NegUrl,
    NegDate,
    NegPrice,
    NegOrder,
    NegIban,
    NegIp,
    NegDigits,
    NegRoadSentence,
    NegPartialLocation,
    NegHeader,
    NegDepartment,
    NegPrompt,
    NegHeading,
    NegCaps,
    NegLaw,
    NegLabel,
    NegHours,
    NegCitation,
    NegLine,
    NegCode,
    NegList,
    NegUnitCode,
    NegRegisterNo,
    Officers,
    Register,
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
            Person | PersonFirst | PersonEponymous => Some(Kind::Person),
            Org | OrgEponymous | OrgSchool | OrgGov | OrgAcronym | OrgUnit | OrgRegistry => {
                Some(Kind::Org)
            }
            Address | AddressMultiline | AddressPersonStreet => Some(Kind::Address),
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
            "person_eponymous" => PersonEponymous,
            "org" => Org,
            "org_eponymous" => OrgEponymous,
            "org_school" => OrgSchool,
            "org_gov" => OrgGov,
            "org_acr" => OrgAcronym,
            "org_unit" => OrgUnit,
            "org_registry" => OrgRegistry,
            "address" => Address,
            "address_ml" => AddressMultiline,
            "address_person_street" => AddressPersonStreet,
            "email" => Email,
            "phone" => Phone,
            "phone_local" => PhoneLocal,
            "neg_place" => NegPlace,
            "neg_word_name" => NegWordName,
            "neg_handle" => NegHandle,
            "neg_url" => NegUrl,
            "neg_date" => NegDate,
            "neg_price" => NegPrice,
            "neg_order" => NegOrder,
            "neg_iban" => NegIban,
            "neg_ip" => NegIp,
            "neg_digits" => NegDigits,
            "neg_road_sentence" => NegRoadSentence,
            "neg_partial_location" => NegPartialLocation,
            "neg_header" => NegHeader,
            "neg_department" => NegDepartment,
            "neg_prompt" => NegPrompt,
            "neg_heading" => NegHeading,
            "neg_caps" => NegCaps,
            "neg_law" => NegLaw,
            "neg_label" => NegLabel,
            "neg_hours" => NegHours,
            "neg_citation" => NegCitation,
            "neg_line" => NegLine,
            "neg_code" => NegCode,
            "neg_list" => NegList,
            "neg_unit_code" => NegUnitCode,
            "neg_register_no" => NegRegisterNo,
            "officers" => Officers,
            "register" => Register,
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
        self != Slot::OrgAcronym
            && matches!(
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

/// An address from the parser shards: one-line and multi-line renderings and its city.
#[derive(Debug, Clone)]
pub struct PoolAddress {
    pub text: String,
    pub city: Option<String>,
}

impl PoolAddress {
    fn from_example(e: &LabelledExample) -> PoolAddress {
        let city = e
            .spans
            .iter()
            .find(|s| s.label == AddressLabel::City)
            .map(|s| e.text[s.start as usize..s.end as usize].to_string());
        PoolAddress {
            text: pool_filter::with_local_country(e, &pool_filter::without_venue_lines(e)),
            city,
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
    /// Orgs added from other countries because the country's own pool was too small.
    pub borrowed_orgs: usize,
    pub bodies: Bodies,
}

/// Per-document fill state: the pools of the document's split and country, and the values
/// bound to link groups so far.
pub struct Ctx<'a> {
    pub country: &'static str,
    pools: &'a Pools,
    bound: HashMap<(Kind, u8), String>,
    bound_person: HashMap<u8, PoolPerson>,
    bound_acronym: HashMap<u8, String>,
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
            plain: false,
        }
    }

    fn fill(&mut self, slot: Slot, group: u8, rng: &mut ChaCha8Rng) -> anyhow::Result<Filled> {
        if group > 0
            && slot.bindable()
            && let Some(kind) = slot.kind()
            && let Some(v) = self.bound.get(&(kind, group))
            && slot != Slot::PersonFirst
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
            Slot::Org => Filled::plain(self.org(rng)?),
            Slot::OrgEponymous => Filled::plain(*pick(templates::EPONYMOUS_COMPANIES, rng)),
            Slot::OrgSchool => Filled::plain(*pick(templates::PERSON_NAMED_INSTITUTIONS, rng)),
            Slot::OrgGov => {
                let body = self.pools.bodies.body(group > 0, rng);
                if let Some(acronym) = body.acronym.filter(|_| group > 0) {
                    self.bound_acronym.insert(group, acronym);
                }
                Filled::plain(body.name)
            }
            Slot::OrgAcronym => Filled::plain(self.acronym(group, rng)),
            Slot::OrgUnit => Filled::plain(self.pools.bodies.unit(rng)),
            Slot::OrgRegistry => Filled::plain(self.pools.bodies.registry(rng)),
            Slot::Address => {
                let a = self.address(false, rng)?.one_line();
                Filled::plain(self.with_room(a, ", ", rng))
            }
            Slot::AddressMultiline => {
                let a = self.address(true, rng)?.text.clone();
                Filled::plain(self.with_room(a, "\n", rng))
            }
            Slot::AddressPersonStreet => {
                let (_, capital, streets) = templates::PERSON_NAMED_STREETS
                    .iter()
                    .find(|(c, _, _)| *c == self.country)
                    .with_context(|| format!("no person-named streets for {}", self.country))?;
                let city = self
                    .address(false, rng)?
                    .city
                    .clone()
                    .unwrap_or_else(|| capital.to_string());
                let number = rng.random_range(1..=250);
                Filled::plain(format!("{number} {}, {city}", pick(streets, rng)))
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
            Slot::NegWordName => Filled::plain(*pick(templates::NEG_WORD_NAMES, rng)),
            Slot::NegHandle => Filled::plain(*pick(templates::NEG_HANDLES, rng)),
            Slot::NegUrl => Filled::plain(*pick(templates::NEG_URLS, rng)),
            Slot::NegDate => Filled::plain(*pick(templates::NEG_DATES, rng)),
            Slot::NegPrice => Filled::plain(*pick(templates::NEG_PRICES, rng)),
            Slot::NegOrder => Filled::plain(*pick(templates::NEG_ORDERS, rng)),
            Slot::NegIban => Filled::plain(*pick(templates::NEG_IBANS, rng)),
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

    fn person(&self, rng: &mut ChaCha8Rng) -> anyhow::Result<PoolPerson> {
        self.pools
            .people
            .choose(rng)
            .cloned()
            .with_context(|| format!("no people for {}", self.country))
    }

    /// An honorific before a person's name and a post-nominal or name suffix after it, each
    /// drawn in the script and country the name fits: `Dr`, `Herr`, `ქალბატონი`, `OBE`, `様`.
    fn honorifics(
        &self,
        p: &PoolPerson,
        rng: &mut ChaCha8Rng,
    ) -> (Option<&'static str>, Option<&'static str>) {
        if p.cjk() {
            let suffix = rng
                .random_bool(0.3)
                .then(|| *pick(templates::NAME_SUFFIXES_JP, rng));
            return (None, suffix);
        }
        if !p.latin() {
            let prefix = (p.script == "georgian" && rng.random_bool(0.2))
                .then(|| *pick(templates::HONORIFICS_GE, rng));
            return (prefix, None);
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
    fn acronym(&mut self, group: u8, rng: &mut ChaCha8Rng) -> String {
        if let Some(a) = self.bound_acronym.get(&group) {
            return a.clone();
        }
        let body = self.pools.bodies.body(true, rng);
        let acronym = body.acronym.unwrap_or_else(|| body.name.clone());
        if group > 0 {
            self.bound_acronym.insert(group, acronym.clone());
            self.bound.entry((Kind::Org, group)).or_insert(body.name);
        }
        acronym
    }

    /// `address` with, one time in ten, a room, suite, or mail stop line before it, which
    /// real addresses carry and gold includes.
    fn with_room(&self, address: String, sep: &str, rng: &mut ChaCha8Rng) -> String {
        if self.plain || !rng.random_bool(0.1) {
            return address;
        }
        let n = rng.random_range(2..=950);
        let room = match self.country {
            "US" | "GB" => match rng.random_range(0..5) {
                0 => format!("Suite {n}"),
                1 => format!("Room {}{}", n, ["", "A", "B"][rng.random_range(0..3)]),
                2 => format!("Mail Stop {}", rng.random_range(1000..9999)),
                3 => format!("Floor {}", rng.random_range(2..=30)),
                _ => pick(templates::BUILDINGS, rng).to_string(),
            },
            "DE" => match rng.random_range(0..3) {
                0 => format!(
                    "Raum {}.{}",
                    rng.random_range(0..6),
                    rng.random_range(1..40)
                ),
                1 => format!("Gebäude {}", ["A", "B", "C", "D"][rng.random_range(0..4)]),
                _ => format!("{}. OG", rng.random_range(1..=6)),
            },
            _ => return address,
        };
        format!("{room}{sep}{address}")
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
        if rng.random_bool(0.25) {
            return Ok(self.pools.bodies.body(false, rng).name);
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
        let candidates: Vec<&PoolAddress> = if multiline {
            pool.iter().filter(|a| a.multiline()).collect()
        } else {
            pool.iter().collect()
        };
        candidates
            .choose(rng)
            .copied()
            .with_context(|| format!("no addresses for {}", self.country))
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

/// A phone number for `country` in international or national form, and whether its range is
/// reserved for fiction. See the plan's reserved-contact table.
fn phone(
    country: &str,
    international: bool,
    rng: &mut ChaCha8Rng,
) -> anyhow::Result<(String, bool)> {
    let digits = |rng: &mut ChaCha8Rng, n: usize| -> String {
        (0..n)
            .map(|_| char::from(b'0' + rng.random_range(0..10u8)))
            .collect()
    };
    Ok(match country {
        "GB" => {
            // Ofcom drama ranges as (international, national) prefixes; three digits follow.
            const RANGES: &[(&str, &str)] = &[
                ("20 7946 0", "020 7946 0"),
                ("113 496 0", "0113 496 0"),
                ("161 496 0", "0161 496 0"),
                ("7700 900", "07700 900"),
                ("808 157 0", "0808 157 0"),
            ];
            let (intl, national) = pick(RANGES, rng);
            let tail = digits(rng, 3);
            let value = if international {
                if rng.random_bool(0.3) && !intl.starts_with('7') {
                    format!("+44 (0){intl}{tail}")
                } else {
                    format!("+44 {intl}{tail}")
                }
            } else if rng.random_bool(0.2) {
                format!("{national}{tail}").replace(' ', "-")
            } else {
                format!("{national}{tail}")
            };
            (value, true)
        }
        "US" | "CA" => {
            // NANP area codes of the right country; 416, 604, and 514 are Canadian, so the
            // rules layer correctly assigns them region CA.
            const US_AREA: &[&str] = &["212", "415", "312", "617", "206", "305", "718"];
            const CA_AREA: &[&str] = &["416", "604", "514"];
            let area = pick(if country == "CA" { CA_AREA } else { US_AREA }, rng);
            let line = format!("01{}", digits(rng, 2));
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
            (value, true)
        }
        "DE" => {
            if rng.random_bool(0.2) {
                let rest = digits(rng, 7);
                let value = if international {
                    format!("+49 151 {rest}")
                } else {
                    format!("0151 {rest}")
                };
                (value, false)
            } else {
                let ext = digits(rng, 3);
                let value = if international {
                    format!("+49 30 23125 {ext}")
                } else {
                    match rng.random_range(0..3) {
                        0 => format!("030 23125 {ext}"),
                        1 => format!("(030) 23125-{ext}"),
                        _ => format!("030/23125{ext}"),
                    }
                };
                (value, true)
            }
        }
        "GE" => {
            let value = if rng.random_bool(0.6) {
                let (a, b, c, d) = (
                    digits(rng, 2),
                    digits(rng, 2),
                    digits(rng, 2),
                    digits(rng, 2),
                );
                if international {
                    format!("+995 5{a} {b} {c} {d}")
                } else {
                    format!("5{a} {b} {c} {d}")
                }
            } else {
                let (a, b, c) = (digits(rng, 2), digits(rng, 2), digits(rng, 2));
                if international {
                    format!("+995 32 2{a} {b}{c}")
                } else {
                    format!("032 2 {a} {b} {c}")
                }
            };
            (value, false)
        }
        "JP" => {
            let (a, b) = (digits(rng, 4), digits(rng, 4));
            let value = match (rng.random_bool(0.5), international) {
                (true, true) => format!("+81 3 {a} {b}"),
                (true, false) => format!("03-{a}-{b}"),
                (false, true) => format!("+81 90 {a} {b}"),
                (false, false) => format!("090-{a}-{b}"),
            };
            (value, false)
        }
        other => bail!("no phone generator for {other}"),
    })
}

/// Renders a template, measuring every labelled value's span on the string being built, so
/// the offsets are exact by construction. Honorifics are written before the span starts.
pub fn render(template: &Template, ctx: &mut Ctx, rng: &mut ChaCha8Rng) -> anyhow::Result<Doc> {
    let mut text = String::new();
    let mut entities = Vec::new();
    let mut links = Vec::new();
    let mut phones_fixture_safe = true;
    let mut rest = template.text;
    for (slot, group) in template.slots()? {
        let open = rest
            .find('{')
            .context("slot list and template text disagree")?;
        let close = open + rest[open..].find('}').context("unclosed slot")?;
        text.push_str(&rest[..open]);
        let filled = ctx.fill(slot, group, rng)?;
        phones_fixture_safe &= filled.fixture_safe;
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

/// `doc` padded with filler text in its country's language (see `filler`), or `doc` as it was
/// when the padded one would not fit in `max_tokens`.
fn wrapped(doc: Doc, ctx: &Ctx, max_tokens: usize, rng: &mut ChaCha8Rng) -> anyhow::Result<Doc> {
    let author = ctx.person(rng)?.name;
    let padded = filler::Wrap::draw(ctx.country, &author, rng).apply(doc.clone());
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
}

impl Drop {
    fn name(self) -> &'static str {
        match self {
            Drop::BoundaryMisaligned => "boundary_misaligned",
            Drop::Overlap => "overlap",
            Drop::TooLong => "too_long",
            Drop::EmptyEntity => "empty_entity",
        }
    }
}

/// The post-render checks: every span starts and ends on a token boundary, spans do not
/// overlap or come out empty, and the document fits the model without chunking.
pub fn check(doc: &Doc, max_tokens: usize) -> Result<(), Drop> {
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

/// Pools below these sizes borrow organizations from other countries, so one small pool is
/// not repeated in thousands of documents.
fn min_org_pool(split: Split) -> usize {
    match split {
        Split::Train => 2000,
        Split::Valid | Split::Test => 200,
    }
}

struct Row {
    id: u64,
    split: Split,
    subset: Subset,
    doc: Doc,
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
    let pools = load_pools(
        Path::new(&names.out),
        Path::new(&gen_cfg.addresses),
        &countries,
        cfg.seed,
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
            let country = *countries.choose(&mut rng).context("no countries")?;
            let template = chosen.choose(&mut rng).context("no templates")?;
            let mut ctx = Ctx::new(country, &pools[&(Split::Train, country)]);
            let doc = render(template, &mut ctx, &mut rng)?;
            let doc = wrapped(doc, &ctx, usize::MAX, &mut rng)?;
            println!(
                "--- {} #{} {}",
                template.family.name(),
                template.id,
                country
            );
            println!("{}\n", bracketed(&doc));
        }
        return Ok(());
    }

    let (rows, dropped) = generate(gen_cfg, &countries, &pools, &all, &mut rng)?;
    let out = PathBuf::from(&cfg.data.processed);
    std::fs::create_dir_all(&out).with_context(|| format!("creating {}", out.display()))?;
    for split in Split::ALL {
        let subset: Vec<&Row> = rows.iter().filter(|r| r.split == split).collect();
        write_split(&out.join(format!("{}.parquet", split.name())), &subset)?;
    }
    write_manifest(&cfg, gen_cfg, &rows, &dropped, &pools, &all)?;
    print_summary(&rows, &dropped);
    Ok(())
}

fn static_countries(codes: &[String]) -> anyhow::Result<Vec<&'static str>> {
    const KNOWN: &[&str] = &["US", "CA", "GB", "DE", "GE", "JP"];
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
    pool.iter()
        .filter(|t| t.category == category)
        .collect::<Vec<_>>()
        .choose(rng)
        .map(|t| **t)
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
    rng: &mut ChaCha8Rng,
) -> anyhow::Result<(Vec<Row>, Dropped)> {
    let mut rows = Vec::new();
    let mut dropped = Dropped::new();
    for (split, subset, count) in plan(cfg) {
        let pool = eligible(all, subset);
        for _ in 0..count {
            let mut attempts = 0;
            let doc = loop {
                attempts += 1;
                if attempts > 100 {
                    bail!("100 rendered documents in a row failed the checks; see `dropped`");
                }
                let template = choose_template(&pool, subset, rng)?;
                let country = *countries.choose(rng).context("no countries")?;
                let mut ctx = Ctx::new(country, &pools[&(split, country)]);
                let doc = wrapped(render(template, &mut ctx, rng)?, &ctx, cfg.max_tokens, rng)?;
                match check(&doc, cfg.max_tokens) {
                    Ok(()) => break doc,
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

/// Reads the names and address pools per (split, country). Organization pools below
/// `min_org_pool` borrow Latin-script names from the other countries' pools of the same
/// split, with their legal form swapped for a Georgian one; only GE has such forms, and a
/// short pool of any other country is used as it is.
fn load_pools(
    names: &Path,
    addresses: &Path,
    countries: &[&'static str],
    seed: u64,
) -> anyhow::Result<HashMap<(Split, &'static str), Pools>> {
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
        if not_a_person {
            continue;
        }
        if let Some(p) = pool_mut(&mut pools, &row[1], &row[3]) {
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
            p.orgs.push(PoolOrg {
                name,
                legal_form: row[2].clone(),
            });
        }
    }
    for split in Split::ALL {
        let path = addresses.join(format!("{}.parquet", split.name()));
        for e in read_shard(&path, split)? {
            if e.augmented
                || pool_filter::hydrant(&e.text)
                || (e.country == "GE" && pool_filter::unusable_ge_address(&e))
            {
                continue;
            }
            if !pool_filter::postal(&e) {
                continue;
            }
            if let Some(p) = pool_mut(&mut pools, &e.country, split.name()) {
                p.addresses.push(PoolAddress::from_example(&e));
            }
        }
    }
    for split in Split::ALL {
        for &c in countries {
            let native = pools[&(split, c)].orgs.len();
            let mut need = min_org_pool(split).saturating_sub(native);
            if need == 0 || c != "GE" {
                continue;
            }
            // At most as many borrowed names as native ones where the native pool allows it:
            // a Georgian org should mostly be a Georgian name.
            if split == Split::Train {
                need = need.min(native);
            }
            let mut donors: Vec<PoolOrg> = countries
                .iter()
                .filter(|&&d| d != c)
                .flat_map(|&d| pools[&(split, d)].orgs.iter().cloned())
                .filter(|o| {
                    o.name
                        .chars()
                        .filter(|ch| ch.is_alphabetic())
                        .all(|ch| ch.is_ascii())
                        && !pool_filter::private_arrangement(&o.name)
                        && !pool_filter::address_like(&o.name)
                        && !pool_filter::institution(&o.name)
                })
                // A donor keeping a second legal form would still read as a foreign company.
                .filter_map(|o| {
                    pool_filter::without_legal_form(&o.name).map(|base| PoolOrg {
                        name: base,
                        legal_form: String::new(),
                    })
                })
                .collect();
            let mut rng = ChaCha8Rng::seed_from_u64(seed ^ fnv1a(c.as_bytes(), 0) ^ split as u64);
            donors.shuffle(&mut rng);
            let borrowed: Vec<PoolOrg> = donors
                .into_iter()
                .take(need)
                .map(|o| {
                    let base = o.name;
                    let &(form, before) = pick(templates::GE_LEGAL_FORMS, &mut rng);
                    let name = if before {
                        format!("{form} {base}")
                    } else {
                        format!("{base} {form}")
                    };
                    PoolOrg {
                        name,
                        legal_form: form.to_string(),
                    }
                })
                .collect();
            let p = pools.get_mut(&(split, c)).context("pool")?;
            p.borrowed_orgs = borrowed.len();
            p.orgs.extend(borrowed);
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
                "borrowed_orgs": p.borrowed_orgs,
                "addresses": p.addresses.len(),
            }),
        );
    }
    let safe = rows.iter().filter(|r| r.doc.phones_fixture_safe).count();
    let manifest = serde_json::json!({
        "source": "tessera-generator",
        "version": 1,
        "seed": cfg.seed,
        "templates": all.len(),
        "max_tokens": gen_cfg.max_tokens,
        "names": cfg.names.as_ref().map(|n| n.out.clone()),
        "addresses": gen_cfg.addresses,
        "use": "detector training data only: documents contain real Wikidata and GLEIF names and never enter versioned fixtures",
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
    use std::collections::HashSet;

    use super::*;

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
                name: "Kavkaz Freight LLC".into(),
                legal_form: "LLC".into(),
            }],
            addresses: vec![
                PoolAddress {
                    text: "12 Rustaveli Avenue\n0108 Tbilisi\nGeorgia".into(),
                    city: Some("Tbilisi".into()),
                },
                PoolAddress {
                    text: "4 Misty Wood Circle".into(),
                    city: None,
                },
            ],
            borrowed_orgs: 0,
            bodies: Bodies::new("GE", Split::Train),
        }
    }

    fn template(text: &'static str) -> Template {
        Template {
            family: Family::Prose,
            id: 0,
            category: Category::Nothing,
            text,
        }
    }

    #[test]
    fn render_records_exact_offsets() {
        let pools = Pools {
            orgs: Vec::new(),
            ..stub_pools(&[("Nino Beridze", "latin")])
        };
        let mut ctx = Ctx::new("GE", &pools);
        ctx.plain = true;
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let doc = render(
            &template("Hi {person#1}, mail {email#1}."),
            &mut ctx,
            &mut rng,
        )
        .unwrap();
        assert!(doc.text.starts_with("Hi Nino Beridze, mail nino.beridze@"));
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
        assert_eq!(&doc.text[3..15], "Nino Beridze");
        assert_eq!(check(&doc, 900), Ok(()));
    }

    #[test]
    fn multiline_addresses_keep_their_lines_inside_the_span() {
        let pools = stub_pools(&[("Nino Beridze", "latin")]);
        let mut ctx = Ctx::new("GE", &pools);
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let doc = render(
            &template("Ship to:\n{address_ml}\nThanks"),
            &mut ctx,
            &mut rng,
        )
        .unwrap();
        let e = &doc.entities[0];
        assert_eq!(
            &doc.text[e.start..e.end],
            "12 Rustaveli Avenue\n0108 Tbilisi\nGeorgia"
        );
        assert_eq!(check(&doc, 900), Ok(()));
    }

    #[test]
    fn single_line_addresses_join_lines_with_commas() {
        let a = PoolAddress {
            text: "12 Rustaveli Avenue\n0108 Tbilisi\nGeorgia".into(),
            city: None,
        };
        assert_eq!(a.one_line(), "12 Rustaveli Avenue, 0108 Tbilisi, Georgia");
    }

    #[test]
    fn honorifics_stay_outside_the_person_span() {
        let pools = stub_pools(&[("Nino Beridze", "latin")]);
        let t = template("Dear {person}.");
        let mut seen = false;
        for seed in 0..50 {
            let mut ctx = Ctx::new("GE", &pools);
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let doc = render(&t, &mut ctx, &mut rng).unwrap();
            let e = &doc.entities[0];
            assert_eq!(doc.text[e.start..e.end].to_lowercase(), "nino beridze");
            seen |= e.start > "Dear ".len();
        }
        assert!(seen, "no honorific in 50 renders");
    }

    #[test]
    fn every_template_renders_for_every_country() {
        let pools = stub_pools(&[("Nino Beridze", "latin"), ("山田太郎", "han")]);
        let all = templates::all().unwrap();
        for country in ["US", "GB", "DE", "GE", "JP"] {
            for (i, t) in all.iter().enumerate() {
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

    #[test]
    fn linked_slots_reuse_their_values() {
        let pools = stub_pools(&[("Nino Beridze", "latin"), ("Anna Schmidt", "latin")]);
        let mut ctx = Ctx::new("GE", &pools);
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
    fn phone_generators_mark_only_reserved_ranges_safe() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        for _ in 0..200 {
            for c in ["GB", "US"] {
                assert!(phone(c, rng.random_bool(0.5), &mut rng).unwrap().1);
            }
            for c in ["GE", "JP"] {
                assert!(!phone(c, rng.random_bool(0.5), &mut rng).unwrap().1);
            }
            let (value, safe) = phone("DE", rng.random_bool(0.5), &mut rng).unwrap();
            assert_eq!(safe, value.contains("23125"), "{value}");
        }
    }

    #[test]
    fn safe_phones_are_found_by_the_rules_with_their_region() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let mut missed = Vec::new();
        for c in ["GB", "US", "CA", "DE"] {
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
            pools.insert((split, "GB"), stub_pools(&[(name, "latin")]));
        }
        let cfg = GenerateConfig {
            train: 200,
            valid: 50,
            test_seen: 20,
            test_heldout_templates: 15,
            test_heldout_families: 15,
            addresses: String::new(),
            max_tokens: 900,
        };
        let all = templates::all().unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let (rows, _) = generate(&cfg, &["GB"], &pools, &all, &mut rng).unwrap();
        assert_eq!(rows.len(), 300);
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
            strip_legal_form("Kavkaz Freight LLC", "LLC").as_deref(),
            Some("Kavkaz Freight")
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
    fn email_parts_are_ascii() {
        assert_eq!(
            ascii_words("José García-Pérez"),
            vec!["jose", "garcia", "perez"]
        );
        assert_eq!(slug("Kavkaz Freight LLC"), "kavkaz-freight-llc");
    }
}
