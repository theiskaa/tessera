//! Find the people, organizations, addresses, emails, and phone numbers inside
//! text, assemble them into contacts, and return exact source spans. Nothing
//! leaves the process.
//!
//! Offsets are UTF-8 byte offsets into the source string and `end` is
//! exclusive. The `wasm` feature converts them to UTF-16 code units at the
//! binding boundary.

use std::ops::BitOr;

/// Entity kinds the library can detect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    Person,
    Org,
    Address,
    Email,
    Phone,
}

impl Kind {
    /// Every kind, in taxonomy order.
    pub const ALL: [Kind; 5] = [
        Kind::Person,
        Kind::Org,
        Kind::Address,
        Kind::Email,
        Kind::Phone,
    ];

    /// The set containing every kind.
    pub fn all() -> KindSet {
        Kind::ALL
            .iter()
            .fold(KindSet::EMPTY, |set, kind| set | *kind)
    }

    const fn bit(self) -> u8 {
        1 << self as u8
    }
}

/// A set of [`Kind`]s, built with `|`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KindSet(u8);

impl KindSet {
    /// The empty set.
    pub const EMPTY: KindSet = KindSet(0);

    /// Whether `kind` is in the set.
    pub fn contains(self, kind: Kind) -> bool {
        self.0 & kind.bit() != 0
    }

    /// Whether the set needs no weight bundle, that is, contains only rule-found kinds.
    pub fn is_rules_only(self) -> bool {
        self.0 & !(Kind::Email.bit() | Kind::Phone.bit()) == 0
    }
}

impl BitOr for KindSet {
    type Output = KindSet;
    fn bitor(self, rhs: KindSet) -> KindSet {
        KindSet(self.0 | rhs.0)
    }
}

impl BitOr<Kind> for KindSet {
    type Output = KindSet;
    fn bitor(self, rhs: Kind) -> KindSet {
        KindSet(self.0 | rhs.bit())
    }
}

impl BitOr for Kind {
    type Output = KindSet;
    fn bitor(self, rhs: Kind) -> KindSet {
        KindSet(self.bit() | rhs.bit())
    }
}

impl From<Kind> for KindSet {
    fn from(kind: Kind) -> KindSet {
        KindSet(kind.bit())
    }
}

/// Which stage produced an entity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Model,
    Rules,
}

/// Address component labels, in taxonomy order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AddressLabel {
    HouseNumber,
    Road,
    Unit,
    Level,
    Suburb,
    City,
    District,
    Region,
    Postcode,
    Country,
    PoBox,
    Unknown,
}

/// One labelled component of an address span.
#[derive(Debug, Clone, PartialEq)]
pub struct Component {
    pub label: AddressLabel,
    pub start: usize,
    pub end: usize,
    pub confidence: f32,
}

impl Component {
    /// The component's text, sliced from the source the offsets refer to.
    pub fn text<'a>(&self, source: &'a str) -> &'a str {
        &source[self.start..self.end]
    }
}

/// One detected span.
#[derive(Debug, Clone, PartialEq)]
pub struct Entity {
    pub kind: Kind,
    pub start: usize,
    pub end: usize,
    pub confidence: f32,
    pub source: Source,
    /// Address only; empty for every other kind.
    pub components: Vec<Component>,
    /// E.164 for phones, lowercased domain for emails; never a rewritten name or address.
    pub normalized: Option<String>,
    /// Phone only, ISO 3166-1 alpha-2.
    pub region: Option<String>,
}

impl Entity {
    /// The entity's text, sliced from the source the offsets refer to.
    pub fn text<'a>(&self, source: &'a str) -> &'a str {
        &source[self.start..self.end]
    }
}

/// A person or organization with the details that belong to them.
#[derive(Debug, Clone, PartialEq)]
pub struct Contact {
    pub start: usize,
    pub end: usize,
    /// Bounded by the weakest assignment, never averaged.
    pub confidence: f32,
    pub person: Option<Entity>,
    pub org: Option<Entity>,
    pub addresses: Vec<Entity>,
    pub emails: Vec<Entity>,
    pub phones: Vec<Entity>,
}

/// Result of [`Tessera::extract_contacts`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Extraction {
    pub contacts: Vec<Contact>,
    /// Entities that could not be assigned to a contact with confidence.
    pub unassigned: Vec<Entity>,
}

/// Failures are always typed; an empty successful result means inference ran and found nothing.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("weight bundle is invalid")]
    BundleInvalid,
    #[error("weight bundle checksum does not match")]
    ChecksumMismatch,
    #[error("weight bundle is newer than this library")]
    UnsupportedVersion,
    #[error("input exceeds the supported size")]
    InputTooLarge,
    #[error("inference failed in stage `{stage}`")]
    Inference { stage: &'static str },
}

/// Load-time configuration.
#[derive(Debug, Clone)]
pub struct Config<'a> {
    /// Kinds to detect; stages for other kinds are skipped.
    pub kinds: KindSet,
    /// `sha256-…` digest the bundle must match before initialization.
    pub expected_checksum: Option<&'a str>,
}

/// Per-call options.
#[derive(Debug, Clone, Default)]
pub struct Query<'a> {
    /// Default regions for phone numbers written without a country code.
    pub country_hint: &'a [&'a str],
    /// Return low-confidence results instead of omitting them.
    pub include_uncertain: bool,
}

/// A loaded extractor. Immutable after load; one instance serves many calls and threads.
#[derive(Debug)]
pub struct Tessera {
    kinds: KindSet,
}

impl Tessera {
    /// Parse and verify a weight bundle. Pass an empty slice for a rules-only configuration.
    pub fn load(bundle: &[u8], config: Config<'_>) -> Result<Tessera, Error> {
        let _ = (bundle, config);
        todo!("bundle loading")
    }

    /// Every supported entity in `text`, as non-overlapping spans sorted by position.
    pub fn detect(&self, text: &str, query: &Query<'_>) -> Result<Vec<Entity>, Error> {
        let _ = (text, query);
        todo!("detect")
    }

    /// Entities grouped into contacts, plus everything that could not be assigned.
    pub fn extract_contacts(&self, text: &str, query: &Query<'_>) -> Result<Extraction, Error> {
        let _ = (text, query);
        todo!("extract_contacts")
    }

    /// Split text already known to be an address into components. Offsets are relative to `text`.
    pub fn parse_address(&self, text: &str, query: &Query<'_>) -> Result<Entity, Error> {
        let _ = (text, query);
        todo!("parse_address")
    }

    /// The kinds this instance was loaded for.
    pub fn kinds(&self) -> KindSet {
        self.kinds
    }
}
