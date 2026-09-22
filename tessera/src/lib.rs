//! Find the people, organizations, addresses, emails, and phone numbers inside
//! text, assemble them into contacts, and return exact source spans. Nothing
//! leaves the process.
//!
//! Offsets are UTF-8 byte offsets into the source string and `end` is
//! exclusive. The `wasm` feature converts them to UTF-16 code units at the
//! binding boundary.

mod features;
mod policy;
mod rules;
mod token;
#[cfg(feature = "wasm")]
pub mod wasm;

/// Internals shared with the trainer and the integration tests. Not a stable API.
#[doc(hidden)]
pub mod internal {
    pub use crate::features::{
        FeatureConfig, TokenFeatures, featurize, flag, is_content, line_ranges,
    };
    pub use crate::policy::display_confidence;
    pub use crate::rules::email::scan as scan_email;
    pub use crate::rules::phone::scan as scan_phone;
    pub use crate::rules::scan as scan_rules;
    pub use crate::token::{Script, Token, TokenClass, tokenize, utf16_offsets};
}

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

    /// Lowercase label used by fixtures, the CLI, the bindings, and the bundle manifest.
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Person => "person",
            Kind::Org => "org",
            Kind::Address => "address",
            Kind::Email => "email",
            Kind::Phone => "phone",
        }
    }

    /// Inverse of [`Kind::as_str`].
    pub fn from_str_label(s: &str) -> Option<Kind> {
        Kind::ALL.iter().copied().find(|k| k.as_str() == s)
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

impl Source {
    /// Both sources, in declaration order.
    pub const ALL: [Source; 2] = [Source::Model, Source::Rules];

    /// Lowercase label used by fixtures, the CLI, and the bindings.
    pub fn as_str(self) -> &'static str {
        match self {
            Source::Model => "model",
            Source::Rules => "rules",
        }
    }

    /// Inverse of [`Source::as_str`].
    pub fn from_str_label(s: &str) -> Option<Source> {
        Source::ALL.iter().copied().find(|v| v.as_str() == s)
    }
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

impl AddressLabel {
    /// Every label, in taxonomy order.
    pub const ALL: [AddressLabel; 12] = [
        AddressLabel::HouseNumber,
        AddressLabel::Road,
        AddressLabel::Unit,
        AddressLabel::Level,
        AddressLabel::Suburb,
        AddressLabel::City,
        AddressLabel::District,
        AddressLabel::Region,
        AddressLabel::Postcode,
        AddressLabel::Country,
        AddressLabel::PoBox,
        AddressLabel::Unknown,
    ];

    /// Snake-case label used by fixtures, the CLI, the bindings, and the bundle manifest.
    pub fn as_str(self) -> &'static str {
        match self {
            AddressLabel::HouseNumber => "house_number",
            AddressLabel::Road => "road",
            AddressLabel::Unit => "unit",
            AddressLabel::Level => "level",
            AddressLabel::Suburb => "suburb",
            AddressLabel::City => "city",
            AddressLabel::District => "district",
            AddressLabel::Region => "region",
            AddressLabel::Postcode => "postcode",
            AddressLabel::Country => "country",
            AddressLabel::PoBox => "po_box",
            AddressLabel::Unknown => "unknown",
        }
    }

    /// Inverse of [`AddressLabel::as_str`].
    pub fn from_str_label(s: &str) -> Option<AddressLabel> {
        AddressLabel::ALL.iter().copied().find(|v| v.as_str() == s)
    }
}

/// One labelled component of an address span.
#[derive(Debug, Clone, PartialEq)]
pub struct Component {
    /// Which part of the address this is.
    pub label: AddressLabel,
    /// UTF-8 byte offset where the span begins.
    pub start: usize,
    /// UTF-8 byte offset where the span ends, exclusive.
    pub end: usize,
    /// Model confidence in this component, from 0 to 1.
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
    /// What was found.
    pub kind: Kind,
    /// UTF-8 byte offset where the span begins.
    pub start: usize,
    /// UTF-8 byte offset where the span ends, exclusive.
    pub end: usize,
    /// Confidence from 0 to 1; see `review_recommended` for the band it falls in.
    pub confidence: f32,
    /// Medium confidence: return it, but a person should look before acting on it.
    pub review_recommended: bool,
    /// Whether a model or the rules layer found it.
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
    /// UTF-8 byte offset of the first byte of any field of the contact.
    pub start: usize,
    /// UTF-8 byte offset after the last field of the contact, exclusive.
    pub end: usize,
    /// Bounded by the weakest assignment, never averaged.
    pub confidence: f32,
    /// Medium confidence: return it, but a person should look before acting on it.
    pub review_recommended: bool,
    /// The person the contact is anchored on, if any.
    pub person: Option<Entity>,
    /// The organization, as anchor when there is no person, or as the person's employer.
    pub org: Option<Entity>,
    /// Addresses assigned to this contact.
    pub addresses: Vec<Entity>,
    /// Email addresses assigned to this contact.
    pub emails: Vec<Entity>,
    /// Phone numbers assigned to this contact.
    pub phones: Vec<Entity>,
}

/// Result of [`Tessera::extract_contacts`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Extraction {
    /// Contacts in document order.
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
    /// Parse and verify a weight bundle.
    ///
    /// A rules-only configuration (`config.kinds.is_rules_only()`) needs no
    /// bundle: `bundle` and `expected_checksum` are ignored and may be empty.
    pub fn load(bundle: &[u8], config: Config<'_>) -> Result<Tessera, Error> {
        if config.kinds.is_rules_only() {
            return Ok(Tessera {
                kinds: config.kinds,
            });
        }
        // No bundle format exists before the parser ships; any bytes are invalid.
        let _ = bundle;
        Err(Error::BundleInvalid)
    }

    /// Every supported entity in `text`, as non-overlapping spans sorted by position.
    pub fn detect(&self, text: &str, query: &Query<'_>) -> Result<Vec<Entity>, Error> {
        if !self.kinds.is_rules_only() {
            return Err(Error::Inference {
                stage: policy::STAGE_DETECT,
            });
        }
        let found = rules::scan(text, query.country_hint)
            .into_iter()
            .filter(|e| self.kinds.contains(e.kind))
            .collect();
        Ok(policy::apply(found, query.include_uncertain))
    }

    /// Entities grouped into contacts, plus everything that could not be assigned.
    pub fn extract_contacts(&self, text: &str, query: &Query<'_>) -> Result<Extraction, Error> {
        let _ = (text, query);
        Err(Error::Inference {
            stage: policy::STAGE_GROUP,
        })
    }

    /// Split text already known to be an address into components. Offsets are relative to `text`.
    pub fn parse_address(&self, text: &str, query: &Query<'_>) -> Result<Entity, Error> {
        let _ = (text, query);
        Err(Error::Inference {
            stage: policy::STAGE_PARSE,
        })
    }

    /// The kinds this instance was loaded for.
    pub fn kinds(&self) -> KindSet {
        self.kinds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels_round_trip() {
        for k in Kind::ALL {
            assert_eq!(Kind::from_str_label(k.as_str()), Some(k));
        }
        for l in AddressLabel::ALL {
            assert_eq!(AddressLabel::from_str_label(l.as_str()), Some(l));
        }
        assert_eq!(Source::from_str_label("rules"), Some(Source::Rules));
        assert_eq!(Kind::from_str_label("Person"), None);
    }
}
