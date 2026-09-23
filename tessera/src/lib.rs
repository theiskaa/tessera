//! Find the people, organizations, addresses, emails, and phone numbers inside
//! text, assemble them into contacts, and return exact source spans. Nothing
//! leaves the process.
//!
//! Offsets are UTF-8 byte offsets into the source string and `end` is
//! exclusive. The `wasm` feature converts them to UTF-16 code units at the
//! binding boundary.

mod features;
mod model;
mod policy;
mod rules;
mod token;
#[cfg(feature = "wasm")]
pub mod wasm;

/// Internals shared with the trainer and the integration tests. Not a stable API.
#[doc(hidden)]
pub mod internal {
    pub use crate::features::{
        FeatureConfig, MAX_NGRAMS_PER_TOKEN, TokenFeatures, featurize, flag, fnv1a, is_content,
        line_ranges,
    };
    pub use crate::model::bio::parser_label_strings;
    pub use crate::model::{FLAG_BITS, PARSER_LABELS, SCRIPT_ROWS, SHAPE_ROWS};
    pub use crate::policy::display_confidence;
    pub use crate::rules::email::scan as scan_email;
    pub use crate::rules::phone::scan as scan_phone;
    pub use crate::rules::scan as scan_rules;
    pub use crate::token::{Script, Token, TokenClass, tokenize, utf16_offsets};

    /// The intermediates of one `parse_address` call, for the golden-vector gate.
    #[derive(Debug, Clone)]
    pub struct ParseTrace {
        /// Byte spans of the retained (non-whitespace) tokens.
        pub token_spans: Vec<(usize, usize)>,
        /// Features of the retained tokens.
        pub features: Vec<TokenFeatures>,
        /// Logits `[tokens, labels]`, row-major, before softmax.
        pub logits: Vec<f32>,
        /// Decoded label id per retained token.
        pub decoded: Vec<u8>,
        /// Probability of each decoded label.
        pub probabilities: Vec<f32>,
    }

    /// `parse_address` up to decoding, keeping every intermediate.
    pub fn parse_address_trace(
        tessera: &crate::Tessera,
        text: &str,
    ) -> Result<ParseTrace, crate::Error> {
        tessera.trace(text)
    }
}

use std::ops::BitOr;

/// Entity kinds the library can detect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    /// A person's name.
    Person,
    /// An organization's name.
    Org,
    /// A postal address.
    Address,
    /// An email address.
    Email,
    /// A phone number.
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
    /// A trained network.
    Model,
    /// The deterministic rules layer.
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
    /// House or building number, `221B`.
    HouseNumber,
    /// Street name with its type, `Baker Street`.
    Road,
    /// Flat, apartment, or unit, `Flat 4`.
    Unit,
    /// Floor, `3rd Floor`, `2. OG`.
    Level,
    /// Neighbourhood within a city.
    Suburb,
    /// City, town, or village.
    City,
    /// County or city district.
    District,
    /// State, province, or constituent country.
    Region,
    /// Postal code.
    Postcode,
    /// Country name.
    Country,
    /// Post office box.
    PoBox,
    /// A span the model found but could not label with confidence.
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
    /// The component's text, sliced from the source the offsets refer to. Panics when given
    /// any other string whose char boundaries differ.
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
    /// E.164 for phones; for emails the address with its domain lowercased. Never a rewritten
    /// name or address.
    pub normalized: Option<String>,
    /// Phone only, ISO 3166-1 alpha-2.
    pub region: Option<String>,
}

impl Entity {
    /// The entity's text, sliced from the source the offsets refer to. Panics when given any
    /// other string whose char boundaries differ.
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
    /// The bundle is malformed, or lacks a network the requested kinds need.
    #[error("weight bundle is invalid or lacks a network for the requested kinds")]
    BundleInvalid,
    /// The bundle's bytes do not match `Config::expected_checksum`.
    #[error("weight bundle checksum does not match")]
    ChecksumMismatch,
    /// The bundle is well formed but was built for another format or library version.
    #[error("weight bundle was built for a different library version")]
    UnsupportedVersion,
    /// The input is longer than the operation accepts.
    #[error("input exceeds the supported size")]
    InputTooLarge,
    /// An operation ran that this instance cannot serve, or an internal invariant broke.
    #[error("inference failed in stage `{stage}`")]
    Inference {
        /// The pipeline stage that failed.
        stage: &'static str,
    },
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
    /// Default regions for phone numbers written without a country code. `parse_address`
    /// ignores it.
    pub country_hint: &'a [&'a str],
    /// Return low-confidence results instead of omitting them.
    pub include_uncertain: bool,
}

/// A loaded extractor. Immutable after load; one instance serves many calls and threads.
#[derive(Debug)]
pub struct Tessera {
    kinds: KindSet,
    model: Option<Model>,
}

/// What the bundle contributes once loaded: the networks and the settings their inputs need.
#[derive(Debug)]
struct Model {
    parser: Option<model::Parser>,
    feature_config: features::FeatureConfig,
    version: String,
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
                model: None,
            });
        }
        let bundle = model::weights::Bundle::parse(bundle, config.expected_checksum)?;
        let needs_detector = [Kind::Person, Kind::Org]
            .iter()
            .any(|&k| config.kinds.contains(k));
        if needs_detector && !bundle.has_net("detector") {
            return Err(Error::BundleInvalid);
        }
        let parser = if config.kinds.contains(Kind::Address) {
            if !bundle.has_net("parser") {
                return Err(Error::BundleInvalid);
            }
            Some(model::Parser::new(&bundle)?)
        } else {
            None
        };
        Ok(Tessera {
            kinds: config.kinds,
            model: Some(Model {
                parser,
                feature_config: bundle.manifest.feature_config,
                version: bundle.manifest.model_version,
            }),
        })
    }

    /// Every supported entity in `text`, as non-overlapping spans sorted by position.
    ///
    /// Until the detector ships, only a rules-only instance (emails and phones) can detect;
    /// an instance loaded with a model kind returns `Error::Inference` here and serves
    /// `parse_address` instead.
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
    ///
    /// The input must be a single address of at most 256 non-whitespace tokens and 8 KiB;
    /// longer input is `InputTooLarge`. Empty input gives an entity with no components and
    /// confidence 0. Of `query`, only `include_uncertain` applies.
    pub fn parse_address(&self, text: &str, query: &Query<'_>) -> Result<Entity, Error> {
        let trace = self.trace(text)?;
        let found = model::bio::components(
            &trace.token_spans,
            &trace
                .decoded
                .iter()
                .copied()
                .zip(trace.probabilities.iter().copied())
                .collect::<Vec<_>>(),
        );
        let components = policy::address_components(
            policy::merge_touching_suburbs(policy::trim_separators(text, found)),
            query.include_uncertain,
        );
        let confidence = policy::address_confidence(&components);
        Ok(Entity {
            kind: Kind::Address,
            start: 0,
            end: text.len(),
            confidence,
            review_recommended: confidence < policy::HIGH,
            source: Source::Model,
            components,
            normalized: None,
            region: None,
        })
    }

    /// `parse_address` up to decoding, keeping every intermediate.
    fn trace(&self, text: &str) -> Result<internal::ParseTrace, Error> {
        let stage = Error::Inference {
            stage: policy::STAGE_PARSE,
        };
        let model = self.model.as_ref().ok_or(stage.clone())?;
        let parser = model.parser.as_ref().ok_or(stage)?;
        if text.len() > model::MAX_PARSE_BYTES {
            return Err(Error::InputTooLarge);
        }
        let tokens = token::tokenize(text);
        let retained = |t: &token::Token| {
            !matches!(
                t.class,
                token::TokenClass::Space | token::TokenClass::Newline
            )
        };
        if tokens.iter().filter(|t| retained(t)).count() > model::MAX_PARSE_TOKENS {
            return Err(Error::InputTooLarge);
        }
        // The trainer featurizes addresses without a country, so the library must too.
        let feats = features::featurize(text, &tokens, &[], None, &model.feature_config);
        let (token_spans, features): (Vec<_>, Vec<_>) = tokens
            .iter()
            .zip(feats)
            .filter(|(t, _)| retained(t))
            .map(|(t, f)| ((t.start, t.end), f))
            .unzip();
        let logits = parser.forward(&features);
        let mut probs = logits.clone();
        model::kernels::softmax_rows(&mut probs, model::PARSER_LABELS);
        let (decoded, probabilities) = model::bio::decode_probs(&probs, model::PARSER_LABELS)
            .into_iter()
            .unzip();
        Ok(internal::ParseTrace {
            token_spans,
            features,
            logits,
            decoded,
            probabilities,
        })
    }

    /// The loaded bundle's model version, or `None` for a rules-only instance.
    pub fn model_version(&self) -> Option<&str> {
        self.model.as_ref().map(|m| m.version.as_str())
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
