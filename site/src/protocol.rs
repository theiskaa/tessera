//! Messages between the page and the inference worker. The library's types have no serde
//! derives, so the parts the page renders are mirrored here. Offsets are UTF-8 bytes.

use serde::{Deserialize, Serialize};

/// What the page asks the worker.
#[derive(Debug, Serialize, Deserialize)]
pub enum Request {
    /// Detect emails and phones in `text`, and parse each address span.
    Analyze {
        /// Echoed in the answer, so the page can drop answers to text it no longer shows.
        id: u64,
        /// The document.
        text: String,
        /// Regions for phone numbers written without a country code.
        country_hint: Vec<String>,
        /// Byte ranges of `text` known to hold one address each.
        addresses: Vec<(usize, usize)>,
    },
    /// Split `text`, known to be one address, into its parts.
    ParseAddress {
        /// Echoed in the answer, so the page can drop answers to text it no longer shows.
        id: u64,
        /// The address.
        text: String,
    },
}

/// What the worker answers.
#[derive(Debug, Serialize, Deserialize)]
pub enum Response {
    /// The bundle could not be fetched, verified, or loaded, so no request can be served.
    LoadFailed(String),
    /// The answer to `Request::Analyze` with the same `id`.
    Analyzed {
        /// The request's `id`.
        id: u64,
        /// Emails, phones, and parsed addresses, in document order, or the library's error.
        result: Result<Vec<Found>, String>,
    },
    /// The answer to `Request::ParseAddress` with the same `id`.
    Parsed {
        /// The request's `id`.
        id: u64,
        /// The parsed address, or the parser's error as its variant name and message.
        result: Result<Found, String>,
    },
}

/// The kinds the page shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FoundKind {
    /// Found by the email rules.
    Email,
    /// Found by the phone rules.
    Phone,
    /// A given address span, split by the parser.
    Address,
}

/// One entity the library returned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Found {
    /// What it is.
    pub kind: FoundKind,
    /// Byte offset into the document where the span begins.
    pub start: usize,
    /// Byte offset where the span ends, exclusive.
    pub end: usize,
    /// The library's confidence, from 0 to 1.
    pub confidence: f32,
    /// Medium confidence: a person should look before acting on it.
    pub review_recommended: bool,
    /// Which stage found it, `rules` or `model`, as the library prints it.
    pub source: String,
    /// E.164 for phones; for emails the address with its domain lowercased.
    pub normalized: Option<String>,
    /// Phone only, ISO 3166-1 alpha-2.
    pub region: Option<String>,
    /// Address only: the parser's components.
    pub components: Vec<Part>,
}

/// One component of an address.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Part {
    /// Snake-case label, as the library prints it.
    pub label: String,
    /// Byte offset into the document where the component begins.
    pub start: usize,
    /// Byte offset where it ends, exclusive.
    pub end: usize,
    /// The parser's confidence in this component.
    pub confidence: f32,
}

impl Found {
    /// Mirrors `entity`, whose offsets are relative to a span starting `offset` bytes into the
    /// document. `None` for kinds other than the ones the page shows.
    pub fn from_entity(entity: &tessera::Entity, offset: usize) -> Option<Found> {
        let kind = match entity.kind {
            tessera::Kind::Email => FoundKind::Email,
            tessera::Kind::Phone => FoundKind::Phone,
            tessera::Kind::Address => FoundKind::Address,
            tessera::Kind::Person | tessera::Kind::Org => return None,
        };
        Some(Found {
            kind,
            start: offset + entity.start,
            end: offset + entity.end,
            confidence: entity.confidence,
            review_recommended: entity.review_recommended,
            source: entity.source.as_str().to_string(),
            normalized: entity.normalized.clone(),
            region: entity.region.clone(),
            components: entity
                .components
                .iter()
                .map(|c| Part {
                    label: c.label.as_str().to_string(),
                    start: offset + c.start,
                    end: offset + c.end,
                    confidence: c.confidence,
                })
                .collect(),
        })
    }
}
