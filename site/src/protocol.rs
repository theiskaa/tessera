//! Messages between the page and the inference worker. The library's types have no serde
//! derives, so the parts the page renders are mirrored here. Offsets are UTF-8 bytes.

use serde::{Deserialize, Serialize};

/// What the page asks the worker.
#[derive(Debug, Serialize, Deserialize)]
pub enum Request {
    /// Detect every kind in `text` and group the entities into contacts.
    Analyze {
        /// Echoed in the answer, so the page can drop answers to text it no longer shows.
        id: u64,
        /// The document.
        text: String,
        /// Regions for phone numbers written without a country code.
        country_hint: Vec<String>,
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
        /// What `extract_contacts` returned, or the library's error.
        result: Result<Analysis, String>,
    },
    /// The answer to `Request::ParseAddress` with the same `id`.
    Parsed {
        /// The request's `id`.
        id: u64,
        /// The parsed address, or the parser's error as its variant name and message.
        result: Result<Found, String>,
    },
}

/// One document's result: every entity `detect` finds, in document order, the contacts
/// `extract_contacts` builds from them, and what no contact took, as indices into `found`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Analysis {
    pub found: Vec<Found>,
    pub contacts: Vec<Card>,
    pub unassigned: Vec<usize>,
}

/// One contact: its members as indices into `Analysis::found`, in the order
/// `Contact::entities` gives them (the person, the organization, then addresses, emails, and
/// phones).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Card {
    pub members: Vec<usize>,
    /// The weakest assignment's confidence.
    pub confidence: f32,
    pub review_recommended: bool,
}

impl Analysis {
    /// Flattens an extraction: every entity once, sorted by position, with each contact's
    /// members and the unassigned entities as positions in that list.
    pub fn from_extraction(x: &tessera::Extraction) -> Analysis {
        let mut entities: Vec<&tessera::Entity> = x
            .contacts
            .iter()
            .flat_map(|c| c.entities())
            .chain(&x.unassigned)
            .collect();
        entities.sort_by_key(|e| (e.start, e.end));
        let position = |e: &tessera::Entity| {
            entities
                .binary_search_by_key(&(e.start, e.end), |f| (f.start, f.end))
                .ok()
        };
        let contacts = x
            .contacts
            .iter()
            .map(|c| Card {
                members: c.entities().filter_map(position).collect(),
                confidence: c.confidence,
                review_recommended: c.review_recommended,
            })
            .collect();
        let unassigned = x.unassigned.iter().filter_map(position).collect();
        Analysis {
            found: entities.iter().map(|e| Found::from_entity(e)).collect(),
            contacts,
            unassigned,
        }
    }
}

/// The kinds the page shows, mirroring `tessera::Kind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FoundKind {
    /// Found by the detector.
    Person,
    /// Found by the detector.
    Org,
    /// Found by the detector and split by the parser.
    Address,
    /// Found by the email rules.
    Email,
    /// Found by the phone rules.
    Phone,
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
    /// The same entity found in a slice that starts at byte `by` of the document, with its
    /// offsets moved into the document.
    pub fn shifted(mut self, by: usize) -> Found {
        self.start += by;
        self.end += by;
        for part in &mut self.components {
            part.start += by;
            part.end += by;
        }
        self
    }

    /// Mirrors `entity`, whose offsets are into the document.
    pub fn from_entity(entity: &tessera::Entity) -> Found {
        let kind = match entity.kind {
            tessera::Kind::Person => FoundKind::Person,
            tessera::Kind::Org => FoundKind::Org,
            tessera::Kind::Address => FoundKind::Address,
            tessera::Kind::Email => FoundKind::Email,
            tessera::Kind::Phone => FoundKind::Phone,
        };
        Found {
            kind,
            start: entity.start,
            end: entity.end,
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
                    start: c.start,
                    end: c.end,
                    confidence: c.confidence,
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tessera::{Contact, Entity, Extraction, Kind, Source};

    fn entity(kind: Kind, start: usize) -> Entity {
        Entity {
            kind,
            start,
            end: start + 3,
            confidence: 0.9,
            review_recommended: false,
            source: Source::Model,
            components: Vec::new(),
            normalized: None,
            region: None,
        }
    }

    #[test]
    fn an_extraction_flattens_into_sorted_entities_and_member_indices() {
        let x = Extraction {
            contacts: vec![Contact {
                start: 10,
                end: 43,
                confidence: 0.7,
                review_recommended: true,
                person: Some(entity(Kind::Person, 20)),
                org: None,
                addresses: Vec::new(),
                emails: vec![entity(Kind::Email, 40)],
                phones: vec![entity(Kind::Phone, 10)],
            }],
            unassigned: vec![entity(Kind::Phone, 0)],
        };
        let a = Analysis::from_extraction(&x);
        let starts: Vec<usize> = a.found.iter().map(|f| f.start).collect();
        assert_eq!(starts, vec![0, 10, 20, 40]);
        assert_eq!(a.contacts.len(), 1);
        // The anchor first, then the email, then the phone, whatever their order in the text.
        assert_eq!(a.contacts[0].members, vec![2, 3, 1]);
        assert_eq!(a.unassigned, vec![0]);
        assert!(a.contacts[0].review_recommended);
    }
}
