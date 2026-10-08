//! The JSON shapes the CLI prints: entities, contacts, and their offsets.

use serde::Deserialize;
use serde_json::{Map, Value, json};
use tessera::internal::display_confidence;
use tessera::{Contact, Entity, Extraction};

/// The unit `start` and `end` count.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Offsets {
    /// UTF-8 bytes, as the library reports them.
    #[default]
    Utf8,
    /// UTF-16 code units, as JavaScript strings index.
    Utf16,
}

/// The source text results are rendered against, with byte offsets converted to the requested
/// unit.
pub(crate) struct Document<'a> {
    text: &'a str,
    /// `(byte, unit)` pairs sorted by byte for the offsets the results carry; `None` for UTF-8.
    units: Option<Vec<(usize, usize)>>,
}

impl<'a> Document<'a> {
    /// `needed` lists the byte offsets that will be rendered, so a UTF-16 conversion walks
    /// `text` once instead of once per offset.
    pub(crate) fn new(
        text: &'a str,
        offsets: Offsets,
        needed: impl IntoIterator<Item = usize>,
    ) -> Self {
        let units = match offsets {
            Offsets::Utf8 => None,
            Offsets::Utf16 => Some(utf16_units(text, needed)),
        };
        Document { text, units }
    }

    fn offset(&self, byte: usize) -> usize {
        let Some(units) = &self.units else {
            return byte;
        };
        match units.binary_search_by_key(&byte, |&(b, _)| b) {
            Ok(i) => units[i].1,
            Err(_) => self
                .text
                .get(..byte)
                .map_or(0, |prefix| prefix.encode_utf16().count()),
        }
    }

    /// One entity, with its address components when it has any.
    pub(crate) fn entity(&self, e: &Entity) -> Value {
        let mut m = Map::new();
        m.insert("kind".into(), e.kind.as_str().into());
        m.insert("text".into(), e.text(self.text).into());
        m.insert("start".into(), self.offset(e.start).into());
        m.insert("end".into(), self.offset(e.end).into());
        m.insert(
            "confidence".into(),
            Value::from(display_confidence(e.confidence)),
        );
        m.insert("review_recommended".into(), e.review_recommended.into());
        m.insert("source".into(), e.source.as_str().into());
        if let Some(n) = &e.normalized {
            m.insert("normalized".into(), n.as_str().into());
        }
        if let Some(r) = &e.region {
            m.insert("region".into(), r.as_str().into());
        }
        if !e.components.is_empty() {
            let comps: Vec<Value> = e
                .components
                .iter()
                .map(|c| {
                    json!({
                        "label": c.label.as_str(),
                        "text": c.text(self.text),
                        "start": self.offset(c.start),
                        "end": self.offset(c.end),
                        "confidence": display_confidence(c.confidence),
                    })
                })
                .collect();
            m.insert("components".into(), comps.into());
        }
        Value::Object(m)
    }

    /// The entities as a JSON array, in their given order.
    pub(crate) fn entities(&self, entities: &[Entity]) -> Value {
        entities.iter().map(|e| self.entity(e)).collect()
    }

    /// One contact. `person` and `org` are absent rather than null when the contact has none.
    pub(crate) fn contact(&self, c: &Contact) -> Value {
        let mut m = Map::new();
        m.insert("start".into(), self.offset(c.start).into());
        m.insert("end".into(), self.offset(c.end).into());
        m.insert(
            "confidence".into(),
            Value::from(display_confidence(c.confidence)),
        );
        m.insert("review_recommended".into(), c.review_recommended.into());
        if let Some(p) = &c.person {
            m.insert("person".into(), self.entity(p));
        }
        if let Some(o) = &c.org {
            m.insert("org".into(), self.entity(o));
        }
        m.insert("addresses".into(), self.entities(&c.addresses));
        m.insert("emails".into(), self.entities(&c.emails));
        m.insert("phones".into(), self.entities(&c.phones));
        Value::Object(m)
    }

    /// The contacts and the unassigned entities, as two JSON arrays.
    pub(crate) fn extraction(&self, x: &Extraction) -> (Value, Value) {
        let contacts = x.contacts.iter().map(|c| self.contact(c)).collect();
        (contacts, self.entities(&x.unassigned))
    }
}

/// Every byte offset `Document::entity` renders for `e`.
pub(crate) fn entity_offsets(e: &Entity) -> impl Iterator<Item = usize> + '_ {
    [e.start, e.end]
        .into_iter()
        .chain(e.components.iter().flat_map(|c| [c.start, c.end]))
}

/// Every byte offset `Document::extraction` renders for `x`.
pub(crate) fn extraction_offsets(x: &Extraction) -> impl Iterator<Item = usize> + '_ {
    x.contacts
        .iter()
        .flat_map(|c| {
            [c.start, c.end]
                .into_iter()
                .chain(c.entities().flat_map(entity_offsets))
        })
        .chain(x.unassigned.iter().flat_map(entity_offsets))
}

/// `(byte, UTF-16 units before it)` for each wanted byte offset, sorted and deduplicated.
fn utf16_units(text: &str, wanted: impl IntoIterator<Item = usize>) -> Vec<(usize, usize)> {
    let mut wanted: Vec<usize> = wanted.into_iter().collect();
    wanted.sort_unstable();
    wanted.dedup();
    let mut out = Vec::with_capacity(wanted.len());
    let (mut units, mut at, mut chars) = (0, 0, text.char_indices());
    for byte in wanted {
        while at < byte {
            let Some((b, c)) = chars.next() else {
                break;
            };
            units += c.len_utf16();
            at = b + c.len_utf8();
        }
        out.push((byte, units));
    }
    out
}
