//! The demo's JSON view: each entity as the CLI prints it, one object per entity so the view
//! can add them one at a time.

use tessera::internal::display_confidence;

use crate::protocol::{Found, FoundKind};

fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// A confidence as the CLI prints it: rounded to four decimals, printed as a JSON float.
fn confidence(c: f32) -> String {
    let rounded = display_confidence(c).to_string();
    if rounded.contains(['.', 'e']) {
        rounded
    } else {
        format!("{rounded}.0")
    }
}

pub(crate) fn kind_name(kind: FoundKind) -> &'static str {
    match kind {
        FoundKind::Address => "address",
        FoundKind::Phone => "phone",
        FoundKind::Email => "email",
    }
}

/// The opening of the array, before any object.
pub(crate) const OPEN: &str = "[";
/// What goes before the first object, and before each later one.
pub(crate) const FIRST: &str = "\n";
pub(crate) const NEXT: &str = ",\n";
/// The close of the array, after the last object.
pub(crate) const CLOSE: &str = "\n]";

/// `e` as an indented object; `OPEN`, the objects with their separators, and `CLOSE` make one
/// JSON array. Fields and their order are the CLI's; offsets are into `text`.
pub(crate) fn object(text: &str, e: &Found) -> String {
    let slice = |start: usize, end: usize| quote(text.get(start..end).unwrap_or(""));
    let mut fields = vec![
        format!("\"kind\": \"{}\"", kind_name(e.kind)),
        format!("\"text\": {}", slice(e.start, e.end)),
        format!("\"start\": {}", e.start),
        format!("\"end\": {}", e.end),
        format!("\"confidence\": {}", confidence(e.confidence)),
        format!("\"review_recommended\": {}", e.review_recommended),
        format!("\"source\": {}", quote(&e.source)),
    ];
    if let Some(n) = &e.normalized {
        fields.push(format!("\"normalized\": {}", quote(n)));
    }
    if let Some(r) = &e.region {
        fields.push(format!("\"region\": {}", quote(r)));
    }
    if !e.components.is_empty() {
        let parts: Vec<String> = e
            .components
            .iter()
            .map(|c| {
                format!(
                    "      {{ \"label\": {}, \"text\": {}, \"start\": {}, \"end\": {}, \"confidence\": {} }}",
                    quote(&c.label),
                    slice(c.start, c.end),
                    c.start,
                    c.end,
                    confidence(c.confidence)
                )
            })
            .collect();
        fields.push(format!("\"components\": [\n{}\n    ]", parts.join(",\n")));
    }
    format!("  {{\n    {}\n  }}", fields.join(",\n    "))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn whole(text: &str, found: &[Found]) -> String {
        let objects: String = found
            .iter()
            .enumerate()
            .map(|(i, e)| format!("{}{}", if i == 0 { FIRST } else { NEXT }, object(text, e)))
            .collect();
        format!("{OPEN}{objects}{CLOSE}")
    }

    /// The same object the CLI's `entity_json` builds, keys in its order.
    fn cli_shape(e: &tessera::Entity, text: &str) -> serde_json::Value {
        let mut m = serde_json::Map::new();
        m.insert("kind".into(), e.kind.as_str().into());
        m.insert("text".into(), e.text(text).into());
        m.insert("start".into(), e.start.into());
        m.insert("end".into(), e.end.into());
        m.insert("confidence".into(), display_confidence(e.confidence).into());
        m.insert("review_recommended".into(), e.review_recommended.into());
        m.insert("source".into(), e.source.as_str().into());
        if let Some(n) = &e.normalized {
            m.insert("normalized".into(), n.as_str().into());
        }
        if let Some(r) = &e.region {
            m.insert("region".into(), r.as_str().into());
        }
        if !e.components.is_empty() {
            let comps: Vec<serde_json::Value> = e
                .components
                .iter()
                .map(|c| serde_json::json!({ "label": c.label.as_str(), "text": c.text(text), "start": c.start, "end": c.end, "confidence": display_confidence(c.confidence) }))
                .collect();
            m.insert("components".into(), comps.into());
        }
        serde_json::Value::Object(m)
    }

    #[test]
    fn objects_join_into_the_cli_output() {
        let text = "Call +44 20 7946 0321 or ops@Example.com.";
        let rules = tessera::Tessera::load(
            &[],
            tessera::Config {
                kinds: tessera::Kind::Email | tessera::Kind::Phone,
                expected_checksum: None,
            },
        )
        .unwrap();
        let entities = rules.detect(text, &tessera::Query::default()).unwrap();
        assert_eq!(entities.len(), 2);
        let address = tessera::Entity {
            kind: tessera::Kind::Address,
            start: 0,
            end: 4,
            confidence: 0.99554205,
            review_recommended: false,
            source: tessera::Source::Model,
            components: vec![tessera::Component {
                label: tessera::AddressLabel::Road,
                start: 0,
                end: 4,
                confidence: 1.0,
            }],
            normalized: None,
            region: None,
        };
        let mut all: Vec<Found> = entities
            .iter()
            .filter_map(|e| Found::from_entity(e, 0))
            .collect();
        all.extend(Found::from_entity(&address, 0));
        let printed = whole(text, &all);
        let ours: serde_json::Value = serde_json::from_str(&printed).unwrap();
        let want: Vec<serde_json::Value> = entities
            .iter()
            .chain([&address])
            .map(|e| cli_shape(e, text))
            .collect();
        // Compared as strings too, because `Value` equality ignores key order.
        assert_eq!(ours, serde_json::Value::Array(want.clone()));
        assert_eq!(ours.to_string(), serde_json::Value::Array(want).to_string());
        assert!(printed.contains("\"confidence\": 1.0 }"));
        assert_eq!(whole(text, &[]), "[\n]");
    }
}
