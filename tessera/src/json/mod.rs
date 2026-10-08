//! Feature `json`: one JSON request in, one JSON response out, the contract `tessera json`
//! speaks over stdin and stdout and that a program embedding the library can answer in process.
//!
//! A request names an `operation` (`detect`, `contacts`, or `address`) and the `text`, with
//! optional `kinds`, `country_hint`, `include_uncertain`, `format`, and `offsets`; without a
//! `country_hint`, the one the bundle's detector was trained to expect is used. A response has
//! `model` and `operation`, then `entities` for `detect`, `contacts` and `unassigned` for
//! `contacts`, or `address` with its `components` for `address`. A [`Fault`] says whether the
//! request or the run is to blame.

mod bundle;
mod render;

use std::path::Path;

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::{Config, Entity, Error, Format, Kind, KindSet, MarkdownOptions, Query, Tessera};
use render::{Document, Offsets, entity_offsets, extraction_offsets};

/// Why a request was not answered.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Fault {
    /// The request is at fault and will fail again unchanged: malformed JSON, an unknown field
    /// or value, or input the library rejects as too large.
    #[error("{0}")]
    Request(String),
    /// The run failed for a reason the caller could not have prevented: reading the model
    /// directory, a missing or mismatched bundle, or inference.
    #[error("{0}")]
    Run(String),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    operation: Operation,
    text: String,
    kinds: Option<Vec<String>>,
    country_hint: Option<Vec<String>>,
    include_uncertain: Option<bool>,
    format: Option<FormatName>,
    offsets: Option<Offsets>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Operation {
    Detect,
    Contacts,
    Address,
}

impl Operation {
    fn as_str(self) -> &'static str {
        match self {
            Operation::Detect => "detect",
            Operation::Contacts => "contacts",
            Operation::Address => "address",
        }
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "lowercase")]
enum FormatName {
    Text,
    Markdown,
}

/// Answer one JSON `request` with the model in `dir`, a directory holding a `bundle.json` and
/// the weights file it names, which is verified against the recorded digest before loading.
/// The request is checked before the model is read, so a request at fault says so even when the
/// model directory is also broken.
pub fn answer(dir: &Path, request: &[u8]) -> Result<Value, Fault> {
    let request: Request = serde_json::from_slice(request)
        .map_err(|e| Fault::Request(format!("invalid request: {e}")))?;
    let kinds = kinds(request.kinds.as_deref())?;
    if matches!(request.operation, Operation::Address) && !kinds.contains(Kind::Address) {
        return Err(Fault::Request(
            "operation `address` needs kind `address`".into(),
        ));
    }
    let requested_hints = request.country_hint.map(country_hint).transpose()?;
    let format = format(request.format.unwrap_or(FormatName::Text))?;

    let bundle = bundle::read(dir).map_err(Fault::Run)?;
    let hints = requested_hints.unwrap_or_else(|| bundle.country_hint.clone());
    let hints: Vec<&str> = hints.iter().map(String::as_str).collect();
    let query = Query {
        country_hint: &hints,
        include_uncertain: request.include_uncertain.unwrap_or(false),
        format,
    };
    let tessera = Tessera::load(
        &bundle.bytes,
        Config {
            kinds,
            expected_checksum: Some(&bundle.checksum),
        },
    )
    .map_err(|e| Fault::Run(format!("{}: {e}", bundle.path.display())))?;

    respond(
        &tessera,
        request.operation,
        &request.text,
        &query,
        request.offsets.unwrap_or_default(),
    )
}

/// `entities` found in `text` as the JSON array a `detect` response carries, with UTF-8 byte
/// offsets.
pub fn entities(text: &str, entities: &[Entity]) -> Value {
    Document::new(text, Offsets::Utf8, []).entities(entities)
}

/// The response object for one operation over `text`.
fn respond(
    tessera: &Tessera,
    operation: Operation,
    text: &str,
    query: &Query<'_>,
    offsets: Offsets,
) -> Result<Value, Fault> {
    let mut out = Map::new();
    out.insert("model".into(), "tessera".into());
    out.insert("operation".into(), operation.as_str().into());
    match operation {
        Operation::Detect => {
            let entities = tessera.detect(text, query).map_err(library_fault)?;
            let doc = Document::new(text, offsets, entities.iter().flat_map(entity_offsets));
            out.insert("entities".into(), doc.entities(&entities));
        }
        Operation::Contacts => {
            let extraction = tessera
                .extract_contacts(text, query)
                .map_err(library_fault)?;
            let doc = Document::new(text, offsets, extraction_offsets(&extraction));
            let (contacts, unassigned) = doc.extraction(&extraction);
            out.insert("contacts".into(), contacts);
            out.insert("unassigned".into(), unassigned);
        }
        Operation::Address => {
            let address = tessera.parse_address(text, query).map_err(library_fault)?;
            let doc = Document::new(text, offsets, entity_offsets(&address));
            let mut rendered = doc.entity(&address);
            // An address the parser could not split still answers with its (empty) components.
            if let Value::Object(m) = &mut rendered {
                m.entry("components")
                    .or_insert_with(|| Value::Array(Vec::new()));
            }
            out.insert("address".into(), rendered);
        }
    }
    Ok(Value::Object(out))
}

/// The requested kinds, every kind when absent.
fn kinds(requested: Option<&[String]>) -> Result<KindSet, Fault> {
    let Some(requested) = requested else {
        return Ok(Kind::all());
    };
    if requested.is_empty() {
        return Err(Fault::Request("kinds must name at least one kind".into()));
    }
    requested.iter().try_fold(KindSet::EMPTY, |set, k| {
        Kind::from_str_label(k)
            .map(|kind| set | kind)
            .ok_or_else(|| Fault::Request(format!("unknown kind `{k}`")))
    })
}

/// Region hints as ISO 3166-1 alpha-2 codes, uppercased.
fn country_hint(requested: Vec<String>) -> Result<Vec<String>, Fault> {
    requested
        .into_iter()
        .map(|c| {
            let code = c.trim().to_ascii_uppercase();
            if code.len() == 2 && code.bytes().all(|b| b.is_ascii_uppercase()) {
                Ok(code)
            } else {
                Err(Fault::Request(format!(
                    "country_hint `{c}` is not a two-letter region code"
                )))
            }
        })
        .collect()
}

fn format(name: FormatName) -> Result<Format, Fault> {
    match name {
        FormatName::Text => Ok(Format::Text),
        FormatName::Markdown if cfg!(feature = "markdown") => {
            Ok(Format::Markdown(MarkdownOptions::default()))
        }
        FormatName::Markdown => Err(Fault::Request(
            "format `markdown` requires the `markdown` feature".into(),
        )),
    }
}

/// Input the library rejects is the caller's to fix; anything else is a failed run.
fn library_fault(e: Error) -> Fault {
    match e {
        Error::InputTooLarge | Error::UnsupportedFormat => Fault::Request(e.to_string()),
        Error::BundleInvalid
        | Error::ChecksumMismatch
        | Error::UnsupportedVersion
        | Error::Inference { .. } => Fault::Run(e.to_string()),
    }
}
