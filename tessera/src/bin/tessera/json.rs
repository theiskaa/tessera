//! `tessera json`: one JSON request on stdin, one JSON response on stdout.
//!
//! The text only ever arrives on stdin, since a process's arguments are visible to other users
//! through `ps`. Exit status 2 means the caller is at fault: bad arguments, a malformed or
//! invalid request, or input the library rejects. Exit status 1 means the run failed: I/O, the
//! model directory, or the library itself.

use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use serde::Deserialize;
use serde_json::{Map, Value};
use tessera::{Config, Error, Format, Kind, KindSet, MarkdownOptions, Query, Tessera};

use crate::bundle;
use crate::render::{Document, Offsets, entity_offsets, extraction_offsets};

const USAGE: &str = "usage: tessera json --bundle DIR [--pretty] < request.json";

/// Why a run failed, which decides its exit status.
enum Fault {
    /// Bad arguments: exit 2, with the usage line.
    Usage(String),
    /// A request the caller must fix: exit 2.
    Caller(String),
    /// A failure the caller could not have prevented: exit 1.
    Runtime(String),
}

struct Args {
    bundle: PathBuf,
    pretty: bool,
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

/// Run `tessera json` with the arguments after `json`.
pub(crate) fn main(args: impl Iterator<Item = String>) -> ExitCode {
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(Fault::Usage(m)) => {
            eprintln!("error: {m}");
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
        Err(Fault::Caller(m)) => {
            eprintln!("error: {m}");
            ExitCode::from(2)
        }
        Err(Fault::Runtime(m)) => {
            eprintln!("error: {m}");
            ExitCode::from(1)
        }
    }
}

/// `Ok(None)` when the caller asked for help.
fn parse_args(mut it: impl Iterator<Item = String>) -> Result<Option<Args>, Fault> {
    let mut bundle = None;
    let mut pretty = false;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--bundle" => {
                let dir = it
                    .next()
                    .ok_or_else(|| Fault::Usage("--bundle needs a value".into()))?;
                bundle = Some(PathBuf::from(dir));
            }
            "--pretty" => pretty = true,
            "-h" | "--help" => return Ok(None),
            s if s.starts_with("--") => return Err(Fault::Usage(format!("unknown flag `{s}`"))),
            // Never echoed: a stray argument may be text meant for stdin.
            _ => {
                return Err(Fault::Usage(
                    "unexpected argument; the request is read from stdin".into(),
                ));
            }
        }
    }
    let bundle = bundle.ok_or_else(|| Fault::Usage("--bundle is required".into()))?;
    Ok(Some(Args { bundle, pretty }))
}

fn run(args: impl Iterator<Item = String>) -> Result<(), Fault> {
    let Some(args) = parse_args(args)? else {
        println!("{USAGE}");
        return Ok(());
    };
    let mut input = Vec::new();
    io::stdin()
        .read_to_end(&mut input)
        .map_err(|e| Fault::Runtime(format!("reading stdin: {e}")))?;
    let request: Request = serde_json::from_slice(&input)
        .map_err(|e| Fault::Caller(format!("invalid request: {e}")))?;
    let kinds = kinds(request.kinds.as_deref())?;
    if matches!(request.operation, Operation::Address) && !kinds.contains(Kind::Address) {
        return Err(Fault::Caller(
            "operation `address` needs kind `address`".into(),
        ));
    }
    let hints = country_hint(request.country_hint.unwrap_or_default())?;
    let hints: Vec<&str> = hints.iter().map(String::as_str).collect();
    let query = Query {
        country_hint: &hints,
        include_uncertain: request.include_uncertain.unwrap_or(false),
        format: format(request.format.unwrap_or(FormatName::Text))?,
    };

    let bundle = bundle::read(&args.bundle).map_err(Fault::Runtime)?;
    let tessera = Tessera::load(
        &bundle.bytes,
        Config {
            kinds,
            expected_checksum: Some(&bundle.checksum),
        },
    )
    .map_err(|e| Fault::Runtime(format!("{}: {e}", bundle.path.display())))?;

    let response = answer(
        &tessera,
        request.operation,
        &request.text,
        &query,
        request.offsets.unwrap_or_default(),
    )?;
    let rendered = if args.pretty {
        serde_json::to_string_pretty(&response)
    } else {
        serde_json::to_string(&response)
    }
    .map_err(|e| Fault::Runtime(e.to_string()))?;
    let mut stdout = io::stdout().lock();
    writeln!(stdout, "{rendered}").map_err(|e| Fault::Runtime(format!("writing stdout: {e}")))
}

/// The response object for one operation over `text`.
fn answer(
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
        return Err(Fault::Caller("kinds must name at least one kind".into()));
    }
    requested.iter().try_fold(KindSet::EMPTY, |set, k| {
        Kind::from_str_label(k)
            .map(|kind| set | kind)
            .ok_or_else(|| Fault::Caller(format!("unknown kind `{k}`")))
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
                Err(Fault::Caller(format!(
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
        FormatName::Markdown => Err(Fault::Caller(
            "format `markdown` requires the `markdown` feature".into(),
        )),
    }
}

/// Input the library rejects is the caller's to fix; anything else is a failed run.
fn library_fault(e: Error) -> Fault {
    match e {
        Error::InputTooLarge | Error::UnsupportedFormat => Fault::Caller(e.to_string()),
        Error::BundleInvalid
        | Error::ChecksumMismatch
        | Error::UnsupportedVersion
        | Error::Inference { .. } => Fault::Runtime(e.to_string()),
    }
}
