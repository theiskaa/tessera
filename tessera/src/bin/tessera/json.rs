//! `tessera json`: one JSON request on stdin, one JSON response on stdout, answered by
//! [`tessera::json::answer`].
//!
//! The text only ever arrives on stdin, since a process's arguments are visible to other users
//! through `ps`. Exit status 2 means the caller is at fault: bad arguments, a malformed or
//! invalid request, or input the library rejects. Exit status 1 means the run failed: I/O, the
//! model directory, or the library itself.

use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use tessera::json::Fault as Answered;

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

impl From<Answered> for Fault {
    fn from(fault: Answered) -> Self {
        match fault {
            Answered::Request(m) => Fault::Caller(m),
            Answered::Run(m) => Fault::Runtime(m),
        }
    }
}

struct Args {
    bundle: PathBuf,
    pretty: bool,
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
    let response = tessera::json::answer(&args.bundle, &input)?;
    let rendered = if args.pretty {
        serde_json::to_string_pretty(&response)
    } else {
        serde_json::to_string(&response)
    }
    .map_err(|e| Fault::Runtime(e.to_string()))?;
    let mut stdout = io::stdout().lock();
    writeln!(stdout, "{rendered}").map_err(|e| Fault::Runtime(format!("writing stdout: {e}")))
}
