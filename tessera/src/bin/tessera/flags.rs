//! `tessera [flags] [FILE]`: entities in a file or stdin, as `{"entities": [...]}`.

use std::io::{self, Read, Write};
use std::process::ExitCode;

use serde_json::json;
use tessera::{Config, Format, Kind, KindSet, MarkdownOptions, Query, Tessera};

use crate::render::{Document, Offsets};

const USAGE: &str = "usage: tessera [--kinds email,phone] [--country GB,GE] [--include-uncertain] [--format text|markdown] [--include-code] [--include-html] [--no-gfm-tables] [--model PATH] [--pretty] [FILE]
       tessera json --bundle DIR [--pretty] < request.json";

struct Args {
    kinds: KindSet,
    country: Vec<String>,
    include_uncertain: bool,
    format: Format,
    model: Option<String>,
    pretty: bool,
    file: Option<String>,
}

/// `Ok(None)` when the caller asked for help.
fn parse_args(mut it: impl Iterator<Item = String>) -> Result<Option<Args>, String> {
    let mut args = Args {
        kinds: Kind::Email | Kind::Phone,
        country: Vec::new(),
        include_uncertain: false,
        format: Format::Text,
        model: None,
        pretty: false,
        file: None,
    };
    let mut markdown = MarkdownOptions::default();
    // The first Markdown-only flag seen, rejected unless `--format markdown` is given too.
    let mut markdown_flag = None;
    while let Some(a) = it.next() {
        match a.as_str() {
            "--kinds" => {
                let v = it.next().ok_or("--kinds needs a value")?;
                let mut set = KindSet::EMPTY;
                for k in v.split(',') {
                    set = set
                        | Kind::from_str_label(k.trim())
                            .ok_or_else(|| format!("unknown kind `{k}`"))?;
                }
                args.kinds = set;
            }
            "--country" => {
                let v = it.next().ok_or("--country needs a value")?;
                args.country = v
                    .split(',')
                    .map(|c| c.trim().to_ascii_uppercase())
                    .collect();
            }
            "--include-uncertain" => args.include_uncertain = true,
            "--format" => {
                args.format = match it.next().ok_or("--format needs a value")?.as_str() {
                    "text" => Format::Text,
                    "markdown" if cfg!(feature = "markdown") => {
                        Format::Markdown(MarkdownOptions::default())
                    }
                    "markdown" => {
                        return Err("--format markdown requires the `markdown` feature".into());
                    }
                    v => return Err(format!("unknown format `{v}`")),
                }
            }
            "--include-code" | "--include-html" | "--no-gfm-tables" => {
                match a.as_str() {
                    "--include-code" => markdown.include_code = true,
                    "--include-html" => markdown.include_html = true,
                    _ => markdown.gfm_tables = false,
                }
                markdown_flag.get_or_insert(a);
            }
            "--model" => args.model = Some(it.next().ok_or("--model needs a value")?),
            "--pretty" => args.pretty = true,
            "-h" | "--help" => return Ok(None),
            s if s.starts_with("--") => return Err(format!("unknown flag `{s}`")),
            s => args.file = Some(s.to_string()),
        }
    }
    if let Format::Markdown(options) = &mut args.format {
        *options = markdown;
    } else if let Some(flag) = markdown_flag {
        return Err(format!("{flag} requires --format markdown"));
    }
    if !args.kinds.is_rules_only() && args.model.is_none() {
        let k = Kind::ALL
            .iter()
            .find(|k| args.kinds.contains(**k) && !matches!(k, Kind::Email | Kind::Phone));
        return Err(format!(
            "--model is required for kind `{}`",
            k.map(|k| k.as_str()).unwrap_or("?")
        ));
    }
    Ok(Some(args))
}

fn run(args: impl Iterator<Item = String>) -> Result<(), (u8, String)> {
    let Some(args) = parse_args(args).map_err(|m| (2, m))? else {
        println!("{USAGE}");
        return Ok(());
    };
    let bundle = match &args.model {
        Some(p) => std::fs::read(p).map_err(|e| (1, format!("{p}: {e}")))?,
        None => Vec::new(),
    };
    let text = match args.file.as_deref() {
        None | Some("-") => {
            let mut s = String::new();
            io::stdin()
                .read_to_string(&mut s)
                .map_err(|e| (1, e.to_string()))?;
            s
        }
        Some(p) => std::fs::read_to_string(p).map_err(|e| (1, format!("{p}: {e}")))?,
    };
    let tessera = Tessera::load(
        &bundle,
        Config {
            kinds: args.kinds,
            expected_checksum: None,
        },
    )
    .map_err(|e| (1, e.to_string()))?;
    let hints: Vec<&str> = args.country.iter().map(String::as_str).collect();
    let entities = tessera
        .detect(
            &text,
            &Query {
                country_hint: &hints,
                include_uncertain: args.include_uncertain,
                format: args.format.clone(),
            },
        )
        .map_err(|e| (1, e.to_string()))?;
    let out = json!({ "entities": Document::new(&text, Offsets::Utf8, []).entities(&entities) });
    let rendered = if args.pretty {
        serde_json::to_string_pretty(&out)
    } else {
        serde_json::to_string(&out)
    };
    let rendered = rendered.map_err(|e| (1, e.to_string()))?;
    let mut stdout = io::stdout().lock();
    writeln!(stdout, "{rendered}").map_err(|e| (1, e.to_string()))
}

/// Run the flag-based mode with every argument after the program name.
pub(crate) fn main(args: impl Iterator<Item = String>) -> ExitCode {
    match run(args) {
        Ok(()) => ExitCode::SUCCESS,
        Err((2, m)) => {
            if !m.is_empty() {
                eprintln!("error: {m}");
            }
            eprintln!("{USAGE}");
            ExitCode::from(2)
        }
        Err((code, m)) => {
            eprintln!("error: {m}");
            ExitCode::from(code)
        }
    }
}
