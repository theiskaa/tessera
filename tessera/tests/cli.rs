//! End-to-end test of the `tessera` binary. Requires the `cli` feature and a native target,
//! since it spawns the binary as a process.
#![cfg(all(feature = "cli", not(target_arch = "wasm32")))]

use std::io::Write;
use std::process::{Command, Stdio};

fn run(args: &[&str], stdin: &str) -> (i32, String, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_tessera"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(stdin.as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    (
        out.status.code().unwrap(),
        String::from_utf8(out.stdout).unwrap(),
        String::from_utf8(out.stderr).unwrap(),
    )
}

#[cfg(feature = "phone-metadata")]
#[test]
fn emails_and_phones_as_json() {
    let (code, stdout, _) = run(
        &["--country", "GE"],
        "Nino <nino@kavkaz-freight.example>, +995 32 212 3456\n",
    );
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let ents = v["entities"].as_array().unwrap();
    assert_eq!(ents.len(), 2);
    assert_eq!(ents[0]["kind"], "email");
    assert_eq!(ents[0]["text"], "nino@kavkaz-freight.example");
    assert_eq!(ents[0]["start"], 6);
    assert_eq!(ents[0]["review_recommended"], false);
    assert_eq!(ents[1]["kind"], "phone");
    assert_eq!(ents[1]["normalized"], "+995322123456");
}

#[test]
fn model_kind_without_model_is_usage_error() {
    let (code, _, stderr) = run(&["--kinds", "person"], "");
    assert_eq!(code, 2);
    assert!(stderr.contains("--model is required for kind `person`"));
}

#[test]
fn unknown_flag() {
    let (code, _, stderr) = run(&["--nope"], "");
    assert_eq!(code, 2);
    assert!(stderr.contains("unknown flag"));
}

#[test]
fn help_succeeds_and_prints_usage() {
    let (code, stdout, _) = run(&["--help"], "");
    assert_eq!(code, 0);
    assert!(stdout.starts_with("usage: tessera"));
}

#[test]
fn unknown_format_is_usage_error() {
    let (code, _, stderr) = run(&["--format", "html"], "");
    assert_eq!(code, 2);
    assert!(stderr.contains("unknown format `html`"), "{stderr}");
}

#[test]
fn markdown_flags_require_markdown_format() {
    for flag in ["--include-code", "--include-html", "--no-gfm-tables"] {
        let (code, _, stderr) = run(&[flag], "");
        assert_eq!(code, 2, "{flag}");
        assert!(
            stderr.contains(&format!("{flag} requires --format markdown")),
            "{stderr}"
        );
    }
}

#[cfg(feature = "markdown")]
#[test]
fn markdown_link_text_carries_the_destination() {
    let input =
        "[Nino](mailto:nino@kavkaz-freight.example)\n\n```\nhidden@kavkaz-freight.example\n```\n";
    let (code, stdout, _) = run(&["--format", "markdown"], input);
    assert_eq!(code, 0);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let ents = v["entities"].as_array().unwrap();
    assert_eq!(ents.len(), 1, "{stdout}");
    assert_eq!(ents[0]["text"], "Nino");
    assert_eq!(ents[0]["normalized"], "nino@kavkaz-freight.example");
    let (_, stdout, _) = run(&["--format", "markdown", "--include-code"], input);
    let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["entities"].as_array().unwrap().len(), 2, "{stdout}");
}

#[cfg(not(feature = "markdown"))]
#[test]
fn markdown_format_without_the_feature_is_usage_error() {
    let (code, _, stderr) = run(&["--format", "markdown"], "");
    assert_eq!(code, 2);
    assert!(
        stderr.contains("--format markdown requires the `markdown` feature"),
        "{stderr}"
    );
}
