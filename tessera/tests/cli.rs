//! End-to-end test of the `tessera` binary. Requires the `cli` feature.
#![cfg(feature = "cli")]

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
