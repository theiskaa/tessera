//! End-to-end tests of `tessera json`, the one-shot request mode, against the committed model
//! directory. Requires the `cli` feature and a native target, since it spawns the binary.
#![cfg(all(feature = "cli", not(target_arch = "wasm32")))]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use serde_json::{Value, json};

const MODELS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models");

const CARD: &str = "Jordan Avery, Project Coordinator\nAcme Corporation\n500 Main St, Springfield, IL 62701\n+1 202-555-0199\njordan@acme.example";

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

fn request(bundle: &str, body: &Value) -> (i32, String, String) {
    run(&["json", "--bundle", bundle], &body.to_string())
}

/// The response to a request that must succeed.
fn answer(body: Value) -> Value {
    let (code, stdout, stderr) = request(MODELS, &body);
    assert_eq!(code, 0, "{stderr}");
    assert_eq!(stdout.lines().count(), 1, "{stdout}");
    let v: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["model"], "tessera");
    assert_eq!(v["operation"], body["operation"]);
    v
}

/// The stderr of a request that must fail with `code`.
fn fault(bundle: &str, body: &str, code: i32) -> String {
    let (got, stdout, stderr) = run(&["json", "--bundle", bundle], body);
    assert_eq!(got, code, "{stderr}");
    assert!(stdout.is_empty(), "{stdout}");
    stderr
}

fn kinds(entities: &Value) -> Vec<&str> {
    entities
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e["kind"].as_str().unwrap())
        .collect()
}

/// Every object with `start`, `end`, and `text` under `v`, checked to slice `text` in `units`.
fn assert_spans_slice(v: &Value, source: &str, units: &str) {
    match v {
        Value::Object(m) => {
            if let (Some(start), Some(end), Some(Value::String(text))) =
                (m.get("start"), m.get("end"), m.get("text"))
            {
                let (start, end) = (start.as_u64().unwrap(), end.as_u64().unwrap());
                let (start, end) = (start as usize, end as usize);
                let sliced = match units {
                    "utf16" => {
                        String::from_utf16(&source.encode_utf16().collect::<Vec<_>>()[start..end])
                            .unwrap()
                    }
                    _ => source[start..end].to_string(),
                };
                assert_eq!(&sliced, text, "{units} span {start}..{end}");
            }
            m.values()
                .for_each(|x| assert_spans_slice(x, source, units));
        }
        Value::Array(a) => a.iter().for_each(|x| assert_spans_slice(x, source, units)),
        _ => {}
    }
}

/// A copy of the committed model directory, so a test can alter its `bundle.json`.
struct ModelCopy(PathBuf);

impl ModelCopy {
    fn new(name: &str, manifest: impl FnOnce(&mut Value)) -> Self {
        let dir = std::env::temp_dir().join(format!("tessera-cli-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut bundle: Value =
            serde_json::from_slice(&std::fs::read(Path::new(MODELS).join("bundle.json")).unwrap())
                .unwrap();
        let file = bundle["file"].as_str().unwrap().to_string();
        std::fs::copy(Path::new(MODELS).join(&file), dir.join(&file)).unwrap();
        manifest(&mut bundle);
        std::fs::write(dir.join("bundle.json"), bundle.to_string()).unwrap();
        ModelCopy(dir)
    }

    fn path(&self) -> &str {
        self.0.to_str().unwrap()
    }
}

impl Drop for ModelCopy {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[cfg(feature = "phone-metadata")]
#[test]
fn detect_finds_person_email_and_phone() {
    let text = "Please contact Jordan Avery at jordan.avery@acme.example or +1 202-555-0199.";
    let v = answer(json!({ "operation": "detect", "text": text, "country_hint": ["US"] }));
    let found = kinds(&v["entities"]);
    for kind in ["person", "email", "phone"] {
        assert!(found.contains(&kind), "{kind} missing from {v}");
    }
    let email = &v["entities"][found.iter().position(|k| *k == "email").unwrap()];
    assert_eq!(email["text"], "jordan.avery@acme.example");
    assert_eq!(email["start"], 31);
    assert_eq!(email["source"], "rules");
    assert_spans_slice(&v, text, "utf8");
}

#[test]
fn detect_limited_to_requested_kinds() {
    let text = "Jordan Avery, jordan@acme.example";
    let v = answer(json!({ "operation": "detect", "text": text, "kinds": ["email"] }));
    assert_eq!(kinds(&v["entities"]), ["email"]);
}

#[cfg(feature = "phone-metadata")]
#[test]
fn contacts_group_the_card() {
    let v = answer(json!({ "operation": "contacts", "text": CARD, "country_hint": ["US"] }));
    let contacts = v["contacts"].as_array().unwrap();
    assert_eq!(contacts.len(), 1, "{v}");
    let c = &contacts[0];
    assert_eq!(c["person"]["text"], "Jordan Avery");
    assert_eq!(c["emails"][0]["text"], "jordan@acme.example");
    assert_eq!(c["phones"][0]["normalized"], "+12025550199");
    assert_eq!(c["addresses"].as_array().unwrap().len(), 1);
    for key in ["start", "end", "confidence", "review_recommended"] {
        assert!(c.get(key).is_some(), "{key} missing from {c}");
    }
    assert!(v["unassigned"].is_array());
    assert_spans_slice(&v, CARD, "utf8");
}

#[test]
fn address_is_split_into_components() {
    let text = "400 Broad St, Seattle, WA 98109";
    let v = answer(json!({ "operation": "address", "text": text }));
    let address = &v["address"];
    assert_eq!(address["kind"], "address");
    assert_eq!(address["start"], 0);
    assert_eq!(address["end"], text.len());
    let labels: Vec<&str> = address["components"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c["label"].as_str().unwrap())
        .collect();
    for label in ["house_number", "road", "city", "region", "postcode"] {
        assert!(labels.contains(&label), "{label} missing from {address}");
    }
    assert_spans_slice(&v, text, "utf8");
}

#[test]
fn empty_address_still_has_components() {
    let v = answer(json!({ "operation": "address", "text": "" }));
    assert_eq!(v["address"]["components"], json!([]));
}

#[cfg(feature = "phone-metadata")]
#[test]
fn utf16_offsets_count_code_units() {
    let text = "Café 😀 — write to jordan@acme.example or +1 202-555-0199";
    let v = answer(json!({
        "operation": "detect",
        "text": text,
        "country_hint": ["US"],
        "offsets": "utf16",
    }));
    let email = &v["entities"][0];
    assert_eq!(email["kind"], "email");
    let byte = text.find("jordan@").unwrap();
    assert_eq!(email["start"], text[..byte].encode_utf16().count());
    assert_ne!(email["start"], byte);
    assert_spans_slice(&v, text, "utf16");

    let card = format!("Ünïcødé 😀 header\n\n{CARD}");
    for operation in ["contacts", "detect"] {
        let v = answer(json!({ "operation": operation, "text": card, "offsets": "utf16" }));
        assert_spans_slice(&v, &card, "utf16");
    }
    let address = "Straße 😀 400 Broad St, Seattle, WA 98109";
    let v = answer(json!({ "operation": "address", "text": address, "offsets": "utf16" }));
    assert_eq!(v["address"]["end"], address.encode_utf16().count());
    assert_spans_slice(&v, address, "utf16");
}

#[test]
fn pretty_output_spans_lines() {
    let (code, stdout, _) = run(
        &["json", "--bundle", MODELS, "--pretty"],
        r#"{"operation":"detect","text":"a@b.example","kinds":["email"]}"#,
    );
    assert_eq!(code, 0);
    assert!(stdout.lines().count() > 1);
    let v: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v["entities"][0]["text"], "a@b.example");
}

#[test]
fn checksum_mismatch_is_a_runtime_fault() {
    let copy = ModelCopy::new("checksum", |m| m["sha256"] = json!("0".repeat(64)));
    let stderr = fault(
        copy.path(),
        r#"{"operation":"detect","text":"Jordan Avery"}"#,
        1,
    );
    assert!(stderr.contains("checksum does not match"), "{stderr}");
}

#[test]
fn bundle_problems_are_runtime_faults() {
    let body = r#"{"operation":"detect","text":"x"}"#;
    let stderr = fault("/nonexistent/tessera-models", body, 1);
    assert!(stderr.contains("bundle.json"), "{stderr}");

    let copy = ModelCopy::new("missing-file", |m| m["file"] = json!("absent.safetensors"));
    let stderr = fault(copy.path(), body, 1);
    assert!(stderr.contains("absent.safetensors"), "{stderr}");

    let copy = ModelCopy::new("escape", |m| m["file"] = json!("../tessera-v1.safetensors"));
    let stderr = fault(copy.path(), body, 1);
    assert!(stderr.contains("does not name a file"), "{stderr}");

    let copy = ModelCopy::new("runtime", |m| m["runtime"] = json!("other"));
    let stderr = fault(copy.path(), body, 1);
    assert!(stderr.contains("runtime `other`"), "{stderr}");
}

#[test]
fn malformed_requests_are_caller_faults() {
    for (body, expected) in [
        ("{not json", "invalid request"),
        ("", "invalid request"),
        ("[]", "invalid request"),
        (r#"{"operation":"detect","text":"a"} {}"#, "invalid request"),
        (r#"{"operation":"detect"}"#, "missing field `text`"),
        (
            r#"{"operation":"translate","text":"a"}"#,
            "unknown variant `translate`",
        ),
        (
            r#"{"operation":"detect","text":"a","kinds":["planet"]}"#,
            "unknown kind `planet`",
        ),
        (
            r#"{"operation":"detect","text":"a","kinds":[]}"#,
            "at least one kind",
        ),
        (
            r#"{"operation":"detect","text":"a","extra":1}"#,
            "unknown field `extra`",
        ),
        (r#"{"operation":"detect","text":7}"#, "invalid type"),
        (
            r#"{"operation":"detect","text":"a","include_uncertain":"yes"}"#,
            "invalid type",
        ),
        (
            r#"{"operation":"detect","text":"a","format":"html"}"#,
            "unknown variant `html`",
        ),
        (
            r#"{"operation":"detect","text":"a","offsets":"utf32"}"#,
            "unknown variant `utf32`",
        ),
        (
            r#"{"operation":"detect","text":"a","country_hint":["USA"]}"#,
            "two-letter",
        ),
        (
            r#"{"operation":"address","text":"a","kinds":["email"]}"#,
            "needs kind `address`",
        ),
    ] {
        let stderr = fault(MODELS, body, 2);
        assert!(stderr.contains(expected), "{body}: {stderr}");
    }
}

#[test]
fn oversized_address_is_a_caller_fault() {
    let text = "1 Main St ".repeat(1000);
    let stderr = fault(
        MODELS,
        &json!({ "operation": "address", "text": text }).to_string(),
        2,
    );
    assert!(stderr.contains("exceeds the supported size"), "{stderr}");
}

#[cfg(not(feature = "markdown"))]
#[test]
fn markdown_without_the_feature_is_a_caller_fault() {
    let stderr = fault(
        MODELS,
        r#"{"operation":"detect","text":"a","format":"markdown"}"#,
        2,
    );
    assert!(
        stderr.contains("requires the `markdown` feature"),
        "{stderr}"
    );
}

#[cfg(feature = "markdown")]
#[test]
fn markdown_reads_link_destinations() {
    let text = "[Nino](mailto:nino@kavkaz-freight.example)\n";
    let v = answer(json!({
        "operation": "detect",
        "text": text,
        "kinds": ["email"],
        "format": "markdown",
    }));
    assert_eq!(v["entities"][0]["text"], "Nino");
    assert_eq!(
        v["entities"][0]["normalized"],
        "nino@kavkaz-freight.example"
    );
}

#[test]
fn arguments_never_carry_text() {
    let (code, _, stderr) = run(
        &["json", "--bundle", MODELS, "Jordan Avery"],
        r#"{"operation":"detect","text":"a"}"#,
    );
    assert_eq!(code, 2);
    assert!(stderr.contains("read from stdin"), "{stderr}");
    assert!(!stderr.contains("Jordan"), "{stderr}");
}

#[test]
fn bundle_flag_is_required() {
    let (code, _, stderr) = run(&["json"], r#"{"operation":"detect","text":"a"}"#);
    assert_eq!(code, 2);
    assert!(stderr.contains("--bundle is required"), "{stderr}");
}

#[test]
fn json_help_succeeds() {
    let (code, stdout, _) = run(&["json", "--help"], "");
    assert_eq!(code, 0);
    assert!(stdout.starts_with("usage: tessera json"));
}

#[test]
fn flag_mode_is_unchanged() {
    let (code, stdout, _) = run(&["--kinds", "email"], "write to jordan@acme.example\n");
    assert_eq!(code, 0);
    let v: Value = serde_json::from_str(&stdout).unwrap();
    assert_eq!(v.as_object().unwrap().len(), 1, "{stdout}");
    assert_eq!(
        v["entities"],
        json!([{
            "kind": "email",
            "text": "jordan@acme.example",
            "start": 9,
            "end": 28,
            "confidence": 0.99,
            "review_recommended": false,
            "source": "rules",
            "normalized": "jordan@acme.example",
        }])
    );
    let model = format!("{MODELS}/tessera-v1.safetensors");
    let (code, stdout, stderr) = run(
        &["--kinds", "person,email", "--model", &model],
        "Please contact Jordan Avery at jordan.avery@acme.example.",
    );
    assert_eq!(code, 0, "{stderr}");
    let v: Value = serde_json::from_str(&stdout).unwrap();
    assert!(kinds(&v["entities"]).contains(&"email"), "{stdout}");
}
