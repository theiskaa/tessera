use super::*;
use crate::reviewed_data::digest;
use serde_json::json;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

const POLICY: &str = "synthetic-test-policy";

fn receipt(path: &str, bytes: &[u8]) -> Receipt {
    Receipt {
        path: PathBuf::from(path),
        sha256: digest(bytes),
    }
}

fn row(text: &str, expected: Vec<Span>) -> Row {
    Row {
        name: "authored-example".to_owned(),
        country: "US".to_owned(),
        input: text.to_owned(),
        expected,
        status: "resolved".to_owned(),
        uncertainties: vec![],
        unresolved: vec![],
        annotation_policy_sha256: POLICY.to_owned(),
        authored_source: AuthoredSource {
            kind: "whole_source_format_variant".to_owned(),
            native_dataset: receipt("fixture/prepared.jsonl", b"fixture"),
            native_row_index: 0,
            native_name: "native-example".to_owned(),
            native_text_sha256: digest(text.as_bytes()),
            format: "pipe_table".to_owned(),
        },
        real_source_text: false,
    }
}

fn span(kind: &str, text: &str, start: usize, end: usize) -> Span {
    Span {
        kind: kind.to_owned(),
        text: text[start..end].to_owned(),
        start: start as u32,
        end: end as u32,
    }
}

fn named_span(kind: &str, input: &str, name: &str) -> Span {
    let start = input.find(name).unwrap();
    span(kind, input, start, start + name.len())
}

fn proposal(row: &Row) -> Proposal {
    Proposal {
        name: row.name.clone(),
        country: row.country.clone(),
        input: row.input.clone(),
        expected: row.expected.clone(),
        uncertainties: vec![],
        annotation_policy_sha256: Some(POLICY.to_owned()),
    }
}

fn manifest_value() -> Value {
    json!({
        "schema":"reviewed-whole-source-authored-data-v1", "origin":"authored_train_only",
        "split":"train", "documents":1, "counts":{},
        "prepared":receipt("fixture/prepared.jsonl", b"prepared"),
        "policy":receipt("fixture/policy.json", b"policy"), "receipt_pins":{},
        "finalizer_sha256":digest(b"synthetic finalizer"),
        "all_source_semantics_and_438_notes_reviewed":true,
        "all_source_bytes_and388_notes_reviewed":false,
        "label_passes":"synthetic", "semantic_review":"synthetic",
        "native_train_documents_changed":false, "fit_route_ready":false, "fit_allowed":false,
        "remaining":["test fixture only"], "limits":["invented text; not training data"]
    })
}

#[derive(Default)]
struct MemoryFiles(BTreeMap<PathBuf, Vec<u8>>);

impl MemoryFiles {
    fn insert(&mut self, path: &str, bytes: Vec<u8>) -> Receipt {
        let pin = receipt(path, &bytes);
        assert!(self.0.insert(pin.path.clone(), bytes).is_none());
        pin
    }

    fn read(&self, pin: &Receipt) -> anyhow::Result<Vec<u8>> {
        let bytes = self
            .0
            .get(&pin.path)
            .ok_or_else(|| anyhow::anyhow!("synthetic dependency absent"))?;
        anyhow::ensure!(digest(bytes) == pin.sha256, "synthetic dependency changed");
        Ok(bytes.clone())
    }
}

fn json_line(value: &Value) -> Vec<u8> {
    let mut bytes = serde_json::to_vec(value).unwrap();
    bytes.push(b'\n');
    bytes
}

struct SourceFixture {
    row: Row,
    map: Value,
    prepared: Value,
    files: MemoryFiles,
}

impl SourceFixture {
    fn verify(&self) -> anyhow::Result<Value> {
        let map: source::CopyMap = serde_json::from_value(self.map.clone())?;
        source::verify(&self.row, &map, &self.prepared, &mut |pin| {
            self.files.read(pin)
        })
    }
}

fn staff_fixture() -> SourceFixture {
    let input = "Example Field Office\nName\tPosition\tPhone Number\tEmail\nJane Doe\tChief\t207-555-0100\tjane@example.org";
    let mut authored = String::new();
    let mut copies = vec![];
    let mut cursor = 0;
    for (line_index, line) in input.split('\n').enumerate() {
        if line_index > 0 {
            authored.push('\n');
        }
        for (cell_index, cell) in line.split('\t').enumerate() {
            if cell_index > 0 {
                authored.push_str(" | ");
            }
            let output_start = authored.len();
            authored.push_str(cell);
            copies.push(
                json!({"source_start":cursor, "source_end":cursor+cell.len(),
                "output_start":output_start, "output_end":authored.len()}),
            );
            cursor += cell.len() + 1;
        }
    }
    let targets = [
        ("org", "Example Field Office"),
        ("person", "Jane Doe"),
        ("phone", "207-555-0100"),
        ("email", "jane@example.org"),
    ];
    let mut files = MemoryFiles::default();
    let raw = b"<table><caption>Example Field Office</caption><tr><th>Name</th><th>Position</th><th>Phone Number</th><th>Email</th></tr><tr><td>Jane Doe</td><td>Chief</td><td>207-555-0100</td><td>jane@example.org</td></tr></table>";
    let raw_pin = files.insert("fixture/capture.html", raw.to_vec());
    let projection = files.insert("fixture/project.py", b"# synthetic fixture\n".to_vec());
    let original = json!({"name":"native-example", "input":input, "country":"US",
        "source_document_key":"https://example.org/directory#office",
        "parent_source_id":"https://example.org/directory", "source_family_id":"example-office-tables",
        "publisher_host":"example.org", "raw_source":raw_pin,
        "source_block":{"complete_native_table":true, "raw_html_start":0,
            "raw_html_end":raw.len(), "raw_html_sha256":digest(raw)},
        "projection":{"script":projection.path, "script_sha256":projection.sha256}});
    let original_pin = files.insert("fixture/native.jsonl", json_line(&original));
    let prepared = json!({"name":"native-example", "country":"US", "input":input,
        "status":"resolved", "uncertainties":[], "unresolved":[], "annotation_policy_sha256":POLICY,
        "expected":targets.iter().map(|&(kind, name)|named_span(kind,input,name)).collect::<Vec<_>>(),
        "origin":{"dataset":original_pin, "row_index":0,"row_id":"native-example",
            "text_sha256":digest(input.as_bytes()),
            "source_document_id":"https://example.org/directory#office",
            "parent_source_id":"https://example.org/directory", "source_family_id":"example-office-tables",
            "domain":"example.org"}});
    let prepared_pin = files.insert("fixture/prepared.jsonl", json_line(&prepared));
    let mut row = row(
        &authored,
        targets
            .iter()
            .map(|&(kind, name)| named_span(kind, &authored, name))
            .collect(),
    );
    row.authored_source.native_dataset = prepared_pin;
    row.authored_source.native_text_sha256 = digest(input.as_bytes());
    let map = json!({"name":row.name,"native_source_document":"native-example",
        "native_input_sha256":digest(input.as_bytes()),"input_sha256":digest(authored.as_bytes()),
        "literal_copies":copies,"all_nonwhitespace_source_bytes_retained_exactly_once":true,
        "original_gold_occurrences_retained":4,"meaning_changed":false});
    SourceFixture {
        row,
        map,
        prepared,
        files,
    }
}

fn reflow_fixture(record_url: bool) -> SourceFixture {
    let input = "Jane Doe, Chief; Example Office.";
    let expected = vec![
        named_span("person", input, "Jane Doe"),
        named_span("org", input, "Example Office"),
    ];
    let mut files = MemoryFiles::default();
    let mut original = json!({"id":"native-id", "country":"US", "text":input,
        "source":"synthetic directory fixture"});
    if record_url {
        original["source_url"] = json!("https://example.org/directory");
    }
    let original_pin = files.insert("fixture/native.jsonl", json_line(&original));
    let prepared = json!({"name":"native-example", "country":"US", "input":input,
        "status":"resolved", "uncertainties":[], "unresolved":[], "annotation_policy_sha256":POLICY,
        "expected":expected,"origin":{"dataset":original_pin,"row_index":0,
            "row_id":"native-id","text_sha256":digest(input.as_bytes())}});
    let prepared_line = serde_json::to_string(&prepared).unwrap();
    let prepared_pin = files.insert("fixture/prepared.jsonl", json_line(&prepared));
    let mut output = String::new();
    let mut copies = vec![];
    let mut insertions = vec![];
    let mut previous = 0;
    for end in input
        .bytes()
        .enumerate()
        .filter_map(|(index, byte)| matches!(byte, b',' | b';').then_some(index + 1))
        .chain(std::iter::once(input.len()))
    {
        let start = output.len();
        output.push_str(&input[previous..end]);
        copies.push(json!({"source_start":previous,"source_end":end,
            "output_start":start,"output_end":output.len()}));
        if end < input.len() {
            output.push('\n');
            insertions.push(json!({"literal":"\n","source_boundary":end}));
        }
        previous = end;
    }
    let mut row = row(
        &output,
        vec![
            named_span("person", &output, "Jane Doe"),
            named_span("org", &output, "Example Office"),
        ],
    );
    row.authored_source.native_dataset = prepared_pin.clone();
    row.authored_source.native_text_sha256 = digest(input.as_bytes());
    row.authored_source.format = "contact_lines".to_owned();
    let map = json!({"name":row.name,"native_source_document":"native-example",
        "source_dataset":prepared_pin,"source_row_index":0,
        "source_row_sha256":digest(prepared_line.as_bytes()),
        "source_context_evidence":[{"start":0,"end":input.len(),"quote":input}],
        "forbidden_inferences":["do not invent an employer"],
        "all_source_bytes_in_order_exactly_once":true,"literal_copy_map":copies,
        "inserted_whitespace_or_delimiters":insertions,"contact_regions":[[0,input.len()]]});
    SourceFixture {
        row,
        map,
        prepared,
        files,
    }
}

#[test]
fn resolved_authored_rows_reject_uncertainty_wrong_policy_and_native_claims() {
    let original = row(
        "Jane Doe",
        vec![named_span("person", "Jane Doe", "Jane Doe")],
    );
    loader::verify_row(&original, POLICY).unwrap();
    let value = serde_json::to_value(json!({
        "name":original.name,"country":original.country,"input":original.input,
        "expected":original.expected,"status":original.status,"uncertainties":[],"unresolved":[],
        "annotation_policy_sha256":POLICY,"authored_source":original.authored_source,"real_source_text":false
    })).unwrap();
    for (key, replacement) in [
        ("status", json!("quarantine")),
        ("uncertainties", json!(["hold"])),
        ("unresolved", json!(["hold"])),
        ("annotation_policy_sha256", json!("different")),
        ("country", json!("CA")),
        ("real_source_text", json!(true)),
    ] {
        let mut changed = value.clone();
        changed[key] = replacement;
        assert!(loader::verify_row(&serde_json::from_value(changed).unwrap(), POLICY).is_err());
    }
}

#[test]
fn authored_schema_does_not_accept_fabricated_native_origin() {
    let mut value = json!({"name":"example","country":"US","input":"Jane Doe",
        "expected":[],"status":"resolved","uncertainties":[],"unresolved":[],
        "annotation_policy_sha256":POLICY,"authored_source":row("Jane Doe",vec![]).authored_source,
        "real_source_text":false});
    assert!(serde_json::from_value::<Row>(value.clone()).is_ok());
    value["origin"] = json!({"dataset":"pretend-native"});
    assert!(serde_json::from_value::<Row>(value).is_err());
}

#[test]
fn spans_require_exact_utf8_nonoverlap_and_known_kinds() {
    let text = "Zoë Example";
    let good = named_span("person", text, text);
    loader::validate_spans(text, std::slice::from_ref(&good)).unwrap();
    let mut split = good.clone();
    split.start = 3;
    let mut unknown = good.clone();
    unknown.kind = "role".to_owned();
    let mut wrong = good.clone();
    wrong.text = "Zoe Example".to_owned();
    for spans in [
        vec![split],
        vec![unknown],
        vec![wrong],
        vec![good.clone(), good],
    ] {
        assert!(loader::validate_spans(text, &spans).is_err());
    }
}

#[test]
fn exact_duplicates_and_native_names_fail_but_normalized_collisions_remain_distinct() {
    let base = row("Jane Doe, Chief", vec![]);
    let mut variant = row("Jane Doe, \tChief", vec![]);
    variant.name = "authored-variant".to_owned();
    assert_eq!(
        loader::normalized(&base.input),
        loader::normalized(&variant.input)
    );
    let native_texts = BTreeSet::from([base.input.clone()]);
    let mut names = BTreeSet::new();
    let mut texts = BTreeSet::new();
    loader::verify_identity(
        &variant,
        &BTreeSet::new(),
        &native_texts,
        &mut names,
        &mut texts,
    )
    .unwrap();
    let mut duplicate = variant.clone();
    duplicate.name.push_str("-other");
    assert!(
        loader::verify_identity(
            &duplicate,
            &BTreeSet::new(),
            &native_texts,
            &mut names,
            &mut texts
        )
        .is_err()
    );
    assert!(
        loader::verify_identity(
            &base,
            &BTreeSet::new(),
            &native_texts,
            &mut BTreeSet::new(),
            &mut BTreeSet::new()
        )
        .is_err()
    );
    assert!(
        loader::verify_identity(
            &variant,
            &BTreeSet::from([variant.name.clone()]),
            &BTreeSet::new(),
            &mut BTreeSet::new(),
            &mut BTreeSet::new()
        )
        .is_err()
    );
}

#[test]
fn original_proposals_bind_whole_text_identity_and_entity_boundaries() {
    let row = row(
        "Jane Doe",
        vec![named_span("person", "Jane Doe", "Jane Doe")],
    );
    let original = proposal(&row);
    loader::verify_proposal(&row, &original).unwrap();
    let mut missing = proposal(&row);
    missing.expected.clear();
    let mut wrong_text = proposal(&row);
    wrong_text.input.push('\n');
    let mut wrong_name = proposal(&row);
    wrong_name.name.push_str("-different");
    for changed in [missing, wrong_text, wrong_name] {
        assert!(loader::verify_proposal(&row, &changed).is_err());
    }
}

#[test]
fn optional_proposal_policy_requires_a_string_when_present_and_exact_match() {
    let row = row("Jane Doe", vec![]);
    let mut value =
        json!({"name":row.name,"country":"US","input":row.input,"expected":[],"uncertainties":[]});
    assert!(
        serde_json::from_value::<Proposal>(value.clone())
            .unwrap()
            .annotation_policy_sha256
            .is_none()
    );
    value["annotation_policy_sha256"] = Value::Null;
    assert!(serde_json::from_value::<Proposal>(value.clone()).is_err());
    value["annotation_policy_sha256"] = json!(POLICY);
    loader::verify_proposal(&row, &serde_json::from_value(value.clone()).unwrap()).unwrap();
    value["annotation_policy_sha256"] = json!("wrong");
    assert!(loader::verify_proposal(&row, &serde_json::from_value(value).unwrap()).is_err());
}

#[test]
fn every_original_note_requires_a_unique_hash_bound_disposition() {
    let row = row("Jane Doe", vec![]);
    let note = json!({"end":8,"start":0,"text":"Jane Doe"});
    let mut a = proposal(&row);
    a.uncertainties.push(note.clone());
    let mut b = proposal(&row);
    b.uncertainties.push(note.clone());
    let hash = digest(&serde_json::to_vec(&note).unwrap());
    let review = json!({"counts":{"all_notes":2},"note_dispositions":[
        {"reviewer":"a","name":row.name,"note_index":0,"canonical_note_sha256":hash,"action":"retain person"},
        {"reviewer":"b","name":row.name,"note_index":0,"canonical_note_sha256":hash,"action":"retain person"}]});
    loader::verify_notes(&[a], &[b], &review).unwrap();
    for mode in 0..3 {
        let mut failed = review.clone();
        match mode {
            0 => {
                failed["note_dispositions"].as_array_mut().unwrap().pop();
            }
            1 => {
                failed["note_dispositions"][0]["canonical_note_sha256"] = json!("wrong");
            }
            _ => {
                let duplicate = failed["note_dispositions"][0].clone();
                failed["note_dispositions"]
                    .as_array_mut()
                    .unwrap()
                    .push(duplicate);
            }
        }
        let mut a = proposal(&row);
        a.uncertainties.push(note.clone());
        let mut b = proposal(&row);
        b.uncertainties.push(note.clone());
        assert!(loader::verify_notes(&[a], &[b], &failed).is_err());
    }
}

#[test]
fn semantic_review_requires_passed_status_bound_draft_and_all_source_guards() {
    let draft = receipt("fixture/draft.json", b"draft");
    let required = [
        "every_selected_complete_raw_table_caption_header_body_cell_exact",
        "every_nonwhitespace_source_byte_once_in_copy_maps",
        "all32_independent_text_reconstructions_exact",
        "every_original_expected_span_mapped_once",
        "triplicate_408_spans_equal",
        "all_utf8_spans_notes_valid",
        "all438_original_notes_retained_exact",
        "a_subject_rulings_explicitly_match",
        "b_status_rulings_all_supported",
        "no_new_proper_entity_fields_inserted",
        "builder_reads_only_reviewed_TRAIN_source_policy_not_DEV",
        "input_source_hashes_unchanged",
    ];
    let mut validation = serde_json::Map::new();
    for key in required {
        validation.insert(key.to_owned(), json!(true));
    }
    let review = json!({"status":"passed","blockers":[],"draft_sha256":draft.sha256,"validation":validation});
    loader::verify_semantic_status(&review, &draft).unwrap();
    for key in required {
        let mut failed = review.clone();
        failed["validation"][key] = json!(false);
        assert!(loader::verify_semantic_status(&failed, &draft).is_err());
    }
    for (key, bad) in [
        ("status", json!("preliminary")),
        ("blockers", json!(["hold"])),
        ("draft_sha256", json!("wrong")),
    ] {
        let mut failed = review.clone();
        failed[key] = bad;
        assert!(loader::verify_semantic_status(&failed, &draft).is_err());
    }
}

#[test]
fn authored_manifest_rejects_dev_native_origin_and_ambiguous_review_routes() {
    let good = manifest_value();
    loader::verify_manifest(&serde_json::from_value(good.clone()).unwrap()).unwrap();
    for (key, bad) in [
        ("split", json!("dev")),
        ("origin", json!("native")),
        ("documents", json!(0)),
        ("native_train_documents_changed", json!(true)),
        ("all_source_semantics_and_438_notes_reviewed", json!(false)),
        ("all_source_bytes_and388_notes_reviewed", json!(true)),
    ] {
        let mut failed = good.clone();
        failed[key] = bad;
        assert!(loader::verify_manifest(&serde_json::from_value(failed).unwrap()).is_err());
    }
    let mut unknown = good;
    unknown["fake_native_origin"] = json!({});
    assert!(serde_json::from_value::<Manifest>(unknown).is_err());
}

#[test]
fn absent_stale_and_inconsistently_pinned_dependencies_fail() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("receipt.json");
    std::fs::write(&path, b"original").unwrap();
    let pin = Receipt {
        path: path.clone(),
        sha256: digest(b"original"),
    };
    let mut deps = loader::Dependencies::default();
    assert_eq!(deps.read(&pin).unwrap(), b"original");
    let mut wrong = pin.clone();
    wrong.sha256 = digest(b"wrong");
    assert!(deps.read(&wrong).is_err());
    std::fs::write(&path, b"mutated").unwrap();
    assert!(loader::Dependencies::default().read(&pin).is_err());
    std::fs::remove_file(path).unwrap();
    assert!(loader::Dependencies::default().read(&pin).is_err());
    assert!(
        load(
            &pin,
            dir.path(),
            &[],
            &FeatureConfig::default(),
            InputContract::Legacy23
        )
        .is_err()
    );
}

#[test]
fn jsonl_rejects_non_utf8_and_unknown_row_fields() {
    assert!(loader::jsonl::<Row>(&[0xff]).is_err());
    assert!(loader::jsonl::<Row>(b"{\"origin\":\"native\"}\n").is_err());
    let line = json_line(
        &json!({"name":"example","country":"US","input":"Zoë Example",
        "expected":[],"status":"resolved","uncertainties":[],"unresolved":[],
        "annotation_policy_sha256":POLICY,"authored_source":row("Zoë Example",vec![]).authored_source,"real_source_text":false}),
    );
    assert_eq!(loader::jsonl::<Row>(&line).unwrap()[0].input, "Zoë Example");
}

#[test]
fn complete_staff_table_source_copy_and_original_capture_are_bound() {
    let fixture = staff_fixture();
    let identity = fixture.verify().unwrap();
    assert_eq!(identity["full_source_copy_reconstructed"], true);
    assert_eq!(identity["not_native_origin"], true);
    assert_eq!(fixture.row.expected.len(), 4);
}

#[test]
fn staff_maps_reject_omitted_cells_forged_origin_and_gold() {
    for mode in 0..5 {
        let mut fixture = staff_fixture();
        fixture.verify().unwrap();
        match mode {
            0 => {
                fixture.map["literal_copies"].as_array_mut().unwrap().pop();
            }
            1 => {
                fixture.prepared["origin"]["source_family_id"] = json!("invented-family");
            }
            2 => {
                fixture.row.expected.remove(0);
            }
            3 => {
                fixture.row.authored_source.native_name.push_str("-wrong");
            }
            _ => {
                fixture.map["meaning_changed"] = json!(true);
            }
        }
        assert!(fixture.verify().is_err());
    }
}

#[test]
fn whole_source_reflow_retains_all_bytes_and_mapped_person_org_spans() {
    let fixture = reflow_fixture(true);
    let identity = fixture.verify().unwrap();
    assert_eq!(identity["full_source_copy_reconstructed"], true);
    assert_eq!(identity["not_native_origin"], true);
    assert_eq!(identity["original_source_url_recorded"], true);
    assert_eq!(
        fixture.row.input.replace('\n', ""),
        fixture.prepared["input"].as_str().unwrap()
    );
}

#[test]
fn reflow_rejects_partial_source_maps_added_content_and_moved_gold() {
    for mode in 0..5 {
        let mut fixture = reflow_fixture(true);
        fixture.verify().unwrap();
        match mode {
            0 => {
                fixture.map["literal_copy_map"]
                    .as_array_mut()
                    .unwrap()
                    .pop();
            }
            1 => {
                fixture.map["inserted_whitespace_or_delimiters"][0]["literal"] = json!(" NEW ORG ");
            }
            2 => {
                fixture.map["source_row_sha256"] = json!(digest(b"wrong row"));
            }
            3 => {
                fixture.row.expected[0].start += 1;
            }
            _ => {
                fixture.map["source_context_evidence"][0]["quote"] = json!("invented context");
            }
        }
        assert!(fixture.verify().is_err());
    }
}

#[test]
fn original_native_id_is_required_and_absent_source_url_is_not_fabricated() {
    let mut fixture = reflow_fixture(false);
    let identity = fixture.verify().unwrap();
    assert_eq!(identity["original_source_url_recorded"], false);
    assert!(identity["original_source_url"].is_null());
    fixture.prepared["origin"]["row_id"] = json!("invented-id");
    let bytes = json_line(&fixture.prepared);
    let pin = &mut fixture.row.authored_source.native_dataset;
    pin.sha256 = digest(&bytes);
    fixture.files.0.insert(pin.path.clone(), bytes);
    fixture.map["source_dataset"] = serde_json::to_value(pin).unwrap();
    fixture.map["source_row_sha256"] = json!(digest(
        serde_json::to_string(&fixture.prepared).unwrap().as_bytes()
    ));
    let error = fixture.verify().unwrap_err();
    assert!(
        error
            .to_string()
            .contains("full native original id/text/source URL binding differs")
    );
}

#[test]
fn token_mask_paragraph_limit_and_postal_end_failures_are_rejected() {
    let long = vec!["Name"; 257].join(" ");
    let postal = "123 Main Street, Denver, CO 80202-1234";
    for bad in [
        row("Jane Doe", vec![span("person", "Jane Doe", 1, 4)]),
        row(
            "jane@example.org",
            vec![span("org", "jane@example.org", 0, 16)],
        ),
        row("Jane\n\nDoe", vec![span("person", "Jane\n\nDoe", 0, 9)]),
        row(&long, vec![span("person", &long, 0, long.len())]),
        row(postal, vec![span("address", postal, 0, postal.len() - 5)]),
    ] {
        for contract in [InputContract::Legacy23, InputContract::TabCells25] {
            assert!(loader::encode(&bad, &FeatureConfig::default(), contract).is_err());
        }
    }
}

#[test]
fn canonical_features_are_gold_independent_and_contacts_never_become_neural_targets() {
    let text = "Jane Doe\tChief\nExample Office\nEmail jane@example.org\nPhone 207-555-0100";
    let supervised = row(
        text,
        vec![
            named_span("person", text, "Jane Doe"),
            named_span("org", text, "Example Office"),
            named_span("email", text, "jane@example.org"),
            named_span("phone", text, "207-555-0100"),
        ],
    );
    let blank = row(text, vec![]);
    for contract in [InputContract::Legacy23, InputContract::TabCells25] {
        let (doc, proof) =
            loader::encode(&supervised, &FeatureConfig::default(), contract).unwrap();
        let (empty, _) = loader::encode(&blank, &FeatureConfig::default(), contract).unwrap();
        assert_eq!(doc.gold.len(), 2);
        let mut enc = doc.enc;
        enc.labels.fill(0);
        assert_eq!(enc, empty.enc);
        assert_eq!(proof["gold_independent_all_input_arrays"], true);
        assert_eq!(proof["all_neural_gold_reachable"], true);
    }
}

#[test]
fn tab_only_variants_require_actual_selected_contract_feature_changes() {
    let variant = row("Jane Doe, \tChief", vec![]);
    assert!(
        loader::compare_native_features(
            &variant,
            "Jane Doe, Chief",
            &FeatureConfig::default(),
            InputContract::Legacy23
        )
        .is_err()
    );
    let proof = loader::compare_native_features(
        &variant,
        "Jane Doe, Chief",
        &FeatureConfig::default(),
        InputContract::TabCells25,
    )
    .unwrap();
    assert!(proof["difference_counts"]["flags"].as_u64().unwrap() > 0);
    let (legacy, _) =
        loader::encode(&variant, &FeatureConfig::default(), InputContract::Legacy23).unwrap();
    let (tabs, _) = loader::encode(
        &variant,
        &FeatureConfig::default(),
        InputContract::TabCells25,
    )
    .unwrap();
    assert_eq!(legacy.enc.ngram_ids, tabs.enc.ngram_ids);
    assert_eq!(legacy.enc.token_spans, tabs.enc.token_spans);
    assert_eq!(
        legacy.enc.flags,
        tabs.enc
            .flags
            .iter()
            .map(|flag| flag & ((1 << 23) - 1))
            .collect::<Vec<_>>()
    );
}
