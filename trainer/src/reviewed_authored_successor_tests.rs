use super::*;
use std::path::{Path, PathBuf};
use tessera::internal::FeatureConfig;

fn pin(path: &str) -> Receipt {
    Receipt {
        path: PathBuf::from(path),
        sha256: digest(path.as_bytes()),
    }
}

fn span(kind: &str, input: &str, text: &str) -> Span {
    let start = input.find(text).unwrap();
    Span {
        kind: kind.to_owned(),
        text: text.to_owned(),
        start: start as u32,
        end: (start + text.len()) as u32,
    }
}

fn jsonl_bytes(rows: &[Value]) -> Vec<u8> {
    rows.iter()
        .map(|row| serde_json::to_string(row).unwrap() + "\n")
        .collect::<String>()
        .into_bytes()
}

/// Insert a newline at each source boundary, with the literal copy map of the result.
fn reflow(input: &str, boundaries: &[usize]) -> (String, Vec<Value>) {
    let mut output = String::new();
    let mut copies = vec![];
    let mut previous = 0;
    for &end in boundaries.iter().chain(std::iter::once(&input.len())) {
        let start = output.len();
        output.push_str(&input[previous..end]);
        copies.push(json!({"source_start":previous,"source_end":end,
            "output_start":start,"output_end":output.len()}));
        if end < input.len() {
            output.push('\n');
        }
        previous = end;
    }
    (output, copies)
}

/// Contact-line boundaries: after every comma and semicolon.
fn punctuation(input: &str) -> Vec<usize> {
    input
        .bytes()
        .enumerate()
        .filter_map(|(index, byte)| matches!(byte, b',' | b';').then_some(index + 1))
        .collect()
}

const CHANGED: &str = "Jane Doe, Chief; Example Office. See the Board site.";
const UNCHANGED: &str = "Mary Major, Clerk; Other Office.";
const BOARD_ID: &str = "native-a:add:board:org";
const BOARD_BASIS: &str = "Established alias of the named board.";

struct Fixture {
    successor: Successor,
    native: Vec<Value>,
    corrected: Vec<Value>,
    reviewed: Vec<Value>,
    prepared: Vec<Value>,
    maps: Vec<Value>,
    case_decisions: Vec<Value>,
}

impl Fixture {
    fn verify(&self) -> anyhow::Result<Vec<Value>> {
        self.successor.verify(&Evidence {
            source_bytes: &jsonl_bytes(&self.native),
            corrected_bytes: &jsonl_bytes(&self.corrected),
            reviewed_bytes: &jsonl_bytes(&self.reviewed),
            prepared_bytes: &jsonl_bytes(&self.prepared),
            maps: &self.maps,
            case_decisions: &self.case_decisions,
        })
    }

    fn error(&self) -> String {
        self.verify().unwrap_err().to_string()
    }

    fn edit_transfer(&mut self, edit: impl FnOnce(&mut Value)) {
        let mut value = serde_json::to_value(&self.successor.transfer).unwrap();
        edit(&mut value);
        self.successor.transfer = serde_json::from_value(value).unwrap();
    }
}

fn native_row(name: &str, input: &str, expected: Vec<Span>) -> Value {
    json!({"name":name,"country":"US","input":input,"expected":expected,"status":"resolved",
        "origin":{"row_id":name}})
}

fn authored_row(name: &str, native: &str, index: usize, input: &str, expected: Vec<Span>) -> Value {
    json!({"name":name,"country":"US","input":input,"expected":expected,"status":"resolved",
        "authored_source":{"native_name":native,"native_row_index":index,"format":"contact_lines"}})
}

fn line_sha(row: &Value) -> String {
    digest(serde_json::to_string(row).unwrap().as_bytes())
}

/// Real-shaped amendment application with one segment and the given changes.
fn application_value(adjudication: &Receipt, manifest: &Receipt, changes: Value) -> Value {
    json!({"version":APPLICATION_VERSION, "status":"label successor pending canonical rebuild",
        "manifest":manifest, "adjudication":adjudication, "native_documents":2,
        "segments":[{"original":pin("/fixture/native.jsonl"),"corrected":pin("/fixture/corrected.jsonl"),
            "documents":2}],
        "changes":changes, "before_counts":{"org":2,"person":2}, "after_counts":{"org":3,"person":2},
        "canonical_features_regenerated":false, "independent_successor_validation_complete":false,
        "training_ready":false, "training_started":false, "limits":["synthetic"]})
}

fn board_change(input: &str) -> Value {
    json!([{"name":"native-a","input_sha256":digest(input.as_bytes()),"added":[span("org", input, "Board")],
        "removed":[],"reviewed_kinds":["org"],"adjudication_ids":[BOARD_ID]}])
}

fn fixture() -> Fixture {
    fixture_from(CHANGED, punctuation)
}

fn fixture_from(changed: &str, boundaries: fn(&str) -> Vec<usize>) -> Fixture {
    let board = span("org", changed, "Board");
    let person = changed.split(',').next().unwrap();
    let native_a = vec![
        span("person", changed, person),
        span("org", changed, "Example Office"),
    ];
    let native_b = vec![
        span("person", UNCHANGED, "Mary Major"),
        span("org", UNCHANGED, "Other Office"),
    ];
    let native = vec![
        native_row("native-a", changed, native_a.clone()),
        native_row("native-b", UNCHANGED, native_b.clone()),
    ];
    let mut corrected = native.clone();
    corrected[0]["expected"] = json!([native_a[0], native_a[1], board]);
    let (authored_a, copies_a) = reflow(changed, &boundaries(changed));
    let (authored_b, copies_b) = reflow(UNCHANGED, &punctuation(UNCHANGED));
    let authored_board = span("org", &authored_a, "Board");
    let reviewed = vec![
        authored_row(
            "authored-a",
            "native-a",
            0,
            &authored_a,
            vec![
                span("person", &authored_a, person),
                span("org", &authored_a, "Example Office"),
            ],
        ),
        authored_row(
            "authored-b",
            "native-b",
            1,
            &authored_b,
            vec![
                span("person", &authored_b, "Mary Major"),
                span("org", &authored_b, "Other Office"),
            ],
        ),
    ];
    let mut prepared = reviewed.clone();
    prepared[0]["expected"] = json!([
        span("person", &authored_a, person),
        span("org", &authored_a, "Example Office"),
        authored_board
    ]);
    let maps = vec![
        json!({"name":"authored-a","native_source_document":"native-a","literal_copy_map":copies_a}),
        json!({"name":"authored-b","native_source_document":"native-b","literal_copy_map":copies_b}),
    ];
    let case_decisions = vec![json!({"authored_name":"authored-a","span":authored_board,
        "decision":"exclude_generic_anaphoric_reference"})];
    let source = pin("/fixture/native.jsonl");
    let corrected_pin = pin("/fixture/corrected.jsonl");
    let changed_sha = digest(changed.as_bytes());
    let application: Application = serde_json::from_value(application_value(
        &pin("/fixture/adjudication.jsonl"),
        &pin("/fixture/amendment-manifest.json"),
        board_change(changed),
    ))
    .unwrap();
    let native_decisions = vec![
        serde_json::from_value(json!({"id":BOARD_ID,"name":"native-a","decision":"add",
            "span":board,"input_sha256":changed_sha,"basis":BOARD_BASIS}))
        .unwrap(),
    ];
    let transfer: Transfer = serde_json::from_value(json!({
        "schema":TRANSFER_SCHEMA, "original_manifest":pin("/fixture/original-manifest.json"),
        "original_prepared":pin("/fixture/original.jsonl"),
        "amendment_application":pin("/fixture/application.json"),
        "native_adjudication":pin("/fixture/adjudication.jsonl"),
        "source_segment":{"original":source,"corrected":corrected_pin},
        "parents":[
            {"name":"native-a","native_row_index":0,"input_sha256":changed_sha,
             "original_line_sha256":line_sha(&native[0]),"successor_line_sha256":line_sha(&corrected[0]),
             "added":[board],"removed":[],"adjudication_ids":[BOARD_ID]},
            {"name":"native-b","native_row_index":1,"input_sha256":digest(UNCHANGED.as_bytes()),
             "original_line_sha256":line_sha(&native[1]),"successor_line_sha256":line_sha(&corrected[1]),
             "added":[],"removed":[],"adjudication_ids":[]}],
        "rows":[{"name":"authored-a","native_name":"native-a","spans":[
            {"action":"add","adjudication_id":BOARD_ID,"native":board,"authored":authored_board}]}],
        "superseded_case_decisions":[{"authored_name":"authored-a","span":authored_board,
            "superseded_decision":"exclude_generic_anaphoric_reference","superseded_by":BOARD_ID,
            "superseding_basis":BOARD_BASIS}]}))
    .unwrap();
    let original_counts = BTreeMap::from([("person".to_owned(), 2), ("org".to_owned(), 2)]);
    Fixture {
        successor: Successor {
            transfer_pin: pin("/fixture/transfer.json"),
            application_record: json!({"status":"synthetic"}),
            application,
            native_decisions,
            transfer,
            original_counts,
        },
        native,
        corrected,
        reviewed,
        prepared,
        maps,
        case_decisions,
    }
}

fn manifest_value() -> Value {
    json!({
        "schema":"reviewed-whole-source-authored-data-v1", "origin":"authored_train_only",
        "split":"train", "documents":1, "counts":{},
        "prepared":pin("/fixture/original.jsonl"), "policy":pin("/fixture/policy.json"),
        "receipt_pins":{"/fixture/final-adjudication-root.json":digest(b"adjudication")},
        "finalizer_sha256":digest(b"finalizer"), "all_source_bytes_and388_notes_reviewed":true,
        "label_passes":"synthetic", "semantic_review":"synthetic",
        "native_train_documents_changed":false, "fit_route_ready":false, "fit_allowed":false,
        "remaining":["test fixture only"], "limits":["invented text"]
    })
}

fn successor_manifest_value() -> Value {
    let mut value = manifest_value();
    value["prepared"] = json!(pin("/fixture/successor.jsonl"));
    value["native_label_successor"] = json!({"amendment_application":pin("/fixture/application.json"),
        "transfer":pin("/fixture/transfer.json")});
    value["limits"] = json!(["invented text", "native labels transferred"]);
    value
}

#[test]
fn legacy_manifest_without_successor_keeps_the_original_route() {
    let manifest: Manifest = serde_json::from_value(manifest_value()).unwrap();
    assert!(manifest.native_label_successor.is_none());
    super::super::loader::verify_manifest(&manifest).unwrap();
    assert!(
        open(&manifest, &mut Dependencies::default())
            .unwrap()
            .is_none()
    );
    let mut null = manifest_value();
    null["native_label_successor"] = Value::Null;
    assert!(serde_json::from_value::<Manifest>(null).is_err());
    let mut unknown = successor_manifest_value();
    unknown["native_label_successor"]["original_rows"] = json!(pin("/fixture/other.jsonl"));
    assert!(serde_json::from_value::<Manifest>(unknown).is_err());
}

#[test]
fn successor_manifest_may_change_only_rows_counts_finalizer_and_added_limits() {
    let original: Manifest = serde_json::from_value(manifest_value()).unwrap();
    let successor: Manifest = serde_json::from_value(successor_manifest_value()).unwrap();
    verify_original_manifest(&successor, &original, &original.prepared).unwrap();
    for (key, bad) in [
        (
            "receipt_pins",
            json!({"/fixture/other.json":digest(b"other")}),
        ),
        ("policy", json!(pin("/fixture/other-policy.json"))),
        ("limits", json!(["native labels transferred"])),
        ("fit_allowed", json!(true)),
        ("prepared", json!(pin("/fixture/original.jsonl"))),
    ] {
        let mut changed = successor_manifest_value();
        changed[key] = bad;
        let changed: Manifest = serde_json::from_value(changed).unwrap();
        assert!(verify_original_manifest(&changed, &original, &original.prepared).is_err());
    }
    let chained: Manifest = serde_json::from_value(successor_manifest_value()).unwrap();
    assert!(verify_original_manifest(&successor, &chained, &chained.prepared).is_err());
}

#[test]
fn successor_with_exact_transfer_is_accepted_and_records_superseded_rulings() {
    let fixture = fixture();
    let identities = fixture.verify().unwrap();
    assert_eq!(identities.len(), 2);
    let superseded = identities[0]["superseded_case_decisions"]
        .as_array()
        .unwrap();
    assert_eq!(superseded.len(), 1);
    assert_eq!(
        superseded[0]["superseded_decision"],
        "exclude_generic_anaphoric_reference"
    );
    assert_eq!(superseded[0]["superseded_by"], BOARD_ID);
    assert_eq!(
        identities[0]["transferred_native_changes"][0]["authored"]["start"],
        CHANGED.find("Board").unwrap() + 2
    );
    assert!(
        identities[1]["transferred_native_changes"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn successor_segment_replaces_only_the_original_it_was_derived_from() {
    let fixture = fixture();
    let source = pin("/fixture/native.jsonl");
    let corrected = pin("/fixture/corrected.jsonl");
    let chosen = fixture
        .successor
        .native_source(&source, std::slice::from_ref(&corrected))
        .unwrap();
    assert!(same(&chosen, &corrected));
    let other = pin("/fixture/other-native.jsonl");
    assert!(
        fixture
            .successor
            .native_source(&other, std::slice::from_ref(&corrected))
            .is_err()
    );
    assert!(
        fixture
            .successor
            .native_source(&source, &[source.clone(), corrected.clone()])
            .is_err()
    );
    let mut remapped = fixture;
    remapped.successor.application.segments[0].original = other;
    let error = remapped
        .successor
        .native_source(&source, &[corrected])
        .unwrap_err();
    assert!(error.to_string().contains("does not map the original"));
}

#[test]
fn extra_missing_or_shifted_authored_labels_are_rejected() {
    let mut extra = fixture();
    let input = extra.prepared[1]["input"].as_str().unwrap().to_owned();
    extra.prepared[1]["expected"]
        .as_array_mut()
        .unwrap()
        .push(json!(span("person", &input, "Clerk")));
    assert!(extra.error().contains("successor authored row differs"));

    let mut missing = fixture();
    missing.prepared[0]["expected"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert!(missing.error().contains("successor authored row differs"));

    let mut shifted = fixture();
    let input = shifted.prepared[0]["input"].as_str().unwrap().to_owned();
    let start = CHANGED.find("Board").unwrap();
    assert_ne!(&input[start..start + 5], "Board");
    shifted.prepared[0]["expected"][2] = json!({"kind":"org","text":&input[start..start + 5],
        "start":start,"end":start + 5});
    assert!(shifted.error().contains("successor authored row differs"));

    let mut field = fixture();
    field.prepared[0]["country"] = json!("CA");
    assert!(field.error().contains("successor authored row differs"));

    let mut reserialized = fixture();
    reserialized.prepared[1]["expected"]
        .as_array_mut()
        .unwrap()
        .reverse();
    assert!(reserialized.error().contains("not byte identical"));
}

#[test]
fn successor_native_rows_may_change_only_their_amended_labels() {
    let mut changed_field = fixture();
    changed_field.corrected[0]["status"] = json!("quarantine");
    assert!(
        changed_field
            .error()
            .contains("more than its amended labels")
    );

    let mut unchanged_parent = fixture();
    unchanged_parent.corrected[1]["origin"]["row_id"] = json!("other");
    assert!(unchanged_parent.error().contains("not byte identical"));

    let mut stale_labels = fixture();
    stale_labels.corrected[0]["expected"] = stale_labels.native[0]["expected"].clone();
    assert!(
        stale_labels
            .error()
            .contains("original plus the application change")
    );
}

#[test]
fn application_change_without_authored_transfer_is_rejected() {
    let mut fixture = fixture();
    let chief = span("org", CHANGED, "Chief");
    let id = "native-a:add:chief:org";
    fixture.successor.application.changes[0]
        .added
        .push(chief.clone());
    fixture.successor.application.changes[0]
        .adjudication_ids
        .push(id.to_owned());
    fixture.successor.native_decisions.push(NativeDecision {
        id: id.to_owned(),
        name: "native-a".to_owned(),
        decision: "add".to_owned(),
        span: chief.clone(),
        input_sha256: digest(CHANGED.as_bytes()),
        basis: "Reviewed.".to_owned(),
    });
    fixture.corrected[0]["expected"]
        .as_array_mut()
        .unwrap()
        .push(json!(chief));
    assert!(fixture.error().contains("successor authored row differs"));
}

#[test]
fn unadjudicated_changes_and_unrecorded_supersessions_are_rejected() {
    let mut unadjudicated = fixture();
    unadjudicated.successor.native_decisions[0].decision = "remove".to_owned();
    assert!(
        unadjudicated
            .error()
            .contains("lacks exactly one adjudicated decision")
    );

    let mut unrecorded = fixture();
    unrecorded
        .successor
        .transfer
        .superseded_case_decisions
        .clear();
    assert!(
        unrecorded
            .error()
            .contains("declared native label transfer")
    );

    let mut outside_copy = fixture();
    outside_copy.maps[0]["literal_copy_map"]
        .as_array_mut()
        .unwrap()
        .pop();
    assert!(outside_copy.error().contains("not inside one literal copy"));
}

const OFFICE_ID: &str = "native-a:remove:office:org";
const OFFICE_BASIS: &str = "Generic office reference, not a proper name.";

/// Extend the fixture with a reviewed removal of "Example Office" that contradicts a retain ruling.
fn with_removal(fixture: &mut Fixture) {
    let input = fixture.reviewed[0]["input"].as_str().unwrap().to_owned();
    let native = span("org", CHANGED, "Example Office");
    let authored = span("org", &input, "Example Office");
    let change = &mut fixture.successor.application.changes[0];
    change.removed.push(native.clone());
    change.adjudication_ids.push(OFFICE_ID.to_owned());
    fixture.successor.native_decisions.push(NativeDecision {
        id: OFFICE_ID.to_owned(),
        name: "native-a".to_owned(),
        decision: "remove".to_owned(),
        span: native.clone(),
        input_sha256: digest(CHANGED.as_bytes()),
        basis: OFFICE_BASIS.to_owned(),
    });
    for rows in [&mut fixture.corrected, &mut fixture.prepared] {
        rows[0]["expected"]
            .as_array_mut()
            .unwrap()
            .retain(|span| span["text"] != "Example Office");
    }
    fixture
        .case_decisions
        .push(json!({"authored_name":"authored-a","span":authored,
        "decision":"retain_established_distinctive_proper_short_name"}));
    let successor_line = line_sha(&fixture.corrected[0]);
    fixture.edit_transfer(|transfer| {
        transfer["parents"][0]["removed"] = json!([native]);
        transfer["parents"][0]["adjudication_ids"] = json!([BOARD_ID, OFFICE_ID]);
        transfer["parents"][0]["successor_line_sha256"] = json!(successor_line);
        transfer["rows"][0]["spans"].as_array_mut().unwrap().insert(
            0,
            json!({"action":"remove","adjudication_id":OFFICE_ID,"native":native,"authored":authored}),
        );
        transfer["superseded_case_decisions"]
            .as_array_mut()
            .unwrap()
            .insert(
                0,
                json!({"authored_name":"authored-a","span":authored,
                    "superseded_decision":"retain_established_distinctive_proper_short_name",
                    "superseded_by":OFFICE_ID,"superseding_basis":OFFICE_BASIS}),
            );
    });
}

#[test]
fn reviewed_removals_transfer_and_supersede_retain_rulings() {
    let mut fixture = fixture();
    with_removal(&mut fixture);
    let identities = fixture.verify().unwrap();
    let superseded = identities[0]["superseded_case_decisions"]
        .as_array()
        .unwrap();
    assert_eq!(superseded.len(), 2);
    assert_eq!(
        superseded[0]["superseded_decision"],
        "retain_established_distinctive_proper_short_name"
    );
    assert_eq!(superseded[0]["superseding_basis"], OFFICE_BASIS);
    assert_eq!(superseded[1]["superseding_basis"], BOARD_BASIS);

    let mut kept = fixture;
    let input = kept.prepared[0]["input"].as_str().unwrap().to_owned();
    kept.prepared[0]["expected"]
        .as_array_mut()
        .unwrap()
        .push(json!(span("org", &input, "Example Office")));
    assert!(kept.error().contains("successor authored row differs"));
}

#[test]
fn transferred_change_agreeing_with_an_existing_ruling_is_rejected() {
    let mut fixture = fixture();
    fixture.case_decisions[0]["decision"] =
        json!("retain_established_distinctive_proper_short_name");
    assert!(fixture.error().contains("repeats an existing case ruling"));
}

#[test]
fn edited_transfer_records_and_original_counts_are_rejected() {
    let mut shifted = fixture();
    shifted.edit_transfer(|transfer| {
        let authored = &mut transfer["rows"][0]["spans"][0]["authored"];
        authored["start"] = json!(authored["start"].as_u64().unwrap() + 1);
        authored["end"] = json!(authored["end"].as_u64().unwrap() + 1);
    });
    assert!(shifted.error().contains("declared native label transfer"));

    let mut line = fixture();
    line.edit_transfer(|transfer| {
        transfer["parents"][0]["successor_line_sha256"] = json!(digest(b"other"));
    });
    assert!(line.error().contains("declared native label transfer"));

    let mut counts = fixture();
    counts.successor.original_counts.insert("org".to_owned(), 3);
    assert!(counts.error().contains("original authored counts differ"));
}

#[test]
fn non_ascii_text_before_a_soft_wrapped_change_uses_byte_offsets() {
    const WRAPPED: &str = "Zoë Doe, Chief; Example Office. See the Board site.";
    let fixture = fixture_from(WRAPPED, |input| {
        vec![
            input.find("Chief").unwrap(),
            input.find("the Board").unwrap(),
        ]
    });
    let identities = fixture.verify().unwrap();
    let native = WRAPPED.find("Board").unwrap();
    assert_ne!(native, WRAPPED[..native].chars().count());
    let authored = &identities[0]["transferred_native_changes"][0]["authored"];
    assert_eq!(authored["start"], native + 2);
    assert_eq!(authored["text"], "Board");
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Break {
    Nothing,
    TransferApplicationPin,
    AdjudicationPin,
    TransferSchema,
    ApplicationVersion,
    ApplicationManifestPin,
    UnknownApplicationField,
    ApplicationCounts,
    DuplicateChange,
    DuplicateDecisionId,
    NonAdjudicatedRoute,
    OriginalPrepared,
}

/// Write a complete successor pin set to disk, broken in one way, and open it.
fn open_case(broken: Break) -> anyhow::Result<Option<Successor>> {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("original")).unwrap();
    let write = |name: &str, bytes: &[u8]| {
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).unwrap();
        Receipt {
            path,
            sha256: digest(bytes),
        }
    };
    let decision = json!({"id":BOARD_ID,"name":"native-a","decision":"add",
        "span":span("org", CHANGED, "Board"),"input_sha256":digest(CHANGED.as_bytes()),"basis":BOARD_BASIS});
    let mut decisions = vec![decision.clone()];
    if broken == Break::DuplicateDecisionId {
        decisions.push(decision);
    }
    let adjudication = write("adjudication.jsonl", &jsonl_bytes(&decisions));
    let mut amendment_manifest = write("amendment-manifest.json", b"{}");
    if broken == Break::ApplicationManifestPin {
        amendment_manifest.sha256 = digest(b"other");
    }
    let mut application =
        application_value(&adjudication, &amendment_manifest, board_change(CHANGED));
    match broken {
        Break::ApplicationVersion => application["version"] = json!("native-label-amendments-v1"),
        Break::UnknownApplicationField => application["unreviewed"] = json!(true),
        Break::ApplicationCounts => application["after_counts"]["org"] = json!(4),
        Break::DuplicateChange => {
            let change = application["changes"][0].clone();
            application["changes"].as_array_mut().unwrap().push(change);
        }
        _ => {}
    }
    let application = write(
        "application.json",
        &serde_json::to_vec(&application).unwrap(),
    );
    let finalizer = write("original/finalize.py", b"# synthetic finalizer\n");
    let original_prepared = write("original/rows.jsonl", b"{}\n");
    let mut original = manifest_value();
    original["prepared"] = json!(original_prepared);
    original["finalizer_sha256"] = json!(finalizer.sha256);
    let original = write(
        "original/manifest.json",
        &serde_json::to_vec(&original).unwrap(),
    );
    let mut transfer = json!({"schema":TRANSFER_SCHEMA,"original_manifest":original,
        "original_prepared":original_prepared,"amendment_application":application,
        "native_adjudication":adjudication,
        "source_segment":{"original":pin("/fixture/native.jsonl"),"corrected":pin("/fixture/corrected.jsonl")},
        "parents":[],"rows":[],"superseded_case_decisions":[]});
    match broken {
        Break::TransferApplicationPin => {
            transfer["amendment_application"]["sha256"] = json!(digest(b"other"))
        }
        Break::AdjudicationPin => {
            transfer["native_adjudication"] = json!(pin("/fixture/other.jsonl"))
        }
        Break::TransferSchema => {
            transfer["schema"] = json!("reviewed-authored-native-label-transfer-v0")
        }
        Break::OriginalPrepared => transfer["original_prepared"] = json!(finalizer),
        _ => {}
    }
    let transfer = write("transfer.json", &serde_json::to_vec(&transfer).unwrap());
    let mut successor = successor_manifest_value();
    successor["native_label_successor"] =
        json!({"amendment_application":application,"transfer":transfer});
    if broken == Break::NonAdjudicatedRoute {
        successor["all_source_bytes_and388_notes_reviewed"] = json!(false);
        successor["all_source_semantics_and_438_notes_reviewed"] = json!(true);
    }
    let manifest: Manifest = serde_json::from_value(successor).unwrap();
    open(&manifest, &mut Dependencies::default())
}

#[test]
fn open_binds_transfer_application_adjudication_and_original_packet() {
    let opened = open_case(Break::Nothing).unwrap().unwrap();
    assert_eq!(
        opened.application_record["status"],
        "label successor pending canonical rebuild"
    );
    assert_eq!(
        opened.application_record["independent_successor_validation_complete"],
        false
    );
    let binding = "schema or amendment application binding differs";
    for (broken, reason) in [
        (Break::TransferApplicationPin, binding),
        (Break::AdjudicationPin, binding),
        (Break::TransferSchema, binding),
        (Break::ApplicationVersion, binding),
        (Break::ApplicationManifestPin, "receipt changed"),
        (Break::UnknownApplicationField, "unknown field `unreviewed`"),
        (Break::ApplicationCounts, "counts differ from its changes"),
        (Break::DuplicateChange, "repeats a native document"),
        (Break::DuplicateDecisionId, "repeats a decision id"),
        (
            Break::NonAdjudicatedRoute,
            "source-adjudicated authored route",
        ),
        (Break::OriginalPrepared, "beyond its rows"),
    ] {
        let error = open_case(broken).err().map(|error| format!("{error:#}"));
        assert!(
            error.as_deref().is_some_and(|error| error.contains(reason)),
            "{broken:?}: {error:?}"
        );
    }
}

fn repository() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_owned()
}

fn receipt(path: &str) -> Receipt {
    let path = repository().join(path);
    let sha256 = digest(&std::fs::read(&path).unwrap());
    Receipt { path, sha256 }
}

fn fixed1200_features() -> FeatureConfig {
    FeatureConfig {
        ngram_sizes: vec![2, 3, 4],
        hash_buckets: 32768,
        hash_seed: 0,
    }
}

/// Uses the local `runs/us95` packets with the fixed1200 run's feature configuration.
#[test]
#[ignore = "requires the local runs/us95 authored and native packets"]
fn real_successor_and_original_packets_load_against_their_native_compositions() {
    let root = repository();
    let contract = super::super::InputContract::TabCells25;
    let successor_manifest =
        receipt("runs/us95/unit-contact-reflow-successor-v1/reviewed-authored-manifest.json");
    let original_manifest =
        receipt("runs/us95/unit-contact-reflow-root-v1/reviewed-authored-manifest.json");
    let successor_native = [
        receipt("runs/us95/native-label-successor-v1/corrected/native-segment-00.jsonl"),
        receipt("runs/us95/native-label-successor-v1/corrected/native-segment-01.jsonl"),
        receipt("runs/us95/native-label-successor-v1/corrected/native-segment-02.jsonl"),
    ];
    let original_native = [
        receipt("runs/us95/pilot-inputs/train-prepared.jsonl"),
        receipt("runs/us95/format-gap-candidates/native-data-check-119/train119-prepared.jsonl"),
        receipt(
            "runs/us95/no-org-data-candidates-v1/native-data-check-washington43-b/train43-prepared.jsonl",
        ),
    ];
    let load = |manifest: &Receipt, native: &[Receipt]| {
        super::super::load(manifest, &root, native, &fixed1200_features(), contract)
    };
    let successor = load(&successor_manifest, &successor_native).unwrap();
    let original = load(&original_manifest, &original_native).unwrap();
    assert!(load(&successor_manifest, &original_native).is_err());
    assert!(load(&original_manifest, &successor_native).is_err());
    assert_eq!(successor.docs.len(), 24);
    assert_eq!(original.docs.len(), 24);
    let changed: Vec<_> = (0..24)
        .filter(|&i| successor.expected[i] != original.expected[i])
        .map(|i| successor.names[i].clone())
        .collect();
    assert_eq!(changed.len(), 4);
    assert!(
        changed
            .iter()
            .all(|name| name.ends_with("seen-real-30-0012"))
    );
    for i in 0..24 {
        let added: Vec<_> = successor.expected[i]
            .iter()
            .filter(|span| !original.expected[i].contains(span))
            .collect();
        assert!(
            original.expected[i]
                .iter()
                .all(|span| successor.expected[i].contains(span))
        );
        if changed.contains(&successor.names[i]) {
            assert_eq!(added.len(), 1);
            assert_eq!(
                (added[0].kind.as_str(), added[0].text.as_str()),
                ("org", "Board")
            );
            assert_eq!(
                successor.identities[i]["native_label_successor"]["superseded_case_decisions"]
                    .as_array()
                    .unwrap()
                    .len(),
                1
            );
        } else {
            assert!(added.is_empty());
        }
        assert!(
            original.identities[i]
                .get("native_label_successor")
                .is_none()
        );
    }
}
