use super::*;
use crate::detector::KindSpan;
use serde_json::json;

const TEXT: &str = "the Respondent asked Jane Doe at 5551234567 for Respondent 555-0000 Acme Corp";
const SOLO: &str = "Respondent";

/// Whitespace tokens of `text`; the network sees only these retained spans.
fn doc(text: &str, labelled: &[(&str, u8)], ruled: &[&str]) -> DetectorDoc {
    let mut spans = Vec::new();
    let mut offset = 0;
    for word in text.split(' ') {
        spans.push((offset as u32, (offset + word.len()) as u32));
        offset += word.len() + 1;
    }
    let words: Vec<_> = text.split(' ').collect();
    let n = spans.len();
    DetectorDoc {
        text: text.into(),
        gold: Vec::new(),
        breaks: vec![false; n],
        enc: Encoded {
            token_spans: spans,
            ngram_ids: vec![vec![1]; n],
            script: vec![0; n],
            shape: vec![0; n],
            flags: words
                .iter()
                .map(|w| {
                    if ruled.contains(w) {
                        flag::IN_RULE_SPAN
                    } else {
                        0
                    }
                })
                .collect(),
            labels: words
                .iter()
                .map(|w| {
                    labelled
                        .iter()
                        .find(|(l, _)| l == w)
                        .map_or(0, |(_, id)| *id)
                })
                .collect(),
            country: "US".into(),
        },
    }
}

/// An additional named TRAIN case with its expected positive byte ranges.
type Extra = (&'static str, DetectorDoc, Vec<(u32, u32)>);

struct Fixture {
    dir: tempfile::TempDir,
    main: DetectorDoc,
    solo: DetectorDoc,
    main_positives: Vec<(u32, u32)>,
    extra: Vec<Extra>,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        for (name, body) in [
            ("reviewer-a.json", "{\"reviewer\":\"a\"}"),
            ("reviewer-b.json", "{\"reviewer\":\"b\"}"),
            ("adjudication.json", "{\"decision\":\"unresolved\"}"),
        ] {
            std::fs::write(dir.path().join(name), body).unwrap();
        }
        let mut main = doc(
            TEXT,
            &[("Jane", 1), ("Doe", 2), ("Acme", 3), ("Corp", 4)],
            &["5551234567", "555-0000"],
        );
        main.gold = vec![KindSpan {
            kind: 0,
            start: 21,
            end: 29,
        }];
        Self {
            dir,
            main,
            solo: doc(SOLO, &[], &[]),
            main_positives: vec![(21, 29), (33, 43), (68, 77)],
            extra: Vec::new(),
        }
    }

    fn receipt(&self, name: &str) -> Receipt {
        let path = self.dir.path().join(name);
        Receipt {
            sha256: digest(&std::fs::read(&path).unwrap()),
            path,
        }
    }

    fn cases(&self) -> Vec<Case<'_>> {
        let extra = self.extra.iter().map(|(name, doc, positives)| Case {
            name,
            doc,
            positives: positives.clone(),
        });
        vec![
            Case {
                name: "solo",
                doc: &self.solo,
                positives: Vec::new(),
            },
            Case {
                name: "seen-real-24-0004",
                doc: &self.main,
                positives: self.main_positives.clone(),
            },
        ]
        .into_iter()
        .chain(extra)
        .collect()
    }

    fn entry(&self, name: &str, start: u32, end: u32) -> Value {
        let text = match name {
            "solo" => SOLO,
            "seen-real-24-0004" => TEXT,
            other => self
                .extra
                .iter()
                .find(|(n, _, _)| *n == other)
                .map(|(_, d, _)| d.text.as_str())
                .unwrap(),
        };
        json!({"name":name,"input_sha256":digest(text.as_bytes()),"start":start,"end":end,
            "quote":text.get(start as usize..end as usize).unwrap_or("x"),
            "reason":"reviewers could not decide ORG versus generic party role"})
    }

    fn sidecar(&self, entries: Vec<Value>) -> Value {
        json!({"version":"reviewed-source-ambiguity-exclusions-v1",
            "reviewers":[self.receipt("reviewer-a.json"),self.receipt("reviewer-b.json")],
            "adjudication":self.receipt("adjudication.json"),"exclusions":entries})
    }

    fn write(&self, value: &Value) -> Receipt {
        let bytes = serde_json::to_vec(value).unwrap();
        let path = self
            .dir
            .path()
            .join(format!("sidecar-{}.json", digest(&bytes)));
        std::fs::write(&path, &bytes).unwrap();
        Receipt {
            path,
            sha256: digest(&bytes),
        }
    }

    fn resolve(&self, value: &Value) -> anyhow::Result<Resolved> {
        let receipt = self.write(value);
        load(&receipt)?.resolve(&receipt, &self.cases())
    }

    fn rejects(&self, value: &Value, message: &str) {
        let err = match self.resolve(value) {
            Ok(_) => panic!("accepted sidecar that should fail with {message}"),
            Err(err) => format!("{err:#}"),
        };
        assert!(err.contains(message), "{err} lacks {message}");
    }
}

#[test]
fn resolves_reviewed_respondent_mentions_to_whole_o_tokens_only() {
    let fx = Fixture::new();
    let resolved = fx
        .resolve(&fx.sidecar(vec![
            fx.entry("seen-real-24-0004", 48, 58),
            fx.entry("seen-real-24-0004", 4, 14),
        ]))
        .unwrap();
    let mut expected = vec![false; 12];
    expected[1] = true;
    expected[8] = true;
    assert_eq!(resolved.row(Some(1), 12).unwrap(), expected);
    assert_eq!(resolved.row(Some(0), 1).unwrap(), vec![false]);
    assert_eq!(resolved.row(None, 3).unwrap(), vec![false; 3]);
    assert!(resolved.row(Some(1), 11).is_err());
    assert_eq!(resolved.identity["excluded_unique_tokens"], 2);
    assert_eq!(resolved.identity["entries"][1]["tokens"], json!([1, 2]));
    assert_eq!(resolved.dependencies.len(), 4);
    resolved.verify().unwrap();
    assert_eq!(fx.main.enc.labels[1], 0);
}

#[test]
fn rejects_known_positive_rule_and_supervised_overlap() {
    let fx = Fixture::new();
    let positive = "overlaps a known expected positive span";
    fx.rejects(
        &fx.sidecar(vec![fx.entry("seen-real-24-0004", 21, 25)]),
        positive,
    );
    fx.rejects(
        &fx.sidecar(vec![fx.entry("seen-real-24-0004", 33, 43)]),
        positive,
    );
    fx.rejects(
        &fx.sidecar(vec![fx.entry("seen-real-24-0004", 68, 72)]),
        positive,
    );
    fx.rejects(
        &fx.sidecar(vec![fx.entry("seen-real-24-0004", 30, 43)]),
        positive,
    );
    fx.rejects(
        &fx.sidecar(vec![fx.entry("seen-real-24-0004", 59, 67)]),
        "rule-owned token",
    );
    let mut labelled = Fixture::new();
    labelled.main.enc.labels[2] = 5;
    labelled.rejects(
        &labelled.sidecar(vec![labelled.entry("seen-real-24-0004", 15, 20)]),
        "positively supervised token",
    );
}

#[test]
fn rejects_partial_tokens_quotes_unknown_cases_and_wrong_inputs() {
    let fx = Fixture::new();
    fx.rejects(
        &fx.sidecar(vec![fx.entry("seen-real-24-0004", 5, 14)]),
        "splits a token",
    );
    fx.rejects(
        &fx.sidecar(vec![fx.entry("seen-real-24-0004", 4, 13)]),
        "splits a token",
    );
    let mut quote = fx.entry("seen-real-24-0004", 4, 14);
    quote["quote"] = json!("respondent");
    fx.rejects(&fx.sidecar(vec![quote]), "quote differs");
    let mut unknown = fx.entry("seen-real-24-0004", 4, 14);
    unknown["name"] = json!("dev-case-0001");
    fx.rejects(&fx.sidecar(vec![unknown]), "unknown TRAIN case");
    let mut hash = fx.entry("seen-real-24-0004", 4, 14);
    hash["input_sha256"] = json!(digest(b"other input"));
    fx.rejects(&fx.sidecar(vec![hash]), "input hash differs");
}

#[test]
fn rejects_duplicate_overlapping_and_whole_row_exclusions() {
    let fx = Fixture::new();
    let entry = fx.entry("seen-real-24-0004", 4, 14);
    fx.rejects(&fx.sidecar(vec![entry.clone(), entry]), "repeat or overlap");
    fx.rejects(
        &fx.sidecar(vec![
            fx.entry("seen-real-24-0004", 4, 20),
            fx.entry("seen-real-24-0004", 15, 20),
        ]),
        "repeat or overlap",
    );
    let mut split = fx.entry("seen-real-24-0004", 48, 58);
    split["input_sha256"] = json!(digest(b"other input"));
    fx.rejects(
        &fx.sidecar(vec![fx.entry("seen-real-24-0004", 4, 14), split]),
        "bind different inputs",
    );
    fx.rejects(
        &fx.sidecar(vec![fx.entry("solo", 0, 10)]),
        "remove all loss",
    );
    fx.rejects(&fx.sidecar(Vec::new()), "declares no exclusions");
    let mut ruled = Fixture::new();
    ruled.solo = doc("Respondent 555-0000", &[], &["555-0000"]);
    let mut entry = ruled.entry("solo", 0, 10);
    entry["input_sha256"] = json!(digest(b"Respondent 555-0000"));
    entry["quote"] = json!("Respondent");
    ruled.rejects(&ruled.sidecar(vec![entry]), "remove all loss");
}

#[test]
fn rejects_unknown_fields_bad_receipts_and_repeated_reviewers() {
    let fx = Fixture::new();
    let valid = fx.sidecar(vec![fx.entry("seen-real-24-0004", 4, 14)]);
    fx.resolve(&valid).unwrap();
    let mut extra = valid.clone();
    extra["model_scores"] = json!([0.5]);
    fx.rejects(&extra, "strict versioned schema");
    let mut extra = valid.clone();
    extra["exclusions"][0]["label"] = json!("org");
    fx.rejects(&extra, "strict versioned schema");
    let mut version = valid.clone();
    version["version"] = json!("reviewed-source-ambiguity-exclusions-v2");
    fx.rejects(&version, "strict versioned schema");
    let mut reason = valid.clone();
    reason["exclusions"][0]["reason"] = json!(" ");
    fx.rejects(&reason, "reason");
    let mut bad = valid.clone();
    bad["reviewers"][1]["sha256"] = json!("0".repeat(64));
    fx.rejects(&bad, "receipt changed");
    let mut twice = valid.clone();
    twice["reviewers"][1] = twice["reviewers"][0].clone();
    fx.rejects(&twice, "three distinct files");
    let copy = fx.dir.path().join("reviewer-copy.json");
    std::fs::copy(fx.receipt("reviewer-a.json").path, &copy).unwrap();
    let mut copied = valid.clone();
    copied["reviewers"][1] = serde_json::to_value(fx.receipt("reviewer-copy.json")).unwrap();
    fx.rejects(&copied, "three distinct files");
    let mut adjudication = valid;
    adjudication["adjudication"] = adjudication["reviewers"][0].clone();
    fx.rejects(&adjudication, "three distinct files");
    let receipt = fx.write(&fx.sidecar(vec![fx.entry("seen-real-24-0004", 4, 14)]));
    std::fs::write(
        fx.dir.path().join("reviewer-b.json"),
        "{\"reviewer\":\"changed\"}",
    )
    .unwrap();
    assert!(load(&receipt).is_err());
}

#[test]
fn batch_rows_follow_native_owners_not_draw_positions() {
    let fx = Fixture::new();
    let resolved = fx
        .resolve(&fx.sidecar(vec![fx.entry("seen-real-24-0004", 4, 14)]))
        .unwrap();
    let items = vec![
        fx.solo.enc.clone(),
        fx.main.enc.clone(),
        fx.solo.enc.clone(),
    ];
    let rows = resolved.batch(&[Some(0), Some(1), None], &items).unwrap();
    assert_eq!(rows[0], vec![false]);
    assert!(rows[1][1] && rows[1].iter().filter(|&&x| x).count() == 1);
    assert_eq!(rows[2], vec![false]);
    let swapped = vec![fx.main.enc.clone(), fx.solo.enc.clone()];
    let rows = resolved.batch(&[Some(1), Some(0)], &swapped).unwrap();
    assert!(rows[0][1] && rows[1] == vec![false]);
    assert!(resolved.batch(&[Some(0), Some(1)], &items[..1]).is_err());
    assert!(
        resolved
            .batch(&[Some(1)], std::slice::from_ref(&fx.solo.enc))
            .is_err()
    );
}

/// A real canonical encoding, so separators and punctuation follow the actual tokenizer.
fn encoded(text: &str) -> DetectorDoc {
    let fc = crate::config::learning_test_config().features.to_tessera();
    crate::detector::annotated_document(text, "US", "[]", &fc).unwrap()
}

const ENCODED: &str = "Café señor: the Respondent Group, then call 555 0100.";

fn at(text: &str, quote: &str) -> (u32, u32) {
    let start = text.find(quote).unwrap();
    (start as u32, (start + quote.len()) as u32)
}

#[test]
fn real_encoding_excludes_multi_token_span_after_multibyte_text() {
    let mut fx = Fixture::new();
    let doc = encoded(ENCODED);
    let (start, end) = at(ENCODED, "Respondent Group");
    let before = &ENCODED[..start as usize];
    assert!(
        before.len() > before.chars().count(),
        "offsets are UTF-8 bytes"
    );
    let first = doc
        .enc
        .token_spans
        .iter()
        .position(|t| t.0 == start)
        .unwrap();
    let last = doc.enc.token_spans.iter().position(|t| t.1 == end).unwrap();
    assert_eq!(
        last,
        first + 1,
        "separator space must not be a retained token"
    );
    assert!(doc.enc.token_spans[first].1 < doc.enc.token_spans[last].0);
    fx.extra.push(("encoded", doc, Vec::new()));
    let resolved = fx
        .resolve(&fx.sidecar(vec![fx.entry("encoded", start, end)]))
        .unwrap();
    let row = resolved
        .row(Some(2), fx.extra[0].1.enc.token_spans.len())
        .unwrap();
    assert_eq!(
        row.iter()
            .enumerate()
            .filter(|(_, x)| **x)
            .map(|(i, _)| i)
            .collect::<Vec<_>>(),
        vec![first, last]
    );
    assert_eq!(
        resolved.identity["entries"][0]["tokens"],
        json!([first, last + 1])
    );
    let inside = ENCODED.find('é').unwrap() as u32 + 1;
    let mut split = fx.entry("encoded", inside, inside + 2);
    split["quote"] = json!("xx");
    fx.rejects(&fx.sidecar(vec![split]), "quote differs");
    let (start, _) = at(ENCODED, "Group");
    fx.rejects(
        &fx.sidecar(vec![fx.entry("encoded", start, start + 3)]),
        "splits a token",
    );
}

#[test]
fn rejects_expected_contact_spans_the_rules_did_not_flag() {
    let mut fx = Fixture::new();
    let doc = encoded(ENCODED);
    let phone = at(ENCODED, "555 0100");
    let unflagged = doc
        .enc
        .token_spans
        .iter()
        .zip(&doc.enc.flags)
        .filter(|((s, e), _)| *s >= phone.0 && *e <= phone.1)
        .all(|(_, f)| f & flag::IN_RULE_SPAN == 0);
    assert!(unflagged, "fixture phone must be missed by the rules");
    fx.extra.push(("encoded", doc, vec![phone]));
    fx.rejects(
        &fx.sidecar(vec![fx.entry("encoded", phone.0, phone.1)]),
        "overlaps a known expected positive span",
    );
    let mut fx = Fixture::new();
    fx.main_positives.push((44, 47));
    assert_eq!(fx.main.enc.flags[7] & flag::IN_RULE_SPAN, 0);
    fx.rejects(
        &fx.sidecar(vec![fx.entry("seen-real-24-0004", 44, 47)]),
        "overlaps a known expected positive span",
    );
}

#[test]
fn mixed_batch_excludes_the_native_row_but_not_its_authored_copy() {
    use super::super::{contract, mixed, objective::Objective};
    let fx = Fixture::new();
    let resolved = fx
        .resolve(&fx.sidecar(vec![fx.entry("seen-real-24-0004", 4, 14)]))
        .unwrap();
    let loaded = mixed::LoadedMixed::with_owners(&[None, Some(1), Some(0)]);
    let indices = [1, 0, 2];
    let owners = contract::owners(Some(&loaded), 2, &indices).unwrap();
    assert_eq!(owners, vec![Some(1), None, Some(0)]);
    let items = vec![
        fx.main.enc.clone(),
        fx.main.enc.clone(),
        fx.solo.enc.clone(),
    ];
    let mask = contract::batch_mask(
        Objective::CanonicalRuleAndReviewedAmbiguityExcludedV1,
        Some(&resolved),
        &owners,
        &items,
        &[1.0; 7],
    )
    .unwrap()
    .unwrap();
    let width = 12;
    assert_eq!(mask.values[1], 0.0, "native Respondent stays excluded");
    assert_eq!(
        mask.values[width + 1],
        1.0,
        "authored copy keeps its O loss"
    );
    assert_eq!(mask.values[2 * width], 1.0);
    assert_eq!(mask.report.excluded_reviewed_ambiguity_o_tokens, Some(1));
    assert!(contract::require_unexcluded_parents(&["seen-real-24-0004"], &resolved).is_err());
    contract::require_unexcluded_parents(&["solo"], &resolved).unwrap();
}

#[test]
fn sidecar_must_equal_the_adjudicated_declared_set_in_both_directions() {
    let fx = Fixture::new();
    let resolved = fx
        .resolve(&fx.sidecar(vec![
            fx.entry("seen-real-24-0004", 4, 14),
            fx.entry("seen-real-24-0004", 48, 58),
        ]))
        .unwrap();
    let declared = |start: u32, end: u32| DeclaredExclusion {
        name: "seen-real-24-0004".into(),
        input_sha256: digest(TEXT.as_bytes()),
        start,
        end,
        quote: TEXT[start as usize..end as usize].into(),
        reason: "adjudicated wording may differ".into(),
    };
    resolved
        .require_declared(&[declared(48, 58), declared(4, 14)])
        .unwrap();
    assert!(resolved.require_declared(&[declared(4, 14)]).is_err());
    assert!(
        resolved
            .require_declared(&[declared(4, 14), declared(48, 58), declared(15, 20)])
            .is_err()
    );
    assert!(resolved.require_declared(&[]).is_err());
    let mut hash = declared(48, 58);
    hash.input_sha256 = digest(b"other");
    assert!(resolved.require_declared(&[declared(4, 14), hash]).is_err());
    let mut renamed = declared(48, 58);
    renamed.name = "solo".into();
    assert!(
        resolved
            .require_declared(&[declared(4, 14), renamed])
            .is_err()
    );
}
