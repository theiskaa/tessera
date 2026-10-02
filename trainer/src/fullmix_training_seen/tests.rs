use super::*;

fn span(kind: usize, start: u32, end: u32) -> KindSpan {
    KindSpan { kind, start, end }
}

fn fixture(
    group: &str,
    texts: &[&str],
    gold: &[Vec<KindSpan>],
    families: &[(&str, &[usize])],
) -> (tempfile::TempDir, Seen) {
    assert_eq!(texts.len(), gold.len());
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("gold.jsonl");
    assert!(path.is_absolute());
    let names: Vec<_> = (0..texts.len()).map(|i| format!("row-{i}")).collect();
    let rows: Vec<_> = texts
        .iter()
        .zip(gold)
        .zip(&names)
        .map(|((text, spans), name)| {
            json!({
                "name":name,
                "country":"US",
                "input":text,
                "expected":spans.iter().map(|s| json!({
                    "kind":crate::detector::KINDS[s.kind].as_str(),
                    "start":s.start,
                    "end":s.end,
                    "text":text.get(s.start as usize..s.end as usize).unwrap()
                })).collect::<Vec<_>>()
            })
        })
        .collect();
    let bytes: String = rows
        .iter()
        .map(|row| format!("{}\n", serde_json::to_string(row).unwrap()))
        .collect();
    std::fs::write(&path, bytes).unwrap();
    let family_names: BTreeMap<_, Vec<_>> = families
        .iter()
        .map(|(family, indices)| {
            (
                (*family).to_owned(),
                indices.iter().map(|&i| names[i].clone()).collect(),
            )
        })
        .collect();
    let sha256 = hash(&path).unwrap();
    let seen = Seen {
        plan: json!({"families":family_names}),
        gold: BTreeMap::from([(group.to_owned(), Receipt { path, sha256 })]),
        names: BTreeMap::from([(group.to_owned(), names)]),
    };
    (dir, seen)
}

fn counts(value: &Value, kind: &str, tp: usize, fp: usize, missed: usize) {
    assert_eq!(
        value["per_kind"][kind]["exact_counts"],
        json!({"tp":tp,"fp":fp,"fn":missed,"gold":tp+missed,"predicted":tp+fp})
    );
}

#[test]
fn raw_duplicates_and_boundary_errors_have_hand_counted_family_totals() {
    let gold = vec![
        vec![span(0, 0, 4), span(1, 8, 12), span(2, 16, 20)],
        vec![span(0, 0, 3), span(1, 7, 11)],
        vec![],
    ];
    let (_dir, seen) = fixture(
        "authored",
        &["Anna at Acme on Main.", "Ben at Beta.", "Other"],
        &gold,
        &[("mixed", &[0]), ("remaining", &[1, 2])],
    );
    let raw = vec![
        vec![
            span(0, 0, 4),
            span(0, 0, 4),
            span(1, 8, 12),
            span(2, 15, 20),
        ],
        vec![span(1, 7, 11)],
        vec![span(1, 0, 5)],
    ];
    let value = seen.metrics("authored", &gold, &raw, &gold).unwrap();
    let raw_total = &value["aggregate"]["raw"];
    counts(raw_total, "person", 1, 1, 1);
    counts(raw_total, "org", 2, 1, 0);
    counts(raw_total, "address", 0, 1, 1);
    assert_eq!(raw_total["per_kind"]["person"]["exact"]["f1"], 0.5);
    assert_eq!(raw_total["per_kind"]["org"]["exact"]["f1"], 0.8);
    assert_eq!(raw_total["per_kind"]["address"]["exact"]["f1"], 0.0);
    let filtered = &value["aggregate"]["filtered"];
    counts(filtered, "person", 2, 0, 0);
    counts(filtered, "org", 2, 0, 0);
    counts(filtered, "address", 1, 0, 0);
    assert_eq!(value["all_required_supported_kind_targets_passed"], true);
    let mixed = &value["families"]["mixed"];
    assert_eq!(mixed["documents"], 1);
    counts(&mixed["raw"], "person", 1, 1, 0);
    counts(&mixed["raw"], "address", 0, 1, 1);
    let remaining = &value["families"]["remaining"];
    assert_eq!(remaining["documents"], 2);
    counts(&remaining["raw"], "person", 0, 0, 1);
    counts(&remaining["raw"], "org", 1, 1, 0);
    assert!(remaining["raw"]["per_kind"]["address"]["exact"]["f1"].is_null());
    assert_eq!(value["negative_documents"].as_array().unwrap().len(), 1);
    assert_eq!(
        value["negative_documents"][0]["raw_false_positives"],
        json!([{"kind":"org","start":0,"end":5,"text":"Other"}])
    );
    assert_eq!(
        value["negative_documents"][0]["filtered_false_positives"],
        json!([])
    );
}

#[test]
fn unsupported_kinds_keep_false_positives_and_every_utf8_negative_row() {
    let gold = vec![vec![], vec![]];
    let (_dir, seen) = fixture(
        "authored",
        &["Élodie", "Empty"],
        &gold,
        &[("controls", &[0, 1])],
    );
    let raw = vec![vec![span(0, 0, 7), span(0, 0, 7)], vec![]];
    let filtered = vec![vec![span(0, 0, 7)], vec![]];
    let value = seen.metrics("authored", &gold, &raw, &filtered).unwrap();
    counts(&value["aggregate"]["raw"], "person", 0, 2, 0);
    counts(&value["aggregate"]["filtered"], "person", 0, 1, 0);
    for prediction in ["raw", "filtered"] {
        let scores = &value["aggregate"][prediction];
        assert_eq!(scores["supported_kind_count"], 0);
        assert!(scores["macro_exact_f1"].is_null());
        for kind in ["person", "org", "address"] {
            for mode in ["exact", "lenient"] {
                for field in ["precision", "recall", "f1"] {
                    assert!(scores["per_kind"][kind][mode][field].is_null());
                }
            }
        }
    }
    assert_eq!(value["all_required_supported_kind_targets_passed"], false);
    let negatives = value["negative_documents"].as_array().unwrap();
    assert_eq!(negatives.len(), 2);
    assert_eq!(negatives[0]["name"], "row-0");
    assert_eq!(
        negatives[0]["raw_false_positives"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(
        negatives[0]["filtered_false_positives"],
        json!([{"kind":"person","start":0,"end":7,"text":"Élodie"}])
    );
    assert_eq!(
        negatives[1],
        json!({"name":"row-1","raw_false_positives":[],"filtered_false_positives":[]})
    );
}

#[test]
fn supported_aggregate_accepts_exactly_95_percent_and_rejects_extra_false_positive() {
    let gold = vec![vec![span(0, 0, 1), span(1, 2, 3), span(2, 4, 5)]; 20];
    let texts = ["P O A"; 20];
    let indices: Vec<_> = (0..20).collect();
    let (_dir, seen) = fixture("authored", &texts, &gold, &[("supported", &indices)]);
    let mut predicted = gold.clone();
    predicted[0].clear();
    predicted[19].extend_from_slice(&gold[19]);
    let value = seen
        .metrics("authored", &gold, &predicted, &predicted)
        .unwrap();
    for kind in ["person", "org", "address"] {
        counts(&value["aggregate"]["filtered"], kind, 19, 1, 1);
        let f1 = value["aggregate"]["filtered"]["per_kind"][kind]["exact"]["f1"]
            .as_f64()
            .unwrap();
        assert!((f1 - 0.95).abs() < 1e-12);
    }
    assert_eq!(value["all_required_supported_kind_targets_passed"], true);
    predicted[19].push(span(0, 0, 1));
    let failed = seen
        .metrics("authored", &gold, &predicted, &predicted)
        .unwrap();
    counts(&failed["aggregate"]["filtered"], "person", 19, 2, 1);
    assert_eq!(failed["all_required_supported_kind_targets_passed"], false);
}

#[test]
fn real_group_family_matches_aggregate_without_losing_raw_duplicates() {
    let gold = vec![vec![span(0, 0, 1), span(1, 2, 3), span(2, 4, 5)]];
    let (_dir, seen) = fixture("real", &["P O A"], &gold, &[]);
    let mut raw = gold.clone();
    raw[0].push(span(0, 0, 1));
    let value = seen.metrics("real", &gold, &raw, &gold).unwrap();
    let family = &value["families"]["all_corrected_real"];
    assert_eq!(family["documents"], 1);
    assert_eq!(family["raw"], value["aggregate"]["raw"]);
    assert_eq!(family["filtered"], value["aggregate"]["filtered"]);
    counts(&family["raw"], "person", 1, 1, 0);
    assert_eq!(value["negative_documents"], json!([]));
    assert_eq!(value["all_required_supported_kind_targets_passed"], true);
}

#[test]
fn false_positive_slices_refuse_utf8_cuts_and_out_of_bounds_offsets() {
    let gold = vec![vec![]];
    let (_dir, seen) = fixture("authored", &["Élodie"], &gold, &[("controls", &[0])]);
    for invalid in [span(0, 0, 1), span(0, 0, 8)] {
        assert!(
            seen.metrics("authored", &gold, &[vec![invalid]], &gold)
                .is_err()
        );
    }
}
