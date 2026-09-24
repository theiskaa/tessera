//! Golden-vector gate: the hand-written forward passes must reproduce the trainer's quantized
//! outputs from `models/golden/parser/` and `models/golden/detector/`, natively and in the
//! browser. Features, logits, and decoded labels are checked in that order because each fails
//! for a different reason.

mod common;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

/// The contents of `models/golden/<path>`, embedded.
macro_rules! golden {
    ($path:literal) => {
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../models/golden/",
            $path
        ))
    };
}

// A browser cannot list a directory, so every golden file is named here;
// `the_case_list_names_every_golden_file` fails natively when the two drift apart.
const PARSER_CASES: &[(&str, &str)] = &[
    ("de-01", golden!("parser/de-01.json")),
    ("de-02", golden!("parser/de-02.json")),
    ("de-03", golden!("parser/de-03.json")),
    ("de-04", golden!("parser/de-04.json")),
    ("edge-long-word", golden!("parser/edge-long-word.json")),
    ("edge-long", golden!("parser/edge-long.json")),
    ("edge-one-char", golden!("parser/edge-one-char.json")),
    (
        "edge-single-token",
        golden!("parser/edge-single-token.json"),
    ),
    ("gb-01", golden!("parser/gb-01.json")),
    ("gb-02", golden!("parser/gb-02.json")),
    ("gb-03", golden!("parser/gb-03.json")),
    ("gb-04", golden!("parser/gb-04.json")),
    ("ge-01", golden!("parser/ge-01.json")),
    ("ge-02", golden!("parser/ge-02.json")),
    ("ge-03", golden!("parser/ge-03.json")),
    ("ge-04", golden!("parser/ge-04.json")),
    ("jp-01", golden!("parser/jp-01.json")),
    ("jp-02", golden!("parser/jp-02.json")),
    ("jp-03", golden!("parser/jp-03.json")),
    ("jp-04", golden!("parser/jp-04.json")),
    ("us-01", golden!("parser/us-01.json")),
    ("us-02", golden!("parser/us-02.json")),
    ("us-03", golden!("parser/us-03.json")),
    ("us-04", golden!("parser/us-04.json")),
];

const DETECTOR_CASES: &[(&str, &str)] = &[
    ("letterhead-de", golden!("detector/letterhead-de.json")),
    ("one-word", golden!("detector/one-word.json")),
    ("prose-negatives", golden!("detector/prose-negatives.json")),
    ("rules-only", golden!("detector/rules-only.json")),
    ("signature-gb", golden!("detector/signature-gb.json")),
    ("signature-ge", golden!("detector/signature-ge.json")),
    ("signature-jp", golden!("detector/signature-jp.json")),
    ("table-tab", golden!("detector/table-tab.json")),
];

/// Checks one trace against its golden case in the order features, logits, decoded labels, and
/// returns the largest logit difference.
fn check_case(
    name: &str,
    case: &common::Golden,
    token_spans: &[(usize, usize)],
    features: &[tessera::internal::TokenFeatures],
    logits: &[f32],
    decoded: &[u8],
) -> f32 {
    assert_eq!(token_spans, case.input.tokens, "{name}: token spans differ");
    assert_eq!(
        features.len(),
        case.input.features.len(),
        "{name}: token count differs"
    );
    for (i, (got, want)) in features.iter().zip(&case.input.features).enumerate() {
        assert_eq!(
            got.ngram_ids, want.ngram_ids,
            "{name}: token {i} n-gram ids differ"
        );
        assert_eq!(
            (got.script, got.shape, got.flags),
            (want.script, want.shape, want.flags),
            "{name}: token {i} script, shape, or flags differ"
        );
    }
    let want: Vec<f32> = case.int8_logits.iter().flatten().copied().collect();
    assert_eq!(logits.len(), want.len(), "{name}: logit count differs");
    assert!(
        logits.iter().all(|v| v.is_finite()),
        "{name}: a logit is not finite"
    );
    let max_diff = logits
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        max_diff <= case.tolerance,
        "{name}: max logit difference {max_diff} exceeds {}",
        case.tolerance
    );
    assert_eq!(decoded, case.decoded, "{name}: decoded labels differ");
    max_diff
}

#[test]
fn parser_golden_vectors() {
    let tessera = common::load_tessera();
    let mut checked = 0;
    let mut longest = 0;
    let mut worst = 0f32;
    for &(name, json) in PARSER_CASES {
        let case: common::Golden = common::parse(name, json);
        let trace = tessera::internal::parse_address_trace(&tessera, &case.input.text).unwrap();
        let diff = check_case(
            name,
            &case,
            &trace.token_spans,
            &trace.features,
            &trace.logits,
            &trace.decoded,
        );
        worst = worst.max(diff);
        longest = trace
            .features
            .iter()
            .map(|f| f.ngram_ids.len())
            .fold(longest, usize::max);
        checked += 1;
    }
    assert_eq!(
        longest,
        tessera::internal::MAX_NGRAMS_PER_TOKEN,
        "no golden token reaches the n-gram cap"
    );
    assert!(
        checked >= 16,
        "expected at least 16 golden cases, found {checked}"
    );
    common::report!("golden parser: {checked} cases, max logit difference {worst:e}");
}

/// The vectors were exported from the default build, whose phone rule spans are detector
/// input features.
#[cfg(feature = "phone-metadata")]
#[test]
fn detector_golden_vectors() {
    let tessera = common::load_all();
    let mut worst = 0f32;
    for &(name, json) in DETECTOR_CASES {
        let case: common::Golden = common::parse(name, json);
        let trace = tessera::internal::detect_trace(&tessera, &case.input.text).unwrap();
        assert_eq!(trace.masked, case.masked, "{name}: decode mask differs");
        let diff = check_case(
            name,
            &case,
            &trace.token_spans,
            &trace.features,
            &trace.logits,
            &trace.decoded,
        );
        worst = worst.max(diff);
    }
    common::report!(
        "golden detector: {} cases, max logit difference {worst:e}",
        DETECTOR_CASES.len()
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn the_case_lists_name_every_golden_file() {
    for (net, cases) in [("parser", PARSER_CASES), ("detector", DETECTOR_CASES)] {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../models/golden")
            .join(net);
        let mut on_disk: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        on_disk.sort();
        let mut listed: Vec<String> = cases
            .iter()
            .map(|(name, _)| format!("{name}.json"))
            .collect();
        listed.sort();
        assert_eq!(listed, on_disk, "models/golden/{net}");
    }
}
