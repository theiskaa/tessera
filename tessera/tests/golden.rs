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
    ("edge-long", golden!("parser/edge-long.json")),
    ("edge-long-word", golden!("parser/edge-long-word.json")),
    ("edge-one-char", golden!("parser/edge-one-char.json")),
    (
        "edge-single-token",
        golden!("parser/edge-single-token.json"),
    ),
    ("us-01", golden!("parser/us-01.json")),
    ("us-02", golden!("parser/us-02.json")),
    ("us-03", golden!("parser/us-03.json")),
    ("us-04", golden!("parser/us-04.json")),
];

const DETECTOR_CASES: &[(&str, &str)] = &[
    ("address-us", golden!("detector/address-us.json")),
    ("letterhead-us", golden!("detector/letterhead-us.json")),
    ("one-word", golden!("detector/one-word.json")),
    ("prose-negatives", golden!("detector/prose-negatives.json")),
    ("rules-only", golden!("detector/rules-only.json")),
    ("signature-us", golden!("detector/signature-us.json")),
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

/// Checks one parser golden case against `tessera`, returning the largest logit difference and
/// the longest n-gram list of any token.
fn check_parser(tessera: &tessera::Tessera, name: &str, json: &str) -> (f32, usize) {
    let case: common::Golden = common::parse(name, json);
    assert_eq!(
        tessera.model_version(),
        Some(case.bundle_version.as_str()),
        "{name}: golden bundle version differs"
    );
    let trace = tessera::internal::parse_address_trace(tessera, &case.input.text).unwrap();
    let diff = check_case(
        name,
        &case,
        &trace.token_spans,
        &trace.features,
        &trace.logits,
        &trace.decoded,
    );
    let longest = trace
        .features
        .iter()
        .map(|f| f.ngram_ids.len())
        .max()
        .unwrap_or(0);
    (diff, longest)
}

/// Checks one detector golden case against `tessera`, tracing its rules with the country hint
/// of the input policy the case records, or with none for a legacy case.
fn check_detector(tessera: &tessera::Tessera, name: &str, json: &str) -> f32 {
    let case: common::Golden = common::parse(name, json);
    assert_eq!(
        tessera.model_version(),
        Some(case.bundle_version.as_str()),
        "{name}: golden bundle version differs"
    );
    let trace = match &case.input_policy {
        Some(_) => {
            let hint: Vec<&str> = case.country_hint.iter().map(String::as_str).collect();
            tessera.detect_trace_with_country_hint(&case.input.text, &hint)
        }
        None => tessera::internal::detect_trace(tessera, &case.input.text),
    }
    .unwrap();
    assert_eq!(trace.masked, case.masked, "{name}: decode mask differs");
    check_case(
        name,
        &case,
        &trace.token_spans,
        &trace.features,
        &trace.logits,
        &trace.decoded,
    )
}

#[test]
fn parser_golden_vectors() {
    let tessera = common::load_tessera();
    let mut longest = 0;
    let mut worst = 0f32;
    for &(name, json) in PARSER_CASES {
        let (diff, tokens) = check_parser(&tessera, name, json);
        worst = worst.max(diff);
        longest = longest.max(tokens);
    }
    assert_eq!(
        longest,
        tessera::internal::MAX_NGRAMS_PER_TOKEN,
        "no golden token reaches the n-gram cap"
    );
    let checked = PARSER_CASES.len();
    assert!(
        checked >= 6,
        "expected at least 6 golden cases, found {checked}"
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
        worst = worst.max(check_detector(&tessera, name, json));
    }
    common::report!(
        "golden detector: {} cases, max logit difference {worst:e}",
        DETECTOR_CASES.len()
    );
}

/// The same gate over a bundle not yet installed in `models/`: set `TESSERA_GOLDEN_BUNDLE_DIR`
/// to the `--out` directory of `trainer export`, holding `tessera-v1.safetensors`, its
/// `.sha256`, and `golden/{parser,detector}/*.json`.
#[cfg(all(not(target_arch = "wasm32"), feature = "phone-metadata"))]
#[test]
#[ignore = "set TESSERA_GOLDEN_BUNDLE_DIR to an exported bundle directory"]
fn exported_bundle_golden_vectors() {
    let dir = std::path::PathBuf::from(
        std::env::var("TESSERA_GOLDEN_BUNDLE_DIR").expect("TESSERA_GOLDEN_BUNDLE_DIR is set"),
    );
    let bundle = std::fs::read(dir.join("tessera-v1.safetensors")).unwrap();
    let checksum = std::fs::read_to_string(dir.join("tessera-v1.sha256")).unwrap();
    let tessera = tessera::Tessera::load(
        &bundle,
        tessera::Config {
            kinds: tessera::Kind::all(),
            expected_checksum: Some(checksum.trim()),
        },
    )
    .expect("exported bundle loads");
    let cases = |net: &str| {
        let mut cases: Vec<(String, String)> = std::fs::read_dir(dir.join("golden").join(net))
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .map(|p| {
                let name = p.file_stem().unwrap().to_string_lossy().into_owned();
                (name, std::fs::read_to_string(p).unwrap())
            })
            .collect();
        cases.sort();
        cases
    };
    let (parser, detector) = (cases("parser"), cases("detector"));
    assert!(
        parser.len() >= 6 && detector.len() >= 7,
        "golden cases missing"
    );
    let mut worst = 0f32;
    for (name, json) in &parser {
        worst = worst.max(check_parser(&tessera, name, json).0);
    }
    for (name, json) in &detector {
        worst = worst.max(check_detector(&tessera, name, json));
    }
    common::report!(
        "exported golden: {} parser and {} detector cases, max logit difference {worst:e}",
        parser.len(),
        detector.len()
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
