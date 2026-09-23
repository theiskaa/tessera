//! Golden-vector gate: the hand-written forward pass must reproduce the trainer's quantized
//! outputs from `models/golden/parser/`, natively and in the browser. Features, logits, and
//! decoded labels are checked in that order because each fails for a different reason.

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

/// Prints a line in the test output: stderr natively, the browser console under wasm, where
/// `wasm-bindgen-test` relays it.
macro_rules! report {
    ($($arg:tt)*) => {{
        #[cfg(target_arch = "wasm32")]
        wasm_bindgen_test::console_log!($($arg)*);
        #[cfg(not(target_arch = "wasm32"))]
        eprintln!($($arg)*);
    }};
}

// A browser cannot list a directory, so every file in models/golden/parser/ is named here;
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

#[test]
fn parser_golden_vectors() {
    let tessera = common::load_tessera();
    let mut checked = 0;
    let mut longest = 0;
    let mut worst = 0f32;
    for &(name, json) in PARSER_CASES {
        let case: common::Golden = common::parse(name, json);
        let trace = tessera::internal::parse_address_trace(&tessera, &case.input.text).unwrap();
        assert_eq!(
            trace.token_spans, case.input.tokens,
            "{name}: token spans differ"
        );
        assert_eq!(
            trace.features.len(),
            case.input.features.len(),
            "{name}: token count differs"
        );
        for (i, (got, want)) in trace.features.iter().zip(&case.input.features).enumerate() {
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
        assert_eq!(
            trace.logits.len(),
            want.len(),
            "{name}: logit count differs"
        );
        assert!(
            trace.logits.iter().all(|v| v.is_finite()),
            "{name}: a logit is not finite"
        );
        let max_diff = trace
            .logits
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(
            max_diff <= case.tolerance,
            "{name}: max logit difference {max_diff} exceeds {}",
            case.tolerance
        );
        worst = worst.max(max_diff);
        assert_eq!(trace.decoded, case.decoded, "{name}: decoded labels differ");
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
    report!("golden: {checked} cases, max logit difference {worst:e}");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn the_case_list_names_every_golden_file() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/golden/parser");
    let mut on_disk: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .collect();
    on_disk.sort();
    let mut listed: Vec<String> = PARSER_CASES
        .iter()
        .map(|(name, _)| format!("{name}.json"))
        .collect();
    listed.sort();
    assert_eq!(listed, on_disk);
}
