//! Golden-vector gate: the hand-written forward pass must reproduce the trainer's quantized
//! outputs from `models/golden/parser/`. Features, logits, and decoded labels are checked in
//! that order because each fails for a different reason.

mod common;

#[test]
fn parser_golden_vectors() {
    let tessera = common::load_tessera();
    let mut checked = 0;
    let mut longest = 0;
    let mut worst = 0f32;
    for path in common::golden_files("parser") {
        let name = path.display();
        let case: common::Golden = common::load_json(&path);
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
    eprintln!("golden: {checked} cases, max logit difference {worst:e}");
}
