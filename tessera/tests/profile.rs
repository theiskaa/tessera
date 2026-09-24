//! The profiling harness times the real pipeline: its stages are finite, fit inside the total,
//! and the timed call returns exactly what an untimed one does.

#![cfg(all(feature = "profile", not(target_arch = "wasm32")))]

mod common;

use std::time::Instant;

use tessera::Query;
use tessera::profile::{StageTimings, extract_timed, parse_timed};

const DOCUMENT: &str = include_str!("../../fixtures/profile/document-10k.txt");
const ADDRESS: &str = include_str!("../../fixtures/profile/address-gb.txt");

fn clock() -> impl Fn() -> f64 + 'static {
    let origin = Instant::now();
    move || origin.elapsed().as_secs_f64() * 1000.0
}

fn assert_consistent(t: &StageTimings) {
    let stages = [
        t.tokenize_ms,
        t.featurize_ms,
        t.rules_ms,
        t.detect_ms,
        t.parse_ms,
        t.group_ms,
    ];
    for v in stages.iter().chain([&t.total_ms]) {
        assert!(v.is_finite() && *v >= 0.0, "{t:?}");
    }
    assert!(stages.iter().sum::<f64>() <= t.total_ms + 0.5, "{t:?}");
}

#[test]
fn extract_timed_measures_the_real_pipeline() {
    let tessera = common::load_all();
    let query = Query::default();
    let (t, timed) = extract_timed(&tessera, DOCUMENT, &query, clock()).unwrap();
    assert_consistent(&t);
    assert!(
        t.detect_ms > 0.0 && t.parse_ms > 0.0 && t.tokenize_ms > 0.0,
        "{t:?}"
    );
    assert_eq!(timed, tessera.extract_contacts(DOCUMENT, &query).unwrap());
}

#[test]
fn parse_timed_measures_the_real_parser() {
    let tessera = common::load_all();
    let query = Query::default();
    let address = ADDRESS.trim_end();
    let (t, timed) = parse_timed(&tessera, address, &query, clock()).unwrap();
    assert_consistent(&t);
    assert_eq!(t.detect_ms, 0.0);
    assert_eq!(timed, tessera.parse_address(address, &query).unwrap());
}
