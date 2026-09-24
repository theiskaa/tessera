//! Fixture-driven tests for `detect` against the versioned bundle: rules, windows, merge, and
//! thresholds together, on documents whose expected entities a person wrote down.

mod common;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

#[cfg(feature = "phone-metadata")]
use tessera::{Entity, Query};

/// Why a case fails, or `None` when every expected entity is found at its exact offsets and no
/// forbidden entity is present.
#[cfg(feature = "phone-metadata")]
fn failure(case: &common::DetectorCase, got: &[Entity]) -> Option<String> {
    let found: Vec<(&str, usize, usize)> = got
        .iter()
        .map(|e| (e.kind.as_str(), e.start, e.end))
        .collect();
    let mut problems = Vec::new();
    for e in &case.expected {
        if !found.contains(&(e.kind.as_str(), e.start, e.end)) {
            problems.push(format!("missing {} {:?}", e.kind, e.text));
        }
    }
    for m in &case.must_not {
        if got
            .iter()
            .any(|e| e.kind.as_str() == m.kind && e.text(&case.input) == m.text)
        {
            problems.push(format!("forbidden {} {:?}", m.kind, m.text));
        }
    }
    if problems.is_empty() {
        return None;
    }
    let got: Vec<String> = got
        .iter()
        .map(|e| format!("{} {:?}", e.kind.as_str(), e.text(&case.input)))
        .collect();
    Some(format!("{}; got [{}]", problems.join(", "), got.join(", ")))
}

/// The fixtures describe the default build: without the phone tables no number is scanned, and
/// the missing rule spans change the detector's input features as well.
#[cfg(feature = "phone-metadata")]
#[test]
fn detector_fixtures() {
    let tessera = common::load_all();
    let mut unexpected = Vec::new();
    let mut known = Vec::new();
    let mut cases = 0;
    for (file, fixture) in common::detector_fixtures() {
        for case in &fixture.cases {
            cases += 1;
            let hint: Vec<&str> = case.country.as_deref().into_iter().collect();
            let query = Query {
                country_hint: &hint,
                ..Query::default()
            };
            let got = tessera.detect(&case.input, &query).unwrap();
            let id = format!("{file}/{}", case.name);
            match (failure(case, &got), case.known_failure) {
                (None, false) => {}
                (Some(why), false) => unexpected.push(format!("{id}: {why}")),
                (Some(_), true) => known.push(id),
                (None, true) => unexpected.push(format!(
                    "{id}: passes but is marked known_failure; remove the flag"
                )),
            }
        }
    }
    for id in &known {
        common::report!("known failure: {id}");
    }
    common::report!(
        "detector fixtures: {cases} cases, {} known failures",
        known.len()
    );
    assert!(unexpected.is_empty(), "{}", unexpected.join("\n"));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn the_fixture_list_names_every_detector_file() {
    common::assert_lists_every_fixture("detector", &common::detector_fixtures());
}
