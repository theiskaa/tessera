//! Fixture-driven grouping tests over gold entities from `fixtures/grouper/`: the grouper alone,
//! with detection taken out of the measurement.

mod common;

use std::collections::BTreeSet;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

use tessera::internal::group;
use tessera::{Entity, Extraction, Kind, Source};

type Triple = (&'static str, usize, usize);

/// Gold entities as the grouper would get them from `detect`: confidence 0.95, emails and
/// phones from the rules with a normalized form (the email name match reads it).
fn gold_entities(case: &common::GrouperCase) -> Vec<Entity> {
    case.entities
        .iter()
        .map(|e| {
            let kind = Kind::from_str_label(&e.kind).expect("known kind");
            Entity {
                kind,
                start: e.start,
                end: e.end,
                confidence: 0.95,
                source: if matches!(kind, Kind::Email | Kind::Phone) {
                    Source::Rules
                } else {
                    Source::Model
                },
                components: Vec::new(),
                normalized: match kind {
                    Kind::Email => Some(e.text.to_ascii_lowercase()),
                    Kind::Phone => Some(e.text.replace(' ', "")),
                    _ => None,
                },
                region: None,
                review_recommended: false,
            }
        })
        .collect()
}

fn triple(e: &Entity) -> Triple {
    (e.kind.as_str(), e.start, e.end)
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ContactShape {
    person: Option<Triple>,
    org: Option<Triple>,
    addresses: BTreeSet<Triple>,
    emails: BTreeSet<Triple>,
    phones: BTreeSet<Triple>,
}

type Shape = (BTreeSet<ContactShape>, Vec<Triple>);

fn expected_shape(case: &common::GrouperCase, gold: &[Entity]) -> Shape {
    let at = |i: usize| triple(&gold[i]);
    let set = |l: &[usize]| l.iter().map(|&i| at(i)).collect();
    let contacts = case
        .contacts
        .iter()
        .map(|c| ContactShape {
            person: c.person.map(at),
            org: c.org.map(at),
            addresses: set(&c.addresses),
            emails: set(&c.emails),
            phones: set(&c.phones),
        })
        .collect();
    let mut unassigned: Vec<Triple> = case.unassigned.iter().map(|&i| at(i)).collect();
    unassigned.sort_by_key(|t| t.1);
    (contacts, unassigned)
}

fn predicted_shape(out: &Extraction) -> Shape {
    let contacts = out
        .contacts
        .iter()
        .map(|c| ContactShape {
            person: c.person.as_ref().map(triple),
            org: c.org.as_ref().map(triple),
            addresses: c.addresses.iter().map(triple).collect(),
            emails: c.emails.iter().map(triple).collect(),
            phones: c.phones.iter().map(triple).collect(),
        })
        .collect();
    (contacts, out.unassigned.iter().map(triple).collect())
}

#[test]
fn grouper_fixtures() {
    let mut cases = 0;
    for (file, fixture) in common::grouper_fixtures() {
        for case in &fixture.cases {
            cases += 1;
            let gold = gold_entities(case);
            let out = group(&case.input, gold.clone());
            let expected = expected_shape(case, &gold);
            let predicted = predicted_shape(&out);
            assert_eq!(
                predicted, expected,
                "{file}: {}\npredicted: {predicted:#?}\nexpected: {expected:#?}",
                case.name
            );
            for c in &out.contacts {
                assert!(c.start < c.end, "{file}: {}: empty contact hull", case.name);
                assert!(
                    c.confidence > 0.0 && c.confidence <= 0.95,
                    "{file}: {}: confidence {} out of range",
                    case.name,
                    c.confidence
                );
            }
        }
    }
    common::report!("grouper fixtures: {cases} cases");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn the_fixture_list_names_every_grouper_file() {
    common::assert_lists_every_fixture("grouper", &common::grouper_fixtures());
}
