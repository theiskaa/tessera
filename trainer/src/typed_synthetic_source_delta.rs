//! Pinned entities-only successor files with an ordered, exact multi-row delta.

use super::{Pin, valid_hash};
use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Deserialize, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
struct Entity {
    kind: String,
    start: usize,
    end: usize,
}

impl Entity {
    fn validate(&self, text: &str) -> anyhow::Result<()> {
        ensure!(
            ["person", "org", "address", "email", "phone"].contains(&self.kind.as_str())
                && self.start < self.end
                && text.get(self.start..self.end).is_some(),
            "invalid entity kind or UTF8 endpoints"
        );
        Ok(())
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RowDelta {
    physical_row_1_based: usize,
    id: String,
    text_sha256: String,
    added: Vec<Entity>,
    removed: Vec<Entity>,
    boundary_replacements: Option<Vec<BoundaryReplacement>>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BoundaryReplacement {
    removed: Entity,
    added: Entity,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SourceDelta {
    schema: String,
    before: Pin,
    after: Pin,
    rows: usize,
    changes: Vec<RowDelta>,
}

fn pinned_text(pin: &Pin) -> anyhow::Result<String> {
    ensure!(
        Path::new(&pin.path).is_absolute() && valid_hash(&pin.sha256),
        "source delta requires an absolute path and lowercase SHA256"
    );
    let bytes = std::fs::read(&pin.path)?;
    ensure!(
        crate::export::sha256_hex(&bytes) == pin.sha256,
        "source delta file changed"
    );
    Ok(String::from_utf8(bytes)?)
}

fn nonoverlapping(entities: &[Entity]) -> anyhow::Result<()> {
    let mut spans: Vec<_> = entities.iter().map(|e| (e.start, e.end)).collect();
    spans.sort_unstable();
    ensure!(
        spans.windows(2).all(|pair| pair[0].1 <= pair[1].0),
        "source delta entities overlap"
    );
    Ok(())
}

fn validate_replacements(change: &RowDelta) -> anyhow::Result<()> {
    let replacements = change
        .boundary_replacements
        .as_ref()
        .context("v2 row lacks explicit boundary replacements")?;
    nonoverlapping(&change.added)?;
    nonoverlapping(&change.removed)?;
    let mut pairs = BTreeSet::new();
    let mut removed = BTreeSet::new();
    let mut added = BTreeSet::new();
    for pair in replacements {
        ensure!(
            pair.removed.kind == pair.added.kind
                && pair.removed.start < pair.added.end
                && pair.added.start < pair.removed.end
                && pair.removed != pair.added
                && change.removed.contains(&pair.removed)
                && change.added.contains(&pair.added)
                && removed.insert(pair.removed.clone())
                && added.insert(pair.added.clone())
                && pairs.insert((pair.removed.clone(), pair.added.clone())),
            "invalid, duplicate or ambiguous boundary replacement"
        );
    }
    for old in &change.removed {
        for new in &change.added {
            if old.start < new.end && new.start < old.end {
                ensure!(
                    pairs.contains(&(old.clone(), new.clone())),
                    "undeclared overlapping boundary replacement"
                );
            }
        }
    }
    Ok(())
}

fn changed_row(before: &Value, change: &RowDelta, boundary_schema: bool) -> anyhow::Result<Value> {
    let text = before["text"].as_str().context("source text absent")?;
    ensure!(
        before["id"] == change.id
            && valid_hash(&change.text_sha256)
            && crate::export::sha256_hex(text.as_bytes()) == change.text_sha256,
        "source delta row id or text changed"
    );
    ensure!(
        !change.added.is_empty() || !change.removed.is_empty(),
        "empty row entity delta"
    );
    let mut delta_keys = BTreeSet::new();
    let mut delta_spans = Vec::new();
    for entity in change.added.iter().chain(&change.removed) {
        entity.validate(text)?;
        ensure!(
            delta_keys.insert(entity.clone()),
            "duplicate or cancelling entity delta"
        );
        delta_spans.push(entity.clone());
    }
    if boundary_schema {
        validate_replacements(change)?;
    } else {
        nonoverlapping(&delta_spans)?;
    }
    let mut entities: Vec<Entity> = serde_json::from_value(before["entities"].clone())?;
    if boundary_schema {
        nonoverlapping(&entities)?;
    }
    let mut original_keys = BTreeSet::new();
    for entity in &entities {
        entity.validate(text)?;
        ensure!(
            original_keys.insert(entity.clone()),
            "duplicate original entity"
        );
    }
    for removed in &change.removed {
        let index = entities
            .iter()
            .position(|entity| entity == removed)
            .context("removed entity absent from original")?;
        entities.remove(index);
    }
    for added in &change.added {
        ensure!(!entities.contains(added), "added entity already exists");
        entities.push(added.clone());
    }
    if !change.added.is_empty() {
        entities.sort_by_key(|entity| (entity.start, entity.end));
    }
    nonoverlapping(&entities)?;
    let mut expected = before.clone();
    expected["entities"] = serde_json::to_value(entities)?;
    Ok(expected)
}

/// Verify exact declared annotation changes without activating any manifest or fit schema.
pub(super) fn verify(value: &Value) -> anyhow::Result<()> {
    let delta: SourceDelta = serde_json::from_value(value.clone())?;
    let boundary_schema = delta.schema == "entities_only_source_delta_v2";
    ensure!(
        (delta.schema == "entities_only_source_delta_v1" || boundary_schema)
            && delta.rows > 0
            && !delta.changes.is_empty(),
        "unsupported or empty source delta contract"
    );
    if boundary_schema {
        ensure!(
            delta
                .changes
                .iter()
                .all(|c| c.boundary_replacements.is_some())
                && delta.changes.iter().any(|c| c
                    .boundary_replacements
                    .as_ref()
                    .is_some_and(|r| !r.is_empty())),
            "v2 needs explicit rows and at least one boundary replacement"
        );
    } else {
        ensure!(
            value["changes"].as_array().is_some_and(|changes| changes
                .iter()
                .all(|c| c.get("boundary_replacements").is_none())),
            "v1 cannot declare boundary replacements"
        );
    }
    ensure!(
        std::fs::canonicalize(&delta.before.path)? != std::fs::canonicalize(&delta.after.path)?
            && delta.before.sha256 != delta.after.sha256,
        "source delta requires distinct before and successor files"
    );
    let before = pinned_text(&delta.before)?;
    let after = pinned_text(&delta.after)?;
    let old: Vec<_> = before.split_inclusive('\n').collect();
    let new: Vec<_> = after.split_inclusive('\n').collect();
    ensure!(
        old.len() == delta.rows && new.len() == delta.rows,
        "source delta physical row count changed"
    );
    let mut previous = 0;
    let mut changed_ids = BTreeSet::new();
    for change in &delta.changes {
        ensure!(
            change.physical_row_1_based > previous
                && change.physical_row_1_based <= delta.rows
                && !change.id.is_empty()
                && changed_ids.insert(&change.id),
            "source delta rows must have unique ordered identities"
        );
        previous = change.physical_row_1_based;
    }
    let mut ids = BTreeSet::new();
    let mut changes = delta.changes.iter().peekable();
    for (index, (old, new)) in old.iter().zip(&new).enumerate() {
        let original: Value = serde_json::from_str(old)?;
        let id = original["id"].as_str().context("source row id absent")?;
        ensure!(
            !id.is_empty() && ids.insert(id.to_owned()),
            "duplicate source row id"
        );
        if changes
            .peek()
            .is_some_and(|change| change.physical_row_1_based == index + 1)
        {
            let change = changes.next().context("source delta row absent")?;
            ensure!(
                old.ends_with('\n') == new.ends_with('\n') && old != new,
                "declared row is unchanged or its line ending changed"
            );
            let successor: Value = serde_json::from_str(new)?;
            ensure!(
                changed_row(&original, change, boundary_schema)? == successor,
                "successor is not the exact declared entities-only row"
            );
        } else {
            ensure!(old == new, "undeclared source row bytes changed");
        }
    }
    ensure!(changes.next().is_none(), "source delta row was not visited");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    struct Fixture {
        _dir: tempfile::TempDir,
        before: String,
        after: String,
        contract: Value,
    }

    impl Fixture {
        fn new() -> Self {
            let dir = tempfile::tempdir().unwrap();
            let rows = [
                json!({"id":"a","text":"The Service works.","country":"US","metadata":{"raw":"pinned"},"entities":[{"kind":"org","start":4,"end":11}]}),
                json!({"id":"b","text":"Élodie at Agency and BLS.","country":"US","entities":[{"kind":"person","start":0,"end":7},{"kind":"org","start":11,"end":17}]}),
                json!({"id":"c","text":"Unchanged row.","source":"fixture","entities":[]}),
            ];
            let before = rows.iter().map(|r| format!("{r}\n")).collect::<String>();
            let mut corrected = rows.clone();
            corrected[0]["entities"] = json!([]);
            corrected[1]["entities"] = json!([{"kind":"person","start":0,"end":7}]);
            let after = corrected
                .iter()
                .map(|r| format!("{r}\n"))
                .collect::<String>();
            let changes = rows[..2].iter().enumerate().map(|(i,r)| json!({
                "physical_row_1_based":i+1,"id":r["id"],
                "text_sha256":crate::export::sha256_hex(r["text"].as_str().unwrap().as_bytes()),
                "added":[],"removed":[r["entities"].as_array().unwrap().last().unwrap()],
            })).collect::<Vec<_>>();
            let contract = json!({"schema":"entities_only_source_delta_v1","rows":3,
                "before":{"path":dir.path().join("before.jsonl"),"sha256":crate::export::sha256_hex(before.as_bytes())},
                "after":{"path":dir.path().join("after.jsonl"),"sha256":crate::export::sha256_hex(after.as_bytes())},"changes":changes});
            let fixture = Self {
                _dir: dir,
                before,
                after,
                contract,
            };
            fixture.write();
            fixture
        }

        fn write(&self) {
            std::fs::write(
                self.contract["before"]["path"].as_str().unwrap(),
                &self.before,
            )
            .unwrap();
            std::fs::write(
                self.contract["after"]["path"].as_str().unwrap(),
                &self.after,
            )
            .unwrap();
        }

        fn repin_after(&mut self) {
            self.contract["after"]["sha256"] =
                json!(crate::export::sha256_hex(self.after.as_bytes()));
            self.write();
        }
    }

    fn boundary_fixture() -> Fixture {
        let mut fixture = Fixture::new();
        let text = "Jamestown Baptist Church Life Center";
        let old = json!({"id":"venue","text":text,"country":"US","entities":[{"kind":"org","start":0,"end":36}]});
        let mut new = old.clone();
        new["entities"][0]["end"] = json!(24);
        fixture.before = format!("{old}\n");
        fixture.after = format!("{new}\n");
        fixture.contract["schema"] = json!("entities_only_source_delta_v2");
        fixture.contract["rows"] = json!(1);
        fixture.contract["before"]["sha256"] =
            json!(crate::export::sha256_hex(fixture.before.as_bytes()));
        fixture.contract["changes"] = json!([{
            "physical_row_1_based":1,"id":"venue",
            "text_sha256":crate::export::sha256_hex(text.as_bytes()),
            "removed":[old["entities"][0]],"added":[new["entities"][0]],
            "boundary_replacements":[{"removed":old["entities"][0],"added":new["entities"][0]}],
        }]);
        fixture.repin_after();
        fixture
    }

    #[test]
    fn v2_shrinks_one_declared_same_kind_boundary_without_changing_text() {
        verify(&boundary_fixture().contract).unwrap();
    }

    #[test]
    fn v1_keeps_refusing_boundary_replacements_with_or_without_declarations() {
        let mut fixture = boundary_fixture();
        fixture.contract["schema"] = json!("entities_only_source_delta_v1");
        assert!(verify(&fixture.contract).is_err());
        fixture.contract["changes"][0]
            .as_object_mut()
            .unwrap()
            .remove("boundary_replacements");
        assert!(verify(&fixture.contract).is_err());
        fixture.contract["changes"][0]["boundary_replacements"] = Value::Null;
        assert!(verify(&fixture.contract).is_err());
    }

    #[test]
    fn v2_requires_exact_unique_one_to_one_overlapping_replacements() {
        let fixture = boundary_fixture();
        for replacement in [
            Value::Null,
            json!([]),
            json!([{"removed":{"kind":"person","start":0,"end":36},"added":{"kind":"org","start":0,"end":24}}]),
            json!([{"removed":{"kind":"org","start":0,"end":35},"added":{"kind":"org","start":0,"end":24}}]),
            json!([{"removed":{"kind":"org","start":0,"end":36},"added":{"kind":"org","start":25,"end":36}}]),
            json!([{"removed":{"kind":"org","start":0,"end":36},"added":{"kind":"org","start":0,"end":24},"unknown":true}]),
        ] {
            let mut contract = fixture.contract.clone();
            contract["changes"][0]["boundary_replacements"] = replacement;
            assert!(verify(&contract).is_err());
        }
        let mut contract = fixture.contract.clone();
        let pair = contract["changes"][0]["boundary_replacements"][0].clone();
        contract["changes"][0]["boundary_replacements"]
            .as_array_mut()
            .unwrap()
            .push(pair);
        assert!(verify(&contract).is_err());
        contract["changes"][0]
            .as_object_mut()
            .unwrap()
            .remove("boundary_replacements");
        assert!(verify(&contract).is_err());
    }

    #[test]
    fn v2_replacement_cannot_overlap_a_retained_entity() {
        let mut fixture = boundary_fixture();
        let text = "Jamestown Baptist Church Life Center Anne";
        let old = json!({"id":"venue","text":text,"country":"US","entities":[{"kind":"org","start":0,"end":36},{"kind":"person","start":37,"end":41}]});
        let mut new = old.clone();
        new["entities"][0]["end"] = json!(39);
        fixture.before = format!("{old}\n");
        fixture.after = format!("{new}\n");
        fixture.contract["before"]["sha256"] =
            json!(crate::export::sha256_hex(fixture.before.as_bytes()));
        fixture.contract["changes"][0]["text_sha256"] =
            json!(crate::export::sha256_hex(text.as_bytes()));
        fixture.contract["changes"][0]["added"][0] = new["entities"][0].clone();
        fixture.contract["changes"][0]["boundary_replacements"][0]["added"] =
            new["entities"][0].clone();
        fixture.repin_after();
        assert!(verify(&fixture.contract).is_err());
    }

    #[test]
    fn two_removals_in_one_source_preserve_utf8_person_metadata_and_unchanged_bytes() {
        let fixture = Fixture::new();
        verify(&fixture.contract).unwrap();
    }

    #[test]
    fn multiple_additions_in_one_row_have_exact_order_and_preserve_other_rows() {
        let mut fixture = Fixture::new();
        fixture.contract["changes"][1]["added"] =
            json!([{"kind":"org","start":8,"end":10},{"kind":"org","start":22,"end":25}]);
        let mut rows: Vec<Value> = fixture
            .after
            .lines()
            .map(|s| serde_json::from_str(s).unwrap())
            .collect();
        rows[1]["entities"] = json!([{"kind":"person","start":0,"end":7},{"kind":"org","start":8,"end":10},{"kind":"org","start":22,"end":25}]);
        fixture.after = rows.iter().map(|r| format!("{r}\n")).collect();
        fixture.repin_after();
        verify(&fixture.contract).unwrap();
        rows[1]["entities"].as_array_mut().unwrap().reverse();
        fixture.after = rows.iter().map(|r| format!("{r}\n")).collect();
        fixture.repin_after();
        assert!(verify(&fixture.contract).is_err());
    }

    #[test]
    fn pin_schema_and_unknown_contract_fields_refuse() {
        let fixture = Fixture::new();
        for (pointer, value) in [
            ("/schema", json!("entities_only_source_delta_v2")),
            ("/before/sha256", json!("0".repeat(64))),
            ("/after/path", fixture.contract["before"]["path"].clone()),
            ("/before/path", json!("before.jsonl")),
        ] {
            let mut bad = fixture.contract.clone();
            *bad.pointer_mut(pointer).unwrap() = value;
            assert!(verify(&bad).is_err());
        }
        for pointer in ["", "/before", "/changes/0", "/changes/0/removed/0"] {
            let mut bad = fixture.contract.clone();
            bad.pointer_mut(pointer).unwrap()["unknown"] = json!(true);
            assert!(verify(&bad).is_err());
        }
    }

    #[test]
    fn changed_text_metadata_unreviewed_format_and_row_counts_refuse_even_when_repinned() {
        for mutation in ["text", "metadata", "format", "append", "ending"] {
            let mut fixture = Fixture::new();
            match mutation {
                "text" => {
                    fixture.after = fixture
                        .after
                        .replace("The Service works.", "The Service changed.")
                }
                "metadata" => fixture.after = fixture.after.replace("pinned", "changed"),
                "format" => fixture.after = fixture.after.replace("\"id\":\"c\"", "\"id\": \"c\""),
                "append" => fixture.after.push_str("{}\n"),
                _ => {
                    fixture.after.pop();
                }
            }
            fixture.repin_after();
            assert!(verify(&fixture.contract).is_err(), "{mutation}");
        }
    }

    #[test]
    fn row_order_identity_hash_and_empty_delta_refuse() {
        let fixture = Fixture::new();
        for field in ["order", "duplicate", "id", "hash", "empty", "outside"] {
            let mut bad = fixture.contract.clone();
            match field {
                "order" => bad["changes"].as_array_mut().unwrap().reverse(),
                "duplicate" => bad["changes"][1] = bad["changes"][0].clone(),
                "id" => bad["changes"][0]["id"] = json!("b"),
                "hash" => bad["changes"][0]["text_sha256"] = json!("0".repeat(64)),
                "empty" => bad["changes"][0]["removed"] = json!([]),
                _ => bad["changes"][1]["physical_row_1_based"] = json!(4),
            }
            assert!(verify(&bad).is_err(), "{field}");
        }
    }

    #[test]
    fn duplicate_cancelling_overlapping_missing_and_bad_utf8_entities_refuse() {
        let fixture = Fixture::new();
        for field in [
            "duplicate",
            "cancel",
            "overlap",
            "retained_overlap",
            "missing",
            "utf8",
            "kind",
        ] {
            let mut bad = fixture.contract.clone();
            match field {
                "duplicate" => {
                    let e = bad["changes"][0]["removed"][0].clone();
                    bad["changes"][0]["removed"].as_array_mut().unwrap().push(e);
                }
                "cancel" => bad["changes"][0]["added"] = bad["changes"][0]["removed"].clone(),
                "overlap" => {
                    bad["changes"][0]["added"] = json!([{"kind":"person","start":5,"end":8}])
                }
                "retained_overlap" => {
                    bad["changes"][1]["added"] = json!([{"kind":"org","start":2,"end":5}])
                }
                "missing" => bad["changes"][0]["removed"][0]["kind"] = json!("person"),
                "utf8" => bad["changes"][1]["added"] = json!([{"kind":"org","start":1,"end":2}]),
                _ => bad["changes"][0]["removed"][0]["kind"] = json!("location"),
            }
            assert!(verify(&bad).is_err(), "{field}");
        }
    }
}
