//! Exact multi-entity row deltas extending the preserved three-source correction chain.

use super::*;

pub(super) fn verify_prior(corrections: &Value) -> anyhow::Result<Pin> {
    ensure!(
        corrections["schema"] == "reviewed_entity_deltas_v4"
            && corrections["correction_count"] == 5
            && corrections["affected_shard_count"] == 5,
        "unsupported multi-entity correction contract"
    );
    let prior_pin = file_pin(&corrections["prior_corrections"])?;
    let prior = read(&prior_pin)?;
    let old_changes = prior["changes"]
        .as_array()
        .context("prior changes absent")?;
    let changes = corrections["changes"]
        .as_array()
        .context("changes absent")?;
    ensure!(
        prior["correction_count"] == 3 && old_changes.len() == 3 && changes.len() == 5,
        "new correction chain must preserve original three and add two sources"
    );
    let source_paths: BTreeSet<_> = changes
        .iter()
        .map(|change| {
            change["original_path"]
                .as_str()
                .context("original path absent")
        })
        .collect::<anyhow::Result<_>>()?;
    ensure!(source_paths.len() == 5, "correction sources repeat");
    for old in old_changes {
        let mut expected = old.clone();
        let entity = old["entity"].clone();
        expected["entity_delta"] = match old["action"].as_str() {
            Some("add") => json!({"added":[entity], "removed":[]}),
            Some("remove") => json!({"added":[], "removed":[entity]}),
            _ => anyhow::bail!("prior action unsupported"),
        };
        ensure!(
            changes.iter().any(|change| change == &expected),
            "original correction changed or omitted"
        );
    }
    Ok(prior_pin)
}

fn span_key(entity: &Value, text: &str) -> anyhow::Result<(String, u64, u64)> {
    let object = entity.as_object().context("entity not object")?;
    ensure!(object.len() == 3, "entity fields differ");
    let kind = entity["kind"].as_str().context("entity kind absent")?;
    ensure!(
        ["person", "org", "address", "email", "phone"].contains(&kind),
        "entity kind unsupported"
    );
    let start = entity["start"].as_u64().context("entity start invalid")?;
    let end = entity["end"].as_u64().context("entity end invalid")?;
    let start_index = usize::try_from(start)?;
    let end_index = usize::try_from(end)?;
    ensure!(
        start < end && text.get(start_index..end_index).is_some(),
        "entity endpoints invalid UTF8"
    );
    Ok((kind.to_owned(), start, end))
}

fn expected_row(original: &Value, change: &Value) -> anyhow::Result<Value> {
    let text = original["text"]
        .as_str()
        .context("correction text absent")?;
    ensure!(
        original["id"] == change["id"]
            && crate::export::sha256_hex(text.as_bytes()) == change["text_sha256"],
        "correction identity/text differs"
    );
    let delta = change["entity_delta"]
        .as_object()
        .context("entity delta absent")?;
    ensure!(delta.len() == 2, "entity delta fields differ");
    let added = delta
        .get("added")
        .and_then(Value::as_array)
        .context("added entities absent")?;
    let removed = delta
        .get("removed")
        .and_then(Value::as_array)
        .context("removed entities absent")?;
    ensure!(
        !added.is_empty() || !removed.is_empty(),
        "empty entity delta"
    );
    let mut keys = BTreeSet::new();
    for entity in added.iter().chain(removed) {
        ensure!(
            keys.insert(span_key(entity, text)?),
            "repeated or cancelling entity delta"
        );
    }
    let mut expected = original.clone();
    let entities = expected["entities"]
        .as_array_mut()
        .context("entities absent")?;
    for entity in removed {
        let positions: Vec<_> = entities
            .iter()
            .enumerate()
            .filter(|(_, candidate)| *candidate == entity)
            .map(|(index, _)| index)
            .collect();
        ensure!(
            positions.len() == 1,
            "removed entity not unique in original"
        );
        entities.remove(positions[0]);
    }
    for entity in added {
        ensure!(!entities.contains(entity), "added entity already exists");
        entities.push(entity.clone());
    }
    if !added.is_empty() {
        entities.sort_by_key(|entity| (entity["start"].as_u64(), entity["end"].as_u64()));
    }
    let mut spans = entities
        .iter()
        .map(|entity| span_key(entity, text))
        .collect::<anyhow::Result<Vec<_>>>()?;
    spans.sort_by_key(|(_, start, end)| (*start, *end));
    ensure!(
        spans.windows(2).all(|pair| pair[0].2 <= pair[1].1),
        "corrected entities overlap"
    );
    Ok(expected)
}

pub(super) fn verify(change: &Value) -> anyhow::Result<()> {
    let original = change["original_path"]
        .as_str()
        .context("original path absent")?;
    let candidate = change["candidate_path"]
        .as_str()
        .context("candidate path absent")?;
    let old_text = std::fs::read_to_string(original)?;
    let new_text = std::fs::read_to_string(candidate)?;
    let old: Vec<_> = old_text.split_inclusive('\n').collect();
    let new: Vec<_> = new_text.split_inclusive('\n').collect();
    ensure!(
        old.len() == new.len() && change["rows"] == old.len(),
        "correction row count differs"
    );
    let row = usize::try_from(
        change["physical_row_1_based"]
            .as_u64()
            .context("physical row absent")?,
    )?;
    ensure!(
        row > 0 && row <= old.len(),
        "correction physical row invalid"
    );
    for (index, (before, after)) in old.iter().zip(new).enumerate() {
        if index + 1 != row {
            ensure!(*before == after, "unreviewed changed row");
        } else {
            let before: Value = serde_json::from_str(before)?;
            let after: Value = serde_json::from_str(after)?;
            ensure!(
                expected_row(&before, change)? == after,
                "not exact entities-only delta"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Value, Value) {
        let original = json!({"id":"fixture","country":"US","text":"John met BLS and STB.",
                              "source":"synthetic-test-only","entities":[{"kind":"person","start":0,"end":4}]});
        let change = json!({"id":"fixture","text_sha256":crate::export::sha256_hex(b"John met BLS and STB."),
                            "entity_delta":{"added":[{"kind":"org","start":9,"end":12},{"kind":"org","start":17,"end":20}],"removed":[]}});
        (original, change)
    }

    #[test]
    fn multiple_additions_preserve_person_and_metadata() {
        let (original, change) = fixture();
        let expected = expected_row(&original, &change).unwrap();
        assert_eq!(expected["entities"].as_array().unwrap().len(), 3);
        assert_eq!(expected["entities"][0], original["entities"][0]);
        for field in ["id", "country", "text", "source"] {
            assert_eq!(expected[field], original[field]);
        }
    }

    #[test]
    fn duplicate_missing_removal_overlap_and_bad_utf8_refuse() {
        let (original, change) = fixture();
        let mut duplicate = change.clone();
        duplicate["entity_delta"]["added"][1] = duplicate["entity_delta"]["added"][0].clone();
        assert!(expected_row(&original, &duplicate).is_err());
        let mut missing = change.clone();
        missing["entity_delta"]["removed"] = json!([{"kind":"org","start":0,"end":4}]);
        assert!(expected_row(&original, &missing).is_err());
        let mut overlap = change.clone();
        overlap["entity_delta"]["added"][0] = json!({"kind":"org","start":2,"end":7});
        assert!(expected_row(&original, &overlap).is_err());
        let mut unicode = original.clone();
        unicode["text"] = json!("Élodie");
        let bad = json!({"id":"fixture","text_sha256":crate::export::sha256_hex("Élodie".as_bytes()),
                        "entity_delta":{"added":[{"kind":"org","start":1,"end":2}],"removed":[]}});
        assert!(expected_row(&unicode, &bad).is_err());
    }
}
