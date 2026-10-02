//! Reviewed real-label successors for zero-update data checks only.

use super::*;

const PRIOR_CONFIG_SHA256: &str =
    "9722c10e0218e12fe4bc4f064f681366fc9fbdf8e37475770d402440cc0c6beb";

#[cfg(test)]
#[path = "typed_synthetic_successor_tests.rs"]
mod tests;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Successor {
    schema: String,
    prior_config: Pin,
    source_deltas: Pin,
    reviews: Vec<Pin>,
    gold: Pin,
}

#[derive(Debug)]
/// Validated original generation selection and additional snapshot artifacts.
pub(super) struct Verified {
    pub(super) prior_sources: Vec<String>,
    pub(super) paths: Vec<PathBuf>,
}

fn settings(cfg: &Config) -> anyhow::Result<&Settings> {
    cfg.detector
        .as_ref()
        .and_then(|d| d.typed_synthetic.as_ref())
        .context("successor needs typed settings")
}

fn review_bindings(value: &Value) -> anyhow::Result<BTreeMap<String, String>> {
    let files = match (value.get("reviewed_files"), value.get("checked_hashes")) {
        (Some(files), None) | (None, Some(files)) => files,
        _ => anyhow::bail!("successor review needs one explicit file binding set"),
    };
    let mut bindings = BTreeMap::new();
    let mut insert = |pin: Pin| -> anyhow::Result<()> {
        ensure!(
            Path::new(&pin.path).is_absolute() && valid_hash(&pin.sha256),
            "review binding needs an absolute path and lowercase SHA256"
        );
        if let Some(previous) = bindings.insert(pin.path, pin.sha256.clone()) {
            ensure!(previous == pin.sha256, "conflicting reviewed file hashes");
        }
        Ok(())
    };
    if let Some(files) = files.as_object() {
        for (path, hash) in files {
            insert(Pin {
                path: path.clone(),
                sha256: hash.as_str().context("review hash not string")?.to_owned(),
            })?;
        }
    } else {
        for row in files
            .as_array()
            .context("review file binding set invalid")?
        {
            insert(serde_json::from_value(row.clone())?)?;
        }
    }
    Ok(bindings)
}

fn verify_reviews(reviews: &[Pin], required: &[Pin]) -> anyhow::Result<()> {
    ensure!(
        reviews.len() == 2
            && reviews[0].sha256 != reviews[1].sha256
            && std::fs::canonicalize(&reviews[0].path)? != std::fs::canonicalize(&reviews[1].path)?,
        "successor needs two distinct independent source/label review receipts"
    );
    for pin in reviews {
        let review = read(pin)?;
        ensure!(
            review["passed"] == true
                && review["blockers"].as_array().is_some_and(Vec::is_empty)
                && review["fit_approved"] == false,
            "successor review failed or does not retain fit refusal"
        );
        let bindings = review_bindings(&review)?;
        for artifact in required {
            ensure!(
                bindings.get(&artifact.path) == Some(&artifact.sha256),
                "successor review does not bind {}",
                artifact.path
            );
            artifact.verify()?;
        }
    }
    Ok(())
}

fn verify_config(cfg: &Config, prior: &Config, selected: &[String]) -> anyhow::Result<()> {
    let mut expected = serde_json::to_value(prior)?;
    expected["name"] = Value::String(cfg.name.clone());
    expected["detector"]["silver"] = serde_json::to_value(selected)?;
    expected["detector"]["typed_synthetic"]["corrected_real_sources"] =
        serde_json::to_value(&settings(cfg)?.corrected_real_sources)?;
    ensure!(
        serde_json::to_value(cfg)? == expected,
        "successor configuration changes more than name, reviewed sources and receipt"
    );
    Ok(())
}

fn verify_gold(gold: &Pin, selected: &[String]) -> anyhow::Result<()> {
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct GoldRow {
        name: String,
        country: String,
        input: String,
        expected: Vec<GoldEntity>,
    }
    #[derive(Deserialize)]
    #[serde(deny_unknown_fields)]
    struct GoldEntity {
        kind: String,
        start: usize,
        end: usize,
        text: String,
    }
    gold.verify()?;
    let rows = std::fs::read_to_string(&gold.path)?;
    let mut gold_rows = BTreeMap::new();
    for line in rows.lines() {
        let row: GoldRow = serde_json::from_str(line)?;
        ensure!(
            gold_rows.insert(row.name.clone(), row).is_none(),
            "duplicate successor gold name"
        );
    }
    ensure!(
        gold_rows.len() == 1799,
        "successor gold needs all 1799 rows"
    );
    for (slot, path) in selected.iter().enumerate() {
        for (index, line) in std::fs::read_to_string(path)?.lines().enumerate() {
            let source: Value = serde_json::from_str(line)?;
            let name = format!("seen-real-{slot:02}-{index:04}");
            let row = gold_rows.remove(&name).context("source gold row absent")?;
            ensure!(
                source["text"] == row.input && source["country"] == row.country,
                "successor gold text/country differs from selected source"
            );
            let mut expected = Vec::new();
            let mut spans = Vec::new();
            for entity in row.expected {
                ensure!(
                    ["person", "org", "address", "email", "phone"].contains(&entity.kind.as_str())
                        && entity.start < entity.end
                        && row.input.get(entity.start..entity.end) == Some(entity.text.as_str()),
                    "successor gold entity kind, UTF8 span or text invalid"
                );
                spans.push((entity.start, entity.end));
                expected.push(json!({"kind":entity.kind,"start":entity.start,"end":entity.end}));
            }
            spans.sort_unstable();
            ensure!(
                spans.windows(2).all(|pair| pair[0].1 <= pair[1].0),
                "successor gold entities overlap"
            );
            ensure!(
                source["entities"] == Value::Array(expected),
                "successor gold differs from all five selected annotation kinds"
            );
        }
    }
    ensure!(gold_rows.is_empty(), "successor gold has extra membership");
    Ok(())
}

struct Selection {
    sources: Vec<String>,
    files: Vec<Pin>,
}

fn source_selection(prior_sources: &[String], contracts: &[Value]) -> anyhow::Result<Selection> {
    ensure!(
        prior_sources.len() == 34 && (1..=34).contains(&contracts.len()),
        "successor needs 1..=34 distinct reviewed source contracts"
    );
    let mut selected = prior_sources.to_vec();
    let mut changed_slots = BTreeSet::new();
    let mut canonical_after = BTreeSet::new();
    let mut files = Vec::new();
    for contract in contracts {
        super::source_delta::verify(contract)?;
        let before = file_pin(&contract["before"])?;
        let after = file_pin(&contract["after"])?;
        let slots: Vec<_> = prior_sources
            .iter()
            .enumerate()
            .filter(|(_, path)| *path == &before.path)
            .map(|(slot, _)| slot)
            .collect();
        ensure!(
            slots.len() == 1,
            "source delta does not bind one exact prior slot"
        );
        let slot = slots[0];
        ensure!(
            changed_slots.insert(slot)
                && canonical_after.insert(std::fs::canonicalize(&after.path)?),
            "source successor repeats a slot or aliases another replacement"
        );
        selected[slot] = after.path.clone();
        files.extend([before, after]);
    }
    Ok(Selection {
        sources: selected,
        files,
    })
}

/// Verify a data-only successor without authorizing a fullmix recipe or model operation.
pub(super) fn verify(cfg: &Config) -> anyhow::Result<Option<Verified>> {
    let Some(pin) = cfg
        .detector
        .as_ref()
        .and_then(|d| d.typed_synthetic.as_ref())
        .and_then(|s| s.corrected_real_sources.as_ref())
    else {
        return Ok(None);
    };
    let receipt: Successor = serde_json::from_value(read(pin)?)?;
    ensure!(
        receipt.schema == "reviewed_real_source_successor_v1"
            && receipt.prior_config.sha256 == PRIOR_CONFIG_SHA256,
        "successor schema or exact reviewed prior config differs"
    );
    receipt.prior_config.verify()?;
    let prior = crate::config::load(Path::new(&receipt.prior_config.path))?;
    ensure!(
        settings(&prior)?.corrected_real_sources.is_none(),
        "successor cannot chain an unreviewed prior successor"
    );
    let prior_sources = &prior
        .detector
        .as_ref()
        .context("prior detector absent")?
        .silver;
    let actual = &cfg
        .detector
        .as_ref()
        .context("actual detector absent")?
        .silver;
    ensure!(
        prior_sources.len() == 34 && actual.len() == 34,
        "successor needs the exact 34 source slots"
    );
    let deltas = read(&receipt.source_deltas)?;
    ensure!(
        deltas["gold"] == serde_json::to_value(&receipt.gold)?
            && deltas["texts_changed"] == 0
            && deltas["historical_inputs_modified"] == false
            && deltas["training_ready"] == false
            && deltas["optimizer_updates"] == 0,
        "source delta gold or zero-update contract differs"
    );
    let contracts = deltas["source_contracts"]
        .as_array()
        .context("source contracts absent")?;
    let selection = source_selection(prior_sources, contracts)?;
    let selected = selection.sources;
    let mut required = vec![
        receipt.prior_config.clone(),
        receipt.source_deltas.clone(),
        receipt.gold.clone(),
    ];
    required.extend(selection.files);
    ensure!(
        actual == &selected,
        "actual silver sources differ from reviewed successors"
    );
    let distinct = selected
        .iter()
        .map(std::fs::canonicalize)
        .collect::<Result<BTreeSet<_>, _>>()?;
    ensure!(
        distinct.len() == 34,
        "successor source slots alias the same file"
    );
    verify_config(cfg, &prior, &selected)?;
    verify_reviews(&receipt.reviews, &required)?;
    verify_gold(&receipt.gold, &selected)?;
    let mut paths: Vec<_> = required.iter().map(|p| PathBuf::from(&p.path)).collect();
    paths.push(PathBuf::from(&pin.path));
    paths.extend(receipt.reviews.iter().map(|p| PathBuf::from(&p.path)));
    paths.extend(prior_sources.iter().chain(&selected).map(PathBuf::from));
    paths.sort();
    paths.dedup();
    Ok(Some(Verified {
        prior_sources: prior_sources.clone(),
        paths,
    }))
}
