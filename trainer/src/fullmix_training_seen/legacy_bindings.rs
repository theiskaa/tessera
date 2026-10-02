//! Exact current-input bindings for the separately versioned 1904-ORG seen plan.

use anyhow::{Context, ensure};
use serde_json::{Value, json};

use super::Receipt;

const SCHEMA: &str = "training_seen_evaluation_v4_1904_v1";

/// Fixed support contracts selected only by an explicit reviewed plan version.
pub(super) fn supports(plan: &Value, review: &Value) -> anyhow::Result<[usize; 3]> {
    match plan.get("schema") {
        None => Ok([9817, 1900, 600]),
        Some(schema) if schema == SCHEMA => {
            validate(plan, review)?;
            Ok([9817, 1904, 600])
        }
        Some(_) => anyhow::bail!("unsupported training-seen plan schema"),
    }
}

fn receipt(value: &Value, review: &Value) -> anyhow::Result<Receipt> {
    let pin: Receipt = serde_json::from_value(value.clone())?;
    ensure!(
        pin.path.is_absolute()
            && review["reviewed_files"][pin.path.to_string_lossy().as_ref()] == pin.sha256,
        "v4 training-seen artifact is not independently reviewed"
    );
    pin.bytes()?;
    Ok(pin)
}

fn read(value: &Value, review: &Value) -> anyhow::Result<Value> {
    Ok(serde_json::from_slice(&receipt(value, review)?.bytes()?)?)
}

fn rows(value: &Value, review: &Value) -> anyhow::Result<Vec<Value>> {
    let pin = receipt(value, review)?;
    std::str::from_utf8(&pin.bytes()?)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()
        .map_err(Into::into)
}

fn review_set(value: &Value, review: &Value) -> anyhow::Result<()> {
    let pins = value.as_array().context("v4 review set absent")?;
    ensure!(
        pins.len() == 2
            && pins[0]["sha256"] != pins[1]["sha256"]
            && std::fs::canonicalize(receipt(&pins[0], review)?.path)?
                != std::fs::canonicalize(receipt(&pins[1], review)?.path)?,
        "v4 review set must contain two distinct reviewed receipts"
    );
    for pin in pins {
        let value = read(pin, review)?;
        ensure!(
            value["passed"] == true && value["blockers"].as_array().is_some_and(Vec::is_empty),
            "v4 input review did not pass"
        );
    }
    Ok(())
}

fn validate(plan: &Value, review: &Value) -> anyhow::Result<()> {
    ensure!(
        plan["model_gold_support"]
            == json!({"authored":{"person":600,"org":600,"address":120},
                "real":{"person":9817,"org":1904,"address":600}}),
        "v4 training-seen exact supports differ"
    );
    for key in ["inputs", "phase_evidence"] {
        for pin in plan[key].as_array().context("v4 plan lineage absent")? {
            receipt(pin, review)?;
        }
    }
    let binding = &plan["current_input_binding"];
    let config_pin = receipt(&binding["config"], review)?;
    let cfg = crate::config::load(&config_pin.path)?;
    let detector = cfg.detector.as_ref().context("v4 detector config absent")?;
    let settings = detector
        .typed_synthetic
        .as_ref()
        .context("v4 typed inputs absent")?;
    ensure!(
        serde_json::to_value(&settings.manifest)? == binding["typed_manifest"],
        "v4 configuration typed manifest crosslink differs"
    );
    let typed = read(&binding["typed_manifest"], review)?;
    ensure!(
        typed["schema"] == "typed_synthetic_train_only_v1"
            && typed["real_amendments"] == binding["real_amendments"]
            && typed["real_corrections"] == binding["correction_manifest"]
            && typed["assembly_reviews"] == binding["metadata_reviews"],
        "v4 typed source crosslinks differ"
    );
    let amendments = read(&binding["real_amendments"], review)?;
    let corrections = read(&binding["correction_manifest"], review)?;
    ensure!(
        amendments["schema"] == "reviewed_training_amendments_v4"
            && corrections["schema"] == "reviewed_entity_deltas_v4"
            && corrections["correction_count"] == 5
            && corrections["changes"]
                .as_array()
                .is_some_and(|v| v.len() == 5),
        "v4 training-seen requires the exact five-row amendment version"
    );
    crate::typed_synthetic::input_paths(&cfg)?;
    review_set(&binding["metadata_reviews"], review)?;
    review_set(&binding["source_delta_reviews"], review)?;
    let selected = binding["selected_real_sources"]
        .as_array()
        .context("v4 selected sources absent")?;
    let sources = amendments["training_real_sources"]
        .as_array()
        .context("v4 amended sources absent")?;
    ensure!(
        selected.len() == 34 && sources.len() == 34 && detector.silver.len() == 34,
        "v4 training-seen source count differs"
    );
    for (slot, ((selected, source), path)) in selected
        .iter()
        .zip(sources)
        .zip(&detector.silver)
        .enumerate()
    {
        let pin = receipt(
            &json!({"path":selected["path"],"sha256":selected["sha256"]}),
            review,
        )?;
        ensure!(
            selected["source_slot"] == slot
                && source["source_slot"] == slot
                && source["path"] == *path
                && selected["path"] == source["path"]
                && selected["sha256"] == source["sha256"]
                && pin.path == std::path::Path::new(path),
            "v4 selected source order/hash differs"
        );
    }
    let result = read(&binding["native_data_check"], review)?;
    let preview = read(&binding["native_preview"], review)?;
    let native = &preview["typed_synthetic"];
    let supports = json!({"person":9817,"org":1904,"address":600});
    ensure!(
        result["passed"] == true
            && result["config"] == binding["config"]
            && result["typed_manifest"] == binding["typed_manifest"]
            && result["native_preview"] == binding["native_preview"]
            && preview["config_sha256"] == config_pin.sha256
            && preview["silver"]["unreachable_spans"] == 0
            && preview["silver"]["documents"] == 1799
            && preview["silver"]["pieces"] == 1799
            && preview["silver"]["by_country"]["US"]["person"] == supports["person"]
            && preview["silver"]["by_country"]["US"]["org"] == supports["org"]
            && preview["silver"]["by_country"]["US"]["address"] == supports["address"]
            && native["model_initialized"] == false
            && native["model_forwards"] == 0
            && native["optimizer_updates_executed"] == 0
            && native["planned_updates"] == 4000
            && native["uniform_synthetic_pool"] == 170000
            && native["full_synthetic_boundary"] == 170569
            && native["real_documents"] == 1799
            && native["batches"]
                .as_array()
                .is_some_and(|v| v.len() == 4000),
        "v4 native input preview binding/support differs"
    );
    let dose = read(&binding["target14_dose"], review)?;
    ensure!(
        dose["passed"] == true
            && dose["config"] == binding["config"]
            && dose["native_preview"] == binding["native_preview"]
            && dose["batch_sequence_sha256"] == native["batch_sequence_sha256"]
            && dose["target_count"] == 14
            && dose["current_training_support"] == supports
            && dose["model_initialized"] == false
            && dose["model_forwards"] == 0
            && dose["optimizer_updates_executed"] == 0,
        "v4 target14 dose crosslink differs"
    );
    let membership = read(&binding["membership"], review)?;
    let delta = read(&binding["gold_delta"], review)?;
    ensure!(
        delta["current_gold"] == plan["gold"]["real"]
            && delta["complete_five_physical_rows"]
                .as_array()
                .is_some_and(|v| v.len() == 5)
            && delta["changed_from_v2"]
                .as_array()
                .is_some_and(|v| v.len() == 2),
        "v4 gold amendment crosslink differs"
    );
    let gold = rows(&plan["gold"]["real"], review)?;
    let members = membership["real"]
        .as_array()
        .context("v4 real membership absent")?;
    ensure!(
        gold.len() == 1799 && members.len() == 1799,
        "v4 real gold membership count differs"
    );
    let mut index = 0usize;
    for (slot, source) in sources.iter().enumerate() {
        let pin = json!({"path":source["path"],"sha256":source["sha256"]});
        for (physical, row) in rows(&pin, review)?.iter().enumerate() {
            let document = gold.get(index).context("v4 gold row absent")?;
            let member = members.get(index).context("v4 member row absent")?;
            ensure!(
                document["input"] == row["text"]
                    && document["country"] == row["country"]
                    && member["name"] == document["name"]
                    && member["id"] == row["id"]
                    && member["source_slot"] == slot
                    && member["physical_row_1_based"] == physical + 1
                    && member["native_index"] == 170569 + index
                    && member["source"] == pin,
                "v4 gold source membership/order differs"
            );
            let text = row["text"].as_str().context("v4 source text absent")?;
            let expected = row["entities"]
                .as_array()
                .context("v4 source annotations absent")?
                .iter()
                .map(|entity| -> anyhow::Result<Value> {
                    let start = usize::try_from(
                        entity["start"].as_u64().context("v4 source start absent")?,
                    )?;
                    let end =
                        usize::try_from(entity["end"].as_u64().context("v4 source end absent")?)?;
                    let span = text
                        .get(start..end)
                        .context("v4 source UTF8 span differs")?;
                    let mut entity = entity.clone();
                    entity["text"] = json!(span);
                    Ok(entity)
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            ensure!(
                document["expected"] == json!(expected),
                "v4 gold differs from selected annotations"
            );
            index += 1;
        }
    }
    ensure!(index == gold.len(), "v4 source/gold row count differs");
    Ok(())
}

#[cfg(test)]
#[path = "legacy_tests.rs"]
mod tests;
