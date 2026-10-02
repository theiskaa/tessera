//! Reviewed corrected full gold and native hard-objective Seen bindings.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::Receipt;

pub(super) const SCHEMA: &str = "training_seen_evaluation_v5_1912_hard_v1";
const GROUPS: [&str; 3] = ["person", "org_contrast", "address"];
const KINDS: [&str; 5] = ["person", "org", "address", "email", "phone"];

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Binding {
    config: Receipt,
    typed_manifest: Receipt,
    source_successor_receipt: Receipt,
    selected_real_sources: Vec<Source>,
    membership: Receipt,
    base_native_preview: Receipt,
    selector_draft: Receipt,
    native_selector_proof: Receipt,
    hard_dose_preview: Receipt,
    hard_objective: Receipt,
    learning_acceptance: Receipt,
    learning_spec: Receipt,
    learning_invariants: Receipt,
    learning_checker: Receipt,
    learning_checker_successor: Receipt,
    promotion_checker: Receipt,
    same_gold_baseline: Receipt,
    source_delta_reviews: Vec<Receipt>,
    production_source_reviews: Vec<Receipt>,
    native_dose_reviews: Vec<Receipt>,
    evaluation_reviews: Vec<Receipt>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Source {
    source_slot: usize,
    path: PathBuf,
    sha256: String,
}

fn reviewed(pin: &Receipt, review: &Value) -> anyhow::Result<()> {
    ensure!(
        pin.path.is_absolute() && crate::fullmix_rms::valid_sha(&pin.sha256) && binds(review, pin),
        "v5 Seen artifact lacks an exact independent review binding"
    );
    pin.bytes()?;
    Ok(())
}

fn binds(review: &Value, pin: &Receipt) -> bool {
    let path = pin.path.to_string_lossy();
    let files = review
        .get("reviewed_files")
        .or_else(|| review.get("checked_hashes"));
    files.is_some_and(|files| {
        files[path.as_ref()] == pin.sha256
            || files.as_array().is_some_and(|rows| {
                rows.iter()
                    .any(|row| row["path"] == path.as_ref() && row["sha256"] == pin.sha256)
            })
    })
}

fn read(pin: &Receipt, review: &Value) -> anyhow::Result<Value> {
    reviewed(pin, review)?;
    Ok(serde_json::from_slice(&pin.bytes()?)?)
}

fn rows(pin: &Receipt, review: &Value) -> anyhow::Result<Vec<Value>> {
    reviewed(pin, review)?;
    std::str::from_utf8(&pin.bytes()?)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()
        .map_err(Into::into)
}

fn pins(value: &Value) -> anyhow::Result<Vec<Receipt>> {
    let mut found = Vec::new();
    if let Some(object) = value.as_object() {
        if object.contains_key("path") && object.contains_key("sha256") {
            found.push(serde_json::from_value(
                json!({"path":value["path"],"sha256":value["sha256"]}),
            )?);
        } else {
            for value in object.values() {
                found.extend(pins(value)?);
            }
        }
    } else if let Some(array) = value.as_array() {
        for value in array {
            found.extend(pins(value)?);
        }
    }
    Ok(found)
}

fn in_closure(plan: &Value, pin: &Receipt) -> anyhow::Result<()> {
    let closure: Vec<Receipt> = serde_json::from_value(plan["inputs"].clone())?;
    ensure!(
        closure
            .iter()
            .any(|input| input.path == pin.path && input.sha256 == pin.sha256),
        "v5 Seen referenced artifact omitted from input closure"
    );
    Ok(())
}

fn review_pair(reviews: &[Receipt], required: &[&Receipt], review: &Value) -> anyhow::Result<()> {
    ensure!(
        reviews.len() == 2
            && reviews[0].sha256 != reviews[1].sha256
            && std::fs::canonicalize(&reviews[0].path)? != std::fs::canonicalize(&reviews[1].path)?,
        "v5 Seen requires two distinct independent receipts"
    );
    for pin in reviews {
        let value = read(pin, review)?;
        ensure!(
            value["passed"] == true
                && value["blockers"].as_array().is_some_and(Vec::is_empty)
                && value["fit_approved"] == false,
            "v5 Seen evidence review failed or grants fitting approval"
        );
        for artifact in required {
            ensure!(
                binds(&value, artifact),
                "v5 Seen review omits a required exact artifact"
            );
        }
    }
    Ok(())
}

/// Fixed model-kind support selected only through the distinct hard Seen schema.
pub(super) fn supports(plan: &Value, review: &Value) -> anyhow::Result<[usize; 3]> {
    ensure!(
        plan["schema"] == SCHEMA,
        "unsupported v5 training-seen plan schema"
    );
    validate_support(plan)?;
    let binding: Binding = serde_json::from_value(plan["current_input_binding"].clone())?;
    let required = pins(&serde_json::to_value(&binding)?)?;
    let closure: Vec<Receipt> = serde_json::from_value(plan["inputs"].clone())?;
    for pin in &required {
        reviewed(pin, review)?;
        ensure!(
            closure
                .iter()
                .any(|input| input.path == pin.path && input.sha256 == pin.sha256),
            "v5 Seen current binding omitted from input closure"
        );
    }
    for pin in pins(&plan["phase_evidence"])? {
        reviewed(&pin, review)?;
    }
    let cfg = crate::config::load(&binding.config.path)?;
    let detector = cfg.detector.as_ref().context("v5 Seen detector absent")?;
    let settings = detector
        .typed_synthetic
        .as_ref()
        .context("v5 Seen typed settings absent")?;
    ensure!(
        serde_json::to_value(&settings.manifest)? == serde_json::to_value(&binding.typed_manifest)?
            && serde_json::to_value(&settings.corrected_real_sources)?
                == serde_json::to_value(&binding.source_successor_receipt)?,
        "v5 Seen config typed/corrected source crosslink differs"
    );
    crate::typed_synthetic::input_paths(&cfg)?;
    let successor = read(&binding.source_successor_receipt, review)?;
    ensure!(
        successor["schema"] == "reviewed_real_source_successor_v1"
            && successor["gold"] == plan["gold"]["real"]
            && successor["reviews"] == serde_json::to_value(&binding.source_delta_reviews)?,
        "v5 Seen source successor/gold/review crosslink differs"
    );
    for pin in pins(&successor)? {
        reviewed(&pin, review)?;
        ensure!(
            closure
                .iter()
                .any(|input| input.path == pin.path && input.sha256 == pin.sha256),
            "v5 Seen successor artifact omitted from input closure"
        );
    }
    let real_pin: Receipt = serde_json::from_value(plan["gold"]["real"].clone())?;
    let authored_pin: Receipt = serde_json::from_value(plan["gold"]["authored"].clone())?;
    let gold = rows(&real_pin, review)?;
    let members = read(&binding.membership, review)?;
    validate_sources(
        &binding.selected_real_sources,
        &detector.silver,
        &gold,
        &members,
        review,
    )?;
    review_pair(&binding.source_delta_reviews, &[&real_pin], review)?;
    let base = read(&binding.base_native_preview, review)?;
    let draft = read(&binding.selector_draft, review)?;
    let proof = read(&binding.native_selector_proof, review)?;
    let dose = read(&binding.hard_dose_preview, review)?;
    validate_native(
        &binding,
        &base,
        &draft,
        &proof,
        &dose,
        &gold,
        &rows(&authored_pin, review)?,
    )?;
    review_pair(
        &binding.native_dose_reviews,
        &[
            &binding.config,
            &binding.base_native_preview,
            &binding.selector_draft,
            &binding.native_selector_proof,
            &binding.hard_dose_preview,
            &real_pin,
            &authored_pin,
        ],
        review,
    )?;
    let objective = read(&binding.hard_objective, review)?;
    let contract: crate::fullmix_hard::Contract = serde_json::from_value(objective.clone())?;
    contract.validate_identity()?;
    reviewed(&contract.measured_batch_source, review)?;
    ensure!(
        closure
            .iter()
            .any(|input| input.path == contract.measured_batch_source.path
                && input.sha256 == contract.measured_batch_source.sha256),
        "v5 Seen measured calculator omitted from input closure"
    );
    ensure!(
        objective["selector_draft"] == serde_json::to_value(&binding.selector_draft)?
            && objective["native_selector_proof"]
                == serde_json::to_value(&binding.native_selector_proof)?
            && objective["hard_dose_preview"] == serde_json::to_value(&binding.hard_dose_preview)?
            && objective["corrected_sources"]
                == serde_json::to_value(&binding.source_successor_receipt)?
            && objective["corrected_gold"] == plan["gold"]["real"]
            && objective["source_reviews"]
                == serde_json::to_value(&binding.production_source_reviews)?
            && objective["native_dose_reviews"]
                == serde_json::to_value(&binding.native_dose_reviews)?,
        "v5 Seen hard objective crosslinks differ"
    );
    ensure!(
        binding.production_source_reviews.iter().all(|pin| binding
            .source_delta_reviews
            .iter()
            .all(|data| pin.sha256 != data.sha256)),
        "v5 Seen production source review reused a data-label review"
    );
    review_pair(
        &binding.production_source_reviews,
        &[
            &binding.config,
            &binding.base_native_preview,
            &binding.selector_draft,
            &binding.native_selector_proof,
            &binding.hard_dose_preview,
            &contract.measured_batch_source,
            &real_pin,
            &authored_pin,
        ],
        review,
    )?;
    let acceptance = read(&binding.learning_acceptance, review)?;
    let spec = read(&binding.learning_spec, review)?;
    let invariants = read(&binding.learning_invariants, review)?;
    let baseline = read(&binding.same_gold_baseline, review)?;
    validate_learning(
        plan,
        &binding,
        &spec,
        &acceptance,
        &invariants,
        &baseline,
        review,
    )?;
    review_pair(
        &binding.evaluation_reviews,
        &[
            &binding.learning_acceptance,
            &binding.learning_spec,
            &binding.learning_invariants,
            &binding.same_gold_baseline,
            &binding.learning_checker,
            &binding.learning_checker_successor,
            &binding.promotion_checker,
            &real_pin,
            &authored_pin,
        ],
        review,
    )?;
    Ok([9817, 1912, 600])
}

fn validate_support(plan: &Value) -> anyhow::Result<()> {
    ensure!(
        plan["model_gold_support"]
            == json!({"authored":{"person":600,"org":600,"address":120},
            "real":{"person":9817,"org":1912,"address":600}})
            && plan["declared_real_gold_support"]
                == json!({"person":9817,"org":1912,"address":600,"email":226,"phone":335}),
        "v5 Seen exact model and five-kind declared supports differ"
    );
    Ok(())
}

fn validate_sources(
    sources: &[Source],
    selected: &[String],
    gold: &[Value],
    membership: &Value,
    review: &Value,
) -> anyhow::Result<()> {
    ensure!(
        sources.len() == 34 && selected.len() == 34 && gold.len() == 1799,
        "v5 Seen requires the complete 34-source/1799-row pool"
    );
    let members = membership["real"]
        .as_array()
        .context("v5 Seen membership absent")?;
    ensure!(
        members.len() == gold.len(),
        "v5 Seen membership count differs"
    );
    let mut index = 0;
    let mut counts = [0usize; 5];
    let mut distinct = BTreeSet::new();
    for (slot, source) in sources.iter().enumerate() {
        ensure!(
            source.source_slot == slot
                && source.path == std::path::Path::new(&selected[slot])
                && distinct.insert(std::fs::canonicalize(&source.path)?),
            "v5 Seen selected source order or uniqueness differs"
        );
        let pin = Receipt {
            path: source.path.clone(),
            sha256: source.sha256.clone(),
        };
        for (physical, row) in rows(&pin, review)?.iter().enumerate() {
            let document = gold.get(index).context("v5 Seen gold row absent")?;
            let member = members.get(index).context("v5 Seen extra source row")?;
            ensure!(
                document["name"] == format!("seen-real-{slot:02}-{physical:04}")
                    && document["input"] == row["text"]
                    && document["country"] == "US"
                    && row["country"] == "US"
                    && member["name"] == document["name"]
                    && member["id"] == row["id"]
                    && member["source_slot"] == slot
                    && member["physical_row_1_based"] == physical + 1
                    && member["native_index"] == 170569 + index
                    && member["source"] == serde_json::to_value(&pin)?,
                "v5 Seen exact source/gold/native membership differs"
            );
            counts
                .iter_mut()
                .zip(source_entities(row, document)?)
                .for_each(|(total, count)| *total += count);
            index += 1;
        }
    }
    ensure!(
        index == gold.len() && counts == [9817, 1912, 600, 226, 335],
        "v5 Seen full declared source/gold supports differ"
    );
    Ok(())
}

fn source_entities(row: &Value, document: &Value) -> anyhow::Result<[usize; 5]> {
    let text = row["text"].as_str().context("v5 Seen source text absent")?;
    let mut counts = [0usize; 5];
    let mut expected = Vec::new();
    let mut spans = Vec::new();
    for entity in row["entities"]
        .as_array()
        .context("v5 Seen annotations absent")?
    {
        let kind = KINDS
            .iter()
            .position(|kind| entity["kind"] == *kind)
            .context("v5 Seen unsupported declared kind")?;
        let start = usize::try_from(entity["start"].as_u64().context("v5 Seen start absent")?)?;
        let end = usize::try_from(entity["end"].as_u64().context("v5 Seen end absent")?)?;
        let span = text
            .get(start..end)
            .filter(|span| !span.is_empty())
            .context("v5 Seen nonempty UTF8 source span differs")?;
        ensure!(
            entity.as_object().is_some_and(|entity| entity.len() == 3),
            "v5 Seen source entity shape differs"
        );
        expected.push(json!({"kind":KINDS[kind],"start":start,"end":end,"text":span}));
        spans.push((start, end));
        counts[kind] += 1;
    }
    spans.sort_unstable();
    ensure!(
        spans.windows(2).all(|pair| pair[0].1 <= pair[1].0)
            && document["expected"] == json!(expected),
        "v5 Seen five-kind gold span/order/overlap differs"
    );
    Ok(counts)
}

fn canonical(value: &Value) -> Value {
    match value {
        Value::Object(object) => {
            let sorted: BTreeMap<_, _> = object
                .iter()
                .map(|(key, value)| (key.clone(), canonical(value)))
                .collect();
            Value::Object(sorted.into_iter().collect())
        }
        Value::Array(values) => Value::Array(values.iter().map(canonical).collect()),
        value => value.clone(),
    }
}

fn expected_hash(value: &Value) -> anyhow::Result<String> {
    Ok(crate::export::sha256_hex(&serde_json::to_vec(&canonical(
        value,
    ))?))
}

fn validate_native(
    binding: &Binding,
    base: &Value,
    draft: &Value,
    proof: &Value,
    dose: &Value,
    real: &[Value],
    authored: &[Value],
) -> anyhow::Result<()> {
    let native = &base["typed_synthetic"];
    ensure!(
        base["config_sha256"] == binding.config.sha256
            && base["silver"]["unreachable_spans"] == 0
            && base["silver"]["documents"] == 1799
            && base["silver"]["pieces"] == 1799
            && base["silver"]["by_country"]["US"]["person"] == 9817
            && base["silver"]["by_country"]["US"]["org"] == 1912
            && base["silver"]["by_country"]["US"]["address"] == 600
            && native["uniform_synthetic_pool"] == 170000
            && native["full_synthetic_boundary"] == 170569
            && native["real_documents"] == 1799
            && native["planned_updates"] == 4000
            && native["model_initialized"] == false
            && native["model_forwards"] == 0
            && native["optimizer_updates_executed"] == 0
            && native["batches"]
                .as_array()
                .is_some_and(|rows| rows.len() == 4000),
        "v5 Seen native base preview config/pool/budget differs"
    );
    let evidence = &draft["evidence"];
    ensure!(
        proof["draft"] == serde_json::to_value(&binding.selector_draft)?,
        "v5 Seen native selector draft differs"
    );
    ensure!(
        draft["groups"] == json!(GROUPS)
            && draft["lambda_each_proposal"] == 0.0625
            && draft["model_operations"] == 0
            && draft["training_ready"] == false
            && draft["records"]
                .as_array()
                .is_some_and(|rows| rows.len() == 541)
            && proof["passed"] == true
            && proof["evidence"] == *evidence
            && proof["full_parent_rows"] == 541
            && proof["group_order"] == json!(GROUPS)
            && proof["lambda_each_proposal"] == 0.0625
            && proof["model_initialized"] == false
            && proof["model_forwards"] == 0
            && proof["optimizer_updates"] == 0
            && proof["training_ready"] == false
            && proof["rows"]
                .as_array()
                .is_some_and(|rows| rows.len() == 541)
            && dose["passed"] == true
            && dose["fit_approved"] == false
            && dose["training_ready"] == false
            && dose["model_initialized"] == false
            && dose["model_forwards"] == 0
            && dose["optimizer_updates"] == 0
            && dose["inputs"]
                == json!([
                    binding.config,
                    binding.selector_draft,
                    binding.native_selector_proof
                ])
            && dose["native_pool_documents"] == 172368
            && dose["uniform_synthetic_pool"] == 170000
            && dose["authored_documents"] == 569
            && dose["real_documents"] == 1799
            && dose["selected_full_parent_rows"] == 541
            && dose["planned_updates"] == 4000
            && dose["learning_rate_horizon"] == 7000
            && dose["group_order"] == json!(GROUPS)
            && dose["lambda_each"] == 0.0625
            && dose["base_batch_sequence_sha256"] == native["batch_sequence_sha256"]
            && dose["base_denominators_learning_rates_order_identical"] == true
            && dose["full_original_native_context_features_breaks_labels_verified"] == true
            && dose["batches"]
                .as_array()
                .is_some_and(|rows| rows.len() == 4000),
        "v5 Seen full native hard preview/mask crosslink differs"
    );
    for (base, hard) in native["batches"]
        .as_array()
        .context("v5 Seen base batches absent")?
        .iter()
        .zip(
            dose["batches"]
                .as_array()
                .context("v5 Seen hard batches absent")?,
        )
    {
        ensure!(
            base["optimizer_step_index"] == hard["update_index"]
                && base["indices"] == hard["ordered_native_indices"]
                && base["native_weighted_token_denominator"] == hard["base_denominator"]
                && base["scheduled_learning_rate"] == hard["scheduled_learning_rate"],
            "v5 Seen hard/base native batch order/denominator/LR differs"
        );
    }
    let proof_rows = proof["rows"]
        .as_array()
        .context("v5 Seen mask rows absent")?;
    let dose_rows = dose["rows"]
        .as_array()
        .context("v5 Seen dose rows absent")?;
    ensure!(
        dose_rows.len() == proof_rows.len(),
        "v5 Seen selected dose rows differ"
    );
    let names: BTreeMap<_, _> = authored
        .iter()
        .enumerate()
        .map(|(i, row)| (row["name"].clone(), (170000 + i, row)))
        .chain(
            real.iter()
                .enumerate()
                .map(|(i, row)| (row["name"].clone(), (170569 + i, row))),
        )
        .map(|(name, row)| {
            Ok((
                name.as_str()
                    .context("v5 Seen gold name absent")?
                    .to_owned(),
                row,
            ))
        })
        .collect::<anyhow::Result<_>>()?;
    let mut selected = BTreeSet::new();
    for (mask, row) in proof_rows.iter().zip(dose_rows) {
        let name = mask["name"]
            .as_str()
            .context("v5 Seen selected name absent")?;
        let (index, gold) = names
            .get(name)
            .context("v5 Seen selected parent absent from full gold")?;
        ensure!(
            selected.insert(name)
                && row["name"] == name
                && row["actual_native_index"] == *index
                && mask["text_sha256"]
                    == crate::export::sha256_hex(
                        gold["input"]
                            .as_str()
                            .context("v5 Seen selected full text absent")?
                            .as_bytes()
                    )
                && mask["expected_sha256"] == expected_hash(&gold["expected"])?
                && row["native_encoding_sha256"] == mask["native_encoding_sha256"]
                && row["binary_masks_sha256"] == mask["binary_masks_sha256"]
                && row["presentations"].as_u64().is_some_and(|n| n > 0),
            "v5 Seen selected parent/current index/native mask identity differs"
        );
    }
    let summaries = dose["summaries"]
        .as_array()
        .context("v5 Seen hard dose summaries absent")?;
    ensure!(
        summaries.len() == 3,
        "v5 Seen hard dose summary groups differ"
    );
    for (summary, group) in summaries.iter().zip(GROUPS) {
        ensure!(
            summary["group"] == group,
            "v5 Seen hard dose summary order differs"
        );
        for role in ["positive", "O"] {
            ensure!(
                summary[role]["selected_tokens"]
                    .as_u64()
                    .is_some_and(|n| n > 0)
                    && summary[role]["zero_base_dose_tokens"] == 0
                    && summary[role]["zero_auxiliary_dose_tokens"] == 0,
                "v5 Seen hard dose has unpresented selected tokens"
            );
        }
    }
    Ok(())
}

fn original_learning_freeze<'a>(
    evidence: &'a [(Receipt, Value)],
    sha256: &str,
    scope: &str,
) -> anyhow::Result<&'a Value> {
    let mut selected = None;
    for (pin, value) in evidence {
        if pin.sha256 == sha256 {
            ensure!(
                value["scope"] == scope,
                "v5 Seen original learning freeze scope differs"
            );
            ensure!(
                selected.replace(value).is_none(),
                "v5 Seen original learning freeze duplicated"
            );
        }
    }
    selected.context("v5 Seen original learning freeze absent")
}

fn validate_learning(
    plan: &Value,
    binding: &Binding,
    spec: &Value,
    acceptance: &Value,
    invariants: &Value,
    baseline: &Value,
    review: &Value,
) -> anyhow::Result<()> {
    ensure!(
        binding.learning_checker.sha256
            == "793e1a0a479a40d751363c13c6d41e0e2eaccac957ce37e33a46085c3daa6434",
        "v5 Seen requires the reviewed hard-scope learning checker successor"
    );
    for pin in [&binding.learning_checker, &binding.promotion_checker] {
        reviewed(pin, review)?;
        ensure!(
            std::str::from_utf8(&pin.bytes()?)?.contains("context96-rms-hard-fit-v1"),
            "v5 Seen evaluator lacks explicit hard-scope identity"
        );
    }
    let successor = read(&binding.learning_checker_successor, review)?;
    let old_checker: Receipt = serde_json::from_value(successor["old_checker"].clone())?;
    reviewed(&old_checker, review)?;
    in_closure(plan, &old_checker)?;
    ensure!(
        successor["schema"] == "hard_objective_learning_checker_successor_v1"
            && successor["new_checker"] == serde_json::to_value(&binding.learning_checker)?
            && old_checker.sha256
                == "b9f23f8fa82926986f771985ebc254122d840c730b8c5cc5ae37632717595a2c"
            && successor["calculation_changes"] == 0
            && successor["threshold_changes"] == 0
            && successor["acceptance_bytes_identical"] == true
            && successor["model_operations"] == 0
            && successor["fit_approved"] == false
            && successor["release_ready"] == false,
        "v5 Seen learning checker successor crosslink differs"
    );
    let previous = String::from_utf8(old_checker.bytes()?)?;
    ensure!(
        previous.contains("'context96-rms-fit-v1'")
            && binding.learning_checker.bytes()?
                == previous
                    .replace("'context96-rms-fit-v1'", "'context96-rms-hard-fit-v1'")
                    .as_bytes(),
        "v5 Seen learning checker changed more than diagnostic scope"
    );
    ensure!(
        spec["scope"] == "frozen training-seen learning acceptance; no unseen accuracy claim"
            && spec["gold"] == plan["gold"]
            && spec["required_filtered_f1"] == 0.95
            && spec["required_raw_and_filtered_reporting"] == true
            && spec["require_no_empty_or_missing_groups"] == true
            && spec["release_ready_claim"] == false
            && spec["automatic_training_retry"] == false
            && spec["authored_negative_names"] == plan["negative_document_names"]
            && spec["groups"]
                .as_object()
                .is_some_and(|groups| groups.len() == 9)
            && spec["organization_controls"]
                .as_array()
                .is_some_and(|controls| controls.len() == 54)
            && invariants["passed"] == true
            && invariants["training_ready"] == false
            && invariants["release_ready"] == false
            && invariants["model_operations"] == 0,
        "v5 Seen corrected learning spec/invariant contract differs"
    );
    let lineage: Vec<Receipt> = serde_json::from_value(invariants["evidence"].clone())?;
    let mut evidence = Vec::new();
    for pin in lineage {
        reviewed(&pin, review)?;
        in_closure(plan, &pin)?;
        if pin
            .path
            .extension()
            .is_some_and(|extension| extension == "json")
        {
            let value = read(&pin, review)?;
            evidence.push((pin, value));
        }
    }
    let original_spec = original_learning_freeze(
        &evidence,
        "6007235876d2a008376b66edea3ce07361294460757cb78e29e7d637ba2866fd",
        "frozen training-seen learning acceptance; no unseen accuracy claim",
    )?;
    let original_acceptance = original_learning_freeze(
        &evidence,
        "f4b1c2bbe6a2a4683ab27bed3ba99dcf793be2fe6dc7916b28156ee9497ee9a2",
        "next candidate promotion requirements fixed before changing training data or fitting",
    )?;
    for key in [
        "groups",
        "organization_controls",
        "authored_negative_names",
        "required_filtered_f1",
    ] {
        ensure!(
            spec[key] == original_spec[key],
            "v5 Seen original learning groups/controls changed"
        );
    }
    for key in [
        "goal_exact_f1_each_supported_kind",
        "confidence_policy",
        "next_candidate_must",
        "required_before_fit",
    ] {
        ensure!(
            acceptance[key] == original_acceptance[key],
            "v5 Seen original promotion gates changed"
        );
    }
    let outputs: Vec<Receipt> = serde_json::from_value(invariants["outputs"].clone())?;
    for pin in [
        &binding.learning_spec,
        &binding.learning_acceptance,
        &binding.same_gold_baseline,
    ] {
        ensure!(
            outputs
                .iter()
                .any(|output| output.path == pin.path && output.sha256 == pin.sha256),
            "v5 Seen corrected learning invariant output differs"
        );
    }
    ensure!(
        acceptance["scope"]
            == "same corrected gold next-candidate acceptance basis; no fitting approval"
            && acceptance["baseline_result"] == serde_json::to_value(&binding.same_gold_baseline)?
            && acceptance["training_authorized_by_this_file"] == false
            && acceptance["automatic_retry"] == false
            && baseline["scope"]
                == "frozen training-seen learning acceptance; no unseen accuracy claim"
            && baseline["release_ready_claim"] == false
            && baseline["automatic_training_retry"] == false
            && baseline["rescore_lineage"]["corrected_gold"] == plan["gold"]["real"]
            && baseline["rescore_lineage"]["native_predictions_unchanged"] == true
            && baseline["rescore_lineage"]["original_native_gold_binding_retained"] == true
            && baseline["rescore_lineage"]["model_forwards"] == 0
            && baseline["rescore_lineage"]["optimizer_updates"] == 0
            && baseline["rescore_lineage"]["label_correction_effect_is_model_improvement"] == false,
        "v5 Seen same-gold baseline lineage differs"
    );
    let original_result: Receipt =
        serde_json::from_value(baseline["rescore_lineage"]["original_result"].clone())?;
    in_closure(plan, &original_result)?;
    ensure!(
        baseline["provenance"] == read(&original_result, review)?["provenance"],
        "v5 Seen baseline native provenance was rewritten"
    );
    let groups = spec["groups"]
        .as_object()
        .context("v5 Seen learning groups absent")?;
    ensure!(
        baseline["groups"]
            .as_object()
            .is_some_and(|actual| actual.len() == groups.len())
            && acceptance["baseline_groups"]
                .as_object()
                .is_some_and(|actual| actual.len() == groups.len()),
        "v5 Seen same-gold baseline group coverage differs"
    );
    for group in groups.keys() {
        for kind in ["person", "org", "address"] {
            ensure!(
                acceptance["baseline_groups"][group][kind]
                    == baseline["groups"][group][kind]["filtered"],
                "v5 Seen promotion baseline differs from corrected rescore"
            );
        }
    }
    ensure!(
        baseline["groups"]["real:all"]["person"]["filtered"]["gold"] == 9817
            && baseline["groups"]["real:all"]["org"]["filtered"]["gold"] == 1912
            && baseline["groups"]["real:all"]["address"]["filtered"]["gold"] == 600,
        "v5 Seen baseline retains historical gold supports"
    );
    Ok(())
}

/// Complete v5 receipt closure; legacy plans use their original path collection.
pub(super) fn input_paths(plan: &Value) -> anyhow::Result<Vec<PathBuf>> {
    let binding: Binding = serde_json::from_value(plan["current_input_binding"].clone())?;
    Ok(pins(&serde_json::to_value(binding)?)?
        .into_iter()
        .map(|pin| pin.path)
        .collect())
}

#[cfg(test)]
#[path = "hard_tests.rs"]
mod tests;
