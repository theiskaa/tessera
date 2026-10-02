//! Forward-only loss inspection of a frozen training memorization checkpoint.

use std::collections::BTreeSet;
use std::path::Path;

use anyhow::{Context, ensure};
use burn::backend::{Autodiff, NdArray};
use burn::data::dataloader::batcher::Batcher;
use burn::module::AutodiffModule;
use burn::prelude::*;
use serde_json::{Value, json};

use crate::config::Config;
use crate::dataset::{ParserBatch, ParserBatcher};
use crate::detector::{self, DETECTOR_LABELS, DetectorDoc};
use crate::net::TaggerNet;

fn hash(path: &Path) -> anyhow::Result<String> {
    Ok(crate::export::sha256_hex(&std::fs::read(path)?))
}

fn validate_marker(marker: &Value) -> anyhow::Result<()> {
    ensure!(
        ((marker["scope"] == "fixed training-only memorization diagnostic; never export"
            && marker["max_steps"].as_u64().is_some_and(|steps| steps > 0))
            || (crate::fullmix_rms::is_fullmix(marker)
                && crate::fullmix_rms::validate_marker_scope(marker).is_ok()))
            && marker["release_quality_claim"] == false,
        "loss inspection requires a training-only memorization diagnostic marker"
    );
    Ok(())
}

fn load_cases(
    gold: &Path,
    cfg: &Config,
    selection: &Value,
) -> anyhow::Result<(Vec<String>, Vec<DetectorDoc>)> {
    load_cases_with_limits(gold, cfg, selection, 32, 128)
}

pub(crate) fn load_cases_with_limits(
    gold: &Path,
    cfg: &Config,
    selection: &Value,
    count: usize,
    max_tokens: usize,
) -> anyhow::Result<(Vec<String>, Vec<DetectorDoc>)> {
    let text = std::fs::read_to_string(gold)?;
    let rows = text
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str::<Value>)
        .collect::<Result<Vec<_>, _>>()?;
    ensure!(
        rows.len() == count
            && rows
                .iter()
                .all(|row| row["doc_type"] == "training_memorization"),
        "loss inspection requires exactly {count} training_memorization cases"
    );
    let expected = selection["documents"]
        .as_array()
        .context("loss inspection requires the original selection manifest")?;
    ensure!(
        selection["selected_documents"].as_u64() == Some(count as u64)
            && expected.len() == rows.len(),
        "loss inspection selection document count differs from training cases"
    );
    let features = cfg.features.to_tessera();
    let mut docs = detector::load_development_gold(gold, &features)?;
    ensure!(
        docs.len() == rows.len(),
        "native loss case alignment changed"
    );
    let mut names = Vec::new();
    let mut texts = BTreeSet::new();
    for ((doc, row), selected) in docs.iter_mut().zip(&rows).zip(expected) {
        ensure!(
            texts.insert(doc.text.as_str()),
            "loss inspection duplicates training text"
        );
        ensure!(
            selected["text_sha256"] == crate::export::sha256_hex(doc.text.as_bytes())
                && selected["original_source_row"] == true
                && selected["derived_piece"] == false,
            "loss inspection case does not match its original training selection"
        );
        let mut declared: Vec<_> = doc
            .gold
            .iter()
            .map(|span| {
                json!({
                    "kind": detector::KINDS[span.kind].as_str(),
                    "start": span.start,
                    "end": span.end,
                })
            })
            .collect();
        let mut original = selected["gold"]
            .as_array()
            .context("selection is missing native gold")?
            .clone();
        let order = |span: &Value| {
            (
                span["start"].as_u64(),
                span["end"].as_u64(),
                span["kind"].as_str().map(str::to_owned),
            )
        };
        declared.sort_by_key(order);
        original.sort_by_key(order);
        ensure!(
            declared == original,
            "loss inspection native gold differs from training selection"
        );
        let mut encoded = detector::encode_document(&doc.text, &doc.gold, &features)?;
        encoded.country = "US".into();
        ensure!(
            !encoded.token_spans.is_empty()
                && encoded.token_spans.len() <= max_tokens
                && selected["content_tokens"].as_u64() == Some(encoded.token_spans.len() as u64),
            "loss inspection native token count differs from training selection"
        );
        doc.enc = encoded;
        names.push(
            row["name"]
                .as_str()
                .context("training case has no name")?
                .to_owned(),
        );
    }
    Ok((names, docs))
}

fn measure<B: Backend>(
    model: &TaggerNet<B>,
    docs: &[DetectorDoc],
    names: &[String],
    weights: &[f32],
    batch_size: usize,
    device: &B::Device,
    operator: crate::diagnostic_operator::ForwardOperator,
) -> anyhow::Result<Value> {
    ensure!(
        weights.len() == DETECTOR_LABELS
            && weights
                .iter()
                .all(|weight| weight.is_finite() && *weight > 0.0)
            && docs.len() == names.len()
            && batch_size > 0,
        "loss inspection has invalid dimensions or class weights"
    );
    let class_weights = Tensor::<B, 1>::from_data(
        burn::tensor::TensorData::new(weights.to_vec(), [weights.len()]),
        device,
    );
    let mut batches = Vec::new();
    let mut documents = Vec::new();
    let mut total_nll = 0.0;
    let mut total_mass = 0.0;
    for (batch_index, chunk) in docs.chunks(batch_size).enumerate() {
        let batch: ParserBatch<B> =
            ParserBatcher.batch(chunk.iter().map(|doc| doc.enc.clone()).collect(), device);
        let (logits, layer_activations) = model.diagnostic_forward_with_operator(
            batch.ngram_ids,
            batch.script,
            batch.shape,
            batch.flags,
            batch.mask.clone(),
            operator,
        )?;
        let [size, padded_length, labels] = logits.dims();
        ensure!(
            labels == DETECTOR_LABELS,
            "loss inspection label count changed"
        );
        let native_loss: f64 = crate::train::masked_loss(
            logits.clone(),
            batch.labels,
            batch.mask,
            class_weights.clone(),
        )
        .into_scalar()
        .elem();
        let values = logits
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|error| anyhow::anyhow!("reading loss diagnostic logits: {error:?}"))?;
        ensure!(
            native_loss.is_finite() && values.iter().all(|value| value.is_finite()),
            "loss inspection has non-finite logits or loss"
        );
        let mut batch_nll = 0.0;
        let mut batch_mass = 0.0;
        let mut batch_max = 0.0f64;
        let mut batch_centered_squares = 0.0;
        let mut batch_tokens = 0usize;
        let mut batch_class_range = 0.0f64;
        let mut padded_max = 0.0f64;
        for (index, doc) in chunk.iter().enumerate() {
            let mut nll = 0.0;
            let mut mass = 0.0;
            let mut max_abs = 0.0f64;
            let mut squares = 0.0;
            let mut centered_squares = 0.0;
            let mut max_class_range = 0.0f64;
            let mut margin_sum = 0.0;
            let mut minimum_gold_margin = f64::INFINITY;
            let mut negative_gold_margins = 0usize;
            let mut person_tokens = Vec::new();
            for position in 0..padded_length {
                let start = (index * padded_length + position) * labels;
                let token = &values[start..start + labels];
                if position >= doc.enc.labels.len() {
                    padded_max = token
                        .iter()
                        .fold(padded_max, |max, value| max.max(f64::from(value.abs())));
                    continue;
                }
                let target = usize::from(doc.enc.labels[position]);
                let weight = f64::from(*weights.get(target).context("invalid native gold label")?);
                let max = token
                    .iter()
                    .map(|value| f64::from(*value))
                    .fold(f64::NEG_INFINITY, f64::max);
                let min = token
                    .iter()
                    .map(|value| f64::from(*value))
                    .fold(f64::INFINITY, f64::min);
                let mean = token.iter().map(|value| f64::from(*value)).sum::<f64>() / labels as f64;
                let maximum_other = token
                    .iter()
                    .enumerate()
                    .filter(|(label, _)| *label != target)
                    .map(|(_, value)| f64::from(*value))
                    .fold(f64::NEG_INFINITY, f64::max);
                let gold_margin = f64::from(token[target]) - maximum_other;
                margin_sum += gold_margin;
                minimum_gold_margin = minimum_gold_margin.min(gold_margin);
                negative_gold_margins += usize::from(gold_margin < 0.);
                max_class_range = max_class_range.max(max - min);
                let shifted_logsumexp = token
                    .iter()
                    .map(|value| (f64::from(*value) - max).exp())
                    .sum::<f64>()
                    .ln();
                if matches!(target, 1 | 2) {
                    let &(start, end) = doc
                        .enc
                        .token_spans
                        .get(position)
                        .context("PERSON loss token bounds missing")?;
                    let token_nll = shifted_logsumexp - (f64::from(token[target]) - max);
                    person_tokens.push(json!({
                        "index": position, "start": start, "end": end,
                        "native_label": target, "unweighted_nll_f64": token_nll,
                        "gold_margin": gold_margin, "gold_probability_f64": (-token_nll).exp(),
                    }));
                }
                nll += weight * (shifted_logsumexp - (f64::from(token[target]) - max));
                mass += weight;
                for &value in token {
                    max_abs = max_abs.max(f64::from(value.abs()));
                    squares += f64::from(value).powi(2);
                    centered_squares += (f64::from(value) - mean).powi(2);
                }
            }
            ensure!(
                mass > 0.0,
                "loss inspection document has no weighted tokens"
            );
            let doc_index = batch_index * batch_size + index;
            documents.push(json!({
                "name": names[doc_index], "index": doc_index,
                "content_tokens": doc.enc.labels.len(), "weighted_token_mass": mass,
                "weighted_nll_sum_f64_reference": nll, "weighted_loss_f64_reference": nll / mass,
                "max_abs_logit": max_abs,
                "rms_logit": (squares / (doc.enc.labels.len() * labels) as f64).sqrt(),
                "centered_rms_logit": (centered_squares / (doc.enc.labels.len() * labels) as f64).sqrt(),
                "max_class_range": max_class_range,
                "mean_gold_margin": margin_sum / doc.enc.labels.len() as f64,
                "minimum_gold_margin": minimum_gold_margin,
                "negative_gold_margin_tokens": negative_gold_margins,
                "person_native_target_tokens": person_tokens,
            }));
            batch_nll += nll;
            batch_mass += mass;
            batch_max = batch_max.max(max_abs);
            batch_centered_squares += centered_squares;
            batch_tokens += doc.enc.labels.len();
            batch_class_range = batch_class_range.max(max_class_range);
        }
        batches.push(json!({
            "index": batch_index, "documents": size, "padded_content_length": padded_length,
            "weighted_token_mass": batch_mass, "native_masked_weighted_loss": native_loss,
            "weighted_loss_f64_reference": batch_nll / batch_mass,
            "native_minus_reference_loss": native_loss - batch_nll / batch_mass,
            "max_abs_logit": batch_max, "max_abs_padding_logit": padded_max,
            "centered_rms_logit": (batch_centered_squares / (batch_tokens * labels) as f64).sqrt(),
            "max_class_range": batch_class_range,
            "document_names": &names[batch_index * batch_size..batch_index * batch_size + size],
            "layer_activations": layer_activations,
        }));
        total_nll += batch_nll;
        total_mass += batch_mass;
    }
    Ok(json!({
        "batch_size": batch_size,
        "aggregate_weighted_loss_f64_reference": total_nll / total_mass,
        "weighted_token_mass": total_mass,
        "documents": documents, "batches": batches,
    }))
}

/// Inspect evaluation and seeded dropout-active forwards without updating a checkpoint.
pub(crate) fn run(run: &Path, gold: &Path, out: &Path, samples: usize) -> anyhow::Result<()> {
    ensure!(
        (1..=16).contains(&samples),
        "loss inspection needs 1–16 dropout samples"
    );
    ensure!(
        matches!(std::fs::symlink_metadata(out), Err(error) if error.kind() == std::io::ErrorKind::NotFound),
        "loss inspection output must be a new directory"
    );
    let marker_path = run.join("diagnostic.json");
    let marker: Value = serde_json::from_slice(
        &std::fs::read(&marker_path).context("loss inspection requires diagnostic.json")?,
    )?;
    validate_marker(&marker)?;
    let selection_path = run.join("selection.json");
    let selection: Value = serde_json::from_slice(&std::fs::read(&selection_path)?)?;
    let config_path = run.join("config.toml");
    let checkpoint = run.join("best.mpk");
    let cfg = crate::config::load(&config_path)?;
    ensure!(
        cfg.task == crate::config::Task::Detector && cfg.train.batch_size == 32,
        "loss inspection requires the native batch32 detector configuration"
    );
    let operator = crate::diagnostic_operator::load(run, &cfg)?;
    let operator_path = run.join(match operator {
        crate::diagnostic_operator::ForwardOperator::ContextRmsV2 => {
            crate::diagnostic_operator::CONTEXT_FILE
        }
        _ => crate::diagnostic_operator::FILE,
    });
    let executable = std::env::current_exe()?;
    let fullmix_path = run.join(crate::fullmix_rms::FILE);
    let checkpoint_proof = run.join("checkpoint.json");
    let checkpoint_events = run.join(crate::fullmix_rms::CHECKPOINT_LEDGER);
    let mut paths = vec![
        &config_path,
        &checkpoint,
        &marker_path,
        &selection_path,
        gold,
        &executable,
    ];
    if matches!(
        operator,
        crate::diagnostic_operator::ForwardOperator::ResidualRmsV1
            | crate::diagnostic_operator::ForwardOperator::ContextRmsV2
    ) {
        paths.push(&operator_path);
    }
    if crate::fullmix_rms::is_fullmix(&marker) {
        paths.push(&fullmix_path);
        paths.push(&checkpoint_proof);
        paths.push(&checkpoint_events);
        crate::fullmix_rms::checkpoint_provenance(run)?;
    }
    let hashes = paths
        .iter()
        .map(|path| hash(path))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let (names, docs) = if crate::fullmix_rms::is_fullmix(&marker) {
        crate::fullmix_rms::load_inspection(run, &cfg, gold, &selection)?
    } else if marker.get("frozen_cohort_scope").is_some() {
        crate::memorization::verify_frozen_selection(&cfg, &marker, &selection)?;
        let loaded = load_cases_with_limits(gold, &cfg, &selection, 15, 900)?;
        for (name, selected) in loaded.0.iter().zip(
            selection["documents"]
                .as_array()
                .context("missing frozen documents")?,
        ) {
            ensure!(
                selected["name"] == *name,
                "frozen inspection document name changed"
            );
        }
        loaded
    } else {
        load_cases(gold, &cfg, &selection)?
    };
    let weights = &cfg
        .detector
        .as_ref()
        .context("missing detector loss weights")?
        .class_weights;
    let device = Default::default();
    let model =
        crate::quantize::load_best_for_operator::<Autodiff<NdArray>>(run, &cfg, &device, operator)?;
    let tensors = crate::quantize::extract(&model.valid(), cfg.net_name());
    let parameter_sha = crate::training_diagnostic::parameter_sha256(&tensors);
    if crate::fullmix_rms::is_fullmix(&marker) {
        crate::fullmix_rms::verify_checkpoint_parameters(run, &parameter_sha)?;
    }
    let mut parameters = Vec::new();
    for tensor in &tensors {
        ensure!(
            tensor.data.iter().all(|value| value.is_finite()),
            "checkpoint parameter {} is non-finite",
            tensor.name
        );
        let max_abs = tensor
            .data
            .iter()
            .fold(0.0f32, |max, value| max.max(value.abs()));
        parameters.push(json!({"name": tensor.name, "shape": tensor.shape, "max_abs": max_abs}));
    }
    let evaluation = [1, 32]
        .into_iter()
        .map(|size| {
            measure(
                &model.valid(),
                &docs,
                &names,
                weights,
                size,
                &device,
                operator,
            )
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut dropout = Vec::new();
    for sample in 0..samples {
        let seed = cfg.seed.wrapping_add(sample as u64);
        for size in [1, 32] {
            <Autodiff<NdArray> as Backend>::seed(&device, seed);
            dropout.push(json!({
                "sample": sample, "seed": seed,
                "measurement": measure(&model, &docs, &names, weights, size, &device, operator)?,
            }));
        }
    }
    let final_sha = crate::training_diagnostic::parameter_sha256(&crate::quantize::extract(
        &model.valid(),
        cfg.net_name(),
    ));
    ensure!(
        parameter_sha == final_sha,
        "forward-only inspection changed model parameters"
    );
    for (path, expected) in paths.iter().zip(&hashes) {
        ensure!(
            hash(path)? == *expected,
            "loss inspection input changed: {}",
            path.display()
        );
    }
    let mut result = json!({
        "scope": "same frozen weights and original training cases; forward-only evaluation versus dropout-active loss; no optimization or accuracy claim",
        "backend": "NdArray f32 / Autodiff<NdArray> f32",
        "run": run, "gold": gold,
        "provenance": paths.iter().zip(hashes).map(|(path, sha)| json!({"path": path, "sha256": sha})).collect::<Vec<_>>(),
        "parameter_sha256_before": parameter_sha, "parameter_sha256_after": final_sha,
        "parameters_finite": true, "parameters": parameters,
        "class_weights": weights, "dropout_probability": cfg.net.dropout,
        "dropout_samples": samples,
        "normalization": "sum(class_weight * negative log probability * real-token mask) / sum(class_weight * real-token mask); f64 reference from actual f32 logits",
        "layer_activation_normalization": "unweighted mean, root mean square, and maximum absolute actual f32 activations over real-token positions and all channels; document order matches each batch's document_names; an empty real-token mask has zero moments",
        "gold_margin_definition": "actual target-label logit minus maximum other-label logit, excluding padding; negative margins indicate a higher-scored alternative label",
        "person_native_target_token_definition": "native BIO labels1/2 only, with original byte bounds; unweighted stable-f64 NLL, gold-label probability and margin from the same actual f32 forward; no extra forward or RNG",
        "evaluation_dropout_disabled": evaluation, "dropout_active": dropout,
    });
    if matches!(
        operator,
        crate::diagnostic_operator::ForwardOperator::ResidualRmsV1
            | crate::diagnostic_operator::ForwardOperator::ContextRmsV2
    ) {
        result["forward_operator"] = json!(match operator {
            crate::diagnostic_operator::ForwardOperator::ContextRmsV2 =>
                crate::diagnostic_operator::CONTEXT_NAME,
            _ => crate::diagnostic_operator::NAME,
        });
    }
    if crate::fullmix_rms::is_fullmix(&marker) {
        result["diagnostic_scope"] = json!(crate::fullmix_rms::diagnostic_scope(&marker)?);
        result["preflight"] = marker["preflight"].clone();
        result["checkpoint"] = crate::fullmix_rms::checkpoint_provenance(run)?;
        result["release_quality_claim"] = json!(false);
    }
    std::fs::create_dir(out)?;
    std::fs::write(
        out.join("loss-inspection.json"),
        serde_json::to_vec_pretty(&result)?,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        crate::config::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/detector-shared-v6.toml"),
        )
        .unwrap()
    }

    #[test]
    fn identical_zero_dropout_weights_have_equal_training_and_evaluation_loss() {
        let mut cfg = config();
        cfg.features.hash_buckets = 16;
        cfg.net.hidden = 4;
        cfg.net.dropout = 0.0;
        let features = cfg.features.to_tessera();
        let docs: Vec<_> = ["Ada Lane", "Ada Lane works here today"]
            .into_iter()
            .map(|text| {
                let gold = vec![detector::KindSpan {
                    kind: 0,
                    start: 0,
                    end: 8,
                }];
                DetectorDoc {
                    text: text.into(),
                    enc: detector::encode_document(text, &gold, &features).unwrap(),
                    gold,
                    breaks: detector::breaks_of(text),
                }
            })
            .collect();
        let names = vec!["short".into(), "long".into()];
        let device = Default::default();
        let model = cfg.detector_net_config().init::<Autodiff<NdArray>>(&device);
        for size in [1, 32] {
            let eval = measure(
                &model.valid(),
                &docs,
                &names,
                &[1.; 7],
                size,
                &device,
                crate::diagnostic_operator::ForwardOperator::Standard,
            )
            .unwrap();
            let train = measure(
                &model,
                &docs,
                &names,
                &[1.; 7],
                size,
                &device,
                crate::diagnostic_operator::ForwardOperator::Standard,
            )
            .unwrap();
            assert_eq!(eval, train);
            for batch in eval["batches"].as_array().unwrap() {
                assert!(batch["native_minus_reference_loss"].as_f64().unwrap().abs() < 1e-5);
                let layers = batch["layer_activations"].as_array().unwrap();
                assert_eq!(layers.len(), 3 + 5 * cfg.net.dilations.len());
                let batch_names = batch["document_names"].as_array().unwrap();
                let tokens: usize = batch_names
                    .iter()
                    .map(|name| {
                        let index = names
                            .iter()
                            .position(|value| value == name.as_str().unwrap())
                            .unwrap();
                        docs[index].enc.labels.len()
                    })
                    .sum();
                for layer in layers {
                    assert_eq!(
                        layer["aggregate"]["real_tokens"].as_u64().unwrap(),
                        tokens as u64
                    );
                    assert_eq!(
                        layer["documents"].as_array().unwrap().len(),
                        batch_names.len()
                    );
                }
                let head = layers.last().unwrap();
                assert_eq!(head["layer"], "head");
                for (local, name) in batch_names.iter().enumerate() {
                    let doc = eval["documents"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|doc| doc["name"] == *name)
                        .unwrap();
                    assert_eq!(
                        head["documents"][local]["real_tokens"],
                        doc["content_tokens"]
                    );
                    assert_eq!(head["documents"][local]["rms"], doc["rms_logit"]);
                    assert!(doc["mean_gold_margin"].as_f64().unwrap().is_finite());
                    assert!(doc["minimum_gold_margin"].as_f64().unwrap().is_finite());
                    let person = doc["person_native_target_tokens"].as_array().unwrap();
                    assert_eq!(person.len(), 2);
                    assert_eq!(person[0]["native_label"], 1);
                    assert_eq!(person[1]["native_label"], 2);
                    for token in person {
                        let nll = token["unweighted_nll_f64"].as_f64().unwrap();
                        let probability = token["gold_probability_f64"].as_f64().unwrap();
                        assert!(nll.is_finite() && nll >= 0.0);
                        assert!((0.0..=1.0).contains(&probability));
                        assert!((probability - (-nll).exp()).abs() < 1e-12);
                    }
                }
            }
        }
        let single = measure(
            &model.valid(),
            &docs,
            &names,
            &[1.; 7],
            1,
            &device,
            crate::diagnostic_operator::ForwardOperator::Standard,
        )
        .unwrap();
        let padded = measure(
            &model.valid(),
            &docs,
            &names,
            &[1.; 7],
            32,
            &device,
            crate::diagnostic_operator::ForwardOperator::Standard,
        )
        .unwrap();
        for index in 0..docs.len() {
            let loss = |v: &Value| {
                v["documents"][index]["weighted_loss_f64_reference"]
                    .as_f64()
                    .unwrap()
            };
            assert!((loss(&single) - loss(&padded)).abs() < 1e-5);
        }
    }

    #[test]
    fn context96_loss_marker_requires_the_exact_bounded_scope() {
        let marker = json!({"scope":crate::fullmix_rms::CONTEXT_MARKER_SCOPE,
            "diagnostic_scope":crate::fullmix_rms::CONTEXT_SCOPE,
            "max_steps":4000,"release_quality_claim":false});
        validate_marker(&marker).unwrap();
        for (key, value) in [
            ("diagnostic_scope", json!(crate::fullmix_rms::SCOPE)),
            ("scope", json!(crate::fullmix_rms::MARKER_SCOPE)),
            ("max_steps", json!(6000)),
            ("release_quality_claim", json!(true)),
        ] {
            let mut wrong = marker.clone();
            wrong[key] = value;
            assert!(validate_marker(&wrong).is_err());
        }
    }

    #[test]
    fn refuses_non_diagnostic_and_non_training_inputs() {
        assert!(validate_marker(&json!({"scope": "normal training", "max_steps": 1000, "release_quality_claim": false})).is_err());
        assert!(validate_marker(&json!({"scope": "fixed training-only memorization diagnostic; never export", "max_steps": 1000, "release_quality_claim": true})).is_err());
        let directory = tempfile::tempdir().unwrap();
        let gold = directory.path().join("held.jsonl");
        let rows = (0..32).map(|index| json!({"name": format!("held-{index}"), "doc_type": "notice", "country": "US", "input": "Ada Lane", "expected": []}).to_string()).collect::<Vec<_>>().join("\n");
        std::fs::write(&gold, rows).unwrap();
        assert!(
            load_cases(&gold, &config(), &json!({}))
                .unwrap_err()
                .to_string()
                .contains("training_memorization")
        );
    }

    #[test]
    fn native_targets_match_original_selection_and_changed_gold_is_rejected() {
        let cfg = config();
        let directory = tempfile::tempdir().unwrap();
        let gold = directory.path().join("cases.jsonl");
        let mut rows = Vec::new();
        let mut selected = Vec::new();
        for index in 0..32 {
            let text = format!("Ada Lane item {index}");
            let spans = vec![detector::KindSpan {
                kind: 0,
                start: 0,
                end: 8,
            }];
            let encoded =
                detector::encode_document(&text, &spans, &cfg.features.to_tessera()).unwrap();
            rows.push(json!({
                "name": format!("training-{index}"), "country": "US", "doc_type": "training_memorization",
                "input": text, "expected": [{"kind": "person", "start": 0, "end": 8, "text": "Ada Lane"}],
            }));
            selected.push(json!({
                "text_sha256": crate::export::sha256_hex(text.as_bytes()),
                "original_source_row": true, "derived_piece": false,
                "content_tokens": encoded.token_spans.len(),
                "gold": [{"kind": "person", "start": 0, "end": 8}],
            }));
        }
        std::fs::write(
            &gold,
            rows.iter()
                .map(Value::to_string)
                .collect::<Vec<_>>()
                .join("\n"),
        )
        .unwrap();
        let mut selection = json!({"selected_documents": 32, "documents": selected});
        let (names, docs) = load_cases(&gold, &cfg, &selection).unwrap();
        assert_eq!(names.len(), 32);
        for doc in docs {
            assert_eq!(&doc.enc.labels[..2], &[1, 2]);
            let mut encoded =
                detector::encode_document(&doc.text, &doc.gold, &cfg.features.to_tessera())
                    .unwrap();
            encoded.country = "US".into();
            assert_eq!(doc.enc, encoded);
        }
        selection["documents"][0]["gold"][0]["kind"] = json!("org");
        assert!(
            load_cases(&gold, &cfg, &selection)
                .unwrap_err()
                .to_string()
                .contains("native gold differs")
        );
        selection["documents"][0]["gold"][0]["kind"] = json!("person");
        selection["documents"][0]["text_sha256"] = json!("incorrect");
        assert!(load_cases(&gold, &cfg, &selection).is_err());
    }

    #[test]
    fn default32_count_and128_limit_are_not_widened_by_frozen_policy() {
        let cfg = config();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cases.jsonl");
        let mut rows = Vec::new();
        let mut selected = Vec::new();
        for index in 0..32 {
            let text = format!("item{index} {}", "word ".repeat(150));
            let enc = detector::encode_document(&text, &[], &cfg.features.to_tessera()).unwrap();
            rows.push(json!({"name": format!("training-{index}"), "country": "US",
                "doc_type": "training_memorization", "input": text, "expected": []}));
            selected.push(
                json!({"text_sha256": crate::export::sha256_hex(text.as_bytes()),
                "original_source_row": true, "derived_piece": false,
                "content_tokens": enc.token_spans.len(), "gold": []}),
            );
        }
        let write = |count: usize| {
            std::fs::write(
                &path,
                rows[..count]
                    .iter()
                    .map(Value::to_string)
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
            .unwrap()
        };
        write(32);
        let selection = json!({"selected_documents": 32, "documents": selected});
        assert!(load_cases(&path, &cfg, &selection).is_err());
        write(15);
        let selection = json!({"selected_documents": 15,
            "documents": selection["documents"].as_array().unwrap()[..15]});
        assert!(load_cases(&path, &cfg, &selection).is_err());
        assert!(load_cases_with_limits(&path, &cfg, &selection, 15, 900).is_ok());
    }
}
