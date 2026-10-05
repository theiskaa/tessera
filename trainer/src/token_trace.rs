//! Forward-only token decisions from a checksum-verified frozen detector checkpoint.

use std::collections::BTreeSet;
use std::fs::OpenOptions;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::{Context, ensure};
use burn::backend::NdArray;
use burn::data::dataloader::batcher::Batcher;
use burn::tensor::activation::softmax;
use serde_json::{Value, json};
use tessera::internal::{
    DETECTOR_LABELS, argmax, detector_label_strings, detector_sequence_labels, flag,
};

use crate::dataset::{ParserBatch, ParserBatcher};
use crate::detector::{self, DetectorDoc};

fn hash(path: &Path) -> anyhow::Result<String> {
    Ok(crate::export::sha256_hex(&std::fs::read(path)?))
}

fn document_record(
    name: &str,
    doc: &DetectorDoc,
    labels: Option<&[u8]>,
    logits: &[f32],
    probs: &[f32],
    decoder: Option<crate::diagnostic_decode::Decoder>,
) -> anyhow::Result<Value> {
    let n = doc.enc.token_spans.len();
    ensure!(
        logits.len() == n * DETECTOR_LABELS
            && probs.len() == logits.len()
            && doc.enc.flags.len() == n
            && doc.breaks.len() == n,
        "token trace dimensions differ"
    );
    ensure!(
        logits.iter().all(|v| v.is_finite())
            && probs
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0),
        "token trace contains nonfinite or invalid values"
    );
    ensure!(
        labels.is_none_or(|labels| labels.len() == n
            && labels
                .iter()
                .all(|label| usize::from(*label) < DETECTOR_LABELS)),
        "gold BIO labels differ"
    );
    let names = detector_label_strings();
    let masked: Vec<_> = doc
        .enc
        .flags
        .iter()
        .map(|f| f & flag::IN_RULE_SPAN != 0)
        .collect();
    let sequence = detector_sequence_labels(probs, &masked, &doc.breaks);
    let raw = detector::decode_scored(doc, probs, decoder);
    let filtered = detector::apply_confidence_policy(std::slice::from_ref(&raw));
    let as_json = |span: &detector::KindSpan| json!({ "kind": detector::KINDS[span.kind].as_str(), "start": span.start, "end": span.end });
    let tokens: Vec<_> = doc.enc.token_spans.iter().enumerate().map(|(i, &(start, end))| {
        let row = &probs[i * DETECTOR_LABELS..(i + 1) * DETECTOR_LABELS];
        let gold = labels.map(|labels| usize::from(labels[i]));
        let chosen = argmax(row);
        json!({
            "start": start, "end": end, "text": doc.text.get(start as usize..end as usize),
            "gold_label": gold.map(|index| names[index].as_str()),
            "argmax_label": names[chosen], "sequence_label": names[sequence[i]],
            "masked_by_rule": masked[i], "paragraph_break": doc.breaks[i],
            "logits": &logits[i * DETECTOR_LABELS..(i + 1) * DETECTOR_LABELS], "probabilities": row,
        })
    }).collect();
    Ok(json!({
        "record": "document", "name": name, "text_sha256": crate::export::sha256_hex(doc.text.as_bytes()),
        "gold": doc.gold.iter().map(as_json).collect::<Vec<_>>(), "tokens": tokens,
        "raw_candidates": raw.iter().map(|candidate| { let mut span = as_json(&candidate.span); span["confidence"] = json!(candidate.confidence); span }).collect::<Vec<_>>(),
        "filtered": filtered[0].iter().map(as_json).collect::<Vec<_>>(),
    }))
}

fn write_record(writer: &mut impl Write, record: &Value) -> anyhow::Result<()> {
    serde_json::to_writer(&mut *writer, record)?;
    writer.write_all(b"\n")?;
    Ok(())
}

/// Inspect at most 200 whole documents, preserving original weights and data.
pub(crate) fn run(run: &Path, gold: &Path, out: &Path) -> anyhow::Result<()> {
    eprintln!("token inspection: validating cases and checkpoint; no optimizer");
    let cfg_path = run.join("config.toml");
    let cfg = crate::config::load(&cfg_path)?;
    ensure!(
        cfg.net_name() == "detector",
        "token inspection requires a detector"
    );
    ensure!(!out.try_exists()?, "token inspection output already exists");
    let input = std::fs::read(gold)?;
    let rows: Vec<Value> = std::str::from_utf8(&input)?
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(serde_json::from_str)
        .collect::<Result<_, _>>()?;
    ensure!(
        !rows.is_empty() && rows.len() <= 200,
        "token inspection requires 1–200 documents"
    );
    let mut names = BTreeSet::new();
    for row in &rows {
        let name = row["name"].as_str().context("missing case name")?;
        ensure!(
            row["country"] == "US" && !name.is_empty() && names.insert(name),
            "inspection cases must be distinct US documents"
        );
    }
    let fc = cfg.features.to_tessera();
    let docs = detector::load_development_gold(gold, &fc)?;
    ensure!(
        docs.len() == rows.len()
            && rows.iter().zip(&docs).all(|(row, doc)| {
                row["input"].as_str() == Some(doc.text.as_str()) && doc.enc.token_spans.len() <= 900
            }),
        "inspection requires aligned whole documents with at most 900 retained tokens"
    );
    let config_sha256 = hash(&cfg_path)?;
    let best_path = run.join("best.mpk");
    let best_sha256 = hash(&best_path)?;
    let operator = crate::diagnostic_operator::deployable(&cfg);
    let device = Default::default();
    let model = crate::quantize::load_best_for_operator::<NdArray>(run, &cfg, &device, operator)?;
    let parameter_sha256 = crate::training_diagnostic::parameter_sha256(&crate::quantize::extract(
        &model,
        cfg.net_name(),
    ));
    let checkpoint = if run.join("diagnostic.json").try_exists()? {
        let marker: Value = serde_json::from_slice(&std::fs::read(run.join("diagnostic.json"))?)?;
        if crate::fullmix_rms::is_fullmix(&marker) {
            Some(crate::fullmix_rms::verify_checkpoint_parameters(
                run,
                &parameter_sha256,
            )?)
        } else {
            None
        }
    } else {
        None
    };
    if let Some(parent) = out.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut writer = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(out)?);
    write_record(
        &mut writer,
        &json!({
            "record": "provenance", "scope": "frozen-checkpoint token diagnosis; no optimizer updates or release claim",
            "gold_sha256": crate::export::sha256_hex(&input), "config_sha256": config_sha256,
            "best_sha256": best_sha256, "parameter_sha256": parameter_sha256, "checkpoint": checkpoint,
            "cases": rows.len(), "offset_unit": "UTF-8 bytes", "label_order": detector_label_strings(),
            "evaluator_binary_sha256": hash(&std::env::current_exe()?)?,
        }),
    )?;
    for (index, (row, doc)) in rows.iter().zip(&docs).enumerate() {
        let batch: ParserBatch<NdArray> = ParserBatcher.batch(vec![doc.enc.clone()], &device);
        let output = operator.forward(
            &model,
            batch.ngram_ids,
            batch.script,
            batch.shape,
            batch.flags,
            batch.mask,
        )?;
        let [b, l, c] = output.dims();
        ensure!(
            b == 1 && c == DETECTOR_LABELS && l >= doc.enc.token_spans.len(),
            "native output dimensions differ"
        );
        let logits = output
            .clone()
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| anyhow::anyhow!("reading logits: {error:?}"))?;
        let probs = softmax(output, 2)
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| anyhow::anyhow!("reading probabilities: {error:?}"))?;
        let gold_encoding = detector::encode_document(&doc.text, &doc.gold, &fc);
        let labels = gold_encoding.as_ref().ok().map(|enc| enc.labels.as_slice());
        let length = doc.enc.token_spans.len() * DETECTOR_LABELS;
        let mut record = document_record(
            row["name"].as_str().context("missing case name")?,
            doc,
            labels,
            &logits[..length],
            &probs[..length],
            operator.decoder(None),
        )?;
        record["gold_encoding_error"] = json!(gold_encoding.err().map(|error| error.to_string()));
        write_record(&mut writer, &record)?;
        if (index + 1).is_multiple_of(20) {
            eprintln!("token inspection: {}/{} documents", index + 1, docs.len());
        }
    }
    ensure!(
        hash(&best_path)? == best_sha256
            && hash(&cfg_path)? == config_sha256
            && std::fs::read(gold)? == input,
        "inspection inputs changed during the forward pass"
    );
    write_record(
        &mut writer,
        &json!({ "record": "complete", "documents": docs.len(), "optimizer_updates_executed": 0, "release_quality_claim": false }),
    )?;
    writer.flush()?;
    eprintln!(
        "token inspection complete: {} documents; zero optimizer updates",
        docs.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tessera::internal::FeatureConfig;

    #[test]
    fn trace_distinguishes_local_argmax_from_legal_sequence() {
        let text = "Acme";
        let gold = vec![detector::KindSpan {
            kind: 1,
            start: 0,
            end: 4,
        }];
        let enc = detector::encode_document(text, &gold, &FeatureConfig::default()).unwrap();
        let doc = DetectorDoc {
            text: text.into(),
            enc,
            gold,
            breaks: vec![false],
        };
        let probs = [0.01, 0.01, 0.01, 0.25, 0.70, 0.01, 0.01];
        let record =
            document_record("acme", &doc, Some(&doc.enc.labels), &[0.0; 7], &probs, None).unwrap();
        assert_eq!(record["tokens"][0]["gold_label"], "B-ORG");
        assert_eq!(record["tokens"][0]["argmax_label"], "I-ORG");
        assert_eq!(record["tokens"][0]["sequence_label"], "B-ORG");
        assert_eq!(record["raw_candidates"].as_array().unwrap().len(), 1);
        assert!(record["filtered"].as_array().unwrap().is_empty());
    }

    #[test]
    fn damaged_rows_cannot_be_valid_traces() {
        let enc = detector::encode_document("Acme", &[], &FeatureConfig::default()).unwrap();
        let doc = DetectorDoc {
            text: "Acme".into(),
            enc,
            gold: vec![],
            breaks: vec![false],
        };
        assert!(document_record("acme", &doc, None, &[0.0; 6], &[0.0; 7], None).is_err());
        assert!(document_record("acme", &doc, None, &[f32::NAN; 7], &[0.0; 7], None).is_err());
    }
}
