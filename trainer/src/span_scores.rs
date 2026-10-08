//! Read-only raw PERSON/ORG/ADDRESS confidences from a reviewed-fit checkpoint, so confidence
//! cutoffs can be calibrated on a separate labeled set. Never trains or changes the checkpoint.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, ensure};
use burn::backend::NdArray;
use burn::module::Module;
use burn::record::{FullPrecisionSettings, NamedMpkFileRecorder};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::config::{Config, Task};
use crate::detector::{self, DetectorDoc, DetectorInputPolicy, KINDS, ScoredSpan};
use crate::exact_metrics::Counts;
use crate::net::{TaggerNet, TaggerNetConfig};
use crate::reviewed_data::digest;
use crate::reviewed_fit::Postprocess;

/// The reviewed fit scores its cohorts in chunks of this size; padding follows the same batches.
const BATCH_SIZE: usize = 32;
const INPUT_POLICY: DetectorInputPolicy = DetectorInputPolicy::KnownUs;

#[derive(Deserialize)]
struct Row {
    name: String,
    input: String,
    expected: Vec<Gold>,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Gold {
    kind: String,
    text: String,
    start: u32,
    end: u32,
}

#[derive(Serialize)]
struct Scored<'a> {
    kind: &'static str,
    start: u32,
    end: u32,
    text: &'a str,
    confidence: f32,
}

#[derive(Serialize)]
struct Record<'a> {
    name: &'a str,
    spans: Vec<Scored<'a>>,
    gold: &'a [Gold],
}

/// Score `documents` with `checkpoint` under `config`, writing one span record per document to
/// `out` and the provenance and exact counts to `<out>.summary.json`.
pub(crate) fn run(
    config: &Path,
    checkpoint: &Path,
    documents: &Path,
    out: &Path,
) -> anyhow::Result<()> {
    let summary_path = summary_path(out)?;
    for path in [out, summary_path.as_path()] {
        ensure!(!path.try_exists()?, "{} already exists", path.display());
    }
    let config_bytes = read(config)?;
    let cfg = parse_config(&config_bytes, config)?;
    let postprocess = cfg
        .net
        .detector_postprocess
        .unwrap_or(Postprocess::AddressContinuationV1);
    let device = Default::default();
    let checkpoint_sha256 = digest(&read(checkpoint)?);
    let (model, parameter_sha256) =
        load_checkpoint(&cfg.detector_net_config(), checkpoint, &device)?;
    ensure!(
        model.flag_bits() == cfg.detector_feature_contract().flag_bits(),
        "checkpoint flag width differs from the configured detector features"
    );
    let document_bytes = read(documents)?;
    let rows = parse_rows(&document_bytes, documents)?;
    let docs = encode(&rows, &cfg)?;
    let candidates =
        detector::predict_scored_with_postprocess(&model, &docs, BATCH_SIZE, &device, postprocess)?;
    ensure!(
        digest(&read(checkpoint)?) == checkpoint_sha256,
        "checkpoint changed during scoring"
    );
    let mut summary = summary(&rows, &candidates);
    summary["pipeline"] = json!({"forward_operator":"context_rms_v2",
        "base_decoder":"address_continuation_v1","postprocess":postprocess,
        "input_policy":INPUT_POLICY,"detector_features":cfg.detector_feature_contract().name(),
        "batch_size":BATCH_SIZE});
    summary["config"] = json!({"path":config,"sha256":digest(&config_bytes)});
    summary["checkpoint"] = json!({"path":checkpoint,"sha256":checkpoint_sha256,
        "parameter_sha256":parameter_sha256});
    summary["documents_file"] = json!({"path":documents,"sha256":digest(&document_bytes)});
    summary["records"] = json!(out);
    let mut lines = Vec::new();
    for record in records(&rows, &candidates)? {
        serde_json::to_writer(&mut lines, &record)?;
        lines.push(b'\n');
    }
    let mut pretty = serde_json::to_vec_pretty(&summary)?;
    pretty.push(b'\n');
    write_new(out, &lines)?;
    write_new(&summary_path, &pretty)?;
    println!("{}", String::from_utf8(pretty)?.trim_end());
    Ok(())
}

fn summary_path(out: &Path) -> anyhow::Result<PathBuf> {
    let name = out
        .file_name()
        .context("--out must name a file")?
        .to_string_lossy();
    Ok(out.with_file_name(format!("{name}.summary.json")))
}

fn read(path: &Path) -> anyhow::Result<Vec<u8>> {
    std::fs::read(path).with_context(|| format!("reading {}", path.display()))
}

/// Parse a reviewed-route config, which legacy `config::load` refuses for TabCells25.
fn parse_config(bytes: &[u8], path: &Path) -> anyhow::Result<Config> {
    let cfg: Config = toml::from_str(std::str::from_utf8(bytes)?)
        .with_context(|| format!("parsing {}", path.display()))?;
    cfg.validate()
        .with_context(|| format!("validating {}", path.display()))?;
    ensure!(
        cfg.task == Task::Detector && cfg.context96_rms(),
        "span scoring requires a context96 RMS detector config"
    );
    Ok(cfg)
}

/// Load native f32 weights, refusing any tensor whose name or shape the config does not declare,
/// and return the finite parameter digest.
pub(crate) fn load_checkpoint(
    net: &TaggerNetConfig,
    path: &Path,
    device: &burn::backend::ndarray::NdArrayDevice,
) -> anyhow::Result<(TaggerNet<NdArray>, String)> {
    ensure!(
        path.extension().is_some_and(|e| e == "mpk") && path.is_file(),
        "checkpoint must be an existing native .mpk file"
    );
    let fresh = net.init::<NdArray>(device);
    let declared = layout(&crate::quantize::extract(&fresh, "detector"));
    // The recorder sets the `.mpk` extension itself, which leaves this exact path unchanged.
    let model = fresh
        .load_file(
            path,
            &NamedMpkFileRecorder::<FullPrecisionSettings>::new(),
            device,
        )
        .with_context(|| format!("loading {}", path.display()))?;
    let tensors = crate::quantize::extract(&model, "detector");
    ensure!(
        layout(&tensors) == declared,
        "checkpoint tensors differ from the configured architecture"
    );
    let parameters = crate::native_checkpoint::parameter_sha256(&tensors)?;
    Ok((model, parameters))
}

fn layout(tensors: &[crate::quantize::F32Tensor]) -> Vec<(String, Vec<usize>)> {
    tensors
        .iter()
        .map(|t| (t.name.clone(), t.shape.clone()))
        .collect()
}

fn parse_rows(bytes: &[u8], path: &Path) -> anyhow::Result<Vec<Row>> {
    let text = std::str::from_utf8(bytes)?;
    let mut names = std::collections::BTreeSet::new();
    let mut rows = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let at = || format!("{}:{}", path.display(), index + 1);
        let row: Row = serde_json::from_str(line).with_context(at)?;
        ensure!(
            names.insert(row.name.clone()),
            "{} repeats {}",
            at(),
            row.name
        );
        for gold in &row.expected {
            ensure!(
                gold.start < gold.end
                    && row.input.get(gold.start as usize..gold.end as usize)
                        == Some(gold.text.as_str())
                    && ["person", "org", "address", "email", "phone"].contains(&gold.kind.as_str()),
                "{} has a gold span that differs from its input",
                at()
            );
        }
        rows.push(row);
    }
    ensure!(!rows.is_empty(), "{} has no documents", path.display());
    Ok(rows)
}

/// Gold-independent features, exactly as the reviewed route encodes its DEV cohort.
fn encode(rows: &[Row], cfg: &Config) -> anyhow::Result<Vec<DetectorDoc>> {
    let fc = cfg.features.to_tessera();
    rows.iter()
        .map(|row| {
            let enc = detector::encode_document_with_feature_contract(
                &row.input,
                &[],
                &fc,
                INPUT_POLICY,
                cfg.detector_feature_contract(),
            )
            .with_context(|| format!("{} cannot be encoded", row.name))?;
            let breaks = detector::breaks_of(&row.input);
            ensure!(
                !enc.token_spans.is_empty() && breaks.len() == enc.token_spans.len(),
                "{} has no retained tokens",
                row.name
            );
            Ok(DetectorDoc {
                text: row.input.clone(),
                enc,
                gold: Vec::new(),
                breaks,
            })
        })
        .collect()
}

fn records<'a>(rows: &'a [Row], candidates: &[Vec<ScoredSpan>]) -> anyhow::Result<Vec<Record<'a>>> {
    ensure!(rows.len() == candidates.len(), "scored documents differ");
    rows.iter()
        .zip(candidates)
        .map(|(row, spans)| {
            let spans = spans
                .iter()
                .map(|s| {
                    Ok(Scored {
                        kind: KINDS[s.span.kind].as_str(),
                        start: s.span.start,
                        end: s.span.end,
                        text: row
                            .input
                            .get(s.span.start as usize..s.span.end as usize)
                            .context("decoded span is not on a UTF-8 boundary")?,
                        confidence: s.confidence,
                    })
                })
                .collect::<anyhow::Result<_>>()?;
            Ok(Record {
                name: &row.name,
                spans,
                gold: &row.expected,
            })
        })
        .collect()
}

/// Exact per-kind counts for every raw span and for those the current cutoffs keep.
fn summary(rows: &[Row], candidates: &[Vec<ScoredSpan>]) -> serde_json::Value {
    type Entity = (&'static str, u32, u32);
    let entity = |s: detector::KindSpan| -> Entity { (KINDS[s.kind].as_str(), s.start, s.end) };
    let raw: Vec<Vec<_>> = candidates
        .iter()
        .map(|doc| doc.iter().map(|s| entity(s.span)).collect())
        .collect();
    let kept: Vec<Vec<_>> = detector::apply_confidence_policy(candidates)
        .into_iter()
        .map(|doc| doc.into_iter().map(entity).collect())
        .collect();
    let mut kinds = BTreeMap::new();
    for (index, kind) in KINDS.iter().enumerate() {
        let kind = kind.as_str();
        let count = |predicted: &[Vec<Entity>]| {
            let mut total = Counts::default();
            for (row, predicted) in rows.iter().zip(predicted) {
                let gold: Vec<Entity> = row
                    .expected
                    .iter()
                    .filter(|g| g.kind == kind)
                    .map(|g| (kind, g.start, g.end))
                    .collect();
                let predicted: Vec<Entity> =
                    predicted.iter().copied().filter(|p| p.0 == kind).collect();
                total.add(&Counts {
                    tp: crate::exact_metrics::matches(&gold, &predicted)
                        .into_iter()
                        .filter(|&yes| yes)
                        .count(),
                    predicted: predicted.len(),
                    gold: gold.len(),
                });
            }
            total
        };
        kinds.insert(
            kind,
            json!({"cutoff":detector::DETECT_MIN[index],"raw":count(&raw),"kept_at_cutoff":count(&kept)}),
        );
    }
    json!({"documents":rows.len(),"per_kind":kinds,
        "spans":candidates.iter().map(Vec::len).sum::<usize>()})
}

/// Commit new bytes without ever replacing an existing file.
fn write_new(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .with_context(|| format!("creating new {}", path.display()))?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reviewed_data::{Cohort, Cohorts};
    use tessera::internal::DetectorFeatureContract;

    fn tab_cells_config() -> Config {
        let mut cfg = crate::config::learning_test_config();
        cfg.net.detector_features = Some(crate::config::DetectorFeatures::TabCells25);
        cfg
    }

    fn row(text: &str, expected: Vec<Gold>) -> Row {
        Row {
            name: "case".into(),
            input: text.into(),
            expected,
        }
    }

    #[test]
    fn spans_kept_at_the_cutoffs_are_exactly_the_filtered_decoding() {
        let text = "Contact Mary Jones at Acme Bank or Lee Park of Initech today.";
        let rows = vec![row(
            text,
            vec![Gold {
                kind: "person".into(),
                text: "Mary Jones".into(),
                start: 8,
                end: 18,
            }],
        )];
        let doc = encode(&rows, &tab_cells_config()).unwrap().remove(0);
        let cutoff = |kind: &str| {
            detector::DETECT_MIN[KINDS.iter().position(|k| k.as_str() == kind).unwrap()]
        };
        let mut probs = Vec::new();
        for &(start, end) in &doc.enc.token_spans {
            let mut token = vec![0.0f32; detector::DETECTOR_LABELS];
            let (label, p) = match &text[start as usize..end as usize] {
                "Mary" => (1, 0.95),
                "Jones" => (2, 0.93),
                "Acme" => (3, 0.70),
                "Bank" => (4, 0.72),
                "Lee" => (1, 0.80),
                "Park" => (2, 0.84),
                "Initech" => (3, cutoff("org")),
                _ => (0, 1.0),
            };
            token[label] = p;
            token[0] += 1.0 - p;
            probs.extend(token);
        }
        let candidates = vec![
            detector::decode_scored_with_postprocess(
                &doc,
                &probs,
                Postprocess::AddressLabeledFieldsV1,
            )
            .unwrap(),
        ];
        let records = records(&rows, &candidates).unwrap();
        let written: serde_json::Value =
            serde_json::from_str(&serde_json::to_string(&records[0]).unwrap()).unwrap();
        let all: Vec<_> = written["spans"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| {
                (
                    s["kind"].as_str().unwrap().to_owned(),
                    s["start"].as_u64().unwrap() as u32,
                    s["end"].as_u64().unwrap() as u32,
                    s["confidence"].as_f64().unwrap() as f32,
                )
            })
            .collect();
        let kept: Vec<_> = all
            .iter()
            .filter(|(kind, .., confidence)| *confidence >= cutoff(kind))
            .map(|(kind, start, end, _)| (kind.clone(), *start, *end))
            .collect();
        let filtered: Vec<_> = detector::apply_confidence_policy(&candidates)[0]
            .iter()
            .map(|s| (KINDS[s.kind].as_str().to_owned(), s.start, s.end))
            .collect();
        let initech = text.find("Initech").unwrap() as u32;
        assert_eq!(kept, filtered);
        assert_eq!(
            kept,
            [
                ("person".to_owned(), 8, 18),
                ("org".to_owned(), initech, initech + 7)
            ]
        );
        assert!(
            all.iter()
                .any(|s| s.0 == "org" && s.1 == initech && s.3 == cutoff("org")),
            "a span exactly at its cutoff is kept"
        );
        assert_eq!(
            all.len(),
            4,
            "rejected ORG and PERSON stay in the raw output"
        );
        assert!(all.iter().any(|s| s.0 == "org" && s.3 < cutoff("org")));
        assert!(
            all.iter()
                .any(|s| s.0 == "person" && s.3 < cutoff("person"))
        );
        assert_eq!(written["spans"][0]["text"], "Mary Jones");
        assert_eq!(written["gold"][0]["text"], "Mary Jones");

        let summary = summary(&rows, &candidates);
        let person = &summary["per_kind"]["person"];
        assert_eq!(person["raw"], json!({"tp":1,"predicted":2,"gold":1}));
        assert_eq!(
            person["kept_at_cutoff"],
            json!({"tp":1,"predicted":1,"gold":1})
        );
        assert_eq!(
            summary["per_kind"]["org"]["kept_at_cutoff"],
            json!({"tp":0,"predicted":1,"gold":0})
        );
    }

    #[test]
    fn encoding_equals_the_reviewed_dev_reencoding() {
        let cfg = tab_cells_config();
        let text = "Clerk\tBrenda Tuttle\nPhone\t(435) 381-3550\n\nMailing: P.O. Box 907 Castle Dale, UT 84513";
        let fc = cfg.features.to_tessera();
        let legacy = DetectorDoc {
            text: text.into(),
            enc: detector::encode_document(text, &[], &fc).unwrap(),
            gold: Vec::new(),
            breaks: detector::breaks_of(text),
        };
        let cohorts = Cohorts {
            train: Cohort::unreviewed_for_test(Vec::new(), Vec::new()),
            dev: Cohort::unreviewed_for_test(vec!["case".into()], vec![legacy]),
        };
        let (reviewed, _) =
            crate::reviewed_data::reencode_with_input_policy(cohorts, &cfg, INPUT_POLICY).unwrap();
        let reviewed = &reviewed.dev.docs[0];
        let ours = encode(&[row(text, Vec::new())], &cfg).unwrap().remove(0);
        assert_eq!(
            cfg.detector_feature_contract(),
            DetectorFeatureContract::TabCells25
        );
        assert_eq!(ours.text, reviewed.text);
        assert_eq!(ours.enc.token_spans, reviewed.enc.token_spans);
        assert_eq!(ours.enc.ngram_ids, reviewed.enc.ngram_ids);
        assert_eq!(ours.enc.script, reviewed.enc.script);
        assert_eq!(ours.enc.shape, reviewed.enc.shape);
        assert_eq!(ours.enc.flags, reviewed.enc.flags);
        assert_eq!(ours.enc.labels, reviewed.enc.labels);
        assert_eq!(ours.breaks, reviewed.breaks);
        assert!(ours.breaks.iter().any(|&b| b));
    }

    #[test]
    fn outputs_never_replace_existing_files() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("spans.jsonl");
        std::fs::write(&out, b"kept").unwrap();
        assert!(write_new(&out, b"replacement").is_err());
        assert_eq!(std::fs::read(&out).unwrap(), b"kept");
        let fresh = dir.path().join("fresh.jsonl");
        write_new(&fresh, b"{}\n").unwrap();
        assert_eq!(std::fs::read(&fresh).unwrap(), b"{}\n");
        assert_eq!(
            summary_path(&out).unwrap(),
            dir.path().join("spans.jsonl.summary.json")
        );
        let missing = dir.path().join("absent.toml");
        let summary = dir.path().join("new.jsonl.summary.json");
        std::fs::write(&summary, b"earlier").unwrap();
        let error = run(&missing, &missing, &missing, &dir.path().join("new.jsonl")).unwrap_err();
        assert!(error.to_string().contains("already exists"), "{error}");
        assert!(!dir.path().join("new.jsonl").exists());
        assert_eq!(std::fs::read(&summary).unwrap(), b"earlier");
    }

    #[test]
    fn checkpoint_must_match_the_configured_architecture() {
        let net = |hidden| {
            TaggerNetConfig::new(16, vec![1, 2], detector::DETECTOR_LABELS)
                .with_ngram_dim(3)
                .with_script_dim(2)
                .with_shape_dim(2)
                .with_hidden(hidden)
        };
        let device = Default::default();
        let dir = tempfile::tempdir().unwrap();
        let values = |model: &TaggerNet<NdArray>| {
            crate::quantize::extract(model, "detector")
                .into_iter()
                .map(|t| t.data)
                .collect::<Vec<_>>()
        };
        let save = |name: &str| {
            let model = net(4).init::<NdArray>(&device);
            // Reading materializes the lazily initialized parameters before they are saved.
            let saved = values(&model);
            let path = dir.path().join(name);
            model
                .save_file(&path, &NamedMpkFileRecorder::<FullPrecisionSettings>::new())
                .unwrap();
            assert!(path.is_file());
            (path, saved)
        };
        let (plain, plain_values) = save("step-1.mpk");
        let (dotted, dotted_values) = save("step-1.final.mpk");
        assert_ne!(plain_values, dotted_values);
        let (loaded, parameters) = load_checkpoint(&net(4), &plain, &device).unwrap();
        assert_eq!(values(&loaded), plain_values);
        let (loaded, dotted_parameters) = load_checkpoint(&net(4), &dotted, &device).unwrap();
        assert_eq!(
            values(&loaded),
            dotted_values,
            "the dotted name loads itself"
        );
        assert_ne!(parameters, dotted_parameters);
        assert!(load_checkpoint(&net(6), &plain, &device).is_err());
        assert!(load_checkpoint(&net(4), &plain.with_extension(""), &device).is_err());
    }
}
