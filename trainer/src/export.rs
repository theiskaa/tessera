//! `trainer export`: the shipping bundle and the golden vectors.
//!
//! The bundle is one safetensors file holding both quantized networks, `parser.*` and
//! `detector.*`, with the manifest in its metadata header, so one fetch and one checksum cover
//! everything the library needs. Golden vectors hold, per case, the features the trainer
//! computed, the f32 logits, the dequantized int8 logits, and the decoded labels (and for the
//! detector the rule-span mask), so the library can check its features and its forward pass
//! separately.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Context;
use burn::backend::NdArray;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use burn::tensor::activation::softmax;
use safetensors::SafeTensors;
use sha2::{Digest, Sha256};
use tessera::internal::{
    DECODER_CONTRACT, TOKENIZER_CONTRACT, argmax, detector_label_strings, parser_label_strings,
};

use crate::config::{Config, Task};
use crate::data::{Split, read_shard};
use crate::dataset::{
    Encoded, FLAG_BITS, MAX_NGRAMS_PER_TOKEN, PARSER_LABELS, ParserBatch, ParserBatcher,
    SCRIPT_ROWS, SHAPE_ROWS, encode,
};
use crate::detector::{self, DETECTOR_LABELS};
use crate::model_eval::decode_probs;
use crate::net::TaggerNet;
use crate::quantize::{self, F32Tensor, QTensor};

/// Bundle layout version; the library checks it before anything else.
pub const FORMAT: &str = "1";
/// Version recorded in the bundle; patch versions after V17 require explicit input contracts.
pub const MODEL_VERSION: &str = "0.2.1";
/// Logit difference the golden gate allows between the trainer and the library.
pub const GOLDEN_TOLERANCE: f64 = 1e-3;
const GOLDEN_PER_COUNTRY: usize = 4;
const GOLDEN_LONG_TOKENS: usize = 200;

/// Detector golden inputs for US documents, including contact and negative layouts.
const DETECTOR_GOLDEN_TEXTS: [(&str, &str); 7] = [
    (
        "signature-us",
        "Thanks for the update.\n\nGrace Liu\nMeridian Health Partners\n1200 Market Street, Philadelphia, PA 19107\n(215) 555-0142\ngrace@meridian.example",
    ),
    (
        "letterhead-us",
        "Blue Ridge Supply Co\n88 Commerce Drive\nAsheville, NC 28801\nPhone: (828) 555-0142\n\nSeptember 18, 2026\n\nDear Ms. Liu,\n\nYour order has shipped.\n\nDaniel Price",
    ),
    (
        "address-us",
        "Send returns to Suite 400, 1200 Market Street, Philadelphia, PA 19107. Questions go to billing@meridian.example.",
    ),
    (
        "table-tab",
        "Name\tCompany\tEmail\nGrace Liu\tMeridian Health Partners\tgrace@meridian.example\nDaniel Price\tBlue Ridge Supply Co\tdaniel@blueridge.example",
    ),
    (
        "prose-negatives",
        "In April the team in Georgia met Ford engineers in Jordan. Grace Liu of Meridian Health Partners said the Abraham Lincoln High School project was on track, and May will review the plan.",
    ),
    (
        "rules-only",
        "Reach us at help@desk.example or (212) 555-0142.",
    ),
    ("one-word", "Hello"),
];

pub(crate) fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// Lowercase hexadecimal of a digest.
pub(crate) fn hex(digest: &[u8]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// The bundle's `__metadata__`: format, versions, label sets, feature settings, and data
/// provenance.
pub fn metadata(cfg: &Config, snapshot: &str, date: &str) -> BTreeMap<String, String> {
    let features = serde_json::json!({
        "ngram_sizes": cfg.features.ngram_sizes,
        "hash_buckets": cfg.features.hash_buckets,
        "hash_seed": cfg.features.hash_seed,
        "max_ngrams_per_token": MAX_NGRAMS_PER_TOKEN,
        "flag_bits": FLAG_BITS,
        "script_rows": SCRIPT_ROWS,
        "shape_rows": SHAPE_ROWS,
    });
    let experimental: Vec<&str> = cfg.data.countries.iter().map(String::as_str).collect();
    BTreeMap::from([
        ("model_name".to_string(), "Tessera".to_string()),
        (
            "description".to_string(),
            "Experimental contact extraction for United States documents".to_string(),
        ),
        ("model_scope".to_string(), "US".to_string()),
        ("format".to_string(), FORMAT.to_string()),
        ("model_version".to_string(), MODEL_VERSION.to_string()),
        (
            "tokenizer_contract".to_string(),
            TOKENIZER_CONTRACT.to_string(),
        ),
        ("decoder_contract".to_string(), DECODER_CONTRACT.to_string()),
        ("nets".to_string(), "parser,detector".to_string()),
        (
            "detector_labels".to_string(),
            json(&detector_label_strings()),
        ),
        ("parser_labels".to_string(), json(&parser_label_strings())),
        ("feature_config".to_string(), features.to_string()),
        (
            "phone_metadata_version".to_string(),
            "libphonenumber v9.0.33 generated tables".to_string(),
        ),
        ("supported_regions".to_string(), "[]".to_string()),
        ("experimental_regions".to_string(), json(&experimental)),
        ("training_snapshot".to_string(), snapshot.to_string()),
        ("report_url".to_string(), String::new()),
        ("created".to_string(), date.to_string()),
    ])
}

/// A label or region list as the manifest stores it, a JSON array in a string.
fn json<T: serde::Serialize>(v: &T) -> String {
    serde_json::to_string(v).expect("lists of strings serialize")
}

/// Reads the quantized run file back into int8 tensors and f32 biases.
fn read_quantized(path: &Path) -> anyhow::Result<(Vec<QTensor>, Vec<F32Tensor>)> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes)?;
    let f32s = |data: &[u8]| -> Vec<f32> {
        data.as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    };
    let mut q = Vec::new();
    let mut biases = Vec::new();
    for (name, view) in st.tensors() {
        if name.ends_with(".scale") {
            continue;
        }
        if quantize::is_quantized(&name) {
            let scales = f32s(st.tensor(&format!("{name}.scale"))?.data());
            q.push(QTensor {
                name,
                shape: view.shape().to_vec(),
                data: view.data().iter().map(|&b| b as i8).collect(),
                scales,
            });
        } else {
            biases.push(F32Tensor {
                name,
                shape: view.shape().to_vec(),
                data: f32s(view.data()),
            });
        }
    }
    Ok((q, biases))
}

/// The library's forward passes are written for one architecture per network, except for the
/// hidden width, which it reads from the bundle; a run with another one must not be exported.
fn check_library_shape(cfg: &Config) -> anyhow::Result<()> {
    let net = cfg.tagger_net_config();
    let dilations: &[usize] = match cfg.task {
        Task::Parser => &[1, 2, 4, 8],
        Task::Detector => &[1, 2, 4, 8, 16, 1],
    };
    let ok = net.dilations == dilations
        && net.kernel == 3
        && (1..=1024).contains(&net.hidden)
        && net.ngram_dim == 48
        && net.script_dim == 8
        && net.shape_dim == 8;
    anyhow::ensure!(
        ok,
        "the library runs the {} with dilations {dilations:?}, kernel 3, hidden 1 to 1024, and \
         embedding dims 48, 8, 8; this run has {:?}, {}, {}, and {}, {}, {}",
        cfg.net_name(),
        net.dilations,
        net.kernel,
        net.hidden,
        net.ngram_dim,
        net.script_dim,
        net.shape_dim
    );
    Ok(())
}

/// A trained run ready to ship: its config and its quantized weights.
struct ShippedRun {
    dir: PathBuf,
    cfg: Config,
    q: Vec<QTensor>,
    biases: Vec<F32Tensor>,
}

fn load_run(dir: &Path, task: Task) -> anyhow::Result<ShippedRun> {
    crate::quantize::verify_gate(dir)?;
    crate::train::verify_input_snapshot(dir)?;
    let cfg = crate::config::load(&dir.join("config.toml"))?;
    anyhow::ensure!(
        cfg.task == task,
        "{} is a {:?} run, not a {task:?} run",
        dir.display(),
        cfg.task
    );
    check_library_shape(&cfg)?;
    let (q, biases) = read_quantized(&dir.join("quantized.safetensors"))?;
    Ok(ShippedRun {
        dir: dir.to_path_buf(),
        cfg,
        q,
        biases,
    })
}

/// Whether the detector was trained on this parser's n-gram table, frozen, in which case the
/// bundle carries the table once and the library gives it to both networks. The table must be
/// the same bytes in both runs, which also proves training left it untouched.
fn shares_parser_ngram(parser: &ShippedRun, detector: &ShippedRun) -> anyhow::Result<bool> {
    let Some(from) = &detector.cfg.net.ngram_from else {
        return Ok(false);
    };
    let same_run = std::fs::canonicalize(from).with_context(|| format!("resolving {from}"))?
        == std::fs::canonicalize(&parser.dir)?;
    anyhow::ensure!(
        same_run,
        "{} was trained on the n-gram table of {from}, not of {}",
        detector.dir.display(),
        parser.dir.display()
    );
    let table = |run: &ShippedRun, name: &str| {
        run.q
            .iter()
            .find(|t| t.name == name)
            .map(|t| (t.data.clone(), t.scales.clone()))
            .with_context(|| format!("{} has no {name}", run.dir.display()))
    };
    anyhow::ensure!(
        table(parser, "parser.embed.ngram")? == table(detector, "detector.embed.ngram")?,
        "{}'s n-gram table differs from {}'s: it was not frozen",
        detector.dir.display(),
        parser.dir.display()
    );
    Ok(true)
}

/// Both networks read features from one computation at inference, so their feature settings
/// must be the same; the error names the first key that differs.
fn check_same_features(parser: &Config, detector: &Config) -> anyhow::Result<()> {
    let (a, b) = (
        serde_json::to_value(&parser.features)?,
        serde_json::to_value(&detector.features)?,
    );
    let empty = serde_json::Map::new();
    let (a, b) = (
        a.as_object().unwrap_or(&empty),
        b.as_object().unwrap_or(&empty),
    );
    for key in a.keys().chain(b.keys()) {
        anyhow::ensure!(
            a.get(key) == b.get(key),
            "the parser and detector runs differ in features.{key}: {} against {}",
            a.get(key).cloned().unwrap_or_default(),
            b.get(key).cloned().unwrap_or_default()
        );
    }
    Ok(())
}

/// Run-bound hashes of the exact train/validation shards, silver files, and frozen n-gram
/// source recorded before training. Current mutable source manifests cannot prove what a run saw.
fn training_snapshot(parser_run: &Path, detector_run: &Path) -> anyhow::Result<String> {
    let mut parts = Vec::new();
    for (name, run) in [
        ("parser-inputs", parser_run),
        ("detector-inputs", detector_run),
    ] {
        crate::train::load_input_snapshot(run)?;
        let bytes = std::fs::read(run.join("input_snapshot.json"))?;
        parts.push(format!("{name}:{}", sha256_hex(&bytes)));
    }
    Ok(parts.join(","))
}

/// Bind the combined bundle to both gated checkpoints and their quantized weights.
fn bind_artifact_hashes(
    meta: &mut BTreeMap<String, String>,
    parser_run: &Path,
    detector_run: &Path,
) -> anyhow::Result<()> {
    for (net, run) in [("parser", parser_run), ("detector", detector_run)] {
        for (name, file) in [
            ("best_sha256", "best.mpk"),
            ("quantized_sha256", "quantized.safetensors"),
            ("input_snapshot_sha256", "input_snapshot.json"),
        ] {
            let path = run.join(file);
            let bytes =
                std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
            meta.insert(format!("{net}_{name}"), sha256_hex(&bytes));
        }
    }
    Ok(())
}

/// `trainer export`: writes the two-network bundle, its checksum file, and the golden vectors
/// of both networks.
pub fn run(parser_run: &Path, detector_run: &Path, out: &Path, date: &str) -> anyhow::Result<()> {
    let parser = load_run(parser_run, Task::Parser)?;
    let detector = load_run(detector_run, Task::Detector)?;
    check_same_features(&parser.cfg, &detector.cfg)?;
    let mut meta = metadata(
        &parser.cfg,
        &training_snapshot(parser_run, detector_run)?,
        date,
    );
    bind_artifact_hashes(&mut meta, parser_run, detector_run)?;
    std::fs::create_dir_all(out)?;
    let bundle = out.join("tessera-v1.safetensors");
    let shared = shares_parser_ngram(&parser, &detector)?;
    let q: Vec<QTensor> = parser
        .q
        .iter()
        .chain(
            detector
                .q
                .iter()
                .filter(|t| !(shared && t.name == "detector.embed.ngram")),
        )
        .cloned()
        .collect();
    let biases: Vec<F32Tensor> = parser
        .biases
        .iter()
        .chain(&detector.biases)
        .cloned()
        .collect();
    quantize::write_safetensors(&bundle, &q, &biases, Some(meta))?;
    let bytes = std::fs::read(&bundle)?;
    let checksum = format!("sha256-{}", sha256_hex(&bytes));
    std::fs::write(out.join("tessera-v1.sha256"), format!("{checksum}\n"))?;
    println!("{} ({} bytes): {checksum}", bundle.display(), bytes.len());

    let device = Default::default();
    let fc = parser.cfg.features.to_tessera();
    let (f32_parser, int8_parser) = models(&parser, &device)?;
    let golden_dir = out.join("golden").join("parser");
    std::fs::create_dir_all(&golden_dir)?;
    for (name, text) in golden_cases(&parser.cfg, &fc)? {
        let enc = encode(&text, &[], &fc).map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
        let case = golden_case(
            Task::Parser,
            &f32_parser,
            &int8_parser,
            &text,
            &enc,
            &device,
        );
        std::fs::write(
            golden_dir.join(format!("{name}.json")),
            golden_json(&case, 0) + "\n",
        )?;
    }
    println!("golden vectors: {}", golden_dir.display());
    let (f32_detector, int8_detector) = models(&detector, &device)?;
    let golden_dir = out.join("golden").join("detector");
    std::fs::create_dir_all(&golden_dir)?;
    for (name, text) in DETECTOR_GOLDEN_TEXTS {
        let enc = detector::encode_document(text, &[], &fc)
            .map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
        let case = golden_case(
            Task::Detector,
            &f32_detector,
            &int8_detector,
            text,
            &enc,
            &device,
        );
        std::fs::write(
            golden_dir.join(format!("{name}.json")),
            golden_json(&case, 0) + "\n",
        )?;
    }
    println!("golden vectors: {}", golden_dir.display());

    let test = read_shard(
        &PathBuf::from(&parser.cfg.data.processed).join("test.parquet"),
        Split::Test,
    )?;
    let shipped = crate::model_eval::score_shipped(&bytes, Some(&checksum), &test)?;
    let eval_dir = parser.dir.join("eval");
    std::fs::create_dir_all(&eval_dir)?;
    std::fs::write(
        eval_dir.join("test-shipped.json"),
        serde_json::to_string_pretty(&shipped.json())? + "\n",
    )?;
    println!(
        "shipped bundle on test, through parse_address: component F1 {:.4}, exact {:.4}",
        shipped.overall.f1(),
        shipped.exact_rate()
    );
    Ok(())
}

/// A run's best f32 model and the same model with its quantized weights dequantized, which is
/// what the library computes.
fn models(
    run: &ShippedRun,
    device: &burn::tensor::Device<NdArray>,
) -> anyhow::Result<(TaggerNet<NdArray>, TaggerNet<NdArray>)> {
    let f32_model = quantize::load_best::<NdArray>(&run.dir, &run.cfg, device)?;
    let dequantized: Vec<F32Tensor> = run
        .q
        .iter()
        .map(|t| quantize::dequantize(t, quantize::channel_axis(&t.name)))
        .chain(run.biases.iter().cloned())
        .collect();
    let int8_model = quantize::inject::<NdArray>(
        &run.cfg.tagger_net_config(),
        &dequantized,
        run.cfg.net_name(),
        device,
    )?;
    Ok((f32_model, int8_model))
}

/// Indented JSON with every array of numbers or strings on one line, so a 200-token case is
/// one line per token rather than one line per logit.
fn golden_json(v: &serde_json::Value, indent: usize) -> String {
    use serde_json::Value;
    let pad = "  ".repeat(indent + 1);
    let close = "  ".repeat(indent);
    match v {
        Value::Object(m) if !m.is_empty() => {
            let fields: Vec<String> = m
                .iter()
                .map(|(k, v)| {
                    format!(
                        "{pad}{}: {}",
                        Value::from(k.as_str()),
                        golden_json(v, indent + 1)
                    )
                })
                .collect();
            format!("{{\n{}\n{close}}}", fields.join(",\n"))
        }
        Value::Array(a) if a.iter().any(|x| x.is_array() || x.is_object()) => {
            let items: Vec<String> = a
                .iter()
                .map(|x| format!("{pad}{}", golden_json(x, indent + 1)))
                .collect();
            format!("[\n{}\n{close}]", items.join(",\n"))
        }
        v => v.to_string(),
    }
}

/// The golden inputs: the first rows per country of the test split with 4 to 30 tokens, and
/// four edge cases: a single token, a single character, a word past the n-gram cap, and a
/// 200-token address.
fn golden_cases(
    cfg: &crate::config::Config,
    fc: &tessera::internal::FeatureConfig,
) -> anyhow::Result<Vec<(String, String)>> {
    let test = read_shard(
        &PathBuf::from(&cfg.data.processed).join("test.parquet"),
        Split::Test,
    )?;
    let mut out = Vec::new();
    for country in &cfg.data.countries {
        let picked = test
            .iter()
            .filter(|e| &e.country == country)
            .filter(|e| {
                encode(&e.text, &[], fc).is_ok_and(|x| (4..=30).contains(&x.token_spans.len()))
            })
            .take(GOLDEN_PER_COUNTRY);
        for (i, e) in picked.enumerate() {
            out.push((
                format!("{}-{:02}", country.to_lowercase(), i + 1),
                e.text.clone(),
            ));
        }
    }
    out.push(("edge-single-token".into(), "Austin".into()));
    out.push(("edge-one-char".into(), "1".into()));
    // 41 letters, past the 21 at which a token's n-grams exceed MAX_NGRAMS_PER_TOKEN.
    out.push((
        "edge-long-word".into(),
        "Pneumonoultramicroscopicsilicovolcanoconiosis, Austin, TX 78701".into(),
    ));
    let mut long = String::new();
    for e in &test {
        if !long.is_empty() {
            long.push('\n');
        }
        long.push_str(&e.text);
        let enc = encode(&long, &[], fc).map_err(|e| anyhow::anyhow!("long case: {e}"))?;
        if enc.token_spans.len() >= GOLDEN_LONG_TOKENS {
            long.truncate(enc.token_spans[GOLDEN_LONG_TOKENS - 1].1 as usize);
            break;
        }
    }
    let tokens = encode(&long, &[], fc)
        .map_err(|e| anyhow::anyhow!("long case: {e}"))?
        .token_spans
        .len();
    anyhow::ensure!(
        tokens == GOLDEN_LONG_TOKENS,
        "the test split holds only {tokens} tokens, short of the {GOLDEN_LONG_TOKENS}-token golden case"
    );
    out.push(("edge-long".into(), long));
    Ok(out)
}

fn logits<B: Backend>(
    model: &TaggerNet<B>,
    labels: usize,
    enc: &Encoded,
    device: &B::Device,
) -> Vec<Vec<f32>> {
    let batch: ParserBatch<B> = ParserBatcher.batch(vec![enc.clone()], device);
    let out = model.forward(
        batch.ngram_ids,
        batch.script,
        batch.shape,
        batch.flags,
        batch.mask.clone(),
    );
    let flat: Vec<f32> = out.into_data().to_vec().expect("logits are f32");
    flat.chunks(labels)
        .take(enc.token_spans.len())
        .map(<[f32]>::to_vec)
        .collect()
}

/// One golden case. The parser's decoded labels are its transition-masked decode; the
/// detector's are the most probable label per position, `O` where the rule-span mask is set,
/// which the file also records.
fn golden_case(
    task: Task,
    f32_model: &TaggerNet<NdArray>,
    int8_model: &TaggerNet<NdArray>,
    text: &str,
    enc: &Encoded,
    device: &burn::tensor::Device<NdArray>,
) -> serde_json::Value {
    let labels = match task {
        Task::Detector => DETECTOR_LABELS,
        Task::Parser => PARSER_LABELS,
    };
    let f32_logits = logits(f32_model, labels, enc, device);
    let int8_logits = logits(int8_model, labels, enc, device);
    let probs: Vec<Vec<f32>> = int8_logits
        .iter()
        .map(|row| {
            let t = Tensor::<NdArray, 1>::from_floats(row.as_slice(), device);
            softmax(t, 0)
                .into_data()
                .to_vec()
                .expect("probabilities are f32")
        })
        .collect();
    let masked: Vec<bool> = enc
        .flags
        .iter()
        .map(|f| f & tessera::internal::flag::IN_RULE_SPAN != 0)
        .collect();
    let decoded: Vec<u8> = match task {
        Task::Detector => probs
            .iter()
            .zip(&masked)
            .map(|(p, &m)| if m { 0 } else { argmax(p) as u8 })
            .collect(),
        Task::Parser => decode_probs(&probs).into_iter().map(|d| d.0).collect(),
    };
    let features: Vec<serde_json::Value> = (0..enc.token_spans.len())
        .map(|t| {
            let ids: Vec<u32> = enc.ngram_ids[t].iter().map(|i| i - 1).collect();
            serde_json::json!({ "ngram_ids": ids, "script": enc.script[t], "shape": enc.shape[t], "flags": enc.flags[t] })
        })
        .collect();
    let mut case = serde_json::json!({
        "net": task.name(),
        "bundle_version": MODEL_VERSION,
        "input": {
            "text": text,
            "tokens": enc.token_spans.iter().map(|(a, b)| [a, b]).collect::<Vec<_>>(),
            "features": features,
        },
        "f32_logits": f32_logits,
        "int8_logits": int8_logits,
        "decoded": decoded,
        "tolerance": GOLDEN_TOLERANCE,
    });
    if task == Task::Detector {
        case["masked"] = serde_json::json!(masked);
    }
    case
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gated_run(task: Task) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let config_name = match task {
            Task::Parser => "parser-small.toml",
            Task::Detector => "detector-shared.toml",
        };
        let mut cfg = crate::config::load(
            &Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("../configs")
                .join(config_name),
        )
        .unwrap();
        cfg.data.processed = dir.path().join("processed").display().to_string();
        cfg.data.sample_manifest = dir.path().join("sample.json").display().to_string();
        if let Some(detector) = cfg.detector.as_mut() {
            detector.silver = vec![dir.path().join("silver.jsonl").display().to_string()];
            cfg.net.ngram_from = None;
        }
        let config = toml::to_string(&cfg).unwrap();
        std::fs::write(dir.path().join("config.toml"), &config).unwrap();
        std::fs::write(dir.path().join("best.mpk"), b"checkpoint").unwrap();
        for (_, path) in crate::train::input_paths(&cfg) {
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, b"training data").unwrap();
        }
        crate::train::capture_input_snapshot(&cfg, dir.path()).unwrap();
        let snapshot_bytes = std::fs::read(dir.path().join("input_snapshot.json")).unwrap();
        quantize::write_safetensors(&dir.path().join("quantized.safetensors"), &[], &[], None)
            .unwrap();
        let quantized = std::fs::read(dir.path().join("quantized.safetensors")).unwrap();
        let gate = serde_json::json!({
            "gate_version": 2,
            "passed": true,
            "config_sha256": sha256_hex(config.as_bytes()),
            "best_sha256": sha256_hex(b"checkpoint"),
            "input_snapshot_sha256": sha256_hex(&snapshot_bytes),
            "quantized_sha256": sha256_hex(&quantized),
        });
        std::fs::write(dir.path().join("quantize.json"), gate.to_string()).unwrap();
        dir
    }

    fn rewrite_snapshot_and_rebind_gate(run_dir: &Path, edit: impl FnOnce(&mut serde_json::Value)) {
        let snapshot_path = run_dir.join("input_snapshot.json");
        let mut snapshot: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&snapshot_path).unwrap()).unwrap();
        edit(&mut snapshot);
        let snapshot_bytes = serde_json::to_vec(&snapshot).unwrap();
        std::fs::write(&snapshot_path, &snapshot_bytes).unwrap();
        let gate_path = run_dir.join("quantize.json");
        let mut gate: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&gate_path).unwrap()).unwrap();
        gate["input_snapshot_sha256"] = sha256_hex(&snapshot_bytes).into();
        std::fs::write(gate_path, serde_json::to_vec(&gate).unwrap()).unwrap();
    }

    #[test]
    fn export_rejects_old_or_mistagged_training_snapshots() {
        for task in [Task::Parser, Task::Detector] {
            let old = gated_run(task);
            rewrite_snapshot_and_rebind_gate(old.path(), |snapshot| {
                snapshot["version"] = 1.into();
                snapshot
                    .as_object_mut()
                    .unwrap()
                    .remove("tokenizer_contract");
                snapshot.as_object_mut().unwrap().remove("decoder_contract");
            });
            let error = load_run(old.path(), task).err().unwrap();
            assert!(
                error
                    .to_string()
                    .contains("unsupported or empty input snapshot"),
                "{error:#}"
            );

            for key in ["tokenizer_contract", "decoder_contract"] {
                let wrong = gated_run(task);
                rewrite_snapshot_and_rebind_gate(wrong.path(), |snapshot| {
                    snapshot[key] = "future-contract".into();
                });
                let error = load_run(wrong.path(), task).err().unwrap();
                assert!(
                    error.to_string().contains("contract differs"),
                    "{key}: {error:#}"
                );

                let missing = gated_run(task);
                rewrite_snapshot_and_rebind_gate(missing.path(), |snapshot| {
                    snapshot.as_object_mut().unwrap().remove(key);
                });
                let error = load_run(missing.path(), task).err().unwrap();
                assert!(
                    error.to_string().contains("contract differs"),
                    "{key}: {error:#}"
                );
            }
        }
    }

    #[test]
    fn both_networks_require_current_artifact_hashes() {
        for task in [Task::Parser, Task::Detector] {
            let dir = gated_run(task);
            load_run(dir.path(), task).unwrap();
            for (file, key) in [
                ("config.toml", "config_sha256"),
                ("best.mpk", "best_sha256"),
                ("input_snapshot.json", "input_snapshot_sha256"),
                ("quantized.safetensors", "quantized_sha256"),
            ] {
                let path = dir.path().join(file);
                let original = std::fs::read(&path).unwrap();
                let mut changed = original.clone();
                changed.push(b' ');
                std::fs::write(&path, changed).unwrap();
                let err = match load_run(dir.path(), task) {
                    Ok(_) => panic!("{file} mutation was accepted for {task:?}"),
                    Err(err) => err,
                };
                assert!(err.to_string().contains(key), "{err:#}");
                std::fs::write(&path, original).unwrap();
            }
        }
    }

    #[test]
    fn export_rechecks_training_file_bytes_after_quantization() {
        let dir = gated_run(Task::Detector);
        load_run(dir.path(), Task::Detector).unwrap();
        std::fs::write(
            dir.path().join("processed/train.parquet"),
            b"changed training data",
        )
        .unwrap();
        let error = match load_run(dir.path(), Task::Detector) {
            Ok(_) => panic!("changed training input was accepted"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("train_shard"), "{error:#}");
        assert!(
            error.to_string().contains("changed during training"),
            "{error:#}"
        );
    }

    #[test]
    fn legacy_unbound_gate_requires_requantization() {
        let dir = gated_run(Task::Parser);
        std::fs::write(dir.path().join("quantize.json"), r#"{"passed":true}"#).unwrap();
        let err = match load_run(dir.path(), Task::Parser) {
            Ok(_) => panic!("legacy gate was accepted"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("requantize this run"), "{err:#}");
    }

    #[test]
    fn export_rejects_runs_without_training_input_snapshots() {
        let parser = gated_run(Task::Parser);
        let detector = gated_run(Task::Detector);
        std::fs::remove_file(parser.path().join("input_snapshot.json")).unwrap();
        let error = training_snapshot(parser.path(), detector.path()).unwrap_err();
        assert!(
            error.to_string().contains("no run-bound input snapshot"),
            "{error:#}"
        );
    }

    #[test]
    fn golden_json_is_compact_and_round_trips() {
        let v = serde_json::json!({
            "tolerance": GOLDEN_TOLERANCE,
            "decoded": [0, 3, 4],
            "int8_logits": [[0.5, -1.25], [2.0, 3.0]],
            "input": { "features": [{ "ngram_ids": [1, 2], "flags": 0 }], "text": "a \"b\"" },
        });
        let s = golden_json(&v, 0);
        assert!(s.contains("\"tolerance\": 0.001"), "{s}");
        assert!(s.contains("[0.5,-1.25]"), "{s}");
        assert_eq!(serde_json::from_str::<serde_json::Value>(&s).unwrap(), v);
    }

    #[test]
    fn label_strings_follow_the_id_table() {
        let l = parser_label_strings();
        assert_eq!(l.len(), PARSER_LABELS);
        assert_eq!(l[1], "B-house_number");
        assert_eq!(l[22], "I-po_box");
    }

    #[test]
    fn metadata_has_every_key() {
        let cfg = crate::config::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/parser-small.toml"),
        )
        .unwrap();
        let m = metadata(&cfg, "parser-sample:abc", "2026-09-22");
        for key in [
            "model_name",
            "description",
            "model_scope",
            "format",
            "model_version",
            "tokenizer_contract",
            "decoder_contract",
            "nets",
            "detector_labels",
            "parser_labels",
            "feature_config",
            "phone_metadata_version",
            "supported_regions",
            "experimental_regions",
            "training_snapshot",
            "report_url",
            "created",
        ] {
            assert!(m.contains_key(key), "{key}");
        }
        assert_eq!(m["model_version"], "0.2.1");
        assert_eq!(m["model_scope"], "US");
        assert_eq!(m["tokenizer_contract"], "tessera-tokenize-legacy-v1");
        assert_eq!(m["decoder_contract"], "tessera-bio-legacy-v1");
    }

    #[test]
    fn saved_safetensors_exposes_export_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = crate::config::load(
            &Path::new(env!("CARGO_MANIFEST_DIR")).join("../configs/parser-small.toml"),
        )
        .unwrap();
        let expected = metadata(&cfg, "parser-inputs:abc,detector-inputs:def", "2026-09-28");
        let path = dir.path().join("model.safetensors");
        quantize::write_safetensors(&path, &[], &[], Some(expected.clone())).unwrap();
        let bytes = std::fs::read(path).unwrap();
        let (_, header) = SafeTensors::read_metadata(&bytes).unwrap();
        assert_eq!(
            header.metadata().as_ref(),
            Some(&expected.into_iter().collect())
        );
    }

    #[test]
    fn artifact_metadata_binds_both_checkpoints_and_weight_files() {
        let parser = tempfile::tempdir().unwrap();
        let detector = tempfile::tempdir().unwrap();
        for (run, net) in [(parser.path(), "parser"), (detector.path(), "detector")] {
            for file in ["best.mpk", "quantized.safetensors", "input_snapshot.json"] {
                let bytes = format!("{net} {file}");
                std::fs::write(run.join(file), &bytes).unwrap();
            }
        }
        let mut meta = BTreeMap::new();
        bind_artifact_hashes(&mut meta, parser.path(), detector.path()).unwrap();
        for (run, net) in [(parser.path(), "parser"), (detector.path(), "detector")] {
            for (key, file) in [
                ("best_sha256", "best.mpk"),
                ("quantized_sha256", "quantized.safetensors"),
                ("input_snapshot_sha256", "input_snapshot.json"),
            ] {
                let expected = sha256_hex(&std::fs::read(run.join(file)).unwrap());
                assert_eq!(meta[&format!("{net}_{key}")], expected);
            }
        }
    }
}
