//! `trainer export`: the shipping bundle and the golden vectors.
//!
//! The bundle is the quantized safetensors file with the manifest in its metadata header, so
//! one fetch and one checksum cover everything the library needs. Golden vectors hold, per
//! case, the features the trainer computed, the f32 logits, the dequantized int8 logits, and
//! the decoded labels, so the library can check its features and its forward pass separately.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Context;
use burn::backend::NdArray;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use burn::tensor::activation::softmax;
use safetensors::SafeTensors;
use sha2::{Digest, Sha256};
use tessera::internal::parser_label_strings;

use crate::data::{Split, read_shard};
use crate::dataset::{
    Encoded, FLAG_BITS, MAX_NGRAMS_PER_TOKEN, PARSER_LABELS, ParserBatch, ParserBatcher,
    SCRIPT_ROWS, SHAPE_ROWS, encode,
};
use crate::model_eval::decode_probs;
use crate::net::TaggerNet;
use crate::quantize::{self, F32Tensor, QTensor};

/// Bundle layout version; the library checks it before anything else.
pub const FORMAT: &str = "1";
/// Version recorded in the bundle; the library accepts bundles of its own `0.minor` series.
pub const MODEL_VERSION: &str = "0.2.0";
/// Logit difference the golden gate allows between the trainer and the library.
pub const GOLDEN_TOLERANCE: f64 = 1e-3;
const GOLDEN_PER_COUNTRY: usize = 4;
const GOLDEN_LONG_TOKENS: usize = 200;

fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

/// Lowercase hexadecimal of a digest.
pub(crate) fn hex(digest: &[u8]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

/// The bundle's `__metadata__`: format, versions, label sets, feature settings, and data
/// provenance.
pub fn metadata(
    cfg: &crate::config::Config,
    snapshot: &str,
    date: &str,
) -> BTreeMap<String, String> {
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
        ("format".to_string(), FORMAT.to_string()),
        ("model_version".to_string(), MODEL_VERSION.to_string()),
        ("nets".to_string(), "parser".to_string()),
        ("detector_labels".to_string(), "[]".to_string()),
        (
            "parser_labels".to_string(),
            serde_json::to_string(&parser_label_strings()).expect("strings serialize"),
        ),
        ("feature_config".to_string(), features.to_string()),
        (
            "phone_metadata_version".to_string(),
            "libphonenumber v9.0.33 generated tables".to_string(),
        ),
        ("supported_regions".to_string(), "[]".to_string()),
        (
            "experimental_regions".to_string(),
            serde_json::to_string(&experimental).expect("strings serialize"),
        ),
        (
            "training_snapshot".to_string(),
            format!("parser-sample:{snapshot}"),
        ),
        ("report_url".to_string(), String::new()),
        ("created".to_string(), date.to_string()),
    ])
}

/// Reads the quantized run file back into int8 tensors and f32 biases.
fn read_quantized(path: &Path) -> anyhow::Result<(Vec<QTensor>, Vec<F32Tensor>)> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let st = SafeTensors::deserialize(&bytes)?;
    let f32s = |data: &[u8]| -> Vec<f32> {
        data.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
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

/// The library's forward pass is written for one architecture; the bundle does not describe
/// it, so a run with another one must not be exported.
fn check_library_shape(cfg: &crate::config::Config) -> anyhow::Result<()> {
    let net = cfg.parser_net_config();
    let ok = net.dilations == [1, 2, 4, 8]
        && net.kernel == 3
        && net.hidden == 96
        && net.ngram_dim == 48
        && net.script_dim == 8
        && net.shape_dim == 8;
    anyhow::ensure!(
        ok,
        "the library runs dilations [1, 2, 4, 8], kernel 3, hidden 96, and embedding dims 48, 8, 8; \
         this run has {:?}, {}, {}, and {}, {}, {}",
        net.dilations,
        net.kernel,
        net.hidden,
        net.ngram_dim,
        net.script_dim,
        net.shape_dim
    );
    Ok(())
}

/// `trainer export`: writes the bundle, its checksum file, and the golden vectors.
pub fn run(run_dir: &Path, out: &Path, date: &str) -> anyhow::Result<()> {
    let cfg = crate::config::load(&run_dir.join("config.toml"))?;
    let gate: serde_json::Value = serde_json::from_slice(
        &std::fs::read(run_dir.join("quantize.json"))
            .with_context(|| format!("reading {}", run_dir.join("quantize.json").display()))?,
    )?;
    anyhow::ensure!(
        gate["passed"] == true,
        "{} did not pass the quantization gate; run `trainer quantize` first",
        run_dir.display()
    );
    let (q, biases) = read_quantized(&run_dir.join("quantized.safetensors"))?;
    check_library_shape(&cfg)?;
    let manifest = std::fs::read(&cfg.data.sample_manifest)
        .with_context(|| format!("reading {}", cfg.data.sample_manifest))?;
    let meta = metadata(&cfg, &sha256_hex(&manifest), date);

    std::fs::create_dir_all(out)?;
    let bundle = out.join("tessera-v1.safetensors");
    quantize::write_safetensors(&bundle, &q, &biases, Some(meta))?;
    let bytes = std::fs::read(&bundle)?;
    let checksum = format!("sha256-{}", sha256_hex(&bytes));
    std::fs::write(out.join("tessera-v1.sha256"), format!("{checksum}\n"))?;
    println!("{} ({} bytes): {checksum}", bundle.display(), bytes.len());

    let device = Default::default();
    let f32_model = quantize::load_best::<NdArray>(run_dir, &cfg, &device)?;
    let dequantized: Vec<F32Tensor> = q
        .iter()
        .map(|t| quantize::dequantize(t, quantize::channel_axis(&t.name)))
        .chain(biases.iter().cloned())
        .collect();
    let int8_model = quantize::inject::<NdArray>(&cfg.parser_net_config(), &dequantized, &device)?;
    let golden_dir = out.join("golden").join("parser");
    std::fs::create_dir_all(&golden_dir)?;
    let fc = cfg.features.to_tessera();
    for (name, text) in golden_cases(&cfg, &fc)? {
        let enc = encode(&text, &[], &fc).map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
        let case = golden_case(&f32_model, &int8_model, &text, &enc, &device);
        std::fs::write(
            golden_dir.join(format!("{name}.json")),
            golden_json(&case, 0) + "\n",
        )?;
    }
    println!("golden vectors: {}", golden_dir.display());

    let test = read_shard(
        &PathBuf::from(&cfg.data.processed).join("test.parquet"),
        Split::Test,
    )?;
    let shipped = crate::model_eval::score_shipped(&bytes, &checksum, &test)?;
    let eval_dir = run_dir.join("eval");
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
    out.push(("edge-single-token".into(), "London".into()));
    out.push(("edge-one-char".into(), "1".into()));
    // 41 letters, past the 21 at which a token's n-grams exceed MAX_NGRAMS_PER_TOKEN.
    out.push((
        "edge-long-word".into(),
        "Donaudampfschifffahrtsgesellschaftsstraße 12, 10115 Berlin".into(),
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

fn logits<B: Backend>(model: &TaggerNet<B>, enc: &Encoded, device: &B::Device) -> Vec<Vec<f32>> {
    let batch: ParserBatch<B> = ParserBatcher.batch(vec![enc.clone()], device);
    let out = model.forward(
        batch.ngram_ids,
        batch.script,
        batch.shape,
        batch.flags,
        batch.mask.clone(),
    );
    let flat: Vec<f32> = out.into_data().to_vec().expect("logits are f32");
    flat.chunks(PARSER_LABELS)
        .take(enc.token_spans.len())
        .map(<[f32]>::to_vec)
        .collect()
}

fn golden_case(
    f32_model: &TaggerNet<NdArray>,
    int8_model: &TaggerNet<NdArray>,
    text: &str,
    enc: &Encoded,
    device: &burn::tensor::Device<NdArray>,
) -> serde_json::Value {
    let f32_logits = logits(f32_model, enc, device);
    let int8_logits = logits(int8_model, enc, device);
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
    let decoded: Vec<u8> = decode_probs(&probs).into_iter().map(|d| d.0).collect();
    let features: Vec<serde_json::Value> = (0..enc.token_spans.len())
        .map(|t| {
            let ids: Vec<u32> = enc.ngram_ids[t].iter().map(|i| i - 1).collect();
            serde_json::json!({ "ngram_ids": ids, "script": enc.script[t], "shape": enc.shape[t], "flags": enc.flags[t] })
        })
        .collect();
    serde_json::json!({
        "net": "parser",
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
    })
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let m = metadata(&cfg, "abc", "2026-09-22");
        for key in [
            "format",
            "model_version",
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
    }
}
