//! Bundles with controlled contents for unit tests: the committed bundle cut down to its
//! parser, and that parser with small pseudo-random `detector.*` tensors added. Outputs from the
//! random detector are meaningless; the shapes, masking, and pipeline invariants are what tests
//! can check with it.

use serde_json::{Map, Value, json};

use super::{DETECTOR_LABELS, HIDDEN, INPUT_DIM, KERNEL, NGRAM_DIM, SCRIPT_DIM, SHAPE_DIM};
use super::{SCRIPT_ROWS, SHAPE_ROWS, bio};

/// A safetensors header and the tensor bytes it indexes.
struct Parts {
    header: Map<String, Value>,
    data: Vec<u8>,
}

impl Parts {
    fn read(bytes: &[u8]) -> Parts {
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        Parts {
            header: serde_json::from_slice(&bytes[8..8 + n]).unwrap(),
            data: bytes[8 + n..].to_vec(),
        }
    }

    fn write(&self) -> Vec<u8> {
        let header = serde_json::to_vec(&self.header).unwrap();
        let mut out = (header.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&header);
        out.extend_from_slice(&self.data);
        out
    }

    fn metadata(&mut self) -> &mut Map<String, Value> {
        self.header["__metadata__"].as_object_mut().unwrap()
    }

    fn add(&mut self, name: &str, dtype: &str, shape: &[usize], payload: &[u8]) {
        let begin = self.data.len();
        self.data.extend_from_slice(payload);
        let end = self.data.len();
        self.header.insert(
            name.to_string(),
            json!({ "dtype": dtype, "shape": shape, "data_offsets": [begin, end] }),
        );
    }
}

/// The committed bundle with every `detector.*` tensor removed and the manifest listing only
/// the parser, as a parser-only bundle from Milestone 2 would be.
pub(crate) fn parser_bundle() -> Vec<u8> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../models/tessera-v1.safetensors"
    );
    let full = Parts::read(&std::fs::read(path).unwrap());
    let mut kept: Vec<(&String, &Value)> = full
        .header
        .iter()
        .filter(|(name, _)| name.starts_with("parser."))
        .collect();
    kept.sort_by_key(|(_, entry)| entry["data_offsets"][0].as_u64().unwrap());
    let mut out = Parts {
        header: Map::new(),
        data: Vec::new(),
    };
    out.header
        .insert("__metadata__".into(), full.header["__metadata__"].clone());
    for (name, entry) in kept {
        let range = |i: usize| entry["data_offsets"][i].as_u64().unwrap() as usize;
        let shape: Vec<usize> = entry["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_u64().unwrap() as usize)
            .collect();
        out.add(
            name,
            entry["dtype"].as_str().unwrap(),
            &shape,
            &full.data[range(0)..range(1)],
        );
    }
    let metadata = out.metadata();
    metadata.insert("nets".into(), json!("parser"));
    metadata.insert("detector_labels".into(), json!("[]"));
    out.write()
}

/// The parser-only bundle plus a detector with `blocks` random residual blocks.
pub(crate) fn with_random_detector(seed: u64, blocks: usize) -> Vec<u8> {
    random_detector(seed, blocks, true)
}

/// The same, with no `detector.embed.ngram`: a detector trained on the parser's table.
pub(crate) fn with_random_sharing_detector(seed: u64) -> Vec<u8> {
    random_detector(seed, 6, false)
}

fn random_detector(seed: u64, blocks: usize, own_ngram: bool) -> Vec<u8> {
    let mut parts = Parts::read(&parser_bundle());
    let feature_config: Value =
        serde_json::from_str(parts.metadata()["feature_config"].as_str().unwrap()).unwrap();
    let buckets = feature_config["hash_buckets"].as_u64().unwrap() as usize;
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut int8 = |name: String, shape: &[usize], channels: usize| {
        let count: usize = shape.iter().product();
        let values: Vec<u8> = (0..count)
            .map(|_| ((next() % 7) as u8).wrapping_sub(3))
            .collect();
        let scales: Vec<u8> = std::iter::repeat_n(0.02f32, channels)
            .flat_map(f32::to_le_bytes)
            .collect();
        (name, shape.to_vec(), values, scales)
    };
    let mut tensors = Vec::new();
    if own_ngram {
        tensors.push(int8(
            "detector.embed.ngram".into(),
            &[buckets + 1, NGRAM_DIM],
            NGRAM_DIM,
        ));
    }
    tensors.extend([
        int8(
            "detector.embed.script".into(),
            &[SCRIPT_ROWS, SCRIPT_DIM],
            SCRIPT_DIM,
        ),
        int8(
            "detector.embed.shape".into(),
            &[SHAPE_ROWS, SHAPE_DIM],
            SHAPE_DIM,
        ),
        int8("detector.proj.weight".into(), &[HIDDEN, INPUT_DIM], HIDDEN),
        int8(
            "detector.head.weight".into(),
            &[DETECTOR_LABELS, HIDDEN],
            DETECTOR_LABELS,
        ),
    ]);
    for i in 0..blocks {
        tensors.push(int8(
            format!("detector.block{i}.conv.weight"),
            &[HIDDEN, HIDDEN, KERNEL],
            HIDDEN,
        ));
    }
    for (name, shape, values, scales) in tensors {
        parts.add(&name, "I8", &shape, &values);
        parts.add(
            &format!("{name}.scale"),
            "F32",
            &[scales.len() / 4],
            &scales,
        );
    }
    let mut zeros = vec![("detector.proj.bias".to_string(), HIDDEN)];
    zeros.push(("detector.head.bias".to_string(), DETECTOR_LABELS));
    for i in 0..blocks {
        zeros.push((format!("detector.block{i}.conv.bias"), HIDDEN));
    }
    for (name, len) in zeros {
        parts.add(&name, "F32", &[len], &vec![0u8; len * 4]);
    }
    let labels = serde_json::to_string(&bio::detector_label_strings()).unwrap();
    let metadata = parts.metadata();
    metadata.insert("nets".into(), json!("parser,detector"));
    metadata.insert("detector_labels".into(), json!(labels));
    parts.write()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use crate::features::{FeatureConfig, featurize};
    use crate::model::weights::Bundle;
    use crate::model::{Tagger, kernels};
    use crate::token::tokenize;

    #[test]
    fn a_detector_with_six_blocks_loads_and_a_missing_block_is_invalid() {
        let bytes = with_random_detector(7, 6);
        let bundle = Bundle::parse(&bytes, None).unwrap();
        assert!(Tagger::detector(&bundle, None).is_ok());
        let bytes = with_random_detector(7, 5);
        let bundle = Bundle::parse(&bytes, None).unwrap();
        assert_eq!(
            Tagger::detector(&bundle, None).unwrap_err(),
            Error::BundleInvalid
        );
    }

    #[test]
    fn a_detector_without_its_own_table_shares_the_parsers() {
        let bytes = with_random_sharing_detector(5);
        let bundle = Bundle::parse(&bytes, None).unwrap();
        let parser = Tagger::parser(&bundle).unwrap();
        let shared = Tagger::detector(&bundle, Some(&parser)).unwrap();
        assert!(shared.shares_ngram_with(&parser));
        let alone = Tagger::detector(&bundle, None).unwrap();
        assert!(!alone.shares_ngram_with(&parser));
        assert_eq!(
            (&alone.ngram.data, &alone.ngram.scales),
            (&shared.ngram.data, &shared.ngram.scales)
        );
        let own = with_random_detector(5, 6);
        let bundle = Bundle::parse(&own, None).unwrap();
        let detector = Tagger::detector(&bundle, Some(&parser)).unwrap();
        assert!(!detector.shares_ngram_with(&parser));
    }

    #[test]
    fn forward_gives_one_distribution_per_position() {
        let bytes = with_random_detector(11, 6);
        let bundle = Bundle::parse(&bytes, None).unwrap();
        let detector = Tagger::detector(&bundle, None).unwrap();
        let text = (0..37)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let tokens = tokenize(&text);
        let feats: Vec<_> = featurize(&text, &tokens, &[], None, &FeatureConfig::default(), None)
            .into_iter()
            .zip(&tokens)
            .filter(|(_, t)| crate::features::is_content(t))
            .map(|(f, _)| f)
            .collect();
        assert_eq!(feats.len(), 37);
        let mut probs = detector.forward(&feats);
        kernels::softmax_rows(&mut probs, DETECTOR_LABELS);
        assert_eq!(probs.len(), 37 * DETECTOR_LABELS);
        for row in probs.chunks(DETECTOR_LABELS) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
    }
}
