//! A bundle with detector weights for tests, until a trained detector ships: the committed
//! parser bundle with small pseudo-random `detector.*` tensors appended and the manifest
//! extended to list the detector. The outputs are meaningless; the shapes, masking, and
//! pipeline invariants are what tests can check with it.

use super::{DETECTOR_LABELS, HIDDEN, INPUT_DIM, KERNEL, NGRAM_DIM, SCRIPT_DIM, SHAPE_DIM};
use super::{SCRIPT_ROWS, SHAPE_ROWS, bio};

/// The committed bundle.
pub(crate) fn parser_bundle() -> Vec<u8> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../models/tessera-v1.safetensors"
    );
    std::fs::read(path).unwrap()
}

/// The committed bundle plus a detector with `blocks` random residual blocks.
pub(crate) fn with_random_detector(seed: u64, blocks: usize) -> Vec<u8> {
    let bytes = parser_bundle();
    let n = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
    let header = std::str::from_utf8(&bytes[8..8 + n]).unwrap();
    let mut data = bytes[8 + n..].to_vec();
    let buckets = header
        .split("\\\"hash_buckets\\\":")
        .nth(1)
        .and_then(|rest| rest.split(|c: char| !c.is_ascii_digit()).next())
        .and_then(|d| d.parse::<usize>().ok())
        .unwrap();
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let mut entries = String::new();
    let mut add = |name: &str, dtype: &str, shape: &[usize], payload: Vec<u8>| {
        let begin = data.len();
        data.extend_from_slice(&payload);
        let dims: Vec<String> = shape.iter().map(ToString::to_string).collect();
        entries.push_str(&format!(
            r#","{name}":{{"dtype":"{dtype}","shape":[{}],"data_offsets":[{begin},{}]}}"#,
            dims.join(","),
            data.len()
        ));
    };
    let mut int8 = |name: &str, shape: &[usize], channels: usize| {
        let count: usize = shape.iter().product();
        let values: Vec<u8> = (0..count)
            .map(|_| ((next() % 7) as u8).wrapping_sub(3))
            .collect();
        let scales: Vec<u8> = std::iter::repeat_n(0.02f32, channels)
            .flat_map(f32::to_le_bytes)
            .collect();
        (values, scales, name.to_string(), shape.to_vec(), channels)
    };
    let zeros = |len: usize| vec![0u8; len * 4];
    let mut tensors = vec![
        int8("detector.embed.ngram", &[buckets + 1, NGRAM_DIM], NGRAM_DIM),
        int8(
            "detector.embed.script",
            &[SCRIPT_ROWS, SCRIPT_DIM],
            SCRIPT_DIM,
        ),
        int8("detector.embed.shape", &[SHAPE_ROWS, SHAPE_DIM], SHAPE_DIM),
        int8("detector.proj.weight", &[HIDDEN, INPUT_DIM], HIDDEN),
        int8(
            "detector.head.weight",
            &[DETECTOR_LABELS, HIDDEN],
            DETECTOR_LABELS,
        ),
    ];
    for i in 0..blocks {
        tensors.push(int8(
            &format!("detector.block{i}.conv.weight"),
            &[HIDDEN, HIDDEN, KERNEL],
            HIDDEN,
        ));
    }
    for (values, scales, name, shape, _) in tensors {
        add(&name, "I8", &shape, values);
        add(&format!("{name}.scale"), "F32", &[scales.len() / 4], scales);
    }
    add("detector.proj.bias", "F32", &[HIDDEN], zeros(HIDDEN));
    add(
        "detector.head.bias",
        "F32",
        &[DETECTOR_LABELS],
        zeros(DETECTOR_LABELS),
    );
    for i in 0..blocks {
        add(
            &format!("detector.block{i}.conv.bias"),
            "F32",
            &[HIDDEN],
            zeros(HIDDEN),
        );
    }
    let labels: Vec<String> = bio::detector_label_strings()
        .iter()
        .map(|l| format!("\\\"{l}\\\""))
        .collect();
    let header = header
        .replacen(r#""nets":"parser""#, r#""nets":"parser,detector""#, 1)
        .replacen(
            r#""detector_labels":"[]""#,
            &format!(r#""detector_labels":"[{}]""#, labels.join(",")),
            1,
        );
    let body = header.trim_end().strip_suffix('}').unwrap();
    let header = format!("{body}{entries}}}");
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(header.as_bytes());
    out.extend_from_slice(&data);
    out
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
        assert!(Tagger::detector(&bundle).is_ok());
        let bytes = with_random_detector(7, 5);
        let bundle = Bundle::parse(&bytes, None).unwrap();
        assert_eq!(Tagger::detector(&bundle).unwrap_err(), Error::BundleInvalid);
    }

    #[test]
    fn forward_gives_one_distribution_per_position() {
        let bytes = with_random_detector(11, 6);
        let bundle = Bundle::parse(&bytes, None).unwrap();
        let detector = Tagger::detector(&bundle).unwrap();
        let text = (0..37)
            .map(|i| format!("w{i}"))
            .collect::<Vec<_>>()
            .join(" ");
        let tokens = tokenize(&text);
        let feats: Vec<_> = featurize(&text, &tokens, &[], None, &FeatureConfig::default())
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
