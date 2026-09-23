//! Encoding labelled examples into the network's input tensors and BIO labels, and batching
//! them with padding. Features come from the library's own `featurize`, so the model learns
//! from exactly what inference will compute.

use burn::data::dataloader::batcher::Batcher;
use burn::prelude::*;
use burn::tensor::TensorData;
use tessera::AddressLabel;
use tessera::internal::{FeatureConfig, TokenClass, featurize, tokenize};

use crate::data::{LabelledExample, Span};

// The network's input and output widths come from the library, so the two cannot drift.
pub use tessera::internal::{
    FLAG_BITS, MAX_NGRAMS_PER_TOKEN, PARSER_LABELS, SCRIPT_ROWS, SHAPE_ROWS,
};

/// One example as the network sees it: only non-whitespace tokens, in order.
#[derive(Clone, Debug, PartialEq)]
pub struct Encoded {
    /// Byte offsets of the retained tokens.
    pub token_spans: Vec<(u32, u32)>,
    /// Per token, `id + 1` for each n-gram id, at most `MAX_NGRAMS_PER_TOKEN`. Batches pad
    /// with 0 to the longest token in the batch.
    pub ngram_ids: Vec<Vec<u32>>,
    /// Script id per retained token.
    pub script: Vec<u8>,
    /// Shape id per retained token.
    pub shape: Vec<u8>,
    /// Flag bits per retained token, `FLAG_BITS` of them used.
    pub flags: Vec<u32>,
    /// Parser label id per retained token; all `O` when encoding for inference.
    pub labels: Vec<u8>,
    /// Country of the example, for per-country evaluation.
    pub country: String,
}

/// Why an example could not be encoded.
#[derive(Debug, thiserror::Error)]
pub enum EncodeError {
    #[error("span {span:?} cuts token {token:?}")]
    PartialToken { token: (u32, u32), span: (u32, u32) },
    #[error("span {span:?} is labelled `unknown`, which is not a training label")]
    UnknownLabel { span: (u32, u32) },
}

/// `B-x` is `1 + 2 * index(x)` and `I-x` is `2 + 2 * index(x)`, over the eleven semantic labels
/// in `AddressLabel` order; 0 is `O`. `Unknown` is an output of the library's policy, never a
/// training label, and has no id.
pub fn label_id(label: AddressLabel, begin: bool) -> Option<u8> {
    let i = AddressLabel::ALL.iter().position(|l| *l == label)? as u8;
    let semantic = (PARSER_LABELS as u8 - 1) / 2;
    (i < semantic).then(|| if begin { 1 + 2 * i } else { 2 + 2 * i })
}

/// Encodes one address. Postcode shapes are matched for any known country, since inference
/// gets no reliable country for a single address.
pub fn encode(text: &str, spans: &[Span], fc: &FeatureConfig) -> Result<Encoded, EncodeError> {
    let tokens = tokenize(text);
    let feats = featurize(text, &tokens, &[], None, fc);
    let mut enc = Encoded {
        token_spans: Vec::new(),
        ngram_ids: Vec::new(),
        script: Vec::new(),
        shape: Vec::new(),
        flags: Vec::new(),
        labels: Vec::new(),
        country: String::new(),
    };
    let mut span_idx = 0usize;
    for (t, f) in tokens.iter().zip(&feats) {
        if matches!(t.class, TokenClass::Space | TokenClass::Newline) {
            continue;
        }
        let (start, end) = (t.start as u32, t.end as u32);
        while span_idx < spans.len() && spans[span_idx].end <= start {
            span_idx += 1;
        }
        let label = match spans.get(span_idx) {
            Some(s) if s.start <= start && end <= s.end => label_id(s.label, s.start == start)
                .ok_or(EncodeError::UnknownLabel {
                    span: (s.start, s.end),
                })?,
            Some(s) if start < s.end && s.start < end => {
                return Err(EncodeError::PartialToken {
                    token: (start, end),
                    span: (s.start, s.end),
                });
            }
            _ => 0,
        };
        enc.token_spans.push((start, end));
        enc.ngram_ids
            .push(f.ngram_ids.iter().map(|id| id + 1).collect());
        enc.script.push(f.script);
        enc.shape.push(f.shape);
        enc.flags.push(f.flags);
        enc.labels.push(label);
    }
    Ok(enc)
}

/// Greedy BIO decoding with transition masking: an `I-x` that does not follow `B-x` or `I-x`
/// starts a new span, as if it were `B-x`.
pub fn decode_labels(token_spans: &[(u32, u32)], labels: &[u8]) -> Vec<Span> {
    let mut out: Vec<Span> = Vec::new();
    let mut prev: u8 = 0;
    for (i, &l) in labels.iter().enumerate() {
        if l == 0 {
            prev = 0;
            continue;
        }
        let (label_idx, is_begin) = ((l - 1) / 2, (l - 1) % 2 == 0);
        let label = AddressLabel::ALL[label_idx as usize];
        let continues = !is_begin && prev != 0 && (prev - 1) / 2 == label_idx;
        match out.last_mut() {
            Some(last) if continues => last.end = token_spans[i].1,
            _ => out.push(Span {
                label,
                start: token_spans[i].0,
                end: token_spans[i].1,
            }),
        }
        prev = l;
    }
    out
}

/// How many rows encoded and how many were skipped because a span cut a token.
#[derive(Debug, Default, Clone)]
pub struct EncodeStats {
    /// Rows encoded.
    pub encoded: usize,
    /// Rows skipped because a span boundary falls inside a token.
    pub cuts_token: usize,
    /// Encoded original (not augmented) rows per country.
    pub originals: std::collections::BTreeMap<String, usize>,
}

/// Encoded training examples, in shard order.
pub struct ParserDataset {
    /// The encoded rows, in shard order.
    pub items: Vec<Encoded>,
}

impl ParserDataset {
    /// Encodes rows. `limit_per_country` keeps the first N original rows per country, in
    /// row-id order, plus their augmented copies, for the learning curves. `copies` is the
    /// augmentation count the rows were prepared with, which recovers a copy's original.
    pub fn from_rows(
        rows: &[LabelledExample],
        fc: &FeatureConfig,
        limit_per_country: Option<usize>,
        copies: usize,
    ) -> (ParserDataset, EncodeStats) {
        let kept = limit_per_country.map(|limit| {
            let mut originals: Vec<&LabelledExample> =
                rows.iter().filter(|r| !r.augmented).collect();
            originals.sort_by_key(|r| r.id);
            let mut per_country: std::collections::HashMap<&str, usize> = Default::default();
            let mut ids = std::collections::HashSet::new();
            for r in originals {
                let n = per_country.entry(r.country.as_str()).or_default();
                if *n < limit {
                    *n += 1;
                    ids.insert(r.id);
                }
            }
            ids
        });
        let mut stats = EncodeStats::default();
        let mut items = Vec::with_capacity(rows.len());
        for r in rows {
            if let Some(kept) = &kept {
                let original = if r.augmented {
                    crate::data::augment::original_id(r.id, copies)
                } else {
                    r.id
                };
                if !kept.contains(&original) {
                    continue;
                }
            }
            match encode(&r.text, &r.spans, fc) {
                Ok(mut e) => {
                    e.country = r.country.clone();
                    items.push(e);
                    stats.encoded += 1;
                    if !r.augmented {
                        *stats.originals.entry(r.country.clone()).or_default() += 1;
                    }
                }
                Err(_) => stats.cuts_token += 1,
            }
        }
        (ParserDataset { items }, stats)
    }
}

/// A padded batch of encoded examples, ready for the network.
#[derive(Clone, Debug)]
pub struct ParserBatch<B: Backend> {
    /// `[B, L, K]`, 0-padded.
    pub ngram_ids: Tensor<B, 3, Int>,
    /// `[B, L]`.
    pub script: Tensor<B, 2, Int>,
    /// `[B, L]`.
    pub shape: Tensor<B, 2, Int>,
    /// `[B, L, 22]`.
    pub flags: Tensor<B, 3>,
    /// `[B, L]`, 0 where padded.
    pub labels: Tensor<B, 2, Int>,
    /// `[B, L]`, 1.0 for real tokens.
    pub mask: Tensor<B, 2>,
    /// Real token count per example.
    pub lengths: Vec<usize>,
}

/// Builds a [`ParserBatch`] from encoded examples.
#[derive(Clone, Default)]
pub struct ParserBatcher;

impl<B: Backend> Batcher<B, Encoded, ParserBatch<B>> for ParserBatcher {
    fn batch(&self, items: Vec<Encoded>, device: &B::Device) -> ParserBatch<B> {
        let b = items.len();
        let l = items
            .iter()
            .map(|e| e.token_spans.len())
            .max()
            .unwrap_or(1)
            .max(1);
        let k = items
            .iter()
            .flat_map(|e| e.ngram_ids.iter().map(Vec::len))
            .max()
            .unwrap_or(1)
            .max(1);
        let mut ngram = vec![0i64; b * l * k];
        let mut script = vec![0i64; b * l];
        let mut shape = vec![0i64; b * l];
        let mut flags = vec![0f32; b * l * FLAG_BITS];
        let mut labels = vec![0i64; b * l];
        let mut mask = vec![0f32; b * l];
        for (i, e) in items.iter().enumerate() {
            for t in 0..e.token_spans.len() {
                let base = (i * l + t) * k;
                for (slot, &id) in ngram[base..base + k].iter_mut().zip(&e.ngram_ids[t]) {
                    *slot = i64::from(id);
                }
                script[i * l + t] = i64::from(e.script[t]);
                shape[i * l + t] = i64::from(e.shape[t]);
                for bit in 0..FLAG_BITS {
                    flags[(i * l + t) * FLAG_BITS + bit] = ((e.flags[t] >> bit) & 1) as f32;
                }
                labels[i * l + t] = i64::from(e.labels.get(t).copied().unwrap_or(0));
                mask[i * l + t] = 1.0;
            }
        }
        ParserBatch {
            ngram_ids: Tensor::from_data(TensorData::new(ngram, [b, l, k]), device),
            script: Tensor::from_data(TensorData::new(script, [b, l]), device),
            shape: Tensor::from_data(TensorData::new(shape, [b, l]), device),
            flags: Tensor::from_data(TensorData::new(flags, [b, l, FLAG_BITS]), device),
            labels: Tensor::from_data(TensorData::new(labels, [b, l]), device),
            mask: Tensor::from_data(TensorData::new(mask, [b, l]), device),
            lengths: items.iter().map(|e| e.token_spans.len()).collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;

    use super::*;

    fn span(label: AddressLabel, text: &str, part: &str) -> Span {
        let start = text.find(part).unwrap() as u32;
        Span {
            label,
            start,
            end: start + part.len() as u32,
        }
    }

    /// A row of `tokens` one-letter words, so an encoded item names its row by its length.
    fn row(id: u64, country: &str, augmented: bool, tokens: usize) -> LabelledExample {
        LabelledExample {
            id,
            group_id: 0,
            country: country.into(),
            language: "en".into(),
            spans: Vec::new(),
            text: vec!["a"; tokens].join(" "),
            split: crate::data::Split::Train,
            augmented,
        }
    }

    #[test]
    fn row_limit_counts_originals_per_country() {
        let copies = 2;
        let copy = |id| crate::data::augment::copy_id(id, copies, 1);
        let rows = vec![
            row(5, "GB", false, 1),
            row(3, "GB", false, 2),
            row(4, "GB", false, 3),
            row(1, "DE", false, 4),
            row(copy(3), "GB", true, 7),
            row(copy(5), "GB", true, 8),
        ];
        let (ds, stats) =
            ParserDataset::from_rows(&rows, &FeatureConfig::default(), Some(2), copies);
        // GB keeps its two lowest ids, 3 and 4, and the copy of 3; DE keeps 1.
        let mut kept: Vec<usize> = ds.items.iter().map(|e| e.token_spans.len()).collect();
        kept.sort_unstable();
        assert_eq!(kept, [2, 3, 4, 7]);
        assert_eq!(stats.originals.get("GB"), Some(&2));
    }

    #[test]
    fn encodes_the_spec_address() {
        use AddressLabel as L;
        let text = "Flat 4, 221B Baker Street, London NW1 6XE";
        let spans = vec![
            span(L::Unit, text, "Flat 4"),
            span(L::HouseNumber, text, "221B"),
            span(L::Road, text, "Baker Street"),
            span(L::City, text, "London"),
            span(L::Postcode, text, "NW1 6XE"),
        ];
        let e = encode(text, &spans, &FeatureConfig::default()).unwrap();
        let b = |l| label_id(l, true).unwrap();
        let i = |l| label_id(l, false).unwrap();
        assert_eq!(
            e.labels,
            vec![
                b(L::Unit),
                i(L::Unit),
                0,
                b(L::HouseNumber),
                b(L::Road),
                i(L::Road),
                0,
                b(L::City),
                b(L::Postcode),
                i(L::Postcode)
            ]
        );
        assert_eq!(decode_labels(&e.token_spans, &e.labels), spans);
    }

    #[test]
    fn label_ids_follow_the_table() {
        assert_eq!(label_id(AddressLabel::HouseNumber, true), Some(1));
        assert_eq!(label_id(AddressLabel::Road, false), Some(4));
        assert_eq!(label_id(AddressLabel::PoBox, false), Some(22));
        assert_eq!(label_id(AddressLabel::Unknown, true), None);
    }

    #[test]
    fn stray_inside_label_starts_a_span() {
        let spans = decode_labels(&[(0, 2), (3, 5)], &[4, 4]);
        assert_eq!(
            spans,
            vec![Span {
                label: AddressLabel::Road,
                start: 0,
                end: 5
            }]
        );
        let spans = decode_labels(&[(0, 2), (3, 5)], &[0, 4]);
        assert_eq!(
            spans,
            vec![Span {
                label: AddressLabel::Road,
                start: 3,
                end: 5
            }]
        );
    }

    #[test]
    fn batches_pad_to_the_longest() {
        let fc = FeatureConfig::default();
        let a = encode("10 Downing St", &[], &fc).unwrap();
        let b = encode("Flat 4, 221B Baker Street", &[], &fc).unwrap();
        assert_eq!((a.token_spans.len(), b.token_spans.len()), (3, 6));
        // `^Downing$` has 8 bigrams, 7 trigrams, and 6 four-grams.
        let longest = a.ngram_ids.iter().chain(&b.ngram_ids).map(Vec::len).max();
        assert_eq!(longest, Some(21));
        let batch: ParserBatch<NdArray> = ParserBatcher.batch(vec![a, b], &Default::default());
        assert_eq!(batch.ngram_ids.dims(), [2, 6, 21]);
        assert_eq!(batch.script.dims(), [2, 6]);
        assert_eq!(batch.flags.dims(), [2, 6, FLAG_BITS]);
        assert_eq!(batch.mask.sum().into_scalar(), 9.0);
    }
}
