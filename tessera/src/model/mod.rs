//! The learned networks as hand-written forward passes over int8 weights, and the bundle they
//! load from. The address parser and the entity detector are one tagger architecture with
//! different depths and label counts, defined in the trainer with Burn as `TaggerNet`; the
//! golden vectors keep the two in agreement.

pub(crate) mod bio;
pub(crate) mod json;
pub(crate) mod kernels;
pub(crate) mod sha256;
#[cfg(test)]
pub(crate) mod testing;
pub(crate) mod weights;

// The trainer builds its network from the constants below through `internal`; a bundle
// recording other values is rejected at load.

/// Bundle layout version this library reads.
pub(crate) const SUPPORTED_FORMAT: &str = "1";
/// Model version series this library runs: under semver before 1.0 a minor bump is breaking,
/// so bundles `0.2.*` load and `0.1.*` or `0.3.0` do not.
pub(crate) const SUPPORTED_MODEL_SERIES: (u32, u32) = (0, 2);
/// Width of the flag input: one entry per bit of `TokenFeatures::flags`. Bits 0 to 21 are the
/// constants in `features::flag`; bit 22 is reserved for marking text hidden by Markdown syntax
/// and is always 0 until Markdown input sets it, so bundles need no retraining then.
pub const FLAG_BITS: usize = 23;
/// Rows of the script embedding table.
pub const SCRIPT_ROWS: usize = 13;
/// Rows of the shape embedding table.
pub const SHAPE_ROWS: usize = 64;
/// `O`, then a `B` and an `I` label for each of the eleven semantic address labels.
pub const PARSER_LABELS: usize = 23;
/// `O`, then a `B` and an `I` label for person, org, and address.
pub const DETECTOR_LABELS: usize = 7;

use crate::Error;
use crate::features::TokenFeatures;
use weights::{Bundle, QTensor};

/// More retained tokens than this is `InputTooLarge`; `parse_address` takes one address.
pub(crate) const MAX_PARSE_TOKENS: usize = 256;
/// Checked before tokenizing, so one enormous token cannot pass the token limit.
pub(crate) const MAX_PARSE_BYTES: usize = 8 * 1024;
// The architecture is fixed by format 1. A bundle whose tensors have other shapes is invalid,
// so its size bounds the work a call does.
/// One residual block per dilation, for each network.
const PARSER_DILATIONS: [usize; 4] = [1, 2, 4, 8];
const DETECTOR_DILATIONS: [usize; 6] = [1, 2, 4, 8, 16, 1];
/// Convolution kernel width.
const KERNEL: usize = 3;
/// Width of the mean n-gram embedding.
const NGRAM_DIM: usize = 48;
/// Width of the script embedding.
const SCRIPT_DIM: usize = 8;
/// Width of the shape embedding.
const SHAPE_DIM: usize = 8;
/// Width of the hidden layers.
const HIDDEN: usize = 96;
/// Width of one token's input to the projection.
const INPUT_DIM: usize = NGRAM_DIM + SCRIPT_DIM + SHAPE_DIM + FLAG_BITS;

/// A tagger network with its weights, validated once so the kernels can trust every shape.
#[derive(Debug)]
pub(crate) struct Tagger {
    ngram: QTensor,
    script: QTensor,
    shape: QTensor,
    proj_w: QTensor,
    proj_b: Vec<f32>,
    blocks: Vec<(QTensor, Vec<f32>, usize)>,
    head_w: QTensor,
    head_b: Vec<f32>,
    labels: usize,
}

impl Tagger {
    /// The address parser: `parser.*` tensors, four blocks, `PARSER_LABELS` outputs.
    pub(crate) fn parser(bundle: &Bundle<'_>) -> Result<Tagger, Error> {
        if bundle.manifest.parser_labels != bio::parser_label_strings() {
            return Err(Error::BundleInvalid);
        }
        Tagger::load(bundle, "parser", &PARSER_DILATIONS, PARSER_LABELS)
    }

    /// The entity detector: `detector.*` tensors, six blocks, `DETECTOR_LABELS` outputs.
    pub(crate) fn detector(bundle: &Bundle<'_>) -> Result<Tagger, Error> {
        if bundle.manifest.detector_labels != bio::detector_label_strings() {
            return Err(Error::BundleInvalid);
        }
        Tagger::load(bundle, "detector", &DETECTOR_DILATIONS, DETECTOR_LABELS)
    }

    /// Copies the tensors named `<net>.*` out of `bundle`, checking dtype and shape of each.
    fn load(
        bundle: &Bundle<'_>,
        net: &str,
        dilations: &[usize],
        labels: usize,
    ) -> Result<Tagger, Error> {
        let rows = (bundle.manifest.feature_config.hash_buckets as usize)
            .checked_add(1)
            .ok_or(Error::BundleInvalid)?;
        let mut blocks = Vec::with_capacity(dilations.len());
        for (i, &d) in dilations.iter().enumerate() {
            blocks.push((
                bundle.take_i8(
                    &format!("{net}.block{i}.conv.weight"),
                    &[HIDDEN, HIDDEN, KERNEL],
                    0,
                )?,
                bundle.take_f32(&format!("{net}.block{i}.conv.bias"), HIDDEN)?,
                d,
            ));
        }
        Ok(Tagger {
            ngram: bundle.take_i8(&format!("{net}.embed.ngram"), &[rows, NGRAM_DIM], 1)?,
            script: bundle.take_i8(
                &format!("{net}.embed.script"),
                &[SCRIPT_ROWS, SCRIPT_DIM],
                1,
            )?,
            shape: bundle.take_i8(&format!("{net}.embed.shape"), &[SHAPE_ROWS, SHAPE_DIM], 1)?,
            proj_w: bundle.take_i8(&format!("{net}.proj.weight"), &[HIDDEN, INPUT_DIM], 0)?,
            proj_b: bundle.take_f32(&format!("{net}.proj.bias"), HIDDEN)?,
            blocks,
            head_w: bundle.take_i8(&format!("{net}.head.weight"), &[labels, HIDDEN], 0)?,
            head_b: bundle.take_f32(&format!("{net}.head.bias"), labels)?,
            labels,
        })
    }

    /// Output labels per token.
    pub(crate) fn labels(&self) -> usize {
        self.labels
    }

    /// Logits `[len, labels]` for the retained tokens' features.
    pub(crate) fn forward(&self, feats: &[TokenFeatures]) -> Vec<f32> {
        let len = feats.len();
        let mut x = vec![0f32; len * INPUT_DIM];
        for (row, f) in x.chunks_mut(INPUT_DIM).zip(feats) {
            let (ngram, rest) = row.split_at_mut(NGRAM_DIM);
            let (script, rest) = rest.split_at_mut(SCRIPT_DIM);
            let (shape, flags) = rest.split_at_mut(SHAPE_DIM);
            kernels::embedding_mean(&self.ngram, &f.ngram_ids, ngram);
            kernels::embedding_row(&self.script, usize::from(f.script), script);
            kernels::embedding_row(&self.shape, usize::from(f.shape), shape);
            for (bit, v) in flags.iter_mut().enumerate() {
                *v = ((f.flags >> bit) & 1) as f32;
            }
        }
        let mut h = vec![0f32; len * HIDDEN];
        kernels::linear(&x, len, &self.proj_w, &self.proj_b, &mut h);
        let mut conv = vec![0f32; len * HIDDEN];
        for (w, b, d) in &self.blocks {
            kernels::conv1d_same(&h, len, HIDDEN, w, b, *d, &mut conv);
            kernels::relu_inplace(&mut conv);
            kernels::add_inplace(&mut h, &conv);
        }
        let mut logits = vec![0f32; len * self.labels];
        kernels::linear(&h, len, &self.head_w, &self.head_b, &mut logits);
        logits
    }
}
