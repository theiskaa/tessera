//! Burn definition of the address parser. The inference twin is `tessera::model`, and the
//! golden-vector gate keeps the two in agreement.
//!
//! Every operation here (embedding lookup and mean, concatenation, linear layers, dilated
//! 1-D convolution, ReLU, residual addition) has a hand-written counterpart in the library,
//! which is why there is no normalization layer, attention, or CRF.

use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig, PaddingConfig1d,
};
use burn::prelude::*;
use burn::tensor::activation::relu;

use crate::dataset::{FLAG_BITS, PARSER_LABELS, SCRIPT_ROWS, SHAPE_ROWS};

/// Shapes of the parser network; the library runs only the one `export` accepts.
#[derive(burn::config::Config, Debug)]
pub struct ParserNetConfig {
    /// N-gram hash buckets; the table has one more row, for padding.
    pub hash_buckets: usize,
    /// Width of the n-gram embedding.
    #[config(default = 48)]
    pub ngram_dim: usize,
    /// Width of the script embedding.
    #[config(default = 8)]
    pub script_dim: usize,
    /// Width of the shape embedding.
    #[config(default = 8)]
    pub shape_dim: usize,
    /// Channels through the convolution stack.
    #[config(default = 96)]
    pub hidden: usize,
    /// Convolution kernel width.
    #[config(default = 3)]
    pub kernel: usize,
    /// One residual block per dilation.
    pub dilations: Vec<usize>,
    /// Dropout after each block, during training only.
    #[config(default = 0.1)]
    pub dropout: f64,
}

/// `y = x + dropout(relu(conv_d(x)))`, keeping the sequence length.
#[derive(Module, Debug)]
pub struct ConvBlock<B: Backend> {
    /// The dilated convolution.
    pub conv: Conv1d<B>,
    dropout: Dropout,
}

impl<B: Backend> ConvBlock<B> {
    /// `x`: `[B, C, L]` to `[B, C, L]`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let h = relu(self.conv.forward(x.clone()));
        x + self.dropout.forward(h)
    }
}

/// The parser network: n-gram, script, and shape embeddings plus flags, a projection, four
/// residual dilated convolutions, and a per-token label head.
#[derive(Module, Debug)]
pub struct ParserNet<B: Backend> {
    /// `[hash_buckets + 1, ngram_dim]`; row 0 is padding and never contributes.
    pub ngram: Embedding<B>,
    /// `[SCRIPT_ROWS, script_dim]`.
    pub script: Embedding<B>,
    /// `[SHAPE_ROWS, shape_dim]`.
    pub shape: Embedding<B>,
    /// `ngram_dim + script_dim + shape_dim + FLAG_BITS` to `hidden`.
    pub proj: Linear<B>,
    proj_dropout: Dropout,
    pub blocks: Vec<ConvBlock<B>>,
    /// `hidden` to `PARSER_LABELS`.
    pub head: Linear<B>,
}

impl ParserNetConfig {
    /// A freshly initialized network with these shapes.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ParserNet<B> {
        let blocks = self
            .dilations
            .iter()
            .map(|&d| {
                let pad = d * (self.kernel - 1) / 2;
                ConvBlock {
                    conv: Conv1dConfig::new(self.hidden, self.hidden, self.kernel)
                        .with_dilation(d)
                        .with_padding(PaddingConfig1d::Explicit(pad, pad))
                        .with_bias(true)
                        .init(device),
                    dropout: DropoutConfig::new(self.dropout).init(),
                }
            })
            .collect();
        ParserNet {
            ngram: EmbeddingConfig::new(self.hash_buckets + 1, self.ngram_dim).init(device),
            script: EmbeddingConfig::new(SCRIPT_ROWS, self.script_dim).init(device),
            shape: EmbeddingConfig::new(SHAPE_ROWS, self.shape_dim).init(device),
            proj: LinearConfig::new(
                self.ngram_dim + self.script_dim + self.shape_dim + FLAG_BITS,
                self.hidden,
            )
            .init(device),
            proj_dropout: DropoutConfig::new(self.dropout).init(),
            blocks,
            head: LinearConfig::new(self.hidden, PARSER_LABELS).init(device),
        }
    }
}

impl<B: Backend> ParserNet<B> {
    /// Logits `[B, L, PARSER_LABELS]` for n-gram ids `[B, L, K]`, script and shape ids
    /// `[B, L]`, flags `[B, L, FLAG_BITS]`, and `mask` `[B, L]` (1 for real tokens).
    ///
    /// Padded positions are zeroed before every convolution, so a sequence's output does not
    /// depend on what it was batched with and equals the library, which runs one sequence at a
    /// time with zeros beyond its ends.
    pub fn forward(
        &self,
        ngram_ids: Tensor<B, 3, Int>,
        script: Tensor<B, 2, Int>,
        shape: Tensor<B, 2, Int>,
        flags: Tensor<B, 3>,
        mask: Tensor<B, 2>,
    ) -> Tensor<B, 3> {
        let [b, l, k] = ngram_ids.dims();
        let dim = self.ngram.weight.dims()[1];
        let present = ngram_ids.clone().greater_elem(0).float();
        let count = present.clone().sum_dim(2).clamp_min(1.0);
        let rows = self
            .ngram
            .forward(ngram_ids.reshape([b, l * k]))
            .reshape([b, l, k, dim]);
        let summed: Tensor<B, 3> = (rows * present.unsqueeze_dim::<4>(3))
            .sum_dim(2)
            .squeeze_dim(2);
        let ngram = summed / count;
        let script = self.script.forward(script);
        let shape = self.shape.forward(shape);
        let x = Tensor::cat(vec![ngram, script, shape, flags], 2);
        let x = self.proj_dropout.forward(self.proj.forward(x));
        let keep = mask.unsqueeze_dim::<3>(1);
        let mut x = x.swap_dims(1, 2) * keep.clone();
        for block in &self.blocks {
            x = block.forward(x) * keep.clone();
        }
        self.head.forward(x.swap_dims(1, 2))
    }

    /// Parameters outside the three embedding tables.
    pub fn non_embedding_params(&self) -> usize {
        self.num_params()
            - self.ngram.num_params()
            - self.script.num_params()
            - self.shape.num_params()
    }
}

/// Checks the size budget: non-embedding parameters against `max_params`, and the n-gram
/// table's int8 bytes against `max_embedding_bytes`. The hashed table is most of the model,
/// so one limit for everything would either forbid it or say nothing about the rest.
pub fn assert_size<B: Backend>(
    model: &ParserNet<B>,
    max_params: usize,
    max_embedding_bytes: usize,
) -> anyhow::Result<(usize, usize)> {
    let dense = model.non_embedding_params();
    anyhow::ensure!(
        dense <= max_params,
        "parser has {dense} non-embedding parameters, over the {max_params} budget"
    );
    let embedding_bytes = model.ngram.num_params();
    anyhow::ensure!(
        embedding_bytes <= max_embedding_bytes,
        "the n-gram table needs {embedding_bytes} int8 bytes, over the {max_embedding_bytes} budget"
    );
    Ok((dense, embedding_bytes))
}

#[cfg(test)]
mod tests {
    use burn::backend::NdArray;

    use super::*;

    fn config() -> ParserNetConfig {
        ParserNetConfig::new(1024, vec![1, 2, 4, 8])
    }

    #[test]
    fn forward_shapes() {
        let device = Default::default();
        let net = config().init::<NdArray>(&device);
        let ids = Tensor::<NdArray, 3, Int>::random(
            [2, 7, 24],
            burn::tensor::Distribution::Uniform(0.0, 1024.0),
            &device,
        );
        let script = Tensor::<NdArray, 2, Int>::zeros([2, 7], &device);
        let shape = Tensor::<NdArray, 2, Int>::zeros([2, 7], &device);
        let flags = Tensor::<NdArray, 3>::zeros([2, 7, FLAG_BITS], &device);
        assert_eq!(
            net.forward(ids, script, shape, flags, Tensor::ones([2, 7], &device))
                .dims(),
            [2, 7, PARSER_LABELS]
        );
    }

    #[test]
    fn non_embedding_parameter_count() {
        let net = config().init::<NdArray>(&Default::default());
        // proj 86*96 + 96, four blocks of 96*96*3 + 96, head 96*23 + 23.
        assert_eq!(net.non_embedding_params(), 8_352 + 4 * 27_744 + 2_231);
    }

    #[test]
    fn all_padding_ids_do_not_produce_nan() {
        let device = Default::default();
        let net = config().init::<NdArray>(&device);
        let out = net.forward(
            Tensor::zeros([1, 3, 24], &device),
            Tensor::zeros([1, 3], &device),
            Tensor::zeros([1, 3], &device),
            Tensor::zeros([1, 3, FLAG_BITS], &device),
            Tensor::ones([1, 3], &device),
        );
        assert!(!out.is_nan().any().into_scalar());
    }

    #[test]
    fn padding_does_not_change_a_sequence() {
        let device = Default::default();
        let net = config().init::<NdArray>(&device);
        let ids = Tensor::<NdArray, 3, Int>::random(
            [1, 5, 24],
            burn::tensor::Distribution::Uniform(1.0, 1024.0),
            &device,
        );
        let script = Tensor::<NdArray, 2, Int>::ones([1, 5], &device);
        let shape = Tensor::<NdArray, 2, Int>::ones([1, 5], &device);
        let flags = Tensor::<NdArray, 3>::ones([1, 5, FLAG_BITS], &device);
        let alone = net.forward(
            ids.clone(),
            script.clone(),
            shape.clone(),
            flags.clone(),
            Tensor::ones([1, 5], &device),
        );
        let padded = net
            .forward(
                Tensor::cat(vec![ids, Tensor::zeros([1, 11, 24], &device)], 1),
                Tensor::cat(vec![script, Tensor::zeros([1, 11], &device)], 1),
                Tensor::cat(vec![shape, Tensor::zeros([1, 11], &device)], 1),
                Tensor::cat(vec![flags, Tensor::zeros([1, 11, FLAG_BITS], &device)], 1),
                Tensor::cat(
                    vec![
                        Tensor::ones([1, 5], &device),
                        Tensor::zeros([1, 11], &device),
                    ],
                    1,
                ),
            )
            .slice([0..1, 0..5]);
        let diff: f32 = (alone - padded).abs().max().into_scalar();
        assert!(diff < 1e-5, "padding changed the output by {diff}");
    }
}
