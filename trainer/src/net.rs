//! Burn definition of the token taggers: the address parser and the entity detector are the
//! same network with different depths and label counts. The inference twin is
//! `tessera::model`, and the golden-vector gate keeps the two in agreement.
//!
//! Every operation here (embedding lookup and mean, concatenation, linear layers, dilated
//! 1-D convolution, ReLU, residual addition, and the named context detector's post-residual
//! channel RMS normalization) has a hand-written counterpart in the library.

use burn::module::Module;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{
    Dropout, DropoutConfig, Embedding, EmbeddingConfig, Linear, LinearConfig, PaddingConfig1d,
};
use burn::prelude::*;
use burn::tensor::activation::relu;

#[cfg(test)]
use crate::dataset::FLAG_BITS;
use crate::dataset::{SCRIPT_ROWS, SHAPE_ROWS};

#[path = "activation_trace.rs"]
mod activation_trace;

use activation_trace::{LayerActivation, layer_activation};

/// Shapes of a tagger network; the library runs only the ones `export` accepts.
#[derive(burn::config::Config, Debug)]
pub struct TaggerNetConfig {
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
    /// Output labels per token: `PARSER_LABELS` or `DETECTOR_LABELS`.
    pub labels: usize,
    /// Versioned numeric flag width; original graphs default to 23.
    #[config(default = 23)]
    pub flag_bits: usize,
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

/// A tagger: n-gram, script, and shape embeddings plus flags, a projection, residual dilated
/// convolutions, and a per-token label head.
#[derive(Module, Debug)]
pub struct TaggerNet<B: Backend> {
    /// `[hash_buckets + 1, ngram_dim]`; row 0 is padding and never contributes.
    pub ngram: Embedding<B>,
    /// `[SCRIPT_ROWS, script_dim]`.
    pub script: Embedding<B>,
    /// `[SHAPE_ROWS, shape_dim]`.
    pub shape: Embedding<B>,
    /// `ngram_dim + script_dim + shape_dim + flag_bits` to `hidden`.
    pub proj: Linear<B>,
    proj_dropout: Dropout,
    pub blocks: Vec<ConvBlock<B>>,
    /// `hidden` to `labels`.
    pub head: Linear<B>,
}

impl TaggerNetConfig {
    /// A freshly initialized network with these shapes.
    pub fn init<B: Backend>(&self, device: &B::Device) -> TaggerNet<B> {
        assert!(
            matches!(self.flag_bits, 23 | 25),
            "unsupported numeric flag width"
        );
        assert!(
            self.labels != crate::dataset::PARSER_LABELS || self.flag_bits == 23,
            "parser flags must stay legacy23"
        );
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
        TaggerNet {
            ngram: EmbeddingConfig::new(self.hash_buckets + 1, self.ngram_dim).init(device),
            script: EmbeddingConfig::new(SCRIPT_ROWS, self.script_dim).init(device),
            shape: EmbeddingConfig::new(SHAPE_ROWS, self.shape_dim).init(device),
            proj: LinearConfig::new(
                self.ngram_dim + self.script_dim + self.shape_dim + self.flag_bits,
                self.hidden,
            )
            .init(device),
            proj_dropout: DropoutConfig::new(self.dropout).init(),
            blocks,
            head: LinearConfig::new(self.hidden, self.labels).init(device),
        }
    }
}

impl<B: Backend> TaggerNet<B> {
    /// Flag width read from the actual parameter shapes, including migrated native records.
    pub fn flag_bits(&self) -> usize {
        self.proj.weight.dims()[0]
            - self.ngram.weight.dims()[1]
            - self.script.weight.dims()[1]
            - self.shape.weight.dims()[1]
    }
    /// Logits `[B, L, labels]` for n-gram ids `[B, L, K]`, script and shape ids
    /// `[B, L]`, flags `[B, L, self.flag_bits()]`, and `mask` `[B, L]` (1 for real tokens).
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
        assert_eq!(
            flags.dims()[2],
            self.flag_bits(),
            "numeric flags differ from projection contract"
        );
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

    /// Trace one forward with unchanged parameters and dropout calls, excluding padding.
    pub(crate) fn diagnostic_forward(
        &self,
        ngram_ids: Tensor<B, 3, Int>,
        script: Tensor<B, 2, Int>,
        shape: Tensor<B, 2, Int>,
        flags: Tensor<B, 3>,
        mask: Tensor<B, 2>,
    ) -> anyhow::Result<(Tensor<B, 3>, Vec<LayerActivation>)> {
        assert_eq!(
            flags.dims()[2],
            self.flag_bits(),
            "numeric flags differ from projection contract"
        );
        let [b, l, k] = ngram_ids.dims();
        anyhow::ensure!(
            mask.dims() == [b, l],
            "layer trace input mask dimensions differ"
        );
        let mask_values = mask
            .clone()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|error| anyhow::anyhow!("reading layer trace mask: {error:?}"))?;
        anyhow::ensure!(
            mask_values.iter().all(|&value| value == 0. || value == 1.),
            "layer trace needs a binary real-token mask"
        );
        let real: Vec<_> = mask_values.into_iter().map(|value| value == 1.).collect();
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
        let x = self.proj.forward(x);
        let mut trace = vec![layer_activation(
            "projection.before_dropout",
            x.clone(),
            &real,
        )?];
        let x = self.proj_dropout.forward(x);
        trace.push(layer_activation(
            "projection.after_dropout",
            x.clone(),
            &real,
        )?);
        let keep = mask.unsqueeze_dim::<3>(1);
        let mut x = x.swap_dims(1, 2) * keep.clone();
        for (index, block) in self.blocks.iter().enumerate() {
            let prefix = format!("block.{index}");
            trace.push(layer_activation(
                &format!("{prefix}.input"),
                x.clone().swap_dims(1, 2),
                &real,
            )?);
            let h = block.conv.forward(x.clone());
            trace.push(layer_activation(
                &format!("{prefix}.convolution"),
                h.clone().swap_dims(1, 2),
                &real,
            )?);
            let h = relu(h);
            trace.push(layer_activation(
                &format!("{prefix}.relu"),
                h.clone().swap_dims(1, 2),
                &real,
            )?);
            let h = block.dropout.forward(h);
            trace.push(layer_activation(
                &format!("{prefix}.dropout_branch"),
                h.clone().swap_dims(1, 2),
                &real,
            )?);
            x = (x + h) * keep.clone();
            trace.push(layer_activation(
                &format!("{prefix}.residual_output"),
                x.clone().swap_dims(1, 2),
                &real,
            )?);
        }
        let logits = self.head.forward(x.swap_dims(1, 2));
        trace.push(layer_activation("head", logits.clone(), &real)?);
        Ok((logits, trace))
    }

    /// Executes the separately bound seven-block deployment operator.
    pub(crate) fn context96_rms_forward(
        &self,
        ids: Tensor<B, 3, Int>,
        script: Tensor<B, 2, Int>,
        shape: Tensor<B, 2, Int>,
        flags: Tensor<B, 3>,
        mask: Tensor<B, 2>,
    ) -> anyhow::Result<Tensor<B, 3>> {
        self.validate_context96_graph()?;
        let logits = self.residual_rms_forward(ids, script, shape, flags, mask)?;
        anyhow::ensure!(
            logits
                .clone()
                .is_finite()
                .all()
                .into_scalar()
                .elem::<bool>(),
            "non-finite context RMS logits"
        );
        Ok(logits)
    }

    fn validate_context96_graph(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            matches!(self.flag_bits(), 23 | 25)
                && self.blocks.len() == 7
                && self.proj.weight.dims()[1] == 96
                && self.head.weight.dims() == [96, 7]
                && self
                    .blocks
                    .iter()
                    .zip(tessera::internal::CONTEXT96_RMS_DILATIONS)
                    .all(|(block, dilation)| block.conv.weight.dims() == [96, 96, 3]
                        && block.conv.dilation == dilation
                        && block.conv.stride == 1
                        && block.conv.groups == 1
                        && block.conv.padding == PaddingConfig1d::Explicit(dilation, dilation)),
            "context RMS model graph differs from its declared deployment operator"
        );
        Ok(())
    }

    /// Parameter-free post-residual RMS, restricted by external diagnostic metadata.
    pub(crate) fn residual_rms_forward(
        &self,
        ngram_ids: Tensor<B, 3, Int>,
        script: Tensor<B, 2, Int>,
        shape: Tensor<B, 2, Int>,
        flags: Tensor<B, 3>,
        mask: Tensor<B, 2>,
    ) -> anyhow::Result<Tensor<B, 3>> {
        assert_eq!(
            flags.dims()[2],
            self.flag_bits(),
            "numeric flags differ from projection contract"
        );
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
        let mut x = x.swap_dims(1, 2).mask_fill(keep.clone().equal_elem(0.), 0.);
        for block in &self.blocks {
            x = crate::residual_rms::normalize(block.forward(x), keep.clone())?;
        }
        Ok(self.head.forward(x.swap_dims(1, 2)))
    }

    /// Trace the already verified diagnostic operator without changing RNG order.
    pub(crate) fn diagnostic_forward_with_operator(
        &self,
        ids: Tensor<B, 3, Int>,
        script: Tensor<B, 2, Int>,
        shape: Tensor<B, 2, Int>,
        flags: Tensor<B, 3>,
        mask: Tensor<B, 2>,
        operator: crate::diagnostic_operator::ForwardOperator,
    ) -> anyhow::Result<(Tensor<B, 3>, Vec<LayerActivation>)> {
        match operator {
            crate::diagnostic_operator::ForwardOperator::Standard => {
                self.diagnostic_forward(ids, script, shape, flags, mask)
            }
            crate::diagnostic_operator::ForwardOperator::ContextRmsV2 => {
                self.validate_context96_graph()?;
                self.residual_rms_diagnostic_forward(ids, script, shape, flags, mask)
            }
            crate::diagnostic_operator::ForwardOperator::ResidualRmsV1 => {
                self.residual_rms_diagnostic_forward(ids, script, shape, flags, mask)
            }
        }
    }

    /// Trace the parameter-free diagnostic using the same RMS and dropout operations.
    pub(crate) fn residual_rms_diagnostic_forward(
        &self,
        ngram_ids: Tensor<B, 3, Int>,
        script: Tensor<B, 2, Int>,
        shape: Tensor<B, 2, Int>,
        flags: Tensor<B, 3>,
        mask: Tensor<B, 2>,
    ) -> anyhow::Result<(Tensor<B, 3>, Vec<LayerActivation>)> {
        assert_eq!(
            flags.dims()[2],
            self.flag_bits(),
            "numeric flags differ from projection contract"
        );
        let [b, l, k] = ngram_ids.dims();
        anyhow::ensure!(
            mask.dims() == [b, l],
            "layer trace input mask dimensions differ"
        );
        let mask_values = mask
            .clone()
            .into_data()
            .convert::<f32>()
            .to_vec::<f32>()
            .map_err(|error| anyhow::anyhow!("reading layer trace mask: {error:?}"))?;
        anyhow::ensure!(
            mask_values.iter().all(|&value| value == 0. || value == 1.),
            "layer trace needs a binary real-token mask"
        );
        let real: Vec<_> = mask_values.into_iter().map(|value| value == 1.).collect();
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
        let x = self.proj.forward(x);
        let mut trace = vec![layer_activation(
            "projection.before_dropout",
            x.clone(),
            &real,
        )?];
        let x = self.proj_dropout.forward(x);
        trace.push(layer_activation(
            "projection.after_dropout",
            x.clone(),
            &real,
        )?);
        let keep = mask.unsqueeze_dim::<3>(1);
        let mut x = x.swap_dims(1, 2).mask_fill(keep.clone().equal_elem(0.), 0.);
        for (index, block) in self.blocks.iter().enumerate() {
            let prefix = format!("block.{index}");
            trace.push(layer_activation(
                &format!("{prefix}.input"),
                x.clone().swap_dims(1, 2),
                &real,
            )?);
            let h = block.conv.forward(x.clone());
            trace.push(layer_activation(
                &format!("{prefix}.convolution"),
                h.clone().swap_dims(1, 2),
                &real,
            )?);
            let h = relu(h);
            trace.push(layer_activation(
                &format!("{prefix}.relu"),
                h.clone().swap_dims(1, 2),
                &real,
            )?);
            let h = block.dropout.forward(h);
            trace.push(layer_activation(
                &format!("{prefix}.dropout_branch"),
                h.clone().swap_dims(1, 2),
                &real,
            )?);
            let residual = x + h;
            x = residual.clone() * keep.clone();
            trace.push(layer_activation(
                &format!("{prefix}.residual_output"),
                x.clone().swap_dims(1, 2),
                &real,
            )?);
            let (normalized, normalization) =
                crate::residual_rms::normalize_traced(residual, keep.clone())?;
            x = normalized;
            let mut activation = layer_activation(
                &format!("{prefix}.normalized_output"),
                x.clone().swap_dims(1, 2),
                &real,
            )?;
            activation.normalization = Some(normalization);
            trace.push(activation);
        }
        let logits = self.head.forward(x.swap_dims(1, 2));
        trace.push(layer_activation("head", logits.clone(), &real)?);
        Ok((logits, trace))
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
    model: &TaggerNet<B>,
    max_params: usize,
    max_embedding_bytes: usize,
) -> anyhow::Result<(usize, usize)> {
    let dense = model.non_embedding_params();
    anyhow::ensure!(
        dense <= max_params,
        "the network has {dense} non-embedding parameters, over the {max_params} budget"
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
    use burn::backend::{Autodiff, NdArray};
    use burn::module::AutodiffModule;

    use super::*;
    use crate::dataset::PARSER_LABELS;

    fn config() -> TaggerNetConfig {
        TaggerNetConfig::new(1024, vec![1, 2, 4, 8], PARSER_LABELS)
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
        // proj 87*96 + 96, four blocks of 96*96*3 + 96, head 96*23 + 23.
        assert_eq!(net.non_embedding_params(), 8_448 + 4 * 27_744 + 2_231);
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

    #[test]
    fn diagnostic_forward_preserves_logits_and_rng_for_unequal_masked_sequences() {
        // NdArray's process-wide RNG can be reseeded by unrelated parallel tests.
        if std::env::var_os("TESSERA_LAYER_TRACE_TEST_ISOLATED").is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "net::tests::diagnostic_forward_preserves_logits_and_rng_for_unequal_masked_sequences",
                    "--test-threads=1",
                ])
                .env("TESSERA_LAYER_TRACE_TEST_ISOLATED", "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }
        type B = Autodiff<NdArray>;
        let device = Default::default();
        for dropout in [0., 0.4] {
            let cfg = TaggerNetConfig::new(16, vec![1, 2], 7)
                .with_ngram_dim(3)
                .with_script_dim(2)
                .with_shape_dim(2)
                .with_hidden(4)
                .with_dropout(dropout);
            B::seed(&device, 19);
            let model = cfg.init::<B>(&device);
            let _ = crate::quantize::extract(&model.valid(), "detector");
            let ids = Tensor::<B, 3, Int>::from_data(
                [[[1, 2], [3, 0], [15, 15]], [[4, 0], [15, 15], [15, 15]]],
                &device,
            );
            let script = Tensor::<B, 2, Int>::ones([2, 3], &device);
            let shape = Tensor::<B, 2, Int>::ones([2, 3], &device);
            let flags = Tensor::<B, 3>::ones([2, 3, FLAG_BITS], &device);
            let mask = Tensor::<B, 2>::from_data([[1., 1., 0.], [1., 0., 0.]], &device);
            for seed in 42..46 {
                B::seed(&device, seed);
                let ordinary = model
                    .forward(
                        ids.clone(),
                        script.clone(),
                        shape.clone(),
                        flags.clone(),
                        mask.clone(),
                    )
                    .into_data()
                    .to_vec::<f32>()
                    .unwrap();
                let next_ordinary =
                    Tensor::<B, 1>::random([8], burn::tensor::Distribution::Default, &device)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap();
                B::seed(&device, seed);
                let (diagnostic, trace) = model
                    .diagnostic_forward(
                        ids.clone(),
                        script.clone(),
                        shape.clone(),
                        flags.clone(),
                        mask.clone(),
                    )
                    .unwrap();
                assert_eq!(ordinary, diagnostic.into_data().to_vec::<f32>().unwrap());
                let next_diagnostic =
                    Tensor::<B, 1>::random([8], burn::tensor::Distribution::Default, &device)
                        .into_data()
                        .to_vec::<f32>()
                        .unwrap();
                assert_eq!(next_ordinary, next_diagnostic);
                assert_eq!(trace.len(), 13);
                assert_eq!(trace[0].layer, "projection.before_dropout");
                assert_eq!(trace[12].layer, "head");
                for layer in trace {
                    assert_eq!(layer.aggregate.real_tokens, 3);
                    assert_eq!(layer.aggregate.elements, 3 * layer.channels);
                    assert_eq!(layer.documents[0].real_tokens, 2);
                    assert_eq!(layer.documents[1].real_tokens, 1);
                }
            }
            let valid = model.valid();
            let ordinary = valid.forward(
                ids.clone().inner(),
                script.clone().inner(),
                shape.clone().inner(),
                flags.clone().inner(),
                mask.clone().inner(),
            );
            let (diagnostic, _) = valid
                .diagnostic_forward(
                    ids.clone().inner(),
                    script.clone().inner(),
                    shape.clone().inner(),
                    flags.clone().inner(),
                    mask.inner(),
                )
                .unwrap();
            assert_eq!(ordinary.into_data(), diagnostic.into_data());
            let (_, trace) = model
                .diagnostic_forward(ids, script, shape, flags, Tensor::zeros([2, 3], &device))
                .unwrap();
            for layer in trace {
                assert_eq!(layer.aggregate.real_tokens, 0);
                assert_eq!(layer.aggregate.elements, 0);
                assert_eq!(layer.aggregate.mean, 0.);
                assert_eq!(layer.aggregate.rms, 0.);
                assert_eq!(layer.aggregate.max_abs, 0.);
            }
        }
    }
    #[test]
    fn residual_rms_routes_preserve_dropout_rng_and_traced_logits() {
        if std::env::var_os("TESSERA_RESIDUAL_RMS_TEST_ISOLATED").is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "net::tests::residual_rms_routes_preserve_dropout_rng_and_traced_logits",
                    "--test-threads=1",
                ])
                .env("TESSERA_RESIDUAL_RMS_TEST_ISOLATED", "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }
        type B = Autodiff<NdArray>;
        use crate::diagnostic_operator::ForwardOperator;
        let device = Default::default();
        for dropout in [0., 0.4] {
            let cfg = TaggerNetConfig::new(16, vec![1, 2], 7)
                .with_ngram_dim(3)
                .with_script_dim(2)
                .with_shape_dim(2)
                .with_hidden(4)
                .with_dropout(dropout);
            B::seed(&device, 19);
            let model = cfg.init::<B>(&device);
            let _ = crate::quantize::extract(&model.valid(), "detector");
            let ids = Tensor::<B, 3, Int>::from_data(
                [[[1, 2], [3, 0], [15, 15]], [[4, 0], [15, 15], [15, 15]]],
                &device,
            );
            let script = Tensor::<B, 2, Int>::ones([2, 3], &device);
            let shape = script.clone();
            let flags = Tensor::<B, 3>::ones([2, 3, FLAG_BITS], &device);
            let mask = Tensor::<B, 2>::from_data([[1., 1., 0.], [1., 0., 0.]], &device);
            for seed in 42..46 {
                B::seed(&device, seed);
                let ordinary = model
                    .forward(
                        ids.clone(),
                        script.clone(),
                        shape.clone(),
                        flags.clone(),
                        mask.clone(),
                    )
                    .into_data();
                let next_standard =
                    Tensor::<B, 1>::random([8], burn::tensor::Distribution::Default, &device)
                        .into_data();
                B::seed(&device, seed);
                let routed = ForwardOperator::Standard
                    .forward(
                        &model,
                        ids.clone(),
                        script.clone(),
                        shape.clone(),
                        flags.clone(),
                        mask.clone(),
                    )
                    .unwrap()
                    .into_data();
                let next_routed =
                    Tensor::<B, 1>::random([8], burn::tensor::Distribution::Default, &device)
                        .into_data();
                assert_eq!(ordinary, routed);
                assert_eq!(next_standard, next_routed);
                B::seed(&device, seed);
                let candidate = ForwardOperator::ResidualRmsV1
                    .forward(
                        &model,
                        ids.clone(),
                        script.clone(),
                        shape.clone(),
                        flags.clone(),
                        mask.clone(),
                    )
                    .unwrap()
                    .into_data();
                let next_candidate =
                    Tensor::<B, 1>::random([8], burn::tensor::Distribution::Default, &device)
                        .into_data();
                assert_eq!(next_standard, next_candidate);
                B::seed(&device, seed);
                let (traced, layers) = model
                    .diagnostic_forward_with_operator(
                        ids.clone(),
                        script.clone(),
                        shape.clone(),
                        flags.clone(),
                        mask.clone(),
                        ForwardOperator::ResidualRmsV1,
                    )
                    .unwrap();
                assert_eq!(candidate, traced.into_data());
                let next_trace =
                    Tensor::<B, 1>::random([8], burn::tensor::Distribution::Default, &device)
                        .into_data();
                assert_eq!(next_candidate, next_trace);
                assert_eq!(layers.len(), 15);
                let normalized: Vec<_> = layers
                    .iter()
                    .filter_map(|layer| layer.normalization.as_ref())
                    .collect();
                assert_eq!(normalized.len(), 2);
                for measurement in normalized {
                    assert_eq!(measurement.real_tokens, 3);
                    assert!(measurement.max_token_channel_rms <= 1.00001);
                    assert!(
                        measurement.native_mean_square_finite
                            && measurement.native_denominator_finite
                    );
                }
            }
            let valid = model.valid();
            let a = valid
                .residual_rms_forward(
                    ids.clone().inner(),
                    script.clone().inner(),
                    shape.clone().inner(),
                    flags.clone().inner(),
                    mask.clone().inner(),
                )
                .unwrap();
            let (b, _) = valid
                .diagnostic_forward_with_operator(
                    ids.inner(),
                    script.inner(),
                    shape.inner(),
                    flags.inner(),
                    mask.inner(),
                    ForwardOperator::ResidualRmsV1,
                )
                .unwrap();
            assert_eq!(a.into_data(), b.into_data());
            let (_, empty) = model
                .diagnostic_forward_with_operator(
                    Tensor::ones([2, 3, 2], &device),
                    Tensor::ones([2, 3], &device),
                    Tensor::ones([2, 3], &device),
                    Tensor::ones([2, 3, FLAG_BITS], &device),
                    Tensor::zeros([2, 3], &device),
                    ForwardOperator::ResidualRmsV1,
                )
                .unwrap();
            for normalized in empty
                .iter()
                .filter_map(|layer| layer.normalization.as_ref())
            {
                assert_eq!(normalized.real_tokens, 0);
                assert_eq!(normalized.max_token_channel_rms, 0.);
            }
        }
    }

    #[test]
    fn residual_rms_has_no_batch_or_padding_dependence() {
        let device = Default::default();
        let cfg = TaggerNetConfig::new(16, vec![1, 2, 4, 8, 16, 1], 7)
            .with_ngram_dim(3)
            .with_script_dim(2)
            .with_shape_dim(2)
            .with_hidden(4)
            .with_dropout(0.);
        let model = cfg.init::<NdArray>(&device);
        let ids = Tensor::<NdArray, 3, Int>::ones([1, 5, 2], &device);
        let script = Tensor::<NdArray, 2, Int>::ones([1, 5], &device);
        let flags = Tensor::<NdArray, 3>::ones([1, 5, FLAG_BITS], &device);
        let alone = model
            .residual_rms_forward(
                ids.clone(),
                script.clone(),
                script.clone(),
                flags.clone(),
                Tensor::ones([1, 5], &device),
            )
            .unwrap();
        let padded_ids = Tensor::cat(vec![ids, Tensor::ones([1, 11, 2], &device)], 1);
        let padded_script = Tensor::cat(vec![script, Tensor::ones([1, 11], &device)], 1);
        let padded_flags = Tensor::cat(vec![flags, Tensor::ones([1, 11, FLAG_BITS], &device)], 1);
        let padded_mask = Tensor::cat(
            vec![
                Tensor::ones([1, 5], &device),
                Tensor::zeros([1, 11], &device),
            ],
            1,
        );
        let padded = model
            .residual_rms_forward(
                padded_ids.clone(),
                padded_script.clone(),
                padded_script.clone(),
                padded_flags.clone(),
                padded_mask.clone(),
            )
            .unwrap()
            .slice([0..1, 0..5]);
        let together = model
            .residual_rms_forward(
                Tensor::cat(vec![padded_ids, Tensor::ones([1, 16, 2], &device)], 0),
                Tensor::cat(
                    vec![padded_script.clone(), Tensor::ones([1, 16], &device)],
                    0,
                ),
                Tensor::cat(vec![padded_script, Tensor::ones([1, 16], &device)], 0),
                Tensor::cat(
                    vec![padded_flags, Tensor::ones([1, 16, FLAG_BITS], &device)],
                    0,
                ),
                Tensor::cat(vec![padded_mask, Tensor::ones([1, 16], &device)], 0),
            )
            .unwrap()
            .slice([0..1, 0..5]);
        assert!((alone.clone() - padded).abs().max().into_scalar() < 1e-4);
        assert!((alone - together).abs().max().into_scalar() < 1e-4);
    }
}
