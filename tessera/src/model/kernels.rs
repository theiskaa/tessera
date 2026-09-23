//! Scalar kernels over int8 weights and f32 activations. Activations are token-major
//! `[len, channels]`. Formulas match the trainer's dequantized model up to floating-point
//! summation order: the per-channel scale is applied once, after the integer-weight dot product.
//!
//! Every tensor's dtype and shape is validated once in `Tagger::load`; the kernels trust it.

use super::weights::QTensor;

/// Mean of the rows `ids` of an int8 `[rows, dim]` table into `out[dim]`. Ids are raw n-gram
/// ids; row 0 of the table is the trainer's padding row, so id `i` reads row `i + 1`. Empty
/// ids give zeros. The featurizer hashes into the manifest's bucket count and the table has
/// one more row than that, so every id has a row; one that did not would count as zeros.
pub(crate) fn embedding_mean(table: &QTensor, ids: &[u32], out: &mut [f32]) {
    let dim = table.shape[1];
    debug_assert_eq!(out.len(), dim);
    out.fill(0.0);
    for &id in ids {
        let row = (id as usize + 1) * dim;
        if let Some(row) = table.data.get(row..row + dim) {
            for (o, &q) in out.iter_mut().zip(row) {
                *o += q as f32;
            }
        }
    }
    let n = ids.len().max(1) as f32;
    for (o, &s) in out.iter_mut().zip(&table.scales) {
        *o = *o / n * s;
    }
}

/// Row `id` of an int8 `[rows, dim]` table, dequantized into `out[dim]`; zeros past the table.
pub(crate) fn embedding_row(table: &QTensor, id: usize, out: &mut [f32]) {
    let dim = table.shape[1];
    debug_assert_eq!(out.len(), dim);
    out.fill(0.0);
    if let Some(row) = table.data.get(id * dim..(id + 1) * dim) {
        for ((o, &q), &s) in out.iter_mut().zip(row).zip(&table.scales) {
            *o = q as f32 * s;
        }
    }
}

/// A linear or convolution layer: int8 weights `[out, in]` or `[out, in, taps]` with one scale
/// per output, laid out once at load as f32 `[taps, in, out]`. The kernels then add one input's
/// contribution to every output at a time over contiguous memory, a loop of independent sums
/// that the compiler vectorizes; the scale is still applied once, after the sum.
#[derive(Debug)]
pub(crate) struct Dense {
    w: Vec<f32>,
    scales: Vec<f32>,
    bias: Vec<f32>,
    taps: usize,
    input: usize,
    output: usize,
}

impl Dense {
    pub(crate) fn new(q: &QTensor, bias: Vec<f32>) -> Dense {
        let (output, input) = (q.shape[0], q.shape[1]);
        let taps = q.shape.get(2).copied().unwrap_or(1);
        let mut w = vec![0f32; taps * input * output];
        for o in 0..output {
            for i in 0..input {
                for k in 0..taps {
                    w[(k * input + i) * output + o] = f32::from(q.data[(o * input + i) * taps + k]);
                }
            }
        }
        Dense {
            w,
            scales: q.scales.clone(),
            bias,
            taps,
            input,
            output,
        }
    }

    /// Adds `sum_i w[tap, i, o] * x[i]` to `y[o]` for every output.
    fn accumulate(&self, tap: usize, x: &[f32], y: &mut [f32]) {
        let weights = &self.w[tap * self.input * self.output..(tap + 1) * self.input * self.output];
        for (&v, row) in x.iter().zip(weights.chunks_exact(self.output)) {
            for (acc, &w) in y.iter_mut().zip(row) {
                *acc += w * v;
            }
        }
    }

    fn finish(&self, y: &mut [f32]) {
        for ((acc, &s), &b) in y.iter_mut().zip(&self.scales).zip(&self.bias) {
            *acc = b + s * *acc;
        }
    }
}

/// `y[t, o] = b[o] + scale[o] * sum_i q[o, i] * x[t, i]`.
pub(crate) fn linear(x: &[f32], len: usize, layer: &Dense, out: &mut [f32]) {
    debug_assert_eq!(x.len(), len * layer.input);
    debug_assert_eq!(out.len(), len * layer.output);
    for (xr, yr) in x
        .chunks_exact(layer.input)
        .zip(out.chunks_exact_mut(layer.output))
    {
        yr.fill(0.0);
        layer.accumulate(0, xr, yr);
        layer.finish(yr);
    }
}

/// Same-length dilated convolution with zero padding `dilation * (taps - 1) / 2` on each side,
/// which is what the trainer's `Conv1d` uses:
/// `y[t, o] = b[o] + scale[o] * sum_k sum_i q[o, i, k] * x[t + (k - (taps - 1) / 2) * d, i]`.
pub(crate) fn conv1d_same(x: &[f32], len: usize, layer: &Dense, dilation: usize, out: &mut [f32]) {
    debug_assert_eq!(x.len(), len * layer.input);
    debug_assert_eq!(out.len(), len * layer.output);
    let half = (layer.taps - 1) / 2;
    for (t, yr) in out.chunks_exact_mut(layer.output).enumerate() {
        yr.fill(0.0);
        for k in 0..layer.taps {
            let Some(u) = (t + k * dilation).checked_sub(half * dilation) else {
                continue;
            };
            if u < len {
                layer.accumulate(k, &x[u * layer.input..(u + 1) * layer.input], yr);
            }
        }
        layer.finish(yr);
    }
}

pub(crate) fn relu_inplace(x: &mut [f32]) {
    for v in x {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

pub(crate) fn add_inplace(x: &mut [f32], y: &[f32]) {
    for (a, b) in x.iter_mut().zip(y) {
        *a += *b;
    }
}

/// Row-wise softmax over `[len, classes]`, in place. Subtracting the row maximum keeps large
/// logits from overflowing.
pub(crate) fn softmax_rows(x: &mut [f32], classes: usize) {
    for row in x.chunks_mut(classes) {
        let max = row.iter().copied().fold(f32::MIN, f32::max);
        let mut sum = 0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct XorShift(u64);

    impl XorShift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }

        fn f32(&mut self) -> f32 {
            (self.next() % 2001) as f32 / 1000.0 - 1.0
        }

        fn i8(&mut self) -> i8 {
            (self.next() % 255) as i8
        }
    }

    fn int8_tensor(rng: &mut XorShift, shape: &[usize], channel_axis: usize) -> QTensor {
        let n = shape.iter().product();
        let data = (0..n).map(|_| rng.i8()).collect();
        let scales = (0..shape[channel_axis])
            .map(|_| 0.001 + rng.f32().abs() * 0.01)
            .collect();
        QTensor {
            shape: shape.to_vec(),
            data,
            scales,
        }
    }

    fn f32_tensor(rng: &mut XorShift, n: usize) -> Vec<f32> {
        (0..n).map(|_| rng.f32()).collect()
    }

    /// Dequantizes a `[out, ...]` tensor quantized along axis 0.
    fn dequant_rows(t: &QTensor) -> Vec<f32> {
        let per = t.data.len() / t.scales.len();
        t.data
            .iter()
            .enumerate()
            .map(|(j, &q)| q as f32 * t.scales[j / per])
            .collect()
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-4 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn linear_matches_dequantized_reference() {
        let mut rng = XorShift(0x9e37_79b9_7f4a_7c15);
        let (len, in_dim, out_dim) = (7, 5, 4);
        let w = int8_tensor(&mut rng, &[out_dim, in_dim], 0);
        let b = f32_tensor(&mut rng, out_dim);
        let x: Vec<f32> = (0..len * in_dim).map(|_| rng.f32()).collect();
        let mut y = vec![0f32; len * out_dim];
        linear(&x, len, &Dense::new(&w, b.clone()), &mut y);
        let wf = dequant_rows(&w);
        let bf = &b;
        for t in 0..len {
            for o in 0..out_dim {
                let mut want = bf[o];
                for i in 0..in_dim {
                    want += wf[o * in_dim + i] * x[t * in_dim + i];
                }
                assert!(close(y[t * out_dim + o], want), "t={t} o={o}");
            }
        }
    }

    #[test]
    fn conv_matches_dequantized_reference() {
        let mut rng = XorShift(0x2545_f491_4f6c_dd1d);
        let (len, c, k) = (7, 8, 3);
        for d in [1usize, 2, 4] {
            let w = int8_tensor(&mut rng, &[c, c, k], 0);
            let b = f32_tensor(&mut rng, c);
            let x: Vec<f32> = (0..len * c).map(|_| rng.f32()).collect();
            let mut y = vec![0f32; len * c];
            conv1d_same(&x, len, &Dense::new(&w, b.clone()), d, &mut y);
            let wf = dequant_rows(&w);
            let bf = &b;
            // Padded formulation, as Burn computes it: x_padded[t + k * d] with d zeros each side.
            let mut padded = vec![0f32; (len + 2 * d) * c];
            padded[d * c..(d + len) * c].copy_from_slice(&x);
            for t in 0..len {
                for o in 0..c {
                    let mut want = bf[o];
                    for i in 0..c {
                        for kk in 0..k {
                            want += wf[(o * c + i) * k + kk] * padded[(t + kk * d) * c + i];
                        }
                    }
                    assert!(close(y[t * c + o], want), "d={d} t={t} o={o}");
                }
            }
        }
    }

    #[test]
    fn conv_taps_outside_the_sequence_are_zero() {
        let (len, c, d) = (3, 1, 2);
        let w = QTensor {
            shape: vec![1, 1, 3],
            data: vec![1, 10, 100],
            scales: vec![1.0],
        };
        let x = [1.0, 2.0, 3.0];
        let mut y = vec![0f32; len * c];
        conv1d_same(&x, len, &Dense::new(&w, vec![0.0]), d, &mut y);
        assert_eq!(y[0], 10.0 * 1.0 + 100.0 * 3.0);
        assert_eq!(y[1], 10.0 * 2.0);
        assert_eq!(y[2], 1.0 * 1.0 + 10.0 * 3.0);
    }

    #[test]
    fn embedding_mean_reads_shifted_rows() {
        let mut rng = XorShift(7);
        let table = int8_tensor(&mut rng, &[6, 4], 1);
        let mut mean = vec![0f32; 4];
        let mut row = vec![0f32; 4];
        embedding_mean(&table, &[3, 3], &mut mean);
        embedding_row(&table, 4, &mut row);
        for (a, b) in mean.iter().zip(&row) {
            assert!(close(*a, *b));
        }
        embedding_mean(&table, &[], &mut mean);
        assert!(mean.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn ids_past_the_table_read_as_zeros() {
        let table = QTensor {
            shape: vec![2, 2],
            data: vec![1, 2, 3, 4],
            scales: vec![1.0, 1.0],
        };
        let mut out = vec![9f32; 2];
        embedding_row(&table, 5, &mut out);
        assert_eq!(out, [0.0, 0.0]);
        embedding_mean(&table, &[0, 7], &mut out);
        assert_eq!(out, [1.5, 2.0]);
    }

    #[test]
    fn softmax_rows_sum_to_one_without_overflow() {
        let mut x = vec![1.0, 2.0, 3.0, 1000.0, 1001.0, 999.0];
        softmax_rows(&mut x, 3);
        for row in x.chunks(3) {
            assert!(row.iter().all(|v| v.is_finite()));
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
    }
}
