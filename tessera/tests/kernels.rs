//! The simd128 dense kernel must equal the scalar one bit for bit, for every output width
//! including those that are not multiples of four or sixteen. Natively both calls are the scalar
//! path; the test earns its keep in a browser build with simd128.

#![cfg(feature = "profile")]

#[cfg(target_arch = "wasm32")]
use wasm_bindgen_test::wasm_bindgen_test as test;

use tessera::kernel_probe::{accumulate, accumulate_scalar};

#[cfg(target_arch = "wasm32")]
wasm_bindgen_test::wasm_bindgen_test_configure!(run_in_browser);

struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0
    }

    /// An int8 weight widened to f32, as `Dense` stores it.
    fn weight(&mut self) -> f32 {
        f32::from((self.next() >> 56) as i8)
    }
}

#[test]
fn simd_matches_scalar_bit_for_bit() {
    let mut rng = XorShift(0x9E37_79B9_7F4A_7C15);
    for output in 0..=70usize {
        for input in [0usize, 1, 3, 17, 64] {
            let w: Vec<f32> = (0..input * output).map(|_| rng.weight()).collect();
            let x: Vec<f32> = (0..input).map(|_| rng.f32()).collect();
            let start: Vec<f32> = (0..output).map(|_| rng.f32()).collect();
            let (mut a, mut b) = (start.clone(), start);
            accumulate(&w, &x, &mut a);
            accumulate_scalar(&w, &x, &mut b);
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&a), bits(&b), "output {output}, input {input}");
        }
    }
}

#[test]
fn scalar_matches_a_naive_sum() {
    let mut rng = XorShift(7);
    let (input, output) = (33, 21);
    let w: Vec<f32> = (0..input * output).map(|_| rng.weight()).collect();
    let x: Vec<f32> = (0..input).map(|_| rng.f32()).collect();
    let mut y = vec![0.0; output];
    accumulate_scalar(&w, &x, &mut y);
    for (o, got) in y.iter().enumerate() {
        let naive: f64 = (0..input)
            .map(|i| f64::from(w[i * output + o]) * f64::from(x[i]))
            .sum();
        assert!(
            (naive - f64::from(*got)).abs() <= 1e-3 * naive.abs().max(1.0),
            "output {o}"
        );
    }
}
