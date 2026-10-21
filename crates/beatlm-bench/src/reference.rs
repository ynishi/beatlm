//! CPU reference implementations of the kernel IR's operations, and an
//! interpreter that runs a whole graph with them.
//!
//! They define what each generated kernel must compute, in the layouts
//! `beatlm_core::kernel` documents, and accumulate in `f64` so the reference is
//! the more accurate side of a comparison. Each operation's result is rounded
//! to `f32` before the next reads it, as an unfused program stores it; a fused
//! kernel keeps the value in a register instead, which the tolerance absorbs.
//! The GPU tests compare against them within [`TOLERANCE_ABS`] plus
//! [`TOLERANCE_REL`] of the reference value.

use std::collections::BTreeMap;

use beatlm_core::kernel::{Graph, LAYER_NORM_EPSILON, OpKind};
use beatlm_core::memory::TensorId;

/// Absolute part of the comparison tolerance.
///
/// The kernels accumulate in `f32` in a different order than the reference, so
/// results differ by rounding; sums here are of at most a few thousand terms of
/// magnitude near one, whose `f32` rounding error is far below this.
pub const TOLERANCE_ABS: f32 = 1e-4;
/// Relative part of the comparison tolerance.
pub const TOLERANCE_REL: f32 = 1e-4;

/// 3x3 convolution, stride 1, zero padding of 1, plus `bias` per output
/// channel when given.
///
/// `input` is `[h][w][cin]`, `weights` is `[3][3][cin][cout]` and the result is
/// `[h][w][cout]`.
pub fn conv3x3(
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    (h, w, cin, cout): (usize, usize, usize, usize),
) -> Vec<f32> {
    let mut out = vec![0.0; h * w * cout];
    for y in 0..h {
        for x in 0..w {
            for co in 0..cout {
                let mut acc = bias.map_or(0.0, |b| f64::from(b[co]));
                for ky in 0..3 {
                    for kx in 0..3 {
                        // The tap reads (y + ky - 1, x + kx - 1); outside the
                        // image it reads zero and adds nothing.
                        let (Some(iy), Some(ix)) =
                            ((y + ky).checked_sub(1), (x + kx).checked_sub(1))
                        else {
                            continue;
                        };
                        if iy >= h || ix >= w {
                            continue;
                        }
                        for ci in 0..cin {
                            let a = input[(iy * w + ix) * cin + ci];
                            let b = weights[((ky * 3 + kx) * cin + ci) * cout + co];
                            acc += f64::from(a) * f64::from(b);
                        }
                    }
                }
                out[(y * w + x) * cout + co] = acc as f32;
            }
        }
    }
    out
}

/// Layer normalisation over the `c` channels of each of `rows` rows, scaled by
/// `gamma` and shifted by `beta`.
pub fn layernorm(input: &[f32], gamma: &[f32], beta: &[f32], rows: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows * c];
    for p in 0..rows {
        let row = &input[p * c..(p + 1) * c];
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / c as f64;
        let var = row
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / c as f64;
        let inv = 1.0 / (var + f64::from(LAYER_NORM_EPSILON)).sqrt();
        for (i, &v) in row.iter().enumerate() {
            let y = (f64::from(v) - mean) * inv * f64::from(gamma[i]) + f64::from(beta[i]);
            out[p * c + i] = y as f32;
        }
    }
    out
}

/// `C[m][n] = A[m][k] * B[k][n] + bias[n]`, all row-major.
pub fn matmul(
    a: &[f32],
    b: &[f32],
    bias: Option<&[f32]>,
    (m, k, n): (usize, usize, usize),
) -> Vec<f32> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = bias.map_or(0.0, |b| f64::from(b[j]));
            for l in 0..k {
                acc += f64::from(a[i * k + l]) * f64::from(b[l * n + j]);
            }
            out[i * n + j] = acc as f32;
        }
    }
    out
}

/// `a + b`, element by element.
pub fn add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| (f64::from(x) + f64::from(y)) as f32)
        .collect()
}

/// `max(x, 0)`.
pub fn relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.max(0.0)).collect()
}

/// GELU, tanh approximation: `0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3)))`.
pub fn gelu(x: &[f32]) -> Vec<f32> {
    let k = (2.0 / std::f64::consts::PI).sqrt();
    x.iter()
        .map(|&v| {
            let v = f64::from(v);
            (0.5 * v * (1.0 + (k * (v + 0.044_715 * v * v * v)).tanh())) as f32
        })
        .collect()
}

/// `x * factor`.
pub fn scale(x: &[f32], factor: f32) -> Vec<f32> {
    x.iter()
        .map(|&v| (f64::from(v) * f64::from(factor)) as f32)
        .collect()
}

/// Runs every operation of `graph` in program order, reading inputs and
/// weights from `values` and inserting each operation's result under its
/// output tensor.
///
/// Panics when `values` lacks a tensor an operation reads: a test that forgot
/// an input.
pub fn run(graph: &Graph, values: &mut BTreeMap<TensorId, Vec<f32>>) {
    let n = |v: u32| v as usize;
    for op in graph.ops() {
        let get = |id: TensorId| values[&id].as_slice();
        let result = match op.kind {
            OpKind::Conv2d3x3(c) => conv3x3(
                get(c.input),
                get(c.weights),
                c.bias.map(get),
                (n(c.height), n(c.width), n(c.cin), n(c.cout)),
            ),
            OpKind::LayerNorm(l) => layernorm(
                get(l.input),
                get(l.gamma),
                get(l.beta),
                n(l.rows),
                n(l.channels),
            ),
            OpKind::MatMul(mm) => matmul(
                get(mm.a),
                get(mm.b),
                mm.bias.map(get),
                (n(mm.m), n(mm.k), n(mm.n)),
            ),
            OpKind::Add { a, b, .. } => add(get(a), get(b)),
            OpKind::Relu { input, .. } => relu(get(input)),
            OpKind::Gelu { input, .. } => gelu(get(input)),
            OpKind::Scale { input, factor, .. } => scale(get(input), factor),
        };
        values.insert(op.output, result);
    }
}

/// Whether `got` is within the tolerance of `want`; false for a NaN on either
/// side.
fn within_tolerance(got: f32, want: f32) -> bool {
    (got - want).abs() <= TOLERANCE_ABS + TOLERANCE_REL * want.abs()
}

/// The first element of `got` that is outside the tolerance of `want`, as
/// `(index, got, want)`, or `None` when all agree.
pub fn first_mismatch(got: &[f32], want: &[f32]) -> Option<(usize, f32, f32)> {
    if got.len() != want.len() {
        return Some((got.len().min(want.len()), f32::NAN, f32::NAN));
    }
    got.iter()
        .zip(want)
        .enumerate()
        .find(|&(_, (&g, &w))| !within_tolerance(g, w))
        .map(|(i, (&g, &w))| (i, g, w))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use beatlm_core::kernel::{Fusion, TensorRole};

    #[test]
    fn a_one_hot_kernel_copies_the_input_through_the_centre_tap() {
        // Weights that pick the centre tap and the same channel are the identity.
        let (h, w, c) = (3, 4, 2);
        let mut weights = vec![0.0; 9 * c * c];
        for ci in 0..c {
            weights[((4) * c + ci) * c + ci] = 1.0;
        }
        let input: Vec<f32> = (0..h * w * c).map(|i| i as f32).collect();
        assert_eq!(conv3x3(&input, &weights, None, (h, w, c, c)), input);
    }

    #[test]
    fn a_corner_pixel_sums_only_the_taps_inside_the_image() {
        // All-ones 1-channel 2x2 image and all-ones kernel: the corner sees a
        // 2x2 neighbourhood, i.e. four taps.
        let out = conv3x3(&[1.0; 4], &[1.0; 9], None, (2, 2, 1, 1));
        assert_eq!(out, vec![4.0; 4]);
        let out = conv3x3(&[1.0; 4], &[1.0; 9], Some(&[0.5]), (2, 2, 1, 1));
        assert_eq!(out, vec![4.5; 4]);
    }

    #[test]
    fn layernorm_gives_zero_mean_unit_variance_before_gamma_and_beta() {
        let c = 4;
        let out = layernorm(&[1.0, 2.0, 3.0, 4.0], &[1.0; 4], &[0.0; 4], 1, c);
        let mean: f32 = out.iter().sum::<f32>() / c as f32;
        let var: f32 = out.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / c as f32;
        assert!(mean.abs() < 1e-6);
        assert!((var - 1.0).abs() < 1e-3);
    }

    #[test]
    fn matmul_multiplies_and_adds_the_bias_per_column() {
        let (a, b) = ([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]);
        assert_eq!(
            matmul(&a, &b, None, (2, 2, 2)),
            vec![19.0, 22.0, 43.0, 50.0]
        );
        assert_eq!(
            matmul(&a, &b, Some(&[1.0, -1.0]), (2, 2, 2)),
            vec![20.0, 21.0, 44.0, 49.0]
        );
    }

    #[test]
    fn element_wise_operations_compute_what_they_say() {
        assert_eq!(add(&[1.0, -2.0], &[0.5, 2.0]), vec![1.5, 0.0]);
        assert_eq!(relu(&[-1.0, 0.0, 3.0]), vec![0.0, 0.0, 3.0]);
        assert_eq!(scale(&[2.0, -4.0], 0.5), vec![1.0, -2.0]);
        let g = gelu(&[0.0, 1.0, -1.0, 10.0, -10.0]);
        assert_eq!(g[0], 0.0);
        // The tanh approximation's published values at 1 and -1.
        assert!((g[1] - 0.841_192).abs() < 1e-5, "{}", g[1]);
        assert!((g[2] + 0.158_808).abs() < 1e-5, "{}", g[2]);
        assert!((g[3] - 10.0).abs() < 1e-6 && g[4].abs() < 1e-6);
    }

    #[test]
    fn the_interpreter_runs_a_graph_in_program_order() {
        let (x, y, z) = (TensorId::new(0), TensorId::new(1), TensorId::new(2));
        let mut g = Graph::new();
        g.tensor(x, TensorRole::Input, 2).unwrap();
        g.tensor(y, TensorRole::Activation, 2).unwrap();
        g.tensor(z, TensorRole::Output, 2).unwrap();
        g.push(
            OpKind::Relu {
                input: x,
                elements: 2,
            },
            y,
        )
        .unwrap();
        g.push(
            OpKind::Add {
                a: y,
                b: x,
                elements: 2,
            },
            z,
        )
        .unwrap();
        assert!(g.lower(Fusion::On).is_ok());
        let mut values = BTreeMap::from([(x, vec![-1.0, 2.0])]);
        run(&g, &mut values);
        assert_eq!(values[&z], vec![-1.0, 4.0]);
    }

    #[test]
    fn mismatches_are_found_and_lengths_compared() {
        assert_eq!(first_mismatch(&[1.0, 2.0], &[1.0, 2.0]), None);
        assert_eq!(
            first_mismatch(&[1.0, 2.5], &[1.0, 2.0]),
            Some((1, 2.5, 2.0))
        );
        assert!(first_mismatch(&[1.0], &[1.0, 2.0]).is_some());
        assert!(first_mismatch(&[f32::NAN], &[1.0]).is_some());
    }
}
