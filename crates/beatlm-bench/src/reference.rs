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

use beatlm_core::kernel::{Graph, OpKind};
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

/// Layer normalisation over the `c` channels of each of `rows` rows, with
/// `epsilon` added to the variance, scaled by `gamma` and shifted by `beta`.
pub fn layernorm(
    input: &[f32],
    gamma: &[f32],
    beta: &[f32],
    (rows, c): (usize, usize),
    epsilon: f32,
) -> Vec<f32> {
    let mut out = vec![0.0; rows * c];
    for p in 0..rows {
        let row = &input[p * c..(p + 1) * c];
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / c as f64;
        let var = row
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / c as f64;
        let inv = 1.0 / (var + f64::from(epsilon)).sqrt();
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
    // A row of accumulators, so `b` is read along its rows; each element is
    // still the sum over `l` in increasing order.
    let mut acc = vec![0.0_f64; n];
    for i in 0..m {
        for (j, a) in acc.iter_mut().enumerate() {
            *a = bias.map_or(0.0, |b| f64::from(b[j]));
        }
        for l in 0..k {
            let x = f64::from(a[i * k + l]);
            for (acc, &w) in acc.iter_mut().zip(&b[l * n..(l + 1) * n]) {
                *acc += x * f64::from(w);
            }
        }
        for (o, &v) in out[i * n..(i + 1) * n].iter_mut().zip(&acc) {
            *o = v as f32;
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

/// The error function, to about `1e-13`: its Maclaurin series below 3, where
/// the terms stay below `e^9` and so lose at most four of `f64`'s sixteen
/// digits, and the continued fraction of `erfc` from 3, which converges fast
/// there. Six orders of magnitude below the approximation it checks. `std`
/// has no `erf`, and the reference must not be the approximation the kernels
/// use.
pub fn erf(x: f64) -> f64 {
    let z = x.abs();
    let value = if z < 3.0 {
        // erf z = 2 / sqrt(pi) * sum (-1)^n z^(2n+1) / (n! (2n + 1)).
        let mut term = z;
        let mut sum = z;
        let mut n = 0.0;
        while term.abs() > 1e-17 * sum.abs() {
            n += 1.0;
            term *= -z * z / n;
            sum += term / (2.0 * n + 1.0);
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    } else {
        // erfc z = exp(-z^2) / sqrt(pi) / (z + (1/2) / (z + 1 / (z + (3/2) / (z + ...)))),
        // evaluated from the tail; 60 levels are far more than z >= 3 needs.
        let mut tail = z;
        for k in (1..=60).rev() {
            tail = z + f64::from(k) / 2.0 / tail;
        }
        1.0 - (-z * z).exp() / std::f64::consts::PI.sqrt() / tail
    };
    value.copysign(x)
}

/// The grid the erf approximation's error is stated over (the core's
/// `GELU_ERF_P`): `[-10, 10]` in steps of `2^-12`, every point exact in `f32`.
pub fn erf_error_grid() -> Vec<f32> {
    (0..=81_920).map(|i| -10.0 + i as f32 / 4096.0).collect()
}

/// GELU as defined, `0.5 x (1 + erf(x / sqrt 2))`, through [`erf`].
pub fn gelu_erf(x: &[f32]) -> Vec<f32> {
    x.iter()
        .map(|&v| {
            let v = f64::from(v);
            (0.5 * v * (1.0 + erf(v / std::f64::consts::SQRT_2))) as f32
        })
        .collect()
}

/// Element `i` times `scale[i % channels]`.
pub fn channel_scale(x: &[f32], scale: &[f32]) -> Vec<f32> {
    x.iter()
        .zip(scale.iter().cycle())
        .map(|(&v, &s)| (f64::from(v) * f64::from(s)) as f32)
        .collect()
}

/// A patch embedding: `image` is `[gh * p][gw * p][cin]`, `weights`
/// `[p][p][cin][cout]`, the result `[gh * gw][cout]`.
pub fn patch_embed(
    image: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    (gh, gw, p, cin, cout): (usize, usize, usize, usize, usize),
) -> Vec<f32> {
    let width = gw * p;
    let mut out = vec![0.0; gh * gw * cout];
    let mut acc = vec![0.0_f64; cout];
    for gy in 0..gh {
        for gx in 0..gw {
            for (co, a) in acc.iter_mut().enumerate() {
                *a = bias.map_or(0.0, |b| f64::from(b[co]));
            }
            for ky in 0..p {
                for kx in 0..p {
                    let pixel = ((gy * p + ky) * width + gx * p + kx) * cin;
                    for ci in 0..cin {
                        let x = f64::from(image[pixel + ci]);
                        let row = &weights[((ky * p + kx) * cin + ci) * cout..][..cout];
                        for (a, &w) in acc.iter_mut().zip(row) {
                            *a += x * f64::from(w);
                        }
                    }
                }
            }
            for (o, &a) in out[(gy * gw + gx) * cout..][..cout].iter_mut().zip(&acc) {
                *o = a as f32;
            }
        }
    }
    out
}

/// `token` in front of `tokens`.
pub fn prepend_token(token: &[f32], tokens: &[f32]) -> Vec<f32> {
    token.iter().chain(tokens).copied().collect()
}

/// `input` without its first `channels` elements: the first token dropped.
pub fn tokens_to_map(input: &[f32], channels: usize) -> Vec<f32> {
    input[channels..].to_vec()
}

/// The shape of multi-head attention: tokens, heads, head width.
pub type AttentionShape = (usize, usize, usize);

/// `S[i][h][j] = Q[i][h] . K[j][h]` from `qkv`, `[t][3][h][d]`; the result is
/// `[t][h][t]`.
pub fn attention_scores(qkv: &[f32], (t, h, d): AttentionShape) -> Vec<f32> {
    let row = 3 * h * d;
    let mut out = vec![0.0; t * h * t];
    for i in 0..t {
        for head in 0..h {
            let q = &qkv[i * row + head * d..][..d];
            for j in 0..t {
                let k = &qkv[j * row + h * d + head * d..][..d];
                let dot: f64 = q
                    .iter()
                    .zip(k)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                out[(i * h + head) * t + j] = dot as f32;
            }
        }
    }
    out
}

/// Softmax over the `cols` elements of each row, the row's maximum subtracted
/// first.
pub fn softmax(x: &[f32], cols: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(x.len());
    for row in x.chunks_exact(cols) {
        let max = row
            .iter()
            .fold(f64::NEG_INFINITY, |m, &v| m.max(f64::from(v)));
        let exps: Vec<f64> = row.iter().map(|&v| (f64::from(v) - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        out.extend(exps.iter().map(|e| (e / sum) as f32));
    }
    out
}

/// `O[i][h] = sum over j of P[i][h][j] V[j][h]` from `probs`, `[t][h][t]`, and
/// `qkv`, `[t][3][h][d]`; the result is `[t][h * d]`.
pub fn attention_values(probs: &[f32], qkv: &[f32], (t, h, d): AttentionShape) -> Vec<f32> {
    let row = 3 * h * d;
    let mut out = vec![0.0; t * h * d];
    let mut acc = vec![0.0_f64; d];
    for i in 0..t {
        for head in 0..h {
            acc.iter_mut().for_each(|a| *a = 0.0);
            let p = &probs[(i * h + head) * t..][..t];
            for (j, &w) in p.iter().enumerate() {
                let v = &qkv[j * row + 2 * h * d + head * d..][..d];
                for (a, &x) in acc.iter_mut().zip(v) {
                    *a += f64::from(w) * f64::from(x);
                }
            }
            for (o, &a) in out[i * h * d + head * d..][..d].iter_mut().zip(&acc) {
                *o = a as f32;
            }
        }
    }
    out
}

/// The weights of one pre-norm vision transformer block, in the IR's layouts:
/// linear weights `[in][out]`, the queries', keys' and values' side by side.
pub struct BlockWeights<'a> {
    /// The layer norm before attention, gamma and beta.
    pub norm1: (&'a [f32], &'a [f32]),
    /// `[width][3 * width]` and `[3 * width]`.
    pub qkv: (&'a [f32], &'a [f32]),
    /// The output projection, `[width][width]` and `[width]`.
    pub projection: (&'a [f32], &'a [f32]),
    /// LayerScale after attention.
    pub layer_scale1: &'a [f32],
    /// The layer norm before the MLP.
    pub norm2: (&'a [f32], &'a [f32]),
    /// `[width][mlp]` and `[mlp]`.
    pub fc1: (&'a [f32], &'a [f32]),
    /// `[mlp][width]` and `[width]`.
    pub fc2: (&'a [f32], &'a [f32]),
    /// LayerScale after the MLP.
    pub layer_scale2: &'a [f32],
}

/// One pre-norm block of DINOv2, written as its forward pass, from the
/// operations above and not from a graph:
/// `x1 = x + ls1 * proj(attention(norm1(x)))`, `out = x1 + ls2 * fc2(gelu(fc1(norm2(x1))))`.
pub fn vit_block(
    x: &[f32],
    w: &BlockWeights<'_>,
    (t, h, d): AttentionShape,
    mlp: usize,
    epsilon: f32,
) -> Vec<f32> {
    let width = h * d;
    let n1 = layernorm(x, w.norm1.0, w.norm1.1, (t, width), epsilon);
    let qkv = matmul(&n1, w.qkv.0, Some(w.qkv.1), (t, width, 3 * width));
    let scores = scale(&attention_scores(&qkv, (t, h, d)), 1.0 / (d as f32).sqrt());
    let probs = softmax(&scores, t);
    let values = attention_values(&probs, &qkv, (t, h, d));
    let projected = matmul(
        &values,
        w.projection.0,
        Some(w.projection.1),
        (t, width, width),
    );
    let x1 = add(x, &channel_scale(&projected, w.layer_scale1));
    let n2 = layernorm(&x1, w.norm2.0, w.norm2.1, (t, width), epsilon);
    let hidden = gelu_erf(&matmul(&n2, w.fc1.0, Some(w.fc1.1), (t, width, mlp)));
    let out = matmul(&hidden, w.fc2.0, Some(w.fc2.1), (t, mlp, width));
    add(&x1, &channel_scale(&out, w.layer_scale2))
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
                (n(l.rows), n(l.channels)),
                l.epsilon,
            ),
            OpKind::PatchEmbed(p) => patch_embed(
                get(p.input),
                get(p.weights),
                p.bias.map(get),
                (n(p.grid_h), n(p.grid_w), n(p.patch), n(p.cin), n(p.cout)),
            ),
            OpKind::PrependToken(t) => prepend_token(get(t.token), get(t.tokens)),
            OpKind::TokensToMap(t) => tokens_to_map(get(t.input), n(t.channels)),
            OpKind::AttentionScores(a) => {
                attention_scores(get(a.qkv), (n(a.tokens), n(a.heads), n(a.head_dim)))
            }
            OpKind::Softmax(s) => softmax(get(s.input), n(s.cols)),
            OpKind::AttentionValues(a) => attention_values(
                get(a.probs),
                get(a.qkv),
                (n(a.tokens), n(a.heads), n(a.head_dim)),
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
            OpKind::GeluErf { input, .. } => gelu_erf(get(input)),
            OpKind::Scale { input, factor, .. } => scale(get(input), factor),
            OpKind::ChannelScale { input, scale, .. } => channel_scale(get(input), get(scale)),
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
        let out = layernorm(&[1.0, 2.0, 3.0, 4.0], &[1.0; 4], &[0.0; 4], (1, c), 1e-5);
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
    fn erf_has_its_tabulated_values_on_both_sides_of_the_switch() {
        // Abramowitz and Stegun, table 7.1, and erf(-x) = -erf(x).
        for (x, want) in [
            (0.0, 0.0),
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (2.0, 0.995_322_265_018_952_7),
            // erf(3) less erf'(3) = 2 exp(-9) / sqrt(pi) times 1e-6.
            (2.999_999, 0.999_977_909_363_748),
            (3.0, 0.999_977_909_503_001_4),
            (4.0, 0.999_999_984_582_742_1),
            (-1.0, -0.842_700_792_949_714_9),
        ] {
            assert!((erf(x) - want).abs() < 1e-12, "erf({x}) = {}", erf(x));
        }
        assert_eq!(erf(30.0), 1.0);
    }

    /// GELU through erf as a generated kernel evaluates it, in `f32`, with the
    /// core's constants: the WGSL of `codegen`, written in Rust.
    fn gelu_erf_as_kernels_compute(x: f32) -> f32 {
        use beatlm_core::kernel::{GELU_ERF_A, GELU_ERF_P};
        let [a1, a2, a3, a4, a5] = GELU_ERF_A;
        let z = x.abs() * std::f32::consts::FRAC_1_SQRT_2;
        let t = 1.0 / (1.0 + GELU_ERF_P * z);
        let poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))));
        let q = 0.5 * poly * (-z * z).exp();
        x * if x >= 0.0 { 1.0 - q } else { q }
    }

    #[test]
    fn the_erf_approximation_is_within_its_stated_error() {
        use beatlm_core::kernel::GELU_ERF_MAX_ERROR;
        let grid = erf_error_grid();
        let exact = |x: f32| {
            let x = f64::from(x);
            0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2))
        };
        let (worst, at) = grid
            .iter()
            .map(|&x| {
                (
                    (f64::from(gelu_erf_as_kernels_compute(x)) - exact(x)).abs(),
                    x,
                )
            })
            .fold((0.0, 0.0), |a, b| if b.0 > a.0 { b } else { a });
        println!("GELU (erf) in f32 on the CPU: max abs error {worst:e} at x = {at}");
        assert!(worst <= f64::from(GELU_ERF_MAX_ERROR), "{worst:e} at {at}");
        // The reference itself is GELU through f64 erf.
        assert_eq!(gelu_erf(&[0.0])[0], 0.0);
        assert!((f64::from(gelu_erf(&[1.0])[0]) - 0.841_344_746_068_543).abs() < 1e-7);
    }

    #[test]
    fn softmax_rows_sum_to_one_and_survive_large_scores() {
        let p = softmax(&[1.0, 2.0, 3.0, 1000.0, 1000.0, -1000.0], 3);
        assert!((p[..3].iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[2] > p[1] && p[1] > p[0]);
        assert_eq!(&p[3..], &[0.5, 0.5, 0.0]);
    }

    #[test]
    fn attention_reads_its_head_of_the_packed_rows() {
        // Two tokens, two heads of width one: rows [q0 q1 | k0 k1 | v0 v1].
        let qkv = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let s = attention_scores(&qkv, (2, 2, 1));
        // S[i][h][j] = q(i, h) k(j, h).
        assert_eq!(s, vec![3.0, 9.0, 8.0, 20.0, 21.0, 63.0, 32.0, 80.0]);
        // Weights that pick key 1 for head 0 and key 0 for head 1.
        let p = [0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0];
        assert_eq!(
            attention_values(&p, &qkv, (2, 2, 1)),
            vec![11.0, 6.0, 11.0, 6.0]
        );
    }

    #[test]
    fn a_patch_embedding_of_one_hot_weights_picks_a_pixel_of_each_patch() {
        // A 4x4 one-channel image in 2x2 patches; the weights pick pixel
        // (1, 0) of the patch into channel 0 and (0, 1) into channel 1.
        let image: Vec<f32> = (0..16).map(|i| i as f32).collect();
        // HWIO: weight (ky, kx, ci = 0, co) is at (ky * 2 + kx) * 2 + co.
        let mut w = vec![0.0; 2 * 2 * 2];
        w[2 * 2] = 1.0;
        w[2 + 1] = 1.0;
        let out = patch_embed(&image, &w, Some(&[0.5, 0.0]), (2, 2, 2, 1, 2));
        assert_eq!(out, vec![4.5, 1.0, 6.5, 3.0, 12.5, 9.0, 14.5, 11.0]);
        assert_eq!(tokens_to_map(&prepend_token(&[7.0, 7.0], &out), 2), out);
        assert_eq!(
            channel_scale(&[1.0, 2.0, 3.0, 4.0], &[2.0, -1.0]),
            vec![2.0, -2.0, 6.0, -4.0]
        );
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
