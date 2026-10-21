//! CPU reference implementations of the network's kernels.
//!
//! They define what each hand-written WGSL kernel in `net.wgsl` must compute,
//! in the same layouts, and accumulate in `f64` so the reference is the more
//! accurate side of a comparison. The GPU tests compare against them within
//! [`TOLERANCE_ABS`] plus [`TOLERANCE_REL`] of the reference value.

/// Absolute part of the comparison tolerance.
///
/// The kernels accumulate in `f32` in a different order than the reference, so
/// results differ by rounding; sums here are of at most a few thousand terms of
/// magnitude near one, whose `f32` rounding error is far below this.
pub const TOLERANCE_ABS: f32 = 1e-4;
/// Relative part of the comparison tolerance.
pub const TOLERANCE_REL: f32 = 1e-4;

/// Layer normalisation's epsilon; the kernel's `LN_EPS`.
const LN_EPS: f64 = 1e-5;

/// 3x3 convolution, stride 1, zero padding of 1.
///
/// `input` is `[h][w][cin]`, `weights` is `[3][3][cin][cout]` and the result is
/// `[h][w][cout]`.
pub fn conv3x3(
    input: &[f32],
    weights: &[f32],
    h: usize,
    w: usize,
    cin: usize,
    cout: usize,
) -> Vec<f32> {
    let mut out = vec![0.0; h * w * cout];
    for y in 0..h {
        for x in 0..w {
            for co in 0..cout {
                let mut acc = 0.0_f64;
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

/// Layer normalisation over the `c` channels of each of `pixels` pixels.
///
/// `params` holds gamma in `[0, c)` and beta in `[c, 2c)`.
pub fn layernorm(input: &[f32], params: &[f32], pixels: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0.0; pixels * c];
    for p in 0..pixels {
        let row = &input[p * c..(p + 1) * c];
        let mean = row.iter().map(|&v| f64::from(v)).sum::<f64>() / c as f64;
        let var = row
            .iter()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / c as f64;
        let inv = 1.0 / (var + LN_EPS).sqrt();
        for (i, &v) in row.iter().enumerate() {
            let y = (f64::from(v) - mean) * inv * f64::from(params[i]) + f64::from(params[c + i]);
            out[p * c + i] = y as f32;
        }
    }
    out
}

/// `C[m][n] = A[m][k] * B[k][n]`, all row-major.
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for l in 0..k {
                acc += f64::from(a[i * k + l]) * f64::from(b[l * n + j]);
            }
            out[i * n + j] = acc as f32;
        }
    }
    out
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

    #[test]
    fn a_one_hot_kernel_copies_the_input_through_the_centre_tap() {
        // Weights that pick the centre tap and the same channel are the identity.
        let (h, w, c) = (3, 4, 2);
        let mut weights = vec![0.0; 9 * c * c];
        for ci in 0..c {
            weights[((4) * c + ci) * c + ci] = 1.0;
        }
        let input: Vec<f32> = (0..h * w * c).map(|i| i as f32).collect();
        assert_eq!(conv3x3(&input, &weights, h, w, c, c), input);
    }

    #[test]
    fn a_corner_pixel_sums_only_the_taps_inside_the_image() {
        // All-ones 1-channel 2x2 image and all-ones kernel: the corner sees a
        // 2x2 neighbourhood, i.e. four taps.
        let out = conv3x3(&[1.0; 4], &[1.0; 9], 2, 2, 1, 1);
        assert_eq!(out, vec![4.0; 4]);
    }

    #[test]
    fn layernorm_gives_zero_mean_unit_variance_before_gamma_and_beta() {
        let c = 4;
        let mut params = vec![1.0; c];
        params.extend(vec![0.0; c]);
        let out = layernorm(&[1.0, 2.0, 3.0, 4.0], &params, 1, c);
        let mean: f32 = out.iter().sum::<f32>() / c as f32;
        let var: f32 = out.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / c as f32;
        assert!(mean.abs() < 1e-6);
        assert!((var - 1.0).abs() < 1e-3);
    }

    #[test]
    fn matmul_multiplies() {
        let out = matmul(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0], 2, 2, 2);
        assert_eq!(out, vec![19.0, 22.0, 43.0, 50.0]);
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
