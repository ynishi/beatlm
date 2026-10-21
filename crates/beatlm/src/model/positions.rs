//! A vision transformer's learned position embedding, resampled on the host to
//! the patch grid of another image size: [`Convert::InterpolatePositions`].
//!
//! It is the arithmetic of `transformers` 5.18.0's
//! `Dinov2Embeddings.interpolate_pos_encoding` (`models/dinov2/modeling_dinov2.py`)
//! and of the PyTorch kernel it calls, `torch.nn.functional.interpolate` with
//! `mode="bicubic"`, `align_corners=False`, an output `size` (not a scale
//! factor) and no antialiasing, in `float32`:
//!
//! - the embedding is stored `[1][1 + g * g][C]`, the class token's row first,
//!   then a `g x g` grid of patch rows (`g` is the integer square root of the
//!   row count less one, `torch_int(num_positions ** 0.5)`);
//! - the class token's row is kept as stored, and the grid is resampled to
//!   `height x width`, each channel on its own, then laid out patch by patch
//!   again behind the class row;
//! - along one axis of `n` samples resampled to `m`, PyTorch's
//!   `area_pixel_compute_scale` gives the ratio `n / m` (in `float32`, since a
//!   size and no scale factor was passed), and output `o` reads the source
//!   position `real = ratio * (o + 0.5) - 0.5`, kept negative where it is
//!   (`area_pixel_compute_source_index` with `cubic`); `guard_index_and_lambda`
//!   takes `i = min(floor(real), n - 1)` and `t = clamp(real - i, 0, 1)`;
//! - the four samples `i - 1 .. i + 2` are read with their indices clamped to
//!   `0 .. n - 1` (`upsample_get_value_bounded`), weighted by Keys' cubic
//!   convolution with `A = -0.75` (`get_cubic_upsample_coefficients`, which is
//!   not the `-0.5` of antialiased resampling): `w0 = c2(t + 1)`, `w1 = c1(t)`,
//!   `w2 = c1(1 - t)`, `w3 = c2((1 - t) + 1)` with
//!   `c1(x) = ((A + 2) x - (A + 3)) x x + 1` and
//!   `c2(x) = ((A x - 5 A) x + 8 A) x - 4 A`;
//! - the two axes are separable: each of the four rows is interpolated along
//!   the width first, and the four results along the height.
//!
//! All of it is done in `f32`, in that order. The result is not promised to be
//! bit-identical to PyTorch's: its CPU kernel is compiled C++, which may
//! contract a multiply and an add into one rounding, and it accumulates in the
//! order its loops give. The difference is a rounding of the embedding, which
//! the comparison with the reference at another size measures with the rest
//! of the model.
//!
//! # Why on the host, at load time
//!
//! The resampled embedding depends on the weights and the image size and on
//! nothing a frame brings, so it is a constant of the loaded model, as a
//! transposed weight is ([`Convert`]'s rule: layouts change once, on the
//! host). As a kernel it would either run in every frame, a dispatch and a
//! read of the stored embedding for a result that never changes, or be a
//! one-off program run before the first frame, which needs a bicubic kernel in
//! the IR that nothing else uses and a second run path to keep correct. On
//! the host it is `1 + g * g` rows of `C` numbers, sixteen multiply-adds each,
//! done once while the weights are read; and as plain arithmetic it is tested
//! against hand-computed values without an adapter.
//!
//! [`Convert`]: super::Convert
//! [`Convert::InterpolatePositions`]: super::Convert::InterpolatePositions

/// The `A` of Keys' cubic convolution PyTorch's bicubic resampling uses.
const A: f32 = -0.75;

/// The cubic convolution's weight at a distance `x` of at most one.
fn near(x: f32) -> f32 {
    ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
}

/// The cubic convolution's weight at a distance `x` between one and two.
fn far(x: f32) -> f32 {
    ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
}

/// For each of `output` samples resampled from `input` along one axis, the
/// four source indices it reads, clamped to the axis, and their weights.
///
/// `input` and `output` are positive.
fn axis(input: usize, output: usize) -> Vec<([usize; 4], [f32; 4])> {
    let last = i64::try_from(input).unwrap_or(i64::MAX) - 1;
    // Small integers, exact in `f32`.
    let ratio = input as f32 / output as f32;
    (0..output)
        .map(|o| {
            let real = ratio * (o as f32 + 0.5) - 0.5;
            // `real` is at least -0.5 and below `input`, so the cast is exact.
            let index = (real.floor() as i64).min(last);
            let t = (real - index as f32).clamp(0.0, 1.0);
            let rest = 1.0 - t;
            let weights = [far(t + 1.0), near(t), near(rest), far(rest + 1.0)];
            let at = |j: i64| usize::try_from((index - 1 + j).clamp(0, last)).unwrap_or(0);
            ([at(0), at(1), at(2), at(3)], weights)
        })
        .collect()
}

/// `grid`, `from.0 x from.1` positions of `channels` numbers each (row-major,
/// a position's channels together), resampled bicubically to `to.0 x to.1`
/// in the same layout.
fn resample(grid: &[f32], from: (usize, usize), to: (usize, usize), channels: usize) -> Vec<f32> {
    let rows = axis(from.0, to.0);
    let cols = axis(from.1, to.1);
    let at = |y: usize, x: usize, c: usize| grid[(y * from.1 + x) * channels + c];
    let mut out = Vec::with_capacity(to.0 * to.1 * channels);
    for (ys, wy) in &rows {
        for (xs, wx) in &cols {
            for c in 0..channels {
                let along = |y: usize| {
                    at(y, xs[0], c) * wx[0]
                        + at(y, xs[1], c) * wx[1]
                        + at(y, xs[2], c) * wx[2]
                        + at(y, xs[3], c) * wx[3]
                };
                out.push(
                    along(ys[0]) * wy[0]
                        + along(ys[1]) * wy[1]
                        + along(ys[2]) * wy[2]
                        + along(ys[3]) * wy[3],
                );
            }
        }
    }
    out
}

/// The embedding `values`, `[rows][channels]` with the class token's row first
/// and a square grid after it, with the grid resampled to `to` (height,
/// width); `None` when `rows - 1` is not a square or a size is zero.
pub(super) fn interpolate(
    values: &[f32],
    rows: usize,
    channels: usize,
    to: (usize, usize),
) -> Option<Vec<f32>> {
    let positions = rows.checked_sub(1)?;
    let side = positions.isqrt();
    if side * side != positions
        || side == 0
        || channels == 0
        || to.0 == 0
        || to.1 == 0
        || values.len() != rows * channels
    {
        return None;
    }
    let (class, grid) = values.split_at(channels);
    let mut out = Vec::with_capacity((1 + to.0 * to.1) * channels);
    out.extend_from_slice(class);
    out.extend(resample(grid, (side, side), to, channels));
    Some(out)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn the_weights_are_keys_cubic_with_a_of_minus_three_quarters() {
        // Halving four samples: output 0 reads position 0.5, between samples
        // 0 and 1, with the sample before clamped to 0.
        let halved = axis(4, 2);
        assert_eq!(halved[0].0, [0, 0, 1, 2]);
        assert_eq!(halved[0].1, [-0.09375, 0.59375, 0.59375, -0.09375]);
        // Output 1 reads position 2.5, the sample after clamped to 3.
        assert_eq!(halved[1].0, [1, 2, 3, 3]);
        // Doubling two samples: output 0 reads position -0.25, so the floor is
        // -1 and every index below 0 is clamped to 0.
        let doubled = axis(2, 4);
        assert_eq!(doubled[0].0, [0, 0, 0, 1]);
        let t = 0.75f32;
        assert_eq!(
            doubled[0].1,
            [far(t + 1.0), near(t), near(1.0 - t), far(2.0 - t)]
        );
        for (_, w) in halved.iter().chain(&doubled) {
            assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn the_same_size_is_the_identity_and_a_constant_stays_constant() {
        let grid: Vec<f32> = (0..3 * 3 * 2).map(|v| v as f32 * 0.25 - 1.0).collect();
        assert_eq!(resample(&grid, (3, 3), (3, 3), 2), grid);
        let flat = vec![1.5f32; 5 * 5 * 3];
        for &to in &[(2, 2), (7, 7), (3, 6)] {
            let out = resample(&flat, (5, 5), to, 3);
            assert_eq!(out.len(), to.0 * to.1 * 3);
            // Weights that are not dyadic sum to one within rounding.
            assert!(out.iter().all(|&v| (v - 1.5).abs() < 1e-5), "{to:?}");
        }
    }

    #[test]
    fn a_two_by_two_grid_halved_is_the_mean_of_its_four_positions() {
        // Each axis reads [0, 0, 1, 1] at t = 0.5, weights that sum to one
        // half per sample.
        let values = vec![9.0, -9.0, 1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0];
        let out = interpolate(&values, 5, 2, (1, 1)).unwrap();
        assert_eq!(&out[..2], &[9.0, -9.0], "the class token's row is kept");
        assert!((out[2] - 2.5).abs() < 1e-6 && (out[3] - 25.0).abs() < 1e-5);
    }

    #[test]
    fn halving_a_row_weights_its_neighbours_by_the_cubic() {
        // One channel, a 4 x 4 grid whose value is its column: every row
        // halved alike, and output column 0 is
        // 0.5 * v0 + 0.59375 * v1 - 0.09375 * v2.
        let grid: Vec<f32> = (0..16).map(|i| (i % 4) as f32).collect();
        let out = resample(&grid, (4, 4), (2, 2), 1);
        let want0 = 0.59375 - 0.09375 * 2.0;
        assert!((out[0] - want0).abs() < 1e-6, "{}", out[0]);
        assert!((out[2] - want0).abs() < 1e-6);
    }

    #[test]
    fn a_grid_that_is_not_square_is_refused() {
        assert_eq!(interpolate(&[0.0; 4], 4, 1, (2, 2)), None);
        assert_eq!(interpolate(&[0.0; 5], 5, 1, (0, 2)), None);
        assert_eq!(interpolate(&[0.0; 4], 5, 1, (2, 2)), None);
        assert_eq!(interpolate(&[], 0, 1, (2, 2)), None);
    }
}
