//! What a port of Depth Anything V2 Small is compared with: the fixed input,
//! where the reference outputs are kept, and how far from them a port may be.
//!
//! `just reference-depth-anything` runs the original `transformers`
//! implementation on [`input`] and writes the input, the four encoder features
//! the DPT head reads and the predicted depth next to the weights (the
//! [`load` module documentation](crate::load#getting-the-depth-anything-v2-small-weights)
//! says how). Everything a comparison needs to agree on with that script and
//! with every other comparison is here, once: the native tests
//! (`crates/beatlm/tests/depth_anything_v2.rs`) and the browser host
//! (`beatlm-web`) both read it, so a page and a native run check a port by the
//! same input, the same error measures and the same tolerances.
//!
//! # Where the files are
//!
//! Under a models directory, [`MODEL_DIR`] holds [`WEIGHTS_FILE`], and the
//! reference made at side `s` is in [`dir`]`(config, s)`: `reference/` at the
//! side the position embedding is stored for, `reference/<s>/` at any other.
//! The paths are relative and use `/`, so a native test joins them to a
//! directory and a page appends them to a URL.
//!
//! # The input
//!
//! Defined by a formula in integers and single-precision arithmetic, so that
//! Rust computes the same bits as the script's NumPy without reading a file:
//! see [`level`] and [`input`]. The script's header states the same formula,
//! and a native test compares the file it wrote with [`input`] bit for bit.

use std::fmt::Write as _;

use super::Config;

/// The model's directory under a models directory, which
/// `just setup-depth-anything` writes.
pub const MODEL_DIR: &str = "depth-anything-v2-small";

/// The weights file in [`MODEL_DIR`].
pub const WEIGHTS_FILE: &str = "model.safetensors";

/// The input the reference ran on, NCHW `[1][3][side][side]`, raw
/// little-endian `f32`.
pub const INPUT_FILE: &str = "input.f32";

/// The depth the reference predicted, `[1][side][side]`, raw little-endian
/// `f32`.
pub const DEPTH_FILE: &str = "predicted_depth.f32";

/// The JSON the script writes beside its outputs: one object per tensor with
/// its `name`, `file` and `shape`.
pub const MANIFEST_FILE: &str = "manifest.json";

/// The manifest's name for the predicted depth.
pub const DEPTH_NAME: &str = "predicted_depth";

/// The channel means of the checkpoint's image processor
/// (`preprocessor_config.json`), which put the reference input in the space
/// the model sees after preprocessing.
pub const MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// The channel standard deviations of the checkpoint's image processor; see
/// [`MEAN`].
pub const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// The side of the checkerboard's cells, in pixels: 14 of them across at 518.
const CELL: usize = 37;

/// The directory of the reference made at `side` pixels, relative to
/// [`MODEL_DIR`]: `reference` at the side `config`'s position embedding is
/// stored for, `reference/<side>` at any other.
#[must_use]
pub fn dir(config: &Config, side: usize) -> String {
    if side == config.image_size {
        "reference".to_owned()
    } else {
        format!("reference/{side}")
    }
}

/// The `just` recipe that writes the reference at `side` pixels.
#[must_use]
pub fn recipe(config: &Config, side: usize) -> String {
    if side == config.image_size {
        "reference-depth-anything".to_owned()
    } else {
        format!("reference-depth-anything {side}")
    }
}

/// The file of the encoder feature the reference recorded after layer
/// `stage`, one of the configuration's `out_indices`.
#[must_use]
pub fn feature_file(stage: usize) -> String {
    format!("encoder_stage_{stage}.f32")
}

/// The 8-bit level of one pixel of the reference image of `side` pixels,
/// before normalisation.
///
/// Channel 0 is a horizontal ramp, channel 1 a vertical ramp, channel 2 (and
/// any later one) a checkerboard of 37-pixel cells, so the image has structure
/// at every scale of the patch grid. Integer arithmetic only: a level is the
/// same in any language. A side of one pixel is read as two, so the ramps have
/// no zero divisor; the model has no such side.
#[must_use]
pub fn level(side: usize, channel: usize, y: usize, x: usize) -> u32 {
    let last = side.saturating_sub(1).max(1);
    let level = match channel {
        0 => x * 255 / last,
        1 => y * 255 / last,
        _ => ((x / CELL + y / CELL) % 2) * 255,
    };
    u32::try_from(level).unwrap_or(u32::MAX)
}

/// The reference input of `side` pixels, channel-first (`[1, 3, side, side]`,
/// row-major).
///
/// Each value is `(level / 255 - mean) / std`, computed in single precision,
/// one rounded operation after another in that order. Division, subtraction
/// and conversion of a small integer are correctly rounded in IEEE 754
/// binary32, so the same sequence gives the same bits in NumPy float32, which
/// is what `scripts/reference/depth_anything_v2.py` does. No random numbers,
/// no transcendental functions, no image file.
#[must_use]
pub fn input(side: usize) -> Vec<f32> {
    let mut values = Vec::with_capacity(3 * side * side);
    for (channel, (mean, std)) in MEAN.iter().zip(&STD).enumerate() {
        for y in 0..side {
            for x in 0..side {
                // A level is at most 255, exact in `f32`.
                #[allow(clippy::cast_precision_loss)]
                let unit = level(side, channel, y, x) as f32 / 255.0f32;
                values.push((unit - mean) / std);
            }
        }
    }
    values
}

/// A channel-first image of three channels and `side` pixels as the graph's
/// NHWC image, `[side][side][3]`; `None` when `nchw` is not that size.
#[must_use]
pub fn to_nhwc(nchw: &[f32], side: usize) -> Option<Vec<f32>> {
    let pixels = side.checked_mul(side)?;
    if nchw.len() != pixels.checked_mul(3)? {
        return None;
    }
    let mut image = vec![0.0; nchw.len()];
    for c in 0..3 {
        for p in 0..pixels {
            image[p * 3 + c] = nchw[c * pixels + p];
        }
    }
    Some(image)
}

/// Raw little-endian `f32`s; a trailing partial number is dropped.
#[must_use]
pub fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// The first byte at which `bytes` differs from `values` written as
/// little-endian `f32`s, or `None` when they are the same bytes. A length that
/// differs is a difference at the shorter one's end.
#[must_use]
pub fn first_difference(bytes: &[u8], values: &[f32]) -> Option<usize> {
    let expected: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    match bytes.iter().zip(&expected).position(|(a, b)| a != b) {
        Some(at) => Some(at),
        None if bytes.len() == expected.len() => None,
        None => Some(bytes.len().min(expected.len())),
    }
}

/// The shape the reference's manifest records for tensor `name`, or `None`
/// when the manifest has no such entry or no shape for it.
///
/// It reads the first `[...]` after the `"shape"` that follows the name's
/// entry, which is that tensor's shape in the JSON the script writes, rather
/// than take a JSON parser as a dependency for one array of integers.
#[must_use]
pub fn manifest_shape(manifest: &str, name: &str) -> Option<Vec<usize>> {
    let entry = format!("\"name\": \"{name}\"");
    let rest = &manifest[manifest.find(&entry)?..];
    let shape = rest.find("\"shape\"")?;
    let open = rest[shape..].find('[')? + shape;
    let close = rest[open..].find(']')? + open;
    rest[open + 1..close]
        .split(',')
        .map(|d| d.trim().parse().ok())
        .collect()
}

/// The tolerance of the comparison with the reference features: the largest
/// absolute difference of any element. The reference ran in PyTorch on the
/// CPU in `f32`; the GPU sums in other orders and evaluates `exp`, `erf` and
/// the inverse square root its own way, and the differences grow through
/// twelve residual layers.
///
/// Set against what the native test measured at 518 pixels on an Apple M1
/// Pro, Metal, on 2026-10-21, in two runs: a largest difference of 5.9e-4 and
/// 6.3e-4 (stage 6; the reference's largest magnitude there is 73) and a mean
/// of at most 1.2e-5, over stages 3, 6, 9 and 12. The GPU's result is not
/// bit-identical from run to run. The comparison at another side applies the
/// same tolerances; it adds the host's resampling of the position embedding,
/// which rounds differently from PyTorch's by single-precision rounding only.
pub const FEATURE_MAX_ERROR: f32 = 2e-3;

/// The tolerance on the mean absolute difference over a feature; see
/// [`FEATURE_MAX_ERROR`].
pub const FEATURE_MEAN_ERROR: f32 = 5e-5;

/// The tolerance on the largest absolute difference of the depth.
///
/// The depth inherits the encoder's differences from the reference (see
/// [`FEATURE_MAX_ERROR`]) and adds the head's own, of convolutions summing in
/// another order than PyTorch's. Set against what the native test measured at
/// 518 pixels on an Apple M1 Pro, Metal, on 2026-10-21, in two invocations,
/// identical in both: a largest difference of 5.96e-6 fused and 6.68e-6
/// unfused, a mean of 1.02e-6 and 1.01e-6, and a largest relative difference
/// of 2.05e-6 and 2.12e-6, on a reference depth from 1.48 to 3.64. The
/// tolerances leave room for an encoder at the edge of its own, three times
/// what it measured, and as much again for the head on another adapter. The
/// comparison at another side applies them unchanged.
pub const DEPTH_MAX_ERROR: f32 = 5e-5;

/// The tolerance on the mean absolute difference of the depth; see
/// [`DEPTH_MAX_ERROR`].
pub const DEPTH_MEAN_ERROR: f32 = 1e-5;

/// The tolerance on the largest relative difference of the depth, measured
/// over the pixels whose reference depth is at least [`RELATIVE_FLOOR`] of
/// its largest; see [`DEPTH_MAX_ERROR`].
pub const DEPTH_MAX_RELATIVE_ERROR: f32 = 2e-5;

/// The share of the largest reference depth below which a pixel's relative
/// error is not counted: the final ReLU clips depth to zero, and near zero a
/// difference of rounding size is a large ratio that says nothing about the
/// port.
pub const RELATIVE_FLOOR: f64 = 0.01;

/// The largest reference magnitude of a set of numbers.
fn scale_of(want: &[f32]) -> f64 {
    want.iter().map(|w| f64::from(w.abs())).fold(0.0, f64::max)
}

/// What a feature map differs from the reference's by.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureError {
    /// The largest absolute difference of any element.
    pub max: f64,
    /// The mean absolute difference.
    pub mean: f64,
    /// The reference's largest magnitude.
    pub scale: f64,
}

impl FeatureError {
    /// The differences of `got` from `want`; `None` when they are not the
    /// same, non-zero length.
    #[must_use]
    pub fn of(got: &[f32], want: &[f32]) -> Option<Self> {
        if got.len() != want.len() || got.is_empty() {
            return None;
        }
        let (mut max, mut sum) = (0.0_f64, 0.0_f64);
        for (&g, &w) in got.iter().zip(want) {
            let d = f64::from(g - w).abs();
            max = max.max(d);
            sum += d;
        }
        #[allow(clippy::cast_precision_loss)]
        let mean = sum / got.len() as f64;
        Some(Self {
            max,
            mean,
            scale: scale_of(want),
        })
    }

    /// Whether both measures are within [`FEATURE_MAX_ERROR`] and
    /// [`FEATURE_MEAN_ERROR`]. A difference that is not a number is not.
    #[must_use]
    pub fn within_tolerance(&self) -> bool {
        self.max <= f64::from(FEATURE_MAX_ERROR) && self.mean <= f64::from(FEATURE_MEAN_ERROR)
    }
}

/// What a depth differs from the reference's by.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DepthError {
    /// The largest absolute difference of any pixel.
    pub max: f64,
    /// The mean absolute difference.
    pub mean: f64,
    /// The largest relative difference over the pixels whose reference depth
    /// is at least [`RELATIVE_FLOOR`] of [`scale`](Self::scale).
    pub max_relative: f64,
    /// The reference's largest depth.
    pub scale: f64,
}

impl DepthError {
    /// The differences of `got` from `want`; `None` when they are not the
    /// same, non-zero length.
    #[must_use]
    pub fn of(got: &[f32], want: &[f32]) -> Option<Self> {
        if got.len() != want.len() || got.is_empty() {
            return None;
        }
        let scale = scale_of(want);
        let (mut max, mut sum, mut max_relative) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (&g, &w) in got.iter().zip(want) {
            let d = f64::from(g - w).abs();
            max = max.max(d);
            sum += d;
            let w = f64::from(w).abs();
            if w >= RELATIVE_FLOOR * scale {
                max_relative = max_relative.max(d / w);
            }
        }
        #[allow(clippy::cast_precision_loss)]
        let mean = sum / got.len() as f64;
        Some(Self {
            max,
            mean,
            max_relative,
            scale,
        })
    }

    /// Whether all three measures are within [`DEPTH_MAX_ERROR`],
    /// [`DEPTH_MEAN_ERROR`] and [`DEPTH_MAX_RELATIVE_ERROR`]. A difference
    /// that is not a number is not.
    #[must_use]
    pub fn within_tolerance(&self) -> bool {
        self.max <= f64::from(DEPTH_MAX_ERROR)
            && self.mean <= f64::from(DEPTH_MEAN_ERROR)
            && self.max_relative <= f64::from(DEPTH_MAX_RELATIVE_ERROR)
    }

    /// The measures as `name=value` pairs on one line, in the order the native
    /// test prints them, for a report.
    #[must_use]
    pub fn describe(&self) -> String {
        let mut out = String::new();
        // Writing to a String cannot fail.
        let _ = write!(
            out,
            "max_abs_error={:.3e} mean_abs_error={:.3e} max_relative_error={:.3e} \
             reference_scale={:.4}",
            self.max, self.mean, self.max_relative, self.scale
        );
        out
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// The side the position embedding is stored for.
    const NATIVE: usize = 518;
    /// Another side the port is compared at.
    const OTHER: usize = 266;

    #[test]
    fn the_reference_input_formula_is_pinned() {
        assert_eq!(level(NATIVE, 0, 5, 0), 0);
        assert_eq!(level(NATIVE, 0, 5, 36), 17);
        assert_eq!(level(NATIVE, 0, 5, 517), 255);
        assert_eq!(level(NATIVE, 1, 517, 5), 255);
        assert_eq!(level(NATIVE, 1, 36, 5), 17);
        assert_eq!((level(NATIVE, 2, 0, 36), level(NATIVE, 2, 0, 37)), (0, 255));
        assert_eq!(
            (level(NATIVE, 2, 37, 37), level(NATIVE, 2, 37, 36)),
            (0, 255)
        );
        // At another side the ramps span that side, and the cells stay 37 pixels.
        assert_eq!(level(OTHER, 0, 5, OTHER - 1), 255);
        assert_eq!(level(OTHER, 1, OTHER / 2, 5), 127);
        assert_eq!((level(OTHER, 2, 0, 36), level(OTHER, 2, 0, 37)), (0, 255));

        let values = input(NATIVE);
        assert_eq!(values.len(), 3 * NATIVE * NATIVE);
        assert_eq!(values[0], (0.0f32 - 0.485f32) / 0.229f32);
        assert_eq!(values[NATIVE - 1], (1.0f32 - 0.485f32) / 0.229f32);
        let green_start = NATIVE * NATIVE;
        assert_eq!(values[green_start], (0.0f32 - 0.456f32) / 0.224f32);
        assert_eq!(input(OTHER).len(), 3 * OTHER * OTHER);
    }

    #[test]
    fn the_nhwc_image_interleaves_the_channels() {
        let side = 2;
        let nchw: Vec<f32> = (0..12).map(|v| v as f32).collect();
        let nhwc = to_nhwc(&nchw, side).unwrap();
        assert_eq!(&nhwc[..6], &[0.0, 4.0, 8.0, 1.0, 5.0, 9.0]);
        assert_eq!(to_nhwc(&nchw[..11], side), None);
        assert_eq!(
            to_nhwc(&input(OTHER), OTHER).unwrap().len(),
            3 * OTHER * OTHER
        );
    }

    #[test]
    fn the_reference_directory_depends_on_the_side() {
        let config = Config::small();
        assert_eq!(dir(&config, 518), "reference");
        assert_eq!(dir(&config, 266), "reference/266");
        assert_eq!(recipe(&config, 518), "reference-depth-anything");
        assert_eq!(recipe(&config, 266), "reference-depth-anything 266");
        assert_eq!(feature_file(12), "encoder_stage_12.f32");
    }

    #[test]
    fn bytes_are_compared_with_the_numbers_they_should_hold() {
        let values = [1.0f32, -2.5];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(floats(&bytes), values);
        assert_eq!(first_difference(&bytes, &values), None);
        assert_eq!(first_difference(&bytes[..7], &values), Some(7));
        let mut changed = bytes.clone();
        changed[5] ^= 1;
        assert_eq!(first_difference(&changed, &values), Some(5));
        assert_eq!(first_difference(&bytes, &[1.0]), Some(4));
    }

    #[test]
    fn the_manifest_gives_each_tensor_its_shape() {
        let manifest = r#"{"tensors": [
            {"name": "input", "file": "input.f32", "shape": [1, 3, 266, 266]},
            {"name": "predicted_depth", "file": "predicted_depth.f32", "shape": [1, 266, 266]}
        ]}"#;
        assert_eq!(
            manifest_shape(manifest, DEPTH_NAME),
            Some(vec![1, 266, 266])
        );
        assert_eq!(
            manifest_shape(manifest, "input"),
            Some(vec![1, 3, 266, 266])
        );
        assert_eq!(manifest_shape(manifest, "missing"), None);
        assert_eq!(manifest_shape(r#""name": "x", "shape": [1, a]"#, "x"), None);
    }

    #[test]
    fn the_depth_error_measures_absolute_mean_and_relative_differences() {
        let want = [2.0f32, 1.0, 0.001];
        let got = [2.0f32 + 4e-5, 1.0, 0.002];
        let error = DepthError::of(&got, &want).unwrap();
        assert!((error.max - 1e-3).abs() < 1e-6, "{error:?}");
        assert_eq!(error.scale, 2.0);
        // The last pixel is below the floor of 1% of 2.0, so its ratio of
        // one is not counted; the first one's is.
        assert!((error.max_relative - 2e-5).abs() < 1e-6, "{error:?}");
        assert!(!error.within_tolerance());
        let close = DepthError::of(&want, &want).unwrap();
        assert!(close.within_tolerance());
        assert_eq!(DepthError::of(&got[..2], &want), None);
        assert_eq!(DepthError::of(&[], &[]), None);
        let nan = DepthError::of(&[f32::NAN, 1.0, 0.001], &want).unwrap();
        assert!(!nan.within_tolerance());
        assert!(close.describe().starts_with("max_abs_error=0.000e0 "));
    }

    #[test]
    fn the_feature_error_measures_absolute_and_mean_differences() {
        let want = [70.0f32, -3.0];
        let error = FeatureError::of(&[70.001, -3.0], &want).unwrap();
        assert!(error.max > 9e-4 && error.max < 1.1e-3, "{error:?}");
        assert_eq!(error.scale, 70.0);
        assert!(!error.within_tolerance());
        assert!(FeatureError::of(&want, &want).unwrap().within_tolerance());
        assert_eq!(FeatureError::of(&want[..1], &want), None);
    }
}
