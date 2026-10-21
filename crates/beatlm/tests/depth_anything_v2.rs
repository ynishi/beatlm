//! Depth Anything V2 Small against the files `just setup-depth-anything` and
//! `just reference-depth-anything` put under `models/`.
//!
//! The tests that read those files are `#[ignore]`d, because the files are
//! third-party material fetched on request and never part of the tree
//! (PUBLIC_DEVELOPMENT.md, "Third-party material"); run them by hand with
//! `cargo test -p beatlm --test depth_anything_v2 -- --ignored`. Run without
//! the files they fail, and the message names the recipe that produces them: a
//! test that skipped itself would report a pass that checked nothing.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::path::{Path, PathBuf};

use beatlm::load::depth_anything_v2::{Config, inventory};
use beatlm::load::{Archive, read_file};

/// The parameter count the model is published with.
const PUBLISHED_PARAMETERS: u64 = 24_785_089;

/// Side of the reference input, in pixels: the size the position embeddings
/// are stored for.
const SIDE: usize = 518;

/// The channel means and standard deviations of the checkpoint's image
/// processor (`preprocessor_config.json`), used to put the reference input in
/// the space the model sees after preprocessing.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

fn model_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/depth-anything-v2-small")
}

/// Reads a file under `models/`, or fails naming the recipe that creates it.
fn read_or_explain(path: &Path, recipe: &str) -> Vec<u8> {
    read_file(path).unwrap_or_else(|err| {
        let cause = std::error::Error::source(&err)
            .map(|source| format!(" ({source})"))
            .unwrap_or_default();
        panic!("{err}{cause}; run `just {recipe}` from the repository root first")
    })
}

/// The 8-bit level of one pixel of the reference image, before normalisation.
///
/// Channel 0 is a horizontal ramp, channel 1 a vertical ramp, channel 2 a
/// checkerboard of 37-pixel cells (14 of them across), so the image has
/// structure at every scale of the patch grid. Integer arithmetic only: a
/// level is the same in any language.
fn level(channel: usize, y: usize, x: usize) -> u32 {
    match channel {
        0 => (x * 255 / (SIDE - 1)) as u32,
        1 => (y * 255 / (SIDE - 1)) as u32,
        _ => (((x / 37 + y / 37) % 2) * 255) as u32,
    }
}

/// The reference input, channel-first (`[1, 3, 518, 518]`, row-major).
///
/// Each value is `(level / 255 - mean) / std`, computed in single precision,
/// one rounded operation after another in that order. Division, subtraction
/// and conversion of a small integer are correctly rounded in IEEE 754 binary32,
/// so the same sequence gives the same bits in NumPy float32, which is what
/// `scripts/reference/depth_anything_v2.py` does. No random numbers, no
/// transcendental functions, no image file.
fn reference_input() -> Vec<f32> {
    let mut values = Vec::with_capacity(3 * SIDE * SIDE);
    for channel in 0..3 {
        for y in 0..SIDE {
            for x in 0..SIDE {
                let unit = level(channel, y, x) as f32 / 255.0f32;
                values.push((unit - MEAN[channel]) / STD[channel]);
            }
        }
    }
    values
}

#[test]
fn the_reference_input_formula_is_pinned() {
    assert_eq!(level(0, 5, 0), 0);
    assert_eq!(level(0, 5, 36), 17);
    assert_eq!(level(0, 5, 517), 255);
    assert_eq!(level(1, 517, 5), 255);
    assert_eq!(level(1, 36, 5), 17);
    assert_eq!((level(2, 0, 36), level(2, 0, 37)), (0, 255));
    assert_eq!((level(2, 37, 37), level(2, 37, 36)), (0, 255));

    let input = reference_input();
    assert_eq!(input.len(), 3 * SIDE * SIDE);
    assert_eq!(input[0], (0.0f32 - 0.485f32) / 0.229f32);
    assert_eq!(input[SIDE - 1], (1.0f32 - 0.485f32) / 0.229f32);
    let green_start = SIDE * SIDE;
    assert_eq!(input[green_start], (0.0f32 - 0.456f32) / 0.224f32);
}

#[test]
#[ignore = "needs `just setup-depth-anything`"]
fn the_inventory_matches_the_real_file_exactly() {
    let bytes = read_or_explain(
        &model_dir().join("model.safetensors"),
        "setup-depth-anything",
    );
    let archive = Archive::parse(&bytes).expect("the downloaded file is a safetensors file");
    let inventory = inventory(&Config::small()).expect("the small configuration is valid");

    let report = inventory.check(&archive);

    assert!(report.is_exact(), "{report}");
    assert_eq!(report.file_parameters, PUBLISHED_PARAMETERS, "{report}");
    assert_eq!(report.expected_parameters, PUBLISHED_PARAMETERS, "{report}");
}

#[test]
#[ignore = "needs `just reference-depth-anything`"]
fn the_rust_input_is_bit_identical_to_the_one_python_wrote() {
    let bytes = read_or_explain(
        &model_dir().join("reference/input.f32"),
        "reference-depth-anything",
    );
    let expected: Vec<u8> = reference_input()
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();

    assert_eq!(bytes.len(), expected.len(), "the file has the wrong size");
    let first_difference = bytes.iter().zip(&expected).position(|(a, b)| a != b);
    assert_eq!(
        first_difference, None,
        "the input differs from the formula at this byte offset"
    );
}
