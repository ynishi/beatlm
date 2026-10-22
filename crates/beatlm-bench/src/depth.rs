//! Depth Anything V2 Small as the job: the `depth` workload.
//!
//! The graph is `beatlm::model::depth_anything_v2::model` at `--size` pixels
//! a side, lowered with `--fusion`, and every weight's numbers are read from
//! the real weights file, which [`load`] checks against the model's inventory
//! before anything is built. A job is one inference: its slices are the
//! program's kernels, or their parts once calibration has split them
//! (`calibrate.rs`), exactly as for the `net` workload, and it runs through
//! the same [`Net`](crate::net::Net), so the slots, bind groups and memory
//! plan follow the rules `net.rs` documents. Nothing here is a simplification
//! of the model: the numbers a slot computes are the depth of its input.
//!
//! What the workload simplifies is the frame around it:
//!
//! - **The input is resident.** Every slot holds the same fixed image
//!   ([`image`]), uploaded once before the loop. A host would refill a slot
//!   with the frame it submits, ordered on the queue before the job's first
//!   slice; the bench measures the model, not that copy.
//! - **The depth is not consumed.** The bench reads nothing back; the
//!   comparison with the reference is `beatlm`'s test
//!   (`crates/beatlm/tests/depth_anything_v2.rs`).
//!
//! The weights are third-party material fetched on request: [`load`] reads
//! them under `BEATLM_MODELS_DIR`, or under `models/` at the workspace root
//! when it is unset, as the `beatlm` tests do, and fails naming
//! `just setup-depth-anything` when they are not there.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use beatlm::load::depth_anything_v2::{Config, Role, inventory, reference};
use beatlm::load::{Archive, read_file};
use beatlm::model::Builder;
use beatlm::model::depth_anything_v2::{DepthModel, model};
use beatlm_core::kernel::{Fusion, Program};
use beatlm_host::slots::Numbers;

/// The model's weights file, under `BEATLM_MODELS_DIR` or the workspace's
/// `models/`.
fn weights_file() -> PathBuf {
    std::env::var_os("BEATLM_MODELS_DIR")
        .map_or_else(
            || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models"),
            PathBuf::from,
        )
        .join(reference::MODEL_DIR)
        .join(reference::WEIGHTS_FILE)
}

/// The model at `side` pixels, built into a fresh builder.
fn built(side: u32) -> Result<(Builder<Role>, DepthModel)> {
    let side = usize::try_from(side).context("--size does not fit this machine")?;
    let mut builder = Builder::new();
    let tensors = model(&mut builder, &Config::small(), side)
        .with_context(|| format!("cannot build Depth Anything V2 Small at {side} pixels"))?;
    Ok((builder, tensors))
}

/// The model at `side` pixels lowered with `fusion`, without its weights.
///
/// # Errors
///
/// A side that is not whole patches, or a graph that does not lower.
pub fn program(side: u32, fusion: Fusion) -> Result<Program> {
    lowered(&built(side)?.0, fusion)
}

/// The graph `builder` holds, lowered with `fusion`.
fn lowered(builder: &Builder<Role>, fusion: Fusion) -> Result<Program> {
    builder
        .graph()
        .lower(fusion)
        .context("cannot lower the depth model")
}

/// How many kernels, hence slices before splitting, a job of the model at
/// `side` pixels has with `fusion`.
///
/// # Errors
///
/// As [`program`].
pub fn kernel_count(side: u32, fusion: Fusion) -> Result<u32> {
    let kernels = program(side, fusion)?.kernels().len();
    u32::try_from(kernels).context("more kernels than a u32 counts")
}

/// The fixed image every slot holds: NHWC `[side][side][3]`, already in the
/// model's normalised input space. Channel 0 is a horizontal ramp and channel
/// 1 a vertical one, both from -2 to 2, and channel 2 a checkerboard of
/// 37-pixel cells at -1 and 1. The kernels have no data-dependent control
/// flow, so what the image shows does not change what a job costs.
pub fn image(side: u32) -> Vec<f32> {
    let side = usize::try_from(side).unwrap_or(0);
    let ramp = |at: usize| {
        let last = side.saturating_sub(1).max(1);
        // Pixel indices are small integers, exact in `f32`.
        4.0 * at as f32 / last as f32 - 2.0
    };
    let mut values = Vec::with_capacity(side * side * 3);
    for y in 0..side {
        for x in 0..side {
            let cell = if (x / 37 + y / 37) % 2 == 0 {
                -1.0
            } else {
                1.0
            };
            values.extend([ramp(x), ramp(y), cell]);
        }
    }
    values
}

/// The model at `side` pixels lowered with `fusion`, and the numbers it is
/// realised with: every weight read from the weights file, which is first
/// checked against the inventory, and [`image`] as every slot's input.
///
/// # Errors
///
/// No weights file (the message names `just setup-depth-anything`), a file
/// that does not match the inventory, a weight that cannot be read, or a model
/// that cannot be built at `side`.
pub fn load(side: u32, fusion: Fusion) -> Result<(Program, Numbers)> {
    let path = weights_file();
    let bytes = read_file(&path).with_context(|| {
        format!(
            "cannot read {}; run `just setup-depth-anything` from the repository root, or \
             point BEATLM_MODELS_DIR at a models directory that has it",
            path.display()
        )
    })?;
    let archive = Archive::parse(&bytes).context("the weights file is not a safetensors file")?;
    let config = Config::small();
    let inventory = inventory(&config).context("the Small configuration is not valid")?;
    let report = inventory.check(&archive);
    anyhow::ensure!(
        report.is_exact(),
        "the weights file does not match the model:\n{report}"
    );

    let (builder, tensors) = built(side)?;
    let program = lowered(&builder, fusion)?;
    let mut weights = Vec::with_capacity(builder.weights().len());
    for weight in builder.weights() {
        let numbers = weight
            .numbers(&inventory, &archive)
            .with_context(|| format!("cannot read weight {}", weight.id.get()))?;
        weights.push((weight.id, numbers));
    }
    let input = image(side);
    let numbers = Numbers {
        input: tensors.image,
        output: tensors.depth,
        weights,
        slot_input: Box::new(move |_| input.clone()),
    };
    Ok((program, numbers))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn the_image_is_the_side_squared_in_three_channels_within_the_normalised_range() {
        let image = image(28);
        assert_eq!(image.len(), 28 * 28 * 3);
        assert_eq!(&image[..3], &[-2.0, -2.0, -1.0]);
        assert_eq!(image[(27 * 28 + 27) * 3], 2.0);
        assert!(image.iter().all(|v| (-2.0..=2.0).contains(v)));
    }

    #[test]
    fn the_model_lowers_at_the_stored_side_and_at_smaller_ones_but_not_between_patches() {
        let native = kernel_count(518, Fusion::On).unwrap();
        assert_eq!(kernel_count(266, Fusion::On).unwrap(), native);
        assert!(kernel_count(266, Fusion::Off).unwrap() > native);
        assert!(kernel_count(520, Fusion::On).is_err());
    }
}
