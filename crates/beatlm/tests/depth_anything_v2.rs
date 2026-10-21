//! Depth Anything V2 Small against the files `just setup-depth-anything` and
//! `just reference-depth-anything` put under `models/`.
//!
//! The tests that read those files are `#[ignore]`d, because the files are
//! third-party material fetched on request and never part of the tree
//! (PUBLIC_DEVELOPMENT.md, "Third-party material"); run them by hand with
//! `cargo test -p beatlm --test depth_anything_v2 -- --ignored`. Run without
//! the files they fail, and the message names the recipe that produces them: a
//! test that skipped itself would report a pass that checked nothing.
//!
//! The files are looked for under the directory `BEATLM_MODELS_DIR` names,
//! and under `models/` at the workspace root when it is unset, which is where
//! the recipes write them. A second worktree has no `models/` of its own, so a
//! run there points the variable at the main checkout's:
//! `BEATLM_MODELS_DIR=<main checkout>/models cargo test ...`.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use beatlm::beatlm_core::device::DeviceProfile;
use beatlm::beatlm_core::kernel::{Fusion, TensorRole};
use beatlm::beatlm_core::memory::{MemoryPlan, TensorId};
use beatlm::beatlm_core::time::{FrameIndex, Micros};
use beatlm::codegen::Kernels;
use beatlm::gpu::{Arenas, SliceTimer};
use beatlm::load::depth_anything_v2::{Config, inventory};
use beatlm::load::{Archive, read_file};
use beatlm::model::Builder;
use beatlm::model::depth_anything_v2::encoder;

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

/// The model's directory: `depth-anything-v2-small` under `BEATLM_MODELS_DIR`,
/// or under the workspace's `models/` when the variable is unset.
fn model_dir() -> PathBuf {
    std::env::var_os("BEATLM_MODELS_DIR")
        .map_or_else(
            || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models"),
            PathBuf::from,
        )
        .join("depth-anything-v2-small")
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

/// Side of the patch grid, and the channels of a feature.
const GRID: usize = 37;
const HIDDEN: usize = 384;

/// The tolerance of the comparison with the reference features: the largest
/// absolute difference of any element. The reference ran in PyTorch on the
/// CPU in `f32`; the GPU sums in other orders and evaluates `exp`, `erf` and
/// the inverse square root its own way, and the differences grow through
/// twelve residual layers.
///
/// Set against what the test measured on an Apple M1 Pro, Metal, on
/// 2026-10-21, in two runs: a largest difference of 5.9e-4 and 6.3e-4 (stage
/// 6; the reference's largest magnitude there is 73) and a mean of at most
/// 1.2e-5, over stages 3, 6, 9 and 12. The GPU's result is not bit-identical
/// from run to run. The test prints the figures of every run.
const FEATURE_MAX_ERROR: f32 = 2e-3;

/// The tolerance on the mean absolute difference over a feature; see
/// [`FEATURE_MAX_ERROR`].
const FEATURE_MEAN_ERROR: f32 = 5e-5;

/// Raw little-endian `f32`s.
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Copies `count` floats from `offset` of `buffer` and waits for them.
fn read_back(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    offset: u64,
    count: usize,
) -> Vec<f32> {
    let bytes = (count * 4) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(buffer, offset, &staging, 0, bytes);
    queue.submit([encoder.finish()]);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let mapped = staging.slice(..).get_mapped_range().unwrap();
    floats(&mapped)
}

/// The encoder of Depth Anything V2 Small, built from the real weights by
/// `beatlm::model`, run on the GPU on the reference input, against the four
/// features the original implementation computed.
///
/// The reference features are `[1][1369][384]`: the patch tokens of one image
/// in row-major patch order, which is the `[37][37][384]` NHWC map the graph
/// writes, so the comparison is element by element. The input is stored NCHW
/// and is transposed to NHWC for the graph.
///
/// With timestamp queries on the device, the GPU time of each kernel is
/// measured over a few runs after the one that is compared, and the sum per run
/// is printed with the adapter and backend.
#[test]
#[ignore = "needs `just setup-depth-anything` and `just reference-depth-anything`"]
fn the_encoder_on_the_gpu_matches_the_reference_features() {
    let dir = model_dir();
    let bytes = read_or_explain(&dir.join("model.safetensors"), "setup-depth-anything");
    let archive = Archive::parse(&bytes).expect("the downloaded file is a safetensors file");
    let config = Config::small();
    let inv = inventory(&config).expect("the small configuration is valid");
    let report = inv.check(&archive);
    assert!(report.is_exact(), "{report}");

    let mut builder = Builder::new();
    let enc = encoder(&mut builder, &config, TensorRole::Output).expect("the encoder builds");
    let (graph, weights) = builder.finish();
    let mut uploads: Vec<(TensorId, Vec<f32>)> = weights
        .iter()
        .map(|w| (w.id, w.numbers(&inv, &archive).expect("the weight reads")))
        .collect();

    let nchw = floats(&read_or_explain(
        &dir.join("reference/input.f32"),
        "reference-depth-anything",
    ));
    assert_eq!(nchw.len(), 3 * SIDE * SIDE);
    let mut nhwc = vec![0.0; nchw.len()];
    for c in 0..3 {
        for p in 0..SIDE * SIDE {
            nhwc[p * 3 + c] = nchw[c * SIDE * SIDE + p];
        }
    }
    uploads.push((enc.image, nhwc));

    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .expect("a GPU adapter");
    let info = adapter.get_info();
    let timed = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features: if timed {
            wgpu::Features::TIMESTAMP_QUERY
        } else {
            wgpu::Features::empty()
        },
        ..Default::default()
    }))
    .expect("a device");

    let program = graph.lower(Fusion::On).expect("the encoder lowers");
    let plan = MemoryPlan::new(program.tensors(), &DeviceProfile::webgpu_baseline())
        .expect("the encoder plans on the baseline");
    let arenas = Arenas::new(&device, &plan).expect("the arenas");
    for (id, numbers) in &uploads {
        let p = plan.placement(*id).expect("a placement");
        let data: Vec<u8> = numbers.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert_eq!(data.len() as u64, p.size, "tensor {}", id.get());
        queue.write_buffer(arenas.buffer(p.arena).unwrap(), p.offset, &data);
    }
    let kernels =
        Kernels::new(&device, &plan, &arenas, program.kernels(), &[]).expect("the kernels build");
    let run = |timer: Option<&mut SliceTimer<usize>>, frame: u64| {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        match timer {
            Some(timer) => {
                let mut stamps = timer.begin_frame(FrameIndex::new(frame));
                for k in 0..kernels.len() {
                    let writes = stamps.compute_pass(k).unwrap();
                    kernels.encode(&mut encoder, k, 0, writes).unwrap();
                }
                stamps.resolve(&mut encoder);
                queue.submit([encoder.finish()]);
                timer.start_readback();
            }
            None => {
                for k in 0..kernels.len() {
                    kernels.encode(&mut encoder, k, 0, None).unwrap();
                }
                queue.submit([encoder.finish()]);
            }
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    };
    run(None, 0);

    println!(
        "adapter {} ({:?}, {:?}); {} kernels, {} arenas of {:?} bytes",
        info.name,
        info.backend,
        info.device_type,
        kernels.len(),
        plan.arenas().len(),
        plan.arenas()
    );
    // The reference recorded `reference/encoder_stage_{n}.f32` for each layer
    // `n` of the checkpoint's `out_indices`, which are the encoder's features.
    assert_eq!(enc.features.len(), config.out_indices.len());
    let mut failures = Vec::new();
    for (&stage, &feature) in config.out_indices.iter().zip(&enc.features) {
        let p = plan.placement(feature).unwrap();
        let got = read_back(
            &device,
            &queue,
            arenas.buffer(p.arena).unwrap(),
            p.offset,
            GRID * GRID * HIDDEN,
        );
        let name = format!("reference/encoder_stage_{stage}.f32");
        let want = floats(&read_or_explain(
            &dir.join(&name),
            "reference-depth-anything",
        ));
        assert_eq!(want.len(), GRID * GRID * HIDDEN, "{name}");
        let diffs: Vec<f64> = got
            .iter()
            .zip(&want)
            .map(|(&g, &w)| f64::from((g - w).abs()))
            .collect();
        let max = diffs.iter().copied().fold(0.0, f64::max);
        let mean = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let scale = want.iter().map(|w| f64::from(w.abs())).fold(0.0, f64::max);
        println!(
            "stage {stage:2}: max abs error {max:.3e}, mean abs error {mean:.3e} \
             (largest reference magnitude {scale:.3})"
        );
        if !(max <= f64::from(FEATURE_MAX_ERROR) && mean <= f64::from(FEATURE_MEAN_ERROR)) {
            failures.push(stage);
        }
    }

    if timed {
        let mut timer = SliceTimer::new(&device, &queue, u32::try_from(kernels.len()).unwrap())
            .expect("a timer");
        let mut totals = Vec::new();
        let mut per_kernel: BTreeMap<usize, Vec<Micros>> = BTreeMap::new();
        for frame in 1..=5 {
            run(Some(&mut timer), frame);
            for timing in timer.collect_ready() {
                let total: u64 = timing.compute.iter().map(|(_, d)| d.get()).sum();
                totals.push(total);
                for (k, d) in timing.compute {
                    per_kernel.entry(k).or_default().push(d);
                }
            }
        }
        totals.sort_unstable();
        println!(
            "GPU time of the encoder, summed over its {} kernels' timestamp pairs, per run \
             (sorted, microseconds): {totals:?}",
            kernels.len()
        );
        let mut by_name: BTreeMap<String, u64> = BTreeMap::new();
        for (k, mut ds) in per_kernel {
            ds.sort_unstable();
            let median = ds[ds.len() / 2].get();
            *by_name
                .entry(kernels.label(k).unwrap_or("?").to_owned())
                .or_default() += median;
        }
        println!("median GPU time per kernel name, summed over the encoder (microseconds):");
        for (name, us) in by_name {
            println!("  {name:45} {us}");
        }
    }
    assert!(
        failures.is_empty(),
        "stages {failures:?} exceed max {FEATURE_MAX_ERROR} or mean {FEATURE_MEAN_ERROR}"
    );
}
