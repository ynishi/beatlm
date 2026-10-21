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
use beatlm::beatlm_core::kernel::{Fusion, Graph, Program, TensorRole};
use beatlm::beatlm_core::memory::{MemoryPlan, TensorId};
use beatlm::beatlm_core::time::FrameIndex;
use beatlm::codegen::Kernels;
use beatlm::gpu::{Arenas, SliceTimer};
use beatlm::load::depth_anything_v2::{Config, Role, inventory};
use beatlm::load::{Archive, Inventory, read_file};
use beatlm::model::Builder;
use beatlm::model::depth_anything_v2::{encoder, model};

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

/// The checked weights file, its inventory, and the reference input as the
/// graph's NHWC image. The input is stored NCHW.
struct Files {
    weights: Vec<u8>,
    inventory: Inventory<Role>,
    image: Vec<f32>,
}

impl Files {
    fn read(config: &Config) -> Self {
        let dir = model_dir();
        let weights = read_or_explain(&dir.join("model.safetensors"), "setup-depth-anything");
        let inventory = inventory(config).expect("the small configuration is valid");
        let report = inventory.check(&Archive::parse(&weights).expect("a safetensors file"));
        assert!(report.is_exact(), "{report}");
        let nchw = floats(&read_or_explain(
            &dir.join("reference/input.f32"),
            "reference-depth-anything",
        ));
        assert_eq!(nchw.len(), 3 * SIDE * SIDE);
        let mut image = vec![0.0; nchw.len()];
        for c in 0..3 {
            for p in 0..SIDE * SIDE {
                image[p * 3 + c] = nchw[c * SIDE * SIDE + p];
            }
        }
        Self {
            weights,
            inventory,
            image,
        }
    }

    /// Every weight of a builder's graph with its numbers, and `image` with
    /// the reference input.
    fn uploads(&self, b: Builder<Role>, image: TensorId) -> (Graph, Vec<(TensorId, Vec<f32>)>) {
        let archive = Archive::parse(&self.weights).expect("a safetensors file");
        let (graph, weights) = b.finish();
        let mut uploads: Vec<(TensorId, Vec<f32>)> = weights
            .iter()
            .map(|w| {
                let numbers = w.numbers(&self.inventory, &archive);
                (w.id, numbers.expect("the weight reads"))
            })
            .collect();
        uploads.push((image, self.image.clone()));
        (graph, uploads)
    }
}

/// The adapter's device, with timestamp queries when the adapter has them.
struct Gpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    timed: bool,
}

impl Gpu {
    fn new() -> Self {
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
        Self {
            device,
            queue,
            info,
            timed,
        }
    }

    /// Copies `count` floats from `offset` of `buffer` and waits for them.
    fn read_back(&self, buffer: &wgpu::Buffer, offset: u64, count: usize) -> Vec<f32> {
        let bytes = (count * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer, offset, &staging, 0, bytes);
        self.queue.submit([encoder.finish()]);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.unwrap());
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let mapped = staging.slice(..).get_mapped_range().unwrap();
        floats(&mapped)
    }
}

/// A graph lowered with one fusion setting, planned on the baseline, its
/// inputs and weights written and its kernels built: ready to run.
struct Loaded<'g> {
    gpu: &'g Gpu,
    program: Program,
    plan: MemoryPlan,
    arenas: Arenas,
    kernels: Kernels,
}

impl<'g> Loaded<'g> {
    fn new(gpu: &'g Gpu, graph: &Graph, fusion: Fusion, uploads: &[(TensorId, Vec<f32>)]) -> Self {
        let program = graph.lower(fusion).expect("the graph lowers");
        let plan = MemoryPlan::new(program.tensors(), &DeviceProfile::webgpu_baseline())
            .expect("the graph plans on the baseline");
        let arenas = Arenas::new(&gpu.device, &plan).expect("the arenas");
        for (id, numbers) in uploads {
            let p = plan.placement(*id).expect("a placement");
            let data: Vec<u8> = numbers.iter().flat_map(|v| v.to_le_bytes()).collect();
            assert_eq!(data.len() as u64, p.size, "tensor {}", id.get());
            gpu.queue
                .write_buffer(arenas.buffer(p.arena).unwrap(), p.offset, &data);
        }
        let kernels = Kernels::new(&gpu.device, &plan, &arenas, program.kernels(), &[])
            .expect("the kernels build");
        Self {
            gpu,
            program,
            plan,
            arenas,
            kernels,
        }
    }

    /// Encodes every kernel once, in one submission, timed when `timer` is
    /// given, and waits for it.
    fn run(&self, timer: Option<&mut SliceTimer<usize>>, frame: u64) {
        let device = &self.gpu.device;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        match timer {
            Some(timer) => {
                let mut stamps = timer.begin_frame(FrameIndex::new(frame));
                for k in 0..self.kernels.len() {
                    let writes = stamps.compute_pass(k).unwrap();
                    self.kernels.encode(&mut encoder, k, 0, writes).unwrap();
                }
                stamps.resolve(&mut encoder);
                self.gpu.queue.submit([encoder.finish()]);
                timer.start_readback();
            }
            None => {
                for k in 0..self.kernels.len() {
                    self.kernels.encode(&mut encoder, k, 0, None).unwrap();
                }
                self.gpu.queue.submit([encoder.finish()]);
            }
        }
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }

    /// The numbers of tensor `id` after a run.
    fn read(&self, id: TensorId) -> Vec<f32> {
        let p = self.plan.placement(id).expect("a placement");
        let elements = self
            .program
            .tensors()
            .iter()
            .find(|t| t.id == id)
            .expect("a tensor of the program")
            .elements;
        let buffer = self.arenas.buffer(p.arena).unwrap();
        self.gpu.read_back(buffer, p.offset, elements as usize)
    }

    /// With timestamp queries, five timed runs: the GPU time of each run, the
    /// sum of its kernels' timestamp pairs, sorted, and each kernel's median
    /// over the runs, in microseconds. `None` without timestamp queries.
    fn timings(&self) -> Option<(Vec<u64>, Vec<u64>)> {
        if !self.gpu.timed {
            return None;
        }
        let count = u32::try_from(self.kernels.len()).unwrap();
        let mut timer = SliceTimer::new(&self.gpu.device, &self.gpu.queue, count).expect("a timer");
        let mut totals = Vec::new();
        let mut per_kernel: Vec<Vec<u64>> = vec![Vec::new(); self.kernels.len()];
        for frame in 1..=5 {
            self.run(Some(&mut timer), frame);
            for timing in timer.collect_ready() {
                totals.push(timing.compute.iter().map(|(_, d)| d.get()).sum());
                for (k, d) in timing.compute {
                    per_kernel[k].push(d.get());
                }
            }
        }
        totals.sort_unstable();
        let medians = per_kernel
            .into_iter()
            .map(|mut ds| {
                ds.sort_unstable();
                ds.get(ds.len() / 2).copied().unwrap_or(0)
            })
            .collect();
        Some((totals, medians))
    }

    /// The adapter and the memory plan, for the report.
    fn describe(&self) -> String {
        format!(
            "adapter {} ({:?}, {:?}); {} kernels, {} tensors; {} arenas of {:?} bytes, {} in all, \
             {} bytes live at most",
            self.gpu.info.name,
            self.gpu.info.backend,
            self.gpu.info.device_type,
            self.kernels.len(),
            self.program.tensors().len(),
            self.plan.arenas().len(),
            self.plan.arenas(),
            self.plan.total_bytes(),
            self.plan.peak_live_bytes()
        )
    }
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
    let config = Config::small();
    let files = Files::read(&config);
    let mut builder = Builder::new();
    let enc = encoder(&mut builder, &config, TensorRole::Output).expect("the encoder builds");
    let (graph, uploads) = files.uploads(builder, enc.image);

    let gpu = Gpu::new();
    let loaded = Loaded::new(&gpu, &graph, Fusion::On, &uploads);
    loaded.run(None, 0);
    println!("{}", loaded.describe());

    // The reference recorded `reference/encoder_stage_{n}.f32` for each layer
    // `n` of the checkpoint's `out_indices`, which are the encoder's features.
    assert_eq!(enc.features.len(), config.out_indices.len());
    let mut failures = Vec::new();
    for (&stage, &feature) in config.out_indices.iter().zip(&enc.features) {
        let got = loaded.read(feature);
        assert_eq!(got.len(), GRID * GRID * HIDDEN);
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

    if let Some((totals, medians)) = loaded.timings() {
        println!(
            "GPU time of the encoder, summed over its {} kernels' timestamp pairs, per run \
             (sorted, microseconds): {totals:?}",
            loaded.kernels.len()
        );
        let mut by_name: BTreeMap<String, u64> = BTreeMap::new();
        for (k, median) in medians.into_iter().enumerate() {
            *by_name
                .entry(loaded.kernels.label(k).unwrap_or("?").to_owned())
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

/// The shape the reference's manifest records for tensor `name`.
///
/// The manifest is the JSON `scripts/reference/depth_anything_v2.py` writes,
/// one object per tensor with its `name`, `file` and `shape`; this reads the
/// first `[...]` after the name's entry, which is that tensor's shape, rather
/// than take a JSON parser as a dependency for one array of integers.
fn manifest_shape(manifest: &str, name: &str) -> Vec<usize> {
    let entry = format!("\"name\": \"{name}\"");
    let at = manifest
        .find(&entry)
        .unwrap_or_else(|| panic!("the manifest lists no {name}"));
    let rest = &manifest[at..];
    let shape = rest.find("\"shape\"").expect("the entry has a shape");
    let open = rest[shape..].find('[').unwrap() + shape;
    let close = rest[open..].find(']').unwrap() + open;
    rest[open + 1..close]
        .split(',')
        .map(|d| d.trim().parse().expect("a dimension"))
        .collect()
}

/// The tolerance on the largest absolute difference of the depth.
///
/// The depth inherits the encoder's differences from the reference (see
/// [`FEATURE_MAX_ERROR`]) and adds the head's own, of convolutions summing in
/// another order than PyTorch's. Set against what the test measured on an
/// Apple M1 Pro, Metal, on 2026-10-21, in two invocations, identical in both:
/// a largest difference of 5.96e-6 fused and 6.68e-6 unfused, a mean of
/// 1.02e-6 and 1.01e-6, and a largest relative difference of 2.05e-6 and
/// 2.12e-6, on a reference depth from 1.48 to 3.64. The tolerances leave room
/// for an encoder at the edge of its own, three times what it measured, and
/// as much again for the head on another adapter.
const DEPTH_MAX_ERROR: f32 = 5e-5;

/// The tolerance on the mean absolute difference of the depth; see
/// [`DEPTH_MAX_ERROR`].
const DEPTH_MEAN_ERROR: f32 = 1e-5;

/// The tolerance on the largest relative difference of the depth, measured
/// over the pixels whose reference depth is at least [`RELATIVE_FLOOR`] of
/// its largest; see [`DEPTH_MAX_ERROR`].
const DEPTH_MAX_RELATIVE_ERROR: f32 = 2e-5;

/// The share of the largest reference depth below which a pixel's relative
/// error is not counted: the final ReLU clips depth to zero, and near zero a
/// difference of rounding size is a large ratio that says nothing about the
/// port.
const RELATIVE_FLOOR: f64 = 0.01;

/// What a [`Loaded`] model's depth differs from the reference by.
struct DepthError {
    max: f64,
    mean: f64,
    max_relative: f64,
    /// The reference's largest depth.
    scale: f64,
}

impl DepthError {
    fn of(got: &[f32], want: &[f32]) -> Self {
        assert_eq!(got.len(), want.len());
        let scale = want.iter().map(|w| f64::from(w.abs())).fold(0.0, f64::max);
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
        Self {
            max,
            mean: sum / got.len() as f64,
            max_relative,
            scale,
        }
    }

    fn within_tolerance(&self) -> bool {
        self.max <= f64::from(DEPTH_MAX_ERROR)
            && self.mean <= f64::from(DEPTH_MEAN_ERROR)
            && self.max_relative <= f64::from(DEPTH_MAX_RELATIVE_ERROR)
    }
}

/// The whole of Depth Anything V2 Small, the encoder and the DPT head built
/// from the real weights by `beatlm::model::depth_anything_v2::model`, run on
/// the GPU on the reference input, fused and unfused, against the depth the
/// original implementation predicted.
///
/// The reference depth is `[1][518][518]`, the shape the manifest records;
/// the graph's is `[518][518][1]`, the same numbers in the same order.
///
/// With timestamp queries on the device, it reports the GPU time of the whole
/// model per run, by kind of kernel for the encoder and the decoder apart, and
/// for the unfused program what the standalone ReLUs that the fused program
/// takes as prologues took.
#[test]
#[ignore = "needs `just setup-depth-anything` and `just reference-depth-anything`"]
fn the_whole_model_on_the_gpu_matches_the_reference_depth() {
    let dir = model_dir();
    let config = Config::small();
    let files = Files::read(&config);
    let manifest = String::from_utf8(read_or_explain(
        &dir.join("reference/manifest.json"),
        "reference-depth-anything",
    ))
    .expect("the manifest is text");
    let shape = manifest_shape(&manifest, "predicted_depth");
    assert_eq!(shape, [1, SIDE, SIDE]);
    let want = floats(&read_or_explain(
        &dir.join("reference/predicted_depth.f32"),
        "reference-depth-anything",
    ));
    assert_eq!(want.len(), shape.iter().product::<usize>());
    let (low, high) = want
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| {
            (l.min(v), h.max(v))
        });
    let mean = want.iter().map(|&v| f64::from(v)).sum::<f64>() / want.len() as f64;
    println!("reference depth: from {low:.4} to {high:.4}, mean {mean:.4}");

    // The encoder's operations come first in the model's graph, so a kernel
    // whose operations start past them is the decoder's.
    let encoder_ops = {
        let mut b = Builder::new();
        encoder(&mut b, &config, TensorRole::Activation).expect("the encoder builds");
        b.graph().ops().len()
    };
    let mut builder = Builder::new();
    let m = model(&mut builder, &config).expect("the model builds");
    let (graph, uploads) = files.uploads(builder, m.image);

    let gpu = Gpu::new();
    let mut failures = Vec::new();
    // The graph operations the fused program applies as convolution
    // prologues: in the unfused program, which has one kernel per operation,
    // they are the kernels of those indices.
    let mut prologue_ops: Vec<usize> = Vec::new();
    for fusion in [Fusion::On, Fusion::Off] {
        let loaded = Loaded::new(&gpu, &graph, fusion, &uploads);
        loaded.run(None, 0);
        println!("{fusion:?}: {}", loaded.describe());
        let error = DepthError::of(&loaded.read(m.depth), &want);
        println!(
            "{fusion:?}: depth max abs error {:.3e}, mean abs error {:.3e}, max relative error \
             {:.3e} (over pixels at least {RELATIVE_FLOOR} of the largest reference depth, {:.4})",
            error.max, error.mean, error.max_relative, error.scale
        );
        if !error.within_tolerance() {
            failures.push(fusion);
        }
        let kernels = loaded.program.kernels();
        if fusion == Fusion::On {
            prologue_ops = kernels
                .iter()
                .filter(|k| !k.prologue().is_empty())
                .map(|k| k.ops().start)
                .collect();
        }

        let Some((totals, medians)) = loaded.timings() else {
            continue;
        };
        println!(
            "{fusion:?}: GPU time of the model, summed over its {} kernels' timestamp pairs, per \
             run (sorted, microseconds): {totals:?}",
            kernels.len()
        );
        // Kind of kernel, by encoder or decoder: count and summed medians.
        let mut by_kind: BTreeMap<(&str, &str), (usize, u64)> = BTreeMap::new();
        for (spec, &median) in kernels.iter().zip(&medians) {
            let part = if spec.ops().start < encoder_ops {
                "encoder"
            } else {
                "decoder"
            };
            let entry = by_kind.entry((part, spec.main().name())).or_default();
            entry.0 += 1;
            entry.1 += median;
        }
        let mut per_part: BTreeMap<&str, u64> = BTreeMap::new();
        println!("{fusion:?}: median GPU time per kind of kernel (microseconds):");
        for ((part, kind), (count, us)) in &by_kind {
            *per_part.entry(part).or_default() += us;
            println!("  {part:8} {kind:18} {count:4} kernels {us:8}");
        }
        println!("{fusion:?}: summed medians by part (microseconds): {per_part:?}");
        if fusion == Fusion::Off {
            let relus: Vec<u64> = prologue_ops.iter().map(|&op| medians[op]).collect();
            println!(
                "Off: the {} standalone ReLUs the fused program takes as prologues took {} \
                 microseconds in all (medians, each {relus:?})",
                relus.len(),
                relus.iter().sum::<u64>()
            );
        }
    }
    assert!(
        failures.is_empty(),
        "{failures:?} exceed max {DEPTH_MAX_ERROR}, mean {DEPTH_MEAN_ERROR} or relative \
         {DEPTH_MAX_RELATIVE_ERROR}"
    );
}
