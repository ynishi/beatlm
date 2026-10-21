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
//!
//! The reference is made at one input side per run of the recipe:
//! `just reference-depth-anything` at [`NATIVE`], the side the position
//! embedding is stored for, into `reference/`, and
//! `just reference-depth-anything <side>` at any other multiple of 14 into
//! `reference/<side>/`. The comparison at [`OTHER`] needs the second. The
//! input's formula, the file layout, the error measures and the tolerances are
//! `beatlm::load::depth_anything_v2::reference`'s, which the browser host
//! checks a port with too.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use beatlm::beatlm_core::device::DeviceProfile;
use beatlm::beatlm_core::kernel::{Fusion, Graph, Program, TensorRole};
use beatlm::beatlm_core::memory::{MemoryPlan, TensorId};
use beatlm::beatlm_core::time::FrameIndex;
use beatlm::codegen::Kernels;
use beatlm::gpu::{Arenas, SliceTimer};
use beatlm::load::depth_anything_v2::reference::{
    self, DEPTH_MAX_ERROR, DEPTH_MAX_RELATIVE_ERROR, DEPTH_MEAN_ERROR, DepthError,
    FEATURE_MAX_ERROR, FEATURE_MEAN_ERROR, FeatureError, RELATIVE_FLOOR, floats,
};
use beatlm::load::depth_anything_v2::{Config, Role, inventory};
use beatlm::load::{Archive, Inventory, read_file};
use beatlm::model::Builder;
use beatlm::model::depth_anything_v2::{encoder, model};

/// The parameter count the model is published with.
const PUBLISHED_PARAMETERS: u64 = 24_785_089;

/// Side of the reference input, in pixels, that the position embeddings are
/// stored for; the graph uses them as stored.
const NATIVE: usize = 518;

/// The other side the port is compared at, where the graph resamples the
/// position embedding: the input side `beatlm-bench` recommends for an Apple
/// M1 Pro (its crate documentation, "The depth workload", has the
/// measurements).
const OTHER: usize = 266;

/// The sides the GPU timing test runs the whole model at, multiples of 14
/// from the stored side down.
const TIMED_SIDES: [usize; 5] = [518, 434, 350, 266, 182];

/// The model's patch side, in pixels.
const PATCH: usize = 14;

/// The channels of a feature.
const HIDDEN: usize = 384;

/// The model's directory: [`reference::MODEL_DIR`] under `BEATLM_MODELS_DIR`,
/// or under the workspace's `models/` when the variable is unset.
fn model_dir() -> PathBuf {
    std::env::var_os("BEATLM_MODELS_DIR")
        .map_or_else(
            || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models"),
            PathBuf::from,
        )
        .join(reference::MODEL_DIR)
}

/// Where the reference at `side` is ([`reference::dir`]).
fn reference_dir(side: usize) -> PathBuf {
    model_dir().join(reference::dir(&Config::small(), side))
}

/// The recipe that writes the reference at `side`.
fn reference_recipe(side: usize) -> String {
    reference::recipe(&Config::small(), side)
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

/// The reference input of `side` pixels ([`reference::input`]) as the graph's
/// NHWC image.
fn reference_image(side: usize) -> Vec<f32> {
    to_nhwc(&reference::input(side), side)
}

/// A channel-first image of `side` pixels as the graph's NHWC image.
fn to_nhwc(nchw: &[f32], side: usize) -> Vec<f32> {
    reference::to_nhwc(nchw, side).expect("an image of three channels and `side` pixels")
}

#[test]
#[ignore = "needs `just setup-depth-anything`"]
fn the_inventory_matches_the_real_file_exactly() {
    let bytes = read_or_explain(
        &model_dir().join(reference::WEIGHTS_FILE),
        "setup-depth-anything",
    );
    let archive = Archive::parse(&bytes).expect("the downloaded file is a safetensors file");
    let inventory = inventory(&Config::small()).expect("the small configuration is valid");

    let report = inventory.check(&archive);

    assert!(report.is_exact(), "{report}");
    assert_eq!(report.file_parameters, PUBLISHED_PARAMETERS, "{report}");
    assert_eq!(report.expected_parameters, PUBLISHED_PARAMETERS, "{report}");
}

/// Checks that the input the reference script wrote at `side` is the formula's,
/// bit for bit.
fn assert_input_matches_the_formula(side: usize) {
    let bytes = read_or_explain(
        &reference_dir(side).join(reference::INPUT_FILE),
        &reference_recipe(side),
    );
    let expected = reference::input(side);

    assert_eq!(
        bytes.len(),
        expected.len() * 4,
        "the file has the wrong size"
    );
    assert_eq!(
        reference::first_difference(&bytes, &expected),
        None,
        "the input differs from the formula at this byte offset"
    );
}

#[test]
#[ignore = "needs `just reference-depth-anything`"]
fn the_rust_input_is_bit_identical_to_the_one_python_wrote() {
    assert_input_matches_the_formula(NATIVE);
}

/// The checked weights file, its inventory, and an input as the graph's NHWC
/// image.
struct Files {
    weights: Vec<u8>,
    inventory: Inventory<Role>,
    image: Vec<f32>,
}

impl Files {
    /// The weights, and the input the reference script wrote at `side`, which
    /// is stored NCHW.
    fn read(config: &Config, side: usize) -> Self {
        let nchw = floats(&read_or_explain(
            &reference_dir(side).join(reference::INPUT_FILE),
            &reference_recipe(side),
        ));
        Self::with_image(config, to_nhwc(&nchw, side))
    }

    /// The weights, and the reference input's formula at `side`: for a run
    /// that needs no reference.
    fn formula(config: &Config, side: usize) -> Self {
        Self::with_image(config, reference_image(side))
    }

    fn with_image(config: &Config, image: Vec<f32>) -> Self {
        let weights = read_or_explain(
            &model_dir().join(reference::WEIGHTS_FILE),
            "setup-depth-anything",
        );
        let inventory = inventory(config).expect("the small configuration is valid");
        let report = inventory.check(&Archive::parse(&weights).expect("a safetensors file"));
        assert!(report.is_exact(), "{report}");
        Self {
            weights,
            inventory,
            image,
        }
    }

    /// Every weight of a builder's graph with its numbers, and `image` with
    /// the input.
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

/// The encoder of Depth Anything V2 Small at `side` pixels, built from the
/// real weights by `beatlm::model`, run on the GPU on the reference input of
/// that side, against the four features the original implementation computed
/// at that side; panics naming the stages beyond the tolerances.
///
/// The reference features are `[1][G * G][384]`: the patch tokens of one image
/// in row-major patch order, which is the `[G][G][384]` NHWC map the graph
/// writes, so the comparison is element by element. The input is stored NCHW
/// and is transposed to NHWC for the graph.
///
/// With timestamp queries on the device, the GPU time of each kernel is
/// measured over a few runs after the one that is compared, and the sum per run
/// is printed with the adapter and backend.
fn encoder_matches_the_reference_features(side: usize) {
    let dir = reference_dir(side);
    let recipe = reference_recipe(side);
    let grid = side / PATCH;
    let config = Config::small();
    let files = Files::read(&config, side);
    let mut builder = Builder::new();
    let enc = encoder(&mut builder, &config, side, TensorRole::Output).expect("the encoder builds");
    let (graph, uploads) = files.uploads(builder, enc.image);

    let gpu = Gpu::new();
    let loaded = Loaded::new(&gpu, &graph, Fusion::On, &uploads);
    loaded.run(None, 0);
    println!("{side} pixels: {}", loaded.describe());

    // The reference recorded `encoder_stage_{n}.f32` for each layer `n` of the
    // checkpoint's `out_indices`, which are the encoder's features.
    assert_eq!(enc.features.len(), config.out_indices.len());
    let mut failures = Vec::new();
    for (&stage, &feature) in config.out_indices.iter().zip(&enc.features) {
        let got = loaded.read(feature);
        assert_eq!(got.len(), grid * grid * HIDDEN);
        let name = reference::feature_file(stage);
        let want = floats(&read_or_explain(&dir.join(&name), &recipe));
        assert_eq!(want.len(), grid * grid * HIDDEN, "{name}");
        let error = FeatureError::of(&got, &want).expect("a feature of the reference's size");
        println!(
            "{side} pixels, stage {stage:2}: max abs error {:.3e}, mean abs error {:.3e} \
             (largest reference magnitude {:.3})",
            error.max, error.mean, error.scale
        );
        if !error.within_tolerance() {
            failures.push(stage);
        }
    }

    if let Some((totals, medians)) = loaded.timings() {
        println!(
            "{side} pixels: GPU time of the encoder, summed over its {} kernels' timestamp pairs, \
             per run (sorted, microseconds): {totals:?}",
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
        "{side} pixels: stages {failures:?} exceed max {FEATURE_MAX_ERROR} or mean \
         {FEATURE_MEAN_ERROR}"
    );
}

#[test]
#[ignore = "needs `just setup-depth-anything` and `just reference-depth-anything`"]
fn the_encoder_on_the_gpu_matches_the_reference_features() {
    encoder_matches_the_reference_features(NATIVE);
}

/// The whole of Depth Anything V2 Small at `side` pixels, the encoder and the
/// DPT head built from the real weights by
/// `beatlm::model::depth_anything_v2::model`, run on the GPU on the reference
/// input of that side, fused and unfused, against the depth the original
/// implementation predicted at that side; panics naming the fusion settings
/// beyond the tolerances.
///
/// The reference depth is `[1][side][side]`, the shape the manifest records;
/// the graph's is `[side][side][1]`, the same numbers in the same order.
///
/// With timestamp queries on the device, it reports the GPU time of the whole
/// model per run, by kind of kernel for the encoder and the decoder apart, and
/// for the unfused program what the standalone ReLUs that the fused program
/// takes as prologues took.
fn whole_model_matches_the_reference_depth(side: usize) {
    let dir = reference_dir(side);
    let recipe = reference_recipe(side);
    let config = Config::small();
    let files = Files::read(&config, side);
    let manifest = String::from_utf8(read_or_explain(
        &dir.join(reference::MANIFEST_FILE),
        &recipe,
    ))
    .expect("the manifest is text");
    let shape = reference::manifest_shape(&manifest, reference::DEPTH_NAME)
        .expect("the manifest records the depth's shape");
    assert_eq!(shape, [1, side, side]);
    let want = floats(&read_or_explain(&dir.join(reference::DEPTH_FILE), &recipe));
    assert_eq!(want.len(), shape.iter().product::<usize>());
    let (low, high) = want
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| {
            (l.min(v), h.max(v))
        });
    let mean = want.iter().map(|&v| f64::from(v)).sum::<f64>() / want.len() as f64;
    println!("{side} pixels: reference depth from {low:.4} to {high:.4}, mean {mean:.4}");

    // The encoder's operations come first in the model's graph, so a kernel
    // whose operations start past them is the decoder's.
    let encoder_ops = {
        let mut b = Builder::new();
        encoder(&mut b, &config, side, TensorRole::Activation).expect("the encoder builds");
        b.graph().ops().len()
    };
    let mut builder = Builder::new();
    let m = model(&mut builder, &config, side).expect("the model builds");
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
        println!("{side} pixels, {fusion:?}: {}", loaded.describe());
        let error =
            DepthError::of(&loaded.read(m.depth), &want).expect("a depth of the reference's size");
        println!(
            "{side} pixels, {fusion:?}: depth max abs error {:.3e}, mean abs error {:.3e}, max \
             relative error {:.3e} (over pixels at least {RELATIVE_FLOOR} of the largest \
             reference depth, {:.4})",
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
            "{side} pixels, {fusion:?}: GPU time of the model, summed over its {} kernels' \
             timestamp pairs, per run (sorted, microseconds): {totals:?}",
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
        "{side} pixels: {failures:?} exceed max {DEPTH_MAX_ERROR}, mean {DEPTH_MEAN_ERROR} or \
         relative {DEPTH_MAX_RELATIVE_ERROR}"
    );
}

#[test]
#[ignore = "needs `just setup-depth-anything` and `just reference-depth-anything`"]
fn the_whole_model_on_the_gpu_matches_the_reference_depth() {
    whole_model_matches_the_reference_depth(NATIVE);
}

/// The model at [`OTHER`] pixels, where the graph resamples the position
/// embedding on the host and every size of the decoder follows the smaller
/// patch grid, against the reference made at that side: the input bit for
/// bit, the four features and the depth, within the tolerances of the stored
/// side.
#[test]
#[ignore = "needs `just setup-depth-anything` and `just reference-depth-anything 266`"]
fn the_model_at_another_side_matches_the_reference_made_at_that_side() {
    assert_input_matches_the_formula(OTHER);
    encoder_matches_the_reference_features(OTHER);
    whole_model_matches_the_reference_depth(OTHER);
}

/// The whole model, fused, from the real weights at each of [`TIMED_SIDES`],
/// on the reference input's formula at that side: every side builds, plans on
/// the baseline and runs, and gives a depth of `side * side` finite,
/// non-negative numbers. With timestamp queries it prints the GPU time of five
/// runs back to back after one untimed run, which is what a side costs per
/// result before the frame loop spreads it; `beatlm-bench --workload depth`
/// measures it paced. For the times, run it alone (name it after
/// `--ignored`): run with this file's other GPU tests, which the test harness
/// runs on other threads at the same time, it shares the GPU with them.
#[test]
#[ignore = "needs `just setup-depth-anything` and a GPU adapter"]
fn the_whole_model_runs_on_the_gpu_at_every_timed_side() {
    let config = Config::small();
    let gpu = Gpu::new();
    // The weights are read and checked once; only the image changes with the
    // side.
    let mut files = Files::formula(&config, TIMED_SIDES[0]);
    for side in TIMED_SIDES {
        files.image = reference_image(side);
        let mut builder = Builder::new();
        let m = model(&mut builder, &config, side).expect("the model builds");
        let (graph, uploads) = files.uploads(builder, m.image);
        let loaded = Loaded::new(&gpu, &graph, Fusion::On, &uploads);
        loaded.run(None, 0);
        let depth = loaded.read(m.depth);
        assert_eq!(depth.len(), side * side, "{side} pixels");
        assert!(
            depth.iter().all(|v| v.is_finite() && *v >= 0.0),
            "{side} pixels: a depth that is not a finite, non-negative number"
        );
        let (low, high) = depth
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| {
                (l.min(v), h.max(v))
            });
        println!(
            "{side} pixels: {}; depth from {low:.4} to {high:.4}",
            loaded.describe()
        );
        if let Some((totals, _)) = loaded.timings() {
            println!(
                "{side} pixels: GPU time of the model per run, summed over its kernels' \
                 timestamp pairs (sorted, microseconds): {totals:?}"
            );
        }
    }
}
