//! The depth model's setup in the browser: fetch and check the weights of the
//! model built before the fetch ([`crate::depth::graph`], so that its memory
//! plan can refuse a run first), calibrate it, realise it, and check its
//! depth against the reference, all before the first frame of the loop.
//!
//! Each step is the native one's, from the same code: the reader and the
//! inventory check of `beatlm::load`, the graph of `beatlm::model`, the core's
//! lowering, split and memory plan, the bench's `Net` (`slots.rs`), and
//! `beatlm::load::depth_anything_v2::reference` for the input, the error
//! measures and the tolerances. What changes is how the page waits: on the
//! event loop, between animation frames, where native code polls the device.
//! The crate documentation ("The depth model") says what each step is for.

use anyhow::{Context, Result, bail, ensure};
use beatlm::gpu::SliceTimer;
use beatlm::load::Archive;
use beatlm::load::depth_anything_v2::reference::{self, DepthError};
use beatlm::load::depth_anything_v2::{Config, inventory};
use beatlm_core::cost::{CostModel, Estimator, SliceId};
use beatlm_core::kernel::Program;
use beatlm_core::time::{FrameIndex, Micros};

use super::device::{self, Gpu};
use super::fetch;
use super::frame::SLICE_PRIOR;
use super::wait::{self, callback, next_frame};
use crate::depth::{self, CALIBRATION_FRAMES, MAX_DISPATCH_US, WholeRuns};
use crate::pace::ms_to_us;
use crate::report::Check;
use crate::slots::{Net, Numbers};
use crate::workload::Workload;

/// How long calibration waits for one frame's timings before it gives up, in
/// microseconds: a minute. It is there so a run whose timings never come back
/// fails instead of waiting for ever, not to bound a slow frame: the first
/// frame of each pass is the first use of every pipeline, which a browser may
/// compile then, and that was not measured.
const TIMING_TIMEOUT_US: u64 = 60_000_000;

/// The model at one side, built and given its numbers, with what the report
/// says about the weights.
pub struct Loaded {
    /// The side in pixels.
    pub side: u32,
    /// The model lowered with fusion, not split.
    pub program: Program,
    /// Every weight's numbers, and the reference input in every slot.
    pub numbers: Numbers,
    /// The weights file's size.
    pub weights_bytes: u64,
    /// The tensors the inventory lists.
    pub tensors: usize,
    /// The parameters of the file.
    pub parameters: u64,
}

/// Fetches the weights, checks them against the inventory derived from the
/// Small configuration, and gives `graph`, the model built at its side
/// before the fetch ([`depth::graph`]), its numbers, with every slot's input
/// the reference input of that side.
///
/// # Errors
///
/// No weights file on the server (the message names the recipes), a file that
/// does not parse or does not match the inventory exactly, or a weight that
/// cannot be read from it.
pub async fn load(graph: depth::Graph) -> Result<Loaded> {
    let url = depth::weights_url();
    let bytes = fetch::bytes(&url).await?.with_context(|| {
        format!(
            "the server has no {url}: run `just setup-depth-anything`, then serve with `just \
             serve-web`, which serves the models directory (BEATLM_MODELS_DIR, or models/ at the \
             workspace root) under {}",
            depth::MODELS_URL
        )
    })?;
    let archive =
        Archive::parse(&bytes).with_context(|| format!("{url} is not a safetensors file"))?;
    let config = Config::small();
    let inventory = inventory(&config).context("the Small configuration is not valid")?;
    let report = inventory.check(&archive);
    ensure!(
        report.is_exact(),
        "{url} does not match the model's inventory: {}",
        crate::report::one_line(&report.to_string())
    );

    let depth::Graph {
        side,
        builder,
        program,
        image: input,
        depth: output,
    } = graph;
    let pixels = usize::try_from(side).context("the side does not fit this machine")?;
    let mut weights = Vec::with_capacity(builder.weights().len());
    for weight in builder.weights() {
        let numbers = weight
            .numbers(&inventory, &archive)
            .with_context(|| format!("cannot read weight {}", weight.id.get()))?;
        weights.push((weight.id, numbers));
    }
    let image = reference::to_nhwc(&reference::input(pixels), pixels)
        .context("the reference input is not an image of its side")?;
    Ok(Loaded {
        side,
        program,
        numbers: Numbers {
            input,
            output,
            weights,
            slot_input: Box::new(move |_| image.clone()),
        },
        weights_bytes: u64::try_from(bytes.len()).unwrap_or(u64::MAX),
        tensors: inventory.len(),
        parameters: archive.parameter_count(),
    })
}

/// What calibration measured and decided.
pub struct Calibration {
    /// The median of each whole kernel's counted measurements, in program
    /// order.
    pub durations: Vec<Micros>,
    /// The GPU time of each counted run of the whole program.
    pub runs_us: Vec<u64>,
    /// The program split so each part is expected within
    /// [`MAX_DISPATCH_US`].
    pub program: Program,
    /// The split program's parts measured into the cost model the loop
    /// starts from: slice `i` is part `i`.
    pub cost: CostModel,
    /// The measurements the cost model was given, part by part, in
    /// microseconds, for the report to compare the loop with.
    pub parts_us: Vec<Vec<u64>>,
}

/// Measures each kernel of `whole` realised with `numbers`, splits it so each
/// part is expected within [`MAX_DISPATCH_US`], and measures the parts, each
/// pass [`CALIBRATION_FRAMES`] timed frames of one whole job followed by the
/// synthetic render pass, as the bench's `calibrate.rs` does and for its
/// reasons.
///
/// The bench paces each frame on its grid and waits on the device for its
/// timings; here each frame starts at an animation frame, and the next one
/// starts at the first animation frame after its timings came back, so the
/// GPU runs one job at a time with the display's idle between them, as in the
/// loop.
///
/// The cost model the parts are measured into, which the loop then plans with,
/// reads with `estimator`, the page's `estimator` parameter, as the bench's
/// does with `--estimator`. The split is decided before it from the medians of
/// the whole kernels ([`WholeRuns::durations`]), which no estimator touches, so
/// the parts are the same whichever is chosen.
///
/// # Errors
///
/// A device without timestamp queries, a network or timer that cannot be
/// built, a kernel or part calibration never measured, a lost device, or
/// timings that do not come back.
pub async fn calibrate(
    gpu: &Gpu,
    workload: &Workload,
    whole: &Program,
    numbers: &Numbers,
    estimator: Estimator,
) -> Result<Calibration> {
    ensure!(
        gpu.timestamp_query,
        "the depth model needs timestamp queries, which this device does not have: the loop's \
         scheduler plans each job from measured kernel times, and without them every job would be \
         estimated at the prior for good"
    );
    let mut first = WholeRuns::new(whole.kernels().len());
    {
        let net = device::accepted(&gpu.device, "the network to calibrate", || {
            Net::new(&gpu.device, &gpu.queue, numbers, 1, whole.clone())
        })
        .await?;
        measure(gpu, workload, &net, |frame, compute| {
            first.record(frame, compute);
        })
        .await?;
    }
    let durations = first.durations()?;
    let parts = whole
        .parts_within(&durations, Micros::new(MAX_DISPATCH_US))
        .context("cannot decide the parts")?;
    let program = whole
        .split(&parts)
        .context("cannot split the depth model")?;

    let mut cost = CostModel::new(SLICE_PRIOR, estimator).context("cannot build the cost model")?;
    let mut parts_us = vec![Vec::new(); program.kernels().len()];
    {
        let net = device::accepted(&gpu.device, "the split network to calibrate", || {
            Net::new(&gpu.device, &gpu.queue, numbers, 1, program.clone())
        })
        .await?;
        measure(gpu, workload, &net, |frame, compute| {
            if depth::counted(frame) {
                for &(slice, duration) in compute {
                    // Calibration's, which the frame-scaled estimate keeps
                    // out of the median it scales; the bench's call.
                    cost.record_calibration(slice, duration);
                    let part = usize::try_from(slice.get())
                        .ok()
                        .and_then(|i| parts_us.get_mut(i));
                    if let Some(part) = part {
                        part.push(duration.get());
                    }
                }
            }
        })
        .await?;
    }
    let slices = u32::try_from(program.kernels().len()).context("more parts than a u32 counts")?;
    for part in 0..slices {
        if cost.samples(SliceId::new(part)) == 0 {
            bail!("calibration measured part {part} of the split model no time");
        }
    }
    Ok(Calibration {
        durations,
        runs_us: first.runs().to_vec(),
        program,
        cost,
        parts_us,
    })
}

/// Runs [`CALIBRATION_FRAMES`] timed frames of one whole job on `net`'s slot 0
/// followed by `workload`'s render pass into the canvas, after one untimed
/// frame, and hands `record` each frame's number (counting from 1 for the
/// first timed frame) with its kernels' measured durations.
async fn measure(
    gpu: &Gpu,
    workload: &Workload,
    net: &Net,
    mut record: impl FnMut(u64, &[(SliceId, Micros)]),
) -> Result<()> {
    let kernels = net.kernel_count();
    let mut timer: SliceTimer<SliceId> = SliceTimer::new(&gpu.device, &gpu.queue, kernels)
        .context("cannot build the calibration timer")?;
    for index in 0..=CALIBRATION_FRAMES {
        let started = ms_to_us(next_frame().await?);
        lost()?;
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("beatlm-web calibration"),
            });
        let canvas = canvas_texture(gpu);
        let resolved = {
            // Frame 0 is untimed: it compiles and first-touches everything.
            let timed = index > 0;
            let mut timestamps = timer.begin_frame(FrameIndex::new(index));
            for kernel in 0..kernels {
                let slice = SliceId::new(kernel);
                let writes = if timed {
                    timestamps.compute_pass(slice)?
                } else {
                    None
                };
                net.encode(&mut encoder, 0, slice, writes)?;
            }
            if let Some((_, view)) = &canvas {
                let render = if timed {
                    timestamps.render_pass()?
                } else {
                    None
                };
                workload.encode_render(&mut encoder, view, render);
            }
            timestamps.resolve(&mut encoder)
        };
        gpu.queue.submit([encoder.finish()]);
        timer.submitted(resolved);
        if let Some((texture, _)) = canvas {
            gpu.queue.present(texture);
        }
        // The next frame starts once this one's timings are back, as the
        // bench's waits on the device; the maps complete between callbacks.
        while timer.in_flight() > 0 {
            let now = ms_to_us(next_frame().await?);
            lost()?;
            for timing in timer.collect_ready() {
                record(timing.frame.get(), &timing.compute);
            }
            if now.saturating_sub(started) > TIMING_TIMEOUT_US {
                bail!("calibration frame {index}'s timings did not come back");
            }
        }
    }
    Ok(())
}

/// Fails once the device has been lost.
pub(super) fn lost() -> Result<()> {
    match device::device_lost() {
        Some(lost) => bail!("the device was lost: {lost}"),
        None => Ok(()),
    }
}

/// The canvas's texture for this frame and a view of it, when the canvas has
/// one to give.
pub(super) fn canvas_texture(gpu: &Gpu) -> Option<(wgpu::SurfaceTexture, wgpu::TextureView)> {
    match gpu.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(texture)
        | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => {
            let view = texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default());
            Some((texture, view))
        }
        _ => None,
    }
}

/// Runs every slot's job once, each slot its own submission waited for before
/// the next, so first use is not charged to the loop's first frame; the bench
/// does the same, and says why one submission of every slot is not safe.
///
/// # Errors
///
/// A lost device.
pub async fn warm_up(gpu: &Gpu, net: &Net) -> Result<()> {
    for slot in 0..net.slots() {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("beatlm-web warm-up"),
            });
        for slice in 0..net.kernel_count() {
            net.encode(&mut encoder, slot, SliceId::new(slice), None)?;
        }
        gpu.queue.submit([encoder.finish()]);
        wait::work_done(&gpu.queue).await;
        lost()?;
    }
    Ok(())
}

/// Compares slot 0's depth, which [`warm_up`] computed from the reference
/// input, with the reference made at the same side, when the server has it.
///
/// A reference the server does not have is reported in the [`Check`], and the
/// run goes on: only the comparison is skipped.
///
/// # Errors
///
/// A depth that cannot be read back, or a server that fails in another way
/// than not having a file.
pub async fn check(gpu: &Gpu, net: &Net, numbers: &Numbers, side: u32) -> Result<Check> {
    let pixels = usize::try_from(side).context("the side does not fit this machine")?;
    let count = pixels
        .checked_mul(pixels)
        .context("the side is too large")?;
    let depth = read_back(gpu, net, numbers.output_of(0), count).await?;
    let mut check = Check {
        reference_url: depth::reference_url(side),
        recipe: depth::reference_recipe(side),
        depth_range: range(&depth),
        ..Check::default()
    };
    let url = |file: &str| format!("{}{file}", check.reference_url);

    let manifest_url = url(reference::MANIFEST_FILE);
    let Some(manifest) = fetch::text(&manifest_url).await? else {
        check.missing = Some(manifest_url);
        return Ok(check);
    };
    let shape = reference::manifest_shape(&manifest, reference::DEPTH_NAME);
    if shape.as_deref() != Some(&[1, pixels, pixels][..]) {
        check.problem = Some(format!(
            "{manifest_url} records the depth's shape as {shape:?}, not [1, {side}, {side}]"
        ));
        return Ok(check);
    }
    let input_url = url(reference::INPUT_FILE);
    let Some(input) = fetch::bytes(&input_url).await? else {
        check.missing = Some(input_url);
        return Ok(check);
    };
    check.input_difference = Some(reference::first_difference(
        &input,
        &reference::input(pixels),
    ));
    let depth_url = url(reference::DEPTH_FILE);
    let Some(want) = fetch::bytes(&depth_url).await? else {
        check.missing = Some(depth_url);
        return Ok(check);
    };
    let want = reference::floats(&want);
    check.depth = DepthError::of(&depth, &want);
    if check.depth.is_none() {
        check.problem = Some(format!(
            "{depth_url} holds {} numbers, not the {count} of a depth of {side} pixels",
            want.len()
        ));
    }
    Ok(check)
}

/// The smallest and the largest finite depth, or `(0, 1)` when there is none.
fn range(depth: &[f32]) -> (f32, f32) {
    let (low, high) = depth
        .iter()
        .filter(|v| v.is_finite())
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &v| {
            (l.min(v), h.max(v))
        });
    if low <= high { (low, high) } else { (0.0, 1.0) }
}

/// Copies `count` numbers of tensor `id` out of `net`'s arenas and waits for
/// them.
async fn read_back(
    gpu: &Gpu,
    net: &Net,
    id: beatlm_core::memory::TensorId,
    count: usize,
) -> Result<Vec<f32>> {
    let placement = net
        .plan()
        .placement(id)
        .context("the depth has no placement")?;
    let source = net
        .arenas()
        .buffer(placement.arena)
        .context("the depth's placement names no arena")?;
    let bytes = u64::try_from(count)
        .ok()
        .and_then(|count| count.checked_mul(4))
        .filter(|&bytes| bytes <= placement.size)
        .context("the depth is larger than its placement")?;
    let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("beatlm-web depth readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("beatlm-web depth readback"),
        });
    encoder.copy_buffer_to_buffer(source, placement.offset, &staging, 0, bytes);
    gpu.queue.submit([encoder.finish()]);
    let slice = staging.slice(..);
    callback(|done| slice.map_async(wgpu::MapMode::Read, done))
        .await
        .map_err(|error| anyhow::anyhow!("cannot map the depth for reading: {error}"))?;
    let numbers = {
        let mapped = slice
            .get_mapped_range()
            .map_err(|error| anyhow::anyhow!("cannot read the mapped depth: {error}"))?;
        reference::floats(&mapped)
    };
    staging.unmap();
    Ok(numbers)
}
