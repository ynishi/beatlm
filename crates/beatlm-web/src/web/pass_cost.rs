//! The page's host of the bench's `pass-cost` workload
//! ([`beatlm_host::pass_cost`]): the device, the frames from
//! `requestAnimationFrame`, and the CPU clock, `performance.now()`.
//!
//! It is a measurement loop like calibration's ([`super::model`]): an `async`
//! task that awaits each animation frame, collects the timings that have come
//! back, and runs the next frame only once nothing is in flight, so one frame
//! is in flight at a time, as natively (`beatlm_host::pass_cost` says why). The map
//! completes between callbacks, so a frame whose GPU work fits the period
//! comes back by the next callback and the frames keep the display's rate;
//! a longer one gives up the callbacks it overran. Nothing is scheduled.

use anyhow::{Context, Result, ensure};
use beatlm::gpu::SliceTimer;
use beatlm_core::time::FrameIndex;
use beatlm_host::pass_cost::{self, Dispatches, Measurements, Settings, Source, Trivial};
use beatlm_host::slots::{self, Net};
use beatlm_host::workload::{DEFAULT_RENDER_ITERS, DEFAULT_SLICE_ITERS, Workload};

use super::device::{self, Need};
use super::model::{self, canvas_texture, lost};
use super::page;
use super::wait::next_frame;
use crate::config::Config;
use crate::pace::ms_to_us;
use crate::report::{PassCostOutcome, render_pass_cost};

/// How long the page waits for a frame's timings before it gives up, in
/// milliseconds of `performance.now()`: far past any frame here, so only a
/// readback that never completes ends the run.
const TIMING_TIMEOUT_MS: f64 = 5_000.0;

/// Opens the device, builds the dispatches `settings` names, runs every cell
/// for `config.frames` timed frames and writes the report into the page.
///
/// # Errors
///
/// A device without timestamp queries, a depth model that cannot be fetched,
/// built or holds fewer kernels than a pass count, a timer that cannot be
/// built, or a lost device.
pub async fn run(config: &Config, settings: &Settings) -> Result<()> {
    // Every figure is read from timestamps, so a device without them is
    // refused as soon as it opens, before the model is built on it.
    let timed = |gpu: &device::Gpu| {
        ensure!(
            gpu.timestamp_query,
            "the pass-cost measurement needs timestamp queries, which this device does not \
             have: every figure it reports is read from them"
        );
        Ok(())
    };
    let (gpu, net) = match settings.dispatches {
        Dispatches::Trivial { .. } => {
            let gpu = device::open(page::canvas()?, Need::Defaults).await?;
            timed(&gpu)?;
            (gpu, None)
        }
        Dispatches::Depth { side } => {
            page::show(&format!("status: fetching {}", crate::depth::weights_url()));
            let loaded = model::load(crate::depth::graph(side)?).await?;
            settings.check_kernels(loaded.program.kernels().len())?;
            let plan = slots::memory_plan(&loaded.program, &loaded.numbers, 1)?;
            let gpu = device::open(page::canvas()?, Need::Plan(plan)).await?;
            timed(&gpu)?;
            let net = device::accepted(&gpu.device, "the depth model's network", || {
                Net::new(&gpu.device, &gpu.queue, &loaded.numbers, 1, loaded.program)
            })
            .await?;
            model::warm_up(&gpu, &net).await?;
            (gpu, Some(net))
        }
    };
    let workload = Workload::new(&gpu.device, DEFAULT_SLICE_ITERS, DEFAULT_RENDER_ITERS);
    super::warm_up(&gpu, &workload).await?;
    let env = super::environment(&gpu)?;
    let trivial = match settings.dispatches {
        Dispatches::Trivial { groups } => Some(Trivial::new(&gpu.device, groups)),
        Dispatches::Depth { .. } => None,
    };
    let source = match (&trivial, &net) {
        (Some(trivial), _) => Source::Trivial(trivial),
        (None, Some(net)) => Source::Net(net),
        (None, None) => anyhow::bail!("nothing to dispatch"),
    };

    let mut measurements = Measurements::new(settings.cells(), config.frames);
    let cells = measurements.cells().to_vec();
    let mut timer: SliceTimer<usize> =
        SliceTimer::new(&gpu.device, &gpu.queue, settings.capacity())
            .context("cannot build the pass-cost timer")?;
    let mut errors = device::ErrorWatch::from_now();
    let performance = page::window()?
        .performance()
        .context("the page has no performance clock")?;
    let total = measurements.total_frames();
    page::show(&format!(
        "status: running model=pass-cost frames={} {}: {total} frames",
        config.frames,
        settings.describe()
    ));
    let mut outcome = PassCostOutcome::default();
    let mut submitted_at = performance.now();
    let mut n = 0;
    while n < total || timer.in_flight() > 0 {
        next_frame().await?;
        lost()?;
        collect(&mut errors, &mut timer, &mut measurements);
        if timer.in_flight() > 0 {
            if performance.now() - submitted_at > TIMING_TIMEOUT_MS {
                anyhow::bail!("frame {}'s timings did not come back", n.saturating_sub(1));
            }
            continue;
        }
        if n >= total {
            break;
        }
        let (index, untimed) = measurements.cell_of(n);
        n += 1;
        let Some(&cell) = cells.get(index) else {
            continue;
        };
        // Before the encode span, so the push is not charged to the cell's
        // `encode` time, which the native bench's has no push in.
        let submission = device::Submission::open(&gpu.device);
        let encoding = performance.now();
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("beatlm-web pass-cost"),
            });
        let canvas = if settings.render {
            canvas_texture(&gpu)
        } else {
            None
        };
        if settings.render && canvas.is_none() {
            outcome.canvas_unavailable += 1;
        }
        let mut timestamps = timer.begin_frame(FrameIndex::new(n - 1));
        let timed = (!untimed).then_some(&mut timestamps);
        pass_cost::encode(&mut encoder, source, cell, index, timed)?;
        if let Some((_, view)) = &canvas {
            workload.encode_render(&mut encoder, view, None);
        }
        let resolved = timestamps.resolve(&mut encoder);
        let commands = encoder.finish();
        let submitting = performance.now();
        gpu.queue.submit([commands]);
        let submitted = performance.now();
        submission.submitted(Some(&mut timer), Some(resolved));
        if let Some((texture, _)) = canvas {
            gpu.queue.present(texture);
        }
        if !untimed {
            measurements.record_cpu(
                index,
                nanos(submitting - encoding),
                nanos(submitted - submitting),
            );
        }
        submitted_at = submitted;
    }
    outcome.frames = total;
    outcome.timer = timer.stats();
    // A frame the timer did not time is not in its count in flight, so its
    // scope may still be settling; `gpu_errors` waits for every frame's.
    let settling = performance.now();
    while device::scopes_pending() > 0 && performance.now() - settling <= TIMING_TIMEOUT_MS {
        next_frame().await?;
    }
    outcome.gpu_errors = device::gpu_errors();
    page::show(&render_pass_cost(
        &env,
        config,
        settings,
        &measurements,
        &outcome,
    ));
    Ok(())
}

/// Records every frame whose timing has come back, under the cell its passes
/// are labelled with; a frame the device may have rejected is never among
/// them ([`device::ErrorWatch`]).
fn collect(
    errors: &mut device::ErrorWatch,
    timer: &mut SliceTimer<usize>,
    measurements: &mut Measurements,
) {
    for timing in errors.collect_ready(timer) {
        if let Some(&(index, _)) = timing.compute.first() {
            measurements.record_spans(index, &timing.compute_spans);
        }
    }
}

/// A duration on `performance.now()`'s clock, in milliseconds, as whole
/// nanoseconds; the clock's own resolution is far coarser.
fn nanos(ms: f64) -> u64 {
    ms_to_us(ms).saturating_mul(1_000)
}
