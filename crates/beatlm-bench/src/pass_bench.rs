//! The native host of the `pass-cost` workload (`pass_cost.rs`): the frames,
//! paced on the bench's 60 Hz grid, and the CPU clock.
//!
//! Each frame is paced as the loop's are (`pace.rs`), so the GPU is as idle
//! between frames as in the loop, and runs the cell `pass_cost.rs` assigns it.
//! After its submission the frame waits for the device and collects its own
//! timings, as calibration's frames do, so one frame is in flight at a time
//! (`pass_cost.rs` says why). The wait comes after the CPU times are read.
//! Nothing here schedules: the frame's work is fixed by its cell.

use std::time::Instant;

use anyhow::{Context, Result};
use beatlm::gpu::{SliceTimer, TimerStats};
use beatlm_core::time::FrameIndex;

use crate::args::Args;
use crate::device::Gpu;
use crate::engine::offscreen_target;
use crate::pace::Pacer;
use crate::pass_cost::{self, Dispatches, Measurements, Source, Trivial};
use crate::slots::Net;
use crate::workload::Workload;

/// What a run of the workload measured.
pub struct PassOutcome {
    /// Every cell's samples.
    pub measurements: Measurements,
    /// Frames encoded, the untimed round included.
    pub frames: u64,
    /// Frame periods given up because a frame began a whole period late.
    pub dropped: u64,
    /// What the timer could and could not read.
    pub timer: TimerStats,
}

/// Runs `args.pass_cost`'s cells for `args.frames` timed frames each on
/// `gpu`.
///
/// # Errors
///
/// No weights file for `--kernels depth`, a pass count above its kernels, a
/// network or timer that cannot be built, or a device that cannot be polled.
pub fn run(gpu: &Gpu, args: &Args) -> Result<PassOutcome> {
    let settings = &args.pass_cost;
    let (device, queue) = (&gpu.device, &gpu.queue);
    let workload = Workload::new(device, args.slice_iters, args.render_iters);
    let target = offscreen_target(device);
    let trivial;
    let net;
    let source = match settings.dispatches {
        Dispatches::Trivial { groups } => {
            trivial = Trivial::new(device, groups);
            Source::Trivial(&trivial)
        }
        Dispatches::Depth { side } => {
            let (program, numbers) = crate::depth::load(side, args.fusion)?;
            settings.check_kernels(program.kernels().len())?;
            net = Net::new(device, queue, &numbers, 1, program)
                .context("cannot build the depth model")?;
            Source::Net(&net)
        }
    };
    let mut measurements = Measurements::new(settings.cells(), args.frames);
    let cells = measurements.cells().to_vec();
    let mut timer: SliceTimer<usize> = SliceTimer::new(device, queue, settings.capacity())
        .context("cannot build the pass-cost timer")?;

    // The pacer counts periods, and a frame longer than the period gives up
    // the ones it overran; the run counts frames, so every cell gets its own.
    let total = measurements.total_frames();
    let mut pacer = Pacer::start(u64::MAX);
    let mut frames = 0;
    while frames < total {
        let Some(tick) = pacer.next_frame() else {
            break;
        };
        let (index, untimed) = measurements.cell_of(frames);
        frames += 1;
        let Some(&cell) = cells.get(index) else {
            continue;
        };
        let encoding = Instant::now();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("beatlm-bench pass-cost"),
        });
        let mut timestamps = timer.begin_frame(FrameIndex::new(tick.index));
        let timed = (!untimed).then_some(&mut timestamps);
        pass_cost::encode(&mut encoder, source, cell, index, timed)?;
        if settings.render {
            workload.encode_render(&mut encoder, &target, None);
        }
        timestamps.resolve(&mut encoder);
        let commands = encoder.finish();
        let submitting = Instant::now();
        queue.submit([commands]);
        let submitted = Instant::now();
        if !untimed {
            measurements.record_cpu(
                index,
                nanos_between(encoding, submitting),
                nanos_between(submitting, submitted),
            );
        }
        timer.start_readback();
        read_back(gpu, &mut timer, &mut measurements)?;
    }
    let (dropped, _) = pacer.finish();
    Ok(PassOutcome {
        measurements,
        frames,
        dropped,
        timer: timer.stats(),
    })
}

/// Waits for the device and records every frame whose timing has come back,
/// under the cell its passes are labelled with.
fn read_back(
    gpu: &Gpu,
    timer: &mut SliceTimer<usize>,
    measurements: &mut Measurements,
) -> Result<()> {
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .context("device poll failed")?;
    for timing in timer.collect_ready() {
        if let Some(&(index, _)) = timing.compute.first() {
            measurements.record_spans(index, &timing.compute_spans);
        }
    }
    Ok(())
}

/// Nanoseconds from `start` to `end`, zero if `end` is not later.
fn nanos_between(start: Instant, end: Instant) -> u64 {
    u64::try_from(end.saturating_duration_since(start).as_nanos()).unwrap_or(u64::MAX)
}
