//! Calibration: what each kernel of the network, or of the depth model, takes
//! run whole on this device, the parts that keep each dispatch within
//! `--max-dispatch-us`, and what each part takes, all measured before the loop.
//! "The network" below is either; both are realised through `Net`.
//!
//! Two paced passes, each [`CALIBRATION_FRAMES`] timed frames after one untimed
//! frame that pays for first use:
//!
//! 1. **Whole kernels, to decide.** The network is built unsplit on one slot
//!    and run as one whole job followed by the synthetic render pass per frame,
//!    each frame submitted and waited for, reading every kernel's timed compute
//!    pass. A kernel's duration is the [`median`] of its measurements in the
//!    second half of the timed frames, whatever `--estimator` is. The decision
//!    is the core's (`Program::parts_within`, which owns the rule); this module
//!    only measures. A kernel is split for what it usually takes after the GPU
//!    has run the paced load for half a second, not for what an estimator that
//!    also reads the first half, or pads by the deviation, would have made of
//!    it.
//! 2. **The split program, to start the loop from.** The program the decision
//!    produced is built on one slot and run the same way, and every part's
//!    measured durations in the second half of the timed frames, as in pass 1,
//!    go into a [`CostModel`] built with the run's `--estimator`, which the loop
//!    then uses. Each part is a new slice to the
//!    loop's cost model, which starts every unmeasured slice at the same prior
//!    whatever it costs, so a job of many parts would have been estimated at
//!    many priors until each had been measured; measured here, each part starts
//!    at what it took. The cost model's rule is unchanged: its estimates still
//!    change only through measurements, and these are measurements.
//!
//! Both passes are paced at the loop's frame period, because a kernel is
//! measured as the loop will run it: behind the same render pass, with the GPU
//! as busy per frame and as idle between frames as in the loop. The pacing is
//! not optional. Run back to back without it, on an Apple M1 Pro (Metal) on
//! 2026-10-21, every kernel of the fused network measured two to four times
//! shorter than the same kernel in the paced loop that followed (the first
//! convolution 0.20 to 0.33 ms against 0.66 to 1.09 ms in the loop, the second
//! matrix multiply 1.04 ms against 2.26 ms), so a bound of 1 ms split the
//! matrix multiply into two parts that each still took about 1.1 ms in the
//! loop. That is consistent with the GPU clocking down when it idles between
//! frames, which was not checked by reading a clock.
//!
//! Everything calibration creates except the cost model is dropped before the
//! loop's network is built, and the loop never sees it.

use anyhow::{Context, Result, bail};
use beatlm::gpu::{MAX_SLICES, SliceTimer};
use beatlm_core::cost::{CostModel, Estimator, SliceId};
use beatlm_core::kernel::Program;
use beatlm_core::time::{FrameIndex, Micros};

use crate::device::Gpu;
use crate::engine::{COST_WINDOW, timer_capacity};
use crate::net::{Net, Numbers};
use crate::pace::Pacer;
use crate::workload::Workload;

/// Timed frames of whole jobs in each calibration pass, after one untimed
/// frame: twice the quantile estimator's window, so that the half the
/// decision reads is as long as the window the loop's estimate reads, and
/// starts after the GPU has run the paced load for half a second.
pub const CALIBRATION_FRAMES: u64 = 2 * COST_WINDOW as u64;

/// What calibration is told.
#[derive(Debug, Clone, Copy)]
pub struct Settings {
    /// The statistic the loop's cost model reads with; the model pass 2 fills
    /// is built with it. The decision does not use it.
    pub estimator: Estimator,
    /// What the cost model estimates for a slice nothing measured.
    pub prior: Micros,
    /// The longest a dispatch should take.
    pub bound: Micros,
    /// How many jobs can have slices left at once in the loop, whose timer
    /// must time the slices of all of them in one frame
    /// (`engine::jobs_with_slices` says how many for each `--submit`).
    pub jobs: u32,
}

/// What calibration measured, and what it decided.
pub struct Calibration {
    /// The median of each kernel's measurements in the second half of pass 1,
    /// for the unsplit network, in program order.
    pub durations: Vec<Micros>,
    /// How many parts each kernel is split into, in program order.
    pub parts: Vec<u32>,
    /// The network split by `parts`.
    pub program: Program,
    /// The cost model pass 2 measured the split program's parts into: slice
    /// `i` is part `i` of `program`.
    pub cost: CostModel,
}

/// The middle of `samples` once sorted, the upper one when there are two;
/// `None` when there are none.
///
/// It is the statistic the decision reads: it ignores a few slow frames at
/// either end without a window, a deviation or a quantile to choose.
pub fn median(samples: &mut [u64]) -> Option<Micros> {
    samples.sort_unstable();
    samples.get(samples.len() / 2).copied().map(Micros::new)
}

/// Measures each kernel of `whole`, the network or model lowered and not
/// split, realised with `numbers`, splits it so each part is expected within
/// `settings.bound`, and measures the parts.
///
/// # Errors
///
/// A network or timer that cannot be built, a device that cannot be polled, a
/// kernel or part calibration never measured, an already split program, a
/// bound of zero, or a split that makes more slices than the loop's timer can
/// time.
pub fn calibrate(
    gpu: &Gpu,
    workload: &Workload,
    numbers: &Numbers,
    whole: &Program,
    settings: &Settings,
) -> Result<Calibration> {
    // Refused before pass 1 when the unsplit job alone is more than the loop's
    // timer can time: no bound fixes that, only fewer jobs with slices left.
    let kernels = u32::try_from(whole.kernels().len()).unwrap_or(u32::MAX);
    if timer_capacity(kernels, settings.jobs).is_none() {
        // With one job, `--submit free` is already in effect and no latency
        // lowers the count; only a smaller program would fit.
        let remedy = if settings.jobs > 1 {
            "Lower --latency, or pass --submit free"
        } else {
            "No --latency or --submit setting makes it fit"
        };
        bail!(
            "{kernels} kernels for each of {} jobs with slices left are more than the GPU \
             timer can time ({MAX_SLICES} slices per frame) before any split. {remedy}",
            settings.jobs,
        );
    }

    // Pass 1: whole kernels, to decide.
    let mut samples: Vec<Vec<u64>> = vec![Vec::new(); whole.kernels().len()];
    {
        let net = Net::new(&gpu.device, &gpu.queue, numbers, 1, whole.clone())
            .context("cannot build the network to calibrate")?;
        measure(gpu, workload, &net, |frame, slice, duration| {
            let list = usize::try_from(slice.get())
                .ok()
                .and_then(|i| samples.get_mut(i));
            if let (true, Some(list)) = (frame > CALIBRATION_FRAMES / 2, list) {
                list.push(duration.get());
            }
        })?;
    }
    let mut durations = Vec::with_capacity(samples.len());
    for (kernel, list) in samples.iter_mut().enumerate() {
        durations.push(
            median(list)
                .with_context(|| format!("calibration measured kernel {kernel} no time"))?,
        );
    }

    let parts = whole
        .parts_within(&durations, settings.bound)
        .context("cannot decide the parts")?;
    let program = whole.split(&parts).context("cannot split the network")?;

    // Refused before pass 2 is built: the loop's timer times the slices of
    // every job that can be in flight, and the unsplit job fit, so the split
    // made it too many.
    let slices = u32::try_from(program.kernels().len()).unwrap_or(u32::MAX);
    if timer_capacity(slices, settings.jobs).is_none() {
        bail!(
            "--max-dispatch-us {} splits a job into {slices} slices; up to {} jobs can have \
             slices left at once and the timer can time {MAX_SLICES} slices per frame. \
             Raise --max-dispatch-us",
            settings.bound.get(),
            settings.jobs,
        );
    }

    // Pass 2: the split program, into the cost model the loop will use.
    let mut cost = CostModel::new(settings.prior, settings.estimator)
        .context("cannot build the cost model")?;
    {
        let net = Net::new(&gpu.device, &gpu.queue, numbers, 1, program.clone())
            .context("cannot build the split network to calibrate")?;
        // The same second half as pass 1: the first half is the GPU settling
        // into the paced load, and an estimator that keeps every sample would
        // otherwise start the loop from it.
        measure(gpu, workload, &net, |frame, slice, duration| {
            if frame > CALIBRATION_FRAMES / 2 {
                cost.record(slice, duration);
            }
        })?;
    }
    for part in 0..slices {
        if cost.samples(SliceId::new(part)) == 0 {
            bail!("calibration measured part {part} of the split network no time");
        }
    }
    Ok(Calibration {
        durations,
        parts,
        program,
        cost,
    })
}

/// Runs [`CALIBRATION_FRAMES`] timed frames of one whole job on `net`'s slot 0
/// followed by the render pass, paced, after one untimed frame, and hands
/// `record` each kernel's measured duration with the number of its frame
/// (counting from 1 for the first timed frame) and its slice, which is its
/// index in `net`'s program.
///
/// # Errors
///
/// A timer that cannot be built or used, or a device that cannot be polled.
fn measure(
    gpu: &Gpu,
    workload: &Workload,
    net: &Net,
    mut record: impl FnMut(u64, SliceId, Micros),
) -> Result<()> {
    let kernels = net.kernel_count();
    let mut timer: SliceTimer<SliceId> = SliceTimer::new(&gpu.device, &gpu.queue, kernels)
        .context("cannot build the calibration timer")?;

    // Frame 0 is untimed: it compiles and first-touches everything.
    let mut pacer = Pacer::start(CALIBRATION_FRAMES + 1);
    while let Some(tick) = pacer.next_frame() {
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("beatlm-bench calibration"),
            });
        let mut timestamps = timer.begin_frame(FrameIndex::new(tick.index));
        let timed = tick.index > 0;
        for kernel in 0..kernels {
            let slice = SliceId::new(kernel);
            let writes = if timed {
                timestamps.compute_pass(slice)?
            } else {
                None
            };
            net.encode(&mut encoder, 0, slice, writes)?;
        }
        let render = if timed {
            timestamps.render_pass()?
        } else {
            None
        };
        workload.encode_render(&mut encoder, render);
        timestamps.resolve(&mut encoder);
        gpu.queue.submit([encoder.finish()]);
        timer.start_readback();
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("device poll failed")?;
        for timing in timer.collect_ready() {
            for (slice, duration) in timing.compute {
                record(timing.frame.get(), slice, duration);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_median_is_the_middle_sample_and_the_upper_of_two() {
        assert_eq!(median(&mut []), None);
        assert_eq!(median(&mut [7]), Some(Micros::new(7)));
        assert_eq!(median(&mut [9, 1, 5]), Some(Micros::new(5)));
        assert_eq!(median(&mut [4, 1, 3, 2]), Some(Micros::new(3)));
    }

    #[test]
    fn the_median_ignores_slow_frames_that_move_a_mean() {
        // Three early under-clocked frames and one slow one among 28 steady.
        let mut samples = vec![1_000; 28];
        samples.extend([9_000, 9_000, 9_000, 20_000]);
        assert_eq!(median(&mut samples), Some(Micros::new(1_000)));
    }
}
