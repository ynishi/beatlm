//! Calibration: what each kernel of the network takes run whole on this
//! device, measured before the loop, and the parts that keep each dispatch
//! within `--max-dispatch-us`.
//!
//! The decision is the core's (`Program::parts_within`, which owns the rule);
//! this module only measures. It builds the network unsplit on one slot and,
//! paced at the loop's frame period, runs [`CALIBRATION_FRAMES`] frames of one
//! whole job followed by the synthetic render pass, each submitted and waited
//! for within its frame, reading every kernel's timed compute pass. A kernel
//! is measured as the loop will run it: behind the same render pass, with the
//! GPU as busy per frame and as idle between frames as in the loop.
//!
//! The pacing is not optional. Run back to back without it, on an Apple M1
//! Pro (Metal) on 2026-10-21, every kernel of the fused network measured two
//! to four times shorter than the same kernel in the paced loop that followed
//! (the first convolution 0.20 to 0.33 ms against 0.66 to 1.09 ms in the loop,
//! the second matrix multiply 1.04 ms against 2.26 ms), so a bound of 1 ms
//! split the matrix multiply into two parts that each still took about 1.1 ms
//! in the loop. That is consistent with the GPU clocking down when it idles
//! between frames, which was not checked by reading a clock.
//!
//! The measurements go into a cost model with the run's `--estimator`, and its
//! estimate per kernel is the duration the decision reads, so a kernel is
//! split for what it usually takes and not for one slow frame. Everything
//! calibration creates is dropped before the loop's network is built, and the
//! loop never sees it.

use anyhow::{Context, Result, bail};
use beatlm::gpu::SliceTimer;
use beatlm_core::cost::{CostModel, Estimator, SliceId};
use beatlm_core::kernel::Program;
use beatlm_core::time::{FrameIndex, Micros};

use crate::device::Gpu;
use crate::engine::COST_WINDOW;
use crate::net::{Net, Shape};
use crate::pace::Pacer;
use crate::workload::Workload;

/// Frames of whole jobs calibration times, after one untimed frame that pays
/// for first use: twice the quantile estimator's window, so its estimate reads
/// only the second half, after the GPU has run the paced load for half a
/// second.
pub const CALIBRATION_FRAMES: u64 = 2 * COST_WINDOW as u64;

/// What calibration measured, and what it decided.
pub struct Calibration {
    /// The cost model's estimate for each kernel of the unsplit network, in
    /// program order.
    pub durations: Vec<Micros>,
    /// How many parts each kernel is split into, in program order.
    pub parts: Vec<u32>,
    /// The network split by `parts`.
    pub program: Program,
}

/// Measures each kernel of `whole`, the network at `shape` lowered and not
/// split, and splits it so each part is expected within `bound`.
///
/// # Errors
///
/// A network or timer that cannot be built, a device that cannot be polled, a
/// kernel calibration never measured, an already split program, or a bound of
/// zero.
pub fn calibrate(
    gpu: &Gpu,
    workload: &Workload,
    shape: &Shape,
    whole: &Program,
    estimator: Estimator,
    prior: Micros,
    bound: Micros,
) -> Result<Calibration> {
    let kernels = u32::try_from(whole.kernels().len()).context("too many kernels to time")?;
    let net = Net::new(&gpu.device, &gpu.queue, shape, 1, whole.clone())
        .context("cannot build the network to calibrate")?;
    let mut timer: SliceTimer<SliceId> = SliceTimer::new(&gpu.device, &gpu.queue, kernels)
        .context("cannot build the calibration timer")?;
    let mut cost = CostModel::new(prior, estimator).context("cannot build the cost model")?;

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
                cost.record(slice, duration);
            }
        }
    }

    let mut durations = Vec::with_capacity(whole.kernels().len());
    for kernel in 0..kernels {
        let slice = SliceId::new(kernel);
        if cost.samples(slice) == 0 {
            bail!("calibration measured kernel {kernel} no time");
        }
        durations.push(cost.estimate(slice));
    }
    let parts = whole
        .parts_within(&durations, bound)
        .context("cannot decide the parts")?;
    let program = whole.split(&parts).context("cannot split the network")?;
    Ok(Calibration {
        durations,
        parts,
        program,
    })
}
