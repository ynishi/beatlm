//! The frame loop: candidate 1 of the concurrency model, a synchronous `tick`
//! on the calling thread.
//!
//! Each frame the loop sleeps until the frame's start, asks the core what to
//! run, encodes it, submits it, and returns; the GPU's answer arrives on a later
//! frame through the timer's readback ring and is fed to the cost model then.
//! The loop owns the host's side of a frame, in the order the umbrella crate
//! documents, and reads the clock; the core and `beatlm::gpu` do neither.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use beatlm::gpu::{FrameTiming, SliceTimer, TimerStats};
use beatlm_core::budget::Budget;
use beatlm_core::cost::{CostModel, SliceId};
use beatlm_core::schedule::{JobId, Scheduler};
use beatlm_core::sync::{MissPolicy, SyncPolicy};
use beatlm_core::time::{FrameIndex, FrameInfo, Micros, TimePoint};

use crate::args::Args;
use crate::device::Gpu;
use crate::workload::Workload;

/// The frame period: 60 Hz.
pub const FRAME_PERIOD: Micros = Micros::new(16_667);

/// What a slice is assumed to cost until it has been measured.
const SLICE_PRIOR: Micros = Micros::new(2_000);
/// What the render pass is assumed to cost until it has been measured.
const RENDER_PRIOR: Micros = Micros::new(2_000);
/// Weight of a new measurement in a slice's estimate.
const COST_ALPHA: f64 = 0.2;
/// Standard deviations added to a slice's mean when budgeting it.
const COST_SIGMAS: f64 = 2.0;
/// How long before a frame's start the loop stops sleeping and spins. A sleep
/// overshoots by up to a few milliseconds on macOS, so the window has to cover
/// the overshoot or the frame starts late; the report's
/// `frame_start_lateness_us` shows whether it does.
const SPIN_WINDOW: Duration = Duration::from_millis(4);
/// How many times the final drain waits for the device before giving up on the
/// timings still in flight.
const DRAIN_ATTEMPTS: usize = 8;

/// The label a timed slice carries from the scheduler to the cost model.
type Tag = (JobId, SliceId);

/// Everything a run measured.
pub struct Outcome {
    /// Time from the loop beginning each frame to the return of `submit`, in
    /// microseconds.
    pub cpu_us: Vec<u64>,
    /// How long after its period's boundary each frame actually began, in
    /// microseconds; nonzero when the previous frame overran or the sleep
    /// overshot.
    pub lateness_us: Vec<u64>,
    /// Frame periods given up because a frame began a whole period late.
    pub dropped: u64,
    /// Render plus every slice measured in the same frame, per timed frame, in
    /// microseconds.
    pub gpu_frame_us: Vec<u64>,
    /// Latency in frames of each delivered result.
    pub latencies: Vec<u64>,
    /// Results delivered.
    pub delivered: u64,
    /// Deadlines that arrived without a complete result.
    pub missed: u64,
    /// Deliveries whose latency was not the promised one.
    pub violations: u64,
    /// Jobs still awaiting their deadline when the loop ended.
    pub pending_at_end: usize,
    /// Per slice: sum and count of measured microseconds.
    pub measured: BTreeMap<SliceId, (u64, u64)>,
    /// Per slice: the cost model's final estimate.
    pub estimates: BTreeMap<SliceId, Micros>,
    /// What the timer could not measure.
    pub timer: TimerStats,
}

struct State {
    cost: CostModel,
    render_estimate: Micros,
    outcome: Outcome,
}

impl State {
    fn new(prior: Micros) -> Result<Self> {
        let cost = CostModel::new(prior, COST_ALPHA, COST_SIGMAS)
            .context("cannot build the cost model")?;
        Ok(Self {
            cost,
            render_estimate: RENDER_PRIOR,
            outcome: Outcome {
                cpu_us: Vec::new(),
                lateness_us: Vec::new(),
                dropped: 0,
                gpu_frame_us: Vec::new(),
                latencies: Vec::new(),
                delivered: 0,
                missed: 0,
                violations: 0,
                pending_at_end: 0,
                measured: BTreeMap::new(),
                estimates: BTreeMap::new(),
                timer: TimerStats::default(),
            },
        })
    }

    /// Feeds one frame's measurements to the cost model and the report.
    fn absorb(&mut self, timing: FrameTiming<Tag>) {
        let mut gpu = Micros::ZERO;
        for ((_, slice), duration) in &timing.compute {
            self.cost.record(*slice, *duration);
            let entry = self.outcome.measured.entry(*slice).or_insert((0, 0));
            entry.0 = entry.0.saturating_add(duration.get());
            entry.1 += 1;
            gpu = gpu.saturating_add(*duration);
        }
        if let Some(render) = timing.render {
            self.render_estimate = render;
            self.outcome
                .gpu_frame_us
                .push(gpu.saturating_add(render).get());
        }
    }
}

/// Runs `args.frames` frames at [`FRAME_PERIOD`] against `gpu`.
///
/// # Errors
///
/// A device that cannot be polled, a timer that cannot be built, or a scheduler
/// that rejects the configuration.
pub fn run(gpu: &Gpu, args: &Args) -> Result<Outcome> {
    let workload = Workload::new(&gpu.device, args.slice_iters, args.render_iters);
    // A frame can run slices of every job still pending, not only its own: a
    // job lives `latency` frames past its submission, so `latency + 1` jobs of
    // `slices` slices each can be pending at once.
    let capacity = args.slices.saturating_mul(args.latency.saturating_add(1));
    let mut timer: SliceTimer<Tag> = SliceTimer::new(&gpu.device, &gpu.queue, capacity)
        .with_context(|| {
            format!(
                "cannot build the GPU timer for --slices {} with --latency {}",
                args.slices, args.latency
            )
        })?;
    let mut scheduler = Scheduler::new(
        SyncPolicy::FixedLatency {
            frames: args.latency,
        },
        MissPolicy::HoldPrevious,
        None,
    )
    .context("cannot build the scheduler")?;
    let mut state = State::new(SLICE_PRIOR)?;
    let slices: Vec<SliceId> = (0..args.slices).map(SliceId::new).collect();
    let margin = Micros::new(args.margin_us);

    warm_up(gpu, &workload)?;

    let period = FRAME_PERIOD.get();
    let epoch = Instant::now();
    let mut n: u64 = 0;
    while n < args.frames {
        wait_until(epoch + Duration::from_micros(n.saturating_mul(period)));
        let began = Instant::now();
        let elapsed = micros_between(epoch, began);
        // A display-driven host never runs two frames in one period. A frame
        // that begins a whole period or more late gives up the periods it
        // overran, as a display would drop them, and the core sees those frame
        // indices as skipped; catching up back to back instead would pile work
        // onto the GPU and charge the stall to the frames after it.
        let current = elapsed / period;
        if current > n {
            state.outcome.dropped += current - n;
            n = current;
            if n >= args.frames {
                break;
            }
        }
        state
            .outcome
            .lateness_us
            .push(elapsed.saturating_sub(n.saturating_mul(period)));

        // The frame's start is when it actually began, so a late frame is not
        // planned as if it had started on time.
        let frame = FrameInfo {
            index: FrameIndex::new(n),
            start: TimePoint::new(elapsed),
            period: FRAME_PERIOD,
        };
        let budget = Budget::for_frame(FRAME_PERIOD, state.render_estimate, margin);
        scheduler
            .submit(frame, slices.clone(), &state.cost)
            .context("the scheduler rejected the frame's job")?;
        let plan = scheduler.plan_frame(frame, budget, &state.cost);

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("beatlm-bench frame"),
            });
        let mut timestamps = timer.begin_frame(frame.index);
        for run in &plan.runs {
            let writes = timestamps.compute_pass((run.job, run.slice))?;
            workload.encode_slice(&mut encoder, writes);
        }
        workload.encode_render(&mut encoder, timestamps.render_pass()?);
        timestamps.resolve(&mut encoder);
        gpu.queue.submit([encoder.finish()]);
        state
            .outcome
            .cpu_us
            .push(micros_between(began, Instant::now()));

        timer.start_readback();
        gpu.device
            .poll(wgpu::PollType::Poll)
            .context("device poll failed")?;
        for timing in timer.collect_ready() {
            state.absorb(timing);
        }

        for delivery in &plan.delivered {
            let latency = frame.index.get().saturating_sub(delivery.input.get());
            state.outcome.latencies.push(latency);
            state.outcome.delivered += 1;
            if latency != u64::from(args.latency) {
                state.outcome.violations += 1;
            }
        }
        state.outcome.missed += plan.missed.len() as u64;
        n += 1;
    }

    for _ in 0..DRAIN_ATTEMPTS {
        if timer.in_flight() == 0 {
            break;
        }
        timer.start_readback();
        gpu.device
            .poll(wgpu::PollType::wait_indefinitely())
            .context("device poll failed")?;
        for timing in timer.collect_ready() {
            state.absorb(timing);
        }
    }

    state.outcome.pending_at_end = scheduler.pending();
    state.outcome.timer = timer.stats();
    for slice in slices {
        state
            .outcome
            .estimates
            .insert(slice, state.cost.estimate(slice));
    }
    Ok(state.outcome)
}

/// Runs one slice and one render pass to completion before the clock starts, so
/// shader compilation and first-use costs are not charged to frame 0.
fn warm_up(gpu: &Gpu, workload: &Workload) -> Result<()> {
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("beatlm-bench warm-up"),
        });
    workload.encode_slice(&mut encoder, None);
    workload.encode_render(&mut encoder, None);
    gpu.queue.submit([encoder.finish()]);
    gpu.device
        .poll(wgpu::PollType::wait_indefinitely())
        .context("device poll failed")?;
    Ok(())
}

/// Sleeps until shortly before `target`, then spins, so the frame starts on
/// time without a core pinned for the whole period.
fn wait_until(target: Instant) {
    let now = Instant::now();
    if let Some(sleep) = target.checked_duration_since(now + SPIN_WINDOW) {
        std::thread::sleep(sleep);
    }
    while Instant::now() < target {
        std::hint::spin_loop();
    }
}

/// Microseconds from `start` to `end`, zero if `end` is not later.
fn micros_between(start: Instant, end: Instant) -> u64 {
    u64::try_from(end.saturating_duration_since(start).as_micros()).unwrap_or(u64::MAX)
}
