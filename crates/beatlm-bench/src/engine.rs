//! The host's side of one frame, in the order the umbrella crate documents,
//! shared by every concurrency candidate.
//!
//! A candidate differs in which thread owns an [`Engine`] and in how it is
//! called: [`Engine::tick_sync`] is a plain call, [`Engine::tick_async`] is a
//! future, and the threaded candidate calls `tick_sync` from a thread of its
//! own. The decisions, the encoding, the submission and the readback are the
//! same code in all three, which is what lets their timings be compared. The
//! engine reads the clock to measure itself; the core and `beatlm::gpu` read
//! none.

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::task::{Context as TaskContext, Poll};
use std::time::Instant;

use anyhow::{Context, Result};
use beatlm::gpu::{FrameTiming, SliceTimer, TimerStats};
use beatlm_core::budget::Budget;
use beatlm_core::cost::{CostModel, Estimator, SliceId};
use beatlm_core::schedule::{FramePlan, JobId, Scheduler};
use beatlm_core::sync::{MissPolicy, SyncPolicy};
use beatlm_core::time::{FrameIndex, FrameInfo, Micros, TimePoint};

use crate::args::{Args, EstimatorKind, WorkloadKind};
use crate::counters::{self, Watch};
use crate::device::Gpu;
use crate::net::{self, Net};
use crate::pace::{FRAME_PERIOD, Tick, micros_between};
use crate::workload::Workload;

/// What a slice is assumed to cost until it has been measured.
const SLICE_PRIOR: Micros = Micros::new(2_000);
/// What the render pass is assumed to cost until it has been measured.
const RENDER_PRIOR: Micros = Micros::new(2_000);
/// Weight of a new measurement in a slice's exponentially weighted mean.
const COST_ALPHA: f64 = 0.2;
/// Standard deviations added to a slice's mean when budgeting it.
const COST_SIGMAS: f64 = 2.0;
/// Measurements per slice the windowed quantile reads, about half a second of
/// frames at 60 Hz.
const COST_WINDOW: usize = 32;
/// The quantile of the window the windowed estimate is.
const COST_QUANTILE: f64 = 0.9;
/// The most the capped estimate pads the mean, as a multiple of the mean.
const COST_CAP: f64 = 0.5;
/// How many times the final drain waits for the device before giving up on the
/// timings still in flight.
const DRAIN_ATTEMPTS: usize = 8;

/// The cost model's statistic for the `--estimator` choice, with this
/// harness's parameters.
const fn estimator(kind: EstimatorKind) -> Estimator {
    match kind {
        EstimatorKind::MeanSigmas => Estimator::MeanPlusSigmas {
            alpha: COST_ALPHA,
            sigmas: COST_SIGMAS,
        },
        EstimatorKind::Quantile => Estimator::WindowedQuantile {
            window: COST_WINDOW,
            quantile: COST_QUANTILE,
        },
        EstimatorKind::Capped => Estimator::CappedDeviation {
            alpha: COST_ALPHA,
            sigmas: COST_SIGMAS,
            cap: COST_CAP,
        },
    }
}

/// The label a timed slice carries from the scheduler to the cost model.
type Tag = (JobId, SliceId);

/// Everything a run measured.
///
/// The per-frame vectors hold one sample per frame the engine handled, except
/// where a field says otherwise.
#[derive(Default)]
pub struct Outcome {
    /// Time from the pacer beginning each frame to the return of
    /// `queue.submit`, in microseconds. On the threaded candidate the clock
    /// starts on the frame-loop thread and stops on the GPU thread.
    pub cpu_us: Vec<u64>,
    /// Time from the pacer beginning each frame to the return of the driver's
    /// call for it: submit, the readback poll, the delivery bookkeeping and, in
    /// `async` mode, the executor's return path, in microseconds. The excess
    /// over `cpu_us` is what follows the submit.
    pub tick_us: Vec<u64>,
    /// Time from the pacer beginning each frame to the GPU thread receiving its
    /// event, in microseconds; empty unless a channel carries the events.
    pub handoff_us: Vec<u64>,
    /// Time spent deciding the frame (`Scheduler::submit` plus `plan_frame`),
    /// in microseconds.
    pub plan_us: Vec<u64>,
    /// Time spent recording the frame's command buffer, from creating the
    /// encoder to finishing it, in microseconds.
    pub encode_us: Vec<u64>,
    /// Time inside `queue.submit`, in microseconds.
    pub submit_us: Vec<u64>,
    /// Compute dispatches encoded in each frame: one per slice the plan
    /// handed out.
    pub dispatches_per_frame: Vec<u64>,
    /// Compute dispatches encoded for each delivered job, over every frame
    /// that ran one of its slices.
    pub dispatches_per_job: Vec<u64>,
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
    /// Dispatches the competing load submitted, from the one it completed
    /// before the run's first frame to the last one queued when it was stopped
    /// after the run; `None` when the run had no competing load.
    pub background_dispatches: Option<u64>,
    /// The planned network's memory and allocation readings; `None` for the
    /// synthetic workload.
    pub net: Option<NetReport>,
}

/// What the `net` workload reports about memory.
pub struct NetReport {
    /// Bytes of each arena.
    pub arenas: Vec<u64>,
    /// Bytes of all arenas.
    pub total_bytes: u64,
    /// The most bytes live at one operation: the lower bound on `total_bytes`.
    pub peak_live_bytes: u64,
    /// The sum of every tensor's size: what the model needs with no reuse.
    pub unshared_bytes: u64,
    /// Tensors in the plan.
    pub tensors: usize,
    /// Input and output slots: the jobs the network keeps apart.
    pub slots: u32,
    /// The name of each kernel of a job, in dispatch order.
    pub kernels: Vec<String>,
    /// wgpu's object counters across the loop.
    pub counters: counters::Report,
}

/// Decides, encodes, submits and reads back the GPU work of frames.
///
/// It holds the scheduler, the cost model, the timer and the workload of one
/// run, so a fresh engine is a fresh run on the same device. Whichever thread
/// calls it is the thread that touches the device.
pub struct Engine<'g> {
    gpu: &'g Gpu,
    workload: Workload,
    net: Option<Net>,
    watch: Option<Watch>,
    timer: SliceTimer<Tag>,
    scheduler: Scheduler,
    cost: CostModel,
    render_estimate: Micros,
    slices: Vec<SliceId>,
    margin: Micros,
    latency: u32,
    /// Dispatches encoded so far for each job that is not yet delivered or
    /// missed, by submission index.
    job_dispatches: BTreeMap<u64, u64>,
    outcome: Outcome,
}

impl<'g> Engine<'g> {
    /// Builds the run's workload, timer, scheduler and cost model on `gpu`, and
    /// runs every slice the workload has once and one render pass to completion
    /// so shader compilation and first-use costs are not charged to frame 0.
    ///
    /// # Errors
    ///
    /// A timer that cannot be built, a scheduler that rejects the
    /// configuration, or a device that cannot be polled.
    pub fn new(gpu: &'g Gpu, args: &Args) -> Result<Self> {
        let workload = Workload::new(&gpu.device, args.slice_iters, args.render_iters);
        let policy = SyncPolicy::FixedLatency {
            frames: args.latency,
        };
        let net = match args.workload {
            WorkloadKind::Synthetic => None,
            WorkloadKind::Net => {
                // One input and output slot per job that can be in flight: a
                // job's output must survive until its delivery frame.
                let slots = policy
                    .max_in_flight()
                    .context("the policy does not bound the jobs in flight")?;
                Some(
                    Net::new(
                        &gpu.device,
                        &gpu.queue,
                        &net::DEFAULT_SHAPE,
                        slots,
                        args.fusion,
                    )
                    .context("cannot build the planned network")?,
                )
            }
        };
        // The job's slices are the network's kernels; `args` counted them from
        // its own lowering, so check they agree before a slice names no kernel.
        if let Some(net) = &net {
            if net.kernel_count() != args.slices {
                anyhow::bail!(
                    "--slices is {} but the network has {} kernels",
                    args.slices,
                    net.kernel_count()
                );
            }
        }
        // A frame can run slices of every job still pending, not only its own: a
        // job lives `latency` frames past its submission, so `latency + 1` jobs
        // of `slices` slices each can be pending at once.
        let capacity = args.slices.saturating_mul(args.latency.saturating_add(1));
        let timer: SliceTimer<Tag> = SliceTimer::new(&gpu.device, &gpu.queue, capacity)
            .with_context(|| {
                format!(
                    "cannot build the GPU timer for --slices {} with --latency {}",
                    args.slices, args.latency
                )
            })?;
        let scheduler = Scheduler::new(policy, MissPolicy::HoldPrevious, None)
            .context("cannot build the scheduler")?;
        let cost = CostModel::new(SLICE_PRIOR, estimator(args.estimator))
            .context("cannot build the cost model")?;
        let mut engine = Self {
            gpu,
            workload,
            net,
            watch: None,
            timer,
            scheduler,
            cost,
            render_estimate: RENDER_PRIOR,
            slices: (0..args.slices).map(SliceId::new).collect(),
            margin: Micros::new(args.margin_us),
            latency: args.latency,
            job_dispatches: BTreeMap::new(),
            outcome: Outcome::default(),
        };
        engine.warm_up()?;
        // Everything the network allocates exists by now, so this reading is the
        // state the loop starts from.
        if engine.net.is_some() {
            engine.watch = Some(Watch::start(&gpu.device));
        }
        Ok(engine)
    }

    /// Runs one frame to its end on the calling thread: decide, encode, submit,
    /// poll the device without waiting, collect what has completed, deliver.
    ///
    /// # Errors
    ///
    /// A scheduler that rejects the frame's job, a timer that cannot time the
    /// frame, or a device that cannot be polled.
    pub fn tick_sync(&mut self, tick: &Tick) -> Result<()> {
        let plan = self.submit_frame(tick)?;
        self.read_back(wgpu::PollType::Poll)?;
        self.complete(tick, &plan);
        Ok(())
    }

    /// The same frame as [`tick_sync`](Self::tick_sync), with the readback
    /// awaited as a [`ReadbackTurn`] instead of called.
    ///
    /// Everything before the await is the synchronous path's code, so the
    /// submit happens on the first poll and the await sits where a browser would
    /// have to return to its event loop for map callbacks to run.
    ///
    /// # Errors
    ///
    /// As [`tick_sync`](Self::tick_sync).
    pub async fn tick_async(&mut self, tick: &Tick) -> Result<()> {
        let plan = self.submit_frame(tick)?;
        ReadbackTurn { engine: self }.await?;
        self.complete(tick, &plan);
        Ok(())
    }

    /// Records how long an event took to reach the thread running this engine,
    /// in microseconds.
    pub fn record_handoff(&mut self, micros: u64) {
        self.outcome.handoff_us.push(micros);
    }

    /// Records that the driver's call for `tick` has returned. Every driver
    /// calls it after the tick, so `tick_us` covers the same span in every
    /// candidate, the executor around an `async` tick included.
    pub fn end_tick(&mut self, tick: &Tick) {
        self.outcome
            .tick_us
            .push(micros_between(tick.began, Instant::now()));
        if let Some(watch) = &mut self.watch {
            watch.sample(&self.gpu.device);
        }
    }

    /// Starts the readback, polls the device with `poll`, and feeds what has
    /// completed to the cost model and the report.
    fn read_back(&mut self, poll: wgpu::PollType) -> Result<()> {
        self.timer.start_readback();
        self.gpu.device.poll(poll).context("device poll failed")?;
        for timing in self.timer.collect_ready() {
            self.absorb(timing);
        }
        Ok(())
    }

    /// Decides the frame, encodes it and submits it, and records where the CPU
    /// time went.
    fn submit_frame(&mut self, tick: &Tick) -> Result<FramePlan> {
        let frame = FrameInfo {
            index: FrameIndex::new(tick.index),
            start: TimePoint::new(tick.start_us),
            period: FRAME_PERIOD,
        };

        let planning = Instant::now();
        let budget = Budget::for_frame(FRAME_PERIOD, self.render_estimate, self.margin);
        self.scheduler
            .submit(frame, self.slices.clone(), &self.cost)
            .context("the scheduler rejected the frame's job")?;
        let plan = self.scheduler.plan_frame(frame, budget, &self.cost);

        let encoding = Instant::now();
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("beatlm-bench frame"),
            });
        let mut timestamps = self.timer.begin_frame(frame.index);
        for run in &plan.runs {
            let writes = timestamps.compute_pass((run.job, run.slice))?;
            match &self.net {
                Some(net) => {
                    net.encode(&mut encoder, net.slot_of(run.job.get()), run.slice, writes)?;
                }
                None => self.workload.encode_slice(&mut encoder, writes),
            }
        }
        self.workload
            .encode_render(&mut encoder, timestamps.render_pass()?);
        timestamps.resolve(&mut encoder);
        let commands = encoder.finish();

        let submitting = Instant::now();
        self.gpu.queue.submit([commands]);
        let submitted = Instant::now();

        self.outcome
            .plan_us
            .push(micros_between(planning, encoding));
        self.outcome
            .encode_us
            .push(micros_between(encoding, submitting));
        self.outcome
            .submit_us
            .push(micros_between(submitting, submitted));
        self.outcome
            .dispatches_per_frame
            .push(plan.runs.len() as u64);
        // Counted after the timed intervals, so the bookkeeping is not charged
        // to the encode time it is compared with. Every slice is one compute
        // pass with one dispatch, in either workload.
        for run in &plan.runs {
            *self.job_dispatches.entry(run.job.get()).or_default() += 1;
        }
        self.outcome
            .cpu_us
            .push(micros_between(tick.began, submitted));
        Ok(plan)
    }

    /// Delivers the plan's results and counts its misses.
    fn complete(&mut self, tick: &Tick, plan: &FramePlan) {
        for delivery in &plan.delivered {
            let latency = tick.index.saturating_sub(delivery.input.get());
            self.outcome.latencies.push(latency);
            self.outcome.delivered += 1;
            if latency != u64::from(self.latency) {
                self.outcome.violations += 1;
            }
            let dispatches = self.job_dispatches.remove(&delivery.job.get()).unwrap_or(0);
            self.outcome.dispatches_per_job.push(dispatches);
        }
        for missed in &plan.missed {
            self.job_dispatches.remove(&missed.job.get());
        }
        self.outcome.missed += plan.missed.len() as u64;
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

    /// Waits for the timings still in flight, then closes the run.
    ///
    /// The wait is outside every measured interval, and blocks: the run is over
    /// and there is no frame to protect.
    ///
    /// # Errors
    ///
    /// A device that cannot be polled.
    pub fn finish(mut self) -> Result<Outcome> {
        // Read before the drain, so the reading is of the loop and not of the
        // wait that follows it.
        let counters = self.watch.take().map(|w| w.finish(&self.gpu.device));
        for _ in 0..DRAIN_ATTEMPTS {
            if self.timer.in_flight() == 0 {
                break;
            }
            self.read_back(wgpu::PollType::wait_indefinitely())?;
        }
        self.outcome.pending_at_end = self.scheduler.pending();
        self.outcome.timer = self.timer.stats();
        if let (Some(net), Some(counters)) = (&self.net, counters) {
            let plan = net.plan();
            self.outcome.net = Some(NetReport {
                arenas: plan.arenas().to_vec(),
                total_bytes: plan.total_bytes(),
                peak_live_bytes: plan.peak_live_bytes(),
                unshared_bytes: plan.placements().map(|(_, p)| p.size).sum(),
                tensors: plan.placements().count(),
                slots: net.slots(),
                kernels: net.program().kernels().iter().map(|k| k.name()).collect(),
                counters,
            });
        }
        for &slice in &self.slices {
            self.outcome
                .estimates
                .insert(slice, self.cost.estimate(slice));
        }
        Ok(self.outcome)
    }

    /// Runs every slice the workload has once, and one render pass, to
    /// completion, so first-use costs are not charged to frame 0.
    fn warm_up(&self) -> Result<()> {
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("beatlm-bench warm-up"),
            });
        match &self.net {
            Some(net) => {
                for slot in 0..net.slots() {
                    for slice in 0..net.kernel_count() {
                        net.encode(&mut encoder, slot, SliceId::new(slice), None)?;
                    }
                }
            }
            None => self.workload.encode_slice(&mut encoder, None),
        }
        self.workload.encode_render(&mut encoder, None);
        self.gpu.queue.submit([encoder.finish()]);
        // The network's uploads leave staging memory that wgpu releases on a
        // poll after the work completes, and the allocation counters are read
        // next, so it gets a second poll. The synthetic load has no uploads.
        let polls = if self.net.is_some() { 2 } else { 1 };
        for _ in 0..polls {
            self.gpu
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .context("device poll failed")?;
        }
        Ok(())
    }
}

/// The readback step of an async tick: gives the device its turn to run map
/// callbacks, then yields what has completed.
///
/// Natively [`wgpu::Device::poll`] runs the callbacks of every submission
/// already finished before it returns, so the future is ready on its first
/// poll and never returns `Pending`. In a browser `poll` does nothing and
/// callbacks run between event-loop turns, so there the same step would find
/// nothing completed for the frame just submitted and report it on a later one.
/// Either way the timer's `collect_ready` is a check, not a wait: the timer
/// exposes completion as a flag set by the map callback, not as a future.
struct ReadbackTurn<'e, 'g> {
    engine: &'e mut Engine<'g>,
}

impl Future for ReadbackTurn<'_, '_> {
    type Output = Result<()>;

    fn poll(self: Pin<&mut Self>, _cx: &mut TaskContext<'_>) -> Poll<Self::Output> {
        Poll::Ready(self.get_mut().engine.read_back(wgpu::PollType::Poll))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_estimator_is_the_cores_default() {
        assert_eq!(estimator(Args::default().estimator), Estimator::default());
    }

    #[test]
    fn every_estimator_the_options_name_is_valid() {
        for kind in [
            EstimatorKind::MeanSigmas,
            EstimatorKind::Quantile,
            EstimatorKind::Capped,
        ] {
            assert!(CostModel::new(SLICE_PRIOR, estimator(kind)).is_ok());
        }
    }
}
