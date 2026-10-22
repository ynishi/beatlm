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
use beatlm::gpu::{FrameTiming, MAX_SLICES, SliceTimer, TimerStats};
use beatlm_core::budget::Budget;
use beatlm_core::cost::{CostModel, Estimator, FrameScale, SliceId};
use beatlm_core::kernel::Program;
use beatlm_core::schedule::{FramePlan, Scheduler};
use beatlm_core::sync::{MissPolicy, SyncPolicy};
use beatlm_core::time::{FrameIndex, FrameInfo, Micros, TimePoint};
use beatlm_host::frame::{Admission, InFlight, RenderCost, Tag, jobs_with_slices, timer_capacity};
use beatlm_host::pass_overlap;
use beatlm_host::presets::{self, RENDER_PRIOR, SLICE_PRIOR, STARTUP_FRAMES};
use beatlm_host::slots;
use beatlm_host::stats::frame_ratio;
use beatlm_host::workload::{
    DEFAULT_STAGE_STORAGE_BUFFERS, RENDER_HEIGHT, RENDER_WIDTH, RenderOrder, TARGET_FORMAT,
    Workload, written_fits,
};

use crate::args::{Args, EstimatorKind, WorkloadKind};
use crate::calibrate::{Settings, calibrate};
use crate::counters::{self, Watch};
use crate::depth;
use crate::device::Gpu;
use crate::net::{self, Net, Numbers};
use crate::pace::{FRAME_PERIOD, Tick, micros_between};

/// How many times the final drain waits for the device before giving up on the
/// timings still in flight.
const DRAIN_ATTEMPTS: usize = 8;

/// The cost model's statistic for the `--estimator` choice: the preset of
/// that name ([`presets`]).
const fn estimator(kind: EstimatorKind) -> Estimator {
    match kind {
        EstimatorKind::MeanSigmas => presets::mean_sigmas(),
        EstimatorKind::Quantile => presets::quantile(),
        EstimatorKind::Capped => presets::capped(),
        EstimatorKind::FrameScaled => presets::frame_scaled(),
    }
}

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
    /// The longest single compute pass measured in each timed frame that ran
    /// one, in microseconds: what one dispatch held up the queue for.
    pub longest_dispatch_us: Vec<u64>,
    /// Latency in frames of each delivered result.
    pub latencies: Vec<u64>,
    /// Inputs submitted as jobs: one per frame with `--submit every`, fewer
    /// with `--submit free`.
    pub submitted: u64,
    /// Results delivered.
    pub delivered: u64,
    /// Deadlines that arrived without a complete result.
    pub missed: u64,
    /// The part of `missed` that belongs to jobs submitted in the loop's first
    /// [`STARTUP_FRAMES`] frames, so a cascade of misses while estimates are
    /// new shows apart from misses in steady state.
    pub startup_missed: u64,
    /// Deliveries whose latency was not the promised one.
    pub violations: u64,
    /// Jobs still awaiting their deadline when the loop ended.
    pub pending_at_end: usize,
    /// Per slice: sum and count of measured microseconds.
    pub measured: BTreeMap<SliceId, (u64, u64)>,
    /// Per slice: the cost model's final estimate.
    pub estimates: BTreeMap<SliceId, Micros>,
    /// Per timed frame whose slices with a frame estimate were planned at
    /// more than zero: their measured time over the sum of those estimates
    /// when the frame was planned ([`CostModel::frame_estimate`]: every slice
    /// for every estimator but the frame-scaled one, its warm slices for
    /// that one).
    pub frame_ratios: Vec<f64>,
    /// The frame scale of each plan made with a measured one.
    pub frame_scales: Vec<f64>,
    /// The frame scale when the run ended.
    pub frame_scale: FrameScale,
    /// Frames the cost model folded into its frame scale.
    pub frames_recorded: u64,
    /// What the timer could not measure.
    pub timer: TimerStats,
    /// Every timed frame's passes with their timestamps, for the
    /// `pass_overlap_*` lines (`beatlm_host::pass_overlap`).
    pub pass_frames: Vec<pass_overlap::Frame>,
    /// Dispatches the competing load submitted, from the one it completed
    /// before the run's first frame to the last one queued when it was stopped
    /// after the run; `None` when the run had no competing load.
    pub background_dispatches: Option<u64>,
    /// The planned program's memory and allocation readings; `None` for the
    /// synthetic workload.
    pub net: Option<NetReport>,
}

/// What the `net` and `depth` workloads report about memory.
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
    /// The name of each kernel of a job, in dispatch order, a part of a split
    /// kernel as `name[i/n]`, counting from one.
    pub kernels: Vec<String>,
    /// Each kernel of the unsplit network, in program order.
    pub kernel_parts: Vec<KernelParts>,
    /// wgpu's object counters across the loop.
    pub counters: counters::Report,
}

/// One kernel of the unsplit network and what calibration made of it.
pub struct KernelParts {
    /// The kernel's name.
    pub name: String,
    /// The parts it runs as; one when it is not split.
    pub parts: u32,
    /// The median calibration measured for it run whole; `None` when
    /// splitting is off and nothing was calibrated.
    pub calibrated: Option<Micros>,
}

/// Decides, encodes, submits and reads back the GPU work of frames.
///
/// It holds the scheduler, the cost model, the timer and the workload of one
/// run, so a fresh engine is a fresh run on the same device. Whichever thread
/// calls it is the thread that touches the device.
pub struct Engine<'g> {
    gpu: &'g Gpu,
    workload: Workload,
    /// The offscreen texture the render pass draws into.
    target: wgpu::TextureView,
    net: Option<Net>,
    kernel_parts: Vec<KernelParts>,
    watch: Option<Watch>,
    timer: SliceTimer<Tag>,
    scheduler: Scheduler,
    cost: CostModel,
    /// The render's measurements, which set the budgets.
    render: RenderCost,
    slices: Vec<SliceId>,
    margin: Micros,
    latency: u32,
    admission: Admission,
    /// Dispatches encoded so far for each job that is not yet delivered or
    /// missed, by submission index.
    job_dispatches: BTreeMap<u64, u64>,
    outcome: Outcome,
}

/// The unsplit program of a planned workload, and the numbers it is realised
/// with.
fn planned_program(args: &Args) -> Result<(Program, Numbers)> {
    match args.workload {
        WorkloadKind::Depth => depth::load(args.size, args.fusion),
        WorkloadKind::Net => {
            let shape = net::DEFAULT_SHAPE;
            Ok((net::program(&shape, args.fusion)?, net::numbers(&shape)))
        }
        WorkloadKind::Synthetic | WorkloadKind::PassCost => {
            anyhow::bail!("the {} workload has no program", args.workload.name())
        }
    }
}

impl<'g> Engine<'g> {
    /// Builds the run's workload, timer, scheduler and cost model on `gpu`;
    /// for the network or the depth model with splitting on, first calibrates
    /// its kernels, splits them and measures the parts into the cost model the
    /// run then uses (`calibrate.rs`); then runs every slice the workload has
    /// once and one render pass to completion so shader compilation and
    /// first-use costs are not charged to frame 0.
    ///
    /// # Errors
    ///
    /// A calibration, network or timer that cannot be built, or a device that
    /// cannot be polled.
    pub fn new(gpu: &'g Gpu, args: &Args) -> Result<Self> {
        let mut workload = Workload::new(&gpu.device, args.slice_iters, args.render_iters);
        let target = offscreen_target(&gpu.device);
        let policy = SyncPolicy::FixedLatency {
            frames: args.latency,
        };
        let jobs = jobs_with_slices(args.submit, InFlight::Latency(args.latency))
            .context("--latency is too large to count the jobs in flight")?;
        let mut kernel_parts = Vec::new();
        // The cost model calibration filled with the split program's parts, when
        // there was a calibration.
        let mut measured_cost = None;
        let net = match args.workload {
            // `main` runs `pass-cost` through `pass_bench.rs`, never here.
            WorkloadKind::Synthetic | WorkloadKind::PassCost => None,
            WorkloadKind::Net | WorkloadKind::Depth => {
                let (whole, numbers) = planned_program(args)?;
                // The job's kernels before splitting; `args` counted them
                // from its own lowering, so check they agree.
                if u32::try_from(whole.kernels().len()).ok() != Some(args.slices) {
                    anyhow::bail!(
                        "--slices is {} but the {} workload has {} kernels",
                        args.slices,
                        args.workload.name(),
                        whole.kernels().len()
                    );
                }
                // `after` reads every arena; refused before calibration
                // spends its seconds when the unsplit plan has more than a
                // stage binds. The split plan is checked again below.
                // One input and output slot per job that can be in flight: a
                // job's output must survive until its delivery frame.
                let slots = policy
                    .max_in_flight()
                    .context("the policy does not bound the jobs in flight")?;
                if args.render_order == RenderOrder::After {
                    let arenas = slots::memory_plan(&whole, &numbers, slots)?.arenas().len();
                    written_fits(arenas, DEFAULT_STAGE_STORAGE_BUFFERS).with_context(|| {
                        format!(
                            "--render-order after cannot be honoured for the plan of {slots} \
                             slots, which has {arenas} arenas; run with --render-order free, or \
                             a shorter --latency or smaller --size, whose plan has fewer"
                        )
                    })?;
                }
                let names = whole.kernels().iter().map(|k| k.name());
                let program = match args.max_dispatch_us {
                    None => {
                        kernel_parts = names
                            .map(|name| KernelParts {
                                name,
                                parts: 1,
                                calibrated: None,
                            })
                            .collect();
                        whole
                    }
                    Some(bound) => {
                        let calibration = calibrate(
                            gpu,
                            &workload,
                            &target,
                            &numbers,
                            &whole,
                            &Settings {
                                estimator: estimator(args.estimator),
                                prior: SLICE_PRIOR,
                                bound: Micros::new(bound),
                                jobs,
                            },
                        )?;
                        kernel_parts = names
                            .zip(&calibration.parts)
                            .zip(&calibration.durations)
                            .map(|((name, &parts), &duration)| KernelParts {
                                name,
                                parts,
                                calibrated: Some(duration),
                            })
                            .collect();
                        measured_cost = Some(calibration.cost);
                        calibration.program
                    }
                };
                Some(
                    Net::new(&gpu.device, &gpu.queue, &numbers, slots, program)
                        .context("cannot build the planned program")?,
                )
            }
        };
        // After calibration, whose render passes stay the free ones, so the
        // loop starts from the same measurements under either order. `after`
        // reads every arena of the plan, so it refuses a plan of more than a
        // stage binds; the synthetic slices write the load's own sink.
        let written = match (args.render_order, &net) {
            (RenderOrder::After, Some(net)) => net.written()?,
            _ => Vec::new(),
        };
        workload
            .order_render(&gpu.device, args.render_order, &written)
            .with_context(|| {
                format!(
                    "--render-order after cannot be honoured for this plan of {} arenas; run \
                     with --render-order free, or a shorter --latency or smaller --size, whose \
                     plan has fewer",
                    written.len()
                )
            })?;
        // A job's slices are the program's kernels, each part of a split one
        // included.
        let slices = net.as_ref().map_or(args.slices, Net::kernel_count);
        let capacity = timer_capacity(slices, jobs).with_context(|| {
            format!(
                "{slices} slices for each of {jobs} jobs with slices left (--latency {}, --submit \
                 {}) are more than the GPU timer can time ({MAX_SLICES})",
                args.latency,
                args.submit.name()
            )
        })?;
        let timer: SliceTimer<Tag> = SliceTimer::new(&gpu.device, &gpu.queue, capacity)
            .with_context(|| {
                format!(
                    "cannot build the GPU timer for {slices} slices with --latency {}",
                    args.latency
                )
            })?;
        let margin = Micros::new(args.margin_us);
        let scheduler = Scheduler::new(
            policy,
            MissPolicy::HoldPrevious,
            Budget::for_frame(FRAME_PERIOD, RENDER_PRIOR, margin),
        );
        let cost = match measured_cost {
            Some(cost) => cost,
            None => CostModel::new(SLICE_PRIOR, estimator(args.estimator))
                .context("cannot build the cost model")?,
        };
        let mut engine = Self {
            gpu,
            workload,
            target,
            net,
            kernel_parts,
            watch: None,
            timer,
            scheduler,
            cost,
            render: RenderCost::default(),
            slices: (0..slices).map(SliceId::new).collect(),
            margin,
            latency: args.latency,
            admission: Admission::new(args.submit),
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

    /// Polls the device with `poll`, and feeds what has completed to the cost
    /// model and the report. The frame's map was requested when it was
    /// submitted.
    fn read_back(&mut self, poll: wgpu::PollType) -> Result<()> {
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
        // This frame's budget and the forecast of the frames to come, at the
        // bench's fixed period; `RenderCost` says why the two differ.
        let budget = self.render.budget(FRAME_PERIOD, self.margin);
        let forecast = self.render.expected(FRAME_PERIOD, self.margin);
        self.scheduler.set_expected_budget(forecast);
        let new_job = if self.admission.takes_input() {
            let job = self
                .scheduler
                .submit(frame, self.slices.clone(), &self.cost)
                .context("the scheduler rejected the frame's job")?;
            Some(job)
        } else {
            None
        };
        let plan = self.scheduler.plan_frame(frame, budget, &self.cost);

        let encoding = Instant::now();
        // The estimates the plan was made with; nothing records into the
        // cost model before the frame is encoded.
        let cost = &self.cost;
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("beatlm-bench frame"),
            });
        let mut timestamps = self.timer.begin_frame(frame.index);
        for run in &plan.runs {
            let writes =
                timestamps.compute_pass((run.job, run.slice, cost.frame_estimate(run.slice)))?;
            match &self.net {
                Some(net) => {
                    net.encode(&mut encoder, net.slot_of(run.job.get()), run.slice, writes)?;
                }
                None => self.workload.encode_slice(&mut encoder, writes),
            }
        }
        self.workload
            .encode_render(&mut encoder, &self.target, timestamps.render_pass()?);
        let resolved = timestamps.resolve(&mut encoder);
        let commands = encoder.finish();

        let submitting = Instant::now();
        self.gpu.queue.submit([commands]);
        let submitted = Instant::now();
        // Straight after the submit, with nothing that can fail in between,
        // so a submitted frame always comes back to the timer; after the
        // instant is taken, so the map request is not charged to the submit.
        self.timer.submitted(resolved);

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
        if let FrameScale::Measured(scale) = self.cost.frame_scale() {
            self.outcome.frame_scales.push(scale);
        }
        // Counted after the timed intervals, so the bookkeeping is not charged
        // to the encode time it is compared with. Every slice is one compute
        // pass with one dispatch, in every workload.
        if new_job.is_some() {
            self.outcome.submitted += 1;
        }
        self.admission
            .planned(new_job.map(|job| (job, self.slices.len())), &plan);
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
            if missed.input.get() < STARTUP_FRAMES {
                self.outcome.startup_missed += 1;
            }
        }
        self.outcome.missed += plan.missed.len() as u64;
    }

    /// Feeds one frame's measurements to the cost model and the report: each
    /// slice's, then the frame's ratio over the timed slices that had a frame
    /// estimate when it was planned: what they were planned at against what
    /// they measured.
    fn absorb(&mut self, timing: FrameTiming<Tag>) {
        self.outcome
            .pass_frames
            .push(pass_overlap::Frame::of(&timing, |&(_, slice, _)| slice));
        let mut gpu = Micros::ZERO;
        let (mut predicted, mut in_ratio) = (Micros::ZERO, Micros::ZERO);
        for ((_, slice, planned), duration) in &timing.compute {
            self.cost.record(*slice, *duration);
            let entry = self.outcome.measured.entry(*slice).or_insert((0, 0));
            entry.0 = entry.0.saturating_add(duration.get());
            entry.1 += 1;
            gpu = gpu.saturating_add(*duration);
            if let Some(planned) = planned {
                predicted = predicted.saturating_add(*planned);
                in_ratio = in_ratio.saturating_add(*duration);
            }
        }
        if !timing.compute.is_empty() {
            self.cost.record_frame(predicted, in_ratio);
            if let Some(ratio) = frame_ratio(predicted, in_ratio) {
                self.outcome.frame_ratios.push(ratio);
            }
        }
        if let Some(longest) = timing.compute.iter().map(|(_, d)| d.get()).max() {
            self.outcome.longest_dispatch_us.push(longest);
        }
        if let Some(render) = timing.render {
            self.render.record(render);
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
                kernels: net
                    .program()
                    .kernels()
                    .iter()
                    .map(|k| {
                        let part = k.part();
                        if part.is_whole() {
                            k.name()
                        } else {
                            format!("{}[{}/{}]", k.name(), part.index() + 1, part.count())
                        }
                    })
                    .collect(),
                kernel_parts: std::mem::take(&mut self.kernel_parts),
                counters,
            });
        }
        for &slice in &self.slices {
            self.outcome
                .estimates
                .insert(slice, self.cost.estimate(slice));
        }
        self.outcome.frame_scale = self.cost.frame_scale();
        self.outcome.frames_recorded = self.cost.frames_recorded();
        Ok(self.outcome)
    }

    /// Runs every slice the workload has once, and one render pass, to
    /// completion, so first-use costs are not charged to frame 0.
    ///
    /// A planned program runs once per slot, each slot its own submission
    /// waited for before the next. One submission of every slot was the
    /// depth model at 434 pixels and latency 13 run 14 times, about two
    /// seconds of GPU work in one command buffer, and the device was lost
    /// before the loop began (Apple M1 Pro, Metal, 2026-10-22); that is
    /// consistent with the system ending a command buffer that runs too long,
    /// which was not confirmed from a driver log.
    fn warm_up(&self) -> Result<()> {
        let device = &self.gpu.device;
        let submit_and_wait = |encoder: wgpu::CommandEncoder| -> Result<()> {
            self.gpu.queue.submit([encoder.finish()]);
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .context("device poll failed")?;
            Ok(())
        };
        let encoder = || {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("beatlm-bench warm-up"),
            })
        };
        match &self.net {
            Some(net) => {
                for slot in 0..net.slots() {
                    let mut slot_encoder = encoder();
                    for slice in 0..net.kernel_count() {
                        net.encode(&mut slot_encoder, slot, SliceId::new(slice), None)?;
                    }
                    submit_and_wait(slot_encoder)?;
                }
            }
            None => {
                let mut slice_encoder = encoder();
                self.workload.encode_slice(&mut slice_encoder, None);
                submit_and_wait(slice_encoder)?;
            }
        }
        let mut render_encoder = encoder();
        self.workload
            .encode_render(&mut render_encoder, &self.target, None);
        submit_and_wait(render_encoder)?;
        // The network's uploads leave staging memory that wgpu releases on a
        // poll after the work completes, and the allocation counters are read
        // next, so it gets a second poll. The synthetic load has no uploads.
        if self.net.is_some() {
            device
                .poll(wgpu::PollType::wait_indefinitely())
                .context("device poll failed")?;
        }
        Ok(())
    }
}

/// The texture the render pass draws into: the bench has no window, so an
/// offscreen [`TARGET_FORMAT`] texture of [`RENDER_WIDTH`] by
/// [`RENDER_HEIGHT`] pixels that nothing reads.
pub fn offscreen_target(device: &wgpu::Device) -> wgpu::TextureView {
    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("beatlm-bench target"),
        size: wgpu::Extent3d {
            width: RENDER_WIDTH,
            height: RENDER_HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TARGET_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    target.create_view(&wgpu::TextureViewDescriptor::default())
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
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn the_default_estimator_is_the_cores_default() {
        assert_eq!(estimator(Args::default().estimator), Estimator::default());
    }

    #[test]
    fn every_estimator_the_options_name_is_valid() {
        // Each is valid (the unwrap), and only the frame-scaled one has a
        // frame scale.
        for kind in EstimatorKind::ALL {
            let cost = CostModel::new(SLICE_PRIOR, estimator(kind)).unwrap();
            let warming = matches!(cost.frame_scale(), FrameScale::WarmingUp { .. });
            assert_eq!(warming, kind == EstimatorKind::FrameScaled, "{kind:?}");
        }
    }
}
