//! The host's side of one frame, in the order the umbrella crate documents,
//! shared by every concurrency candidate.
//!
//! A candidate differs in which thread owns an [`Engine`] and in how it is
//! called: [`Engine::tick_sync`] is a plain call, [`Engine::tick_async`] is a
//! future, and the threaded candidate calls `tick_sync` from a thread of its
//! own. The decisions, the encoding, the submission and the readback are the
//! same code in all three, which is what lets their timings be compared: the
//! frame's step is [`FrameStep`], which the browser host drives too, and the
//! engine adds the device, the timer and the clock. The engine reads the clock
//! to measure itself; the step, the core and `beatlm::gpu` read none.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context as TaskContext, Poll};
use std::time::Instant;

use anyhow::{Context, Result};
use beatlm::gpu::{MAX_SLICES, SliceTimer, TimerStats};
use beatlm_core::cost::{CostModel, Estimator, SliceId};
use beatlm_core::kernel::Program;
use beatlm_core::sync::{MissPolicy, SyncPolicy};
use beatlm_core::time::{FrameIndex, FrameInfo, Micros, TimePoint};
use beatlm_host::frame::{InFlight, Tag, jobs_with_slices, timer_capacity};
use beatlm_host::presets::{self, SLICE_PRIOR};
use beatlm_host::slots;
use beatlm_host::step::{FrameStep, StepConfig, Submitted, Tally};
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

/// Everything a run measured: what the frame's step counted ([`Tally`]) and
/// what the bench measured around it.
///
/// The per-frame vectors hold one sample per frame the engine handled, except
/// where a field says otherwise.
#[derive(Default)]
pub struct Outcome {
    /// The step's counts and samples, which the report's shared lines are
    /// written from.
    pub tally: Tally,
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
    /// How long after its period's boundary each frame actually began, in
    /// microseconds; nonzero when the previous frame overran or the sleep
    /// overshot.
    pub lateness_us: Vec<u64>,
    /// Frame periods given up because a frame began a whole period late.
    pub dropped: u64,
    /// What the timer could not measure.
    pub timer: TimerStats,
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
/// It holds the frame's step (the scheduler and the cost model), the timer
/// and the workload of one run, so a fresh engine is a fresh run on the same
/// device. Whichever thread calls it is the thread that touches the device.
pub struct Engine<'g> {
    gpu: &'g Gpu,
    workload: Workload,
    /// The offscreen texture the render pass draws into.
    target: wgpu::TextureView,
    net: Option<Net>,
    kernel_parts: Vec<KernelParts>,
    watch: Option<Watch>,
    timer: SliceTimer<Tag>,
    step: FrameStep,
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
        let cost = match measured_cost {
            Some(cost) => cost,
            None => CostModel::new(SLICE_PRIOR, estimator(args.estimator))
                .context("cannot build the cost model")?,
        };
        let step = FrameStep::new(
            StepConfig {
                policy,
                miss: MissPolicy::HoldPrevious,
                slices,
                margin: Micros::new(args.margin_us),
                prior_period: FRAME_PERIOD,
                submit: args.submit,
            },
            cost,
        );
        let mut engine = Self {
            gpu,
            workload,
            target,
            net,
            kernel_parts,
            watch: None,
            timer,
            step,
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
        let submitted = self.submit_frame(tick)?;
        self.read_back(wgpu::PollType::Poll)?;
        self.step.complete(submitted);
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
        let submitted = self.submit_frame(tick)?;
        ReadbackTurn { engine: self }.await?;
        self.step.complete(submitted);
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

    /// Polls the device with `poll`, and feeds what has completed to the
    /// frame's step. The frame's map was requested when it was submitted.
    fn read_back(&mut self, poll: wgpu::PollType) -> Result<()> {
        self.gpu.device.poll(poll).context("device poll failed")?;
        for timing in self.timer.collect_ready() {
            self.step.absorb(&timing);
        }
        Ok(())
    }

    /// Decides the frame, encodes it and submits it, and records where the CPU
    /// time went.
    fn submit_frame(&mut self, tick: &Tick) -> Result<Submitted> {
        let frame = FrameInfo {
            index: FrameIndex::new(tick.index),
            start: TimePoint::new(tick.start_us),
            period: FRAME_PERIOD,
        };

        let planning = Instant::now();
        // At the bench's fixed period; every frame's input is owed.
        let planned = self.step.plan(frame, true)?;
        let plan = planned.plan();

        let encoding = Instant::now();
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("beatlm-bench frame"),
            });
        let mut timestamps = self.timer.begin_frame(frame.index);
        for run in &plan.runs {
            let writes = timestamps.compute_pass(self.step.tag(run))?;
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
        // Counted after the timed intervals, so the bookkeeping is not charged
        // to the encode time it is compared with.
        let submitted_frame = self.step.after_submit(planned);
        self.outcome
            .cpu_us
            .push(micros_between(tick.began, submitted));
        Ok(submitted_frame)
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
        self.outcome.tally = self.step.finish();
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
    use beatlm_core::cost::FrameScale;

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
