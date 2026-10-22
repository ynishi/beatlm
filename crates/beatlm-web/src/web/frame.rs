//! The host's side of each frame, run from `requestAnimationFrame`.
//!
//! [`run`] registers one callback with the browser, and each callback is one
//! [`Host::turn`], which is one frame: the [`Pacer`] reads the callback's
//! timestamp, the frame's start, to say which frame it is and how long the
//! display's period is, and the frame runs to its end before the turn returns,
//! in the order the `beatlm` crate documentation gives; then the turn asks for
//! the next callback. Nothing in a turn waits. The map callbacks of the
//! timestamp readbacks run from the event loop between turns, and a later
//! turn collects what they completed. `performance.now()` is still read
//! inside a frame, for the CPU times the report quotes, which are durations on
//! one clock and need no frame time.
//!
//! The frame's work is the bench's synchronous `tick` (`beatlm-bench`'s
//! `engine.rs`): the same budget and forecast, the same scheduler and cost
//! model, one timed compute pass per slice handed out and one timed render
//! pass; the crate documentation says what differs and why. The job is the
//! synthetic one or the depth model ([`Load`]). The depth model's slices are
//! its kernels' parts on the slot of their job, a frame submits its input only
//! when the job before it has had all its slices (the bench's `--submit free`,
//! its default for the depth workload), and the render pass is the display,
//! showing the result the frame delivers or, under the miss policy, the last
//! one delivered. A result is copied into the display in the frame that
//! delivers it, after the frame's slices and before its render pass, never
//! earlier ([`super::display`] says why).
//!
//! An on-beat run ([`OnBeatRun`]) adds one step at the start of each frame,
//! before the frame is decided: read the audio clock, let [`Beats`] decide
//! what the frame does about the beats, hand the scheduler the grids it
//! returns, and schedule the clicks about to come. The frame submits its input
//! only when an input is owed for a beat ([`crate::onbeat`], "Submission")
//! and its output slot is free ([`Beats::slot_free`]), besides the depth
//! model's own admission rule, and every submission, completed job, delivery
//! and miss is recorded for the report.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use anyhow::{Context, Result, anyhow, bail};
use beatlm::gpu::{FrameTiming, MAX_SLICES, PassSpan, SliceTimer};
use beatlm_core::budget::Budget;
use beatlm_core::cost::{CostModel, Estimator, FrameScale, SliceId};
use beatlm_core::schedule::{Due, FramePlan, JobId, Scheduler, SliceRun};
use beatlm_core::sync::{MissPolicy, SyncPolicy};
use beatlm_core::time::{BeatGrid, FrameIndex, FrameInfo, Micros, TimePoint};
use beatlm_host::slots::Net;
use beatlm_host::stats::frame_ratio;
use beatlm_host::workload::Workload;
use wasm_bindgen::JsCast;
use wasm_bindgen::closure::Closure;

use super::audio::Audio;
use super::device::{self, Gpu};
use super::display::Display;
use super::page;
use crate::config::{Config, MARGIN_US, OnBeat, SLICES};
use crate::depth::{delivered_now, frames_for, still_busy};
use crate::diagnosis::tail::{Pass as TailPass, TimedFrame};
use crate::diagnosis::{self, Reason, Unstarted};
use crate::onbeat::{self, Beats};
use crate::pace::{FrameStart, NOMINAL_PERIOD, Pace, Pacer, ms_to_us};
use crate::render_cost::{RENDER_PRIOR, RenderCost};
use crate::report::{DepthReport, Environment, Outcome, STARTUP_FRAMES, render};

/// What a slice is assumed to cost until it has been measured: the bench's.
pub(super) const SLICE_PRIOR: Micros = Micros::new(2_000);
/// How long after the last frame the host waits for timings still in flight
/// before it reports without them, in microseconds: a second, whatever the
/// display's rate. The bench blocks on the device instead, which a page
/// cannot.
const DRAIN_US: u64 = 1_000_000;

/// The label a timed slice carries from the scheduler to the cost model: its
/// job, the slice, and what a frame's ratio takes the slice at when the frame
/// was planned ([`CostModel::frame_estimate`], `None` for a slice left out of
/// it), the bench's label and for its reason: the frame's measured slice time
/// is compared with what the same slices were planned at
/// ([`CostModel::record_frame`]), however many frames later the timing comes
/// back.
type Tag = (JobId, SliceId, Option<Micros>);

/// Whether the loop asks for another callback.
enum Turn {
    Continue,
    Finished,
}

/// The job each frame submits.
pub enum Load {
    /// The bench's synthetic slices and render pass.
    Synthetic,
    /// The depth model, realised and calibrated.
    Depth(Box<DepthLoad>),
}

/// The depth model as the loop runs it.
pub struct DepthLoad {
    /// The split program, with one slot per job that can be in flight.
    pub net: Net,
    /// The render pass that shows a slot's input and depth.
    pub display: Display,
    /// The cost model calibration measured the parts into.
    pub cost: CostModel,
    /// What the setup measured, for the report.
    pub report: DepthReport,
}

/// What a run is, past the device and the load.
pub struct Run {
    /// Frame indices to run.
    pub frames: u64,
    /// The promised latency in frames.
    pub latency: u32,
    /// The latency `latency_ms` asked for, when it was asked in time.
    pub asked: Option<Micros>,
    /// The display's period measured before the run, when it was: the period
    /// a latency in time was turned into frames at, and the pacer's prior.
    pub period: Option<Micros>,
    /// The job.
    pub load: Load,
    /// The statistic the loop's cost model estimates a slice with: the model
    /// a synthetic run builds here, and the placeholder beside the depth
    /// model's, which calibration built with the same estimator.
    pub estimator: Estimator,
    /// The on-beat promise and its audio clock; `None` for fixed latency.
    pub on_beat: Option<OnBeatRun>,
}

/// What an on-beat run starts from: the query's promise, and the audio
/// context the start button created.
pub struct OnBeatRun {
    /// `sync=onbeat` and its parameters.
    pub config: OnBeat,
    /// The click track on the context's clock.
    pub audio: Audio,
}

/// An on-beat run while it runs.
struct OnBeatHost {
    beats: Beats,
    audio: Audio,
}

/// The run: the device, the load, the decisions and what was measured.
pub struct Host {
    gpu: Gpu,
    workload: Workload,
    load: Load,
    /// The depth model's job submitted last, with its slices not yet handed
    /// out, while it has some and has not left the scheduler: a frame submits
    /// no input while there is one.
    busy: Option<(JobId, usize)>,
    /// Whether the display holds a delivered depth, the last one delivered:
    /// false until the first delivery.
    shown: bool,
    asked: Option<Micros>,
    period: Option<Micros>,
    /// `None` when the device has no timestamp queries.
    timer: Option<SliceTimer<Tag>>,
    scheduler: Scheduler,
    cost: CostModel,
    /// The render's measurements, which set the budgets.
    render: RenderCost,
    slices: Vec<SliceId>,
    margin: Micros,
    latency: u32,
    /// Dispatches encoded so far for each job not yet delivered or missed.
    job_dispatches: BTreeMap<u64, u64>,
    pacer: Pacer,
    performance: web_sys::Performance,
    /// When the first callback after the last frame began, in microseconds.
    drain_from_us: Option<u64>,
    outcome: Outcome,
    /// The index of the frame last run, which timings are collected against.
    current_frame: u64,
    /// The start of the frame last run, which a delivery's frame is checked
    /// against: it must have started before the beat.
    previous_start_us: Option<u64>,
    /// The beats of an on-beat run.
    on_beat: Option<OnBeatHost>,
}

impl Host {
    /// Builds the run's timer, scheduler and cost model for `run` on `gpu`,
    /// which `workload` was built on. The cost model reads with
    /// `run.estimator`.
    ///
    /// # Errors
    ///
    /// A device with timestamp queries and a latency whose jobs in flight
    /// have more slices than the timer can time, a timer or cost model that
    /// cannot be built, or a page without `performance`.
    pub fn new(gpu: Gpu, workload: Workload, mut run: Run) -> Result<Self> {
        let estimator = run.estimator;
        let on_beat = run.on_beat.is_some();
        let (slices, jobs, cost) = match &mut run.load {
            // Every frame submits a job, and a job lives `latency` frames past
            // its submission, so `latency + 1` jobs can have slices left in a
            // frame. On beat, no more than `SLOTS` are pending.
            Load::Synthetic => (
                SLICES,
                if on_beat {
                    Some(onbeat::SLOTS)
                } else {
                    run.latency.checked_add(1)
                },
                CostModel::new(SLICE_PRIOR, estimator).context("cannot build the cost model")?,
            ),
            // A frame submits only once the job before has had its slices, so
            // one job at a time has slices left. The model is calibration's,
            // built with the run's estimator.
            Load::Depth(depth) => (
                depth.net.kernel_count(),
                Some(1),
                std::mem::replace(
                    &mut depth.cost,
                    CostModel::new(SLICE_PRIOR, estimator)
                        .context("cannot build the cost model")?,
                ),
            ),
        };
        let timer = if gpu.timestamp_query {
            let capacity = jobs
                .and_then(|jobs| jobs.checked_mul(slices))
                .filter(|&capacity| capacity <= MAX_SLICES)
                .with_context(|| {
                    format!(
                        "{slices} slices for each of the jobs with slices left at latency {} \
                         are more than the GPU timer can time ({MAX_SLICES})",
                        run.latency
                    )
                })?;
            Some(
                SliceTimer::new(&gpu.device, &gpu.queue, capacity)
                    .context("cannot build the GPU timer")?,
            )
        } else {
            None
        };
        let margin = Micros::new(MARGIN_US);
        let prior = run.period.unwrap_or(NOMINAL_PERIOD);
        let policy = match &run.on_beat {
            Some(on_beat) => SyncPolicy::OnBeat {
                margin: on_beat.config.margin,
                // A placeholder that places no job: no input is submitted
                // before the first beat, and that beat's frame puts the grid
                // made from the audio clock in effect from the epoch, which
                // replaces this one ([`crate::onbeat`], "Drift").
                grid: BeatGrid::new(TimePoint::new(0), Micros::new(1_000_000))
                    .map_err(|error| anyhow!("cannot build the placeholder grid: {error}"))?,
            },
            None => SyncPolicy::FixedLatency {
                frames: run.latency,
            },
        };
        let scheduler = Scheduler::new(
            policy,
            MissPolicy::HoldPrevious,
            // Before the display's period is known, the prior; every frame
            // sets its own budget from the period measured by then.
            Budget::for_frame(prior, RENDER_PRIOR, margin),
        );
        let performance = page::window()?
            .performance()
            .context("the window has no performance clock")?;
        let mut host = Self {
            gpu,
            workload,
            load: run.load,
            busy: None,
            shown: false,
            asked: run.asked,
            period: run.period,
            timer,
            scheduler,
            cost,
            render: RenderCost::default(),
            slices: (0..slices).map(SliceId::new).collect(),
            margin,
            latency: run.latency,
            job_dispatches: BTreeMap::new(),
            pacer: Pacer::with_prior(run.frames, prior),
            performance,
            drain_from_us: None,
            outcome: Outcome::default(),
            current_frame: 0,
            previous_start_us: None,
            on_beat: run.on_beat.map(|on_beat| OnBeatHost {
                beats: Beats::new(
                    on_beat.config.tempo,
                    on_beat.config.margin,
                    on_beat.config.beats,
                ),
                audio: on_beat.audio,
            }),
        };
        host.outcome.diagnosis.job_estimate_us.0 = host.job_estimate();
        Ok(host)
    }

    /// One animation-frame callback, which is one frame, and once the frames
    /// are done a turn of the drain. `timestamp` is the callback's argument,
    /// the display's frame time in milliseconds.
    ///
    /// # Errors
    ///
    /// A lost device, whose frames from then on would measure nothing, or a
    /// frame that fails.
    fn turn(&mut self, timestamp: f64) -> Result<Turn> {
        if let Some(lost) = device::device_lost() {
            bail!("the device was lost: {lost}");
        }
        let began = self.performance.now();
        let now_us = ms_to_us(timestamp);
        match self.pacer.frame_at(now_us) {
            Pace::Frame(start) => {
                self.tick(start, began)?;
                Ok(Turn::Continue)
            }
            Pace::Done => {
                self.read_back();
                let drain_from = *self.drain_from_us.get_or_insert(now_us);
                let in_flight = self.timer.as_ref().map_or(0, SliceTimer::in_flight);
                if in_flight == 0 || now_us.saturating_sub(drain_from) >= DRAIN_US {
                    Ok(Turn::Finished)
                } else {
                    Ok(Turn::Continue)
                }
            }
        }
    }

    /// Runs one frame to its end: decide, encode, submit, start the readback,
    /// collect what has completed, deliver.
    fn tick(&mut self, start: FrameStart, began: f64) -> Result<()> {
        let frame = FrameInfo {
            index: FrameIndex::new(start.index),
            start: TimePoint::new(start.start_us),
            period: start.period,
        };
        // Whether this frame's input is owed: every frame's under fixed
        // latency, a beat's on beat.
        let owed = match &mut self.on_beat {
            Some(on_beat) => {
                let owed = beat_step(
                    on_beat,
                    &mut self.scheduler,
                    &self.pacer,
                    &self.performance,
                    start,
                )?;
                // Jobs are numbered in submission order from zero, so the
                // next one's number is the count submitted.
                owed && on_beat.beats.slot_free(self.outcome.submitted)
            }
            None => true,
        };

        let planning = self.performance.now();
        // This frame's budget follows the last render measured and the
        // display's period; the frames to come are forecast from the median of
        // the recent renders, as the bench does and for its reason
        // (`Scheduler::set_expected_budget`), at the same period.
        let budget = self.render.budget(start.period, self.margin);
        let forecast = self.render.expected(start.period, self.margin);
        self.scheduler.set_expected_budget(forecast);
        self.outcome.expected_budget_us.push(forecast.get().get());
        let takes_input = owed
            && match self.load {
                Load::Synthetic => true,
                Load::Depth(_) => self.busy.is_none(),
            };
        let new_job = if takes_input {
            let job = self
                .scheduler
                .submit(frame, self.slices.clone(), &self.cost)
                .context("the scheduler rejected the frame's job")?;
            if let (Some(on_beat), Some(Due::Beat(due))) =
                (&mut self.on_beat, self.scheduler.due(job))
            {
                on_beat.beats.submitted(job.get(), due.get());
            }
            Some(job)
        } else {
            None
        };
        let plan = self.scheduler.plan_frame(frame, budget, &self.cost);
        self.current_frame = start.index;
        // The render pass shows what this frame delivers: the slot of the job
        // the display copies in this frame, after the slices that may finish
        // it and before the render pass.
        let delivering = match &self.load {
            Load::Depth(depth) => delivered_now(&plan).map(|job| depth.net.slot_of(job.get())),
            Load::Synthetic => None,
        };
        self.shown |= delivering.is_some();

        let encoding = self.performance.now();
        let mut encoder = self
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("beatlm-web frame"),
            });
        // The canvas hands out a new texture every frame; the browser shows it
        // once the callback returns.
        let canvas = match self.gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture)
            | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => Some(texture),
            _ => None,
        };
        let view = canvas.as_ref().map(|texture| {
            texture
                .texture
                .create_view(&wgpu::TextureViewDescriptor::default())
        });
        // The estimates the plan was made with; nothing records into the cost
        // model before the frame is encoded.
        let cost = &self.cost;
        let resolved = match &mut self.timer {
            Some(timer) => {
                let mut timestamps = timer.begin_frame(frame.index);
                for run in &plan.runs {
                    let writes = timestamps.compute_pass((
                        run.job,
                        run.slice,
                        cost.frame_estimate(run.slice),
                    ))?;
                    encode_slice(&self.load, &self.workload, &mut encoder, run, writes)?;
                }
                encode_delivery(&self.load, &mut encoder, delivering);
                if let Some(view) = &view {
                    let writes = timestamps.render_pass()?;
                    encode_render(
                        &self.load,
                        &self.workload,
                        self.shown,
                        &mut encoder,
                        view,
                        writes,
                    );
                }
                Some(timestamps.resolve(&mut encoder))
            }
            None => {
                for run in &plan.runs {
                    encode_slice(&self.load, &self.workload, &mut encoder, run, None)?;
                }
                encode_delivery(&self.load, &mut encoder, delivering);
                if let Some(view) = &view {
                    encode_render(
                        &self.load,
                        &self.workload,
                        self.shown,
                        &mut encoder,
                        view,
                        None,
                    );
                }
                None
            }
        };
        let commands = encoder.finish();

        let submitting = self.performance.now();
        self.gpu.queue.submit([commands]);
        let submitted = self.performance.now();
        // Straight after the submit, with nothing that can fail in between,
        // so a submitted frame always comes back to the timer.
        if let (Some(timer), Some(resolved)) = (&mut self.timer, resolved) {
            timer.submitted(resolved);
        }
        match canvas {
            Some(texture) => self.gpu.queue.present(texture),
            None => self.outcome.canvas_unavailable += 1,
        }

        self.outcome.plan_us.push(between(planning, encoding));
        self.outcome.encode_us.push(between(encoding, submitting));
        self.outcome.submit_us.push(between(submitting, submitted));
        self.outcome
            .dispatches_per_frame
            .push(plan.runs.len() as u64);
        if let FrameScale::Measured(scale) = self.cost.frame_scale() {
            self.outcome.diagnosis.frame_scales.push(scale);
        }
        if new_job.is_some() {
            self.outcome.submitted += 1;
        }
        if let Load::Depth(_) = self.load {
            self.busy = still_busy(
                self.busy,
                new_job.map(|job| (job, self.slices.len())),
                &plan,
            );
        }
        // After the frame's CPU times are taken, so the diagnosis is not
        // charged to `cpu_plan_us`, the bench's line; before the slices are
        // counted, which it reads.
        self.diagnose(start.index, &plan, budget.get(), forecast.get());
        for run in &plan.runs {
            *self.job_dispatches.entry(run.job.get()).or_default() += 1;
        }
        self.outcome.cpu_us.push(between(began, submitted));

        self.read_back();
        self.complete(start, &plan);
        self.previous_start_us = Some(start.start_us);
        self.outcome
            .tick_us
            .push(between(began, self.performance.now()));
        Ok(())
    }

    /// Feeds every frame whose map has completed to the cost model and the
    /// report. The map of the frame just submitted was requested with its
    /// submission.
    ///
    /// `Device::poll` is not called: in a browser it does nothing, and the map
    /// callbacks have run from the event loop before this callback began.
    fn read_back(&mut self) {
        let ready = match &mut self.timer {
            Some(timer) => timer.collect_ready(),
            None => return,
        };
        for timing in ready {
            let lag = self.current_frame.saturating_sub(timing.frame.get());
            self.outcome.diagnosis.lag_frames.push(lag);
            self.absorb(timing);
        }
    }

    /// Records why the plan's misses were missed, what the projection read for
    /// a job abandoned before it started, and which jobs started, before the
    /// frame's slices are counted ([`crate::diagnosis`]).
    fn diagnose(&mut self, index: u64, plan: &FramePlan, now: Micros, expected: Micros) {
        for missed in &plan.missed {
            let dispatched = self
                .job_dispatches
                .get(&missed.job.get())
                .copied()
                .unwrap_or(0);
            let due = match missed.due {
                Due::Frame(due) => due.get(),
                Due::Beat(_) => index,
            };
            let reason = diagnosis::reason(dispatched, due, index);
            self.outcome.diagnosis.missed(reason);
            if reason == Reason::Unstarted {
                // What the projection read: the planning estimates.
                let estimates: Vec<Micros> = self
                    .slices
                    .iter()
                    .map(|&s| self.cost.planning_estimate(s))
                    .collect();
                self.outcome.diagnosis.abandoned.push(Unstarted {
                    frame: index,
                    now_us: now.get(),
                    expected_us: expected.get(),
                    largest_us: estimates.iter().map(|e| e.get()).max().unwrap_or(0),
                    sum_us: estimates.iter().map(|e| e.get()).sum(),
                    frames_left: due.saturating_sub(index),
                    projected_end: diagnosis::project(&estimates, now, expected),
                });
            }
        }
        let mut started: Vec<u64> = Vec::new();
        for run in &plan.runs {
            let job = run.job.get();
            if !self.job_dispatches.contains_key(&job) && !started.contains(&job) {
                started.push(job);
                self.outcome.diagnosis.starts.push(index);
            }
        }
    }

    /// The sum of every slice's planning estimate now: the job as the
    /// projection sees it.
    fn job_estimate(&self) -> u64 {
        self.slices
            .iter()
            .map(|&s| self.cost.planning_estimate(s).get())
            .sum()
    }

    /// Delivers the plan's results and counts its misses; on beat, records
    /// the jobs whose last slice ran, and where each result landed.
    fn complete(&mut self, start: FrameStart, plan: &FramePlan) {
        let index = start.index;
        if let Some(on_beat) = &mut self.on_beat {
            for run in &plan.runs {
                if run.position + 1 == self.slices.len() {
                    on_beat.beats.completed(run.job.get(), start.start_us);
                }
            }
            for delivery in &plan.delivered {
                on_beat.beats.delivered(
                    delivery.job.get(),
                    start.start_us,
                    self.previous_start_us,
                    start.period.get(),
                );
            }
            for missed in &plan.missed {
                if let Due::Beat(due) = missed.due {
                    on_beat.beats.missed(missed.job.get(), due.get());
                }
            }
        }
        for delivery in &plan.delivered {
            let latency = index.saturating_sub(delivery.input.get());
            self.outcome.latencies.push(latency);
            self.outcome.delivered += 1;
            // On beat there is no latency in frames to keep.
            if self.on_beat.is_none() && latency != u64::from(self.latency) {
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
    /// slice's, then the frame's measured slice time against the estimates
    /// its timed slices were planned with.
    fn absorb(&mut self, timing: FrameTiming<Tag>) {
        let passes: Vec<(SliceId, Micros)> = timing
            .compute
            .iter()
            .map(|&((_, slice, _), duration)| (slice, duration))
            .collect();
        self.outcome.diagnosis.passes(&passes);
        self.outcome
            .pass_frames
            .push(beatlm_host::pass_overlap::Frame::of(
                &timing,
                |&(_, slice, _)| slice,
            ));
        // Only a depth run writes the `tail_*` lines that read them.
        if matches!(self.load, Load::Depth(_)) {
            self.outcome.diagnosis.frames.push(timed(&timing));
        }
        let mut gpu = Micros::ZERO;
        // The ratio's two sides, over the timed slices that had a frame
        // estimate when the frame was planned.
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
                self.outcome.diagnosis.frame_ratios.push(ratio);
            }
        }
        if let Some(longest) = timing.compute.iter().map(|(_, d)| d.get()).max() {
            self.outcome.longest_dispatch_us.push(longest);
        }
        if let Some(render) = timing.render {
            self.render.record(render);
            if timing.compute.is_empty() {
                self.outcome.render_alone_us.push(render.get());
            } else {
                self.outcome.render_with_slices_us.push(render.get());
            }
            self.outcome
                .gpu_frame_us
                .push(gpu.saturating_add(render).get());
        }
    }

    /// Closes the run and returns what it measured.
    fn finish(mut self) -> Outcome {
        self.outcome.diagnosis.job_estimate_us.1 = self.job_estimate();
        self.outcome.latency = self.latency;
        if let Some(asked) = self.asked {
            self.outcome.latency_asked_us = Some(asked.get());
            self.outcome.latency_period_us = self.period.map(Micros::get);
        }
        if let Load::Depth(depth) = self.load {
            self.outcome.depth = Some(depth.report);
        }
        self.outcome.dropped = self.pacer.dropped();
        self.outcome.lateness_us = self.pacer.lateness_us().to_vec();
        self.outcome.display_period_us = self.pacer.measured_period().map(Micros::get);
        self.outcome.interval_us = self.pacer.intervals_us().to_vec();
        self.outcome.pending_at_end = self.scheduler.pending();
        if let Some(timer) = &self.timer {
            self.outcome.timer = timer.stats();
        }
        for &slice in &self.slices {
            self.outcome
                .estimates
                .insert(slice, self.cost.estimate(slice));
        }
        self.outcome.diagnosis.frame_scale = self.cost.frame_scale();
        self.outcome.diagnosis.frames_recorded = self.cost.frames_recorded();
        self.outcome.gpu_errors = device::gpu_errors();
        if let Some(on_beat) = &self.on_beat {
            on_beat.audio.close();
            let mut summary = on_beat.beats.summary();
            (summary.clicks_scheduled, summary.clicks_late) = on_beat.audio.clicks();
            // What `capacity_frames_needed` holds a job to: the frames of the
            // plan's shortest beat, before or after a tempo change, less the
            // margin.
            let shortest = on_beat.beats.tempo().shortest_beat().get();
            let room = Micros::new(shortest.saturating_sub(summary.margin_us));
            self.outcome.latency = frames_for(room, self.pacer.period());
            self.outcome.on_beat = Some(summary);
        }
        self.outcome
    }
}

/// The on-beat step at the start of a frame: reads the audio clock, puts in
/// effect the grids the beats decide on, and schedules the clicks to come.
/// Whether an input is owed.
///
/// # Errors
///
/// A grid the scheduler or the core refuses, or a click the audio context
/// refuses.
fn beat_step(
    on_beat: &mut OnBeatHost,
    scheduler: &mut Scheduler,
    pacer: &Pacer,
    performance: &web_sys::Performance,
    start: FrameStart,
) -> Result<bool> {
    let beats = &mut on_beat.beats;
    if beats.audio_time_s(0).is_none() {
        beats.begin(on_beat.audio.current_time(), on_beat.audio.sample_rate());
    }
    // The pacer has read this frame's callback, so its epoch is known.
    let epoch = pacer.epoch_us().unwrap_or(0);
    beats.record(&on_beat.audio.reading(performance), epoch);
    let decided = beats
        .frame(start.start_us)
        .map_err(|error| anyhow!("cannot make the beat grid: {error}"))?;
    for (from, grid) in decided.grids {
        scheduler
            .replace_beat_grid(from, grid)
            .context("the scheduler refused the beat grid")?;
    }
    // Inputs at beats 0 to `beats - 1` are placed up to beat `beats`.
    on_beat.audio.schedule(beats, beats.beats_asked())?;
    Ok(decided.owed)
}

/// One frame's timing as the `tail_*` lines read it: each pass's duration, as
/// the cost model receives it, with its timestamps.
fn timed(timing: &FrameTiming<Tag>) -> TimedFrame {
    let pass = |duration: Micros, span: &PassSpan| TailPass {
        us: duration.get(),
        begin_ns: span.begin_ns,
        end_ns: span.end_ns,
    };
    TimedFrame {
        frame: timing.frame.get(),
        compute: timing
            .compute
            .iter()
            .zip(&timing.compute_spans)
            .map(|(&((_, slice, _), duration), span)| (slice, pass(duration, span)))
            .collect(),
        render: timing
            .render
            .zip(timing.render_span.as_ref())
            .map(|(duration, span)| pass(duration, span)),
    }
}

/// Records the slice `run` hands out as its own compute pass.
fn encode_slice(
    load: &Load,
    workload: &Workload,
    encoder: &mut wgpu::CommandEncoder,
    run: &SliceRun,
    writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
) -> Result<()> {
    match load {
        Load::Synthetic => {
            workload.encode_slice(encoder, writes);
            Ok(())
        }
        Load::Depth(depth) => {
            let slot = depth.net.slot_of(run.job.get());
            depth.net.encode(encoder, slot, run.slice, writes)?;
            Ok(())
        }
    }
}

/// Records the copy of the depth of slot `delivering`, whose job the frame
/// delivers, into the display: between the frame's slices and its render
/// pass, untimed. Nothing for a frame that delivers nothing or a synthetic
/// run.
fn encode_delivery(load: &Load, encoder: &mut wgpu::CommandEncoder, delivering: Option<u32>) {
    if let (Load::Depth(depth), Some(slot)) = (load, delivering) {
        depth.display.deliver(encoder, slot);
    }
}

/// Records the frame's render pass into `view`: the synthetic one, or the
/// display, with the depth delivered last once `shown`.
fn encode_render(
    load: &Load,
    workload: &Workload,
    shown: bool,
    encoder: &mut wgpu::CommandEncoder,
    view: &wgpu::TextureView,
    writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
) {
    match load {
        Load::Synthetic => workload.encode_render(encoder, view, writes),
        Load::Depth(depth) => depth.display.encode(encoder, view, shown, writes),
    }
}

/// Microseconds from `start` to `end`, both `performance.now()` readings.
fn between(start: f64, end: f64) -> u64 {
    ms_to_us(end - start)
}

/// The callback the loop registers, kept where the callback itself can reach
/// it to register itself again.
type Callback = Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>;

/// Drives `host` from `requestAnimationFrame` until its frames are done and
/// their timings drained, then writes the report into the page.
///
/// The callback holds a handle on itself so each turn can ask for the next;
/// that cycle is never broken, so the callback lives as long as the page,
/// which runs once.
///
/// # Errors
///
/// A page that refuses the first callback; later failures are written into
/// the page, since nothing is left to return them to.
pub fn run(host: Host, env: Environment, config: Config) -> Result<()> {
    let callback: Callback = Rc::new(RefCell::new(None));
    let again = Rc::clone(&callback);
    let mut host = Some(host);
    *callback.borrow_mut() = Some(Closure::new(move |timestamp: f64| {
        let Some(running) = host.as_mut() else {
            return;
        };
        let next = match running.turn(timestamp) {
            Ok(Turn::Continue) => request(&again),
            Ok(Turn::Finished) => {
                if let Some(done) = host.take() {
                    page::show(&render(&env, &config, &done.finish()));
                }
                return;
            }
            Err(error) => Err(error),
        };
        if let Err(error) = next {
            host = None;
            page::fail(&error);
        }
    }));
    request(&callback)
}

/// Asks the browser to call `callback` before the next repaint.
fn request(callback: &Callback) -> Result<()> {
    let slot = callback.borrow();
    let Some(closure) = slot.as_ref() else {
        bail!("the frame callback is gone");
    };
    page::window()?
        .request_animation_frame(closure.as_ref().unchecked_ref())
        .map_err(|error| anyhow!("requestAnimationFrame failed: {error:?}"))?;
    Ok(())
}
