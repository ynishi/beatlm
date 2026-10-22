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
//! The frame's work is the bench's synchronous `tick`: the same step
//! ([`FrameStep`], which [`beatlm_host::step`] says how to drive), one timed
//! compute pass per slice handed out and one timed render pass; the crate
//! documentation says what differs and why. The job is the synthetic one or
//! the depth model ([`Load`]). The depth model's slices are its kernels' parts
//! on the slot of their job, a frame submits its input only when the job
//! before it has had all its slices ([`DEPTH_SUBMIT`]), and the render pass is
//! the display, showing the result the frame delivers or, under the miss
//! policy, the last one delivered. A result is copied into the display in the
//! frame that delivers it, after the frame's slices and before its render pass, never
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
use std::rc::Rc;

use anyhow::{Context, Result, anyhow, bail};
use beatlm::gpu::{FrameTiming, MAX_SLICES, PassSpan, SliceTimer};
use beatlm_core::cost::{CostModel, Estimator, SliceId};
use beatlm_core::schedule::{Due, FramePlan, SliceRun};
use beatlm_core::sync::{MissPolicy, SyncPolicy};
use beatlm_core::time::{BeatGrid, FrameIndex, FrameInfo, Micros, TimePoint};
use beatlm_host::frame::{InFlight, Submit, Tag, jobs_with_slices, timer_capacity};
use beatlm_host::presets::{MARGIN_US, SLICE_PRIOR};
use beatlm_host::slots::Net;
use beatlm_host::step::{FrameStep, StepConfig, Submitted};
use beatlm_host::workload::Workload;
use wasm_bindgen::JsCast;
use wasm_bindgen::closure::Closure;

use super::audio::Audio;
use super::device::{self, Gpu};
use super::display::Display;
use super::page;
use crate::config::{Config, OnBeat, SLICES};
use crate::depth::{delivered_now, frames_for};
use crate::diagnosis::tail::{Pass as TailPass, TimedFrame};
use crate::diagnosis::{self, Reason, Unstarted};
use crate::onbeat::{self, Beats};
use crate::pace::{FrameStart, NOMINAL_PERIOD, Pace, Pacer, ms_to_us};
use crate::report::{DepthReport, Environment, Outcome, render};

/// How long after the last frame the host waits for timings still in flight
/// before it reports without them, in microseconds: a second, whatever the
/// display's rate. The bench blocks on the device instead, which a page
/// cannot.
const DRAIN_US: u64 = 1_000_000;

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

/// When a frame of the depth model submits its input: only once the job
/// before it has had all its slices, the bench's default for its `depth`
/// workload.
pub const DEPTH_SUBMIT: Submit = Submit::Free;

impl Load {
    /// When a frame submits its input: every frame for the synthetic job, as
    /// the bench's default, and [`DEPTH_SUBMIT`] for the depth model.
    const fn submit(&self) -> Submit {
        match self {
            Self::Synthetic => Submit::Every,
            Self::Depth(_) => DEPTH_SUBMIT,
        }
    }
}

/// How many jobs a run's promise lets be pending at once: `latency + 1` at a
/// fixed latency, and on beat the [`onbeat::SLOTS`] the host keeps.
pub const fn in_flight(on_beat: bool, latency: u32) -> InFlight {
    if on_beat {
        InFlight::Bounded(onbeat::SLOTS)
    } else {
        InFlight::Latency(latency)
    }
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
    /// Whether the display holds a delivered depth, the last one delivered:
    /// false until the first delivery.
    shown: bool,
    asked: Option<Micros>,
    period: Option<Micros>,
    /// `None` when the device has no timestamp queries.
    timer: Option<SliceTimer<Tag>>,
    /// The scheduler, the cost model and what they counted. Its admission is
    /// the load's rule; on beat, an input is also owed and its slot free.
    step: FrameStep,
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
        let submit = run.load.submit();
        let jobs = jobs_with_slices(submit, in_flight(run.on_beat.is_some(), run.latency));
        let (slices, cost) = match &mut run.load {
            Load::Synthetic => (
                SLICES,
                CostModel::new(SLICE_PRIOR, estimator).context("cannot build the cost model")?,
            ),
            // The model is calibration's, built with the run's estimator.
            Load::Depth(depth) => (
                depth.net.kernel_count(),
                std::mem::replace(
                    &mut depth.cost,
                    CostModel::new(SLICE_PRIOR, estimator)
                        .context("cannot build the cost model")?,
                ),
            ),
        };
        let timer = if gpu.timestamp_query {
            let capacity = jobs
                .and_then(|jobs| timer_capacity(slices, jobs))
                .with_context(|| {
                    let jobs =
                        jobs.map_or_else(|| String::from("too many"), |jobs| jobs.to_string());
                    format!(
                        "{slices} slices for each of {jobs} jobs with slices left (latency {}) \
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
        let step = FrameStep::new(
            StepConfig {
                policy,
                miss: MissPolicy::HoldPrevious,
                slices,
                margin: Micros::new(MARGIN_US),
                // Before the display's period is known, the prior; every frame
                // sets its own budget from the period measured by then.
                prior_period: prior,
                submit,
            },
            cost,
        );
        let performance = page::window()?
            .performance()
            .context("the window has no performance clock")?;
        let mut host = Self {
            gpu,
            workload,
            load: run.load,
            shown: false,
            asked: run.asked,
            period: run.period,
            timer,
            step,
            pacer: Pacer::with_prior(run.frames, prior),
            performance,
            drain_from_us: None,
            // On beat, replaced at the end by the frames the beats allow.
            outcome: Outcome {
                latency: run.latency,
                ..Outcome::default()
            },
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
                    &mut self.step,
                    &self.pacer,
                    &self.performance,
                    start,
                )?;
                owed && on_beat.beats.slot_free(self.step.submitted())
            }
            None => true,
        };

        let planning = self.performance.now();
        // At the display's measured period.
        let planned = self.step.plan(frame, owed)?;
        self.outcome
            .expected_budget_us
            .push(planned.expected().get().get());
        if let (Some(on_beat), Some(job), Some(Due::Beat(due))) =
            (&mut self.on_beat, planned.job(), planned.due())
        {
            on_beat.beats.submitted(job.get(), due.get());
        }
        let plan = planned.plan();
        self.current_frame = start.index;
        // The render pass shows what this frame delivers: the slot of the job
        // the display copies in this frame, after the slices that may finish
        // it and before the render pass.
        let delivering = match &self.load {
            Load::Depth(depth) => delivered_now(plan).map(|job| depth.net.slot_of(job.get())),
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
        let resolved = match &mut self.timer {
            Some(timer) => {
                let mut timestamps = timer.begin_frame(frame.index);
                for run in &plan.runs {
                    let writes = timestamps.compute_pass(self.step.tag(run))?;
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
        // After the frame's CPU times are taken, so the diagnosis is not
        // charged to `cpu_plan_us`, the bench's line; before the step counts
        // the frame, since it reads the dispatches of the frames before.
        self.diagnose(
            start.index,
            plan,
            planned.budget().get(),
            planned.expected().get(),
        );
        let submitted_frame = self.step.after_submit(planned);
        self.outcome.cpu_us.push(between(began, submitted));

        self.read_back();
        self.complete(start, submitted_frame);
        self.previous_start_us = Some(start.start_us);
        self.outcome
            .tick_us
            .push(between(began, self.performance.now()));
        Ok(())
    }

    /// Feeds every frame whose map has completed to the frame's step and the
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
            self.absorb(&timing);
        }
    }

    /// Records why the plan's misses were missed, what the projection read for
    /// a job abandoned before it started, and which jobs started, before the
    /// step counts the frame's slices ([`FrameStep::dispatched`];
    /// [`crate::diagnosis`]).
    fn diagnose(&mut self, index: u64, plan: &FramePlan, now: Micros, expected: Micros) {
        for missed in &plan.missed {
            let dispatched = self.step.dispatched(missed.job);
            let due = match missed.due {
                Due::Frame(due) => due.get(),
                Due::Beat(_) => index,
            };
            let reason = diagnosis::reason(dispatched, due, index);
            self.outcome.diagnosis.missed(reason);
            if reason == Reason::Unstarted {
                // What the projection read.
                let estimates = planning_estimates(&self.step);
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
            if self.step.dispatched(run.job) == 0 && !started.contains(&job) {
                started.push(job);
                self.outcome.diagnosis.starts.push(index);
            }
        }
    }

    /// The sum of every slice's planning estimate now: the job as the
    /// projection sees it.
    fn job_estimate(&self) -> u64 {
        planning_estimates(&self.step).iter().map(|e| e.get()).sum()
    }

    /// Has the step count the frame's deliveries and misses; on beat, records
    /// the jobs whose last slice ran, and where each result landed.
    fn complete(&mut self, start: FrameStart, submitted: Submitted) {
        let plan = self.step.complete(submitted);
        if let Some(on_beat) = &mut self.on_beat {
            let slices = self.step.slices().len();
            for run in &plan.runs {
                if run.position + 1 == slices {
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
    }

    /// Feeds one frame's measurements to the frame's step, and records the
    /// page's own readings of it: how late it was collected, its passes for
    /// the diagnosis and the `tail_*` lines, and its render alone or with
    /// slices.
    fn absorb(&mut self, timing: &FrameTiming<Tag>) {
        let passes: Vec<(SliceId, Micros)> = timing
            .compute
            .iter()
            .map(|&((_, slice, _), duration)| (slice, duration))
            .collect();
        self.outcome.diagnosis.passes(&passes);
        // Only a depth run writes the `tail_*` lines that read them.
        if matches!(self.load, Load::Depth(_)) {
            self.outcome.diagnosis.frames.push(timed(timing));
        }
        let absorbed = self.step.absorb(timing);
        let lag = self.current_frame.saturating_sub(absorbed.frame.get());
        self.outcome.diagnosis.lag_frames.push(lag);
        if let Some(render) = absorbed.render {
            if absorbed.ran_slices {
                self.outcome.render_with_slices_us.push(render.get());
            } else {
                self.outcome.render_alone_us.push(render.get());
            }
        }
    }

    /// Closes the run and returns what it measured.
    fn finish(mut self) -> Outcome {
        self.outcome.diagnosis.job_estimate_us.1 = self.job_estimate();
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
        if let Some(timer) = &self.timer {
            self.outcome.timer = timer.stats();
        }
        self.outcome.tally = self.step.finish();
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

/// Each slice's planning estimate now, in order: the job as the projection
/// reads it.
fn planning_estimates(step: &FrameStep) -> Vec<Micros> {
    let cost = step.cost();
    step.slices()
        .iter()
        .map(|&slice| cost.planning_estimate(slice))
        .collect()
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
    step: &mut FrameStep,
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
        step.replace_beat_grid(from, grid)
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
