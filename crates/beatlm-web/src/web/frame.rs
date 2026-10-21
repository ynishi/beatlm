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
//! `engine.rs`) for the synthetic workload: the same budget and forecast, the
//! same scheduler and cost model, one timed compute pass per slice handed out
//! and one timed render pass; the crate documentation says what differs and
//! why.

use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::rc::Rc;

use anyhow::{Context, Result, anyhow, bail};
use beatlm::gpu::{FrameTiming, MAX_SLICES, SliceTimer};
use beatlm_core::budget::Budget;
use beatlm_core::cost::{CostModel, Estimator, SliceId};
use beatlm_core::schedule::{FramePlan, JobId, Scheduler};
use beatlm_core::sync::{MissPolicy, SyncPolicy};
use beatlm_core::time::{FrameIndex, FrameInfo, Micros, TimePoint};
use wasm_bindgen::JsCast;
use wasm_bindgen::closure::Closure;

use super::device::{self, Gpu};
use super::page;
use crate::config::{Config, MARGIN_US, SLICES};
use crate::pace::{FrameStart, NOMINAL_PERIOD, Pace, Pacer, ms_to_us};
use crate::report::{Environment, Outcome, STARTUP_FRAMES, render};
use crate::workload::Workload;

/// What a slice is assumed to cost until it has been measured: the bench's.
const SLICE_PRIOR: Micros = Micros::new(2_000);
/// What the render pass is assumed to cost until it has been measured: the
/// bench's.
const RENDER_PRIOR: Micros = Micros::new(2_000);
/// How many recent render measurements the budget forecast takes its median
/// over: the bench's, the window of the cost model's default estimate.
const RENDER_WINDOW: usize = 32;
/// How long after the last frame the host waits for timings still in flight
/// before it reports without them, in microseconds: a second, whatever the
/// display's rate. The bench blocks on the device instead, which a page
/// cannot.
const DRAIN_US: u64 = 1_000_000;

/// The label a timed slice carries from the scheduler to the cost model.
type Tag = (JobId, SliceId);

/// Whether the loop asks for another callback.
enum Turn {
    Continue,
    Finished,
}

/// The run: the device, the load, the decisions and what was measured.
pub struct Host {
    gpu: Gpu,
    workload: Workload,
    /// `None` when the device has no timestamp queries.
    timer: Option<SliceTimer<Tag>>,
    scheduler: Scheduler,
    cost: CostModel,
    render_estimate: Micros,
    /// The most recent render measurements, at most [`RENDER_WINDOW`], whose
    /// median forecasts the frames to come.
    render_recent: VecDeque<Micros>,
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
}

impl Host {
    /// Builds the run's timer, scheduler and cost model for `config` on `gpu`,
    /// which `workload` was built on.
    ///
    /// # Errors
    ///
    /// A device with timestamp queries and a latency whose jobs in flight
    /// have more slices than the timer can time, a timer or cost model that
    /// cannot be built, or a page without `performance`.
    pub fn new(gpu: Gpu, workload: Workload, config: &Config) -> Result<Self> {
        let timer = if gpu.timestamp_query {
            // Every frame submits a job, and a job lives `latency` frames past
            // its submission, so `latency + 1` jobs can have slices left in a
            // frame.
            let capacity = config
                .latency
                .checked_add(1)
                .and_then(|jobs| jobs.checked_mul(SLICES))
                .filter(|&capacity| capacity <= MAX_SLICES)
                .with_context(|| {
                    format!(
                        "{SLICES} slices for each of the latency + 1 jobs in flight at latency \
                         {} are more than the GPU timer can time ({MAX_SLICES})",
                        config.latency
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
        let scheduler = Scheduler::new(
            SyncPolicy::FixedLatency {
                frames: config.latency,
            },
            MissPolicy::HoldPrevious,
            // Before the display's period is known, the nominal one; every
            // frame sets its own budget from the period measured by then.
            Budget::for_frame(NOMINAL_PERIOD, RENDER_PRIOR, margin),
        );
        let cost = CostModel::new(SLICE_PRIOR, Estimator::default())
            .context("cannot build the cost model")?;
        let performance = page::window()?
            .performance()
            .context("the window has no performance clock")?;
        Ok(Self {
            gpu,
            workload,
            timer,
            scheduler,
            cost,
            render_estimate: RENDER_PRIOR,
            render_recent: VecDeque::with_capacity(RENDER_WINDOW),
            slices: (0..SLICES).map(SliceId::new).collect(),
            margin,
            latency: config.latency,
            job_dispatches: BTreeMap::new(),
            pacer: Pacer::new(config.frames),
            performance,
            drain_from_us: None,
            outcome: Outcome::default(),
        })
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

        let planning = self.performance.now();
        // This frame's budget follows the last render measured and the
        // display's period; the frames to come are forecast from the median of
        // the recent renders, as the bench does and for its reason
        // (`Scheduler::set_expected_budget`), at the same period.
        let budget = Budget::for_frame(start.period, self.render_estimate, self.margin);
        let forecast = Budget::for_frame(start.period, self.render_forecast(), self.margin);
        self.scheduler.set_expected_budget(forecast);
        self.scheduler
            .submit(frame, self.slices.clone(), &self.cost)
            .context("the scheduler rejected the frame's job")?;
        let plan = self.scheduler.plan_frame(frame, budget, &self.cost);

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
        match &mut self.timer {
            Some(timer) => {
                let mut timestamps = timer.begin_frame(frame.index);
                for run in &plan.runs {
                    let writes = timestamps.compute_pass((run.job, run.slice))?;
                    self.workload.encode_slice(&mut encoder, writes);
                }
                if let Some(view) = &view {
                    self.workload
                        .encode_render(&mut encoder, view, timestamps.render_pass()?);
                }
                timestamps.resolve(&mut encoder);
            }
            None => {
                for _ in &plan.runs {
                    self.workload.encode_slice(&mut encoder, None);
                }
                if let Some(view) = &view {
                    self.workload.encode_render(&mut encoder, view, None);
                }
            }
        }
        let commands = encoder.finish();

        let submitting = self.performance.now();
        self.gpu.queue.submit([commands]);
        let submitted = self.performance.now();
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
        self.outcome.submitted += 1;
        for run in &plan.runs {
            *self.job_dispatches.entry(run.job.get()).or_default() += 1;
        }
        self.outcome.cpu_us.push(between(began, submitted));

        self.read_back();
        self.complete(start.index, &plan);
        self.outcome
            .tick_us
            .push(between(began, self.performance.now()));
        Ok(())
    }

    /// Requests the maps of the frames just submitted and feeds every frame
    /// whose map has completed to the cost model and the report.
    ///
    /// `Device::poll` is not called: in a browser it does nothing, and the map
    /// callbacks have run from the event loop before this callback began.
    fn read_back(&mut self) {
        let ready = match &mut self.timer {
            Some(timer) => {
                timer.start_readback();
                timer.collect_ready()
            }
            None => return,
        };
        for timing in ready {
            self.absorb(timing);
        }
    }

    /// The render time the frames to come are forecast with: the median of the
    /// recent measurements (the upper one of an even count), or the prior
    /// before any has arrived.
    fn render_forecast(&self) -> Micros {
        let mut recent: Vec<Micros> = self.render_recent.iter().copied().collect();
        recent.sort_unstable();
        recent
            .get(recent.len() / 2)
            .copied()
            .unwrap_or(RENDER_PRIOR)
    }

    /// Delivers the plan's results and counts its misses.
    fn complete(&mut self, index: u64, plan: &FramePlan) {
        for delivery in &plan.delivered {
            let latency = index.saturating_sub(delivery.input.get());
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
        if let Some(longest) = timing.compute.iter().map(|(_, d)| d.get()).max() {
            self.outcome.longest_dispatch_us.push(longest);
        }
        if let Some(render) = timing.render {
            self.render_estimate = render;
            if self.render_recent.len() == RENDER_WINDOW {
                self.render_recent.pop_front();
            }
            self.render_recent.push_back(render);
            self.outcome
                .gpu_frame_us
                .push(gpu.saturating_add(render).get());
        }
    }

    /// Closes the run and returns what it measured.
    fn finish(mut self) -> Outcome {
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
        self.outcome.gpu_errors = device::gpu_errors();
        self.outcome
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
