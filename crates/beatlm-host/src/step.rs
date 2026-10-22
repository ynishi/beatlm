//! A frame's step as both hosts take it, reading no clock: [`FrameStep`]
//! plans a frame, labels its timed passes, records what was submitted, takes
//! the timings that come back, counts the frame's deliveries and misses, and
//! closes the run into a [`Tally`] that the shared report lines
//! ([`crate::report`]) are written from.
//!
//! # A frame, from the host's side
//!
//! The host owns the clock, the device and the timer; [`FrameStep`] owns the
//! scheduler, the cost model and what they decided. One frame is, in order:
//!
//! 1. The host's pacer says which frame it is, when it starts and how long its
//!    period is; a host whose promise is on beat brings the scheduler's beats
//!    up to date ([`FrameStep::replace_beat_grid`]) and says whether an input
//!    is owed.
//! 2. [`FrameStep::plan`] decides the frame: its budget, the forecast, the
//!    input's admission and the plan ([`Planned`]).
//! 3. The host encodes the plan's slices, each compute pass labelled with
//!    [`FrameStep::tag`], then its render pass and the timer's resolve, calls
//!    `queue.submit`, and hands the resolved frame to the timer straight
//!    after ([`SliceTimer::submitted`] has that rule).
//! 4. Whatever the host reads of the plan before it is counted (the
//!    dispatches so far, [`FrameStep::dispatched`]) it reads now; then
//!    [`FrameStep::after_submit`] counts the frame, outside the host's timed
//!    intervals.
//! 5. Each timing the timer has collected goes to [`FrameStep::absorb`], and
//!    the frame is closed by [`FrameStep::complete`].
//!
//! At the end the host drains what its timer still holds through
//! [`FrameStep::absorb`] and takes the [`Tally`] from [`FrameStep::finish`].
//!
//! The order of steps 2, 4 and 5 is held by the types: [`after_submit`]
//! takes the [`Planned`] by value and returns the [`Submitted`] that only
//! [`complete`] takes, so neither can be called twice for a frame or out of
//! order, and [`plan`] refuses a frame while the last one has not been
//! through [`after_submit`]:
//!
//! ```compile_fail
//! # use beatlm_host::step::FrameStep;
//! # use beatlm_core::time::FrameInfo;
//! # fn frame(step: &mut FrameStep, info: FrameInfo) -> anyhow::Result<()> {
//! let planned = step.plan(info, true)?;
//! step.complete(planned); // a `Planned` is not a `Submitted`
//! # Ok(())
//! # }
//! ```
//!
//! ```
//! # use beatlm_host::step::FrameStep;
//! # use beatlm_core::time::FrameInfo;
//! # fn frame(step: &mut FrameStep, info: FrameInfo) -> anyhow::Result<()> {
//! let planned = step.plan(info, true)?;
//! // ...encode with `step.tag(run)`, `queue.submit`, `timer.submitted`...
//! let submitted = step.after_submit(planned);
//! // ...`step.absorb(&timing)` for each timing collected...
//! step.complete(submitted);
//! # Ok(())
//! # }
//! ```
//!
//! [`plan`]: FrameStep::plan
//! [`after_submit`]: FrameStep::after_submit
//! [`complete`]: FrameStep::complete
//! [`SliceTimer::submitted`]: beatlm::gpu::SliceTimer::submitted

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use beatlm::gpu::FrameTiming;
use beatlm_core::budget::Budget;
use beatlm_core::cost::{CostModel, FrameScale, SliceId};
use beatlm_core::schedule::{Due, FramePlan, JobId, ScheduleError, Scheduler, SliceRun};
use beatlm_core::sync::{MissPolicy, SyncPolicy};
use beatlm_core::time::{BeatGrid, FrameIndex, FrameInfo, Micros, TimePoint};

use crate::frame::{Admission, RenderCost, Submit, Tag};
use crate::pass_overlap;
use crate::presets::{RENDER_PRIOR, STARTUP_FRAMES};
use crate::stats::frame_ratio;

/// What a run's step is built with.
#[derive(Debug, Clone, Copy)]
pub struct StepConfig {
    /// The sync promise. A fixed delay in frames
    /// ([`SyncPolicy::source_delay`]) is what a delivery's latency is held to
    /// for [`Tally::violations`]; a promise without one counts none.
    pub policy: SyncPolicy,
    /// What the host shows for a missed input.
    pub miss: MissPolicy,
    /// Slices per job.
    pub slices: u32,
    /// The margin each frame's budget keeps besides the render.
    pub margin: Micros,
    /// The period the scheduler's first expected budget is made at, before
    /// any frame gives its own.
    pub prior_period: Micros,
    /// When a frame's input becomes a job.
    pub submit: Submit,
}

/// A frame as [`FrameStep::plan`] decided it. Only `plan` makes one, so the
/// frame [`FrameStep::after_submit`] counts is the one the step planned.
#[derive(Debug)]
#[must_use = "a planned frame not handed to `FrameStep::after_submit` is never counted, \
              and the step plans no frame after it"]
pub struct Planned {
    frame: FrameInfo,
    plan: FramePlan,
    job: Option<JobId>,
    due: Option<Due>,
    budget: Budget,
    expected: Budget,
}

impl Planned {
    /// The frame.
    #[inline]
    pub const fn frame(&self) -> FrameInfo {
        self.frame
    }

    /// What the frame runs, delivers and misses.
    #[inline]
    pub const fn plan(&self) -> &FramePlan {
        &self.plan
    }

    /// The job the frame's input became, if it became one.
    #[inline]
    pub const fn job(&self) -> Option<JobId> {
        self.job
    }

    /// When that job is due, read before the plan, which can deliver or miss
    /// it in the same frame.
    #[inline]
    pub const fn due(&self) -> Option<Due> {
        self.due
    }

    /// This frame's budget.
    #[inline]
    pub const fn budget(&self) -> Budget {
        self.budget
    }

    /// The budget forecast for the frames to come.
    #[inline]
    pub const fn expected(&self) -> Budget {
        self.expected
    }
}

/// A frame [`FrameStep::after_submit`] has counted, which only
/// [`FrameStep::complete`] closes.
#[derive(Debug)]
#[must_use = "a submitted frame not handed to `FrameStep::complete` has its deliveries \
              and misses uncounted"]
pub struct Submitted(Planned);

impl Submitted {
    /// The frame as it was planned.
    #[inline]
    pub const fn planned(&self) -> &Planned {
        &self.0
    }
}

/// What [`FrameStep::absorb`] read from one returned frame, for a host that
/// keeps lines of its own about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Absorbed {
    /// The frame the timing is of.
    pub frame: FrameIndex,
    /// Its render pass, when that was timed.
    pub render: Option<Micros>,
    /// Whether any of its compute passes was timed.
    pub ran_slices: bool,
}

/// What a run's step counted and sampled: the part of each report both hosts
/// write from the same numbers ([`crate::report::Lines`] names each).
///
/// Per-frame vectors hold one sample per frame through
/// [`FrameStep::after_submit`], unless a field says otherwise.
#[derive(Debug, Clone, Default)]
pub struct Tally {
    /// Inputs submitted as jobs.
    pub submitted: u64,
    /// Results delivered.
    pub delivered: u64,
    /// Deadlines that arrived without a complete result.
    pub missed: u64,
    /// The part of `missed` that belongs to jobs submitted in the run's first
    /// [`STARTUP_FRAMES`] frames, so a cascade of misses while estimates are
    /// new shows apart from misses in steady state.
    pub startup_missed: u64,
    /// Deliveries whose latency was not the promised one; none under a
    /// promise without a fixed delay ([`StepConfig::policy`]).
    pub violations: u64,
    /// Jobs still awaiting their deadline when the run ended.
    pub pending_at_end: usize,
    /// Latency in frames of each delivered result.
    pub latencies: Vec<u64>,
    /// Compute dispatches encoded in each frame: one per slice the plan
    /// handed out.
    pub dispatches_per_frame: Vec<u64>,
    /// Compute dispatches encoded for each delivered job, over every frame
    /// that ran one of its slices.
    pub dispatches_per_job: Vec<u64>,
    /// Render plus every slice measured in the same frame, per timed frame
    /// with a timed render, in microseconds.
    pub gpu_frame_us: Vec<u64>,
    /// The longest single compute pass in each timed frame that ran one, in
    /// microseconds: what one dispatch held up the queue for.
    pub longest_dispatch_us: Vec<u64>,
    /// Per slice: sum and count of measured microseconds.
    pub measured: BTreeMap<SliceId, (u64, u64)>,
    /// Per slice: the cost model's estimate when the run ended.
    pub estimates: BTreeMap<SliceId, Micros>,
    /// Per timed frame whose slices with a frame estimate were planned at
    /// more than zero: their measured time over the sum of those estimates
    /// when the frame was planned ([`CostModel::frame_estimate`]).
    pub frame_ratios: Vec<f64>,
    /// The frame scale of each plan made with a measured one.
    pub frame_scales: Vec<f64>,
    /// The frame scale when the run ended.
    pub frame_scale: FrameScale,
    /// Frames the cost model folded into its frame scale.
    pub frames_recorded: u64,
    /// Every timed frame's passes with their timestamps, for the
    /// `pass_overlap_*` lines ([`pass_overlap`]).
    pub pass_frames: Vec<pass_overlap::Frame>,
}

/// One run's scheduler, cost model and render measurements, and what they
/// counted: the frame's step without its clock, its device or its timer.
///
/// [The module documentation](self) gives the order a host calls it in.
/// It holds no wgpu or browser type, so a host may move it to another thread.
#[derive(Debug)]
pub struct FrameStep {
    scheduler: Scheduler,
    cost: CostModel,
    /// The render's measurements, which set the budgets.
    render: RenderCost,
    slices: Vec<SliceId>,
    margin: Micros,
    /// The latency deliveries are held to, when the promise has one.
    promised: Option<u32>,
    admission: Admission,
    /// Dispatches encoded so far for each job not yet delivered or missed,
    /// by submission index.
    job_dispatches: BTreeMap<u64, u64>,
    /// The frame planned and not yet through `after_submit`.
    unsubmitted: Option<FrameIndex>,
    tally: Tally,
}

impl FrameStep {
    /// A run under `config` whose slices `cost` estimates.
    #[must_use]
    pub fn new(config: StepConfig, cost: CostModel) -> Self {
        let scheduler = Scheduler::new(
            config.policy,
            config.miss,
            Budget::for_frame(config.prior_period, RENDER_PRIOR, config.margin),
        );
        Self {
            scheduler,
            cost,
            render: RenderCost::default(),
            slices: (0..config.slices).map(SliceId::new).collect(),
            margin: config.margin,
            promised: config.policy.source_delay(),
            admission: Admission::new(config.submit),
            job_dispatches: BTreeMap::new(),
            unsubmitted: None,
            tally: Tally::default(),
        }
    }

    /// Puts `grid` in effect from `from` on, for a host that brings its beats
    /// up to date before [`plan`](Self::plan): the scheduler's
    /// [`Scheduler::replace_beat_grid`], which says what it changes.
    ///
    /// # Errors
    ///
    /// As [`Scheduler::replace_beat_grid`]: the promise is not on beat.
    #[inline]
    pub fn replace_beat_grid(
        &mut self,
        from: TimePoint,
        grid: BeatGrid,
    ) -> Result<(), ScheduleError> {
        self.scheduler.replace_beat_grid(from, grid)
    }

    /// The cost model, as the next plan reads it.
    #[inline]
    #[must_use]
    pub const fn cost(&self) -> &CostModel {
        &self.cost
    }

    /// A job's slices, in order.
    #[inline]
    #[must_use]
    pub fn slices(&self) -> &[SliceId] {
        &self.slices
    }

    /// Inputs submitted as jobs so far. Jobs are numbered in submission order
    /// from zero, so this is the number the next one gets.
    #[inline]
    #[must_use]
    pub const fn submitted(&self) -> u64 {
        self.tally.submitted
    }

    /// Dispatches encoded for `job` in the frames counted so far; zero for a
    /// job none of whose slices has run, or that has been delivered or
    /// missed. A frame's own dispatches count from its
    /// [`after_submit`](Self::after_submit), so a host reading what came
    /// before the frame reads it before that.
    #[inline]
    #[must_use]
    pub fn dispatched(&self, job: JobId) -> u64 {
        self.job_dispatches.get(&job.get()).copied().unwrap_or(0)
    }

    /// Decides `frame`: its budget from the render measured last, the
    /// forecast of the frames to come (both at `frame.period`, as
    /// [`RenderCost`] says), the input's admission when one is `owed`, and
    /// the plan.
    ///
    /// An error between a frame's `plan` and its
    /// [`after_submit`](Self::after_submit) ends the step: nothing takes the
    /// planned frame back, so every later `plan` is refused, naming that
    /// frame.
    ///
    /// # Errors
    ///
    /// The frame planned before has not been through
    /// [`after_submit`](Self::after_submit), or the scheduler rejects the
    /// frame's job.
    pub fn plan(&mut self, frame: FrameInfo, owed: bool) -> Result<Planned> {
        if let Some(unsubmitted) = self.unsubmitted {
            bail!(
                "frame {} was planned and never counted as submitted, so frame {} cannot be \
                 planned: an error between `plan` and `after_submit` ends the step",
                unsubmitted.get(),
                frame.index.get()
            );
        }
        let budget = self.render.budget(frame.period, self.margin);
        let expected = self.render.expected(frame.period, self.margin);
        self.scheduler.set_expected_budget(expected);
        let job = if owed && self.admission.takes_input() {
            let job = self
                .scheduler
                .submit(frame, self.slices.clone(), &self.cost)
                .context("the scheduler rejected the frame's job")?;
            Some(job)
        } else {
            None
        };
        let due = job.and_then(|job| self.scheduler.due(job));
        let plan = self.scheduler.plan_frame(frame, budget, &self.cost);
        self.unsubmitted = Some(frame.index);
        Ok(Planned {
            frame,
            plan,
            job,
            due,
            budget,
            expected,
        })
    }

    /// The label of `run`'s compute pass: its job, its slice and the estimate
    /// the frame's ratio takes it at ([`Tag`]). Read before
    /// [`absorb`](Self::absorb) records into the cost model, as it is while
    /// the planned frame is encoded.
    #[inline]
    #[must_use]
    pub fn tag(&self, run: &SliceRun) -> Tag {
        (run.job, run.slice, self.cost.frame_estimate(run.slice))
    }

    /// Counts the frame `planned`: its input, its dispatches, its frame scale,
    /// and what the admission of the next input reads. A host calls it after
    /// the frame's `queue.submit` and the timer's `submitted` that follows it,
    /// outside its timed intervals, and after anything it reads of the frames
    /// before ([`dispatched`](Self::dispatched)).
    pub fn after_submit(&mut self, planned: Planned) -> Submitted {
        // Only `plan` makes a `Planned`, and it plans no frame while one is
        // pending; a `Planned` from another step is what this catches.
        debug_assert_eq!(
            self.unsubmitted,
            Some(planned.frame.index),
            "the frame counted is not the frame this step planned"
        );
        let plan = &planned.plan;
        self.tally.dispatches_per_frame.push(plan.runs.len() as u64);
        if let FrameScale::Measured(scale) = self.cost.frame_scale() {
            self.tally.frame_scales.push(scale);
        }
        if planned.job.is_some() {
            self.tally.submitted += 1;
        }
        self.admission
            .planned(planned.job.map(|job| (job, self.slices.len())), plan);
        // Every slice is one compute pass with one dispatch.
        for run in &plan.runs {
            *self.job_dispatches.entry(run.job.get()).or_default() += 1;
        }
        self.unsubmitted = None;
        Submitted(planned)
    }

    /// Feeds one returned frame's measurements to the cost model and the
    /// tally: each slice's, then the frame's ratio over the timed slices that
    /// had a frame estimate when it was planned, then the render, which sets
    /// the budgets of the frames planned from now on.
    pub fn absorb(&mut self, timing: &FrameTiming<Tag>) -> Absorbed {
        self.tally
            .pass_frames
            .push(pass_overlap::Frame::of(timing, |&(_, slice, _)| slice));
        let mut gpu = Micros::ZERO;
        let (mut predicted, mut in_ratio) = (Micros::ZERO, Micros::ZERO);
        for ((_, slice, planned), duration) in &timing.compute {
            self.cost.record(*slice, *duration);
            let entry = self.tally.measured.entry(*slice).or_insert((0, 0));
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
                self.tally.frame_ratios.push(ratio);
            }
        }
        if let Some(longest) = timing.compute.iter().map(|(_, d)| d.get()).max() {
            self.tally.longest_dispatch_us.push(longest);
        }
        if let Some(render) = timing.render {
            self.render.record(render);
            self.tally
                .gpu_frame_us
                .push(gpu.saturating_add(render).get());
        }
        Absorbed {
            frame: timing.frame,
            render: timing.render,
            ran_slices: !timing.compute.is_empty(),
        }
    }

    /// Counts the frame's deliveries, with their latency and dispatches, and
    /// its misses, and returns its plan for a host that records more of it.
    pub fn complete(&mut self, submitted: Submitted) -> FramePlan {
        let Planned { frame, plan, .. } = submitted.0;
        for delivery in &plan.delivered {
            let latency = frame.index.get().saturating_sub(delivery.input.get());
            self.tally.latencies.push(latency);
            self.tally.delivered += 1;
            if self
                .promised
                .is_some_and(|promised| latency != u64::from(promised))
            {
                self.tally.violations += 1;
            }
            let dispatches = self.job_dispatches.remove(&delivery.job.get()).unwrap_or(0);
            self.tally.dispatches_per_job.push(dispatches);
        }
        for missed in &plan.missed {
            self.job_dispatches.remove(&missed.job.get());
            if missed.input.get() < STARTUP_FRAMES {
                self.tally.startup_missed += 1;
            }
        }
        self.tally.missed += plan.missed.len() as u64;
        plan
    }

    /// Closes the run: the jobs still pending, and each slice's estimate and
    /// the frame scale as the run left them.
    #[must_use]
    pub fn finish(mut self) -> Tally {
        self.tally.pending_at_end = self.scheduler.pending();
        for &slice in &self.slices {
            self.tally
                .estimates
                .insert(slice, self.cost.estimate(slice));
        }
        self.tally.frame_scale = self.cost.frame_scale();
        self.tally.frames_recorded = self.cost.frames_recorded();
        self.tally
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use beatlm_core::cost::Estimator;
    use beatlm_core::time::TimePoint;

    use crate::presets::SLICE_PRIOR;

    /// Frame `index` of a grid at `period`.
    fn frame_at(index: u64, period: Micros) -> FrameInfo {
        FrameInfo {
            index: FrameIndex::new(index),
            start: TimePoint::new(index * period.get()),
            period,
        }
    }

    /// The timing of frame `frame` whose slices each measured `slice_us` and
    /// whose render measured `render_us`, without timestamps.
    fn timing(
        step: &FrameStep,
        frame: &Planned,
        slice_us: u64,
        render_us: u64,
    ) -> FrameTiming<Tag> {
        FrameTiming {
            frame: frame.frame.index,
            compute: frame
                .plan
                .runs
                .iter()
                .map(|run| (step.tag(run), Micros::new(slice_us)))
                .collect(),
            render: Some(Micros::new(render_us)),
            compute_spans: Vec::new(),
            render_span: None,
        }
    }

    #[test]
    fn the_step_holds_no_type_that_keeps_it_on_one_thread() {
        const fn send<T: Send>() {}
        send::<FrameStep>();
        send::<Submitted>();
    }

    /// Seventy frames of one-slice jobs at latency 1 whose frames each return
    /// before the next is planned, as a native poll returns them. Each job
    /// runs in its own frame and is delivered in the next, except where two
    /// renders of 20 ms (frames 2 and 3, then 65 and 66) leave the frames
    /// after them no budget: frames 3 and 4 run nothing, so job 3 reaches its
    /// deadline, frame 4, unrun and is missed, and job 4 runs and is
    /// delivered in frame 5 with job 5 beside it; and the same at 66. Job 3 is
    /// of the run's first [`STARTUP_FRAMES`] frames and job 66 is not. Job
    /// 69's deadline is past the run.
    #[test]
    fn the_tally_counts_what_a_scripted_run_planned_delivered_and_missed() {
        const PERIOD: Micros = Micros::new(16_667);
        assert_eq!(STARTUP_FRAMES, 64);
        let cost = CostModel::new(SLICE_PRIOR, Estimator::default()).unwrap();
        let mut step = FrameStep::new(
            StepConfig {
                policy: SyncPolicy::FixedLatency { frames: 1 },
                miss: MissPolicy::HoldPrevious,
                slices: 1,
                margin: Micros::new(2_000),
                prior_period: PERIOD,
                submit: Submit::Every,
            },
            cost,
        );
        let slow = [2, 3, 65, 66];
        for index in 0..70 {
            let planned = step.plan(frame_at(index, PERIOD), true).unwrap();
            let render = if slow.contains(&index) { 20_000 } else { 1_000 };
            let returned = timing(&step, &planned, 1_500, render);
            let submitted = step.after_submit(planned);
            let absorbed = step.absorb(&returned);
            assert_eq!(absorbed.frame, FrameIndex::new(index));
            assert_eq!(absorbed.render, Some(Micros::new(render)));
            step.complete(submitted);
        }
        let tally = step.finish();
        assert_eq!(tally.submitted, 70);
        assert_eq!(tally.delivered, 67);
        assert_eq!(tally.missed, 2);
        assert_eq!(tally.startup_missed, 1);
        assert_eq!(tally.violations, 0);
        assert_eq!(tally.pending_at_end, 1);
        assert_eq!(tally.latencies, vec![1; 67]);
        assert_eq!(tally.dispatches_per_job, vec![1; 67]);
        let mut per_frame = vec![1; 70];
        for (frame, dispatches) in [(3, 0), (4, 0), (5, 2), (66, 0), (67, 0), (68, 2)] {
            per_frame[frame] = dispatches;
        }
        assert_eq!(tally.dispatches_per_frame, per_frame);
        // Every frame's render was timed; 66 of them ran a slice.
        assert_eq!(tally.gpu_frame_us.len(), 70);
        assert_eq!(tally.gpu_frame_us[0], 2_500);
        assert_eq!(tally.gpu_frame_us[3], 20_000);
        assert_eq!(tally.gpu_frame_us[5], 4_000);
        assert_eq!(tally.longest_dispatch_us, vec![1_500; 66]);
        assert_eq!(tally.measured, [(SliceId::new(0), (68 * 1_500, 68))].into());
        assert_eq!(tally.frame_ratios.len(), 66);
        assert_eq!(tally.pass_frames.len(), 70);
        assert_eq!(tally.estimates.len(), 1);
    }

    #[test]
    fn a_frame_planned_before_the_last_was_counted_is_refused() {
        let cost = CostModel::new(SLICE_PRIOR, Estimator::default()).unwrap();
        let mut step = FrameStep::new(
            StepConfig {
                policy: SyncPolicy::FixedLatency { frames: 2 },
                miss: MissPolicy::HoldPrevious,
                slices: 2,
                margin: Micros::new(2_000),
                prior_period: Micros::new(16_667),
                submit: Submit::Every,
            },
            cost,
        );
        let period = Micros::new(16_667);
        let first = step.plan(frame_at(0, period), true).unwrap();
        let job = first.job.unwrap();
        // Before the frame is counted its own slices are not: what a host
        // reads of the frames before it.
        assert_eq!(first.plan.runs.len(), 2);
        assert_eq!(step.dispatched(job), 0);
        let refused = step.plan(frame_at(1, period), true).unwrap_err();
        assert!(
            refused
                .to_string()
                .starts_with("frame 0 was planned and never counted"),
            "{refused}"
        );
        let submitted = step.after_submit(first);
        // From the count on, they are.
        assert_eq!(step.dispatched(job), 2);
        assert_eq!(step.submitted(), 1);
        step.complete(submitted);
        assert!(step.plan(frame_at(1, period), true).is_ok());
    }

    /// An estimator that reads a slice at its last measurement, or the prior.
    const LAST: Estimator = Estimator::MeanPlusSigmas {
        alpha: 1.0,
        sigmas: 0.0,
    };

    /// The timing of frame `index` whose render measured `render_us` and whose
    /// slices were not timed, so their estimates stay the prior.
    fn render_only(index: u64, render_us: u64) -> FrameTiming<Tag> {
        FrameTiming {
            frame: FrameIndex::new(index),
            compute: Vec::new(),
            render: Some(Micros::new(render_us)),
            compute_spans: Vec::new(),
            render_span: None,
        }
    }

    /// Twenty-one frames of 10 ms on beats every 50 ms, put in effect through
    /// the step, with an input owed on each beat: frames 0, 5, 10, 15 and 20.
    /// A job is three slices estimated at 2 000 µs, and a frame's budget is its
    /// 10 000 µs less a render of 2 000 µs and a margin of 3 000 µs: two
    /// slices. A job runs two slices in its input's frame and one in the next,
    /// ending 12 000 µs after its input's start, and with the beat margin of
    /// 1 000 µs is due on the next beat, 50 000 µs after its input: it is
    /// delivered at latency 5 by the frame starting on that beat. Renders of
    /// 20 ms in frames 9 to 14 leave frames 10 to 15 no budget while the
    /// forecast, their median, stays at 2 000 µs: job 2 (input frame 10, due
    /// 150 000 µs) runs no slice and is missed by frame 15, which is of the
    /// run's first [`STARTUP_FRAMES`]. Job 3 (input frame 15) runs in frames 16
    /// and 17 and is delivered by frame 20; job 4 (input frame 20) is pending
    /// when the run ends.
    #[test]
    fn the_tally_counts_what_a_scripted_on_beat_run_delivered_and_missed() {
        const PERIOD: Micros = Micros::new(10_000);
        const BEAT: u64 = 50_000;
        let cost = CostModel::new(Micros::new(2_000), LAST).unwrap();
        let mut step = FrameStep::new(
            StepConfig {
                policy: SyncPolicy::OnBeat {
                    margin: Micros::new(1_000),
                    // No beat of the run is on it: the grid below replaces it.
                    grid: BeatGrid::new(TimePoint::new(0), Micros::new(1_000_000)).unwrap(),
                },
                miss: MissPolicy::HoldPrevious,
                slices: 3,
                margin: Micros::new(3_000),
                prior_period: PERIOD,
                submit: Submit::Every,
            },
            cost,
        );
        let beats = BeatGrid::new(TimePoint::new(0), Micros::new(BEAT)).unwrap();
        step.replace_beat_grid(TimePoint::new(0), beats).unwrap();
        let mut due = BTreeMap::new();
        let (mut delivered, mut missed) = (Vec::new(), Vec::new());
        for index in 0..21 {
            let frame = frame_at(index, PERIOD);
            let planned = step.plan(frame, index % 5 == 0).unwrap();
            let budget = if (10..=15).contains(&index) { 0 } else { 5_000 };
            let budgets = (planned.budget().get().get(), planned.expected().get().get());
            assert_eq!(budgets, (budget, 5_000), "{index}");
            if let Some(job) = planned.job() {
                let beat = TimePoint::new(frame.start.get() + BEAT);
                assert_eq!(planned.due(), Some(Due::Beat(beat)), "{index}");
                due.insert(job.get(), Due::Beat(beat));
            }
            let slow = (9..=14).contains(&index);
            let render = if slow { 20_000 } else { 2_000 };
            let submitted = step.after_submit(planned);
            step.absorb(&render_only(index, render));
            let plan = step.complete(submitted);
            for delivery in &plan.delivered {
                // By the frame that starts on the beat read at submission.
                assert_eq!(due[&delivery.job.get()], Due::Beat(frame.start));
                delivered.push((delivery.job.get(), index));
            }
            for miss in &plan.missed {
                assert_eq!(due[&miss.job.get()], miss.due);
                missed.push((miss.job.get(), index));
            }
        }
        assert_eq!(delivered, vec![(0, 5), (1, 10), (3, 20)]);
        assert_eq!(missed, vec![(2, 15)]);
        let tally = step.finish();
        assert_eq!(tally.submitted, 5);
        assert_eq!(tally.delivered, 3);
        assert_eq!(tally.missed, 1);
        assert_eq!(tally.startup_missed, 1);
        // On beat the promise has no fixed delay to hold a latency to.
        assert_eq!(tally.violations, 0);
        assert_eq!(tally.pending_at_end, 1);
        assert_eq!(tally.latencies, vec![5; 3]);
        assert_eq!(tally.dispatches_per_job, vec![3; 3]);
        let mut per_frame = vec![0; 21];
        for (frame, dispatches) in [(0, 2), (1, 1), (5, 2), (6, 1), (16, 2), (17, 1), (20, 2)] {
            per_frame[frame] = dispatches;
        }
        assert_eq!(tally.dispatches_per_frame, per_frame);
    }

    /// A job of no estimated work, owed with no beat margin in a frame that
    /// starts on a beat, is due on that beat and delivered by the same frame's
    /// plan, so it has left the scheduler once the plan is made: its beat is
    /// there to read because [`FrameStep::plan`] reads it between the
    /// submission and the plan.
    #[test]
    fn a_job_its_own_frame_delivers_keeps_the_beat_read_before_the_plan() {
        const PERIOD: Micros = Micros::new(10_000);
        let cost = CostModel::new(Micros::ZERO, LAST).unwrap();
        let mut step = FrameStep::new(
            StepConfig {
                policy: SyncPolicy::OnBeat {
                    margin: Micros::ZERO,
                    grid: BeatGrid::new(TimePoint::new(0), Micros::new(50_000)).unwrap(),
                },
                miss: MissPolicy::HoldPrevious,
                slices: 1,
                margin: Micros::new(3_000),
                prior_period: PERIOD,
                submit: Submit::Every,
            },
            cost,
        );
        let planned = step.plan(frame_at(5, PERIOD), true).unwrap();
        let job = planned.job().unwrap();
        assert_eq!(planned.due(), Some(Due::Beat(TimePoint::new(50_000))));
        assert_eq!(planned.plan().delivered.len(), 1);
        assert_eq!(planned.plan().delivered[0].job, job);
        assert_eq!(step.scheduler.due(job), None);
        let submitted = step.after_submit(planned);
        step.complete(submitted);
        let tally = step.finish();
        assert_eq!((tally.delivered, tally.latencies), (1, vec![0]));
    }

    /// What a simulated run delivered and missed.
    #[derive(Debug)]
    struct Delivery {
        delivered: u64,
        missed: u64,
    }

    /// The depth model's loop on the step alone, at the period, margin and
    /// latency of the Chrome run that collapsed (8 300 µs, 2 ms, 25 frames),
    /// with the [`Submit::Free`] admission, a job of 160 slices of about
    /// 29 ms, estimated at what they take, and a render pass of 1 900 µs whose
    /// measurement also counts the frame's slices when `waits` is set. Each
    /// frame's render reaches the budget the next frame, as readbacks arrive
    /// between callbacks; its slices' times are not fed back, so the
    /// estimates stay what they were seeded at.
    fn simulate(waits: bool, frames: u64) -> Delivery {
        let (period, margin, render_us) = (Micros::new(8_300), Micros::new(2_000), 1_900);
        let costs: Vec<u64> = (0..160).map(|i| [60, 120, 180, 240, 300][i % 5]).collect();
        let mut cost = CostModel::new(Micros::new(2_000), Estimator::default()).unwrap();
        for _ in 0..32 {
            for (slice, &us) in costs.iter().enumerate() {
                cost.record(SliceId::new(slice as u32), Micros::new(us));
            }
        }
        let mut step = FrameStep::new(
            StepConfig {
                policy: SyncPolicy::FixedLatency { frames: 25 },
                miss: MissPolicy::HoldPrevious,
                slices: 160,
                margin,
                prior_period: period,
                submit: Submit::Free,
            },
            cost,
        );
        for index in 0..frames {
            let planned = step.plan(frame_at(index, period), true).unwrap();
            let ran: u64 = planned
                .plan
                .runs
                .iter()
                .map(|run| costs[run.slice.get() as usize])
                .sum();
            let returned = FrameTiming {
                frame: planned.frame.index,
                compute: Vec::new(),
                render: Some(Micros::new(render_us + if waits { ran } else { 0 })),
                compute_spans: Vec::new(),
                render_span: None,
            };
            let submitted = step.after_submit(planned);
            step.absorb(&returned);
            step.complete(submitted);
        }
        let tally = step.finish();
        Delivery {
            delivered: tally.delivered,
            missed: tally.missed,
        }
    }

    #[test]
    fn a_render_that_waits_for_the_slices_slows_the_jobs() {
        // A job of 29 ms over budgets of 4.4 ms takes 7 frames, so a 25-frame
        // promise is met by every job when the render is measured alone...
        let alone = simulate(false, 3_000);
        assert_eq!((alone.delivered, alone.missed), (425, 0), "{alone:?}");
        // ...and when its measurement carries the slices the frames alternate
        // between full and empty, and the forecast median, a render that fills
        // the period, leaves no expected frame room for a slice. The job, with
        // none behind it, is kept and runs in the full frames: every promise
        // is still met, at about half the rate. Before #41 it was abandoned
        // and more than three jobs were missed for each delivered.
        let waits = simulate(true, 3_000);
        assert_eq!((waits.delivered, waits.missed), (218, 0), "{waits:?}");
    }

    /// The step alone, for `frames` frames of a 10 ms budget, expected and
    /// real, with jobs of `slices` slices each estimated at 1 ms, under
    /// latency `k` and `submit`: how many inputs were submitted, delivered and
    /// missed. No timing returns, so both budgets stay the period of the
    /// bench less the render's prior and a margin that leaves 10 ms.
    fn schedule_alone(submit: Submit, slices: u32, k: u32, frames: u64) -> (u64, u64, u64) {
        const PERIOD: Micros = Micros::new(16_667);
        let margin = Micros::new(PERIOD.get() - RENDER_PRIOR.get() - 10_000);
        let cost = CostModel::new(
            Micros::new(1_000),
            Estimator::MeanPlusSigmas {
                alpha: 1.0,
                sigmas: 0.0,
            },
        )
        .unwrap();
        let mut step = FrameStep::new(
            StepConfig {
                policy: SyncPolicy::FixedLatency { frames: k },
                miss: MissPolicy::HoldPrevious,
                slices,
                margin,
                prior_period: PERIOD,
                submit,
            },
            cost,
        );
        for i in 0..frames {
            let planned = step.plan(frame_at(i, PERIOD), true).unwrap();
            assert_eq!(planned.budget, Budget::new(Micros::new(10_000)));
            assert_eq!(planned.expected, planned.budget);
            let submitted = step.after_submit(planned);
            step.complete(submitted);
        }
        let tally = step.finish();
        (tally.submitted, tally.delivered, tally.missed)
    }

    #[test]
    fn a_job_longer_than_a_frame_misses_most_inputs_when_every_frame_submits_and_none_when_free() {
        // 45 ms of work at 10 ms a frame is five frames, and latency 5 gives
        // a job six. Every frame's input: a job whose projection behind the
        // unfinished work ahead does not fit its six frames is abandoned once
        // the next frame's input queues behind it, and the inputs that fit are
        // delivered; the last input, with nothing behind it, is still pending
        // when the run ends. Before jobs were abandoned each kept job held the
        // budget to its deadline, the one behind started too late, and this
        // delivered 1 and missed 114. Until #41 a job was abandoned in the
        // frame it arrived, the last one too, and this missed 93.
        assert_eq!(schedule_alone(Submit::Every, 45, 5, 120), (120, 26, 92));
        // Only once the last job's slices are all handed out: one input every
        // five frames, and every deadline that arrived is met.
        assert_eq!(schedule_alone(Submit::Free, 45, 5, 120), (24, 23, 0));
        // At latency 3 a job gets four frames and needs five. With no job
        // behind it, it is kept: it runs four frames' slices, so they are
        // measured, and is missed at its deadline, and the frame after takes
        // the next input: one in four is submitted, and every one is missed.
        // Until #41 each was missed in the frame it was submitted, without
        // running a slice, and every frame took an input: 120 submitted, 120
        // missed.
        assert_eq!(schedule_alone(Submit::Free, 45, 3, 120), (30, 0, 30));
    }
}
