//! The pieces of a frame's step that read no clock, which both hosts take
//! each frame or at setup: the label a timed slice carries ([`Tag`]), the
//! render pass's budget and forecast ([`RenderCost`]), whether a frame's input
//! becomes a job ([`Admission`]), and how many slices the loop's timer must
//! hold ([`timer_capacity`], [`jobs_with_slices`]).
//!
//! The frame's period comes in as an argument: the bench paces a fixed grid,
//! a page follows the display's measured period, and the rules here are the
//! same at either.

use std::collections::VecDeque;

use beatlm::gpu::MAX_SLICES;
use beatlm_core::budget::Budget;
use beatlm_core::cost::SliceId;
use beatlm_core::schedule::{FramePlan, JobId};
use beatlm_core::time::Micros;

use crate::presets::{RENDER_PRIOR, RENDER_WINDOW};

/// The label a timed slice carries from the scheduler to the cost model: its
/// job, the slice, and what a frame's ratio takes the slice at when the frame
/// was planned (`CostModel::frame_estimate`, `None` for a slice left out of
/// it). The timer hands the label back with the measurement, so a frame's
/// measured slice time is compared with what those same slices were planned
/// at (`CostModel::record_frame`), however many frames later the timing
/// arrives and whichever of its passes were timed.
pub type Tag = (JobId, SliceId, Option<Micros>);

/// What the render pass is budgeted at: the frame's own budget from the
/// render measured last, and the forecast of the frames to come from the
/// median of the recent ones.
///
/// This frame's budget follows the last render measured, as the host's order
/// of a frame (the `beatlm` crate documentation) says. The frames to come are
/// forecast from the median of the recent renders instead: one slow render
/// would otherwise lower the forecast for every later frame, and the
/// scheduler would abandon jobs those frames finish
/// (`Scheduler::set_expected_budget` says why the forecast matters).
///
/// # A render measurement that carries the slices
///
/// The rule assumes the render pass's measured time is the render's own. If a
/// render pass waits for the frame's slices and its timestamps count the wait,
/// a frame that ran `s` of slices measures a render of about `render + s`, and
/// the rule feeds that back twice: the next frame's budget shrinks by `s`, so
/// slice-free frames alternate with full ones; and once more than half of the
/// recent frames ran slices, the median forecast is a render that fills the
/// period and no slice fits an expected frame. Before #41 the scheduler then
/// abandoned the job it was running, and jobs starved: more than three missed
/// for each delivered. Since, a job with no job behind it is kept, as every
/// job of the [`Submit::Free`] admission is, and runs in the frames whose own
/// budget has room, so the alternation halves what is delivered instead. The
/// test `a_render_that_waits_for_the_slices_slows_the_jobs` shows it on the
/// scheduler alone. It was suspected of the depth model's
/// collapse in Chrome and measured not to be it: with the display reading no
/// buffer a slice writes, the render measured p50 1 245 µs in frames with
/// slices and 1 901 µs without (Chrome 154, WebGPU, Apple M1 Pro, 120 Hz,
/// 2026-10-22, Depth Anything V2 Small at 266 pixels, `latency_ms=200`, the
/// run that introduced the page's `gpu_render_alone_us` and
/// `gpu_render_with_slices_us` lines), and the collapse remained;
/// `beatlm-web`'s `diagnosis` module has what was found next. Its run, a later
/// one of the same configuration, measured 1 573 µs with slices: the two runs
/// differ, and neither shows the render carrying the slices.
#[derive(Debug, Clone)]
pub struct RenderCost {
    /// The render measured last, or the prior.
    last: Micros,
    /// The most recent measurements, at most [`RENDER_WINDOW`].
    recent: VecDeque<Micros>,
}

impl Default for RenderCost {
    fn default() -> Self {
        Self {
            last: RENDER_PRIOR,
            recent: VecDeque::with_capacity(RENDER_WINDOW),
        }
    }
}

impl RenderCost {
    /// Takes one frame's measured render.
    #[inline]
    pub fn record(&mut self, render: Micros) {
        self.last = render;
        if self.recent.len() == RENDER_WINDOW {
            self.recent.pop_front();
        }
        self.recent.push_back(render);
    }

    /// The render measured last, or the prior before any.
    #[inline]
    #[must_use]
    pub fn last(&self) -> Micros {
        self.last
    }

    /// The render the frames to come are forecast with: the median of the
    /// recent measurements (the upper one of an even count), or the prior
    /// before any has arrived.
    #[inline]
    #[must_use]
    pub fn forecast(&self) -> Micros {
        // On the stack: both hosts read it every frame.
        let mut window = [Micros::ZERO; RENDER_WINDOW];
        let recent = &mut window[..self.recent.len()];
        for (slot, &render) in recent.iter_mut().zip(&self.recent) {
            *slot = render;
        }
        if recent.is_empty() {
            return RENDER_PRIOR;
        }
        let middle = recent.len() / 2;
        *recent.select_nth_unstable(middle).1
    }

    /// This frame's budget, at `period` less the render measured last and
    /// `margin`.
    #[inline]
    #[must_use]
    pub fn budget(&self, period: Micros, margin: Micros) -> Budget {
        Budget::for_frame(period, self.last, margin)
    }

    /// The budget expected of the frames to come, at `period` less the
    /// forecast render and `margin`.
    #[inline]
    #[must_use]
    pub fn expected(&self, period: Micros, margin: Micros) -> Budget {
        Budget::for_frame(period, self.forecast(), margin)
    }
}

/// When a frame submits its input as a job: the bench's `--submit`.
/// `beatlm-bench`'s crate documentation, under "The `depth` workload", says
/// why the depth model needs [`Free`](Self::Free).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Submit {
    /// Every frame submits one job.
    Every,
    /// A frame submits a job only when the job submitted last has had all its
    /// slices handed out, or has left the scheduler; otherwise its input is
    /// not run, as a host feeding a model slower than its frame rate skips
    /// frames of the source.
    Free,
}

impl Submit {
    /// The spelling the bench's `--submit` accepts and its report prints.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Every => "every",
            Self::Free => "free",
        }
    }
}

/// Whether a frame's input becomes a job: the [`Submit`] rule, kept apart
/// from the device so a test can drive it against the scheduler alone. A
/// host may hold an input back for reasons of its own besides (a page's
/// on-beat run submits only when a beat is owed one); this says only what the
/// rule allows.
#[derive(Debug, Clone, Copy)]
pub struct Admission {
    submit: Submit,
    /// The job submitted last, with how many of its slices are not yet handed
    /// out, while it has slices left and has not left the scheduler: what
    /// [`Submit::Free`] waits for.
    busy: Option<(JobId, usize)>,
}

impl Admission {
    /// The rule `submit`, with no job submitted yet.
    #[must_use]
    pub const fn new(submit: Submit) -> Self {
        Self { submit, busy: None }
    }

    /// Whether the rule takes this frame's input.
    #[inline]
    #[must_use]
    pub const fn takes_input(&self) -> bool {
        match self.submit {
            Submit::Every => true,
            Submit::Free => self.busy.is_none(),
        }
    }

    /// Records the frame's plan, after `job` of its slice count was submitted
    /// for it if one was: the slices handed out, and the jobs that left.
    #[inline]
    pub fn planned(&mut self, job: Option<(JobId, usize)>, plan: &FramePlan) {
        // Only `Free` reads what is left of the job submitted last.
        if self.submit == Submit::Every {
            return;
        }
        if job.is_some() {
            self.busy = job;
        }
        if let Some((busy, left)) = &mut self.busy {
            let handed_out = plan.runs.iter().filter(|run| run.job == *busy).count();
            *left = left.saturating_sub(handed_out);
            // A job that left the scheduler, delivered or missed with slices
            // never handed out, no longer holds back the next input.
            let gone = plan.delivered.iter().any(|d| d.job == *busy)
                || plan.missed.iter().any(|m| m.job == *busy);
            if *left == 0 || gone {
                self.busy = None;
            }
        }
    }
}

/// How many jobs the sync promise lets be pending at once, which under
/// [`Submit::Every`] is how many can have slices left.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InFlight {
    /// A fixed latency of this many frames: a job lives that many frames past
    /// its submission, so one more than that can be pending.
    Latency(u32),
    /// A promise that does not bound them, whose host keeps at most this many
    /// pending (a page's on-beat run, one per output slot).
    Bounded(u32),
}

/// How many timed slices the loop's timer must hold for jobs of `slices`
/// slices when `jobs` of them can have slices left at once
/// ([`jobs_with_slices`]): a frame can run slices of every such job, not only
/// its own. `None` when that is more than one query set can time
/// ([`MAX_SLICES`]) or overflows.
///
/// A host checks it for the loop's split program before building the loop's
/// timer, and, for a program it splits by calibration, before calibration's
/// first pass with the unsplit program and before its second with the split
/// one, so a job the loop could not time fails before calibration spends its
/// seconds.
#[must_use]
pub fn timer_capacity(slices: u32, jobs: u32) -> Option<u32> {
    slices
        .checked_mul(jobs)
        .filter(|&capacity| capacity <= MAX_SLICES)
}

/// How many jobs can have slices left at once under `submit`, with
/// `in_flight` the most the promise lets be pending. Submitting every frame,
/// each pending job can. Submitting only once the last job's slices are all
/// handed out, or it has left, one job at most has slices left. `None` on
/// overflow.
#[must_use]
pub const fn jobs_with_slices(submit: Submit, in_flight: InFlight) -> Option<u32> {
    match (submit, in_flight) {
        (Submit::Every, InFlight::Latency(frames)) => frames.checked_add(1),
        (Submit::Every, InFlight::Bounded(jobs)) => Some(jobs),
        (Submit::Free, _) => Some(1),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use beatlm_core::cost::{CostModel, Estimator};
    use beatlm_core::schedule::Scheduler;
    use beatlm_core::sync::{MissPolicy, SyncPolicy};
    use beatlm_core::time::{FrameIndex, FrameInfo, TimePoint};

    #[test]
    fn the_budget_follows_the_last_render_and_the_forecast_the_median() {
        let mut render = RenderCost::default();
        let (period, margin) = (Micros::new(8_300), Micros::new(2_000));
        assert_eq!(
            render.budget(period, margin),
            Budget::new(Micros::new(4_300))
        );
        for us in [1_900, 1_900, 9_000] {
            render.record(Micros::new(us));
        }
        assert_eq!(render.last(), Micros::new(9_000));
        assert_eq!(render.budget(period, margin), Budget::new(Micros::ZERO));
        assert_eq!(render.forecast(), Micros::new(1_900));
        assert_eq!(
            render.expected(period, margin),
            Budget::new(Micros::new(4_400))
        );
        for _ in 0..RENDER_WINDOW {
            render.record(Micros::new(2_000));
        }
        assert_eq!(render.forecast(), Micros::new(2_000));
    }

    /// What a simulated run delivered and missed.
    #[derive(Debug)]
    struct Delivery {
        delivered: u64,
        missed: u64,
    }

    /// The depth model's loop on the scheduler alone, at the period, margin
    /// and latency of the Chrome run that collapsed (8 300 µs, 2 ms, 25
    /// frames), with the [`Submit::Free`] admission, a job of 160 slices of
    /// about 29 ms, estimated at what they take, and a render pass of
    /// 1 900 µs whose measurement also counts the frame's slices when
    /// `waits` is set. Each frame's measurements reach the budget the next
    /// frame, as readbacks arrive between callbacks.
    fn simulate(waits: bool, frames: u64) -> Delivery {
        let (period, margin, render_us) = (Micros::new(8_300), Micros::new(2_000), 1_900);
        let costs: Vec<u64> = (0..160).map(|i| [60, 120, 180, 240, 300][i % 5]).collect();
        let slices: Vec<SliceId> = (0..160).map(SliceId::new).collect();
        let mut cost = CostModel::new(Micros::new(2_000), Estimator::default()).unwrap();
        for _ in 0..32 {
            for (slice, &us) in slices.iter().zip(&costs) {
                cost.record(*slice, Micros::new(us));
            }
        }
        let mut scheduler = Scheduler::new(
            SyncPolicy::FixedLatency { frames: 25 },
            MissPolicy::HoldPrevious,
            Budget::for_frame(period, RENDER_PRIOR, margin),
        );
        let mut render = RenderCost::default();
        let mut admission = Admission::new(Submit::Free);
        let mut measured = None;
        let mut delivery = Delivery {
            delivered: 0,
            missed: 0,
        };
        for index in 0..frames {
            if let Some(us) = measured.take() {
                render.record(Micros::new(us));
            }
            let frame = FrameInfo {
                index: FrameIndex::new(index),
                start: TimePoint::new(index * period.get()),
                period,
            };
            scheduler.set_expected_budget(render.expected(period, margin));
            let job = if admission.takes_input() {
                Some(scheduler.submit(frame, slices.clone(), &cost).unwrap())
            } else {
                None
            };
            let plan = scheduler.plan_frame(frame, render.budget(period, margin), &cost);
            let ran: u64 = plan
                .runs
                .iter()
                .map(|run| costs[run.slice.get() as usize])
                .sum();
            measured = Some(render_us + if waits { ran } else { 0 });
            admission.planned(job.map(|job| (job, slices.len())), &plan);
            delivery.delivered += plan.delivered.len() as u64;
            delivery.missed += plan.missed.len() as u64;
        }
        delivery
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

    #[test]
    fn the_timer_must_hold_the_slices_of_every_job_with_slices_left() {
        assert_eq!(timer_capacity(6, 3), Some(18));
        assert_eq!(timer_capacity(MAX_SLICES, 1), Some(MAX_SLICES));
        assert_eq!(timer_capacity(MAX_SLICES, 2), None);
        assert_eq!(timer_capacity(1, u32::MAX), None);
        assert_eq!(timer_capacity(u32::MAX, 3), None);
        // Every frame's input: the latency plus one jobs, or as many as the
        // host keeps pending. Only when the last is handed out: one, whatever
        // the latency.
        let every = |frames| jobs_with_slices(Submit::Every, InFlight::Latency(frames));
        let free = |frames| jobs_with_slices(Submit::Free, InFlight::Latency(frames));
        assert_eq!(every(2), Some(3));
        assert_eq!(every(u32::MAX), None);
        assert_eq!(free(17), Some(1));
        assert_eq!(
            jobs_with_slices(Submit::Every, InFlight::Bounded(3)),
            Some(3)
        );
        assert_eq!(
            jobs_with_slices(Submit::Free, InFlight::Bounded(3)),
            Some(1)
        );
        assert_eq!(timer_capacity(300, free(17).unwrap()), Some(300));
        assert_eq!(timer_capacity(300, every(17).unwrap()), None);
    }

    #[test]
    fn a_split_into_more_slices_than_the_timer_holds_is_refused_with_one_job() {
        // Calibration's check before its second pass: the unsplit job fit, and
        // a split of it into more slices than one frame can time is refused
        // even with one job at a time, which no latency lowers.
        let one = jobs_with_slices(Submit::Free, InFlight::Latency(10)).unwrap();
        assert_eq!(timer_capacity(155, one), Some(155));
        assert_eq!(timer_capacity(MAX_SLICES + 1, one), None);
    }

    /// The scheduler alone, for `frames` frames of a 10 ms budget, expected
    /// and real, with jobs of `slices` slices each estimated at 1 ms, under
    /// latency `k` and `submit`: how many inputs were submitted, delivered and
    /// missed. The period, which a budget given outright does not read, is
    /// the bench's.
    fn schedule_alone(submit: Submit, slices: u32, k: u32, frames: u64) -> (u64, u64, u64) {
        const PERIOD: Micros = Micros::new(16_667);
        let cost = CostModel::new(
            Micros::new(1_000),
            Estimator::MeanPlusSigmas {
                alpha: 1.0,
                sigmas: 0.0,
            },
        )
        .unwrap();
        let policy = SyncPolicy::FixedLatency { frames: k };
        let budget = Budget::new(Micros::new(10_000));
        let mut scheduler = Scheduler::new(policy, MissPolicy::HoldPrevious, budget);
        let mut admission = Admission::new(submit);
        let job: Vec<SliceId> = (0..slices).map(SliceId::new).collect();
        let (mut submitted, mut delivered, mut missed) = (0, 0, 0);
        for i in 0..frames {
            let frame = FrameInfo {
                index: FrameIndex::new(i),
                start: TimePoint::new(i * PERIOD.get()),
                period: PERIOD,
            };
            let new_job = admission.takes_input().then(|| {
                submitted += 1;
                let id = scheduler.submit(frame, job.clone(), &cost).unwrap();
                (id, job.len())
            });
            let plan = scheduler.plan_frame(frame, budget, &cost);
            admission.planned(new_job, &plan);
            delivered += plan.delivered.len() as u64;
            missed += plan.missed.len() as u64;
        }
        (submitted, delivered, missed)
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
