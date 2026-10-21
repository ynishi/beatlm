//! Deciding which slices run in a frame and when each result is delivered.
//!
//! The scheduler is where [the queue is the
//! synchronisation](crate#the-queue-is-the-synchronisation) is applied: a job
//! is complete once all its slices were handed out in this or an earlier
//! frame. Measured GPU timings flow back only into the [`CostModel`]; the
//! scheduler itself is a pure function of the frames, budgets and estimates it
//! is given.
//!
//! A [`Scheduler`] turns each submitted input into a job with a deadline fixed
//! by its [`SyncPolicy`]. Each frame it spends the [`Budget`] on pending jobs
//! in submission order, delivers the jobs whose deadline arrives and are
//! complete, and reports the ones that are not as missed.
//!
//! # Jobs run as one sequence
//!
//! Jobs of one scheduler share one [`MemoryPlan`](crate::memory::MemoryPlan),
//! which lets tensors of non-overlapping lifetimes share bytes over one job's
//! operations in program order. A later job's first slice would overwrite
//! bytes an earlier job's remaining slices still read, so slices never
//! interleave: no job receives a slice while an earlier, not-yet-missed job
//! has slices left. This is the rule even where it is not earliest deadline
//! first: an `OnBeat` job with little work can target an earlier beat than an
//! older, larger one, and it still waits. A job the budget cannot finish is
//! missed (and reported) rather than overtaken.
//!
//! # Projecting the queue onto frames to come
//!
//! Two decisions depend on frames the scheduler has not seen: the beat an
//! `OnBeat` job is promised at submission, and whether a pending job can still
//! make its deadline. Both read one projection, so that with the budget as
//! expected a job placed on a beat at submission is never abandoned for it in
//! the same frame.
//!
//! The projection lays the remaining slices of the pending jobs, in submission
//! order, onto consecutive frames from the one being planned, each starting a
//! period after the last. The planned frame has its own budget and every later
//! frame the budget the host expects
//! ([`Scheduler::set_expected_budget`]). A frame is filled the way
//! [`Scheduler::plan_frame`] fills one: it takes slices while they fit, and the
//! first that does not opens the next frame, for whichever job it belongs to.
//! A job is projected to finish in the frame its last slice lands in, at that
//! frame's start plus the estimates landed in the frame up to that slice:
//! slices are submitted before the frame's render pass, so they run first.
//!
//! Before this, an on-beat deadline was the first beat after the input's start
//! plus the job's summed estimates, as though the job ran back to back. A job
//! spread over several frames by the budget, or queued behind another, missed
//! that beat every time; the projection is what replaced it.
//!
//! The expected budget is a setting of the scheduler, not an argument of each
//! call, for two reasons. Submission and planning must project with the same
//! figure, or one would place a job on a beat that the other then abandons it
//! for; one value held in one place makes them agree. And it is a different
//! figure from a frame's budget: the budget a frame gets is that frame's
//! measurement, and one slow render would, read as the forecast for every
//! frame after it, abandon a job the following frames would have finished.
//! The host keeps the expected budget current, as it keeps the frame's.
//!
//! # A job that cannot make its deadline
//!
//! When the projection puts a job's last slice past the frame that keeps its
//! deadline, or a slice of it fits no frame of the expected budget at all, the
//! job is abandoned: it receives no more slices, leaves, and is reported
//! missed in that plan, with the deadline the miss is for
//! ([`Missed::due`]). The projection then goes on without it.
//!
//! Keeping it instead would spend GPU time on a result that is reported missed
//! anyway, and — because jobs run in submission order — hold every job behind
//! it until its deadline, so they start late and can miss in turn. A host that
//! submits faster than the budget can serve then misses nearly everything
//! (the bench's `engine` tests show it on the scheduler alone). Abandoning
//! keeps both promises the rules make: the miss is reported, earlier than its
//! deadline but never later, and `due` tells the host which frame or beat to
//! show [`MissPolicy`] on; and nothing is delivered late, because an abandoned
//! job is never delivered at all.
//!
//! The price is trusting the projection. A job abandoned because the expected
//! budget was set too low might have made its deadline; one kept because it was
//! set too high is decided in a later frame, at the latest the frame that keeps
//! its deadline, where only that frame's budget counts and the answer is
//! certain. Either way no deadline is stretched: it is fixed at submission.
//!
//! # Tempo changes
//!
//! The beats of an `OnBeat` scheduler are a [`BeatMap`]: the grid of the
//! policy, then each grid the host puts in effect from an instant
//! ([`Scheduler::replace_beat_grid`]). A job submitted after a change takes its
//! beat from the grid in effect at that beat. A deadline already fixed stays
//! as it was: it is the promise the job was placed under, and moving it to the
//! new grid would either stretch it to a later beat, which the rules forbid, or
//! pull it to an earlier one its work was never projected to make.

use thiserror::Error;

use crate::budget::Budget;
use crate::cost::{CostModel, SliceId};
use crate::sync::{MissPolicy, SyncPolicy};
use crate::time::{BeatGrid, BeatMap, FrameIndex, FrameInfo, Micros, TimePoint};

/// Identifies one submitted input.
///
/// Ids are assigned by the scheduler in submission order, starting at zero, so
/// ordering by id is ordering by submission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct JobId(u64);

impl JobId {
    /// The job's sequence number.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// One slice the host submits to the GPU queue this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SliceRun {
    /// The job the slice belongs to.
    pub job: JobId,
    /// Which slice of the plan to run.
    pub slice: SliceId,
    /// The slice's index within its job; slices of a job run in this order.
    pub position: usize,
}

/// A result the host delivers this frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Delivery {
    /// The job whose result is ready.
    pub job: JobId,
    /// The frame whose input produced the result.
    pub input: FrameIndex,
}

/// When a job's result is due: a frame, or a beat.
///
/// A scheduler has one policy, so all its deadlines are the same kind. There
/// is no ordering: a frame and a beat are not comparable, and a host holding
/// [`Missed`] reports from more than one scheduler can meet both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Due {
    /// Delivered on exactly this frame (`FixedLatency`, `SameFrame`).
    Frame(FrameIndex),
    /// Delivered on the first frame starting at or after this beat (`OnBeat`).
    Beat(TimePoint),
}

impl Due {
    /// The deadline arrives with this frame, so a decision about the job is
    /// due now.
    fn reached(self, frame: FrameInfo) -> bool {
        match self {
            Self::Frame(d) => frame.index >= d,
            Self::Beat(d) => frame.start >= d,
        }
    }

    /// Delivering in this frame keeps the promise. A frame deadline is kept
    /// only on its own frame: the host delays its source by exactly the
    /// promised number of frames, so a result shown on any later frame is
    /// misaligned even though it is finished. A beat deadline is kept only by
    /// the first frame that starts at or after the beat: one starting less than
    /// a period after it. A frame starting a whole period or more after the
    /// beat means the host skipped the frame that would have kept it.
    fn on_time(self, frame: FrameInfo) -> bool {
        match self {
            Self::Frame(d) => frame.index == d,
            Self::Beat(d) => {
                frame.start == d
                    || (frame.start > d && frame.start.saturating_since(d) < frame.period)
            }
        }
    }

    /// How many frames after `frame` the frame that keeps this deadline is,
    /// when the frames from `frame` on follow each other at its period; `None`
    /// when the deadline is already behind `frame`. The frame that keeps a
    /// deadline still counts: its slices are submitted before its render pass.
    ///
    /// A zero period never reaches a later beat; it is read as unbounded, so
    /// the job is decided on the frame its beat arrives with.
    fn frames_left(self, frame: FrameInfo) -> Option<u64> {
        match self {
            Self::Frame(d) => d.get().checked_sub(frame.index.get()),
            Self::Beat(d) if d <= frame.start => self.on_time(frame).then_some(0),
            Self::Beat(d) => {
                let ahead = d.saturating_since(frame.start).get();
                Some(match frame.period.get() {
                    0 => u64::MAX,
                    period => ahead.div_ceil(period),
                })
            }
        }
    }
}

/// A deadline the scheduler could not keep.
///
/// It is reported so that the miss is never silent; `policy` tells the host
/// what to show instead, on the frame or beat `due` names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Missed {
    /// The job that did not finish in time.
    pub job: JobId,
    /// The frame whose input has no result.
    pub input: FrameIndex,
    /// The deadline the miss is for. A job abandoned because it cannot make
    /// its deadline is reported before the deadline arrives (the [module
    /// documentation](self#a-job-that-cannot-make-its-deadline)), so the host
    /// reads here when the result it will not get was due.
    pub due: Due,
    /// What the host shows for this input.
    pub policy: MissPolicy,
}

/// Everything the host does for one frame, as decided by
/// [`Scheduler::plan_frame`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FramePlan {
    /// Slices to submit before the frame's render pass, in submission order.
    pub runs: Vec<SliceRun>,
    /// Results to deliver this frame.
    pub delivered: Vec<Delivery>,
    /// Jobs reported missed by this plan: their deadline arrived without a
    /// complete result, or they were abandoned as unable to make it.
    pub missed: Vec<Missed>,
    /// The summed estimate of `runs`; never more than `budget`.
    pub spent_estimate: Micros,
    /// The budget the plan was made against.
    pub budget: Budget,
}

/// A rejected job or scheduler request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ScheduleError {
    /// A job with no slices has nothing to run and nothing to deliver.
    #[error("a job needs at least one slice")]
    EmptyJob,
    /// Only an `OnBeat` scheduler has beats to replace.
    #[error("only an OnBeat scheduler has a beat grid")]
    NotOnBeat,
}

/// Where the projected work of the queue ends: a frame, counted from the one
/// being planned, and the estimate already placed in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Cursor {
    frame: u64,
    spent: Micros,
}

impl Cursor {
    const START: Self = Self {
        frame: 0,
        spent: Micros::ZERO,
    };

    /// When the work up to the cursor is projected to finish, the frames from
    /// `planned` on following each other at its period.
    fn instant(self, planned: FrameInfo) -> TimePoint {
        planned
            .start
            .saturating_add(Micros::new(self.frame.saturating_mul(planned.period.get())))
            .saturating_add(self.spent)
    }
}

/// Slices laid onto frames by [`Budgets::place`].
#[derive(Debug, Clone, Copy)]
struct Placed {
    /// Where the last slice landed.
    end: Cursor,
    /// How many of the slices landed in the planned frame.
    now: usize,
}

/// The budgets a projection fills: the planned frame's own, and the expected
/// budget of every frame after it.
#[derive(Debug, Clone, Copy)]
struct Budgets {
    now: Budget,
    later: Budget,
}

impl Budgets {
    fn of(self, frame: u64) -> Micros {
        if frame == 0 {
            self.now.get()
        } else {
            self.later.get()
        }
    }

    /// Lays `slices` in order onto frames from `from`: a frame takes slices
    /// while they fit, and the first that does not goes to the next frame.
    /// `None` when a slice does not fit even an empty frame after the planned
    /// one, so no number of frames finishes it.
    fn place(self, from: Cursor, slices: &[SliceId], cost: &CostModel) -> Option<Placed> {
        let mut at = from;
        let mut now = 0;
        for &slice in slices {
            let estimate = cost.estimate(slice);
            loop {
                let after = at.spent.saturating_add(estimate);
                if after <= self.of(at.frame) {
                    at.spent = after;
                    break;
                }
                if at.frame > 0 && at.spent == Micros::ZERO {
                    return None;
                }
                at = Cursor {
                    frame: at.frame.saturating_add(1),
                    spent: Micros::ZERO,
                };
            }
            if at.frame == 0 {
                now += 1;
            }
        }
        Some(Placed { end: at, now })
    }
}

/// What the projection decides for one pending job.
#[derive(Debug, Clone, Copy)]
enum Fate {
    /// The deadline is behind the planned frame: the job gets nothing and is
    /// reported missed by this plan.
    Passed,
    /// The job is projected to finish after the frame that keeps its deadline:
    /// it gets nothing, is reported missed by this plan, and the projection
    /// goes on without it.
    Abandoned,
    /// The job is projected to finish in time: `now` of its next slices run in
    /// the planned frame, and the projection goes on from `end`.
    Kept { now: usize, end: Cursor },
}

#[derive(Debug, Clone)]
struct Job {
    id: JobId,
    input: FrameIndex,
    slices: Vec<SliceId>,
    /// How many leading slices have been handed out.
    scheduled: usize,
    deadline: Due,
    /// Set by the plan that abandons the job, which also removes it.
    abandoned: bool,
}

impl Job {
    fn is_complete(&self) -> bool {
        self.scheduled >= self.slices.len()
    }

    /// The job's fate when the work kept ahead of it ends at `from`.
    fn fate(&self, from: Cursor, frame: FrameInfo, budgets: Budgets, cost: &CostModel) -> Fate {
        let Some(frames_left) = self.deadline.frames_left(frame) else {
            return Fate::Passed;
        };
        let rest = self.slices.get(self.scheduled..).unwrap_or_default();
        if rest.is_empty() {
            return Fate::Kept { now: 0, end: from };
        }
        match budgets.place(from, rest, cost) {
            Some(placed) if placed.end.frame <= frames_left => Fate::Kept {
                now: placed.now,
                end: placed.end,
            },
            _ => Fate::Abandoned,
        }
    }
}

/// How a scheduler turns an input into a deadline: its [`SyncPolicy`], with
/// the beats an `OnBeat` policy carries grown into a [`BeatMap`] the host can
/// change.
#[derive(Debug, Clone)]
enum Promise {
    /// Due this many frames after the input frame; `SameFrame` is zero.
    Frames(u32),
    /// Due on a beat of `beats`, placed with `margin`.
    Beats { margin: Micros, beats: BeatMap },
}

/// Decides, frame by frame, which slices run and which results are delivered.
///
/// It owns the promise made by a [`SyncPolicy`]: it fixes each job's deadline
/// at submission, spends the frame budget on jobs in submission order (see the
/// [module documentation](self#jobs-run-as-one-sequence)), abandons a job that
/// cannot make its deadline, and reports every deadline it could not keep with
/// the configured [`MissPolicy`].
#[derive(Debug, Clone)]
pub struct Scheduler {
    promise: Promise,
    miss: MissPolicy,
    expected: Budget,
    next_job: u64,
    pending: Vec<Job>,
}

impl Scheduler {
    /// A scheduler that keeps `policy`, reports misses with `miss`, and expects
    /// each frame to come to have `expected` to spend
    /// ([`set_expected_budget`](Self::set_expected_budget)).
    #[must_use]
    pub fn new(policy: SyncPolicy, miss: MissPolicy, expected: Budget) -> Self {
        let promise = match policy {
            SyncPolicy::FixedLatency { frames } => Promise::Frames(frames),
            SyncPolicy::SameFrame => Promise::Frames(0),
            SyncPolicy::OnBeat { margin, grid } => Promise::Beats {
                margin,
                beats: BeatMap::new(grid),
            },
        };
        Self {
            promise,
            miss,
            expected,
            next_job: 0,
            pending: Vec::new(),
        }
    }

    /// Sets the budget the host expects each frame after the one being planned
    /// to have, for submissions and plans from now on.
    ///
    /// The [projection](self#projecting-the-queue-onto-frames-to-come) reads
    /// it for every frame it has not seen: [`submit`](Self::submit) places an
    /// on-beat deadline with it, and [`plan_frame`](Self::plan_frame) abandons
    /// a job that cannot make its deadline at it. It is a forecast and not the
    /// frame's measurement: set from a single slow frame it abandons jobs the
    /// following frames would finish. Deadlines already fixed are not moved by
    /// a new value.
    pub fn set_expected_budget(&mut self, expected: Budget) {
        self.expected = expected;
    }

    /// The budget the scheduler expects each frame to come to have.
    #[must_use]
    pub const fn expected_budget(&self) -> Budget {
        self.expected
    }

    /// Puts `grid` in effect from `from` on, for jobs submitted from now on;
    /// a tempo change.
    ///
    /// Deadlines already fixed are not moved (the [module
    /// documentation](self#tempo-changes) says why), and every grid that took
    /// effect at or after `from` is dropped ([`BeatMap::replace_from`]).
    ///
    /// # Errors
    ///
    /// [`ScheduleError::NotOnBeat`] when the scheduler's policy is not
    /// `OnBeat`.
    pub fn replace_beat_grid(
        &mut self,
        from: TimePoint,
        grid: BeatGrid,
    ) -> Result<(), ScheduleError> {
        match &mut self.promise {
            Promise::Beats { beats, .. } => {
                beats.replace_from(from, grid);
                Ok(())
            }
            Promise::Frames(_) => Err(ScheduleError::NotOnBeat),
        }
    }

    /// Registers the input of frame `input` as a job that runs `slices` in
    /// order, and fixes its deadline.
    ///
    /// The host calls it in the input's frame, before that frame's
    /// [`plan_frame`](Self::plan_frame), the order the `beatlm` crate gives
    /// the host's side of a frame, so the job can run in the frame it was
    /// submitted in.
    ///
    /// The deadline is `input + frames` for `FixedLatency` and the input frame
    /// for `SameFrame`. For `OnBeat` it is the first beat at or after the job's
    /// projected finish plus the margin, on the grid in effect at that beat:
    /// the [projection](self#projecting-the-queue-onto-frames-to-come) lays the
    /// slices of the jobs ahead that it keeps, then this job's, onto frames
    /// from the input's on, each of the expected budget. When a slice of the
    /// job fits no frame of that budget, the finish is taken as the input's
    /// start plus the job's summed estimates, the earliest any budget could
    /// make, and the job is abandoned when planned unless that frame's budget
    /// runs it.
    ///
    /// The deadline is fixed here: neither a later budget nor a tempo change
    /// moves it. A frame budget below the expected one is a miss reported
    /// when the projection shows it, never a later deadline.
    ///
    /// # Errors
    ///
    /// [`ScheduleError::EmptyJob`] when `slices` is empty.
    pub fn submit(
        &mut self,
        input: FrameInfo,
        slices: Vec<SliceId>,
        cost: &CostModel,
    ) -> Result<JobId, ScheduleError> {
        if slices.is_empty() {
            return Err(ScheduleError::EmptyJob);
        }
        let deadline = match &self.promise {
            Promise::Frames(frames) => Due::Frame(input.index.offset(*frames)),
            Promise::Beats { margin, beats } => {
                let budgets = Budgets {
                    now: self.expected,
                    later: self.expected,
                };
                let ahead = self.projected_end(input, budgets, cost);
                let finish = match budgets.place(ahead, &slices, cost) {
                    Some(placed) => placed.end.instant(input),
                    None => input.start.saturating_add(
                        slices
                            .iter()
                            .fold(Micros::ZERO, |sum, s| sum.saturating_add(cost.estimate(*s))),
                    ),
                };
                Due::Beat(beats.next_beat_at_or_after(finish.saturating_add(*margin)))
            }
        };
        let id = JobId(self.next_job);
        self.next_job = self.next_job.saturating_add(1);
        self.pending.push(Job {
            id,
            input: input.index,
            slices,
            scheduled: 0,
            deadline,
            abandoned: false,
        });
        Ok(id)
    }

    /// Where the projected work of the jobs kept ahead ends, planning `frame`
    /// with `budgets`.
    fn projected_end(&self, frame: FrameInfo, budgets: Budgets, cost: &CostModel) -> Cursor {
        self.pending.iter().fold(Cursor::START, |at, job| {
            match job.fate(at, frame, budgets, cost) {
                Fate::Kept { end, .. } => end,
                Fate::Passed | Fate::Abandoned => at,
            }
        })
    }

    /// Plans `frame`: hands out slices within `budget`, then delivers or
    /// reports the jobs whose deadline arrives or that cannot make it.
    ///
    /// Pending jobs are served in submission ([`JobId`]) order and never
    /// interleave, because they share one memory plan: a job's slices go out in
    /// order while they fit, and when its next slice does not fit the remaining
    /// budget no slice of any later job is handed out this frame, however
    /// small. The summed estimate of the runs never exceeds `budget`.
    ///
    /// Each job's remaining slices are
    /// [projected](self#projecting-the-queue-onto-frames-to-come) after the
    /// work kept ahead of it, onto this frame's `budget` and the expected
    /// budget of the frames after it. A job projected to finish after the
    /// frame that keeps its deadline is abandoned: it gets no slices, does not
    /// hold up the jobs after it, and is reported missed by this plan. So is a
    /// job whose deadline is already behind `frame`.
    ///
    /// A job is complete once all its slices were handed out in this or an
    /// earlier frame. A job whose deadline arrives with `frame` is delivered if
    /// it is complete and `frame` keeps its promise, and reported as missed
    /// otherwise; either way it leaves. Only the deadline frame itself — or,
    /// for a beat, the first frame starting at or after it — keeps the
    /// promise, so a job the host first plans after a skipped deadline frame
    /// is missed even when finished. A complete job whose deadline has not
    /// arrived stays, which is what keeps latency fixed when work finishes
    /// early.
    pub fn plan_frame(&mut self, frame: FrameInfo, budget: Budget, cost: &CostModel) -> FramePlan {
        if let Promise::Beats { beats, .. } = &mut self.promise {
            // Inputs are submitted in their frame, so none from here on finishes
            // before `frame` starts and needs a grid whose span ended by then.
            beats.forget_before(frame.start);
        }
        let budgets = Budgets {
            now: budget,
            later: self.expected,
        };
        // `pending` is in submission order: jobs are only ever pushed, and
        // removal keeps the order of the rest.
        let mut runs = Vec::new();
        let mut spent = Micros::ZERO;
        let mut at = Cursor::START;
        for job in &mut self.pending {
            match job.fate(at, frame, budgets, cost) {
                Fate::Passed => {}
                Fate::Abandoned => job.abandoned = true,
                Fate::Kept { now, end } => {
                    for _ in 0..now {
                        let Some(&slice) = job.slices.get(job.scheduled) else {
                            break;
                        };
                        runs.push(SliceRun {
                            job: job.id,
                            slice,
                            position: job.scheduled,
                        });
                        spent = spent.saturating_add(cost.estimate(slice));
                        job.scheduled += 1;
                    }
                    at = end;
                }
            }
        }

        let mut delivered = Vec::new();
        let mut missed = Vec::new();
        for job in self
            .pending
            .extract_if(.., |job| job.abandoned || job.deadline.reached(frame))
        {
            if !job.abandoned && job.is_complete() && job.deadline.on_time(frame) {
                delivered.push(Delivery {
                    job: job.id,
                    input: job.input,
                });
            } else {
                missed.push(Missed {
                    job: job.id,
                    input: job.input,
                    due: job.deadline,
                    policy: self.miss,
                });
            }
        }

        FramePlan {
            runs,
            delivered,
            missed,
            spent_estimate: spent,
            budget,
        }
    }

    /// How many jobs are submitted and not yet delivered or reported missed.
    #[must_use]
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    /// The deadline fixed for `job`, while it is pending.
    #[must_use]
    pub fn due(&self, job: JobId) -> Option<Due> {
        self.pending
            .binary_search_by_key(&job, |j| j.id)
            .ok()
            .and_then(|i| self.pending.get(i))
            .map(|j| j.deadline)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::cost::Estimator;

    const PERIOD: u64 = 1_000;

    fn frame(i: u64) -> FrameInfo {
        FrameInfo {
            index: FrameIndex::new(i),
            start: TimePoint::new(i * PERIOD),
            period: Micros::new(PERIOD),
        }
    }

    fn at(start: u64) -> FrameInfo {
        FrameInfo {
            index: FrameIndex::new(start / PERIOD),
            start: TimePoint::new(start),
            period: Micros::new(PERIOD),
        }
    }

    fn budget(us: u64) -> Budget {
        Budget::new(Micros::new(us))
    }

    /// Every unmeasured slice costs 100 us.
    fn cost() -> CostModel {
        CostModel::new(
            Micros::new(100),
            Estimator::MeanPlusSigmas {
                alpha: 1.0,
                sigmas: 0.0,
            },
        )
        .unwrap()
    }

    fn slices(ids: &[u32]) -> Vec<SliceId> {
        ids.iter().copied().map(SliceId::new).collect()
    }

    /// A scheduler expecting `expected` us of every frame to come.
    fn sched(policy: SyncPolicy, miss: MissPolicy, expected: u64) -> Scheduler {
        Scheduler::new(policy, miss, budget(expected))
    }

    fn grid() -> BeatGrid {
        BeatGrid::new(TimePoint::new(0), Micros::new(1_000)).unwrap()
    }

    fn on_beat(margin: u64) -> SyncPolicy {
        SyncPolicy::OnBeat {
            margin: Micros::new(margin),
            grid: grid(),
        }
    }

    fn beat(us: u64) -> Due {
        Due::Beat(TimePoint::new(us))
    }

    #[test]
    fn replacing_beats_needs_an_on_beat_scheduler() {
        let mut s = sched(SyncPolicy::SameFrame, MissPolicy::Drop, 100);
        let r = s.replace_beat_grid(TimePoint::new(0), grid());
        assert_eq!(r.unwrap_err(), ScheduleError::NotOnBeat);
        let mut s = sched(on_beat(0), MissPolicy::Drop, 100);
        assert!(s.replace_beat_grid(TimePoint::new(0), grid()).is_ok());
    }

    #[test]
    fn empty_job_is_rejected() {
        let mut s = sched(SyncPolicy::SameFrame, MissPolicy::Drop, 100);
        let r = s.submit(frame(0), Vec::new(), &cost());
        assert_eq!(r.unwrap_err(), ScheduleError::EmptyJob);
        assert_eq!(s.pending(), 0);
    }

    #[test]
    fn job_ids_count_from_zero() {
        let mut s = sched(SyncPolicy::SameFrame, MissPolicy::Drop, 100);
        let a = s.submit(frame(0), slices(&[0]), &cost()).unwrap();
        let b = s.submit(frame(0), slices(&[0]), &cost()).unwrap();
        assert_eq!((a.get(), b.get()), (0, 1));
    }

    #[test]
    fn fixed_latency_delivers_exactly_on_its_frame() {
        let mut s = sched(
            SyncPolicy::FixedLatency { frames: 2 },
            MissPolicy::Drop,
            1_000,
        );
        let c = cost();
        let job = s.submit(frame(10), slices(&[0]), &c).unwrap();
        assert_eq!(s.due(job), Some(Due::Frame(FrameIndex::new(12))));

        let p10 = s.plan_frame(frame(10), budget(1_000), &c);
        assert_eq!(p10.runs.len(), 1);
        assert!(p10.delivered.is_empty() && p10.missed.is_empty());

        let p11 = s.plan_frame(frame(11), budget(1_000), &c);
        assert!(p11.runs.is_empty());
        assert!(p11.delivered.is_empty() && p11.missed.is_empty());
        assert_eq!(s.pending(), 1);

        let p12 = s.plan_frame(frame(12), budget(1_000), &c);
        assert_eq!(
            p12.delivered,
            vec![Delivery {
                job,
                input: FrameIndex::new(10)
            }]
        );
        assert!(p12.missed.is_empty());
        assert_eq!(s.pending(), 0);
        assert_eq!(s.due(job), None);
    }

    #[test]
    fn fixed_latency_job_that_cannot_finish_is_missed_at_once_without_running() {
        // Four slices of 100 us against 100 us a frame need four frames;
        // latency 2 gives three. Before jobs were abandoned this one ran three
        // slices and was reported at frame 2.
        let mut s = sched(
            SyncPolicy::FixedLatency { frames: 2 },
            MissPolicy::HoldPrevious,
            100,
        );
        let c = cost();
        let job = s.submit(frame(0), slices(&[0, 1, 2, 3]), &c).unwrap();
        let p = s.plan_frame(frame(0), budget(100), &c);
        assert!(p.runs.is_empty());
        assert!(p.delivered.is_empty());
        assert_eq!(
            p.missed,
            vec![Missed {
                job,
                input: FrameIndex::new(0),
                due: Due::Frame(FrameIndex::new(2)),
                policy: MissPolicy::HoldPrevious
            }]
        );
        assert_eq!(s.pending(), 0);

        // The job is gone: nothing more runs for it.
        for f in 1..4 {
            let later = s.plan_frame(frame(f), budget(100), &c);
            assert!(later.runs.is_empty() && later.missed.is_empty());
        }
    }

    #[test]
    fn a_forecast_above_the_real_budget_delays_the_miss_to_its_deadline() {
        // Expected 200 us a frame, four 100 us slices fit frames 0 and 1 of
        // latency 2's three. Every frame gets 100: at frame 2 two slices are
        // left for one frame, and the miss is reported then, at its deadline,
        // with nothing delivered.
        let mut s = sched(
            SyncPolicy::FixedLatency { frames: 2 },
            MissPolicy::HoldPrevious,
            200,
        );
        let c = cost();
        let job = s.submit(frame(0), slices(&[0, 1, 2, 3]), &c).unwrap();
        let mut ran = 0;
        for f in 0..2 {
            let p = s.plan_frame(frame(f), budget(100), &c);
            ran += p.runs.len();
            assert!(p.delivered.is_empty() && p.missed.is_empty(), "frame {f}");
        }
        let p = s.plan_frame(frame(2), budget(100), &c);
        assert_eq!(ran, 2);
        assert!(p.runs.is_empty(), "the last frame cannot finish it");
        assert!(p.delivered.is_empty());
        assert_eq!(p.missed.len(), 1);
        assert_eq!(p.missed[0].job, job);
        assert_eq!(p.missed[0].due, Due::Frame(FrameIndex::new(2)));
    }

    #[test]
    fn a_dip_in_one_frame_does_not_abandon_a_job_the_later_frames_finish() {
        // Frame 0 gets 100 us instead of the expected 200. The projection reads
        // that for frame 0 only, so the job is kept and finishes at frame 2.
        let mut s = sched(
            SyncPolicy::FixedLatency { frames: 2 },
            MissPolicy::Drop,
            200,
        );
        let c = cost();
        let job = s.submit(frame(0), slices(&[0, 1, 2, 3]), &c).unwrap();
        let mut ran = 0;
        for (f, b) in [(0, 100), (1, 200)] {
            let p = s.plan_frame(frame(f), budget(b), &c);
            ran += p.runs.len();
            assert!(p.missed.is_empty(), "frame {f}");
        }
        let p = s.plan_frame(frame(2), budget(200), &c);
        ran += p.runs.len();
        assert_eq!(ran, 4);
        assert_eq!(p.delivered.len(), 1);
        assert_eq!(p.delivered[0].job, job);
    }

    #[test]
    fn an_abandoned_job_frees_the_job_behind_it() {
        // Job A needs three frames and latency 1 gives two; job B, behind it,
        // fits frame 0. A is reported at once and B runs in A's place.
        let mut s = sched(
            SyncPolicy::FixedLatency { frames: 1 },
            MissPolicy::Drop,
            100,
        );
        let c = cost();
        let a = s.submit(frame(0), slices(&[0, 1, 2]), &c).unwrap();
        let b = s.submit(frame(0), slices(&[3]), &c).unwrap();
        let p0 = s.plan_frame(frame(0), budget(100), &c);
        let ran: Vec<JobId> = p0.runs.iter().map(|r| r.job).collect();
        assert_eq!(ran, vec![b]);
        assert_eq!(
            p0.missed,
            vec![Missed {
                job: a,
                input: FrameIndex::new(0),
                due: Due::Frame(FrameIndex::new(1)),
                policy: MissPolicy::Drop
            }]
        );
        let p1 = s.plan_frame(frame(1), budget(100), &c);
        assert_eq!(p1.delivered.len(), 1);
        assert_eq!(p1.delivered[0].job, b);
        // A is never delivered, on its frame or later.
        for f in 2..5 {
            let p = s.plan_frame(frame(f), budget(100), &c);
            assert!(p.delivered.is_empty() && p.missed.is_empty());
        }
    }

    #[test]
    fn miss_policy_is_carried_through() {
        let mut s = sched(SyncPolicy::SameFrame, MissPolicy::Drop, 50);
        let c = cost();
        s.submit(frame(0), slices(&[0]), &c).unwrap();
        let p = s.plan_frame(frame(0), budget(50), &c);
        assert_eq!(p.missed[0].policy, MissPolicy::Drop);
    }

    #[test]
    fn budget_is_never_exceeded_and_leftover_is_not_given_to_a_later_job() {
        let mut c = cost();
        c.record(SliceId::new(2), Micros::new(40));
        let mut s = sched(
            SyncPolicy::FixedLatency { frames: 5 },
            MissPolicy::Drop,
            150,
        );
        // Job A is earlier, with two 100 us slices; job B has one 40 us slice.
        let a = s.submit(frame(0), slices(&[0, 1]), &c).unwrap();
        let b = s.submit(frame(1), slices(&[2]), &c).unwrap();

        let p = s.plan_frame(frame(1), budget(150), &c);
        let used: Vec<(JobId, usize)> = p.runs.iter().map(|r| (r.job, r.position)).collect();
        // A's first slice fits, A's second does not; B's 40 us slice would fit
        // the 50 us left but must wait for A to finish.
        assert_eq!(used, vec![(a, 0)]);
        assert_eq!(p.spent_estimate, Micros::new(100));
        assert!(p.spent_estimate <= p.budget.get());

        let p = s.plan_frame(frame(2), budget(150), &c);
        let used: Vec<(JobId, usize)> = p.runs.iter().map(|r| (r.job, r.position)).collect();
        assert_eq!(used, vec![(a, 1), (b, 0)]);
        assert_eq!(p.spent_estimate, Micros::new(140));

        let sum = p
            .runs
            .iter()
            .map(|r| c.estimate(r.slice))
            .fold(Micros::ZERO, Micros::saturating_add);
        assert_eq!(sum, p.spent_estimate);
    }

    #[test]
    fn budget_holds_across_many_frames() {
        let c = cost();
        let mut s = sched(
            SyncPolicy::FixedLatency { frames: 20 },
            MissPolicy::Drop,
            250,
        );
        for i in 0..4 {
            s.submit(frame(i), slices(&[0, 1, 2]), &c).unwrap();
        }
        for f in 0..25 {
            let p = s.plan_frame(frame(f), budget(250), &c);
            assert!(p.spent_estimate <= Micros::new(250), "frame {f}");
            assert!(p.runs.len() <= 2, "frame {f}");
            assert!(p.missed.is_empty(), "frame {f}");
        }
        assert_eq!(s.pending(), 0);
    }

    #[test]
    fn oldest_job_gets_slices_first_even_with_the_later_deadline() {
        // Inputs submitted out of frame order: the older job has the later
        // deadline, and still runs first.
        let mut s = sched(
            SyncPolicy::FixedLatency { frames: 5 },
            MissPolicy::Drop,
            100,
        );
        let c = cost();
        let first = s.submit(frame(3), slices(&[0]), &c).unwrap();
        let second = s.submit(frame(1), slices(&[0]), &c).unwrap();
        assert!(first < second);

        let p = s.plan_frame(frame(3), budget(100), &c);
        assert_eq!(p.runs.len(), 1);
        assert_eq!(p.runs[0].job, first);
    }

    #[test]
    fn later_job_never_gets_a_slice_while_an_earlier_job_has_slices_left() {
        let mut c = cost();
        // Job B's slices are cheap enough to fit every leftover.
        c.record(SliceId::new(10), Micros::new(10));
        c.record(SliceId::new(11), Micros::new(10));
        let mut s = sched(
            SyncPolicy::FixedLatency { frames: 6 },
            MissPolicy::Drop,
            250,
        );
        let a = s.submit(frame(0), slices(&[0, 1, 2, 3]), &c).unwrap();
        let b = s.submit(frame(1), slices(&[10, 11]), &c).unwrap();

        let mut a_left = 4;
        let mut b_ran = 0;
        // 250 us fits two of A's 100 us slices and leaves 50 us over.
        for f in 1..8 {
            let p = s.plan_frame(frame(f), budget(250), &c);
            for run in &p.runs {
                if run.job == a {
                    a_left -= 1;
                } else {
                    assert_eq!(run.job, b);
                    assert_eq!(a_left, 0, "frame {f}: B ran while A had slices left");
                    b_ran += 1;
                }
            }
            // Within a frame A's slices also precede B's.
            let first_b = p.runs.iter().position(|r| r.job == b);
            let last_a = p.runs.iter().rposition(|r| r.job == a);
            if let (Some(fb), Some(la)) = (first_b, last_a) {
                assert!(la < fb, "frame {f}");
            }
        }
        assert_eq!((a_left, b_ran), (0, 2));
    }

    #[test]
    fn a_missed_job_does_not_hold_up_the_jobs_after_it() {
        let c = cost();
        let mut s = sched(SyncPolicy::SameFrame, MissPolicy::Drop, 1_000);
        // The host skipped frame 0, so job A's deadline is behind frame 1; B's
        // is frame 1 itself.
        let a = s.submit(frame(0), slices(&[0, 1]), &c).unwrap();
        let b = s.submit(frame(1), slices(&[0]), &c).unwrap();
        let p = s.plan_frame(frame(1), budget(1_000), &c);
        assert_eq!(p.runs.len(), 1);
        assert_eq!(p.runs[0].job, b);
        assert_eq!(p.missed.len(), 1);
        assert_eq!(p.missed[0].job, a);
        assert_eq!(p.delivered.len(), 1);
        assert_eq!(p.delivered[0].job, b);
    }

    #[test]
    fn equal_deadlines_go_to_the_lower_job_id() {
        let mut s = sched(SyncPolicy::SameFrame, MissPolicy::Drop, 100);
        let c = cost();
        let first = s.submit(frame(0), slices(&[0]), &c).unwrap();
        let _second = s.submit(frame(0), slices(&[0]), &c).unwrap();
        let p = s.plan_frame(frame(0), budget(100), &c);
        assert_eq!(p.runs.len(), 1);
        assert_eq!(p.runs[0].job, first);
        assert_eq!(p.delivered.len(), 1);
        assert_eq!(p.missed.len(), 1);
    }

    #[test]
    fn same_frame_fitting_job_is_delivered_in_the_same_plan() {
        let mut s = sched(SyncPolicy::SameFrame, MissPolicy::HoldPrevious, 200);
        let c = cost();
        let job = s.submit(frame(4), slices(&[0, 1]), &c).unwrap();
        let p = s.plan_frame(frame(4), budget(200), &c);
        assert_eq!(p.runs.len(), 2);
        assert_eq!(
            p.delivered,
            vec![Delivery {
                job,
                input: FrameIndex::new(4)
            }]
        );
        assert!(p.missed.is_empty());
        assert_eq!(s.pending(), 0);
    }

    #[test]
    fn same_frame_oversized_job_is_missed_in_the_same_plan_without_running() {
        // Before jobs were abandoned the first slice ran, for a result that
        // was reported missed in the same plan.
        let mut s = sched(SyncPolicy::SameFrame, MissPolicy::Drop, 150);
        let c = cost();
        let job = s.submit(frame(4), slices(&[0, 1]), &c).unwrap();
        let p = s.plan_frame(frame(4), budget(150), &c);
        assert!(p.runs.is_empty());
        assert_eq!(p.spent_estimate, Micros::ZERO);
        assert!(p.delivered.is_empty());
        assert_eq!(
            p.missed,
            vec![Missed {
                job,
                input: FrameIndex::new(4),
                due: Due::Frame(FrameIndex::new(4)),
                policy: MissPolicy::Drop
            }]
        );
        assert_eq!(s.pending(), 0);
    }

    #[test]
    fn on_beat_deadline_follows_the_grid() {
        let c = cost();
        let mut s = sched(on_beat(50), MissPolicy::Drop, 1_000);

        // Input at 250, two 100 us slices, 50 us margin: ready at 500, so the
        // beat at 1000. Nothing is delivered before it.
        let job = s.submit(at(250), slices(&[0, 1]), &c).unwrap();
        assert_eq!(s.due(job), Some(beat(1_000)));
        let p = s.plan_frame(at(250), budget(1_000), &c);
        assert_eq!(p.runs.len(), 2);
        for start in [500, 750, 999] {
            let p = s.plan_frame(at(start), budget(1_000), &c);
            assert!(p.delivered.is_empty() && p.missed.is_empty(), "at {start}");
        }
        assert_eq!(s.pending(), 1);
        let p = s.plan_frame(at(1_000), budget(1_000), &c);
        assert_eq!(
            p.delivered,
            vec![Delivery {
                job,
                input: FrameIndex::new(0)
            }]
        );
        assert_eq!(s.pending(), 0);
    }

    #[test]
    fn on_beat_skips_to_the_next_beat_when_work_does_not_fit_before_it() {
        let c = cost();
        let mut s = sched(on_beat(50), MissPolicy::Drop, 1_000);

        // Input at 900 is ready at 1050, past the beat at 1000, so it targets
        // the beat at 2000.
        let job = s.submit(at(900), slices(&[0]), &c).unwrap();
        s.plan_frame(at(900), budget(1_000), &c);
        let p = s.plan_frame(at(1_000), budget(1_000), &c);
        assert!(p.delivered.is_empty() && p.missed.is_empty());
        let p = s.plan_frame(at(1_999), budget(1_000), &c);
        assert!(p.delivered.is_empty() && p.missed.is_empty());
        let p = s.plan_frame(at(2_000), budget(1_000), &c);
        assert_eq!(p.delivered.len(), 1);
        assert_eq!(p.delivered[0].job, job);
    }

    #[test]
    fn on_beat_multi_frame_job_gets_a_beat_it_can_make() {
        // Three 100 us slices at 100 us a frame need frames 0, 1 and 2, and
        // the last finishes at 2100. Placed back to back (ready at 300) the job
        // would have targeted the beat at 1000, which frame 1 keeps with one
        // slice still to run; it targets the beat at 3000 instead.
        let c = cost();
        let mut s = sched(on_beat(0), MissPolicy::Drop, 100);
        let job = s.submit(frame(0), slices(&[0, 1, 2]), &c).unwrap();
        assert_eq!(s.due(job), Some(beat(3_000)));
        for f in 0..3 {
            let p = s.plan_frame(frame(f), budget(100), &c);
            assert_eq!(p.runs.len(), 1, "frame {f}");
            assert!(p.delivered.is_empty() && p.missed.is_empty(), "frame {f}");
        }
        let p = s.plan_frame(frame(3), budget(100), &c);
        assert_eq!(p.delivered.len(), 1);
        assert_eq!(p.delivered[0].job, job);
        assert!(p.missed.is_empty());
    }

    #[test]
    fn on_beat_work_queued_ahead_pushes_the_beat_later() {
        // Alone, one 100 us slice from 0 is ready at 100: the beat at 1000.
        // Behind a job that fills frames 0 to 2 it lands in frame 3, ready at
        // 3100: the beat at 4000.
        let c = cost();
        let mut alone = sched(on_beat(0), MissPolicy::Drop, 100);
        let solo = alone.submit(frame(0), slices(&[3]), &c).unwrap();
        assert_eq!(alone.due(solo), Some(beat(1_000)));

        let mut s = sched(on_beat(0), MissPolicy::Drop, 100);
        let a = s.submit(frame(0), slices(&[0, 1, 2]), &c).unwrap();
        let b = s.submit(frame(0), slices(&[3]), &c).unwrap();
        assert_eq!(s.due(a), Some(beat(3_000)));
        assert_eq!(s.due(b), Some(beat(4_000)));

        let mut delivered = Vec::new();
        for f in 0..5 {
            let p = s.plan_frame(frame(f), budget(100), &c);
            assert!(p.missed.is_empty(), "frame {f}");
            delivered.extend(p.delivered.iter().map(|d| (f, d.job)));
        }
        assert_eq!(delivered, vec![(3, a), (4, b)]);
    }

    #[test]
    fn on_beat_job_no_frame_can_run_is_missed_at_once() {
        // A 100 us slice fits no frame of 50 us. The deadline falls back to the
        // work back to back from the input, the beat at 1000, and the miss is
        // reported in the first plan, not at the beat.
        let c = cost();
        let mut s = sched(on_beat(0), MissPolicy::HoldPrevious, 50);
        let job = s.submit(at(0), slices(&[0]), &c).unwrap();
        assert_eq!(s.due(job), Some(beat(1_000)));
        let p = s.plan_frame(at(500), budget(50), &c);
        assert!(p.runs.is_empty());
        assert_eq!(p.missed.len(), 1);
        assert_eq!(p.missed[0].job, job);
        assert_eq!(p.missed[0].due, beat(1_000));
    }

    #[test]
    fn on_beat_miss_is_reported_at_the_beat() {
        let c = cost();
        let mut s = sched(on_beat(0), MissPolicy::HoldPrevious, 1_000);
        // The forecast says the slice runs in the input frame, so the target is
        // 1000, but the frames get 50 us. At 500 the frame after might still
        // run it; at 1000 nothing after counts, and it is missed.
        let job = s.submit(at(0), slices(&[0]), &c).unwrap();
        let p = s.plan_frame(at(500), budget(50), &c);
        assert!(p.missed.is_empty());
        let p = s.plan_frame(at(1_000), budget(50), &c);
        assert_eq!(p.missed.len(), 1);
        assert_eq!(p.missed[0].job, job);
    }

    #[test]
    fn on_beat_frame_starting_after_the_beat_still_runs_and_delivers() {
        let c = cost();
        // The forecast of 200 us places both 100 us slices in the input frame,
        // ready at 210: the beat at 1000. The input frame gets 100 us and runs
        // one. Frames start 10 us after each beat, so the frame at 1010 keeps
        // the beat, and runs the second slice before delivering.
        let mut s = sched(on_beat(0), MissPolicy::Drop, 200);
        let job = s.submit(at(10), slices(&[0, 1]), &c).unwrap();
        assert_eq!(s.due(job), Some(beat(1_000)));
        let p = s.plan_frame(at(10), budget(100), &c);
        assert_eq!(p.runs.len(), 1);
        assert!(p.missed.is_empty());
        let p = s.plan_frame(at(1_010), budget(100), &c);
        assert_eq!(p.runs.len(), 1);
        assert_eq!(p.delivered.len(), 1);
        assert_eq!(p.delivered[0].job, job);
        assert!(p.missed.is_empty());
    }

    #[test]
    fn on_beat_result_seen_after_a_skipped_beat_frame_is_missed() {
        let c = cost();
        let mut s = sched(on_beat(0), MissPolicy::HoldPrevious, 100);
        let job = s.submit(at(10), slices(&[0]), &c).unwrap();
        let p = s.plan_frame(at(10), budget(100), &c);
        assert_eq!(p.runs.len(), 1);
        // The host skipped the frame at 1010, the first after the beat.
        let p = s.plan_frame(at(2_010), budget(100), &c);
        assert!(p.delivered.is_empty());
        assert_eq!(p.missed.len(), 1);
        assert_eq!(p.missed[0].job, job);
    }

    #[test]
    fn a_tempo_change_moves_only_the_deadlines_fixed_after_it() {
        let c = cost();
        let mut s = sched(on_beat(0), MissPolicy::Drop, 1_000);
        // Fixed on the 1000 us grid.
        let a = s.submit(at(0), slices(&[0]), &c).unwrap();
        assert_eq!(s.due(a), Some(beat(1_000)));
        s.plan_frame(at(0), budget(1_000), &c);

        // From 500, beats every 300 us from 500: 500, 800, 1100, ...
        s.replace_beat_grid(
            TimePoint::new(500),
            BeatGrid::new(TimePoint::new(500), Micros::new(300)).unwrap(),
        )
        .unwrap();
        // A keeps its beat; on the new grid it would have been 500.
        assert_eq!(s.due(a), Some(beat(1_000)));
        let b = s.submit(at(600), slices(&[1]), &c).unwrap();
        assert_eq!(s.due(b), Some(beat(800)));

        let p = s.plan_frame(at(600), budget(1_000), &c);
        assert!(p.delivered.is_empty() && p.missed.is_empty());
        let p = s.plan_frame(at(800), budget(1_000), &c);
        assert_eq!(p.delivered.len(), 1);
        assert_eq!(p.delivered[0].job, b);
        let p = s.plan_frame(at(1_000), budget(1_000), &c);
        assert_eq!(p.delivered.len(), 1);
        assert_eq!(p.delivered[0].job, a);
        assert!(p.missed.is_empty());
    }

    #[test]
    fn slice_larger_than_the_budget_is_never_run_and_its_job_is_missed() {
        // Before jobs were abandoned the miss waited for frame 1, the deadline.
        let mut c = cost();
        c.record(SliceId::new(0), Micros::new(500));
        let mut s = sched(
            SyncPolicy::FixedLatency { frames: 1 },
            MissPolicy::Drop,
            499,
        );
        let job = s.submit(frame(0), slices(&[0]), &c).unwrap();

        let p0 = s.plan_frame(frame(0), budget(499), &c);
        assert!(p0.runs.is_empty());
        assert_eq!(p0.spent_estimate, Micros::ZERO);
        assert_eq!(p0.missed.len(), 1);
        assert_eq!(p0.missed[0].job, job);
        assert!(p0.delivered.is_empty());
        let p1 = s.plan_frame(frame(1), budget(499), &c);
        assert!(p1.runs.is_empty() && p1.missed.is_empty() && p1.delivered.is_empty());
    }

    #[test]
    fn job_past_its_deadline_gets_no_slices() {
        let mut s = sched(SyncPolicy::SameFrame, MissPolicy::Drop, 1_000);
        let c = cost();
        let job = s.submit(frame(0), slices(&[0]), &c).unwrap();
        // The host skipped frame 0; frame 1 is already past the deadline.
        let p = s.plan_frame(frame(1), budget(1_000), &c);
        assert!(p.runs.is_empty());
        assert_eq!(p.missed.len(), 1);
        assert_eq!(p.missed[0].job, job);
    }

    #[test]
    fn finished_job_seen_after_its_frame_is_missed_not_delivered_late() {
        let mut s = sched(
            SyncPolicy::FixedLatency { frames: 1 },
            MissPolicy::HoldPrevious,
            100,
        );
        let c = cost();
        let job = s.submit(frame(0), slices(&[0]), &c).unwrap();
        let p0 = s.plan_frame(frame(0), budget(100), &c);
        assert_eq!(p0.runs.len(), 1);
        // The host skipped frame 1, the job's frame; the source it was aligned
        // with has already been shown.
        let p2 = s.plan_frame(frame(2), budget(100), &c);
        assert!(p2.delivered.is_empty());
        assert_eq!(p2.missed.len(), 1);
        assert_eq!(p2.missed[0].job, job);
        assert_eq!(s.pending(), 0);
    }

    #[test]
    fn slices_of_a_job_run_in_order_across_frames() {
        let mut s = sched(
            SyncPolicy::FixedLatency { frames: 3 },
            MissPolicy::Drop,
            100,
        );
        let c = cost();
        s.submit(frame(0), slices(&[7, 8, 9]), &c).unwrap();
        let mut seen = Vec::new();
        for f in 0..3 {
            let p = s.plan_frame(frame(f), budget(100), &c);
            seen.extend(p.runs.iter().map(|r| (r.position, r.slice.get())));
        }
        assert_eq!(seen, vec![(0, 7), (1, 8), (2, 9)]);
        let p = s.plan_frame(frame(3), budget(100), &c);
        assert_eq!(p.delivered.len(), 1);
    }

    /// A session at 60 Hz for 600 frames, on beats at 120 bpm that change to
    /// 150 bpm at frame 300. Every fourth frame submits a job of 16, 24 or 30
    /// slices of 1 ms, two to four frames of the expected 8 ms. Every tenth
    /// frame a slow render leaves 5 ms, and frames 450 to 469 leave 2 ms, a
    /// third of a second the host did not forecast. Nothing is delivered off
    /// its beat, no miss is reported after its deadline, and the figures are
    /// exact because the scheduler is deterministic.
    #[test]
    fn simulated_session_delivers_on_its_beats_through_budget_dips_and_a_tempo_change() {
        const FRAME_US: u64 = 16_667;
        const FRAMES: u64 = 600;
        const EXPECTED: u64 = 8_000;
        const CHANGE: u64 = 300;
        const SLOW: std::ops::Range<u64> = 450..470;
        let slow = BeatGrid::new(TimePoint::new(0), Micros::new(500_000)).unwrap();
        let change_at = TimePoint::new(CHANGE * FRAME_US);
        let fast = BeatGrid::new(change_at, Micros::new(400_000)).unwrap();
        let c = CostModel::new(
            Micros::new(1_000),
            Estimator::MeanPlusSigmas {
                alpha: 1.0,
                sigmas: 0.0,
            },
        )
        .unwrap();
        let mut s = Scheduler::new(
            SyncPolicy::OnBeat {
                margin: Micros::new(2_000),
                grid: slow,
            },
            MissPolicy::HoldPrevious,
            budget(EXPECTED),
        );

        let sizes = [16, 24, 30];
        // Each job's deadline as fixed at submission, and whether it was
        // submitted after the tempo change.
        let mut fixed: BTreeMap<JobId, (TimePoint, bool)> = BTreeMap::new();
        let mut ran: BTreeMap<JobId, u64> = BTreeMap::new();
        let (mut delivered, mut missed) = (0_u64, 0_u64);
        // Misses reported in a frame before the one that keeps their beat, and
        // the slices run for jobs that were then missed.
        let (mut early, mut wasted) = (0_u64, 0_u64);
        let mut missed_in = Vec::new();
        for i in 0..FRAMES {
            let f = FrameInfo {
                index: FrameIndex::new(i),
                start: TimePoint::new(i * FRAME_US),
                period: Micros::new(FRAME_US),
            };
            if i == CHANGE {
                s.replace_beat_grid(change_at, fast).unwrap();
            }
            if i % 4 == 0 {
                let size = sizes[usize::try_from(i / 4).unwrap() % sizes.len()];
                let job = s
                    .submit(f, (0..size).map(SliceId::new).collect(), &c)
                    .unwrap();
                let Some(Due::Beat(due)) = s.due(job) else {
                    panic!("an on-beat job has a beat");
                };
                fixed.insert(job, (due, i >= CHANGE));
            }
            let b = if SLOW.contains(&i) {
                2_000
            } else if i % 10 == 9 {
                5_000
            } else {
                EXPECTED
            };
            let p = s.plan_frame(f, budget(b), &c);
            assert!(p.spent_estimate <= p.budget.get(), "frame {i}");
            for r in &p.runs {
                *ran.entry(r.job).or_default() += 1;
            }
            for d in &p.delivered {
                let (due, _) = fixed[&d.job];
                // The first frame starting at or after the beat, and no later.
                assert!(f.start >= due, "frame {i}: delivered before its beat");
                assert!(
                    f.start.saturating_since(due) < f.period,
                    "frame {i}: delivered late"
                );
                delivered += 1;
            }
            for m in &p.missed {
                let (due, _) = fixed[&m.job];
                assert_eq!(m.due, Due::Beat(due), "frame {i}");
                // Reported no later than the frame that keeps the beat.
                assert!(
                    f.start < due || f.start.saturating_since(due) < f.period,
                    "frame {i}: miss reported late"
                );
                if f.start < due {
                    early += 1;
                }
                wasted += ran.get(&m.job).copied().unwrap_or(0);
                missed_in.push(i);
                missed += 1;
            }
        }

        for (job, &(due, after_change)) in &fixed {
            let on = |g_origin: u64, g_beat: u64| {
                due.get() >= g_origin && (due.get() - g_origin).is_multiple_of(g_beat)
            };
            if after_change {
                assert!(on(change_at.get(), 400_000), "job {}", job.get());
            } else {
                assert!(on(0, 500_000), "job {}", job.get());
            }
        }
        let submitted = fixed.len() as u64;
        let pending = s.pending() as u64;
        assert_eq!(submitted, delivered + missed + pending);
        let rate = delivered as f64 / (delivered + missed) as f64;
        assert!(rate >= 0.95, "delivery rate {rate}");
        // The three misses are reported in the slow patch; two before their
        // beat. Their 25 slices ran while the projection still had them
        // finishing in time.
        assert!(missed_in.iter().all(|i| SLOW.contains(i)), "{missed_in:?}");
        assert_eq!(
            (submitted, delivered, missed, pending, early, wasted),
            (150, 144, 3, 3, 2, 25)
        );
    }
}
