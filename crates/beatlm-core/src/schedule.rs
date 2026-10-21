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

use thiserror::Error;

use crate::budget::Budget;
use crate::cost::{CostModel, SliceId};
use crate::sync::{MissPolicy, SyncPolicy};
use crate::time::{BeatGrid, FrameIndex, FrameInfo, Micros, TimePoint};

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

/// A deadline that arrived before the job was complete.
///
/// It is reported so that the miss is never silent; `policy` tells the host
/// what to show instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Missed {
    /// The job that did not finish in time.
    pub job: JobId,
    /// The frame whose input has no result.
    pub input: FrameIndex,
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
    /// Inputs whose deadline arrived without a complete result.
    pub missed: Vec<Missed>,
    /// The summed estimate of `runs`; never more than `budget`.
    pub spent_estimate: Micros,
    /// The budget the plan was made against.
    pub budget: Budget,
}

/// A rejected scheduler configuration or job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ScheduleError {
    /// A job with no slices has nothing to run and nothing to deliver.
    #[error("a job needs at least one slice")]
    EmptyJob,
    /// `OnBeat` delivery has no deadline without a beat grid.
    #[error("the OnBeat policy needs a beat grid")]
    MissingBeatGrid,
}

/// When a job's result is due: a frame, or an instant.
///
/// A scheduler has one policy, so all its deadlines are the same kind and the
/// derived ordering is the ordering of due dates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Deadline {
    Frame(FrameIndex),
    Time(TimePoint),
}

impl Deadline {
    /// The deadline arrives with this frame, so a decision about the job is
    /// due now.
    fn reached(self, frame: FrameInfo) -> bool {
        match self {
            Self::Frame(d) => frame.index >= d,
            Self::Time(d) => frame.start >= d,
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
            Self::Time(d) => {
                frame.start == d
                    || (frame.start > d && frame.start.saturating_since(d) < frame.period)
            }
        }
    }

    /// The deadline arrived in an earlier frame than this one, so this frame
    /// cannot keep it and slices scheduled now are wasted. The frame that keeps
    /// a deadline still gets slices: they are submitted before its render pass.
    fn passed(self, frame: FrameInfo) -> bool {
        self.reached(frame) && !self.on_time(frame)
    }
}

#[derive(Debug, Clone)]
struct Job {
    id: JobId,
    input: FrameIndex,
    slices: Vec<SliceId>,
    /// How many leading slices have been handed out.
    scheduled: usize,
    deadline: Deadline,
}

impl Job {
    fn is_complete(&self) -> bool {
        self.scheduled >= self.slices.len()
    }
}

/// Decides, frame by frame, which slices run and which results are delivered.
///
/// It owns the promise made by a [`SyncPolicy`]: it fixes each job's deadline
/// at submission, spends the frame budget on jobs in submission order (see the
/// [module documentation](self#jobs-run-as-one-sequence)), and reports every
/// deadline it could not keep with the configured [`MissPolicy`].
#[derive(Debug, Clone)]
pub struct Scheduler {
    policy: SyncPolicy,
    miss: MissPolicy,
    beats: Option<BeatGrid>,
    next_job: u64,
    pending: Vec<Job>,
}

impl Scheduler {
    /// A scheduler that keeps `policy`, reports misses with `miss`, and reads
    /// beat times from `beats`.
    ///
    /// # Errors
    ///
    /// [`ScheduleError::MissingBeatGrid`] when `policy` is `OnBeat` and `beats`
    /// is `None`.
    pub fn new(
        policy: SyncPolicy,
        miss: MissPolicy,
        beats: Option<BeatGrid>,
    ) -> Result<Self, ScheduleError> {
        if matches!(policy, SyncPolicy::OnBeat { .. }) && beats.is_none() {
            return Err(ScheduleError::MissingBeatGrid);
        }
        Ok(Self {
            policy,
            miss,
            beats,
            next_job: 0,
            pending: Vec::new(),
        })
    }

    /// Registers the input of frame `input` as a job that runs `slices` in
    /// order, and fixes its deadline.
    ///
    /// The deadline is `input + frames` for `FixedLatency`, the input frame for
    /// `SameFrame`, and for `OnBeat` the first beat at or after the input's
    /// start plus the summed estimates of `slices` plus the margin.
    ///
    /// # Errors
    ///
    /// [`ScheduleError::EmptyJob`] when `slices` is empty;
    /// [`ScheduleError::MissingBeatGrid`] when an `OnBeat` scheduler has no
    /// grid.
    pub fn submit(
        &mut self,
        input: FrameInfo,
        slices: Vec<SliceId>,
        cost: &CostModel,
    ) -> Result<JobId, ScheduleError> {
        if slices.is_empty() {
            return Err(ScheduleError::EmptyJob);
        }
        let deadline = match self.policy {
            SyncPolicy::FixedLatency { frames } => Deadline::Frame(input.index.offset(frames)),
            SyncPolicy::SameFrame => Deadline::Frame(input.index),
            SyncPolicy::OnBeat { margin } => {
                let grid = self.beats.ok_or(ScheduleError::MissingBeatGrid)?;
                let work = slices
                    .iter()
                    .fold(Micros::ZERO, |sum, s| sum.saturating_add(cost.estimate(*s)));
                let ready = input.start.saturating_add(work).saturating_add(margin);
                Deadline::Time(grid.next_beat_at_or_after(ready))
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
        });
        Ok(id)
    }

    /// Plans `frame`: hands out slices within `budget`, then delivers or
    /// reports the jobs whose deadline arrives.
    ///
    /// Pending jobs are served in submission ([`JobId`]) order and never
    /// interleave, because they share one memory plan: a job's slices go out in
    /// order while they fit, and when its next slice does not fit the remaining
    /// budget no slice of any later job is handed out this frame, however
    /// small. The summed estimate of the runs never exceeds `budget`. A job
    /// whose deadline is already behind `frame` gets no slices, does not hold
    /// up the jobs after it, and is reported missed by this plan.
    ///
    /// A job is complete once all its slices were handed out in this or an
    /// earlier frame. A job whose deadline arrives with `frame` is delivered if
    /// it is complete and `frame` keeps its promise, and reported as missed
    /// otherwise; either way it leaves. Only the deadline frame itself — or,
    /// for a beat, the first frame starting at or after it — keeps the
    /// promise, so a job the host first plans after a skipped deadline frame
    /// is missed even when finished. A
    /// complete job whose deadline has not arrived stays, which is what keeps
    /// latency fixed when work finishes early.
    pub fn plan_frame(&mut self, frame: FrameInfo, budget: Budget, cost: &CostModel) -> FramePlan {
        // `pending` is in submission order: jobs are only ever pushed, and
        // removal keeps the order of the rest.
        let mut runs = Vec::new();
        let mut spent = Micros::ZERO;
        'jobs: for job in &mut self.pending {
            if job.deadline.passed(frame) {
                continue;
            }
            while let Some(&slice) = job.slices.get(job.scheduled) {
                let after = spent.saturating_add(cost.estimate(slice));
                if after > budget.get() {
                    // No later job may run before this one has finished.
                    break 'jobs;
                }
                runs.push(SliceRun {
                    job: job.id,
                    slice,
                    position: job.scheduled,
                });
                spent = after;
                job.scheduled += 1;
            }
        }

        let mut delivered = Vec::new();
        let mut missed = Vec::new();
        for job in self
            .pending
            .extract_if(.., |job| job.deadline.reached(frame))
        {
            if job.is_complete() && job.deadline.on_time(frame) {
                delivered.push(Delivery {
                    job: job.id,
                    input: job.input,
                });
            } else {
                missed.push(Missed {
                    job: job.id,
                    input: job.input,
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
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

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
        CostModel::new(Micros::new(100), 1.0, 0.0).unwrap()
    }

    fn slices(ids: &[u32]) -> Vec<SliceId> {
        ids.iter().copied().map(SliceId::new).collect()
    }

    fn sched(policy: SyncPolicy, miss: MissPolicy) -> Scheduler {
        Scheduler::new(policy, miss, None).unwrap()
    }

    fn grid() -> BeatGrid {
        BeatGrid::new(TimePoint::new(0), Micros::new(1_000)).unwrap()
    }

    #[test]
    fn on_beat_without_grid_is_rejected() {
        let r = Scheduler::new(
            SyncPolicy::OnBeat {
                margin: Micros::ZERO,
            },
            MissPolicy::Drop,
            None,
        );
        assert_eq!(r.unwrap_err(), ScheduleError::MissingBeatGrid);
    }

    #[test]
    fn empty_job_is_rejected() {
        let mut s = sched(SyncPolicy::SameFrame, MissPolicy::Drop);
        let r = s.submit(frame(0), Vec::new(), &cost());
        assert_eq!(r.unwrap_err(), ScheduleError::EmptyJob);
        assert_eq!(s.pending(), 0);
    }

    #[test]
    fn job_ids_count_from_zero() {
        let mut s = sched(SyncPolicy::SameFrame, MissPolicy::Drop);
        let a = s.submit(frame(0), slices(&[0]), &cost()).unwrap();
        let b = s.submit(frame(0), slices(&[0]), &cost()).unwrap();
        assert_eq!((a.get(), b.get()), (0, 1));
    }

    #[test]
    fn fixed_latency_delivers_exactly_on_its_frame() {
        let mut s = sched(SyncPolicy::FixedLatency { frames: 2 }, MissPolicy::Drop);
        let c = cost();
        let job = s.submit(frame(10), slices(&[0]), &c).unwrap();

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
    }

    #[test]
    fn fixed_latency_miss_is_reported_with_the_policy() {
        let mut s = sched(
            SyncPolicy::FixedLatency { frames: 2 },
            MissPolicy::HoldPrevious,
        );
        let c = cost();
        // Four slices of 100 us against a 100 us budget: three frames hand
        // out three slices.
        let job = s.submit(frame(0), slices(&[0, 1, 2, 3]), &c).unwrap();
        let mut ran = 0;
        for f in 0..2 {
            let p = s.plan_frame(frame(f), budget(100), &c);
            ran += p.runs.len();
            assert!(p.delivered.is_empty() && p.missed.is_empty());
        }
        let p = s.plan_frame(frame(2), budget(100), &c);
        ran += p.runs.len();
        assert_eq!(ran, 3);
        assert!(p.delivered.is_empty());
        assert_eq!(
            p.missed,
            vec![Missed {
                job,
                input: FrameIndex::new(0),
                policy: MissPolicy::HoldPrevious
            }]
        );
        assert_eq!(s.pending(), 0);

        // The job is gone: nothing more runs for it.
        let later = s.plan_frame(frame(3), budget(100), &c);
        assert!(later.runs.is_empty() && later.missed.is_empty());
    }

    #[test]
    fn miss_policy_is_carried_through() {
        let mut s = sched(SyncPolicy::SameFrame, MissPolicy::Drop);
        let c = cost();
        s.submit(frame(0), slices(&[0]), &c).unwrap();
        let p = s.plan_frame(frame(0), budget(50), &c);
        assert_eq!(p.missed[0].policy, MissPolicy::Drop);
    }

    #[test]
    fn budget_is_never_exceeded_and_leftover_is_not_given_to_a_later_job() {
        let mut c = cost();
        c.record(SliceId::new(2), Micros::new(40));
        let mut s = sched(SyncPolicy::FixedLatency { frames: 5 }, MissPolicy::Drop);
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
        let mut s = sched(SyncPolicy::FixedLatency { frames: 20 }, MissPolicy::Drop);
        for i in 0..4 {
            s.submit(frame(i), slices(&[0, 1, 2]), &c).unwrap();
        }
        for f in 0..25 {
            let p = s.plan_frame(frame(f), budget(250), &c);
            assert!(p.spent_estimate <= Micros::new(250), "frame {f}");
            assert!(p.runs.len() <= 2, "frame {f}");
        }
        assert_eq!(s.pending(), 0);
    }

    #[test]
    fn oldest_job_gets_slices_first_even_with_the_later_deadline() {
        // Inputs submitted out of frame order: the older job has the later
        // deadline, and still runs first.
        let mut s = sched(SyncPolicy::FixedLatency { frames: 5 }, MissPolicy::Drop);
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
        let mut s = sched(SyncPolicy::FixedLatency { frames: 6 }, MissPolicy::Drop);
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
        let mut s = sched(SyncPolicy::SameFrame, MissPolicy::Drop);
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
        let mut s = sched(SyncPolicy::SameFrame, MissPolicy::Drop);
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
        let mut s = sched(SyncPolicy::SameFrame, MissPolicy::HoldPrevious);
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
    fn same_frame_oversized_job_is_missed_in_the_same_plan() {
        let mut s = sched(SyncPolicy::SameFrame, MissPolicy::Drop);
        let c = cost();
        let job = s.submit(frame(4), slices(&[0, 1]), &c).unwrap();
        let p = s.plan_frame(frame(4), budget(150), &c);
        assert_eq!(p.runs.len(), 1);
        assert!(p.delivered.is_empty());
        assert_eq!(
            p.missed,
            vec![Missed {
                job,
                input: FrameIndex::new(4),
                policy: MissPolicy::Drop
            }]
        );
        assert_eq!(s.pending(), 0);
    }

    #[test]
    fn on_beat_deadline_follows_the_grid() {
        let c = cost();
        let policy = SyncPolicy::OnBeat {
            margin: Micros::new(50),
        };
        let mut s = Scheduler::new(policy, MissPolicy::Drop, Some(grid())).unwrap();

        // Input at 250, two 100 us slices, 50 us margin: ready at 500, so the
        // beat at 1000. Nothing is delivered before it.
        let job = s.submit(at(250), slices(&[0, 1]), &c).unwrap();
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
        let policy = SyncPolicy::OnBeat {
            margin: Micros::new(50),
        };
        let mut s = Scheduler::new(policy, MissPolicy::Drop, Some(grid())).unwrap();

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
    fn on_beat_miss_is_reported_at_the_beat() {
        let c = cost();
        let policy = SyncPolicy::OnBeat {
            margin: Micros::ZERO,
        };
        let mut s = Scheduler::new(policy, MissPolicy::HoldPrevious, Some(grid())).unwrap();
        // Estimate says 100 us, so the target is 1000, but the budget never
        // lets the slice run.
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
        let policy = SyncPolicy::OnBeat {
            margin: Micros::ZERO,
        };
        let mut s = Scheduler::new(policy, MissPolicy::Drop, Some(grid())).unwrap();
        // Two 100 us slices, ready at 200: the beat at 1000. Frames start 10 us
        // after each beat, so the frame at 1010 is the one that keeps it.
        let job = s.submit(at(10), slices(&[0, 1]), &c).unwrap();
        let p = s.plan_frame(at(10), budget(100), &c);
        assert_eq!(p.runs.len(), 1);
        let p = s.plan_frame(at(1_010), budget(100), &c);
        assert_eq!(p.runs.len(), 1);
        assert_eq!(p.delivered.len(), 1);
        assert_eq!(p.delivered[0].job, job);
        assert!(p.missed.is_empty());
    }

    #[test]
    fn on_beat_result_seen_after_a_skipped_beat_frame_is_missed() {
        let c = cost();
        let policy = SyncPolicy::OnBeat {
            margin: Micros::ZERO,
        };
        let mut s = Scheduler::new(policy, MissPolicy::HoldPrevious, Some(grid())).unwrap();
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
    fn slice_larger_than_the_budget_is_never_run_and_its_job_is_missed() {
        let mut c = cost();
        c.record(SliceId::new(0), Micros::new(500));
        let mut s = sched(SyncPolicy::FixedLatency { frames: 1 }, MissPolicy::Drop);
        let job = s.submit(frame(0), slices(&[0]), &c).unwrap();

        let p0 = s.plan_frame(frame(0), budget(499), &c);
        assert!(p0.runs.is_empty());
        assert_eq!(p0.spent_estimate, Micros::ZERO);
        let p1 = s.plan_frame(frame(1), budget(499), &c);
        assert!(p1.runs.is_empty());
        assert_eq!(p1.missed.len(), 1);
        assert_eq!(p1.missed[0].job, job);
        assert!(p1.delivered.is_empty());
    }

    #[test]
    fn job_past_its_deadline_gets_no_slices() {
        let mut s = sched(SyncPolicy::SameFrame, MissPolicy::Drop);
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
        let mut s = sched(SyncPolicy::FixedLatency { frames: 3 }, MissPolicy::Drop);
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
}
