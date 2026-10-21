//! Why jobs were missed, and what the estimates they were judged by were
//! made of: the report's diagnosis lines.
//!
//! The scheduler reports a miss without its reason. A host can tell three
//! apart: a job abandoned before any of its slices ran, one abandoned after
//! some ran, and one missed at its deadline frame. For a job abandoned before
//! it started, the host also knows everything the projection read: this
//! frame's budget, the expected budget of the frames after it, the estimate
//! of each of its slices, and the frames left to its deadline. [`project`]
//! restates the projection's packing rule (`beatlm_core::schedule`, "Projecting
//! the queue onto frames to come"), so the report can say which side of the
//! rule a job fell on; the scheduler's own copy is private, and this one is
//! only read for the report. It projects from an empty frame, which is exact
//! for the depth model: with the `--submit free` admission no job ahead of an
//! unstarted one has slices left.
//!
//! The estimates are the cost model's, which change only when a slice is
//! measured, and a slice is measured only when a job runs it. Each timed
//! frame's first compute pass is recorded apart from the later ones
//! ([`Diagnosis::passes`]), to see whether the frame's first pass measures
//! long: on an Apple adapter a pass boundary can include time the pass waited
//! behind earlier work (`beatlm_core::cost`'s table, natively).
//!
//! # What it found in Chrome, before #41
//!
//! Chrome 154 (WebGPU) on an Apple M1 Pro driving a 120 Hz display (period
//! 8 300 µs), 2026-10-22, Depth Anything V2 Small,
//! `?model=depth&side=266&frames=3000&latency_ms=200` (k = 25):
//!
//! - 31 results delivered of 2 592 inputs submitted. Of the misses, 2 543 were
//!   jobs abandoned before their first slice, 18 after some ran, none at a
//!   deadline. Jobs started 49 times, from frame 0 to frame 2 964; the first
//!   job abandoned before it started was in frame 341.
//! - Those jobs were refused on their total: the projection placed the last
//!   slice in frame 26 at the median (28 at most) with 25 frames left, the
//!   expected budget 4 334 µs at the median and the largest slice estimate
//!   2 949 µs, so every slice fitted a frame (`unstarted_slice_fits_no_frame`
//!   0). The job's estimate, the sum of its slices', was 101 722 µs at the
//!   median, where a whole run of the model measured 28 716 µs in calibration
//!   (`gpu_run_us` p50, `calibrated_job_us` 28 243).
//! - The estimate rose during the loop: 34 236 µs when the loop began,
//!   101 984 µs when it ended (`job_estimate_us`).
//! - Timings came back 1 frame later at the median, 3 at most; a frame's first
//!   pass measured 1.11 times the same slice's later passes (median over
//!   slices), so the first pass is not where the estimates grew.
//! - The render pass measured 1 901 µs (p50) in frames without slices and
//!   1 573 µs in frames with them.
//!
//! Two explanations were tested and refuted by these lines: a render pass
//! whose measurement carried the frame's slices, which would have shrunk the
//! budget (`crate::render_cost`), and a single slice estimated above the
//! expected budget, which no plan hands out whatever the latency (the test
//! `a_slice_estimated_over_every_expected_frame_keeps_jobs_from_finishing`
//! shows that rule). What remained was why the slices measure slower in the
//! loop than in calibration, so that the estimates climb until whole jobs no
//! longer fit their deadline, and that the estimates of slices no job runs any
//! more are never measured again. One reading, unverified, is the GPU's clock:
//! most frames of the loop leave the GPU idle, and the render pass, the same
//! work every frame, measured faster in the frames that also ran slices.
//!
//! These runs were before #41, and every figure above is of the scheduler
//! then, which abandoned every job projected past its deadline. The second
//! half, the feedback between the estimates and the jobs they let run, was the
//! core's to settle, and #41 settled it: a job with no job waiting behind it,
//! which with the `--submit free` admission is every job, is no longer
//! abandoned but kept to its deadline, so its slices run and are measured
//! (`beatlm_core::schedule`, "A job that cannot make its deadline"). Under it
//! a miss of this page is reported at its deadline, not before its first
//! slice. The test
//! `a_job_estimated_past_its_frames_runs_alone_and_its_estimates_come_down`
//! shows the recovery on the scheduler alone. Why the slices measure slower
//! in the loop stays open.

use std::collections::BTreeMap;

use beatlm_core::cost::SliceId;
use beatlm_core::time::Micros;

/// Why a job was missed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    /// Abandoned by the projection before any of its slices ran.
    Unstarted,
    /// Abandoned by the projection after some of its slices ran.
    Started,
    /// Its deadline frame came without a complete result.
    AtDeadline,
}

/// The reason of a miss reported in frame `frame` for a job due in frame
/// `due` that had `dispatched` slices run.
#[must_use]
pub fn reason(dispatched: u64, due: u64, frame: u64) -> Reason {
    if frame >= due {
        Reason::AtDeadline
    } else if dispatched == 0 {
        Reason::Unstarted
    } else {
        Reason::Started
    }
}

/// The frame, counted from the planned one as 0, in which slices of
/// `estimates`, laid in order onto the planned frame's budget `now` and then
/// frames of the expected budget `later`, would finish; `None` when one does
/// not fit even an empty later frame. The rule is the projection's: a frame
/// takes slices while they fit, and the first that does not opens the next.
#[must_use]
pub fn project(estimates: &[Micros], now: Micros, later: Micros) -> Option<u64> {
    let (mut frame, mut spent) = (0_u64, 0_u64);
    for estimate in estimates {
        loop {
            let budget = if frame == 0 { now } else { later };
            let after = spent.saturating_add(estimate.get());
            if after <= budget.get() {
                spent = after;
                break;
            }
            if frame > 0 && spent == 0 {
                return None;
            }
            frame += 1;
            spent = 0;
        }
    }
    Some(frame)
}

/// What the projection read for one job abandoned before it started.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Unstarted {
    /// The frame it was abandoned in.
    pub frame: u64,
    /// This frame's budget, in microseconds.
    pub now_us: u64,
    /// The expected budget of the frames after it, in microseconds.
    pub expected_us: u64,
    /// The largest estimate of its slices, in microseconds.
    pub largest_us: u64,
    /// The sum of its slices' estimates, in microseconds.
    pub sum_us: u64,
    /// Frames from this one to its deadline: the last frame the projection
    /// may finish it in, counted from this one as 0.
    pub frames_left: u64,
    /// The frame [`project`] finishes it in; `None` when a slice fits no
    /// expected frame.
    pub projected_end: Option<u64>,
}

/// Measurements of one kind of pass for one slice: sum and count.
type Sums = (u64, u64);

/// The diagnosis of a run.
#[derive(Debug, Clone, Default)]
pub struct Diagnosis {
    /// Misses by [`Reason`]: unstarted, started, at the deadline.
    pub unstarted: u64,
    /// See [`Diagnosis::unstarted`].
    pub started: u64,
    /// See [`Diagnosis::unstarted`].
    pub at_deadline: u64,
    /// What the projection read for each job abandoned before it started.
    pub abandoned: Vec<Unstarted>,
    /// The frames in which a job's first slice ran.
    pub starts: Vec<u64>,
    /// How many frames after its own each timed frame's timings were
    /// collected.
    pub lag_frames: Vec<u64>,
    /// Each timed frame's first compute pass, in microseconds.
    pub first_pass_us: Vec<u64>,
    /// Every later compute pass, in microseconds.
    pub later_pass_us: Vec<u64>,
    /// Per slice: its measurements as a frame's first pass, and as a later
    /// one.
    pub passes: BTreeMap<SliceId, (Sums, Sums)>,
    /// The sum of every slice's estimate when the loop began, and when it
    /// ended, in microseconds: the job as the projection saw it.
    pub job_estimate_us: (u64, u64),
}

impl Diagnosis {
    /// Counts a miss of `reason`.
    pub fn missed(&mut self, reason: Reason) {
        match reason {
            Reason::Unstarted => self.unstarted += 1,
            Reason::Started => self.started += 1,
            Reason::AtDeadline => self.at_deadline += 1,
        }
    }

    /// Takes one timed frame's compute passes, in the order they ran.
    pub fn passes(&mut self, compute: &[(SliceId, Micros)]) {
        for (index, &(slice, duration)) in compute.iter().enumerate() {
            let entry = self.passes.entry(slice).or_default();
            let (sums, list) = if index == 0 {
                (&mut entry.0, &mut self.first_pass_us)
            } else {
                (&mut entry.1, &mut self.later_pass_us)
            };
            sums.0 = sums.0.saturating_add(duration.get());
            sums.1 += 1;
            list.push(duration.get());
        }
    }

    /// Over the slices measured both as a frame's first pass and as a later
    /// one, the median of their first-pass mean over their later-pass mean,
    /// with how many slices it is over; `None` when no slice was measured
    /// both ways.
    #[must_use]
    pub fn first_over_later(&self) -> Option<(f64, usize)> {
        let mut ratios: Vec<f64> = self
            .passes
            .values()
            .filter_map(|&((first, n_first), (later, n_later))| {
                if n_first == 0 || n_later == 0 || later == 0 {
                    return None;
                }
                #[allow(clippy::cast_precision_loss)]
                let ratio = (first as f64 / n_first as f64) / (later as f64 / n_later as f64);
                Some(ratio)
            })
            .collect();
        ratios.sort_by(f64::total_cmp);
        let count = ratios.len();
        ratios.get(count / 2).map(|&ratio| (ratio, count))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn us(values: &[u64]) -> Vec<Micros> {
        values.iter().copied().map(Micros::new).collect()
    }

    #[test]
    fn a_miss_is_unstarted_started_or_at_its_deadline() {
        assert_eq!(reason(0, 30, 5), Reason::Unstarted);
        assert_eq!(reason(3, 30, 5), Reason::Started);
        assert_eq!(reason(0, 30, 30), Reason::AtDeadline);
        assert_eq!(reason(160, 30, 31), Reason::AtDeadline);
    }

    #[test]
    fn the_projection_packs_frames_and_refuses_a_slice_no_frame_holds() {
        let later = Micros::new(4_400);
        assert_eq!(
            project(&us(&[1_000, 1_000]), Micros::new(2_000), later),
            Some(0)
        );
        assert_eq!(
            project(&us(&[1_000, 1_500]), Micros::new(2_000), later),
            Some(1)
        );
        // A frame's leftover is not used by a slice that does not fit it.
        assert_eq!(
            project(&us(&[3_000, 3_000, 3_000]), Micros::ZERO, later),
            Some(3)
        );
        // A slice larger than this frame's budget waits for the next; one
        // larger than any expected frame finishes in none.
        assert_eq!(project(&us(&[4_400]), Micros::ZERO, later), Some(1));
        assert_eq!(
            project(&us(&[100, 4_401]), Micros::new(8_000), later),
            Some(0)
        );
        assert_eq!(project(&us(&[4_401]), Micros::ZERO, later), None);
        assert_eq!(project(&[], Micros::ZERO, later), Some(0));
    }

    /// What a simulated run did.
    #[derive(Debug)]
    struct Run {
        delivered: u64,
        diagnosis: Diagnosis,
        /// The last frame any slice ran in.
        last_slice_frame: Option<u64>,
        /// The sum of the slices' estimates at the start and at the end.
        job_estimate_us: (u64, u64),
    }

    /// A loop on the scheduler alone, with the page's budget
    /// (`crate::render_cost`), its `--submit free` admission and the core's
    /// default estimator: 160 slices of a 29 ms job, each seeded with 32
    /// measurements of `seed_scale` times what it takes, as calibration seeds
    /// the model. The render is 1 900 µs, measured alone, the margin 2 ms and
    /// the latency 25 frames. One pass in `1 / every` measures `slow_us`
    /// longer than it ran, at pseudo-random. It models the rules, not the
    /// browser: what makes the page's estimates climb is not in it.
    fn simulate(period_us: u64, every: u64, slow_us: u64, seed_scale: u64, frames: u64) -> Run {
        use crate::depth::still_busy;
        use crate::render_cost::RenderCost;
        use beatlm_core::cost::{CostModel, Estimator};
        use beatlm_core::schedule::{Due, Scheduler};
        use beatlm_core::sync::{MissPolicy, SyncPolicy};
        use beatlm_core::time::{FrameIndex, FrameInfo, TimePoint};

        let (period, margin) = (Micros::new(period_us), Micros::new(2_000));
        let costs: Vec<u64> = (0..160).map(|i| [60, 120, 180, 240, 300][i % 5]).collect();
        let slices: Vec<SliceId> = (0..160).map(SliceId::new).collect();
        let mut cost = CostModel::new(Micros::new(2_000), Estimator::default()).unwrap();
        for _ in 0..32 {
            for (slice, &us) in slices.iter().zip(&costs) {
                cost.record(*slice, Micros::new(us * seed_scale));
            }
        }
        let estimate =
            |cost: &CostModel| -> u64 { slices.iter().map(|&s| cost.estimate(s).get()).sum() };
        let start_estimate = estimate(&cost);
        let mut scheduler = Scheduler::new(
            SyncPolicy::FixedLatency { frames: 25 },
            MissPolicy::HoldPrevious,
            render_budget(period, margin),
        );
        let mut render = RenderCost::default();
        let (mut busy, mut seed) = (None, 1_u64);
        let mut dispatched: BTreeMap<u64, u64> = BTreeMap::new();
        let mut pending: Vec<(SliceId, Micros)> = Vec::new();
        let mut run = Run {
            delivered: 0,
            diagnosis: Diagnosis::default(),
            last_slice_frame: None,
            job_estimate_us: (start_estimate, 0),
        };
        for index in 0..frames {
            // Last frame's measurements arrive now, as readbacks do.
            for &(slice, measured) in &pending {
                cost.record(slice, measured);
            }
            pending.clear();
            render.record(Micros::new(1_900));
            let frame = FrameInfo {
                index: FrameIndex::new(index),
                start: TimePoint::new(index * period_us),
                period,
            };
            let expected = render.expected(period, margin);
            scheduler.set_expected_budget(expected);
            let job = if busy.is_none() {
                Some(scheduler.submit(frame, slices.clone(), &cost).unwrap())
            } else {
                None
            };
            let now = render.budget(period, margin);
            let plan = scheduler.plan_frame(frame, now, &cost);
            for missed in &plan.missed {
                let Due::Frame(due) = missed.due else {
                    panic!("a fixed latency is due on a frame");
                };
                let sent = dispatched.get(&missed.job.get()).copied().unwrap_or(0);
                run.diagnosis.missed(reason(sent, due.get(), index));
            }
            for slice_run in &plan.runs {
                *dispatched.entry(slice_run.job.get()).or_default() += 1;
                seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let slow = if (seed >> 33) % every == 0 {
                    slow_us
                } else {
                    0
                };
                let ran = costs[slice_run.slice.get() as usize];
                pending.push((slice_run.slice, Micros::new(ran + slow)));
                run.last_slice_frame = Some(index);
            }
            busy = still_busy(busy, job.map(|job| (job, slices.len())), &plan);
            run.delivered += plan.delivered.len() as u64;
        }
        run.job_estimate_us.1 = estimate(&cost);
        run
    }

    fn render_budget(period: Micros, margin: Micros) -> beatlm_core::budget::Budget {
        beatlm_core::budget::Budget::for_frame(period, crate::render_cost::RENDER_PRIOR, margin)
    }

    /// A property of the core's rules, not the page's finding: a slice whose
    /// estimate exceeds every frame's budget is handed out by no plan, so it
    /// is never measured again and no job that has it finishes. Rare slow
    /// measurements can put a slice there at 120 Hz, where the expected
    /// budget is 4 400 µs, and not at 60 Hz, where it is 12 767 µs. The Chrome
    /// runs refuted this as the page's collapse
    /// (`unstarted_slice_fits_no_frame` 0; the module documentation).
    ///
    /// Before #41 the projection refused every such job at arrival, so after
    /// the first frames no slice ran at all (more than 1 000 jobs abandoned
    /// before they started, the last slice before frame 1 500). Since, a job
    /// with no job behind it is kept: every job starts and runs its slices up
    /// to that one, and is missed at its deadline, so the lock that is left is
    /// that slice's alone.
    #[test]
    fn a_slice_estimated_over_every_expected_frame_keeps_jobs_from_finishing() {
        // One pass in 20 measuring 5 ms longer than it ran.
        let at_120 = simulate(8_300, 20, 5_000, 1, 3_000);
        let last = at_120.last_slice_frame.expect("jobs ran");
        assert!(last > 2_900, "{at_120:?}");
        assert_eq!(at_120.diagnosis.unstarted, 0, "{at_120:?}");
        assert_eq!(at_120.diagnosis.started, 0, "{at_120:?}");
        assert_eq!(
            (at_120.delivered, at_120.diagnosis.at_deadline),
            (8, 113),
            "{at_120:?}"
        );
        let at_60 = simulate(16_667, 20, 5_000, 1, 3_000);
        assert_eq!(at_60.diagnosis.unstarted, 0);
        assert_eq!(at_60.diagnosis.started + at_60.diagnosis.at_deadline, 0);
        assert!(at_60.delivered > 300, "{}", at_60.delivered);
        let steady = simulate(8_300, u64::MAX, 0, 1, 3_000);
        assert_eq!(
            steady.diagnosis.unstarted, 0,
            "{:?}",
            steady.diagnosis.unstarted
        );
        assert!(steady.delivered > 300, "{}", steady.delivered);
    }

    /// The refusal the Chrome run measured, as a property of the rules, and
    /// its recovery since #41. Seeded at four times its 29 ms (the page's had
    /// grown to about three and a half), the job's estimates sum past what its
    /// 25 frames of 4 400 µs hold, although every slice fits a frame.
    ///
    /// Before #41 the job was abandoned at arrival, nothing ran and the
    /// estimates stayed where they were: every one of more than 2 900 inputs
    /// was missed before its first slice, none was delivered, and the job's
    /// estimate ended as it began. Since, a job with no job behind it is kept:
    /// it runs what its frames hold, those slices are measured at what they
    /// take, and once enough are the job fits and is delivered. Estimated at
    /// what it takes, the same job is delivered every time. What made the
    /// page's estimates grow is not modelled.
    #[test]
    fn a_job_estimated_past_its_frames_runs_alone_and_its_estimates_come_down() {
        let inflated = simulate(8_300, u64::MAX, 0, 4, 3_000);
        assert_eq!(inflated.diagnosis.unstarted, 0, "{inflated:?}");
        assert_eq!(inflated.diagnosis.started, 0, "{inflated:?}");
        // 29 jobs are missed, each at its deadline, while the estimates come
        // down to what the slices take.
        assert_eq!(
            (inflated.delivered, inflated.diagnosis.at_deadline),
            (297, 29),
            "{inflated:?}"
        );
        assert_eq!(inflated.job_estimate_us, (115_200, 28_800));
        let measured = simulate(8_300, u64::MAX, 0, 1, 3_000);
        assert_eq!(measured.diagnosis.unstarted, 0);
        assert_eq!(
            measured.diagnosis.started + measured.diagnosis.at_deadline,
            0
        );
        assert_eq!(measured.delivered, 425);
    }

    #[test]
    fn first_passes_are_kept_apart_from_later_ones() {
        let mut diagnosis = Diagnosis::default();
        let s = SliceId::new;
        diagnosis.passes(&[(s(0), Micros::new(2_000)), (s(1), Micros::new(500))]);
        diagnosis.passes(&[(s(1), Micros::new(1_500)), (s(0), Micros::new(1_000))]);
        diagnosis.passes(&[(s(2), Micros::new(300))]);
        assert_eq!(diagnosis.first_pass_us, vec![2_000, 1_500, 300]);
        assert_eq!(diagnosis.later_pass_us, vec![500, 1_000]);
        // Slice 0: 2000 / 1000; slice 1: 1500 / 500; slice 2 only first.
        assert_eq!(diagnosis.first_over_later(), Some((3.0, 2)));
        assert_eq!(Diagnosis::default().first_over_later(), None);
        diagnosis.missed(Reason::Unstarted);
        diagnosis.missed(Reason::AtDeadline);
        assert_eq!(
            (
                diagnosis.unstarted,
                diagnosis.started,
                diagnosis.at_deadline
            ),
            (1, 0, 1)
        );
    }
}
