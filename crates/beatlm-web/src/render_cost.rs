//! What the render pass is budgeted at: the frame's own budget from the render
//! measured last, and the forecast of the frames to come from the median of
//! the recent ones, as the bench's `engine.rs` keeps them and for its reasons
//! (`Scheduler::set_expected_budget`).
//!
//! Kept apart from the frame loop, which only the browser builds, so the rule
//! and what it does to delivery can be tested natively.
//!
//! # A render measurement that carries the slices
//!
//! The rule assumes the render pass's measured time is the render's own. If a
//! render pass waits for the frame's slices and its timestamps count the wait,
//! a frame that ran `s` of slices measures a render of about `render + s`, and
//! the rule feeds that back twice: the next frame's budget shrinks by `s`, so
//! slice-free frames alternate with full ones; and once more than half of the
//! recent frames ran slices, the median forecast is a render that fills the
//! period and no slice fits an expected frame. Before #41 the scheduler then
//! abandoned the job it was running, and jobs starved: more than three missed
//! for each delivered. Since, a job with no job behind it is kept, as every
//! job of the `--submit free` admission is, and runs in the frames whose own
//! budget has room, so the alternation halves what is delivered instead. The
//! test `a_render_that_waits_for_the_slices_slows_the_jobs` shows it on the
//! scheduler alone. It was suspected of the depth model's
//! collapse in Chrome and measured not to be it: with the display reading no
//! buffer a slice writes, the render measured p50 1 245 µs in frames with
//! slices and 1 901 µs without (Chrome 154, WebGPU, Apple M1 Pro, 120 Hz,
//! 2026-10-22, Depth Anything V2 Small at 266 pixels, `latency_ms=200`, the
//! run that introduced these two lines), and the collapse remained;
//! [`crate::diagnosis`] has what was found next. Its run, a later one of the
//! same configuration, measured 1 573 µs with slices: the two runs differ, and
//! neither shows the render carrying the slices.

use std::collections::VecDeque;

use beatlm_core::budget::Budget;
use beatlm_core::time::Micros;
use beatlm_host::presets::{RENDER_PRIOR, RENDER_WINDOW};

/// The render's measurements, as the budget reads them.
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
    pub fn record(&mut self, render: Micros) {
        self.last = render;
        if self.recent.len() == RENDER_WINDOW {
            self.recent.pop_front();
        }
        self.recent.push_back(render);
    }

    /// The render measured last, or the prior before any.
    #[must_use]
    pub fn last(&self) -> Micros {
        self.last
    }

    /// The render the frames to come are forecast with: the median of the
    /// recent measurements (the upper one of an even count), or the prior
    /// before any has arrived.
    #[must_use]
    pub fn forecast(&self) -> Micros {
        let mut recent: Vec<Micros> = self.recent.iter().copied().collect();
        recent.sort_unstable();
        recent
            .get(recent.len() / 2)
            .copied()
            .unwrap_or(RENDER_PRIOR)
    }

    /// This frame's budget, at `period` less the render measured last and
    /// `margin`.
    #[must_use]
    pub fn budget(&self, period: Micros, margin: Micros) -> Budget {
        Budget::for_frame(period, self.last, margin)
    }

    /// The budget expected of the frames to come, at `period` less the
    /// forecast render and `margin`.
    #[must_use]
    pub fn expected(&self, period: Micros, margin: Micros) -> Budget {
        Budget::for_frame(period, self.forecast(), margin)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::depth::still_busy;
    use beatlm_core::cost::{CostModel, Estimator, SliceId};
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
    /// frames), with the `--submit free` admission, a job of 160 slices of
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
        let mut busy = None;
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
            let job = if busy.is_none() {
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
            busy = still_busy(busy, job.map(|job| (job, slices.len())), &plan);
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
}
