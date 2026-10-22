//! Measured slice durations, as an estimate the scheduler can plan with.
//!
//! The cost model is updated only from measurements. Nothing else, such as a
//! plan's shape or a guess, feeds it; until a slice has been measured the
//! configured prior stands in. Which statistic of the measurements is the
//! estimate is an [`Estimator`], chosen when the [`CostModel`] is built.
//!
//! # Choosing the estimator
//!
//! The scheduler sums the estimates of the slices it hands out and keeps the
//! sum within the frame budget, so an estimate above what a slice usually takes
//! costs deadlines even when the frame had room. Three estimators were compared
//! with `beatlm-bench` ([`Estimator::MeanPlusSigmas`] with `alpha` 0.2 and
//! `sigmas` 2; [`Estimator::WindowedQuantile`] with a window of 32 and the 0.9
//! quantile; [`Estimator::CappedDeviation`] with the same `alpha` and `sigmas`
//! and a cap of 0.5), idle and with a competing load from a second device on
//! the same adapter (`--background-load`, 4000 loop iterations per invocation
//! over 16 384 workgroups; the bench crate documentation describes it).
//!
//! Configuration of every run: `--workload synthetic --frames 600 --runs 2`,
//! latency 2, four slices of `--slice-iters 500`, render 50 iterations, margin
//! 2 ms, release build, on an Apple M1 Pro, Metal, wgpu 30.0.1, on 2026-10-21.
//! Every run is one row; no run was discarded. Delivery rate is delivered over
//! delivered plus missed; "over" is timed frames whose render plus slices took
//! longer than the 16 667 us period (`gpu_frames_over_period`); "wrong" is
//! deliveries at a latency other than the promised 2 frames; slice 0 is the
//! first slice of a job, shown because its pass boundary can include time the
//! pass waited behind earlier work (the other three are in the bench output).
//!
//! | Estimator | Load | Run | Delivered | Missed | Rate | Over | Wrong | Slice 0 estimate / mean (us) |
//! |---|---|---|---|---|---|---|---|---|
//! | mean + sigmas | idle | 1 | 450 | 148 | 0.7525 | 0 | 0 | 2481 / 1420 |
//! | mean + sigmas | idle | 2 | 546 | 52 | 0.9130 | 0 | 0 | 2976 / 1550 |
//! | mean + sigmas | load | 1 | 586 | 12 | 0.9799 | 3 | 0 | 1118 / 794 |
//! | mean + sigmas | load | 2 | 595 | 3 | 0.9950 | 1 | 0 | 1025 / 715 |
//! | quantile | idle | 1 | 598 | 0 | 1.0000 | 0 | 0 | 2406 / 1141 |
//! | quantile | idle | 2 | 598 | 0 | 1.0000 | 0 | 0 | 3046 / 1421 |
//! | quantile | load | 1 | 598 | 0 | 1.0000 | 0 | 0 | 830 / 455 |
//! | quantile | load | 2 | 598 | 0 | 1.0000 | 0 | 0 | 1006 / 600 |
//! | capped | idle | 1 | 598 | 0 | 1.0000 | 0 | 0 | 2016 / 1430 |
//! | capped | idle | 2 | 585 | 6 | 0.9898 | 0 | 0 | 1986 / 1373 |
//! | capped | load | 1 | 598 | 0 | 1.0000 | 0 | 0 | 994 / 719 |
//! | capped | load | 2 | 598 | 0 | 1.0000 | 0 | 0 | 866 / 726 |
//!
//! What the numbers say, and what they do not:
//!
//! - **The windowed quantile is the default.** It delivered every deadline in
//!   all four of its runs, idle and loaded, with no frame over the period and
//!   no wrong-latency delivery, which is no worse than idle mean-plus-sigmas on
//!   either of those counts and better on delivery. Capped deviation was close
//!   (one run in four missed 6), and mean-plus-sigmas was the only estimator to
//!   miss in bulk.
//! - **The condition that separated the estimators was the idle one.** At idle
//!   the slices measured 0.7 to 1.6 ms (means per slice, in these runs),
//!   under the competing load 0.3 to 0.8 ms; the load did not raise the means.
//!   Both are consistent with the GPU running at a lower clock when little
//!   else uses it, but no clock was read, so that is a reading of the figures
//!   and not a measurement. The mean-plus-sigmas estimate was up to 2.9 times
//!   the measured mean at idle and its summed estimates did not fit the budget
//!   (52 to 148 of 598 missed); under the load, where the slices were quicker
//!   and steadier, it missed 3 to 12 and was the only estimator with frames over
//!   the period (3 and 1).
//! - **The estimate inflating far past the mean was not reproduced in the
//!   table.** It appeared once in the sweep that tuned the load (an estimate of
//!   8.3 ms against a measured mean of 0.8 ms for slice 0, one 300-frame run
//!   at 4000 iterations) and not in these runs. Two runs per cell is too few
//!   to rank quantile against capped; the evidence is that mean-plus-sigmas is
//!   worse here, and that the quantile did not miss in four runs.
//! - **The quantile can under-budget a rare slow slice.** The 0.9 quantile of
//!   32 ignores the slowest three measurements in the window, so a slice slower
//!   than that one time in ten is budgeted below its slow case. None of these
//!   runs had a frame over the period under it; a workload whose slow cases
//!   are more frequent, or more costly, than these is not covered by this
//!   table.
//!
//! # Planning a frame against its total
//!
//! The scheduler sums the estimates of a frame's slices against its budget.
//! That sum stands for the frame's total only while the slices' slow
//! measurements are independent of each other, and in a browser they were
//! not. Chrome 154 (WebGPU) on an Apple M1 Pro driving a 120 Hz display
//! (period 8 300 µs), 2026-10-22, Depth Anything V2 Small at 266 pixels,
//! latency 200 ms (25 frames), one run per estimator (`beatlm-web`'s `config`
//! module has the table, the URLs and the diagnosis lines quoted here):
//!
//! - Slow measurements came by frame. 175 frames had half or more of their
//!   measurements above twice their slice's median, against 26.7 if each
//!   measurement were slow on its own at the run's share, and each slice's
//!   0.9 quantile was twice its median (the median over slices).
//! - Planned at the sum of the slices' 0.9 quantiles, the default, the run
//!   delivered 19 of 67 deadlines (0.28) and 57 of 1 496 frames ran over the
//!   period. Planned at the sum of their medians it delivered 114 of 118
//!   (0.97), and 408 of 1 477 frames (28%) ran over the period.
//!
//! A sum of per-slice quantiles is too high for the typical frame, since most
//! of a slice's slow measurements fall in frames where its neighbours were
//! not slow, and it is still no bound on a frame where all of them are.
//! [`Estimator::FrameScaled`] models the frame instead: its total is the sum
//! of its slices' typical costs times a slowdown common to the frame. A
//! slice's estimate is the median of its window. The model also keeps, for
//! each recent frame, the ratio of the slice time it measured to the sum of
//! the estimates its slices were planned with ([`CostModel::record_frame`]),
//! and the frame scale F is a quantile of those ratios. The scheduler plans a
//! slice at its estimate times F ([`CostModel::planning_estimate`]), so a
//! frame filled to its budget overruns it about as often as a frame's ratio
//! exceeds F: in `1 - frame_quantile` of the frames, as far as the frames to
//! come are like the ones in the window. The scheduler's rule is unchanged:
//! the planning estimates it sums never exceed the budget.
//!
//! The choices, and why:
//!
//! - **The ratio is taken against the estimates the frame was planned
//!   with,** which the host hands back with the frame's measurements, and not
//!   against the estimates when the measurements arrive. F scales the planned
//!   figures, so the ratio it reads must be of the same figures; and it then
//!   does not matter whether the host records the frame's slices before or
//!   after the frame.
//! - **A frame counts in proportion to its predicted time.** F is the
//!   weighted nearest-rank quantile of the ratios, each weighted by the
//!   frame's predicted total. A frame that ran one small slice, a job's last
//!   say, has a ratio made largely of pass overhead and timestamp steps
//!   (65.5 µs in Chrome); weighted, it cannot move F the way a full frame
//!   does, and no threshold for a frame too small to count has to be chosen.
//!   Frames filled to their budget weigh about the same, so for them the
//!   quantile is of frames. A frame predicted at zero has no ratio and is not
//!   recorded.
//! - **F is at least 1.** Below 1 the frames measured faster than their
//!   slices' medians, which is what medians that lag a speed-up show, and
//!   the medians correct that themselves as their windows turn over. An F
//!   below 1 would plan a slice under its own median, below anything that
//!   slice was measured to take typically, and stay in force after the
//!   medians caught up, until the frames that made it aged out.
//! - **The median F multiplies is made of the loop's measurements alone.**
//!   Calibration's back-to-back runs of a job are recorded apart
//!   ([`CostModel::record_calibration`]), and the slice's loop median does
//!   not read them. A first version let them into the median, as every other
//!   estimator does, and in Chrome (the configuration above, 2026-10-22, one
//!   run each, back to back) it delivered 8 and missed 54, against the
//!   default's 17 and 49, with F ending at 2.53 (p90 6.00 over the run).
//!   Slices late in the job, which run only in the jobs that get that far,
//!   had 8 measurements in the loop and medians still calibration's, 918 µs
//!   against a loop mean of 2 343 µs for one, so the frame ratio measured
//!   that staleness, not a slowdown common to the frame, and F carried it to
//!   every slice, those measured often included. Every other estimator still
//!   folds calibration's measurements as the loop's.
//! - **A slice is warm after [`WARM_SAMPLES`] (5) loop measurements.** Until
//!   then it is planned at the `frame_quantile` of all its measurements,
//!   calibration's included, which with the hosts' parameters below is what
//!   the default estimator plans it at: no worse than today for a slice the
//!   loop has hardly measured, and from measurements rather than a constant
//!   guess. Five is the fewest whose median is still a typical measurement
//!   when slow ones are as common as the page measured them (a share of 0.054
//!   to 0.200 by how busy the GPU had been, `beatlm-web`'s `config` module):
//!   at 0.2 the median of three is slow one time in ten, of five one time in
//!   seventeen. More would hold a slice the loop runs once a job, as the
//!   depth model's are, at the default's plan for longer.
//! - **A frame's ratio is taken over its warm slices only**
//!   ([`CostModel::frame_estimate`]), both what they were planned at and
//!   what they measured. A slowdown common to the frame shows in any subset
//!   of its slices, and a cold median's own error stays out of F. Leaving out
//!   only the cold slices, rather than every frame that ran one, keeps the
//!   frames late in a job, which run the slices measured least, in the ratio.
//! - **Until `frame_window` frames have been recorded every measured slice
//!   is planned as a cold one is,** at the `frame_quantile` of all its
//!   measurements: the per-slice bound that kept the frames above within the
//!   period, at the cost of delivery, for the half second or so the warm-up
//!   lasts. F is read only from a full window, so it is a quantile of as many
//!   frames at its first reading as later.
//! - **An unmeasured slice is planned at the prior** in either phase: the
//!   prior stands in for a measurement and is not a median, so F does not
//!   scale it.
//! - **The hosts' bare `frame-scaled`** (the bench's `--estimator`, the
//!   page's `estimator=`) is `window` 32, `frame_window` 64 and
//!   `frame_quantile` 0.9. The window is the default's, so the medians look
//!   back as far as its quantile does. At the 0.9 quantile a window of 64
//!   frames leaves six above the rank read, so one or two slow frames do not
//!   move F, as three slow measurements of 32 do not move the default's
//!   estimate; and it turns over in about half a second at 120 Hz. The
//!   quantile is the default's, so the share of frames the estimate leaves to
//!   overrun is the share the default leaves each slice.
//!
//! # What the frame scale measured
//!
//! In Chrome, in the configuration above (Chrome 154, WebGPU, Apple M1 Pro,
//! 120 Hz, Depth Anything V2 Small at 266 pixels,
//! `?model=depth&side=266&frames=1500&latency_ms=200&estimator=<e>`, 25
//! frames), 2026-10-22, each pair run back to back, one run per cell, no run
//! discarded. "First" is the version that let calibration's measurements
//! into the median; "second" is this one. "Over" is `gpu_frames_over_period`;
//! ratio and F are the report's `frame_ratio` and `frame_scale`.
//!
//! | Version | Estimator | Delivered | Missed | Rate | Over | Ratio p50 / p90 | F p50 / p90 (end) |
//! |---|---|---|---|---|---|---|---|
//! | first | `quantile` (default) | 17 | 49 | 0.2576 | 65 of 1 497 | | |
//! | first | `frame-scaled` | 8 | 54 | 0.1290 | 28 of 1 498 | 1.93 / 4.44 | 2.80 / 6.00 (2.53) |
//! | second | `quantile` (default) | 16 | 51 | 0.2388 | 64 of 1 494 | | |
//! | second | `frame-scaled` | 11 | 52 | 0.1746 | 25 of 1 491 | 1.23 / 2.45 | 2.04 / 3.91 (1.58) |
//!
//! From the `frame-scaled` runs' diagnosis lines (`beatlm-web`'s
//! `diagnosis` module defines them), first and second:
//! `tail_outlier_frames` 99 against 99.3 expected if independent, and 53
//! against 85.0; `tail_loop_over_calibration` median ratio 1.98 and 1.83,
//! mean ratio 2.43 and 2.27; `capacity_frames_needed` 23 and 24 of the 26
//! frames available (a run of the model in the loop of 86.0 and 87.4 ms,
//! against 3 875 and 3 679 µs of slice budget a frame).
//!
//! Natively, with `beatlm-bench` on an Apple M1 Pro, Metal, wgpu 30.0.1,
//! release build, 2026-10-22, the second version: `--workload depth --size
//! 266 --latency 5 --submit free --frames 600`, `--estimator quantile` and
//! then `frame-scaled`, one run each, no run discarded. The ratio is over the
//! frame-scaled run's warm slices, in 575 of the 600 frames.
//!
//! | Estimator | Delivered | Missed | Rate | Over | Dispatches a frame, p50 | Ratio p50 / p90 | F p50 / p90 (end) |
//! |---|---|---|---|---|---|---|---|
//! | quantile | 101 | 12 | 0.8938 | 1 of 600 | 34 | 0.92 / 1.01 | |
//! | frame-scaled | 116 | 0 | 1.0000 | 2 of 600 | 36 | 1.05 / 1.16 | 1.08 / 1.25 (1.29) |
//!
//! Calibration measured a job at 53.7 ms in both, and the slices' loop means
//! summed to 61.9 and 60.1 ms. The machine was slower that day than when #41
//! was measured: `main` at that time delivered 97 of 101, with calibration
//! at 53.6 ms. The first version's native runs, two pairs, swung with the
//! state of the GPU: in a pair where the loop's slices summed to 68 and 83 ms
//! both estimators delivered about half and F reached a p90 of 2.52; in one
//! at 62 and 60 ms the frame scale delivered 117 of 118 and the default 98 of
//! 109.
//!
//! What these runs show, and what they do not:
//!
//! - **In Chrome the frame scale did not deliver more than the default, in
//!   either version.** It delivered 8 against 17 and 11 against 16.
//! - **It cut the frames over the period,** to 28 of the default's 65 in the
//!   first pair and 25 of 64 in the second, about 40%. In the second pair
//!   its frames took less GPU time (`gpu_frame_us` p50 4 457 µs and p99
//!   9 440 µs, against the default's 4 916 and 10 028). F over the run (p50
//!   2.80 and 2.04) was at or above the ratio of a slice's 0.9 quantile to
//!   its median that the default plans with, about 2 in #43; that it is why
//!   the frames held less is a reading, not a measurement.
//! - **The clustering it was designed for was absent from both Chrome
//!   runs.** Frames with half or more of their measurements slow were as
//!   many as independent measurements would give, or fewer (99 against 99.3,
//!   53 against 85.0), where #43 had measured 175 against 26.7. The loop
//!   instead ran 1.8 to 2.4 times calibration (medians and means above). Read
//!   together, the frame ratios measured a slowdown of the whole loop rather
//!   than of some frames, and a quantile of them is then a margin on every
//!   frame; that reading is not a measurement either.
//! - **Capacity was near its limit,** 23 and 24 of the 26 frames a job had
//!   even with exact estimates, on that machine at that time. That caps the
//!   delivery of any estimator, so these runs cannot show how much delivery a
//!   frame-level estimate would buy with room to spare.
//! - **Natively, one run delivered 116 of 116 against the default's 101 of
//!   113,** with 2 frames over the period against 1. One run each ranks
//!   nothing, and natively the clustering was not measured.
//!
//! **The frame scale stays opt-in, and the default stays the windowed
//! quantile.** Neither browser comparison met the aim it was built for,
//! delivery near the medians' rate with no more frames over the period than
//! the default, and the condition it models was not present in them. What
//! would reopen it is a browser run in which slow measurements cluster by
//! frame again and capacity leaves room, compared with the default there.

use std::collections::BTreeMap;
use std::collections::btree_map::Entry;

use thiserror::Error;

use crate::time::Micros;

/// Identifies one slice of a compiled plan.
///
/// A slice is the unit the scheduler hands out and the cost model measures, so
/// both sides name it with the same type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SliceId(u32);

impl SliceId {
    /// Wraps a slice number.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// The slice number.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// The longest window [`Estimator::WindowedQuantile`] accepts.
///
/// Each slice keeps its window, and every measurement scans a copy of it, so an
/// unbounded window is an unbounded cost per measurement and per slice. The
/// bound is generous for a slice measured once per frame: it covers over a
/// minute of history at 60 Hz.
pub const MAX_WINDOW: usize = 4096;

/// A rejected [`CostModel`] configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum CostError {
    /// The smoothing factor must lie in `(0, 1]`; outside it the mean either
    /// never moves or diverges.
    #[error("alpha must be finite and in (0, 1]")]
    InvalidAlpha,
    /// The safety factor must be a non-negative finite number; a negative one
    /// would estimate below the mean.
    #[error("sigmas must be finite and non-negative")]
    InvalidSigmas,
    /// The window must hold at least one measurement and at most
    /// [`MAX_WINDOW`].
    #[error("window must be in 1..=MAX_WINDOW")]
    InvalidWindow,
    /// The quantile must lie in `[0, 1]`; for [`Estimator::FrameScaled`]
    /// this is also the frame quantile's range.
    #[error("quantile must be finite and in [0, 1]")]
    InvalidQuantile,
    /// The frame window must hold at least one frame and at most
    /// [`MAX_WINDOW`], for the reason the slice window has that bound.
    #[error("frame window must be in 1..=MAX_WINDOW")]
    InvalidFrameWindow,
    /// The cap must be a non-negative finite number; a negative one would
    /// estimate below the mean.
    #[error("cap must be finite and non-negative")]
    InvalidCap,
}

/// The statistic a [`CostModel`] turns a slice's measurements into.
///
/// The scheduler trusts the estimate: it sums the estimates of the slices it
/// hands out and keeps the sum within the frame budget. An estimate far above
/// what the slice usually takes therefore wastes budget and misses deadlines
/// as surely as one below it. Which statistic is the right one is a
/// measurement, and the module documentation records the one that settled the
/// default.
///
/// Every variant gives the prior until a slice has been measured, changes only
/// when [`CostModel::record`], [`CostModel::record_calibration`] or
/// [`CostModel::record_frame`] is called, and is deterministic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Estimator {
    /// Exponentially weighted mean plus `sigmas` exponentially weighted
    /// standard deviations. A slice that is usually fast but sometimes slow is
    /// planned for its slow side; under noise the deviation can grow faster
    /// than the mean, and the estimate with it.
    MeanPlusSigmas {
        /// Weight of a new measurement, in `(0, 1]`.
        alpha: f64,
        /// Standard deviations added to the mean, not negative.
        sigmas: f64,
    },
    /// The `quantile` of the last `window` measurements of the slice (nearest
    /// rank, so the estimate is a measurement that was seen). One slow
    /// measurement in a window moves the estimate only when it takes up more
    /// than `1 - quantile` of the window, and leaves it once it ages out.
    WindowedQuantile {
        /// Measurements kept per slice, in `1..=`[`MAX_WINDOW`].
        window: usize,
        /// Rank to read, in `[0, 1]`: 0.5 is the median, 1 is the window's
        /// maximum.
        quantile: f64,
    },
    /// Exponentially weighted mean plus the smaller of `sigmas` standard
    /// deviations and `cap` times the mean, so the padding can never exceed a
    /// fixed fraction of what the slice takes on average.
    CappedDeviation {
        /// Weight of a new measurement, in `(0, 1]`.
        alpha: f64,
        /// Standard deviations added to the mean, not negative.
        sigmas: f64,
        /// The most the padding may be, as a multiple of the mean, not
        /// negative.
        cap: f64,
    },
    /// The median of the last `window` measurements of the slice, planned
    /// at that times the frame scale: the `frame_quantile` of the ratios of
    /// the last `frame_window` frames' measured slice time to what their
    /// slices were planned at, and never below 1. The module documentation
    /// ("Planning a frame against its total") says why, and what holds while
    /// fewer frames have been recorded.
    ///
    /// The only estimator whose [`CostModel::planning_estimate`] differs from
    /// its [`CostModel::estimate`], and the only one that reads
    /// [`CostModel::record_frame`].
    FrameScaled {
        /// Measurements kept per slice, in `1..=`[`MAX_WINDOW`]; the slice's
        /// estimate is their median (nearest rank).
        window: usize,
        /// Frames whose ratio is kept, in `1..=`[`MAX_WINDOW`]; the frame
        /// scale is read once this many have been recorded.
        frame_window: usize,
        /// Rank of the frame ratios to read, in `[0, 1]`; also the rank of
        /// its own window a slice is planned at before the frame scale is
        /// read.
        frame_quantile: f64,
    },
}

/// What [`CostModel::planning_estimate`] applies to a measured slice's
/// estimate: the frame scale of [`Estimator::FrameScaled`], or nothing.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum FrameScale {
    /// The estimator has no frame scale: a slice is planned at its estimate.
    #[default]
    None,
    /// Fewer frames than the frame window holds have been recorded: a
    /// measured slice is planned at the frame quantile of its own window.
    WarmingUp {
        /// Frames recorded so far.
        recorded: usize,
        /// Frames the window holds, after which the scale is read.
        needed: usize,
    },
    /// A measured slice is planned at its estimate times this, at least 1.
    Measured(f64),
}

impl Default for Estimator {
    /// The windowed quantile, 0.9 of the last 32 measurements: the estimator
    /// the measurements in the module documentation chose.
    fn default() -> Self {
        Self::WindowedQuantile {
            window: 32,
            quantile: 0.9,
        }
    }
}

impl Estimator {
    /// Checks the parameters of the chosen statistic.
    ///
    /// # Errors
    ///
    /// The [`CostError`] naming the first parameter that is out of range.
    fn validate(self) -> Result<(), CostError> {
        let unit = |x: f64| x.is_finite() && x > 0.0 && x <= 1.0;
        let non_negative = |x: f64| x.is_finite() && x >= 0.0;
        let rank = |q: f64| q.is_finite() && (0.0..=1.0).contains(&q);
        match self {
            Self::MeanPlusSigmas { alpha, sigmas } => {
                if !unit(alpha) {
                    return Err(CostError::InvalidAlpha);
                }
                if !non_negative(sigmas) {
                    return Err(CostError::InvalidSigmas);
                }
            }
            Self::WindowedQuantile { window, quantile } => {
                if !(1..=MAX_WINDOW).contains(&window) {
                    return Err(CostError::InvalidWindow);
                }
                if !rank(quantile) {
                    return Err(CostError::InvalidQuantile);
                }
            }
            Self::FrameScaled {
                window,
                frame_window,
                frame_quantile,
            } => {
                if !(1..=MAX_WINDOW).contains(&window) {
                    return Err(CostError::InvalidWindow);
                }
                if !(1..=MAX_WINDOW).contains(&frame_window) {
                    return Err(CostError::InvalidFrameWindow);
                }
                if !rank(frame_quantile) {
                    return Err(CostError::InvalidQuantile);
                }
            }
            Self::CappedDeviation { alpha, sigmas, cap } => {
                if !unit(alpha) {
                    return Err(CostError::InvalidAlpha);
                }
                if !non_negative(sigmas) {
                    return Err(CostError::InvalidSigmas);
                }
                if !non_negative(cap) {
                    return Err(CostError::InvalidCap);
                }
            }
        }
        Ok(())
    }
}

/// Exponentially weighted mean and variance of one slice.
#[derive(Debug, Clone, Copy)]
struct Ewma {
    mean: f64,
    var: f64,
    samples: u32,
}

impl Ewma {
    const fn first(x: f64) -> Self {
        Self {
            mean: x,
            var: 0.0,
            samples: 1,
        }
    }

    fn fold(&mut self, x: f64, alpha: f64) {
        let diff = x - self.mean;
        let incr = alpha * diff;
        self.mean += incr;
        self.var = (1.0 - alpha) * (self.var + diff * incr);
        self.samples = self.samples.saturating_add(1);
    }

    fn stddev(&self) -> f64 {
        self.var.max(0.0).sqrt()
    }
}

/// The last measurements of one slice, and the quantile read from them.
///
/// The estimate is computed when a measurement is recorded and kept, and the
/// buffers are sized when the slice is first seen, so reading the estimate
/// costs nothing and recording allocates only the first time.
#[derive(Debug, Clone)]
struct Window {
    ring: Vec<u64>,
    next: usize,
    scratch: Vec<u64>,
    estimate: u64,
    /// A second quantile of the window, for [`Estimator::FrameScaled`];
    /// the estimate itself when only one is read.
    second: u64,
    samples: u32,
}

impl Window {
    fn new(capacity: usize) -> Self {
        Self {
            ring: Vec::with_capacity(capacity),
            next: 0,
            scratch: Vec::with_capacity(capacity),
            estimate: 0,
            second: 0,
            samples: 0,
        }
    }

    /// Adds `x`, then reads the estimate at `quantile` and, when given, the
    /// second value at `second`.
    fn fold(&mut self, x: u64, capacity: usize, quantile: f64, second: Option<f64>) {
        if self.ring.len() < capacity {
            self.ring.push(x);
        } else if let Some(slot) = self.ring.get_mut(self.next) {
            *slot = x;
        }
        self.next = (self.next + 1) % capacity;
        self.samples = self.samples.saturating_add(1);

        self.scratch.clear();
        self.scratch.extend_from_slice(&self.ring);
        self.estimate = self.select(quantile).unwrap_or(x);
        self.second = match second {
            Some(q) => self.select(q).unwrap_or(x),
            None => self.estimate,
        };
    }

    /// The value at `quantile`'s nearest rank in the scratch copy, `None`
    /// when it is empty. Only that value is needed, so it is selected rather
    /// than the window sorted: linear rather than n log n per measurement.
    /// Selecting reorders the copy and keeps its values, so a second call
    /// reads the same window.
    fn select(&mut self, quantile: f64) -> Option<u64> {
        if self.scratch.is_empty() {
            return None;
        }
        let rank = nearest_rank(quantile, self.scratch.len());
        Some(*self.scratch.select_nth_unstable(rank - 1).1)
    }
}

/// One recorded frame: its measured slice time over its predicted, and the
/// predicted time, which is its weight.
#[derive(Debug, Clone, Copy)]
struct Ratio {
    ratio: f64,
    weight: f64,
}

/// The last frames' ratios and the frame scale read from them, kept by
/// [`Estimator::FrameScaled`].
#[derive(Debug, Clone)]
struct Frames {
    ring: Vec<Ratio>,
    next: usize,
    scratch: Vec<Ratio>,
    /// The frame scale; read only once the ring is full.
    scale: f64,
    recorded: u64,
}

impl Frames {
    fn new(capacity: usize) -> Self {
        Self {
            ring: Vec::with_capacity(capacity),
            next: 0,
            scratch: Vec::with_capacity(capacity),
            scale: 1.0,
            recorded: 0,
        }
    }

    fn fold(&mut self, frame: Ratio, capacity: usize, quantile: f64) {
        if self.ring.len() < capacity {
            self.ring.push(frame);
        } else if let Some(slot) = self.ring.get_mut(self.next) {
            *slot = frame;
        }
        self.next = (self.next + 1) % capacity;
        self.recorded = self.recorded.saturating_add(1);

        // The scale is read only from a full window.
        if self.ring.len() < capacity {
            return;
        }
        self.scratch.clear();
        self.scratch.extend_from_slice(&self.ring);
        self.scratch
            .sort_unstable_by(|a, b| a.ratio.total_cmp(&b.ratio));
        self.scale = weighted_rank(&self.scratch, quantile).max(1.0);
    }

    fn scale(&self, capacity: usize) -> FrameScale {
        if self.ring.len() < capacity {
            FrameScale::WarmingUp {
                recorded: self.ring.len(),
                needed: capacity,
            }
        } else {
            FrameScale::Measured(self.scale)
        }
    }
}

/// The weighted nearest rank of `quantile` among `sorted`, ascending by
/// ratio: the first ratio at which the weight up to and including it reaches
/// `quantile` of the total. With equal weights it is [`nearest_rank`]'s
/// value, and the same tolerance keeps a whole rank from being pushed up by
/// rounding. 1 when there are no ratios.
fn weighted_rank(sorted: &[Ratio], quantile: f64) -> f64 {
    let total: f64 = sorted.iter().map(|r| r.weight).sum();
    let target = quantile.mul_add(total, -(total * 1e-9));
    let mut reached = 0.0;
    for r in sorted {
        reached += r.weight;
        if reached >= target {
            return r.ratio;
        }
    }
    sorted.last().map_or(1.0, |r| r.ratio)
}

/// The 1-based nearest rank of `quantile` among `n` sorted values, clamped to
/// `1..=n`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn nearest_rank(quantile: f64, n: usize) -> usize {
    // `n` is at most MAX_WINDOW and `quantile` is in [0, 1], so the product is
    // non-negative. It can land a rounding error above a whole number (0.55 *
    // 100 is 55.000000000000007), which `ceil` would turn into the next rank, so
    // a product within a small tolerance of a whole number is that number.
    let product = quantile * n as f64;
    let nearest = product.round();
    let rank = if (product - nearest).abs() < 1e-9 {
        nearest
    } else {
        product.ceil()
    } as usize;
    rank.clamp(1, n.max(1))
}

/// Loop measurements a slice needs before [`Estimator::FrameScaled`] plans it
/// at its loop median times the frame scale, and takes it into a frame's
/// ratio. The module documentation ("Planning a frame against its total")
/// says why five.
pub const WARM_SAMPLES: u32 = 5;

/// One slice of [`Estimator::FrameScaled`].
#[derive(Debug, Clone)]
struct Scaled {
    /// Every measurement, calibration's included: `estimate` is the frame
    /// quantile, which plans the slice until it is warm, as the windowed
    /// quantile would; `second` is the median.
    all: Window,
    /// The loop's measurements alone, from [`CostModel::record`]:
    /// `estimate` is the median the frame scale multiplies.
    looped: Window,
}

impl Scaled {
    fn new(capacity: usize) -> Self {
        Self {
            all: Window::new(capacity),
            looped: Window::new(capacity),
        }
    }

    const fn warm(&self) -> bool {
        self.looped.samples >= WARM_SAMPLES
    }
}

#[derive(Debug, Clone)]
enum State {
    Ewma(Ewma),
    Window(Window),
    Scaled(Scaled),
}

/// Per-slice statistics of measured durations, read as the estimate the
/// scheduler plans with.
///
/// Which statistic is read is the [`Estimator`] the model was built with.
/// Slices are kept in an ordered map so that identical measurements always give
/// identical estimates.
#[derive(Debug, Clone)]
pub struct CostModel {
    prior: Micros,
    estimator: Estimator,
    stats: BTreeMap<SliceId, State>,
    /// The frame ratios, for [`Estimator::FrameScaled`] only.
    frames: Option<Frames>,
}

impl CostModel {
    /// A model that estimates `prior` for unmeasured slices and reads
    /// measured ones with `estimator`.
    ///
    /// # Errors
    ///
    /// The [`CostError`] for the first parameter of `estimator` that is out of
    /// range: [`CostError::InvalidAlpha`] unless `alpha` is finite and in
    /// `(0, 1]`; [`CostError::InvalidSigmas`] and [`CostError::InvalidCap`]
    /// unless finite and not negative; [`CostError::InvalidWindow`] unless the
    /// window is in `1..=`[`MAX_WINDOW`]; [`CostError::InvalidQuantile`]
    /// unless the quantile is finite and in `[0, 1]`, the frame quantile
    /// included; [`CostError::InvalidFrameWindow`] unless the frame window is
    /// in `1..=`[`MAX_WINDOW`].
    pub fn new(prior: Micros, estimator: Estimator) -> Result<Self, CostError> {
        estimator.validate()?;
        let frames = match estimator {
            Estimator::FrameScaled { frame_window, .. } => Some(Frames::new(frame_window)),
            _ => None,
        };
        Ok(Self {
            prior,
            estimator,
            stats: BTreeMap::new(),
            frames,
        })
    }

    /// The statistic this model estimates with.
    #[must_use]
    pub const fn estimator(&self) -> Estimator {
        self.estimator
    }

    /// Folds one measured duration of `slice` into its statistics.
    ///
    /// For the exponentially weighted estimators the first sample sets the mean
    /// and leaves the variance at zero; later samples move the mean by `alpha`
    /// of the difference and decay the variance accordingly. For the windowed
    /// quantile and the frame-scaled median the sample replaces the oldest
    /// one once the window is full.
    ///
    /// It is a measurement of the loop the model plans: the frame-scaled
    /// estimator's median is made of these alone. A measurement taken apart
    /// from the loop goes to [`record_calibration`](Self::record_calibration).
    pub fn record(&mut self, slice: SliceId, measured: Micros) {
        self.fold(slice, measured, true);
    }

    /// Folds one duration of `slice` measured apart from the loop the model
    /// plans, such as calibration's runs of a job back to back.
    ///
    /// Every estimator but [`Estimator::FrameScaled`] folds it exactly as
    /// [`record`](Self::record). That one keeps it out of the median the
    /// frame scale multiplies, and plans the slice with it, at the frame
    /// quantile of all its measurements, only until the slice has
    /// [`WARM_SAMPLES`] loop measurements: the module documentation
    /// ("Planning a frame against its total") says why.
    pub fn record_calibration(&mut self, slice: SliceId, measured: Micros) {
        self.fold(slice, measured, false);
    }

    fn fold(&mut self, slice: SliceId, measured: Micros, in_loop: bool) {
        match self.estimator {
            Estimator::MeanPlusSigmas { alpha, .. } | Estimator::CappedDeviation { alpha, .. } => {
                let x = micros_to_f64(measured);
                match self.stats.entry(slice) {
                    Entry::Vacant(v) => {
                        v.insert(State::Ewma(Ewma::first(x)));
                    }
                    Entry::Occupied(mut o) => {
                        if let State::Ewma(s) = o.get_mut() {
                            s.fold(x, alpha);
                        }
                    }
                }
            }
            Estimator::WindowedQuantile { window, quantile } => {
                let state = self
                    .stats
                    .entry(slice)
                    .or_insert_with(|| State::Window(Window::new(window)));
                if let State::Window(w) = state {
                    w.fold(measured.get(), window, quantile, None);
                }
            }
            Estimator::FrameScaled {
                window,
                frame_quantile,
                ..
            } => {
                let state = self
                    .stats
                    .entry(slice)
                    .or_insert_with(|| State::Scaled(Scaled::new(window)));
                if let State::Scaled(s) = state {
                    let x = measured.get();
                    s.all.fold(x, window, frame_quantile, Some(0.5));
                    if in_loop {
                        s.looped.fold(x, window, 0.5, None);
                    }
                }
            }
        }
    }

    /// What a frame's ratio takes `slice` at, read when the frame is planned
    /// and handed back to [`record_frame`](Self::record_frame) with the
    /// slice's measurement; `None` when the slice is left out of the ratio.
    ///
    /// For [`Estimator::FrameScaled`] it is the slice's loop median once it
    /// has [`WARM_SAMPLES`] loop measurements, and `None` before: a median
    /// still made of calibration's measurements, or of too few of the
    /// loop's, would put its own error into the ratio, and the frame scale
    /// would carry it to every slice. For every other estimator it is the
    /// estimate, so the report's ratio covers every slice.
    #[must_use]
    pub fn frame_estimate(&self, slice: SliceId) -> Option<Micros> {
        match (self.stats.get(&slice), self.estimator) {
            (Some(State::Scaled(s)), _) => s.warm().then(|| Micros::new(s.looped.estimate)),
            (None, Estimator::FrameScaled { .. }) => None,
            _ => Some(self.estimate(slice)),
        }
    }

    /// Folds one frame into the frame scale: `predicted`, the sum of
    /// [`frame_estimate`](Self::frame_estimate) of the slices it measured, as
    /// they stood when the frame was planned and over the slices where it was
    /// not `None`, and `measured`, the sum of those same slices' measured
    /// durations. The module documentation ("Planning a frame against its
    /// total") says why the estimates are the planned ones, why only warm
    /// slices count and why the frame weighs as much as `predicted`.
    ///
    /// Only [`Estimator::FrameScaled`] reads it; every other estimator
    /// ignores it, so a host can call it whatever the estimator. A frame
    /// predicted at zero has no ratio and is not recorded. The order against
    /// [`record`](Self::record) of the same frame's slices does not matter.
    pub fn record_frame(&mut self, predicted: Micros, measured: Micros) {
        let (
            Estimator::FrameScaled {
                frame_window,
                frame_quantile,
                ..
            },
            Some(frames),
        ) = (self.estimator, &mut self.frames)
        else {
            return;
        };
        if predicted == Micros::ZERO {
            return;
        }
        let weight = micros_to_f64(predicted);
        let frame = Ratio {
            ratio: micros_to_f64(measured) / weight,
            weight,
        };
        frames.fold(frame, frame_window, frame_quantile);
    }

    /// The frame scale [`planning_estimate`](Self::planning_estimate)
    /// applies now: [`FrameScale::None`] for every estimator but
    /// [`Estimator::FrameScaled`].
    #[must_use]
    pub fn frame_scale(&self) -> FrameScale {
        match (self.estimator, &self.frames) {
            (Estimator::FrameScaled { frame_window, .. }, Some(frames)) => {
                frames.scale(frame_window)
            }
            _ => FrameScale::None,
        }
    }

    /// How many frames [`record_frame`](Self::record_frame) has folded in,
    /// saturating; zero for an estimator that ignores it.
    #[must_use]
    pub fn frames_recorded(&self) -> u64 {
        self.frames.as_ref().map_or(0, |frames| frames.recorded)
    }

    /// The duration the scheduler plans `slice` at, and sums against a
    /// frame's budget.
    ///
    /// For every estimator but [`Estimator::FrameScaled`] it is
    /// [`estimate`](Self::estimate) exactly. For that one it is the prior
    /// until the slice has been measured; then, until the slice has
    /// [`WARM_SAMPLES`] loop measurements or while the model is warming up
    /// ([`FrameScale::WarmingUp`]), the frame quantile of all the slice's
    /// measurements, calibration's included, which is what the windowed
    /// quantile at that quantile would plan; and after both the loop median
    /// times the frame scale, rounded up to whole microseconds, never below
    /// the median.
    #[must_use]
    pub fn planning_estimate(&self, slice: SliceId) -> Micros {
        match (self.stats.get(&slice), self.frame_scale()) {
            (Some(State::Scaled(s)), FrameScale::Measured(scale)) if s.warm() => {
                scaled_ceil(s.looped.estimate, scale)
            }
            (Some(State::Scaled(s)), _) => Micros::new(s.all.estimate),
            _ => self.estimate(slice),
        }
    }

    /// The duration to budget for `slice`: the prior until it has been
    /// measured, afterwards what the [`Estimator`] reads from its
    /// measurements, rounded up to whole microseconds. For
    /// [`Estimator::FrameScaled`] it is the median of the loop's
    /// measurements once the slice has [`WARM_SAMPLES`] of them, and the
    /// median of all its measurements before.
    #[must_use]
    pub fn estimate(&self, slice: SliceId) -> Micros {
        match (self.stats.get(&slice), self.estimator) {
            (Some(State::Ewma(s)), Estimator::MeanPlusSigmas { sigmas, .. }) => {
                f64_to_micros_ceil(sigmas.mul_add(s.stddev(), s.mean))
            }
            (Some(State::Ewma(s)), Estimator::CappedDeviation { sigmas, cap, .. }) => {
                let padding = (sigmas * s.stddev()).min(cap * s.mean.max(0.0));
                f64_to_micros_ceil(s.mean + padding)
            }
            (Some(State::Window(w)), Estimator::WindowedQuantile { .. }) => Micros::new(w.estimate),
            (Some(State::Scaled(s)), Estimator::FrameScaled { .. }) => Micros::new(if s.warm() {
                s.looped.estimate
            } else {
                s.all.second
            }),
            _ => self.prior,
        }
    }

    /// How many measurements of `slice` the model has seen, saturating;
    /// calibration's included.
    #[must_use]
    pub fn samples(&self, slice: SliceId) -> u32 {
        match self.stats.get(&slice) {
            None => 0,
            Some(State::Ewma(s)) => s.samples,
            Some(State::Window(w)) => w.samples,
            Some(State::Scaled(s)) => s.all.samples,
        }
    }
}

/// Microsecond counts beyond 2^53 lose low bits; at that size (centuries) the
/// error is irrelevant to a frame budget.
#[allow(clippy::cast_precision_loss)]
const fn micros_to_f64(m: Micros) -> f64 {
    m.get() as f64
}

/// `median` times `scale`, rounded up to whole microseconds. A product within
/// a billionth of itself of a whole number is that number, as in
/// [`nearest_rank`]: 100 times a scale of 1.1 (1 100 measured over 1 000
/// planned) is 110.00000000000001 in floating point, which `ceil` alone would
/// plan at 111.
fn scaled_ceil(median: u64, scale: f64) -> Micros {
    let product = micros_to_f64(Micros::new(median)) * scale;
    let nearest = product.round();
    if (product - nearest).abs() <= product.abs() * 1e-9 {
        f64_to_micros_ceil(nearest)
    } else {
        f64_to_micros_ceil(product)
    }
}

/// Rounds up and converts, clamping first so the cast cannot truncate or
/// change sign: NaN and negatives become zero, values at or above 2^64 become
/// `u64::MAX`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn f64_to_micros_ceil(v: f64) -> Micros {
    const U64_LIMIT: f64 = 18_446_744_073_709_551_616.0; // 2^64
    let v = v.max(0.0).ceil();
    if v >= U64_LIMIT {
        Micros::new(u64::MAX)
    } else {
        Micros::new(v as u64)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const S: SliceId = SliceId::new(7);

    fn mean_sigmas(prior: u64, alpha: f64, sigmas: f64) -> CostModel {
        CostModel::new(
            Micros::new(prior),
            Estimator::MeanPlusSigmas { alpha, sigmas },
        )
        .unwrap()
    }

    fn quantile(prior: u64, window: usize, quantile: f64) -> CostModel {
        CostModel::new(
            Micros::new(prior),
            Estimator::WindowedQuantile { window, quantile },
        )
        .unwrap()
    }

    fn capped(prior: u64, alpha: f64, sigmas: f64, cap: f64) -> CostModel {
        CostModel::new(
            Micros::new(prior),
            Estimator::CappedDeviation { alpha, sigmas, cap },
        )
        .unwrap()
    }

    fn rejected(estimator: Estimator) -> CostError {
        CostModel::new(Micros::ZERO, estimator).unwrap_err()
    }

    #[test]
    fn invalid_alpha_is_rejected() {
        for alpha in [0.0, -0.5, 1.5, f64::NAN, f64::INFINITY] {
            assert_eq!(
                rejected(Estimator::MeanPlusSigmas { alpha, sigmas: 1.0 }),
                CostError::InvalidAlpha,
                "alpha {alpha}"
            );
            assert_eq!(
                rejected(Estimator::CappedDeviation {
                    alpha,
                    sigmas: 1.0,
                    cap: 1.0
                }),
                CostError::InvalidAlpha,
                "alpha {alpha}"
            );
        }
    }

    #[test]
    fn invalid_sigmas_is_rejected() {
        for sigmas in [-0.1, f64::NAN, f64::INFINITY] {
            assert_eq!(
                rejected(Estimator::MeanPlusSigmas { alpha: 0.5, sigmas }),
                CostError::InvalidSigmas,
                "sigmas {sigmas}"
            );
            assert_eq!(
                rejected(Estimator::CappedDeviation {
                    alpha: 0.5,
                    sigmas,
                    cap: 1.0
                }),
                CostError::InvalidSigmas,
                "sigmas {sigmas}"
            );
        }
    }

    #[test]
    fn invalid_window_and_quantile_are_rejected() {
        for window in [0, MAX_WINDOW + 1] {
            assert_eq!(
                rejected(Estimator::WindowedQuantile {
                    window,
                    quantile: 0.9
                }),
                CostError::InvalidWindow,
                "window {window}"
            );
        }
        for q in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
            assert_eq!(
                rejected(Estimator::WindowedQuantile {
                    window: 8,
                    quantile: q
                }),
                CostError::InvalidQuantile,
                "quantile {q}"
            );
        }
        for (window, q) in [(1, 0.0), (MAX_WINDOW, 1.0), (8, 0.5)] {
            assert!(
                CostModel::new(
                    Micros::ZERO,
                    Estimator::WindowedQuantile {
                        window,
                        quantile: q
                    }
                )
                .is_ok()
            );
        }
    }

    #[test]
    fn invalid_cap_is_rejected() {
        for cap in [-0.1, f64::NAN, f64::INFINITY] {
            assert_eq!(
                rejected(Estimator::CappedDeviation {
                    alpha: 0.5,
                    sigmas: 1.0,
                    cap
                }),
                CostError::InvalidCap,
                "cap {cap}"
            );
        }
    }

    #[test]
    fn prior_is_used_without_samples() {
        for mut m in [
            mean_sigmas(123, 0.5, 2.0),
            quantile(123, 8, 0.9),
            capped(123, 0.5, 2.0, 0.5),
            frame_scaled(123, 8, 4, 0.9),
        ] {
            assert_eq!(m.estimate(S), Micros::new(123));
            assert_eq!(m.samples(S), 0);
            // Measuring another slice leaves this one at the prior.
            m.record(SliceId::new(1), Micros::new(5));
            assert_eq!(m.estimate(S), Micros::new(123));
        }
    }

    #[test]
    fn the_model_reports_its_estimator() {
        let e = Estimator::WindowedQuantile {
            window: 4,
            quantile: 0.5,
        };
        assert_eq!(CostModel::new(Micros::ZERO, e).unwrap().estimator(), e);
    }

    #[test]
    fn alpha_one_tracks_the_last_sample() {
        let mut m = mean_sigmas(1, 1.0, 3.0);
        m.record(S, Micros::new(100));
        m.record(S, Micros::new(400));
        m.record(S, Micros::new(250));
        assert_eq!(m.estimate(S), Micros::new(250));
    }

    #[test]
    fn estimate_pads_the_mean_by_sigmas() {
        let mut m = mean_sigmas(1, 0.5, 2.0);
        m.record(S, Micros::new(100));
        assert_eq!(m.estimate(S), Micros::new(100));
        m.record(S, Micros::new(120));
        m.record(S, Micros::new(80));
        // mean 95, var 275: ceil(95 + 2 * sqrt(275)) = 129.
        assert_eq!(m.estimate(S), Micros::new(129));
        assert!(m.estimate(S) >= Micros::new(95));
    }

    #[test]
    fn samples_count_per_slice() {
        for mut m in [
            mean_sigmas(1, 0.5, 0.0),
            quantile(1, 3, 0.5),
            capped(1, 0.5, 0.0, 0.0),
            frame_scaled(1, 3, 2, 0.5),
        ] {
            m.record(S, Micros::new(10));
            m.record(S, Micros::new(10));
            m.record(SliceId::new(8), Micros::new(10));
            assert_eq!(m.samples(S), 2);
            assert_eq!(m.samples(SliceId::new(8)), 1);
            assert_eq!(m.samples(SliceId::new(9)), 0);
        }
    }

    #[test]
    fn huge_estimates_saturate() {
        let mut m = mean_sigmas(0, 1.0, 0.0);
        m.record(S, Micros::new(u64::MAX));
        assert_eq!(m.estimate(S), Micros::new(u64::MAX));

        let mut m = capped(0, 1.0, 0.0, 0.0);
        m.record(S, Micros::new(u64::MAX));
        assert_eq!(m.estimate(S), Micros::new(u64::MAX));

        let mut m = quantile(0, 2, 1.0);
        m.record(S, Micros::new(u64::MAX));
        assert_eq!(m.estimate(S), Micros::new(u64::MAX));
    }

    #[test]
    fn capped_pads_by_the_smaller_of_sigmas_and_the_cap() {
        // The same measurements as `estimate_pads_the_mean_by_sigmas`: mean
        // 95, standard deviation sqrt(275) = 16.58, so two sigmas pad by 33.2.
        let feed = |m: &mut CostModel| {
            m.record(S, Micros::new(100));
            m.record(S, Micros::new(120));
            m.record(S, Micros::new(80));
        };
        // A cap above the sigmas padding changes nothing: 95 + 33.17 -> 129.
        let mut loose = capped(1, 0.5, 2.0, 1.0);
        feed(&mut loose);
        assert_eq!(loose.estimate(S), Micros::new(129));
        // A cap of 0.1 of the mean pads by 9.5: ceil(104.5) = 105.
        let mut tight = capped(1, 0.5, 2.0, 0.1);
        feed(&mut tight);
        assert_eq!(tight.estimate(S), Micros::new(105));
        // A cap of zero is the mean.
        let mut none = capped(1, 0.5, 2.0, 0.0);
        feed(&mut none);
        assert_eq!(none.estimate(S), Micros::new(95));
    }

    #[test]
    fn quantile_reads_the_nearest_rank_of_the_window() {
        let feed = |m: &mut CostModel| {
            for x in [50, 10, 40, 20, 30] {
                m.record(S, Micros::new(x));
            }
        };
        // Sorted: 10 20 30 40 50. Rank = ceil(q * 5).
        for (q, expected) in [(0.0, 10), (0.2, 10), (0.5, 30), (0.9, 50), (1.0, 50)] {
            let mut m = quantile(1, 8, q);
            feed(&mut m);
            assert_eq!(m.estimate(S), Micros::new(expected), "quantile {q}");
        }
    }

    #[test]
    fn a_whole_rank_is_not_pushed_up_by_rounding() {
        // 0.55 * 100 is a rounding error above 55.
        assert_eq!(nearest_rank(0.55, 100), 55);
        assert_eq!(nearest_rank(0.9, 32), 29);
        assert_eq!(nearest_rank(0.0, 5), 1);
        assert_eq!(nearest_rank(1.0, 5), 5);
    }

    #[test]
    fn quantile_forgets_what_leaves_the_window() {
        let mut m = quantile(1, 3, 1.0);
        for x in [900, 10, 20] {
            m.record(S, Micros::new(x));
        }
        assert_eq!(m.estimate(S), Micros::new(900));
        // The fourth sample evicts the oldest, the 900.
        m.record(S, Micros::new(30));
        assert_eq!(m.estimate(S), Micros::new(30));
        m.record(S, Micros::new(5));
        assert_eq!(m.estimate(S), Micros::new(30));
        m.record(S, Micros::new(6));
        m.record(S, Micros::new(7));
        assert_eq!(m.estimate(S), Micros::new(7));
        assert_eq!(m.samples(S), 7);
    }

    #[test]
    fn a_window_of_one_tracks_the_last_sample() {
        let mut m = quantile(1, 1, 0.9);
        m.record(S, Micros::new(100));
        m.record(S, Micros::new(400));
        assert_eq!(m.estimate(S), Micros::new(400));
    }

    #[test]
    fn windows_are_per_slice() {
        let mut m = quantile(1, 2, 1.0);
        m.record(S, Micros::new(100));
        m.record(SliceId::new(8), Micros::new(5));
        assert_eq!(m.estimate(S), Micros::new(100));
        assert_eq!(m.estimate(SliceId::new(8)), Micros::new(5));
    }

    #[test]
    fn estimates_change_only_through_record() {
        for mut m in [
            mean_sigmas(1, 0.2, 2.0),
            quantile(1, 8, 0.9),
            capped(1, 0.2, 2.0, 0.5),
        ] {
            m.record(S, Micros::new(100));
            m.record(S, Micros::new(300));
            let before = m.estimate(S);
            assert_eq!(m.estimate(S), before);
            assert_eq!(m.clone().estimate(S), before);
        }
    }

    #[test]
    fn identical_measurements_give_identical_estimates() {
        let estimators = [
            Estimator::MeanPlusSigmas {
                alpha: 0.2,
                sigmas: 2.0,
            },
            Estimator::WindowedQuantile {
                window: 6,
                quantile: 0.9,
            },
            Estimator::CappedDeviation {
                alpha: 0.2,
                sigmas: 2.0,
                cap: 0.5,
            },
        ];
        for e in estimators {
            let mut a = CostModel::new(Micros::new(9), e).unwrap();
            let mut b = CostModel::new(Micros::new(9), e).unwrap();
            for i in 0..40_u64 {
                let x = Micros::new(100 + (i * 37) % 11 * 13);
                a.record(S, x);
                b.record(S, x);
                assert_eq!(a.estimate(S), b.estimate(S));
            }
        }
    }

    /// Measurements with a bulk near 1000 us and one slow outlier in ten, as a
    /// slice sees under competing GPU load.
    fn noisy(i: u64) -> Micros {
        if i % 10 == 9 {
            Micros::new(8_000)
        } else {
            Micros::new(1_000 + (i % 5) * 10)
        }
    }

    /// The largest estimate seen after the first 30 measurements.
    fn peak_estimate(mut m: CostModel) -> Micros {
        let mut peak = Micros::ZERO;
        for i in 0..200 {
            m.record(S, noisy(i));
            if i >= 30 {
                peak = peak.max(m.estimate(S));
            }
        }
        peak
    }

    #[test]
    fn noise_inflates_mean_plus_sigmas_but_not_the_robust_estimators() {
        let inflated = peak_estimate(mean_sigmas(1, 0.05, 2.0));
        let windowed = peak_estimate(quantile(1, 20, 0.9));
        let bounded = peak_estimate(capped(1, 0.05, 2.0, 0.5));
        // The bulk is 1000 to 1040 us. Two slow outliers in a window of 20 are
        // the top 10 %, so the 0.9 quantile stays in the bulk.
        assert!(windowed <= Micros::new(1_040), "windowed {windowed:?}");
        // The mean includes the outliers (the samples average 1 700 us), and
        // the padding is at most half of it, so the estimate is bounded by
        // about 3 000 us however large the deviation grows.
        assert!(bounded <= Micros::new(3_000), "capped {bounded:?}");
        // Mean plus two deviations is several times the bulk.
        assert!(inflated >= Micros::new(6_000), "mean+sigmas {inflated:?}");
        assert!(windowed < bounded && bounded < inflated);
    }

    fn frame_scaled(prior: u64, window: usize, frame_window: usize, q: f64) -> CostModel {
        CostModel::new(
            Micros::new(prior),
            Estimator::FrameScaled {
                window,
                frame_window,
                frame_quantile: q,
            },
        )
        .unwrap()
    }

    /// Records frames of `(predicted, measured)` microseconds.
    fn frames(m: &mut CostModel, frames: &[(u64, u64)]) {
        for &(predicted, measured) in frames {
            m.record_frame(Micros::new(predicted), Micros::new(measured));
        }
    }

    #[test]
    fn invalid_frame_scaled_parameters_are_rejected() {
        let e = |window, frame_window, frame_quantile| Estimator::FrameScaled {
            window,
            frame_window,
            frame_quantile,
        };
        for window in [0, MAX_WINDOW + 1] {
            assert_eq!(rejected(e(window, 8, 0.9)), CostError::InvalidWindow);
        }
        for frame_window in [0, MAX_WINDOW + 1] {
            assert_eq!(
                rejected(e(8, frame_window, 0.9)),
                CostError::InvalidFrameWindow
            );
        }
        for q in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
            assert_eq!(rejected(e(8, 8, q)), CostError::InvalidQuantile, "{q}");
        }
        for (window, frame_window, q) in [(1, 1, 0.0), (MAX_WINDOW, MAX_WINDOW, 1.0)] {
            assert!(CostModel::new(Micros::ZERO, e(window, frame_window, q)).is_ok());
        }
    }

    #[test]
    fn frame_scaled_estimates_the_median_and_warms_up_at_the_frame_quantile() {
        let mut m = frame_scaled(123, 8, 3, 0.9);
        // Unmeasured: the prior, which the frame scale never touches.
        assert_eq!(m.estimate(S), Micros::new(123));
        assert_eq!(m.planning_estimate(S), Micros::new(123));
        for x in [50, 10, 40, 20, 30] {
            m.record(S, Micros::new(x));
        }
        // Sorted 10 20 30 40 50: the median is rank 3, the 0.9 rank 5.
        assert_eq!(m.estimate(S), Micros::new(30));
        assert_eq!(
            m.frame_scale(),
            FrameScale::WarmingUp {
                recorded: 0,
                needed: 3
            }
        );
        assert_eq!(m.planning_estimate(S), Micros::new(50));
        frames(&mut m, &[(100, 150), (100, 150)]);
        assert_eq!(
            m.frame_scale(),
            FrameScale::WarmingUp {
                recorded: 2,
                needed: 3
            }
        );
        assert_eq!(m.planning_estimate(S), Micros::new(50));
        // The third frame fills the window: from here on the median is scaled.
        frames(&mut m, &[(100, 150)]);
        assert_eq!(m.frame_scale(), FrameScale::Measured(1.5));
        assert_eq!(m.planning_estimate(S), Micros::new(45));
        assert_eq!(m.estimate(S), Micros::new(30));
        // Another slice still at its prior is not scaled.
        assert_eq!(m.planning_estimate(SliceId::new(1)), Micros::new(123));
        assert_eq!(m.frames_recorded(), 3);
    }

    #[test]
    fn the_frame_scale_is_the_nearest_rank_of_the_last_frames() {
        // Equal weights: the ratios 1.1 1.2 1.5 2.0, the 0.75 rank is the
        // third, as `nearest_rank` would pick.
        let mut m = frame_scaled(1, 4, 4, 0.75);
        frames(&mut m, &[(100, 150), (100, 120), (100, 200), (100, 110)]);
        assert_eq!(m.frame_scale(), FrameScale::Measured(1.5));
        // A fifth frame evicts the oldest (1.5): 1.1 1.2 1.3 2.0.
        frames(&mut m, &[(100, 130)]);
        assert_eq!(m.frame_scale(), FrameScale::Measured(1.3));
        // The 0 quantile is the smallest, the 1 quantile the largest.
        for (q, expected) in [(0.0, 1.1), (1.0, 2.0)] {
            let mut m = frame_scaled(1, 4, 4, q);
            frames(&mut m, &[(100, 150), (100, 120), (100, 200), (100, 110)]);
            assert_eq!(m.frame_scale(), FrameScale::Measured(expected), "{q}");
        }
        // A whole rank is not pushed up by rounding: 0.55 of 100 is the 55th.
        let mut m = frame_scaled(1, 4, 100, 0.55);
        let ratios: Vec<(u64, u64)> = (1..=100).map(|i| (100, 100 + i)).collect();
        frames(&mut m, &ratios);
        assert_eq!(m.frame_scale(), FrameScale::Measured(1.55));
    }

    #[test]
    fn the_planning_estimate_is_the_median_times_the_scale_rounded_up() {
        let mut m = frame_scaled(1, 5, 1, 1.0);
        for x in [101; 5] {
            m.record(S, Micros::new(x));
        }
        frames(&mut m, &[(1_000, 1_234)]);
        // 101 * 1.234 = 124.634, rounded up.
        assert_eq!(m.planning_estimate(S), Micros::new(125));
        assert!(m.planning_estimate(S) >= m.estimate(S));
        // A whole product is not pushed up by rounding: 100 * 1.1 is 110.
        let mut m = frame_scaled(1, 5, 1, 1.0);
        for x in [100; 5] {
            m.record(S, Micros::new(x));
        }
        frames(&mut m, &[(1_000, 1_100)]);
        assert_eq!(m.planning_estimate(S), Micros::new(110));
    }

    #[test]
    fn the_frame_scale_never_plans_a_slice_below_its_median() {
        let mut m = frame_scaled(1, 8, 4, 0.9);
        for x in [200; 5] {
            m.record(S, Micros::new(x));
        }
        // Every frame measured half its prediction.
        frames(&mut m, &[(400, 200); 4]);
        assert_eq!(m.frame_scale(), FrameScale::Measured(1.0));
        assert_eq!(m.planning_estimate(S), m.estimate(S));
        assert_eq!(m.planning_estimate(S), Micros::new(200));
    }

    #[test]
    fn a_frame_counts_as_much_as_its_predicted_time() {
        // Twenty full frames at 1.2, then five frames of one 20 us slice
        // whose pass overhead made them 10 times their prediction. Counted
        // as frames, a fifth of the window would put the 0.9 rank at 10;
        // weighted, the small frames are half a percent of the window's
        // predicted time, and the scale stays at the full frames'.
        let mut m = frame_scaled(1, 4, 25, 0.9);
        frames(&mut m, &[(4_000, 4_800); 20]);
        frames(&mut m, &[(20, 200); 5]);
        assert_eq!(m.frame_scale(), FrameScale::Measured(1.2));
        assert_eq!(m.frames_recorded(), 25);
        // Unweighted, the same window would read 10.
        let mut unweighted = frame_scaled(1, 4, 25, 0.9);
        frames(&mut unweighted, &[(100, 120); 20]);
        frames(&mut unweighted, &[(100, 1_000); 5]);
        assert_eq!(unweighted.frame_scale(), FrameScale::Measured(10.0));
    }

    #[test]
    fn a_frame_predicted_at_zero_is_not_recorded() {
        let mut m = frame_scaled(1, 4, 2, 0.9);
        frames(&mut m, &[(0, 500), (0, 0)]);
        assert_eq!(m.frames_recorded(), 0);
        assert_eq!(
            m.frame_scale(),
            FrameScale::WarmingUp {
                recorded: 0,
                needed: 2
            }
        );
        // A frame measured at zero is a ratio of zero, floored to 1.
        frames(&mut m, &[(100, 0), (100, 0)]);
        assert_eq!(m.frame_scale(), FrameScale::Measured(1.0));
    }

    #[test]
    fn the_other_estimators_ignore_frames_and_plan_at_their_estimate() {
        for mut m in [
            mean_sigmas(1, 0.2, 2.0),
            quantile(1, 8, 0.9),
            capped(1, 0.2, 2.0, 0.5),
        ] {
            for x in [100, 300, 200, 900] {
                m.record(S, Micros::new(x));
            }
            let before = m.clone();
            frames(&mut m, &[(100, 300); 50]);
            assert_eq!(m.frame_scale(), FrameScale::None);
            assert_eq!(m.frames_recorded(), 0);
            assert_eq!(m.estimate(S), before.estimate(S));
            assert_eq!(m.planning_estimate(S), m.estimate(S));
            assert_eq!(m.planning_estimate(SliceId::new(1)), Micros::new(1));
        }
    }

    #[test]
    fn calibration_does_not_seed_the_frame_scaled_median() {
        let mut m = frame_scaled(1, 32, 1, 0.9);
        // Calibration measured the slice at 100 us, back to back.
        for _ in 0..20 {
            m.record_calibration(S, Micros::new(100));
        }
        assert_eq!(m.samples(S), 20);
        assert_eq!(m.frame_estimate(S), None);
        // Cold, it is planned as the default would plan it: the 0.9 quantile
        // of everything measured, calibration included.
        assert_eq!(m.planning_estimate(S), Micros::new(100));
        assert_eq!(m.estimate(S), Micros::new(100));
        m.record_frame(Micros::new(100), Micros::new(150));
        assert_eq!(m.frame_scale(), FrameScale::Measured(1.5));
        // The loop measures it at 300. Four loop measurements leave it cold,
        // planned at the quantile of all 24, still unscaled.
        for _ in 0..4 {
            m.record(S, Micros::new(300));
        }
        assert_eq!(m.frame_estimate(S), None);
        assert_eq!(m.planning_estimate(S), Micros::new(300));
        // The fifth warms it: its median is the loop's alone, 300, where the
        // median of all 25 would still be calibration's 100; scaled by 1.5.
        m.record(S, Micros::new(300));
        assert_eq!(m.frame_estimate(S), Some(Micros::new(300)));
        assert_eq!(m.estimate(S), Micros::new(300));
        assert_eq!(m.planning_estimate(S), Micros::new(450));
        assert_eq!(m.samples(S), 25);
    }

    #[test]
    fn the_frame_scale_reads_warm_slices_and_not_stale_ones() {
        // Slice A is warm at 100 us. Slice B ran in the loop twice at 200 us
        // and calibration measured it at 50. Every frame runs both and
        // measures each at its loop time, so frames are not slow at all.
        let a = SliceId::new(1);
        let b = SliceId::new(2);
        let mut m = frame_scaled(1, 32, 8, 0.9);
        for _ in 0..WARM_SAMPLES {
            m.record(a, Micros::new(100));
        }
        for _ in 0..10 {
            m.record_calibration(b, Micros::new(50));
        }
        m.record(b, Micros::new(200));
        m.record(b, Micros::new(200));
        for _ in 0..8 {
            // As a host does: the frame's ratio over the slices with a frame
            // estimate, read when the frame is planned.
            let mut predicted = Micros::ZERO;
            let mut measured = Micros::ZERO;
            for (slice, took) in [(a, 100), (b, 200)] {
                if let Some(planned) = m.frame_estimate(slice) {
                    predicted = predicted.saturating_add(planned);
                    measured = measured.saturating_add(Micros::new(took));
                }
            }
            m.record_frame(predicted, measured);
        }
        assert_eq!(m.frame_scale(), FrameScale::Measured(1.0));
        assert_eq!(m.planning_estimate(a), Micros::new(100));
        // Taken at B's stale median of all its measurements the same frames
        // would have read (100 + 200) / (100 + 50) = 2 and doubled A.
        assert_eq!(m.estimate(b), Micros::new(50));
    }

    #[test]
    fn the_other_estimators_fold_calibration_as_the_loop() {
        let estimators = [
            Estimator::MeanPlusSigmas {
                alpha: 0.2,
                sigmas: 2.0,
            },
            Estimator::WindowedQuantile {
                window: 6,
                quantile: 0.9,
            },
            Estimator::CappedDeviation {
                alpha: 0.2,
                sigmas: 2.0,
                cap: 0.5,
            },
        ];
        for e in estimators {
            let mut by_record = CostModel::new(Micros::new(9), e).unwrap();
            let mut by_calibration = CostModel::new(Micros::new(9), e).unwrap();
            for i in 0..40_u64 {
                let x = Micros::new(100 + (i * 37) % 11 * 13);
                by_record.record(S, x);
                if i < 20 {
                    by_calibration.record_calibration(S, x);
                } else {
                    by_calibration.record(S, x);
                }
                assert_eq!(by_record.estimate(S), by_calibration.estimate(S));
                assert_eq!(
                    by_calibration.frame_estimate(S),
                    Some(by_calibration.estimate(S))
                );
            }
            assert_eq!(by_record.samples(S), by_calibration.samples(S));
            assert_eq!(
                by_calibration.frame_estimate(SliceId::new(3)),
                Some(Micros::new(9))
            );
        }
        // The frame-scaled one leaves an unmeasured slice out of the ratio.
        assert_eq!(frame_scaled(9, 4, 4, 0.9).frame_estimate(S), None);
    }

    #[test]
    fn frame_scaled_is_deterministic_and_changes_only_through_records() {
        let e = Estimator::FrameScaled {
            window: 6,
            frame_window: 5,
            frame_quantile: 0.9,
        };
        let mut a = CostModel::new(Micros::new(9), e).unwrap();
        let mut b = CostModel::new(Micros::new(9), e).unwrap();
        for i in 0..40_u64 {
            let x = Micros::new(100 + (i * 37) % 11 * 13);
            let measured = Micros::new(1_000 + (i * 53) % 17 * 31);
            for m in [&mut a, &mut b] {
                m.record(S, x);
                m.record_frame(Micros::new(1_000), measured);
            }
            assert_eq!(a.planning_estimate(S), b.planning_estimate(S));
            assert_eq!(a.frame_scale(), b.frame_scale());
            assert_eq!(a.clone().planning_estimate(S), a.planning_estimate(S));
        }
    }
}
