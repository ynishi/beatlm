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
    /// The quantile must lie in `[0, 1]`.
    #[error("quantile must be finite and in [0, 1]")]
    InvalidQuantile,
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
/// when [`CostModel::record`] is called, and is deterministic.
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
                if !(quantile.is_finite() && (0.0..=1.0).contains(&quantile)) {
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
    samples: u32,
}

impl Window {
    fn new(capacity: usize) -> Self {
        Self {
            ring: Vec::with_capacity(capacity),
            next: 0,
            scratch: Vec::with_capacity(capacity),
            estimate: 0,
            samples: 0,
        }
    }

    fn fold(&mut self, x: u64, capacity: usize, quantile: f64) {
        if self.ring.len() < capacity {
            self.ring.push(x);
        } else if let Some(slot) = self.ring.get_mut(self.next) {
            *slot = x;
        }
        self.next = (self.next + 1) % capacity;
        self.samples = self.samples.saturating_add(1);

        self.scratch.clear();
        self.scratch.extend_from_slice(&self.ring);
        // Only the value at the rank is needed, so select it rather than sort
        // the whole window: linear rather than n log n per measurement.
        let rank = nearest_rank(quantile, self.scratch.len());
        self.estimate = if self.scratch.is_empty() {
            x
        } else {
            *self.scratch.select_nth_unstable(rank - 1).1
        };
    }
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

#[derive(Debug, Clone)]
enum State {
    Ewma(Ewma),
    Window(Window),
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
    /// unless the quantile is finite and in `[0, 1]`.
    pub fn new(prior: Micros, estimator: Estimator) -> Result<Self, CostError> {
        estimator.validate()?;
        Ok(Self {
            prior,
            estimator,
            stats: BTreeMap::new(),
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
    /// quantile the sample replaces the oldest one once the window is full.
    pub fn record(&mut self, slice: SliceId, measured: Micros) {
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
                    w.fold(measured.get(), window, quantile);
                }
            }
        }
    }

    /// The duration to budget for `slice`: the prior until it has been
    /// measured, afterwards what the [`Estimator`] reads from its
    /// measurements, rounded up to whole microseconds.
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
            _ => self.prior,
        }
    }

    /// How many measurements of `slice` the model has seen, saturating.
    #[must_use]
    pub fn samples(&self, slice: SliceId) -> u32 {
        match self.stats.get(&slice) {
            None => 0,
            Some(State::Ewma(s)) => s.samples,
            Some(State::Window(w)) => w.samples,
        }
    }
}

/// Microsecond counts beyond 2^53 lose low bits; at that size (centuries) the
/// error is irrelevant to a frame budget.
#[allow(clippy::cast_precision_loss)]
const fn micros_to_f64(m: Micros) -> f64 {
    m.get() as f64
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
}
