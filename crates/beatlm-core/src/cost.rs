//! Measured slice durations, as an estimate the scheduler can plan with.
//!
//! The cost model is updated only from measurements. Nothing else, such as a
//! plan's shape or a guess, feeds it; until a slice has been measured the
//! configured prior stands in.

use std::collections::BTreeMap;

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
}

#[derive(Debug, Clone, Copy)]
struct Stats {
    mean: f64,
    var: f64,
    samples: u32,
}

/// Per-slice exponentially weighted mean and variance of measured durations.
///
/// The scheduler budgets with `mean + sigmas * stddev`, so a slice that is
/// usually fast but sometimes slow is planned for its slow side. Slices are
/// kept in an ordered map so that identical measurements always give identical
/// estimates.
#[derive(Debug, Clone)]
pub struct CostModel {
    prior: Micros,
    alpha: f64,
    sigmas: f64,
    stats: BTreeMap<SliceId, Stats>,
}

impl CostModel {
    /// A model that estimates `prior` for unmeasured slices, smooths new
    /// measurements with weight `alpha`, and pads estimates by `sigmas`
    /// standard deviations.
    ///
    /// # Errors
    ///
    /// [`CostError::InvalidAlpha`] unless `alpha` is finite and in `(0, 1]`;
    /// [`CostError::InvalidSigmas`] unless `sigmas` is finite and not negative.
    pub fn new(prior: Micros, alpha: f64, sigmas: f64) -> Result<Self, CostError> {
        if !(alpha.is_finite() && alpha > 0.0 && alpha <= 1.0) {
            return Err(CostError::InvalidAlpha);
        }
        if !(sigmas.is_finite() && sigmas >= 0.0) {
            return Err(CostError::InvalidSigmas);
        }
        Ok(Self {
            prior,
            alpha,
            sigmas,
            stats: BTreeMap::new(),
        })
    }

    /// Folds one measured duration of `slice` into its statistics.
    ///
    /// The first sample sets the mean and leaves the variance at zero; later
    /// samples move the mean by `alpha` of the difference and decay the
    /// variance accordingly.
    pub fn record(&mut self, slice: SliceId, measured: Micros) {
        let x = micros_to_f64(measured);
        let alpha = self.alpha;
        self.stats
            .entry(slice)
            .and_modify(|s| {
                let diff = x - s.mean;
                let incr = alpha * diff;
                s.mean += incr;
                s.var = (1.0 - alpha) * (s.var + diff * incr);
                s.samples = s.samples.saturating_add(1);
            })
            .or_insert(Stats {
                mean: x,
                var: 0.0,
                samples: 1,
            });
    }

    /// The duration to budget for `slice`: the prior until it has been
    /// measured, afterwards `ceil(mean + sigmas * stddev)`.
    #[must_use]
    pub fn estimate(&self, slice: SliceId) -> Micros {
        match self.stats.get(&slice) {
            None => self.prior,
            Some(s) => f64_to_micros_ceil(self.sigmas.mul_add(s.var.max(0.0).sqrt(), s.mean)),
        }
    }

    /// How many measurements of `slice` the model has seen, saturating.
    #[must_use]
    pub fn samples(&self, slice: SliceId) -> u32 {
        self.stats.get(&slice).map_or(0, |s| s.samples)
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

    #[test]
    fn invalid_alpha_is_rejected() {
        for alpha in [0.0, -0.5, 1.5, f64::NAN, f64::INFINITY] {
            assert_eq!(
                CostModel::new(Micros::ZERO, alpha, 1.0).unwrap_err(),
                CostError::InvalidAlpha,
                "alpha {alpha}"
            );
        }
    }

    #[test]
    fn invalid_sigmas_is_rejected() {
        for sigmas in [-0.1, f64::NAN, f64::INFINITY] {
            assert_eq!(
                CostModel::new(Micros::ZERO, 0.5, sigmas).unwrap_err(),
                CostError::InvalidSigmas,
                "sigmas {sigmas}"
            );
        }
    }

    #[test]
    fn prior_is_used_without_samples() {
        let m = CostModel::new(Micros::new(123), 0.5, 2.0).unwrap();
        assert_eq!(m.estimate(S), Micros::new(123));
        assert_eq!(m.samples(S), 0);
    }

    #[test]
    fn alpha_one_tracks_the_last_sample() {
        let mut m = CostModel::new(Micros::new(1), 1.0, 3.0).unwrap();
        m.record(S, Micros::new(100));
        m.record(S, Micros::new(400));
        m.record(S, Micros::new(250));
        assert_eq!(m.estimate(S), Micros::new(250));
    }

    #[test]
    fn estimate_pads_the_mean_by_sigmas() {
        let mut m = CostModel::new(Micros::new(1), 0.5, 2.0).unwrap();
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
        let mut m = CostModel::new(Micros::new(1), 0.5, 0.0).unwrap();
        m.record(S, Micros::new(10));
        m.record(S, Micros::new(10));
        m.record(SliceId::new(8), Micros::new(10));
        assert_eq!(m.samples(S), 2);
        assert_eq!(m.samples(SliceId::new(8)), 1);
        assert_eq!(m.samples(SliceId::new(9)), 0);
    }

    #[test]
    fn huge_estimates_saturate() {
        let mut m = CostModel::new(Micros::ZERO, 1.0, 0.0).unwrap();
        m.record(S, Micros::new(u64::MAX));
        assert_eq!(m.estimate(S), Micros::new(u64::MAX));
    }
}
