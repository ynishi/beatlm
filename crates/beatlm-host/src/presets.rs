//! The values both hosts plan and count with: the cost model's estimators and
//! priors, the margin, and the windows and frame counts derived from the
//! default estimator's window.
//!
//! A value here is the same in a native run and in a page, so that their
//! reports compare line by line ([the crate documentation](crate)). Each host
//! parses its own text for a choice — the bench's `--estimator`, the page's
//! `estimator=` with its optional parameters — and falls back to these for
//! what the text leaves out; what a bare name means is the preset function of
//! that name ([`quantile`], [`mean_sigmas`], [`capped`], [`frame_scaled`]).

use beatlm_core::cost::Estimator;
use beatlm_core::time::Micros;

/// Measurements per slice the windowed quantile reads, about half a second of
/// frames at 60 Hz: the window of the core's default estimator
/// (`Estimator::default()`). The core writes it as a literal inside
/// `Default`, which a constant cannot read, so a test holds the two equal.
pub const WINDOW: usize = 32;

/// The quantile of the window the windowed estimate is: the core's default's.
pub const QUANTILE: f64 = 0.9;

/// Weight of a new measurement in the exponentially weighted mean of
/// [`mean_sigmas`] and [`capped`]: the value the core's estimators were
/// compared with.
pub const ALPHA: f64 = 0.2;

/// Standard deviations [`mean_sigmas`] and [`capped`] add to the mean.
pub const SIGMAS: f64 = 2.0;

/// The most [`capped`] pads the mean by, as a multiple of the mean.
pub const CAP: f64 = 0.5;

/// The frames whose ratio [`frame_scaled`] reads its frame scale from;
/// `beatlm_core::cost` ("Planning a frame against its total") says why 64.
pub const FRAME_WINDOW: usize = 64;

/// The quantile of the frame ratios [`frame_scaled`] reads: [`QUANTILE`],
/// for the reason the same section gives.
pub const FRAME_QUANTILE: f64 = QUANTILE;

/// What a slice is assumed to cost until it has been measured.
pub const SLICE_PRIOR: Micros = Micros::new(2_000);

/// What the render pass is assumed to cost until it has been measured.
pub const RENDER_PRIOR: Micros = Micros::new(2_000);

/// How many recent render measurements the budget forecast takes its median
/// over: [`WINDOW`], so the forecast and the estimates it is compared with
/// look back as far.
pub const RENDER_WINDOW: usize = WINDOW;

/// The margin kept free in each frame's budget, in microseconds, unless a
/// host's user chooses another.
pub const MARGIN_US: u64 = 2_000;

/// The first frames of the loop, counted from its first frame, whose jobs'
/// misses are reported apart as `startup_missed`: twice [`WINDOW`], so a
/// cascade of misses while the estimates are new shows apart from misses in
/// steady state.
pub const STARTUP_FRAMES: u64 = 2 * WINDOW as u64;

/// Timed frames of whole jobs in each calibration pass, after one untimed
/// frame: twice [`WINDOW`], so the half the decision reads is as long as the
/// window the loop's estimate reads, and starts after the GPU has run the
/// paced load for half a second at 60 Hz (a quarter at 120 Hz). An estimator with a longer window starts the
/// loop with the counted half ([`counted`](crate::calibration::counted)),
/// `CALIBRATION_FRAMES / 2` measurements of each
/// part, and fills the rest from the loop.
pub const CALIBRATION_FRAMES: u64 = 2 * WINDOW as u64;

/// The estimator a bare `quantile` names: the core's default.
#[must_use]
pub const fn quantile() -> Estimator {
    Estimator::WindowedQuantile {
        window: WINDOW,
        quantile: QUANTILE,
    }
}

/// The estimator a bare `mean-sigmas` names.
#[must_use]
pub const fn mean_sigmas() -> Estimator {
    Estimator::MeanPlusSigmas {
        alpha: ALPHA,
        sigmas: SIGMAS,
    }
}

/// The estimator a bare `capped` names.
#[must_use]
pub const fn capped() -> Estimator {
    Estimator::CappedDeviation {
        alpha: ALPHA,
        sigmas: SIGMAS,
        cap: CAP,
    }
}

/// The estimator a bare `frame-scaled` names.
#[must_use]
pub const fn frame_scaled() -> Estimator {
    Estimator::FrameScaled {
        window: WINDOW,
        frame_window: FRAME_WINDOW,
        frame_quantile: FRAME_QUANTILE,
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use beatlm_core::cost::CostModel;

    use super::*;

    #[test]
    fn the_quantile_preset_is_the_cores_default() {
        assert_eq!(quantile(), Estimator::default());
    }

    /// The values the hosts' recorded measurements were taken with; a change
    /// to one changes both hosts' runs and reports, and is a decision of its
    /// own rather than an edit here.
    #[test]
    fn the_values_are_the_ones_both_hosts_ran_with() {
        assert_eq!((WINDOW, RENDER_WINDOW, FRAME_WINDOW), (32, 32, 64));
        assert_eq!((STARTUP_FRAMES, CALIBRATION_FRAMES), (64, 64));
        assert_eq!((QUANTILE, FRAME_QUANTILE), (0.9, 0.9));
        assert_eq!((ALPHA, SIGMAS, CAP), (0.2, 2.0, 0.5));
        assert_eq!(
            (SLICE_PRIOR.get(), RENDER_PRIOR.get(), MARGIN_US),
            (2_000, 2_000, 2_000)
        );
    }

    #[test]
    fn the_core_accepts_every_preset() {
        for preset in [quantile(), mean_sigmas(), capped(), frame_scaled()] {
            assert!(CostModel::new(SLICE_PRIOR, preset).is_ok(), "{preset:?}");
        }
    }
}
