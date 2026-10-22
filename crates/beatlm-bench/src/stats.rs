//! Order statistics and histograms over the samples a run collects.
//!
//! The browser host (`beatlm-web`) compiles this file by path, so its report
//! and its calibration quote the same statistics as the bench's.

use std::collections::BTreeMap;

use beatlm_core::cost::FrameScale;
use beatlm_core::time::Micros;

/// The median, the 99th percentile and the maximum of a set of samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Summary {
    /// The 50th percentile.
    pub p50: u64,
    /// The 99th percentile.
    pub p99: u64,
    /// The largest sample.
    pub max: u64,
}

/// Summarises `samples`, or `None` when there are none.
///
/// Percentiles use the nearest-rank method: the smallest sample such that at
/// least `p` percent of the samples are at or below it. It returns a value that
/// was observed, which is what a latency report should quote.
pub fn summarize(samples: &[u64]) -> Option<Summary> {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    Some(Summary {
        p50: nearest_rank(&sorted, 50)?,
        p99: nearest_rank(&sorted, 99)?,
        max: *sorted.last()?,
    })
}

/// The `percent`th percentile of an ascending slice, by nearest rank; `None`
/// when it is empty.
pub fn nearest_rank<T: Copy>(sorted: &[T], percent: usize) -> Option<T> {
    let rank = (sorted.len() * percent).div_ceil(100).max(1);
    sorted.get(rank - 1).copied()
}

/// The middle of `samples` once sorted, the upper one when there are two;
/// `None` when there are none.
///
/// It is the statistic calibration's decision reads (`calibrate.rs`): it
/// ignores a few slow frames at either end without a window, a deviation or a
/// quantile to choose.
pub fn median(samples: &mut [u64]) -> Option<Micros> {
    samples.sort_unstable();
    samples.get(samples.len() / 2).copied().map(Micros::new)
}

/// How many times each value occurs, in ascending value order.
pub fn histogram<I: IntoIterator<Item = u64>>(values: I) -> BTreeMap<u64, u64> {
    let mut counts = BTreeMap::new();
    for v in values {
        *counts.entry(v).or_insert(0) += 1;
    }
    counts
}

/// A frame's measured slice time over what its slices were planned at, the
/// ratio the frame-scaled estimate reads; `None` when they were planned at
/// zero.
// The browser host calls it from its wasm32-only frame loop, so a native
// build of `beatlm-web` compiles it unused.
#[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
#[allow(clippy::cast_precision_loss)]
pub fn frame_ratio(predicted: Micros, measured: Micros) -> Option<f64> {
    // Microsecond counts this size are exact in an f64.
    (predicted > Micros::ZERO).then(|| measured.get() as f64 / predicted.get() as f64)
}

/// ` min=… p50=… p90=… max=…` of `samples`, to two decimals, by nearest rank
/// as [`summarize`]; empty when there are none.
fn ratio_summary(samples: &[f64]) -> String {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    match (
        sorted.first(),
        nearest_rank(&sorted, 50),
        nearest_rank(&sorted, 90),
        sorted.last(),
    ) {
        (Some(min), Some(p50), Some(p90), Some(max)) => {
            format!(" min={min:.2} p50={p50:.2} p90={p90:.2} max={max:.2}")
        }
        _ => String::new(),
    }
}

/// The value of the report's `frame_ratio` line: over the timed frames, the
/// ratio of the measured time of their slices with a frame estimate
/// (`CostModel::frame_estimate` when the frame was planned) to the sum of
/// those estimates, for the frames where that sum is above zero; `none` when
/// there were none. Every slice has one under every estimator but the
/// frame-scaled one, where only warm slices do, and these are the ratios its
/// frame scale is a quantile of; under another, how far the frames' measured
/// time fell from what was planned.
pub fn frame_ratio_value(ratios: &[f64]) -> String {
    if ratios.is_empty() {
        return "none".to_owned();
    }
    format!("frames={}{}", ratios.len(), ratio_summary(ratios))
}

/// The value of the report's `frame_scale` line: `none` for an estimator
/// without a frame scale; otherwise the scale when the run ended (`final`,
/// `warming_up` while the frame window was not yet full), the frames the
/// cost model recorded, and the plans made with a measured scale
/// (`planned_scaled`) with the spread of the scale over them.
pub fn frame_scale_value(last: FrameScale, recorded: u64, planned: &[f64]) -> String {
    let last = match last {
        FrameScale::None => return "none".to_owned(),
        FrameScale::WarmingUp { .. } => "warming_up".to_owned(),
        FrameScale::Measured(scale) => format!("{scale:.2}"),
    };
    format!(
        "final={last} frames_recorded={recorded} planned_scaled={}{}",
        planned.len(),
        ratio_summary(planned)
    )
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn no_samples_have_no_summary() {
        assert_eq!(summarize(&[]), None);
    }

    #[test]
    fn nearest_rank_returns_an_observed_sample() {
        let samples: Vec<u64> = (1..=100).rev().collect();
        let s = summarize(&samples).unwrap();
        assert_eq!((s.p50, s.p99, s.max), (50, 99, 100));
    }

    #[test]
    fn a_single_sample_is_every_percentile() {
        let s = summarize(&[7]).unwrap();
        assert_eq!((s.p50, s.p99, s.max), (7, 7, 7));
    }

    #[test]
    fn small_sets_round_the_rank_up() {
        let s = summarize(&[10, 20, 30]).unwrap();
        assert_eq!((s.p50, s.p99, s.max), (20, 30, 30));
    }

    #[test]
    fn the_median_is_the_middle_sample_and_the_upper_of_two() {
        assert_eq!(median(&mut []), None);
        assert_eq!(median(&mut [7]), Some(Micros::new(7)));
        assert_eq!(median(&mut [9, 1, 5]), Some(Micros::new(5)));
        assert_eq!(median(&mut [4, 1, 3, 2]), Some(Micros::new(3)));
    }

    #[test]
    fn the_median_ignores_slow_frames_that_move_a_mean() {
        // Three early under-clocked frames and one slow one among 28 steady.
        let mut samples = vec![1_000; 28];
        samples.extend([9_000, 9_000, 9_000, 20_000]);
        assert_eq!(median(&mut samples), Some(Micros::new(1_000)));
    }

    #[test]
    fn a_frame_ratio_needs_a_prediction() {
        assert_eq!(frame_ratio(Micros::ZERO, Micros::new(5)), None);
        assert_eq!(frame_ratio(Micros::new(400), Micros::new(500)), Some(1.25));
    }

    #[test]
    fn the_frame_lines_say_none_and_the_spread() {
        assert_eq!(frame_ratio_value(&[]), "none");
        assert_eq!(
            frame_ratio_value(&[1.5, 1.0, 2.0, 1.25]),
            "frames=4 min=1.00 p50=1.25 p90=2.00 max=2.00"
        );
        assert_eq!(frame_scale_value(FrameScale::None, 9, &[1.0]), "none");
        assert_eq!(
            frame_scale_value(
                FrameScale::WarmingUp {
                    recorded: 3,
                    needed: 64
                },
                3,
                &[]
            ),
            "final=warming_up frames_recorded=3 planned_scaled=0"
        );
        assert_eq!(
            frame_scale_value(FrameScale::Measured(1.234), 70, &[1.0, 1.234]),
            "final=1.23 frames_recorded=70 planned_scaled=2 min=1.00 p50=1.00 p90=1.23 max=1.23"
        );
    }

    #[test]
    fn histogram_counts_in_value_order() {
        let h = histogram([2, 2, 1, 2, 3]);
        assert_eq!(
            h.into_iter().collect::<Vec<_>>(),
            vec![(1, 1), (2, 3), (3, 1)]
        );
    }
}
