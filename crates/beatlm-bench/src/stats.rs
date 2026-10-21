//! Order statistics and histograms over the samples a run collects.
//!
//! The browser host (`beatlm-web`) compiles this file by path, so its report
//! and its calibration quote the same statistics as the bench's.

use std::collections::BTreeMap;

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
    fn histogram_counts_in_value_order() {
        let h = histogram([2, 2, 1, 2, 3]);
        assert_eq!(
            h.into_iter().collect::<Vec<_>>(),
            vec![(1, 1), (2, 3), (3, 1)]
        );
    }
}
