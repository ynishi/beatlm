//! Order statistics and histograms over the samples a run collects.

use std::collections::BTreeMap;

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

/// The `percent`th percentile of an ascending slice, by nearest rank.
fn nearest_rank(sorted: &[u64], percent: usize) -> Option<u64> {
    if sorted.is_empty() {
        return None;
    }
    let rank = (sorted.len() * percent).div_ceil(100).max(1);
    sorted.get(rank - 1).copied()
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
    fn histogram_counts_in_value_order() {
        let h = histogram([2, 2, 1, 2, 3]);
        assert_eq!(
            h.into_iter().collect::<Vec<_>>(),
            vec![(1, 1), (2, 3), (3, 1)]
        );
    }
}
