//! The statistics behind the report's `tail_*` lines: where the time of a
//! slice's slow measurements comes from.
//!
//! The parent module ([`super`], "Where the slow measurements come from") says
//! what each line is and which explanation it tests. This one computes them
//! from what the host recorded of each timed frame ([`TimedFrame`]) and from
//! calibration's measurements, and is pure, so each rule is tested natively;
//! [`analyse`] is the entry point. Nothing here feeds the scheduler or the
//! cost model.
//!
//! Percentiles are nearest-rank, by the same function as the report's other
//! percentiles (the bench's `stats.rs`), so every one is a value that was
//! observed. A slice's median is its 50th percentile by that rule.

use std::collections::BTreeMap;

use beatlm_core::cost::SliceId;

use crate::stats::nearest_rank as percentile;

/// A measurement is an outlier when it is more than this many times its
/// slice's median in the loop.
pub const OUTLIER_FACTOR: u64 = 2;

/// Slices measured fewer times than this in the loop are left out of every
/// per-slice statistic: a median of fewer says little about the slice.
pub const MIN_SAMPLES: usize = 8;

/// How many frames before a measurement's own its GPU busy fraction is read
/// over: four, 33 ms at 120 Hz, long enough to span a few frames with and
/// without slices. A clock that follows the load would follow it over some
/// such time; what Apple's does was not measured.
pub const BUSY_WINDOW: u64 = 4;

/// The upper bounds of the busy buckets, in percent of the window's time; the
/// last bucket has none.
pub const BUSY_BOUNDS: [u64; 3] = [25, 50, 75];

/// The largest gap before a frame's first compute pass, in nanoseconds, that
/// counts as the pass beginning right behind the previous frame's work: a
/// quarter of a millisecond, under four of the 65.5 µs steps Chrome's
/// timestamps were seen to come in.
pub const NEAR_GAP_NS: i64 = 250_000;

/// One timed pass: its duration as the cost model received it, and its
/// timestamps (`beatlm::gpu::PassSpan`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pass {
    /// The measured duration, in microseconds.
    pub us: u64,
    /// The beginning-of-pass timestamp, in nanoseconds.
    pub begin_ns: u64,
    /// The end-of-pass timestamp, in nanoseconds.
    pub end_ns: u64,
}

/// One timed frame's passes, in the order they were encoded: its compute
/// passes, then its render pass.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TimedFrame {
    /// The frame's index.
    pub frame: u64,
    /// Each compute pass with the slice it ran.
    pub compute: Vec<(SliceId, Pass)>,
    /// The render pass, when the frame timed one.
    pub render: Option<Pass>,
}

impl TimedFrame {
    /// The GPU time of every pass of the frame, in microseconds.
    #[must_use]
    pub fn gpu_us(&self) -> u64 {
        self.passes().map(|p| p.us).fold(0, u64::saturating_add)
    }

    /// The frame's passes in encoding order.
    fn passes(&self) -> impl Iterator<Item = &Pass> {
        self.compute
            .iter()
            .map(|(_, p)| p)
            .chain(self.render.iter())
    }

    /// The latest end of any of the frame's passes, in nanoseconds: where the
    /// frame's recorded work was over.
    fn end_ns(&self) -> Option<u64> {
        self.passes().map(|p| p.end_ns).max()
    }
}

/// The median and the 90th percentile of a set of ratios.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ratios {
    /// How many ratios.
    pub samples: usize,
    /// Their median.
    pub median: f64,
    /// Their 90th percentile.
    pub p90: f64,
}

impl Ratios {
    /// The ratios' median and 90th percentile; `None` when there are none.
    #[must_use]
    pub fn of(mut values: Vec<f64>) -> Option<Self> {
        values.sort_by(f64::total_cmp);
        Some(Self {
            samples: values.len(),
            median: percentile(&values, 50)?,
            p90: percentile(&values, 90)?,
        })
    }
}

/// How wide each slice's measurements spread, over the slices measured
/// [`MIN_SAMPLES`] times or more.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spread {
    /// How many slices.
    pub slices: usize,
    /// Each slice's 90th percentile over its median: their median and 90th
    /// percentile over the slices.
    pub p90_over_p50: Ratios,
    /// The median over the slices of each one's largest measurement over its
    /// median.
    pub max_over_p50: f64,
}

/// How much the timestamps' step can account for.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quantization {
    /// The step, in microseconds.
    pub step_us: f64,
    /// Slices whose 90th percentile is within one step of their median, so
    /// that their whole spread up to it could be the step's.
    pub within_step: usize,
    /// The median over the slices of one step over the slice's 90th
    /// percentile: the most the step can add to it, as a share of it.
    pub step_over_p90: Option<f64>,
    /// The median over the slices whose final estimate is above their mean of
    /// one step over that excess, capped at 1: the share of it the step can
    /// account for.
    pub step_over_excess: Option<f64>,
}

/// Each slice in the loop against the same part in calibration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Versus {
    /// How many slices both measured.
    pub slices: usize,
    /// The median over the slices of the loop's mean over calibration's.
    pub mean: f64,
    /// The median over the slices of the loop's median over calibration's.
    pub p50: f64,
}

/// Measurements above [`OUTLIER_FACTOR`] times their slice's median.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Outliers {
    /// Measurements of the slices the statistics are over.
    pub samples: u64,
    /// Of them, outliers.
    pub over: u64,
    /// Measurements that were their frame's first compute pass.
    pub first_samples: u64,
    /// Outliers that were their frame's first compute pass.
    pub first_over: u64,
}

/// Whether outliers come together in frames.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Clustering {
    /// Frames with two measurements or more of the slices the statistics are
    /// over.
    pub frames: u64,
    /// Of them, frames where half their measurements or more were outliers.
    pub at_least_half: u64,
    /// How many such frames there would be if each measurement were an
    /// outlier on its own, at the run's outlier share.
    pub expected: f64,
}

/// The measurements of frames whose previous [`BUSY_WINDOW`] frames kept the
/// GPU busy for a share of their time within one bucket.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Busy {
    /// The bucket's lower bound, in percent of the window's time.
    pub from_percent: u64,
    /// Its upper bound; `None` for the last.
    pub below_percent: Option<u64>,
    /// Measurements in it.
    pub samples: u64,
    /// Of them, outliers.
    pub over: u64,
    /// The median of each measurement over its slice's median; `None` when
    /// the bucket is empty.
    pub ratio: Option<f64>,
}

/// Where a frame's first compute pass began, against the end of the previous
/// frame's recorded work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gap {
    /// Before that work ended.
    Overlap,
    /// Within [`NEAR_GAP_NS`] of it.
    Near,
    /// Later.
    Idle,
    /// The previous frame was not timed.
    NoPrevious,
}

impl Gap {
    /// Every kind, in the order the report lists them.
    pub const ALL: [Self; 4] = [Self::Overlap, Self::Near, Self::Idle, Self::NoPrevious];

    /// The kind of a gap of `ns` nanoseconds, or of none.
    #[must_use]
    pub fn of(ns: Option<i64>) -> Self {
        match ns {
            None => Self::NoPrevious,
            Some(ns) if ns < 0 => Self::Overlap,
            Some(ns) if ns <= NEAR_GAP_NS => Self::Near,
            Some(_) => Self::Idle,
        }
    }

    /// The name the report gives it.
    #[must_use]
    pub fn name(self) -> &'static str {
        match self {
            Self::Overlap => "overlap",
            Self::Near => "near",
            Self::Idle => "idle",
            Self::NoPrevious => "no_previous",
        }
    }

    fn index(self) -> usize {
        match self {
            Self::Overlap => 0,
            Self::Near => 1,
            Self::Idle => 2,
            Self::NoPrevious => 3,
        }
    }
}

/// Everything the `tail_*` lines say.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Tail {
    /// The largest step every timestamp of the run is a whole number of from
    /// the earliest, in nanoseconds ([`step_ns`]).
    pub step_ns: Option<u64>,
    /// How many timestamps it is over.
    pub timestamps: usize,
    /// How wide the slices' measurements spread.
    pub spread: Option<Spread>,
    /// How much the step can account for.
    pub quantization: Option<Quantization>,
    /// The loop against calibration.
    pub versus_calibration: Option<Versus>,
    /// The outliers.
    pub outliers: Outliers,
    /// Whether they come together in frames.
    pub clustering: Clustering,
    /// For each frame with slices whose previous frame was timed, from the end
    /// of that frame's recorded work to this one's first compute pass, in
    /// nanoseconds; negative when it began before that work ended.
    pub gap_before_first_ns: Vec<i64>,
    /// For every pass after the first of a frame, from the end of the pass
    /// encoded before it, in nanoseconds.
    pub gap_between_ns: Vec<i64>,
    /// Each frame's first compute pass over the median of the same slice's
    /// measurements as a later pass.
    pub first_over_later: Option<Ratios>,
    /// The same for the later passes themselves, the baseline the first
    /// passes compare with.
    pub later_over_later: Option<Ratios>,
    /// [`Tail::first_over_later`] by the gap before the pass, in
    /// [`Gap::ALL`]'s order.
    pub first_by_gap: [Option<Ratios>; 4],
    /// The measurements by how busy the GPU was over the frames before
    /// theirs, in the order of [`BUSY_BOUNDS`].
    pub busy: Vec<Busy>,
    /// Measurements left out of [`Tail::busy`]: their frame's window was not
    /// all timed, or the period is unknown.
    pub busy_unknown: u64,
}

/// What [`analyse`] reads besides the frames.
#[derive(Debug, Clone, Copy)]
pub struct Context<'a> {
    /// Calibration's measurements of each part of the split program, slice
    /// `i` being part `i`.
    pub calibration: &'a [Vec<u64>],
    /// Each slice's estimate when the loop ended, in microseconds.
    pub estimates: &'a BTreeMap<SliceId, u64>,
    /// The display's period, in microseconds.
    pub period_us: Option<u64>,
}

/// No estimates, for [`Context`]'s default.
static NO_ESTIMATES: BTreeMap<SliceId, u64> = BTreeMap::new();

impl Default for Context<'_> {
    /// No calibration, no estimates and no period.
    fn default() -> Self {
        Self {
            calibration: &[],
            estimates: &NO_ESTIMATES,
            period_us: None,
        }
    }
}

/// The statistics of the `tail_*` lines over `frames`, in any order.
#[must_use]
pub fn analyse(frames: &[TimedFrame], context: Context<'_>) -> Tail {
    let mut ordered: Vec<&TimedFrame> = frames.iter().collect();
    ordered.sort_by_key(|f| f.frame);
    let by_index: BTreeMap<u64, &TimedFrame> = ordered.iter().map(|f| (f.frame, *f)).collect();

    let mut all: BTreeMap<SliceId, Vec<u64>> = BTreeMap::new();
    let mut later: BTreeMap<SliceId, Vec<u64>> = BTreeMap::new();
    for frame in &ordered {
        for (index, (slice, pass)) in frame.compute.iter().enumerate() {
            all.entry(*slice).or_default().push(pass.us);
            if index > 0 {
                later.entry(*slice).or_default().push(pass.us);
            }
        }
    }
    let medians = medians_of(&mut all);
    let later_medians = medians_of(&mut later);

    let timestamps: Vec<u64> = ordered
        .iter()
        .flat_map(|f| f.passes().flat_map(|p| [p.begin_ns, p.end_ns]))
        .collect();
    let step = step_ns(&timestamps);

    let mut tail = Tail {
        step_ns: step,
        timestamps: timestamps.len(),
        spread: spread(&all),
        quantization: step.map(|step| quantization(step, &all, context.estimates)),
        versus_calibration: versus(&all, context.calibration),
        ..Tail::default()
    };

    let is_over = |slice: &SliceId, us: u64| -> Option<bool> {
        medians
            .get(slice)
            .map(|&median| us > median.saturating_mul(OUTLIER_FACTOR))
    };
    let mut per_frame: Vec<(u64, u64)> = Vec::new();
    let mut first_ratios = Vec::new();
    let mut later_ratios = Vec::new();
    let mut by_gap: [Vec<f64>; 4] = Default::default();
    let mut busy: Vec<(Busy, Vec<f64>)> = buckets();
    for frame in &ordered {
        let previous = frame
            .frame
            .checked_sub(1)
            .and_then(|i| by_index.get(&i).copied());
        let gap = match (previous.and_then(TimedFrame::end_ns), frame.compute.first()) {
            (Some(end), Some((_, first))) => Some(signed_gap(end, first.begin_ns)),
            _ => None,
        };
        if let Some(gap) = gap {
            tail.gap_before_first_ns.push(gap);
        }
        let mut passes = frame.passes();
        if let Some(mut before) = passes.next() {
            for pass in passes {
                tail.gap_between_ns
                    .push(signed_gap(before.end_ns, pass.begin_ns));
                before = pass;
            }
        }
        let busy_percent = busy_percent(&by_index, frame.frame, context.period_us);

        let (mut n, mut k) = (0, 0);
        for (index, (slice, pass)) in frame.compute.iter().enumerate() {
            if let Some(&median) = later_medians.get(slice).filter(|&&m| m > 0) {
                let ratio = ratio(pass.us, median);
                if index == 0 {
                    first_ratios.push(ratio);
                    if let Some(list) = by_gap.get_mut(Gap::of(gap).index()) {
                        list.push(ratio);
                    }
                } else {
                    later_ratios.push(ratio);
                }
            }
            let Some(over) = is_over(slice, pass.us) else {
                continue;
            };
            n += 1;
            tail.outliers.samples += 1;
            if over {
                k += 1;
                tail.outliers.over += 1;
            }
            if index == 0 {
                tail.outliers.first_samples += 1;
                if over {
                    tail.outliers.first_over += 1;
                }
            }
            let bucket =
                busy_percent.and_then(|percent| busy.iter_mut().find(|b| b.0.holds(percent)));
            match bucket {
                Some((bucket, ratios)) => {
                    bucket.samples += 1;
                    if over {
                        bucket.over += 1;
                    }
                    if let Some(&median) = medians.get(slice).filter(|&&m| m > 0) {
                        ratios.push(ratio(pass.us, median));
                    }
                }
                None => tail.busy_unknown += 1,
            }
        }
        if n >= 2 {
            per_frame.push((n, k));
        }
    }
    tail.clustering = clustering(&per_frame, tail.outliers.over, tail.outliers.samples);
    tail.first_over_later = Ratios::of(first_ratios);
    tail.later_over_later = Ratios::of(later_ratios);
    tail.first_by_gap = by_gap.map(Ratios::of);
    tail.busy = busy
        .into_iter()
        .map(|(mut bucket, mut ratios)| {
            ratios.sort_by(f64::total_cmp);
            bucket.ratio = percentile(&ratios, 50);
            bucket
        })
        .collect();
    tail
}

impl Busy {
    /// Whether `percent` falls in the bucket.
    fn holds(&self, percent: u64) -> bool {
        percent >= self.from_percent && self.below_percent.is_none_or(|below| percent < below)
    }
}

/// The empty buckets of [`BUSY_BOUNDS`], each with its ratios.
fn buckets() -> Vec<(Busy, Vec<f64>)> {
    let mut from = 0;
    let mut out = Vec::new();
    for below in BUSY_BOUNDS.iter().copied().map(Some).chain([None]) {
        out.push((
            Busy {
                from_percent: from,
                below_percent: below,
                ..Busy::default()
            },
            Vec::new(),
        ));
        from = below.unwrap_or(from);
    }
    out
}

/// The GPU time of the [`BUSY_WINDOW`] frames before `frame`, in percent of
/// their time at `period_us`; `None` when one of them was not timed, the run
/// had fewer frames before it, or the period is unknown.
fn busy_percent(
    frames: &BTreeMap<u64, &TimedFrame>,
    frame: u64,
    period_us: Option<u64>,
) -> Option<u64> {
    let window_us = period_us?.checked_mul(BUSY_WINDOW).filter(|&w| w > 0)?;
    let first = frame.checked_sub(BUSY_WINDOW)?;
    let mut busy = 0_u64;
    for index in first..frame {
        busy = busy.saturating_add(frames.get(&index)?.gpu_us());
    }
    Some(busy.saturating_mul(100) / window_us)
}

/// `to - from` in nanoseconds, saturating at the ends of `i64`.
fn signed_gap(from: u64, to: u64) -> i64 {
    let difference = i128::from(to) - i128::from(from);
    i64::try_from(difference).unwrap_or(if difference < 0 { i64::MIN } else { i64::MAX })
}

/// `us` over `median`.
#[allow(clippy::cast_precision_loss)]
fn ratio(us: u64, median: u64) -> f64 {
    us as f64 / median as f64
}

/// Each slice's median, over the slices measured [`MIN_SAMPLES`] times or
/// more; sorts every slice's measurements.
fn medians_of(samples: &mut BTreeMap<SliceId, Vec<u64>>) -> BTreeMap<SliceId, u64> {
    samples
        .iter_mut()
        .filter_map(|(slice, list)| {
            list.sort_unstable();
            if list.len() < MIN_SAMPLES {
                return None;
            }
            percentile(list, 50).map(|median| (*slice, median))
        })
        .collect()
}

/// The spread of each slice's sorted measurements, over the slices measured
/// [`MIN_SAMPLES`] times or more.
fn spread(sorted: &BTreeMap<SliceId, Vec<u64>>) -> Option<Spread> {
    let mut p90 = Vec::new();
    let mut max = Vec::new();
    for list in eligible(sorted) {
        let (Some(median), Some(high), Some(&top)) =
            (percentile(list, 50), percentile(list, 90), list.last())
        else {
            continue;
        };
        if median == 0 {
            continue;
        }
        p90.push(ratio(high, median));
        max.push(ratio(top, median));
    }
    let slices = p90.len();
    let max = Ratios::of(max)?.median;
    Some(Spread {
        slices,
        p90_over_p50: Ratios::of(p90)?,
        max_over_p50: max,
    })
}

/// The slices' sorted measurements, of those measured [`MIN_SAMPLES`] times
/// or more.
fn eligible(sorted: &BTreeMap<SliceId, Vec<u64>>) -> impl Iterator<Item = &Vec<u64>> {
    sorted.values().filter(|list| list.len() >= MIN_SAMPLES)
}

/// What a step of `step_ns` can account for in the slices' sorted
/// measurements and in their final `estimates`.
///
/// A duration is the difference of two timestamps, each cut down to the
/// step, so it is off by less than one step either way, and so is any
/// percentile of a slice's durations or an estimate read from them.
#[allow(clippy::cast_precision_loss)]
fn quantization(
    step_ns: u64,
    sorted: &BTreeMap<SliceId, Vec<u64>>,
    estimates: &BTreeMap<SliceId, u64>,
) -> Quantization {
    let step_us = step_ns as f64 / 1_000.0;
    let mut within_step = 0;
    let mut over_p90 = Vec::new();
    let mut over_excess = Vec::new();
    for (slice, list) in sorted.iter().filter(|(_, l)| l.len() >= MIN_SAMPLES) {
        let (Some(median), Some(high)) = (percentile(list, 50), percentile(list, 90)) else {
            continue;
        };
        if (high - median) as f64 <= step_us {
            within_step += 1;
        }
        if high > 0 {
            over_p90.push(step_us / high as f64);
        }
        let count = list.len() as u64;
        let mean = list.iter().sum::<u64>() as f64 / count as f64;
        if let Some(&estimate) = estimates.get(slice) {
            let excess = estimate as f64 - mean;
            if excess > 0.0 {
                over_excess.push((step_us / excess).min(1.0));
            }
        }
    }
    Quantization {
        step_us,
        within_step,
        step_over_p90: Ratios::of(over_p90).map(|r| r.median),
        step_over_excess: Ratios::of(over_excess).map(|r| r.median),
    }
}

/// The loop's sorted measurements of each slice against calibration's of the
/// same part.
#[allow(clippy::cast_precision_loss)]
fn versus(sorted: &BTreeMap<SliceId, Vec<u64>>, calibration: &[Vec<u64>]) -> Option<Versus> {
    let mut means = Vec::new();
    let mut medians = Vec::new();
    for (slice, list) in sorted.iter().filter(|(_, l)| l.len() >= MIN_SAMPLES) {
        let Some(part) = usize::try_from(slice.get())
            .ok()
            .and_then(|i| calibration.get(i))
            .filter(|p| !p.is_empty())
        else {
            continue;
        };
        let mean = |values: &[u64]| values.iter().sum::<u64>() as f64 / values.len() as f64;
        let mut part = part.clone();
        part.sort_unstable();
        let (Some(loop_p50), Some(calibrated_p50)) = (percentile(list, 50), percentile(&part, 50))
        else {
            continue;
        };
        if calibrated_p50 == 0 || mean(&part) <= 0.0 {
            continue;
        }
        means.push(mean(list) / mean(&part));
        medians.push(ratio(loop_p50, calibrated_p50));
    }
    let slices = means.len();
    Some(Versus {
        slices,
        mean: Ratios::of(means)?.median,
        p50: Ratios::of(medians)?.median,
    })
}

/// Clustering of outliers over frames of `(measurements, outliers)`, with
/// `over` outliers among `samples` measurements in the run.
#[allow(clippy::cast_precision_loss)]
fn clustering(per_frame: &[(u64, u64)], over: u64, samples: u64) -> Clustering {
    let share = if samples == 0 {
        0.0
    } else {
        over as f64 / samples as f64
    };
    let mut out = Clustering::default();
    for &(n, k) in per_frame {
        out.frames += 1;
        let half = n.div_ceil(2);
        if k >= half {
            out.at_least_half += 1;
        }
        out.expected += binomial_at_least(n, half, share);
    }
    out
}

/// The probability that `n` independent trials, each a success with
/// probability `p`, have at least `k` successes. Summed in logarithms, so a
/// frame of many slices neither underflows nor overflows.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn binomial_at_least(n: u64, k: u64, p: f64) -> f64 {
    if k == 0 {
        return 1.0;
    }
    if k > n || p.is_nan() || p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    let (ln_p, ln_q) = (p.ln(), (1.0 - p).ln());
    // ln of the probability of exactly i successes, from i = 0 up.
    let mut ln_pmf = n as f64 * ln_q;
    let mut total = 0.0;
    for i in 0..=n {
        if i >= k {
            total += ln_pmf.exp();
        }
        ln_pmf += ((n - i) as f64).ln() - ((i + 1) as f64).ln() + ln_p - ln_q;
    }
    total.min(1.0)
}

/// The largest step every one of `timestamps` is a whole number of from the
/// earliest: their differences' greatest common divisor. `None` with fewer
/// than two distinct timestamps.
///
/// A clock that is cut down to a step before it is read shows that step; an
/// exact one shows a step of a few nanoseconds once there are enough
/// timestamps.
#[must_use]
pub fn step_ns(timestamps: &[u64]) -> Option<u64> {
    let earliest = *timestamps.iter().min()?;
    let step = timestamps
        .iter()
        .fold(0, |step, &t| gcd(step, t - earliest));
    (step > 0).then_some(step)
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const MS: u64 = 1_000_000;

    fn pass(us: u64, begin_ns: u64) -> Pass {
        Pass {
            us,
            begin_ns,
            end_ns: begin_ns + us * 1_000,
        }
    }

    /// A frame at `index` of compute passes of `(slice, µs)` laid one behind
    /// the other from `start_ns`, then a 1 900 µs render.
    fn frame(index: u64, start_ns: u64, slices: &[(u32, u64)]) -> TimedFrame {
        let mut at = start_ns;
        let mut compute = Vec::new();
        for &(slice, us) in slices {
            compute.push((SliceId::new(slice), pass(us, at)));
            at += us * 1_000;
        }
        TimedFrame {
            frame: index,
            compute,
            render: Some(pass(1_900, at)),
        }
    }

    fn analysed(frames: &[TimedFrame]) -> Tail {
        analyse(
            frames,
            Context {
                period_us: Some(8_000),
                ..Context::default()
            },
        )
    }

    #[test]
    fn percentiles_are_nearest_rank_and_ratios_take_them() {
        assert_eq!(percentile::<u64>(&[], 50), None);
        assert_eq!(percentile(&[1, 2, 3, 4], 50), Some(2));
        assert_eq!(percentile(&[1, 2, 3, 4], 90), Some(4));
        let r = Ratios::of(vec![3.0, 1.0, 2.0]).unwrap();
        assert_eq!((r.samples, r.median, r.p90), (3, 2.0, 3.0));
        assert_eq!(Ratios::of(Vec::new()), None);
    }

    #[test]
    fn the_step_is_the_timestamps_common_divisor() {
        let q = 65_536;
        assert_eq!(step_ns(&[5 * q, 9 * q, 12 * q, 5 * q]), Some(q));
        assert_eq!(step_ns(&[1_000, 1_003, 1_010]), Some(1));
        assert_eq!(step_ns(&[7, 7]), None);
        assert_eq!(step_ns(&[]), None);
    }

    #[test]
    fn the_binomial_tail_sums_to_the_exact_probability() {
        assert_eq!(binomial_at_least(4, 0, 0.3), 1.0);
        assert_eq!(binomial_at_least(4, 5, 0.3), 0.0);
        assert_eq!(binomial_at_least(4, 2, 0.0), 0.0);
        assert_eq!(binomial_at_least(4, 2, 1.0), 1.0);
        // P(X >= 1) for 2 fair coins is 3/4; P(X >= 2) for 3 at 0.1 is 0.028.
        assert!((binomial_at_least(2, 1, 0.5) - 0.75).abs() < 1e-12);
        assert!((binomial_at_least(3, 2, 0.1) - 0.028).abs() < 1e-12);
        // Many trials neither underflow nor exceed 1.
        let p = binomial_at_least(2_000, 1_000, 0.5);
        assert!((0.5..=0.52).contains(&p), "{p}");
    }

    #[test]
    fn a_gap_is_overlap_near_idle_or_without_a_previous_frame() {
        assert_eq!(Gap::of(Some(-1)), Gap::Overlap);
        assert_eq!(Gap::of(Some(0)), Gap::Near);
        assert_eq!(Gap::of(Some(NEAR_GAP_NS)), Gap::Near);
        assert_eq!(Gap::of(Some(NEAR_GAP_NS + 1)), Gap::Idle);
        assert_eq!(Gap::of(None), Gap::NoPrevious);
        assert_eq!(signed_gap(10, 4), -6);
        assert_eq!(signed_gap(0, u64::MAX), i64::MAX);
    }

    /// Ten frames, each slice 0 then slice 1 at 1 000 µs, and one frame
    /// where both ran slow: the per-slice lines, the outliers and the
    /// clustering.
    #[test]
    fn outliers_are_counted_against_their_slice_and_found_together() {
        let mut frames: Vec<TimedFrame> = (0..10)
            .map(|i| frame(i, i * 8 * MS, &[(0, 1_000), (1, 1_000)]))
            .collect();
        frames.push(frame(10, 80 * MS, &[(0, 3_000), (1, 2_500)]));
        let tail = analysed(&frames);
        assert_eq!(
            tail.outliers,
            Outliers {
                samples: 22,
                over: 2,
                first_samples: 11,
                first_over: 1,
            }
        );
        // The one frame with both slow, against 11 * P(Bin(2, 2/22) >= 1).
        assert_eq!(
            (tail.clustering.frames, tail.clustering.at_least_half),
            (11, 1)
        );
        let p: f64 = 2.0 / 22.0;
        let expected = 11.0 * (1.0 - (1.0 - p).powi(2));
        assert!((tail.clustering.expected - expected).abs() < 1e-9);
        let spread = tail.spread.unwrap();
        assert_eq!(spread.slices, 2);
        assert_eq!(spread.p90_over_p50.median, 1.0);
        assert_eq!(spread.max_over_p50, 2.5);
        // Slice 1 is always later, slice 0 always first: no slice was
        // measured both ways, so first passes have nothing to compare with.
        assert_eq!(tail.first_over_later, None);
    }

    /// Slice 0 runs first in some frames and later in others; as a first
    /// pass it measures twice its later median, and only when it begins
    /// right behind the previous frame's work.
    #[test]
    fn a_first_pass_is_compared_with_the_same_slice_later_by_its_gap() {
        let mut frames = Vec::new();
        for i in 0..10 {
            frames.push(frame(i, i * 8 * MS, &[(1, 500), (0, 1_000)]));
        }
        // Frame 10 begins right where frame 9's render ended, and its first
        // pass measures 2 000 µs; frame 12, after an untimed frame 11, 1 000.
        let end_9 = frames[9].end_ns().unwrap();
        frames.push(frame(10, end_9, &[(0, 2_000)]));
        frames.push(frame(12, 96 * MS, &[(0, 1_000)]));
        let tail = analysed(&frames);
        let near = tail.first_by_gap[Gap::Near.index()].unwrap();
        assert_eq!((near.samples, near.median), (1, 2.0));
        let none = tail.first_by_gap[Gap::NoPrevious.index()].unwrap();
        assert_eq!((none.samples, none.median), (1, 1.0));
        assert_eq!(tail.first_by_gap[Gap::Overlap.index()], None);
        // Frames 1-9 began 8 ms after the previous frame began, so after it
        // ended: idle gaps. Slice 1, their first pass, has no later
        // measurements, so only frames 10 and 12 compare.
        assert_eq!(tail.first_over_later.unwrap().samples, 2);
        assert_eq!(tail.later_over_later.unwrap().median, 1.0);
        // Gaps before a first pass: frames 1-9 and 10 (frame 0 and 12 have
        // no timed previous frame).
        assert_eq!(tail.gap_before_first_ns.len(), 10);
        assert_eq!(tail.gap_before_first_ns[9], 0);
        let idle = 8 * MS - (500 + 1_000 + 1_900) * 1_000;
        assert_eq!(tail.gap_before_first_ns[0], i64::try_from(idle).unwrap());
        // Passes laid end to end: every gap between them is zero.
        assert!(tail.gap_between_ns.iter().all(|&g| g == 0));
        assert_eq!(tail.gap_between_ns.len(), 10 * 2 + 1 + 1);
    }

    #[test]
    fn a_pass_that_began_before_the_previous_frame_ended_overlaps() {
        let first = frame(0, 0, &[(0, 1_000)]);
        let end = first.end_ns().unwrap();
        let second = frame(1, end - 131_072, &[(0, 1_000)]);
        let tail = analysed(&[second, first]);
        assert_eq!(tail.gap_before_first_ns, vec![-131_072]);
    }

    /// Measurements are bucketed by the GPU time of the four frames before
    /// theirs, at an 8 000 µs period: a frame after four of render alone
    /// (1 900 µs) is 23 % busy, one after four that also ran 5 000 µs of
    /// slices 86 %.
    #[test]
    fn measurements_are_bucketed_by_how_busy_the_frames_before_were() {
        let mut frames = Vec::new();
        for i in 0..4 {
            frames.push(frame(i, i * 8 * MS, &[]));
        }
        for i in 4..13 {
            frames.push(frame(i, i * 8 * MS, &[(0, 2_500), (1, 2_500)]));
        }
        let tail = analysed(&frames);
        let samples: Vec<u64> = tail.busy.iter().map(|b| b.samples).collect();
        // Frame 4 reads frames 0-3 (23 %); 5 to 7 read a mix (39, 55 and
        // 70 %); 8 to 12 read four busy frames (86 %).
        assert_eq!(samples, vec![2, 2, 2 + 2, 2 * 5]);
        assert_eq!(tail.busy_unknown, 0);
        assert_eq!(
            tail.busy.iter().map(|b| b.from_percent).collect::<Vec<_>>(),
            vec![0, 25, 50, 75]
        );
        assert_eq!(tail.busy[3].below_percent, None);
        assert_eq!(tail.busy[0].ratio, Some(1.0));
        // Without a period, nothing is bucketed.
        let unknown = analyse(&frames, Context::default());
        assert_eq!(unknown.busy_unknown, 18);
    }

    #[test]
    fn the_step_bounds_what_quantization_adds_to_quantiles_and_estimates() {
        let q = 65_536;
        // Slice 0 alternates 1 step and 2 steps; slice 1 is 1 000 µs with one
        // in five at 2 000.
        let frames: Vec<TimedFrame> = (0..10)
            .map(|i| {
                let us0 = if i % 2 == 0 { 66 } else { 131 };
                let us1 = if i % 5 == 0 { 2_000 } else { 1_000 };
                TimedFrame {
                    frame: i,
                    compute: vec![
                        (
                            SliceId::new(0),
                            Pass {
                                us: us0,
                                begin_ns: 100 * i * q,
                                end_ns: (100 * i + 1 + i % 2) * q,
                            },
                        ),
                        (
                            SliceId::new(1),
                            Pass {
                                us: us1,
                                begin_ns: (100 * i + 10) * q,
                                end_ns: (100 * i + 40) * q,
                            },
                        ),
                    ],
                    render: None,
                }
            })
            .collect();
        let estimates = BTreeMap::from([(SliceId::new(0), 131), (SliceId::new(1), 2_000)]);
        let tail = analyse(
            &frames,
            Context {
                estimates: &estimates,
                ..Context::default()
            },
        );
        assert_eq!(tail.step_ns, Some(q));
        let quantization = tail.quantization.unwrap();
        assert_eq!(quantization.within_step, 1);
        // Slice 0: 65.536 / 131, and capped at 1 against its 32.5 µs excess
        // (estimate 131, mean 98.5); slice 1: 65.536 / 2 000, and against
        // 800 µs (estimate 2 000, mean 1 200). The median of two is the
        // lower.
        let step_over_p90 = quantization.step_over_p90.unwrap();
        assert!(
            (step_over_p90 - 65.536 / 2_000.0).abs() < 1e-12,
            "{step_over_p90}"
        );
        let excess = quantization.step_over_excess.unwrap();
        assert!((excess - 65.536 / 800.0).abs() < 1e-12, "{excess}");
    }

    #[test]
    fn the_loop_is_compared_with_calibration_slice_by_slice() {
        let frames: Vec<TimedFrame> = (0..8)
            .map(|i| frame(i, i * 8 * MS, &[(0, 1_200), (1, 3_000)]))
            .collect();
        let calibration = vec![vec![1_000; 4], vec![1_000, 1_000, 1_000, 5_000], vec![]];
        let tail = analyse(
            &frames,
            Context {
                calibration: &calibration,
                ..Context::default()
            },
        );
        let versus = tail.versus_calibration.unwrap();
        assert_eq!(versus.slices, 2);
        // Means: 1.2 and 3 000 / 2 000 = 1.5; medians: 1.2 and 3.0. The
        // median of two is the lower.
        assert_eq!((versus.mean, versus.p50), (1.2, 1.2));
        let none = analyse(&frames, Context::default());
        assert_eq!(none.versus_calibration, None);
    }

    #[test]
    fn slices_measured_too_rarely_are_left_out() {
        let frames: Vec<TimedFrame> = (0..u64::try_from(MIN_SAMPLES).unwrap() - 1)
            .map(|i| frame(i, i * 8 * MS, &[(0, 1_000)]))
            .collect();
        let tail = analysed(&frames);
        assert_eq!(tail.outliers, Outliers::default());
        assert_eq!(tail.spread, None);
        assert_eq!(frames[0].gpu_us(), 2_900);
    }
}
