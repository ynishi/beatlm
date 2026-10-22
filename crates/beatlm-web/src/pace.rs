//! Which frame an animation-frame callback is, and how long its period is,
//! from the timestamp the browser gave it.
//!
//! The display decides when a callback runs, so the host does not impose a
//! frame grid on it, as the native bench does with its fixed 60 Hz one
//! (`beatlm-bench`'s `pace.rs`). That grid is anchored once, at the start of a
//! run, and a display is driven by its own refresh, which is rarely exactly
//! 60.000 Hz. On a 59.94 Hz panel each callback lands about 16 µs later per
//! period than the grid expects, so the lateness climbs through a run and,
//! after several hundred frames, a callback crosses into the next period and
//! is counted as a dropped frame on a healthy display; a faster display gets
//! a periodic extra callback instead. The rule here has no grid to drift from:
//!
//! - **The frame is the callback.** Every callback is one frame, whatever the
//!   display's rate; there is no callback that is not a frame. On a 120 Hz
//!   display that is a frame every 8.3 ms with a budget to match, where the
//!   grid ran every other callback.
//! - **Its start is the callback's timestamp**, the `DOMHighResTimeStamp` the
//!   browser passes to a `requestAnimationFrame` callback, in microseconds
//!   from the run's first callback. That value is the time the display's frame
//!   began, the same for every callback of that frame, on the clock of
//!   `performance.now()`. A fresh `performance.now()` would add to it
//!   whenever the main thread got round to running the callback, and the run's
//!   own work before it, so the intervals the period is measured from would
//!   carry the host's jitter rather than the display's cadence.
//! - **Its period is measured from the display**: the median of the last
//!   [`WINDOW`] per-period intervals, once [`MIN_INTERVALS`] of them exist, and
//!   the prior before that: [`NOMINAL_PERIOD`] (60 Hz, the bench's), or the
//!   period a pacer of its own measured before the run, when the host measured
//!   one ([`Pacer::with_prior`]). A host that knows the display's period gives
//!   it, so the first frames of a 120 Hz run are not budgeted for 60 Hz and
//!   overrun their period while the window fills. Each callback's gap
//!   since the previous one is divided by the number of periods it counts as
//!   (the next bullet), and that quotient, in whole microseconds, is the
//!   window's one sample for the gap. With `n` samples sorted ascending the
//!   period is the one at zero-based index `n / 2`: the upper median, the rule
//!   the render forecast uses.
//!
//!   The samples are per period, and not the raw gaps, so that the host cannot
//!   move the period: a gap of 2 periods, where the host overran its frame or
//!   a vsync was missed, contributes one period, the same sample as a gap that
//!   was on time. A median of the raw gaps would read a run that overran more
//!   than half of a window as a display of twice the period, and the budget
//!   would follow it. And it is a median, and not a low quantile, because the
//!   timestamps jitter. The nearest-rank 10th percentile of the raw gaps was
//!   the rule first, and it read 7,700 µs on Chrome 154 (WebGPU) on an Apple
//!   M1 Pro adapter driving a 120 Hz display (2026-10-22, synthetic workload,
//!   `?frames=3000&latency=2`),
//!   where the callback intervals were p50 8,300 µs and p99 9,400 µs: the
//!   timestamps jitter by about ±0.6 ms, so the short end of a window is the
//!   jitter and not the display, 7% short. The budget shrank by the difference,
//!   `frame_start_lateness_us` read p50 600 µs where it should be 0, and the
//!   1.5-period threshold moved with the period. The median of per-period
//!   samples averages the jitter out, and one short interval is one sample of
//!   the window and not its period.
//!
//!   What the timestamps cannot tell apart is a host that overruns every frame
//!   and a display at half the rate: both are gaps of 2 periods, one after
//!   another. This rule measures both at the faster rate, counts every gap as
//!   one dropped frame, and gives the frames the shorter period's budget. The
//!   failure is visible, in `frames_dropped`, and conservative, in a budget
//!   that is smaller and not larger; a period that silently doubled would pack
//!   twice the work into each frame and report no drop at all. The same holds
//!   for a prior that is too fast: a 30 Hz display under the 60 Hz prior is
//!   measured at 60 Hz, with every other period dropped, because each of its
//!   gaps counts as 2 prior periods and contributes one. A display faster than
//!   the prior is learned: at 120 Hz under the 60 Hz prior a gap of half a
//!   period counts as 1 period and contributes the real interval. A display
//!   that changes rate (a window moved to another monitor) is followed the
//!   same way: a faster one, or a slower one by less than 1.5 times, is
//!   learned once half of the window is new, and the gaps before then are
//!   judged against the old period; one slower by more than that is the
//!   half-rate case again and stays measured at the old period.
//! - **A frame is dropped when the gap since the previous callback is more than
//!   [`DROP_THRESHOLD_HALVES`] half periods** (1.5 periods): below that the
//!   callback is the next vsync, late by some jitter, above it a vsync was
//!   missed. The gap then counts as the whole number of periods nearest to it,
//!   and every one of them but the one the callback is in is dropped: a missed
//!   vsync (a gap of 2 periods) is one dropped frame, a 100 ms stall at 60 Hz
//!   is five, and a host that overran its frame is the same missed vsync and
//!   is counted as one. The core sees the frame indices that were skipped, as
//!   it does from the bench. The period is read once per callback, before its
//!   gap is recorded, and is the one the gap is judged against and the one the
//!   callback's frame is given, so the interval being judged cannot excuse
//!   itself; the cost is that the first [`MIN_INTERVALS`] intervals are judged
//!   against the nominal period, and on a display far from 60 Hz those read
//!   wrong.
//! - **Lateness is the part of the gap the periods do not explain**: the gap
//!   less the periods it counted as, floored at zero, so a callback that comes
//!   exactly when the display's cadence says reads zero. A stall is counted
//!   once, in the dropped frames, and not again as lateness. This is not the
//!   native bench's number, which is measured from a fixed grid; it keeps the
//!   name `frame_start_lateness_us` because it answers the same question, how
//!   far from its expected time a frame began, with the expected time taken
//!   from the display rather than from a clock the display does not keep.
//!
//! A run lasts for a number of frame indices, as the bench's does: the periods
//! a long gap gives up count towards it, so a run with a stall runs fewer
//! frames than it was asked for and `frames` in the report says how many. Frames
//! plus dropped frames is the number asked for, so a gap that carries the index
//! past the run's last drops only the indices of the run; the periods beyond it
//! were never the run's, and that gap is not recorded as an interval.
//!
//! Consequence for latency promised in frames: a promise of `k` frames is `k`
//! display periods, so it is shorter in time on a faster display (2 frames is
//! 33.3 ms at 60 Hz and 16.7 ms at 120 Hz), and the frame's budget, taken from
//! the period, is shorter in the same proportion. A run is comparable with
//! another only at the same display period, which the report's
//! `display_period_us` states.
//!
//! Nothing here reads a clock: the timestamp is an argument, which is what lets
//! the rule be tested natively.

use std::collections::VecDeque;

use beatlm_core::time::Micros;

/// The frame period assumed until the display's has been measured: 60 Hz, the
/// native bench's.
pub const NOMINAL_PERIOD: Micros = Micros::new(16_667);

/// How many recent per-period intervals the display's period is measured from.
pub const WINDOW: usize = 64;

/// How many intervals must have been seen before the measured period replaces
/// [`NOMINAL_PERIOD`]. With this many the upper median is the fifth smallest of
/// eight, so up to three samples on either side of the display's, a stall or an
/// irregular start, cannot be the period; fewer leave the median a vote or two
/// from an outlier. More only delays the measured period, and until it exists
/// every gap is judged against the nominal one, which on a display far from
/// 60 Hz reads wrong.
pub const MIN_INTERVALS: usize = 8;

/// The gap, in half periods, above which a callback is late by a vsync: 3, so
/// 1.5 periods. Midway between "the next vsync" and "the one after" is where
/// jitter stops being the likelier explanation.
pub const DROP_THRESHOLD_HALVES: u64 = 3;

/// One frame, as the pacer starts it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameStart {
    /// The frame's index; periods given up leave gaps in the sequence.
    pub index: u64,
    /// Microseconds from the run's first callback to this one.
    pub start_us: u64,
    /// The frame's period: the display's, as far as it has been measured.
    pub period: Micros,
}

/// What a callback is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pace {
    /// A frame to run.
    Frame(FrameStart),
    /// The run's frames are done.
    Done,
}

/// Turns callback timestamps into frames for a run of a fixed number of frame
/// indices.
#[derive(Debug, Clone)]
pub struct Pacer {
    frames: u64,
    first_us: Option<u64>,
    /// The previous callback's timestamp; `None` before the first.
    previous_us: Option<u64>,
    /// The index the next callback would have if it came one period after the
    /// previous.
    next: u64,
    dropped: u64,
    /// The last [`WINDOW`] per-period intervals, the display's period being
    /// their median.
    recent: VecDeque<u64>,
    intervals_us: Vec<u64>,
    lateness_us: Vec<u64>,
    /// The period frames are given before [`MIN_INTERVALS`] have been seen.
    prior: Micros,
}

impl Pacer {
    /// A pacer for `frames` frame indices, whose time zero is the first
    /// callback, with [`NOMINAL_PERIOD`] as its prior.
    #[must_use]
    pub fn new(frames: u64) -> Self {
        Self::with_prior(frames, NOMINAL_PERIOD)
    }

    /// A pacer for `frames` frame indices whose frames are given `prior` until
    /// the display's period has been measured: for a host that measured it
    /// before the run, by this rule. A prior of zero is the nominal period.
    #[must_use]
    pub fn with_prior(frames: u64, prior: Micros) -> Self {
        let prior = if prior == Micros::ZERO {
            NOMINAL_PERIOD
        } else {
            prior
        };
        Self {
            frames,
            first_us: None,
            previous_us: None,
            next: 0,
            dropped: 0,
            recent: VecDeque::with_capacity(WINDOW),
            intervals_us: Vec::new(),
            lateness_us: Vec::new(),
            prior,
        }
    }

    /// What a callback is, `timestamp_us` being its timestamp in microseconds
    /// on the clock every callback's is read from.
    pub fn frame_at(&mut self, timestamp_us: u64) -> Pace {
        if self.next >= self.frames {
            return Pace::Done;
        }
        let first = *self.first_us.get_or_insert(timestamp_us);
        // Read before this callback's gap is recorded: it is the period the gap
        // is judged against and the one the frame is given.
        let period = self.period();
        let mut lateness = 0;
        if let Some(previous) = self.previous_us {
            let gap = timestamp_us.saturating_sub(previous);
            let periods = periods_in(gap, period.get());
            let given_up = periods - 1;
            let remaining = self.frames - self.next;
            if given_up >= remaining {
                // The callback is past the run's last index, so only the
                // indices of the run are given up, and a gap that ends the run
                // is no interval of the display's.
                self.dropped += remaining;
                self.next = self.frames;
                return Pace::Done;
            }
            self.dropped += given_up;
            self.next += given_up;
            lateness = gap.saturating_sub(periods.saturating_mul(period.get()));
            // A timestamp that did not advance is no interval of any display.
            if gap > 0 {
                if self.recent.len() == WINDOW {
                    self.recent.pop_front();
                }
                // The period's sample is the gap per period it counted as, so
                // an overrun gap of two periods is one period here and cannot
                // move the median; the report's intervals stay the raw gaps.
                self.recent.push_back(gap / periods);
                self.intervals_us.push(gap);
            }
        }
        self.previous_us = Some(timestamp_us);
        self.lateness_us.push(lateness);
        let start = FrameStart {
            index: self.next,
            start_us: timestamp_us.saturating_sub(first),
            period,
        };
        self.next += 1;
        Pace::Frame(start)
    }

    /// The period frames are given now: the measured one, or the prior before
    /// enough intervals have been seen.
    #[must_use]
    pub fn period(&self) -> Micros {
        self.measured_period().unwrap_or(self.prior)
    }

    /// The display's period, the median of the recent per-period callback
    /// intervals: of `n` sorted ascending, the one at zero-based index `n / 2`
    /// (the upper one of an even count). `None` before [`MIN_INTERVALS`] have
    /// been seen. The module documentation says why per period and why a
    /// median.
    #[must_use]
    pub fn measured_period(&self) -> Option<Micros> {
        if self.recent.len() < MIN_INTERVALS {
            return None;
        }
        let mut sorted = [0_u64; WINDOW];
        let sorted = sorted.get_mut(..self.recent.len())?;
        for (slot, &interval) in sorted.iter_mut().zip(&self.recent) {
            *slot = interval;
        }
        sorted.sort_unstable();
        sorted.get(sorted.len() / 2).copied().map(Micros::new)
    }

    /// The run's first callback timestamp, in microseconds on the clock the
    /// timestamps are read from: the epoch of every [`FrameStart::start_us`],
    /// which a host maps another clock onto the frames' with; `None` before
    /// the first callback.
    #[must_use]
    pub const fn epoch_us(&self) -> Option<u64> {
        self.first_us
    }

    /// Frame periods of the run given up because a callback came more than 1.5
    /// periods after the one before; with the frames started, the number the
    /// run was asked for once it is done.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Every interval between consecutive callbacks, in microseconds, in the
    /// order they came; the gaps that were dropped periods included, and the
    /// one that carried the index past the end of the run not.
    #[must_use]
    pub fn intervals_us(&self) -> &[u64] {
        &self.intervals_us
    }

    /// How late each started frame's callback came, in microseconds, after the
    /// time the previous callback and the periods its gap counted as put it
    /// at; zero for the first frame.
    #[must_use]
    pub fn lateness_us(&self) -> &[u64] {
        &self.lateness_us
    }
}

/// How many periods a `gap` between two callbacks counts as: one up to
/// [`DROP_THRESHOLD_HALVES`] half periods, and above that the whole number of
/// periods nearest to it, which is at least two.
fn periods_in(gap: u64, period: u64) -> u64 {
    let period = period.max(1);
    let halves = gap.saturating_mul(2);
    if halves <= period.saturating_mul(DROP_THRESHOLD_HALVES) {
        1
    } else {
        // Round to nearest: (gap / period + 1/2), in halves.
        halves.saturating_add(period) / period.saturating_mul(2)
    }
}

/// `ms` milliseconds, as a `performance.now()` reading or a callback timestamp
/// is, in whole microseconds, rounded to nearest; zero for a negative or
/// non-finite value.
#[must_use]
pub fn ms_to_us(ms: f64) -> u64 {
    if ms.is_finite() && ms > 0.0 {
        // A float-to-integer `as` saturates, and microseconds since a page
        // loaded are far below where an `f64` stops being exact.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let us = (ms * 1_000.0).round() as u64;
        us
    } else {
        0
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    const P: u64 = NOMINAL_PERIOD.get();

    /// Timestamps of `count` callbacks of a display at `hz`, from `start_us`,
    /// each rounded to whole microseconds from its exact time.
    fn display(hz: f64, count: u64, start_us: u64) -> Vec<u64> {
        (0..count)
            .map(|i| start_us + (i as f64 * 1_000_000.0 / hz).round() as u64)
            .collect()
    }

    /// The same, as a browser that coarsens timestamps to 100 µs reports them.
    fn display_coarse(hz: f64, count: u64, start_us: u64) -> Vec<u64> {
        display(hz, count, start_us)
            .into_iter()
            .map(|t| (t + 50) / 100 * 100)
            .collect()
    }

    /// Timestamps from `start_us` with `gaps` between consecutive ones.
    fn from_gaps(start_us: u64, gaps: &[u64]) -> Vec<u64> {
        let mut t = start_us;
        let mut out = vec![t];
        for gap in gaps {
            t += gap;
            out.push(t);
        }
        out
    }

    /// Feeds `timestamps` to a pacer of `frames` indices and returns it with
    /// the frames it started.
    fn run(frames: u64, timestamps: &[u64]) -> (Pacer, Vec<FrameStart>) {
        let mut pacer = Pacer::new(frames);
        let mut started = Vec::new();
        for &t in timestamps {
            if let Pace::Frame(start) = pacer.frame_at(t) {
                started.push(start);
            }
        }
        (pacer, started)
    }

    fn max(samples: &[u64]) -> u64 {
        samples.iter().copied().max().unwrap_or(0)
    }

    /// A run whose last index has been reached has started or given up every
    /// index it was asked for.
    fn assert_accounted(pacer: &Pacer, started: &[FrameStart], requested: u64) {
        assert_eq!(
            started.len() as u64 + pacer.dropped(),
            requested,
            "frames and dropped frames do not add up to the frames asked for"
        );
    }

    /// Every frame was given exactly `period`.
    fn assert_every_period(started: &[FrameStart], period: Micros) {
        for (i, start) in started.iter().enumerate() {
            assert_eq!(start.period, period, "frame {i}");
        }
    }

    fn assert_steady(hz: f64, timestamps: &[u64], lateness_bound: u64) {
        let count = timestamps.len() as u64;
        let (pacer, started) = run(count, timestamps);
        assert_eq!(pacer.dropped(), 0, "{hz} Hz dropped a frame");
        assert_accounted(&pacer, &started, count);
        assert_eq!(started.len() as u64, count, "{hz} Hz skipped a callback");
        for (i, start) in started.iter().enumerate() {
            assert_eq!(start.index, i as u64, "{hz} Hz: frame {i}");
        }
        // Until the display's period is measured the prior stands in for it,
        // and a display that is not exactly 60 Hz reads late by the difference.
        assert!(
            max(pacer.lateness_us()) <= lateness_bound + 50,
            "{hz} Hz: lateness max {}",
            max(pacer.lateness_us())
        );
        let settled = pacer.lateness_us().get(MIN_INTERVALS + 1..).unwrap();
        assert!(
            max(settled) <= lateness_bound,
            "{hz} Hz: lateness max once measured {}",
            max(settled)
        );
        let exact = 1_000_000.0 / hz;
        let measured = pacer.measured_period().unwrap().get() as f64;
        assert!(
            (measured - exact).abs() <= 100.0,
            "{hz} Hz measured as {measured} µs, exact {exact}"
        );
    }

    #[test]
    fn a_60_hz_display_is_every_callback_a_frame_with_no_lateness() {
        assert_steady(60.0, &display(60.0, 5_000, 5_000_000), 2);
    }

    #[test]
    fn a_59_94_hz_display_does_not_drift_into_dropped_frames() {
        // Against a 60 Hz grid this sequence is 16 µs later per period, 80 ms
        // late by frame 5000 and several periods dropped on the way.
        assert_steady(
            60_000.0 / 1_001.0,
            &display(60_000.0 / 1_001.0, 5_000, 1),
            2,
        );
    }

    #[test]
    fn a_120_hz_display_runs_every_callback_with_half_the_period() {
        let timestamps = display(120.0, 6_000, 2_000_000);
        assert_steady(120.0, &timestamps, 2);
        let (pacer, started) = run(6_000, &timestamps);
        assert_eq!(started.len(), 6_000);
        let late = started.last().unwrap().period.get();
        assert!((8_332..=8_334).contains(&late), "{late}");
        assert!(pacer.measured_period().unwrap() < NOMINAL_PERIOD);
    }

    #[test]
    fn timestamps_coarsened_to_100_us_neither_drop_nor_run_late() {
        for hz in [60.0, 60_000.0 / 1_001.0, 120.0] {
            assert_steady(hz, &display_coarse(hz, 4_000, 3_000_000), 200);
        }
    }

    #[test]
    fn a_coarsened_display_drops_nothing_and_is_measured_within_one_step() {
        // 59.94 Hz and 120 Hz as 100 µs timestamps alternate their intervals
        // between two neighbouring values.
        for (low, high) in [(16_600, 16_700), (8_300, 8_400)] {
            let gaps: Vec<u64> = (0..2_000)
                .map(|i| if i % 2 == 0 { low } else { high })
                .collect();
            let (pacer, started) = run(2_001, &from_gaps(1_000_000, &gaps));
            assert_eq!(pacer.dropped(), 0, "{low}/{high} µs dropped a frame");
            assert_accounted(&pacer, &started, 2_001);
            // A window of 64 consecutive intervals is 32 of each, and the
            // upper median of an even split is the higher value.
            let period = pacer.measured_period().unwrap();
            assert_eq!(period.get(), high);
            assert!(max(pacer.lateness_us().get(MIN_INTERVALS + 1..).unwrap()) <= 100);
        }
    }

    /// A pseudo-random sequence from `seed`, a 64-bit LCG's high bits, so a
    /// test that needs jitter is the same on every run.
    fn jitter(seed: u64, count: usize, centre: u64, half_width: u64) -> Vec<u64> {
        let mut state = seed;
        (0..count)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                centre - half_width + (state >> 33) % (2 * half_width + 1)
            })
            .collect()
    }

    #[test]
    fn a_jittered_120_hz_display_is_measured_without_a_bias_and_drops_nothing() {
        // Intervals uniform in 8,333 ± 600 µs, as Chrome's timestamps on a
        // 120 Hz display jitter; a low quantile of them reads about 7,700.
        let gaps = jitter(0x5eed_1234, 5_000, 8_333, 600);
        let (pacer, started) = run(5_001, &from_gaps(2_000_000, &gaps));
        assert_eq!(pacer.dropped(), 0);
        assert_accounted(&pacer, &started, 5_001);
        let measured = pacer.measured_period().unwrap().get();
        assert!(measured.abs_diff(8_333) <= 100, "measured {measured}");
        // Once the window is full the median of 64 samples stays near the
        // display's period throughout the run, not only at its end.
        for start in &started[WINDOW..] {
            assert!(start.period.get().abs_diff(8_333) <= 250, "{start:?}");
        }
        let mut lateness = pacer.lateness_us().to_vec();
        lateness.sort_unstable();
        let p50 = lateness[lateness.len() / 2];
        assert!(p50 < 300, "lateness p50 {p50}");
    }

    #[test]
    fn an_overrun_gap_adds_one_period_to_the_window_and_its_raw_gap_to_the_report() {
        // Gaps of 1, 2 and 3 periods of a steady display: all three are one
        // period per period.
        let gaps = [P, 2 * P, P, 3 * P, P, 2 * P, P, P, P, P];
        let requested = gaps.len() as u64 + 1 + (1 + 2 + 1);
        let (pacer, started) = run(requested, &from_gaps(1_000_000, &gaps));
        assert_eq!(pacer.intervals_us(), gaps);
        assert_eq!(pacer.dropped(), 1 + 2 + 1);
        assert_accounted(&pacer, &started, requested);
        assert_eq!(pacer.measured_period(), Some(NOMINAL_PERIOD));
        // A gap that three periods do not divide evenly rounds down.
        let mut gaps = vec![P; 7];
        gaps.push(3 * P + 2);
        let (pacer, _) = run(100, &from_gaps(1_000_000, &gaps));
        assert_eq!(pacer.recent.back().copied(), Some(P));
    }

    #[test]
    fn a_short_interval_is_one_sample_not_the_period() {
        // Fewer than MIN_INTERVALS samples carry the nominal period; from the
        // eighth on, one short sample is not the median of any window.
        for count in [MIN_INTERVALS, MIN_INTERVALS + 1, WINDOW] {
            for at in 0..count {
                let mut gaps = vec![P; count];
                gaps[at] = 12_000;
                let (pacer, _) = run(1_000, &from_gaps(1_000_000, &gaps));
                assert_eq!(pacer.dropped(), 0, "{at}/{count}");
                assert_eq!(
                    pacer.measured_period(),
                    Some(NOMINAL_PERIOD),
                    "{at}/{count}"
                );
            }
        }
    }

    #[test]
    fn the_period_is_the_upper_median_of_the_window() {
        // 12 ms and 16 ms are each one period of the other, so every gap is
        // one sample of its own length whatever the period was.
        let window = |short: usize| {
            let mut gaps = vec![16_000; WINDOW - short];
            gaps.extend(vec![12_000; short]);
            let (pacer, _) = run(1_000, &from_gaps(1_000_000, &gaps));
            pacer.measured_period().unwrap().get()
        };
        // 32 of 64: the 33rd smallest is the first of the longer ones.
        assert_eq!(window(32), 16_000);
        assert_eq!(window(33), 12_000);
        // 4 of 8 is the same rule at MIN_INTERVALS.
        let mut gaps = vec![12_000; MIN_INTERVALS / 2];
        gaps.extend(vec![16_000; MIN_INTERVALS / 2]);
        let (pacer, _) = run(1_000, &from_gaps(1_000_000, &gaps));
        assert_eq!(pacer.measured_period().map(Micros::get), Some(16_000));
        // Three samples of a stall, a missed vsync or an irregular start on
        // either side of the display's cannot be the period at MIN_INTERVALS.
        for outlier in [4_000, 24_000] {
            let mut gaps = vec![outlier; 3];
            gaps.extend(vec![P; MIN_INTERVALS - 3]);
            let (pacer, _) = run(1_000, &from_gaps(1_000_000, &gaps));
            assert_eq!(pacer.measured_period(), Some(NOMINAL_PERIOD), "{outlier}");
        }
    }

    #[test]
    fn a_host_that_overruns_most_callbacks_does_not_change_the_measured_period() {
        // 24 callbacks of 64 arrive a period after the one before and 40 two
        // periods after: more than half of every window is an overrun, and the
        // median of the raw gaps would read as 2P.
        let mut cycle = vec![P; 24];
        cycle.extend(vec![2 * P; 40]);
        let gaps = cycle.repeat(5);
        let timestamps = from_gaps(1_000_000, &gaps);
        let overruns = 5 * 40;
        let requested = timestamps.len() as u64 + overruns;
        let (pacer, started) = run(requested, &timestamps);
        // Each overrun is a missed vsync and is counted as one.
        assert_eq!(pacer.dropped(), overruns);
        assert_accounted(&pacer, &started, requested);
        assert_every_period(&started, NOMINAL_PERIOD);
        assert_eq!(pacer.measured_period(), Some(NOMINAL_PERIOD));
        assert_eq!(max(pacer.lateness_us()), 0);
    }

    #[test]
    fn a_run_that_overruns_for_a_while_and_recovers_measures_the_display_throughout() {
        let mut gaps = [P, 2 * P, 2 * P, 2 * P].repeat(50);
        gaps.extend(vec![P; 100]);
        let timestamps = from_gaps(1_000_000, &gaps);
        let overruns = 150;
        let requested = timestamps.len() as u64 + overruns;
        let (pacer, started) = run(requested, &timestamps);
        assert_eq!(pacer.dropped(), overruns);
        assert_accounted(&pacer, &started, requested);
        assert_every_period(&started, NOMINAL_PERIOD);
        assert_eq!(pacer.measured_period(), Some(NOMINAL_PERIOD));
        // Nothing is dropped once the host has recovered: the 200 intervals
        // that were overrun hold every drop.
        let (overrun_only, _) = run(requested, &timestamps[..=200]);
        assert_eq!(overrun_only.dropped(), overruns);
    }

    #[test]
    fn a_missed_vsync_is_one_dropped_frame() {
        let mut gaps = vec![P; 40];
        gaps.push(2 * P);
        gaps.extend(vec![P; 40]);
        let (pacer, started) = run(83, &from_gaps(1_000_000, &gaps));
        assert_eq!(pacer.dropped(), 1);
        assert_eq!(started.len(), 82);
        assert_accounted(&pacer, &started, 83);
        assert_eq!(started[40].index, 40);
        assert_eq!(started[41].index, 42);
        // The gap is two whole periods, so nothing is left to call lateness.
        assert_eq!(max(pacer.lateness_us()), 0);
    }

    #[test]
    fn a_100_ms_stall_drops_five_frames_and_leaves_the_period_alone() {
        let mut gaps = vec![P; 100];
        gaps.push(100_000);
        gaps.extend(vec![P; 100]);
        let (pacer, started) = run(206, &from_gaps(1_000_000, &gaps));
        // 100 ms is six periods of 16.667 ms: the callback is in the sixth, and
        // the five before it are given up.
        assert_eq!(pacer.dropped(), 5);
        assert_accounted(&pacer, &started, 206);
        assert_eq!(started[101].index, 101 + 5);
        assert_eq!(started[101].start_us - started[100].start_us, 100_000);
        assert!(max(pacer.lateness_us()) <= 2);
        assert_eq!(pacer.measured_period(), Some(NOMINAL_PERIOD));
        assert_eq!(started.last().unwrap().period, NOMINAL_PERIOD);
    }

    #[test]
    fn real_gaps_in_a_59_94_hz_run_are_counted_and_the_rest_is_not() {
        let period = 16_683;
        let mut gaps = vec![period; 500];
        gaps.push(2 * period);
        gaps.extend(vec![period; 500]);
        gaps.push(100_000);
        gaps.extend(vec![period; 500]);
        let (pacer, started) = run(1_509, &from_gaps(7_000_000, &gaps));
        assert_eq!(pacer.dropped(), 1 + 5);
        assert_accounted(&pacer, &started, 1_509);
    }

    #[test]
    fn a_gap_of_one_and_a_half_periods_is_late_and_beyond_it_is_a_drop() {
        let prefix = vec![P; 20];
        let at = |gap: u64| {
            let mut gaps = prefix.clone();
            gaps.push(gap);
            run(100, &from_gaps(1_000_000, &gaps))
        };
        let (pacer, _) = at(P * 3 / 2);
        assert_eq!(pacer.dropped(), 0);
        assert_eq!(pacer.lateness_us().last().copied(), Some(P * 3 / 2 - P));
        let (pacer, _) = at(P * 3 / 2 + 2);
        assert_eq!(pacer.dropped(), 1);
        let (pacer, _) = at(P * 5 / 2 + 2);
        assert_eq!(pacer.dropped(), 2);
    }

    #[test]
    fn a_sequence_that_starts_irregular_settles_on_the_display() {
        let start = [40_000, 5_000, 30_000, 12_000, 22_000, 9_000];
        let mut gaps = start.to_vec();
        gaps.extend(vec![8_333; 300]);
        let timestamps = from_gaps(500_000, &gaps);
        let mut pacer = Pacer::new(1_000);
        let mut dropped_after_settling = None;
        for (i, &t) in timestamps.iter().enumerate() {
            pacer.frame_at(t);
            // Once the irregular intervals are a minority of the window and
            // behind us, the period must be the display's.
            if i == 40 {
                dropped_after_settling = Some(pacer.dropped());
                let period = pacer.measured_period().unwrap().get();
                assert!((8_332..=8_334).contains(&period), "{period}");
            }
        }
        let period = pacer.measured_period().unwrap().get();
        assert!((8_332..=8_334).contains(&period), "{period}");
        // Drops belong to the irregular start; none come after it. The two are
        // the 40 ms and the 30 ms gap judged against the nominal period: the
        // 5 ms one does not make the period 5 ms for the callbacks after it.
        assert_eq!(Some(pacer.dropped()), dropped_after_settling);
        assert_eq!(pacer.dropped(), 2);
    }

    #[test]
    fn the_prior_is_the_period_until_enough_intervals_exist() {
        let mut pacer = Pacer::new(100);
        for (i, t) in display(120.0, MIN_INTERVALS as u64 + 2, 0)
            .into_iter()
            .enumerate()
        {
            let Pace::Frame(start) = pacer.frame_at(t) else {
                panic!("expected a frame");
            };
            // A frame is given the period measured before its own gap, so the
            // first frame to get the measured one is the one after the
            // MIN_INTERVALS-th interval.
            let expected = if i <= MIN_INTERVALS {
                NOMINAL_PERIOD.get()
            } else {
                8_333
            };
            assert!(
                start.period.get().abs_diff(expected) <= 1,
                "frame {i}: {}",
                start.period.get()
            );
        }
        assert_eq!(pacer.dropped(), 0);
    }

    #[test]
    fn a_measured_prior_gives_the_first_frames_the_displays_period() {
        let mut pacer = Pacer::with_prior(100, Micros::new(8_333));
        for (i, t) in display(120.0, MIN_INTERVALS as u64 + 4, 0)
            .into_iter()
            .enumerate()
        {
            let Pace::Frame(start) = pacer.frame_at(t) else {
                panic!("expected a frame");
            };
            assert!(start.period.get().abs_diff(8_333) <= 1, "frame {i}");
        }
        assert_eq!(pacer.dropped(), 0);
        assert_eq!(Pacer::with_prior(1, Micros::ZERO).period(), NOMINAL_PERIOD);
        assert_eq!(Pacer::new(1).period(), NOMINAL_PERIOD);
    }

    #[test]
    fn a_30_hz_display_under_the_60_hz_prior_is_measured_at_60_hz_with_every_gap_dropped() {
        // The limit of the rule: a host that overruns every frame and a display
        // at half the rate send the same timestamps. Each 33.3 ms gap counts as
        // two prior periods and contributes one, so the period stays the
        // faster one and the failure shows in `dropped`.
        let timestamps = display(30.0, 200, 0);
        let (pacer, started) = run(1_000, &timestamps);
        assert_eq!(pacer.dropped(), 199);
        assert_accounted(&pacer, &started, 199 + 200);
        let measured = pacer.measured_period().unwrap().get();
        assert!(measured.abs_diff(P) <= 1, "{measured}");
        assert!(started.iter().all(|s| s.period.get().abs_diff(P) <= 1));
        assert_eq!(pacer.intervals_us().len(), 199);
    }

    #[test]
    fn a_120_hz_display_under_the_60_hz_prior_is_learned_from_its_first_gaps() {
        // A gap of half the prior counts as one period and contributes the real
        // interval, so nothing is dropped on the way.
        let (pacer, started) = run(1_000, &display(120.0, 100, 0));
        assert_eq!(pacer.dropped(), 0);
        assert_eq!(started.len(), 100);
        let measured = pacer.measured_period().unwrap().get();
        assert!(measured.abs_diff(8_333) <= 1, "{measured}");
    }

    #[test]
    fn a_frame_starts_at_its_callbacks_timestamp_from_the_first() {
        let (_, started) = run(3, &[9_000_000, 9_000_000 + P, 9_000_000 + 2 * P + 7]);
        let starts: Vec<u64> = started.iter().map(|s| s.start_us).collect();
        assert_eq!(starts, [0, P, 2 * P + 7]);
    }

    #[test]
    fn a_timestamp_that_does_not_advance_is_a_frame_without_an_interval() {
        let (pacer, started) = run(10, &[1_000_000, 1_000_000]);
        assert_eq!(started.len(), 2);
        assert!(pacer.intervals_us().is_empty());
        assert_eq!(pacer.dropped(), 0);
    }

    #[test]
    fn the_run_ends_after_its_frames_or_when_a_gap_overruns_them() {
        let mut pacer = Pacer::new(3);
        for i in 0..3 {
            assert!(matches!(
                pacer.frame_at(1_000_000 + i * P),
                Pace::Frame(f) if f.index == i
            ));
        }
        assert_eq!(pacer.frame_at(1_000_000 + 3 * P), Pace::Done);
        assert_eq!(pacer.dropped(), 0);

        // A gap past the end gives up only the indices the run had left, and
        // is no interval of the display's.
        let mut pacer = Pacer::new(2);
        assert!(matches!(pacer.frame_at(1_000_000), Pace::Frame(_)));
        assert_eq!(pacer.frame_at(1_000_000 + 5 * P), Pace::Done);
        assert_eq!(pacer.dropped(), 1);
        assert!(pacer.intervals_us().is_empty());
        assert_eq!(pacer.lateness_us().len(), 1);
        assert_eq!(pacer.frame_at(1_000_000 + 6 * P), Pace::Done);
        assert_eq!(pacer.dropped(), 1);

        // A gap that lands on the first index past the end.
        let mut pacer = Pacer::new(3);
        assert!(matches!(pacer.frame_at(1_000_000), Pace::Frame(_)));
        assert_eq!(pacer.frame_at(1_000_000 + 3 * P), Pace::Done);
        assert_eq!(pacer.dropped(), 2);

        // A gap that lands on the last index still starts it.
        let mut pacer = Pacer::new(3);
        assert!(matches!(pacer.frame_at(1_000_000), Pace::Frame(_)));
        assert!(matches!(
            pacer.frame_at(1_000_000 + 2 * P),
            Pace::Frame(f) if f.index == 2
        ));
        assert_eq!(pacer.dropped(), 1);
        assert_eq!(pacer.intervals_us(), [2 * P]);
    }

    #[test]
    fn a_run_that_a_stall_ends_accounts_for_every_index_it_was_asked_for() {
        let mut gaps = vec![P; 60];
        gaps.push(50 * P);
        let (pacer, started) = run(100, &from_gaps(1_000_000, &gaps));
        assert_eq!(started.len(), 60 + 1);
        assert_eq!(pacer.dropped(), 100 - 61);
        assert_accounted(&pacer, &started, 100);
        assert_eq!(pacer.intervals_us().len(), 60);
    }

    #[test]
    fn periods_in_rounds_to_the_nearest_whole_period_above_the_threshold() {
        assert_eq!(periods_in(0, 100), 1);
        assert_eq!(periods_in(150, 100), 1);
        assert_eq!(periods_in(151, 100), 2);
        assert_eq!(periods_in(249, 100), 2);
        assert_eq!(periods_in(251, 100), 3);
        // A zero period is taken as one microsecond rather than divided by.
        assert_eq!(periods_in(1_000, 0), 1_000);
    }

    #[test]
    fn milliseconds_convert_to_rounded_microseconds() {
        assert_eq!(ms_to_us(16.6674), 16_667);
        assert_eq!(ms_to_us(0.0), 0);
        assert_eq!(ms_to_us(-3.0), 0);
        assert_eq!(ms_to_us(f64::NAN), 0);
    }
}
