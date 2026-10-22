//! On-beat delivery from a Web Audio clock: the half that needs no browser.
//!
//! `?sync=onbeat` runs the scheduler under the core's `OnBeat` promise
//! (`beatlm_core::sync::SyncPolicy::OnBeat`): a result lands on the first frame
//! starting at or after a beat. The beats are music the page plays itself, a
//! click on every beat scheduled on an `AudioContext`'s clock, and what this
//! module decides is how those beats become a [`BeatGrid`] on the frame clock
//! the scheduler reads, how the grid is kept current, when inputs are
//! submitted, and what the report says about where results landed. The
//! browser side (`web/audio.rs`) reads the audio clock and plays the clicks;
//! everything it reads arrives here as numbers, so the rules below are tested
//! natively.
//!
//! # The clock chain
//!
//! ```text
//!   audio clock        AudioContext.currentTime: where the audio graph renders.
//!     │                Beat n is the audio time a(n); its click is scheduled there.
//!     │ getOutputTimestamp(): { contextTime, performanceTime }
//!     ▼
//!   performance clock  performance.now(), the clock requestAnimationFrame
//!     │                timestamps are on: a(n) is heard at
//!     │                performanceTime + (a(n) − contextTime)
//!     │ − the run's first callback timestamp (crate::pace)
//!     ▼
//!   frame clock        µs since the first callback: FrameInfo::start, BeatGrid,
//!     │                the scheduler's Due::Beat
//!     ▼
//!   delivery           the first frame starting at or after the beat
//! ```
//!
//! So beat `n` is at `f(n) = a(n) + m` on the frame clock, in microseconds,
//! where the mapping `m` is `performanceTime − contextTime − epoch`
//! ([`Reading::offset_us`], [`Clock`]). The frame clock is the one
//! [`crate::pace`] gives each frame, so a beat and a frame start are compared
//! on one clock and the core needs nothing but the grid.
//!
//! # Which latency, and why
//!
//! `currentTime` is where the graph renders, and that is ahead of what is
//! heard by everything between the graph and the speaker: the browser's and
//! the operating system's buffers and the device. A click scheduled at `a` is
//! heard when the device presents the audio at `a`, so the mapping must be to
//! the presented position, not to the rendered one.
//!
//! - **Used: `getOutputTimestamp()` alone.** It pairs the position the device
//!   presents (`contextTime`) with the `performance.now()` instant it presents
//!   it (`performanceTime`). In Chromium
//!   (`third_party/blink/renderer/platform/audio/audio_destination.cc` and
//!   `modules/webaudio/audio_context.cc`, read 2026-10-22) the position is the
//!   frames rendered less the device's reported output delay, the timestamp is
//!   when that delay was reported, and `outputLatency` returns that same
//!   delay. The pair therefore already contains the output latency, and adding
//!   `outputLatency` to it would count it twice. It is also the more precise
//!   figure: `outputLatency` is rounded to 8 ms for a page without
//!   permission, the timestamp's delay is not.
//! - **Fallback: `currentTime` read beside `performance.now()`, plus
//!   `outputLatency`, or `baseLatency` where `outputLatency` is missing or
//!   zero.** Chromium's timestamp reads zero before the device has reported a
//!   position, and a browser may not have the method; then the rendered
//!   position is moved to the heard one by the output latency, the interval
//!   the specification defines as from the browser handing a buffer to the
//!   system to the device producing it. `baseLatency` is only the graph's own
//!   processing latency, a lower bound of it: a beat mapped with it is mapped
//!   early, so a result can land before its click is heard. Every sample says
//!   which source it came from and the report counts them
//!   (`audio_mapping`), so a run that leaned on the fallback says so.
//! - **Not included, and not measured here:** the air between speaker and ear
//!   (about 3 µs a millimetre), and the display's latency. A frame's start, the
//!   callback's timestamp, is when the display's frame began, not when its
//!   picture reaches the glass: the compositor and the panel add a frame or
//!   more. The landing error is measured on the frame clock, beat to frame
//!   start, which is what the engine promises; how much later the picture is
//!   seen than the click is heard is a question for a camera and a microphone.
//!   The report also gives how far the render position runs ahead of the
//!   heard one (`audio_latency`'s `render_ahead_us`) beside the two
//!   latencies the browser reports, which is what tells whether the browser's
//!   figures agree.
//!
//! # Drift, and keeping the grid current
//!
//! The audio device's clock and the system's are different oscillators, so
//! `m` drifts, and the timestamp jitters, since Chromium refreshes it once per
//! render callback. Every frame adds one sample of `m` to a window of the last
//! [`MAPPING_WINDOW`], and the estimate is their upper median (the rule the
//! pacer's period uses, for the same reason: one outlier is one sample, not the
//! answer). The estimate is adopted at the frame each beat arrives with, when
//! it has moved [`MAPPING_STEP`] or more from the mapping in effect, and only
//! then is a new grid handed to the scheduler
//! (`Scheduler::replace_beat_grid`), in effect from the midpoint between the
//! beat that arrived and the next. A correction smaller than half a beat
//! therefore neither drops nor doubles a beat at the seam, and every beat a job
//! can still be placed on, the next one on, follows the new mapping. A tempo
//! change already announced is put in effect again after it, because the core's
//! `BeatMap::replace_from` drops every grid that takes effect at or after the
//! new one's instant.
//!
//! The step is 500 µs: well under a frame at 120 Hz (8.3 ms), and above the
//! 100 µs to which a page that is not cross-origin isolated may see
//! `performance.now()` coarsened, so quantisation alone does not churn the
//! grid. Deadlines already fixed are not moved by a correction (the core's
//! rule), so the report also measures each delivery against its beat as the
//! mapping at delivery places it (`beat_landing_remapped_us`), which says how
//! much the corrections mattered. A beat's length is rounded to whole
//! microseconds, less than half a microsecond per beat, so 150 µs over 300
//! beats should no correction be adopted: under the step, and reset by every
//! correction.
//!
//! # Submission
//!
//! One input per beat. An input is owed from the first frame starting at or
//! after a beat and is submitted in that frame, or in a later one before the
//! next beat when the job before it still has slices (the depth model's free
//! admission) or its slot is taken ([`Beats::slot_free`]); it is dropped, and
//! counted, when the next beat arrives first. Submitting at a beat gives a job
//! a whole beat before the next one, and the core places it on the first beat
//! its projection can make, normally the next. One result per beat is the use:
//! an effect that changes on the beat. A depth job writes the output slot of
//! its submission index modulo [`SLOTS`], and no input is submitted while the
//! job [`SLOTS`] before it is pending, so a pending job's slot is never
//! reused, whatever order the jobs leave in, and at most `SLOTS - 1` are
//! pending.
//!
//! # The margin
//!
//! The core places a job on the first beat at or after its projected finish
//! plus a margin, and the projection already lays the job's estimated slices
//! over frames of the expected budget behind the work ahead of it, so the
//! margin covers only estimate error and frames budgeted below what was
//! expected. [`DEFAULT_MARGIN`] is 100 ms. What it costs is latency alone: a job
//! whose finish plus margin crosses a beat lands a beat later. At 112 pixels a
//! job's slices spanned about 23 frames in Chrome (the measured base in the
//! crate documentation), about 190 ms at 120 Hz, so an input submitted at a
//! beat of at least 500 ms, the shortest the page accepts, still makes the
//! next beat with 100 ms of margin, which is 12 frames of estimate error at
//! 120 Hz. How much slack the jobs actually had is reported
//! (`beat_slack_us`, the beat less the start of the frame its last slice ran
//! in), which is what a smaller margin would be chosen by; `margin_ms` sets it.
//!
//! # A tempo change
//!
//! `bpm_change=<bpm>@<seconds>` changes the tempo at the first beat at or after
//! `seconds` after the first beat: that beat, the change beat, is the last of
//! the old spacing and the first of the new. The click track plays the whole
//! plan on the audio clock from the start. The scheduler is told on the first
//! frame starting at or after the instant `seconds` names, and the grid of the
//! new tempo is in effect from the change beat on. A deadline fixed before that
//! frame stays where it was (`beatlm_core::schedule`, "Tempo changes"), so a job
//! submitted before the announcement and placed after the change beat lands on
//! a beat of the old grid, which has no click: the report counts those
//! (`tempo_change`'s `fixed_before_after_change`) and how they ended, and checks
//! that every job submitted after the announcement is on a beat of the plan.
//!
//! # Landing
//!
//! A result's landing error is the time from its beat, the deadline the
//! scheduler fixed, to the start of the frame that delivered it. It is early
//! when that frame starts before the beat, and late when it is not the first
//! frame at or after it: the frame before it already started at or after the
//! beat, or it starts a whole period or more after it ([`classify`]). The core
//! delivers only on the first frame at or after a beat, so both counts must be
//! zero; the host checks it with its own record of the deadline and of the
//! frame before.

use std::collections::{BTreeMap, VecDeque};

use beatlm_core::time::{BeatGrid, Micros, TimeError, TimePoint};

use crate::stats::nearest_rank;

/// The tempo when `bpm` is not given, in beats per minute: a beat of 600 ms.
pub const DEFAULT_BPM: f64 = 100.0;

/// The slowest tempo the page accepts, in beats per minute: a beat of 2 s.
/// Slower tempos would work, and only make a run of a few hundred beats long.
pub const MIN_BPM: f64 = 30.0;

/// The fastest tempo the page accepts, in beats per minute: a beat of 500 ms.
/// The demo was sized for it: Depth Anything V2 Small at 112 pixels delivered
/// a result every 23 frames or so in Chrome at 120 Hz, about 190 ms, which with
/// the margin fits a beat of 500 ms and leaves room; at 266 pixels the model
/// is not stable in the browser at all (the crate documentation's measured
/// base).
pub const MAX_BPM: f64 = 120.0;

/// The margin when `margin_ms` is not given (the module documentation, "The
/// margin", says why 100 ms).
pub const DEFAULT_MARGIN: Micros = Micros::new(100_000);

/// The largest margin `margin_ms` accepts: a second, two beats at the fastest
/// tempo, past which every job would land beats late.
pub const MAX_MARGIN: Micros = Micros::new(1_000_000);

/// From the loop's first frame to the first beat, on the audio clock: a
/// second, so the first click is not scheduled in the past whatever the
/// output latency, and the mapping window is full before the first beat's
/// grid is made from it (120 frames at 120 Hz against a window of 64).
pub const LEAD: Micros = Micros::new(1_000_000);

/// What a run asked for in beats lasts past its last beat: a quarter of a
/// second, more than the output latency the first beat is heard after and the
/// frame that delivers on the last beat ([`run_frames`]).
pub const TAIL: Micros = Micros::new(250_000);

/// The samples of the mapping its estimate is the median of: about half a
/// second at 120 Hz.
pub const MAPPING_WINDOW: usize = 64;

/// How far the estimate of the mapping must move before it is adopted and a
/// new grid handed to the scheduler (the module documentation says why
/// 500 µs).
pub const MAPPING_STEP: Micros = Micros::new(500);

/// The output slots of an on-beat depth run, one more than the most jobs
/// pending when an input is submitted ([`Beats::slot_free`]): a job submitted
/// at a beat is normally delivered on the next one, so at that beat's frame it
/// is pending as the next is submitted, and the third slot covers a job placed
/// a beat later.
pub const SLOTS: u32 = 3;

/// The latest a tempo change may be, in seconds after the first beat: an hour,
/// far past any run, so a typing slip is refused rather than planned.
pub const MAX_CHANGE_AFTER_S: f64 = 3_600.0;

/// The most beats `beats` accepts.
pub const MAX_BEATS: u64 = 100_000;

/// How close a deadline must be to a beat of the plan to be on it: a
/// millisecond, far above the rounding of a beat to whole microseconds and far
/// below any beat's length.
const ON_PLAN_TOLERANCE_US: f64 = 1_000.0;

/// How many misses' deadlines the report lists.
pub const DUES_LISTED: usize = 16;

/// A tempo change, `bpm_change=<bpm>@<after_s>`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TempoChange {
    /// The tempo from the change beat on, in beats per minute.
    pub bpm: f64,
    /// When the change is announced, in seconds after the first beat; the
    /// change beat is the first beat at or after it.
    pub after_s: f64,
}

/// The beats a run plays: a tempo, and at most one change of it.
///
/// Times are in seconds after the first beat, beat 0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Tempo {
    /// The tempo from the first beat, in beats per minute.
    pub bpm: f64,
    /// The change, if any.
    pub change: Option<TempoChange>,
}

impl Tempo {
    /// The length of a beat before the change, in seconds.
    #[must_use]
    pub fn beat_s(&self) -> f64 {
        60.0 / self.bpm
    }

    /// The change beat: the first beat of the first tempo at or after the
    /// change's instant.
    #[must_use]
    pub fn change_beat(&self) -> Option<u64> {
        let change = self.change?;
        Some(ceil_index(change.after_s / self.beat_s()))
    }

    /// Seconds from the first beat to beat `n`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn since_first(&self, n: u64) -> f64 {
        match (self.change, self.change_beat()) {
            (Some(change), Some(k)) if n > k => {
                k as f64 * self.beat_s() + (n - k) as f64 * (60.0 / change.bpm)
            }
            _ => n as f64 * self.beat_s(),
        }
    }

    /// The shortest beat of the plan, in whole microseconds: the one before
    /// the change or the one after it.
    #[must_use]
    pub fn shortest_beat(&self) -> Micros {
        let shortest_s = self
            .change
            .map_or(self.beat_s(), |change| self.beat_s().min(60.0 / change.bpm));
        Micros::new(u64::try_from(us_of(shortest_s)).unwrap_or(0))
    }

    /// Seconds from beat `n` to beat `n + 1`.
    #[must_use]
    pub fn beat_after_s(&self, n: u64) -> f64 {
        match (self.change, self.change_beat()) {
            (Some(change), Some(k)) if n >= k => 60.0 / change.bpm,
            _ => self.beat_s(),
        }
    }

    /// The last beat at or before `t_s` seconds after the first, by the float
    /// arithmetic alone; `None` before the first beat. Callers that compare
    /// beats with frame starts correct it in whole microseconds.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn last_at_or_before(&self, t_s: f64) -> Option<u64> {
        if !t_s.is_finite() || t_s < 0.0 {
            return None;
        }
        let old = floor_index(t_s / self.beat_s());
        match (self.change, self.change_beat()) {
            (Some(change), Some(k)) if old >= k => {
                let rest = t_s - k as f64 * self.beat_s();
                Some(k.saturating_add(floor_index(rest / (60.0 / change.bpm))))
            }
            _ => Some(old),
        }
    }

    /// The beat within `tolerance_s` of `t_s` seconds after the first, if one
    /// is.
    #[must_use]
    pub fn beat_near(&self, t_s: f64, tolerance_s: f64) -> Option<u64> {
        let below = self.last_at_or_before(t_s + tolerance_s)?;
        ((self.since_first(below) - t_s).abs() <= tolerance_s).then_some(below)
    }
}

/// The whole number `x` is within a billionth of, if there is one: `60 / 0.6`,
/// which floats compute a little above 100, is 100.
fn whole(x: f64) -> Option<f64> {
    let nearest = x.round();
    ((x - nearest).abs() <= 1e-9 * nearest.abs().max(1.0)).then_some(nearest)
}

/// `x` rounded up to a whole number, taking a value within a billionth of a
/// whole number as that number ([`whole`]), so `60 / 0.6` is 100 and not 101.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn ceil_index(x: f64) -> u64 {
    // A float-to-integer `as` saturates, and clamps NaN and negatives to 0.
    whole(x).unwrap_or_else(|| x.ceil()).max(0.0) as u64
}

/// `x` rounded down to a whole number, with the same tolerance as
/// [`ceil_index`].
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn floor_index(x: f64) -> u64 {
    whole(x).unwrap_or_else(|| x.floor()).max(0.0) as u64
}

/// Seconds as whole microseconds, rounded to nearest.
#[allow(clippy::cast_possible_truncation)]
fn us_of(seconds: f64) -> i64 {
    // A float-to-integer `as` saturates; audio times here are far inside it.
    (seconds * 1e6).round() as i64
}

/// A frame-clock instant in microseconds as a `TimePoint`, clamped at the
/// epoch.
fn point(us: i64) -> TimePoint {
    TimePoint::new(u64::try_from(us).unwrap_or(0))
}

/// The frames a run of `beats` beats lasts at `period`: from the loop's first
/// frame, [`LEAD`] to the first beat, then to beat `beats`, the last that an
/// input (at beat `beats - 1`) is placed on, and [`TAIL`] after it. A zero
/// period is taken as one microsecond.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn run_frames(tempo: &Tempo, beats: u64, period: Micros) -> u64 {
    let span = us_of(tempo.since_first(beats)).max(0) as u64;
    let total = LEAD.get().saturating_add(span).saturating_add(TAIL.get());
    total.div_ceil(period.get().max(1))
}

/// Where a mapping sample came from ([`Reading::offset_us`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    /// `getOutputTimestamp()`, which already holds the output latency.
    OutputTimestamp,
    /// `currentTime` beside `performance.now()`, plus `outputLatency`.
    OutputLatency,
    /// `currentTime` beside `performance.now()`, plus `baseLatency`, a lower
    /// bound of the output latency.
    BaseLatency,
}

/// What the browser said about the audio clock in one frame, as read.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Reading {
    /// `getOutputTimestamp().contextTime`, in seconds; `None` when the method
    /// is missing.
    pub context_time_s: Option<f64>,
    /// `getOutputTimestamp().performanceTime`, in milliseconds on the
    /// `performance.now()` clock.
    pub performance_time_ms: Option<f64>,
    /// `currentTime`, in seconds.
    pub current_time_s: f64,
    /// `performance.now()`, read right after `currentTime`, in milliseconds.
    pub now_ms: f64,
    /// `outputLatency`, in seconds; `None` when the browser does not have it.
    pub output_latency_s: Option<f64>,
    /// `baseLatency`, in seconds; `None` when the browser does not have it.
    pub base_latency_s: Option<f64>,
}

impl Reading {
    /// The timestamp's pair, when it holds a position the device has
    /// presented: both numbers finite and above zero, which Chromium's are not
    /// before the device has reported one.
    fn timestamp(&self) -> Option<(f64, f64)> {
        let context = self.context_time_s.filter(|t| t.is_finite() && *t > 0.0)?;
        let performance = self
            .performance_time_ms
            .filter(|t| t.is_finite() && *t > 0.0)?;
        Some((context, performance))
    }

    /// The mapping this reading gives, in microseconds on the performance
    /// clock: audio time `a` is heard at `a + offset` (both in microseconds),
    /// with where it came from; `None` when the reading gives none.
    ///
    /// The module documentation ("Which latency, and why") says why the
    /// timestamp is taken alone and when the latencies are used.
    #[must_use]
    pub fn offset_us(&self) -> Option<(i64, Source)> {
        if let Some((context, performance)) = self.timestamp() {
            return Some((
                us_of(performance / 1_000.0) - us_of(context),
                Source::OutputTimestamp,
            ));
        }
        if !self.current_time_s.is_finite() || !self.now_ms.is_finite() {
            return None;
        }
        let usable = |latency: Option<f64>| latency.filter(|l| l.is_finite() && *l > 0.0);
        let (latency, source) = match (
            usable(self.output_latency_s),
            self.base_latency_s.filter(|l| l.is_finite() && *l >= 0.0),
        ) {
            (Some(output), _) => (output, Source::OutputLatency),
            (None, Some(base)) => (base, Source::BaseLatency),
            (None, None) => return None,
        };
        Some((
            us_of(self.now_ms / 1_000.0) - us_of(self.current_time_s) + us_of(latency),
            source,
        ))
    }

    /// How far the rendered position runs ahead of the heard one, in
    /// microseconds: `currentTime` less the presented position, less the time
    /// since it was presented. It is what the output path holds, and what
    /// `outputLatency` estimates; `None` without a usable timestamp.
    #[must_use]
    pub fn render_ahead_us(&self) -> Option<i64> {
        let (context, performance) = self.timestamp()?;
        if !self.current_time_s.is_finite() || !self.now_ms.is_finite() {
            return None;
        }
        Some(
            us_of(self.current_time_s)
                - us_of(context)
                - (us_of(self.now_ms / 1_000.0) - us_of(performance / 1_000.0)),
        )
    }
}

/// The mapping from the audio clock to the frame clock, and how it was kept.
#[derive(Debug, Clone, Default)]
pub struct Clock {
    /// The last [`MAPPING_WINDOW`] samples of `m`, frame-clock microseconds
    /// less audio microseconds.
    window: VecDeque<i64>,
    /// The mapping the grids in the scheduler were made with.
    in_effect: Option<i64>,
    timestamp_samples: u64,
    output_latency_samples: u64,
    base_latency_samples: u64,
    unusable: u64,
    updates: u64,
    max_correction_us: u64,
    render_ahead_us: Vec<i64>,
    output_latency_us: Option<i64>,
    base_latency_us: Option<i64>,
}

impl Clock {
    /// Adds one frame's reading, the run's first callback having been at
    /// `epoch_us` on the performance clock.
    pub fn record(&mut self, reading: &Reading, epoch_us: u64) {
        if let Some(latency) = reading.output_latency_s.filter(|l| l.is_finite()) {
            self.output_latency_us = Some(us_of(latency));
        }
        if let Some(latency) = reading.base_latency_s.filter(|l| l.is_finite()) {
            self.base_latency_us = Some(us_of(latency));
        }
        let Some((offset, source)) = reading.offset_us() else {
            self.unusable += 1;
            return;
        };
        match source {
            Source::OutputTimestamp => self.timestamp_samples += 1,
            Source::OutputLatency => self.output_latency_samples += 1,
            Source::BaseLatency => self.base_latency_samples += 1,
        }
        if let Some(ahead) = reading.render_ahead_us() {
            self.render_ahead_us.push(ahead);
        }
        if self.window.len() == MAPPING_WINDOW {
            self.window.pop_front();
        }
        self.window
            .push_back(offset.saturating_sub(i64::try_from(epoch_us).unwrap_or(i64::MAX)));
    }

    /// The estimate of the mapping: the upper median of the window; `None`
    /// before the first usable sample.
    #[must_use]
    pub fn estimate(&self) -> Option<i64> {
        let mut sorted: Vec<i64> = self.window.iter().copied().collect();
        sorted.sort_unstable();
        sorted.get(sorted.len() / 2).copied()
    }

    /// The mapping the scheduler's grids were made with; `None` before the
    /// first was adopted.
    #[must_use]
    pub const fn in_effect(&self) -> Option<i64> {
        self.in_effect
    }

    /// Adopts the estimate when there is no mapping in effect, or when it has
    /// moved [`MAPPING_STEP`] or more from the one in effect, which is then an
    /// update with its correction recorded. Whether the mapping in effect
    /// changed.
    pub fn adopt(&mut self) -> bool {
        let Some(estimate) = self.estimate() else {
            return false;
        };
        match self.in_effect {
            None => {
                self.in_effect = Some(estimate);
                true
            }
            Some(current) => {
                let correction = estimate.abs_diff(current);
                if correction < MAPPING_STEP.get() {
                    return false;
                }
                self.in_effect = Some(estimate);
                self.updates += 1;
                self.max_correction_us = self.max_correction_us.max(correction);
                true
            }
        }
    }

    /// What the report says about the mapping.
    #[must_use]
    pub fn summary(&self) -> MappingSummary {
        MappingSummary {
            timestamp_samples: self.timestamp_samples,
            output_latency_samples: self.output_latency_samples,
            base_latency_samples: self.base_latency_samples,
            unusable: self.unusable,
            updates: self.updates,
            max_correction_us: self.max_correction_us,
            output_latency_us: self.output_latency_us,
            base_latency_us: self.base_latency_us,
            render_ahead_us: Spread::of(&self.render_ahead_us),
        }
    }
}

/// Where a result landed relative to its beat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    /// On a frame that started before the beat.
    Early,
    /// On the first frame starting at or after the beat.
    OnBeat,
    /// On a later frame.
    Late,
}

/// Where a result delivered on a frame starting at `start_us` landed relative
/// to its beat at `beat_us`, the frame before it having started at
/// `previous_start_us` and the frame's period being `period_us`: early before
/// the beat; late when the frame before also started at or after the beat, or
/// when it starts a whole period or more after it, the frame that would have
/// kept it having been skipped.
#[must_use]
pub fn classify(
    beat_us: u64,
    start_us: u64,
    previous_start_us: Option<u64>,
    period_us: u64,
) -> Placement {
    if start_us < beat_us {
        Placement::Early
    } else if previous_start_us.is_some_and(|previous| previous >= beat_us)
        || start_us - beat_us >= period_us
    {
        Placement::Late
    } else {
        Placement::OnBeat
    }
}

/// The smallest, median, 99th percentile and largest of signed samples, by
/// nearest rank, the report's statistic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Spread {
    /// How many samples.
    pub count: usize,
    /// The smallest.
    pub min: i64,
    /// The 50th percentile.
    pub p50: i64,
    /// The 99th percentile.
    pub p99: i64,
    /// The largest.
    pub max: i64,
}

impl Spread {
    /// The spread of `samples`; `None` when there are none.
    #[must_use]
    pub fn of(samples: &[i64]) -> Option<Self> {
        let mut sorted = samples.to_vec();
        sorted.sort_unstable();
        Some(Self {
            count: sorted.len(),
            min: *sorted.first()?,
            p50: nearest_rank(&sorted, 50)?,
            p99: nearest_rank(&sorted, 99)?,
            max: *sorted.last()?,
        })
    }
}

/// What the report says about the audio clock's mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MappingSummary {
    /// Samples taken from `getOutputTimestamp()`.
    pub timestamp_samples: u64,
    /// Samples taken from `currentTime` and `outputLatency`.
    pub output_latency_samples: u64,
    /// Samples taken from `currentTime` and `baseLatency`.
    pub base_latency_samples: u64,
    /// Frames whose reading gave no sample.
    pub unusable: u64,
    /// Corrections adopted after the first mapping.
    pub updates: u64,
    /// The largest of them, in microseconds.
    pub max_correction_us: u64,
    /// The last `outputLatency` read, in microseconds.
    pub output_latency_us: Option<i64>,
    /// The last `baseLatency` read, in microseconds.
    pub base_latency_us: Option<i64>,
    /// How far the rendered position ran ahead of the heard one
    /// ([`Reading::render_ahead_us`]).
    pub render_ahead_us: Option<Spread>,
}

/// What the report says about a tempo change.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ChangeSummary {
    /// The tempo before it.
    pub from_bpm: f64,
    /// The tempo after it.
    pub to_bpm: f64,
    /// When it was announced, in seconds after the first beat.
    pub after_s: f64,
    /// The change beat's index.
    pub change_beat: u64,
    /// The frame-clock start of the frame it was announced on; `None` when the
    /// run ended first.
    pub announced_us: Option<u64>,
    /// The change beat on the frame clock, as mapped when announced.
    pub change_us: Option<u64>,
    /// Jobs pending at the announcement with a deadline after the change beat:
    /// fixed on the old grid, and kept there.
    pub fixed_before_after_change: u64,
    /// How many of them were delivered.
    pub kept_delivered: u64,
    /// How many of them were missed.
    pub kept_missed: u64,
    /// Jobs submitted after the announcement.
    pub later_jobs: u64,
    /// How many of them have a deadline on no beat of the plan; must be zero.
    pub later_off_plan: u64,
}

/// Everything the report's on-beat lines say.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Summary {
    /// The tempo the run started with.
    pub bpm: f64,
    /// Its beat, in whole microseconds.
    pub beat_us: u64,
    /// The margin the scheduler places jobs with.
    pub margin_us: u64,
    /// The beats the run was asked for, if it was asked in beats.
    pub beats_asked: Option<u64>,
    /// The audio context's sample rate.
    pub sample_rate_hz: Option<f32>,
    /// Time from each beat to the start of the frame that delivered its
    /// result, for the results not delivered early.
    pub landing: Option<Spread>,
    /// Results delivered on a frame starting before their beat.
    pub early: u64,
    /// Results delivered on a frame after the first at or after their beat.
    pub late: u64,
    /// The same deliveries, from the beat as the mapping at delivery places
    /// it.
    pub remapped: Option<Spread>,
    /// For each delivered job, its beat less the start of the frame its last
    /// slice ran in.
    pub slack: Option<Spread>,
    /// Beats after the first, up to the run's last frame and the beats asked
    /// for: the ones a job could be placed on.
    pub beats_elapsed: u64,
    /// How many of them no job was placed on.
    pub beats_without_job: u64,
    /// How many had more than one.
    pub beats_shared: u64,
    /// Inputs owed: one per beat that arrived, up to the beats asked for.
    pub inputs_owed: u64,
    /// Inputs submitted.
    pub inputs_submitted: u64,
    /// Inputs dropped because the next beat came before they could be.
    pub inputs_skipped: u64,
    /// Misses reported for jobs this tracked.
    pub missed: u64,
    /// The first [`DUES_LISTED`] misses' deadlines, frame-clock microseconds.
    pub missed_dues_us: Vec<u64>,
    /// Misses whose reported deadline was not the one fixed at submission;
    /// must be zero.
    pub deadlines_moved: u64,
    /// Jobs still pending at the end.
    pub pending: u64,
    /// The mapping.
    pub mapping: MappingSummary,
    /// Clicks the page scheduled on the audio clock; the host sets it.
    pub clicks_scheduled: u64,
    /// Beats whose time had passed before their click could be scheduled, so
    /// that nothing was heard on them; the host sets it.
    pub clicks_late: u64,
    /// The tempo change, if the run had one.
    pub change: Option<ChangeSummary>,
}

/// One job, from its submission until it leaves the scheduler.
#[derive(Debug, Clone, Copy)]
struct Job {
    deadline_us: u64,
    /// The mapping in effect at submission, which the grid its beat came from
    /// was made with.
    offset_us: i64,
    /// The beat of the plan the deadline is on.
    beat: Option<u64>,
    /// The start of the frame its last slice ran in.
    completed_us: Option<u64>,
    /// Pending at the tempo change's announcement with a deadline after the
    /// change beat.
    kept_on_old_grid: bool,
}

/// What one frame does about the beats.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct FrameBeats {
    /// Grids to put in effect, in this order, each from its instant
    /// (`Scheduler::replace_beat_grid`).
    pub grids: Vec<(TimePoint, BeatGrid)>,
    /// Whether an input is owed, and is to be submitted if the job before
    /// lets it.
    pub owed: bool,
}

/// The on-beat side of a run: the beats, the mapping, the inputs owed and
/// where results landed.
#[derive(Debug, Clone)]
pub struct Beats {
    tempo: Tempo,
    margin: Micros,
    beats_asked: Option<u64>,
    /// Beat 0 on the audio clock, in seconds; `None` before [`Beats::begin`].
    origin_s: Option<f64>,
    clock: Clock,
    sample_rate_hz: Option<f32>,
    /// The latest beat that arrived.
    last_beat: Option<u64>,
    /// The beat whose input is owed.
    owed: Option<u64>,
    inputs_owed: u64,
    inputs_submitted: u64,
    inputs_skipped: u64,
    /// The start of the frame the change was announced on, and the change
    /// beat on the frame clock.
    announced: Option<(u64, u64)>,
    jobs: BTreeMap<u64, Job>,
    jobs_on_beat: BTreeMap<u64, u32>,
    landing_us: Vec<i64>,
    early: u64,
    late: u64,
    remapped_us: Vec<i64>,
    slack_us: Vec<i64>,
    missed: u64,
    missed_dues_us: Vec<u64>,
    moved: u64,
    change: ChangeSummary,
    last_start_us: Option<u64>,
}

impl Beats {
    /// The beats of `tempo`, placed with `margin`, for a run of `beats_asked`
    /// beats if it was asked in beats.
    #[must_use]
    pub fn new(tempo: Tempo, margin: Micros, beats_asked: Option<u64>) -> Self {
        let change = tempo
            .change
            .map_or_else(ChangeSummary::default, |change| ChangeSummary {
                from_bpm: tempo.bpm,
                to_bpm: change.bpm,
                after_s: change.after_s,
                change_beat: tempo.change_beat().unwrap_or(0),
                ..ChangeSummary::default()
            });
        Self {
            tempo,
            margin,
            beats_asked,
            origin_s: None,
            clock: Clock::default(),
            sample_rate_hz: None,
            last_beat: None,
            owed: None,
            inputs_owed: 0,
            inputs_submitted: 0,
            inputs_skipped: 0,
            announced: None,
            jobs: BTreeMap::new(),
            jobs_on_beat: BTreeMap::new(),
            landing_us: Vec::new(),
            early: 0,
            late: 0,
            remapped_us: Vec::new(),
            slack_us: Vec::new(),
            missed: 0,
            missed_dues_us: Vec::new(),
            moved: 0,
            change,
            last_start_us: None,
        }
    }

    /// The beats played.
    #[must_use]
    pub const fn tempo(&self) -> &Tempo {
        &self.tempo
    }

    /// The beats asked for, if the run was asked in beats.
    #[must_use]
    pub const fn beats_asked(&self) -> Option<u64> {
        self.beats_asked
    }

    /// Places the first beat [`LEAD`] after `current_time_s` on the audio
    /// clock, read in the loop's first frame, and notes the context's sample
    /// rate for the report.
    #[allow(clippy::cast_precision_loss)]
    pub fn begin(&mut self, current_time_s: f64, sample_rate_hz: f32) {
        self.origin_s = Some(current_time_s + LEAD.get() as f64 / 1e6);
        self.sample_rate_hz = Some(sample_rate_hz);
    }

    /// Beat `n` on the audio clock, in seconds; `None` before
    /// [`Beats::begin`].
    #[must_use]
    pub fn audio_time_s(&self, n: u64) -> Option<f64> {
        Some(self.origin_s? + self.tempo.since_first(n))
    }

    /// Adds one frame's reading of the audio clock.
    pub fn record(&mut self, reading: &Reading, epoch_us: u64) {
        self.clock.record(reading, epoch_us);
    }

    /// Beat `n` on the frame clock with mapping `m`, in microseconds.
    fn mapped(&self, origin_s: f64, n: u64, m: i64) -> i64 {
        us_of(origin_s + self.tempo.since_first(n)).saturating_add(m)
    }

    /// The last beat at or before frame-clock instant `t` with mapping `m`,
    /// exact in whole microseconds.
    #[allow(clippy::cast_precision_loss)]
    fn last_beat_at(&self, origin_s: f64, t: i64, m: i64) -> Option<u64> {
        let guess = self
            .tempo
            .last_at_or_before(t.saturating_sub(m) as f64 / 1e6 - origin_s);
        let mut n = match guess {
            Some(n) => n,
            None if self.mapped(origin_s, 0, m) <= t => 0,
            None => return None,
        };
        while self.mapped(origin_s, n.saturating_add(1), m) <= t {
            n += 1;
        }
        while self.mapped(origin_s, n, m) > t {
            n = n.checked_sub(1)?;
        }
        Some(n)
    }

    /// The grid of the beats from beat `n` on, in effect from `from`, clamped
    /// to its origin. Its spacing is the old tempo's until the change is
    /// announced, so a correction that comes before the announcement does not
    /// tell the scheduler of the change early.
    fn grid(
        &self,
        origin_s: f64,
        from: i64,
        n: u64,
        m: i64,
    ) -> Result<(TimePoint, BeatGrid), TimeError> {
        let origin = self.mapped(origin_s, n, m);
        let spacing_s = if self.announced.is_some() {
            self.tempo.beat_after_s(n)
        } else {
            self.tempo.beat_s()
        };
        let beat = u64::try_from(us_of(spacing_s)).unwrap_or(0);
        Ok((
            point(from.min(origin)),
            BeatGrid::new(point(origin), Micros::new(beat))?,
        ))
    }

    /// The grids that put beat `n` on in effect with mapping `m`: from the
    /// epoch for the first, else from the midpoint between beat `n - 1` and
    /// beat `n`; then the change's again when it is announced and still ahead.
    fn grids_from(
        &self,
        origin_s: f64,
        n: u64,
        m: i64,
        first: bool,
    ) -> Result<Vec<(TimePoint, BeatGrid)>, TimeError> {
        let at = self.mapped(origin_s, n, m);
        let from = if first {
            0
        } else {
            let before = self.mapped(origin_s, n.saturating_sub(1), m);
            before + (at - before) / 2
        };
        let mut grids = vec![self.grid(origin_s, from, n, m)?];
        let ahead = self
            .tempo
            .change_beat()
            .filter(|&k| n < k && self.announced.is_some());
        if let Some(k) = ahead {
            let change = self.mapped(origin_s, k, m);
            grids.push(self.grid(origin_s, change, k, m)?);
        }
        Ok(grids)
    }

    /// Decides what the frame starting at `start_us` does about the beats:
    /// announces the tempo change when its instant has come, and when a beat
    /// has arrived adopts the mapping (the first, or a correction) and owes
    /// its input. Called each frame after [`Beats::record`] and before the
    /// frame's submission.
    ///
    /// # Errors
    ///
    /// A beat that rounds to zero microseconds, which the page's tempo range
    /// rules out.
    pub fn frame(&mut self, start_us: u64) -> Result<FrameBeats, TimeError> {
        self.last_start_us = Some(start_us);
        let mut grids = Vec::new();
        let mapping = self.clock.in_effect().or_else(|| self.clock.estimate());
        if let (Some(origin_s), Some(m)) = (self.origin_s, mapping) {
            let start = i64::try_from(start_us).unwrap_or(i64::MAX);
            self.announce(origin_s, start, start_us, m, &mut grids)?;
            let arrived = self
                .last_beat_at(origin_s, start, m)
                .filter(|&n| self.last_beat.is_none_or(|last| n > last));
            if let Some(n) = arrived {
                let first = self.clock.in_effect().is_none();
                let changed = self.clock.adopt();
                let m = self.clock.in_effect().unwrap_or(m);
                if first {
                    grids = self.grids_from(origin_s, n, m, true)?;
                } else if changed {
                    grids.extend(self.grids_from(origin_s, n.saturating_add(1), m, false)?);
                }
                self.arrived(n);
            }
        }
        Ok(FrameBeats {
            grids,
            owed: self.owed.is_some(),
        })
    }

    /// Announces the tempo change on the first frame at or after its instant:
    /// the grid of the new tempo from the change beat on, and the jobs
    /// pending with a deadline after it marked as kept on the old grid. Before
    /// the first mapping is adopted nothing is handed out: the first grids
    /// carry the change.
    fn announce(
        &mut self,
        origin_s: f64,
        start: i64,
        start_us: u64,
        m: i64,
        grids: &mut Vec<(TimePoint, BeatGrid)>,
    ) -> Result<(), TimeError> {
        let (Some(change), Some(k)) = (self.tempo.change, self.tempo.change_beat()) else {
            return Ok(());
        };
        if self.announced.is_some() || start < us_of(origin_s + change.after_s).saturating_add(m) {
            return Ok(());
        }
        let change_at = self.mapped(origin_s, k, m);
        let change_us = u64::try_from(change_at).unwrap_or(0);
        self.announced = Some((start_us, change_us));
        self.change.announced_us = Some(start_us);
        self.change.change_us = Some(change_us);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let tolerance = ON_PLAN_TOLERANCE_US as u64;
        for job in self.jobs.values_mut() {
            if job.deadline_us > change_us.saturating_add(tolerance) {
                job.kept_on_old_grid = true;
                self.change.fixed_before_after_change += 1;
            }
        }
        if self.clock.in_effect().is_some() {
            grids.push(self.grid(origin_s, change_at, k, m)?);
        }
        Ok(())
    }

    /// Beat `n` arrived: the inputs of the beats since the last one, up to the
    /// beats asked for, are owed, and every one but the latest, with an owed
    /// input not yet submitted, is dropped.
    fn arrived(&mut self, n: u64) {
        let first = self.last_beat.map_or(0, |last| last.saturating_add(1));
        self.last_beat = Some(n);
        let limit = self.beats_asked.unwrap_or(u64::MAX);
        let new = n
            .min(limit.saturating_sub(1))
            .saturating_add(1)
            .saturating_sub(first);
        if new == 0 {
            if self.owed.take().is_some() {
                self.inputs_skipped += 1;
            }
            return;
        }
        // The latest beat's input is owed only while it is one asked for;
        // past them, every new beat's input is dropped.
        let latest_owed = n < limit;
        self.inputs_owed += new;
        self.inputs_skipped += new - u64::from(latest_owed) + u64::from(self.owed.is_some());
        self.owed = latest_owed.then_some(n);
    }

    /// The beat of the plan a deadline fixed under mapping `m` is on.
    #[allow(clippy::cast_precision_loss)]
    fn beat_of(&self, deadline_us: u64, m: i64) -> Option<u64> {
        let origin_s = self.origin_s?;
        let audio_us = i64::try_from(deadline_us).ok()?.saturating_sub(m);
        self.tempo
            .beat_near(audio_us as f64 / 1e6 - origin_s, ON_PLAN_TOLERANCE_US / 1e6)
    }

    /// Records that job `job` was submitted and its deadline fixed at
    /// `deadline_us` on the frame clock; the owed input is no longer owed.
    pub fn submitted(&mut self, job: u64, deadline_us: u64) {
        self.owed = None;
        self.inputs_submitted += 1;
        let offset_us = self.clock.in_effect().unwrap_or(0);
        let beat = self.beat_of(deadline_us, offset_us);
        if self.announced.is_some() {
            self.change.later_jobs += 1;
            if beat.is_none() {
                self.change.later_off_plan += 1;
            }
        }
        self.jobs.insert(
            job,
            Job {
                deadline_us,
                offset_us,
                beat,
                completed_us: None,
                kept_on_old_grid: false,
            },
        );
    }

    /// Whether job `next`, the next to be submitted (jobs are numbered in
    /// submission order from zero), may be: no job submitted [`SLOTS`] or more
    /// before it is still pending, so the slot `next` shares with them
    /// (`next % SLOTS`) is free. A job counts as pending until the plan that
    /// delivers or misses it has been recorded, so in the frame that delivers
    /// a job its slot is not yet given to another.
    #[must_use]
    pub fn slot_free(&self, next: u64) -> bool {
        self.jobs
            .keys()
            .next()
            .is_none_or(|&oldest| next.saturating_sub(oldest) < u64::from(SLOTS))
    }

    /// Records that job `job`'s last slice ran in the frame starting at
    /// `start_us`.
    pub fn completed(&mut self, job: u64, start_us: u64) {
        if let Some(record) = self.jobs.get_mut(&job) {
            record.completed_us = Some(start_us);
        }
    }

    /// Records that job `job` was delivered on the frame starting at
    /// `start_us`, of period `period_us`, the frame before it having started at
    /// `previous_start_us`.
    pub fn delivered(
        &mut self,
        job: u64,
        start_us: u64,
        previous_start_us: Option<u64>,
        period_us: u64,
    ) {
        let Some(record) = self.jobs.remove(&job) else {
            return;
        };
        match classify(record.deadline_us, start_us, previous_start_us, period_us) {
            Placement::Early => self.early += 1,
            Placement::Late => self.late += 1,
            Placement::OnBeat => {}
        }
        let start = i64::try_from(start_us).unwrap_or(i64::MAX);
        let deadline = i64::try_from(record.deadline_us).unwrap_or(i64::MAX);
        if start >= deadline {
            self.landing_us.push(start - deadline);
        }
        if let Some(now) = self.clock.estimate().or(self.clock.in_effect()) {
            let heard = deadline
                .saturating_sub(record.offset_us)
                .saturating_add(now);
            self.remapped_us.push(start.saturating_sub(heard));
        }
        if let Some(completed) = record.completed_us {
            let completed = i64::try_from(completed).unwrap_or(i64::MAX);
            self.slack_us.push(deadline.saturating_sub(completed));
        }
        if record.kept_on_old_grid {
            self.change.kept_delivered += 1;
        }
        self.on_beat(record.beat);
    }

    /// Records that job `job` was reported missed for the deadline `due_us`.
    pub fn missed(&mut self, job: u64, due_us: u64) {
        let Some(record) = self.jobs.remove(&job) else {
            return;
        };
        self.missed += 1;
        if self.missed_dues_us.len() < DUES_LISTED {
            self.missed_dues_us.push(due_us);
        }
        if due_us != record.deadline_us {
            self.moved += 1;
        }
        if record.kept_on_old_grid {
            self.change.kept_missed += 1;
        }
        self.on_beat(record.beat);
    }

    fn on_beat(&mut self, beat: Option<u64>) {
        if let Some(beat) = beat {
            *self.jobs_on_beat.entry(beat).or_default() += 1;
        }
    }

    /// The report's on-beat facts, the run having ended.
    #[must_use]
    pub fn summary(&self) -> Summary {
        let mut on_beat = self.jobs_on_beat.clone();
        for job in self.jobs.values() {
            if let Some(beat) = job.beat {
                *on_beat.entry(beat).or_default() += 1;
            }
        }
        let elapsed = match (self.origin_s, self.clock.in_effect(), self.last_start_us) {
            (Some(origin_s), Some(m), Some(last)) => self
                .last_beat_at(origin_s, i64::try_from(last).unwrap_or(i64::MAX), m)
                .unwrap_or(0),
            _ => 0,
        }
        .min(self.beats_asked.unwrap_or(u64::MAX));
        // Beats 1 to `elapsed`; a half-open range, which is empty rather than
        // refused when no beat after the first has come.
        let targetable = 1..elapsed.saturating_add(1);
        let with_job = on_beat.range(targetable.clone()).count() as u64;
        let shared = on_beat.range(targetable).filter(|(_, n)| **n > 1).count() as u64;
        Summary {
            bpm: self.tempo.bpm,
            beat_us: u64::try_from(us_of(self.tempo.beat_s())).unwrap_or(0),
            margin_us: self.margin.get(),
            beats_asked: self.beats_asked,
            sample_rate_hz: self.sample_rate_hz,
            landing: Spread::of(&self.landing_us),
            early: self.early,
            late: self.late,
            remapped: Spread::of(&self.remapped_us),
            slack: Spread::of(&self.slack_us),
            beats_elapsed: elapsed,
            beats_without_job: elapsed - with_job,
            beats_shared: shared,
            inputs_owed: self.inputs_owed,
            inputs_submitted: self.inputs_submitted,
            inputs_skipped: self.inputs_skipped,
            missed: self.missed,
            missed_dues_us: self.missed_dues_us.clone(),
            deadlines_moved: self.moved,
            pending: self.jobs.len() as u64,
            mapping: self.clock.summary(),
            clicks_scheduled: 0,
            clicks_late: 0,
            change: self.tempo.change.map(|_| self.change),
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::depth::Screen;
    use beatlm_core::budget::Budget;
    use beatlm_core::cost::{CostModel, Estimator, SliceId};
    use beatlm_core::schedule::{Due, Scheduler};
    use beatlm_core::sync::{MissPolicy, SyncPolicy};
    use beatlm_core::time::{FrameIndex, FrameInfo};

    const PERIOD: u64 = 8_333;

    fn tempo(bpm: f64) -> Tempo {
        Tempo { bpm, change: None }
    }

    /// A reading whose timestamp says audio time `audio_s` is heard at
    /// performance time `heard_ms`.
    fn stamped(audio_s: f64, heard_ms: f64) -> Reading {
        Reading {
            context_time_s: Some(audio_s),
            performance_time_ms: Some(heard_ms),
            current_time_s: audio_s + 0.030,
            now_ms: heard_ms + 2.0,
            output_latency_s: Some(0.024),
            base_latency_s: Some(0.005),
        }
    }

    #[test]
    fn the_timestamp_alone_maps_audio_to_heard_time() {
        // Audio second 2 is heard at performance time 10 s: a(n) + 8 s.
        let reading = stamped(2.0, 10_000.0);
        assert_eq!(
            reading.offset_us(),
            Some((8_000_000, Source::OutputTimestamp))
        );
        // The render position is 30 ms ahead of the presented one, read 2 ms
        // after it was presented: 28 ms in the output path.
        assert_eq!(reading.render_ahead_us(), Some(28_000));
    }

    #[test]
    fn without_a_timestamp_the_render_position_is_moved_by_the_output_latency() {
        let reading = Reading {
            context_time_s: Some(0.0),
            performance_time_ms: Some(0.0),
            current_time_s: 2.0,
            now_ms: 10_000.0,
            output_latency_s: Some(0.024),
            base_latency_s: Some(0.005),
        };
        assert_eq!(
            reading.offset_us(),
            Some((8_024_000, Source::OutputLatency))
        );
        assert_eq!(reading.render_ahead_us(), None);
        // No outputLatency, or a zero one: baseLatency, a lower bound.
        for output in [None, Some(0.0)] {
            let reading = Reading {
                output_latency_s: output,
                context_time_s: None,
                performance_time_ms: None,
                ..reading
            };
            assert_eq!(reading.offset_us(), Some((8_005_000, Source::BaseLatency)));
        }
        let nothing = Reading {
            output_latency_s: None,
            base_latency_s: None,
            ..reading
        };
        assert_eq!(nothing.offset_us(), None);
    }

    #[test]
    fn the_mapping_is_the_median_and_moves_only_by_a_step() {
        let mut clock = Clock::default();
        assert!(!clock.adopt());
        // 8 s on the performance clock, less an epoch of 5 s.
        for jitter in [0.0, 0.3, -0.2, 9.0, 0.1] {
            clock.record(&stamped(2.0, 10_000.0 + jitter), 5_000_000);
        }
        assert_eq!(clock.estimate(), Some(3_000_100));
        assert!(clock.adopt());
        assert_eq!(clock.in_effect(), Some(3_000_100));
        // The first mapping is not a correction.
        assert_eq!(clock.summary().updates, 0);
        // Moving 400 µs is below the step and is not adopted.
        for _ in 0..MAPPING_WINDOW {
            clock.record(&stamped(2.0, 10_000.5), 5_000_000);
        }
        assert!(!clock.adopt());
        assert_eq!(clock.in_effect(), Some(3_000_100));
        for _ in 0..MAPPING_WINDOW {
            clock.record(&stamped(2.0, 10_001.3), 5_000_000);
        }
        assert!(clock.adopt());
        assert_eq!(clock.in_effect(), Some(3_001_300));
        let summary = clock.summary();
        assert_eq!(summary.updates, 1);
        assert_eq!(summary.max_correction_us, 1_200);
        assert_eq!(summary.timestamp_samples, 5 + 2 * MAPPING_WINDOW as u64);
        assert_eq!(summary.output_latency_us, Some(24_000));
    }

    #[test]
    fn a_tempo_change_snaps_to_the_first_beat_at_or_after_it() {
        let plan = Tempo {
            bpm: 100.0,
            change: Some(TempoChange {
                bpm: 120.0,
                after_s: 60.0,
            }),
        };
        // 60 s is beat 100 at 600 ms exactly, not 101.
        assert_eq!(plan.change_beat(), Some(100));
        assert!((plan.since_first(100) - 60.0).abs() < 1e-9);
        assert!((plan.since_first(102) - 61.0).abs() < 1e-9);
        assert!((plan.beat_after_s(99) - 0.6).abs() < 1e-12);
        assert!((plan.beat_after_s(100) - 0.5).abs() < 1e-12);
        assert_eq!(plan.last_at_or_before(60.99), Some(101));
        assert_eq!(plan.last_at_or_before(61.0), Some(102));
        assert_eq!(plan.last_at_or_before(-0.1), None);
        let between = Tempo {
            change: Some(TempoChange {
                bpm: 60.0,
                after_s: 1.3,
            }),
            ..plan
        };
        assert_eq!(between.change_beat(), Some(3));
        assert_eq!(between.beat_near(1.8 + 1.0, 0.001), Some(4));
        assert_eq!(between.beat_near(2.4, 0.001), None);
    }

    #[test]
    fn a_run_in_beats_lasts_the_lead_the_beats_and_the_tail() {
        // 300 beats at 100 BPM are 180 s; with the lead and the tail 181.25 s.
        assert_eq!(
            run_frames(&tempo(100.0), 300, Micros::new(8_333)),
            181_250_000_u64.div_ceil(8_333)
        );
        let changed = Tempo {
            bpm: 100.0,
            change: Some(TempoChange {
                bpm: 120.0,
                after_s: 60.0,
            }),
        };
        // 100 beats of 600 ms and 200 of 500 ms: 160 s.
        assert_eq!(
            run_frames(&changed, 300, Micros::new(10_000)),
            (1_000_000 + 160_000_000 + 250_000) / 10_000
        );
    }

    #[test]
    fn landing_is_early_on_beat_or_late() {
        assert_eq!(classify(1_000, 999, Some(990), PERIOD), Placement::Early);
        assert_eq!(classify(1_000, 1_000, Some(990), PERIOD), Placement::OnBeat);
        assert_eq!(classify(1_000, 9_000, Some(999), PERIOD), Placement::OnBeat);
        // The frame before was already at or after the beat.
        assert_eq!(classify(1_000, 9_000, Some(1_000), PERIOD), Placement::Late);
        // A frame was skipped: a whole period after the beat.
        assert_eq!(
            classify(1_000, 1_000 + PERIOD, None, PERIOD),
            Placement::Late
        );
        assert_eq!(
            classify(1_000, 1_000 + PERIOD - 1, None, PERIOD),
            Placement::OnBeat
        );
    }

    #[test]
    fn the_spread_is_by_nearest_rank() {
        assert_eq!(Spread::of(&[]), None);
        let samples: Vec<i64> = (1..=100).rev().collect();
        assert_eq!(
            Spread::of(&samples),
            Some(Spread {
                count: 100,
                min: 1,
                p50: 50,
                p99: 99,
                max: 100
            })
        );
        assert_eq!(
            Spread::of(&[-5, 7]),
            Some(Spread {
                count: 2,
                min: -5,
                p50: -5,
                p99: 7,
                max: 7
            })
        );
    }

    /// A run against the core's scheduler: frames every [`PERIOD`] µs from the
    /// epoch, a steady audio mapping, one input per beat of `slices` slices
    /// of 1 ms each.
    struct Sim {
        beats: Beats,
        scheduler: Scheduler,
        cost: CostModel,
        slices: Vec<SliceId>,
        /// Frame-clock µs less audio µs.
        m_us: i64,
        epoch_us: u64,
        frame: u64,
        previous: Option<u64>,
        installed: Vec<(TimePoint, BeatGrid)>,
        /// The page's output slots and display as the frames write them:
        /// every frame checks that no slot is reused before its job's
        /// delivery and that only a delivered job is drawn.
        screen: Screen,
        /// Frames given no budget.
        starved: std::ops::Range<u64>,
    }

    impl Sim {
        fn new(plan: Tempo, beats: Option<u64>) -> Self {
            let budget =
                Budget::for_frame(Micros::new(PERIOD), Micros::new(2_000), Micros::new(2_000));
            let placeholder = BeatGrid::new(TimePoint::new(0), Micros::new(600_000)).unwrap();
            let mut sim = Self {
                beats: Beats::new(plan, DEFAULT_MARGIN, beats),
                scheduler: Scheduler::new(
                    SyncPolicy::OnBeat {
                        margin: DEFAULT_MARGIN,
                        grid: placeholder,
                    },
                    MissPolicy::HoldPrevious,
                    budget,
                ),
                cost: CostModel::new(Micros::new(1_000), Estimator::default()).unwrap(),
                slices: (0..20).map(SliceId::new).collect(),
                m_us: 0,
                epoch_us: 5_000_000,
                frame: 0,
                previous: None,
                installed: Vec::new(),
                screen: Screen::new(SLOTS),
                starved: 0..0,
            };
            // The audio clock renders 3.0 s at the first frame, frame-clock 0,
            // and what it renders is heard 40 ms later.
            sim.m_us = -3_000_000 + 40_000;
            sim.beats.begin(3.0, 48_000.0);
            sim
        }

        /// The reading of an audio clock at frame-clock instant `t_us`.
        fn reading(&self, t_us: u64) -> Reading {
            let audio_us = t_us as i64 - self.m_us;
            stamped(
                audio_us as f64 / 1e6,
                (t_us + self.epoch_us) as f64 / 1_000.0,
            )
        }

        /// Runs one frame; returns its start.
        fn step(&mut self) -> u64 {
            let start = self.frame * PERIOD;
            let reading = self.reading(start);
            self.beats.record(&reading, self.epoch_us);
            let decided = self.beats.frame(start).unwrap();
            for &(from, grid) in &decided.grids {
                self.scheduler.replace_beat_grid(from, grid).unwrap();
                self.installed.push((from, grid));
            }
            let info = FrameInfo {
                index: FrameIndex::new(self.frame),
                start: TimePoint::new(start),
                period: Micros::new(PERIOD),
            };
            let next = self.beats.summary().inputs_submitted;
            if decided.owed && self.beats.slot_free(next) {
                let job = self
                    .scheduler
                    .submit(info, self.slices.clone(), &self.cost)
                    .unwrap();
                let Some(Due::Beat(due)) = self.scheduler.due(job) else {
                    panic!("an on-beat job has a beat");
                };
                self.beats.submitted(job.get(), due.get());
            }
            let budget = if self.starved.contains(&self.frame) {
                Budget::new(Micros::ZERO)
            } else {
                Budget::for_frame(Micros::new(PERIOD), Micros::new(2_000), Micros::new(2_000))
            };
            let plan = self.scheduler.plan_frame(info, budget, &self.cost);
            self.screen.frame(&plan, self.slices.len());
            for run in &plan.runs {
                if run.position + 1 == self.slices.len() {
                    self.beats.completed(run.job.get(), start);
                }
            }
            for delivery in &plan.delivered {
                self.beats
                    .delivered(delivery.job.get(), start, self.previous, PERIOD);
            }
            for missed in &plan.missed {
                let Due::Beat(due) = missed.due else {
                    panic!("an on-beat miss has a beat");
                };
                self.beats.missed(missed.job.get(), due.get());
            }
            self.previous = Some(start);
            self.frame += 1;
            start
        }

        /// Beat `n` as the page places it, on the frame clock.
        fn beat_us(&self, n: u64) -> u64 {
            (us_of(self.beats.audio_time_s(n).unwrap()) + self.m_us) as u64
        }
    }

    #[test]
    fn the_first_grid_is_anchored_on_the_first_beat_and_every_result_lands_on_a_beat() {
        let mut sim = Sim::new(tempo(100.0), Some(20));
        let frames = run_frames(&tempo(100.0), 20, Micros::new(PERIOD));
        for _ in 0..frames {
            sim.step();
        }
        // The first grid replaces the placeholder from the epoch, its origin
        // the first beat: a second after the first frame plus 40 ms of output.
        let (from, grid) = sim.installed[0];
        assert_eq!(from, TimePoint::new(0));
        assert_eq!(sim.beat_us(0), 1_040_000);
        assert_eq!(
            grid,
            BeatGrid::new(TimePoint::new(1_040_000), Micros::new(600_000)).unwrap()
        );
        // A steady mapping is never corrected.
        assert_eq!(sim.installed.len(), 1);
        let summary = sim.beats.summary();
        assert_eq!(summary.inputs_owed, 20);
        assert_eq!(summary.inputs_submitted, 20);
        assert_eq!(summary.inputs_skipped, 0);
        assert_eq!(summary.early, 0);
        assert_eq!(summary.late, 0);
        assert_eq!(summary.missed, 0);
        assert_eq!(summary.deadlines_moved, 0);
        let landing = summary.landing.unwrap();
        assert_eq!(landing.count, 20);
        assert!(
            landing.min >= 0 && landing.max < PERIOD as i64,
            "{landing:?}"
        );
        // Every beat after the first had its job.
        assert_eq!(summary.beats_elapsed, 20);
        assert_eq!(summary.beats_without_job, 0);
        assert_eq!(summary.beats_shared, 0);
        // 20 slices of 1 ms at about 4.3 ms a frame: five frames, and the
        // rest of the beat is slack.
        let slack = summary.slack.unwrap();
        assert!(slack.min > 500_000 && slack.max < 600_000, "{slack:?}");
        assert_eq!(summary.mapping.updates, 0);
        assert_eq!(summary.mapping.render_ahead_us.unwrap().p50, 28_000);
    }

    #[test]
    fn a_job_on_a_slot_whose_last_delivery_is_on_screen_appears_only_on_its_beat() {
        let mut sim = Sim::new(tempo(100.0), Some(6));
        // No budget from beat 1's frame to beat 3's: job 0 is delivered on
        // beat 1, jobs 1 and 2 miss beats 2 and 3, and job 3, on job 0's
        // slot, finishes in the frames after beat 3 and is delivered on beat
        // 4, which is when it appears.
        let frame_at = |us: u64| us.div_ceil(PERIOD);
        sim.starved = frame_at(sim.beat_us(1))..frame_at(sim.beat_us(3));
        let frames = run_frames(&tempo(100.0), 6, Micros::new(PERIOD));
        let mut shown = Vec::new();
        while sim.frame < frames {
            let start = sim.step();
            shown.push((start, sim.screen.shown()));
        }
        let summary = sim.beats.summary();
        assert_eq!(summary.missed, 2);
        assert_eq!(summary.early, 0);
        let on_screen = |job| {
            shown
                .iter()
                .find(|&&(_, s)| s == Some(job))
                .map(|&(t, _)| t)
        };
        assert!(on_screen(0).unwrap() >= sim.beat_us(1));
        assert_eq!(on_screen(1), None);
        assert_eq!(on_screen(2), None);
        assert!(on_screen(3).unwrap() >= sim.beat_us(4));
        // The copy after the last slice, into job 0's slot's buffer, drew job
        // 3 from beat 3's frames, a beat before its own.
        assert!(sim.screen.early_under_kept > 0);
    }

    #[test]
    fn a_drifting_mapping_is_corrected_from_the_midpoint_and_only_later_beats_move() {
        let mut sim = Sim::new(tempo(100.0), Some(12));
        // Run past beat 3, then move the output path 3 ms later.
        while sim.step() < sim.beat_us(3) + PERIOD {}
        let before = sim.installed.len();
        sim.m_us += 3_000;
        let frames = run_frames(&tempo(100.0), 12, Micros::new(PERIOD));
        while sim.frame < frames {
            sim.step();
        }
        let summary = sim.beats.summary();
        assert_eq!(summary.mapping.updates, 1);
        assert_eq!(summary.mapping.max_correction_us, 3_000);
        let (from, grid) = sim.installed[before];
        // Adopted at beat 4's frame; in effect from the midpoint between beats
        // 4 and 5, the beats from 5 on following the new mapping.
        assert_eq!(
            from.get(),
            (sim.beat_us(4) + sim.beat_us(5)) / 2,
            "{:?}",
            sim.installed
        );
        assert_eq!(
            grid,
            BeatGrid::new(TimePoint::new(sim.beat_us(5)), Micros::new(600_000)).unwrap()
        );
        assert_eq!(summary.early + summary.late + summary.missed, 0);
        assert_eq!(summary.beats_without_job, 0);
        // The jobs placed before the correction landed 3 ms early against the
        // beats as now mapped; the remapped error says so.
        let remapped = summary.remapped.unwrap();
        assert!(remapped.min < 0, "{remapped:?}");
    }

    #[test]
    fn a_tempo_change_moves_only_the_deadlines_fixed_after_its_announcement() {
        // 100 BPM, then 120 from 3.3 s after the first beat: announced there,
        // in effect from beat 6 at 3.6 s.
        let plan = Tempo {
            bpm: 100.0,
            change: Some(TempoChange {
                bpm: 120.0,
                after_s: 3.3,
            }),
        };
        let mut sim = Sim::new(plan, Some(14));
        // 63 slices of 4 ms, one a frame: a job finishes 62 frames and 4 ms,
        // about 521 ms, after its submission, and with the margin is placed
        // two beats ahead. The job submitted at beat 5 (3.0 s), before the
        // announcement, is so placed on beat 7 of the old grid (4.2 s), after
        // the change beat; the one submitted at beat 4 on the change beat
        // itself.
        sim.cost = CostModel::new(Micros::new(4_000), Estimator::default()).unwrap();
        sim.slices = (0..63).map(SliceId::new).collect();
        let frames = run_frames(&plan, 14, Micros::new(PERIOD));
        while sim.frame < frames {
            sim.step();
        }
        let summary = sim.beats.summary();
        let change = summary.change.unwrap();
        assert_eq!(change.change_beat, 6);
        assert_eq!(change.change_us, Some(sim.beat_us(6)));
        let announced = change.announced_us.unwrap();
        assert!(announced >= sim.beat_us(0) + 3_300_000 && announced < sim.beat_us(6));
        // The grid of the new tempo, from the change beat on.
        assert!(sim.installed.contains(&(
            TimePoint::new(sim.beat_us(6)),
            BeatGrid::new(TimePoint::new(sim.beat_us(6)), Micros::new(500_000)).unwrap()
        )));
        // Jobs were pending at the announcement with deadlines past the change
        // beat on the old grid's beats; they kept them.
        assert!(change.fixed_before_after_change >= 1, "{change:?}");
        assert_eq!(
            change.kept_delivered + change.kept_missed,
            change.fixed_before_after_change
        );
        // Every job submitted after the announcement is on a beat of the plan.
        assert!(change.later_jobs >= 1);
        assert_eq!(change.later_off_plan, 0);
        assert_eq!(summary.deadlines_moved, 0);
        assert_eq!(summary.early + summary.late, 0);
    }

    #[test]
    fn inputs_of_beats_that_pass_without_a_submission_are_skipped() {
        let mut beats = Beats::new(tempo(120.0), DEFAULT_MARGIN, Some(5));
        // A run that ends before any beat reports none.
        assert_eq!(beats.summary().beats_elapsed, 0);
        beats.begin(0.0, 48_000.0);
        // m = 0 with an epoch of 0: audio time is the frame clock.
        let reading = |t_us: u64| stamped(t_us as f64 / 1e6, t_us as f64 / 1_000.0);
        beats.record(&reading(10), 0);
        assert!(!beats.frame(10).unwrap().owed);
        // Beat 0 at 1 s.
        let first = beats.frame(1_000_000).unwrap();
        assert!(first.owed);
        assert_eq!(first.grids.len(), 1);
        // Two more beats pass without a submission: 0 and 1 are dropped.
        assert!(beats.frame(2_000_000).unwrap().owed);
        beats.submitted(0, 2_500_000);
        // Job 0 is pending: jobs 1 and 2 have slots of their own, job 3 would
        // take job 0's.
        assert!(beats.slot_free(1) && beats.slot_free(2));
        assert!(!beats.slot_free(3));
        beats.missed(0, 2_500_000);
        assert!(beats.slot_free(3));
        assert!(!beats.frame(2_100_000).unwrap().owed);
        // Past the beats asked for (inputs at beats 0..=4), nothing is owed.
        assert!(beats.frame(3_000_000).unwrap().owed);
        assert!(!beats.frame(3_600_000).unwrap().owed);
        let summary = beats.summary();
        assert_eq!(summary.inputs_owed, 5);
        assert_eq!(summary.inputs_submitted, 1);
        assert_eq!(summary.inputs_skipped, 4);
    }

    #[test]
    fn a_frame_that_reaches_past_the_last_beat_asked_drops_its_inputs() {
        let mut beats = Beats::new(tempo(120.0), DEFAULT_MARGIN, Some(2));
        beats.begin(0.0, 48_000.0);
        let reading = |t_us: u64| stamped(t_us as f64 / 1e6, t_us as f64 / 1_000.0);
        beats.record(&reading(10), 0);
        assert!(beats.frame(1_000_000).unwrap().owed);
        // Beat 1, the last asked for, and beat 2 come in one frame: beat 0's
        // and beat 1's inputs are both dropped, and nothing is owed.
        assert!(!beats.frame(2_000_000).unwrap().owed);
        let summary = beats.summary();
        assert_eq!(summary.inputs_owed, 2);
        assert_eq!(summary.inputs_submitted, 0);
        assert_eq!(summary.inputs_skipped, 2);
    }

    #[test]
    fn a_correction_before_the_announcement_keeps_the_old_tempo() {
        // 120 BPM, then 60 from 1.3 s after the first beat: change beat 3.
        let plan = Tempo {
            bpm: 120.0,
            change: Some(TempoChange {
                bpm: 60.0,
                after_s: 1.3,
            }),
        };
        let mut beats = Beats::new(plan, DEFAULT_MARGIN, None);
        beats.begin(0.0, 48_000.0);
        // m = 0 with an epoch of 0, then 1 ms later.
        let reading =
            |t_us: u64, m_us: u64| stamped(t_us as f64 / 1e6, (t_us + m_us) as f64 / 1_000.0);
        beats.record(&reading(10, 0), 0);
        assert!(beats.frame(1_000_000).unwrap().owed);
        for _ in 0..MAPPING_WINDOW {
            beats.record(&reading(1_900_000, 1_000), 0);
        }
        // Beat 2, the one before the change beat, arrives with the correction,
        // before the change's instant at 2.3 s: the grid from beat 3 on keeps
        // the old beat of 500 ms.
        let decided = beats.frame(2_000_000).unwrap();
        let (_, grid) = *decided.grids.last().unwrap();
        assert_eq!(
            grid,
            BeatGrid::new(TimePoint::new(2_501_000), Micros::new(500_000)).unwrap()
        );
        assert_eq!(beats.summary().change.unwrap().announced_us, None);
    }

    #[test]
    fn the_shortest_beat_is_the_faster_tempos() {
        assert_eq!(tempo(100.0).shortest_beat(), Micros::new(600_000));
        let faster = Tempo {
            bpm: 100.0,
            change: Some(TempoChange {
                bpm: 120.0,
                after_s: 60.0,
            }),
        };
        assert_eq!(faster.shortest_beat(), Micros::new(500_000));
        let slower = Tempo {
            bpm: 120.0,
            change: Some(TempoChange {
                bpm: 100.0,
                after_s: 60.0,
            }),
        };
        assert_eq!(slower.shortest_beat(), Micros::new(500_000));
    }
}
