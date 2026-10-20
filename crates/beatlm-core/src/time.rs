//! Time as values: durations, instants, frame indices and the beat grid.
//!
//! The core never reads a clock. The host measures time and hands it in as
//! these newtypes, so every decision made from them replays identically in a
//! test, on native and on wasm.

use thiserror::Error;

/// A duration in microseconds.
///
/// It exists so a duration cannot be confused with an instant ([`TimePoint`])
/// or a frame count ([`FrameIndex`]). Arithmetic saturates: a budget or an
/// estimate that overflows is clamped rather than wrapped into a small value
/// that would look cheap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Micros(u64);

impl Micros {
    /// The empty duration.
    pub const ZERO: Self = Self(0);

    /// Wraps a number of microseconds.
    #[must_use]
    pub const fn new(micros: u64) -> Self {
        Self(micros)
    }

    /// The number of microseconds.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// Adds two durations, clamping at `u64::MAX` microseconds.
    #[must_use]
    pub const fn saturating_add(self, other: Self) -> Self {
        Self(self.0.saturating_add(other.0))
    }

    /// Subtracts `other`, clamping at zero so a deficit reads as "nothing
    /// left" and never as a huge duration.
    #[must_use]
    pub const fn saturating_sub(self, other: Self) -> Self {
        Self(self.0.saturating_sub(other.0))
    }
}

/// An instant, in microseconds since an epoch the host chooses.
///
/// The host promises the clock behind it is monotonic; the core only compares
/// and offsets instants, so the epoch itself never matters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TimePoint(u64);

impl TimePoint {
    /// Wraps microseconds since the host's epoch.
    #[must_use]
    pub const fn new(micros: u64) -> Self {
        Self(micros)
    }

    /// Microseconds since the host's epoch.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// The instant `duration` later, clamping at the end of representable
    /// time.
    #[must_use]
    pub const fn saturating_add(self, duration: Micros) -> Self {
        Self(self.0.saturating_add(duration.get()))
    }

    /// How long after `earlier` this instant is; zero when `earlier` is not
    /// earlier, so a reordered pair of readings never yields a huge duration.
    #[must_use]
    pub const fn saturating_since(self, earlier: Self) -> Micros {
        Micros::new(self.0.saturating_sub(earlier.0))
    }
}

/// The position of a frame in the host's frame sequence.
///
/// Delivery promises are stated in frames, so frames get their own type
/// instead of travelling as bare integers next to durations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FrameIndex(u64);

impl FrameIndex {
    /// Wraps a frame number.
    #[must_use]
    pub const fn new(index: u64) -> Self {
        Self(index)
    }

    /// The frame number.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }

    /// The following frame, clamping at the last representable one.
    #[must_use]
    pub const fn next(self) -> Self {
        Self(self.0.saturating_add(1))
    }

    /// The frame `frames` later, clamping at the last representable one.
    #[must_use]
    pub const fn offset(self, frames: u32) -> Self {
        Self(self.0.saturating_add(frames as u64))
    }
}

/// What the host knows when a frame begins.
///
/// It is the whole of the core's view of the clock for that frame: which frame
/// it is, when it starts, and how long it is allowed to take.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameInfo {
    /// Position of the frame in the host's sequence.
    pub index: FrameIndex,
    /// When the frame starts.
    pub start: TimePoint,
    /// The time between this frame's start and the next frame's start.
    pub period: Micros,
}

/// A rejected [`BeatGrid`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum TimeError {
    /// A beat of zero length puts every instant on a beat, so no beat could be
    /// told apart from any other.
    #[error("beat length must be greater than zero")]
    ZeroBeat,
}

/// Musical time: beats at `origin + n * beat` for `n >= 0`.
///
/// The `OnBeat` delivery promise needs to know where the next beat boundary is.
/// The host derives the grid from tempo and phase and passes it in; the core
/// never tracks tempo itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BeatGrid {
    origin: TimePoint,
    beat: Micros,
}

impl BeatGrid {
    /// A grid whose first beat is at `origin` and whose beats are `beat`
    /// apart.
    ///
    /// # Errors
    ///
    /// [`TimeError::ZeroBeat`] when `beat` is zero.
    pub const fn new(origin: TimePoint, beat: Micros) -> Result<Self, TimeError> {
        if beat.get() == 0 {
            return Err(TimeError::ZeroBeat);
        }
        Ok(Self { origin, beat })
    }

    /// The first beat at or after `t`.
    ///
    /// An instant before the grid's origin maps to the origin, because beats
    /// exist only from the origin on. A result that cannot be represented
    /// clamps at the end of representable time.
    #[must_use]
    pub fn next_beat_at_or_after(&self, t: TimePoint) -> TimePoint {
        if t <= self.origin {
            return self.origin;
        }
        let since = t.saturating_since(self.origin).get();
        let into_beat = since.checked_rem(self.beat.get()).unwrap_or(0);
        if into_beat == 0 {
            t
        } else {
            t.saturating_add(Micros::new(self.beat.get() - into_beat))
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn grid() -> BeatGrid {
        BeatGrid::new(TimePoint::new(1_000), Micros::new(500)).unwrap()
    }

    #[test]
    fn zero_beat_is_rejected() {
        assert_eq!(
            BeatGrid::new(TimePoint::new(0), Micros::ZERO),
            Err(TimeError::ZeroBeat)
        );
    }

    #[test]
    fn instant_before_origin_maps_to_origin() {
        assert_eq!(
            grid().next_beat_at_or_after(TimePoint::new(10)),
            TimePoint::new(1_000)
        );
    }

    #[test]
    fn instant_on_origin_or_beat_is_its_own_beat() {
        assert_eq!(
            grid().next_beat_at_or_after(TimePoint::new(1_000)),
            TimePoint::new(1_000)
        );
        assert_eq!(
            grid().next_beat_at_or_after(TimePoint::new(2_500)),
            TimePoint::new(2_500)
        );
    }

    #[test]
    fn instant_between_beats_rounds_up() {
        assert_eq!(
            grid().next_beat_at_or_after(TimePoint::new(1_001)),
            TimePoint::new(1_500)
        );
        assert_eq!(
            grid().next_beat_at_or_after(TimePoint::new(1_999)),
            TimePoint::new(2_000)
        );
    }

    #[test]
    fn beat_search_saturates_at_the_end_of_time() {
        let g = BeatGrid::new(TimePoint::new(0), Micros::new(1_000)).unwrap();
        let t = TimePoint::new(u64::MAX - 1);
        assert_eq!(g.next_beat_at_or_after(t), TimePoint::new(u64::MAX));
    }

    #[test]
    fn micros_arithmetic_saturates() {
        assert_eq!(
            Micros::new(u64::MAX).saturating_add(Micros::new(1)),
            Micros::new(u64::MAX)
        );
        assert_eq!(Micros::new(1).saturating_sub(Micros::new(2)), Micros::ZERO);
    }

    #[test]
    fn time_point_arithmetic() {
        let t = TimePoint::new(10);
        assert_eq!(t.saturating_add(Micros::new(5)), TimePoint::new(15));
        assert_eq!(TimePoint::new(15).saturating_since(t), Micros::new(5));
        assert_eq!(t.saturating_since(TimePoint::new(15)), Micros::ZERO);
    }

    #[test]
    fn frame_index_steps_and_saturates() {
        let f = FrameIndex::new(3);
        assert_eq!(f.next(), FrameIndex::new(4));
        assert_eq!(f.offset(2), FrameIndex::new(5));
        assert_eq!(
            FrameIndex::new(u64::MAX).offset(2),
            FrameIndex::new(u64::MAX)
        );
    }
}
