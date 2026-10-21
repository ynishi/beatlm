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
/// never derives tempo itself. A tempo change is a new grid the host hands in
/// with the instant it takes effect from ([`BeatMap`]).
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

/// Beats over a session whose tempo changes: a sequence of [`BeatGrid`]s,
/// each in effect from an instant until the next one's.
///
/// A grid in effect from `from` contributes its beats at or after `from` and
/// before the next grid's `from`; outside that span its beats do not exist. The
/// first grid is in effect from the epoch. Instants are compared, never
/// subtracted across grids, so a change neither has to fall on a beat of the
/// old grid nor be the new grid's origin.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BeatMap {
    /// `(from, grid)` in strictly increasing `from`; never empty.
    grids: Vec<(TimePoint, BeatGrid)>,
}

impl BeatMap {
    /// Beats from `grid` alone, in effect from the epoch.
    #[must_use]
    pub fn new(grid: BeatGrid) -> Self {
        Self {
            grids: vec![(TimePoint::new(0), grid)],
        }
    }

    /// Puts `grid` in effect from `from` on.
    ///
    /// Every grid that took effect at or after `from` is dropped, so the latest
    /// call decides what follows `from`: a host that announced a change and
    /// then changed its mind replaces the announcement.
    pub fn replace_from(&mut self, from: TimePoint, grid: BeatGrid) {
        self.grids.retain(|&(start, _)| start < from);
        self.grids.push((from, grid));
    }

    /// The first beat at or after `t`, on whichever grid is in effect at that
    /// beat.
    ///
    /// An instant before the earliest grid kept maps to that grid's first beat
    /// at or after its `from` ([`forget_before`](Self::forget_before) is what
    /// moves it from the epoch). A result that cannot be represented clamps at
    /// the end of representable time.
    #[must_use]
    pub fn next_beat_at_or_after(&self, t: TimePoint) -> TimePoint {
        let first = self
            .grids
            .partition_point(|&(from, _)| from <= t)
            .saturating_sub(1);
        let mut t = t;
        let mut grids = self.grids.iter().skip(first).peekable();
        while let Some(&(from, grid)) = grids.next() {
            let beat = grid.next_beat_at_or_after(t.max(from));
            match grids.peek() {
                // The beat falls where the next grid is in effect, so the next
                // grid decides.
                Some(&&(next, _)) if beat >= next => t = next,
                _ => return beat,
            }
        }
        // The map is never empty, so the loop returned.
        t
    }

    /// Drops the grids whose whole span ends at or before `t`, keeping the one
    /// in effect at `t`.
    ///
    /// It bounds the map in a session with many tempo changes. A lookup at or
    /// after `t` gives the same beat as before; one before `t` gets a beat at
    /// or after the earliest kept grid's `from`, so never an earlier beat.
    pub fn forget_before(&mut self, t: TimePoint) {
        let in_effect = self
            .grids
            .partition_point(|&(from, _)| from <= t)
            .saturating_sub(1);
        self.grids.drain(..in_effect);
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

    /// Beats every 1000 us from 0, then every 300 us from a change at 2500
    /// whose own origin is 2600.
    fn changed() -> BeatMap {
        let mut map = BeatMap::new(BeatGrid::new(TimePoint::new(0), Micros::new(1_000)).unwrap());
        map.replace_from(
            TimePoint::new(2_500),
            BeatGrid::new(TimePoint::new(2_600), Micros::new(300)).unwrap(),
        );
        map
    }

    #[test]
    fn beat_map_uses_the_old_grid_before_the_change_and_the_new_one_after() {
        let map = changed();
        assert_eq!(
            map.next_beat_at_or_after(TimePoint::new(1_200)),
            TimePoint::new(2_000)
        );
        assert_eq!(
            map.next_beat_at_or_after(TimePoint::new(2_700)),
            TimePoint::new(2_900)
        );
    }

    #[test]
    fn beat_of_the_old_grid_past_the_change_does_not_exist() {
        // The old grid's next beat after 2100 is 3000, past the change at
        // 2500, so the new grid's first beat at or after 2500 is taken.
        assert_eq!(
            changed().next_beat_at_or_after(TimePoint::new(2_100)),
            TimePoint::new(2_600)
        );
    }

    #[test]
    fn replacing_drops_the_grids_announced_at_or_after_it() {
        let mut map = changed();
        map.replace_from(
            TimePoint::new(2_000),
            BeatGrid::new(TimePoint::new(0), Micros::new(250)).unwrap(),
        );
        // The change at 2500 is gone; from 2000 the 250 us grid holds.
        assert_eq!(
            map.next_beat_at_or_after(TimePoint::new(2_700)),
            TimePoint::new(2_750)
        );
        assert_eq!(
            map.next_beat_at_or_after(TimePoint::new(1_500)),
            TimePoint::new(2_000)
        );
    }

    #[test]
    fn forgetting_keeps_every_beat_from_then_on() {
        let mut map = changed();
        let before: Vec<_> = (2_500..4_000)
            .step_by(7)
            .map(|t| map.next_beat_at_or_after(TimePoint::new(t)))
            .collect();
        map.forget_before(TimePoint::new(2_500));
        assert_eq!(map.grids.len(), 1);
        let after: Vec<_> = (2_500..4_000)
            .step_by(7)
            .map(|t| map.next_beat_at_or_after(TimePoint::new(t)))
            .collect();
        assert_eq!(before, after);
        // Before what was kept, a lookup gets the kept grid's first beat:
        // later than the forgotten grid's, never earlier.
        assert_eq!(
            map.next_beat_at_or_after(TimePoint::new(1_200)),
            TimePoint::new(2_600)
        );
        // Forgetting before the change keeps both grids.
        let mut map = changed();
        map.forget_before(TimePoint::new(2_499));
        assert_eq!(map.grids.len(), 2);
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
