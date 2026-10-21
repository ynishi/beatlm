//! Which frame an animation-frame callback is, from the time it began.
//!
//! The browser decides when a callback runs, at the display's refresh rate;
//! the pacer decides which of them are frames. The rule is the bench's pacer
//! (`beatlm-bench`'s `pace.rs`) applied to a time the host did not choose:
//!
//! - **A frame's index is the 60 Hz period it began in**, counted from the
//!   run's epoch.
//! - **A frame that begins whole periods late gives those periods up**, as the
//!   bench drops them and a display would, and the core sees the skipped
//!   indices.
//! - **A callback in a period that already had its frame is not a frame.** The
//!   bench never meets this, because it sleeps until the next period; a
//!   browser on a display faster than 60 Hz calls back more than once a
//!   period, and running a frame on each would double the load.
//!
//! The epoch is a quarter period before the first callback. Callbacks arrive
//! with some jitter, and with the epoch on the first one every later callback
//! would sit on a period boundary, where an early one reads as the period
//! before (and is not a frame) and the next one as a dropped period. A quarter
//! period lets a callback be that much early, or three quarters late, and still
//! read as its own period, at 60 Hz and at 120 Hz, where callbacks alternate
//! between a quarter and three quarters through each period.
//!
//! Lateness is measured from that phase rather than from the boundary, so an
//! on-time frame reads zero as it does natively.
//!
//! Nothing here reads a clock: the time is an argument, which is what lets the
//! rule be tested natively.

use beatlm_core::time::Micros;

/// The frame period: 60 Hz, as the bench's.
pub const FRAME_PERIOD: Micros = Micros::new(16_667);

/// Where in its period a frame on time begins: a quarter period after the
/// period's start.
const PHASE_US: u64 = FRAME_PERIOD.get() / 4;

/// One frame, as the pacer starts it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameStart {
    /// The frame's index; periods given up leave gaps in the sequence.
    pub index: u64,
    /// Microseconds from the run's epoch to the frame's start.
    pub start_us: u64,
}

/// What a callback is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pace {
    /// A frame to run.
    Frame(FrameStart),
    /// Not a frame: this period has had its frame.
    Wait,
    /// The run's frames are done.
    Done,
}

/// Turns callback times into frames for a run of a fixed number of frames.
#[derive(Debug, Clone)]
pub struct Pacer {
    frames: u64,
    epoch_us: Option<u64>,
    next: u64,
    dropped: u64,
    lateness_us: Vec<u64>,
}

impl Pacer {
    /// A pacer for `frames` frames, whose epoch is set by the first callback.
    #[must_use]
    pub fn new(frames: u64) -> Self {
        Self {
            frames,
            epoch_us: None,
            next: 0,
            dropped: 0,
            lateness_us: Vec::new(),
        }
    }

    /// What a callback that began at `now_us` is, `now_us` being microseconds
    /// on the clock every callback is read from.
    pub fn frame_at(&mut self, now_us: u64) -> Pace {
        if self.next >= self.frames {
            return Pace::Done;
        }
        let period = FRAME_PERIOD.get();
        let epoch = *self.epoch_us.get_or_insert(now_us.saturating_sub(PHASE_US));
        let elapsed = now_us.saturating_sub(epoch);
        let current = elapsed / period;
        if current < self.next {
            return Pace::Wait;
        }
        if current > self.next {
            self.dropped += current - self.next;
            self.next = current;
            if self.next >= self.frames {
                return Pace::Done;
            }
        }
        let on_time = self.next.saturating_mul(period).saturating_add(PHASE_US);
        self.lateness_us.push(elapsed.saturating_sub(on_time));
        let start = FrameStart {
            index: self.next,
            start_us: elapsed,
        };
        self.next += 1;
        Pace::Frame(start)
    }

    /// Frame periods given up because a frame began whole periods late.
    #[must_use]
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// How late each started frame began, in microseconds, after the phase an
    /// on-time frame begins at.
    #[must_use]
    pub fn lateness_us(&self) -> &[u64] {
        &self.lateness_us
    }
}

/// `ms` milliseconds, as a `performance.now()` reading is, in whole
/// microseconds, rounded to nearest; zero for a negative or non-finite value.
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

    const P: u64 = FRAME_PERIOD.get();

    fn frame(pace: Pace) -> FrameStart {
        match pace {
            Pace::Frame(start) => start,
            other => panic!("expected a frame, got {other:?}"),
        }
    }

    #[test]
    fn callbacks_at_60_hz_are_consecutive_frames_on_time() {
        let mut pacer = Pacer::new(3);
        let t0 = 5_000_000;
        for i in 0..3 {
            assert_eq!(frame(pacer.frame_at(t0 + i * P)).index, i);
        }
        assert_eq!(pacer.frame_at(t0 + 3 * P), Pace::Done);
        assert_eq!(pacer.dropped(), 0);
        assert_eq!(pacer.lateness_us(), [0, 0, 0]);
    }

    #[test]
    fn jitter_within_a_quarter_period_neither_waits_nor_drops() {
        let mut pacer = Pacer::new(4);
        let t0 = 1_000_000;
        let early = P / 4 - 1;
        let starts = [t0, t0 + P - early, t0 + 2 * P + early, t0 + 3 * P - early];
        for (i, &t) in starts.iter().enumerate() {
            assert_eq!(frame(pacer.frame_at(t)).index, i as u64);
        }
        assert_eq!(pacer.dropped(), 0);
    }

    #[test]
    fn at_120_hz_every_other_callback_is_a_frame() {
        let mut pacer = Pacer::new(3);
        let t0 = 2_000_000;
        let mut frames = Vec::new();
        for i in 0..6 {
            if let Pace::Frame(start) = pacer.frame_at(t0 + i * P / 2) {
                frames.push(start.index);
            }
        }
        assert_eq!(frames, [0, 1, 2]);
        assert_eq!(pacer.dropped(), 0);
    }

    #[test]
    fn a_late_frame_gives_up_the_periods_it_overran() {
        let mut pacer = Pacer::new(10);
        let t0 = 3_000_000;
        assert_eq!(frame(pacer.frame_at(t0)).index, 0);
        // The next callback comes three and a half periods later.
        let late = frame(pacer.frame_at(t0 + 3 * P + P / 2));
        assert_eq!(late.index, 3);
        assert_eq!(pacer.dropped(), 2);
        assert_eq!(pacer.lateness_us()[1], P / 2);
        assert_eq!(frame(pacer.frame_at(t0 + 4 * P)).index, 4);
    }

    #[test]
    fn dropping_past_the_last_frame_ends_the_run() {
        let mut pacer = Pacer::new(2);
        assert!(matches!(pacer.frame_at(1_000_000), Pace::Frame(_)));
        assert_eq!(pacer.frame_at(1_000_000 + 5 * P), Pace::Done);
    }

    #[test]
    fn a_frame_starts_where_the_period_clock_says() {
        let mut pacer = Pacer::new(2);
        let first = frame(pacer.frame_at(1_000_000));
        assert_eq!(first.start_us, P / 4);
        let second = frame(pacer.frame_at(1_000_000 + P));
        assert_eq!(second.start_us, P + P / 4);
    }

    #[test]
    fn milliseconds_convert_to_rounded_microseconds() {
        assert_eq!(ms_to_us(16.6674), 16_667);
        assert_eq!(ms_to_us(0.0), 0);
        assert_eq!(ms_to_us(-3.0), 0);
        assert_eq!(ms_to_us(f64::NAN), 0);
    }
}
