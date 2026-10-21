//! Frame pacing, the part of the loop that is the same whichever thread decides
//! and submits the frame's GPU work.
//!
//! Pacing lives apart from the work so that the three concurrency candidates
//! are compared on one clock: each starts its frames at the same instants, drops
//! overrun periods the same way, and differs only in what happens to the
//! [`Tick`] it produces.

use std::time::{Duration, Instant};

use beatlm_core::time::Micros;

/// The frame period: 60 Hz.
pub const FRAME_PERIOD: Micros = Micros::new(16_667);

/// How long before a frame's start the loop stops sleeping and spins. A sleep
/// overshoots by up to a few milliseconds on macOS, so the window has to cover
/// the overshoot or the frame starts late; the report's
/// `frame_start_lateness_us` shows whether it does.
const SPIN_WINDOW: Duration = Duration::from_millis(4);

/// One frame, as the pacer starts it.
#[derive(Debug, Clone, Copy)]
pub struct Tick {
    /// The frame's index; frames given up as overrun leave gaps in the sequence.
    pub index: u64,
    /// When the frame actually began, on the monotonic clock every candidate
    /// measures against, including across threads.
    pub began: Instant,
    /// Microseconds from the run's epoch to `began`.
    pub start_us: u64,
}

/// Starts frames at [`FRAME_PERIOD`] intervals and records how late each began.
pub struct Pacer {
    epoch: Instant,
    next: u64,
    frames: u64,
    dropped: u64,
    lateness_us: Vec<u64>,
}

impl Pacer {
    /// A pacer for `frames` frames whose first period begins now.
    pub fn start(frames: u64) -> Self {
        Self {
            epoch: Instant::now(),
            next: 0,
            frames,
            dropped: 0,
            lateness_us: Vec::new(),
        }
    }

    /// Waits for the next frame's start and returns it, or `None` once the
    /// run's frames are done.
    pub fn next_frame(&mut self) -> Option<Tick> {
        if self.next >= self.frames {
            return None;
        }
        let period = FRAME_PERIOD.get();
        wait_until(self.epoch + Duration::from_micros(self.next.saturating_mul(period)));
        let began = Instant::now();
        let elapsed = micros_between(self.epoch, began);
        // A display-driven host never runs two frames in one period. A frame
        // that begins a whole period or more late gives up the periods it
        // overran, as a display would drop them, and the core sees those frame
        // indices as skipped; catching up back to back instead would pile work
        // onto the GPU and charge the stall to the frames after it.
        let current = elapsed / period;
        if current > self.next {
            self.dropped += current - self.next;
            self.next = current;
            if self.next >= self.frames {
                return None;
            }
        }
        self.lateness_us
            .push(elapsed.saturating_sub(self.next.saturating_mul(period)));
        // The frame's start is when it actually began, so a late frame is not
        // planned as if it had started on time.
        let tick = Tick {
            index: self.next,
            began,
            start_us: elapsed,
        };
        self.next += 1;
        Some(tick)
    }

    /// Consumes the pacer, returning the frame periods given up and each
    /// started frame's lateness in microseconds.
    pub fn finish(self) -> (u64, Vec<u64>) {
        (self.dropped, self.lateness_us)
    }
}

/// Sleeps until shortly before `target`, then spins, so the frame starts on
/// time without a core pinned for the whole period.
fn wait_until(target: Instant) {
    let now = Instant::now();
    if let Some(sleep) = target.checked_duration_since(now + SPIN_WINDOW) {
        std::thread::sleep(sleep);
    }
    while Instant::now() < target {
        std::hint::spin_loop();
    }
}

/// Microseconds from `start` to `end`, zero if `end` is not later.
pub fn micros_between(start: Instant, end: Instant) -> u64 {
    u64::try_from(end.saturating_duration_since(start).as_micros()).unwrap_or(u64::MAX)
}
