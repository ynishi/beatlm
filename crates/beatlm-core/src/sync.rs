//! The delivery promise and what to do when it breaks.

use crate::time::{BeatGrid, Micros};

/// When the result for an input is delivered.
///
/// The host picks one per model; the scheduler turns it into a deadline for
/// every submitted job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncPolicy {
    /// L1: the result for input frame `t` is delivered on frame `t + frames`,
    /// never earlier and never later. A result finished early waits for its
    /// frame, so latency is a constant the host can align the source to.
    FixedLatency {
        /// The constant delay between input and delivery, in frames.
        frames: u32,
    },
    /// L2: the result is delivered in the frame of its input. Only models that
    /// fit the frame budget can keep this.
    SameFrame,
    /// The result is delivered on the first frame starting at or after a target
    /// beat. The target is the first beat at or after the input's estimated
    /// completion plus `margin`, so work starts early enough to finish before
    /// the beat; [`Scheduler::submit`](crate::schedule::Scheduler::submit)
    /// says how the completion is estimated.
    ///
    /// The beats are part of this promise and of no other, so the grid travels
    /// with it rather than beside every policy.
    OnBeat {
        /// Slack added to the estimated completion before looking for a beat,
        /// absorbing estimate error.
        margin: Micros,
        /// The beats in effect from the start of the session; a tempo change
        /// replaces them from an instant on
        /// ([`Scheduler::replace_beat_grid`](crate::schedule::Scheduler::replace_beat_grid)).
        grid: BeatGrid,
    },
}

impl SyncPolicy {
    /// How many frames the host delays its source so it lines up with the
    /// results, or `None` when the delay is not constant.
    ///
    /// `OnBeat` has no constant delay: its latency varies with where the input
    /// falls relative to the beats.
    #[must_use]
    pub const fn source_delay(&self) -> Option<u32> {
        match self {
            Self::FixedLatency { frames } => Some(*frames),
            Self::SameFrame => Some(0),
            Self::OnBeat { .. } => None,
        }
    }

    /// How many jobs can be in flight at once, or `None` when the policy does
    /// not bound it.
    ///
    /// A job is in flight from its submission until its delivery frame has
    /// been rendered, and its output must stay intact for all of that time:
    /// the next job overwrites a shared output as soon as its last slice runs,
    /// which can be before the earlier job's delivery frame renders. The host
    /// therefore keeps one output slot per in-flight job and reads the slot of
    /// the job being delivered.
    ///
    /// `FixedLatency { frames: k }` has the inputs of `k + 1` consecutive
    /// frames in flight, `SameFrame` one. `OnBeat` does not bound it: how many
    /// jobs wait for a beat depends on the beat spacing and on how fast inputs
    /// arrive, so the host bounds it and decides what happens to an input
    /// beyond the bound.
    #[must_use]
    pub const fn max_in_flight(&self) -> Option<u32> {
        match self {
            Self::FixedLatency { frames } => Some(frames.saturating_add(1)),
            Self::SameFrame => Some(1),
            Self::OnBeat { .. } => None,
        }
    }
}

/// What the host shows for an input whose deadline passed without a complete
/// result.
///
/// A missed deadline is always reported; the policy only tells the host what
/// to show in its place. Nothing is ever delayed silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissPolicy {
    /// The host keeps showing the previous result.
    HoldPrevious,
    /// The host shows no result for this input.
    Drop,
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::time::TimePoint;

    fn on_beat() -> SyncPolicy {
        SyncPolicy::OnBeat {
            margin: Micros::ZERO,
            grid: BeatGrid::new(TimePoint::new(0), Micros::new(500_000)).unwrap(),
        }
    }

    #[test]
    fn source_delay_follows_policy() {
        assert_eq!(
            SyncPolicy::FixedLatency { frames: 3 }.source_delay(),
            Some(3)
        );
        assert_eq!(SyncPolicy::SameFrame.source_delay(), Some(0));
        assert_eq!(on_beat().source_delay(), None);
    }

    #[test]
    fn max_in_flight_follows_policy() {
        assert_eq!(
            SyncPolicy::FixedLatency { frames: 0 }.max_in_flight(),
            Some(1)
        );
        assert_eq!(
            SyncPolicy::FixedLatency { frames: 2 }.max_in_flight(),
            Some(3)
        );
        assert_eq!(
            SyncPolicy::FixedLatency { frames: u32::MAX }.max_in_flight(),
            Some(u32::MAX)
        );
        assert_eq!(SyncPolicy::SameFrame.max_in_flight(), Some(1));
        assert_eq!(on_beat().max_in_flight(), None);
    }
}
