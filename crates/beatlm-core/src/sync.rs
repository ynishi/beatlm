//! The delivery promise and what to do when it breaks.

use crate::time::Micros;

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
    /// the beat.
    OnBeat {
        /// Slack added to the estimated completion before looking for a beat,
        /// absorbing estimate error.
        margin: Micros,
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

    #[test]
    fn source_delay_follows_policy() {
        assert_eq!(
            SyncPolicy::FixedLatency { frames: 3 }.source_delay(),
            Some(3)
        );
        assert_eq!(SyncPolicy::SameFrame.source_delay(), Some(0));
        assert_eq!(
            SyncPolicy::OnBeat {
                margin: Micros::ZERO
            }
            .source_delay(),
            None
        );
    }
}
