//! The time a frame may spend on inference.

use crate::time::Micros;

/// The time a frame may spend on inference.
///
/// The renderer and the model share one frame; this is what remains for the
/// model once rendering and a safety margin are paid for. The scheduler never
/// plans more estimated work than this in a frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Budget(Micros);

impl Budget {
    /// Wraps a duration as a budget.
    #[must_use]
    pub const fn new(time: Micros) -> Self {
        Self(time)
    }

    /// The budgeted duration.
    #[must_use]
    pub const fn get(self) -> Micros {
        self.0
    }

    /// The budget left in a frame of length `period` after `render` and
    /// `margin`, and zero when they exceed the period.
    ///
    /// `render` is the measured cost of rendering the frame; `margin` protects
    /// the frame deadline against estimate error and jitter.
    #[must_use]
    pub const fn for_frame(period: Micros, render: Micros, margin: Micros) -> Self {
        Self(period.saturating_sub(render).saturating_sub(margin))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn for_frame_subtracts_render_and_margin() {
        let b = Budget::for_frame(Micros::new(16_000), Micros::new(10_000), Micros::new(1_000));
        assert_eq!(b.get(), Micros::new(5_000));
    }

    #[test]
    fn for_frame_saturates_at_zero() {
        let b = Budget::for_frame(Micros::new(16_000), Micros::new(15_000), Micros::new(2_000));
        assert_eq!(b.get(), Micros::ZERO);
        let b = Budget::for_frame(Micros::new(1), Micros::new(u64::MAX), Micros::new(u64::MAX));
        assert_eq!(b.get(), Micros::ZERO);
    }
}
