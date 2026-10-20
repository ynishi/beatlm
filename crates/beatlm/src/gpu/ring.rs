//! Which readback slot a frame may use, as bookkeeping with no GPU in it.
//!
//! The ring is the rule that keeps timestamp readback off the frame loop's
//! critical path: a frame takes a slot only if the slot is free, and a frame
//! that finds its slot busy goes untimed instead of waiting for it. Keeping the
//! rule apart from the buffers lets it be tested without an adapter.

/// Where a slot is in the life of one frame's readback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    /// Nothing in flight; the next frame may take the slot.
    Free,
    /// A frame holds the slot and is still recording its passes.
    Recording,
    /// The frame's resolve and copy are recorded; the map is not requested
    /// yet, because the submission that carries them has to come first.
    Encoded,
    /// The map is requested and has not been collected.
    Mapping,
}

/// A fixed ring of slots handed out in order.
#[derive(Debug, Clone)]
pub(crate) struct Ring {
    phases: Vec<Phase>,
    cursor: usize,
}

impl Ring {
    /// A ring of `depth` free slots. A depth of zero is raised to one, so a
    /// ring always has a slot to offer.
    pub(crate) fn new(depth: usize) -> Self {
        Self {
            phases: vec![Phase::Free; depth.max(1)],
            cursor: 0,
        }
    }

    /// Takes the slot at the cursor if it is free, and moves the cursor on.
    ///
    /// A busy slot is not taken and the cursor stays: the slot at the cursor is
    /// the oldest one in flight, so the next frame retries the slot most likely
    /// to have come free.
    pub(crate) fn claim(&mut self) -> Option<usize> {
        let slot = self.cursor;
        match self.phases.get_mut(slot) {
            Some(phase @ Phase::Free) => {
                *phase = Phase::Recording;
                self.cursor = (slot + 1) % self.phases.len();
                Some(slot)
            }
            _ => None,
        }
    }

    /// Moves `slot` to `to` when it is currently in `from`, and reports whether
    /// it moved.
    pub(crate) fn advance(&mut self, slot: usize, from: Phase, to: Phase) -> bool {
        match self.phases.get_mut(slot) {
            Some(phase) if *phase == from => {
                *phase = to;
                true
            }
            _ => false,
        }
    }

    /// Returns `slot` to the free state, whatever state it was in.
    pub(crate) fn release(&mut self, slot: usize) {
        if let Some(phase) = self.phases.get_mut(slot) {
            *phase = Phase::Free;
        }
    }

    /// The slots currently in `phase`, in slot order.
    pub(crate) fn slots_in(&self, phase: Phase) -> impl Iterator<Item = usize> + '_ {
        self.phases
            .iter()
            .enumerate()
            .filter(move |(_, p)| **p == phase)
            .map(|(i, _)| i)
    }

    /// How many slots are not free.
    pub(crate) fn in_flight(&self) -> usize {
        self.phases.iter().filter(|p| **p != Phase::Free).count()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn slots_are_handed_out_in_order_and_wrap() {
        let mut ring = Ring::new(3);
        assert_eq!(ring.claim(), Some(0));
        ring.release(0);
        assert_eq!(ring.claim(), Some(1));
        ring.release(1);
        assert_eq!(ring.claim(), Some(2));
        ring.release(2);
        assert_eq!(ring.claim(), Some(0));
    }

    #[test]
    fn a_busy_slot_is_refused_and_the_cursor_stays() {
        let mut ring = Ring::new(2);
        assert_eq!(ring.claim(), Some(0));
        assert_eq!(ring.claim(), Some(1));
        assert_eq!(ring.claim(), None);
        assert_eq!(ring.claim(), None);
        ring.release(0);
        assert_eq!(ring.claim(), Some(0));
    }

    #[test]
    fn a_slot_moves_only_from_the_phase_it_is_in() {
        let mut ring = Ring::new(2);
        let slot = ring.claim().unwrap();
        assert!(!ring.advance(slot, Phase::Encoded, Phase::Mapping));
        assert!(ring.advance(slot, Phase::Recording, Phase::Encoded));
        assert!(ring.advance(slot, Phase::Encoded, Phase::Mapping));
        assert_eq!(
            ring.slots_in(Phase::Mapping).collect::<Vec<_>>(),
            vec![slot]
        );
        assert_eq!(ring.in_flight(), 1);
        ring.release(slot);
        assert_eq!(ring.in_flight(), 0);
    }

    #[test]
    fn unknown_slots_are_ignored() {
        let mut ring = Ring::new(1);
        assert!(!ring.advance(9, Phase::Free, Phase::Recording));
        ring.release(9);
        assert_eq!(ring.in_flight(), 0);
    }

    #[test]
    fn a_zero_depth_ring_still_has_one_slot() {
        let mut ring = Ring::new(0);
        assert_eq!(ring.claim(), Some(0));
        assert_eq!(ring.claim(), None);
    }
}
