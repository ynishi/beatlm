//! Which readback slot a frame may use, as bookkeeping with no GPU in it.
//!
//! The ring is the rule that keeps timestamp readback off the frame loop's
//! critical path: a frame takes a slot only if the slot is free, and a frame
//! that finds its slot busy goes untimed instead of waiting for it. Keeping the
//! rule apart from the buffers lets it be tested without an adapter.
//!
//! It also holds the second rule: **a slot is read only after the host has
//! said the encoder holding its copy was submitted.** A resolve hands out a
//! [`Ticket`]; only [`Ring::submit`] with that ticket moves the slot to
//! [`Phase::Mapping`], and a ticket dropped without it gives the slot back
//! unread. Without the rule a slot whose encoder was dropped would still be
//! mapped, the map would succeed on whatever an earlier frame left in the
//! buffer, and those durations would be reported as this frame's.
//!
//! And the third: **a slot the host has invalidated is never decoded.**
//! [`Ring::invalidate`] moves every slot whose map was requested to
//! [`Phase::Invalid`], and [`collect`] turns an invalid slot whose map has
//! completed into [`Collect::Drop`], whatever the map read. The slot is not
//! freed before its map completes: a buffer with a map pending cannot take
//! the next lap's copy.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Where a slot is in the life of one frame's readback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    /// Nothing in flight; the next frame may take the slot.
    Free,
    /// A frame holds the slot and is still recording its passes.
    Recording,
    /// The frame's resolve and copy are recorded and its [`Ticket`] is with
    /// the host. The slot waits for the ticket to come back through
    /// [`Ring::submit`], or to be dropped.
    Encoded,
    /// The host has said the encoder was submitted; the map is requested and
    /// has not been collected.
    Mapping,
    /// The map is requested, and the host has since said the frame's
    /// submission is not to be trusted; the slot is dropped unread once the
    /// map completes.
    Invalid,
}

/// Where a slot's map request stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Map {
    /// Not completed.
    Pending,
    /// Completed, and the buffer is mapped.
    Ok,
    /// Completed with an error, and the buffer is not mapped.
    Failed,
}

/// What collecting a slot does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Collect {
    /// Not collectable yet: leave it in flight.
    Wait,
    /// Read the timestamps, unmap, free the slot.
    Read,
    /// Free the slot without reading it, after unmapping it when `unmap`;
    /// the frame's timing is lost.
    Drop {
        /// Whether the buffer is mapped and must be unmapped before reuse.
        unmap: bool,
    },
}

/// What collecting a slot in `phase` whose map stands at `map` does.
///
/// Only a slot in [`Phase::Mapping`] whose map succeeded is read. An
/// [`Phase::Invalid`] slot is dropped once its map completes, so that its
/// buffer is no longer pending when the next lap copies into it. A slot in
/// any other phase has no map to collect.
pub(crate) fn collect(phase: Phase, map: Map) -> Collect {
    match (phase, map) {
        (Phase::Mapping | Phase::Invalid, Map::Pending)
        | (Phase::Free | Phase::Recording | Phase::Encoded, _) => Collect::Wait,
        (Phase::Mapping, Map::Ok) => Collect::Read,
        (Phase::Invalid, Map::Ok) => Collect::Drop { unmap: true },
        (Phase::Mapping | Phase::Invalid, Map::Failed) => Collect::Drop { unmap: false },
    }
}

/// The proof that a slot's resolve was recorded, which the host hands back
/// once the encoder holding it is submitted.
///
/// Dropping it instead marks the slot abandoned, and the ring frees it the
/// next time it is asked for a slot or for what is in flight. A drop cannot
/// tell whether a submission happened, so it is always read as "never
/// submitted": the frame's timing is lost, never misread. That is sound even
/// if the host did submit: the slot is never mapped for that frame, and a
/// later frame that reuses it records its own copy into the buffer on the
/// same queue, after the abandoned one, and requests its map only after its
/// own submission.
#[derive(Debug)]
pub(crate) struct Ticket {
    slot: usize,
    /// The slot's abandon flag, shared with the ring. `None` once the ticket
    /// has been handed back, so its drop then marks nothing.
    abandoned: Option<Arc<AtomicBool>>,
}

impl Drop for Ticket {
    fn drop(&mut self) {
        if let Some(flag) = self.abandoned.take() {
            flag.store(true, Ordering::Release);
        }
    }
}

/// A fixed ring of slots handed out in order.
#[derive(Debug)]
pub(crate) struct Ring {
    phases: Vec<Phase>,
    /// One flag per slot, set by a dropped [`Ticket`]. It is read only while
    /// the slot is [`Phase::Encoded`], and cleared each time a ticket for the
    /// slot is handed out, so one allocation per slot serves every frame.
    abandoned: Vec<Arc<AtomicBool>>,
    cursor: usize,
}

impl Ring {
    /// A ring of `depth` free slots. A depth of zero is raised to one, so a
    /// ring always has a slot to offer.
    pub(crate) fn new(depth: usize) -> Self {
        let depth = depth.max(1);
        Self {
            phases: vec![Phase::Free; depth],
            abandoned: (0..depth)
                .map(|_| Arc::new(AtomicBool::new(false)))
                .collect(),
            cursor: 0,
        }
    }

    /// Takes the slot at the cursor if it is free, and moves the cursor on.
    ///
    /// A busy slot is not taken and the cursor stays: the slot at the cursor is
    /// the oldest one in flight, so the next frame retries the slot most likely
    /// to have come free. Slots whose ticket was dropped are freed first.
    pub(crate) fn claim(&mut self) -> Option<usize> {
        self.reclaim();
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

    /// Records that `slot`'s resolve is encoded, and returns the ticket the
    /// host hands back after submitting. `None` when the slot is not
    /// recording.
    pub(crate) fn encode(&mut self, slot: usize) -> Option<Ticket> {
        let flag = self.abandoned.get(slot)?;
        match self.phases.get_mut(slot) {
            Some(phase @ Phase::Recording) => {
                // No other ticket for the slot is alive: the slot was free
                // when this frame claimed it.
                flag.store(false, Ordering::Release);
                *phase = Phase::Encoded;
                Some(Ticket {
                    slot,
                    abandoned: Some(Arc::clone(flag)),
                })
            }
            _ => None,
        }
    }

    /// Takes a ticket back after its encoder was submitted, moves its slot to
    /// [`Phase::Mapping`] and returns the slot.
    ///
    /// `None` for a ticket this ring did not hand out; it is dropped, which
    /// frees its slot in the ring that did.
    pub(crate) fn submit(&mut self, mut ticket: Ticket) -> Option<usize> {
        let slot = ticket.slot;
        let ours = matches!(
            (&ticket.abandoned, self.abandoned.get(slot)),
            (Some(theirs), Some(mine)) if Arc::ptr_eq(theirs, mine)
        );
        if !ours {
            return None;
        }
        let phase = self.phases.get_mut(slot)?;
        if *phase != Phase::Encoded {
            return None;
        }
        *phase = Phase::Mapping;
        // Handed back: its drop must not mark the slot abandoned.
        ticket.abandoned = None;
        Some(slot)
    }

    /// Moves every slot in [`Phase::Mapping`] to [`Phase::Invalid`] and
    /// returns how many moved. Slots not yet handed back as submitted are
    /// left as they are.
    pub(crate) fn invalidate(&mut self) -> usize {
        let mut moved = 0;
        for phase in &mut self.phases {
            if *phase == Phase::Mapping {
                *phase = Phase::Invalid;
                moved += 1;
            }
        }
        moved
    }

    /// The phase of `slot`, `None` for a slot the ring does not have.
    pub(crate) fn phase(&self, slot: usize) -> Option<Phase> {
        self.phases.get(slot).copied()
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

    /// How many slots are neither free nor waiting on a dropped ticket.
    pub(crate) fn in_flight(&self) -> usize {
        self.phases
            .iter()
            .enumerate()
            .filter(|&(i, p)| *p != Phase::Free && !self.is_abandoned(i))
            .count()
    }

    /// Frees every slot whose ticket was dropped.
    fn reclaim(&mut self) {
        for i in 0..self.phases.len() {
            if self.is_abandoned(i) {
                self.release(i);
            }
        }
    }

    /// Whether `slot` is encoded and its ticket was dropped.
    fn is_abandoned(&self, slot: usize) -> bool {
        self.phases.get(slot) == Some(&Phase::Encoded)
            && self
                .abandoned
                .get(slot)
                .is_some_and(|flag| flag.load(Ordering::Acquire))
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
    fn a_submitted_ticket_makes_its_slot_readable() {
        let mut ring = Ring::new(2);
        let slot = ring.claim().unwrap();
        assert_eq!(ring.slots_in(Phase::Mapping).count(), 0);
        let ticket = ring.encode(slot).unwrap();
        assert_eq!(ticket.slot, slot);
        // Resolved but not submitted: in flight, and not readable.
        assert_eq!(ring.slots_in(Phase::Mapping).count(), 0);
        assert_eq!(ring.in_flight(), 1);
        assert_eq!(ring.submit(ticket), Some(slot));
        assert_eq!(
            ring.slots_in(Phase::Mapping).collect::<Vec<_>>(),
            vec![slot]
        );
        assert_eq!(ring.in_flight(), 1);
        ring.release(slot);
        assert_eq!(ring.in_flight(), 0);
    }

    #[test]
    fn a_dropped_ticket_frees_its_slot_unread() {
        let mut ring = Ring::new(2);
        let slot = ring.claim().unwrap();
        drop(ring.encode(slot).unwrap());
        assert_eq!(ring.in_flight(), 0);
        assert_eq!(ring.slots_in(Phase::Mapping).count(), 0);
        // The slot comes back on the next lap, and is never read on the way.
        assert_eq!(ring.claim(), Some(1));
        assert_eq!(ring.slots_in(Phase::Mapping).count(), 0);
        ring.release(1);
        assert_eq!(ring.claim(), Some(slot));
        assert_eq!(ring.slots_in(Phase::Mapping).count(), 0);
    }

    #[test]
    fn a_ticket_from_an_earlier_lap_does_not_leak_into_the_next() {
        let mut ring = Ring::new(1);
        let slot = ring.claim().unwrap();
        drop(ring.encode(slot).unwrap());
        // The same slot again: the old drop must not abandon the new frame.
        let slot = ring.claim().unwrap();
        let ticket = ring.encode(slot).unwrap();
        assert_eq!(ring.in_flight(), 1);
        assert_eq!(ring.submit(ticket), Some(0));
        assert_eq!(ring.slots_in(Phase::Mapping).collect::<Vec<_>>(), vec![0]);
    }

    #[test]
    fn tickets_may_come_back_out_of_order() {
        let mut ring = Ring::new(3);
        let first = ring.claim().unwrap();
        let first = ring.encode(first).unwrap();
        let second = ring.claim().unwrap();
        let second = ring.encode(second).unwrap();
        let third = ring.claim().unwrap();
        let third = ring.encode(third).unwrap();
        assert_eq!(ring.submit(third), Some(2));
        drop(first);
        assert_eq!(ring.submit(second), Some(1));
        assert_eq!(
            ring.slots_in(Phase::Mapping).collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(ring.in_flight(), 2);
        // The dropped one is free again and the cursor, back at it, takes it.
        assert_eq!(ring.claim(), Some(0));
    }

    #[test]
    fn a_full_ring_refuses_until_a_ticket_is_dropped() {
        let mut ring = Ring::new(2);
        let a = ring.claim().unwrap();
        let a = ring.encode(a).unwrap();
        let b = ring.claim().unwrap();
        let b = ring.encode(b).unwrap();
        // Both slots wait on tickets the host still holds.
        assert_eq!(ring.claim(), None);
        assert_eq!(ring.in_flight(), 2);
        // A drop not at the cursor frees its slot, but the cursor stays on
        // the oldest.
        drop(b);
        assert_eq!(ring.in_flight(), 1);
        assert_eq!(ring.claim(), None);
        drop(a);
        assert_eq!(ring.claim(), Some(0));
        assert_eq!(ring.claim(), Some(1));
    }

    #[test]
    fn a_ticket_from_another_ring_is_refused_and_frees_its_own_slot() {
        let mut mine = Ring::new(2);
        let mut theirs = Ring::new(2);
        let slot = mine.claim().unwrap();
        let _held = mine.encode(slot).unwrap();
        let slot = theirs.claim().unwrap();
        let foreign = theirs.encode(slot).unwrap();
        assert_eq!(mine.submit(foreign), None);
        assert_eq!(mine.slots_in(Phase::Mapping).count(), 0);
        assert_eq!(mine.in_flight(), 1);
        assert_eq!(theirs.slots_in(Phase::Mapping).count(), 0);
        assert_eq!(theirs.in_flight(), 0);
    }

    #[test]
    fn only_a_recording_slot_is_encoded() {
        let mut ring = Ring::new(2);
        assert!(ring.encode(0).is_none());
        let slot = ring.claim().unwrap();
        let ticket = ring.encode(slot).unwrap();
        assert!(ring.encode(slot).is_none());
        assert_eq!(ring.submit(ticket), Some(slot));
        assert!(ring.encode(slot).is_none());
    }

    #[test]
    fn unknown_slots_are_ignored() {
        let mut ring = Ring::new(1);
        assert!(ring.encode(9).is_none());
        ring.release(9);
        assert_eq!(ring.in_flight(), 0);
    }

    /// Claims, encodes and submits one frame, returning its slot.
    fn submit_one(ring: &mut Ring) -> usize {
        let slot = ring.claim().unwrap();
        let ticket = ring.encode(slot).unwrap();
        ring.submit(ticket).unwrap()
    }

    #[test]
    fn only_a_valid_completed_map_is_read() {
        use Collect::{Drop, Read, Wait};
        assert_eq!(collect(Phase::Mapping, Map::Ok), Read);
        assert_eq!(collect(Phase::Mapping, Map::Pending), Wait);
        assert_eq!(collect(Phase::Mapping, Map::Failed), Drop { unmap: false });
        assert_eq!(collect(Phase::Invalid, Map::Pending), Wait);
        assert_eq!(collect(Phase::Invalid, Map::Ok), Drop { unmap: true });
        assert_eq!(collect(Phase::Invalid, Map::Failed), Drop { unmap: false });
        for phase in [Phase::Free, Phase::Recording, Phase::Encoded] {
            for map in [Map::Pending, Map::Ok, Map::Failed] {
                assert_eq!(collect(phase, map), Wait, "{phase:?} {map:?}");
            }
        }
    }

    #[test]
    fn a_slot_invalidated_before_its_map_completes_waits_then_drops() {
        let mut ring = Ring::new(2);
        let slot = submit_one(&mut ring);
        assert_eq!(ring.invalidate(), 1);
        assert_eq!(ring.phase(slot), Some(Phase::Invalid));
        // Still holding the slot: its buffer has a map pending.
        assert_eq!(ring.in_flight(), 1);
        assert_eq!(
            collect(ring.phase(slot).unwrap(), Map::Pending),
            Collect::Wait
        );
        // The map completes on whatever the buffer held; it is not read.
        assert_eq!(
            collect(ring.phase(slot).unwrap(), Map::Ok),
            Collect::Drop { unmap: true }
        );
        ring.release(slot);
        assert_eq!(ring.in_flight(), 0);
    }

    #[test]
    fn a_slot_invalidated_after_its_map_completed_is_still_dropped() {
        let mut ring = Ring::new(2);
        let slot = submit_one(&mut ring);
        // The map has completed (Read, were it collected now), and the host
        // invalidates before the slot is collected.
        assert_eq!(collect(ring.phase(slot).unwrap(), Map::Ok), Collect::Read);
        ring.invalidate();
        assert_eq!(
            collect(ring.phase(slot).unwrap(), Map::Ok),
            Collect::Drop { unmap: true }
        );
    }

    #[test]
    fn invalidation_spares_frames_not_yet_handed_back() {
        let mut ring = Ring::new(3);
        let first = ring.claim().unwrap();
        let first = ring.encode(first).unwrap();
        let second = submit_one(&mut ring);
        let third = ring.claim().unwrap();
        // Out of order: the second is submitted, the first and third are not.
        assert_eq!(ring.invalidate(), 1);
        assert_eq!(ring.phase(second), Some(Phase::Invalid));
        assert_eq!(ring.phase(third), Some(Phase::Recording));
        assert_eq!(ring.submit(first), Some(0));
        assert_eq!(ring.phase(0), Some(Phase::Mapping));
        let third = ring.encode(third).unwrap();
        assert_eq!(ring.submit(third), Some(2));
        assert_eq!(
            ring.slots_in(Phase::Mapping).collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert_eq!(ring.slots_in(Phase::Invalid).collect::<Vec<_>>(), vec![1]);
        // A second invalidation with nothing new in flight moves only the
        // slots submitted since.
        assert_eq!(ring.invalidate(), 2);
        assert_eq!(ring.invalidate(), 0);
    }

    #[test]
    fn an_invalid_slot_holds_its_place_and_the_next_lap_is_readable() {
        let mut ring = Ring::new(2);
        let a = submit_one(&mut ring);
        let b = submit_one(&mut ring);
        ring.invalidate();
        // Both slots wait on their maps: the ring is full, the cursor stays.
        assert_eq!(ring.claim(), None);
        ring.release(a);
        ring.release(b);
        // The next lap on the same slots is a fresh, valid frame.
        let again = submit_one(&mut ring);
        assert_eq!(again, a);
        assert_eq!(ring.phase(again), Some(Phase::Mapping));
        assert_eq!(collect(ring.phase(again).unwrap(), Map::Ok), Collect::Read);
        assert_eq!(submit_one(&mut ring), b);
        assert_eq!(ring.phase(b), Some(Phase::Mapping));
    }

    #[test]
    fn a_zero_depth_ring_still_has_one_slot() {
        let mut ring = Ring::new(0);
        assert_eq!(ring.claim(), Some(0));
        assert_eq!(ring.claim(), None);
    }
}
