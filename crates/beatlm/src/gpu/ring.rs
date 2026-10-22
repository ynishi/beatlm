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
//!
//! And the fourth, for a host that learns of a rejection per submission:
//! **a held slot is collected only once the host has accepted it.**
//! [`Ring::submit_held`] moves the slot to [`Phase::Held`] and hands out a
//! [`Verdict`]; [`Ring::settle`] moves the slot to [`Phase::Mapping`] once
//! the verdict accepts it and to [`Phase::Invalid`] once it rejects it, and a
//! verdict dropped unsettled rejects. Until then [`collect`] waits on it,
//! whatever its map did. Each hold has a cell of its own, so a verdict
//! settled after its slot was invalidated, freed and taken by a later frame
//! reaches nothing.
//!
//! And the fifth: **slots are read in the order they were handed back as
//! submitted.** [`Ring::ready`] holds back a slot it would read while a slot
//! submitted before it may still be read: one [`Phase::Held`], or one in
//! [`Phase::Mapping`] whose map has not completed. A slot that will not be
//! read, [`Phase::Invalid`] or with a failed map, holds back nothing, and is
//! dropped whatever is before it. Submission order is the queue's order, so
//! it is the order the frames ran in; a host that feeds the results to a
//! model that reads the last one as the latest gets them in that order.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

const VERDICT_PENDING: u8 = 0;
const VERDICT_ACCEPTED: u8 = 1;
const VERDICT_REJECTED: u8 = 2;

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
    /// The map is requested, and the host's [`Verdict`] on the submission
    /// is outstanding; not collected until [`Ring::settle`] moves it on.
    Held,
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
/// buffer is no longer pending when the next lap copies into it. A
/// [`Phase::Held`] slot waits for its verdict, and a slot in any other phase
/// has no map to collect.
pub(crate) fn collect(phase: Phase, map: Map) -> Collect {
    match (phase, map) {
        (Phase::Mapping | Phase::Invalid, Map::Pending)
        | (Phase::Free | Phase::Recording | Phase::Encoded | Phase::Held, _) => Collect::Wait,
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

/// The host's word on one held submission, which [`Ring::submit_held`] hands
/// out: accepted, or rejected, which is also what dropping it unsettled says.
///
/// It holds no borrow of the ring, so the host settles it from wherever it
/// learns the outcome.
#[derive(Debug)]
pub(crate) struct Verdict {
    /// The hold's cell, shared with the ring; `None` once settled.
    cell: Option<Arc<AtomicU8>>,
}

impl Verdict {
    /// The device ran the submission: the slot may be read.
    pub(crate) fn accept(mut self) {
        self.settle(VERDICT_ACCEPTED);
    }

    /// The device may have rejected the submission: the slot is dropped
    /// unread.
    pub(crate) fn reject(mut self) {
        self.settle(VERDICT_REJECTED);
    }

    fn settle(&mut self, verdict: u8) {
        if let Some(cell) = self.cell.take() {
            cell.store(verdict, Ordering::Release);
        }
    }
}

impl Drop for Verdict {
    fn drop(&mut self) {
        self.settle(VERDICT_REJECTED);
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
    /// The cell of each [`Phase::Held`] slot's [`Verdict`], `None` in every
    /// other phase. A cell per hold, not per slot, which is what keeps a
    /// late verdict away from the slot's next frame.
    verdicts: Vec<Option<Arc<AtomicU8>>>,
    /// When each slot was last handed back as submitted, as a count of
    /// submissions; meaningful only from [`Phase::Mapping`] on.
    submitted: Vec<u64>,
    submissions: u64,
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
            verdicts: vec![None; depth],
            submitted: vec![0; depth],
            submissions: 0,
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
        self.submissions = self.submissions.saturating_add(1);
        if let Some(order) = self.submitted.get_mut(slot) {
            *order = self.submissions;
        }
        Some(slot)
    }

    /// [`submit`](Self::submit), but the slot moves to [`Phase::Held`] and
    /// is collected only once the returned [`Verdict`] accepts it
    /// ([`settle`](Self::settle)). `None` where `submit` returns `None`.
    pub(crate) fn submit_held(&mut self, ticket: Ticket) -> Option<(usize, Verdict)> {
        let slot = self.submit(ticket)?;
        let cell = Arc::new(AtomicU8::new(VERDICT_PENDING));
        if let (Some(phase), Some(held)) = (self.phases.get_mut(slot), self.verdicts.get_mut(slot))
        {
            *phase = Phase::Held;
            *held = Some(Arc::clone(&cell));
        }
        Some((slot, Verdict { cell: Some(cell) }))
    }

    /// Moves every [`Phase::Held`] slot whose verdict has come: to
    /// [`Phase::Mapping`] when accepted, to [`Phase::Invalid`] when rejected.
    /// Slots whose verdict is outstanding stay held.
    pub(crate) fn settle(&mut self) {
        for (phase, held) in self.phases.iter_mut().zip(&mut self.verdicts) {
            if *phase != Phase::Held {
                continue;
            }
            let verdict = held
                .as_ref()
                .map_or(VERDICT_REJECTED, |cell| cell.load(Ordering::Acquire));
            *phase = match verdict {
                VERDICT_ACCEPTED => Phase::Mapping,
                VERDICT_PENDING => continue,
                _ => Phase::Invalid,
            };
            *held = None;
        }
    }

    /// Moves every slot in [`Phase::Mapping`] or [`Phase::Held`] to
    /// [`Phase::Invalid`] and returns how many moved; a held slot's verdict
    /// no longer matters. Slots not yet handed back as submitted are left as
    /// they are.
    pub(crate) fn invalidate(&mut self) -> usize {
        let mut moved = 0;
        for (phase, held) in self.phases.iter_mut().zip(&mut self.verdicts) {
            if matches!(*phase, Phase::Mapping | Phase::Held) {
                *phase = Phase::Invalid;
                *held = None;
                moved += 1;
            }
        }
        moved
    }

    /// What to do with each slot that can be collected now, oldest
    /// submission first, given `map(slot)`, where each slot's map stands.
    ///
    /// A slot [`collect`] would read is left out while a slot submitted
    /// before it is [`Phase::Held`] or waits on its map in
    /// [`Phase::Mapping`] (the module's fifth rule); slots to drop are never
    /// left out. `map` is asked once per slot past submission. Nothing is
    /// allocated when nothing can be collected.
    pub(crate) fn ready(&self, map: impl Fn(usize) -> Map) -> Vec<(usize, Collect)> {
        let mut ready = Vec::new();
        // The oldest submission that may still be read and cannot be yet.
        let mut barrier = u64::MAX;
        for (i, &phase) in self.phases.iter().enumerate() {
            if !matches!(phase, Phase::Mapping | Phase::Held | Phase::Invalid) {
                continue;
            }
            let order = self.order(i);
            match collect(phase, map(i)) {
                Collect::Wait if phase != Phase::Invalid => barrier = barrier.min(order),
                Collect::Wait => {}
                verdict => ready.push((i, verdict)),
            }
        }
        ready.retain(|&(i, verdict)| verdict != Collect::Read || self.order(i) < barrier);
        ready.sort_by_key(|&(i, _)| self.order(i));
        ready
    }

    /// When `slot` was handed back as submitted; 0 for one the ring lacks.
    fn order(&self, slot: usize) -> u64 {
        self.submitted.get(slot).copied().unwrap_or(0)
    }

    /// The phase of `slot`, `None` for a slot the ring does not have.
    #[cfg(test)]
    pub(crate) fn phase(&self, slot: usize) -> Option<Phase> {
        self.phases.get(slot).copied()
    }

    /// Returns `slot` to the free state, whatever state it was in.
    pub(crate) fn release(&mut self, slot: usize) {
        if let Some(phase) = self.phases.get_mut(slot) {
            *phase = Phase::Free;
        }
        if let Some(held) = self.verdicts.get_mut(slot) {
            *held = None;
        }
    }

    /// The slots currently in `phase`, in slot order.
    #[cfg(test)]
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
        for phase in [Phase::Free, Phase::Recording, Phase::Encoded, Phase::Held] {
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

    /// Claims, encodes and submits one frame held for a verdict.
    fn hold_one(ring: &mut Ring) -> (usize, Verdict) {
        let slot = ring.claim().unwrap();
        let ticket = ring.encode(slot).unwrap();
        ring.submit_held(ticket).unwrap()
    }

    #[test]
    fn a_held_slot_waits_for_its_verdict_whatever_its_map_did() {
        let mut ring = Ring::new(2);
        let (slot, verdict) = hold_one(&mut ring);
        assert_eq!(ring.phase(slot), Some(Phase::Held));
        assert_eq!(ring.in_flight(), 1);
        ring.settle();
        assert_eq!(ring.phase(slot), Some(Phase::Held));
        for map in [Map::Pending, Map::Ok, Map::Failed] {
            assert_eq!(collect(ring.phase(slot).unwrap(), map), Collect::Wait);
        }
        verdict.accept();
        // Nothing moves until the ring is asked to settle.
        assert_eq!(ring.phase(slot), Some(Phase::Held));
        ring.settle();
        assert_eq!(ring.phase(slot), Some(Phase::Mapping));
        assert_eq!(collect(ring.phase(slot).unwrap(), Map::Ok), Collect::Read);
    }

    #[test]
    fn a_rejected_or_dropped_verdict_invalidates_only_its_own_slot() {
        let mut ring = Ring::new(3);
        let (rejected, rejected_verdict) = hold_one(&mut ring);
        let (dropped, dropped_verdict) = hold_one(&mut ring);
        let (accepted, accepted_verdict) = hold_one(&mut ring);
        rejected_verdict.reject();
        drop(dropped_verdict);
        accepted_verdict.accept();
        ring.settle();
        assert_eq!(ring.phase(rejected), Some(Phase::Invalid));
        assert_eq!(ring.phase(dropped), Some(Phase::Invalid));
        assert_eq!(ring.phase(accepted), Some(Phase::Mapping));
        assert_eq!(
            collect(ring.phase(rejected).unwrap(), Map::Ok),
            Collect::Drop { unmap: true }
        );
        assert_eq!(
            collect(ring.phase(accepted).unwrap(), Map::Ok),
            Collect::Read
        );
    }

    /// Every map completed.
    fn all_ok(_: usize) -> Map {
        Map::Ok
    }

    #[test]
    fn a_later_frame_waits_while_an_earlier_one_is_held() {
        let mut ring = Ring::new(3);
        let (earlier, verdict) = hold_one(&mut ring);
        let (later, later_verdict) = hold_one(&mut ring);
        later_verdict.accept();
        ring.settle();
        assert_eq!(ring.phase(later), Some(Phase::Mapping));
        // Readable on its own, but the earlier frame may still be read.
        assert!(ring.ready(all_ok).is_empty());
        // Across calls too: still nothing until the earlier one settles.
        ring.settle();
        assert!(ring.ready(all_ok).is_empty());
        verdict.accept();
        ring.settle();
        assert_eq!(
            ring.ready(all_ok),
            vec![(earlier, Collect::Read), (later, Collect::Read)]
        );
    }

    #[test]
    fn a_later_frame_waits_while_an_earlier_map_is_pending() {
        let mut ring = Ring::new(3);
        let earlier = submit_one(&mut ring);
        let later = submit_one(&mut ring);
        let map =
            |pending: usize| move |i: usize| if i == pending { Map::Pending } else { Map::Ok };
        assert!(ring.ready(map(earlier)).is_empty());
        // The other way round, the earlier frame is read and the later waits
        // for its own map.
        assert_eq!(ring.ready(map(later)), vec![(earlier, Collect::Read)]);
        assert_eq!(
            ring.ready(all_ok),
            vec![(earlier, Collect::Read), (later, Collect::Read)]
        );
    }

    #[test]
    fn a_rejected_earlier_frame_holds_back_nothing() {
        let mut ring = Ring::new(3);
        let (earlier, verdict) = hold_one(&mut ring);
        let (later, later_verdict) = hold_one(&mut ring);
        later_verdict.accept();
        verdict.reject();
        ring.settle();
        assert_eq!(
            ring.ready(all_ok),
            vec![
                (earlier, Collect::Drop { unmap: true }),
                (later, Collect::Read)
            ]
        );
        // Not even while its map is pending: it will never be read.
        let pending_earlier = |i: usize| if i == earlier { Map::Pending } else { Map::Ok };
        assert_eq!(ring.ready(pending_earlier), vec![(later, Collect::Read)]);
        // Nor once it has been dropped.
        ring.release(earlier);
        assert_eq!(ring.ready(all_ok), vec![(later, Collect::Read)]);
    }

    #[test]
    fn a_failed_earlier_map_holds_back_nothing() {
        let mut ring = Ring::new(3);
        let earlier = submit_one(&mut ring);
        let later = submit_one(&mut ring);
        let failed_earlier = |i: usize| if i == earlier { Map::Failed } else { Map::Ok };
        assert_eq!(
            ring.ready(failed_earlier),
            vec![
                (earlier, Collect::Drop { unmap: false }),
                (later, Collect::Read)
            ]
        );
    }

    #[test]
    fn a_frame_held_later_holds_back_only_what_follows_it() {
        let mut ring = Ring::new(3);
        let earlier = submit_one(&mut ring);
        let (held, _verdict) = hold_one(&mut ring);
        let later = submit_one(&mut ring);
        assert_eq!(held, 1);
        assert_eq!(ring.ready(all_ok), vec![(earlier, Collect::Read)]);
        assert_eq!(ring.phase(later), Some(Phase::Mapping));
    }

    #[test]
    fn unheld_frames_are_ready_in_submission_order_as_their_maps_complete() {
        let mut ring = Ring::new(3);
        let a = ring.claim().unwrap();
        let a = ring.encode(a).unwrap();
        let b = ring.claim().unwrap();
        let b = ring.encode(b).unwrap();
        let c = ring.claim().unwrap();
        let c = ring.encode(c).unwrap();
        // Handed back out of claim order: the queue runs them as submitted.
        let c = ring.submit(c).unwrap();
        let a = ring.submit(a).unwrap();
        let b = ring.submit(b).unwrap();
        assert!(ring.ready(|_| Map::Pending).is_empty());
        assert_eq!(
            ring.ready(all_ok),
            vec![(c, Collect::Read), (a, Collect::Read), (b, Collect::Read)]
        );
        // Collected one call at a time, as maps complete in queue order.
        let done = |upto: usize| {
            move |i: usize| {
                if i == upto || i == c {
                    Map::Ok
                } else {
                    Map::Pending
                }
            }
        };
        assert_eq!(ring.ready(done(c)), vec![(c, Collect::Read)]);
        ring.release(c);
        assert_eq!(ring.ready(done(a)), vec![(a, Collect::Read)]);
        ring.release(a);
        assert_eq!(ring.ready(all_ok), vec![(b, Collect::Read)]);
    }

    #[test]
    fn a_held_slot_keeps_the_cursor_as_a_mapping_one_does() {
        let mut ring = Ring::new(2);
        let (first, verdict) = hold_one(&mut ring);
        let _second = submit_one(&mut ring);
        // The cursor is back at the held slot, which is not free.
        assert_eq!(ring.claim(), None);
        verdict.accept();
        ring.settle();
        ring.release(first);
        assert_eq!(ring.claim(), Some(first));
    }

    #[test]
    fn invalidation_takes_held_slots_and_a_late_verdict_reaches_nothing() {
        let mut ring = Ring::new(1);
        let (slot, late) = hold_one(&mut ring);
        assert_eq!(ring.invalidate(), 1);
        assert_eq!(ring.phase(slot), Some(Phase::Invalid));
        ring.release(slot);
        // The slot's next frame is held in turn, and the first frame's
        // verdict, settled now, must neither accept nor reject it.
        let (again, verdict) = hold_one(&mut ring);
        assert_eq!(again, slot);
        late.accept();
        ring.settle();
        assert_eq!(ring.phase(again), Some(Phase::Held));
        verdict.reject();
        ring.settle();
        assert_eq!(ring.phase(again), Some(Phase::Invalid));
    }

    #[test]
    fn a_held_ticket_from_another_ring_is_refused() {
        let mut mine = Ring::new(1);
        let mut theirs = Ring::new(1);
        let slot = theirs.claim().unwrap();
        let foreign = theirs.encode(slot).unwrap();
        assert!(mine.submit_held(foreign).is_none());
        assert_eq!(mine.slots_in(Phase::Held).count(), 0);
        assert_eq!(theirs.in_flight(), 0);
    }

    #[test]
    fn a_zero_depth_ring_still_has_one_slot() {
        let mut ring = Ring::new(0);
        assert_eq!(ring.claim(), Some(0));
        assert_eq!(ring.claim(), None);
    }
}
