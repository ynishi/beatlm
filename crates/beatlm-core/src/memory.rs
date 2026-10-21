//! The memory plan: where every tensor lives for the whole run.
//!
//! A model's tensors are known before the first frame, and so is the order of
//! the operations that use them. This module turns that knowledge into a
//! [`MemoryPlan`]: every tensor gets a byte range in an *arena*, and the arenas
//! are the only GPU buffers a plan needs. Because all memory is decided up
//! front, nothing is allocated while frames run, which is what the "steady
//! state allocates nothing" rule in the [crate documentation](crate#rules)
//! rests on. Realising a plan as buffers is the umbrella crate's job; this
//! module only decides, from values, and replays identically on native and on
//! wasm.
//!
//! # Rules
//!
//! The planner owns these, and each has a test:
//!
//! 1. **Overlapping lifetimes never share a byte.** Two tensors whose use
//!    ranges intersect (inclusive on both ends) are placed at disjoint byte
//!    ranges when they are in the same arena. Tensors in different arenas are
//!    different buffers and share nothing.
//! 2. **Offsets and sizes fit the hardware.** Every offset is a multiple of the
//!    profile's [`min_storage_buffer_offset_alignment`], so a placement can be
//!    bound as it is. Every placement's size is the tensor's byte size rounded
//!    up to a multiple of four bytes, the granularity at which WGSL accesses a
//!    storage buffer.
//! 3. **An arena never exceeds what one binding can cover.** No arena is larger
//!    than [`DeviceProfile::max_arena_size`], the smaller of the largest
//!    storage binding and the largest buffer. A tensor larger than that has no
//!    placement and is [`PlanError::TensorTooLarge`]: a tensor is never split
//!    across arenas, because a kernel addresses a tensor through one binding
//!    and splitting would make every kernel know the split.
//! 4. **The plan is deterministic.** The same tensors and profile give the same
//!    plan, whatever order the tensors are listed in.
//! 5. **Malformed input is an error, not a guess.** A repeated [`TensorId`], a
//!    tensor whose `first_use` is after its `last_use`, a tensor of no
//!    elements, and sizes that overflow `u64` are reported as [`PlanError`].
//!
//! A plan made against [`DeviceProfile::webgpu_baseline`] is valid on every
//! adapter; [`MemoryPlan::valid_on`] states the check for any other profile.
//!
//! # Lifetimes
//!
//! A use is an operation's index in program order. A tensor is live from its
//! `first_use` to its `last_use`, both inclusive: an operation reads its inputs
//! and writes its outputs while all of them are live, so an operation's input
//! and output are never given the same bytes even when the input dies at that
//! operation. A weight is a tensor whose lifetime covers the whole program,
//! from operation 0 to the last operation.
//!
//! The planner reasons about one run of the program. Whoever runs the program
//! more than once (a frame loop does) must not start the next run's first
//! operation while the previous run's later operations still read their
//! tensors, or give a tensor that is rewritten between runs a short lifetime:
//! such a tensor, an input the host refills, is live for the whole program.
//!
//! # Algorithm
//!
//! **Greedy by size, descending, ties by id; each tensor goes to the lowest
//! aligned offset in the first arena where it fits without overlapping an
//! already-placed tensor whose lifetime intersects its own; a new arena opens
//! when none fits.** Large tensors are placed first because they constrain the
//! layout most and small ones fill the gaps they leave.
//!
//! It is the choice here because it is simple, deterministic without any
//! tie-breaking beyond the id, and close to the lower bound on the graphs
//! inference produces, where lifetimes are short and nested along a chain of
//! operations. [`MemoryPlan::peak_live_bytes`] is that lower bound (no layout
//! can use fewer bytes than the most that is live at once, ignoring alignment),
//! and [`MemoryPlan::total_bytes`] is what the layout used, so the gap is
//! measurable on any model. This is the decision expected to move: a better
//! packer (best-fit, a search over orders, or splitting by lifetime class) only
//! has to keep the five rules above, and callers see only
//! [`MemoryPlan::arenas`] and [`MemoryPlan::placement`].
//!
//! [`min_storage_buffer_offset_alignment`]: DeviceProfile::min_storage_buffer_offset_alignment

use std::collections::{BTreeMap, BTreeSet};

use thiserror::Error;

use crate::device::DeviceProfile;

/// Bytes WGSL reads and writes a storage buffer in; every placement's size is a
/// multiple of it.
const STORAGE_GRANULARITY: u64 = 4;

/// Identifies one tensor of the model.
///
/// The model graph assigns ids; the plan keys placements by them and orders
/// ties between equal-sized tensors by them, which is what makes a plan
/// independent of the order tensors are listed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TensorId(u32);

impl TensorId {
    /// Wraps a tensor number.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// The tensor number.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Identifies one arena of a plan: its position in [`MemoryPlan::arenas`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ArenaId(u32);

impl ArenaId {
    /// Wraps an arena number.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// The arena number.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// The element type of a tensor, which fixes how many bytes it occupies.
///
/// Storage is bytes whatever the type, so a type needs no device feature to be
/// planned; `F16` is available to a kernel only when
/// [`DeviceFeatures::shader_f16`](crate::device::DeviceFeatures::shader_f16)
/// is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DType {
    /// 32-bit float.
    F32,
    /// 16-bit float.
    F16,
}

impl DType {
    /// Bytes per element.
    #[must_use]
    pub const fn size_bytes(self) -> u64 {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
        }
    }
}

/// What the planner needs to know about a tensor: how big it is and when it is
/// in use.
///
/// The shape is the model graph's business; the planner sees only the element
/// count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorDesc {
    /// The tensor's identity; unique within one plan.
    pub id: TensorId,
    /// The element type.
    pub dtype: DType,
    /// The number of elements; at least one.
    pub elements: u64,
    /// The first operation that uses the tensor, as an index in program order.
    pub first_use: u32,
    /// The last operation that uses the tensor; not before `first_use`.
    pub last_use: u32,
}

impl TensorDesc {
    /// Describes a tensor used from operation `first_use` to operation
    /// `last_use`, both inclusive.
    #[must_use]
    pub const fn new(
        id: TensorId,
        dtype: DType,
        elements: u64,
        first_use: u32,
        last_use: u32,
    ) -> Self {
        Self {
            id,
            dtype,
            elements,
            first_use,
            last_use,
        }
    }

    /// Whether the two tensors are in use at the same operation, in which case
    /// they must not share bytes.
    const fn overlaps(&self, other: &Self) -> bool {
        self.first_use <= other.last_use && other.first_use <= self.last_use
    }
}

/// Where a tensor lives: a byte range of an arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Placement {
    /// The arena the tensor is in.
    pub arena: ArenaId,
    /// The byte offset of the tensor in the arena; a multiple of the profile's
    /// storage offset alignment.
    pub offset: u64,
    /// The bytes reserved: the tensor's size rounded up to a multiple of four.
    pub size: u64,
}

impl Placement {
    /// The offset one past the last reserved byte.
    #[must_use]
    pub const fn end(&self) -> u64 {
        self.offset.saturating_add(self.size)
    }
}

/// Why a set of tensors could not be planned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum PlanError {
    /// Two tensors share an id, so a placement could not be looked up by it.
    #[error("tensor {} is described more than once", .0.get())]
    DuplicateTensor(TensorId),
    /// A tensor whose first use is after its last use has no lifetime.
    #[error("tensor {} is first used at operation {first_use}, after its last use at {last_use}", id.get())]
    InvalidLifetime {
        /// The offending tensor.
        id: TensorId,
        /// Its first use.
        first_use: u32,
        /// Its last use.
        last_use: u32,
    },
    /// A tensor of no elements has nothing to place and cannot be bound.
    #[error("tensor {} has no elements", .0.get())]
    EmptyTensor(TensorId),
    /// The tensor's byte size does not fit a `u64`.
    #[error("the size of tensor {} overflows", .0.get())]
    SizeOverflow(TensorId),
    /// The tensor is larger than any one arena may be, and tensors are never
    /// split.
    #[error("tensor {} needs {size} bytes but an arena holds at most {limit}", id.get())]
    TensorTooLarge {
        /// The offending tensor.
        id: TensorId,
        /// Its size in bytes after rounding.
        size: u64,
        /// The largest arena the profile allows.
        limit: u64,
    },
    /// The profile cannot be planned against: a zero alignment, or no room in
    /// an arena.
    #[error("the device profile has a zero offset alignment or a zero arena size")]
    InvalidProfile,
    /// More tensors than an [`ArenaId`] or an operation count can index.
    #[error("too many tensors to plan")]
    TooManyTensors,
}

/// Where every tensor lives, and how big each arena is.
///
/// Built by [`MemoryPlan::new`]; read through [`placement`](Self::placement)
/// and [`arenas`](Self::arenas). The rules the plan keeps are in the
/// [module documentation](self#rules).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemoryPlan {
    arenas: Vec<u64>,
    placements: BTreeMap<TensorId, Placement>,
    peak_live_bytes: u64,
    offset_alignment: u64,
}

impl MemoryPlan {
    /// Plans `tensors` against `profile`.
    ///
    /// The tensors may be listed in any order; the plan depends only on the set.
    /// The algorithm and the rules it keeps are in the [module
    /// documentation](self).
    ///
    /// # Errors
    ///
    /// A [`PlanError`] for a repeated id, an empty tensor, a first use after the
    /// last, a size that overflows, a tensor larger than an arena may be, or a
    /// profile with a zero alignment or arena size.
    pub fn new(tensors: &[TensorDesc], profile: &DeviceProfile) -> Result<Self, PlanError> {
        let alignment = u64::from(profile.min_storage_buffer_offset_alignment);
        let cap = profile.max_arena_size();
        if alignment == 0 || cap == 0 {
            return Err(PlanError::InvalidProfile);
        }
        // One arena per tensor at most, and arenas are indexed by `u32`.
        if u32::try_from(tensors.len()).is_err() {
            return Err(PlanError::TooManyTensors);
        }

        let sized = Self::validate(tensors, cap)?;
        let peak_live_bytes = peak_live(&sized);

        let mut order = sized;
        order.sort_by(|a, b| b.size.cmp(&a.size).then(a.desc.id.cmp(&b.desc.id)));

        let mut arenas: Vec<Arena> = Vec::new();
        let mut placements = BTreeMap::new();
        for item in &order {
            let slot = arenas
                .iter()
                .enumerate()
                .find_map(|(i, arena)| arena.lowest_fit(item, alignment, cap).map(|o| (i, o)));
            let (index, offset) = match slot {
                Some(found) => found,
                None => {
                    arenas.push(Arena::default());
                    (arenas.len() - 1, 0)
                }
            };
            // `arenas.len()` fits a `u32` because `tensors.len()` does.
            let arena_id =
                ArenaId::new(u32::try_from(index).map_err(|_| PlanError::TooManyTensors)?);
            if let Some(arena) = arenas.get_mut(index) {
                arena.place(*item, offset);
            }
            placements.insert(
                item.desc.id,
                Placement {
                    arena: arena_id,
                    offset,
                    size: item.size,
                },
            );
        }

        Ok(Self {
            arenas: arenas.iter().map(|a| a.size).collect(),
            placements,
            peak_live_bytes,
            offset_alignment: alignment,
        })
    }

    /// Checks the input and attaches each tensor's rounded byte size.
    fn validate(tensors: &[TensorDesc], cap: u64) -> Result<Vec<Sized>, PlanError> {
        let mut seen = BTreeSet::new();
        let mut sized = Vec::with_capacity(tensors.len());
        for desc in tensors {
            if !seen.insert(desc.id) {
                return Err(PlanError::DuplicateTensor(desc.id));
            }
            if desc.first_use > desc.last_use {
                return Err(PlanError::InvalidLifetime {
                    id: desc.id,
                    first_use: desc.first_use,
                    last_use: desc.last_use,
                });
            }
            if desc.elements == 0 {
                return Err(PlanError::EmptyTensor(desc.id));
            }
            let size = desc
                .elements
                .checked_mul(desc.dtype.size_bytes())
                .and_then(|bytes| round_up(bytes, STORAGE_GRANULARITY))
                .ok_or(PlanError::SizeOverflow(desc.id))?;
            if size > cap {
                return Err(PlanError::TensorTooLarge {
                    id: desc.id,
                    size,
                    limit: cap,
                });
            }
            sized.push(Sized { desc: *desc, size });
        }
        Ok(sized)
    }

    /// The size in bytes of each arena, indexed by [`ArenaId`]. Each is the end
    /// of the last byte any tensor in it uses, so it is a multiple of four and
    /// no larger than the profile allows.
    #[must_use]
    pub fn arenas(&self) -> &[u64] {
        &self.arenas
    }

    /// Where `id` lives, or `None` for a tensor the plan was not made with.
    #[must_use]
    pub fn placement(&self, id: TensorId) -> Option<Placement> {
        self.placements.get(&id).copied()
    }

    /// Every tensor's placement, in id order.
    pub fn placements(&self) -> impl Iterator<Item = (TensorId, Placement)> + '_ {
        self.placements.iter().map(|(id, p)| (*id, *p))
    }

    /// The most bytes live at any one operation: the maximum over operation
    /// indices of the summed placement sizes of the tensors in use there.
    ///
    /// No layout can use less, so it is the lower bound that [`total_bytes`]
    /// is measured against. It ignores alignment padding, which a layout cannot
    /// avoid in general.
    ///
    /// [`total_bytes`]: Self::total_bytes
    #[must_use]
    pub const fn peak_live_bytes(&self) -> u64 {
        self.peak_live_bytes
    }

    /// The bytes of all arenas together: what realising the plan allocates.
    /// The gap to [`peak_live_bytes`](Self::peak_live_bytes) is what the layout
    /// wastes to fragmentation, alignment and arena boundaries.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.arenas.iter().fold(0, |sum, &a| sum.saturating_add(a))
    }

    /// Whether this plan can be realised on a device described by `profile`:
    /// every arena fits one binding and one buffer there, and every offset is
    /// aligned for the profile's alignment.
    ///
    /// True for any profile when the plan was made against
    /// [`DeviceProfile::webgpu_baseline`], which is the guarantee that makes the
    /// baseline the profile to plan against when the device is not known.
    #[must_use]
    pub fn valid_on(&self, profile: &DeviceProfile) -> bool {
        let alignment = u64::from(profile.min_storage_buffer_offset_alignment);
        // A zero alignment has no remainder, so it is not valid either.
        self.offset_alignment.checked_rem(alignment) == Some(0)
            && self.arenas.iter().all(|&a| a <= profile.max_arena_size())
    }
}

/// A tensor with its rounded byte size.
#[derive(Debug, Clone, Copy)]
struct Sized {
    desc: TensorDesc,
    size: u64,
}

/// An arena being filled: the tensors placed so far and the extent they use.
#[derive(Debug, Default)]
struct Arena {
    placed: Vec<(TensorDesc, u64, u64)>,
    size: u64,
}

impl Arena {
    /// The lowest aligned offset at which `item` fits here without overlapping
    /// a tensor in use at the same time, or `None` when it would pass `cap`.
    fn lowest_fit(&self, item: &Sized, alignment: u64, cap: u64) -> Option<u64> {
        let mut busy: Vec<(u64, u64)> = self
            .placed
            .iter()
            .filter(|(other, _, _)| other.overlaps(&item.desc))
            .map(|&(_, offset, size)| (offset, offset.saturating_add(size)))
            .collect();
        busy.sort_unstable();

        let mut candidate = 0;
        for (start, end) in busy {
            let aligned = round_up(candidate, alignment)?;
            if aligned.checked_add(item.size)? <= start {
                break;
            }
            candidate = candidate.max(end);
        }
        let offset = round_up(candidate, alignment)?;
        (offset.checked_add(item.size)? <= cap).then_some(offset)
    }

    fn place(&mut self, item: Sized, offset: u64) {
        let end = offset.saturating_add(item.size);
        self.placed.push((item.desc, offset, item.size));
        self.size = self.size.max(end);
    }
}

/// `value` rounded up to a multiple of `multiple`, or `None` on overflow.
fn round_up(value: u64, multiple: u64) -> Option<u64> {
    let rem = value % multiple;
    if rem == 0 {
        Some(value)
    } else {
        value.checked_add(multiple - rem)
    }
}

/// The largest summed size of tensors in use at one operation.
fn peak_live(sized: &[Sized]) -> u64 {
    // A tensor adds its size at its first use and gives it back one operation
    // after its last; widened to `u64` so `last_use + 1` cannot overflow.
    let mut deltas: BTreeMap<u64, (u64, u64)> = BTreeMap::new();
    for item in sized {
        let born = deltas.entry(u64::from(item.desc.first_use)).or_default();
        born.0 = born.0.saturating_add(item.size);
        let died = deltas.entry(u64::from(item.desc.last_use) + 1).or_default();
        died.1 = died.1.saturating_add(item.size);
    }
    let mut live: u64 = 0;
    let mut peak: u64 = 0;
    for (added, removed) in deltas.into_values() {
        live = live.saturating_add(added).saturating_sub(removed);
        peak = peak.max(live);
    }
    peak
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn id(n: u32) -> TensorId {
        TensorId::new(n)
    }

    fn f32s(n: u32, elements: u64, first: u32, last: u32) -> TensorDesc {
        TensorDesc::new(id(n), DType::F32, elements, first, last)
    }

    fn baseline() -> DeviceProfile {
        DeviceProfile::webgpu_baseline()
    }

    /// A small deterministic generator so the property tests need no
    /// dependency and replay identically.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 33
        }

        fn below(&mut self, bound: u64) -> u64 {
            self.next() % bound
        }
    }

    fn random_graph(rng: &mut Lcg, max_elements: u64) -> Vec<TensorDesc> {
        let count = 1 + rng.below(40) as u32;
        let ops = 1 + rng.below(30) as u32;
        (0..count)
            .map(|n| {
                let first = rng.below(u64::from(ops)) as u32;
                let last = first + rng.below(u64::from(ops - first)) as u32;
                let dtype = if rng.below(2) == 0 {
                    DType::F32
                } else {
                    DType::F16
                };
                TensorDesc::new(id(n), dtype, 1 + rng.below(max_elements), first, last)
            })
            .collect()
    }

    fn random_profile(rng: &mut Lcg) -> DeviceProfile {
        let mut p = baseline();
        p.min_storage_buffer_offset_alignment = [4, 16, 64, 256][rng.below(4) as usize];
        // Small caps force several arenas out of graphs of a few kilobytes.
        p.max_storage_buffer_binding_size = 4096 * (1 + rng.below(8));
        p.max_buffer_size = p.max_storage_buffer_binding_size * 2;
        p
    }

    /// Asserts rules 1 to 3 and the accessors' consistency for `plan`.
    fn assert_plan_holds(tensors: &[TensorDesc], profile: &DeviceProfile, plan: &MemoryPlan) {
        let alignment = u64::from(profile.min_storage_buffer_offset_alignment);
        let cap = profile.max_arena_size();
        for desc in tensors {
            let p = plan.placement(desc.id).unwrap();
            let bytes = desc.elements * desc.dtype.size_bytes();
            // Rule 2.
            assert_eq!(p.offset % alignment, 0, "offset of {:?}", desc.id);
            assert_eq!(p.size % 4, 0);
            assert!(p.size >= bytes && p.size < bytes + 4);
            // Rule 3.
            assert!(p.end() <= plan.arenas()[p.arena.get() as usize]);
        }
        for &size in plan.arenas() {
            assert!(size <= cap, "arena of {size} bytes exceeds {cap}");
        }
        // Rule 1.
        for (i, a) in tensors.iter().enumerate() {
            for b in &tensors[i + 1..] {
                let (pa, pb) = (plan.placement(a.id).unwrap(), plan.placement(b.id).unwrap());
                if a.overlaps(b) && pa.arena == pb.arena {
                    assert!(
                        pa.end() <= pb.offset || pb.end() <= pa.offset,
                        "{a:?} at {pa:?} and {b:?} at {pb:?} share bytes"
                    );
                }
            }
        }
        assert!(plan.peak_live_bytes() <= plan.total_bytes());
        assert!(plan.valid_on(profile));
    }

    #[test]
    fn disjoint_lifetimes_share_the_same_bytes() {
        let t = [f32s(0, 1024, 0, 1), f32s(1, 1024, 2, 3)];
        let plan = MemoryPlan::new(&t, &baseline()).unwrap();
        let (a, b) = (
            plan.placement(id(0)).unwrap(),
            plan.placement(id(1)).unwrap(),
        );
        assert_eq!((a.arena, a.offset), (b.arena, b.offset));
        assert_eq!(plan.total_bytes(), 4096);
        assert_eq!(plan.peak_live_bytes(), 4096);
    }

    #[test]
    fn lifetimes_are_inclusive_at_both_ends() {
        // Tensor 0 dies at operation 2 and tensor 1 is born at operation 2:
        // the operation reads one and writes the other, so they must not share.
        let t = [f32s(0, 1024, 0, 2), f32s(1, 1024, 2, 4)];
        let plan = MemoryPlan::new(&t, &baseline()).unwrap();
        let (a, b) = (
            plan.placement(id(0)).unwrap(),
            plan.placement(id(1)).unwrap(),
        );
        assert!(a.end() <= b.offset || b.end() <= a.offset);
        assert_eq!(plan.peak_live_bytes(), 8192);
    }

    #[test]
    fn overlapping_lifetimes_do_not_share_bytes() {
        let t = [f32s(0, 100, 0, 5), f32s(1, 50, 3, 8), f32s(2, 70, 4, 4)];
        let plan = MemoryPlan::new(&t, &baseline()).unwrap();
        assert_plan_holds(&t, &baseline(), &plan);
    }

    #[test]
    fn offsets_are_aligned_and_sizes_rounded_to_four_bytes() {
        // Three f16 elements are 6 bytes, reserved as 8; the next placement
        // that overlaps it starts on the next 256-byte boundary.
        let t = [
            TensorDesc::new(id(0), DType::F16, 3, 0, 1),
            TensorDesc::new(id(1), DType::F16, 1, 0, 1),
        ];
        let plan = MemoryPlan::new(&t, &baseline()).unwrap();
        let a = plan.placement(id(0)).unwrap();
        let b = plan.placement(id(1)).unwrap();
        assert_eq!(a.size, 8);
        assert_eq!(b.size, 4);
        assert_eq!((a.offset, b.offset), (0, 256));
        assert_eq!(plan.arenas(), &[260]);
    }

    #[test]
    fn a_tensor_larger_than_an_arena_is_an_error() {
        let mut p = baseline();
        p.max_storage_buffer_binding_size = 1024;
        let t = [f32s(0, 256, 0, 0), f32s(1, 257, 0, 0)];
        assert_eq!(
            MemoryPlan::new(&t, &p),
            Err(PlanError::TensorTooLarge {
                id: id(1),
                size: 1028,
                limit: 1024
            })
        );
        // The buffer size bounds an arena as well.
        let mut p = baseline();
        p.max_buffer_size = 1024;
        assert!(matches!(
            MemoryPlan::new(&[f32s(0, 257, 0, 0)], &p),
            Err(PlanError::TensorTooLarge { .. })
        ));
    }

    #[test]
    fn a_tensor_exactly_the_arena_size_fits() {
        let mut p = baseline();
        p.max_storage_buffer_binding_size = 1024;
        let plan = MemoryPlan::new(&[f32s(0, 256, 0, 0)], &p).unwrap();
        assert_eq!(plan.arenas(), &[1024]);
    }

    #[test]
    fn a_full_arena_opens_a_new_one() {
        let mut p = baseline();
        p.max_storage_buffer_binding_size = 1024;
        let t = [f32s(0, 200, 0, 1), f32s(1, 200, 0, 1)];
        let plan = MemoryPlan::new(&t, &p).unwrap();
        assert_eq!(plan.arenas().len(), 2);
        assert_ne!(
            plan.placement(id(0)).unwrap().arena,
            plan.placement(id(1)).unwrap().arena
        );
        assert_plan_holds(&t, &p, &plan);
    }

    #[test]
    fn duplicate_ids_are_an_error() {
        let t = [f32s(7, 4, 0, 0), f32s(7, 8, 1, 1)];
        assert_eq!(
            MemoryPlan::new(&t, &baseline()),
            Err(PlanError::DuplicateTensor(id(7)))
        );
    }

    #[test]
    fn a_first_use_after_the_last_is_an_error() {
        let t = [f32s(3, 4, 5, 4)];
        assert_eq!(
            MemoryPlan::new(&t, &baseline()),
            Err(PlanError::InvalidLifetime {
                id: id(3),
                first_use: 5,
                last_use: 4
            })
        );
    }

    #[test]
    fn empty_tensors_overflow_and_bad_profiles_are_errors() {
        assert_eq!(
            MemoryPlan::new(&[f32s(0, 0, 0, 0)], &baseline()),
            Err(PlanError::EmptyTensor(id(0)))
        );
        assert_eq!(
            MemoryPlan::new(&[f32s(0, u64::MAX, 0, 0)], &baseline()),
            Err(PlanError::SizeOverflow(id(0)))
        );
        let mut p = baseline();
        p.min_storage_buffer_offset_alignment = 0;
        assert_eq!(
            MemoryPlan::new(&[f32s(0, 1, 0, 0)], &p),
            Err(PlanError::InvalidProfile)
        );
        let mut p = baseline();
        p.max_buffer_size = 0;
        assert_eq!(
            MemoryPlan::new(&[f32s(0, 1, 0, 0)], &p),
            Err(PlanError::InvalidProfile)
        );
    }

    #[test]
    fn no_tensors_is_an_empty_plan() {
        let plan = MemoryPlan::new(&[], &baseline()).unwrap();
        assert!(plan.arenas().is_empty());
        assert_eq!(plan.total_bytes(), 0);
        assert_eq!(plan.peak_live_bytes(), 0);
        assert_eq!(plan.placement(id(0)), None);
    }

    #[test]
    fn the_plan_does_not_depend_on_the_order_tensors_are_listed_in() {
        let mut rng = Lcg(7);
        for _ in 0..50 {
            let profile = random_profile(&mut rng);
            let mut tensors = random_graph(&mut rng, 600);
            let forward = MemoryPlan::new(&tensors, &profile);
            tensors.reverse();
            let backward = MemoryPlan::new(&tensors, &profile);
            assert_eq!(forward, backward);
            assert_eq!(forward, MemoryPlan::new(&tensors, &profile));
        }
    }

    #[test]
    fn peak_live_bytes_is_the_most_live_at_one_operation() {
        let t = [f32s(0, 100, 0, 2), f32s(1, 50, 1, 1), f32s(2, 25, 3, 3)];
        let plan = MemoryPlan::new(&t, &baseline()).unwrap();
        // Operation 1 holds tensors 0 and 1: 400 + 200 bytes.
        assert_eq!(plan.peak_live_bytes(), 600);
    }

    #[test]
    fn a_chain_of_operations_reuses_memory_down_to_two_activations() {
        // x -> a -> b -> c -> d: each activation is read by the next operation
        // only, so at most two are live at once.
        let activation = 1 << 20;
        let t: Vec<TensorDesc> = (0..8).map(|n| f32s(n, activation, n, n + 1)).collect();
        let plan = MemoryPlan::new(&t, &baseline()).unwrap();
        assert_eq!(plan.peak_live_bytes(), 2 * activation * 4);
        assert_eq!(plan.total_bytes(), 2 * activation * 4);
    }

    #[test]
    fn rules_one_to_three_hold_on_many_generated_graphs() {
        let mut rng = Lcg(0x5eed);
        let (mut multi_arena, mut reused, mut too_large) = (0, 0, 0);
        for _ in 0..500 {
            let profile = random_profile(&mut rng);
            let tensors = random_graph(&mut rng, 3000);
            match MemoryPlan::new(&tensors, &profile) {
                Ok(plan) => {
                    assert_plan_holds(&tensors, &profile, &plan);
                    multi_arena += usize::from(plan.arenas().len() > 1);
                    let summed: u64 = plan.placements().map(|(_, p)| p.size).sum();
                    reused += usize::from(plan.total_bytes() < summed);
                }
                Err(PlanError::TensorTooLarge { id, size, limit }) => {
                    too_large += 1;
                    assert!(size > limit);
                    let desc = tensors.iter().find(|t| t.id == id).unwrap();
                    assert!(desc.elements * desc.dtype.size_bytes() + 3 >= size);
                }
                Err(other) => panic!("unexpected error {other:?}"),
            }
        }
        // The generator reaches the interesting cases, so the loop above is
        // not passing on easy inputs only.
        assert!(
            multi_arena > 20 && reused > 20 && too_large > 5,
            "{multi_arena} {reused} {too_large}"
        );
    }

    #[test]
    fn rules_one_to_three_hold_on_wide_graphs_against_the_baseline() {
        let mut rng = Lcg(99);
        let profile = baseline();
        for _ in 0..200 {
            let tensors = random_graph(&mut rng, 1 << 22);
            let plan = MemoryPlan::new(&tensors, &profile).unwrap();
            assert_plan_holds(&tensors, &profile, &plan);
        }
    }

    #[test]
    fn a_baseline_plan_is_valid_on_a_larger_device_but_not_the_reverse() {
        let t = [f32s(0, 1 << 20, 0, 1)];
        let on_baseline = MemoryPlan::new(&t, &baseline()).unwrap();
        let mut big = baseline();
        big.max_storage_buffer_binding_size = 1 << 30;
        big.max_buffer_size = 1 << 31;
        big.min_storage_buffer_offset_alignment = 64;
        assert!(on_baseline.valid_on(&big));

        let on_big = MemoryPlan::new(&[f32s(0, 40 << 20, 0, 1)], &big).unwrap();
        assert!(!on_big.valid_on(&baseline()));
        // An alignment that does not divide the plan's is not valid either.
        let mut odd = baseline();
        odd.min_storage_buffer_offset_alignment = 512;
        assert!(!on_baseline.valid_on(&odd));
    }
}
