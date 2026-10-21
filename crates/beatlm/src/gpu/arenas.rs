//! A memory plan realised as GPU buffers.
//!
//! [`Arenas`] creates one storage buffer per arena of a
//! [`MemoryPlan`], once, and hands out bindings of the placements inside them.
//! That creation is the only allocation a plan needs: every tensor already has
//! a byte range, so the steady-state loop must not create a buffer, and nothing
//! in this module lets it (a binding is a borrow of an existing buffer).
//! `beatlm-bench` measures the rule against wgpu's own counters.

use beatlm_core::device::DeviceProfile;
use beatlm_core::memory::{ArenaId, MemoryPlan, Placement};
use thiserror::Error;

use super::profile::profile;

/// Why arenas could not be created or a placement could not be bound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum ArenaError {
    /// The plan was made for a device with larger limits or a finer offset
    /// alignment than this one provides.
    #[error("the memory plan is not valid on this device: {device:?}")]
    PlanNotValidOnDevice {
        /// What the device provides.
        device: DeviceProfile,
    },
    /// The placement names an arena this set does not have: it came from
    /// another plan.
    #[error("arena {} does not exist in this set", .0.get())]
    UnknownArena(ArenaId),
    /// The placement reaches past the end of its arena, or is empty: it came
    /// from another plan.
    #[error("placement at offset {offset} of {size} bytes does not lie inside its arena")]
    PlacementOutOfRange {
        /// The placement's offset.
        offset: u64,
        /// The placement's size.
        size: u64,
    },
    /// The placement's offset is not a multiple of the device's storage offset
    /// alignment: it came from another plan.
    #[error("placement at offset {offset} is not aligned to {alignment} bytes")]
    MisalignedPlacement {
        /// The placement's offset.
        offset: u64,
        /// The alignment the device requires.
        alignment: u64,
    },
}

/// The buffers of a realised [`MemoryPlan`], one per arena.
///
/// Created once, and the only allocation the plan needs: the steady-state loop
/// binds placements of these buffers and creates no buffer of its own.
///
/// # One access mode per arena in a dispatch
///
/// Tensors that share an arena share a buffer, and wgpu checks resource usage
/// per buffer, not per byte range: a dispatch that binds one range of a buffer
/// read-only and another range of the same buffer read-write is rejected with a
/// usage conflict, although the ranges are disjoint. A kernel that takes inputs
/// and an output from the same arena therefore declares all of them with one
/// access mode (`var<storage, read_write>`), and the planner's rule that
/// overlapping lifetimes never share bytes is what makes that safe.
#[derive(Debug)]
pub struct Arenas {
    buffers: Vec<wgpu::Buffer>,
    /// The device's storage offset alignment, which every binding's offset
    /// must be a multiple of.
    offset_alignment: u64,
}

impl Arenas {
    /// The usages of an arena: bound as storage by kernels, written from the
    /// host for weights and inputs, and copied from for reading results back.
    const USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
        .union(wgpu::BufferUsages::COPY_DST)
        .union(wgpu::BufferUsages::COPY_SRC);

    /// Creates one buffer per arena of `plan` on `device`.
    ///
    /// # Errors
    ///
    /// [`ArenaError::PlanNotValidOnDevice`] when the device's limits are below
    /// what the plan was made against, which a plan made against
    /// [`DeviceProfile::webgpu_baseline`] never is. A device created with
    /// [`required_limits`](super::required_limits) for the plan is never
    /// below it.
    pub fn new(device: &wgpu::Device, plan: &MemoryPlan) -> Result<Self, ArenaError> {
        let device_profile = profile(&device.limits(), device.features());
        if !plan.valid_on(&device_profile) {
            return Err(ArenaError::PlanNotValidOnDevice {
                device: device_profile,
            });
        }
        let buffers = plan
            .arenas()
            .iter()
            .enumerate()
            .map(|(index, &size)| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("beatlm arena {index}")),
                    size,
                    usage: Self::USAGE,
                    mapped_at_creation: false,
                })
            })
            .collect();
        Ok(Self {
            buffers,
            offset_alignment: u64::from(device_profile.min_storage_buffer_offset_alignment),
        })
    }

    /// The buffer behind an arena, for writes from the host and copies out of
    /// it; `None` for an id this set does not have.
    #[must_use]
    pub fn buffer(&self, arena: ArenaId) -> Option<&wgpu::Buffer> {
        self.buffers.get(usize::try_from(arena.get()).ok()?)
    }

    /// The binding of a placement: its arena's buffer, its offset and its size,
    /// ready for a bind group entry.
    ///
    /// The offset is a multiple of the device's storage offset alignment
    /// because the plan guarantees it. Every binding of one arena in a dispatch
    /// must use the same access mode; see [`Arenas`].
    ///
    /// # Errors
    ///
    /// A placement that did not come from this set's plan.
    pub fn binding(&self, placement: &Placement) -> Result<wgpu::BufferBinding<'_>, ArenaError> {
        let buffer = self
            .buffer(placement.arena)
            .ok_or(ArenaError::UnknownArena(placement.arena))?;
        let out_of_range = ArenaError::PlacementOutOfRange {
            offset: placement.offset,
            size: placement.size,
        };
        if placement.end() > buffer.size() {
            return Err(out_of_range);
        }
        let size = wgpu::BufferSize::new(placement.size).ok_or(out_of_range)?;
        // `valid_on` refused a zero alignment in `new`, so the remainder exists.
        if placement.offset.checked_rem(self.offset_alignment) != Some(0) {
            return Err(ArenaError::MisalignedPlacement {
                offset: placement.offset,
                alignment: self.offset_alignment,
            });
        }
        Ok(wgpu::BufferBinding {
            buffer,
            offset: placement.offset,
            size: Some(size),
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use beatlm_core::memory::{DType, TensorDesc, TensorId};

    async fn device() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .unwrap()
    }

    fn plan() -> MemoryPlan {
        let tensors = [
            TensorDesc::new(TensorId::new(0), DType::F32, 1000, 0, 1),
            TensorDesc::new(TensorId::new(1), DType::F32, 1000, 1, 2),
            TensorDesc::new(TensorId::new(2), DType::F32, 1000, 3, 4),
        ];
        MemoryPlan::new(&tensors, &DeviceProfile::webgpu_baseline()).unwrap()
    }

    #[test]
    #[ignore = "needs a GPU adapter"]
    fn one_buffer_per_arena_is_created_and_placements_bind_inside_it() {
        let (device, _queue) = pollster::block_on(device());
        let plan = plan();
        let arenas = Arenas::new(&device, &plan).unwrap();
        for (id, placement) in plan.placements() {
            let binding = arenas.binding(&placement).unwrap();
            assert_eq!(binding.offset, placement.offset, "{id:?}");
            assert_eq!(binding.size.unwrap().get(), placement.size);
            assert_eq!(
                binding.buffer.size(),
                plan.arenas()[placement.arena.get() as usize]
            );
        }
    }

    #[test]
    #[ignore = "needs a GPU adapter"]
    fn a_placement_from_another_plan_is_rejected() {
        let (device, _queue) = pollster::block_on(device());
        let arenas = Arenas::new(&device, &plan()).unwrap();
        let elsewhere = Placement {
            arena: ArenaId::new(9),
            offset: 0,
            size: 4,
        };
        assert_eq!(
            arenas.binding(&elsewhere).unwrap_err(),
            ArenaError::UnknownArena(ArenaId::new(9))
        );
        let past_the_end = Placement {
            arena: ArenaId::new(0),
            offset: 0,
            size: 1 << 40,
        };
        assert!(matches!(
            arenas.binding(&past_the_end),
            Err(ArenaError::PlacementOutOfRange { .. })
        ));
        let misaligned = Placement {
            arena: ArenaId::new(0),
            offset: 4,
            size: 4,
        };
        assert!(matches!(
            arenas.binding(&misaligned),
            Err(ArenaError::MisalignedPlacement { offset: 4, .. })
        ));
    }

    #[test]
    #[ignore = "needs a GPU adapter"]
    fn a_plan_for_a_larger_device_is_refused() {
        let (device, _queue) = pollster::block_on(device());
        let mut big = DeviceProfile::webgpu_baseline();
        big.max_storage_buffer_binding_size = 1 << 40;
        big.max_buffer_size = 1 << 41;
        let tensors = [TensorDesc::new(TensorId::new(0), DType::F32, 1 << 36, 0, 0)];
        let plan = MemoryPlan::new(&tensors, &big).unwrap();
        assert!(matches!(
            Arenas::new(&device, &plan),
            Err(ArenaError::PlanNotValidOnDevice { .. })
        ));
    }
}
