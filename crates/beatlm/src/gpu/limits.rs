//! The limits a device must be created with for a memory plan to be realised
//! on it.
//!
//! A device has the WebGPU default limits unless more is requested when it is
//! created, whatever the adapter could give; a browser grants what was asked
//! for and nothing more. A plan made against a profile larger than
//! [`DeviceProfile::webgpu_baseline`](beatlm_core::device::DeviceProfile::webgpu_baseline)
//! therefore needs its limits requested before the device exists, and
//! [`Arenas::new`](super::Arenas::new) refuses it on a device created with the
//! defaults.
//!
//! [`required_limits`] answers that from the plan and the adapter. It raises a
//! default only where the plan needs more, and only to what the plan needs,
//! so the request is the same on every adapter that can hold the plan, and a
//! device's limits say what the host relies on rather than what one adapter
//! happened to offer. A plan the adapter cannot hold is refused with the limit,
//! the bytes it needs and the bytes the adapter has: a plan is never cut down
//! to fit, because a tensor is never split across arenas
//! ([`beatlm_core::memory`] says why).

use beatlm_core::memory::MemoryPlan;
use thiserror::Error;

use super::profile::profile;

/// Why a plan cannot be realised on any device of an adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum LimitsError {
    /// An arena of the plan is larger than the adapter allows for one buffer
    /// or one storage binding.
    #[error(
        "the memory plan needs {limit} of at least {needed} bytes, and the adapter provides {available}"
    )]
    AdapterTooSmall {
        /// The WebGPU name of the limit, as `wgpu::Limits` spells it.
        limit: &'static str,
        /// The largest arena of the plan, in bytes.
        needed: u64,
        /// What the adapter reports for the limit, in bytes.
        available: u64,
    },
    /// The plan's offsets are aligned for a storage offset alignment finer
    /// than the adapter's, so some placement cannot be bound there.
    #[error(
        "the memory plan's offsets need a finer storage offset alignment than the adapter's {available} bytes"
    )]
    Alignment {
        /// The adapter's `min_storage_buffer_offset_alignment`, in bytes.
        available: u32,
    },
}

/// The limits to create a device with so that `plan` can be realised on it,
/// for an adapter reporting `adapter`.
///
/// They are [`wgpu::Limits::default`], the WebGPU defaults every adapter
/// provides, with the storage binding size and the buffer size raised to the
/// plan's largest arena where that is above the default, and the storage
/// offset alignment lowered to the adapter's when the plan was made for a
/// finer one. Every other limit stays at its default.
///
/// # Errors
///
/// [`LimitsError::AdapterTooSmall`] when an arena is larger than the adapter's
/// storage binding or buffer size; [`LimitsError::Alignment`] when the plan's
/// offsets are aligned for a finer alignment than the adapter has.
pub fn required_limits(
    plan: &MemoryPlan,
    adapter: &wgpu::Limits,
) -> Result<wgpu::Limits, LimitsError> {
    let mut limits = wgpu::Limits::default();
    let largest = plan.arenas().iter().copied().max().unwrap_or(0);
    limits.max_storage_buffer_binding_size = raised(
        "max_storage_buffer_binding_size",
        limits.max_storage_buffer_binding_size,
        adapter.max_storage_buffer_binding_size,
        largest,
    )?;
    limits.max_buffer_size = raised(
        "max_buffer_size",
        limits.max_buffer_size,
        adapter.max_buffer_size,
        largest,
    )?;
    if !plan.valid_on(&profile(&limits, wgpu::Features::empty())) {
        // Every arena fits by now, so what is left is the alignment: the plan
        // was made for one finer than the default.
        limits.min_storage_buffer_offset_alignment = adapter.min_storage_buffer_offset_alignment;
        if !plan.valid_on(&profile(&limits, wgpu::Features::empty())) {
            return Err(LimitsError::Alignment {
                available: adapter.min_storage_buffer_offset_alignment,
            });
        }
    }
    Ok(limits)
}

/// `default` when `needed` fits it, else `needed` when the adapter's
/// `available` covers it.
fn raised(
    limit: &'static str,
    default: u64,
    available: u64,
    needed: u64,
) -> Result<u64, LimitsError> {
    if needed <= default {
        Ok(default)
    } else if needed <= available {
        Ok(needed)
    } else {
        Err(LimitsError::AdapterTooSmall {
            limit,
            needed,
            available,
        })
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use beatlm_core::device::DeviceProfile;
    use beatlm_core::memory::{DType, TensorDesc, TensorId};

    /// A plan of one `f32` tensor of `bytes` bytes (a multiple of four), made
    /// against `profile`.
    fn one_tensor(bytes: u64, profile: &DeviceProfile) -> MemoryPlan {
        let tensors = [TensorDesc::new(
            TensorId::new(0),
            DType::F32,
            bytes / 4,
            0,
            0,
        )];
        MemoryPlan::new(&tensors, profile).unwrap()
    }

    /// A profile whose arenas may be as large as `bytes`.
    fn profile_of(bytes: u64) -> DeviceProfile {
        DeviceProfile {
            max_storage_buffer_binding_size: bytes,
            max_buffer_size: bytes,
            ..DeviceProfile::webgpu_baseline()
        }
    }

    /// An adapter reporting `bytes` for both size limits.
    fn adapter_of(bytes: u64) -> wgpu::Limits {
        wgpu::Limits {
            max_storage_buffer_binding_size: bytes,
            max_buffer_size: bytes,
            ..wgpu::Limits::default()
        }
    }

    #[test]
    fn a_plan_within_the_defaults_asks_for_nothing_more() {
        let plan = one_tensor(1 << 20, &DeviceProfile::webgpu_baseline());
        assert_eq!(
            required_limits(&plan, &wgpu::Limits::default()),
            Ok(wgpu::Limits::default())
        );
    }

    #[test]
    fn the_largest_arena_of_depth_anything_at_518_pixels_fits_the_default_binding() {
        // 134,217,600 bytes, 128 under the default 128 MiB binding.
        let plan = one_tensor(134_217_600, &profile_of(1 << 30));
        assert_eq!(plan.arenas(), [134_217_600]);
        assert_eq!(
            required_limits(&plan, &wgpu::Limits::default()),
            Ok(wgpu::Limits::default())
        );
    }

    #[test]
    fn an_arena_past_the_default_binding_raises_the_binding_to_exactly_its_size() {
        let arena = (128 << 20) + 256;
        let plan = one_tensor(arena, &profile_of(1 << 30));
        let limits = required_limits(&plan, &adapter_of(1 << 30)).unwrap();
        assert_eq!(limits.max_storage_buffer_binding_size, arena);
        // Still under the default buffer size, which stays.
        assert_eq!(
            limits.max_buffer_size,
            wgpu::Limits::default().max_buffer_size
        );
        assert_eq!(
            wgpu::Limits {
                max_storage_buffer_binding_size: wgpu::Limits::default()
                    .max_storage_buffer_binding_size,
                ..limits
            },
            wgpu::Limits::default(),
            "nothing else is raised"
        );
        assert!(plan.valid_on(&profile(&limits, wgpu::Features::empty())));
    }

    #[test]
    fn an_arena_past_the_default_buffer_raises_both_sizes() {
        let arena = (256 << 20) + 256;
        let plan = one_tensor(arena, &profile_of(1 << 30));
        let limits = required_limits(&plan, &adapter_of(1 << 30)).unwrap();
        assert_eq!(limits.max_storage_buffer_binding_size, arena);
        assert_eq!(limits.max_buffer_size, arena);
    }

    #[test]
    fn an_arena_larger_than_the_adapter_allows_is_refused_and_named() {
        let arena = (128 << 20) + 256;
        let plan = one_tensor(arena, &profile_of(1 << 30));
        let refused = required_limits(&plan, &wgpu::Limits::default()).unwrap_err();
        assert_eq!(
            refused,
            LimitsError::AdapterTooSmall {
                limit: "max_storage_buffer_binding_size",
                needed: arena,
                available: 128 << 20,
            }
        );
        assert_eq!(
            refused.to_string(),
            "the memory plan needs max_storage_buffer_binding_size of at least 134217984 bytes, \
             and the adapter provides 134217728"
        );
    }

    #[test]
    fn a_plan_for_a_finer_alignment_takes_the_adapters_or_is_refused() {
        let fine = DeviceProfile {
            min_storage_buffer_offset_alignment: 32,
            ..DeviceProfile::webgpu_baseline()
        };
        let tensors = [
            TensorDesc::new(TensorId::new(0), DType::F32, 9, 0, 1),
            TensorDesc::new(TensorId::new(1), DType::F32, 9, 1, 2),
        ];
        let plan = MemoryPlan::new(&tensors, &fine).unwrap();
        let adapter = wgpu::Limits {
            min_storage_buffer_offset_alignment: 32,
            ..wgpu::Limits::default()
        };
        let limits = required_limits(&plan, &adapter).unwrap();
        assert_eq!(limits.min_storage_buffer_offset_alignment, 32);
        assert_eq!(
            required_limits(&plan, &wgpu::Limits::default()),
            Err(LimitsError::Alignment { available: 256 })
        );
    }
}
