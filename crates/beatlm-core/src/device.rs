//! What the core knows about an adapter: its limits and features as plain
//! values.
//!
//! The memory plan and, later, kernel generation decide from numbers an adapter
//! reports: how large a storage binding may be, how finely an offset must be
//! aligned, how much workgroup memory a kernel may use. Those numbers arrive
//! here as a [`DeviceProfile`] and nothing in the core names the API that
//! reported them, which is what keeps the core free of wgpu and lets a plan be
//! made, stored and replayed without a device. Translating an adapter into a
//! profile is the umbrella crate's job.
//!
//! # Rules
//!
//! - **The baseline is valid everywhere.** [`DeviceProfile::webgpu_baseline`]
//!   holds the default limits every WebGPU implementation must provide and no
//!   optional feature. A plan made against it is valid on every adapter, because
//!   no adapter reports a limit below its baseline (the specification defines
//!   each default as the worst value a conforming implementation may report).
//!   [`MemoryPlan::valid_on`](crate::memory::MemoryPlan::valid_on) states the
//!   check for a plan made against a different profile.
//! - **A profile is values, not capabilities.** Whether a feature is *enabled*
//!   on the device is what the profile says; whether the adapter *could* enable
//!   it is not the profile's business, so a kernel variant chosen from a profile
//!   runs on the device that produced it.

/// The optional features a device was created with.
///
/// Only features that change what the core decides are listed; each one is
/// something a kernel variant or a timing path branches on. Everything absent
/// is `false`, so a default value is the baseline's "no optional features".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct DeviceFeatures {
    /// 16-bit floats in shaders. Kernels using `f16` need it; weights stored
    /// as [`DType::F16`](crate::memory::DType::F16) do not, because storage is
    /// just bytes.
    pub shader_f16: bool,
    /// Subgroup operations in shaders. Exposed on native backends only, so a
    /// kernel using them always has a variant without.
    pub subgroups: bool,
    /// Timestamp queries inside passes, which the slice timer needs.
    pub timestamp_query: bool,
}

/// An adapter's limits and enabled features, as plain values.
///
/// Field names and units follow the WebGPU specification's limit names, so a
/// number can be checked against the specification without a translation
/// table. Sizes are in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DeviceProfile {
    /// The largest range one storage binding may cover. It bounds the size of
    /// a memory arena, because an arena is bound whole.
    pub max_storage_buffer_binding_size: u64,
    /// The largest buffer the device can create; bounds an arena together with
    /// the binding size.
    pub max_buffer_size: u64,
    /// The alignment a storage binding's offset must have; every placement in
    /// a memory plan is a multiple of it.
    pub min_storage_buffer_offset_alignment: u32,
    /// The workgroup memory one dispatch may declare, which bounds a kernel's
    /// tile size.
    pub max_compute_workgroup_storage_size: u32,
    /// The largest number of invocations in one workgroup (the product of the
    /// workgroup size's three dimensions).
    pub max_compute_invocations_per_workgroup: u32,
    /// The largest workgroup size in x, y and z.
    pub max_compute_workgroup_size: [u32; 3],
    /// The largest workgroup count in any one dimension of a dispatch.
    pub max_compute_workgroups_per_dimension: u32,
    /// How many storage buffers one shader stage may bind, which bounds a
    /// kernel's number of distinct tensor bindings.
    pub max_storage_buffers_per_shader_stage: u32,
    /// The optional features enabled on the device.
    pub features: DeviceFeatures,
}

impl DeviceProfile {
    /// The default WebGPU limits and no optional feature.
    ///
    /// A plan made against this profile is valid on every adapter: the values
    /// are the specification's defaults, which no conforming adapter reports
    /// below. It is the profile to plan against when the target device is not
    /// known, and the one a plan stored for a browser should use.
    #[must_use]
    pub const fn webgpu_baseline() -> Self {
        Self {
            max_storage_buffer_binding_size: 128 << 20,
            max_buffer_size: 256 << 20,
            min_storage_buffer_offset_alignment: 256,
            max_compute_workgroup_storage_size: 16_384,
            max_compute_invocations_per_workgroup: 256,
            max_compute_workgroup_size: [256, 256, 64],
            max_compute_workgroups_per_dimension: 65_535,
            max_storage_buffers_per_shader_stage: 8,
            features: DeviceFeatures {
                shader_f16: false,
                subgroups: false,
                timestamp_query: false,
            },
        }
    }

    /// The largest size one memory arena may have: the smaller of the largest
    /// storage binding and the largest buffer, since an arena is one buffer
    /// bound whole.
    #[must_use]
    pub const fn max_arena_size(&self) -> u64 {
        if self.max_storage_buffer_binding_size < self.max_buffer_size {
            self.max_storage_buffer_binding_size
        } else {
            self.max_buffer_size
        }
    }
}

impl Default for DeviceProfile {
    /// The baseline, so a profile nobody chose is the one valid everywhere.
    fn default() -> Self {
        Self::webgpu_baseline()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn the_baseline_is_the_webgpu_default_limits_with_no_features() {
        let b = DeviceProfile::webgpu_baseline();
        assert_eq!(b.max_storage_buffer_binding_size, 128 * 1024 * 1024);
        assert_eq!(b.max_buffer_size, 256 * 1024 * 1024);
        assert_eq!(b.min_storage_buffer_offset_alignment, 256);
        assert_eq!(b.max_compute_workgroup_storage_size, 16 * 1024);
        assert_eq!(b.max_compute_invocations_per_workgroup, 256);
        assert_eq!(b.max_compute_workgroup_size, [256, 256, 64]);
        assert_eq!(b.max_compute_workgroups_per_dimension, 65_535);
        assert_eq!(b.max_storage_buffers_per_shader_stage, 8);
        assert_eq!(b.features, DeviceFeatures::default());
        assert_eq!(DeviceProfile::default(), b);
    }

    #[test]
    fn an_arena_is_bounded_by_the_smaller_of_binding_and_buffer_size() {
        let mut p = DeviceProfile::webgpu_baseline();
        assert_eq!(p.max_arena_size(), 128 << 20);
        p.max_buffer_size = 64 << 20;
        assert_eq!(p.max_arena_size(), 64 << 20);
    }
}
