//! Turning what wgpu reports about a device into the core's
//! [`DeviceProfile`].
//!
//! This is the one place wgpu's limits and feature vocabulary is read. The
//! core plans memory and picks kernel variants from the profile's plain values,
//! so a wgpu release that renames a limit changes this file and nothing in
//! [`beatlm_core`].
//!
//! The function takes limits and features, not a device or an adapter, so the
//! same translation serves both: pass an adapter's to ask what it could do,
//! pass a device's to ask what it was created with. A plan is realised on a
//! device, so [`Arenas::new`](super::Arenas::new) uses the device's.

use beatlm_core::device::{DeviceFeatures, DeviceProfile};

/// The profile of a device with these limits and features.
///
/// It is a free function because [`DeviceProfile`] belongs to the core, which
/// cannot name wgpu types, and an inherent method on it would have to; and a
/// translation from a foreign type is what this crate is for.
///
/// Every limit has the width wgpu reports it in, so no value is narrowed.
#[must_use]
pub fn profile(limits: &wgpu::Limits, features: wgpu::Features) -> DeviceProfile {
    DeviceProfile {
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        max_buffer_size: limits.max_buffer_size,
        min_storage_buffer_offset_alignment: limits.min_storage_buffer_offset_alignment,
        max_compute_workgroup_storage_size: limits.max_compute_workgroup_storage_size,
        max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
        max_compute_workgroup_size: [
            limits.max_compute_workgroup_size_x,
            limits.max_compute_workgroup_size_y,
            limits.max_compute_workgroup_size_z,
        ],
        max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        max_storage_buffers_per_shader_stage: limits.max_storage_buffers_per_shader_stage,
        features: DeviceFeatures {
            shader_f16: features.contains(wgpu::Features::SHADER_F16),
            subgroups: features.contains(wgpu::Features::SUBGROUP),
            timestamp_query: features.contains(wgpu::Features::TIMESTAMP_QUERY),
        },
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn wgpus_default_limits_translate_to_the_core_baseline() {
        // The two baselines are written independently, so this is the check
        // that they agree.
        assert_eq!(
            profile(&wgpu::Limits::default(), wgpu::Features::empty()),
            DeviceProfile::webgpu_baseline()
        );
    }

    #[test]
    fn features_are_read_one_by_one() {
        let p = profile(
            &wgpu::Limits::default(),
            wgpu::Features::SHADER_F16 | wgpu::Features::TIMESTAMP_QUERY,
        );
        assert_eq!(
            p.features,
            DeviceFeatures {
                shader_f16: true,
                subgroups: false,
                timestamp_query: true
            }
        );
        let p = profile(&wgpu::Limits::default(), wgpu::Features::SUBGROUP);
        assert!(p.features.subgroups && !p.features.shader_f16);
    }

    #[test]
    fn limits_are_carried_over_field_by_field() {
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: 1 << 30,
            max_buffer_size: 1 << 31,
            min_storage_buffer_offset_alignment: 32,
            max_compute_workgroup_storage_size: 32_768,
            max_compute_invocations_per_workgroup: 1024,
            max_compute_workgroup_size_x: 1024,
            max_compute_workgroup_size_y: 512,
            max_compute_workgroup_size_z: 128,
            max_compute_workgroups_per_dimension: 1 << 20,
            max_storage_buffers_per_shader_stage: 16,
            ..wgpu::Limits::default()
        };
        let p = profile(&limits, wgpu::Features::empty());
        assert_eq!(p.max_storage_buffer_binding_size, 1 << 30);
        assert_eq!(p.max_buffer_size, 1 << 31);
        assert_eq!(p.min_storage_buffer_offset_alignment, 32);
        assert_eq!(p.max_compute_workgroup_storage_size, 32_768);
        assert_eq!(p.max_compute_invocations_per_workgroup, 1024);
        assert_eq!(p.max_compute_workgroup_size, [1024, 512, 128]);
        assert_eq!(p.max_compute_workgroups_per_dimension, 1 << 20);
        assert_eq!(p.max_storage_buffers_per_shader_stage, 16);
    }
}
