//! A headless device: an adapter, a device and a queue, and no window.

use anyhow::{Context, Result};

/// The GPU the benchmark runs on.
pub struct Gpu {
    /// The device that owns every resource.
    pub device: wgpu::Device,
    /// The queue the slices and the render pass share.
    pub queue: wgpu::Queue,
    /// Which adapter and backend were chosen, for the report.
    pub info: wgpu::AdapterInfo,
    /// The adapter the device came from, so a second device can share it.
    pub adapter: wgpu::Adapter,
}

impl Gpu {
    /// Requests the high-performance adapter and a device with
    /// [`wgpu::Features::TIMESTAMP_QUERY`].
    ///
    /// # Errors
    ///
    /// No adapter, or an adapter without timestamp queries: the benchmark
    /// measures GPU time and has nothing to report without them.
    pub fn open() -> Result<Self> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .context("no GPU adapter available")?;
        let info = adapter.get_info();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("beatlm-bench"),
            required_features: wgpu::Features::TIMESTAMP_QUERY,
            ..Default::default()
        }))
        .with_context(|| {
            format!(
                "adapter {} ({}) does not provide TIMESTAMP_QUERY",
                info.name, info.backend
            )
        })?;
        Ok(Self {
            device,
            queue,
            info,
            adapter,
        })
    }
}
