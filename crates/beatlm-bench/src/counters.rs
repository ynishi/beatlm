//! wgpu's internal object counters, read around the frame loop to check the
//! rule that the steady state allocates nothing.
//!
//! The rule is stated once, in the core crate documentation, and realised by
//! `beatlm::gpu::Arenas`: every tensor lives in a buffer created before the
//! first frame. This module measures it. wgpu keeps counts of the objects it
//! creates when its `counters` feature is on (enabled for this crate only), and
//! `Device::get_internal_counters` reads them.
//!
//! What a reading means:
//!
//! - **A counter is a live count, not a creation count.** It goes up when an
//!   object is created and down when it is destroyed, so a buffer created and
//!   dropped between two readings is invisible. The loop's per-frame sampling
//!   narrows that gap to what happens inside one frame; it does not close it.
//! - **Backends fill what they fill.** Counters wgpu documents as optional may
//!   read zero on a backend that does not set them. A buffer count of zero
//!   after the arenas exist means the backend sets none, and [`Report::populated`]
//!   says so; `buffer_memory` and `memory_allocations` are set by some backends
//!   only, and the report prints them next to their readings so a constant zero
//!   is not mistaken for a measurement.

/// The counters the report prints, read at one moment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Snapshot {
    /// Live buffers.
    pub buffers: isize,
    /// Bytes of GPU memory attributed to buffers, where the backend sets it.
    pub buffer_memory: isize,
    /// Live GPU memory allocations, where the backend sets it.
    pub memory_allocations: isize,
    /// Live textures.
    pub textures: isize,
    /// Live bind groups.
    pub bind_groups: isize,
    /// Live compute pipelines.
    pub compute_pipelines: isize,
}

impl Snapshot {
    /// Reads the device's counters now.
    pub fn read(device: &wgpu::Device) -> Self {
        let hal = device.get_internal_counters().hal;
        Self {
            buffers: hal.buffers.read(),
            buffer_memory: hal.buffer_memory.read(),
            memory_allocations: hal.memory_allocations.read(),
            textures: hal.textures.read(),
            bind_groups: hal.bind_groups.read(),
            compute_pipelines: hal.compute_pipelines.read(),
        }
    }
}

/// What the counters did across the frame loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Report {
    /// The readings before the first frame.
    pub before: Snapshot,
    /// The readings after the last frame.
    pub after: Snapshot,
    /// The fewest live buffers at any frame boundary in the loop.
    pub min_buffers: isize,
    /// The most live buffers at any frame boundary in the loop.
    pub max_buffers: isize,
    /// Whether the backend sets the buffer counter at all.
    pub populated: bool,
}

/// Samples the counters across a loop: once before it, once at each frame
/// boundary, once after.
pub struct Watch {
    before: Snapshot,
    min_buffers: isize,
    max_buffers: isize,
}

impl Watch {
    /// Takes the reading before the first frame.
    pub fn start(device: &wgpu::Device) -> Self {
        let before = Snapshot::read(device);
        Self {
            before,
            min_buffers: before.buffers,
            max_buffers: before.buffers,
        }
    }

    /// Notes the live buffer count at a frame boundary.
    pub fn sample(&mut self, device: &wgpu::Device) {
        let buffers = device.get_internal_counters().hal.buffers.read();
        self.min_buffers = self.min_buffers.min(buffers);
        self.max_buffers = self.max_buffers.max(buffers);
    }

    /// Takes the reading after the last frame and closes the watch.
    pub fn finish(self, device: &wgpu::Device) -> Report {
        Report {
            before: self.before,
            after: Snapshot::read(device),
            min_buffers: self.min_buffers,
            max_buffers: self.max_buffers,
            populated: self.before.buffers > 0,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// The control for the loop's zero: a creation between two readings moves
    /// the counter, so a delta of zero in the loop is not a counter that never
    /// moves.
    #[test]
    #[ignore = "needs a GPU adapter"]
    fn creating_and_dropping_a_buffer_moves_the_live_count() {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .unwrap();
        let (device, _queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap();

        let before = Snapshot::read(&device);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 1024,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        assert_eq!(Snapshot::read(&device).buffers, before.buffers + 1);
        drop(buffer);
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        assert_eq!(Snapshot::read(&device).buffers, before.buffers);
    }
}
