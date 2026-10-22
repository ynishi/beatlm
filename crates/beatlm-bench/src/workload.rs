//! The synthetic load: one compute kernel standing in for an inference slice
//! and one render pass standing in for the application's frame.
//!
//! Both kernels busy-loop for a configurable number of iterations, so a run can
//! be given any cost; the WGSL is in `shaders.wgsl`. The compute dispatch is
//! sized to fill the GPU, so a slice's time grows with its iteration count and
//! not with how few invocations it launches.
//!
//! The browser host (`beatlm-web`) compiles this file by path, so a browser run
//! and a native run time the same kernels at the same sizes and their reports
//! can be compared. That is why nothing here names a bench type, and why the
//! render target is the caller's: the bench draws into an offscreen texture,
//! the browser into its canvas, both of [`TARGET_FORMAT`] and
//! [`RENDER_WIDTH`] by [`RENDER_HEIGHT`].
//!
//! # Ordering the render pass after the slices
//!
//! A frame encodes its slices, then its render pass, in one command buffer,
//! and nothing makes the render wait for them: it reads nothing they write.
//! On an Apple M1 Pro both hosts ran it beside them (`pass_overlap.rs`).
//! [`RenderOrder::After`] makes it wait, as an opt-in for measuring what that
//! costs: the render pipeline's vertex shader reads one word of the buffer the
//! slices write, bound as read-only storage ([`written_bind_group`]), so the
//! render pass reads what they wrote. The synthetic slices all write this
//! load's `sink`; the depth model's kernels all write its plan's one arena
//! (`slots.rs`, `Net::written`).
//!
//! What that guarantees is WebGPU's rule for a read after a write: the
//! render's vertex shader sees what every earlier pass wrote to the buffer, so
//! no implementation may run that stage before those passes are done. How it
//! is kept differs:
//!
//! - wgpu natively, on Metal, records no barrier at all (`wgpu-hal` 30's
//!   `transition_buffers` is empty there) and creates buffers in Metal's
//!   default, tracked, hazard mode, so Metal itself delays a render encoder's
//!   stage that binds a buffer an earlier encoder wrote until that encoder is
//!   done. wgpu-hal samples a render pass's begin timestamp at the start of
//!   the vertex stage, so with the read in that stage the begin cannot come
//!   before the slices end. A read in the fragment stage only would leave the
//!   vertex stage, and that timestamp, free to start early.
//! - In Chrome it is Dawn's to keep the same rule. Before the switch, both
//!   hosts already showed it kept between the slices, which write and read
//!   the same buffers: no two consecutive compute passes' spans intersected
//!   (`pass_overlap_compute`), while the render's did.
//!
//! What it does not guarantee: anything about the render's fragment work
//! beyond that it starts after the vertex stage; anything in a frame without
//! slices, where the read still waits for whatever earlier submission last
//! wrote the buffer; and that the next frame's slices do not wait for this
//! render, since they write what it read and a write after a read is a
//! hazard too. Splitting the render into a submission of its own was not
//! chosen: wgpu documents a submission only as committed for execution in
//! order (on Metal, one more command buffer on the same queue), and orders
//! nothing that shares no resource. Calibration's render passes are always
//! the free ones, so the loop's estimates start from the same measurements
//! under either order.
//!
//! Natively it did what it promises. Apple M1 Pro, Metal, wgpu 30.0.1,
//! release, 2026-10-22, Depth Anything V2 Small, `--workload depth --size 266
//! --latency 5 --submit free --frames 600 --render-iters 50`, one run each,
//! back to back:
//!
//! | `--render-order` | overlapping frames (largest intersection) | render before / among / after the slices | overlapped slices, ratio p50 / mean | slice 3 / slice 6 mean, µs | `gpu_frame_us` p50 / p99 | `longest_dispatch_us` p50 / p99 | delivered / missed |
//! |---|---|---|---|---|---|---|---|
//! | `free` | 600 of 600 (1 204 208 ns) | 27 / 573 / 0 | 1214, 1.21 / 2.00 | 545 / 231 | 13121 / 14868 | 898 / 1788 | 119 / 0 |
//! | `after` | 0 of 600 (0 ns) | 0 / 0 / 600 | none | 546 / 229 | 13752 / 15363 | 859 / 1435 | 119 / 0 |
//!
//! Under `after` no render span touched a slice's, and every render began
//! once the last slice had ended. The slices that overlapped the render under
//! `free` measured long, but they were about 6 % of the measurements, so the
//! slice means and the delivery did not move; natively the order costs and
//! buys nothing measurable at this setting.

use wgpu::util::DeviceExt;

/// Width of the render target, in pixels.
pub const RENDER_WIDTH: u32 = 1280;
/// Height of the render target, in pixels.
pub const RENDER_HEIGHT: u32 = 720;
/// The render target's format, which the render pipeline is built for. A
/// WebGPU canvas can always be configured with it.
pub const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Loop iterations per slice-kernel invocation that put one slice at 0.8 to
/// 1.4 ms on an Apple M1 Pro (Metal); see the `beatlm-bench` crate
/// documentation.
pub const DEFAULT_SLICE_ITERS: u32 = 500;

/// Loop iterations per pixel that put the render pass at about 2 ms on an
/// Apple M1 Pro (Metal); see the `beatlm-bench` crate documentation.
pub const DEFAULT_RENDER_ITERS: u32 = 50;

const WORKGROUP_SIZE: u32 = 64;
const WORKGROUPS: u32 = 1024;

/// Whether a frame's render pass waits for its slices: the bench's
/// `--render-order`, the page's `render_order`. The module documentation,
/// "Ordering the render pass after the slices", says how and what it
/// guarantees.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RenderOrder {
    /// The render reads nothing the slices write, and may run beside them.
    #[default]
    Free,
    /// The render's vertex shader reads a word of the buffer the slices
    /// write, so it runs after them.
    After,
}

impl RenderOrder {
    /// Every order, as the options list them.
    pub const ALL: [Self; 2] = [Self::Free, Self::After];

    /// The spelling the options accept and the `config:` line prints.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Free => "free",
            Self::After => "after",
        }
    }

    /// The order spelled `text`, or `None`.
    #[must_use]
    pub fn from_name(text: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|order| order.name() == text)
    }
}

/// The bind group group 1 of a render pipeline built for
/// [`RenderOrder::After`] takes: binding 0, the first word of `written`, the
/// buffer the slices write, as read-only storage. Its layout is the
/// pipeline's own, derived from the shader.
pub fn written_bind_group(
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    written: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("beatlm render-after binding"),
        layout: &pipeline.get_bind_group_layout(1),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: written,
                offset: 0,
                size: wgpu::BufferSize::new(4),
            }),
        }],
    })
}

/// The pipelines and bindings of the synthetic load.
pub struct Workload {
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,
    /// The render pass of the order [`Workload::order_render`] last set.
    render: Render,
    module: wgpu::ShaderModule,
    params: wgpu::Buffer,
    sink: wgpu::Buffer,
}

/// A render pass's pipeline and bind groups: group 0, the parameters, and
/// under [`RenderOrder::After`] group 1, the buffer the slices write, which
/// that pipeline's vertex shader reads.
struct Render {
    pipeline: wgpu::RenderPipeline,
    params: wgpu::BindGroup,
    written: Option<wgpu::BindGroup>,
}

impl Workload {
    /// Builds the load with `slice_iters` loop iterations per slice invocation
    /// and `render_iters` per pixel.
    pub fn new(device: &wgpu::Device, slice_iters: u32, render_iters: u32) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("beatlm-bench shaders"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders.wgsl").into()),
        });

        let mut params = Vec::with_capacity(16);
        for word in [slice_iters, render_iters, 0, 0] {
            params.extend_from_slice(&word.to_le_bytes());
        }
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("beatlm-bench params"),
            contents: &params,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let sink = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("beatlm-bench sink"),
            size: u64::from(WORKGROUPS) * u64::from(WORKGROUP_SIZE) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("beatlm-bench slice"),
            layout: None,
            module: &module,
            entry_point: Some("slice_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let compute_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("beatlm-bench slice bindings"),
            layout: &compute_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sink.as_entire_binding(),
                },
            ],
        });

        let pipeline = render_pipeline(device, &module, "render_vs");
        let render = Render {
            params: params_bind_group(device, &pipeline, &params),
            written: None,
            pipeline,
        };

        Self {
            compute_pipeline,
            compute_bind_group,
            render,
            module,
            params,
            sink,
        }
    }

    /// Sets the order of the render pass to `order` for the frames encoded
    /// from now on. [`RenderOrder::After`] orders it after the passes that
    /// write `written`, or this load's own slices when `None`; the pipeline it
    /// needs is built here, once, so the frames allocate nothing.
    pub fn order_render(
        &mut self,
        device: &wgpu::Device,
        order: RenderOrder,
        written: Option<&wgpu::Buffer>,
    ) {
        let vertex = match order {
            // Already the pipeline `new` built.
            RenderOrder::Free if self.render.written.is_none() => return,
            RenderOrder::Free => "render_vs",
            RenderOrder::After => "render_vs_after",
        };
        let pipeline = render_pipeline(device, &self.module, vertex);
        self.render = Render {
            params: params_bind_group(device, &pipeline, &self.params),
            written: (order == RenderOrder::After)
                .then(|| written_bind_group(device, &pipeline, written.unwrap_or(&self.sink))),
            pipeline,
        };
    }

    /// Records one slice as its own compute pass.
    pub fn encode_slice(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("beatlm-bench slice"),
            timestamp_writes,
        });
        pass.set_pipeline(&self.compute_pipeline);
        pass.set_bind_group(0, &self.compute_bind_group, &[]);
        pass.dispatch_workgroups(WORKGROUPS, 1, 1);
    }

    /// Records the render pass: one full-screen triangle into `target`, a
    /// view of a [`TARGET_FORMAT`] texture of [`RENDER_WIDTH`] by
    /// [`RENDER_HEIGHT`] pixels.
    pub fn encode_render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("beatlm-bench render"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.render.pipeline);
        pass.set_bind_group(0, &self.render.params, &[]);
        if let Some(written) = &self.render.written {
            pass.set_bind_group(1, written, &[]);
        }
        pass.draw(0..3, 0..1);
    }
}

/// The render pipeline with vertex entry point `vertex`: `render_vs`, or
/// `render_vs_after` for [`RenderOrder::After`]. Its layout is derived from
/// the shader, so each pipeline's bind groups are made from its own.
fn render_pipeline(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    vertex: &str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("beatlm-bench render"),
        layout: None,
        vertex: wgpu::VertexState {
            module,
            entry_point: Some(vertex),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        fragment: Some(wgpu::FragmentState {
            module,
            entry_point: Some("render_fs"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(TARGET_FORMAT.into())],
        }),
        multiview_mask: None,
        cache: None,
    })
}

/// Group 0 of a render pipeline: the parameters.
fn params_bind_group(
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    params: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("beatlm-bench render bindings"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: params.as_entire_binding(),
        }],
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn a_render_order_is_named_as_the_options_spell_it() {
        assert_eq!(RenderOrder::default(), RenderOrder::Free);
        for order in RenderOrder::ALL {
            assert_eq!(RenderOrder::from_name(order.name()), Some(order));
        }
        assert_eq!(RenderOrder::from_name("after"), Some(RenderOrder::After));
        assert_eq!(RenderOrder::from_name("before"), None);
        assert_eq!(RenderOrder::from_name("After"), None);
    }

    /// The shaders are compiled by the device that runs them, so they are
    /// validated here, by naga with every check on and no optional feature,
    /// as the generated kernels are; `render_vs_after` included.
    #[test]
    fn the_synthetic_shaders_validate_without_optional_features() {
        beatlm::codegen::validate(
            "synthetic",
            include_str!("shaders.wgsl"),
            beatlm_core::device::DeviceFeatures::default(),
        )
        .unwrap();
    }
}
