//! The synthetic load: one compute kernel standing in for an inference slice
//! and one render pass standing in for the application's frame.
//!
//! Both kernels busy-loop for a configurable number of iterations, so a run can
//! be given any cost; the WGSL is in `shaders.wgsl`. The compute dispatch is
//! sized to fill the GPU, so a slice's time grows with its iteration count and
//! not with how few invocations it launches.

use wgpu::util::DeviceExt;

/// Width of the offscreen render target, in pixels.
pub const RENDER_WIDTH: u32 = 1280;
/// Height of the offscreen render target, in pixels.
pub const RENDER_HEIGHT: u32 = 720;

const WORKGROUP_SIZE: u32 = 64;
const WORKGROUPS: u32 = 1024;
const TARGET_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// The pipelines, bindings and targets of the synthetic load.
pub struct Workload {
    compute_pipeline: wgpu::ComputePipeline,
    compute_bind_group: wgpu::BindGroup,
    render_pipeline: wgpu::RenderPipeline,
    render_bind_group: wgpu::BindGroup,
    target_view: wgpu::TextureView,
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

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("beatlm-bench render"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("render_vs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("render_fs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(TARGET_FORMAT.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("beatlm-bench render bindings"),
            layout: &render_pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params.as_entire_binding(),
            }],
        });

        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("beatlm-bench target"),
            size: wgpu::Extent3d {
                width: RENDER_WIDTH,
                height: RENDER_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TARGET_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

        Self {
            compute_pipeline,
            compute_bind_group,
            render_pipeline,
            render_bind_group,
            target_view,
        }
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

    /// Records the render pass: one full-screen triangle into the offscreen
    /// target.
    pub fn encode_render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("beatlm-bench render"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.target_view,
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
        pass.set_pipeline(&self.render_pipeline);
        pass.set_bind_group(0, &self.render_bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}
