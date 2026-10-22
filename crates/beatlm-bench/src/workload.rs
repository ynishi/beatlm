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
//! [`RenderOrder::After`] makes it wait: the render pipeline's vertex shader
//! reads the first word of **every buffer the slices can write**, each bound
//! as read-only storage ([`written_bind_group`], the stage
//! [`after_vertex`] generates), so the render pass reads what they wrote. The
//! synthetic slices all write this load's `sink`; the depth model's kernels
//! write the arenas of its plan, and `Net::written` (`slots.rs`) hands out
//! all of them. Each host chooses its default: the bench keeps `free`
//! (`main.rs`, "What it measures", says why), the browser host uses
//! `after` (`beatlm-web`'s crate documentation has its measurements).
//!
//! Every arena, rather than only those a frame's slices write or the one the
//! last slice wrote, because it is the one set fixed by the plan: one bind
//! group built before the loop, no choice in the frame and nothing allocated
//! there, as for every other binding. A read waits only for passes that wrote
//! its buffer, so an arena no slice of the frame wrote adds a wait only for
//! the earlier writes to it, every one encoded before this frame's slices,
//! and never for a slice of this frame. The last-written arena
//! alone would order the render after the last slice only, and after the
//! others only if every slice is ordered before the last, which holds for a
//! chain of kernels each reading the one before but is not a rule the
//! scheduler or WebGPU keeps for a frame's slices, so it is not relied on.
//! What bounds it is the device: a stage binds at most
//! `maxStorageBuffersPerShaderStage` storage buffers, 8 in the WebGPU
//! defaults, which `beatlm::gpu::required_limits` does not raise and the
//! bench's device does not either; [`written_fits`] refuses more. Each
//! binding is the arena's first four bytes, so no binding-size limit
//! applies. The plans of the browser runs measured for this order fit, and
//! `beatlm-web`'s `depth.rs` tests that they do and that a longer one does
//! not; a plan that does not fit is refused, before the browser fetches its
//! weights, never cut down.
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
//!   done; per buffer, so with several arenas bound it waits for the writers
//!   of each. wgpu-hal samples a render pass's begin timestamp at the start of
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
//! hazard too. WebGPU orders that write after the vertex stage's read, not
//! after the render's end; the `pass_overlap_next_frame` line
//! (`pass_overlap.rs`) measures where the next frame's first slice began
//! against this render's end. Splitting the render into a submission of its
//! own was not chosen: wgpu documents a submission only as committed for execution in
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
//!
//! With every arena bound, plans of two held it the same way (same machine,
//! wgpu and day, `--submit free --frames 300 --render-iters 50
//! --render-order after`, one run each): 266 px at `--latency 25` (26 slots,
//! arenas of 134 217 728 and 10 631 172 bytes) and 518 px at `--latency 10`
//! (11 slots, 134 217 728 and 116 675 332 bytes) each had 0 of 300 frames
//! overlapping and all 300 renders after the last slice, with
//! `gpu_frame_us` p50 14 037 and 14 307 µs. The first delivered 55 and
//! missed none; the second missed all 27 deadlines it reached, as it did
//! under `free` (13 206 µs p50), its job (205 ms calibrated) being longer
//! than the 167 ms promised.

use std::fmt::Write as _;

use anyhow::{Result, bail};
use beatlm_core::device::DeviceProfile;
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
///
/// It has no `Default`: the hosts default to different orders, each for its
/// own measurements, and each names its own (the bench's `Args`, the page's
/// `DEFAULT_RENDER_ORDER`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderOrder {
    /// The render reads nothing the slices write, and may run beside them.
    Free,
    /// The render's vertex shader reads a word of every buffer the slices
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

/// The most storage buffers one shader stage binds on a device created with
/// the WebGPU default limits: `maxStorageBuffersPerShaderStage` of the
/// baseline profile. Both hosts create their devices with this limit at its
/// default, so it is how many buffers [`RenderOrder::After`] can read before
/// a device exists.
pub const DEFAULT_STAGE_STORAGE_BUFFERS: u32 =
    DeviceProfile::webgpu_baseline().max_storage_buffers_per_shader_stage;

/// The bytes of each buffer [`RenderOrder::After`] binds: its first word.
const WRITTEN_WORD_BYTES: u64 = 4;

/// Whether [`RenderOrder::After`] can read `buffers` buffers, those the
/// frame's slices can write, in the render's vertex stage, on a device whose
/// stage binds at most `limit` storage buffers
/// ([`DEFAULT_STAGE_STORAGE_BUFFERS`] before the device exists, the device's
/// own limit after).
///
/// # Errors
///
/// No buffer, which would order nothing, or more than `limit`.
pub fn written_fits(buffers: usize, limit: u32) -> Result<()> {
    if buffers == 0 {
        bail!("the slices write no buffer for the render to read");
    }
    if u32::try_from(buffers).map_or(true, |n| n > limit) {
        bail!(
            "the render is ordered after the slices by reading each of the {buffers} buffers \
             they write in its vertex stage, and the device binds at most {limit} storage \
             buffers to one stage (maxStorageBuffersPerShaderStage)"
        );
    }
    Ok(())
}

/// WGSL for a vertex stage named `entry` that reads the first word of each
/// of `buffers` buffers, bound read-only at group 1, bindings 0 to
/// `buffers - 1`, and otherwise draws `triangle(vi)`: appended to a shader
/// that defines `fn triangle(vi: u32) -> vec4<f32>`, it is that shader's
/// [`RenderOrder::After`] vertex stage.
///
/// The words, combined so that none is dead, move the vertex by at most
/// 2^24 * 1e-30 of clip space, which no rasteriser shows, and from values the
/// compiler cannot know, so no read is removed. The stage is generated
/// because a stage binds a fixed number of buffers and a plan's arena count
/// is known only once it is made; the pipeline's layout is derived from it.
#[must_use]
pub fn after_vertex(buffers: usize, entry: &str) -> String {
    let mut wgsl = String::from(
        "\n// Generated by `workload::after_vertex`: the first word of every buffer the\n\
         // frame's slices write, read so that this stage runs after them.\n",
    );
    for i in 0..buffers {
        // A `String` takes every write.
        let _ = writeln!(
            wgsl,
            "@group(1) @binding({i}) var<storage, read> written_{i}: array<u32>;"
        );
    }
    // [`written_fits`] refuses none before any caller gets here, and a stage
    // of none would have no group 1 for [`written_bind_group`] to fill.
    debug_assert!(buffers > 0, "the render reads at least one buffer");
    let word = (0..buffers)
        .map(|i| format!("written_{i}[0]"))
        .collect::<Vec<_>>()
        .join(" ^ ");
    let _ = write!(
        wgsl,
        "\n@vertex\n\
         fn {entry}(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {{\n\
         \x20   let word = {word};\n\
         \x20   let p = triangle(vi);\n\
         \x20   return vec4<f32>(p.x + f32(word >> 8u) * 1e-30, p.y, p.z, p.w);\n\
         }}\n"
    );
    wgsl
}

/// The bind group group 1 of a render pipeline built for
/// [`RenderOrder::After`] over [`after_vertex`] takes: binding `i`, the first
/// word of `written[i]`, as read-only storage. Its layout is the pipeline's
/// own, derived from the shader.
pub fn written_bind_group(
    device: &wgpu::Device,
    pipeline: &wgpu::RenderPipeline,
    written: &[&wgpu::Buffer],
) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry<'_>> = written
        .iter()
        .zip(0_u32..)
        .map(|(buffer, binding)| wgpu::BindGroupEntry {
            binding,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer,
                offset: 0,
                size: wgpu::BufferSize::new(WRITTEN_WORD_BYTES),
            }),
        })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("beatlm render-after binding"),
        layout: &pipeline.get_bind_group_layout(1),
        entries: &entries,
    })
}

/// `source` with the [`RenderOrder::After`] vertex stage `entry` over
/// `buffers` buffers appended, compiled.
pub fn after_module(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    buffers: usize,
    entry: &str,
) -> wgpu::ShaderModule {
    let wgsl = format!("{source}{}", after_vertex(buffers, entry));
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(wgsl.into()),
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
/// under [`RenderOrder::After`] group 1, the buffers the slices write, which
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
    /// write `written`, every buffer the frame's slices can write, or after
    /// this load's own slices when `written` is empty; the pipeline it needs
    /// is built here, once, so the frames allocate nothing.
    ///
    /// # Errors
    ///
    /// [`RenderOrder::After`] over more buffers than the device binds to one
    /// stage ([`written_fits`]); the order is then left as it was.
    pub fn order_render(
        &mut self,
        device: &wgpu::Device,
        order: RenderOrder,
        written: &[&wgpu::Buffer],
    ) -> Result<()> {
        let own = [&self.sink];
        let written = if written.is_empty() {
            &own[..]
        } else {
            written
        };
        let (pipeline, written) = match order {
            // Already the pipeline `new` built.
            RenderOrder::Free if self.render.written.is_none() => return Ok(()),
            RenderOrder::Free => (render_pipeline(device, &self.module, "render_vs"), None),
            RenderOrder::After => {
                written_fits(
                    written.len(),
                    device.limits().max_storage_buffers_per_shader_stage,
                )?;
                let module = after_module(
                    device,
                    "beatlm-bench shaders, render after",
                    include_str!("shaders.wgsl"),
                    written.len(),
                    "render_vs_after",
                );
                let pipeline = render_pipeline(device, &module, "render_vs_after");
                let group = written_bind_group(device, &pipeline, written);
                (pipeline, Some(group))
            }
        };
        self.render = Render {
            params: params_bind_group(device, &pipeline, &self.params),
            written,
            pipeline,
        };
        Ok(())
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

/// The render pipeline with vertex entry point `vertex` of `module`:
/// `render_vs` of `shaders.wgsl`, or `render_vs_after` of the module
/// [`after_module`] makes from it for [`RenderOrder::After`]. Its layout is
/// derived from the shader, so each pipeline's bind groups are made from its
/// own.
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
        for order in RenderOrder::ALL {
            assert_eq!(RenderOrder::from_name(order.name()), Some(order));
        }
        assert_eq!(RenderOrder::from_name("after"), Some(RenderOrder::After));
        assert_eq!(RenderOrder::from_name("before"), None);
        assert_eq!(RenderOrder::from_name("After"), None);
    }

    /// The shaders are compiled by the device that runs them, so they are
    /// validated here, by naga with every check on and no optional feature,
    /// as the generated kernels are; with `render_vs_after` over every count
    /// of buffers a device with the default limits can bind.
    #[test]
    fn the_synthetic_shaders_validate_without_optional_features() {
        let source = include_str!("shaders.wgsl");
        let features = beatlm_core::device::DeviceFeatures::default();
        beatlm::codegen::validate("synthetic", source, features).unwrap();
        for buffers in 1..=DEFAULT_STAGE_STORAGE_BUFFERS as usize {
            let wgsl = format!("{source}{}", after_vertex(buffers, "render_vs_after"));
            beatlm::codegen::validate("synthetic, render after", &wgsl, features)
                .unwrap_or_else(|e| panic!("{buffers} buffers: {e}"));
        }
    }

    #[test]
    fn the_after_stage_reads_the_first_word_of_every_buffer() {
        let wgsl = after_vertex(3, "v");
        for i in 0..3 {
            assert!(wgsl.contains(&format!(
                "@group(1) @binding({i}) var<storage, read> written_{i}: array<u32>;"
            )));
        }
        assert!(!wgsl.contains("binding(3)"));
        assert!(wgsl.contains("let word = written_0[0] ^ written_1[0] ^ written_2[0];"));
        assert!(wgsl.contains("fn v(@builtin(vertex_index) vi: u32)"));
    }

    /// The baseline profile's limit is the one both hosts' devices get: wgpu's
    /// default limits, which `beatlm::gpu::required_limits` starts from and
    /// does not raise for this one.
    #[test]
    fn the_stage_limit_before_a_device_is_the_default_devices() {
        assert_eq!(DEFAULT_STAGE_STORAGE_BUFFERS, 8);
        assert_eq!(
            wgpu::Limits::default().max_storage_buffers_per_shader_stage,
            DEFAULT_STAGE_STORAGE_BUFFERS
        );
        let plan = beatlm_core::memory::MemoryPlan::new(
            &[beatlm_core::memory::TensorDesc::new(
                beatlm_core::memory::TensorId::new(0),
                beatlm_core::memory::DType::F32,
                1 << 28,
                0,
                0,
            )],
            &DeviceProfile {
                max_storage_buffer_binding_size: 1 << 31,
                max_buffer_size: 1 << 31,
                ..DeviceProfile::webgpu_baseline()
            },
        )
        .unwrap();
        let adapter = wgpu::Limits {
            max_storage_buffer_binding_size: 1 << 31,
            max_buffer_size: 1 << 31,
            max_storage_buffers_per_shader_stage: 31,
            ..wgpu::Limits::default()
        };
        let required = beatlm::gpu::required_limits(&plan, &adapter).unwrap();
        assert_eq!(
            required.max_storage_buffers_per_shader_stage,
            DEFAULT_STAGE_STORAGE_BUFFERS
        );
    }

    #[test]
    fn the_render_reads_at_most_as_many_buffers_as_a_stage_binds() {
        for buffers in 1..=8 {
            assert!(written_fits(buffers, 8).is_ok(), "{buffers}");
        }
        let refused = written_fits(9, 8).unwrap_err().to_string();
        assert!(refused.contains("9 buffers"), "{refused}");
        assert!(refused.contains("at most 8"), "{refused}");
        assert!(
            refused.contains("maxStorageBuffersPerShaderStage"),
            "{refused}"
        );
        assert!(written_fits(0, 8).is_err());
        assert!(written_fits(1, 0).is_err());
        assert!(written_fits(31, 31).is_ok());
    }
}
