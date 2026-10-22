//! The depth model's render pass: the source and the delivered depth.
//!
//! **The screen is part of the delivery promise.** A result may appear only
//! from the frame that delivers it, so the display holds one depth, the one
//! delivered last, in a buffer of its own, and copies a job's depth into it in
//! the frame the scheduler delivers the job ([`Display::deliver`]): after that
//! frame's slices, since a job is complete once its last slice has been handed
//! out in that frame or an earlier one, and before its render pass. Until the
//! first delivery the pass draws the source and no depth. A copy made when a
//! job's last slice was encoded, into a buffer per output slot, put jobs on
//! screen early: once the jobs after a delivered one missed, the screen still
//! showed that slot's buffer, and the next job on the same slot overwrote it
//! when its last slice ran, frames (on beat, a beat or more) before its own
//! delivery, with nothing in the report counting it early. The slot copied
//! from is still reserved when its job is delivered: a slot is handed to
//! another job only after the plan that delivers or misses its job
//! ([`crate::depth::delivered_now`] says why under each policy).
//!
//! The copy is `side * side` numbers recorded between passes, so no timestamp
//! covers it: neither the render's measured time nor `gpu_frame_us` counts
//! it, as neither counted the per-slot copy before it. It does make the
//! render pass of a delivery frame read a buffer written in its own frame,
//! from an arena that frame's slices may write, so in that frame the render
//! can be ordered behind the slices. The render was measured not to carry the
//! slices when it read no such buffer (p50 1 245 µs, and 1 573 µs in a later
//! run, with slices in its frame against 1 901 µs without; Chrome 154,
//! WebGPU, Apple M1 Pro, 120 Hz, Depth Anything V2 Small at 266 pixels,
//! [`crate::render_cost`]), and a delivery frame comes at most once per job,
//! so a wait there would shrink the budget of the frame after it and not the
//! median of recent renders the frames to come are forecast with. The
//! report's `gpu_render_alone_us` and `gpu_render_with_slices_us` show the
//! render's measurement without and with slices in its frame, and would show
//! it if that changed. That paragraph is about `render_order=free`.
//!
//! Under `render_order=after`, the page's default ([`crate::config`]), the
//! pass waits for the slices in every frame, not only in a delivery frame:
//! its vertex stage reads the first word of every arena of the plan, the
//! buffers the slices write, through `display_vs_after`, which the bench's
//! `workload::after_vertex` appends to `display.wgsl` with one binding per
//! arena, and one bind group made here. The bench's `workload.rs` ("Ordering
//! the render pass after the slices") says why every arena, what bounds
//! their number, and what the order guarantees; the page refuses a plan of
//! too many before it fetches the weights (`crate::depth::check_render_order`),
//! and this checks the split model's plan again.
//!
//! Both bind groups, one drawing the depth and one drawn before the first
//! delivery, are made when the display is built, so choosing what to show in
//! a frame creates nothing and uploads nothing, as the rest of the frame
//! loop. The source is a buffer of its own, the input every slot holds,
//! uploaded once. The depth is mapped to grey between the smallest and the
//! largest depth the check's run computed, fixed for the run. The shader
//! (`display.wgsl`) carries the synthetic render pass's per-pixel loop, so
//! the pass costs what the bench's render pass does beside the depth model.

use anyhow::{Context, Result};
use beatlm::load::depth_anything_v2::reference::{MEAN, STD};
use wgpu::util::DeviceExt;

use crate::slots::{Net, Numbers};
use crate::workload::{RenderOrder, TARGET_FORMAT, after_module, written_bind_group, written_fits};

/// The render pass that shows the source and the depth delivered last.
pub struct Display {
    pipeline: wgpu::RenderPipeline,
    /// Per slot, where its output is in the arenas.
    outputs: Vec<Output>,
    /// The depth delivered last, which the pass draws.
    depth: wgpu::Buffer,
    /// The bytes of one depth.
    depth_bytes: u64,
    /// Draws the source and `depth`.
    shown: wgpu::BindGroup,
    /// Drawn before the first delivery: the source, and no depth.
    blank: wgpu::BindGroup,
    /// Under [`RenderOrder::After`], group 1: every arena of the plan, the
    /// buffers the slices write, which the vertex shader reads
    /// (`workload.rs` says why all of them).
    written: Option<wgpu::BindGroup>,
}

impl Display {
    /// Builds the pass for `net`, realised with `numbers`, whose input side is
    /// `side`, mapping depths from `range.0` to `range.1` to black to white,
    /// with `render_iters` loop iterations per pixel, in `order` against the
    /// frame's slices.
    ///
    /// # Errors
    ///
    /// A slot whose output has no placement in the plan or a smaller one than
    /// a depth of `side` pixels, an input of another size, or
    /// [`RenderOrder::After`] for a plan of more arenas than the device binds
    /// to one stage ([`written_fits`]), which the page refuses before the
    /// weights from the unsplit plan and this checks again for the split one.
    pub fn new(
        device: &wgpu::Device,
        net: &Net,
        numbers: &Numbers,
        side: u32,
        range: (f32, f32),
        render_iters: u32,
        order: RenderOrder,
    ) -> Result<Self> {
        const SOURCE: &str = include_str!("display.wgsl");
        let (written, module, vertex) = match order {
            RenderOrder::Free => (
                Vec::new(),
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("beatlm-web display"),
                    source: wgpu::ShaderSource::Wgsl(SOURCE.into()),
                }),
                "display_vs",
            ),
            RenderOrder::After => {
                let written = net.written()?;
                written_fits(
                    written.len(),
                    device.limits().max_storage_buffers_per_shader_stage,
                )
                .context("render_order=after cannot be honoured for the split model's plan")?;
                let module = after_module(
                    device,
                    "beatlm-web display, render after",
                    SOURCE,
                    written.len(),
                    "display_vs_after",
                );
                (written, module, "display_vs_after")
            }
        };
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("beatlm-web display"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some(vertex),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("display_fs"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(TARGET_FORMAT.into())],
            }),
            multiview_mask: None,
            cache: None,
        });
        let params = |shown: u32| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("beatlm-web display params"),
                contents: &params_bytes(side, render_iters, shown, range),
                usage: wgpu::BufferUsages::UNIFORM,
            })
        };
        let shown_params = params(1);
        let blank_params = params(0);
        let pixels = u64::from(side) * u64::from(side);
        let depth_bytes = pixels * 4;
        let input: Vec<u8> = (numbers.slot_input)(0)
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        anyhow::ensure!(
            u64::try_from(input.len()).ok() == Some(pixels * 3 * 4),
            "the input is not an image of {side} pixels"
        );
        // Every slot holds this same input, so one copy is the source of all.
        let image = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("beatlm-web display image"),
            contents: &input,
            usage: wgpu::BufferUsages::STORAGE,
        });
        let outputs = (0..net.slots())
            .map(|slot| {
                let id = numbers.output_of(slot);
                let placement = net
                    .plan()
                    .placement(id)
                    .filter(|p| p.size >= depth_bytes)
                    .with_context(|| format!("the depth of slot {slot} has no placement"))?;
                let arena = net
                    .arenas()
                    .buffer(placement.arena)
                    .with_context(|| format!("the depth of slot {slot} names no arena"))?
                    .clone();
                Ok(Output {
                    arena,
                    offset: placement.offset,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        anyhow::ensure!(!outputs.is_empty(), "the network has no slots");
        let depth = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("beatlm-web display depth"),
            size: depth_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let layout = pipeline.get_bind_group_layout(0);
        let shown = bind_group(device, &layout, &shown_params, &image, &depth);
        let blank = bind_group(device, &layout, &blank_params, &image, &depth);
        let written = match order {
            RenderOrder::Free => None,
            RenderOrder::After => Some(written_bind_group(device, &pipeline, &written)),
        };
        Ok(Self {
            pipeline,
            outputs,
            depth,
            depth_bytes,
            shown,
            blank,
            written,
        })
    }

    /// Copies the depth of `slot` into the buffer the pass draws: recorded in
    /// the frame that delivers the slot's job, after the frame's slices and
    /// before its render pass, so the depth is on screen from that frame and
    /// not before (the module documentation says why). A slot the network
    /// does not have is ignored.
    pub fn deliver(&self, encoder: &mut wgpu::CommandEncoder, slot: u32) {
        if let Some(output) = usize::try_from(slot).ok().and_then(|i| self.outputs.get(i)) {
            encoder.copy_buffer_to_buffer(
                &output.arena,
                output.offset,
                &self.depth,
                0,
                self.depth_bytes,
            );
        }
    }

    /// Records the pass into `target`, showing the depth delivered last when
    /// `delivered`, and the source alone before the first delivery.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        delivered: bool,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let group = if delivered { &self.shown } else { &self.blank };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("beatlm-web display"),
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
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, group, &[]);
        if let Some(written) = &self.written {
            pass.set_bind_group(1, written, &[]);
        }
        pass.draw(0..3, 0..1);
    }
}

/// Where a slot's depth is computed.
struct Output {
    /// The arena holding the slot's output.
    arena: wgpu::Buffer,
    /// The output's offset in it.
    offset: u64,
}

/// The bindings of one picture: the uniform `params`, the source `image` and
/// a `depth`.
fn bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    params: &wgpu::Buffer,
    image: &wgpu::Buffer,
    depth: &wgpu::Buffer,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("beatlm-web display bindings"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: params.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: image.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: depth.as_entire_binding(),
            },
        ],
    })
}

/// The shader's `Params`, in its layout: two `vec4<f32>`, four `u32`, four
/// `f32`, little endian.
fn params_bytes(side: u32, render_iters: u32, shown: u32, range: (f32, f32)) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(64);
    for values in [MEAN, STD] {
        for value in values.into_iter().chain([0.0]) {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    for word in [side, render_iters, shown, 0] {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    for value in [range.0, range.1, 0.0, 0.0] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}
