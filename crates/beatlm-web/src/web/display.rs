//! The depth model's render pass: the source and the delivered depth.
//!
//! **It reads no buffer a compute pass writes**, as the bench's render pass
//! does not, so its measured time, which the frame's budget takes as the
//! render's cost, cannot include waiting for the frame's slices
//! ([`crate::render_cost`] says what such a wait would do). The source is a
//! buffer of its own, the input every slot holds, uploaded once; each slot's
//! depth is copied into a buffer of its own when its job's last slice is
//! encoded ([`Display::keep`]), an untimed copy of `side * side` numbers, and
//! the pass draws from that copy. The report's `gpu_render_alone_us` and
//! `gpu_render_with_slices_us` show the render's measurement with and without
//! slices in its frame.
//!
//! Every slot gets a bind group of its own when the display is built, and one
//! more draws before the first delivery, so choosing what to show in a frame
//! creates nothing and uploads nothing, as the rest of the frame loop. The
//! depth is mapped to grey between the smallest and the largest depth the
//! check's run computed, fixed for the run. The shader (`display.wgsl`)
//! carries the synthetic render pass's per-pixel loop, so the pass costs what
//! the bench's render pass does beside the depth model.

use anyhow::{Context, Result};
use beatlm::load::depth_anything_v2::reference::{MEAN, STD};
use wgpu::util::DeviceExt;

use crate::slots::{Net, Numbers};
use crate::workload::TARGET_FORMAT;

/// The render pass that shows a slot's input and output.
pub struct Display {
    pipeline: wgpu::RenderPipeline,
    /// Per slot: where its output is in the arenas, and the display's copy.
    copies: Vec<Kept>,
    /// The bytes of one depth.
    depth_bytes: u64,
    /// One per slot, drawing that slot's input and output.
    slots: Vec<wgpu::BindGroup>,
    /// Drawn before the first delivery: the input, and no depth.
    blank: wgpu::BindGroup,
}

impl Display {
    /// Builds the pass for `net`, realised with `numbers`, whose input side is
    /// `side`, mapping depths from `range.0` to `range.1` to black to white,
    /// with `render_iters` loop iterations per pixel.
    ///
    /// # Errors
    ///
    /// A slot whose output has no placement in the plan or a smaller one than
    /// a depth of `side` pixels, or an input of another size.
    pub fn new(
        device: &wgpu::Device,
        net: &Net,
        numbers: &Numbers,
        side: u32,
        range: (f32, f32),
        render_iters: u32,
    ) -> Result<Self> {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("beatlm-web display"),
            source: wgpu::ShaderSource::Wgsl(include_str!("display.wgsl").into()),
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("beatlm-web display"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("display_vs"),
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
        let shown = params(1);
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
        let copies = (0..net.slots())
            .map(|slot| {
                let id = numbers.output_of(slot);
                let placement = net
                    .plan()
                    .placement(id)
                    .filter(|p| p.size >= depth_bytes)
                    .with_context(|| format!("the depth of slot {slot} has no placement"))?;
                let source = net
                    .arenas()
                    .buffer(placement.arena)
                    .with_context(|| format!("the depth of slot {slot} names no arena"))?
                    .clone();
                let depth = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("beatlm-web display depth"),
                    size: depth_bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                Ok(Kept {
                    source,
                    offset: placement.offset,
                    depth,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let layout = pipeline.get_bind_group_layout(0);
        let slots = copies
            .iter()
            .map(|copy| bind_group(device, &layout, &shown, &image, &copy.depth))
            .collect();
        let first = copies.first().context("the network has no slots")?;
        let blank = bind_group(device, &layout, &blank_params, &image, &first.depth);
        Ok(Self {
            pipeline,
            copies,
            depth_bytes,
            slots,
            blank,
        })
    }

    /// Copies the depth of `slot` into the display's own buffer: encoded after
    /// the last slice of the slot's job, so the render pass reads a buffer no
    /// slice writes. A slot the network does not have is ignored.
    pub fn keep(&self, encoder: &mut wgpu::CommandEncoder, slot: u32) {
        if let Some(copy) = usize::try_from(slot).ok().and_then(|i| self.copies.get(i)) {
            encoder.copy_buffer_to_buffer(
                &copy.source,
                copy.offset,
                &copy.depth,
                0,
                self.depth_bytes,
            );
        }
    }

    /// Records the pass into `target`, showing the depth of `slot`, or none
    /// before the first delivery.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        slot: Option<u32>,
        timestamp_writes: Option<wgpu::RenderPassTimestampWrites<'_>>,
    ) {
        let group = slot
            .and_then(|slot| self.slots.get(usize::try_from(slot).ok()?))
            .unwrap_or(&self.blank);
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
        pass.draw(0..3, 0..1);
    }
}

/// Where a slot's depth is computed, and the display's copy of it.
struct Kept {
    /// The arena holding the slot's output.
    source: wgpu::Buffer,
    /// The output's offset in it.
    offset: u64,
    /// The display's copy.
    depth: wgpu::Buffer,
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
