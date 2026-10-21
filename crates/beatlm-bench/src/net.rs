//! A small network run through the memory plan and the arenas: the `net`
//! workload.
//!
//! The network is six operations, each one slice of the frame loop, over an
//! `H x W` image:
//!
//! | Slice | Operation | Reads | Writes |
//! |---|---|---|---|
//! | 0 | 3x3 convolution, `C0 -> C1` | input, conv1 weights | activation `a` |
//! | 1 | layer norm over channels, `C1` | `a`, norm1 parameters | `b` |
//! | 2 | 3x3 convolution, `C1 -> C2` | `b`, conv2 weights | `c` |
//! | 3 | matrix multiply, `[HW, C2] x [C2, C3]` | `c`, matmul1 weights | `d` |
//! | 4 | layer norm over channels, `C3` | `d`, norm2 parameters | `e` |
//! | 5 | matrix multiply, `[HW, C3] x [C3, C4]` | `e`, matmul2 weights | output `f` |
//!
//! Each activation is read by the next operation only, so at most two are live
//! at once and the planner reuses their memory: that is what the workload is
//! for, a model whose plan is smaller than the sum of its tensors, run for
//! real. Every weight, and every input and output (below), is live for the
//! whole program.
//!
//! # What the workload stands in for
//!
//! The kernels (`net.wgsl`) are hand-written scaffolding until generated
//! kernels replace them. Activations are NHWC with one image (`[H][W][C]`,
//! channels contiguous), convolution weights are HWIO, and every number is
//! `f32`. Shapes arrive as pipeline-overridable constants, and the tiled matrix
//! multiply uses 2 KiB of workgroup memory, well inside the baseline's 16 KiB.
//! The plan is made against [`DeviceProfile::webgpu_baseline`], so it is valid
//! on every adapter.
//!
//! # Jobs, slots and the shared plan
//!
//! The plan covers one job's six operations, and the activations of every job
//! share its bytes. That is sound only because the scheduler runs jobs as one
//! sequence (the rule is owned by `Scheduler::plan_frame`): job `n + 1`'s first
//! slice never runs before job `n`'s last. What a job must keep from its
//! neighbours is the memory the plan cannot recycle, which is the input and the
//! output.
//!
//! A job's output must survive until its delivery frame renders, and up to
//! `SyncPolicy::max_in_flight` jobs are in flight (the rule is owned there).
//! The network therefore has that many **slots**, and job `i` uses slot
//! `i % slots`, `i` being its submission index (`JobId::get`). A slot is an
//! input tensor and an output tensor, both live for the whole program, so
//! neither shares bytes with an activation or with another slot's. The
//! slot's bind groups for the two operations that touch them (the first reads
//! the input, the last writes the output) are created in [`Net::new`], so
//! choosing a slot in the loop creates nothing. The four operations between
//! touch only activations and weights and have one bind group.
//!
//! Two simplifications are the workload's, not the engine's:
//!
//! - **The inputs are resident.** Each slot's input is uploaded once, from its
//!   own seed, so jobs of different slots compute different results, and the
//!   host never refills an input in the loop. A host that did refill it per
//!   job would need the refill ordered on the queue before the job's first
//!   slice.
//! - **The output is not consumed.** The bench reads nothing back; a test reads
//!   a delivered job's output slot to check it. With one job submitted per
//!   frame, slot `i % slots` is next written by job `i + slots`, which is
//!   submitted after job `i`'s delivery frame.

use anyhow::{Context, Result, bail};
use beatlm::gpu::Arenas;
use beatlm_core::cost::SliceId;
use beatlm_core::device::DeviceProfile;
use beatlm_core::memory::{DType, MemoryPlan, TensorDesc, TensorId};

/// How many operations, hence slices, the network has.
pub const OPS: u32 = 6;

/// The index of the last operation, which ends the lifetime of whatever lives
/// for the whole program.
const LAST_OP: u32 = OPS - 1;

/// Width and height of a matrix-multiply tile; the kernel's `MM_TILE`.
const MM_TILE: u32 = 16;
/// Invocations per workgroup of the convolution; passed to the kernel as
/// `CONV_GROUP`, its `@workgroup_size`.
const CONV_GROUP: u32 = 256;
/// Workgroups per row of the convolution's grid; passed to the kernel as
/// `CONV_ROW`.
const CONV_ROW: u32 = 256;
/// Workgroups per row of the layer-norm grid; passed to the kernel as `LN_ROW`.
const LN_ROW: u32 = 256;

const W1: TensorId = TensorId::new(0);
const A: TensorId = TensorId::new(1);
const P1: TensorId = TensorId::new(2);
const B: TensorId = TensorId::new(3);
const W2: TensorId = TensorId::new(4);
const C: TensorId = TensorId::new(5);
const W3: TensorId = TensorId::new(6);
const D: TensorId = TensorId::new(7);
const P2: TensorId = TensorId::new(8);
const E: TensorId = TensorId::new(9);
const W4: TensorId = TensorId::new(10);

/// Slot tensors are numbered from these bases, so the fixed tensors above and
/// the two kinds of slot tensor never collide.
const INPUT_BASE: u32 = 1 << 16;
const OUTPUT_BASE: u32 = 2 << 16;
/// The most slots a network can have, so that a slot's tensor id stays inside
/// its range.
pub const MAX_SLOTS: u32 = 1 << 16;

/// The input tensor of slot `slot`.
fn input_id(slot: u32) -> TensorId {
    TensorId::new(INPUT_BASE.saturating_add(slot))
}

/// The output tensor of slot `slot`, the network's `f`.
fn output_id(slot: u32) -> TensorId {
    TensorId::new(OUTPUT_BASE.saturating_add(slot))
}

/// The image size and channel counts of the network.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    /// Image height in pixels.
    pub height: u32,
    /// Image width in pixels.
    pub width: u32,
    /// Channels of the input.
    pub c0: u32,
    /// Channels after the first convolution and first layer norm.
    pub c1: u32,
    /// Channels after the second convolution.
    pub c2: u32,
    /// Channels after the first matrix multiply and the second layer norm.
    pub c3: u32,
    /// Channels of the output.
    pub c4: u32,
}

/// The shape the bench runs, chosen so each slice takes roughly 0.5 to 2 ms on
/// an Apple M1 Pro (Metal); the crate documentation has the measurement.
pub const DEFAULT_SHAPE: Shape = Shape {
    height: 128,
    width: 128,
    c0: 4,
    c1: 16,
    c2: 8,
    c3: 64,
    c4: 16,
};

impl Shape {
    fn pixels(&self) -> u64 {
        u64::from(self.height) * u64::from(self.width)
    }
}

/// The host-side numbers of the network: every weight and parameter. The inputs
/// are [`Data::input`], one per slot.
///
/// Generated from fixed seeds, so the GPU run and the CPU reference start from
/// the same values on every run.
pub struct Data {
    /// First convolution, `[3][3][c0][c1]`.
    pub w1: Vec<f32>,
    /// First layer norm: gamma then beta, `2 * c1`.
    pub p1: Vec<f32>,
    /// Second convolution, `[3][3][c1][c2]`.
    pub w2: Vec<f32>,
    /// First matrix multiply, `[c2][c3]`.
    pub w3: Vec<f32>,
    /// Second layer norm: gamma then beta, `2 * c3`.
    pub p2: Vec<f32>,
    /// Second matrix multiply, `[c3][c4]`.
    pub w4: Vec<f32>,
}

impl Data {
    /// The input image of slot `slot`, `[h][w][c0]`. Every slot has its own, so
    /// jobs of different slots compute different results.
    pub fn input(shape: &Shape, slot: u32) -> Vec<f32> {
        let n = |v: u32| usize::try_from(v).unwrap_or(usize::MAX);
        let pixels = n(shape.height).saturating_mul(n(shape.width));
        uniform(
            1 + 1_000 * u64::from(slot),
            pixels.saturating_mul(n(shape.c0)),
            0.0,
            1.0,
        )
    }

    /// The network's weights and parameters for `shape`.
    pub fn generate(shape: &Shape) -> Self {
        let n = |v: u32| usize::try_from(v).unwrap_or(usize::MAX);
        let (c0, c1, c2, c3, c4) = (
            n(shape.c0),
            n(shape.c1),
            n(shape.c2),
            n(shape.c3),
            n(shape.c4),
        );
        Self {
            w1: weights(2, 9 * c0 * c1, 9 * c0),
            p1: norm_params(3, c1),
            w2: weights(4, 9 * c1 * c2, 9 * c1),
            w3: weights(5, c2 * c3, c2),
            p2: norm_params(6, c3),
            w4: weights(7, c3 * c4, c3),
        }
    }
}

/// A tiny deterministic generator, so data needs no dependency and replays.
struct Lcg(u64);

impl Lcg {
    /// The next value in `[0, 1)`.
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // The top 24 bits fit an f32's mantissa exactly.
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// `count` values uniform in `[low, high)`.
fn uniform(seed: u64, count: usize, low: f32, high: f32) -> Vec<f32> {
    let mut rng = Lcg(seed);
    (0..count)
        .map(|_| low + (high - low) * rng.next())
        .collect()
}

/// Weights uniform in `[-1, 1) / sqrt(fan_in)`, which keeps activations near
/// unit size through the linear layers.
fn weights(seed: u64, count: usize, fan_in: usize) -> Vec<f32> {
    let scale = 1.0 / (fan_in as f32).sqrt();
    uniform(seed, count, -scale, scale)
}

/// Gamma near one then beta near zero, for `channels` channels.
fn norm_params(seed: u64, channels: usize) -> Vec<f32> {
    let mut p = uniform(seed, channels, 0.9, 1.1);
    p.extend(uniform(seed + 100, channels, -0.1, 0.1));
    p
}

/// The tensors of the network with their lifetimes, in operation indices, and
/// one input and one output for each of `slots` slots.
///
/// The slot tensors live for the whole program, so no activation and no other
/// slot shares their bytes.
fn tensors(s: &Shape, slots: u32) -> Vec<TensorDesc> {
    let px = s.pixels();
    let ch = |c: u32| px.saturating_mul(u64::from(c));
    let n = u64::from;
    let f32s = |id, elements, first, last| TensorDesc::new(id, DType::F32, elements, first, last);
    let io = (0..slots).flat_map(|slot| {
        [
            f32s(input_id(slot), ch(s.c0), 0, LAST_OP),
            f32s(output_id(slot), ch(s.c4), 0, LAST_OP),
        ]
    });
    vec![
        f32s(W1, 9 * n(s.c0) * n(s.c1), 0, LAST_OP),
        f32s(A, ch(s.c1), 0, 1),
        f32s(P1, 2 * n(s.c1), 0, LAST_OP),
        f32s(B, ch(s.c1), 1, 2),
        f32s(W2, 9 * n(s.c1) * n(s.c2), 0, LAST_OP),
        f32s(C, ch(s.c2), 2, 3),
        f32s(W3, n(s.c2) * n(s.c3), 0, LAST_OP),
        f32s(D, ch(s.c3), 3, 4),
        f32s(P2, 2 * n(s.c3), 0, LAST_OP),
        f32s(E, ch(s.c3), 4, 5),
        f32s(W4, n(s.c3) * n(s.c4), 0, LAST_OP),
    ]
    .into_iter()
    .chain(io)
    .collect()
}

/// One operation: a pipeline, its bindings and the dispatch that runs it.
///
/// An operation that touches a slot's input or output has one bind group per
/// slot; any other has one. All are created with the operation.
struct Op {
    name: &'static str,
    pipeline: wgpu::ComputePipeline,
    bind_groups: Vec<wgpu::BindGroup>,
    dispatch: [u32; 3],
}

impl Op {
    /// Records the operation as its own compute pass, with the bind group of
    /// `slot` if the operation has one per slot.
    fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        slot: usize,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) {
        // Never empty: `build_op` rejects an operation with no bindings.
        let index = slot.checked_rem(self.bind_groups.len()).unwrap_or(0);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(self.name),
            timestamp_writes,
        });
        pass.set_pipeline(&self.pipeline);
        if let Some(group) = self.bind_groups.get(index) {
            pass.set_bind_group(0, group, &[]);
        }
        pass.dispatch_workgroups(self.dispatch[0], self.dispatch[1], self.dispatch[2]);
    }
}

/// The three bindings every kernel takes, in binding order: two inputs and the
/// output.
type Io<'a> = [wgpu::BufferBinding<'a>; 3];

/// Builds one operation: the entry point `entry` with the given constants, one
/// bind group for each set of three bindings in `ios`, and a dispatch checked
/// against the baseline's limit on workgroups per dimension.
fn build_op(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    name: &'static str,
    entry: &'static str,
    constants: &[(&str, f64)],
    ios: &[Io<'_>],
    dispatch: [u64; 3],
) -> Result<Op> {
    if ios.is_empty() {
        bail!("{name}: no bindings");
    }
    let limit = u64::from(DeviceProfile::webgpu_baseline().max_compute_workgroups_per_dimension);
    if dispatch.iter().any(|&d| d > limit) {
        bail!("{name}: dispatch {dispatch:?} exceeds {limit} workgroups in a dimension");
    }
    // Checked just above, so each fits a u32.
    let dispatch = dispatch.map(|d| u32::try_from(d).unwrap_or(u32::MAX));
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(name),
        layout: None,
        module,
        entry_point: Some(entry),
        compilation_options: wgpu::PipelineCompilationOptions {
            constants,
            ..Default::default()
        },
        cache: None,
    });
    let layout = pipeline.get_bind_group_layout(0);
    let bind_groups = ios
        .iter()
        .map(|io| {
            let entries: Vec<wgpu::BindGroupEntry<'_>> = (0..)
                .zip(io.iter().cloned())
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding,
                    resource: wgpu::BindingResource::Buffer(buffer),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(name),
                layout: &layout,
                entries: &entries,
            })
        })
        .collect();
    Ok(Op {
        name,
        pipeline,
        bind_groups,
        dispatch,
    })
}

/// A 3x3 convolution of an `h x w` image from `cin` to `cout` channels.
fn conv_op(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    (h, w, cin, cout): (u32, u32, u32, u32),
    ios: &[Io<'_>],
) -> Result<Op> {
    let groups = (u64::from(h) * u64::from(w) * u64::from(cout)).div_ceil(u64::from(CONV_GROUP));
    let row = u64::from(CONV_ROW);
    build_op(
        device,
        module,
        "conv3x3",
        "conv3x3",
        &[
            ("CONV_H", f64::from(h)),
            ("CONV_W", f64::from(w)),
            ("CONV_CIN", f64::from(cin)),
            ("CONV_COUT", f64::from(cout)),
            ("CONV_GROUP", f64::from(CONV_GROUP)),
            ("CONV_ROW", f64::from(CONV_ROW)),
        ],
        ios,
        [groups.min(row), groups.div_ceil(row), 1],
    )
}

/// Layer normalisation over `c` channels of each of `pixels` pixels.
fn layernorm_op(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    (pixels, c): (u64, u32),
    ios: &[Io<'_>],
) -> Result<Op> {
    let pixels32 = u32::try_from(pixels).context("layernorm: more pixels than a u32 counts")?;
    build_op(
        device,
        module,
        "layernorm",
        "layernorm",
        &[
            ("LN_C", f64::from(c)),
            ("LN_PIXELS", f64::from(pixels32)),
            ("LN_ROW", f64::from(LN_ROW)),
        ],
        ios,
        [
            pixels.min(u64::from(LN_ROW)),
            pixels.div_ceil(u64::from(LN_ROW)),
            1,
        ],
    )
}

/// `[m, k] x [k, n]`.
fn matmul_op(
    device: &wgpu::Device,
    module: &wgpu::ShaderModule,
    (m, k, n): (u32, u32, u32),
    ios: &[Io<'_>],
) -> Result<Op> {
    build_op(
        device,
        module,
        "matmul",
        "matmul",
        &[
            ("MM_M", f64::from(m)),
            ("MM_K", f64::from(k)),
            ("MM_N", f64::from(n)),
        ],
        ios,
        [
            u64::from(n).div_ceil(u64::from(MM_TILE)),
            u64::from(m).div_ceil(u64::from(MM_TILE)),
            1,
        ],
    )
}

fn shader(device: &wgpu::Device) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("beatlm-bench net"),
        source: wgpu::ShaderSource::Wgsl(include_str!("net.wgsl").into()),
    })
}

/// The network, realised on a device: planned memory, uploaded numbers and one
/// pipeline with its bindings per operation.
///
/// Everything that allocates happens in [`Net::new`]; [`Net::encode`] only
/// records passes over existing pipelines and bind groups.
pub struct Net {
    plan: MemoryPlan,
    ops: Vec<Op>,
    slots: u32,
    // Held so the buffers the bind groups point into live as long as the net.
    #[cfg_attr(not(test), allow(dead_code))]
    arenas: Arenas,
}

impl Net {
    /// Plans the network's memory against the baseline profile, creates the
    /// arenas, uploads the weights and the `slots` inputs, and builds the six
    /// operations with a bind group for every slot.
    ///
    /// `slots` is the number of jobs that can be in flight, from
    /// `SyncPolicy::max_in_flight`.
    ///
    /// # Errors
    ///
    /// No slots or more than [`MAX_SLOTS`], a shape whose tensors cannot be
    /// planned or whose dispatch exceeds the baseline's workgroup limit, or a
    /// plan the device cannot hold.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        shape: &Shape,
        slots: u32,
    ) -> Result<Self> {
        if slots == 0 || slots > MAX_SLOTS {
            bail!("{slots} slots; the network has between 1 and {MAX_SLOTS}");
        }
        let plan = MemoryPlan::new(&tensors(shape, slots), &DeviceProfile::webgpu_baseline())
            .context("cannot plan the network's memory")?;
        let arenas = Arenas::new(device, &plan).context("cannot create the arenas")?;

        let data = Data::generate(shape);
        let mut uploads = vec![
            (W1, data.w1),
            (P1, data.p1),
            (W2, data.w2),
            (W3, data.w3),
            (P2, data.p2),
            (W4, data.w4),
        ];
        uploads.extend((0..slots).map(|slot| (input_id(slot), Data::input(shape, slot))));
        for (id, numbers) in &uploads {
            let placement = plan.placement(*id).context("a tensor has no placement")?;
            let buffer = arenas
                .buffer(placement.arena)
                .context("a placement names no arena")?;
            let bytes: Vec<u8> = numbers.iter().flat_map(|v| v.to_le_bytes()).collect();
            // `write_buffer` checks only the buffer's end, so data longer than
            // its placement would overwrite a neighbouring tensor in the arena.
            if u64::try_from(bytes.len()).map_or(true, |len| len > placement.size) {
                bail!(
                    "tensor {}: {} bytes of data for a placement of {}",
                    id.get(),
                    bytes.len(),
                    placement.size
                );
            }
            queue.write_buffer(buffer, placement.offset, &bytes);
        }

        let module = shader(device);
        let bind = |a: TensorId, b: TensorId, out: TensorId| -> Result<Io<'_>> {
            let one = |id: TensorId| -> Result<wgpu::BufferBinding<'_>> {
                let placement = plan.placement(id).context("a tensor has no placement")?;
                arenas
                    .binding(&placement)
                    .context("a placement cannot be bound")
            };
            Ok([one(a)?, one(b)?, one(out)?])
        };
        // Every bind group of the loop exists before it: the operations that
        // read an input or write an output get one per slot.
        let first: Vec<Io<'_>> = (0..slots)
            .map(|slot| bind(input_id(slot), W1, A))
            .collect::<Result<_>>()?;
        let last: Vec<Io<'_>> = (0..slots)
            .map(|slot| bind(E, W4, output_id(slot)))
            .collect::<Result<_>>()?;
        let px = shape.pixels();
        let pixels = u32::try_from(px).context("more pixels than a u32 counts")?;
        let (h, w) = (shape.height, shape.width);
        let ops = vec![
            conv_op(device, &module, (h, w, shape.c0, shape.c1), &first)?,
            layernorm_op(device, &module, (px, shape.c1), &[bind(A, P1, B)?])?,
            conv_op(
                device,
                &module,
                (h, w, shape.c1, shape.c2),
                &[bind(B, W2, C)?],
            )?,
            matmul_op(
                device,
                &module,
                (pixels, shape.c2, shape.c3),
                &[bind(C, W3, D)?],
            )?,
            layernorm_op(device, &module, (px, shape.c3), &[bind(D, P2, E)?])?,
            matmul_op(device, &module, (pixels, shape.c3, shape.c4), &last)?,
        ];
        debug_assert_eq!(u32::try_from(ops.len()), Ok(OPS));
        Ok(Self {
            plan,
            ops,
            slots,
            arenas,
        })
    }

    /// The memory plan the network runs in.
    pub fn plan(&self) -> &MemoryPlan {
        &self.plan
    }

    /// How many jobs the network keeps apart: the number of input and output
    /// slots.
    pub fn slots(&self) -> u32 {
        self.slots
    }

    /// The slot job number `job` (its submission index) uses: the index modulo
    /// the number of slots.
    pub fn slot_of(&self, job: u64) -> u32 {
        // The remainder is below `slots`, a u32.
        u32::try_from(job % u64::from(self.slots)).unwrap_or(0)
    }

    /// Records the operation `slice` names as its own compute pass, reading
    /// the input of `slot` and writing the output of `slot` where it touches
    /// them.
    ///
    /// # Errors
    ///
    /// A slot the network does not have, or a slice that is not one of its
    /// operations.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        slot: u32,
        slice: SliceId,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) -> Result<()> {
        if slot >= self.slots {
            bail!("slot {slot} of a network with {} slots", self.slots);
        }
        let op = usize::try_from(slice.get())
            .ok()
            .and_then(|i| self.ops.get(i))
            .with_context(|| format!("slice {} is not an operation of the network", slice.get()))?;
        op.encode(
            encoder,
            usize::try_from(slot).unwrap_or(usize::MAX),
            timestamp_writes,
        );
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::reference;
    use beatlm_core::budget::Budget;
    use beatlm_core::cost::{CostModel, Estimator};
    use beatlm_core::schedule::Scheduler;
    use beatlm_core::sync::{MissPolicy, SyncPolicy};
    use beatlm_core::time::{FrameIndex, FrameInfo, Micros, TimePoint};
    use std::collections::{BTreeMap, BTreeSet};
    use wgpu::util::DeviceExt;

    fn device() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .unwrap();
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap()
    }

    fn upload(device: &wgpu::Device, numbers: &[f32]) -> wgpu::Buffer {
        let bytes: Vec<u8> = numbers.iter().flat_map(|v| v.to_le_bytes()).collect();
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        })
    }

    fn scratch(device: &wgpu::Device, count: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (count * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Copies `count` floats from `offset` of `buffer` and waits for them.
    fn read(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        buffer: &wgpu::Buffer,
        offset: u64,
        count: usize,
    ) -> Vec<f32> {
        let bytes = (count * 4) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(buffer, offset, &staging, 0, bytes);
        queue.submit([encoder.finish()]);
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, |r| r.unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let mapped = staging.slice(..).get_mapped_range().unwrap();
        mapped
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    }

    fn run(device: &wgpu::Device, queue: &wgpu::Queue, op: &Op) {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        op.encode(&mut encoder, 0, None);
        queue.submit([encoder.finish()]);
    }

    fn whole(buffer: &wgpu::Buffer) -> wgpu::BufferBinding<'_> {
        wgpu::BufferBinding {
            buffer,
            offset: 0,
            size: None,
        }
    }

    fn assert_close(got: &[f32], want: &[f32], what: &str) {
        if let Some((i, g, w)) = reference::first_mismatch(got, want) {
            panic!("{what}: element {i} is {g}, the reference says {w}");
        }
    }

    // Shapes that are not multiples of any tile or workgroup size, so the
    // kernels' bounds handling is exercised.
    const SMALL_H: u32 = 7;
    const SMALL_W: u32 = 9;

    #[test]
    #[ignore = "needs a GPU adapter"]
    fn conv3x3_matches_the_reference() {
        let (device, queue) = device();
        let (cin, cout) = (3, 37);
        let x = uniform(11, (SMALL_H * SMALL_W * cin) as usize, -1.0, 1.0);
        let w = weights(12, (9 * cin * cout) as usize, (9 * cin) as usize);
        let (xb, wb) = (upload(&device, &x), upload(&device, &w));
        let out = scratch(&device, (SMALL_H * SMALL_W * cout) as usize);
        let op = conv_op(
            &device,
            &shader(&device),
            (SMALL_H, SMALL_W, cin, cout),
            &[[whole(&xb), whole(&wb), whole(&out)]],
        )
        .unwrap();
        run(&device, &queue, &op);
        let got = read(
            &device,
            &queue,
            &out,
            0,
            (SMALL_H * SMALL_W * cout) as usize,
        );
        let want = reference::conv3x3(
            &x,
            &w,
            SMALL_H as usize,
            SMALL_W as usize,
            cin as usize,
            cout as usize,
        );
        assert_close(&got, &want, "conv3x3");
    }

    #[test]
    #[ignore = "needs a GPU adapter"]
    fn layernorm_matches_the_reference() {
        let (device, queue) = device();
        // More channels than the workgroup has invocations, and not a multiple.
        let c = 150;
        let pixels = u64::from(SMALL_H * SMALL_W);
        let x = uniform(21, pixels as usize * c as usize, -3.0, 5.0);
        let p = norm_params(22, c as usize);
        let (xb, pb) = (upload(&device, &x), upload(&device, &p));
        let out = scratch(&device, x.len());
        let op = layernorm_op(
            &device,
            &shader(&device),
            (pixels, c),
            &[[whole(&xb), whole(&pb), whole(&out)]],
        )
        .unwrap();
        run(&device, &queue, &op);
        let got = read(&device, &queue, &out, 0, x.len());
        let want = reference::layernorm(&x, &p, pixels as usize, c as usize);
        assert_close(&got, &want, "layernorm");
    }

    #[test]
    #[ignore = "needs a GPU adapter"]
    fn matmul_matches_the_reference() {
        let (device, queue) = device();
        let (m, k, n) = (37, 21, 19);
        let a = uniform(31, (m * k) as usize, -1.0, 1.0);
        let b = uniform(32, (k * n) as usize, -1.0, 1.0);
        let (ab, bb) = (upload(&device, &a), upload(&device, &b));
        let out = scratch(&device, (m * n) as usize);
        let op = matmul_op(
            &device,
            &shader(&device),
            (m, k, n),
            &[[whole(&ab), whole(&bb), whole(&out)]],
        )
        .unwrap();
        run(&device, &queue, &op);
        let got = read(&device, &queue, &out, 0, (m * n) as usize);
        let want = reference::matmul(&a, &b, m as usize, k as usize, n as usize);
        assert_close(&got, &want, "matmul");
    }

    /// The shape of the whole-network tests.
    const TEST_SHAPE: Shape = Shape {
        height: SMALL_H,
        width: SMALL_W,
        c0: 3,
        c1: 20,
        c2: 11,
        c3: 70,
        c4: 5,
    };

    /// The chain of reference kernels over the input `x` and the weights `d`.
    fn reference_output(shape: &Shape, d: &Data, x: &[f32]) -> Vec<f32> {
        let (h, w) = (shape.height as usize, shape.width as usize);
        let (c0, c1, c2, c3, c4) = (
            shape.c0 as usize,
            shape.c1 as usize,
            shape.c2 as usize,
            shape.c3 as usize,
            shape.c4 as usize,
        );
        let a = reference::conv3x3(x, &d.w1, h, w, c0, c1);
        let b = reference::layernorm(&a, &d.p1, h * w, c1);
        let c = reference::conv3x3(&b, &d.w2, h, w, c1, c2);
        let dd = reference::matmul(&c, &d.w3, h * w, c2, c3);
        let e = reference::layernorm(&dd, &d.p2, h * w, c3);
        reference::matmul(&e, &d.w4, h * w, c3, c4)
    }

    /// The output of `slot` of a net of [`TEST_SHAPE`], read back.
    fn read_output(device: &wgpu::Device, queue: &wgpu::Queue, net: &Net, slot: u32) -> Vec<f32> {
        let placement = net.plan().placement(output_id(slot)).unwrap();
        let buffer = net.arenas.buffer(placement.arena).unwrap();
        let count = (TEST_SHAPE.pixels() * u64::from(TEST_SHAPE.c4)) as usize;
        read(device, queue, buffer, placement.offset, count)
    }

    /// The whole network, in planned memory with its activations sharing bytes,
    /// against the chain of reference kernels: a plan that let two live
    /// tensors overlap would corrupt this.
    #[test]
    #[ignore = "needs a GPU adapter"]
    fn the_planned_network_matches_the_reference_chain() {
        let (device, queue) = device();
        let shape = TEST_SHAPE;
        let net = Net::new(&device, &queue, &shape, 1).unwrap();
        let plan = net.plan();
        assert!(plan.total_bytes() < plan.placements().map(|(_, p)| p.size).sum::<u64>());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        for slice in 0..OPS {
            net.encode(&mut encoder, 0, SliceId::new(slice), None)
                .unwrap();
        }
        queue.submit([encoder.finish()]);

        let got = read_output(&device, &queue, &net, 0);
        let want = reference_output(&shape, &Data::generate(&shape), &Data::input(&shape, 0));
        assert_close(&got, &want, "network output");
    }

    /// One run of [`drive`]: a job is submitted at each of the `submit` frames,
    /// and frame `f` has `budgets[f]` microseconds to spend.
    struct Scenario {
        name: &'static str,
        submit: &'static [u64],
        budgets: &'static [u64],
        /// Whether some job's slices must be split across frames.
        split: bool,
    }

    /// Drives the real scheduler, with latency 2 and so three slots, over the
    /// network on the GPU, and reads each delivered job's output slot back in
    /// its delivery frame, after that frame's slices and before any later
    /// frame's, which is when a render pass would read it.
    ///
    /// Every slot has its own input, so a job whose activations or output
    /// another job disturbed shows as a wrong result. Slice 0 costs less than
    /// the others, so when a job's next slice does not fit the rest of a
    /// frame's budget the next job's slice 0 does: the slice an interleaving
    /// scheduler would run early.
    fn drive(scenario: &Scenario) {
        let (device, queue) = device();
        let shape = TEST_SHAPE;
        let policy = SyncPolicy::FixedLatency { frames: 2 };
        let slots = policy.max_in_flight().unwrap();
        assert_eq!(slots, 3);
        let net = Net::new(&device, &queue, &shape, slots).unwrap();
        let weights = Data::generate(&shape);

        let mut cost = CostModel::new(
            Micros::new(100),
            Estimator::MeanPlusSigmas {
                alpha: 1.0,
                sigmas: 0.0,
            },
        )
        .unwrap();
        cost.record(SliceId::new(0), Micros::new(40));
        let mut scheduler = Scheduler::new(policy, MissPolicy::Drop, None).unwrap();
        let all: Vec<SliceId> = (0..OPS).map(SliceId::new).collect();

        let mut frames_with_slices: BTreeMap<u64, u32> = BTreeMap::new();
        let mut most_in_flight = 0;
        let mut delivered = Vec::new();
        for (f, &budget) in scenario.budgets.iter().enumerate() {
            let f = f as u64;
            let frame = FrameInfo {
                index: FrameIndex::new(f),
                start: TimePoint::new(f * 1_000),
                period: Micros::new(1_000),
            };
            if scenario.submit.contains(&f) {
                scheduler.submit(frame, all.clone(), &cost).unwrap();
            }
            most_in_flight = most_in_flight.max(scheduler.pending());
            let plan = scheduler.plan_frame(frame, Budget::new(Micros::new(budget)), &cost);
            assert!(plan.missed.is_empty(), "{}: frame {f}", scenario.name);

            let mut encoder =
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
            let mut touched = BTreeSet::new();
            for run in &plan.runs {
                touched.insert(run.job.get());
                let slot = net.slot_of(run.job.get());
                net.encode(&mut encoder, slot, run.slice, None).unwrap();
            }
            for job in touched {
                *frames_with_slices.entry(job).or_default() += 1;
            }
            queue.submit([encoder.finish()]);

            for d in &plan.delivered {
                let slot = net.slot_of(d.job.get());
                let got = read_output(&device, &queue, &net, slot);
                let want = reference_output(&shape, &weights, &Data::input(&shape, slot));
                assert_close(
                    &got,
                    &want,
                    &format!("{}: job {} at frame {f}", scenario.name, d.job.get()),
                );
                delivered.push(d.job.get());
            }
        }

        assert_eq!(
            delivered,
            (0..scenario.submit.len() as u64).collect::<Vec<_>>(),
            "{}: every job is delivered, in submission order",
            scenario.name
        );
        assert_eq!(scheduler.pending(), 0, "{}", scenario.name);
        assert_eq!(most_in_flight, 3, "{}: jobs in flight", scenario.name);
        let split = frames_with_slices.values().any(|&frames| frames >= 2);
        assert_eq!(
            split, scenario.split,
            "{}: jobs split across frames",
            scenario.name
        );
    }

    /// Jobs whose slices are spread over several frames, with three in flight:
    /// the next job's first slice must wait for the rest of the earlier job,
    /// or it overwrites activations the earlier job still reads.
    #[test]
    #[ignore = "needs a GPU adapter"]
    fn jobs_split_across_frames_do_not_disturb_each_other() {
        // The scenario means something only if the next job's first slice
        // writes bytes a half-finished job still reads: the first activation
        // `a` shares bytes with `d`, which slice 4 reads.
        let all = tensors(&TEST_SHAPE, 3);
        let plan = MemoryPlan::new(&all, &DeviceProfile::webgpu_baseline()).unwrap();
        let (a, d) = (plan.placement(A).unwrap(), plan.placement(D).unwrap());
        assert!(a.arena == d.arena && a.offset < d.end() && d.offset < a.end());

        // A job costs 540 us: 40 for slice 0 and 100 for each other slice. At
        // frame 1, 150 us let job 0 run slice 3 and leave 50 us, too little
        // for its slice 4 but enough for job 1's slice 0. The scheduler must
        // leave the 50 us unused.
        drive(&Scenario {
            name: "split",
            submit: &[0, 1, 2],
            budgets: &[240, 150, 300, 500, 540, 540],
            split: true,
        });
    }

    /// Jobs that finish before their delivery frame, three in flight: the next
    /// job's output must go to a slot of its own, or it replaces the result
    /// the earlier job's delivery frame is about to show.
    #[test]
    #[ignore = "needs a GPU adapter"]
    fn finished_jobs_keep_their_output_until_delivery() {
        // Every frame has room for a whole job, so job 0 is finished at frame
        // 0, job 1 at frame 1 and job 2 at frame 2, and job 0 is delivered at
        // frame 2.
        drive(&Scenario {
            name: "early",
            submit: &[0, 1, 2],
            budgets: &[1_000, 1_000, 1_000, 1_000, 1_000, 1_000],
            split: false,
        });
    }

    #[test]
    fn the_default_shape_plans_and_reuses_memory() {
        let plan = MemoryPlan::new(
            &tensors(&DEFAULT_SHAPE, 3),
            &DeviceProfile::webgpu_baseline(),
        )
        .unwrap();
        let summed: u64 = plan.placements().map(|(_, p)| p.size).sum();
        assert!(plan.total_bytes() < summed);
        assert!(plan.peak_live_bytes() <= plan.total_bytes());
    }

    #[test]
    fn lifetimes_put_activations_in_the_same_bytes_only_when_disjoint() {
        let plan = MemoryPlan::new(
            &tensors(&DEFAULT_SHAPE, 3),
            &DeviceProfile::webgpu_baseline(),
        )
        .unwrap();
        let live: Vec<_> = tensors(&DEFAULT_SHAPE, 3);
        for (i, a) in live.iter().enumerate() {
            for b in &live[i + 1..] {
                let overlapping = a.first_use <= b.last_use && b.first_use <= a.last_use;
                let (pa, pb) = (plan.placement(a.id).unwrap(), plan.placement(b.id).unwrap());
                if overlapping && pa.arena == pb.arena {
                    assert!(pa.end() <= pb.offset || pb.end() <= pa.offset);
                }
            }
        }
    }

    #[test]
    fn the_kernels_stay_inside_the_baseline() {
        let baseline = DeviceProfile::webgpu_baseline();
        let tile_bytes = 2 * MM_TILE * MM_TILE * 4;
        assert!(tile_bytes <= baseline.max_compute_workgroup_storage_size);
        assert!(MM_TILE * MM_TILE <= baseline.max_compute_invocations_per_workgroup);
        assert!(CONV_GROUP <= baseline.max_compute_invocations_per_workgroup);
        assert!(baseline.max_storage_buffers_per_shader_stage >= 3);
    }

    #[test]
    fn the_input_is_deterministic() {
        let a = Data::generate(&DEFAULT_SHAPE);
        let b = Data::generate(&DEFAULT_SHAPE);
        assert_eq!(a.w4, b.w4);
        assert_eq!(
            Data::input(&DEFAULT_SHAPE, 1),
            Data::input(&DEFAULT_SHAPE, 1)
        );
    }

    #[test]
    fn every_slot_has_its_own_input() {
        assert_ne!(
            Data::input(&DEFAULT_SHAPE, 0),
            Data::input(&DEFAULT_SHAPE, 1)
        );
    }

    #[test]
    fn slot_tensors_live_for_the_whole_program_and_share_nothing() {
        let all = tensors(&DEFAULT_SHAPE, 3);
        let plan = MemoryPlan::new(&all, &DeviceProfile::webgpu_baseline()).unwrap();
        let io: Vec<_> = (0..3)
            .flat_map(|slot| [input_id(slot), output_id(slot)])
            .collect();
        assert_eq!(all.len(), 11 + io.len());
        for id in &io {
            let desc = all.iter().find(|t| t.id == *id).unwrap();
            assert_eq!((desc.first_use, desc.last_use), (0, LAST_OP));
            let p = plan.placement(*id).unwrap();
            for other in all.iter().filter(|t| t.id != *id) {
                let q = plan.placement(other.id).unwrap();
                assert!(
                    p.arena != q.arena || p.end() <= q.offset || q.end() <= p.offset,
                    "tensor {} overlaps tensor {}",
                    id.get(),
                    other.id.get()
                );
            }
        }
    }
}
