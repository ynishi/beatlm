//! A lowered program realised on a device with one input and one output slot
//! per job in flight: [`Net`], and the [`Numbers`] it is realised with.
//!
//! The bench's `net` workload (`net.rs`) and `depth` workload (`depth.rs`)
//! both run through [`Net`]. The browser host (`beatlm-web`) uses this module
//! for the depth model, so a page and the bench realise a program with the
//! same slots, bind groups and memory plan.
//!
//! # Jobs, slots and the shared plan
//!
//! The plan covers one job's kernels, and the activations of every job share
//! its bytes. That is sound only because the scheduler runs jobs as one
//! sequence (the rule is owned by `Scheduler::plan_frame`): job `n + 1`'s first
//! slice never runs before job `n`'s last. What a job must keep from its
//! neighbours is the memory the plan cannot recycle, which is the input and the
//! output.
//!
//! A job's output must survive until its delivery frame renders, and up to
//! `SyncPolicy::max_in_flight` jobs are in flight (the rule is owned there).
//! The network therefore has that many **slots**, and job `i` uses slot
//! `i % slots`, `i` being its submission index (`JobId::get`). Slot 0 is the
//! graph's own input and output; every other slot is a pair of stand-ins
//! (`Program::stand_in`), live for the whole program, so no slot shares bytes
//! with an activation or with another slot. Each slot is one binding set of
//! the kernels, and the kernels that read the input or write the output get
//! one bind group per slot, all created in [`Net::new`], so choosing a slot in
//! the loop creates nothing. The kernels between share one bind group.
//! [`Net::new`] takes the program with the [`Numbers`] it is realised with,
//! which is how the depth model gets the same slots as the network.

use std::borrow::Cow;

use anyhow::{Context, Result, bail};
use beatlm::codegen::{Kernels, Substitution};
use beatlm::gpu::Arenas;
use beatlm_core::cost::SliceId;
use beatlm_core::device::DeviceProfile;
use beatlm_core::kernel::Program;
use beatlm_core::memory::{ArenaId, MemoryPlan, TensorDesc, TensorId};

/// Slot tensors are numbered from these bases, so a program's own tensors and
/// the two kinds of slot tensor never collide.
const INPUT_BASE: u32 = 1 << 16;
const OUTPUT_BASE: u32 = 2 << 16;
/// The most slots a network can have, so that a slot's tensor id stays inside
/// its range.
pub const MAX_SLOTS: u32 = 1 << 16;

/// The input tensor of slot `slot`; slot 0's is the graph's input.
pub fn input_id(slot: u32) -> TensorId {
    TensorId::new(INPUT_BASE.saturating_add(slot))
}

/// The output tensor of slot `slot`; slot 0's is the graph's output.
pub fn output_id(slot: u32) -> TensorId {
    TensorId::new(OUTPUT_BASE.saturating_add(slot))
}

/// What a program is realised with besides its kernels: which of its tensors
/// are a job's input and output, every weight's numbers, and each slot's
/// input. The network's are `net::numbers` (the bench's `net.rs`); the depth
/// model's come from its weights file (each host's `depth.rs`), so both run
/// through one [`Net`] with the same slots and bind groups.
pub struct Numbers {
    /// The graph's input, which is slot 0's.
    pub input: TensorId,
    /// The graph's output, which is slot 0's.
    pub output: TensorId,
    /// Every weight of the graph with its numbers.
    pub weights: Vec<(TensorId, Vec<f32>)>,
    /// The numbers of a slot's input, uploaded once per slot before the loop.
    pub slot_input: Box<dyn Fn(u32) -> Vec<f32>>,
}

impl Numbers {
    /// The input tensor of slot `slot`: the graph's own for slot 0, a stand-in
    /// numbered from the input base for every other.
    pub fn input_of(&self, slot: u32) -> TensorId {
        if slot == 0 {
            self.input
        } else {
            input_id(slot)
        }
    }

    /// The output tensor of slot `slot`, as [`input_of`](Self::input_of).
    pub fn output_of(&self, slot: u32) -> TensorId {
        if slot == 0 {
            self.output
        } else {
            output_id(slot)
        }
    }
}

/// The program's tensors with one input and one output stand-in for each slot
/// after the first, all for the memory plan. A stand-in's id that the program
/// already uses is refused by `Program::stand_in`.
pub fn tensors(program: &Program, numbers: &Numbers, slots: u32) -> Result<Vec<TensorDesc>> {
    let mut all = program.tensors().to_vec();
    for slot in 1..slots {
        all.push(program.stand_in(numbers.input, numbers.input_of(slot))?);
        all.push(program.stand_in(numbers.output, numbers.output_of(slot))?);
    }
    Ok(all)
}

/// The memory plan [`Net::new`] realises `program` in with `slots` slots: its
/// tensors and every slot's stand-ins, planned against the baseline profile.
///
/// A host that must create its device with the limits the plan needs
/// (`beatlm::gpu::required_limits`) asks for it before the device exists,
/// and both hosts ask whether its arenas fit the render order before the
/// work that comes before [`Net::new`] (the page's download, the bench's
/// calibration); [`Net::new`] makes the same plan from the same tensors.
///
/// # Errors
///
/// No slots or more than [`MAX_SLOTS`], or tensors that cannot be planned.
pub fn memory_plan(program: &Program, numbers: &Numbers, slots: u32) -> Result<MemoryPlan> {
    planned(program, numbers, slots).map(|(plan, _)| plan)
}

/// [`memory_plan`] with the tensors it planned, so [`Net::new`] checks the
/// numbers against the same descriptions without making them twice.
fn planned(
    program: &Program,
    numbers: &Numbers,
    slots: u32,
) -> Result<(MemoryPlan, Vec<TensorDesc>)> {
    if slots == 0 || slots > MAX_SLOTS {
        bail!("{slots} slots; the network has between 1 and {MAX_SLOTS}");
    }
    let descs = tensors(program, numbers, slots)?;
    let plan = MemoryPlan::new(&descs, &DeviceProfile::webgpu_baseline())
        .context("cannot plan the network's memory")?;
    Ok((plan, descs))
}

/// One binding set per slot: the graph's input and output rebound to the
/// slot's.
fn binding_sets(numbers: &Numbers, slots: u32) -> Vec<Vec<Substitution>> {
    (0..slots)
        .map(|slot| {
            vec![
                (numbers.input, numbers.input_of(slot)),
                (numbers.output, numbers.output_of(slot)),
            ]
        })
        .collect()
}

/// The network, realised on a device: planned memory, uploaded numbers and one
/// generated kernel per spec, each part of a split kernel included, with its
/// bind groups.
///
/// Everything that allocates happens in [`Net::new`]; [`Net::encode`] only
/// records passes over existing pipelines and bind groups.
pub struct Net {
    plan: MemoryPlan,
    program: Program,
    kernels: Kernels,
    slots: u32,
    // Held so the buffers the bind groups point into live as long as the net.
    arenas: Arenas,
}

impl Net {
    /// Plans the memory of `program`, the network or the depth model lowered
    /// and perhaps split, with `slots` input and output slots, against the
    /// baseline profile ([`memory_plan`]), creates the arenas, uploads every
    /// weight and each slot's input from `numbers`, and builds every kernel
    /// with a bind group for every slot that needs one.
    ///
    /// `slots` is the number of jobs that can be in flight, from
    /// `SyncPolicy::max_in_flight`. A split program is planned like any other:
    /// its lifetimes are in its own kernel indices.
    ///
    /// # Errors
    ///
    /// No slots or more than [`MAX_SLOTS`], tensors that cannot be planned,
    /// numbers whose sizes are not the program's tensors' (a program of
    /// another shape), a plan the device cannot hold, or a kernel that cannot
    /// be generated or bound.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        numbers: &Numbers,
        slots: u32,
        program: Program,
    ) -> Result<Self> {
        let (plan, descs) = planned(&program, numbers, slots)?;
        let arenas = Arenas::new(device, &plan).context("cannot create the arenas")?;

        // Each slot's input is made only when it is uploaded, so the host
        // holds one at a time, not one per slot: a page of many slots would
        // otherwise hold them all in its memory at once.
        let weights = numbers
            .weights
            .iter()
            .map(|(id, values)| (*id, Cow::Borrowed(values.as_slice())));
        let inputs = (0..slots).map(|slot| {
            let values: Vec<f32> = (numbers.slot_input)(slot);
            (numbers.input_of(slot), Cow::Owned(values))
        });
        for (id, values) in weights.chain(inputs) {
            let id = &id;
            // `program` is given apart from its numbers, so a program of
            // another shape must not be uploaded into: the data of a smaller
            // shape would leave the rest of a larger tensor unwritten.
            let elements = descs.iter().find(|t| t.id == *id).map(|t| t.elements);
            if elements != u64::try_from(values.len()).ok() {
                bail!(
                    "tensor {}: {} numbers at this shape for a program tensor of {elements:?} elements; the program is of another shape",
                    id.get(),
                    values.len()
                );
            }
            let placement = plan.placement(*id).context("a tensor has no placement")?;
            let buffer = arenas
                .buffer(placement.arena)
                .context("a placement names no arena")?;
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
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

        let kernels = Kernels::new(
            device,
            &plan,
            &arenas,
            program.kernels(),
            &binding_sets(numbers, slots),
        )
        .context("cannot build the network's kernels")?;
        Ok(Self {
            plan,
            program,
            kernels,
            slots,
            arenas,
        })
    }

    /// The memory plan the network runs in.
    pub fn plan(&self) -> &MemoryPlan {
        &self.plan
    }

    /// The buffers the plan's arenas are, for reading a slot's tensors: the
    /// bench's tests read an output back, and the browser host's display binds
    /// a slot's input and output to draw them.
    pub fn arenas(&self) -> &Arenas {
        &self.arenas
    }

    /// Every buffer a kernel of the network can write: all the plan's arenas,
    /// which a render pass reads to be ordered after a frame's slices
    /// (`workload.rs`, "Ordering the render pass after the slices", says why
    /// all of them).
    ///
    /// # Errors
    ///
    /// An arena with no buffer, or one smaller than the word the render
    /// reads of it; neither comes from a plan [`Net::new`] realised.
    pub fn written(&self) -> Result<Vec<&wgpu::Buffer>> {
        self.plan
            .arenas()
            .iter()
            .zip(0_u32..)
            .map(|(&size, index)| {
                if size < 4 {
                    bail!("arena {index} has {size} bytes, less than the word the render reads");
                }
                self.arenas
                    .buffer(ArenaId::new(index))
                    .with_context(|| format!("arena {index} of the plan has no buffer"))
            })
            .collect()
    }

    /// The lowered network the kernels were generated from.
    pub fn program(&self) -> &Program {
        &self.program
    }

    /// How many jobs the network keeps apart: the number of input and output
    /// slots.
    pub fn slots(&self) -> u32 {
        self.slots
    }

    /// How many kernels, hence slices and dispatches, one job has: a split
    /// kernel counts each of its parts.
    pub fn kernel_count(&self) -> u32 {
        u32::try_from(self.kernels.len()).unwrap_or(u32::MAX)
    }

    /// The slot job number `job` (its submission index) uses: the index modulo
    /// the number of slots.
    pub fn slot_of(&self, job: u64) -> u32 {
        // The remainder is below `slots`, a u32.
        u32::try_from(job % u64::from(self.slots)).unwrap_or(0)
    }

    /// Records the kernel `slice` names as its own compute pass, reading the
    /// input of `slot` and writing the output of `slot` where it touches them.
    ///
    /// # Errors
    ///
    /// A slot the network does not have, or a slice that is not one of its
    /// kernels.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        slot: u32,
        slice: SliceId,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) -> Result<()> {
        let (kernel, set) = self.indices(slot, slice)?;
        self.kernels
            .encode(encoder, kernel, set, timestamp_writes)
            .with_context(|| format!("slice {} of the network", slice.get()))
    }

    /// The kernel and binding set indices of `slice` on `slot`.
    fn indices(&self, slot: u32, slice: SliceId) -> Result<(usize, usize)> {
        if slot >= self.slots {
            bail!("slot {slot} of a network with {} slots", self.slots);
        }
        let kernel = usize::try_from(slice.get()).unwrap_or(usize::MAX);
        let set = usize::try_from(slot).unwrap_or(usize::MAX);
        Ok((kernel, set))
    }

    /// Records the dispatch of the kernel `slice` names into `pass`, a compute
    /// pass the caller opened, on `slot` as [`encode`](Self::encode) does.
    /// The loop never does this: a slice is one pass. The `pass-cost`
    /// workload (`pass_cost.rs`) does, to put several kernels in one pass.
    ///
    /// # Errors
    ///
    /// A slot the network does not have, or a slice that is not one of its
    /// kernels.
    pub fn record(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        slot: u32,
        slice: SliceId,
    ) -> Result<()> {
        let (kernel, set) = self.indices(slot, slice)?;
        self.kernels
            .record(pass, kernel, set)
            .with_context(|| format!("slice {} of the network", slice.get()))
    }
}
