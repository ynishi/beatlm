//! The kernel IR: a model as typed operations over tensors, and the kernels a
//! GPU runs for it.
//!
//! A [`Graph`] lists a model's operations in program order, each over
//! [`TensorId`]s with explicit shapes. [`Graph::lower`] turns it into a
//! [`Program`]: the [`KernelSpec`]s to dispatch, one compute pass each, and the
//! [`TensorDesc`]s the [`MemoryPlan`](crate::memory::MemoryPlan) places. A spec
//! says what a kernel computes, which tensors it binds, and what it needs from a
//! device; turning it into shader text is the umbrella crate's job, so nothing
//! here names a shader language.
//!
//! # Why fusion
//!
//! Every dispatch costs the host command encoding and a share of
//! `queue.submit` before any GPU work runs, and the CPU-side variance of a
//! frame lives there (the concurrency model in the `beatlm` crate
//! documentation has the measurement). An element-wise operation is cheap on
//! the GPU and costs a full dispatch anyway, plus a write and a read of its
//! input through memory. Fusing it into the kernel that produced its input
//! removes the dispatch, the round trip and the intermediate tensor. Kernels
//! written by hand cannot be fused per model, which is why the IR exists.
//!
//! What it bought on one machine is measured by `beatlm-bench --workload net`
//! with `--fusion on` and `off`; the table, with its adapter, configuration and
//! date, lives in that crate's documentation under "Fusion". In short, on an
//! Apple M1 Pro, Metal, on 2026-10-21: fusion took a job from ten dispatches
//! to six, and the median encode time
//! was lower in both fused runs than in either unfused one. The 99th
//! percentiles and the delivery rates spread more between two runs of one
//! setting than between the settings, so these runs show no effect of fusion
//! on either; the GPU time per frame moved by less than a millisecond at the
//! median. The plan shrank from 27 tensors in 25 979 424 bytes to 23 in
//! 21 785 120. The `beatlm-bench` crate documentation has the rest of the
//! reports.
//!
//! # Rules
//!
//! This module owns these, and each has a test:
//!
//! 1. **Lifetimes come from the kernels, never from a second list.** A
//!    tensor's [`TensorDesc`] is derived by [`Graph::lower`] from the kernels
//!    of the [`Program`] that use it, so the memory plan and the IR cannot
//!    disagree about when a tensor is live. A use is a kernel's index in the
//!    program, which is the operation index the memory plan reasons in. An
//!    activation is live from the kernel that writes it to the last kernel that
//!    reads it, both inclusive; an input, a weight and an output are live for
//!    the whole program, because the host writes inputs and reads outputs
//!    between runs (see the memory plan's
//!    [lifetimes](crate::memory#lifetimes)). [`Program::stand_in`] describes a
//!    tensor bound in place of an input or output, such as a per-job slot, by
//!    the same rule.
//! 2. **Fusion folds an element-wise operation into the heavy kernel before
//!    it, and only when nothing else needs the tensor between them.** An
//!    element-wise operation ([`OpKind::Add`], [`OpKind::Relu`],
//!    [`OpKind::Gelu`], [`OpKind::Scale`]) becomes part of the previous
//!    kernel's [epilogue](Epilogue) when that kernel's main operation is a
//!    convolution, a layer norm or a matrix multiply, the element-wise
//!    operation's input is the tensor that kernel writes (the heavy
//!    operation's output, or the output of an operation already fused into
//!    it), that tensor has no other consumer and is not a graph
//!    [output](TensorRole::Output), and the kernel's bindings stay within the
//!    baseline's storage-buffer limit. The intermediate tensor then disappears
//!    from the program and from the memory plan. Fusion never crosses a tensor
//!    another operation reads, and it is deterministic: the same graph lowers
//!    to the same program. [`Fusion::Off`] lowers the same graph to one kernel
//!    per operation.
//! 3. **Fusion changes no result beyond floating-point contraction and
//!    reassociation.** A fused kernel applies the same operations in the same
//!    order to the same `f32` values; the only difference is that the
//!    intermediate value stays in a register instead of being stored and
//!    loaded as `f32`, which is exact. A shader compiler may contract a
//!    multiply and an add across the former boundary into one fused
//!    multiply-add, or reorder an addition, and that rounding difference is the
//!    only one fusion allows.
//! 4. **Every kernel runs on the baseline.** Every spec has
//!    [`Variant::Baseline`], which needs no optional feature, and
//!    [`Graph::lower`] refuses a graph whose baseline variant would not fit
//!    [`DeviceProfile::webgpu_baseline`], so every program it returns runs on
//!    every adapter. [`KernelSpec::check`] states what a variant needs of a
//!    device — workgroup memory within the limit, features a subset of the
//!    device's — and [`KernelSpec::select_variant`] picks, from a profile, the
//!    most capable variant that passes.
//! 5. **Malformed graphs are errors, not guesses.** An unknown or repeated
//!    tensor, an element count that disagrees with an operation's shape, a
//!    tensor read before it is written or written twice, a write to an input or
//!    a weight, an activation or output nobody writes, a zero dimension and a
//!    non-finite scale are each a [`KernelError`].
//!
//! # Layouts
//!
//! Every tensor is `f32` and row-major. Activations are NHWC with one image
//! (`[H][W][C]`, channels contiguous), convolution weights are HWIO
//! (`[3][3][Cin][Cout]`), and a layer norm normalises the last dimension.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use thiserror::Error;

use crate::device::{DeviceFeatures, DeviceProfile};
use crate::memory::{DType, TensorDesc, TensorId};

/// The epsilon a layer norm adds to the variance before the square root, so a
/// constant row does not divide by zero.
pub const LAYER_NORM_EPSILON: f32 = 1e-5;

/// The most elements one tensor may have.
///
/// Kernels index tensors with 32-bit integers, and the last row of a
/// dispatch's grid of workgroups computes indices up to a row of workgroups
/// (under 2^24 elements at the baseline's 65 535 workgroups of 256) past the
/// end before it checks them, so the bound leaves that headroom below
/// `u32::MAX`; it is also the largest value a signed 32-bit coordinate holds.
pub const MAX_ELEMENTS: u64 = 0x7fff_ffff;

/// Invocations per workgroup of a kernel that computes one element per
/// invocation (a convolution, a standalone element-wise operation).
const ELEMENT_GROUP: u32 = 256;
/// Invocations per workgroup of a layer norm: one workgroup per row, reducing
/// through workgroup memory. A power of two, so the reduction halves evenly.
const LAYER_NORM_LANES: u32 = 64;
/// Width and height of a matrix-multiply tile, and of its workgroup.
const MATMUL_TILE: u32 = 16;
/// Bytes of one `f32`.
const F32_BYTES: u32 = 4;
/// The profile every program is checked against, so it runs everywhere.
const BASELINE: DeviceProfile = DeviceProfile::webgpu_baseline();

/// What a tensor is to the graph, which fixes its lifetime and who may write
/// it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorRole {
    /// Written by the host before a run; no operation writes it.
    Input,
    /// Written by the host once; no operation writes it.
    Weight,
    /// Written by exactly one operation and read by later ones; fusion may
    /// remove it.
    Activation,
    /// Written by exactly one operation and read by the host after a run;
    /// never removed by fusion.
    Output,
}

impl TensorRole {
    /// Whether a tensor of this role is live for the whole program.
    const fn whole_program(self) -> bool {
        !matches!(self, Self::Activation)
    }

    /// Whether an operation may write a tensor of this role.
    const fn written_by_an_op(self) -> bool {
        matches!(self, Self::Activation | Self::Output)
    }
}

/// A 3x3 convolution, stride 1, zero padding of 1, of one NHWC image.
///
/// Reads `input` (`[height][width][cin]`), `weights` (`[3][3][cin][cout]`)
/// and, when present, `bias` (`[cout]`); writes `[height][width][cout]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Conv2d3x3 {
    /// The image.
    pub input: TensorId,
    /// The HWIO weights.
    pub weights: TensorId,
    /// One value per output channel added to every pixel, or none.
    pub bias: Option<TensorId>,
    /// Image height in pixels.
    pub height: u32,
    /// Image width in pixels.
    pub width: u32,
    /// Input channels.
    pub cin: u32,
    /// Output channels.
    pub cout: u32,
}

/// Layer normalisation over the last dimension:
/// `(x - mean) / sqrt(var + LAYER_NORM_EPSILON) * gamma + beta` per row.
///
/// Reads `input` (`[rows][channels]`), `gamma` and `beta` (`[channels]`);
/// writes `[rows][channels]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LayerNorm {
    /// The rows to normalise.
    pub input: TensorId,
    /// The per-channel scale.
    pub gamma: TensorId,
    /// The per-channel shift.
    pub beta: TensorId,
    /// How many rows (pixels, for an image).
    pub rows: u32,
    /// The length of a row (channels).
    pub channels: u32,
}

/// `C[m][n] = A[m][k] * B[k][n] + bias[n]`, all row-major.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MatMul {
    /// The left matrix, `[m][k]`.
    pub a: TensorId,
    /// The right matrix, `[k][n]`.
    pub b: TensorId,
    /// One value per column added to every row, or none.
    pub bias: Option<TensorId>,
    /// Rows of `a` and of the result.
    pub m: u32,
    /// Columns of `a`, rows of `b`.
    pub k: u32,
    /// Columns of `b` and of the result.
    pub n: u32,
}

/// What an operation computes, with the tensors it reads.
///
/// The first three are heavy operations, each a kernel of its own; the rest
/// are element-wise and may be fused into the kernel before them (rule 2 of
/// the [module documentation](self#rules)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OpKind {
    /// A 3x3 convolution.
    Conv2d3x3(Conv2d3x3),
    /// A layer norm over channels.
    LayerNorm(LayerNorm),
    /// A matrix multiply.
    MatMul(MatMul),
    /// `a + b`, element by element; both have `elements` elements.
    Add {
        /// One operand.
        a: TensorId,
        /// The other operand.
        b: TensorId,
        /// Elements of each operand and of the result.
        elements: u32,
    },
    /// `max(x, 0)`.
    Relu {
        /// The operand.
        input: TensorId,
        /// Elements of the operand and of the result.
        elements: u32,
    },
    /// GELU, tanh approximation:
    /// `0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3)))`.
    Gelu {
        /// The operand.
        input: TensorId,
        /// Elements of the operand and of the result.
        elements: u32,
    },
    /// `x * factor`, for a finite constant `factor`.
    Scale {
        /// The operand.
        input: TensorId,
        /// The constant; must be finite.
        factor: f32,
        /// Elements of the operand and of the result.
        elements: u32,
    },
}

impl OpKind {
    /// Every tensor the operation reads, in binding order, with the element
    /// count its shape gives it. Products saturate, so an overflow reads as
    /// more than [`MAX_ELEMENTS`].
    fn reads(&self) -> Vec<(TensorId, u64)> {
        let mul = |dims: &[u32]| {
            dims.iter()
                .fold(1_u64, |acc, &d| acc.saturating_mul(u64::from(d)))
        };
        match *self {
            Self::Conv2d3x3(c) => {
                let mut r = vec![
                    (c.input, mul(&[c.height, c.width, c.cin])),
                    (c.weights, mul(&[9, c.cin, c.cout])),
                ];
                r.extend(c.bias.map(|b| (b, u64::from(c.cout))));
                r
            }
            Self::LayerNorm(l) => vec![
                (l.input, mul(&[l.rows, l.channels])),
                (l.gamma, u64::from(l.channels)),
                (l.beta, u64::from(l.channels)),
            ],
            Self::MatMul(mm) => {
                let mut r = vec![(mm.a, mul(&[mm.m, mm.k])), (mm.b, mul(&[mm.k, mm.n]))];
                r.extend(mm.bias.map(|b| (b, u64::from(mm.n))));
                r
            }
            Self::Add { a, b, elements } => vec![(a, elements.into()), (b, elements.into())],
            Self::Relu { input, elements }
            | Self::Gelu { input, elements }
            | Self::Scale {
                input, elements, ..
            } => vec![(input, elements.into())],
        }
    }

    /// The element count of the result.
    fn output_elements(&self) -> u64 {
        let mul = |a: u32, b: u32, c: u32| {
            u64::from(a)
                .saturating_mul(u64::from(b))
                .saturating_mul(u64::from(c))
        };
        match *self {
            Self::Conv2d3x3(c) => mul(c.height, c.width, c.cout),
            Self::LayerNorm(l) => mul(l.rows, l.channels, 1),
            Self::MatMul(mm) => mul(mm.m, mm.n, 1),
            Self::Add { elements, .. }
            | Self::Relu { elements, .. }
            | Self::Gelu { elements, .. }
            | Self::Scale { elements, .. } => elements.into(),
        }
    }

    /// Whether any dimension of the operation is zero.
    fn has_zero_dimension(&self) -> bool {
        match *self {
            Self::Conv2d3x3(c) => [c.height, c.width, c.cin, c.cout].contains(&0),
            Self::LayerNorm(l) => l.rows == 0 || l.channels == 0,
            Self::MatMul(mm) => [mm.m, mm.k, mm.n].contains(&0),
            Self::Add { elements, .. }
            | Self::Relu { elements, .. }
            | Self::Gelu { elements, .. }
            | Self::Scale { elements, .. } => elements == 0,
        }
    }

    /// The kernel that runs this operation alone.
    fn standalone(&self) -> (MainOp, Vec<Epilogue>) {
        match *self {
            Self::Conv2d3x3(c) => (MainOp::Conv2d3x3(c), Vec::new()),
            Self::LayerNorm(l) => (MainOp::LayerNorm(l), Vec::new()),
            Self::MatMul(mm) => (MainOp::MatMul(mm), Vec::new()),
            Self::Add { a, b, elements } => {
                (MainOp::Map { input: a, elements }, vec![Epilogue::Add(b)])
            }
            Self::Relu { input, elements } => {
                (MainOp::Map { input, elements }, vec![Epilogue::Relu])
            }
            Self::Gelu { input, elements } => {
                (MainOp::Map { input, elements }, vec![Epilogue::Gelu])
            }
            Self::Scale {
                input,
                factor,
                elements,
            } => (
                MainOp::Map { input, elements },
                vec![Epilogue::Scale(factor)],
            ),
        }
    }

    /// The epilogue step this operation becomes when the value flowing out of
    /// a kernel is `chain`, or `None` when it is not element-wise or does not
    /// read `chain`.
    fn as_epilogue(&self, chain: TensorId) -> Option<Epilogue> {
        match *self {
            Self::Add { a, b, .. } if a == chain && b != chain => Some(Epilogue::Add(b)),
            Self::Add { a, b, .. } if b == chain && a != chain => Some(Epilogue::Add(a)),
            Self::Relu { input, .. } if input == chain => Some(Epilogue::Relu),
            Self::Gelu { input, .. } if input == chain => Some(Epilogue::Gelu),
            Self::Scale { input, factor, .. } if input == chain => Some(Epilogue::Scale(factor)),
            _ => None,
        }
    }
}

/// One operation of a graph: what it computes and the tensor it writes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Op {
    /// What it computes.
    pub kind: OpKind,
    /// The tensor it writes; an [activation](TensorRole::Activation) or an
    /// [output](TensorRole::Output), written by no other operation.
    pub output: TensorId,
}

/// Whether [`Graph::lower`] fuses element-wise operations into the kernel
/// before them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Fusion {
    /// Fuse where rule 2 of the [module documentation](self#rules) allows.
    #[default]
    On,
    /// One kernel per operation, so the effect of fusion can be measured on
    /// the same graph.
    Off,
}

/// A tensor as the graph knows it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TensorInfo {
    role: TensorRole,
    elements: u64,
}

/// A model: its tensors and its operations in program order.
///
/// Built with [`tensor`](Self::tensor) and [`push`](Self::push), each of which
/// checks its input, so a graph that exists is well-formed except for
/// activations nobody has written yet, which [`lower`](Self::lower) checks.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Graph {
    tensors: BTreeMap<TensorId, TensorInfo>,
    ops: Vec<Op>,
    written: BTreeSet<TensorId>,
}

impl Graph {
    /// An empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares a tensor of `elements` `f32`s with the given role.
    ///
    /// # Errors
    ///
    /// A repeated id, no elements, or more than [`MAX_ELEMENTS`].
    pub fn tensor(
        &mut self,
        id: TensorId,
        role: TensorRole,
        elements: u64,
    ) -> Result<(), KernelError> {
        if self.tensors.contains_key(&id) {
            return Err(KernelError::DuplicateTensor(id));
        }
        if elements == 0 {
            return Err(KernelError::EmptyTensor(id));
        }
        if elements > MAX_ELEMENTS {
            return Err(KernelError::TensorTooLarge(id));
        }
        self.tensors.insert(id, TensorInfo { role, elements });
        Ok(())
    }

    /// Appends an operation that computes `kind` into `output`.
    ///
    /// # Errors
    ///
    /// A tensor not declared, an element count that disagrees with the
    /// operation's shape, a tensor read before an operation wrote it, an
    /// output that is an input or a weight or was already written, a zero
    /// dimension, a non-finite scale, or a size past [`MAX_ELEMENTS`].
    pub fn push(&mut self, kind: OpKind, output: TensorId) -> Result<(), KernelError> {
        let op = self.ops.len();
        if u32::try_from(op.saturating_add(1)).is_err() {
            return Err(KernelError::TooManyOps);
        }
        if let OpKind::Scale { factor, .. } = kind {
            if !factor.is_finite() {
                return Err(KernelError::NonFiniteScale { op });
            }
        }
        if kind.has_zero_dimension() {
            return Err(KernelError::ZeroDimension { op });
        }
        let out_elements = kind.output_elements();
        let reads = kind.reads();
        if reads
            .iter()
            .map(|&(_, elements)| elements)
            .chain([out_elements])
            .any(|elements| elements > MAX_ELEMENTS)
        {
            return Err(KernelError::OpTooLarge { op });
        }
        for &(tensor, expected) in &reads {
            let info = self.info(tensor, expected, op)?;
            if info.role.written_by_an_op() && !self.written.contains(&tensor) {
                return Err(KernelError::ReadBeforeWritten { tensor, op });
            }
        }
        let info = self.info(output, out_elements, op)?;
        if !info.role.written_by_an_op() {
            return Err(KernelError::NotWritable { tensor: output, op });
        }
        if !self.written.insert(output) {
            return Err(KernelError::WrittenTwice { tensor: output, op });
        }
        self.ops.push(Op { kind, output });
        Ok(())
    }

    /// The declared tensor `id`, checked to have `expected` elements.
    fn info(&self, id: TensorId, expected: u64, op: usize) -> Result<TensorInfo, KernelError> {
        let info = *self
            .tensors
            .get(&id)
            .ok_or(KernelError::UnknownTensor(id))?;
        if info.elements != expected {
            return Err(KernelError::ElementCount {
                tensor: id,
                declared: info.elements,
                op,
                expected,
            });
        }
        Ok(info)
    }

    /// The operations, in program order.
    #[must_use]
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }

    /// The role of a declared tensor.
    #[must_use]
    pub fn role(&self, id: TensorId) -> Option<TensorRole> {
        self.tensors.get(&id).map(|t| t.role)
    }

    /// The element count of a declared tensor.
    #[must_use]
    pub fn elements(&self, id: TensorId) -> Option<u64> {
        self.tensors.get(&id).map(|t| t.elements)
    }

    /// Lowers the graph to the kernels that run it, fusing as `fusion` says,
    /// and derives every remaining tensor's lifetime from those kernels.
    ///
    /// The rules the result keeps are in the [module documentation](self#rules).
    ///
    /// # Errors
    ///
    /// A graph with no operations, an activation or output no operation
    /// writes, or a kernel whose baseline variant does not fit
    /// [`DeviceProfile::webgpu_baseline`].
    pub fn lower(&self, fusion: Fusion) -> Result<Program, KernelError> {
        if self.ops.is_empty() {
            return Err(KernelError::EmptyGraph);
        }
        if let Some((&id, _)) = self
            .tensors
            .iter()
            .find(|(id, t)| t.role.written_by_an_op() && !self.written.contains(id))
        {
            return Err(KernelError::NeverWritten(id));
        }

        // How many times each tensor is read, over every operation; an
        // operation that reads a tensor twice counts twice.
        let mut consumers: BTreeMap<TensorId, usize> = BTreeMap::new();
        for op in &self.ops {
            for (id, _) in op.kind.reads() {
                *consumers.entry(id).or_default() += 1;
            }
        }

        let mut kernels: Vec<KernelSpec> = Vec::new();
        for (index, op) in self.ops.iter().enumerate() {
            if fusion == Fusion::On {
                if let Some(last) = kernels.last_mut() {
                    if self.fuse(last, op, &consumers) {
                        last.ops.end = index + 1;
                        continue;
                    }
                }
            }
            let (main, epilogue) = op.kind.standalone();
            kernels.push(KernelSpec {
                main,
                epilogue,
                output: op.output,
                ops: index..index + 1,
            });
        }
        for spec in &kernels {
            spec.check(Variant::Baseline, &BASELINE)?;
        }
        Ok(Program::new(kernels, self))
    }

    /// Folds `op` into `kernel`'s epilogue when rule 2 of the module
    /// documentation allows it, and reports whether it did.
    fn fuse(
        &self,
        kernel: &mut KernelSpec,
        op: &Op,
        consumers: &BTreeMap<TensorId, usize>,
    ) -> bool {
        if matches!(kernel.main, MainOp::Map { .. }) {
            return false;
        }
        let chain = kernel.output;
        if self.role(chain) != Some(TensorRole::Activation)
            || consumers.get(&chain).copied() != Some(1)
        {
            return false;
        }
        let Some(step) = op.kind.as_epilogue(chain) else {
            return false;
        };
        let mut fused = kernel.clone();
        fused.epilogue.push(step);
        fused.output = op.output;
        let limit =
            usize::try_from(BASELINE.max_storage_buffers_per_shader_stage).unwrap_or(usize::MAX);
        if fused.bindings().len() > limit {
            return false;
        }
        *kernel = fused;
        true
    }
}

/// The heavy part of a kernel: what it computes before its epilogue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MainOp {
    /// A 3x3 convolution, one invocation per output element.
    Conv2d3x3(Conv2d3x3),
    /// A layer norm, one workgroup per row.
    LayerNorm(LayerNorm),
    /// A tiled matrix multiply, one workgroup per output tile.
    MatMul(MatMul),
    /// Reads `input` element by element: the main operation of a standalone
    /// element-wise kernel, whose epilogue is the operation itself.
    Map {
        /// The operand.
        input: TensorId,
        /// Elements of the operand and of the result.
        elements: u32,
    },
}

impl MainOp {
    /// The tensors the main operation reads, in binding order.
    fn reads(&self) -> Vec<TensorId> {
        match *self {
            Self::Conv2d3x3(c) => [Some(c.input), Some(c.weights), c.bias]
                .into_iter()
                .flatten()
                .collect(),
            Self::LayerNorm(l) => vec![l.input, l.gamma, l.beta],
            Self::MatMul(mm) => [Some(mm.a), Some(mm.b), mm.bias]
                .into_iter()
                .flatten()
                .collect(),
            Self::Map { input, .. } => vec![input],
        }
    }

    /// The element count of the result; fits [`MAX_ELEMENTS`] because the
    /// graph checked it.
    fn output_elements(&self) -> u64 {
        let mul = |a: u32, b: u32| u64::from(a).saturating_mul(u64::from(b));
        match *self {
            Self::Conv2d3x3(c) => mul(c.height, c.width).saturating_mul(u64::from(c.cout)),
            Self::LayerNorm(l) => mul(l.rows, l.channels),
            Self::MatMul(mm) => mul(mm.m, mm.n),
            Self::Map { elements, .. } => elements.into(),
        }
    }

    /// A short name for labels and file names.
    const fn name(&self) -> &'static str {
        match self {
            Self::Conv2d3x3(_) => "conv2d3x3",
            Self::LayerNorm(_) => "layernorm",
            Self::MatMul(_) => "matmul",
            Self::Map { .. } => "map",
        }
    }
}

/// One element-wise step applied to a kernel's value before it is stored.
///
/// Steps run in order on the value of each output element, at that element's
/// index.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Epilogue {
    /// Adds the element of this tensor at the same index; the tensor has as
    /// many elements as the kernel's output.
    Add(TensorId),
    /// `max(v, 0)`.
    Relu,
    /// GELU, tanh approximation, as [`OpKind::Gelu`].
    Gelu,
    /// `v * factor`; finite.
    Scale(f32),
}

impl Epilogue {
    /// A short name for labels and file names.
    const fn name(&self) -> &'static str {
        match self {
            Self::Add(_) => "add",
            Self::Relu => "relu",
            Self::Gelu => "gelu",
            Self::Scale(_) => "scale",
        }
    }
}

/// One way of running a kernel, with what it needs from a device.
///
/// Variants that need an optional feature are listed before the baseline in
/// [`Variant::PREFERENCE`]; the baseline needs none, so every spec has a
/// variant every adapter runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Variant {
    /// Needs no optional feature and fits the default WebGPU limits.
    Baseline,
}

impl Variant {
    /// Every variant, most capable first; [`KernelSpec::select_variant`] takes
    /// the first that passes [`KernelSpec::check`]. The baseline is last, so
    /// it is chosen only when nothing better runs.
    pub const PREFERENCE: [Self; 1] = [Self::Baseline];
}

/// What a kernel variant needs from a device, and how it is launched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Requirements {
    /// The workgroup size, x, y and z.
    pub workgroup_size: [u32; 3],
    /// The workgroup count of the dispatch, x, y and z. Work is numbered over
    /// the grid row by row, a row being `workgroups[0]` workgroups wide.
    pub workgroups: [u32; 3],
    /// Bytes of workgroup memory the kernel declares.
    pub workgroup_storage_bytes: u32,
    /// Storage buffers the kernel binds: one per distinct tensor.
    pub storage_buffers: u32,
    /// Optional features the kernel uses.
    pub features: DeviceFeatures,
}

/// A device limit a kernel can exceed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Limit {
    /// [`DeviceProfile::max_compute_workgroup_storage_size`].
    WorkgroupStorageSize,
    /// [`DeviceProfile::max_compute_invocations_per_workgroup`].
    InvocationsPerWorkgroup,
    /// [`DeviceProfile::max_compute_workgroup_size`], in x, y or z.
    WorkgroupSize,
    /// [`DeviceProfile::max_compute_workgroups_per_dimension`].
    WorkgroupsPerDimension,
    /// [`DeviceProfile::max_storage_buffers_per_shader_stage`].
    StorageBuffersPerShaderStage,
}

/// One kernel of a program: a main operation, its epilogue, and the tensors it
/// binds.
///
/// Made only by [`Graph::lower`], so its operations, shapes and tensors are the
/// graph's and already checked.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelSpec {
    main: MainOp,
    epilogue: Vec<Epilogue>,
    output: TensorId,
    ops: Range<usize>,
}

impl KernelSpec {
    /// The heavy part of the kernel.
    #[must_use]
    pub const fn main(&self) -> &MainOp {
        &self.main
    }

    /// The element-wise steps applied before the result is stored, in order.
    #[must_use]
    pub fn epilogue(&self) -> &[Epilogue] {
        &self.epilogue
    }

    /// The tensor the kernel writes: the last fused operation's output.
    #[must_use]
    pub const fn output(&self) -> TensorId {
        self.output
    }

    /// The graph operations the kernel runs, as a range of indices into
    /// [`Graph::ops`].
    #[must_use]
    pub fn ops(&self) -> Range<usize> {
        self.ops.clone()
    }

    /// The elements of the output.
    #[must_use]
    pub fn output_elements(&self) -> u64 {
        self.main.output_elements()
    }

    /// The tensors the kernel binds, each once, in binding order: the main
    /// operation's reads, then the epilogue's [`Epilogue::Add`] operands, then
    /// the output, which is always last.
    ///
    /// Binding `i` of the generated shader is element `i`.
    #[must_use]
    pub fn bindings(&self) -> Vec<TensorId> {
        let mut out: Vec<TensorId> = Vec::new();
        let added = self.epilogue.iter().filter_map(|e| match e {
            Epilogue::Add(id) => Some(*id),
            _ => None,
        });
        for id in self.main.reads().into_iter().chain(added) {
            if !out.contains(&id) {
                out.push(id);
            }
        }
        out.push(self.output);
        out
    }

    /// A short name naming the main operation and the epilogue, such as
    /// `matmul+add+scale`, for labels and file names.
    #[must_use]
    pub fn name(&self) -> String {
        let mut name = String::from(self.main.name());
        for step in &self.epilogue {
            name.push('+');
            name.push_str(step.name());
        }
        name
    }

    /// What `variant` of this kernel needs and how it is launched.
    #[must_use]
    pub fn requirements(&self, variant: Variant) -> Requirements {
        let Variant::Baseline = variant;
        let storage_buffers = u32::try_from(self.bindings().len()).unwrap_or(u32::MAX);
        let (workgroup_size, workgroups, workgroup_storage_bytes) = match self.main {
            MainOp::Conv2d3x3(_) | MainOp::Map { .. } => (
                [ELEMENT_GROUP, 1, 1],
                rows_of(self.output_elements().div_ceil(u64::from(ELEMENT_GROUP))),
                0,
            ),
            MainOp::LayerNorm(l) => (
                [LAYER_NORM_LANES, 1, 1],
                rows_of(u64::from(l.rows)),
                LAYER_NORM_LANES * F32_BYTES,
            ),
            MainOp::MatMul(mm) => (
                [MATMUL_TILE, MATMUL_TILE, 1],
                [mm.n.div_ceil(MATMUL_TILE), mm.m.div_ceil(MATMUL_TILE), 1],
                2 * MATMUL_TILE * MATMUL_TILE * F32_BYTES,
            ),
        };
        Requirements {
            workgroup_size,
            workgroups,
            workgroup_storage_bytes,
            storage_buffers,
            features: DeviceFeatures::default(),
        }
    }

    /// Checks that `variant` runs on a device described by `profile`: its
    /// workgroup memory, workgroup size, workgroup counts and bindings within
    /// the limits, and its features a subset of the device's.
    ///
    /// # Errors
    ///
    /// [`KernelError::ExceedsLimit`] for the first limit exceeded, or
    /// [`KernelError::MissingFeatures`].
    pub fn check(&self, variant: Variant, profile: &DeviceProfile) -> Result<(), KernelError> {
        let r = self.requirements(variant);
        let over = |limit, needed: u64, available: u64| {
            if needed > available {
                Err(KernelError::ExceedsLimit {
                    limit,
                    needed,
                    available,
                })
            } else {
                Ok(())
            }
        };
        over(
            Limit::WorkgroupStorageSize,
            r.workgroup_storage_bytes.into(),
            profile.max_compute_workgroup_storage_size.into(),
        )?;
        let invocations = r
            .workgroup_size
            .iter()
            .fold(1_u64, |acc, &d| acc.saturating_mul(u64::from(d)));
        over(
            Limit::InvocationsPerWorkgroup,
            invocations,
            profile.max_compute_invocations_per_workgroup.into(),
        )?;
        for (&size, &max) in r
            .workgroup_size
            .iter()
            .zip(&profile.max_compute_workgroup_size)
        {
            over(Limit::WorkgroupSize, size.into(), max.into())?;
        }
        for &count in &r.workgroups {
            over(
                Limit::WorkgroupsPerDimension,
                count.into(),
                profile.max_compute_workgroups_per_dimension.into(),
            )?;
        }
        over(
            Limit::StorageBuffersPerShaderStage,
            r.storage_buffers.into(),
            profile.max_storage_buffers_per_shader_stage.into(),
        )?;
        if !profile.features.contains(r.features) {
            return Err(KernelError::MissingFeatures {
                needed: r.features,
                available: profile.features,
            });
        }
        Ok(())
    }

    /// The most capable variant that runs on a device described by `profile`,
    /// in the order of [`Variant::PREFERENCE`].
    ///
    /// # Errors
    ///
    /// The baseline's [`check`](Self::check) error when no variant runs, which
    /// for a spec from [`Graph::lower`] means a profile below the WebGPU
    /// baseline.
    pub fn select_variant(&self, profile: &DeviceProfile) -> Result<Variant, KernelError> {
        let mut last = Ok(());
        for variant in Variant::PREFERENCE {
            last = self.check(variant, profile);
            if last.is_ok() {
                return Ok(variant);
            }
        }
        last.map(|()| Variant::Baseline)
    }
}

/// A one-dimensional count of workgroups as a grid of rows, each as wide as
/// the baseline allows in one dimension, so a count past that limit is still
/// launched. The row width is the grid's x.
fn rows_of(groups: u64) -> [u32; 3] {
    let width = u64::from(BASELINE.max_compute_workgroups_per_dimension).min(groups.max(1));
    let height = groups.max(1).div_ceil(width);
    let narrow = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
    [narrow(width), narrow(height), 1]
}

/// A lowered graph: the kernels to dispatch, in order, and every tensor they
/// use with its lifetime.
#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    kernels: Vec<KernelSpec>,
    tensors: Vec<TensorDesc>,
    roles: BTreeMap<TensorId, TensorRole>,
}

impl Program {
    /// Derives every tensor's lifetime from the kernels that use it (rule 1 of
    /// the [module documentation](self#rules)).
    fn new(kernels: Vec<KernelSpec>, graph: &Graph) -> Self {
        // `push` bounds the operation count by `u32`, and there are no more
        // kernels than operations.
        let index = |k: usize| u32::try_from(k).unwrap_or(u32::MAX);
        let last = index(kernels.len().saturating_sub(1));
        let mut uses: BTreeMap<TensorId, (u32, u32)> = BTreeMap::new();
        for (k, spec) in kernels.iter().enumerate() {
            for id in spec.bindings() {
                let k = index(k);
                uses.entry(id)
                    .and_modify(|(first, end)| {
                        *first = (*first).min(k);
                        *end = (*end).max(k);
                    })
                    .or_insert((k, k));
            }
        }
        let mut tensors = Vec::new();
        let mut roles = BTreeMap::new();
        for (&id, info) in &graph.tensors {
            let lifetime = if info.role.whole_program() {
                Some((0, last))
            } else {
                // An activation fusion removed is used by no kernel.
                uses.get(&id).copied()
            };
            if let Some((first, end)) = lifetime {
                tensors.push(TensorDesc::new(id, DType::F32, info.elements, first, end));
                roles.insert(id, info.role);
            }
        }
        Self {
            kernels,
            tensors,
            roles,
        }
    }

    /// The kernels, in dispatch order. Kernel `i` is operation `i` of the
    /// memory plan's program order.
    #[must_use]
    pub fn kernels(&self) -> &[KernelSpec] {
        &self.kernels
    }

    /// Every tensor a kernel uses, and every input, weight and output, with
    /// lifetimes in kernel indices, in id order: what
    /// [`MemoryPlan::new`](crate::memory::MemoryPlan::new) is given.
    #[must_use]
    pub fn tensors(&self) -> &[TensorDesc] {
        &self.tensors
    }

    /// The role of a tensor of the program; `None` for one fusion removed or
    /// the graph never declared.
    #[must_use]
    pub fn role(&self, id: TensorId) -> Option<TensorRole> {
        self.roles.get(&id).copied()
    }

    /// Describes tensor `id` as a stand-in for the input or output `of`: the
    /// same size, live for the whole program like `of`, so it can be bound in
    /// its place, as a host does for one input and output slot per job in
    /// flight.
    ///
    /// # Errors
    ///
    /// `of` is not an input or output of the program, or `id` is already one
    /// of its tensors.
    pub fn stand_in(&self, of: TensorId, id: TensorId) -> Result<TensorDesc, KernelError> {
        if self.roles.contains_key(&id) {
            return Err(KernelError::DuplicateTensor(id));
        }
        match (self.role(of), self.tensors.iter().find(|t| t.id == of)) {
            (Some(TensorRole::Input | TensorRole::Output), Some(desc)) => Ok(TensorDesc::new(
                id,
                desc.dtype,
                desc.elements,
                desc.first_use,
                desc.last_use,
            )),
            _ => Err(KernelError::NotInputOrOutput(of)),
        }
    }
}

/// Why a graph could not be built or lowered, or a kernel does not run on a
/// device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum KernelError {
    /// Two tensors share an id.
    #[error("tensor {} is declared more than once", .0.get())]
    DuplicateTensor(TensorId),
    /// A tensor of no elements has nothing to compute or bind.
    #[error("tensor {} has no elements", .0.get())]
    EmptyTensor(TensorId),
    /// A tensor larger than kernels can index.
    #[error("tensor {} has more than {MAX_ELEMENTS} elements", .0.get())]
    TensorTooLarge(TensorId),
    /// An operation names a tensor the graph does not declare.
    #[error("tensor {} is not declared", .0.get())]
    UnknownTensor(TensorId),
    /// A tensor's declared size disagrees with an operation's shape.
    #[error("tensor {} has {declared} elements, but operation {op} needs {expected}", tensor.get())]
    ElementCount {
        /// The tensor.
        tensor: TensorId,
        /// Its declared element count.
        declared: u64,
        /// The operation's index.
        op: usize,
        /// What the operation's shape gives.
        expected: u64,
    },
    /// An operation reads an activation or output no earlier operation wrote.
    #[error("operation {op} reads tensor {} before any operation writes it", tensor.get())]
    ReadBeforeWritten {
        /// The tensor.
        tensor: TensorId,
        /// The operation's index.
        op: usize,
    },
    /// An operation writes a tensor an earlier one wrote.
    #[error("operation {op} writes tensor {}, which an earlier operation wrote", tensor.get())]
    WrittenTwice {
        /// The tensor.
        tensor: TensorId,
        /// The operation's index.
        op: usize,
    },
    /// An operation writes an input or a weight.
    #[error("operation {op} writes tensor {}, an input or weight", tensor.get())]
    NotWritable {
        /// The tensor.
        tensor: TensorId,
        /// The operation's index.
        op: usize,
    },
    /// An operation has a dimension of zero.
    #[error("operation {op} has a zero dimension")]
    ZeroDimension {
        /// The operation's index.
        op: usize,
    },
    /// A shape whose element count passes [`MAX_ELEMENTS`].
    #[error("operation {op} has a tensor of more than {MAX_ELEMENTS} elements")]
    OpTooLarge {
        /// The operation's index.
        op: usize,
    },
    /// A scale by an infinity or a NaN.
    #[error("operation {op} scales by a non-finite factor")]
    NonFiniteScale {
        /// The operation's index.
        op: usize,
    },
    /// More operations than a kernel index counts.
    #[error("too many operations")]
    TooManyOps,
    /// A graph with no operations has no program.
    #[error("the graph has no operations")]
    EmptyGraph,
    /// An activation or output no operation writes.
    #[error("tensor {} is never written", .0.get())]
    NeverWritten(TensorId),
    /// A stand-in was asked for a tensor that is not an input or output.
    #[error("tensor {} is not an input or output of the program", .0.get())]
    NotInputOrOutput(TensorId),
    /// A kernel needs more of a limit than the device provides.
    #[error("the kernel needs {needed} of {limit:?}, the device provides {available}")]
    ExceedsLimit {
        /// The limit.
        limit: Limit,
        /// What the kernel needs.
        needed: u64,
        /// What the device provides.
        available: u64,
    },
    /// A kernel needs features the device was not created with.
    #[error("the kernel needs features {needed:?}, the device has {available:?}")]
    MissingFeatures {
        /// What the kernel needs.
        needed: DeviceFeatures,
        /// What the device has.
        available: DeviceFeatures,
    },
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::memory::MemoryPlan;

    fn id(n: u32) -> TensorId {
        TensorId::new(n)
    }

    const X: TensorId = TensorId::new(0);
    const W: TensorId = TensorId::new(1);
    const BIAS: TensorId = TensorId::new(2);
    const A: TensorId = TensorId::new(3);
    const B: TensorId = TensorId::new(4);
    const C: TensorId = TensorId::new(5);
    const Y: TensorId = TensorId::new(6);

    /// `[4][8] x [8][4] + bias`, the heavy operation of most tests.
    fn matmul(output: TensorId) -> (OpKind, TensorId) {
        (
            OpKind::MatMul(MatMul {
                a: X,
                b: W,
                bias: Some(BIAS),
                m: 4,
                k: 8,
                n: 4,
            }),
            output,
        )
    }

    /// A graph with the matmul's operands, `activations` as activations and
    /// `Y` as the output, all of 16 elements. Every activation must be
    /// written before the graph lowers.
    fn base(activations: &[TensorId]) -> Graph {
        let mut g = Graph::new();
        g.tensor(X, TensorRole::Input, 32).unwrap();
        g.tensor(W, TensorRole::Weight, 32).unwrap();
        g.tensor(BIAS, TensorRole::Weight, 4).unwrap();
        for &t in activations {
            g.tensor(t, TensorRole::Activation, 16).unwrap();
        }
        g.tensor(Y, TensorRole::Output, 16).unwrap();
        g
    }

    fn push(g: &mut Graph, (kind, out): (OpKind, TensorId)) {
        g.push(kind, out).unwrap();
    }

    fn relu(input: TensorId) -> OpKind {
        OpKind::Relu {
            input,
            elements: 16,
        }
    }

    fn lifetime(p: &Program, t: TensorId) -> Option<(u32, u32)> {
        p.tensors()
            .iter()
            .find(|d| d.id == t)
            .map(|d| (d.first_use, d.last_use))
    }

    #[test]
    fn an_element_wise_chain_after_a_heavy_op_becomes_its_epilogue() {
        let mut g = base(&[A, B]);
        push(&mut g, matmul(A));
        push(&mut g, (relu(A), B));
        push(
            &mut g,
            (
                OpKind::Scale {
                    input: B,
                    factor: 0.5,
                    elements: 16,
                },
                Y,
            ),
        );
        let p = g.lower(Fusion::On).unwrap();
        assert_eq!(p.kernels().len(), 1);
        let k = &p.kernels()[0];
        assert_eq!(k.epilogue(), &[Epilogue::Relu, Epilogue::Scale(0.5)]);
        assert_eq!(k.output(), Y);
        assert_eq!(k.ops(), 0..3);
        assert_eq!(k.bindings(), vec![X, W, BIAS, Y]);
        assert_eq!(k.name(), "matmul+relu+scale");
        // The intermediates are gone from the program and so from the plan.
        assert_eq!(lifetime(&p, A), None);
        assert_eq!(lifetime(&p, B), None);
        assert_eq!(p.role(A), None);
        let plan = MemoryPlan::new(p.tensors(), &DeviceProfile::webgpu_baseline()).unwrap();
        assert!(plan.placement(A).is_none() && plan.placement(Y).is_some());
    }

    #[test]
    fn fusion_off_gives_one_kernel_per_op_and_keeps_every_tensor() {
        let mut g = base(&[A, B]);
        push(&mut g, matmul(A));
        push(&mut g, (relu(A), B));
        push(
            &mut g,
            (
                OpKind::Gelu {
                    input: B,
                    elements: 16,
                },
                Y,
            ),
        );
        let p = g.lower(Fusion::Off).unwrap();
        assert_eq!(p.kernels().len(), 3);
        assert_eq!(
            p.kernels().iter().map(KernelSpec::name).collect::<Vec<_>>(),
            vec!["matmul", "map+relu", "map+gelu"]
        );
        // Lifetimes in kernel indices: written by one, read by the next.
        assert_eq!(lifetime(&p, A), Some((0, 1)));
        assert_eq!(lifetime(&p, B), Some((1, 2)));
        // Inputs, weights and outputs live for the whole program.
        for t in [X, W, BIAS, Y] {
            assert_eq!(lifetime(&p, t), Some((0, 2)), "{t:?}");
        }
    }

    #[test]
    fn an_intermediate_with_two_consumers_is_not_fused_through() {
        // A is read by the relu and by the add, so the relu may not swallow it.
        let mut g = base(&[A, B]);
        push(&mut g, matmul(A));
        push(&mut g, (relu(A), B));
        push(
            &mut g,
            (
                OpKind::Add {
                    a: A,
                    b: B,
                    elements: 16,
                },
                Y,
            ),
        );
        let p = g.lower(Fusion::On).unwrap();
        assert_eq!(p.kernels().len(), 3);
        assert_eq!(lifetime(&p, A), Some((0, 2)));
        assert_eq!(lifetime(&p, B), Some((1, 2)));
    }

    #[test]
    fn an_add_reading_the_same_tensor_twice_is_not_fused() {
        let mut g = base(&[A]);
        push(&mut g, matmul(A));
        push(
            &mut g,
            (
                OpKind::Add {
                    a: A,
                    b: A,
                    elements: 16,
                },
                Y,
            ),
        );
        assert_eq!(g.lower(Fusion::On).unwrap().kernels().len(), 2);
    }

    #[test]
    fn a_residual_add_is_fused_and_binds_its_other_operand() {
        // B is written by the first matmul and read by the add after the
        // second: the add fuses into the second and binds B.
        let mut g = base(&[A, B]);
        push(&mut g, matmul(B));
        push(&mut g, matmul(A));
        push(
            &mut g,
            (
                OpKind::Add {
                    a: B,
                    b: A,
                    elements: 16,
                },
                Y,
            ),
        );
        let p = g.lower(Fusion::On).unwrap();
        assert_eq!(p.kernels().len(), 2);
        assert_eq!(p.kernels()[1].epilogue(), &[Epilogue::Add(B)]);
        assert_eq!(p.kernels()[1].bindings(), vec![X, W, BIAS, B, Y]);
        assert_eq!(lifetime(&p, B), Some((0, 1)));
        assert_eq!(lifetime(&p, A), None);
    }

    #[test]
    fn an_element_wise_op_after_a_standalone_one_is_not_fused() {
        // The add of two inputs is a kernel of its own (no heavy op before
        // it), and the relu after it stays one too.
        let mut g = Graph::new();
        g.tensor(X, TensorRole::Input, 16).unwrap();
        g.tensor(W, TensorRole::Input, 16).unwrap();
        g.tensor(A, TensorRole::Activation, 16).unwrap();
        g.tensor(Y, TensorRole::Output, 16).unwrap();
        push(
            &mut g,
            (
                OpKind::Add {
                    a: X,
                    b: W,
                    elements: 16,
                },
                A,
            ),
        );
        push(&mut g, (relu(A), Y));
        assert_eq!(g.lower(Fusion::On).unwrap().kernels().len(), 2);
    }

    #[test]
    fn an_op_whose_input_came_from_an_earlier_kernel_is_not_fused() {
        // matmul -> A, matmul -> B, relu(A): A's producer is not the kernel
        // just before the relu.
        let mut g = base(&[A, B, C]);
        push(&mut g, matmul(A));
        push(&mut g, matmul(B));
        push(&mut g, (relu(A), C));
        push(
            &mut g,
            (
                OpKind::Add {
                    a: B,
                    b: C,
                    elements: 16,
                },
                Y,
            ),
        );
        let p = g.lower(Fusion::On).unwrap();
        assert_eq!(
            p.kernels().iter().map(KernelSpec::name).collect::<Vec<_>>(),
            vec!["matmul", "matmul", "map+relu", "map+add"]
        );
    }

    #[test]
    fn a_graph_output_is_never_fused_away() {
        let mut g = base(&[]);
        g.tensor(id(30), TensorRole::Output, 16).unwrap();
        push(&mut g, matmul(Y));
        push(&mut g, (relu(Y), id(30)));
        let p = g.lower(Fusion::On).unwrap();
        assert_eq!(p.kernels().len(), 2);
        assert!(lifetime(&p, Y).is_some());
    }

    #[test]
    fn fusion_stops_at_the_baselines_storage_buffer_limit() {
        // The matmul binds three tensors and its output; each fused add of a
        // new weight binds one more. The baseline allows eight, so the fifth
        // add (the ninth binding) starts a kernel of its own, and the relu
        // after that standalone kernel is a third.
        let mut g = base(&[A]);
        let addends: Vec<TensorId> = (40..45).map(id).collect();
        let chain: Vec<TensorId> = (50..55).map(id).collect();
        for &t in &addends {
            g.tensor(t, TensorRole::Weight, 16).unwrap();
        }
        for &t in &chain {
            g.tensor(t, TensorRole::Activation, 16).unwrap();
        }
        push(&mut g, matmul(A));
        let mut prev = A;
        for (&w, &out) in addends.iter().zip(&chain) {
            push(
                &mut g,
                (
                    OpKind::Add {
                        a: prev,
                        b: w,
                        elements: 16,
                    },
                    out,
                ),
            );
            prev = out;
        }
        push(&mut g, (relu(prev), Y));
        let p = g.lower(Fusion::On).unwrap();
        let limit = BASELINE.max_storage_buffers_per_shader_stage as usize;
        assert!(p.kernels().iter().all(|k| k.bindings().len() <= limit));
        assert_eq!(p.kernels()[0].bindings().len(), limit);
        assert_eq!(p.kernels()[0].epilogue().len(), 4);
        assert_eq!(
            p.kernels().iter().map(KernelSpec::name).collect::<Vec<_>>(),
            vec!["matmul+add+add+add+add", "map+add", "map+relu"]
        );
    }

    #[test]
    fn lowering_is_deterministic() {
        let build = || {
            let mut g = base(&[A, B]);
            push(&mut g, matmul(A));
            push(&mut g, (relu(A), B));
            push(
                &mut g,
                (
                    OpKind::Gelu {
                        input: B,
                        elements: 16,
                    },
                    Y,
                ),
            );
            g
        };
        for fusion in [Fusion::On, Fusion::Off] {
            assert_eq!(build().lower(fusion), build().lower(fusion));
        }
    }

    #[test]
    fn every_lowered_kernel_has_a_baseline_variant_that_needs_no_feature() {
        let mut g = Graph::new();
        g.tensor(X, TensorRole::Input, 7 * 9 * 3).unwrap();
        g.tensor(W, TensorRole::Weight, 9 * 3 * 5).unwrap();
        g.tensor(BIAS, TensorRole::Weight, 5).unwrap();
        g.tensor(A, TensorRole::Activation, 7 * 9 * 5).unwrap();
        g.tensor(B, TensorRole::Weight, 5).unwrap();
        g.tensor(C, TensorRole::Weight, 5).unwrap();
        g.tensor(Y, TensorRole::Output, 7 * 9 * 5).unwrap();
        push(
            &mut g,
            (
                OpKind::Conv2d3x3(Conv2d3x3 {
                    input: X,
                    weights: W,
                    bias: Some(BIAS),
                    height: 7,
                    width: 9,
                    cin: 3,
                    cout: 5,
                }),
                A,
            ),
        );
        push(
            &mut g,
            (
                OpKind::LayerNorm(LayerNorm {
                    input: A,
                    gamma: B,
                    beta: C,
                    rows: 63,
                    channels: 5,
                }),
                Y,
            ),
        );
        let baseline = DeviceProfile::webgpu_baseline();
        for fusion in [Fusion::On, Fusion::Off] {
            for k in g.lower(fusion).unwrap().kernels() {
                let r = k.requirements(Variant::Baseline);
                assert_eq!(r.features, DeviceFeatures::default());
                assert_eq!(k.check(Variant::Baseline, &baseline), Ok(()));
                assert_eq!(k.select_variant(&baseline), Ok(Variant::Baseline));
            }
        }
    }

    #[test]
    fn a_device_below_a_kernels_needs_is_refused() {
        let mut g = base(&[]);
        push(&mut g, matmul(Y));
        let p = g.lower(Fusion::On).unwrap();
        let k = &p.kernels()[0];
        let r = k.requirements(Variant::Baseline);
        assert_eq!(r.workgroup_size, [16, 16, 1]);
        assert_eq!(r.workgroups, [1, 1, 1]);
        assert_eq!(r.workgroup_storage_bytes, 2048);

        let mut small = DeviceProfile::webgpu_baseline();
        small.max_compute_workgroup_storage_size = 1024;
        assert_eq!(
            k.check(Variant::Baseline, &small),
            Err(KernelError::ExceedsLimit {
                limit: Limit::WorkgroupStorageSize,
                needed: 2048,
                available: 1024
            })
        );
        assert!(k.select_variant(&small).is_err());
        let mut few = DeviceProfile::webgpu_baseline();
        few.max_storage_buffers_per_shader_stage = 3;
        assert!(matches!(
            k.check(Variant::Baseline, &few),
            Err(KernelError::ExceedsLimit {
                limit: Limit::StorageBuffersPerShaderStage,
                ..
            })
        ));
    }

    #[test]
    fn a_long_element_wise_kernel_is_launched_as_rows_of_workgroups() {
        let elements = 300 * 65_535_u32;
        let mut g = Graph::new();
        g.tensor(X, TensorRole::Input, elements.into()).unwrap();
        g.tensor(Y, TensorRole::Output, elements.into()).unwrap();
        push(&mut g, (OpKind::Relu { input: X, elements }, Y));
        let p = g.lower(Fusion::On).unwrap();
        let r = p.kernels()[0].requirements(Variant::Baseline);
        let groups = u64::from(elements).div_ceil(256);
        assert_eq!(r.workgroups[0], 65_535);
        assert_eq!(u64::from(r.workgroups[1]), groups.div_ceil(65_535));
        assert!(u64::from(r.workgroups[0]) * u64::from(r.workgroups[1]) >= groups);
    }

    #[test]
    fn a_stand_in_has_the_size_and_lifetime_of_what_it_stands_in_for() {
        let mut g = base(&[A]);
        push(&mut g, matmul(A));
        push(&mut g, (relu(A), Y));
        let p = g.lower(Fusion::Off).unwrap();
        let s = p.stand_in(Y, id(99)).unwrap();
        assert_eq!(
            (s.id, s.elements, s.first_use, s.last_use),
            (id(99), 16, 0, 1)
        );
        assert_eq!(p.stand_in(X, id(98)).unwrap().elements, 32);
        assert_eq!(p.stand_in(A, id(97)), Err(KernelError::NotInputOrOutput(A)));
        assert_eq!(p.stand_in(W, id(97)), Err(KernelError::NotInputOrOutput(W)));
        assert_eq!(p.stand_in(Y, X), Err(KernelError::DuplicateTensor(X)));
    }

    #[test]
    fn malformed_graphs_are_errors() {
        let mut g = base(&[A, B]);
        assert_eq!(
            g.tensor(X, TensorRole::Input, 4),
            Err(KernelError::DuplicateTensor(X))
        );
        assert_eq!(
            g.tensor(id(9), TensorRole::Input, 0),
            Err(KernelError::EmptyTensor(id(9)))
        );
        assert_eq!(
            g.tensor(id(9), TensorRole::Input, MAX_ELEMENTS + 1),
            Err(KernelError::TensorTooLarge(id(9)))
        );
        assert_eq!(g.lower(Fusion::On), Err(KernelError::EmptyGraph));
        assert_eq!(
            g.push(relu(id(77)), A),
            Err(KernelError::UnknownTensor(id(77)))
        );
        assert_eq!(
            g.push(relu(A), B),
            Err(KernelError::ReadBeforeWritten { tensor: A, op: 0 })
        );
        assert_eq!(
            g.push(
                OpKind::Relu {
                    input: X,
                    elements: 32
                },
                W
            ),
            Err(KernelError::NotWritable { tensor: W, op: 0 })
        );
        assert!(matches!(
            g.push(
                OpKind::Relu {
                    input: X,
                    elements: 32
                },
                A
            ),
            Err(KernelError::ElementCount {
                tensor: A,
                declared: 16,
                op: 0,
                expected: 32
            })
        ));
        assert_eq!(
            g.push(
                OpKind::Scale {
                    input: A,
                    factor: f32::NAN,
                    elements: 16
                },
                B
            ),
            Err(KernelError::NonFiniteScale { op: 0 })
        );
        assert_eq!(
            g.push(
                OpKind::Relu {
                    input: X,
                    elements: 0
                },
                B
            ),
            Err(KernelError::ZeroDimension { op: 0 })
        );
        push(&mut g, matmul(A));
        assert_eq!(
            g.push(matmul(A).0, A),
            Err(KernelError::WrittenTwice { tensor: A, op: 1 })
        );
        // Reading its own output is a read before the write.
        assert_eq!(
            g.push(relu(B), B),
            Err(KernelError::ReadBeforeWritten { tensor: B, op: 1 })
        );
        // B, C and Y are declared and nobody writes them.
        assert_eq!(g.lower(Fusion::On), Err(KernelError::NeverWritten(B)));
    }
}
