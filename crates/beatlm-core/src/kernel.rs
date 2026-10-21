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
//! # Why splitting
//!
//! A GPU cannot be preempted in the middle of a dispatch, so one long dispatch
//! delays everything queued behind it, the frame's render pass included, and
//! the scheduler can place work only at dispatch boundaries. Fusion makes
//! dispatches larger; splitting is the other direction, and the two are
//! traded against each other: every part is a dispatch the host encodes and
//! submits.
//!
//! The rule that one dispatch stays within a bound is owned by
//! [`Program::parts_within`], which turns what each kernel took into a part
//! count, and carried out by [`Program::split`] (rule 6). The decision reads
//! measurements, not shapes: a host measures each kernel run whole, picks a
//! bound, and lowers again with the parts it was given.
//!
//! What it bought on one machine is measured by `beatlm-bench --workload net`
//! with splitting on (a bound of 1 ms) and off; the table, with its adapter,
//! configuration and date, and the sweep that chose the bound, live in that
//! crate's documentation under "Splitting". In short, on an Apple M1 Pro,
//! Metal, on 2026-10-21: calibration split the second matrix multiply, which
//! took about 2.2 ms whole in the loop, into four parts of about 0.6 ms each,
//! and the first convolution into three, so a job went from six dispatches to
//! eleven or twelve. The longest dispatch per frame fell from about 2.1 ms to
//! 1.2 to 1.4 ms at the median and from 3.5 to 4.1 ms to about 2.9 ms at the
//! 99th percentile, but stayed above the bound at the median, so in these runs
//! the bound held for the parts' means and not for every dispatch. The median
//! encode time rose with the dispatch count; the GPU time per frame and the
//! delivery rate did not move beyond the spread between two runs.
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
//!    [`OpKind::Gelu`], [`OpKind::GeluErf`], [`OpKind::Scale`],
//!    [`OpKind::ChannelScale`]) becomes part of the previous kernel's
//!    [epilogue](Epilogue) when that kernel's main operation is a heavy
//!    operation (any [`MainOp`] but [`MainOp::Map`], the standalone
//!    element-wise kernel), the element-wise
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
//!    a weight, an activation or output nobody writes, a zero dimension, a
//!    non-finite scale and a layer norm epsilon that is not finite and positive
//!    are each a [`KernelError`].
//! 6. **A split kernel's parts cover its output exactly once, as consecutive
//!    kernels.** [`Program::split`] replaces a kernel by `k` kernels, its
//!    [`Part`]s, each computing a disjoint, contiguous range of the output
//!    ([`KernelSpec::output_range`]) with the kernel's whole epilogue applied
//!    to that range, and together every output element once. `k` is clamped
//!    to `1..=`[`KernelSpec::max_parts`], so no part is empty. Every part binds
//!    the tensors the kernel bound, and the parts take the kernel's place in
//!    program order, one after another. What that does to lifetimes follows
//!    from rule 1 with nothing added: a tensor the kernel read or wrote is live
//!    across all `k` parts, so nothing shares its bytes while any part runs; a
//!    tensor the parts write is first used by part 1, and every later reader
//!    is a later kernel, so it reads only after all `k` parts. A tensor read
//!    by the kernel and dying there now dies at its last part. The memory plan
//!    of a split program is therefore planned by the same rules, over longer
//!    lifetimes in more kernel indices. Splitting changes no result beyond
//!    what rule 3 allows a shader compiler: each output element is computed by
//!    the same operations, in the same order, as in the whole kernel.
//!
//! # Layouts
//!
//! Every tensor is `f32` and row-major. Activations are NHWC with one image
//! (`[H][W][C]`, channels contiguous), convolution weights are HWIO
//! (`[3][3][Cin][Cout]`, `[P][P][Cin][Cout]` for a patch embedding), and a
//! layer norm and a softmax normalise the last dimension. A sequence of tokens
//! is `[tokens][channels]`; patch tokens are in row-major patch order, so the
//! tokens of an `H x W` grid of patches are the NHWC map `[H][W][C]` as they
//! are. Attention reads the queries, keys and values packed per token,
//! `[tokens][3][heads][head_dim]`, and lays scores out query first,
//! `[tokens][heads][tokens]`.
//!
//! # Vision transformer operations
//!
//! A ViT encoder (DINOv2, the backbone of Depth Anything V2) is these
//! operations. How each fuses and splits follows from rules 2 and 6; the table
//! says what that comes to, so it can be checked against the tests.
//!
//! | Operation | Fuses into the kernel before it | Takes epilogues | Splits along |
//! |---|---|---|---|
//! | [`PatchEmbed`] | no: heavy | yes | patches, in tile rows |
//! | [`PrependToken`] | no: its output index is not its input's | yes | output elements |
//! | position embedding: [`OpKind::Add`] of a weight | yes, as [`Epilogue::Add`] | — | — |
//! | [`AttentionScores`] | no: heavy | yes (the `1 / sqrt(head_dim)` [`OpKind::Scale`]) | query rows, in tile rows |
//! | [`Softmax`] | no: a row reduction | yes | rows, which are (query, head) pairs, query first |
//! | [`AttentionValues`] | no: heavy | yes | query rows, in tile rows |
//! | [`OpKind::GeluErf`] | yes, as [`Epilogue::GeluErf`] | — | — |
//! | [`OpKind::ChannelScale`] (LayerScale) | yes, as [`Epilogue::ChannelScale`] | — | — |
//! | [`TokensToMap`] | no: its output index is not its input's | yes | output elements |
//!
//! An element-wise operation that does not fuse (its input has a second
//! consumer, or no heavy kernel precedes it) is a standalone kernel and splits
//! along its elements, as any other.
//!
//! ## Attention: materialised scores
//!
//! Self-attention is five operations: one [`MatMul`] from the normalised
//! tokens to the packed queries, keys and values (`384 -> 1152` for DINOv2
//! Small, one matrix multiply rather than three, so the normalised tokens are
//! read once and the kernel is one dispatch), [`AttentionScores`] with the
//! scale fused, [`Softmax`], [`AttentionValues`], and the output projection, a
//! [`MatMul`] that LayerScale and the residual add fuse into. The scores are
//! stored in memory between the three middle kernels, rather than kept in
//! registers by one kernel that computes the softmax online, block by block of
//! keys.
//!
//! They fit the baseline. At 1370 tokens and 6 heads the scores are
//! `1370 x 6 x 1370` `f32`s, 45 045 600 bytes (7.5 MB per head), below the
//! baseline's 128 MiB per binding, so the tensor is one binding as rule 3 of
//! the [memory plan](crate::memory#rules) requires. The softmax writes a second
//! tensor of that size while the first is live, so 90 MB of scores are live at
//! that kernel and the planner places them like any other two tensors. No
//! kernel's workgroup memory depends on the sequence length: the two products
//! stage 16 x 16 tiles (2 KiB) and the softmax one value per invocation
//! (1 KiB), against the baseline's 16 KiB.
//!
//! Why materialise, in terms of the frame-budget design ([`Program::parts_within`]
//! and the crate documentation's scheduler rules): work is placed in frames at
//! dispatch boundaries, and a kernel whose measured time exceeds the bound is
//! split into parts. Three kernels give the cost model three measurements and
//! the scheduler three boundaries, each split along query rows on its own
//! measurement; an online-softmax kernel would be the longest dispatch of a
//! block and would be split along the same query rows anyway, which makes its
//! parts dispatches as well. What materialising costs is two dispatches per
//! block (each a share of the host's encode time, which is where the frame's
//! CPU-side variance lives) and the traffic of writing and reading the scores:
//! about 180 MB per block at these shapes, against about 7.7 GFLOP of
//! arithmetic per block. It is also the decomposition each of whose kernels has
//! a reference of its own.
//!
//! What the traffic costs was measured with DINOv2 Small's real weights at 518
//! pixels by the `beatlm` crate's test
//! `the_encoder_on_the_gpu_matches_the_reference_features`, on an Apple M1 Pro,
//! Metal, on 2026-10-21, fused and unsplit: the encoder's 118 kernels took 715
//! to 730 ms of GPU time per run (the sum of every pass's timestamps; two
//! invocations of the test, five runs each). In the first, by per-kernel
//! median, the twelve softmax kernels, which read and write the full scores,
//! took 10 ms of it; the twelve score products 135 ms, the twelve weighted
//! sums 130 ms and the other matrix multiplies 433 ms. The scores' traffic is
//! therefore not where the time is; the tiled products are, and an
//! online-softmax kernel would run the same two products. The decision is
//! expected to move if, once the products are faster, the softmax's share
//! grows to matter: an online-softmax kernel removes the 90 MB and the two
//! dispatches and keeps every other operation as it is.

use std::collections::{BTreeMap, BTreeSet};
use std::ops::Range;

use thiserror::Error;

use crate::device::{DeviceFeatures, DeviceProfile};
use crate::memory::{DType, TensorDesc, TensorId};
use crate::time::Micros;

/// The epsilon `torch.nn.LayerNorm` adds to the variance by default, for a
/// [`LayerNorm`] whose model states none. Each layer norm carries its own
/// ([`LayerNorm::epsilon`]), because models differ: DINOv2 uses `1e-6`.
pub const LAYER_NORM_EPSILON: f32 = 1e-5;

/// `p` of the erf approximation an [`OpKind::GeluErf`] kernel evaluates.
///
/// WGSL has no `erf`, so a kernel computes it as Abramowitz and Stegun,
/// *Handbook of Mathematical Functions*, formula 7.1.26: for `z >= 0`,
/// `erfc(z) = (a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5) exp(-z^2)` with
/// `t = 1 / (1 + p z)`, [`GELU_ERF_A`] holding `a1` to `a5`. The book bounds
/// the formula's absolute error by `1.5e-7` for every `z >= 0`.
///
/// GELU is then `x * Phi(x)` with `Phi(x) = erfc(|x| / sqrt 2) / 2` for
/// negative `x` and `1 - erfc(x / sqrt 2) / 2` otherwise, so the negative side
/// is a product, not `1 + erf` of a number near `-1`, and loses nothing to
/// cancellation. Evaluated in `f32` as a kernel does, its largest absolute
/// error against GELU through an `f64` erf, over `[-10, 10]` in steps of
/// `2^-12`, is [`GELU_ERF_MAX_ERROR`]; that bound is asserted by
/// `beatlm-bench`'s CPU test `the_erf_approximation_is_within_its_stated_error`,
/// and what a GPU's `exp` adds is measured there by its GPU test
/// `gelu_erf_on_the_gpu_is_within_its_stated_error`.
///
/// The constants are here, and not only in the shader text, so that the CPU
/// test evaluates the same numbers the generated kernel does.
pub const GELU_ERF_P: f32 = 0.327_591_1;

/// `a1` to `a5` of the erf approximation; see [`GELU_ERF_P`].
pub const GELU_ERF_A: [f32; 5] = [
    0.254_829_6,
    -0.284_496_72,
    1.421_413_8,
    -1.453_152_1,
    1.061_405_4,
];

/// The largest absolute error of [`OpKind::GeluErf`] evaluated in `f32`
/// against GELU through an `f64` erf, as stated at [`GELU_ERF_P`].
///
/// Measured: `4.13e-7`, at `x = 3.232`, both in `f32` on the CPU and through
/// the generated kernel on an Apple M1 Pro (Metal), on 2026-10-21; the bound
/// leaves a fifth above that for a GPU whose `exp` rounds differently.
pub const GELU_ERF_MAX_ERROR: f32 = 5e-7;

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
/// Invocations per workgroup of a softmax, one workgroup per row as for a layer
/// norm. Wider, because an attention row is the sequence length (1370 for
/// DINOv2 at 518 pixels) rather than a channel count; a power of two for the
/// same reason.
const SOFTMAX_LANES: u32 = 256;
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
/// `(x - mean) / sqrt(var + epsilon) * gamma + beta` per row.
///
/// Reads `input` (`[rows][channels]`), `gamma` and `beta` (`[channels]`);
/// writes `[rows][channels]`.
#[derive(Debug, Clone, Copy, PartialEq)]
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
    /// Added to the variance before the square root, so a constant row does
    /// not divide by zero; finite and positive. The model says which:
    /// [`LAYER_NORM_EPSILON`] is PyTorch's default.
    pub epsilon: f32,
}

/// A patch embedding: a convolution of one NHWC image whose kernel and stride
/// are both `patch`, without padding, written as one token per patch.
///
/// Reads `input` (`[grid_h * patch][grid_w * patch][cin]`), `weights`
/// (`[patch][patch][cin][cout]`, HWIO) and, when present, `bias` (`[cout]`);
/// writes `[grid_h * grid_w][cout]`, patches in row-major order, which is the
/// NHWC map `[grid_h][grid_w][cout]` as well. The image is a whole number of
/// patches by construction, so no pixel is dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PatchEmbed {
    /// The image.
    pub input: TensorId,
    /// The HWIO weights.
    pub weights: TensorId,
    /// One value per output channel added to every token, or none.
    pub bias: Option<TensorId>,
    /// Patches down the image.
    pub grid_h: u32,
    /// Patches across the image.
    pub grid_w: u32,
    /// Side of a patch in pixels, and the stride.
    pub patch: u32,
    /// Input channels.
    pub cin: u32,
    /// Output channels: the token width.
    pub cout: u32,
}

/// A token in front of a sequence: `[1 + rows][channels]`, `token` as row 0
/// and `tokens` after it. A vision transformer's class token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrependToken {
    /// The token put first, `[channels]`.
    pub token: TensorId,
    /// The sequence, `[rows][channels]`.
    pub tokens: TensorId,
    /// Rows of `tokens`.
    pub rows: u32,
    /// Width of a token.
    pub channels: u32,
}

/// A sequence of `1 + height * width` tokens without its first one, as the
/// NHWC map `[height][width][channels]`: how a vision transformer's patch
/// tokens leave it with the class token dropped.
///
/// Reads `input` (`[1 + height * width][channels]`). Patch tokens are in
/// row-major patch order, so the map is rows `1..` of the input as they are;
/// the operation is that copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokensToMap {
    /// The sequence, class token first.
    pub input: TensorId,
    /// Rows of the map.
    pub height: u32,
    /// Columns of the map.
    pub width: u32,
    /// Width of a token, and channels of the map.
    pub channels: u32,
}

/// The attention scores of multi-head self-attention, unscaled:
/// `S[i][h][j] = sum over d of Q[i][h][d] * K[j][h][d]`.
///
/// Reads `qkv` (`[tokens][3 * heads * head_dim]`, each row the query, then the
/// key, then the value of one token, each `[heads][head_dim]`); writes
/// `[tokens][heads][tokens]`: query, head, key. The layout puts the query
/// outermost, so a range of query rows is a contiguous range of the scores and
/// the kernel splits along them (rule 6). The `1 / sqrt(head_dim)` scale is an
/// [`OpKind::Scale`] after it, which fuses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttentionScores {
    /// The packed queries, keys and values.
    pub qkv: TensorId,
    /// Sequence length.
    pub tokens: u32,
    /// Attention heads.
    pub heads: u32,
    /// Width of one head.
    pub head_dim: u32,
}

/// A softmax over the last dimension, `exp(x - max) / sum exp(x - max)` per
/// row: subtracting the row's maximum first keeps every `exp` at most one, so
/// no score overflows.
///
/// Reads `input` (`[rows][cols]`); writes `[rows][cols]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Softmax {
    /// The rows.
    pub input: TensorId,
    /// How many rows.
    pub rows: u32,
    /// The length of a row: what is normalised over.
    pub cols: u32,
}

/// The weighted sum of the values of multi-head self-attention:
/// `O[i][h][d] = sum over j of P[i][h][j] * V[j][h][d]`.
///
/// Reads `probs` (`[tokens][heads][tokens]`, as [`AttentionScores`] lays them
/// out) and `qkv` (as [`AttentionScores`] reads it, of which it uses the
/// values); writes `[tokens][heads * head_dim]`, the heads side by side as the
/// output projection reads them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AttentionValues {
    /// The attention weights, each row summing to one.
    pub probs: TensorId,
    /// The packed queries, keys and values.
    pub qkv: TensorId,
    /// Sequence length.
    pub tokens: u32,
    /// Attention heads.
    pub heads: u32,
    /// Width of one head.
    pub head_dim: u32,
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
/// The variants holding a struct are heavy operations, each the main operation
/// of a kernel of its own; the rest are element-wise and may be fused into the
/// kernel before them (rule 2 of the [module documentation](self#rules)).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OpKind {
    /// A 3x3 convolution.
    Conv2d3x3(Conv2d3x3),
    /// A layer norm over channels.
    LayerNorm(LayerNorm),
    /// A matrix multiply.
    MatMul(MatMul),
    /// A patch embedding.
    PatchEmbed(PatchEmbed),
    /// A token put in front of a sequence.
    PrependToken(PrependToken),
    /// A sequence without its first token, as a map.
    TokensToMap(TokensToMap),
    /// Multi-head attention scores.
    AttentionScores(AttentionScores),
    /// A softmax over rows.
    Softmax(Softmax),
    /// Multi-head attention's weighted sum of the values.
    AttentionValues(AttentionValues),
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
    /// GELU as defined, `x * Phi(x) = 0.5 x (1 + erf(x / sqrt 2))`, which is
    /// what `torch.nn.GELU()` computes by default and DINOv2's MLP uses. erf is
    /// approximated within [`GELU_ERF_MAX_ERROR`]; [`GELU_ERF_P`] says how.
    GeluErf {
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
    /// Element `i` times `scale[i % channels]`: a per-channel multiplier, as a
    /// vision transformer's LayerScale applies to a residual branch.
    ChannelScale {
        /// The operand, `[rows][channels]`.
        input: TensorId,
        /// The multipliers, `[channels]`.
        scale: TensorId,
        /// Rows of the operand.
        rows: u32,
        /// Channels: the operand's last dimension.
        channels: u32,
    },
}

/// The product of `dims`, saturating, so an overflow reads as more than
/// [`MAX_ELEMENTS`].
fn product(dims: &[u32]) -> u64 {
    dims.iter()
        .fold(1_u64, |acc, &d| acc.saturating_mul(u64::from(d)))
}

/// Elements of the packed queries, keys and values attention reads.
fn qkv_elements(tokens: u32, heads: u32, head_dim: u32) -> u64 {
    product(&[tokens, 3, heads, head_dim])
}

impl OpKind {
    /// Every tensor the operation reads, in binding order, with the element
    /// count its shape gives it. Products saturate, so an overflow reads as
    /// more than [`MAX_ELEMENTS`].
    fn reads(&self) -> Vec<(TensorId, u64)> {
        let mul = product;
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
            Self::PatchEmbed(p) => {
                let mut r = vec![
                    (p.input, mul(&[p.grid_h, p.patch, p.grid_w, p.patch, p.cin])),
                    (p.weights, mul(&[p.patch, p.patch, p.cin, p.cout])),
                ];
                r.extend(p.bias.map(|b| (b, u64::from(p.cout))));
                r
            }
            Self::PrependToken(t) => vec![
                (t.token, t.channels.into()),
                (t.tokens, mul(&[t.rows, t.channels])),
            ],
            Self::TokensToMap(t) => vec![(
                t.input,
                mul(&[t.height, t.width])
                    .saturating_add(1)
                    .saturating_mul(t.channels.into()),
            )],
            Self::AttentionScores(s) => vec![(s.qkv, qkv_elements(s.tokens, s.heads, s.head_dim))],
            Self::Softmax(s) => vec![(s.input, mul(&[s.rows, s.cols]))],
            Self::AttentionValues(v) => vec![
                (v.probs, mul(&[v.tokens, v.heads, v.tokens])),
                (v.qkv, qkv_elements(v.tokens, v.heads, v.head_dim)),
            ],
            Self::Add { a, b, elements } => vec![(a, elements.into()), (b, elements.into())],
            Self::Relu { input, elements }
            | Self::Gelu { input, elements }
            | Self::GeluErf { input, elements }
            | Self::Scale {
                input, elements, ..
            } => vec![(input, elements.into())],
            Self::ChannelScale {
                input,
                scale,
                rows,
                channels,
            } => vec![(input, mul(&[rows, channels])), (scale, channels.into())],
        }
    }

    /// The element count of the result, saturating, so an overflow reads as
    /// more than [`MAX_ELEMENTS`]. A graph builder declares the tensor an
    /// operation writes with it.
    #[must_use]
    pub fn output_elements(&self) -> u64 {
        match *self {
            Self::Conv2d3x3(c) => product(&[c.height, c.width, c.cout]),
            Self::LayerNorm(l) => product(&[l.rows, l.channels]),
            Self::MatMul(mm) => product(&[mm.m, mm.n]),
            Self::PatchEmbed(p) => product(&[p.grid_h, p.grid_w, p.cout]),
            Self::PrependToken(t) => u64::from(t.rows)
                .saturating_add(1)
                .saturating_mul(t.channels.into()),
            Self::TokensToMap(t) => product(&[t.height, t.width, t.channels]),
            Self::AttentionScores(s) => product(&[s.tokens, s.heads, s.tokens]),
            Self::Softmax(s) => product(&[s.rows, s.cols]),
            Self::AttentionValues(v) => product(&[v.tokens, v.heads, v.head_dim]),
            Self::Add { elements, .. }
            | Self::Relu { elements, .. }
            | Self::Gelu { elements, .. }
            | Self::GeluErf { elements, .. }
            | Self::Scale { elements, .. } => elements.into(),
            Self::ChannelScale { rows, channels, .. } => product(&[rows, channels]),
        }
    }

    /// Whether any dimension of the operation is zero.
    fn has_zero_dimension(&self) -> bool {
        match *self {
            Self::Conv2d3x3(c) => [c.height, c.width, c.cin, c.cout].contains(&0),
            Self::LayerNorm(l) => l.rows == 0 || l.channels == 0,
            Self::MatMul(mm) => [mm.m, mm.k, mm.n].contains(&0),
            Self::PatchEmbed(p) => [p.grid_h, p.grid_w, p.patch, p.cin, p.cout].contains(&0),
            Self::PrependToken(t) => t.rows == 0 || t.channels == 0,
            Self::TokensToMap(t) => [t.height, t.width, t.channels].contains(&0),
            Self::AttentionScores(AttentionScores {
                tokens,
                heads,
                head_dim,
                ..
            })
            | Self::AttentionValues(AttentionValues {
                tokens,
                heads,
                head_dim,
                ..
            }) => [tokens, heads, head_dim].contains(&0),
            Self::Softmax(s) => s.rows == 0 || s.cols == 0,
            Self::Add { elements, .. }
            | Self::Relu { elements, .. }
            | Self::Gelu { elements, .. }
            | Self::GeluErf { elements, .. }
            | Self::Scale { elements, .. } => elements == 0,
            Self::ChannelScale { rows, channels, .. } => rows == 0 || channels == 0,
        }
    }

    /// The kernel that runs this operation alone.
    fn standalone(&self) -> (MainOp, Vec<Epilogue>) {
        let map = |input, elements| MainOp::Map { input, elements };
        match *self {
            Self::Conv2d3x3(c) => (MainOp::Conv2d3x3(c), Vec::new()),
            Self::LayerNorm(l) => (MainOp::LayerNorm(l), Vec::new()),
            Self::MatMul(mm) => (MainOp::MatMul(mm), Vec::new()),
            Self::PatchEmbed(p) => (MainOp::PatchEmbed(p), Vec::new()),
            Self::PrependToken(t) => (MainOp::PrependToken(t), Vec::new()),
            Self::TokensToMap(t) => (MainOp::TokensToMap(t), Vec::new()),
            Self::AttentionScores(s) => (MainOp::AttentionScores(s), Vec::new()),
            Self::Softmax(s) => (MainOp::Softmax(s), Vec::new()),
            Self::AttentionValues(v) => (MainOp::AttentionValues(v), Vec::new()),
            Self::Add { a, b, elements } => (map(a, elements), vec![Epilogue::Add(b)]),
            Self::Relu { input, elements } => (map(input, elements), vec![Epilogue::Relu]),
            Self::Gelu { input, elements } => (map(input, elements), vec![Epilogue::Gelu]),
            Self::GeluErf { input, elements } => (map(input, elements), vec![Epilogue::GeluErf]),
            Self::Scale {
                input,
                factor,
                elements,
            } => (map(input, elements), vec![Epilogue::Scale(factor)]),
            Self::ChannelScale {
                input,
                scale,
                rows,
                channels,
            } => (
                // The graph bounded the element count by MAX_ELEMENTS.
                map(
                    input,
                    u32::try_from(product(&[rows, channels])).unwrap_or(u32::MAX),
                ),
                vec![Epilogue::ChannelScale { scale, channels }],
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
            Self::GeluErf { input, .. } if input == chain => Some(Epilogue::GeluErf),
            Self::Scale { input, factor, .. } if input == chain => Some(Epilogue::Scale(factor)),
            Self::ChannelScale {
                input,
                scale,
                channels,
                ..
            } if input == chain && scale != chain => {
                Some(Epilogue::ChannelScale { scale, channels })
            }
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
    /// dimension, a non-finite scale, a layer norm epsilon that is not finite
    /// and positive, or a size past [`MAX_ELEMENTS`].
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
        if let OpKind::LayerNorm(l) = kind {
            if !(l.epsilon.is_finite() && l.epsilon > 0.0) {
                return Err(KernelError::InvalidEpsilon { op });
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
            kernels.push(KernelSpec::whole_of(
                main,
                epilogue,
                op.output,
                index..index + 1,
            ));
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
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MainOp {
    /// A 3x3 convolution, one invocation per output element.
    Conv2d3x3(Conv2d3x3),
    /// A layer norm, one workgroup per row.
    LayerNorm(LayerNorm),
    /// A tiled matrix multiply, one workgroup per output tile.
    MatMul(MatMul),
    /// A patch embedding as a tiled matrix multiply of the patches, gathered
    /// from the image as they are loaded, by the weights: one workgroup per
    /// output tile.
    PatchEmbed(PatchEmbed),
    /// A token in front of a sequence, one invocation per output element.
    PrependToken(PrependToken),
    /// A sequence without its first token, one invocation per output element.
    TokensToMap(TokensToMap),
    /// Attention scores as one tiled matrix multiply per head, `Q K^T`: one
    /// workgroup per output tile of a head.
    AttentionScores(AttentionScores),
    /// A softmax, one workgroup per row.
    Softmax(Softmax),
    /// The weighted sum of the values as one tiled matrix multiply per head,
    /// `P V`: one workgroup per output tile of a head.
    AttentionValues(AttentionValues),
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
            Self::PatchEmbed(p) => [Some(p.input), Some(p.weights), p.bias]
                .into_iter()
                .flatten()
                .collect(),
            Self::PrependToken(t) => vec![t.token, t.tokens],
            Self::TokensToMap(t) => vec![t.input],
            Self::AttentionScores(s) => vec![s.qkv],
            Self::Softmax(s) => vec![s.input],
            Self::AttentionValues(v) => vec![v.probs, v.qkv],
            Self::Map { input, .. } => vec![input],
        }
    }

    /// The element count of the result; fits [`MAX_ELEMENTS`] because the
    /// graph checked it.
    fn output_elements(&self) -> u64 {
        match *self {
            Self::Conv2d3x3(c) => OpKind::Conv2d3x3(c).output_elements(),
            Self::LayerNorm(l) => OpKind::LayerNorm(l).output_elements(),
            Self::MatMul(mm) => OpKind::MatMul(mm).output_elements(),
            Self::PatchEmbed(p) => OpKind::PatchEmbed(p).output_elements(),
            Self::PrependToken(t) => OpKind::PrependToken(t).output_elements(),
            Self::TokensToMap(t) => OpKind::TokensToMap(t).output_elements(),
            Self::AttentionScores(s) => OpKind::AttentionScores(s).output_elements(),
            Self::Softmax(s) => OpKind::Softmax(s).output_elements(),
            Self::AttentionValues(v) => OpKind::AttentionValues(v).output_elements(),
            Self::Map { elements, .. } => elements.into(),
        }
    }

    /// A short name for labels and file names.
    const fn name(&self) -> &'static str {
        match self {
            Self::Conv2d3x3(_) => "conv2d3x3",
            Self::LayerNorm(_) => "layernorm",
            Self::MatMul(_) => "matmul",
            Self::PatchEmbed(_) => "patch_embed",
            Self::PrependToken(_) => "prepend_token",
            Self::TokensToMap(_) => "tokens_to_map",
            Self::AttentionScores(_) => "attention_scores",
            Self::Softmax(_) => "softmax",
            Self::AttentionValues(_) => "attention_values",
            Self::Map { .. } => "map",
        }
    }

    /// The size of the dimension a kernel is split along: output pixels,
    /// rows, output rows, query rows or elements (see [`Part`]). Fits a `u32`
    /// because the graph bounds the output by [`MAX_ELEMENTS`].
    fn split_extent(&self) -> u32 {
        let narrow = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
        match *self {
            Self::Conv2d3x3(c) => narrow(product(&[c.height, c.width])),
            Self::LayerNorm(l) => l.rows,
            Self::MatMul(mm) => mm.m,
            Self::PatchEmbed(p) => narrow(product(&[p.grid_h, p.grid_w])),
            Self::AttentionScores(s) => s.tokens,
            Self::Softmax(s) => s.rows,
            Self::AttentionValues(v) => v.tokens,
            Self::PrependToken(_) | Self::TokensToMap(_) | Self::Map { .. } => {
                narrow(self.output_elements())
            }
        }
    }

    /// Units of the split dimension a part boundary moves in: a tile's height
    /// for the kernels tiled like a matrix multiply, one unit otherwise.
    const fn split_grain(&self) -> u32 {
        if self.is_tiled() { MATMUL_TILE } else { 1 }
    }

    /// Whether the kernel is a tiled matrix multiply: one workgroup per output
    /// tile, rows of tiles over the split dimension, columns across the
    /// output's last dimension and, for attention, one layer of tiles per head.
    const fn is_tiled(&self) -> bool {
        matches!(
            self,
            Self::MatMul(_)
                | Self::PatchEmbed(_)
                | Self::AttentionScores(_)
                | Self::AttentionValues(_)
        )
    }

    /// Output elements per unit of the split dimension.
    fn elements_per_unit(&self) -> u64 {
        match *self {
            Self::Conv2d3x3(c) => c.cout.into(),
            Self::LayerNorm(l) => l.channels.into(),
            Self::MatMul(mm) => mm.n.into(),
            Self::PatchEmbed(p) => p.cout.into(),
            Self::AttentionScores(s) => product(&[s.heads, s.tokens]),
            Self::Softmax(s) => s.cols.into(),
            Self::AttentionValues(v) => product(&[v.heads, v.head_dim]),
            Self::PrependToken(_) | Self::TokensToMap(_) | Self::Map { .. } => 1,
        }
    }

    /// The tiled kernel's columns of tiles and layers of tiles: the extent of
    /// the output's columns within one row of the split dimension and one
    /// layer, and how many layers. `None` for a kernel that is not tiled.
    fn tile_columns_and_layers(&self) -> Option<(u32, u32)> {
        match *self {
            Self::MatMul(mm) => Some((mm.n, 1)),
            Self::PatchEmbed(p) => Some((p.cout, 1)),
            Self::AttentionScores(s) => Some((s.tokens, s.heads)),
            Self::AttentionValues(v) => Some((v.head_dim, v.heads)),
            _ => None,
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
    /// GELU through erf, as [`OpKind::GeluErf`].
    GeluErf,
    /// `v * factor`; finite.
    Scale(f32),
    /// `v * scale[index % channels]`, as [`OpKind::ChannelScale`]; the kernel's
    /// output has a multiple of `channels` elements, `channels` last.
    ChannelScale {
        /// The per-channel multipliers, `[channels]`.
        scale: TensorId,
        /// The output's last dimension.
        channels: u32,
    },
}

impl Epilogue {
    /// A short name for labels and file names.
    const fn name(&self) -> &'static str {
        match self {
            Self::Add(_) => "add",
            Self::Relu => "relu",
            Self::Gelu => "gelu",
            Self::GeluErf => "gelu_erf",
            Self::Scale(_) => "scale",
            Self::ChannelScale { .. } => "channel_scale",
        }
    }

    /// The tensor the step reads, if it reads one: it is bound by the kernel.
    const fn operand(&self) -> Option<TensorId> {
        match *self {
            Self::Add(id) | Self::ChannelScale { scale: id, .. } => Some(id),
            Self::Relu | Self::Gelu | Self::GeluErf | Self::Scale(_) => None,
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

/// Which share of a kernel's output one dispatch computes: part `index` of
/// `count`, covering `range` of the kernel's split dimension.
///
/// The split dimension is the one the kernel's grid walks in output order: the
/// output pixels of a convolution, the rows of a layer norm or a softmax, the
/// output rows of a matrix multiply, the patches of a patch embedding, the
/// query rows of the attention scores and of the weighted sum of the values,
/// and the elements of a standalone element-wise kernel and of the token
/// copies ([`PrependToken`], [`TokensToMap`]).
/// Because every layout is row-major with that dimension outermost, a range of
/// it is a contiguous range of the output ([`KernelSpec::output_range`]).
/// Rule 6 of the [module documentation](self#rules) says how a kernel's parts
/// cover it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Part {
    index: u32,
    count: u32,
    range: Range<u32>,
}

impl Part {
    /// Which part this is, from zero.
    #[must_use]
    pub const fn index(&self) -> u32 {
        self.index
    }

    /// How many parts the kernel was split into; one for a kernel run whole.
    #[must_use]
    pub const fn count(&self) -> u32 {
        self.count
    }

    /// The units of the split dimension this part computes, never empty.
    #[must_use]
    pub fn range(&self) -> Range<u32> {
        self.range.clone()
    }

    /// Whether the part is the whole kernel.
    #[must_use]
    pub const fn is_whole(&self) -> bool {
        self.count == 1
    }
}

/// One kernel of a program: a main operation, its epilogue, the tensors it
/// binds, and the [`Part`] of its output one dispatch of it computes.
///
/// Made only by [`Graph::lower`] and [`Program::split`], so its operations,
/// shapes and tensors are the graph's and already checked.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelSpec {
    main: MainOp,
    epilogue: Vec<Epilogue>,
    output: TensorId,
    ops: Range<usize>,
    part: Part,
}

impl KernelSpec {
    /// A spec running `main` whole.
    fn whole_of(
        main: MainOp,
        epilogue: Vec<Epilogue>,
        output: TensorId,
        ops: Range<usize>,
    ) -> Self {
        let part = Part {
            index: 0,
            count: 1,
            range: 0..main.split_extent(),
        };
        Self {
            main,
            epilogue,
            output,
            ops,
            part,
        }
    }

    /// The share of the output this spec's dispatch computes.
    #[must_use]
    pub const fn part(&self) -> &Part {
        &self.part
    }

    /// The kernel this spec is a part of: the same spec over its whole output.
    /// A spec that is whole is its own whole.
    #[must_use]
    pub fn whole(&self) -> Self {
        Self::whole_of(
            self.main,
            self.epilogue.clone(),
            self.output,
            self.ops.clone(),
        )
    }

    /// The most parts the kernel can be split into: one per unit of its split
    /// dimension, except a kernel tiled like a matrix multiply (the matrix
    /// multiply, the patch embedding and the two attention products), whose
    /// parts are whole rows of output tiles. A boundary inside a tile would
    /// make both parts run that tile's workgroups, each with some invocations
    /// idle, so the work would grow with the part count.
    #[must_use]
    pub fn max_parts(&self) -> u32 {
        self.main.split_extent().div_ceil(self.main.split_grain())
    }

    /// The output elements this spec's dispatch writes, a contiguous range of
    /// the output tensor.
    #[must_use]
    pub fn output_range(&self) -> Range<u64> {
        let per_unit = self.main.elements_per_unit();
        u64::from(self.part.range.start).saturating_mul(per_unit)
            ..u64::from(self.part.range.end).saturating_mul(per_unit)
    }

    /// The kernel, whole, as `count` parts in order, `count` clamped to
    /// `1..=`[`max_parts`](Self::max_parts) (rule 6 of the
    /// [module documentation](self#rules)).
    ///
    /// The split dimension is cut into blocks (one unit, or one tile row of a
    /// matrix multiply), and part `i` takes blocks `i * B / count` up to
    /// `(i + 1) * B / count` of the `B` there are, so no two parts differ by
    /// more than one block and none is empty.
    fn split(&self, count: u32) -> Vec<Self> {
        let whole = self.whole();
        let count = count.clamp(1, self.max_parts().max(1));
        let extent = self.main.split_extent();
        let grain = self.main.split_grain();
        let blocks = u64::from(self.max_parts());
        // At most `blocks * grain`, which is below `extent + grain`: it fits a
        // u32 because the graph bounds every dimension by MAX_ELEMENTS.
        let edge = |i: u32| {
            let block = blocks * u64::from(i) / u64::from(count);
            let unit = block.saturating_mul(u64::from(grain));
            u32::try_from(unit).unwrap_or(u32::MAX).min(extent)
        };
        (0..count)
            .map(|index| Self {
                part: Part {
                    index,
                    count,
                    range: edge(index)..edge(index + 1),
                },
                ..whole.clone()
            })
            .collect()
    }

    /// How many parts keep each part of this kernel within `bound`, given that
    /// the whole kernel took `duration`: `ceil(duration / bound)`, at least
    /// one and at most [`max_parts`](Self::max_parts).
    ///
    /// The duration is a measurement, or a [`CostModel`](crate::cost::CostModel)
    /// estimate built from measurements, of the kernel run whole on the device
    /// that will run its parts. The decision follows it rather than the
    /// kernel's shape, so a kernel that is fast on one adapter is not split
    /// there and pays no extra dispatches for another adapter's slowness. Parts
    /// are equal shares of the output, so each is expected to take about
    /// `duration / parts`; that holds only as far as a kernel's time is
    /// proportional to its output, which is why what the parts took is
    /// measured again (the `beatlm-bench` crate documentation, "Splitting").
    ///
    /// # Errors
    ///
    /// [`KernelError::ZeroBound`]: no dispatch fits a bound of zero.
    pub fn parts_within(&self, duration: Micros, bound: Micros) -> Result<u32, KernelError> {
        if bound == Micros::ZERO {
            return Err(KernelError::ZeroBound);
        }
        let wanted = duration.get().div_ceil(bound.get()).max(1);
        Ok(u32::try_from(wanted)
            .unwrap_or(u32::MAX)
            .min(self.max_parts().max(1)))
    }

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
    /// operation's reads, then the tensors the epilogue's steps read
    /// ([`Epilogue::Add`], [`Epilogue::ChannelScale`]) in step order, then the
    /// output, which is always last.
    ///
    /// Binding `i` of the generated shader is element `i`.
    #[must_use]
    pub fn bindings(&self) -> Vec<TensorId> {
        let mut out: Vec<TensorId> = Vec::new();
        let added = self.epilogue.iter().filter_map(Epilogue::operand);
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

    /// What `variant` of this kernel needs and how it is launched; the grid of
    /// workgroups covers the spec's [`Part`], not the whole output.
    #[must_use]
    pub fn requirements(&self, variant: Variant) -> Requirements {
        let Variant::Baseline = variant;
        let storage_buffers = u32::try_from(self.bindings().len()).unwrap_or(u32::MAX);
        // The grid covers the part, not the whole output.
        let units = self.part.range.end - self.part.range.start;
        let (workgroup_size, workgroups, workgroup_storage_bytes) = match self.main {
            MainOp::Conv2d3x3(_)
            | MainOp::PrependToken(_)
            | MainOp::TokensToMap(_)
            | MainOp::Map { .. } => {
                let range = self.output_range();
                (
                    [ELEMENT_GROUP, 1, 1],
                    rows_of((range.end - range.start).div_ceil(u64::from(ELEMENT_GROUP))),
                    0,
                )
            }
            MainOp::LayerNorm(_) => (
                [LAYER_NORM_LANES, 1, 1],
                rows_of(u64::from(units)),
                LAYER_NORM_LANES * F32_BYTES,
            ),
            MainOp::Softmax(_) => (
                [SOFTMAX_LANES, 1, 1],
                rows_of(u64::from(units)),
                SOFTMAX_LANES * F32_BYTES,
            ),
            MainOp::MatMul(_)
            | MainOp::PatchEmbed(_)
            | MainOp::AttentionScores(_)
            | MainOp::AttentionValues(_) => {
                let (columns, layers) = self.main.tile_columns_and_layers().unwrap_or((1, 1));
                (
                    [MATMUL_TILE, MATMUL_TILE, 1],
                    [
                        columns.div_ceil(MATMUL_TILE),
                        units.div_ceil(MATMUL_TILE),
                        layers,
                    ],
                    2 * MATMUL_TILE * MATMUL_TILE * F32_BYTES,
                )
            }
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
    /// The graph's knowledge of every tensor in `tensors`, kept so a split
    /// derives lifetimes by the same rule as the lowering.
    infos: BTreeMap<TensorId, TensorInfo>,
}

impl Program {
    /// Derives every tensor's lifetime from the kernels that use it (rule 1 of
    /// the [module documentation](self#rules)).
    fn new(kernels: Vec<KernelSpec>, graph: &Graph) -> Self {
        Self::derive(kernels, &graph.tensors)
    }

    /// The program of `kernels` over the tensors `infos` describes: each
    /// tensor a kernel binds, and each input, weight and output, with its
    /// lifetime derived from the kernels' indices. The caller keeps the kernel
    /// count within `u32`.
    fn derive(kernels: Vec<KernelSpec>, infos: &BTreeMap<TensorId, TensorInfo>) -> Self {
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
        let mut kept = BTreeMap::new();
        for (&id, info) in infos {
            let lifetime = if info.role.whole_program() {
                Some((0, last))
            } else {
                // An activation fusion removed is used by no kernel.
                uses.get(&id).copied()
            };
            if let Some((first, end)) = lifetime {
                tensors.push(TensorDesc::new(id, DType::F32, info.elements, first, end));
                kept.insert(id, *info);
            }
        }
        Self {
            kernels,
            tensors,
            infos: kept,
        }
    }

    /// The kernels, in dispatch order. Kernel `i` is operation `i` of the
    /// memory plan's program order.
    #[must_use]
    pub fn kernels(&self) -> &[KernelSpec] {
        &self.kernels
    }

    /// How many parts keep each kernel within `bound`, given what each took:
    /// [`KernelSpec::parts_within`] for kernel `i` and `durations[i]`, ready to
    /// be passed to [`split`](Self::split).
    ///
    /// # Errors
    ///
    /// [`KernelError::PartCounts`] when `durations` does not have one entry
    /// per kernel, [`KernelError::ZeroBound`] for a bound of zero.
    pub fn parts_within(
        &self,
        durations: &[Micros],
        bound: Micros,
    ) -> Result<Vec<u32>, KernelError> {
        if durations.len() != self.kernels.len() {
            return Err(KernelError::PartCounts {
                kernels: self.kernels.len(),
                given: durations.len(),
            });
        }
        self.kernels
            .iter()
            .zip(durations)
            .map(|(spec, &duration)| spec.parts_within(duration, bound))
            .collect()
    }

    /// The program with kernel `i` split into `parts[i]` parts (rule 6 of the
    /// [module documentation](self#rules)), each a kernel of its own placed
    /// where the kernel was, in order, so each can be its own slice of a plan
    /// and the scheduler can hand them out in different frames. A count is
    /// clamped to `1..=`[`KernelSpec::max_parts`], so a count past what a
    /// kernel's shape allows, or of zero, is never an error.
    ///
    /// Lifetimes are derived again from the new kernel indices, by rule 1; the
    /// memory plan of the result is a plan of its own.
    ///
    /// # Errors
    ///
    /// [`KernelError::PartCounts`] when `parts` does not have one entry per
    /// kernel, [`KernelError::AlreadySplit`] when a kernel is already a part,
    /// [`KernelError::TooManyOps`] when the parts would be more kernels than a
    /// kernel index counts.
    pub fn split(&self, parts: &[u32]) -> Result<Self, KernelError> {
        if parts.len() != self.kernels.len() {
            return Err(KernelError::PartCounts {
                kernels: self.kernels.len(),
                given: parts.len(),
            });
        }
        let mut total: u64 = 0;
        for (index, (spec, &count)) in self.kernels.iter().zip(parts).enumerate() {
            if !spec.part.is_whole() {
                return Err(KernelError::AlreadySplit { kernel: index });
            }
            total += u64::from(count.clamp(1, spec.max_parts().max(1)));
        }
        if u32::try_from(total).is_err() {
            return Err(KernelError::TooManyOps);
        }
        let kernels = self
            .kernels
            .iter()
            .zip(parts)
            .flat_map(|(spec, &count)| spec.split(count))
            .collect();
        Ok(Self::derive(kernels, &self.infos))
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
        self.infos.get(&id).map(|info| info.role)
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
        if self.infos.contains_key(&id) {
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
    /// A layer norm whose epsilon is not finite and positive, so a constant
    /// row would divide by zero or every row by an infinity.
    #[error("operation {op} is a layer norm whose epsilon is not finite and positive")]
    InvalidEpsilon {
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
    /// A list of part counts or durations that is not one per kernel.
    #[error("{given} entries given for a program of {kernels} kernels")]
    PartCounts {
        /// The program's kernels.
        kernels: usize,
        /// The entries given.
        given: usize,
    },
    /// A kernel that is already a part of a split is not split again: its
    /// part counts would no longer describe the program it came from.
    #[error("kernel {kernel} is already a part of a split kernel")]
    AlreadySplit {
        /// The kernel's index in the program.
        kernel: usize,
    },
    /// No dispatch fits a bound of zero.
    #[error("a dispatch bound of zero cannot be met")]
    ZeroBound,
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
                    epsilon: LAYER_NORM_EPSILON,
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

    /// A graph of one heavy or element-wise operation reading weights of the
    /// sizes its shape needs and writing `Y`.
    fn single(kind: OpKind) -> Graph {
        let mut g = Graph::new();
        for (t, elements) in kind.reads() {
            if g.role(t).is_none() {
                g.tensor(t, TensorRole::Weight, elements).unwrap();
            }
        }
        g.tensor(Y, TensorRole::Output, kind.output_elements())
            .unwrap();
        g.push(kind, Y).unwrap();
        g
    }

    /// One operation of each main-operation kind, at shapes that include ones
    /// of a single unit and ones no tile or workgroup divides.
    fn shapes() -> Vec<OpKind> {
        let mut kinds = Vec::new();
        for (height, width, cin, cout) in [(1, 1, 1, 1), (7, 9, 3, 5), (3, 17, 2, 4)] {
            kinds.push(OpKind::Conv2d3x3(Conv2d3x3 {
                input: X,
                weights: W,
                bias: Some(BIAS),
                height,
                width,
                cin,
                cout,
            }));
        }
        for (rows, channels) in [(1, 1), (63, 5), (100, 3)] {
            kinds.push(OpKind::LayerNorm(LayerNorm {
                input: X,
                gamma: W,
                beta: BIAS,
                rows,
                channels,
                epsilon: 1e-6,
            }));
        }
        for (m, k, n) in [(1, 1, 1), (4, 8, 4), (16, 3, 2), (33, 5, 7), (37, 21, 19)] {
            kinds.push(OpKind::MatMul(MatMul {
                a: X,
                b: W,
                bias: Some(BIAS),
                m,
                k,
                n,
            }));
        }
        for elements in [1, 7, 257, 1000] {
            kinds.push(OpKind::Gelu { input: X, elements });
            kinds.push(OpKind::GeluErf { input: X, elements });
        }
        kinds.extend(encoder_shapes());
        kinds
    }

    /// One operation of each vision transformer kind, at a single unit, at
    /// shapes no tile divides, and at DINOv2 Small's at 518 pixels where the
    /// kernel splits in rows; one split by elements is tried at a width of 384
    /// and fewer rows, since its part counts reach its element count.
    fn encoder_shapes() -> Vec<OpKind> {
        let mut kinds = Vec::new();
        for (grid_h, grid_w, patch, cin, cout) in
            [(1, 1, 1, 1, 1), (5, 7, 3, 2, 19), (37, 37, 14, 3, 384)]
        {
            kinds.push(OpKind::PatchEmbed(PatchEmbed {
                input: X,
                weights: W,
                bias: Some(BIAS),
                grid_h,
                grid_w,
                patch,
                cin,
                cout,
            }));
        }
        for (rows, channels) in [(1, 1), (35, 19), (40, 384)] {
            kinds.push(OpKind::PrependToken(PrependToken {
                token: W,
                tokens: X,
                rows,
                channels,
            }));
        }
        for (height, width, channels) in [(1, 1, 1), (5, 7, 19), (6, 7, 384)] {
            kinds.push(OpKind::TokensToMap(TokensToMap {
                input: X,
                height,
                width,
                channels,
            }));
        }
        for (tokens, heads, head_dim) in [(1, 1, 1), (17, 2, 5), (1370, 6, 64)] {
            kinds.push(OpKind::AttentionScores(AttentionScores {
                qkv: X,
                tokens,
                heads,
                head_dim,
            }));
            kinds.push(OpKind::AttentionValues(AttentionValues {
                probs: W,
                qkv: X,
                tokens,
                heads,
                head_dim,
            }));
        }
        for (rows, cols) in [(1, 1), (19, 300), (1370 * 6, 1370)] {
            kinds.push(OpKind::Softmax(Softmax {
                input: X,
                rows,
                cols,
            }));
        }
        for (rows, channels) in [(1, 1), (35, 19), (40, 384)] {
            kinds.push(OpKind::ChannelScale {
                input: X,
                scale: W,
                rows,
                channels,
            });
        }
        kinds
    }

    /// Part counts to try against a kernel of `max` parts: every count up to
    /// past `max` when that is small, otherwise counts that do and do not
    /// divide it, `max` itself, and counts past it.
    fn counts(max: u32) -> Vec<u32> {
        if max <= 40 {
            return (0..=max + 3).collect();
        }
        vec![
            0,
            1,
            2,
            3,
            5,
            7,
            13,
            max - 1,
            max,
            max + 1,
            2 * max + 3,
            u32::MAX,
        ]
    }

    #[test]
    fn a_split_kernels_parts_cover_its_output_exactly_once() {
        let baseline = DeviceProfile::webgpu_baseline();
        for kind in shapes() {
            let whole = single(kind).lower(Fusion::On).unwrap();
            let spec = &whole.kernels()[0];
            let max = spec.max_parts();
            let extent = spec.main().split_extent();
            let grain = spec.main().split_grain();
            assert_eq!(max, extent.div_ceil(grain), "{}", spec.name());
            assert_eq!(spec.part().count(), 1);
            assert_eq!(spec.output_range(), 0..spec.output_elements());
            for count in counts(max) {
                let split = whole.split(&[count]).unwrap();
                let parts = split.kernels();
                let what = format!("{} split {count} ways", spec.name());
                assert_eq!(parts.len() as u32, count.clamp(1, max), "{what}");
                let mut next_unit = 0;
                let mut next_element = 0;
                let mut sizes = Vec::new();
                for (i, part) in parts.iter().enumerate() {
                    assert_eq!(part.part().index(), i as u32, "{what}");
                    assert_eq!(part.part().count(), parts.len() as u32, "{what}");
                    assert_eq!(part.whole(), *spec, "{what}");
                    // Same tensors, same epilogue, same operations.
                    assert_eq!(part.bindings(), spec.bindings(), "{what}");
                    assert_eq!(part.epilogue(), spec.epilogue(), "{what}");
                    assert_eq!(part.ops(), spec.ops(), "{what}");
                    let range = part.part().range();
                    assert_eq!(range.start, next_unit, "{what}: contiguous units");
                    assert!(range.end > range.start, "{what}: part {i} is empty");
                    if range.end != extent {
                        assert_eq!(range.end % grain, 0, "{what}: a boundary inside a tile");
                    }
                    next_unit = range.end;
                    let out = part.output_range();
                    assert_eq!(out.start, next_element, "{what}: contiguous output");
                    next_element = out.end;
                    sizes.push(u64::from(range.end - range.start));

                    // The part's grid covers the part, and the part runs on the
                    // baseline.
                    let r = part.requirements(Variant::Baseline);
                    let groups = u64::from(r.workgroups[0]) * u64::from(r.workgroups[1]);
                    let covered =
                        groups * u64::from(r.workgroup_size[0]) * u64::from(r.workgroup_size[1]);
                    match spec.main() {
                        MainOp::Conv2d3x3(_)
                        | MainOp::PrependToken(_)
                        | MainOp::TokensToMap(_)
                        | MainOp::Map { .. } => {
                            assert!(covered >= out.end - out.start, "{what}");
                        }
                        MainOp::LayerNorm(_) | MainOp::Softmax(_) => {
                            assert!(groups >= u64::from(range.end - range.start), "{what}");
                        }
                        MainOp::MatMul(_)
                        | MainOp::PatchEmbed(_)
                        | MainOp::AttentionScores(_)
                        | MainOp::AttentionValues(_) => {
                            let (columns, layers) = spec.main().tile_columns_and_layers().unwrap();
                            assert!(r.workgroups[0] * MATMUL_TILE >= columns, "{what}");
                            assert!(
                                r.workgroups[1] * MATMUL_TILE >= range.end - range.start,
                                "{what}"
                            );
                            assert_eq!(r.workgroups[2], layers, "{what}");
                        }
                    }
                    assert_eq!(part.check(Variant::Baseline, &baseline), Ok(()), "{what}");
                }
                assert_eq!(next_unit, extent, "{what}: every unit");
                assert_eq!(
                    next_element,
                    spec.output_elements(),
                    "{what}: every element"
                );
                // Even: no two parts differ by more than one block (the last
                // block of a matrix multiply can be short, which only makes
                // its part smaller).
                let (least, most) = (sizes.iter().min().unwrap(), sizes.iter().max().unwrap());
                assert!(most - least <= 2 * u64::from(grain), "{what}: {sizes:?}");
            }
        }
    }

    #[test]
    fn the_part_count_is_the_duration_over_the_bound_rounded_up_and_clamped() {
        let mut g = base(&[]);
        push(&mut g, matmul(Y));
        let p = g.lower(Fusion::On).unwrap();
        // A 4-row matmul is one tile row, so it cannot be split at all.
        let small = &p.kernels()[0];
        assert_eq!(small.max_parts(), 1);
        let us = Micros::new;
        assert_eq!(small.parts_within(us(10_000), us(1_000)), Ok(1));

        let big = single(OpKind::Relu {
            input: X,
            elements: 1000,
        })
        .lower(Fusion::On)
        .unwrap();
        let k = &big.kernels()[0];
        assert_eq!(k.max_parts(), 1000);
        for (duration, bound, parts) in [
            (0, 1_000, 1),
            (1, 1_000, 1),
            (999, 1_000, 1),
            (1_000, 1_000, 1),
            (1_001, 1_000, 2),
            (2_700, 1_000, 3),
            (3_000, 1_000, 3),
            (3_001, 1_000, 4),
            (5_000_000, 1, 1000),
            (u64::MAX, 1, 1000),
        ] {
            assert_eq!(
                k.parts_within(us(duration), us(bound)),
                Ok(parts),
                "{duration} us within {bound} us"
            );
        }
        assert_eq!(
            k.parts_within(us(1), Micros::ZERO),
            Err(KernelError::ZeroBound)
        );
        assert_eq!(big.parts_within(&[us(2_500)], us(1_000)), Ok(vec![3]));
        assert_eq!(
            big.parts_within(&[], us(1_000)),
            Err(KernelError::PartCounts {
                kernels: 1,
                given: 0
            })
        );
    }

    /// Three kernels, with fusion off: a matmul, then two element-wise
    /// kernels, one of which reads a tensor two kernels back.
    fn three_kernels() -> Program {
        let mut g = Graph::new();
        g.tensor(X, TensorRole::Input, 40 * 8).unwrap();
        g.tensor(W, TensorRole::Weight, 8 * 3).unwrap();
        g.tensor(BIAS, TensorRole::Weight, 3).unwrap();
        g.tensor(A, TensorRole::Activation, 120).unwrap();
        g.tensor(B, TensorRole::Activation, 120).unwrap();
        g.tensor(Y, TensorRole::Output, 120).unwrap();
        push(
            &mut g,
            (
                OpKind::MatMul(MatMul {
                    a: X,
                    b: W,
                    bias: Some(BIAS),
                    m: 40,
                    k: 8,
                    n: 3,
                }),
                A,
            ),
        );
        push(
            &mut g,
            (
                OpKind::Relu {
                    input: A,
                    elements: 120,
                },
                B,
            ),
        );
        push(
            &mut g,
            (
                OpKind::Add {
                    a: B,
                    b: A,
                    elements: 120,
                },
                Y,
            ),
        );
        g.lower(Fusion::Off).unwrap()
    }

    #[test]
    fn splitting_stretches_lifetimes_over_the_parts_and_the_plan_still_holds() {
        let whole = three_kernels();
        assert_eq!(whole.kernels().len(), 3);
        let baseline = DeviceProfile::webgpu_baseline();
        // The matmul has three tile rows; the maps have 120 elements.
        for parts in [[1, 1, 1], [2, 1, 1], [3, 2, 5], [9, 7, 1], [1, 4, 120]] {
            let split = whole.split(&parts).unwrap();
            let counts: Vec<u32> = parts
                .iter()
                .zip(whole.kernels())
                .map(|(&n, k)| n.clamp(1, k.max_parts()))
                .collect();
            let first: Vec<u32> = counts
                .iter()
                .scan(0, |at, &n| {
                    let here = *at;
                    *at += n;
                    Some(here)
                })
                .collect();
            let last_of = |k: usize| first[k] + counts[k] - 1;
            assert_eq!(split.kernels().len() as u32, counts.iter().sum::<u32>());

            // Every tensor's lifetime is the whole program's, mapped: from the
            // first part of its first kernel to the last part of its last.
            for t in whole.tensors() {
                let s = split.tensors().iter().find(|s| s.id == t.id).unwrap();
                assert_eq!(
                    (s.first_use, s.last_use),
                    (first[t.first_use as usize], last_of(t.last_use as usize)),
                    "tensor {} with parts {parts:?}",
                    t.id.get()
                );
                assert_eq!(split.role(t.id), whole.role(t.id));
            }
            assert_eq!(split.tensors().len(), whole.tensors().len());

            // A tensor is read only by kernels after every part that writes it.
            for (j, kernel) in split.kernels().iter().enumerate() {
                for t in kernel.bindings() {
                    if t == kernel.output() {
                        continue;
                    }
                    for (i, writer) in split.kernels().iter().enumerate() {
                        if writer.output() == t {
                            assert!(i < j, "kernel {j} reads {} before part {i}", t.get());
                        }
                    }
                }
            }

            // The plan of the split program keeps every kernel's tensors apart.
            let plan = MemoryPlan::new(split.tensors(), &baseline).unwrap();
            for kernel in split.kernels() {
                let bound = kernel.bindings();
                for (i, &a) in bound.iter().enumerate() {
                    for &b in &bound[i + 1..] {
                        let (p, q) = (plan.placement(a).unwrap(), plan.placement(b).unwrap());
                        assert!(
                            p.arena != q.arena || p.end() <= q.offset || q.end() <= p.offset,
                            "tensors {} and {} share bytes with parts {parts:?}",
                            a.get(),
                            b.get()
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn splitting_a_whole_program_by_ones_changes_nothing_and_bad_counts_are_errors() {
        let whole = three_kernels();
        assert_eq!(whole.split(&[1, 1, 1]).unwrap(), whole);
        // Zero is clamped to one.
        assert_eq!(whole.split(&[0, 0, 0]).unwrap(), whole);
        assert_eq!(
            whole.split(&[1, 1]),
            Err(KernelError::PartCounts {
                kernels: 3,
                given: 2
            })
        );
        let split = whole.split(&[2, 1, 1]).unwrap();
        assert_eq!(
            split.split(&[1, 1, 1, 1]),
            Err(KernelError::AlreadySplit { kernel: 0 })
        );
        // Splitting is deterministic.
        assert_eq!(whole.split(&[3, 2, 5]), whole.split(&[3, 2, 5]));
    }

    /// DINOv2 Small at 518 pixels: tokens, width, heads, head width, MLP width.
    const T: u32 = 1370;
    const D: u32 = 384;
    const HEADS: u32 = 6;
    const HEAD_DIM: u32 = 64;
    const MLP: u32 = 1536;

    /// Declares `id` as a weight of `elements` and returns it.
    fn weight(g: &mut Graph, id: u32, elements: u32) -> TensorId {
        g.tensor(TensorId::new(id), TensorRole::Weight, elements.into())
            .unwrap();
        TensorId::new(id)
    }

    /// Appends `kind`, declaring its output as tensor `id` of `role`.
    fn op(g: &mut Graph, kind: OpKind, id: u32, role: TensorRole) -> TensorId {
        g.tensor(TensorId::new(id), role, kind.output_elements())
            .unwrap();
        g.push(kind, TensorId::new(id)).unwrap();
        TensorId::new(id)
    }

    /// One pre-norm DINOv2 block at the real shapes, reading `x` and writing
    /// the output `Y`, with its tensors numbered from 100.
    ///
    /// The tensors that tests name: the packed queries, keys and values
    /// (110), the scores before and after the scale (111, 112), the
    /// attention weights (113), the values' weighted sum (114) and the block's
    /// residual stream between its halves (118).
    fn encoder_block(g: &mut Graph, x: TensorId) {
        use TensorRole::Activation as Act;
        let ln = |input, gamma, beta| {
            OpKind::LayerNorm(LayerNorm {
                input,
                gamma,
                beta,
                rows: T,
                channels: D,
                epsilon: 1e-6,
            })
        };
        let mm = |a, b, bias, k, n| {
            OpKind::MatMul(MatMul {
                a,
                b,
                bias: Some(bias),
                m: T,
                k,
                n,
            })
        };
        let (g1, b1) = (weight(g, 100, D), weight(g, 101, D));
        let (wqkv, bqkv) = (weight(g, 102, D * 3 * D), weight(g, 103, 3 * D));
        let (wo, bo) = (weight(g, 104, D * D), weight(g, 105, D));
        let ls1 = weight(g, 106, D);
        let (g2, b2) = (weight(g, 107, D), weight(g, 108, D));
        let (w1, bf1) = (weight(g, 120, D * MLP), weight(g, 121, MLP));
        let (w2, bf2) = (weight(g, 122, MLP * D), weight(g, 123, D));
        let ls2 = weight(g, 124, D);

        let n1 = op(g, ln(x, g1, b1), 109, Act);
        let qkv = op(g, mm(n1, wqkv, bqkv, D, 3 * D), 110, Act);
        let attention = AttentionScores {
            qkv,
            tokens: T,
            heads: HEADS,
            head_dim: HEAD_DIM,
        };
        let scores = op(g, OpKind::AttentionScores(attention), 111, Act);
        let scaled = op(
            g,
            OpKind::Scale {
                input: scores,
                factor: 0.125,
                elements: T * HEADS * T,
            },
            112,
            Act,
        );
        let probs = op(
            g,
            OpKind::Softmax(Softmax {
                input: scaled,
                rows: T * HEADS,
                cols: T,
            }),
            113,
            Act,
        );
        let values = op(
            g,
            OpKind::AttentionValues(AttentionValues {
                probs,
                qkv,
                tokens: T,
                heads: HEADS,
                head_dim: HEAD_DIM,
            }),
            114,
            Act,
        );
        let projected = op(g, mm(values, wo, bo, D, D), 115, Act);
        let layer_scaled = op(
            g,
            OpKind::ChannelScale {
                input: projected,
                scale: ls1,
                rows: T,
                channels: D,
            },
            116,
            Act,
        );
        let x1 = op(
            g,
            OpKind::Add {
                a: layer_scaled,
                b: x,
                elements: T * D,
            },
            118,
            Act,
        );
        let n2 = op(g, ln(x1, g2, b2), 119, Act);
        let hidden = op(g, mm(n2, w1, bf1, D, MLP), 125, Act);
        let activated = op(
            g,
            OpKind::GeluErf {
                input: hidden,
                elements: T * MLP,
            },
            126,
            Act,
        );
        let out = op(g, mm(activated, w2, bf2, MLP, D), 127, Act);
        let layer_scaled = op(
            g,
            OpKind::ChannelScale {
                input: out,
                scale: ls2,
                rows: T,
                channels: D,
            },
            128,
            Act,
        );
        g.tensor(Y, TensorRole::Output, u64::from(T * D)).unwrap();
        g.push(
            OpKind::Add {
                a: x1,
                b: layer_scaled,
                elements: T * D,
            },
            Y,
        )
        .unwrap();
    }

    fn block_graph() -> Graph {
        let mut g = Graph::new();
        g.tensor(X, TensorRole::Input, u64::from(T * D)).unwrap();
        encoder_block(&mut g, X);
        g
    }

    /// Checks `plan` against `program`: no two tensors whose lifetimes overlap
    /// share a byte, every arena fits one baseline binding, and each kernel's
    /// tensors are live at that kernel.
    fn assert_plan_holds(program: &Program, plan: &MemoryPlan) {
        let baseline = DeviceProfile::webgpu_baseline();
        assert!(plan.valid_on(&baseline));
        assert!(
            plan.arenas()
                .iter()
                .all(|&a| a <= baseline.max_arena_size())
        );
        let tensors = program.tensors();
        for (i, a) in tensors.iter().enumerate() {
            for b in &tensors[i + 1..] {
                if a.first_use <= b.last_use && b.first_use <= a.last_use {
                    let (p, q) = (plan.placement(a.id).unwrap(), plan.placement(b.id).unwrap());
                    assert!(
                        p.arena != q.arena || p.end() <= q.offset || q.end() <= p.offset,
                        "tensors {} and {} are live together and share bytes",
                        a.id.get(),
                        b.id.get()
                    );
                }
            }
        }
        for (k, kernel) in program.kernels().iter().enumerate() {
            let k = k as u32;
            for t in kernel.bindings() {
                let d = tensors.iter().find(|d| d.id == t).unwrap();
                assert!(d.first_use <= k && k <= d.last_use, "tensor {}", t.get());
            }
        }
    }

    #[test]
    fn an_encoder_block_fuses_layerscale_residuals_gelu_and_the_attention_scale() {
        let p = block_graph().lower(Fusion::On).unwrap();
        assert_eq!(
            p.kernels().iter().map(KernelSpec::name).collect::<Vec<_>>(),
            vec![
                "layernorm",
                "matmul",
                "attention_scores+scale",
                "softmax",
                "attention_values",
                "matmul+channel_scale+add",
                "layernorm",
                "matmul+gelu_erf",
                "matmul+channel_scale+add",
            ]
        );
        let names = |k: usize| p.kernels()[k].bindings();
        // The output projection binds the values, its weights and bias, the
        // LayerScale vector, the residual and its output.
        assert_eq!(
            names(5),
            [id(114), id(104), id(105), id(106), X, id(118)].to_vec()
        );
        assert_eq!(
            p.kernels()[5].epilogue(),
            &[
                Epilogue::ChannelScale {
                    scale: id(106),
                    channels: D
                },
                Epilogue::Add(X)
            ]
        );
        assert_eq!(p.kernels()[2].epilogue(), &[Epilogue::Scale(0.125)]);
        assert_eq!(p.kernels()[7].epilogue(), &[Epilogue::GeluErf]);
        // Fused intermediates are gone; the shared ones stay.
        for gone in [111, 115, 116, 125, 127, 128] {
            assert_eq!(lifetime(&p, id(gone)), None, "tensor {gone}");
        }
        let unfused = block_graph().lower(Fusion::Off).unwrap();
        assert_eq!(unfused.kernels().len(), 15);
        assert_eq!(unfused.kernels()[3].name(), "map+scale");
        assert_eq!(unfused.kernels()[11].name(), "map+gelu_erf");
        assert_eq!(unfused.kernels()[7].name(), "map+channel_scale");
        assert_eq!(
            unfused.kernels()[7].bindings(),
            vec![id(115), id(106), id(116)]
        );
    }

    #[test]
    fn an_encoder_blocks_lifetimes_come_from_its_kernels_and_its_plan_holds_split_or_whole() {
        let whole = block_graph().lower(Fusion::On).unwrap();
        // Kernel indices of the fused block: the queries, keys and values are
        // written by the projection (1) and last read by the weighted sum (4);
        // the scaled scores live from their kernel to the softmax; the
        // residual stream between the halves from the output projection (5)
        // to the second residual add (8).
        assert_eq!(lifetime(&whole, id(110)), Some((1, 4)));
        assert_eq!(lifetime(&whole, id(112)), Some((2, 3)));
        assert_eq!(lifetime(&whole, id(113)), Some((3, 4)));
        assert_eq!(lifetime(&whole, id(118)), Some((5, 8)));
        assert_eq!(lifetime(&whole, X), Some((0, 8)));
        // The scores are one tensor of 45 045 600 bytes, within one binding.
        let scores = whole.tensors().iter().find(|t| t.id == id(112)).unwrap();
        assert_eq!(scores.elements * 4, 45_045_600);
        let baseline = DeviceProfile::webgpu_baseline();
        let plan = MemoryPlan::new(whole.tensors(), &baseline).unwrap();
        assert_plan_holds(&whole, &plan);

        // Every kernel split along its rows (query rows for attention): the
        // plan of the split program holds as well, and each attention kernel's
        // parts are whole tile rows of queries.
        for count in [2, 3, 7, 86, 200] {
            let split = whole.split(&vec![count; whole.kernels().len()]).unwrap();
            let plan = MemoryPlan::new(split.tensors(), &baseline).unwrap();
            assert_plan_holds(&split, &plan);
            for part in split.kernels() {
                if let MainOp::AttentionScores(_) | MainOp::AttentionValues(_) = part.main() {
                    let r = part.part().range();
                    assert!(r.start % MATMUL_TILE == 0 && r.end <= T, "{r:?}");
                }
            }
        }
        let unfused = block_graph().lower(Fusion::Off).unwrap();
        let plan = MemoryPlan::new(unfused.tensors(), &baseline).unwrap();
        assert_plan_holds(&unfused, &plan);
    }

    #[test]
    fn the_embeddings_fuse_the_position_add_into_the_class_token_kernel() {
        let (image, w, b, cls, pos) = (id(0), id(1), id(2), id(3), id(4));
        let mut g = Graph::new();
        g.tensor(image, TensorRole::Input, 518 * 518 * 3).unwrap();
        weight(&mut g, 1, 14 * 14 * 3 * D);
        weight(&mut g, 2, D);
        weight(&mut g, 3, D);
        weight(&mut g, 4, T * D);
        let patches = op(
            &mut g,
            OpKind::PatchEmbed(PatchEmbed {
                input: image,
                weights: w,
                bias: Some(b),
                grid_h: 37,
                grid_w: 37,
                patch: 14,
                cin: 3,
                cout: D,
            }),
            10,
            TensorRole::Activation,
        );
        let tokens = op(
            &mut g,
            OpKind::PrependToken(PrependToken {
                token: cls,
                tokens: patches,
                rows: T - 1,
                channels: D,
            }),
            11,
            TensorRole::Activation,
        );
        let embedded = op(
            &mut g,
            OpKind::Add {
                a: tokens,
                b: pos,
                elements: T * D,
            },
            12,
            TensorRole::Activation,
        );
        op(
            &mut g,
            OpKind::TokensToMap(TokensToMap {
                input: embedded,
                height: 37,
                width: 37,
                channels: D,
            }),
            13,
            TensorRole::Output,
        );
        let p = g.lower(Fusion::On).unwrap();
        assert_eq!(
            p.kernels().iter().map(KernelSpec::name).collect::<Vec<_>>(),
            vec!["patch_embed", "prepend_token+add", "tokens_to_map"]
        );
        assert_eq!(p.kernels()[1].bindings(), vec![cls, patches, pos, id(12)]);
        assert_eq!(p.kernels()[0].output_elements(), u64::from((T - 1) * D));
        assert_eq!(p.kernels()[0].max_parts(), (T - 1).div_ceil(MATMUL_TILE));
        assert_eq!(
            p.kernels()[2].output_elements(),
            u64::from(37 * 37 * D),
            "the class token is dropped"
        );
        assert_eq!(g.lower(Fusion::Off).unwrap().kernels().len(), 4);
    }

    #[test]
    fn a_layerscale_by_its_own_input_is_not_fused() {
        // The multipliers are the tensor flowing out of the kernel, so a fused
        // step would read the value it is computing.
        let mut g = Graph::new();
        g.tensor(X, TensorRole::Input, 4).unwrap();
        g.tensor(W, TensorRole::Weight, 4).unwrap();
        g.tensor(A, TensorRole::Activation, 4).unwrap();
        g.tensor(Y, TensorRole::Output, 4).unwrap();
        g.push(
            OpKind::MatMul(MatMul {
                a: X,
                b: W,
                bias: None,
                m: 2,
                k: 2,
                n: 2,
            }),
            A,
        )
        .unwrap();
        g.push(
            OpKind::ChannelScale {
                input: A,
                scale: A,
                rows: 1,
                channels: 4,
            },
            Y,
        )
        .unwrap();
        assert_eq!(g.lower(Fusion::On).unwrap().kernels().len(), 2);
    }

    #[test]
    fn the_erf_constants_are_the_published_coefficients_in_f32() {
        // Abramowitz and Stegun 7.1.26, as printed.
        assert_eq!(GELU_ERF_P, 0.327_591_1_f64 as f32);
        let published: [f64; 5] = [
            0.254_829_592,
            -0.284_496_736,
            1.421_413_741,
            -1.453_152_027,
            1.061_405_429,
        ];
        assert_eq!(GELU_ERF_A, published.map(|a| a as f32));
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
        for epsilon in [0.0, -1e-6, f32::NAN, f32::INFINITY] {
            assert_eq!(
                g.push(
                    OpKind::LayerNorm(LayerNorm {
                        input: X,
                        gamma: BIAS,
                        beta: BIAS,
                        rows: 8,
                        channels: 4,
                        epsilon,
                    }),
                    id(9)
                ),
                Err(KernelError::InvalidEpsilon { op: 0 }),
                "{epsilon}"
            );
        }
        assert_eq!(
            g.push(
                OpKind::AttentionScores(AttentionScores {
                    qkv: X,
                    tokens: 4,
                    heads: 0,
                    head_dim: 8,
                }),
                A
            ),
            Err(KernelError::ZeroDimension { op: 0 })
        );
        assert_eq!(
            g.push(
                OpKind::PrependToken(PrependToken {
                    token: X,
                    tokens: X,
                    rows: 0,
                    channels: 16,
                }),
                A
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
