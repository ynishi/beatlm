//! Domain core of beatlm.
//!
//! This crate decides *what* GPU work runs in a frame and *when* its result is
//! delivered; it never touches a GPU. Time and measurements arrive as values
//! from the caller, and decisions leave as values, so the same logic can be
//! tested without an adapter and behaves identically on native and on wasm.
//!
//! # Architecture
//!
//! beatlm is three layers, and this crate is the innermost one:
//!
//! ```text
//!   host (frame loop: bench, demo, an application)
//!     │ frame start, measured render time, measured slice times
//!     ▼
//!   beatlm         translates wgpu, model files and shader text into the
//!     │            vocabulary below, and carries the decisions out on a GPU
//!     ▼
//!   beatlm-core    decides; owns every rule listed under "Rules"
//! ```
//!
//! Dependencies point inward only. Nothing here names a wgpu type, a file
//! format or a shader language, and nothing here reads a clock, performs I/O
//! or draws a random number: anything non-deterministic is measured outside
//! and passed in. A change that seems to need one of those here is a sign the
//! boundary is being drawn in the wrong place.
//!
//! # Design
//!
//! ## The queue is the synchronisation
//!
//! Inference and rendering share one GPU queue, and a queue completes work in
//! submission order. A slice handed out for frame `f` is submitted before
//! frame `f`'s render pass, so it is finished before anything frame `f` or a
//! later frame draws. The core therefore treats "handed out in this or an
//! earlier frame" as "complete" and needs no completion callbacks; measured
//! GPU times come back only to refine the [`cost::CostModel`]. This is what
//! lets the engine *promise* a delivery frame instead of compositing whatever
//! result arrived last, which is what tools that run inference in a separate
//! process are left with.
//!
//! ## Promises
//!
//! Each model runs under one [`sync::SyncPolicy`]:
//!
//! - **L1, fixed latency** — the result for input frame `t` is delivered on
//!   frame `t + k`, never earlier and never later. The host delays its source by
//!   the same `k` ([`sync::SyncPolicy::source_delay`]), so picture and result
//!   align exactly. This is the default promise: it costs `k` frames of latency
//!   and works for any model the budget can finish within `k` frames.
//! - **L2, same frame** — the result is delivered in its input frame. Only for
//!   models that fit one frame's budget.
//! - **On beat** — the result lands on the first frame starting at or after a
//!   beat. The beat is chosen when the input is submitted, far enough ahead to
//!   cover the measured cost of the work plus a margin, so generation can be
//!   started early and still hit the bar.
//!
//! A promise that cannot be kept is reported, never stretched: the host is told
//! the deadline was missed and what to show instead ([`sync::MissPolicy`]).
//!
//! ## Rules
//!
//! One concept owns each rule; others refer to it rather than restating it.
//!
//! | Rule | Owner |
//! |---|---|
//! | A result is delivered exactly on its promised frame or beat; a finished result seen after a skipped deadline frame is missed, not delivered late. | [`schedule::Scheduler`] (deadlines from [`sync::SyncPolicy`]) |
//! | The source is delayed by the promised latency so it aligns with results. | [`sync::SyncPolicy::source_delay`] |
//! | Jobs of one scheduler run in submission order, never interleaved, because they share one memory plan. | [`schedule::Scheduler::plan_frame`] |
//! | A job's output stays intact until its delivery frame is rendered: the host keeps one output slot per in-flight job. | [`sync::SyncPolicy::max_in_flight`] |
//! | The summed estimate of the slices run in a frame never exceeds that frame's budget. | [`schedule::Scheduler::plan_frame`] |
//! | A frame's budget is its period minus measured render time minus a margin, never negative. | [`budget::Budget::for_frame`] |
//! | An on-beat deadline is the first beat at or after submission time plus estimated work plus margin. | [`schedule::Scheduler::submit`] |
//! | A missed deadline is always reported, with the policy that says what to show. | [`schedule::Scheduler::plan_frame`] |
//! | Cost estimates change only through measurements. | [`cost::CostModel::record`] |
//! | One dispatch never runs longer than a bound, because a GPU cannot be preempted mid-dispatch. | the plan that slices a model |
//! | Tensors whose lifetimes overlap never share memory; arenas respect the adapter's binding size and offset alignment; a tensor too large for an arena is an error, never split. | [`memory::MemoryPlan`] |
//! | A plan made against the baseline profile is valid on every adapter. | [`device::DeviceProfile::webgpu_baseline`], checked for other profiles by [`memory::MemoryPlan::valid_on`] |
//! | The steady-state loop allocates nothing and reads nothing back except timestamps. | the plan |
//! | A kernel stays within the adapter's workgroup memory and requests only features the adapter has; a kernel using subgroups has a variant without them. | the kernel IR |
//!
//! ## What changes, and where
//!
//! Decisions expected to move are kept behind one owner each: the statistic a
//! cost estimate uses lives in [`cost`], the delivery promises in [`sync`],
//! the order jobs are served in [`schedule`], the packing algorithm that places tensors
//! in memory in [`memory`]. Shader languages, model formats and GPU backends are
//! not decided here at all; what an adapter reports reaches the core only as
//! the plain values of a [`device::DeviceProfile`].

// Also set as a workspace lint. Kept at the crate root so the requirement is
// visible where the crate's documentation starts, and survives the crate
// being built outside this workspace.
#![warn(missing_docs)]

pub mod budget;
pub mod cost;
pub mod device;
pub mod memory;
pub mod schedule;
pub mod sync;
pub mod time;
