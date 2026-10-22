//! What beatlm's two hosts both compile: the headless bench (`beatlm-bench`)
//! and the browser host (`beatlm-web`).
//!
//! ```text
//!   beatlm-bench   beatlm-web     the hosts: clock, device calls, I/O
//!          \         /
//!           ▼       ▼
//!          beatlm-host            what both compile; reads no clock
//!               │
//!               ▼
//!            beatlm               the umbrella: the core's decisions on wgpu
//!               │
//!               ▼
//!          beatlm-core            the domain: decides
//! ```
//!
//! The two hosts run the same loads and must report them the same way, so a
//! browser report compares with a native one line by line. The code that makes
//! that so lives here once:
//!
//! - [`workload`] and its `shaders.wgsl`: the synthetic load, one busy-loop
//!   compute kernel per slice and one render pass, and the render order;
//! - [`slots`]: a lowered program realised with one input and one output slot
//!   per job in flight ([`slots::Net`]), which the depth model runs through;
//! - [`pass_cost`]: the `pass-cost` workload's encoding, cells and lines;
//! - [`pass_overlap`]: the `pass_overlap_*` lines, read from a frame's
//!   timestamps;
//! - [`presets`]: the cost model's estimators and priors, the margin, and the
//!   windows and frame counts derived from the default estimator's window;
//! - [`stats`]: the percentiles, the median and the ratio lines both reports
//!   and calibration quote, and the fewest measurements a slice's median is
//!   read from.
//!
//! # What goes here
//!
//! **Only what both hosts compile.** Code one host uses stays in that host
//! until the other uses it too, so this crate does not grow into a place for
//! whatever has no other home.
//!
//! **It reads no clock, blocks on nothing and performs no I/O**, like the
//! core: frame starts, periods and measured times come in as arguments,
//! and each host measures its CPU times with its own clock (`Instant` in the
//! bench, `performance.now()` in a page) and hands them in. Unlike the core it
//! may use wgpu, to encode the shared loads. It turns on no wgpu feature of its
//! own, needs no executor and starts no thread, so it builds for
//! `wasm32-unknown-unknown`; `just wasm-check` lints it there on every change.
//!
//! # Why a crate of its own
//!
//! These files were the bench's, and the browser host compiled them from the
//! bench's source tree with `#[path]`. Neither existing crate can hold them.
//! The umbrella is published, and a harness's busy-loop load, its slots and
//! its pass-cost measurement would become public API with a semver promise.
//! The bench cannot lend them as a library: it reads `Instant`, which panics
//! on `wasm32-unknown-unknown`, and turns on wgpu's `counters` feature, which
//! a dependency on it would turn on in the browser build too.
//!
//! The rest of the frame's bookkeeping that both hosts do (the budget and
//! forecast, the admission of an input, the delivery and miss counts, the
//! report's shared lines) is written in each host; `beatlm-web`'s crate
//! documentation ("What is shared with the bench") says why.
//!
//! The crate is not published (`publish = false`): its API is the two hosts'
//! and changes with them.

// Also set as a workspace lint; see the same attribute in `beatlm-core`.
#![warn(missing_docs)]

pub mod pass_cost;
pub mod pass_overlap;
pub mod presets;
pub mod slots;
pub mod stats;
pub mod workload;
