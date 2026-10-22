//! beatlm runs neural networks on the same GPU queue as a real-time renderer,
//! inside a per-frame time budget, and delivers each result on a promised
//! frame.
//!
//! The decisions — what runs this frame, when a result is due — belong to
//! [`beatlm_core`], whose crate documentation states the promises and the
//! rules behind them. This crate carries those decisions out on wgpu and
//! translates what comes from outside into the core's terms.
//!
//! # Architecture
//!
//! This crate is the boundary between the core and everything it must not
//! know about. Each kind of outside vocabulary has one place where it is
//! translated, and nothing past that place sees it:
//!
//! | Outside | Translated into | Why the translation is here |
//! |---|---|---|
//! | wgpu adapter limits and features | a [`DeviceProfile`](beatlm_core::device::DeviceProfile) of plain values ([`gpu::profile`]) | the core plans memory and picks kernel variants from values, so it never depends on a wgpu version |
//! | GPU timestamp queries | measured durations ([`gpu`]) | the core's cost model is fed measurements, and only measurements |
//! | model files (safetensors first) | named, shaped tensors checked against what a model's graph needs ([`load`]), then a graph of tensors and operations with each weight's numbers in the IR's layouts ([`model`]) | weight formats change and multiply; the graph does not |
//! | [kernel IR](beatlm_core::kernel) | WGSL text, validated by naga ([`codegen`]) | browsers accept only WGSL, and generated text can be read, diffed and checked by naga's parser |
//!
//! Shaders are generated as WGSL text rather than built as naga IR. Building IR
//! directly makes the generator responsible for invariants naga's parser
//! already keeps (handles, arenas, emission ranges), and a mistake surfaces as
//! a diagnostic with no source location. Text can be inspected as it is.
//!
//! Features are layered over a baseline every adapter has. Kernels always
//! have a variant that needs no subgroups, because wgpu exposes subgroups on
//! native backends only; memory is planned against the default WebGPU limits
//! ([`DeviceProfile::webgpu_baseline`](beatlm_core::device::DeviceProfile::webgpu_baseline))
//! and grows only when the adapter reports more. A memory plan becomes buffers
//! once, in [`gpu::Arenas`].
//!
//! # Design
//!
//! ## The host's side of a frame
//!
//! The promises in [`beatlm_core`] hold only if the host keeps this order
//! every frame, on the queue the renderer uses:
//!
//! 1. Report the frame's start and the render time measured last frame, bring
//!    the scheduler's forecast of later frames' budget and, on a tempo change,
//!    its beats up to date, and submit the frame's input, so a same-frame job
//!    is planned with it and an on-beat job is placed with the current
//!    forecast.
//! 2. Submit the slices the frame's plan hands out.
//! 3. Render, using the results the plan delivers and delaying the source by
//!    the promised latency.
//! 4. Feed the timestamps that came back into the cost model.
//!
//! Step 2 before step 3 is the load-bearing part: queue order is what makes a
//! slice complete before the frame that uses it is drawn.
//!
//! The timestamps of step 4 come from [`gpu::SliceTimer`], and reach it along
//! the same steps: the frame's passes take their timestamp writes from it, the
//! frame's encoder ends with the resolve, and the
//! [`ResolvedFrame`](gpu::ResolvedFrame) that returns goes back to the timer
//! ([`gpu::SliceTimer::submitted`]) straight after the `queue.submit` carrying
//! that encoder. Only frames handed back that way are read; an encoder dropped
//! instead of submitted takes its frame's timing with it, which is lost and
//! never misread as a later frame's ([`gpu`] has the rule, and why the step is
//! the host's). A host that learns the device rejected a submission marks
//! what it submitted before collecting again
//! ([`gpu::SliceTimer::invalidate_submitted`]), or, learning it per
//! submission, holds each frame until it accepts it
//! ([`gpu::SliceTimer::submitted_held`]).
//!
//! ## Concurrency model
//!
//! **A synchronous `tick`, called once per frame on the thread that runs the
//! host's frame loop, which decides, encodes and submits before it returns.**
//! GPU completion is never awaited inside a frame: timestamp readbacks finish
//! through map callbacks and are collected by a later `tick`.
//!
//! The engine is driven by the host's frame loop — winit natively,
//! `requestAnimationFrame` in a browser, where the main thread can never block
//! and GPU completion arrives as promises. The usual default of an async
//! service on a dedicated thread assumes a library that is called into; here
//! the property that matters is the time from frame start to submission, and
//! that one code path runs on native and in a browser. Three candidates were
//! measured with `beatlm-bench` under one configuration (latency 2, four
//! synthetic slices of about 1 ms, a synthetic render of about 2 ms, 600 frames
//! at 60 Hz, three runs each) on an Apple M1 Pro, Metal, on 2026-10-21:
//!
//! | Candidate | frame start to submit, p99 per run | wrong-latency deliveries |
//! |---|---|---|
//! | synchronous `tick` | 2.0 / 2.1 / 1.8 ms | 0 |
//! | async handle over a dedicated GPU thread, fed by a channel | 2.3 / 2.3 / 2.6 ms | 0 |
//! | `async` `tick` awaited by the host | 3.7 / 1.9 / 2.7 ms | 0 |
//!
//! For the thread candidate the figure runs from the frame loop's frame start
//! to the GPU thread's submit, so it includes the channel hand-off (0.7 ms at
//! p99) and any frames queued behind a slow one; the GPU thread does not drop
//! periods the way the frame loop does. It measures that candidate as a host
//! would experience it, not the engine's cost alone.
//!
//! Why the synchronous `tick`:
//!
//! - **The others buy nothing measurable.** The spread between runs of one
//!   candidate is as large as the spread between candidates. The breakdown
//!   says where the time goes: planning in the core stays under 0.1 ms at
//!   p99, while command encoding (1.4–2.2 ms p99) and `queue.submit`
//!   (0.5–1.2 ms p99) carry the variance in every candidate. Moving the same
//!   wgpu and driver calls to another thread, or behind an `await`, does not
//!   change them; issuing fewer of them does.
//! - **The thread does not exist in a browser.** wgpu's WebGPU device is not
//!   `Sync` (it holds an `Rc`), so the dedicated-thread candidate does not
//!   compile for `wasm32-unknown-unknown`, and `std::thread::spawn` is
//!   unsupported there without atomics.
//! - **The `async` tick has nothing to wait for.** Readback completes through
//!   callbacks — natively run inside the `tick`'s non-blocking `device.poll`,
//!   in a browser as promise callbacks between frames — so an `await`
//!   in the frame only adds an executor. A `tick` that must not block the
//!   browser's main thread is synchronous by construction.
//!
//! What follows from it: the CPU-side jitter is attacked by encoding fewer
//! passes per frame — fusion and fewer, larger dispatches — not by
//! concurrency. [`beatlm_core`] is unaffected either way: it takes time and
//! measurements as arguments and has no concurrency model of its own.

// Also set as a workspace lint; see the same attribute in `beatlm-core`.
#![warn(missing_docs)]

pub mod codegen;
pub mod gpu;
pub mod load;
pub mod model;

pub use beatlm_core;
