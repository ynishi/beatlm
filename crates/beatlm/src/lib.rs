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
//! | wgpu adapter limits and features | a device profile of plain values | the core plans memory and picks kernel variants from values, so it never depends on a wgpu version |
//! | GPU timestamp queries | measured durations | the core's cost model is fed measurements, and only measurements |
//! | model files (safetensors first) | a graph of tensors and operations | weight formats change and multiply; the graph does not |
//! | kernel IR | WGSL text, validated by naga | browsers accept only WGSL, and generated text can be read, diffed and checked by naga's parser |
//!
//! Shaders are generated as WGSL text rather than built as naga IR. Building IR
//! directly makes the generator responsible for invariants naga's parser
//! already keeps (handles, arenas, emission ranges), and a mistake surfaces as
//! a diagnostic with no source location. Text can be inspected as it is.
//!
//! Features are layered over a baseline every adapter has. Kernels always
//! have a variant that needs no subgroups, because wgpu exposes subgroups on
//! native backends only; buffers are split against the default WebGPU limits
//! and grow only when the adapter reports more.
//!
//! # Design
//!
//! ## The host's side of a frame
//!
//! The promises in [`beatlm_core`] hold only if the host keeps this order
//! every frame, on the queue the renderer uses:
//!
//! 1. Report the frame's start and the render time measured last frame, and
//!    submit the frame's input, so a same-frame job is planned with it.
//! 2. Submit the slices the frame's plan hands out.
//! 3. Render, using the results the plan delivers and delaying the source by
//!    the promised latency.
//! 4. Feed the timestamps that came back into the cost model.
//!
//! Step 2 before step 3 is the load-bearing part: queue order is what makes a
//! slice complete before the frame that uses it is drawn.
//!
//! ## Concurrency model
//!
//! Undecided, and decided by measurement rather than by default. The engine is
//! driven by the host's frame loop — winit natively, `requestAnimationFrame`
//! in a browser, where the main thread can never block and GPU completion
//! arrives as promises. The usual default of an async service on a dedicated
//! thread assumes the library is called into; here the property that matters
//! is the time from frame start to submission. The candidates:
//!
//! - a synchronous `tick` called once per frame on the render thread, which
//!   decides, encodes and submits before returning;
//! - an async handle over a dedicated GPU thread, fed frame events over a
//!   channel;
//! - an `async` `tick` awaited by the host.
//!
//! They are compared against a synthetic GPU load on the time from frame start
//! to submission, its jitter at the 99th percentile, and whether one code path
//! holds on native and on wasm. The choice and the numbers that made it replace
//! this section. Whichever wins, [`beatlm_core`] is unaffected: it takes time
//! and measurements as arguments and has no concurrency model of its own.

// Also set as a workspace lint; see the same attribute in `beatlm-core`.
#![warn(missing_docs)]

pub use beatlm_core;
