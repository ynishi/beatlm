//! Domain core of beatlm.
//!
//! This crate decides *what* GPU work runs in a frame and *when* its result is
//! delivered; it never touches a GPU. Time and measurements arrive as values
//! from the caller, and decisions leave as values, so the same logic can be
//! tested without an adapter and behaves identically on native and on wasm.
//!
//! The vocabulary it owns:
//!
//! - **Clock** — frame index, frame period and musical time (tempo, beat phase).
//! - **Budget** — the time a frame can spend on inference, after rendering.
//! - **SyncPolicy** — the delivery promise: a fixed latency of `k` frames (L1),
//!   completion inside the same frame (L2), or landing on a beat boundary.
//! - **Plan** — a compiled graph: kernels, the memory plan, the slicing into
//!   per-frame work.
//! - **CostModel** — measured kernel durations, updated only from timestamps.
//!
//! GPU APIs, model file formats and shader languages are translated into this
//! vocabulary by the `beatlm` crate; none of them appear here.
