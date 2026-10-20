//! beatlm runs neural networks on the same GPU queue as a real-time renderer,
//! inside a per-frame time budget, and delivers each result on a promised
//! frame.
//!
//! The decisions — what runs this frame, when a result is due — belong to
//! [`beatlm_core`]. This crate carries them out on wgpu and translates what
//! comes from outside (adapter limits, model files) into the core's terms.

pub use beatlm_core;
