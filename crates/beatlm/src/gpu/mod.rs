//! The GPU side of beatlm: what wgpu reports, translated into the core's
//! vocabulary.
//!
//! This module is the one place wgpu timestamp queries are seen. Everything
//! past it deals in [`beatlm_core::time::Micros`] and
//! [`beatlm_core::time::FrameIndex`], which is what lets the
//! [`CostModel`](beatlm_core::cost::CostModel) be fed from measurements without
//! knowing what a tick is. The rules this module owns:
//!
//! - **A tick is converted once, here.** A timestamp pair becomes a duration in
//!   microseconds using the queue's timestamp period, and nothing outside this
//!   module sees a raw tick.
//! - **Readback never stalls the frame loop.** Results come back through a
//!   ring of buffers, one slot per frame in flight. A frame whose slot is still
//!   in flight is not timed and the skip is counted; waiting for the slot would
//!   move GPU latency onto the host's frame, which is the cost this module
//!   exists to avoid.
//! - **Nothing here blocks or reads a clock.** The host decides when to poll
//!   the device, and the module only reports what has completed by then. That
//!   keeps one code path valid on native, where polling is the host's call, and
//!   in a browser, where it can never block.
//! - **Only timestamps are read back.** Timing data is eight bytes per query;
//!   the steady-state loop reads nothing else from the GPU.
//! - **Only a frame the host says it submitted is read.** The timer cannot see
//!   whether an encoder reached the queue, and mapping a slot whose copy never
//!   ran succeeds on an earlier frame's timestamps. So a resolve returns a
//!   [`ResolvedFrame`] that the host hands back after `queue.submit`, and one
//!   dropped instead — with its encoder, on an error path — frees its slot
//!   unread: a frame's timing may be lost, never misread.
//! - **A frame the host marks as rejected is never decoded.** Submitted is
//!   not accepted: a command buffer the device rejects never runs, and its
//!   slot maps on an earlier frame's timestamps all the same. The host that
//!   learns of a rejection marks what it submitted, and the timer frees those
//!   slots unread and counts them failed
//!   ([`SliceTimer::invalidate_submitted`] has the rule).
//!
//! Memory is the other thing this module realises. [`profile`] is the one place
//! wgpu's limits and features are read, and becomes the core's
//! [`DeviceProfile`](beatlm_core::device::DeviceProfile); [`required_limits`]
//! says what a device must be created with to hold a
//! [`MemoryPlan`](beatlm_core::memory::MemoryPlan), and refuses a plan the
//! adapter cannot hold; [`Arenas`] turns the plan into buffers, once, so the
//! steady-state loop creates none.
//!
//! [`SliceTimer`] is the entry point for timing; the host's side of the frame it plugs into
//! is described in the [crate documentation](crate#the-hosts-side-of-a-frame).

mod arenas;
mod limits;
mod profile;
mod ring;
mod timer;

pub use arenas::{ArenaError, Arenas};
pub use limits::{LimitsError, required_limits};
pub use profile::profile;
pub use timer::{
    FrameTimestamps, FrameTiming, MAX_SLICES, PassSpan, RING_DEPTH, ResolvedFrame, SliceTimer,
    TimerError, TimerStats,
};
