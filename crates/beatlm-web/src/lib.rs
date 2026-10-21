//! Browser host for beatlm: the bench's synthetic workload, driven from
//! `requestAnimationFrame` on WebGPU, with the bench's report written into the
//! page.
//!
//! It is to a browser what `beatlm-bench --workload synthetic` is to a native
//! machine: four slices of a busy-loop kernel submitted every frame under the
//! fixed-latency promise, a render pass of a full-screen triangle, and the
//! GPU and CPU time each took. The point is the host, not the load: a frame
//! loop the browser drives, a clock it provides, a device requested with the
//! limits a plan needs, and readbacks that complete between frames.
//!
//! # Running it
//!
//! From the workspace root:
//!
//! ```text
//! just setup-web    # once: wasm-bindgen-cli at the version Cargo.lock pins, into cargo's bin
//! just build-web    # release build for wasm32-unknown-unknown, glue and page into target/beatlm-web/
//! just serve-web    # serves target/beatlm-web/ on http://127.0.0.1:8000
//! ```
//!
//! then open `http://127.0.0.1:8000/?frames=600&latency=2` in a browser with
//! WebGPU; a run long enough that a display's departure from exactly 60 Hz
//! would show in `frames_dropped` and `frame_start_lateness_us`, were the
//! pacing to drift, is `?frames=3000`. The run starts when the page loads; the
//! report replaces the text under the canvas when it ends, with `status: done`
//! as its last line, and is logged to the console. An error ends the report with `error:` and
//! `status: failed`.
//!
//! The query string takes `frames` (default 600) and `latency` (default 2),
//! the bench's defaults, and `arena_bytes`: plan one arena of that many bytes
//! against the adapter's limits, create the device with the limits it needs,
//! and realise the arena on it before the run, which checks the limit path in
//! a browser. Anything else in the query is refused.
//!
//! The page needs no cross-origin isolation, so a plain static server serves
//! it: the host runs on the page's main thread only and shares no memory with
//! a worker, which is what the `Cross-Origin-Opener-Policy` and
//! `Cross-Origin-Embedder-Policy` headers would be for. One consequence is
//! the clock's resolution: the High Resolution Time specification lets a
//! browser coarsen `performance.now()` to 100 µs in a page that is not
//! isolated (5 µs in one that is), so the report's CPU times may be quantised
//! that coarsely; what Chrome does was not checked here.
//!
//! # The host's side of a frame, in a browser
//!
//! The order is the one the `beatlm` crate documentation sets out ("The host's
//! side of a frame"), and each step is the bench's synchronous `tick`. One
//! `requestAnimationFrame` callback runs:
//!
//! 0. **Pace.** The callback is the frame. Give the pacer the callback's
//!    timestamp argument: it returns the frame's index, its start and the
//!    display's period, which the budget and the core's `FrameInfo` take
//!    ([`pace`] has the rule, and why the timestamp and not a fresh clock
//!    reading).
//! 1. **Decide.** Set this frame's budget from the last render measured and
//!    the forecast of later frames from the median of the recent renders,
//!    submit the frame's input as a job of [`config::SLICES`] slices under
//!    fixed latency `k`, and plan the frame.
//! 2. **Slices.** Encode one compute pass per slice the plan hands out, each
//!    with timestamp writes.
//! 3. **Render.** Take the canvas's texture for this frame and encode the
//!    render pass into it, timed; resolve the frame's timestamps, submit, and
//!    hand the texture back for presentation. The browser shows it once the
//!    callback returns.
//! 4. **Measurements.** Request the map of this frame's timestamps, collect
//!    every earlier frame whose map has completed into the cost model and the
//!    render estimate, then deliver the plan's results and count its misses.
//!
//! Then it asks for the next callback. Nothing in a callback waits.
//!
//! **How readbacks complete.** `mapAsync` returns a promise, and wgpu runs the
//! map callback from that promise's `then`, on the event loop, where it marks
//! the readback slot as done (`beatlm::gpu::SliceTimer`). A promise cannot
//! settle while a callback is running, so a frame's timing is never collected
//! by the callback that submitted it: the earliest is the next one, once the
//! GPU has finished the frame. The timer keeps
//! [`RING_DEPTH`](beatlm::gpu::RING_DEPTH) frames in flight; a frame whose
//! slot is still waiting for its map when the frame needs it is not timed, and
//! `timings_skipped` counts it, as natively. `Device::poll` is not
//! called: in a browser it does nothing.
//!
//! # What differs from the native bench, and why
//!
//! - **Pacing.** The browser chooses when a callback runs, at the display's
//!   rate; the bench sleeps until the next period of a fixed 60 Hz grid. Here
//!   every callback is a frame, it starts at the callback's timestamp, and its
//!   period is the display's, the median of the per-period intervals between
//!   callbacks ([`pace`] has the rule and why a fixed grid was dropped: a display that
//!   is not exactly 60.000 Hz drifts off it and a healthy run is counted as
//!   dropping frames). The native bench keeps its grid, which its published
//!   measurements were made with. The consequences are in the report: the
//!   frame budget is the display's period less the render and the margin, so
//!   at 120 Hz it is about 8.3 ms less than at 60 Hz, and a latency promised in
//!   `k` frames is `k` display periods, half the time at 120 Hz; the report
//!   states the period (`display_period_us`) so runs compare only at the same
//!   one. `frames_dropped`, `frame_start_lateness_us`, `promised_latency_ms`
//!   and `gpu_frames_over_period` keep their names with the meaning the
//!   [`report`] documentation gives. A page in a background tab gets no
//!   callbacks, and its periods are dropped.
//! - **Waiting.** A page can never block, so the warm-up is awaited on the
//!   event loop before the first frame, and the timings still in flight after
//!   the last frame are collected over one second of further callbacks
//!   rather than waited for; any not back by then are left out, and
//!   `timings_collected` says how many came back.
//! - **The render target** is the canvas's texture, acquired every frame,
//!   where the bench draws into an offscreen texture; it has the same format
//!   and size, so the pass is the same work. `encode_us` includes acquiring
//!   it.
//! - **The device** has the WebGPU default limits unless a plan needs more,
//!   and then exactly what the plan needs
//!   ([`beatlm::gpu::required_limits`]); one the adapter cannot give is
//!   refused, never cut down. Timestamp queries are requested when the adapter
//!   has them; without them the run still keeps or misses its deadlines, but
//!   no GPU pass is timed, the budget and the slice estimates stay at their
//!   priors, and the report's `timestamp_query` line says so.
//! - **GPU times.** Chrome is documented to quantise timestamp-query results
//!   (to 100 µs) unless its "WebGPU Developer Features" flag is on; that was
//!   not confirmed from source here, and a report's GPU times should be read
//!   with it in mind.
//! - **One concurrency model.** Only the synchronous `tick` runs here; the
//!   `beatlm` crate documentation ("Concurrency model") says why the other two
//!   candidates do not exist or buy nothing in a browser. The bench's `--mode`,
//!   `--runs`, competing load and planned workloads, and wgpu's object
//!   counters, which the WebGPU backend does not keep, have no counterpart.
//!
//! # What is shared with the bench
//!
//! The load is the bench's own: [`workload`] and its `shaders.wgsl`, and the
//! percentiles of the report, are compiled from `beatlm-bench`'s files by
//! path, so the two hosts time the same kernels at the same sizes and quote
//! the same statistic. The timer, the limits and the arenas come from
//! `beatlm::gpu`, and the decisions from `beatlm_core`.
//!
//! The frame's bookkeeping — the budget and forecast, the delivery and miss
//! counts, what each measurement feeds — and the report's shared lines follow
//! the bench's `engine.rs` and `report.rs` for the synthetic workload, and a
//! change to one is owed to the other. They are written twice because the
//! bench's engine reads `std::time::Instant` and blocks on the device for its
//! warm-up and drain, neither of which exists in a page, and carries the
//! planned workloads and the bench's options; separating a clock-free frame
//! step from it would change the code the bench's published measurements
//! came from, which is a change of its own.
//!
//! # Builds
//!
//! The browser side is compiled only for `wasm32`; a native build of the
//! workspace compiles the query parsing, the pacer and the report, which are
//! tested natively. `just check` lints the crate for `wasm32-unknown-unknown`
//! (`wasm-check`) and needs no `wasm-bindgen` command; only `just build-web`
//! does.

// Also set as a workspace lint; see the same attribute in `beatlm-core`.
#![warn(missing_docs)]

pub mod config;
pub mod pace;
pub mod report;
// The bench's percentiles, by path, so the report quotes the same statistic.
#[path = "../../beatlm-bench/src/stats.rs"]
mod stats;
// The bench's synthetic load, by path; the file says why it is shared.
#[path = "../../beatlm-bench/src/workload.rs"]
pub mod workload;

#[cfg(target_arch = "wasm32")]
mod web;
