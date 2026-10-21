//! Browser host for beatlm: the bench's synthetic workload or Depth Anything
//! V2 Small, driven from `requestAnimationFrame` on WebGPU, with the bench's
//! report written into the page.
//!
//! It is to a browser what `beatlm-bench` is to a native machine. With the
//! synthetic workload, `--workload synthetic`: four slices of a busy-loop
//! kernel submitted every frame under the fixed-latency promise, a render pass
//! of a full-screen triangle, and the GPU and CPU time each took. With the
//! depth model, `--workload depth`: the model built from its real weights,
//! calibrated, split and run as the job under the same promise, beside a render
//! pass that shows its source and its depth, and checked against the original
//! implementation's output ("The depth model" below). The point of the first is
//! the host: a frame loop the browser drives, a clock it provides, a device
//! requested with the limits a plan needs, and readbacks that complete between
//! frames. The point of the second is whether the native default holds there.
//!
//! # Running it
//!
//! From the workspace root:
//!
//! ```text
//! just setup-web    # once: wasm-bindgen-cli at the version Cargo.lock pins, into cargo's bin
//! just build-web    # release build for wasm32-unknown-unknown, glue and page into target/beatlm-web/
//! just serve-web    # serves the page and the models directory on http://127.0.0.1:8000
//! ```
//!
//! then open `http://127.0.0.1:8000/?frames=600&latency=2` in a browser with
//! WebGPU for the synthetic workload, or
//! `http://127.0.0.1:8000/?model=depth&side=266` for the depth model, which
//! needs `just setup-depth-anything` first and, for its check,
//! `just reference-depth-anything 266` (`518` without the argument). A run long
//! enough that a display's departure from exactly 60 Hz would show in
//! `frames_dropped` and `frame_start_lateness_us`, were the pacing to drift, is
//! `?frames=3000`. The run starts when the page loads; the report replaces the
//! text under the canvas when it ends, with `status: done` as its last line,
//! and is logged to the console. An error ends the report with `error:` and
//! `status: failed`.
//!
//! The query string's parameters are [`config`]'s: `frames` (default 600),
//! `model` (`synthetic` or `depth`), `side` (the depth model's, default 266),
//! `latency` in frames or `latency_ms` in time, and `arena_bytes`, which plans
//! one arena of that many bytes against the adapter's limits, creates the
//! device with the limits it needs, and realises the arena on it before the
//! run, to check the limit path in a browser. `estimator` chooses the
//! statistic the cost model plans with, for either job: the bench's
//! `--estimator` names `quantile`, `mean-sigmas` and `capped`, or `quantile:<q>`
//! and `quantile:<q>:<window>` (window 32 by default), `mean-sigmas:<alpha>:<sigmas>`
//! and `capped:<alpha>:<sigmas>:<cap>`; the default is the core's (`quantile`),
//! and [`config`] has every form, the ranges and what
//! the choice reaches, and the four estimators measured in a browser at
//! 120 Hz ("Which estimator, measured in a browser"). Anything else in the
//! query is refused.
//!
//! **Serving.** `just serve-web` runs `scripts/serve_web.py`, which serves the
//! built page at `/` and the models directory at `/models/`
//! ([`depth::MODELS_URL`]) from one origin, bound to 127.0.0.1, read-only and
//! without directory listings; the script says what it refuses. The models
//! directory is `BEATLM_MODELS_DIR`, or `models/` at the workspace root, the
//! rule the native tests read it by. The weights and references are fetched
//! from there and never copied into the page's build output: they are
//! third-party material, and a build output that held them would be
//! redistribution once published. Without a models directory the server still serves the page, and
//! the synthetic workload needs nothing else.
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
//! # The depth model
//!
//! `?model=depth&side=N` runs Depth Anything V2 Small at `N` pixels a side, as
//! `beatlm-bench --workload depth --size N` does natively, from the same code
//! wherever there is code to share. Before the first frame, in order:
//!
//! 1. **Latency.** The display's period is measured over the first callbacks
//!    by the [`pace`] rule, and a latency asked in time (`latency_ms`, and the
//!    depth model's default) is turned into frames at it
//!    ([`depth::frames_for`] says how a time becomes frames). The promise is
//!    fixed in frames for the run, as the scheduler keeps it. The same period
//!    is the loop's prior ([`pace::Pacer::with_prior`]), so on a 120 Hz display
//!    the model's first frames are not budgeted for 60 Hz.
//! 2. **Weights.** The safetensors file is fetched from `/models/` as one
//!    buffer and read by `beatlm::load`, the native reader, which works on
//!    bytes (`Archive::parse`; only `read_file` needs a file system), and
//!    checked against the inventory `beatlm::load::depth_anything_v2`
//!    derives from the Small configuration. Anything but an exact match fails
//!    the run, as natively. The graph is `beatlm::model::depth_anything_v2`'s
//!    at the side, lowered with fusion on, and every weight's numbers come from
//!    the file; every slot's input is the reference input of that side
//!    (`beatlm::load::depth_anything_v2::reference::input`), so the loop
//!    computes what the check compares.
//! 3. **Device.** The bench's `Net` ([`slots`], compiled by path) plans the
//!    model with one input and one output slot per job that can be in flight,
//!    against the baseline profile, before the device exists; the device is
//!    created with what that plan needs (`beatlm::gpu::required_limits`). At
//!    518 pixels the plan needs an arena of 134 217 600 bytes, 128 bytes under
//!    the WebGPU default storage binding size; a plan against the baseline
//!    never has an arena larger than the defaults allow, so the request comes
//!    out as the defaults, and it is made from the plan all the same, so that
//!    the device is asked for what the plan needs and not for what one model
//!    happens to fit.
//! 4. **Calibration**, as the bench's `calibrate.rs`: the unsplit model is run
//!    whole for [`depth::CALIBRATION_FRAMES`] timed frames behind the synthetic
//!    render pass, each kernel's median over the second half decides how it is
//!    split for [`depth::MAX_DISPATCH_US`] (the core's
//!    `Program::parts_within`), and the split program is run the same way into
//!    the cost model the loop starts from, built with the page's `estimator`;
//!    the split is decided from medians before that model exists, so the
//!    estimator does not change it. Without it every slice would start
//!    at the 2 ms prior, every job would be projected past its deadline, and
//!    only the slices the loop's jobs ran could bring the estimates down.
//!    Each calibration frame starts at an animation
//!    frame, and the next at the first animation frame after its timings came
//!    back, where the bench waits on the device. It needs timestamp queries,
//!    and the run fails without them.
//! 5. **Realise and check.** The split program is realised with its slots,
//!    every slot's job is run once, each its own submission, and slot 0's
//!    depth, the reference input's, is read back and compared with the
//!    reference made at the same side, fetched from `/models/` (the
//!    `reference` and `depth_check` report lines). The input, the error
//!    measures and the tolerances are `beatlm::load::depth_anything_v2::reference`'s,
//!    which the native tests use, so the page holds the port to the native
//!    standard; the reference's input file is also compared with the formula,
//!    bit for bit. A reference the server does not have is reported, with the
//!    recipe that makes it, and the timing run goes on.
//!
//! Then the loop runs as for the synthetic workload, with the depth model's
//! kernel parts as the slices, the bench's `--submit free` (an input is
//! submitted only once the job before it has had all its slices; the bench's
//! crate documentation says why that is the depth default), and the display as
//! the render pass: the delivered slot's source on the left and its depth on
//! the right, the depth in grey between the smallest and the largest the check
//! computed. The display's shader carries the synthetic render pass's
//! per-pixel loop, so the render pass costs what the bench's does beside the
//! depth model and a frame's budget is the one the native default was measured
//! under. Like the bench's render pass, it reads no buffer a slice writes: a
//! job's depth is copied into the display's own buffer after its last slice,
//! so the render's measurement, which the budget takes, cannot carry a wait
//! for the slices ([`render_cost`]). Why a job was missed, and what the
//! projection read when it abandoned one, is in the report's diagnosis lines
//! ([`diagnosis`]).
//!
//! **The native default in time.** The bench chose 266 pixels at `k` = 5 on an
//! Apple M1 Pro, a decision for 83.3 ms of latency and 12 results a second at
//! its fixed 60 Hz grid. Frames here follow the display, so 5 frames is 41.7 ms
//! at 120 Hz: another promise, with half the time for a job. The page therefore
//! takes a latency in time (`latency_ms`), and the depth model promises the
//! native default's time unless told otherwise, as many frames as that takes at
//! the measured period (10 at 120 Hz). The report says, whatever the run was,
//! what the native default takes at the display's period
//! (`native_default`), and whether it held (`native_default_holds`): `yes`,
//! `no` with the criteria it failed, or `not tested` when the run was another
//! side or latency. It holds by the native criteria
//! ([`depth::native_default`]): GPU time per frame within the display's period
//! at the 99th percentile, no delivery at another latency, and at least
//! [`depth::DELIVERY_FLOOR`] of the deadlines met. The other figures a side
//! is compared by are the shared lines (`gpu_frame_us`, `delivery_rate`,
//! `promised_latency_ms`) and calibration's GPU time per run of the whole
//! model (`gpu_run_us`, and `calibrated_job_us`, the bench's line).
//!
//! **Whether the promise can be kept at all.** `capacity_frames_needed`
//! compares what one whole job costs in the loop (the sum of its slices' mean
//! measured durations) with the frames the promise gives it (`k + 1`, the
//! deadline frame included) at the median expected budget: the frames a job
//! needs even if every estimate were exact. `verdict=infeasible` means no
//! estimator can deliver the promise on this display with this model;
//! [`depth::Capacity`] defines the inputs, why the count is a lower bound and
//! what `not_ruled_out` does not say.
//!
//! ## What a browser run measured
//!
//! **At 120 Hz in Chrome the native default does not hold.** Chrome 154
//! (WebGPU) on an Apple M1 Pro driving a 120 Hz display (period 8 300 µs),
//! 2026-10-22, Depth Anything V2 Small:
//!
//! | URL | promise | delivered / submitted | build |
//! |---|---|---|---|
//! | `?model=depth&side=266&frames=3000` | 10 frames, 83 ms | 0 / 2 898 | display drawn from the arenas |
//! | `?model=depth&side=266&frames=3000&latency_ms=200` | 25 frames, 207.5 ms | 31 / 2 592 | display with its own buffers, diagnosis lines |
//! | `?model=depth&side=518&latency_ms=600&frames=1200` | 72 frames, 597.6 ms | 3 / 265 | display drawn from the arenas |
//!
//! In each, the depth matched the reference within the native tolerances
//! (266 pixels: largest relative error 2.6e-6; 518: 2.1e-6). A whole run of
//! the model measured 28.7 ms at 266 pixels in calibration (`gpu_run_us`
//! p50), and a frame's budget for slices is about 4.3 ms at 120 Hz (8.3 ms
//! less a 1.9 ms render and the 2 ms margin), against 12.6 ms on the bench's
//! 60 Hz grid. What the diagnosis lines of the second run say is in
//! [`diagnosis`] ("What it found in Chrome, before #41"): almost every miss
//! was a job abandoned before its first slice, refused because its estimate,
//! which had climbed during the loop from 34 ms to 102 ms, projected past its
//! deadline, while the GPU sat idle. These runs are of the scheduler before
//! #41, which abandoned such a job even with none behind it, so its slices
//! were never measured and the estimates could not come down; the core keeps
//! such a job since ([`diagnosis`] says what that changes here). Why the
//! estimates climb belongs to the core's cost model and scheduler rather than
//! to this page, which does not change the scheduling for it. A `no` from
//! `native_default_holds` is therefore a measurement of this configuration,
//! not a fault the page is known to have; the line says which criteria failed
//! and, for delivery, the run's own abandoned jobs and estimates.
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
//!   refused, never cut down. A browser reports a buffer it rejects later
//!   than it hands the buffer out, so the arenas are created inside error
//!   scopes that are awaited. Timestamp queries are requested when the
//!   adapter has them; without them the synthetic run still keeps or misses
//!   its deadlines, but no GPU pass is timed, the budget and the slice
//!   estimates stay at their priors, and the report's `timestamp_query` line
//!   says so. The depth model refuses to run without them (calibration needs
//!   them).
//! - **The depth model's input and output** are the bench's simplifications:
//!   every slot holds the same fixed image, uploaded once, and nothing but the
//!   check's one result is read back. Its depth is drawn, not consumed. A
//!   missed result leaves the last one on the screen, as the miss policy says,
//!   and the display's copy of the slot it is in is replaced when the next job
//!   of that slot completes, so the picture may then show a later result than
//!   the one delivered; the measurements do not depend on it.
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
//! The load is the bench's own: [`workload`] and its `shaders.wgsl`, the
//! depth model's realisation [`slots`], and the percentiles of the report and
//! calibration's median, are compiled from `beatlm-bench`'s files by path, so
//! the two hosts time the same kernels at the same sizes and quote the same
//! statistic. The model, its loader and its reference comparison come from
//! `beatlm::model` and `beatlm::load`, the kernels from `beatlm::codegen`, the
//! timer, the limits and the arenas from `beatlm::gpu`, and the decisions from
//! `beatlm_core`.
//!
//! The frame's bookkeeping — the budget and forecast, the admission of an
//! input, the delivery and miss counts, what each measurement feeds — the
//! report's shared lines and calibration's two passes follow the bench's
//! `engine.rs`, `report.rs` and `calibrate.rs`, and a change to one is owed to
//! the other. They are written twice because the
//! bench's engine reads `std::time::Instant` and blocks on the device for its
//! warm-up and drain, neither of which exists in a page, and carries the
//! planned workloads and the bench's options; separating a clock-free frame
//! step from it would change the code the bench's published measurements
//! came from, which is a change of its own.
//!
//! # Builds
//!
//! The browser side is compiled only for `wasm32`; a native build of the
//! workspace compiles the query parsing, the pacer, the depth model's rules
//! and the report, which are tested natively, and a test validates the
//! display's shader with naga. `just check` lints the crate for `wasm32-unknown-unknown`
//! (`wasm-check`) and needs no `wasm-bindgen` command; only `just build-web`
//! does.

// Also set as a workspace lint; see the same attribute in `beatlm-core`.
#![warn(missing_docs)]

pub mod config;
pub mod depth;
pub mod diagnosis;
pub mod pace;
pub mod render_cost;
pub mod report;
// The bench's realisation of a program with one slot per job in flight, by
// path, so a page and the bench run the depth model through the same `Net`.
#[path = "../../beatlm-bench/src/slots.rs"]
pub mod slots;
// The bench's percentiles and median, by path, so the report and the
// calibration quote the same statistics.
#[path = "../../beatlm-bench/src/stats.rs"]
mod stats;
// The bench's synthetic load, by path; the file says why it is shared.
#[path = "../../beatlm-bench/src/workload.rs"]
pub mod workload;

#[cfg(target_arch = "wasm32")]
mod web;

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use beatlm_core::device::DeviceFeatures;

    /// The display's shader is compiled only by the browser that runs the
    /// page, so it is validated here, by naga with every check on and no
    /// optional feature, as the generated kernels are.
    #[test]
    fn the_display_shader_validates_without_optional_features() {
        beatlm::codegen::validate(
            "display",
            include_str!("web/display.wgsl"),
            DeviceFeatures::default(),
        )
        .unwrap();
    }
}
