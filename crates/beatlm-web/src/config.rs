//! What a run is configured with, read from the page's query string.
//!
//! The query is parsed by hand for the same reason the bench parses its
//! command line by hand: a handful of numbers. Parsing the text, rather than
//! asking the browser's `URLSearchParams`, keeps it testable without a
//! browser. Anything not understood is refused, so a mistyped parameter
//! cannot leave a run on its default without saying so.
//!
//! # Parameters
//!
//! - `frames`: frame indices the run lasts, at least one (default
//!   [`DEFAULT_FRAMES`]).
//! - `model`: the job, `synthetic` (the default; the bench's busy-loop slices)
//!   or `depth` (Depth Anything V2 Small, [`crate::depth`]); or `pass-cost`,
//!   no job at all but the bench's measurement of what a compute pass costs
//!   ([`crate::pass_cost`]), with parameters of its own (below).
//! - `side`: the depth model's input side in pixels, a positive multiple of 14
//!   up to 518 (default [`NATIVE_DEFAULT_SIDE`]).
//!   Only with `model=depth`, or `model=pass-cost&kernels=depth` (default
//!   [`crate::pass_cost::DEFAULT_DEPTH_SIDE`] there).
//! - `latency`: the promised latency in frames, which the display's period
//!   turns into time.
//! - `latency_ms`: the promised latency in milliseconds, a decimal; the page
//!   measures the display's period before the run and promises the frames
//!   that take that long ([`crate::depth::frames_for`] has the rule). Not with
//!   `latency`. Without either, the synthetic workload promises
//!   [`DEFAULT_LATENCY`] frames and the depth model the native default's
//!   latency in time ([`NATIVE_DEFAULT_LATENCY`]).
//! - `arena_bytes`: plan one arena of that many bytes, create the device with
//!   the limits it needs and realise it before the run, to check the limit
//!   path. Only with the synthetic workload: the depth model's own plan sets
//!   the device's limits.
//! - `estimator`: the statistic the cost model estimates a slice with, the
//!   core's [`Estimator`], in the forms the next section lists. Default: the
//!   core's ([`Estimator::default`]).
//! - `render_iters`: loop iterations per pixel of the render pass, a whole
//!   number from 0 up (the bench's `--render-iters`, whose spelling and range
//!   it has; default [`DEFAULT_RENDER_ITERS`], 50). Both jobs, fixed latency
//!   and on beat; refused with `model=pass-cost` (below). The next section
//!   says why it exists.
//! - `render_order`: `after` (the default, [`DEFAULT_RENDER_ORDER`]) or
//!   `free`, the bench's `--render-order`: whether the loop's render pass
//!   reads what the frame's slices write, so that it runs after them. It
//!   came from H7 of #53 (#61): in Chrome the render ran beside the slices
//!   and slowed them, and ordering it after them took that away (the crate
//!   documentation, "What differs from the native bench, and why", has the
//!   measurements and why the bench keeps `free`); `free` is the frame as it
//!   was before, kept to compare against. The bench's `workload.rs`
//!   ("Ordering the render pass after the slices") says how and what it
//!   guarantees. With the depth model the display is the pass that reads
//!   them, every arena of the plan, and a plan of more arenas than the
//!   render can read is refused before the weights are fetched, naming
//!   `render_order=free`. Both jobs; refused with `model=pass-cost`, whose
//!   render is no loop's. The `config:` line prints it (`render_order=`).
//! - `sync`: the delivery promise, `latency` (the default: fixed latency, the
//!   promise every parameter above is about) or `onbeat` (results on the
//!   first frame at or after a beat of a click track the page plays,
//!   [`crate::onbeat`]). With `onbeat`, `latency` and `latency_ms` are
//!   refused, since the promise has no latency in frames, and these apply,
//!   each only with it:
//!   - `bpm`: the tempo, a decimal from [`MIN_BPM`] to [`MAX_BPM`] (default
//!     [`DEFAULT_BPM`]).
//!   - `bpm_change`: `<bpm>@<seconds>`, a tempo in the same range from the
//!     first beat at or after `seconds` after the first beat, a decimal above
//!     zero up to an hour; the scheduler is told at the instant `seconds`
//!     names (the [`crate::onbeat`] documentation, "A tempo change").
//!   - `margin_ms`: the margin the core places a job's beat with, a decimal
//!     from 0 to 1000 milliseconds (default 100, [`crate::onbeat`] says why).
//!   - `beats`: run for this many beats rather than for `frames` (not with
//!     it): inputs at the first `beats` beats, the run lasting to the beat
//!     the last is placed on and a little after
//!     ([`crate::onbeat::run_frames`]), 1 to [`MAX_BEATS`].
//!
//! `model=pass-cost` schedules nothing, so `latency`, `latency_ms`,
//! `arena_bytes`, `estimator` and `sync` are refused with it, and so is
//! `render_iters`: that measurement's render pass is `render=on|off` at the
//! default [`DEFAULT_RENDER_ITERS`], the figure its published cells were
//! measured at, whereas the bench's `pass-cost` does take `--render-iters`. A
//! page that took it there would put a second spelling of the render beside
//! `render`, so it is refused rather than half-applied. `frames` is the timed
//! frames of each cell. These apply, each only with it, and spell their
//! values as the bench's flags of the same names do:
//!
//! - `passes`: the dispatches per frame, a comma-separated list (default
//!   `1,8,32,128`).
//! - `variants`: `per-pass`, `ends`, `one-pass` and `every:<k>`,
//!   comma-separated (default all four, `k` 8).
//! - `kernels`: what each dispatch runs, `trivial` (the default) or `depth`,
//!   the depth model's first kernels at `side`, whose weights the page then
//!   fetches as for `model=depth`.
//! - `groups`: workgroups per trivial dispatch (default 1); not with
//!   `kernels=depth`.
//! - `render`: `on` to end each frame with the synthetic render pass, as the
//!   loop's frames do; `off` (the default).
//!
//! # The render_iters parameter
//!
//! The render pass of a loop run repeats the synthetic per-pixel loop
//! (`render_fs` in the bench's `shaders.wgsl`; the depth model's display,
//! `web/display.wgsl`, carries the same loop before it draws the image and the
//! depth), `render_iters` times per pixel; the budget's prior for the render
//! at the default of 50 is the bench's [`crate::render_cost::RENDER_PRIOR`].
//! It exists to test H2 of #53: that the render pass in the same frame
//! contends with the depth slices on the GPU, so slices measure slower in the
//! loop than in calibration (`tail_loop_over_calibration`). Run at a minimal
//! render (`render_iters=1`, or 0) and compare: if the in-loop over
//! calibration ratio falls and delivery rises, the render pass contends; if the
//! ratio stays, it does not.
//!
//! **0 is accepted.** The shader's loop is `for i in 0..render_iters` over a
//! uniform, so zero trips is well defined; the loop's result still feeds the
//! colour (`a` keeps its starting value, a function of the pixel position), so
//! nothing else in the fragment shader changes: the display still draws the
//! source and the depth once per pixel, and the synthetic render still fills
//! the target. No other minimum applies; the range is the bench's, any `u32`.
//!
//! **What it changes.** It is the render pass's cost, and the render pass's
//! cost sets the frame budget the slices share: each frame's budget is the
//! display's period less the render the forecast expects and the margin
//! ([`crate::render_cost`]). A cheaper render leaves more of the frame to
//! slices, so a run at `render_iters=1` is not the native default's budget (the
//! default's is the one it was measured under) but a larger one, and its
//! delivery is not comparable with a default run's on that account alone; read
//! the `gpu_render_alone_us` and `gpu_render_with_slices_us` lines beside it.
//! The forecast follows the render only as it is measured: until the first
//! measurement it is [`crate::render_cost::RENDER_PRIOR`], the default's, at
//! any value, and without `timestamp_query` the render is never measured, so
//! the budget stays at the default's and the value changes the render's cost
//! but not the slices' share of the frame.
//! The value applies to the whole run: calibration's frames end with the same
//! synthetic render pass, so the same value reaches them, and the `config:`
//! line prints the value used (`render_iters=`).
//!
//! # The estimator parameter
//!
//! The value is a name and, after colons, its parameters. The bare names are
//! the bench's `--estimator` spellings (`quantile`, `mean-sigmas`, `capped`,
//! `frame-scaled`), each meaning exactly the bench's parameters. The report's `config:` line
//! prints (`estimator=`, by [`estimator_name`]) the bare name when the
//! estimator is exactly what it means, as the bench's does, so the default
//! prints `estimator=quantile`, and the parameterised form otherwise; either
//! can be pasted back into a URL.
//!
//! | Value | The core's estimator |
//! |---|---|
//! | `quantile` | `WindowedQuantile`, quantile 0.9, window [`DEFAULT_WINDOW`] (32): the core's default |
//! | `quantile:<q>` |`WindowedQuantile`, window [`DEFAULT_WINDOW`] (32), quantile `q`, a number in `(0, 1]` |
//! | `quantile:<q>:<window>` | the same with a window of `window` measurements, `1..=`[`MAX_WINDOW`] |
//! | `mean-sigmas` | `MeanPlusSigmas`, `alpha` [`COST_ALPHA`] (0.2), `sigmas` [`COST_SIGMAS`] (2) |
//! | `mean-sigmas:<alpha>:<sigmas>` | the same with `alpha` in `(0, 1]` and `sigmas` not negative |
//! | `capped` | `CappedDeviation`, `alpha` 0.2, `sigmas` 2, `cap` [`COST_CAP`] (0.5) |
//! | `capped:<alpha>:<sigmas>:<cap>` | the same with `cap` not negative |
//! | `frame-scaled` | `FrameScaled`, `window` [`DEFAULT_WINDOW`] (32), `frame_window` [`DEFAULT_FRAME_WINDOW`] (64), `frame_quantile` [`DEFAULT_FRAME_QUANTILE`] (0.9) |
//! | `frame-scaled:<q>` | the same with `frame_quantile` `q`, a number in `(0, 1]` |
//! | `frame-scaled:<q>:<frame_window>` | the same with a frame window of `frame_window` frames, `1..=`[`MAX_WINDOW`] |
//! | `frame-scaled:<q>:<frame_window>:<window>` | the same with a slice window of `window` measurements, `1..=`[`MAX_WINDOW`] |
//!
//! The parameters of the bare names are the ones the core's choice of
//! estimator was measured with (`beatlm_core::cost`, "Choosing the
//! estimator"), which are also the bench's; `frame-scaled`'s are the ones
//! `beatlm_core::cost` ("Planning a frame against its total") gives reasons
//! for. The old spelling `mean` is refused. A quantile of 0
//! (the window's minimum) is refused here although the core accepts it, since
//! it plans every slice for its fastest measurement, and a frame quantile of
//! 0 for the same reason. The remaining ranges are the core's, and its error
//! is quoted when one is out of range.
//!
//! It applies to both jobs: the cost model the loop plans with is built with
//! it. For the depth model that model is the one calibration measured the
//! split parts into, so calibration builds it with the estimator, as the
//! bench's does with `--estimator`; calibration's decision, how each kernel is
//! split, reads the median of its whole-kernel measurements and never the cost
//! model, so it is the same whichever estimator is chosen
//! ([`crate::depth::WholeRuns::durations`]). The render pass's own cost model
//! ([`crate::render_cost`]) is not an estimator choice and does not change.
//!
//! # Which estimator, measured in a browser
//!
//! The core chose its default on the native bench at 60 Hz ("Choosing the
//! estimator" in `beatlm_core::cost`). This is the same comparison in a
//! browser at 120 Hz, where the default does not keep the promise
//! ([`crate`], "What a browser run measured").
//!
//! Chrome 154 (WebGPU) on macOS, Apple M1 Pro, 120 Hz display (period
//! 8 300 µs), 2026-10-22, Depth Anything V2 Small at 266 pixels,
//! `?model=depth&side=266&frames=1500&latency_ms=200&estimator=<e>` (the URLs
//! said `quantile:0.5`, `quantile:0.75`, `quantile:0.9` and `capped`, and the
//! `config:` line of that build printed them in full; the table writes each
//! estimator as the `config:` line prints it now, the default as the bare
//! `quantile`, the same estimators): 25 frames,
//! about 207 ms (24 frames for `quantile:0.75:32`, whose run measured the
//! period as 8 400 µs at its start), free admission, margin 2 ms, render 50
//! iterations, one run per estimator, after #41 (a lone late job is kept). No
//! run was discarded.
//!
//! | Estimator | Submitted | Delivered | Missed | Rate | GPU frames over the period | `gpu_frame_us` p50 / p99 |
//! |---|---|---|---|---|---|---|
//! | `quantile:0.5:32` | 119 | 114 | 4 | 0.9661 | 408 of 1477 | 7 343 / 11 862 |
//! | `quantile:0.75:32` | 76 | 31 | 44 | 0.4133 | 165 of 1497 | 6 423 / 10 356 |
//! | `quantile` (default; 0.9, window 32) | 68 | 19 | 48 | 0.2836 | 57 of 1496 | 4 917 / 9 701 |
//! | `capped` (0.2, 2, 0.5) | 61| 7 | 53 | 0.1167 | 26 of 1486 | 4 523 / 8 848 |
//!
//! From the `quantile:0.5:32` run, the lines [`crate::diagnosis`] defines,
//! with the fields not quoted here elided as `…`:
//!
//! ```text
//! capacity_frames_needed: in_loop_run_us=68281 slice_budget_us=4631 frames=15 frames_promised=25 frames_available=26 … verdict=not_ruled_out
//! tail_loop_over_calibration: … median_mean_ratio=2.05 median_p50_ratio=1.57
//! tail_outlier_frames: … at_least_half_outliers=175 expected_if_independent=26.7
//! ```
//!
//! From an earlier 3000-frame run at the native default's 83 ms (10 frames)
//! with the default estimator: `tail_loop_over_calibration
//! median_mean_ratio=1.78 median_p50_ratio=1.50`; outlier frames 185 against
//! 98.5 expected if independent; first passes were 0.233 of the outliers
//! against 0.246 of the samples (H1, that first passes are the slow ones, is
//! refuted); the share of measurements that were outliers rose with the
//! previous frame's busyness, from 0.054 (0 to 25%) to 0.200 (75% and over).
//!
//! What the numbers say, and what they do not (one run per cell throughout):
//!
//! - **No estimator here keeps both promises.** The lighter estimates deliver
//!   (`quantile:0.5:32`: 0.97) by packing more into a frame than the frame
//!   holds: the GPU time of 28% of its frames exceeded the 8.3 ms period,
//!   which breaks the frame budget the renderer shares, the engine's first
//!   promise. The heavier estimates keep frames within the period (`capped`
//!   1.7%, `quantile:0.9:32` 3.8%) and deliver little.
//! - **Why, as far as these runs show.** The per-slice estimates are summed
//!   for a frame, but slow measurements cluster by frame (the frames with
//!   half or more of their measurements slow were 1.9 and 6.6 times as many
//!   as independent measurements would give, in the two runs above), so the
//!   sum of per-slice quantiles is too high on average and still not a bound
//!   on the slow frames. The clustering is measured; that it is the whole cause is not.
//! - **The default stays `quantile:0.9:32`.** The frame budget is the promise
//!   that must hold. Delivery at 120 Hz was left to an estimate made for the
//!   frame rather than the slice, `frame-scaled` (#45); repeated with it,
//!   below, this comparison did not make it the default.
//! - **At 83 ms (10 frames) the native default is infeasible on this display
//!   whatever the estimator.** The `quantile:0.5:32` run measured a run of the
//!   model in the loop of 68 ms against about 4.6 ms of slice budget per frame:
//!   15 frames needed against the 11 that 83 ms gives. That compares one
//!   run's figures with the 11 frames; it is not a run at 83 ms.
//! - **`capped` delivering least, though its estimates are closest to the
//!   means, is one run and unexplained.** Do not rank `capped` against
//!   `quantile:0.9:32` on it.
//!
//! **With the frame scale (#45).** The same configuration, the same day,
//! each pair back to back, one run per cell, no run discarded. "First" is
//! the version of `frame-scaled` that let calibration's measurements into
//! the median it scales; "second" keeps them out
//! (`beatlm_core::cost`, "Planning a frame against its total", says how and
//! why). F is the report's `frame_scale` over the run.
//!
//! | Version | Estimator | Delivered | Missed | Rate | GPU frames over the period | F p50 / p90 (end) | `tail_outlier_frames` (half or more slow) | Loop over calibration, median / mean | Frames needed of 26 |
//! |---|---|---|---|---|---|---|---|---|---|
//! | first | `quantile` | 17 | 49 | 0.2576 | 65 of 1497 | | | | |
//! | first | `frame-scaled` | 8 | 54 | 0.1290 | 28 of 1498 | 2.80 / 6.00 (2.53) | 99 against 99.3 | 1.98 / 2.43 | 23 |
//! | second | `quantile` | 16 | 51 | 0.2388 | 64 of 1494 | | | | |
//! | second | `frame-scaled` | 11 | 52 | 0.1746 | 25 of 1491 | 2.04 / 3.91 (1.58) | 53 against 85.0 | 1.83 / 2.27 | 24 |
//!
//! The diagnosis columns are from the `frame-scaled` runs. What they show,
//! and what they do not:
//!
//! - **It did not deliver more than the default in either run** (8 against
//!   17, 11 against 16).
//! - **It cut the GPU frames over the period to about 40% of the
//!   default's** (28 of 65, 25 of 64).
//! - **The clustering that motivated it was absent from both runs:** frames
//!   with half or more of their measurements slow were as many as
//!   independent measurements would give, or fewer, where the runs above had
//!   175 against 26.7. The loop instead ran 1.8 to 2.4 times calibration.
//! - **Capacity was near its limit,** 23 and 24 of the 26 frames a job had
//!   even with exact estimates, on this machine at that time, which caps any
//!   estimator's delivery.
//! - **Natively** (Apple M1 Pro, Metal, the bench's depth workload at latency
//!   5) one run of the second version delivered 116 of 116 against the
//!   default's 101 of 113; one run each ranks nothing, and that machine was
//!   slower that day than when #41 was measured (`main` delivered 97 of 101,
//!   calibration at 53.6 ms). `beatlm_core::cost` has the table.
//!
//! `frame-scaled` stays opt-in, and the default stays `quantile:0.9:32`.
//!
//! **Why a latency in time.** A promise in frames is worth the display's
//! period, so the native default of 5 frames, a decision made at 60 Hz for
//! 83.3 ms and 12 results a second, is 41.7 ms on a 120 Hz display, a
//! different promise with half the frames for a job. The decision was about
//! time, so the page can be asked for it in time, and the depth model is
//! asked for it unless told otherwise; `latency` stays, in frames, for a run
//! that means frames.

use anyhow::{Context, Result, anyhow, bail};
use beatlm_core::cost::{CostModel, Estimator, MAX_WINDOW};
use beatlm_core::time::Micros;

use crate::depth::{MAX_SIDE, NATIVE_DEFAULT_LATENCY, NATIVE_DEFAULT_SIDE, PATCH};
use crate::onbeat::{
    DEFAULT_BPM, DEFAULT_MARGIN, MAX_BEATS, MAX_BPM, MAX_CHANGE_AFTER_S, MAX_MARGIN, MIN_BPM,
    Tempo, TempoChange,
};
use crate::pass_cost::{self, DEFAULT_DEPTH_SIDE, Dispatches};
use crate::workload::{DEFAULT_RENDER_ITERS, RenderOrder};

/// Frames a run lasts when `frames` is not given: the bench's default.
pub const DEFAULT_FRAMES: u64 = 600;

/// The render order when `render_order` is not given: the render pass waits
/// for the frame's slices. The bench's default is `free`; the crate
/// documentation ("What differs from the native bench, and why") has the
/// measurements that decided each.
pub const DEFAULT_RENDER_ORDER: RenderOrder = RenderOrder::After;

/// The promised latency in frames of the synthetic workload when neither
/// `latency` nor `latency_ms` is given: the bench's default for it.
pub const DEFAULT_LATENCY: u32 = 2;

/// Slices in each frame's job of the synthetic workload: the bench's default
/// for it, which the page does not change.
pub const SLICES: u32 = 4;

/// The margin kept free in each frame's budget, in microseconds: the bench's
/// default, which the page does not change.
pub const MARGIN_US: u64 = 2_000;

/// The longest promised latency `latency_ms` accepts, in milliseconds: a
/// minute, far past any job here, so a typing slip is refused rather than
/// planned.
const MAX_LATENCY_MS: f64 = 60_000.0;

/// The window `estimator=quantile` and `quantile:<q>` read: the core's
/// default, and the bench's `COST_WINDOW`.
pub const DEFAULT_WINDOW: usize = 32;

/// The weight of a new measurement of `estimator=mean-sigmas` and `capped`:
/// the value the core's estimators were compared with, and the bench's.
pub const COST_ALPHA: f64 = 0.2;

/// The standard deviations `estimator=mean-sigmas` and `capped` add to the
/// mean.
pub const COST_SIGMAS: f64 = 2.0;

/// The most `estimator=capped` pads the mean by, as a multiple of it.
pub const COST_CAP: f64 = 0.5;

/// The frames `estimator=frame-scaled` reads its frame scale from: the
/// bench's `COST_FRAME_WINDOW`.
pub const DEFAULT_FRAME_WINDOW: usize = 64;

/// The quantile of the frame ratios `estimator=frame-scaled` reads: the
/// bench's, the default estimator's quantile.
pub const DEFAULT_FRAME_QUANTILE: f64 = 0.9;

/// The parameters the query string accepts, as the error for an unknown one
/// lists them.
const PARAMETERS: &str = "frames, model, side, latency, latency_ms, arena_bytes, estimator, \
                          sync, bpm, bpm_change, margin_ms, beats, render_iters, render_order, \
                          passes, variants, kernels, groups, render";

/// The parameters only `model=pass-cost` reads.
const PASS_COST_PARAMETERS: [&str; 5] = ["passes", "variants", "kernels", "groups", "render"];

/// The parameters of the loop, which `model=pass-cost` refuses.
const LOOP_PARAMETERS: [&str; 6] = [
    "latency",
    "latency_ms",
    "arena_bytes",
    "estimator",
    "sync",
    "render_order",
];

/// The parameters only `sync=onbeat` reads.
const ON_BEAT_PARAMETERS: [&str; 4] = ["bpm", "bpm_change", "margin_ms", "beats"];

/// The forms `estimator` accepts, as the error for another value lists them.
const ESTIMATOR_FORMS: &str = "quantile, quantile:<q>, quantile:<q>:<window>, mean-sigmas, \
                               mean-sigmas:<alpha>:<sigmas>, capped, capped:<alpha>:<sigmas>:<cap>, \
                               frame-scaled, frame-scaled:<q>, frame-scaled:<q>:<frame_window> or \
                               frame-scaled:<q>:<frame_window>:<window>";

/// The job each frame submits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Model {
    /// The bench's synthetic slices, `model=synthetic`.
    Synthetic,
    /// Depth Anything V2 Small at `side` pixels a side, `model=depth`.
    Depth {
        /// The input's side in pixels.
        side: u32,
    },
    /// No job: the bench's measurement of a compute pass's cost,
    /// `model=pass-cost`, with what it measures.
    PassCost(pass_cost::Settings),
}

/// The promise of the fixed-latency policy, as the query gave it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Latency {
    /// A whole number of frames, `latency`.
    Frames(u32),
    /// A time, `latency_ms`, turned into frames at the display's period
    /// measured before the run.
    Time(Micros),
}

/// A run's configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Frames to run, `frames`; at least one.
    pub frames: u64,
    /// The job, `model` and `side`.
    pub model: Model,
    /// The promised latency, `latency` or `latency_ms`.
    pub latency: Latency,
    /// `arena_bytes`: before the run, plan one arena of this many bytes
    /// against the adapter's limits, create the device with the limits it
    /// needs and realise it there, to check the limit path; `None` for a run
    /// that needs only the default limits.
    pub arena_bytes: Option<u64>,
    /// `estimator`: the statistic the loop's cost model estimates a slice
    /// with.
    pub estimator: Estimator,
    /// `render_iters`: loop iterations per pixel of the render pass of a loop
    /// run, calibration's included; [`DEFAULT_RENDER_ITERS`] when not given.
    /// Zero is allowed (the module documentation says why).
    pub render_iters: u32,
    /// `render_order`: whether the loop's render pass waits for the frame's
    /// slices; [`DEFAULT_RENDER_ORDER`] when not given.
    pub render_order: RenderOrder,
    /// `sync=onbeat` and its parameters; `None` for the fixed-latency promise,
    /// whose `latency` is then the one kept.
    pub on_beat: Option<OnBeat>,
}

/// The on-beat promise, `sync=onbeat`, as the query gave it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OnBeat {
    /// `bpm` and `bpm_change`.
    pub tempo: Tempo,
    /// `margin_ms`.
    pub margin: Micros,
    /// `beats`: the run's length in beats, in place of `frames`.
    pub beats: Option<u64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            frames: DEFAULT_FRAMES,
            model: Model::Synthetic,
            latency: Latency::Frames(DEFAULT_LATENCY),
            arena_bytes: None,
            estimator: Estimator::default(),
            render_iters: DEFAULT_RENDER_ITERS,
            render_order: DEFAULT_RENDER_ORDER,
            on_beat: None,
        }
    }
}

/// Parses a query string such as `?frames=600&latency=2` or
/// `?model=depth&side=266&latency_ms=83.3`, with or without the leading `?`.
/// An empty query is the default configuration.
///
/// # Errors
///
/// An unknown or repeated parameter, one without a value, a value that is not
/// a number or not a model, zero frames or an arena of zero bytes, a side that
/// is not a positive multiple of 14 up to 518, `side` without `model=depth`,
/// `arena_bytes` with it, both `latency` and `latency_ms`, or an `estimator`
/// that is not one of the forms [`estimator`] reads or has a parameter out of
/// range; a `render_iters` that is not a whole number or is given with
/// `model=pass-cost`; a `sync` that is not `latency` or `onbeat`, `bpm`, `bpm_change`,
/// `margin_ms` or `beats` without `sync=onbeat` or out of range, and
/// `latency`, `latency_ms`, or `frames` beside `beats`, with it.
pub fn parse(query: &str) -> Result<Config> {
    let mut frames = DEFAULT_FRAMES;
    let mut depth = false;
    let mut measure_passes = false;
    let mut settings = pass_cost::Settings::default();
    let mut kernels_depth = false;
    let mut groups = 1;
    let mut side = None;
    let mut latency = None;
    let mut latency_ms = None;
    let mut arena_bytes = None;
    let mut chosen = None;
    let mut render_iters = DEFAULT_RENDER_ITERS;
    let mut render_order = DEFAULT_RENDER_ORDER;
    let mut on_beat = false;
    let mut bpm = None;
    let mut change = None;
    let mut margin = None;
    let mut beats = None;
    let mut seen: Vec<&str> = Vec::new();
    let query = query.strip_prefix('?').unwrap_or(query);
    for pair in query.split('&').filter(|pair| !pair.is_empty()) {
        let (name, value) = pair
            .split_once('=')
            .with_context(|| format!("query parameter {name:?} needs a value", name = pair))?;
        if seen.contains(&name) {
            bail!("query parameter {name:?} is given twice");
        }
        seen.push(name);
        match name {
            "frames" => frames = number(name, value)?,
            "model" => {
                (depth, measure_passes) = match value {
                    "synthetic" => (false, false),
                    "depth" => (true, false),
                    "pass-cost" => (false, true),
                    _ => bail!(
                        "query parameter model: {value:?} is not synthetic, depth or pass-cost"
                    ),
                };
            }
            "passes" => {
                settings.passes =
                    pass_cost::parse_passes(value).context("query parameter passes")?;
            }
            "variants" => {
                settings.variants =
                    pass_cost::parse_variants(value).context("query parameter variants")?;
            }
            "kernels" => {
                kernels_depth = match value {
                    "trivial" => false,
                    "depth" => true,
                    _ => bail!("query parameter kernels: {value:?} is not trivial or depth"),
                };
            }
            "groups" => {
                groups = pass_cost::parse_groups(value).context("query parameter groups")?;
            }
            "render" => {
                settings.render = match value {
                    "on" => true,
                    "off" => false,
                    _ => bail!("query parameter render: {value:?} is not on or off"),
                };
            }
            "side" => side = Some(number::<u32>(name, value)?),
            "latency" => latency = Some(number(name, value)?),
            "latency_ms" => latency_ms = Some(milliseconds(value)?),
            "arena_bytes" => arena_bytes = Some(number(name, value)?),
            "estimator" => chosen = Some(estimator(value)?),
            "render_iters" => render_iters = number(name, value)?,
            "render_order" => {
                render_order = RenderOrder::from_name(value).with_context(|| {
                    format!("query parameter render_order: {value:?} is not one of free, after")
                })?;
            }
            "sync" => {
                on_beat = match value {
                    "latency" => false,
                    "onbeat" => true,
                    _ => bail!("query parameter sync: {value:?} is not latency or onbeat"),
                };
            }
            "bpm" => bpm = Some(tempo(name, value)?),
            "bpm_change" => change = Some(tempo_change(value)?),
            "margin_ms" => margin = Some(margin_ms(value)?),
            "beats" => beats = Some(number::<u64>(name, value)?),
            _ => bail!("unknown query parameter {name:?}; the page accepts {PARAMETERS}"),
        }
    }
    if frames == 0 {
        bail!("frames must be at least 1");
    }
    if measure_passes {
        // Nothing is scheduled, so the loop's parameters would be silently
        // ignored; they are refused instead.
        if let Some(name) = LOOP_PARAMETERS.iter().find(|name| seen.contains(name)) {
            bail!(
                "query parameter {name} has no effect with model=pass-cost, which schedules nothing"
            );
        }
        if seen.contains(&"render_iters") {
            bail!(
                "query parameter render_iters has no effect with model=pass-cost, whose render \
                 pass is render=on or render=off at the default of {DEFAULT_RENDER_ITERS} iterations"
            );
        }
        if let Some(name) = ON_BEAT_PARAMETERS.iter().find(|name| seen.contains(name)) {
            bail!("query parameter {name} belongs to on-beat delivery; it needs sync=onbeat");
        }
        settings.dispatches = if kernels_depth {
            if seen.contains(&"groups") {
                bail!("groups is the trivial kernel's; it has no effect with kernels=depth");
            }
            Dispatches::Depth {
                side: depth_side(side, DEFAULT_DEPTH_SIDE)?,
            }
        } else {
            if side.is_some() {
                bail!("side is the depth model's input side; it needs kernels=depth here");
            }
            Dispatches::Trivial { groups }
        };
        return Ok(Config {
            frames,
            model: Model::PassCost(settings),
            ..Config::default()
        });
    }
    if let Some(name) = PASS_COST_PARAMETERS.iter().find(|name| seen.contains(name)) {
        bail!(
            "query parameter {name} belongs to the pass-cost measurement; it needs model=pass-cost"
        );
    }
    let on_beat = if on_beat {
        if latency.is_some() || latency_ms.is_some() {
            bail!(
                "sync=onbeat delivers on beats and has no latency in frames; \
                 give bpm (and bpm_change) instead of latency or latency_ms"
            );
        }
        if beats.is_some() && seen.contains(&"frames") {
            bail!("give the run's length in frames or in beats, not both");
        }
        if beats.is_some_and(|beats| !(1..=MAX_BEATS).contains(&beats)) {
            bail!("beats must be from 1 to {MAX_BEATS}");
        }
        Some(OnBeat {
            tempo: Tempo {
                bpm: bpm.unwrap_or(DEFAULT_BPM),
                change,
            },
            margin: margin.unwrap_or(DEFAULT_MARGIN),
            beats,
        })
    } else {
        if let Some(name) = ON_BEAT_PARAMETERS.iter().find(|name| seen.contains(name)) {
            bail!("query parameter {name} belongs to on-beat delivery; it needs sync=onbeat");
        }
        None
    };
    if arena_bytes == Some(0) {
        bail!("arena_bytes must be at least 1");
    }
    let model = if depth {
        if arena_bytes.is_some() {
            bail!(
                "arena_bytes checks the synthetic workload's limits; the depth model's plan sets its own"
            );
        }
        Model::Depth {
            side: depth_side(side, NATIVE_DEFAULT_SIDE)?,
        }
    } else {
        if side.is_some() {
            bail!("side is the depth model's input side; it needs model=depth");
        }
        Model::Synthetic
    };
    let latency = match (latency, latency_ms) {
        (Some(_), Some(_)) => bail!("give latency in frames or latency_ms in time, not both"),
        (Some(frames), None) => Latency::Frames(frames),
        (None, Some(time)) => Latency::Time(time),
        (None, None) => match model {
            Model::Synthetic | Model::PassCost(_) => Latency::Frames(DEFAULT_LATENCY),
            Model::Depth { .. } => Latency::Time(NATIVE_DEFAULT_LATENCY),
        },
    };
    Ok(Config {
        frames,
        model,
        latency,
        arena_bytes,
        estimator: chosen.unwrap_or_default(),
        render_iters,
        render_order,
        on_beat,
    })
}

/// The depth model's input side, `side` or else `default`: a positive
/// multiple of [`PATCH`] up to [`MAX_SIDE`].
fn depth_side(side: Option<u32>, default: u32) -> Result<u32> {
    let side = side.unwrap_or(default);
    if side == 0 || !side.is_multiple_of(PATCH) || side > MAX_SIDE {
        bail!("side must be a positive multiple of {PATCH} up to {MAX_SIDE}, not {side}");
    }
    Ok(side)
}

/// Parses a tempo, the value of `bpm` or the tempo of `bpm_change`: a decimal
/// number of beats per minute from [`MIN_BPM`] to [`MAX_BPM`].
fn tempo(name: &str, value: &str) -> Result<f64> {
    let bpm: f64 = number(name, value)?;
    // The range check is false for NaN.
    if !(MIN_BPM..=MAX_BPM).contains(&bpm) {
        bail!("query parameter {name}: {value:?} is not a tempo from {MIN_BPM} to {MAX_BPM} BPM");
    }
    Ok(bpm)
}

/// Parses `bpm_change`, `<bpm>@<seconds>`.
fn tempo_change(value: &str) -> Result<TempoChange> {
    let (bpm, after) = value
        .split_once('@')
        .with_context(|| format!("query parameter bpm_change: {value:?} is not <bpm>@<seconds>"))?;
    let bpm = tempo("bpm_change", bpm)?;
    let after_s: f64 = number("bpm_change", after)?;
    if !(after_s > 0.0 && after_s <= MAX_CHANGE_AFTER_S) {
        bail!(
            "query parameter bpm_change: the change must come after the first beat, \
             at most {MAX_CHANGE_AFTER_S} seconds, not {after:?}"
        );
    }
    Ok(TempoChange { bpm, after_s })
}

/// Parses `margin_ms`, a decimal number of milliseconds up to [`MAX_MARGIN`],
/// into whole microseconds.
fn margin_ms(value: &str) -> Result<Micros> {
    let ms: f64 = number("margin_ms", value)?;
    #[allow(clippy::cast_precision_loss)]
    let max_ms = MAX_MARGIN.get() as f64 / 1_000.0;
    if !(0.0..=max_ms).contains(&ms) {
        bail!("query parameter margin_ms: {value:?} is not between 0 and {max_ms}");
    }
    // At most a second in microseconds.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let us = (ms * 1_000.0).round() as u64;
    Ok(Micros::new(us))
}

/// Parses the value of `estimator`: a name and its parameters after colons
/// (the module documentation has the table).
///
/// # Errors
///
/// A name or number of parameters that is none of the forms, a parameter that
/// is not a number, a quantile outside `(0, 1]`, a window outside
/// `1..=`[`MAX_WINDOW`], or an `alpha`, `sigmas` or `cap` the core refuses.
pub fn estimator(value: &str) -> Result<Estimator> {
    let mut parts = value.split(':');
    let kind = parts.next().unwrap_or_default();
    let args: Vec<&str> = parts.collect();
    let float = |text: &str| number::<f64>("estimator", text);
    let chosen = match (kind, args.as_slice()) {
        ("quantile", []) => bare_quantile(),
        ("mean-sigmas", []) => bare_mean_sigmas(),
        ("capped", []) => bare_capped(),
        ("quantile", [q]) => Estimator::WindowedQuantile {
            window: DEFAULT_WINDOW,
            quantile: float(q)?,
        },
        ("quantile", [q, window]) => Estimator::WindowedQuantile {
            window: number("estimator", window)?,
            quantile: float(q)?,
        },
        ("mean-sigmas", [alpha, sigmas]) => Estimator::MeanPlusSigmas {
            alpha: float(alpha)?,
            sigmas: float(sigmas)?,
        },
        ("capped", [alpha, sigmas, cap]) => Estimator::CappedDeviation {
            alpha: float(alpha)?,
            sigmas: float(sigmas)?,
            cap: float(cap)?,
        },
        ("frame-scaled", []) => bare_frame_scaled(),
        ("frame-scaled", [q]) => Estimator::FrameScaled {
            frame_quantile: float(q)?,
            frame_window: DEFAULT_FRAME_WINDOW,
            window: DEFAULT_WINDOW,
        },
        ("frame-scaled", [q, frame_window]) => Estimator::FrameScaled {
            frame_quantile: float(q)?,
            frame_window: number("estimator", frame_window)?,
            window: DEFAULT_WINDOW,
        },
        ("frame-scaled", [q, frame_window, window]) => Estimator::FrameScaled {
            frame_quantile: float(q)?,
            frame_window: number("estimator", frame_window)?,
            window: number("estimator", window)?,
        },
        _ => bail!("query parameter estimator: {value:?} is not {ESTIMATOR_FORMS}"),
    };
    // `q > 0.0` is false for NaN, and `<= 1.0` for infinity.
    let in_unit_interval = |name: &str, q: f64| -> Result<()> {
        if q > 0.0 && q <= 1.0 {
            Ok(())
        } else {
            bail!("query parameter estimator: the {name} {q} is not in (0, 1]")
        }
    };
    match chosen {
        Estimator::WindowedQuantile { window, quantile } => {
            in_unit_interval("quantile", quantile)?;
            if !(1..=MAX_WINDOW).contains(&window) {
                bail!("query parameter estimator: the window {window} is not in 1..={MAX_WINDOW}");
            }
        }
        Estimator::FrameScaled { frame_quantile, .. } => {
            in_unit_interval("frame quantile", frame_quantile)?;
        }
        Estimator::MeanPlusSigmas { .. } | Estimator::CappedDeviation { .. } => {}
    }
    // The core owns the ranges of the other parameters; building a model is
    // how its check is asked.
    CostModel::new(Micros::ZERO, chosen)
        .map_err(|error| anyhow!("query parameter estimator: {value:?}: {error}"))?;
    Ok(chosen)
}

/// The estimator the bare `quantile` names: the core's default, which is the
/// bench's `--estimator quantile` (quantile 0.9, window [`DEFAULT_WINDOW`]).
const fn bare_quantile() -> Estimator {
    Estimator::WindowedQuantile {
        window: DEFAULT_WINDOW,
        quantile: 0.9,
    }
}

/// The estimator the bare `mean-sigmas` names, the bench's.
const fn bare_mean_sigmas() -> Estimator {
    Estimator::MeanPlusSigmas {
        alpha: COST_ALPHA,
        sigmas: COST_SIGMAS,
    }
}

/// The estimator the bare `capped` names, the bench's.
const fn bare_capped() -> Estimator {
    Estimator::CappedDeviation {
        alpha: COST_ALPHA,
        sigmas: COST_SIGMAS,
        cap: COST_CAP,
    }
}

/// The estimator the bare `frame-scaled` names, the bench's.
const fn bare_frame_scaled() -> Estimator {
    Estimator::FrameScaled {
        window: DEFAULT_WINDOW,
        frame_window: DEFAULT_FRAME_WINDOW,
        frame_quantile: DEFAULT_FRAME_QUANTILE,
    }
}

/// `estimator` in a form [`estimator`] reads back: the bench's bare name when
/// the estimator is exactly what that name means, else the name with every
/// parameter written. What the report's `config:` line prints.
#[must_use]
pub fn estimator_name(estimator: Estimator) -> String {
    if estimator == bare_quantile() {
        return "quantile".to_owned();
    }
    if estimator == bare_mean_sigmas() {
        return "mean-sigmas".to_owned();
    }
    if estimator == bare_capped() {
        return "capped".to_owned();
    }
    if estimator == bare_frame_scaled() {
        return "frame-scaled".to_owned();
    }
    match estimator {
        Estimator::MeanPlusSigmas { alpha, sigmas } => format!("mean-sigmas:{alpha}:{sigmas}"),
        Estimator::WindowedQuantile { window, quantile } => format!("quantile:{quantile}:{window}"),
        Estimator::CappedDeviation { alpha, sigmas, cap } => {
            format!("capped:{alpha}:{sigmas}:{cap}")
        }
        Estimator::FrameScaled {
            window,
            frame_window,
            frame_quantile,
        } => format!("frame-scaled:{frame_quantile}:{frame_window}:{window}"),
    }
}

/// Parses the value of parameter `name` as a number.
fn number<T: std::str::FromStr>(name: &str, value: &str) -> Result<T> {
    value
        .parse()
        .ok()
        .with_context(|| format!("query parameter {name}: {value:?} is not a number in range"))
}

/// Parses `latency_ms`, a decimal number of milliseconds, into whole
/// microseconds.
fn milliseconds(value: &str) -> Result<Micros> {
    let ms: f64 = number("latency_ms", value)?;
    if !(0.0..=MAX_LATENCY_MS).contains(&ms) {
        bail!("query parameter latency_ms: {value:?} is not between 0 and {MAX_LATENCY_MS}");
    }
    // At most a minute in microseconds, well inside a u64; rounding keeps
    // `83.3` at 83 300 rather than one below it.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let us = (ms * 1_000.0).round() as u64;
    Ok(Micros::new(us))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_query_is_the_benchs_synthetic_defaults() {
        for query in ["", "?"] {
            assert_eq!(parse(query).unwrap(), Config::default());
        }
        assert_eq!(Config::default().frames, 600);
        assert_eq!(Config::default().latency, Latency::Frames(2));
        assert_eq!(Config::default().model, Model::Synthetic);
    }

    #[test]
    fn the_pass_cost_measurement_reads_its_own_parameters_and_refuses_the_loops() {
        let settings = |query: &str| match parse(query).unwrap().model {
            Model::PassCost(settings) => settings,
            other => panic!("expected pass-cost, got {other:?}"),
        };
        assert_eq!(settings("?model=pass-cost"), pass_cost::Settings::default());
        let config = parse("?model=pass-cost&frames=300").unwrap();
        assert_eq!(config.frames, 300);
        let s =
            settings("?model=pass-cost&passes=1,64&variants=one-pass,every:16&groups=8&render=on");
        assert_eq!(s.passes, vec![1, 64]);
        assert_eq!(
            s.variants,
            vec![pass_cost::Variant::OnePass, pass_cost::Variant::Every(16)]
        );
        assert_eq!(s.dispatches, Dispatches::Trivial { groups: 8 });
        assert!(s.render);
        assert_eq!(
            settings("?model=pass-cost&kernels=depth").dispatches,
            Dispatches::Depth {
                side: DEFAULT_DEPTH_SIDE
            }
        );
        assert_eq!(
            settings("?kernels=depth&side=154&model=pass-cost").dispatches,
            Dispatches::Depth { side: 154 }
        );
        for bad in [
            "?model=pass-cost&passes=0",
            "?model=pass-cost&passes=1,1",
            "?model=pass-cost&variants=every:0",
            "?model=pass-cost&variants=all",
            "?model=pass-cost&kernels=net",
            "?model=pass-cost&groups=0",
            "?model=pass-cost&render=yes",
            "?model=pass-cost&side=112",
            "?model=pass-cost&kernels=depth&side=100",
            "?model=pass-cost&kernels=depth&groups=2",
            "?model=pass-cost&frames=0",
            "?model=pass-cost&latency=2",
            "?model=pass-cost&latency_ms=80",
            "?model=pass-cost&arena_bytes=64",
            "?model=pass-cost&estimator=capped",
            "?model=pass-cost&sync=latency",
            "?model=pass-cost&bpm=100",
            "?model=pass-cost&passes=8&passes=8",
            "?passes=8",
            "?model=depth&variants=ends",
            "?kernels=trivial",
            "?model=synthetic&render=off",
        ] {
            assert!(parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn frames_latency_and_the_arena_are_read() {
        assert_eq!(
            parse("?frames=300&latency=5&arena_bytes=134217600").unwrap(),
            Config {
                frames: 300,
                model: Model::Synthetic,
                latency: Latency::Frames(5),
                arena_bytes: Some(134_217_600),
                estimator: Estimator::default(),
                render_iters: DEFAULT_RENDER_ITERS,
                render_order: RenderOrder::After,
                on_beat: None,
            }
        );
        assert_eq!(parse("latency=0").unwrap().latency, Latency::Frames(0));
        assert_eq!(parse("&frames=1&").unwrap().frames, 1);
        assert_eq!(
            parse("?model=synthetic&latency_ms=33.3").unwrap().latency,
            Latency::Time(Micros::new(33_300))
        );
    }

    #[test]
    fn the_depth_model_defaults_to_the_native_default_in_time() {
        let config = parse("?model=depth").unwrap();
        assert_eq!(config.model, Model::Depth { side: 266 });
        assert_eq!(config.latency, Latency::Time(Micros::new(83_335)));
        let config = parse("?model=depth&side=518&frames=1200&latency=40").unwrap();
        assert_eq!(config.model, Model::Depth { side: 518 });
        assert_eq!(config.latency, Latency::Frames(40));
        assert_eq!(config.frames, 1200);
        assert_eq!(
            parse("?latency_ms=83.3&side=182&model=depth").unwrap(),
            Config {
                frames: 600,
                model: Model::Depth { side: 182 },
                latency: Latency::Time(Micros::new(83_300)),
                arena_bytes: None,
                estimator: Estimator::default(),
                render_iters: DEFAULT_RENDER_ITERS,
                render_order: RenderOrder::After,
                on_beat: None,
            }
        );
    }

    #[test]
    fn on_beat_delivery_reads_its_tempo_change_margin_and_beats() {
        let config = parse("?model=depth&side=112&sync=onbeat").unwrap();
        assert_eq!(config.model, Model::Depth { side: 112 });
        assert_eq!(
            config.on_beat,
            Some(OnBeat {
                tempo: Tempo {
                    bpm: 100.0,
                    change: None
                },
                margin: Micros::new(100_000),
                beats: None,
            })
        );
        let config = parse(
            "?model=depth&side=112&sync=onbeat&bpm=90&bpm_change=120@60.5&margin_ms=50.5&beats=300",
        )
        .unwrap();
        assert_eq!(
            config.on_beat,
            Some(OnBeat {
                tempo: Tempo {
                    bpm: 90.0,
                    change: Some(TempoChange {
                        bpm: 120.0,
                        after_s: 60.5
                    })
                },
                margin: Micros::new(50_500),
                beats: Some(300),
            })
        );
        // The fixed-latency promise is the default and can be named.
        assert_eq!(parse("?sync=latency").unwrap(), Config::default());
        assert_eq!(
            parse("?sync=onbeat&bpm=30&margin_ms=0&frames=900")
                .unwrap()
                .on_beat
                .map(|o| (o.tempo.bpm, o.margin, o.beats)),
            Some((30.0, Micros::ZERO, None))
        );
        assert_eq!(
            parse("?sync=onbeat&bpm=120")
                .unwrap()
                .on_beat
                .unwrap()
                .tempo
                .bpm,
            120.0
        );
    }

    #[test]
    fn on_beat_parameters_out_of_place_or_range_are_refused() {
        for query in [
            "?sync=OnBeat",
            "?sync=",
            "?bpm=100",
            "?bpm_change=120@60",
            "?margin_ms=50",
            "?beats=300",
            "?sync=latency&bpm=100",
            "?sync=onbeat&latency=10",
            "?sync=onbeat&latency_ms=400",
            "?sync=onbeat&beats=300&frames=600",
            "?sync=onbeat&beats=0",
            "?sync=onbeat&beats=100001",
            "?sync=onbeat&beats=-1",
            "?sync=onbeat&bpm=29.9",
            "?sync=onbeat&bpm=120.1",
            "?sync=onbeat&bpm=NaN",
            "?sync=onbeat&bpm=inf",
            "?sync=onbeat&bpm=fast",
            "?sync=onbeat&bpm_change=120",
            "?sync=onbeat&bpm_change=120@",
            "?sync=onbeat&bpm_change=@60",
            "?sync=onbeat&bpm_change=121@60",
            "?sync=onbeat&bpm_change=120@0",
            "?sync=onbeat&bpm_change=120@-5",
            "?sync=onbeat&bpm_change=120@3601",
            "?sync=onbeat&bpm_change=120@NaN",
            "?sync=onbeat&bpm_change=120@60@70",
            "?sync=onbeat&margin_ms=-1",
            "?sync=onbeat&margin_ms=1000.1",
            "?sync=onbeat&margin_ms=NaN",
            "?sync=onbeat&sync=latency",
        ] {
            assert!(parse(query).is_err(), "{query}");
        }
        let said = parse("?bpm=100").unwrap_err().to_string();
        assert!(said.contains("needs sync=onbeat"), "{said}");
        let said = parse("?sync=onbeat&bpm=200").unwrap_err().to_string();
        assert!(said.contains("from 30 to 120 BPM"), "{said}");
        let said = parse("?sync=onbeat&latency=5").unwrap_err().to_string();
        assert!(said.contains("no latency in frames"), "{said}");
    }

    #[test]
    fn the_estimator_defaults_to_the_cores_and_applies_to_both_jobs() {
        assert_eq!(Config::default().estimator, Estimator::default());
        assert_eq!(
            parse("?model=depth").unwrap().estimator,
            Estimator::default()
        );
        // The forms spelled out for the default are the default; if the core's
        // default moves, this is where it shows.
        assert_eq!(estimator("quantile").unwrap(), Estimator::default());
        assert_eq!(estimator("quantile:0.9").unwrap(), Estimator::default());
        assert_eq!(estimator("quantile:0.9:32").unwrap(), Estimator::default());
        for query in [
            "?estimator=capped",
            "?model=depth&estimator=capped",
            "?estimator=capped&side=266&model=depth&frames=300",
        ] {
            assert_eq!(
                parse(query).unwrap().estimator,
                Estimator::CappedDeviation {
                    alpha: 0.2,
                    sigmas: 2.0,
                    cap: 0.5
                },
                "{query}"
            );
        }
        // Colons need no escaping in a query string.
        let config = parse("?model=depth&estimator=quantile:0.75:16&latency=10").unwrap();
        assert_eq!(
            config.estimator,
            Estimator::WindowedQuantile {
                window: 16,
                quantile: 0.75
            }
        );
        assert_eq!(config.model, Model::Depth { side: 266 });
    }

    #[test]
    fn every_form_of_the_estimator_is_read() {
        let quantile = |quantile, window| Estimator::WindowedQuantile { window, quantile };
        assert_eq!(estimator("quantile:0.75").unwrap(), quantile(0.75, 32));
        assert_eq!(estimator("quantile:1").unwrap(), quantile(1.0, 32));
        assert_eq!(estimator("quantile:0.5:1").unwrap(), quantile(0.5, 1));
        assert_eq!(estimator("quantile:.5:4096").unwrap(), quantile(0.5, 4096));
        assert_eq!(
            estimator("mean-sigmas").unwrap(),
            Estimator::MeanPlusSigmas {
                alpha: 0.2,
                sigmas: 2.0
            }
        );
        assert_eq!(
            estimator("mean-sigmas:1:0").unwrap(),
            Estimator::MeanPlusSigmas {
                alpha: 1.0,
                sigmas: 0.0
            }
        );
        assert_eq!(
            estimator("capped:0.1:3:0").unwrap(),
            Estimator::CappedDeviation {
                alpha: 0.1,
                sigmas: 3.0,
                cap: 0.0
            }
        );
        let scaled = |frame_quantile, frame_window, window| Estimator::FrameScaled {
            window,
            frame_window,
            frame_quantile,
        };
        assert_eq!(estimator("frame-scaled").unwrap(), scaled(0.9, 64, 32));
        assert_eq!(estimator("frame-scaled:0.8").unwrap(), scaled(0.8, 64, 32));
        assert_eq!(
            estimator("frame-scaled:0.95:128").unwrap(),
            scaled(0.95, 128, 32)
        );
        assert_eq!(
            estimator("frame-scaled:1:1:4096").unwrap(),
            scaled(1.0, 1, 4096)
        );
        assert_eq!(
            parse("?model=depth&side=266&frames=1500&latency_ms=200&estimator=frame-scaled")
                .unwrap()
                .estimator,
            scaled(0.9, 64, 32)
        );
    }

    #[test]
    fn an_estimator_not_understood_or_out_of_range_is_refused() {
        for value in [
            "",
            "quantile:",
            "quantile:0",
            "quantile:-0.5",
            "quantile:1.01",
            "quantile:NaN",
            "quantile:inf",
            "quantile:p90",
            "quantile:0.9:",
            "quantile:0.9:0",
            "quantile:0.9:4097",
            "quantile:0.9:-1",
            "quantile:0.9:3.5",
            "quantile:0.9:32:1",
            "mean",
            "mean:0.2:2",
            "mean-sigmas:0.2",
            "mean-sigmas:0:2",
            "mean-sigmas:1.5:2",
            "mean-sigmas:0.2:-1",
            "mean-sigmas:0.2:2:3",
            "capped:",
            "capped:0.2:2",
            "capped:0.2:2:-0.1",
            "capped:0.2:2:inf",
            "capped:0.2:2:0.5:1",
            "Capped",
            "Quantile",
            "mean-sigmas:",
            "quantile%3A0.9",
            "frame-scaled:",
            "frame-scaled:0",
            "frame-scaled:1.5",
            "frame-scaled:NaN",
            "frame-scaled:0.9:0",
            "frame-scaled:0.9:4097",
            "frame-scaled:0.9:64:0",
            "frame-scaled:0.9:64:4097",
            "frame-scaled:0.9:64:32:1",
            "frame-scaled:0.9:6.5",
            "frame",
            "Frame-scaled",
        ] {
            assert!(estimator(value).is_err(), "{value:?}");
            let query = format!("?estimator={value}");
            assert!(parse(&query).is_err(), "{query}");
        }
        assert!(parse("?estimator").is_err());
        assert!(parse("?estimator=mean-sigmas&estimator=capped").is_err());
        let said = estimator("quantile:0").unwrap_err().to_string();
        assert!(said.contains("not in (0, 1]"), "{said}");
        let said = estimator("quantile:0.9:4097").unwrap_err().to_string();
        assert!(said.contains("1..=4096"), "{said}");
        let said = estimator("mean-sigmas:0:2").unwrap_err().to_string();
        assert!(
            said.contains("alpha must be finite and in (0, 1]"),
            "{said}"
        );
        let said = estimator("p99").unwrap_err().to_string();
        assert!(said.contains("quantile:<q>:<window>"), "{said}");
        assert!(
            said.contains("frame-scaled:<q>:<frame_window>:<window>"),
            "{said}"
        );
        let said = estimator("frame-scaled:0").unwrap_err().to_string();
        assert!(said.contains("frame quantile 0 is not in (0, 1]"), "{said}");
        let said = estimator("frame-scaled:0.9:0").unwrap_err().to_string();
        assert!(said.contains("frame window must be in"), "{said}");
    }

    #[test]
    fn the_estimator_is_named_in_a_form_that_reads_back() {
        assert_eq!(estimator_name(Estimator::default()), "quantile");
        for (value, name) in [
            ("quantile", "quantile"),
            ("quantile:0.9", "quantile"),
            ("quantile:0.9:32", "quantile"),
            ("quantile:0.75", "quantile:0.75:32"),
            ("quantile:0.75:16", "quantile:0.75:16"),
            ("quantile:0.9:16", "quantile:0.9:16"),
            ("quantile:1", "quantile:1:32"),
            ("mean-sigmas", "mean-sigmas"),
            ("mean-sigmas:0.2:2", "mean-sigmas"),
            ("mean-sigmas:0.3:2", "mean-sigmas:0.3:2"),
            ("capped", "capped"),
            ("capped:0.2:2:0.5", "capped"),
            ("capped:0.2:2:0.6", "capped:0.2:2:0.6"),
            ("capped:0.25:1.5:0.75", "capped:0.25:1.5:0.75"),
            ("frame-scaled", "frame-scaled"),
            ("frame-scaled:0.9", "frame-scaled"),
            ("frame-scaled:0.9:64", "frame-scaled"),
            ("frame-scaled:0.9:64:32", "frame-scaled"),
            ("frame-scaled:0.8", "frame-scaled:0.8:64:32"),
            ("frame-scaled:0.9:128", "frame-scaled:0.9:128:32"),
            ("frame-scaled:0.9:64:16", "frame-scaled:0.9:64:16"),
        ] {
            let chosen = estimator(value).unwrap();
            assert_eq!(estimator_name(chosen), name, "{value}");
            assert_eq!(estimator(name).unwrap(), chosen, "{name}");
        }
    }

    #[test]
    fn render_order_is_after_unless_free_is_asked_for() {
        assert_eq!(DEFAULT_RENDER_ORDER, RenderOrder::After);
        assert_eq!(Config::default().render_order, RenderOrder::After);
        for query in [
            "",
            "?model=depth",
            "?model=depth&side=518&latency_ms=600",
            "?sync=onbeat&bpm=120",
            "?model=depth&sync=onbeat",
        ] {
            assert_eq!(
                parse(query).unwrap().render_order,
                RenderOrder::After,
                "{query}"
            );
        }
        for (query, order) in [
            ("?render_order=free", RenderOrder::Free),
            ("?render_order=after", RenderOrder::After),
            (
                "?model=depth&side=266&latency_ms=200&frames=1500&render_iters=50&render_order=free",
                RenderOrder::Free,
            ),
            ("?sync=onbeat&bpm=120&render_order=free", RenderOrder::Free),
        ] {
            assert_eq!(parse(query).unwrap().render_order, order, "{query}");
        }
        // Nothing else moves with it.
        assert_eq!(
            parse("?render_order=free").unwrap(),
            Config {
                render_order: RenderOrder::Free,
                ..Config::default()
            }
        );
        assert_eq!(parse("?render_order=after").unwrap(), Config::default());
        for bad in [
            "?render_order=before",
            "?render_order=",
            "?render_order=after&render_order=free",
            "?model=pass-cost&render_order=after",
        ] {
            assert!(parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn render_iters_defaults_to_the_benchs_and_reads_any_whole_number() {
        assert_eq!(Config::default().render_iters, 50);
        assert_eq!(parse("").unwrap().render_iters, DEFAULT_RENDER_ITERS);
        assert_eq!(
            parse("?model=depth&side=112&latency_ms=200")
                .unwrap()
                .render_iters,
            50
        );
        for (query, iters) in [
            ("?render_iters=1", 1),
            ("?render_iters=0", 0),
            ("?render_iters=50", 50),
            ("?render_iters=4294967295", u32::MAX),
            ("?model=depth&side=266&latency_ms=200&render_iters=1", 1),
            ("?model=depth&side=112&render_iters=0&frames=900", 0),
            ("?render_iters=7&sync=onbeat&bpm=120", 7),
            ("?model=depth&sync=onbeat&render_iters=0&beats=8", 0),
        ] {
            assert_eq!(parse(query).unwrap().render_iters, iters, "{query}");
        }
        // Nothing else moves with it.
        let with = parse("?model=depth&side=112&render_iters=1").unwrap();
        assert_eq!(
            with,
            Config {
                model: Model::Depth { side: 112 },
                latency: Latency::Time(NATIVE_DEFAULT_LATENCY),
                render_iters: 1,
                ..Config::default()
            }
        );
    }

    #[test]
    fn render_iters_out_of_range_or_with_the_pass_cost_measurement_is_refused() {
        for bad in [
            "?render_iters",
            "?render_iters=",
            "?render_iters=-1",
            "?render_iters=1.5",
            "?render_iters=fifty",
            "?render_iters=4294967296",
            "?render_iters=1&render_iters=2",
            "?model=pass-cost&render_iters=1",
            "?model=pass-cost&render_iters=50",
            "?model=pass-cost&render=on&render_iters=0",
            "?model=pass-cost&kernels=depth&render_iters=1",
        ] {
            assert!(parse(bad).is_err(), "{bad}");
        }
        let error = parse("?model=pass-cost&render_iters=1")
            .unwrap_err()
            .to_string();
        assert!(
            error.contains("render_iters") && error.contains("render=on"),
            "{error}"
        );
        // The pass-cost measurement's own render switch is unaffected.
        assert!(parse("?model=pass-cost&render=on").is_ok());
        let unknown = parse("?frame=600").unwrap_err().to_string();
        assert!(unknown.contains("render_iters"), "{unknown}");
    }

    #[test]
    fn anything_not_understood_is_refused() {
        for query in [
            "?frame=600",
            "?frames",
            "?frames=",
            "?frames=six",
            "?frames=-1",
            "?frames=0",
            "?latency=99999999999",
            "?arena_bytes=0",
            "?frames=1&frames=2",
            "?model=Depth",
            "?model=",
            "?side=266",
            "?model=synthetic&side=266",
            "?model=depth&side=0",
            "?model=depth&side=265",
            "?model=depth&side=532",
            "?model=depth&side=-14",
            "?model=depth&arena_bytes=1024",
            "?latency=5&latency_ms=83.3",
            "?latency_ms=-1",
            "?latency_ms=NaN",
            "?latency_ms=inf",
            "?latency_ms=60001",
            "?latency_ms=83,3",
        ] {
            assert!(parse(query).is_err(), "{query}");
        }
        let unknown = parse("?frame=600").unwrap_err().to_string();
        assert!(
            unknown.contains("frames, model, side, latency, latency_ms, arena_bytes, estimator"),
            "{unknown}"
        );
    }
}
