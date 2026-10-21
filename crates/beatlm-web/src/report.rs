//! The plain-text summary of a browser run: one fact per line, `name: value`,
//! in a fixed order, so it can be scraped from the page and diffed against a
//! native run.
//!
//! The lines the two hosts share have the bench's names and formats
//! (`beatlm-bench`'s `report.rs`), so a browser report and a native one compare
//! line by line; percentiles come from the bench's own `stats.rs`. The lines
//! only a browser has say where it ran (`user_agent`, `date`), what the device
//! was given (`timestamp_query`, the limits), what the display did
//! (`display_period_us`, `callback_interval_us`) and what went wrong on the GPU
//! (`gpu_errors`), and the render pass's GPU time apart in frames with and
//! without slices (`gpu_render_alone_us`, `gpu_render_with_slices_us`), which
//! says whether the render's measurement, which the budget takes, counts the
//! slices; and the diagnosis of the misses ([`crate::diagnosis`]): how many
//! were abandoned before they started (`missed_unstarted`), after
//! (`missed_started`) or missed at their deadline (`missed_at_deadline`), when
//! jobs started (`jobs_started`), what the projection read for the jobs
//! abandoned before they started (`unstarted_*`, three of them in full as
//! `unstarted_sample`), how many frames the timings came back after
//! (`timing_lag_frames`), the frame's first compute pass apart from the later
//! ones (`first_pass_us`, `later_pass_us`, `first_pass_over_later_mean`), and
//! a job's estimated GPU time when the loop began and ended
//! (`job_estimate_us`). The last line is `status: done`, so a reader can tell a
//! finished report from one still being written.
//!
//! Four shared lines keep their names and change their meaning, because the
//! browser's frame is the display's callback and not a slot on a 60 Hz grid
//! ([`crate::pace`] has the rule), so they compare with a native run only at a
//! display period of 16.7 ms:
//!
//! - `frames_dropped` counts the periods a callback gap of more than 1.5
//!   periods gave up, at the display's period; with `frames` it is the number
//!   of frames a run that completed was asked for.
//! - `frame_start_lateness_us` is the part of the gap since the previous
//!   callback that the periods it counted as do not explain.
//! - `promised_latency_ms` is the promised frames at the display's period,
//!   which is half as long at 120 Hz.
//! - `gpu_frames_over_period` holds a frame to the display's period.
//!
//! The depth model adds the lines its setup measured and checked, after the
//! shared ones and named as the bench names the same facts where it has them
//! (`calibrated_job_us`, `memory_plan_*`): the weights and their inventory
//! check, calibration's GPU time per run (`gpu_run_us`), the comparison with
//! the reference (`reference*`, `depth_check*`), and whether the native
//! default held (`native_default*`). [`DepthReport`] says what each is. A run
//! asked for its latency in time also says what it asked for and the period it
//! was turned into frames at (`latency_asked_ms`, `latency_period_us`); the
//! frames are `config`'s `latency`.
//!
//! `display_period_us` is the period as measured at the end of the run, the
//! median of the last per-period callback intervals, so that a host which
//! overran its frames does not read as a slower display and timestamp jitter
//! does not read as a faster one ([`crate::pace`] says how); the frames before
//! it was measured were given the prior (the nominal 60 Hz one, or for a run
//! that measured the period before it, as the depth model's and one asked for
//! a latency in time do, that measurement), and frames of a display
//! that changed rate were given the rate of their time. `promised_latency_ms` and
//! `gpu_frames_over_period` take this one period for the whole run.
//! `callback_interval_us` is the spread of the raw intervals, stalls included.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use beatlm::gpu::TimerStats;
use beatlm::load::depth_anything_v2::reference::{
    self, DEPTH_MAX_ERROR, DEPTH_MAX_RELATIVE_ERROR, DEPTH_MEAN_ERROR, DepthError,
};
use beatlm_core::cost::SliceId;
use beatlm_core::time::Micros;

use crate::config::{Config, MARGIN_US, Model, SLICES};
use crate::depth::{self, Facts, MAX_DISPATCH_US, NATIVE_DEFAULT_LATENCY, NATIVE_DEFAULT_SIDE};
use crate::diagnosis::Diagnosis;
use crate::pace::NOMINAL_PERIOD;
use crate::stats::{histogram, summarize};
use crate::workload::{DEFAULT_RENDER_ITERS, DEFAULT_SLICE_ITERS};

/// The two limits a memory plan can need raised, as a device or an adapter
/// reports them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SizeLimits {
    /// The largest range one storage binding may cover, in bytes.
    pub max_storage_buffer_binding_size: u64,
    /// The largest buffer, in bytes.
    pub max_buffer_size: u64,
}

/// Where the run happened, and what the device was created with.
#[derive(Debug, Clone, Default)]
pub struct Environment {
    /// The adapter's description as the browser reports it; browsers may
    /// leave it empty.
    pub adapter: String,
    /// The backend, `webgpu` in a browser.
    pub backend: String,
    /// The browser's user agent string.
    pub user_agent: String,
    /// When the run started, as an ISO 8601 date and time.
    pub date: String,
    /// Whether the device has timestamp queries, without which no GPU time is
    /// measured.
    pub timestamp_query: bool,
    /// What the adapter could give.
    pub adapter_limits: SizeLimits,
    /// What the device was created with.
    pub device_limits: SizeLimits,
    /// The arena `arena_bytes` asked for, as planned and realised on the
    /// device; `None` when none was asked for.
    pub arena_realised_bytes: Option<u64>,
}

/// What the page's run of the depth model compared with the reference.
#[derive(Debug, Clone, Default)]
pub struct Check {
    /// The URL of the reference directory at the run's side.
    pub reference_url: String,
    /// The `just` recipe that writes the reference at the run's side.
    pub recipe: String,
    /// The URL of a reference file the server did not have, when one was
    /// missing; the comparison was then not made, and the run went on.
    pub missing: Option<String>,
    /// Something about the reference's files that keeps them from being
    /// compared, such as a depth of another shape.
    pub problem: Option<String>,
    /// The comparison of the reference's input with the formula's: `None`
    /// when not made, `Some(None)` when they are the same bytes, and
    /// `Some(Some(at))` when they first differ at byte `at`.
    pub input_difference: Option<Option<usize>>,
    /// What the page's depth differs from the reference's by; `None` when
    /// the comparison was not made.
    pub depth: Option<DepthError>,
    /// The smallest and the largest depth the page computed, which the display
    /// maps to black and white.
    pub depth_range: (f32, f32),
}

impl Check {
    /// `pass`, `fail: ` with why, or `missing: ` with what to run.
    #[must_use]
    pub fn result(&self) -> String {
        if let Some(url) = &self.missing {
            return format!(
                "missing: the server has no {url}; run `just {}` and serve with `just serve-web`",
                self.recipe
            );
        }
        if let Some(problem) = &self.problem {
            return format!("fail: {problem}");
        }
        if let Some(Some(at)) = self.input_difference {
            return format!("fail: the reference input differs from the formula at byte {at}");
        }
        match &self.depth {
            Some(error) if error.within_tolerance() => "pass".to_owned(),
            Some(_) => "fail: the depth is beyond the tolerances".to_owned(),
            None => "fail: no comparison was made".to_owned(),
        }
    }
}

/// What the depth model's setup read, measured and decided before the loop.
#[derive(Debug, Clone, Default)]
pub struct DepthReport {
    /// The input side, in pixels.
    pub side: u32,
    /// The URL the weights were fetched from.
    pub weights_url: String,
    /// The size of the weights file.
    pub weights_bytes: u64,
    /// The tensors the inventory lists, which the file matched exactly.
    pub tensors: usize,
    /// The parameters of the file.
    pub parameters: u64,
    /// The kernels of the program before calibration split it: `config`'s
    /// `slices`, as the bench prints it.
    pub kernels: u32,
    /// The slices of a job after the split: one per dispatch.
    pub slices: u32,
    /// Each whole kernel's median in calibration's first pass, in program
    /// order; their sum is `calibrated_job_us`.
    pub calibrated: Vec<Micros>,
    /// The GPU time of each counted run of calibration's first pass, every
    /// kernel of the unsplit program summed, in microseconds: `gpu_run_us`.
    pub runs_us: Vec<u64>,
    /// Bytes of each arena of the loop's plan.
    pub arenas: Vec<u64>,
    /// Bytes of all arenas.
    pub total_bytes: u64,
    /// The most bytes live at one kernel.
    pub peak_live_bytes: u64,
    /// The input and output slots: one per job that can be in flight.
    pub slots: u32,
    /// The comparison with the reference.
    pub check: Check,
}

/// Everything a run measured. The per-frame vectors hold one sample per frame
/// run, except where a field says otherwise.
#[derive(Debug, Clone, Default)]
pub struct Outcome {
    /// The promised latency in frames the run kept: `latency`, or
    /// `latency_ms` at the display's period measured before the run.
    pub latency: u32,
    /// The latency `latency_ms` asked for, in microseconds; `None` when it was
    /// given in frames.
    pub latency_asked_us: Option<u64>,
    /// The display's period `latency_ms` was turned into frames at, in
    /// microseconds.
    pub latency_period_us: Option<u64>,
    /// What the depth model's setup measured; `None` for the synthetic
    /// workload.
    pub depth: Option<DepthReport>,
    /// Why jobs were missed, and what their estimates were made of.
    pub diagnosis: Diagnosis,
    /// From the callback's start to the return of `queue.submit`, in
    /// microseconds.
    pub cpu_us: Vec<u64>,
    /// From the callback's start to the end of the frame's work, delivery
    /// bookkeeping included, in microseconds.
    pub tick_us: Vec<u64>,
    /// Deciding the frame (`Scheduler::submit` plus `plan_frame`), in
    /// microseconds.
    pub plan_us: Vec<u64>,
    /// Recording the frame's command buffer, acquiring the canvas texture
    /// included, in microseconds.
    pub encode_us: Vec<u64>,
    /// Inside `queue.submit`, in microseconds.
    pub submit_us: Vec<u64>,
    /// Compute dispatches encoded in each frame.
    pub dispatches_per_frame: Vec<u64>,
    /// Compute dispatches encoded for each delivered job.
    pub dispatches_per_job: Vec<u64>,
    /// How late each frame's callback came after the time the previous
    /// callback and the periods its gap counted as put it at, in microseconds
    /// ([`crate::pace`] defines it).
    pub lateness_us: Vec<u64>,
    /// Frame periods given up because a callback came more than 1.5 periods
    /// after the one before.
    pub dropped: u64,
    /// The display's period as measured from the callbacks, in microseconds;
    /// `None` when the run was too short to measure it, and then the frames
    /// were given the nominal period.
    pub display_period_us: Option<u64>,
    /// Every interval between consecutive callbacks, in microseconds.
    pub interval_us: Vec<u64>,
    /// Render plus every slice measured in the same frame, per timed frame, in
    /// microseconds.
    pub gpu_frame_us: Vec<u64>,
    /// The render pass's measured GPU time in each timed frame that ran no
    /// slice, in microseconds: the render's own cost.
    pub render_alone_us: Vec<u64>,
    /// The render pass's measured GPU time in each timed frame that ran a
    /// slice, in microseconds. Above `render_alone_us`, the measurement counts
    /// time the pass spent on or behind the slices, and the budget, which
    /// takes the render measured, shrinks by it ([`crate::render_cost`]).
    pub render_with_slices_us: Vec<u64>,
    /// The longest compute pass measured in each timed frame that ran one, in
    /// microseconds.
    pub longest_dispatch_us: Vec<u64>,
    /// Latency in frames of each delivered result.
    pub latencies: Vec<u64>,
    /// Inputs submitted as jobs: one per frame.
    pub submitted: u64,
    /// Results delivered.
    pub delivered: u64,
    /// Deadlines that arrived without a complete result.
    pub missed: u64,
    /// The part of `missed` that belongs to jobs submitted in the first
    /// [`STARTUP_FRAMES`] frames.
    pub startup_missed: u64,
    /// Deliveries whose latency was not the promised one.
    pub violations: u64,
    /// Jobs still awaiting their deadline when the run ended.
    pub pending_at_end: usize,
    /// Per slice: sum and count of measured microseconds.
    pub measured: BTreeMap<SliceId, (u64, u64)>,
    /// Per slice: the cost model's final estimate.
    pub estimates: BTreeMap<SliceId, Micros>,
    /// What the timer could not measure.
    pub timer: TimerStats,
    /// Frames whose canvas texture could not be acquired, so their render pass
    /// did not run.
    pub canvas_unavailable: u64,
    /// Errors the device raised during the run.
    pub gpu_errors: u64,
}

/// The first frames whose jobs' misses are reported apart as
/// `startup_missed`, as the bench counts them: twice the window the cost
/// model's default estimate reads.
pub const STARTUP_FRAMES: u64 = 64;

/// The cost model's estimator, by the name the bench's `--estimator` gives
/// it: the core's default.
const ESTIMATOR: &str = "quantile";

/// Renders the summary of a finished run.
#[must_use]
pub fn render(env: &Environment, config: &Config, outcome: &Outcome) -> String {
    let mut out = String::new();
    // Writing to a String cannot fail.
    let _ = write_report(&mut out, env, config, outcome);
    out
}

fn write_report(
    out: &mut String,
    env: &Environment,
    config: &Config,
    outcome: &Outcome,
) -> std::fmt::Result {
    writeln!(out, "run: 1/1")?;
    writeln!(out, "adapter: {}", reported(&env.adapter))?;
    writeln!(out, "backend: {}", reported(&env.backend))?;
    writeln!(out, "user_agent: {}", reported(&env.user_agent))?;
    writeln!(out, "date: {}", reported(&env.date))?;
    if env.timestamp_query {
        writeln!(out, "timestamp_query: available")?;
    } else {
        writeln!(
            out,
            "timestamp_query: unavailable; no GPU pass is timed, so gpu_frame_us, \
             longest_dispatch_us and the slice measurements are none, and the budget and the \
             slice estimates stay at their priors"
        )?;
    }
    write_limits(out, "adapter_limits", env.adapter_limits)?;
    write_limits(out, "device_limits", env.device_limits)?;
    if let Some(bytes) = env.arena_realised_bytes {
        writeln!(out, "arena_realised_bytes: {bytes}")?;
    }
    let (slices, submit) = match &outcome.depth {
        Some(depth) => (depth.kernels, "free"),
        None => (SLICES, "every"),
    };
    write!(
        out,
        "config: frames={} latency={} slices={slices} slice_iters={DEFAULT_SLICE_ITERS} \
         render_iters={DEFAULT_RENDER_ITERS} margin_us={MARGIN_US} estimator={ESTIMATOR} \
         submit={submit}",
        config.frames, outcome.latency,
    )?;
    if let Some(bytes) = config.arena_bytes {
        write!(out, " arena_bytes={bytes}")?;
    }
    match config.model {
        Model::Depth { side } => writeln!(
            out,
            " workload=depth size={side} fusion=on split=on max_dispatch_us={MAX_DISPATCH_US}"
        )?,
        Model::Synthetic => writeln!(out)?,
    }
    if let Some(us) = outcome.latency_asked_us {
        writeln!(out, "latency_asked_ms: {}", milliseconds(us))?;
    }
    if let Some(us) = outcome.latency_period_us {
        writeln!(out, "latency_period_us: {us}")?;
    }
    match outcome.display_period_us {
        Some(us) => writeln!(out, "display_period_us: {us}")?,
        None => writeln!(out, "display_period_us: none")?,
    }
    let period = period_of(outcome);
    writeln!(
        out,
        "promised_latency_ms: {:.1}",
        promised_latency_ms(outcome.latency, period)
    )?;
    writeln!(out, "frames: {}", outcome.cpu_us.len())?;
    writeln!(out, "frames_dropped: {}", outcome.dropped)?;
    writeln!(out, "submitted: {}", outcome.submitted)?;
    writeln!(out, "delivered: {}", outcome.delivered)?;
    writeln!(out, "missed: {}", outcome.missed)?;
    writeln!(out, "startup_missed: {}", outcome.startup_missed)?;
    match delivery_rate(outcome.delivered, outcome.missed) {
        Some(rate) => writeln!(out, "delivery_rate: {rate:.4}")?,
        None => writeln!(out, "delivery_rate: none")?,
    }
    writeln!(out, "pending_at_end: {}", outcome.pending_at_end)?;
    writeln!(out, "latency_violations: {}", outcome.violations)?;
    writeln!(out, "timings_collected: {}", outcome.timer.frames_collected)?;
    writeln!(out, "timings_skipped: {}", outcome.timer.frames_skipped)?;
    writeln!(out, "timings_failed: {}", outcome.timer.frames_failed)?;
    for (frames, count) in histogram(outcome.latencies.iter().copied()) {
        writeln!(out, "latency_frames[{frames}]: {count}")?;
    }
    write_summary(out, "dispatches_per_job", &outcome.dispatches_per_job)?;
    write_summary(out, "dispatches_per_frame", &outcome.dispatches_per_frame)?;
    write_summary(out, "callback_interval_us", &outcome.interval_us)?;
    write_summary(out, "frame_start_lateness_us", &outcome.lateness_us)?;
    write_summary(out, "cpu_start_to_submit_us", &outcome.cpu_us)?;
    write_summary(out, "cpu_start_to_tick_end_us", &outcome.tick_us)?;
    write_summary(out, "cpu_plan_us", &outcome.plan_us)?;
    write_summary(out, "cpu_encode_us", &outcome.encode_us)?;
    write_summary(out, "cpu_queue_submit_us", &outcome.submit_us)?;
    write_summary(out, "gpu_frame_us", &outcome.gpu_frame_us)?;
    write_summary(out, "gpu_render_alone_us", &outcome.render_alone_us)?;
    write_summary(
        out,
        "gpu_render_with_slices_us",
        &outcome.render_with_slices_us,
    )?;
    write_summary(out, "longest_dispatch_us", &outcome.longest_dispatch_us)?;
    writeln!(
        out,
        "gpu_frames_over_period: {} of {}",
        gpu_frames_over_period(&outcome.gpu_frame_us, period),
        outcome.gpu_frame_us.len()
    )?;
    for (slice, estimate) in &outcome.estimates {
        let (sum, count) = outcome.measured.get(slice).copied().unwrap_or((0, 0));
        write!(
            out,
            "slice[{}]: estimate_us={} measured_samples={}",
            slice.get(),
            estimate.get(),
            count
        )?;
        match sum.checked_div(count) {
            Some(mean) => writeln!(out, " measured_mean_us={mean}")?,
            None => writeln!(out, " measured_mean_us=none")?,
        }
    }
    write_diagnosis(out, &outcome.diagnosis)?;
    if let Some(depth) = &outcome.depth {
        write_depth(out, depth, outcome)?;
    }
    writeln!(out, "canvas_unavailable: {}", outcome.canvas_unavailable)?;
    writeln!(out, "gpu_errors: {}", outcome.gpu_errors)?;
    writeln!(out, "status: done")
}

/// The diagnosis lines ([`crate::diagnosis`]).
fn write_diagnosis(out: &mut String, d: &Diagnosis) -> std::fmt::Result {
    writeln!(out, "missed_unstarted: {}", d.unstarted)?;
    writeln!(out, "missed_started: {}", d.started)?;
    writeln!(out, "missed_at_deadline: {}", d.at_deadline)?;
    write_span(out, "jobs_started", d.starts.iter().copied())?;
    write_span(
        out,
        "missed_unstarted_frames",
        d.abandoned.iter().map(|a| a.frame),
    )?;
    let column = |f: fn(&crate::diagnosis::Unstarted) -> u64| -> Vec<u64> {
        d.abandoned.iter().map(f).collect()
    };
    write_summary(out, "unstarted_now_budget_us", &column(|a| a.now_us))?;
    write_summary(
        out,
        "unstarted_expected_budget_us",
        &column(|a| a.expected_us),
    )?;
    write_summary(
        out,
        "unstarted_largest_estimate_us",
        &column(|a| a.largest_us),
    )?;
    write_summary(out, "unstarted_estimate_sum_us", &column(|a| a.sum_us))?;
    write_summary(out, "unstarted_frames_left", &column(|a| a.frames_left))?;
    let ends: Vec<u64> = d.abandoned.iter().filter_map(|a| a.projected_end).collect();
    write_summary(out, "unstarted_projected_end_frame", &ends)?;
    writeln!(
        out,
        "unstarted_slice_fits_no_frame: {}",
        d.abandoned.len() - ends.len()
    )?;
    for a in d.abandoned.iter().take(3) {
        write!(
            out,
            "unstarted_sample: frame={} now_budget_us={} expected_budget_us={} \
             largest_estimate_us={} estimate_sum_us={} frames_left={}",
            a.frame, a.now_us, a.expected_us, a.largest_us, a.sum_us, a.frames_left
        )?;
        match a.projected_end {
            Some(end) => writeln!(out, " projected_end_frame={end}")?,
            None => writeln!(out, " projected_end_frame=none")?,
        }
    }
    write_summary(out, "timing_lag_frames", &d.lag_frames)?;
    write_summary(out, "first_pass_us", &d.first_pass_us)?;
    write_summary(out, "later_pass_us", &d.later_pass_us)?;
    match d.first_over_later() {
        Some((ratio, slices)) => writeln!(
            out,
            "first_pass_over_later_mean: median={ratio:.2} slices={slices}"
        )?,
        None => writeln!(out, "first_pass_over_later_mean: none")?,
    }
    writeln!(
        out,
        "job_estimate_us: loop_start={} loop_end={}",
        d.job_estimate_us.0, d.job_estimate_us.1
    )
}

/// `name: count first=a last=b` over frame indices, or `name: 0`.
fn write_span(out: &mut String, name: &str, frames: impl Iterator<Item = u64>) -> std::fmt::Result {
    let frames: Vec<u64> = frames.collect();
    match (frames.iter().min(), frames.iter().max()) {
        (Some(first), Some(last)) => writeln!(
            out,
            "{name}: {} first_frame={first} last_frame={last}",
            frames.len()
        ),
        _ => writeln!(out, "{name}: 0"),
    }
}

/// The depth model's lines.
fn write_depth(out: &mut String, depth: &DepthReport, outcome: &Outcome) -> std::fmt::Result {
    writeln!(out, "model: {}", reference::MODEL_DIR)?;
    writeln!(
        out,
        "weights: url={} bytes={}",
        depth.weights_url, depth.weights_bytes
    )?;
    writeln!(
        out,
        "inventory: exact tensors={} parameters={}",
        depth.tensors, depth.parameters
    )?;
    writeln!(out, "job_kernels: {}", depth.kernels)?;
    writeln!(out, "job_slices: {}", depth.slices)?;
    let calibrated: u64 = depth.calibrated.iter().map(|us| us.get()).sum();
    writeln!(out, "calibrated_job_us: {calibrated}")?;
    write_summary(out, "gpu_run_us", &depth.runs_us)?;
    writeln!(out, "memory_plan_io_slots: {}", depth.slots)?;
    writeln!(out, "memory_plan_arenas: {}", depth.arenas.len())?;
    for (index, bytes) in depth.arenas.iter().enumerate() {
        writeln!(out, "memory_plan_arena[{index}]_bytes: {bytes}")?;
    }
    writeln!(out, "memory_plan_total_bytes: {}", depth.total_bytes)?;
    writeln!(
        out,
        "memory_plan_peak_live_bytes: {}",
        depth.peak_live_bytes
    )?;
    let check = &depth.check;
    writeln!(out, "reference: {}", reported(&check.reference_url))?;
    match check.input_difference {
        None => writeln!(out, "reference_input: not compared")?,
        Some(None) => writeln!(out, "reference_input: identical")?,
        Some(Some(at)) => writeln!(out, "reference_input: differs_at_byte={at}")?,
    }
    match &check.depth {
        Some(error) => writeln!(out, "depth_check: {}", error.describe())?,
        None => writeln!(out, "depth_check: none")?,
    }
    writeln!(
        out,
        "depth_check_tolerance: max_abs_error={DEPTH_MAX_ERROR:e} \
         mean_abs_error={DEPTH_MEAN_ERROR:e} max_relative_error={DEPTH_MAX_RELATIVE_ERROR:e}"
    )?;
    writeln!(out, "depth_check_result: {}", one_line(&check.result()))?;
    writeln!(
        out,
        "depth_range: low={:.4} high={:.4}",
        check.depth_range.0, check.depth_range.1
    )?;
    let period = outcome.display_period_us.map(Micros::new);
    let frames = period.map(|period| depth::frames_for(NATIVE_DEFAULT_LATENCY, period));
    write!(
        out,
        "native_default: size={NATIVE_DEFAULT_SIDE} latency_ms={}",
        milliseconds(NATIVE_DEFAULT_LATENCY.get())
    )?;
    match frames {
        Some(frames) => writeln!(out, " latency_frames_at_display_period={frames}")?,
        None => writeln!(out, " latency_frames_at_display_period=none")?,
    }
    let facts = Facts {
        side: depth.side,
        latency: outcome.latency,
        period,
        gpu_frame_p99: summarize(&outcome.gpu_frame_us).map(|s| s.p99),
        violations: outcome.violations,
        delivered: outcome.delivered,
        missed: outcome.missed,
        unstarted: outcome.diagnosis.unstarted,
        job_estimate_us: outcome.diagnosis.job_estimate_us,
        run_us: summarize(&depth.runs_us).map(|s| s.p50),
    };
    writeln!(
        out,
        "native_default_holds: {}",
        one_line(&depth::native_default(&facts))
    )
}

/// Microseconds as milliseconds to one decimal place, the precision the
/// report's other latencies in milliseconds have.
#[allow(clippy::cast_precision_loss)]
pub(crate) fn milliseconds(us: u64) -> String {
    format!("{:.1}", us as f64 / 1_000.0)
}

/// `text` on one line: every run of whitespace, line breaks included, as one
/// space, so a message from the browser keeps the report's one fact per line.
#[must_use]
pub fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// `text` on one line, or `unreported` when the browser gave nothing, so every
/// line has a value; browsers may leave the adapter's description empty.
fn reported(text: &str) -> String {
    let text = one_line(text);
    if text.is_empty() {
        String::from("unreported")
    } else {
        text
    }
}

fn write_limits(out: &mut String, name: &str, limits: SizeLimits) -> std::fmt::Result {
    writeln!(
        out,
        "{name}: max_storage_buffer_binding_size={} max_buffer_size={}",
        limits.max_storage_buffer_binding_size, limits.max_buffer_size
    )
}

fn write_summary(out: &mut String, name: &str, samples: &[u64]) -> std::fmt::Result {
    match summarize(samples) {
        Some(s) => writeln!(out, "{name}: p50={} p99={} max={}", s.p50, s.p99, s.max),
        None => writeln!(out, "{name}: none"),
    }
}

/// Delivered over delivered plus missed: the share of deadlines that arrived
/// with their result. `None` when no deadline arrived.
#[allow(clippy::cast_precision_loss)]
fn delivery_rate(delivered: u64, missed: u64) -> Option<f64> {
    let total = delivered.saturating_add(missed);
    (total > 0).then(|| delivered as f64 / total as f64)
}

/// The period the run's frames were given: the display's as measured, or the
/// nominal one when the run was too short to measure it.
fn period_of(outcome: &Outcome) -> Micros {
    outcome
        .display_period_us
        .map_or(NOMINAL_PERIOD, Micros::new)
}

/// The promised latency of `frames` frames of `period`, in milliseconds. The
/// time a promise in frames is worth follows the display, so this is not the
/// bench's figure unless the display is 60 Hz.
#[allow(clippy::cast_precision_loss)]
fn promised_latency_ms(frames: u32, period: Micros) -> f64 {
    f64::from(frames) * period.get() as f64 / 1_000.0
}

/// Timed frames whose render plus slices took longer than the frame `period`,
/// the display's, so a 120 Hz run is held to 8.3 ms.
fn gpu_frames_over_period(gpu_frame_us: &[u64], period: Micros) -> usize {
    gpu_frame_us.iter().filter(|&&us| us > period.get()).count()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn report(env: &Environment, outcome: &Outcome) -> String {
        let outcome = Outcome {
            latency: 2,
            ..outcome.clone()
        };
        render(env, &Config::default(), &outcome)
    }

    fn line<'a>(text: &'a str, name: &str) -> Option<&'a str> {
        text.lines()
            .find_map(|l| l.strip_prefix(name)?.strip_prefix(": "))
    }

    #[test]
    fn the_lines_shared_with_the_bench_have_its_names_and_formats() {
        let outcome = Outcome {
            cpu_us: vec![900, 1_100, 1_000],
            gpu_frame_us: vec![6_000, 17_000, 7_000],
            latencies: vec![2, 2, 2],
            submitted: 3,
            delivered: 3,
            missed: 1,
            ..Outcome::default()
        };
        let text = report(&Environment::default(), &outcome);
        assert_eq!(line(&text, "delivered"), Some("3"));
        assert_eq!(line(&text, "missed"), Some("1"));
        assert_eq!(line(&text, "delivery_rate"), Some("0.7500"));
        assert_eq!(line(&text, "latency_violations"), Some("0"));
        assert_eq!(line(&text, "latency_frames[2]"), Some("3"));
        assert_eq!(
            line(&text, "cpu_start_to_submit_us"),
            Some("p50=1000 p99=1100 max=1100")
        );
        assert_eq!(
            line(&text, "gpu_frame_us"),
            Some("p50=7000 p99=17000 max=17000")
        );
        assert_eq!(line(&text, "gpu_frames_over_period"), Some("1 of 3"));
        assert_eq!(line(&text, "promised_latency_ms"), Some("33.3"));
        assert_eq!(line(&text, "display_period_us"), Some("none"));
        assert_eq!(
            line(&text, "config"),
            Some(
                "frames=600 latency=2 slices=4 slice_iters=500 render_iters=50 margin_us=2000 \
                 estimator=quantile submit=every"
            )
        );
        assert_eq!(text.lines().last(), Some("status: done"));
    }

    #[test]
    fn every_line_is_one_name_and_one_value() {
        let text = report(&Environment::default(), &Outcome::default());
        for l in text.lines() {
            let (name, value) = l.split_once(": ").unwrap();
            assert!(!name.is_empty() && !name.contains(' '), "{l}");
            assert!(!value.is_empty(), "{l}");
        }
        assert_eq!(line(&text, "adapter"), Some("unreported"));
        assert_eq!(line(&text, "gpu_frame_us"), Some("none"));
        assert_eq!(line(&text, "delivery_rate"), Some("none"));
    }

    #[test]
    fn a_missing_timestamp_query_says_what_is_not_measured() {
        let text = report(&Environment::default(), &Outcome::default());
        let said = line(&text, "timestamp_query").unwrap();
        assert!(said.starts_with("unavailable;"), "{said}");
        assert!(said.contains("gpu_frame_us"), "{said}");
        let env = Environment {
            timestamp_query: true,
            ..Environment::default()
        };
        assert_eq!(
            line(&report(&env, &Outcome::default()), "timestamp_query"),
            Some("available")
        );
    }

    #[test]
    fn the_promise_and_the_period_follow_the_display() {
        assert_eq!(promised_latency_ms(0, NOMINAL_PERIOD), 0.0);
        assert!((promised_latency_ms(5, NOMINAL_PERIOD) - 83.335).abs() < 1e-9);
        assert!((promised_latency_ms(2, Micros::new(8_333)) - 16.666).abs() < 1e-9);
        let period = NOMINAL_PERIOD;
        let over = [
            period.get() - 1,
            period.get(),
            period.get() + 1,
            period.get() * 2,
        ];
        assert_eq!(gpu_frames_over_period(&over, period), 2);
        assert_eq!(gpu_frames_over_period(&over, Micros::new(8_333)), 4);
        assert_eq!(delivery_rate(0, 0), None);
        assert_eq!(delivery_rate(5, 0), Some(1.0));
    }

    #[test]
    fn a_measured_display_period_is_reported_and_sets_the_promise() {
        let outcome = Outcome {
            display_period_us: Some(8_333),
            interval_us: vec![8_300, 8_400, 8_333],
            gpu_frame_us: vec![8_000, 9_000],
            ..Outcome::default()
        };
        let text = report(&Environment::default(), &outcome);
        assert_eq!(line(&text, "display_period_us"), Some("8333"));
        assert_eq!(line(&text, "promised_latency_ms"), Some("16.7"));
        assert_eq!(
            line(&text, "callback_interval_us"),
            Some("p50=8333 p99=8400 max=8400")
        );
        assert_eq!(line(&text, "gpu_frames_over_period"), Some("1 of 2"));
    }

    #[test]
    fn the_render_is_reported_apart_with_and_without_slices() {
        let outcome = Outcome {
            render_alone_us: vec![1_900, 1_950],
            render_with_slices_us: vec![6_100],
            ..Outcome::default()
        };
        let text = report(&Environment::default(), &outcome);
        assert_eq!(
            line(&text, "gpu_render_alone_us"),
            Some("p50=1900 p99=1950 max=1950")
        );
        assert_eq!(
            line(&text, "gpu_render_with_slices_us"),
            Some("p50=6100 p99=6100 max=6100")
        );
    }

    fn depth_run(check: Check) -> String {
        let config = Config {
            model: Model::Depth { side: 266 },
            ..Config::default()
        };
        let outcome = Outcome {
            latency: 10,
            latency_asked_us: Some(83_335),
            latency_period_us: Some(8_333),
            display_period_us: Some(8_333),
            gpu_frame_us: vec![7_000, 7_500],
            delivered: 50,
            depth: Some(DepthReport {
                side: 266,
                weights_url: "models/depth-anything-v2-small/model.safetensors".to_owned(),
                weights_bytes: 99_173_660,
                tensors: 300,
                parameters: 24_785_089,
                kernels: 155,
                slices: 160,
                calibrated: vec![Micros::new(30_000), Micros::new(24_000)],
                runs_us: vec![54_000, 53_000, 55_000],
                arenas: vec![100, 200],
                total_bytes: 300,
                peak_live_bytes: 250,
                slots: 11,
                check,
            }),
            ..Outcome::default()
        };
        render(&Environment::default(), &config, &outcome)
    }

    #[test]
    fn a_depth_run_reports_its_setup_check_and_the_native_default() {
        let want = vec![1.5f32, 2.0, 3.0];
        let check = Check {
            reference_url: "models/depth-anything-v2-small/reference/266/".to_owned(),
            recipe: "reference-depth-anything 266".to_owned(),
            input_difference: Some(None),
            depth: DepthError::of(&want, &want),
            depth_range: (1.5, 3.0),
            ..Check::default()
        };
        let text = depth_run(check);
        assert_eq!(
            line(&text, "config"),
            Some(
                "frames=600 latency=10 slices=155 slice_iters=500 render_iters=50 margin_us=2000 \
                 estimator=quantile submit=free workload=depth size=266 fusion=on split=on \
                 max_dispatch_us=1000"
            )
        );
        assert_eq!(line(&text, "latency_asked_ms"), Some("83.3"));
        assert_eq!(line(&text, "latency_period_us"), Some("8333"));
        assert_eq!(line(&text, "promised_latency_ms"), Some("83.3"));
        assert_eq!(line(&text, "job_slices"), Some("160"));
        assert_eq!(line(&text, "calibrated_job_us"), Some("54000"));
        assert_eq!(
            line(&text, "gpu_run_us"),
            Some("p50=54000 p99=55000 max=55000")
        );
        assert_eq!(line(&text, "memory_plan_arena[1]_bytes"), Some("200"));
        assert_eq!(line(&text, "reference_input"), Some("identical"));
        assert_eq!(line(&text, "depth_check_result"), Some("pass"));
        assert_eq!(
            line(&text, "depth_check_tolerance"),
            Some("max_abs_error=5e-5 mean_abs_error=1e-5 max_relative_error=2e-5")
        );
        assert_eq!(
            line(&text, "native_default"),
            Some("size=266 latency_ms=83.3 latency_frames_at_display_period=10")
        );
        assert_eq!(line(&text, "native_default_holds"), Some("yes"));
        for l in text.lines() {
            let (name, value) = l.split_once(": ").unwrap();
            assert!(!name.is_empty() && !name.contains(' '), "{l}");
            assert!(!value.is_empty(), "{l}");
        }
        assert_eq!(text.lines().last(), Some("status: done"));
    }

    #[test]
    fn a_missing_reference_is_said_and_the_run_still_reports() {
        let check = Check {
            reference_url: "models/depth-anything-v2-small/reference/182/".to_owned(),
            recipe: "reference-depth-anything 182".to_owned(),
            missing: Some("models/depth-anything-v2-small/reference/182/manifest.json".to_owned()),
            ..Check::default()
        };
        let text = depth_run(check);
        let said = line(&text, "depth_check_result").unwrap();
        assert!(
            said.starts_with("missing: the server has no models/"),
            "{said}"
        );
        assert!(
            said.contains("`just reference-depth-anything 182`"),
            "{said}"
        );
        assert_eq!(line(&text, "depth_check"), Some("none"));
        assert_eq!(line(&text, "reference_input"), Some("not compared"));
        assert_eq!(line(&text, "native_default_holds"), Some("yes"));
        assert_eq!(text.lines().last(), Some("status: done"));
    }

    #[test]
    fn a_check_fails_on_a_wrong_input_or_a_depth_beyond_the_tolerances() {
        let want = [2.0f32];
        let wrong_input = Check {
            input_difference: Some(Some(12)),
            depth: DepthError::of(&want, &want),
            ..Check::default()
        };
        assert_eq!(
            wrong_input.result(),
            "fail: the reference input differs from the formula at byte 12"
        );
        let far = Check {
            input_difference: Some(None),
            depth: DepthError::of(&[2.1], &want),
            ..Check::default()
        };
        assert_eq!(far.result(), "fail: the depth is beyond the tolerances");
        let shape = Check {
            problem: Some("a depth of shape [1, 2, 2]".to_owned()),
            ..Check::default()
        };
        assert!(shape.result().starts_with("fail: a depth of shape"));
    }
}
