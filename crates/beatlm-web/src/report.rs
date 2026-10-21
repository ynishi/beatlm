//! The plain-text summary of a browser run: one fact per line, `name: value`,
//! in a fixed order, so it can be scraped from the page and diffed against a
//! native run.
//!
//! The lines the two hosts share have the bench's names and formats
//! (`beatlm-bench`'s `report.rs`), so a browser report and a native one compare
//! line by line; percentiles come from the bench's own `stats.rs`. The lines
//! only a browser has say where it ran (`user_agent`, `date`), what the device
//! was given (`timestamp_query`, the limits) and what went wrong on the GPU
//! (`gpu_errors`). The last line is `status: done`, so a reader can tell a
//! finished report from one still being written.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use beatlm::gpu::TimerStats;
use beatlm_core::cost::SliceId;
use beatlm_core::time::Micros;

use crate::config::{Config, MARGIN_US, SLICES};
use crate::pace::FRAME_PERIOD;
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

/// Everything a run measured. The per-frame vectors hold one sample per frame
/// run, except where a field says otherwise.
#[derive(Debug, Clone, Default)]
pub struct Outcome {
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
    /// How late each frame began after its on-time phase, in microseconds.
    pub lateness_us: Vec<u64>,
    /// Frame periods given up because a frame began whole periods late.
    pub dropped: u64,
    /// Render plus every slice measured in the same frame, per timed frame, in
    /// microseconds.
    pub gpu_frame_us: Vec<u64>,
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
    write!(
        out,
        "config: frames={} latency={} slices={SLICES} slice_iters={DEFAULT_SLICE_ITERS} \
         render_iters={DEFAULT_RENDER_ITERS} margin_us={MARGIN_US} estimator={ESTIMATOR} \
         submit=every",
        config.frames, config.latency,
    )?;
    match config.arena_bytes {
        Some(bytes) => writeln!(out, " arena_bytes={bytes}")?,
        None => writeln!(out)?,
    }
    writeln!(
        out,
        "promised_latency_ms: {:.1}",
        promised_latency_ms(config.latency)
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
    write_summary(out, "frame_start_lateness_us", &outcome.lateness_us)?;
    write_summary(out, "cpu_start_to_submit_us", &outcome.cpu_us)?;
    write_summary(out, "cpu_start_to_tick_end_us", &outcome.tick_us)?;
    write_summary(out, "cpu_plan_us", &outcome.plan_us)?;
    write_summary(out, "cpu_encode_us", &outcome.encode_us)?;
    write_summary(out, "cpu_queue_submit_us", &outcome.submit_us)?;
    write_summary(out, "gpu_frame_us", &outcome.gpu_frame_us)?;
    write_summary(out, "longest_dispatch_us", &outcome.longest_dispatch_us)?;
    writeln!(
        out,
        "gpu_frames_over_period: {} of {}",
        gpu_frames_over_period(&outcome.gpu_frame_us),
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
    writeln!(out, "canvas_unavailable: {}", outcome.canvas_unavailable)?;
    writeln!(out, "gpu_errors: {}", outcome.gpu_errors)?;
    writeln!(out, "status: done")
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

/// The promised latency of `frames` frames at the period, in milliseconds.
#[allow(clippy::cast_precision_loss)]
fn promised_latency_ms(frames: u32) -> f64 {
    f64::from(frames) * FRAME_PERIOD.get() as f64 / 1_000.0
}

/// Timed frames whose render plus slices took longer than the frame period.
fn gpu_frames_over_period(gpu_frame_us: &[u64]) -> usize {
    gpu_frame_us
        .iter()
        .filter(|&&us| us > FRAME_PERIOD.get())
        .count()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn report(env: &Environment, outcome: &Outcome) -> String {
        render(env, &Config::default(), outcome)
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
    fn the_period_and_the_promise_are_the_benchs() {
        assert_eq!(promised_latency_ms(0), 0.0);
        assert!((promised_latency_ms(5) - 83.335).abs() < 1e-9);
        let period = FRAME_PERIOD.get();
        assert_eq!(
            gpu_frames_over_period(&[period - 1, period, period + 1, period * 2]),
            2
        );
        assert_eq!(delivery_rate(0, 0), None);
        assert_eq!(delivery_rate(5, 0), Some(1.0));
    }
}
