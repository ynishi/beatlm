//! The report lines both hosts write, in one order: from `promised_latency_ms`
//! to `frame_scale`, then the `slice[i]` lines.
//!
//! A browser report compares with a native one line by line ([the crate
//! documentation](crate)) because these lines are written here once, from
//! numbers each host hands in ([`Lines`]): their names, their order and how
//! each value is printed. [`write_lines`] writes, in this order:
//!
//! - `promised_latency_ms`, `frames`, `frames_dropped`, `submitted`,
//!   `delivered`, `missed`, `startup_missed`, `delivery_rate`,
//!   `pending_at_end`, `latency_violations`, `timings_collected`,
//!   `timings_skipped`, `timings_failed` and `latency_frames[n]`;
//! - the summaries `dispatches_per_job`, `dispatches_per_frame`,
//!   `frame_start_lateness_us`, `cpu_start_to_submit_us`,
//!   `cpu_start_to_tick_end_us`, `cpu_plan_us`, `cpu_encode_us`,
//!   `cpu_queue_submit_us`, `gpu_frame_us` and `longest_dispatch_us`, with the
//!   host's own summaries at the three places [`Lines`] names;
//! - `gpu_frames_over_period`, the `pass_overlap_*` lines
//!   ([`crate::pass_overlap`]), `frame_ratio` and `frame_scale`
//!   ([`crate::stats::frame_ratio_value`], [`crate::stats::frame_scale_value`]).
//!
//! [`write_slices`] writes the `slice[i]` lines, and [`write_timings`] the
//! `timings_*` lines a `pass-cost` report has too. A host writes its own lines
//! around these, and between the two calls; what each shared line means is
//! the bench's documentation (`beatlm-bench`, "Reading the output"), and where
//! a browser's line keeps the name but not the meaning, the browser host's
//! `report` module says so.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use beatlm::gpu::TimerStats;
use beatlm_core::cost::{FrameScale, SliceId};
use beatlm_core::time::Micros;

use crate::pass_overlap;
use crate::stats::{frame_ratio_value, frame_scale_value, histogram, summarize};

/// A summary line of a host's own: its name and its samples, written as
/// [`write_summary`] writes them.
pub type HostSummary<'a> = (&'a str, &'a [u64]);

/// What the shared lines are written from. The per-frame slices hold one
/// sample per frame run unless a field says otherwise; each host's
/// documentation says how it measures them.
#[derive(Debug, Clone, Copy)]
pub struct Lines<'a> {
    /// The promised latency in frames; `None` for a run that promises none,
    /// whose `promised_latency_ms` and `latency_violations` say `none`.
    pub latency: Option<u32>,
    /// The frame period: what `promised_latency_ms` multiplies the frames by,
    /// and what `gpu_frames_over_period` holds a frame's GPU time to.
    pub period: Micros,
    /// Frame periods given up.
    pub dropped: u64,
    /// Inputs submitted as jobs.
    pub submitted: u64,
    /// Results delivered.
    pub delivered: u64,
    /// Deadlines that arrived without a complete result.
    pub missed: u64,
    /// The part of `missed` that belongs to the first frames' jobs.
    pub startup_missed: u64,
    /// Jobs still awaiting their deadline when the run ended.
    pub pending_at_end: usize,
    /// Deliveries whose latency was not the promised one; not written when
    /// `latency` is `None`.
    pub violations: u64,
    /// What the timer could not measure.
    pub timer: TimerStats,
    /// Latency in frames of each delivered result.
    pub latencies: &'a [u64],
    /// Compute dispatches encoded for each delivered job.
    pub dispatches_per_job: &'a [u64],
    /// Compute dispatches encoded in each frame.
    pub dispatches_per_frame: &'a [u64],
    /// The host's summaries after `dispatches_per_frame`.
    pub after_dispatches: &'a [HostSummary<'a>],
    /// How late each frame began, in microseconds.
    pub lateness_us: &'a [u64],
    /// From each frame's start to the return of `queue.submit`, in
    /// microseconds; `frames` is how many there are.
    pub cpu_us: &'a [u64],
    /// From each frame's start to the end of its work, in microseconds.
    pub tick_us: &'a [u64],
    /// The host's summaries after `cpu_start_to_tick_end_us`.
    pub after_tick: &'a [HostSummary<'a>],
    /// Deciding each frame, in microseconds.
    pub plan_us: &'a [u64],
    /// Recording each frame's command buffer, in microseconds.
    pub encode_us: &'a [u64],
    /// Inside `queue.submit`, in microseconds.
    pub submit_us: &'a [u64],
    /// Render plus every slice measured in the same frame, per timed frame,
    /// in microseconds.
    pub gpu_frame_us: &'a [u64],
    /// The host's summaries after `gpu_frame_us`.
    pub after_gpu_frame: &'a [HostSummary<'a>],
    /// The longest compute pass in each timed frame that ran one, in
    /// microseconds.
    pub longest_dispatch_us: &'a [u64],
    /// Every timed frame's passes with their timestamps.
    pub pass_frames: &'a [pass_overlap::Frame],
    /// The frame ratios [`frame_ratio_value`] reads.
    pub frame_ratios: &'a [f64],
    /// The frame scale when the run ended.
    pub frame_scale: FrameScale,
    /// Frames the cost model folded into its frame scale.
    pub frames_recorded: u64,
    /// The frame scale of each plan made with a measured one.
    pub frame_scales: &'a [f64],
}

/// Writes the shared lines from `promised_latency_ms` to `frame_scale`, in
/// the order [the module documentation](self) lists.
pub fn write_lines(out: &mut String, l: &Lines<'_>) -> std::fmt::Result {
    match l.latency {
        Some(frames) => writeln!(
            out,
            "promised_latency_ms: {:.1}",
            promised_latency_ms(frames, l.period)
        )?,
        None => writeln!(out, "promised_latency_ms: none")?,
    }
    writeln!(out, "frames: {}", l.cpu_us.len())?;
    writeln!(out, "frames_dropped: {}", l.dropped)?;
    writeln!(out, "submitted: {}", l.submitted)?;
    writeln!(out, "delivered: {}", l.delivered)?;
    writeln!(out, "missed: {}", l.missed)?;
    writeln!(out, "startup_missed: {}", l.startup_missed)?;
    match delivery_rate(l.delivered, l.missed) {
        Some(rate) => writeln!(out, "delivery_rate: {rate:.4}")?,
        None => writeln!(out, "delivery_rate: none")?,
    }
    writeln!(out, "pending_at_end: {}", l.pending_at_end)?;
    match l.latency {
        Some(_) => writeln!(out, "latency_violations: {}", l.violations)?,
        None => writeln!(out, "latency_violations: none")?,
    }
    write_timings(out, l.timer)?;
    for (frames, count) in histogram(l.latencies.iter().copied()) {
        writeln!(out, "latency_frames[{frames}]: {count}")?;
    }
    let host = |out: &mut String, summaries: &[HostSummary<'_>]| {
        summaries
            .iter()
            .try_for_each(|&(name, samples)| write_summary(out, name, samples))
    };
    write_summary(out, "dispatches_per_job", l.dispatches_per_job)?;
    write_summary(out, "dispatches_per_frame", l.dispatches_per_frame)?;
    host(out, l.after_dispatches)?;
    write_summary(out, "frame_start_lateness_us", l.lateness_us)?;
    write_summary(out, "cpu_start_to_submit_us", l.cpu_us)?;
    write_summary(out, "cpu_start_to_tick_end_us", l.tick_us)?;
    host(out, l.after_tick)?;
    write_summary(out, "cpu_plan_us", l.plan_us)?;
    write_summary(out, "cpu_encode_us", l.encode_us)?;
    write_summary(out, "cpu_queue_submit_us", l.submit_us)?;
    write_summary(out, "gpu_frame_us", l.gpu_frame_us)?;
    host(out, l.after_gpu_frame)?;
    write_summary(out, "longest_dispatch_us", l.longest_dispatch_us)?;
    writeln!(
        out,
        "gpu_frames_over_period: {} of {}",
        gpu_frames_over_period(l.gpu_frame_us, l.period),
        l.gpu_frame_us.len()
    )?;
    pass_overlap::write_lines(out, &pass_overlap::analyse(l.pass_frames))?;
    writeln!(out, "frame_ratio: {}", frame_ratio_value(l.frame_ratios))?;
    writeln!(
        out,
        "frame_scale: {}",
        frame_scale_value(l.frame_scale, l.frames_recorded, l.frame_scales)
    )
}

/// Writes `timings_collected`, `timings_skipped` and `timings_failed` from
/// `timer`: among the lines [`write_lines`] writes, and the ones both hosts'
/// `pass-cost` reports have too.
pub fn write_timings(out: &mut String, timer: TimerStats) -> std::fmt::Result {
    writeln!(out, "timings_collected: {}", timer.frames_collected)?;
    writeln!(out, "timings_skipped: {}", timer.frames_skipped)?;
    writeln!(out, "timings_failed: {}", timer.frames_failed)
}

/// Writes a `slice[i]` line per slice with an estimate: the cost model's
/// final estimate against the mean of the slice's measurements (`measured`
/// holds their sum and count), `none` when it has none.
pub fn write_slices(
    out: &mut String,
    estimates: &BTreeMap<SliceId, Micros>,
    measured: &BTreeMap<SliceId, (u64, u64)>,
) -> std::fmt::Result {
    for (slice, estimate) in estimates {
        let (sum, count) = measured.get(slice).copied().unwrap_or((0, 0));
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
    Ok(())
}

/// Writes `name: p50=… p99=… max=…` of `samples` ([`summarize`]), or
/// `name: none` when there are none: the form of every summary line of both
/// reports.
pub fn write_summary(out: &mut String, name: &str, samples: &[u64]) -> std::fmt::Result {
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

/// The promised latency of `frames` frames of `period`, in milliseconds.
#[allow(clippy::cast_precision_loss)]
fn promised_latency_ms(frames: u32, period: Micros) -> f64 {
    f64::from(frames) * period.get() as f64 / 1_000.0
}

/// Timed frames whose render plus slices took longer than `period`: an
/// overrun the estimates were supposed to prevent.
fn gpu_frames_over_period(gpu_frame_us: &[u64], period: Micros) -> usize {
    gpu_frame_us.iter().filter(|&&us| us > period.get()).count()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn delivery_rate_is_the_delivered_share_of_the_deadlines_that_arrived() {
        assert_eq!(delivery_rate(0, 0), None);
        assert_eq!(delivery_rate(3, 1), Some(0.75));
        assert_eq!(delivery_rate(0, 5), Some(0.0));
        assert_eq!(delivery_rate(5, 0), Some(1.0));
    }

    #[test]
    fn the_promised_latency_is_the_frames_at_the_period() {
        let period = Micros::new(16_667);
        assert_eq!(promised_latency_ms(0, period), 0.0);
        assert!((promised_latency_ms(5, period) - 83.335).abs() < 1e-9);
        assert!((promised_latency_ms(2, Micros::new(8_333)) - 16.666).abs() < 1e-9);
    }

    #[test]
    fn a_frame_is_over_only_when_it_exceeds_the_period() {
        let period = Micros::new(16_667);
        let p = period.get();
        assert_eq!(gpu_frames_over_period(&[], period), 0);
        let over = [p - 1, p, p + 1, p * 2];
        assert_eq!(gpu_frames_over_period(&over, period), 2);
        assert_eq!(gpu_frames_over_period(&over, Micros::new(8_333)), 4);
    }

    #[test]
    fn a_run_without_a_latency_in_frames_says_none_for_its_lines() {
        let lines = Lines {
            latency: None,
            period: Micros::new(16_667),
            dropped: 0,
            submitted: 0,
            delivered: 0,
            missed: 0,
            startup_missed: 0,
            pending_at_end: 0,
            violations: 3,
            timer: TimerStats::default(),
            latencies: &[],
            dispatches_per_job: &[],
            dispatches_per_frame: &[],
            after_dispatches: &[("host_us", &[5])],
            lateness_us: &[],
            cpu_us: &[],
            tick_us: &[],
            after_tick: &[],
            plan_us: &[],
            encode_us: &[],
            submit_us: &[],
            gpu_frame_us: &[],
            after_gpu_frame: &[],
            longest_dispatch_us: &[],
            pass_frames: &[],
            frame_ratios: &[],
            frame_scale: FrameScale::None,
            frames_recorded: 0,
            frame_scales: &[],
        };
        let mut out = String::new();
        write_lines(&mut out, &lines).unwrap();
        let names: Vec<&str> = out.lines().collect();
        assert_eq!(names[0], "promised_latency_ms: none");
        assert!(names.contains(&"latency_violations: none"));
        let at = names
            .iter()
            .position(|l| l.starts_with("host_us:"))
            .unwrap();
        assert_eq!(names[at - 1], "dispatches_per_frame: none");
        assert_eq!(names[at], "host_us: p50=5 p99=5 max=5");
        assert_eq!(names.last(), Some(&"frame_scale: none"));
    }
}
