//! The plain-text summary of a run: one fact per line, `name: value`, in a
//! fixed order, so a run can be pasted into documentation and two runs diffed.

use std::fmt::Write as _;

use crate::args::Args;
use crate::run::Outcome;
use crate::stats::{histogram, summarize};

/// Where the run happened.
pub struct Environment {
    /// The adapter's name.
    pub adapter: String,
    /// The backend wgpu chose, such as `Metal` or `Vulkan`.
    pub backend: String,
}

/// Renders the summary of `outcome`.
pub fn render(env: &Environment, args: &Args, outcome: &Outcome) -> String {
    let mut out = String::new();
    // Writing to a String cannot fail.
    let _ = write_report(&mut out, env, args, outcome);
    out
}

fn write_report(
    out: &mut String,
    env: &Environment,
    args: &Args,
    outcome: &Outcome,
) -> std::fmt::Result {
    writeln!(out, "adapter: {}", env.adapter)?;
    writeln!(out, "backend: {}", env.backend)?;
    writeln!(
        out,
        "config: frames={} latency={} slices={} slice_iters={} render_iters={} margin_us={}",
        args.frames, args.latency, args.slices, args.slice_iters, args.render_iters, args.margin_us
    )?;
    writeln!(out, "frames: {}", outcome.cpu_us.len())?;
    writeln!(out, "frames_dropped: {}", outcome.dropped)?;
    writeln!(out, "delivered: {}", outcome.delivered)?;
    writeln!(out, "missed: {}", outcome.missed)?;
    writeln!(out, "pending_at_end: {}", outcome.pending_at_end)?;
    writeln!(out, "latency_violations: {}", outcome.violations)?;
    writeln!(out, "timings_collected: {}", outcome.timer.frames_collected)?;
    writeln!(out, "timings_skipped: {}", outcome.timer.frames_skipped)?;
    writeln!(out, "timings_failed: {}", outcome.timer.frames_failed)?;
    for (frames, count) in histogram(outcome.latencies.iter().copied()) {
        writeln!(out, "latency_frames[{frames}]: {count}")?;
    }
    write_summary(out, "frame_start_lateness_us", &outcome.lateness_us)?;
    write_summary(out, "cpu_start_to_submit_us", &outcome.cpu_us)?;
    write_summary(out, "gpu_frame_us", &outcome.gpu_frame_us)?;
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
    Ok(())
}

fn write_summary(out: &mut String, name: &str, samples: &[u64]) -> std::fmt::Result {
    match summarize(samples) {
        Some(s) => writeln!(out, "{name}: p50={} p99={} max={}", s.p50, s.p99, s.max),
        None => writeln!(out, "{name}: none"),
    }
}
