//! The plain-text summary of a run: one fact per line, `name: value`, in a
//! fixed order, so a run can be pasted into documentation and two runs diffed.

use std::fmt::Write as _;

use crate::args::{Args, WorkloadKind, fusion_name};
use crate::engine::{NetReport, Outcome};
use crate::pace::FRAME_PERIOD;
use crate::stats::{histogram, summarize};

/// Where the run happened.
pub struct Environment {
    /// The adapter's name.
    pub adapter: String,
    /// The backend wgpu chose, such as `Metal` or `Vulkan`.
    pub backend: String,
}

/// Renders the summary of run number `run` (counting from 1) of `args.runs`.
pub fn render(env: &Environment, args: &Args, run: u32, outcome: &Outcome) -> String {
    let mut out = String::new();
    // Writing to a String cannot fail.
    let _ = write_report(&mut out, env, args, run, outcome);
    out
}

fn write_report(
    out: &mut String,
    env: &Environment,
    args: &Args,
    run: u32,
    outcome: &Outcome,
) -> std::fmt::Result {
    writeln!(out, "run: {run}/{}", args.runs)?;
    writeln!(out, "adapter: {}", env.adapter)?;
    writeln!(out, "backend: {}", env.backend)?;
    write!(
        out,
        "config: mode={} runs={} frames={} latency={} slices={} slice_iters={} render_iters={} margin_us={} estimator={} background_load={}",
        args.mode.name(),
        args.runs,
        args.frames,
        args.latency,
        args.slices,
        args.slice_iters,
        args.render_iters,
        args.margin_us,
        args.estimator.name(),
        if args.background_load { "on" } else { "off" }
    )?;
    if args.background_load {
        write!(out, " background_iters={}", args.background_iters)?;
    }
    // Only the network adds a workload key to the line; the synthetic workload
    // is the one without it.
    match args.workload {
        WorkloadKind::Synthetic => writeln!(out)?,
        WorkloadKind::Net => {
            write!(out, " workload=net fusion={}", fusion_name(args.fusion))?;
            match args.max_dispatch_us {
                Some(us) => writeln!(out, " split=on max_dispatch_us={us}")?,
                None => writeln!(out, " split=off")?,
            }
        }
    }
    writeln!(out, "frames: {}", outcome.cpu_us.len())?;
    writeln!(out, "frames_dropped: {}", outcome.dropped)?;
    writeln!(out, "delivered: {}", outcome.delivered)?;
    writeln!(out, "missed: {}", outcome.missed)?;
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
    write_summary(out, "channel_handoff_us", &outcome.handoff_us)?;
    write_summary(out, "cpu_plan_us", &outcome.plan_us)?;
    write_summary(out, "cpu_encode_us", &outcome.encode_us)?;
    write_summary(out, "cpu_queue_submit_us", &outcome.submit_us)?;
    write_summary(out, "gpu_frame_us", &outcome.gpu_frame_us)?;
    write_summary(out, "longest_dispatch_us", &outcome.longest_dispatch_us)?;
    let over = gpu_frames_over_period(&outcome.gpu_frame_us);
    writeln!(
        out,
        "gpu_frames_over_period: {over} of {}",
        outcome.gpu_frame_us.len()
    )?;
    if let Some(dispatches) = outcome.background_dispatches {
        writeln!(out, "background_dispatches: {dispatches}")?;
    }
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
    if let Some(net) = &outcome.net {
        write_net(out, net)?;
    }
    Ok(())
}

/// Delivered over delivered plus missed: the share of deadlines that arrived
/// with their result. `None` when no deadline arrived.
#[allow(clippy::cast_precision_loss)]
fn delivery_rate(delivered: u64, missed: u64) -> Option<f64> {
    let total = delivered.saturating_add(missed);
    (total > 0).then(|| delivered as f64 / total as f64)
}

/// Timed frames whose render plus slices took longer than the frame period: an
/// overrun the estimates were supposed to prevent.
fn gpu_frames_over_period(gpu_frame_us: &[u64]) -> usize {
    gpu_frame_us
        .iter()
        .filter(|&&us| us > FRAME_PERIOD.get())
        .count()
}

/// The memory plan and the allocation readings of the `net` workload.
fn write_net(out: &mut String, net: &NetReport) -> std::fmt::Result {
    for (index, kernel) in net.kernel_parts.iter().enumerate() {
        write!(
            out,
            "kernel_parts[{index}]: name={} parts={}",
            kernel.name, kernel.parts
        )?;
        match kernel.calibrated {
            Some(us) => writeln!(out, " calibrated_us={}", us.get())?,
            None => writeln!(out, " calibrated_us=none")?,
        }
    }
    writeln!(out, "kernels: {}", net.kernels.join(" "))?;
    writeln!(out, "memory_plan_tensors: {}", net.tensors)?;
    writeln!(out, "memory_plan_io_slots: {}", net.slots)?;
    writeln!(out, "memory_plan_arenas: {}", net.arenas.len())?;
    for (index, bytes) in net.arenas.iter().enumerate() {
        writeln!(out, "memory_plan_arena[{index}]_bytes: {bytes}")?;
    }
    writeln!(out, "memory_plan_total_bytes: {}", net.total_bytes)?;
    writeln!(out, "memory_plan_peak_live_bytes: {}", net.peak_live_bytes)?;
    writeln!(out, "memory_plan_unshared_bytes: {}", net.unshared_bytes)?;
    let c = &net.counters;
    writeln!(out, "wgpu_counters_populated: {}", c.populated)?;
    if c.populated {
        let delta = |before: isize, after: isize| after - before;
        writeln!(
            out,
            "loop_buffers_live: before={} after={} delta={} min={} max={}",
            c.before.buffers,
            c.after.buffers,
            delta(c.before.buffers, c.after.buffers),
            c.min_buffers,
            c.max_buffers
        )?;
        let others = [
            (
                "buffer_memory_bytes",
                c.before.buffer_memory,
                c.after.buffer_memory,
            ),
            (
                "memory_allocations",
                c.before.memory_allocations,
                c.after.memory_allocations,
            ),
            ("textures", c.before.textures, c.after.textures),
            ("bind_groups", c.before.bind_groups, c.after.bind_groups),
            (
                "compute_pipelines",
                c.before.compute_pipelines,
                c.after.compute_pipelines,
            ),
        ];
        for (name, before, after) in others {
            writeln!(
                out,
                "loop_{name}: before={before} after={after} delta={}",
                delta(before, after)
            )?;
        }
    } else {
        writeln!(
            out,
            "loop_buffers_created: unmeasured; this backend sets no wgpu counters. The loop \
             code calls no buffer-creating function, but creations inside wgpu are not observed."
        )?;
    }
    Ok(())
}

fn write_summary(out: &mut String, name: &str, samples: &[u64]) -> std::fmt::Result {
    match summarize(samples) {
        Some(s) => writeln!(out, "{name}: p50={} p99={} max={}", s.p50, s.p99, s.max),
        None => writeln!(out, "{name}: none"),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
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
    fn a_frame_is_over_only_when_it_exceeds_the_period() {
        let period = FRAME_PERIOD.get();
        assert_eq!(gpu_frames_over_period(&[]), 0);
        assert_eq!(
            gpu_frames_over_period(&[period - 1, period, period + 1, period * 2]),
            2
        );
    }
}
