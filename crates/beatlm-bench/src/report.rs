//! The plain-text summary of a run: one fact per line, `name: value`, in a
//! fixed order, so a run can be pasted into documentation and two runs diffed.

use std::fmt::Write as _;

use beatlm_host::report;

use crate::args::{Args, WorkloadKind, fusion_name};
use crate::engine::{NetReport, Outcome};
use crate::pace::FRAME_PERIOD;
use crate::pass_bench::PassOutcome;

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

/// Renders run number `run` (counting from 1) of `args.runs` of the
/// `pass-cost` workload: the lines every report starts with, then the
/// workload's own (`beatlm_host::pass_cost` defines them).
pub fn render_pass_cost(env: &Environment, args: &Args, run: u32, outcome: &PassOutcome) -> String {
    let mut out = String::new();
    // Writing to a String cannot fail.
    let _ = write_pass_cost(&mut out, env, args, run, outcome);
    out
}

fn write_pass_cost(
    out: &mut String,
    env: &Environment,
    args: &Args,
    run: u32,
    outcome: &PassOutcome,
) -> std::fmt::Result {
    writeln!(out, "run: {run}/{}", args.runs)?;
    writeln!(out, "adapter: {}", env.adapter)?;
    writeln!(out, "backend: {}", env.backend)?;
    write!(
        out,
        "config: workload={} runs={} frames={} render_iters={}",
        args.workload.name(),
        args.runs,
        args.frames,
        args.render_iters
    )?;
    if matches!(
        args.pass_cost.dispatches,
        beatlm_host::pass_cost::Dispatches::Depth { .. }
    ) {
        write!(
            out,
            " size={} fusion={}",
            args.size,
            fusion_name(args.fusion)
        )?;
    }
    writeln!(out)?;
    writeln!(out, "frames: {}", outcome.frames)?;
    writeln!(out, "frames_dropped: {}", outcome.dropped)?;
    report::write_timings(out, outcome.timer)?;
    outcome.measurements.write_report(out, &args.pass_cost)
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
        "config: mode={} runs={} frames={} latency={} slices={} slice_iters={} render_iters={} render_order={} margin_us={} estimator={} submit={} background_load={}",
        args.mode.name(),
        args.runs,
        args.frames,
        args.latency,
        args.slices,
        args.slice_iters,
        args.render_iters,
        args.render_order.name(),
        args.margin_us,
        args.estimator.name(),
        args.submit.name(),
        if args.background_load { "on" } else { "off" }
    )?;
    if args.background_load {
        write!(out, " background_iters={}", args.background_iters)?;
    }
    // Only a planned program adds a workload key to the line; the synthetic
    // workload is the one without it.
    if args.workload.is_planned() {
        write!(out, " workload={}", args.workload.name())?;
        if args.workload == WorkloadKind::Depth {
            write!(out, " size={}", args.size)?;
        }
        write!(out, " fusion={}", fusion_name(args.fusion))?;
        match args.max_dispatch_us {
            Some(us) => writeln!(out, " split=on max_dispatch_us={us}")?,
            None => writeln!(out, " split=off")?,
        }
    } else {
        writeln!(out)?;
    }
    // The lines the browser's report shares, at the loop's period
    // (`beatlm_host::report`); `channel_handoff_us` is the bench's own.
    report::write_lines(
        out,
        &report::Lines {
            latency: Some(args.latency),
            period: FRAME_PERIOD,
            dropped: outcome.dropped,
            submitted: outcome.submitted,
            delivered: outcome.delivered,
            missed: outcome.missed,
            startup_missed: outcome.startup_missed,
            pending_at_end: outcome.pending_at_end,
            violations: outcome.violations,
            timer: outcome.timer,
            latencies: &outcome.latencies,
            dispatches_per_job: &outcome.dispatches_per_job,
            dispatches_per_frame: &outcome.dispatches_per_frame,
            after_dispatches: &[],
            lateness_us: &outcome.lateness_us,
            cpu_us: &outcome.cpu_us,
            tick_us: &outcome.tick_us,
            after_tick: &[("channel_handoff_us", &outcome.handoff_us)],
            plan_us: &outcome.plan_us,
            encode_us: &outcome.encode_us,
            submit_us: &outcome.submit_us,
            gpu_frame_us: &outcome.gpu_frame_us,
            after_gpu_frame: &[],
            longest_dispatch_us: &outcome.longest_dispatch_us,
            pass_frames: &outcome.pass_frames,
            frame_ratios: &outcome.frame_ratios,
            frame_scale: outcome.frame_scale,
            frames_recorded: outcome.frames_recorded,
            frame_scales: &outcome.frame_scales,
        },
    )?;
    if let Some(dispatches) = outcome.background_dispatches {
        writeln!(out, "background_dispatches: {dispatches}")?;
    }
    report::write_slices(out, &outcome.estimates, &outcome.measured)?;
    if let Some(net) = &outcome.net {
        write_net(out, net)?;
    }
    Ok(())
}

/// The memory plan and the allocation readings of the `net` and `depth`
/// workloads.
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
    let calibrated: Option<u64> = net
        .kernel_parts
        .iter()
        .map(|k| k.calibrated.map(|us| us.get()))
        .sum();
    match calibrated {
        Some(us) => writeln!(out, "calibrated_job_us: {us}")?,
        None => writeln!(out, "calibrated_job_us: none")?,
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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn the_config_line_prints_the_render_order() {
        use beatlm_host::workload::RenderOrder;
        let env = Environment {
            adapter: String::from("adapter"),
            backend: String::from("backend"),
        };
        for order in RenderOrder::ALL {
            let args = Args {
                render_order: order,
                ..Args::default()
            };
            let text = render(&env, &args, 1, &Outcome::default());
            let config = text
                .lines()
                .find_map(|l| l.strip_prefix("config: "))
                .unwrap();
            let expected = format!(" render_iters=50 render_order={} margin_us=", order.name());
            assert!(config.contains(&expected), "{config}");
        }
    }

    /// A run with every line the synthetic workload writes, the
    /// `background_dispatches` line included, at fixed numbers.
    fn full_outcome() -> Outcome {
        use beatlm::gpu::{PassSpan, TimerStats};
        use beatlm_core::cost::{FrameScale, SliceId};
        use beatlm_core::time::Micros;
        use beatlm_host::pass_overlap::{Frame, Pass};
        let q = 65_536;
        let span = |begin_ns, end_ns| PassSpan { begin_ns, end_ns };
        Outcome {
            cpu_us: vec![900, 1_100, 1_000],
            tick_us: vec![1_200, 1_300, 1_250],
            handoff_us: Vec::new(),
            plan_us: vec![50, 60, 55],
            encode_us: vec![100, 110, 105],
            submit_us: vec![30, 40, 35],
            dispatches_per_frame: vec![4, 4, 3],
            dispatches_per_job: vec![4, 4],
            lateness_us: vec![0, 10, 500],
            dropped: 1,
            gpu_frame_us: vec![6_000, 17_000, 7_000],
            longest_dispatch_us: vec![1_500, 1_600],
            latencies: vec![2, 2, 3],
            submitted: 3,
            delivered: 2,
            missed: 1,
            startup_missed: 1,
            violations: 1,
            pending_at_end: 2,
            measured: [(SliceId::new(0), (3_000, 2))].into_iter().collect(),
            estimates: [
                (SliceId::new(0), Micros::new(1_600)),
                (SliceId::new(1), Micros::new(2_000)),
            ]
            .into_iter()
            .collect(),
            frame_ratios: vec![1.1, 0.9, 1.25],
            frame_scales: vec![1.0, 1.2],
            frame_scale: FrameScale::Measured(1.2),
            frames_recorded: 70,
            timer: TimerStats {
                frames_skipped: 1,
                frames_collected: 3,
                frames_failed: 0,
            },
            pass_frames: vec![Frame {
                index: 0,
                compute: vec![Pass {
                    slice: SliceId::new(0),
                    us: 655,
                    span: span(3 * q, 13 * q),
                }],
                render: Some(span(10 * q, 41 * q)),
            }],
            background_dispatches: Some(42),
            net: None,
        }
    }

    /// The report's names, order and formatting, held to the text in
    /// `report/`: a run with every value present, and an empty one where
    /// each value that can be absent says `none`.
    #[test]
    fn the_report_is_the_golden_text() {
        let env = Environment {
            adapter: String::from("Test Adapter"),
            backend: String::from("Metal"),
        };
        let args = Args {
            background_load: true,
            ..Args::default()
        };
        assert_eq!(
            render(&env, &args, 1, &full_outcome()),
            include_str!("report/full.txt")
        );
        assert_eq!(
            render(&env, &Args::default(), 1, &Outcome::default()),
            include_str!("report/empty.txt")
        );
    }
}
