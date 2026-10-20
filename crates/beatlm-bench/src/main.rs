//! Headless benchmark for beatlm: a synthetic GPU load driven through the frame
//! loop, with GPU and CPU time measured.
//!
//! # What it measures
//!
//! It stands in for a host application at 60 Hz (a 16 667 microsecond frame)
//! and runs the umbrella crate's side of a frame against a fake workload, to
//! answer the question the concurrency model in the `beatlm` crate documentation
//! leaves open: how long from frame start to submission, and how much that
//! varies. This binary is **candidate 1**, a synchronous `tick` on the calling
//! thread.
//!
//! Each frame it submits one job of `--slices` slices with an L1 promise of
//! `--latency` frames, plans the frame against a budget of the period minus the
//! last measured render time minus `--margin-us`, encodes one timed compute pass
//! per slice the plan hands out and one timed render pass, submits, and starts
//! the timestamp readback without waiting for it. Measurements that have arrived
//! by a later frame feed the cost model.
//!
//! The load is two kernels in `shaders.wgsl`. A slice is a compute dispatch of
//! 65 536 invocations, each running `--slice-iters` rounds of an integer hash.
//! The render pass draws one full-screen triangle into an offscreen 1280x720
//! `Rgba8Unorm` texture whose fragment shader runs `--render-iters` rounds of a
//! float hash. The defaults are tuned by measurement so that a slice and the
//! render pass each take about one to three milliseconds on an Apple Silicon
//! GPU: on an Apple M1 Pro (Metal) a slice measured between 0.8 and 1.4 ms and
//! the render pass about 2 ms. On other adapters they are starting points, and
//! the options exist for that. Pass durations on a tile-based GPU are measured
//! between the pass's boundaries, which can include time the pass waited for
//! earlier work on the queue, so a very short slice following a heavy render
//! pass measures longer than it ran.
//!
//! # Running it
//!
//! ```text
//! cargo run --release -p beatlm-bench -- --frames 300
//! ```
//!
//! Options are listed by `--help`. It needs an adapter with
//! `Features::TIMESTAMP_QUERY` and is not part of `just check`; an adapter is
//! not guaranteed where `check` runs. A run takes about `frames / 60` seconds.
//!
//! # Reading the output
//!
//! The report is plain text, one fact per line, on standard output.
//!
//! | Line | Meaning |
//! |---|---|
//! | `adapter`, `backend` | what the numbers were measured on; quote them with any number |
//! | `config` | the options the run used |
//! | `frames` | frames run |
//! | `frames_dropped` | frame periods given up because a frame began a whole period late; the loop drops them as a display would, rather than running frames back to back to catch up |
//! | `delivered`, `missed` | results delivered on their frame, and deadlines that arrived without a complete result |
//! | `pending_at_end` | jobs submitted whose deadline was after the last frame |
//! | `latency_violations` | deliveries whose latency was not `--latency`; the L1 promise broke when this is not zero |
//! | `timings_collected`, `timings_skipped`, `timings_failed` | frames whose GPU timing was read, was not taken because the readback ring was full, and was lost to a failed map |
//! | `latency_frames[n]` | how many deliveries had a latency of `n` frames |
//! | `frame_start_lateness_us` | per frame, how long after its scheduled start the loop began it: p50, p99, max |
//! | `cpu_start_to_submit_us` | per frame, from the loop beginning the frame to the return of `queue.submit`: p50, p99, max |
//! | `gpu_frame_us` | per timed frame, render plus every slice, on the GPU: p50, p99, max |
//! | `slice[i]` | the cost model's final estimate for slice `i` against the mean of its measured durations |
//!
//! The process exits non-zero when `latency_violations` is not zero. A missed
//! deadline is reported but does not fail the run: it is the documented outcome
//! of a budget that is too small, not a broken promise.
//!
//! `cpu_start_to_submit_us` starts when the loop begins the frame, so it is the
//! engine's own cost; a frame that begins late, because the one before it
//! overran or the sleep overshot, shows that in `frame_start_lateness_us`
//! instead. A slice's estimate is what the scheduler budgets with, so it sits
//! above the measured mean by the cost model's safety factor.

mod args;
mod device;
mod report;
mod run;
mod stats;
mod workload;

use std::process::ExitCode;

use anyhow::Result;

fn main() -> ExitCode {
    match real_main() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::from(2)
        }
    }
}

fn real_main() -> Result<ExitCode> {
    let args = match args::parse(std::env::args().skip(1))? {
        args::Command::Help => {
            println!("{}", args::USAGE);
            return Ok(ExitCode::SUCCESS);
        }
        args::Command::Run(args) => args,
    };

    let gpu = device::Gpu::open()?;
    eprintln!(
        "running {} frames on {} ({})",
        args.frames, gpu.info.name, gpu.info.backend
    );
    let outcome = run::run(&gpu, &args)?;
    let env = report::Environment {
        adapter: gpu.info.name.clone(),
        backend: gpu.info.backend.to_string(),
    };
    print!("{}", report::render(&env, &args, &outcome));

    Ok(if outcome.violations == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}
