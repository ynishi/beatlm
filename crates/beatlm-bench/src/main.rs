//! Headless benchmark for beatlm: a synthetic GPU load driven through the frame
//! loop, with GPU and CPU time measured.
//!
//! # What it measures
//!
//! It stands in for a host application at 60 Hz (a 16 667 microsecond frame)
//! and runs the umbrella crate's side of a frame against a fake workload, to
//! answer the question the concurrency model in the `beatlm` crate documentation
//! leaves open: how long from frame start to submission, and how much that
//! varies. `--mode` picks which of the three candidates drives the frame:
//!
//! | `--mode` | Candidate | Who decides, encodes and submits | How the frame loop calls it |
//! |---|---|---|---|
//! | `sync` (default) | 1 | the frame-loop thread | a plain call to `Engine::tick_sync` |
//! | `thread` | 2 | a dedicated GPU thread: it builds and uses the workload, timer, scheduler and cost model, and is the only thread that touches the device during a run | the loop sends a frame event `(index, start)` over a bounded `std::sync::mpsc` channel |
//! | `async` | 3 | the frame-loop thread | `pollster::block_on(engine.tick_async(..))`, once per frame |
//!
//! Everything else is the same code in all three: the pacing (`pace.rs`), the
//! frame's work (`engine.rs`), the scheduler use, the latency check and the
//! report. The only difference is where and how the frame's GPU work is decided,
//! encoded and submitted.
//!
//! The cost model estimates each slice with the statistic `--estimator` names,
//! and `--background-load` adds a competing GPU load; both are described below.
//!
//! Each frame it submits one job of `--slices` slices with an L1 promise of
//! `--latency` frames, plans the frame against a budget of the period minus the
//! last measured render time minus `--margin-us`, encodes one timed compute pass
//! per slice the plan hands out and one timed render pass, submits, and starts
//! the timestamp readback without waiting for it. Measurements that have arrived
//! by a later frame feed the cost model.
//!
//! In `thread` mode the clock for "frame start to submit" starts on the
//! frame-loop thread when the pacer begins the frame and stops on the GPU thread
//! when `queue.submit` returns; both read `std::time::Instant`, which is one
//! monotonic clock for the process. The time from frame start to the GPU thread
//! receiving the event is reported separately as `channel_handoff_us`. The
//! channel holds two events; a GPU thread further behind than that blocks the
//! sender, which shows as `frame_start_lateness_us`.
//!
//! In `async` mode the await sits where the readback is checked, after the
//! submit. The timer exposes completion as a flag its map callback sets, not as
//! a future, so the awaited step polls the device and collects what has
//! completed; on native `Device::poll` runs the finished callbacks before it
//! returns, so that future is ready on its first poll and the mode measures the
//! cost of an `async fn`, a hand-written future and `block_on` around work that
//! never waits. A tick that really waits for a map callback needs the timer to
//! hand out a future or register a waker, which is a change to `beatlm`.
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
//! # The `net` workload
//!
//! `--workload net` replaces the busy-loop slices with a small network written
//! in the kernel IR (`beatlm_core::kernel`): ten operations over a 128x128
//! image (two 3x3 convolutions, two layer norms and two matrix multiplies, with
//! a ReLU, a GELU, a residual add and a scale between them), lowered to kernels
//! whose WGSL `beatlm::codegen` generates and naga validates. Each kernel is one
//! slice of the frame loop. `--fusion` (default `on`) decides whether
//! element-wise operations are fused into the kernel before them: six kernels
//! with fusion, ten without. `--dump-wgsl <dir>` writes each kernel's text
//! before the run. `net.rs` documents the network, its layouts and what it
//! simplifies; `reference.rs` holds a CPU reference for each operation and an
//! interpreter over the graph, and `cargo test -p beatlm-bench -- --ignored`
//! compares every generated kernel, with and without each epilogue, and the
//! whole network in planned memory, fused and unfused, against it. The
//! scheduler, the timer and the frame loop are the ones the synthetic workload
//! uses, and the job has one slice per kernel, so `--slices` is fixed at the
//! kernel count and `--slice-iters` does not apply. The render pass is the
//! synthetic one.
//!
//! It measures the rule that the steady-state loop allocates nothing. The
//! network's memory is planned against the baseline WebGPU limits, realised as
//! one buffer per arena before the first frame, and the report prints the plan
//! (`memory_plan_*`) next to wgpu's own object counters read before the first
//! frame and after the last (`loop_*`, from `counters.rs`, which also says what
//! a counter does and does not show). On an Apple M1 Pro (Metal), release
//! build, `--frames 600 --runs 2` at the defaults, on 2026-10-21, with fusion
//! on and off:
//!
//! - with fusion the plan has 23 tensors in one arena of 21 785 120 bytes,
//!   576 bytes above the most bytes live at one kernel
//!   (`memory_plan_peak_live_bytes`) and below the 24 405 984 bytes the tensors
//!   need unshared; without, 27 tensors in 25 979 424 bytes, against 38 037 472
//!   unshared. The four tensors fusion removes are the activations it keeps in
//!   registers. Each count includes an input and an output for each of the 3
//!   slots (`memory_plan_io_slots`, `--latency` plus one: one slot per job that
//!   can be in flight, so a job's output survives until its delivery frame);
//! - the live buffer count was 10 before the first frame, 10 after the last,
//!   and 10 at every frame boundary in between, in all four runs; textures (1),
//!   bind groups (13 fused, 17 unfused) and compute pipelines (9 fused, 13
//!   unfused) did not change either, the per-slot bind groups included,
//!   because all are created before the first frame;
//! - the mean measured duration of the six fused kernels was 0.86 to 0.90,
//!   0.90 to 1.02, 0.79 to 0.86, 0.76 to 0.88, 0.65 to 0.70 and 2.68 to 2.80 ms
//!   over the two runs. The last is the second matrix multiply, which is four
//!   times the work of the network's earlier version (it produces 64 channels,
//!   not 16, so the residual add has an operand); without fusion it measured
//!   2.81 to 3.12 ms and the four standalone element-wise kernels 0.04 to 0.41
//!   ms each.
//!
//! The shape was tuned for an earlier, six-operation version of the network
//! with hand-written kernels, by an in-loop figure rather than a dispatch
//! timed alone: at 256x256 each operation took 0.85 to 1.35 ms repeated back to
//! back on an idle queue, while at 128x128, a quarter of the pixels, they
//! measured 0.8 to 1.4 ms in the loop. A job whose summed estimates do not fit
//! a frame's budget misses its deadline; jobs run in submission order and never
//! interleave, so a job the budget cannot finish is missed rather than
//! overtaken.
//!
//! `loop_buffer_memory_bytes` and `loop_memory_allocations` read zero on this
//! backend before and after: Metal does not set those two counters, so their
//! zeros show nothing, and only the object counts are evidence. The control for
//! the buffer count is a test that creates a buffer and sees the count move.
//! Counts are live counts: a buffer created and dropped between two frame
//! boundaries is not seen.
//!
//! # Fusion
//!
//! Fusion exists to issue fewer dispatches (the `beatlm_core::kernel`
//! documentation says why, and owns the rule). The runs above, `sync` mode,
//! latency 2, the quantile estimator; times in microseconds:
//!
//! | `--fusion`, run | `dispatches_per_job` | `dispatches_per_frame` p50 / p99 | `cpu_encode_us` p50 / p99 | `cpu_queue_submit_us` p50 / p99 | `cpu_start_to_submit_us` p50 / p99 | `gpu_frame_us` p50 / p99 | delivered / missed | `delivery_rate` |
//! |---|---|---|---|---|---|---|---|---|
//! | `on`, 1 | 6 | 6 / 8 | 595 / 5893 | 153 / 3067 | 782 / 6721 | 8748 / 15548 | 526 / 72 | 0.8796 |
//! | `on`, 2 | 6 | 6 / 8 | 492 / 2292 | 142 / 1184 | 664 / 2875 | 8498 / 15468 | 511 / 87 | 0.8545 |
//! | `off`, 1 | 10 | 10 / 13 | 630 / 3240 | 165 / 995 | 856 / 4192 | 9152 / 15926 | 598 / 0 | 1.0000 |
//! | `off`, 2 | 10 | 10 / 13 | 719 / 5440 | 173 / 2252 | 992 / 6167 | 8519 / 15737 | 403 / 195 | 0.6739 |
//!
//! Every delivered job had exactly its kernel count of dispatches, and no run
//! had a latency violation. The median CPU figures were lower in both fused
//! runs than in either unfused one. Everything at the 99th percentile, and the
//! delivery rate, spread more between two runs of one setting than between
//! the settings, so two runs each show no effect of fusion there. A miss is a
//! job whose slices' estimates did not fit the budgets of the frames before its
//! deadline, and the largest final estimate in every run was the second matrix
//! multiply's, fused or not.
//!
//! # Running it
//!
//! ```text
//! cargo run --release -p beatlm-bench -- --frames 300
//! cargo run --release -p beatlm-bench -- --frames 600 --runs 3 --mode thread
//! cargo run --release -p beatlm-bench -- --frames 600 --runs 2 --estimator capped --background-load
//! cargo run --release -p beatlm-bench -- --workload net --frames 600 --runs 2 --fusion off
//! cargo run --release -p beatlm-bench -- --workload net --frames 1 --dump-wgsl target/wgsl
//! ```
//!
//! Options are listed by `--help`. It needs an adapter with
//! `Features::TIMESTAMP_QUERY` and is not part of `just check`; an adapter is
//! not guaranteed where `check` runs. A run takes about `frames / 60` seconds.
//! `--runs N` repeats the whole run `N` times in one process on one device, each
//! with a fresh scheduler, cost model, timer and workload, and prints one report
//! per run, so the variance between runs is visible without restarting.
//!
//! # Reading the output
//!
//! The report is plain text, one fact per line, on standard output; with
//! `--runs` the reports follow one another, each starting with its `run` line.
//!
//! | Line | Meaning |
//! |---|---|
//! | `run` | which run of `--runs` this is |
//! | `adapter`, `backend` | what the numbers were measured on; quote them with any number |
//! | `config` | the options the run used, `mode`, `estimator` and `background_load` included (and `background_iters` when the load is on) |
//! | `frames` | frames run |
//! | `frames_dropped` | frame periods given up because a frame began a whole period late; the loop drops them as a display would, rather than running frames back to back to catch up |
//! | `delivered`, `missed` | results delivered on their frame, and deadlines that arrived without a complete result |
//! | `delivery_rate` | `delivered` over `delivered` plus `missed`: the share of deadlines that arrived with their result; `none` when no deadline arrived |
//! | `pending_at_end` | jobs submitted whose deadline was after the last frame |
//! | `latency_violations` | deliveries whose latency was not `--latency`; the L1 promise broke when this is not zero |
//! | `timings_collected`, `timings_skipped`, `timings_failed` | frames whose GPU timing was read, was not taken because the readback ring was full, and was lost to a failed map |
//! | `latency_frames[n]` | how many deliveries had a latency of `n` frames |
//! | `dispatches_per_job` | compute dispatches encoded for each delivered job, over every frame that ran one of its slices: p50, p99, max |
//! | `dispatches_per_frame` | compute dispatches encoded in each frame, every job's slices included: p50, p99, max |
//! | `frame_start_lateness_us` | per frame, how long after its scheduled start the loop began it: p50, p99, max |
//! | `cpu_start_to_submit_us` | per frame, from the loop beginning the frame to the return of `queue.submit`: p50, p99, max |
//! | `cpu_start_to_tick_end_us` | per frame, from the loop beginning the frame to the return of the driver's call for it, readback poll, deliveries and in `async` mode `block_on` included |
//! | `channel_handoff_us` | per frame, from the loop beginning the frame to the GPU thread receiving its event; `none` unless `--mode thread` |
//! | `cpu_plan_us` | per frame, `Scheduler::submit` plus `plan_frame` |
//! | `cpu_encode_us` | per frame, from creating the command encoder to finishing it, timestamp writes and the resolve included |
//! | `cpu_queue_submit_us` | per frame, the time inside `queue.submit` |
//! | `gpu_frame_us` | per timed frame, render plus every slice, on the GPU: p50, p99, max |
//! | `gpu_frames_over_period` | timed frames whose render plus slices took longer than the frame period, of the timed frames: a real overrun that the estimates were supposed to prevent |
//! | `background_dispatches` | dispatches the competing load submitted, from the one completed before the first frame to the last one queued when it was stopped; present only with `--background-load` |
//! | `slice[i]` | the cost model's final estimate for slice `i` against the mean of its measured durations |
//! | `workload`, `fusion` (in `config`) | present as `workload=net fusion=on` or `fusion=off` only for the network; the synthetic report has neither |
//! | `kernels` | `net` only: each kernel of a job, in dispatch order, named by its main operation and epilogue |
//! | `memory_plan_*` | `net` only: tensors, input/output slots, arenas and their bytes, the plan's total, the most bytes live at one kernel, and the bytes with no sharing |
//! | `wgpu_counters_populated` | `net` only: whether the backend sets wgpu's buffer counter; when false the loop's buffer creation is unmeasured and the report says so |
//! | `loop_*` | `net` only: a counter's reading before the first frame and after the last, and the change; for buffers also the least and most seen at a frame boundary |
//!
//! The process exits non-zero when `latency_violations` is not zero in any run.
//! A missed deadline is reported but does not fail the run: it is the documented
//! outcome of a budget that is too small, not a broken promise.
//!
//! `cpu_start_to_submit_us` starts when the loop begins the frame, so it is the
//! engine's own cost plus, in `thread` mode, the hand-off; a frame that begins
//! late, because the one before it overran or the sleep overshot, shows that in
//! `frame_start_lateness_us` instead. The three `cpu_` breakdown lines are
//! measured with one `Instant` read at each boundary, in every mode, and answer
//! whether the 99th-percentile variance of `cpu_start_to_submit_us` comes from
//! this code (`cpu_plan_us`, `cpu_encode_us`) or from the driver and the OS
//! (`cpu_queue_submit_us`, and `channel_handoff_us` in `thread` mode). A slice's
//! estimate is what the scheduler budgets with, so how it compares with the
//! measured mean depends on `--estimator`: the mean-plus-sigmas estimate sits
//! above the mean by its safety factor, and a quantile of the recent
//! measurements can sit on either side of it.
//!
//! # Choosing the cost estimate
//!
//! `--estimator` picks the statistic the cost model reads a slice's
//! measurements with: `mean-sigmas`, `quantile` or `capped` (the parameters are
//! constants in `engine.rs`). The default is the one the measurements recorded
//! in `beatlm_core::cost` chose, with the table of every run.
//!
//! # The competing load
//!
//! `--background-load` opens a second device on the same adapter, with its own
//! queue and no timestamp queries, and a thread (`load.rs`) that keeps one
//! dispatch queued behind the one running. The kernel is `shaders.wgsl`'s slice
//! kernel over 16 384 workgroups (1 048 576 invocations, sixteen times a
//! slice's) at `--background-iters` loop iterations per invocation, default
//! 4000. It starts, and has completed one dispatch, before the engine is built,
//! and is stopped after the run's last timing is drained, once per run of
//! `--runs`.
//!
//! The setting was chosen by a sweep with `--frames 300`, mean-plus-sigmas, on
//! an Apple M1 Pro (Metal), on 2026-10-21. The aim was a load that raises the
//! slices' measured means by 30 to 80 percent; that was not achieved, and the
//! figures say what happened instead. Idle, the four slices' means were 1.3 to
//! 1.6 ms (first slice of a job to last, 300 frames, one run). With the load at
//! 500, 1000, 2000, 4000, 8000 and 16 000 iterations they were 0.3 to 0.8 ms,
//! lower than idle at every setting. A load of 1024 workgroups at 500 or 2000
//! iterations did the same. What the load added was spread: at 4000 iterations
//! one 300-frame run ended with slice 0 estimated at 8.3 ms against a measured
//! mean of 0.83 ms, and one 300-frame run at 2000 iterations delivered 58 of
//! 298 with only 61 slice-0 timings collected, where three runs in a later
//! process delivered 298, 281 and 288. The setting is therefore a stand-in for
//! noisy measurements, not for a slower GPU; that the means fall is consistent
//! with the GPU running faster when it is kept busy, which was not checked by
//! reading a clock.
//!
//! # Which candidates can run in a browser
//!
//! The question is whether the same code path can run in a browser's event loop,
//! where the main thread can never block. Statements below cite the source they
//! come from; a statement not confirmed from source says so.
//!
//! **Compile check.** `cargo check -p beatlm-bench --target
//! wasm32-unknown-unknown` with this workspace's wgpu features (`webgpu`, not
//! `fragile-send-sync-non-atomic-wasm`) fails with exactly one error, and it is
//! the `thread` candidate's: `scope.spawn` requires `&Gpu: Send`, and
//! `wgpu::Device` is not `Sync` there because the WebGPU backend's
//! `WebDevice` holds an `Rc<Cell<u32>>`. The `sync` and `async` paths produce no
//! error on that target. That is a type check only; it shows neither candidate
//! runs, and the harness itself does not (below).
//!
//! - **`sync`: yes.** Nothing in the per-frame path blocks. `Device::poll` is
//!   documented in wgpu 30.0.1 (`src/api/device.rs`) as "a no-op" on WebGPU,
//!   where "`Device`s are automatically polled", and `PollType::Wait` "has no
//!   effect" there (`wgpu-types` `PollType`), so the `PollType::Poll` call
//!   degrades to nothing and the map callbacks run from the event loop. The
//!   blocking `wait_indefinitely` poll is used only by the warm-up and the final
//!   drain, which a browser host would not call. Whether it then behaves in a
//!   running browser as the native runs here do is unconfirmed: nothing was run
//!   in one.
//! - **`async`: the tick, yes; the driver, no.** `tick_async` contains no
//!   blocking call, so a browser can run it as a future. `pollster::block_on`
//!   cannot drive it there: pollster 1.0.1 (`src/lib.rs`) waits for a pending
//!   future with `thread::park`, and on `wasm32-unknown-unknown` without
//!   atomics std's `Parker::park` is empty (`unsupported.rs` under
//!   `std/src/sys/sync/thread_parking`), so a future that is truly pending would be polled again
//!   in a loop that never returns control to the event loop. The awaited step
//!   here never pends, so it would not hang, but the browser driver has to be
//!   a task spawned on the event loop; that `wasm-bindgen-futures::spawn_local`
//!   is the right one for this workspace is unconfirmed, as it is not a
//!   dependency and was not built. With `target_feature = "atomics"` std parks
//!   with a futex (`thread_parking/mod.rs`), and whether a browser allows that
//!   wait on its main thread is unconfirmed from source: the rule is in the
//!   JavaScript specification, which is not in the tree.
//! - **`thread`: no.** `std::thread::spawn` is `Builder::spawn(..).expect(..)`
//!   (`std/src/thread/functions.rs`), and on `wasm32-unknown-unknown` without
//!   atomics `Thread::new` returns `Err(UNSUPPORTED_PLATFORM)`
//!   (`std/src/sys/thread/unsupported.rs`), so it panics. With
//!   `target_feature = "atomics"` the target selects a different thread module
//!   (`std/src/sys/thread/mod.rs`) but its `Thread` is still the unsupported
//!   one, so std still cannot spawn there. Independently of std, wgpu's WebGPU
//!   handle types implement `Send` and `Sync` only under its `send_sync` cfg,
//!   which is on for native targets and for wasm only with the
//!   `fragile-send-sync-non-atomic-wasm` feature and no atomics (wgpu 30.0.1
//!   `build.rs`, `src/backend/webgpu.rs`); the comment there says the types are
//!   left unmarked when wasm threading is enabled because a `JsValue` is not
//!   `Send`. So the design measured here, one device shared by reference with a
//!   second thread, does not type-check in a browser build; a browser design
//!   would be a worker with its own device, which is not built here, and whether
//!   a browser can hand a `GPUDevice` to a worker is unconfirmed.
//!
//! The harness is native only regardless of mode: on `wasm32-unknown-unknown`
//! `Instant::now` panics ("time not implemented on this platform",
//! `std/src/sys/time/unsupported.rs`) and `thread::sleep` panics ("can't sleep",
//! `std/src/sys/thread/unsupported.rs`), and the pacer uses both. The `beatlm`
//! crate reads no clock, which is why only the harness is affected.

mod args;
mod counters;
mod device;
mod engine;
mod load;
mod net;
mod pace;
#[cfg(test)]
mod reference;
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

    if let Some(dir) = &args.dump_wgsl {
        let paths = net::dump_wgsl(dir, args.fusion)?;
        eprintln!(
            "wrote the WGSL of {} kernels to {}",
            paths.len(),
            dir.display()
        );
    }

    let gpu = device::Gpu::open()?;
    let workload = match args.workload {
        args::WorkloadKind::Synthetic => String::from("synthetic"),
        args::WorkloadKind::Net => {
            format!("net with fusion {}", args::fusion_name(args.fusion))
        }
    };
    eprintln!(
        "running {} frames x {} runs, workload {workload}, mode {}, estimator {}, background load {}, on {} ({})",
        args.frames,
        args.runs,
        args.mode.name(),
        args.estimator.name(),
        if args.background_load { "on" } else { "off" },
        gpu.info.name,
        gpu.info.backend
    );
    let env = report::Environment {
        adapter: gpu.info.name.clone(),
        backend: gpu.info.backend.to_string(),
    };
    let mut violations = 0;
    for number in 1..=args.runs {
        // Each run builds its own scheduler, cost model and timer on the same
        // device, so the difference between two reports is variance, not state
        // carried over.
        //
        // The competing load, when asked for, starts before the engine is built
        // and stops after the last frame's timings are drained, so every
        // measured frame ran under it.
        let load = if args.background_load {
            Some(load::BackgroundLoad::start(&gpu, args.background_iters)?)
        } else {
            None
        };
        let ran = run::run(&gpu, &args);
        let dispatches = load.map(load::BackgroundLoad::stop).transpose();
        let mut outcome = ran?;
        outcome.background_dispatches = dispatches?;
        if number > 1 {
            println!();
        }
        print!("{}", report::render(&env, &args, number, &outcome));
        violations += outcome.violations;
    }

    Ok(if violations == 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    })
}
