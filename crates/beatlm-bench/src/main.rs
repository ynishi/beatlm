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
//! kernel count before splitting and `--slice-iters` does not apply. The
//! render pass is the synthetic one.
//!
//! Before the measured loop the network is calibrated and split
//! (`calibrate.rs`, and "Splitting" below): each kernel is timed whole, paced
//! as the loop is, and a kernel whose median exceeds `--max-dispatch-us`
//! (default 1000) is lowered again as several parts, each its own slice; then
//! the split program is timed the same way, and the loop's cost model starts
//! from those measurements. The parts' pipelines and bind groups are built
//! once, after calibration and before the loop. `--max-dispatch-us 0` or
//! `--split off` skips calibration and runs every kernel whole.
//!
//! It measures the rule that the steady-state loop allocates nothing. The
//! network's memory is planned against the baseline WebGPU limits, realised as
//! one buffer per arena before the first frame, and the report prints the plan
//! (`memory_plan_*`) next to wgpu's own object counters read before the first
//! frame and after the last (`loop_*`, from `counters.rs`, which also says what
//! a counter does and does not show). On an Apple M1 Pro (Metal), release
//! build, `--frames 600 --runs 2` at the defaults, on 2026-10-21, with fusion
//! on and off (these runs predate splitting, so they are runs with
//! `--split off`; "Splitting" has the counts with it on):
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
//! latency 2, the quantile estimator, no splitting; times in microseconds:
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
//! # Splitting
//!
//! Splitting exists to bound how long one dispatch holds the queue (the
//! `beatlm_core::kernel` documentation says why, and the core owns the rule
//! and the decision). Configuration of every run: `--workload net --frames 600
//! --runs 2`, `sync` mode, latency 2, the quantile estimator, fusion on,
//! margin 2 ms, release build, on an Apple M1 Pro, Metal, wgpu 30.0.1, on
//! 2026-10-21, with `--max-dispatch-us 1000` (the default) and with `--split
//! off`. Parts are per kernel in program order (`conv2d3x3+relu`,
//! `layernorm`, `conv2d3x3`, `matmul+gelu`, `layernorm`, `matmul+add+scale`);
//! times in microseconds; every run is one row and none was discarded. These
//! runs are from before calibration measured the parts ("Why calibration
//! measures the parts" below), so their parts are the old decision's, and
//! their deadlines were not reported apart for the loop's start.
//!
//! | `--split`, run | parts | `dispatches_per_job` | `longest_dispatch_us` p50 / p99 / max | `gpu_frame_us` p50 / p99 | `cpu_encode_us` p50 / p99 | delivered / missed | `delivery_rate` |
//! |---|---|---|---|---|---|---|---|
//! | on, 1 | 3 2 1 1 1 4 | 12 | 1223 / 2895 / 4432 | 7840 / 12570 | 745 / 3714 | 598 / 0 | 1.0000 |
//! | on, 2 | 3 1 1 1 1 4 | 11 | 1371 / 2911 / 4122 | 7936 / 12972 | 820 / 3138 | 598 / 0 | 1.0000 |
//! | off, 1 | 1 1 1 1 1 1 | 6 | 2086 / 4125 / 6514 | 7752 / 12622 | 407 / 4949 | 598 / 0 | 1.0000 |
//! | off, 2 | 1 1 1 1 1 1 | 6 | 2087 / 3474 / 4271 | 8138 / 11937 | 492 / 4133 | 598 / 0 | 1.0000 |
//!
//! Calibration estimated the second matrix multiply at 3468 and 3472 us, and
//! split it into four parts in both runs; the first convolution at 2427 and
//! 2347 us (three parts), and the first layer norm at 1996 and 867 us, so it
//! was split in one run and not the other. The decision follows what was
//! measured in that process, which is the point of deciding from
//! measurements, and two runs of one setting can differ in their parts.
//! In the loop that followed, the second matrix multiply measured a mean of
//! 2223 and 2187 us whole and 553 to 630 us per part split; its four parts
//! summed to 2351 and 2481 us, a little more than the whole.
//!
//! What the numbers say, and what they do not:
//!
//! - **The longest dispatch per frame came down**: from 2086 and 2087 to 1223
//!   and 1371 at the median, and from 3474 and 4125 to 2895 and 2911 at p99.
//!   No slice's measured mean was above 803 us. The bound held for the means,
//!   not for every dispatch: `longest_dispatch_us` stayed above 1000 at the
//!   median. The slices with the heaviest tails were slices 1 to 4 (the later
//!   parts of the first convolution and the kernels after them), whose final
//!   estimates, the 0.9 quantile of their last 32 measurements, were 1077 to
//!   1645 against means of 362 to 803, while the matrix multiply's parts' were
//!   894 to 897. Why those slices, and not the parts of the longest kernel,
//!   ran long was not established: which slice was the longest in each frame
//!   was not recorded, and a pass's boundaries can include time it waited for
//!   earlier work on the queue (the caveat above).
//! - **It cost CPU encode time at the median and nothing measurable on the
//!   GPU.** The job had 11 or 12 dispatches instead of 6, and the median encode
//!   time was 745 and 820 us against 407 and 492; the 99th percentile spread
//!   more between two runs of one setting than between the settings. The GPU
//!   time per frame and the delivery rate did not move beyond the run-to-run
//!   spread.
//! - **The loop created nothing.** Buffers stayed at 10 at every frame
//!   boundary in all four runs; bind groups (29 and 28) and compute pipelines
//!   (15 and 14) did not change in the split runs, the parts' included,
//!   because every part's pipeline and bind groups are created before the
//!   first frame.
//!
//! **Why calibration measures the parts.** The table above was measured when
//! calibration measured only whole kernels, and the loop's cost model then met
//! every part as a new slice at the same 2 ms prior. A job's estimated cost
//! before its first measurements grew with its number of parts, not with its
//! work, and early jobs of a finely split network could miss their deadline
//! for that reason alone. One 300-frame run with the 1000 us default showed
//! it: calibration measured the last matrix multiply at twice its usual time
//! and split the network into 14 dispatches, and 0.13 of the deadlines were
//! met, against 1.0 in the two runs that followed (Apple M1 Pro, Metal,
//! 2026-10-21). Calibration now times the split program too and starts the
//! loop's cost model from those measurements (`calibrate.rs` owns the two
//! passes), and `startup_missed` reports the misses among the loop's first
//! `2 * COST_WINDOW` frames apart, so a cascade shows. The bound sweep that
//! chose 1000 us was made before that change and is replaced by the one below.
//!
//! **The bound sweep, with the parts measured.** Configuration of every run:
//! `--workload net --frames 300 --runs 2`, `sync` mode, latency 2, the
//! quantile estimator, fusion on, margin 2 ms, release build, on an Apple M1
//! Pro, Metal, wgpu 30.0.1, on 2026-10-21. Parts are per kernel in program
//! order, as above; times in microseconds; every run is one row and none was
//! discarded. `delivered / missed` is out of 298 deadlines except where it
//! sums to 297.
//!
//! | `--max-dispatch-us`, run | parts | `dispatches_per_job` | `longest_dispatch_us` p50 / p99 | `cpu_encode_us` p50 / p99 | delivered / missed | `startup_missed` | `delivery_rate` |
//! |---|---|---|---|---|---|---|---|
//! | 500, 1 | 3 3 2 2 1 5 | 16 | 1121 / 2434 | 691 / 3063 | 296 / 1 | 0 | 0.9966 |
//! | 500, 2 | 2 3 2 2 2 7 | 18 | 1229 / 2056 | 700 / 4624 | 298 / 0 | 0 | 1.0000 |
//! | 1000, 1 | 2 1 1 1 1 4 | 10 | 1328 / 2507 | 737 / 2570 | 298 / 0 | 0 | 1.0000 |
//! | 1000, 2 | 2 2 1 1 1 2 | 9 | 1288 / 2330 | 786 / 2019 | 298 / 0 | 0 | 1.0000 |
//! | 2000, 1 | 1 1 1 1 1 1 | 6 | 2139 / 3451 | 731 / 2336 | 298 / 0 | 0 | 1.0000 |
//! | 2000, 2 | 1 1 1 1 1 1 | 6 | 1975 / 3306 | 740 / 1545 | 298 / 0 | 0 | 1.0000 |
//! | off, 1 | 1 1 1 1 1 1 | 6 | 1948 / 3275 | 867 / 2708 | 298 / 0 | 0 | 1.0000 |
//! | off, 2 | 1 1 1 1 1 1 | 6 | 2087 / 3468 | 802 / 2522 | 298 / 0 | 0 | 1.0000 |
//!
//! What the numbers say, and what they do not:
//!
//! - **No run had a startup miss, at 16 and 18 dispatches per job included.**
//!   The one miss in eight runs (500, run 1) came after the loop's first 64
//!   frames, in a run that also gave up one frame period
//!   (`frames_dropped: 1`); whether the dropped frame caused it was not
//!   checked. Eight runs without a cascade are consistent with the parts now
//!   starting from measurements and do not show a cascade cannot happen: the
//!   0.13 run was one in a session, and the cause that run had, a calibration
//!   that measured twice the usual time, is not something a run can be asked
//!   to repeat.
//! - **The bound moves the longest dispatch and the dispatch count, and the
//!   deliveries did not tell the settings apart.** From unsplit to 1000 us the
//!   median `longest_dispatch_us` fell from 1948 and 2087 to 1328 and 1288 for
//!   3 or 4 more dispatches per job; to 500 us it fell by another 60 to 210
//!   for about 1.8 times the dispatches of 1000 (16 and 18 against 9 and 10),
//!   and p99 fell with it (2056 and 2434 against 2330 and 2507, which
//!   overlap). At 1000 us the median longest dispatch still exceeded the
//!   bound: the bound is a planning target, as the rule says.
//! - **2000 us split nothing on this machine.** No kernel's median was above
//!   2000 in either run (the largest was 1919 and 1810), so the jobs had six
//!   dispatches and the numbers are those of `--split off`.
//! - **Calibration differs between runs of one setting.** The last matrix
//!   multiply's median was 3468 and 1792 us at 1000 (four and two parts) and
//!   2084 and 3469 us at 500 (five and seven parts), and the first
//!   convolution's 1772 and 1204 at 1000. The decision follows what was
//!   measured in that process; the parts a setting gives are not fixed.
//! - **Encode time did not follow the dispatch count in this sweep.** The
//!   median `cpu_encode_us` was 691 to 867 across all four settings with no
//!   order, while the 600-frame table above, on the same machine, had it
//!   rise with the dispatches (745 and 820 against 407 and 492). The cost of
//!   more dispatches on the CPU is therefore not settled by these figures.
//!
//! **The default stays 1000 us.** Nothing measured favours another value, and
//! the cost side of 500 is plain: it nearly doubled the dispatches for a
//! median longest dispatch 60 to 210 us lower, and it had the only miss. That
//! is one miss in eight runs, which two runs per cell cannot rank against
//! 1000. 2000 does not split at all here, and unsplit left the longest
//! dispatch 0.6 to 0.8 ms longer at the median than 1000 did. On another
//! adapter the sweep is the thing to repeat.
//!
//! # The `depth` workload
//!
//! `--workload depth` runs Depth Anything V2 Small as the job: the graph
//! `beatlm::model::depth_anything_v2::model` builds at `--size` pixels a side
//! (a multiple of 14; default 266), lowered with `--fusion`, with every
//! weight read from the real weights file (`depth.rs`; `just
//! setup-depth-anything` fetches it, and `BEATLM_MODELS_DIR` points at a
//! models directory elsewhere, as for the `beatlm` tests). Everything after
//! the graph is the `net` workload's: the program is calibrated and split
//! (`--max-dispatch-us`, `--split`, "Splitting" above), realised through the
//! same `Net` with one input and one output slot per job that can be in
//! flight (`SyncPolicy::max_in_flight`, `--latency` plus one), run before the
//! synthetic render pass under `--latency k`, and watched by the same
//! counters. Every slot holds the same fixed image, uploaded before the loop;
//! the depth is not read back (`depth.rs` says what that leaves out). The
//! report adds `size` to `config` and prints the `net` lines: the kernels and
//! their parts, the memory plan, and the loop's counters.
//!
//! **One input at a time.** A job of this model takes several frames' budget,
//! and jobs run in submission order and never interleave (the core's
//! scheduler owns the rule). Submitting every frame's input then starves every
//! job after the first: each reaches the queue only when the one before it
//! has held the budget up to its own deadline, gets its deadline frame and
//! misses. The engine's test
//! `a_job_longer_than_a_frame_starves_when_every_frame_submits_and_not_when_free`
//! shows it on the scheduler alone, with a job of five frames' estimates under
//! latency 5: every frame's input delivers 1 result in 120 frames and misses
//! 114, where submitting only when the engine is free delivers 23 and misses
//! none. So `--submit` (default `every`, and `free` for `depth`) has a second
//! mode: a frame submits its input only when the job submitted last has had
//! all its slices handed out, or has left the scheduler; otherwise that
//! frame's input is not run, as a host skips frames of its source for a model
//! slower than its frame rate. The promise is unchanged for every input that
//! is run: its result is delivered exactly `k` frames after it. At most one
//! job has slices left at a time, so the loop's timer holds one job's slices
//! (`engine::jobs_with_slices`), and the unsplit model's 155 kernels times
//! `k + 1` jobs, which the timer could not hold for `k` past 12, never has to
//! be timed. `submitted` in the report counts the inputs run; `delivered` per
//! second is the rate depth results arrive at.
//!
//! **The warm-up runs one slot per submission.** It used to encode every slot
//! into one command buffer; at 434 pixels and `--latency 13` that was 14 runs
//! of the model, about two seconds of GPU work in one submission, and the
//! device was lost before the first frame: every timing of that run failed,
//! and the second run panicked creating a buffer on the lost device. Those
//! runs, and the `--latency 14` runs made in the same session, are not in
//! the tables below.
//!
//! ## What a size costs
//!
//! GPU time of the whole model, fused, real weights, per run: by `beatlm`'s
//! test `the_whole_model_runs_on_the_gpu_at_every_timed_side`, five runs back
//! to back after one untimed run, each the sum of its 155 kernels' timestamp
//! pairs; and `calibrated_job_us`, the sum of the kernels' medians that
//! calibration measured paced behind the render pass, from the two runs of
//! each sweep cell below. The memory plan is the model's with one slot, from
//! the test. Apple M1 Pro, Metal, wgpu 30.0.1, release build, on 2026-10-22.
//!
//! | side (pixels) | patch grid | tokens | GPU time per run, 5 runs (ms) | `calibrated_job_us` (ms) | memory plan, 1 slot (bytes) |
//! |---|---|---|---|---|---|
//! | 518 | 37 | 1370 | 202.3 to 202.7 | 203.2 to 203.8 | 207 959 172 |
//! | 434 | 31 | 962 | 137.8 to 138.7 | 138.7 to 138.9 | 156 838 788 |
//! | 350 | 25 | 626 | 85.6 to 86.0 | 86.3 to 86.7 | 131 021 828 |
//! | 266 | 19 | 362 | 53.2 to 53.6 | 54.0 to 54.1 | 116 541 700 |
//! | 182 | 13 | 170 | 35.0 to 35.3 | 35.8 to 35.9 | 106 010 884 |
//!
//! The budget a frame gives the model is the period less the render estimate
//! (about 2 ms) and the 2 ms margin, about 12.6 ms, and the slices that fit it
//! are estimated at their 0.9 quantile: in the runs below a job used 3 frames
//! at 182 pixels, 5 at 266, 8 to 9 at 350, 13 to 14 at 434 and about 20 at
//! 518, which is what `submitted` counts with `--submit free`. A job submitted
//! at frame `f` with latency `k` has frames `f` to `f + k` (the deadline
//! frame's slices run before its render pass), `k + 1` frames, so `k` at
//! least one less than the frames a job uses can deliver every result.
//!
//! ## The sweep
//!
//! Configuration of every run: `--workload depth --frames 600 --runs 2`, the
//! size and `--latency` in the table, `--submit free`, `sync` mode, the
//! quantile estimator, fusion on, `--max-dispatch-us 1000`, margin 2 ms,
//! release build, Apple M1 Pro, Metal, wgpu 30.0.1, 2026-10-22. Latency is
//! `k` times the 16.667 ms period (`promised_latency_ms`). Times in
//! microseconds; each run is one row, and only the lost-device runs named
//! above were left out. Every run had `latency_violations` 0, every delivery
//! at exactly `k` frames, and every `loop_*` counter's delta 0: buffers stayed
//! at 10 to 12 at every frame boundary, and bind groups and compute pipelines
//! did not change.
//!
//! | size, `k` (latency ms), run | `submitted` | delivered / missed | `delivery_rate` | `gpu_frame_us` p50 / p99 / max | `gpu_frames_over_period` | `longest_dispatch_us` p50 / p99 / max | `dispatches_per_job` | `startup_missed` | `frames_dropped` | `memory_plan_total_bytes` (slots) |
//! |---|---|---|---|---|---|---|---|---|---|---|
//! | 182, 2 (33.3), 1 | 200 | 200 / 0 | 1.0000 | 13533 / 14436 / 15245 | 0 of 600 | 1010 / 1515 / 1623 | 156 | 0 | 0 | 107 071 236 (3) |
//! | 182, 2 (33.3), 2 | 200 | 200 / 0 | 1.0000 | 13540 / 14789 / 15302 | 0 of 600 | 1010 / 1534 / 1759 | 156 | 0 | 0 | 107 071 236 (3) |
//! | 182, 3 (50.0), 1 | 200 | 198 / 1 | 0.9950 | 13651 / 14635 / 15021 | 0 of 599 | 1013 / 1539 / 1794 | 156 | 0 | 1 | 107 601 412 (4) |
//! | 182, 3 (50.0), 2 | 200 | 199 / 0 | 1.0000 | 13560 / 14898 / 15511 | 0 of 599 | 1012 / 1535 / 1753 | 156 | 0 | 0 | 107 601 412 (4) |
//! | 266, 4 (66.7), 1 | 120 | 120 / 0 | 1.0000 | 13160 / 14799 / 15285 | 0 of 599 | 901 / 1870 / 1995 | 160 | 0 | 0 | 121 070 852 (5) |
//! | 266, 4 (66.7), 2 | 120 | 118 / 2 | 0.9833 | 13134 / 14979 / 15472 | 0 of 597 | 911 / 1733 / 1866 | 160 | 0 | 2 | 121 070 852 (5) |
//! | 266, 5 (83.3), 1 | 120 | 118 / 1 | 0.9916 | 13210 / 14615 / 16272 | 0 of 592 | 892 / 1818 / 1971 | 160 | 0 | 5 | 122 203 140 (6) |
//! | 266, 5 (83.3), 2 | 120 | 119 / 0 | 1.0000 | 13151 / 14834 / 15082 | 0 of 599 | 896 / 1885 / 1953 | 160 | 0 | 0 | 122 203 140 (6) |
//! | 350, 7 (116.7), 1 | 75 | 48 / 27 | 0.6400 | 13422 / 14889 / 20267 | 1 of 592 | 1175 / 1745 / 2039 | 191 | 0 | 6 | 144 744 964 (8) |
//! | 350, 7 (116.7), 2 | 75 | 47 / 28 | 0.6267 | 13439 / 14950 / 16129 | 0 of 599 | 1178 / 1767 / 2077 | 191 | 3 | 1 | 144 744 964 (8) |
//! | 350, 8 (133.3), 1 | 68 | 66 / 1 | 0.9851 | 13234 / 14929 / 18147 | 1 of 594 | 1176 / 1730 / 1914 | 191 | 0 | 3 | 146 705 412 (9) |
//! | 350, 8 (133.3), 2 | 67 | 61 / 5 | 0.9242 | 12748 / 14512 / 20345 | 1 of 593 | 1101 / 1773 / 1922 | 192 | 2 | 5 | 146 705 412 (9) |
//! | 434, 13 (216.7), 1 | 45 | 41 / 3 | 0.9318 | 12872 / 14393 / 15269 | 0 of 596 | 1228 / 1811 / 2178 | 232 | 1 | 4 | 196 022 788 (14) |
//! | 434, 13 (216.7), 2 | 43 | 6 / 36 | 0.1429 | 12321 / 15590 / 18799 | 1 of 600 | 1395 / 2172 / 2765 | 232 | 2 | 0 | 196 022 788 (14) |
//! | 434, 15 (250.0), 1 | 45 | 44 / 0 | 1.0000 | 12918 / 15156 / 15794 | 0 of 600 | 1304 / 1916 / 2162 | 232 | 0 | 0 | 202 051 076 (16) |
//! | 434, 15 (250.0), 2 | 47 | 45 / 0 | 1.0000 | 13214 / 15001 / 15757 | 0 of 598 | 1272 / 1842 / 2161 | 232 | 0 | 0 | 202 051 076 (16) |
//! | 518, 20 (333.3), 1 | 30 | 27 / 2 | 0.9310 | 12859 / 15269 / 17059 | 1 of 600 | 1410 / 1922 / 2076 | 292 | 0 | 0 | 293 826 820 (21) |
//! | 518, 20 (333.3), 2 | 30 | 28 / 1 | 0.9655 | 13105 / 14474 / 15182 | 0 of 600 | 1379 / 1893 / 1951 | 292 | 0 | 0 | 293 826 820 (21) |
//!
//! What the numbers say, and what they do not:
//!
//! - **The issue's criterion held in every run.** GPU time per frame, render
//!   and slices, was within the 16 667 us period at p99 (14 393 to 15 590),
//!   and no delivery was at the wrong latency. It held at sizes and latencies
//!   that missed most deadlines (350 at `k` 7, 434 at 13 in run 2): the
//!   budget bounds each frame whatever the latency, so the criterion does not
//!   choose `k` alone; the delivery rate does.
//! - **Frames over the period happen, rarely.** Five of the 18 runs had one
//!   frame over the period, at 17 059 to 20 345 us, none at 182 or 266
//!   pixels. Which slices ran long in those frames was not recorded.
//! - **Delivery stops collapsing at `k` one less than the frames a job
//!   uses**: 2 at 182 pixels, 4 at 266, 8 at 350 (0.9851 and 0.9242, its
//!   misses in runs that dropped frames), 15 at 434 (13 was not enough, and 14
//!   was not measured), and more than 20 at 518, where `k` 20 missed 2 and 1
//!   deadlines without dropping a frame. Near that edge a run can fall far:
//!   434 at `k` 13 delivered 0.93 in one run and 0.14 in the other, which
//!   dropped no frame and whose dispatches ran longer (longest p99 2172
//!   against 1811). With `--submit free` each job starts fresh once the one
//!   before has left, so those are jobs that each ran out of frames, not one
//!   job's lateness carried into the next.
//! - **Away from that edge, every miss was in a run that dropped frames.** At
//!   182, 266 and 350 pixels, leaving out 350 at `k` 7, every run with a miss
//!   had `frames_dropped` above zero, and every run without a dropped frame
//!   missed nothing. That is consistent with a dropped period being a
//!   deadline frame, which the scheduler then cannot keep; that the dropped
//!   frame was the deadline frame was not recorded. The drops came with CPU
//!   encode spikes (`cpu_encode_us` max up to 83 ms), not with GPU overruns.
//! - **The longest dispatch grows with the size**, from 0.9 to 1.0 ms at the
//!   median at 182 and 266 pixels to 1.4 ms at 518, over the 1000 us bound at
//!   the median as in "Splitting": the bound is a planning target.
//! - **Memory** is the model's plan plus one input and one output per slot:
//!   from 107 MB at 182 pixels with 3 slots to 294 MB at 518 with 21.
//!
//! ## The decision
//!
//! **266 pixels at `k` 5 is the default for this adapter**: `--size 266
//! --latency 5`, a latency of 83.3 ms, a result every 5 frames (12 a second),
//! from a 19 x 19 patch grid. Both its runs met the criterion (p99 14 615 and
//! 14 834 us, nothing over the period) and delivered 0.9916 and 1.0, the one
//! miss in a run that dropped five frames. `k` 4 met the criterion too and
//! delivered 1.0 and 0.9833, but at `k` 4 a job uses all five of the frames
//! it is given, a margin like the one at which 434 pixels at `k` 13 fell to
//! 0.14 in one run of two: the frame `k` 5 adds is slack against a GPU
//! running slower than calibration measured, bought for 16.7 ms of latency.
//!
//! The trade-off, latency against resolution, from the table:
//!
//! | size | patch grid | `k` | latency (ms) | results per second | delivered in its runs |
//! |---|---|---|---|---|---|
//! | 182 | 13 x 13 | 2 | 33.3 | 20 | 1.0, 1.0 |
//! | 266 | 19 x 19 | 5 (default) | 83.3 | 12 | 0.9916, 1.0 |
//! | 350 | 25 x 25 | 8 | 133.3 | 6.7 | 0.9851, 0.9242 |
//! | 434 | 31 x 31 | 15 | 250.0 | 4.5 | 1.0, 1.0 |
//! | 518 | 37 x 37 | more than 20 | more than 333 | about 3 | 0.9310, 0.9655 at 20 |
//!
//! Why not the others: 182 pixels has the lowest latency and delivered every
//! result, and is the choice where latency matters most, but its depth comes
//! from 169 patches, under half of 266's 361. 350 pixels and up cost 133 ms of
//! latency or more for under 7 results a second, and 350 at `k` 8 still
//! missed 5 deadlines in one run; 518, the size the position embedding is
//! stored for, needs more than a third of a second. On another adapter the
//! costs above move, and the sweep is the thing to repeat. Where to start:
//! the frames a job uses are its `calibrated_job_us` over what a frame's
//! budget held of it, 10 to 12 ms here, rounded up; `k` one less than that is
//! the edge, and one more frame is the slack the default keeps.
//!
//! # Running it
//!
//! ```text
//! cargo run --release -p beatlm-bench -- --frames 300
//! cargo run --release -p beatlm-bench -- --frames 600 --runs 3 --mode thread
//! cargo run --release -p beatlm-bench -- --frames 600 --runs 2 --estimator capped --background-load
//! cargo run --release -p beatlm-bench -- --workload net --frames 600 --runs 2 --fusion off
//! cargo run --release -p beatlm-bench -- --workload net --frames 600 --runs 2 --split off
//! cargo run --release -p beatlm-bench -- --workload net --frames 300 --runs 2 --max-dispatch-us 500
//! cargo run --release -p beatlm-bench -- --workload net --frames 1 --dump-wgsl target/wgsl
//! cargo run --release -p beatlm-bench -- --workload depth --frames 600 --runs 2
//! cargo run --release -p beatlm-bench -- --workload depth --size 182 --latency 2 --frames 600 --runs 2
//! ```
//!
//! The `depth` workload reads the weights `just setup-depth-anything` fetches;
//! from a worktree, set `BEATLM_MODELS_DIR` to the main checkout's `models/`.
//!
//! Options are listed by `--help`. It needs an adapter with
//! `Features::TIMESTAMP_QUERY` and is not part of `just check`; an adapter is
//! not guaranteed where `check` runs. A run takes about `frames / 60` seconds;
//! with splitting on, calibration first runs two passes of `2 * COST_WINDOW`
//! paced frames, each plus one untimed frame. A calibration frame waits for a
//! whole job, so for the depth model a pass takes about 65 times its GPU time
//! per run: about 13 seconds at 518 pixels from the 202 ms per run measured
//! on an Apple M1 Pro, Metal ("What a size costs" above).
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
//! | `config` | the options the run used, `mode`, `estimator`, `submit` and `background_load` included (and `background_iters` when the load is on) |
//! | `promised_latency_ms` | `--latency` times the frame period: what the host delays its source by |
//! | `frames` | frames run |
//! | `frames_dropped` | frame periods given up because a frame began a whole period late; the loop drops them as a display would, rather than running frames back to back to catch up |
//! | `submitted` | inputs run as jobs: one per frame with `--submit every`, fewer with `--submit free` |
//! | `delivered`, `missed` | results delivered on their frame, and deadlines that arrived without a complete result |
//! | `delivery_rate` | `delivered` over `delivered` plus `missed`: the share of deadlines that arrived with their result; `none` when no deadline arrived |
//! | `startup_missed` | the part of `missed` that belongs to jobs submitted in the loop's first `2 * COST_WINDOW` frames, so a cascade of misses while estimates are new shows apart from misses in steady state |
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
//! | `longest_dispatch_us` | per timed frame that ran a slice, the longest single measured compute pass: how long one dispatch held up the queue; p50, p99, max |
//! | `gpu_frames_over_period` | timed frames whose render plus slices took longer than the frame period, of the timed frames: a real overrun that the estimates were supposed to prevent |
//! | `background_dispatches` | dispatches the competing load submitted, from the one completed before the first frame to the last one queued when it was stopped; present only with `--background-load` |
//! | `slice[i]` | the cost model's final estimate for slice `i` against the mean of its measured durations |
//! | `workload`, `size`, `fusion`, `split` (in `config`) | present only for a planned program, as `workload=net` or `workload=depth size=N`, then `fusion=on` or `fusion=off`, then `split=on max_dispatch_us=N` or `split=off`; the synthetic report has none. `slices` there is the kernel count before splitting |
//! | `kernel_parts[i]` | `net` and `depth` only: kernel `i` of the unsplit program, its parts, and the median calibration measured for it run whole (`calibrated_us`, `none` with splitting off) |
//! | `calibrated_job_us` | `net` and `depth` only: the sum of the `calibrated_us` medians, what one job took paced behind the render pass before it was split; `none` with splitting off |
//! | `kernels` | `net` and `depth` only: each kernel of a job, in dispatch order, named by its main operation and epilogue, a part as `name[i/n]`; one slice each |
//! | `memory_plan_*` | `net` and `depth` only: tensors, input/output slots, arenas and their bytes, the plan's total, the most bytes live at one kernel, and the bytes with no sharing |
//! | `wgpu_counters_populated` | `net` and `depth` only: whether the backend sets wgpu's buffer counter; when false the loop's buffer creation is unmeasured and the report says so |
//! | `loop_*` | `net` and `depth` only: a counter's reading before the first frame and after the last, and the change; for buffers also the least and most seen at a frame boundary |
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
mod calibrate;
mod counters;
mod depth;
mod device;
#[cfg(test)]
mod dpt;
mod engine;
#[cfg(test)]
mod harness;
mod load;
mod net;
mod pace;
#[cfg(test)]
mod reference;
mod report;
mod run;
mod stats;
#[cfg(test)]
mod vit;
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
        args::WorkloadKind::Net | args::WorkloadKind::Depth => {
            let split = args
                .max_dispatch_us
                .map_or_else(|| String::from("off"), |us| format!("at {us} us"));
            let size = if args.workload == args::WorkloadKind::Depth {
                format!(" at {} pixels", args.size)
            } else {
                String::new()
            };
            format!(
                "{}{size} with fusion {}, splitting {split}",
                args.workload.name(),
                args::fusion_name(args.fusion)
            )
        }
    };
    eprintln!(
        "running {} frames x {} runs, workload {workload}, latency {}, submit {}, mode {}, estimator {}, background load {}, on {} ({})",
        args.frames,
        args.runs,
        args.latency,
        args.submit.name(),
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
