# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Workspace with `beatlm-core` (pure domain) and `beatlm` (wgpu umbrella),
  building for native targets and `wasm32-unknown-unknown`.
- `beatlm-core` timing domain: frame budgets, a measurement-fed cost model,
  delivery promises (fixed latency, same frame, on beat) and a scheduler that
  delivers results exactly on their promised frame or reports the miss.
- `beatlm::gpu::SliceTimer`: GPU time per compute and render pass from
  timestamp queries, read back without blocking the frame loop.
- `beatlm-bench`: a headless synthetic-load benchmark of the fixed-latency
  promise, with its first baseline recorded in the crate docs.
- `beatlm-bench --mode sync|thread|async` and `--runs`, with a CPU breakdown
  into planning, encoding and `queue.submit`.
- `beatlm-core::device::DeviceProfile` and `beatlm-core::memory::MemoryPlan`:
  adapter limits as values, and tensor placement in shared arenas.
- `beatlm::gpu::profile` and `beatlm::gpu::Arenas`: the translation from wgpu
  limits, and a plan realised as buffers once.
- `SyncPolicy::max_in_flight`: how many output slots a host keeps.
- `beatlm-bench --workload net`: a small planned network with GPU-checked
  kernels and wgpu counters around the frame loop.
- `beatlm-core::cost::Estimator`: mean plus sigmas, windowed quantile and
  capped deviation.
- `beatlm-bench --estimator`, `--background-load`, and the `delivery_rate`
  and `gpu_frames_over_period` report lines.
- `beatlm-core::kernel`: a typed kernel IR (convolution, layer norm, matrix
  multiply and element-wise operations), lowering with epilogue fusion that
  can be switched off, kernel specs checked against a device profile with a
  baseline variant, and tensor lifetimes for the memory plan derived from the
  kernels.
- `beatlm::codegen`: WGSL generated from kernel specs and validated by naga
  before a pipeline is created, and `Kernels`, every pipeline and bind group of
  a program created once.
- `beatlm-bench --fusion on|off` and `--dump-wgsl`, and the
  `dispatches_per_job`, `dispatches_per_frame` and `kernels` report lines.
- Kernel splitting: `kernel::Part`, `Program::split` and
  `Program::parts_within`, with per-part ranges as WGSL override constants.
- `beatlm-bench --max-dispatch-us` and `--split on|off`, a paced calibration
  before the loop, and the `longest_dispatch_us` and `kernel_parts` report
  lines.
- `beatlm::codegen::Generator`, validating each distinct WGSL text once, and
  `beatlm::gpu::MAX_SLICES`.
- `beatlm-bench` calibrates split parts in a second pass so the loop starts
  from their measurements, refuses splits beyond the timer's capacity before
  building anything, and reports `startup_missed`.
- `beatlm::load`: a safetensors reader and a configuration-derived inventory
  of the tensors Depth Anything V2 Small needs, checked against the file.
- `just setup-depth-anything` and `just reference-depth-anything`: user-run
  setup that fetches the Apache-2.0 Small weights from a pinned commit and
  writes reference outputs from the transformers implementation; neither is
  part of the build or `just check`.
- Kernel IR operations for a ViT encoder: patch embedding, class token,
  attention scores, softmax and values, exact GELU, LayerScale, and tokens to
  a feature map; layer norm takes its own epsilon.
- `beatlm::model`: graph building from weights by role, and the Depth Anything
  V2 Small encoder, checked against the transformers reference.
- Kernel IR operations for a DPT decoder: strided 3x3 convolution, 1x1
  convolution, transposed convolution with kernel equal to stride, bilinear
  resize with `align_corners=True`, and a ReLU prologue on 3x3 convolutions.
- The whole Depth Anything V2 Small model (`model::depth_anything_v2::model`),
  whose depth matches the transformers reference on the real weights.

### Changed

- Design is documented in the crate docs; the separate ADR directory is gone.
- The concurrency model is a synchronous `tick` per frame, decided by
  measurement and recorded in the `beatlm` crate docs.
- The scheduler runs jobs strictly in submission order; leftover budget is no
  longer given to a later job's slice, because jobs share one memory plan.
- `CostModel::new` takes an `Estimator`; the default is the 0.9 quantile of
  the last 32 measurements, chosen by measurement (breaking).
- `beatlm-bench --workload net` runs generated kernels; the hand-written
  `net.wgsl` is gone, and the network gained biases, a ReLU, a GELU, a
  residual add and a scale for fusion to absorb.
