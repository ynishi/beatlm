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

### Changed

- Design is documented in the crate docs; the separate ADR directory is gone.
- The concurrency model is a synchronous `tick` per frame, decided by
  measurement and recorded in the `beatlm` crate docs.
- The scheduler runs jobs strictly in submission order; leftover budget is no
  longer given to a later job's slice, because jobs share one memory plan.
