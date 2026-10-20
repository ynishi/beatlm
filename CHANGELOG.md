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

### Changed

- Design is documented in the crate docs; the separate ADR directory is gone.
