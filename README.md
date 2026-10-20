# beatlm

beatlm runs neural networks on the same GPU queue as a real-time renderer,
inside a per-frame time budget, and delivers each result on a promised frame.

It is built for live audio/visual work: a camera feed passing through a depth
or segmentation model before it reaches a shader, a generator whose output must
land on the next bar, an audio model running beside the visuals. Tools in this
space usually run inference in a separate process and composite whatever result
arrived last, so the effect drifts against the picture by a varying number of
frames. beatlm owns the timeline instead.

## Promises

The engine states what it promises about delivery, per model:

| Level  | Promise                                                                                                                                 |
| ------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| L1     | The result for input frame `t` is delivered on frame `t + k`, with `k` fixed. The source is delayed by the same `k`, so the two align. |
| L2     | The result is delivered in the same frame as its input. Only for models that fit the frame budget.                                     |
| Beat   | The result lands on a beat boundary. Work starts early enough, by the measured cost, to finish before the beat.                        |

When a deadline cannot be met, the engine reports it and applies the policy the
caller chose; it never delays silently.

## How

- **Frame-budget scheduling.** A model is compiled into slices of GPU work.
  Each frame runs as many slices as fit in the time left after rendering. Slice
  costs come from GPU timestamps measured on the machine running them, not from
  a static estimate.
- **Static memory planning.** Every buffer a model needs is planned before the
  first frame. The steady-state loop allocates nothing and reads nothing back.
- **Generated, fused kernels.** Kernels are generated as WGSL text, validated
  with naga, and fused to keep dispatch counts low — dispatch overhead is what a
  small model's budget is spent on.

Native backends (Metal first, then Vulkan) and the browser (WebGPU) share one
code path through [wgpu](https://wgpu.rs).

## Crates

| Crate         | Role                                                                                         |
| ------------- | -------------------------------------------------------------------------------------------- |
| `beatlm`      | What you depend on. Carries the core's decisions out on wgpu; loads models.                 |
| `beatlm-core` | The domain: clocks, budgets, sync promises, plans, kernel IR. Pure, with no GPU dependency. |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), and
[PUBLIC_DEVELOPMENT.md](PUBLIC_DEVELOPMENT.md) for what may be published.
Security reports: [SECURITY.md](SECURITY.md).

## Licence

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

Model weights are not part of this repository and are not covered by this
licence. Each model is distributed under its own terms.
