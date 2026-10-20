# ADR 0001: Concurrency model of the engine

Status: Proposed

## Context

beatlm shares one GPU queue with the host's renderer and must hand work to it
every frame within a budget measured in milliseconds. The host owns the frame
loop: a winit event loop natively, `requestAnimationFrame` in the browser.
In the browser the main thread can never block, and GPU completion arrives as
promises (`mapAsync`, `onSubmittedWorkDone`).

The default in our Rust projects is async first, with stateful synchronous
libraries confined to a dedicated thread behind an async handle. That default
assumes the library is a service the application calls into. Here the engine is
driven by the frame loop, and the property that matters is the latency between
a frame starting and its GPU work being submitted.

## Options

1. **Frame-driven synchronous API.** The host calls `tick(frame_context)` once
   per frame on its render thread; the engine decides, encodes and submits
   before returning. GPU completion is observed on later ticks by polling
   (native) or by promise callbacks recorded into state (web).
2. **Async handle over a dedicated GPU thread.** The engine owns the device on
   its own thread; the host sends frame events over a channel and receives
   results.
3. **Async API on the host's runtime.** `tick` is an `async fn` awaited by the
   host.

## Decision

Not yet taken. It is settled by measurement in milestone M1, against a
synthetic GPU load: the time from frame start to submission, its jitter at the
99th percentile, and whether the same code path holds on native and on wasm.
The chosen option and the numbers that chose it replace this section.

## Consequences

Whichever is chosen, `beatlm-core` is unaffected: it takes time and
measurements as arguments and returns decisions, so it has no concurrency model
of its own.
