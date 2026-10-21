//! The three concurrency candidates, as three ways to drive one [`Engine`].
//!
//! Pacing ([`Pacer`]) and the frame's work ([`Engine`]) are the same in every
//! candidate; what differs is which thread decides, encodes and submits the
//! frame, and how the frame loop calls it:
//!
//! - **sync** (candidate 1): the frame-loop thread owns the engine and calls
//!   [`Engine::tick_sync`].
//! - **thread** (candidate 2): a dedicated GPU thread owns the engine. The
//!   frame-loop thread paces frames and sends each [`Tick`] over a bounded
//!   channel; the GPU thread runs `tick_sync` for it.
//! - **async** (candidate 3): the frame-loop thread owns the engine and drives
//!   [`Engine::tick_async`] to completion with `pollster::block_on`, once per
//!   frame.

use std::sync::mpsc;
use std::thread;
use std::time::Instant;

use anyhow::{Result, anyhow};

use crate::args::{Args, Mode};
use crate::device::Gpu;
use crate::engine::{Engine, Outcome};
use crate::pace::{Pacer, Tick, micros_between};

/// How many frame events the channel of the threaded candidate holds before the
/// frame-loop thread blocks in `send`.
///
/// Two frames of slack: a GPU thread more than that behind has already lost the
/// frame the event is for, and the blocked sender shows up as frame-start
/// lateness rather than as an unbounded queue.
const CHANNEL_DEPTH: usize = 2;

/// Runs `args.frames` frames at the pacer's period against `gpu`, driven by the
/// candidate `args.mode` names.
///
/// # Errors
///
/// A device that cannot be polled, a timer that cannot be built, a scheduler
/// that rejects the configuration, or a GPU thread that panicked.
pub fn run(gpu: &Gpu, args: &Args) -> Result<Outcome> {
    match args.mode {
        Mode::Sync => run_sync(gpu, args),
        Mode::Thread => run_thread(gpu, args),
        Mode::Async => run_async(gpu, args),
    }
}

/// Candidate 1: the frame loop calls the engine directly.
fn run_sync(gpu: &Gpu, args: &Args) -> Result<Outcome> {
    let mut engine = Engine::new(gpu, args)?;
    let mut pacer = Pacer::start(args.frames);
    while let Some(tick) = pacer.next_frame() {
        engine.tick_sync(&tick)?;
        engine.end_tick(&tick);
    }
    Ok(close(pacer, engine.finish()?))
}

/// Candidate 3: the frame loop awaits the engine's `async` tick, one
/// `block_on` per frame.
fn run_async(gpu: &Gpu, args: &Args) -> Result<Outcome> {
    let mut engine = Engine::new(gpu, args)?;
    let mut pacer = Pacer::start(args.frames);
    while let Some(tick) = pacer.next_frame() {
        pollster::block_on(engine.tick_async(&tick))?;
        engine.end_tick(&tick);
    }
    Ok(close(pacer, engine.finish()?))
}

/// Candidate 2: the frame loop sends events, a GPU thread runs them.
///
/// The GPU thread builds the engine, so the workload, timer, scheduler and cost
/// model are created and used there; the frame-loop thread never touches the
/// device during the run. The pacer starts only after the engine is built and
/// warmed up, as in the other candidates.
fn run_thread(gpu: &Gpu, args: &Args) -> Result<Outcome> {
    let (events, inbox) = mpsc::sync_channel::<Tick>(CHANNEL_DEPTH);
    let (ready_tx, ready_rx) = mpsc::channel::<()>();
    thread::scope(|scope| {
        let worker = scope.spawn(move || gpu_thread(gpu, args, &inbox, &ready_tx));
        // The GPU thread signals only once its engine is built; if it ends
        // first, its own error (or panic) is what the join returns.
        if ready_rx.recv().is_err() {
            return join(worker).and(Err(anyhow!("the GPU thread ended early")));
        }
        let mut pacer = Pacer::start(args.frames);
        while let Some(tick) = pacer.next_frame() {
            // A failed send means the GPU thread stopped; its error is the
            // result of the join below.
            if events.send(tick).is_err() {
                break;
            }
        }
        drop(events);
        Ok(close(pacer, join(worker)?))
    })
}

/// The body of the GPU thread: build the engine, tell the frame loop it may
/// start, then run each event until the channel closes.
fn gpu_thread(
    gpu: &Gpu,
    args: &Args,
    inbox: &mpsc::Receiver<Tick>,
    ready: &mpsc::Sender<()>,
) -> Result<Outcome> {
    let mut engine = Engine::new(gpu, args)?;
    // The frame loop is waiting on the receiver, so the send cannot fail.
    let _ = ready.send(());
    while let Ok(tick) = inbox.recv() {
        engine.record_handoff(micros_between(tick.began, Instant::now()));
        engine.tick_sync(&tick)?;
        engine.end_tick(&tick);
    }
    engine.finish()
}

/// Waits for the GPU thread and returns what it measured.
fn join(worker: thread::ScopedJoinHandle<'_, Result<Outcome>>) -> Result<Outcome> {
    worker
        .join()
        .map_err(|_| anyhow!("the GPU thread panicked"))?
}

/// Adds the pacer's side of the run to the engine's.
fn close(pacer: Pacer, mut outcome: Outcome) -> Outcome {
    let (dropped, lateness_us) = pacer.finish();
    outcome.dropped = dropped;
    outcome.lateness_us = lateness_us;
    outcome
}
