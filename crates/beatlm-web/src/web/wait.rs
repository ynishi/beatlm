//! Waiting without blocking: a GPU callback or the next animation frame, as a
//! future the setup awaits on the event loop.
//!
//! Only the setup waits. The frame loop never does: each callback is a frame
//! that runs to its end ([`super::frame`]).

use std::sync::{Arc, Mutex, PoisonError};
use std::task::{Context as TaskContext, Poll, Waker};

use anyhow::{Context, Result, anyhow};
use wasm_bindgen_futures::JsFuture;

use super::page;

/// What a callback and the future waiting for it share: the value once the
/// callback has run, and the waker of a future that polled before it.
struct Shared<T> {
    value: Option<T>,
    waker: Option<Waker>,
}

/// The value a callback is called with, once it has been.
///
/// `register` is handed the callback to give the API that calls back. The
/// callback must be `Send` for wgpu, whose callbacks are on native, so the
/// state it shares with the future is behind a mutex rather than in a page
/// object.
pub async fn callback<T: Send + 'static>(register: impl FnOnce(Box<dyn FnOnce(T) + Send>)) -> T {
    let shared = Arc::new(Mutex::new(Shared {
        value: None,
        waker: None,
    }));
    let fired = Arc::clone(&shared);
    register(Box::new(move |value| {
        let mut shared = fired.lock().unwrap_or_else(PoisonError::into_inner);
        shared.value = Some(value);
        if let Some(waker) = shared.waker.take() {
            waker.wake();
        }
    }));
    std::future::poll_fn(|cx: &mut TaskContext<'_>| {
        let mut shared = shared.lock().unwrap_or_else(PoisonError::into_inner);
        match shared.value.take() {
            Some(value) => Poll::Ready(value),
            None => {
                shared.waker = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    })
    .await
}

/// Resolves once the GPU has finished the work submitted to `queue` so far.
pub async fn work_done(queue: &wgpu::Queue) {
    callback(|done| queue.on_submitted_work_done(move || done(()))).await;
}

/// The timestamp of the next animation frame, in milliseconds on the clock of
/// `performance.now()`.
///
/// The code after the `await` runs in that frame: a promise the
/// `requestAnimationFrame` callback resolves settles in the microtask
/// checkpoint right after the callback, before the browser paints, so a
/// texture of the canvas presented there is shown with that frame.
///
/// # Errors
///
/// A page that refuses the callback, or one that passes no timestamp.
pub async fn next_frame() -> Result<f64> {
    let window = page::window()?;
    let mut requested = Ok(0);
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        requested = window.request_animation_frame(&resolve);
    });
    requested.map_err(|error| anyhow!("requestAnimationFrame failed: {error:?}"))?;
    JsFuture::from(promise)
        .await
        .map_err(|error| anyhow!("the animation frame failed: {error:?}"))?
        .as_f64()
        .context("the animation frame passed no timestamp")
}
