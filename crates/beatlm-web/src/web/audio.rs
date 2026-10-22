//! The audio clock and the click track of an on-beat run.
//!
//! A browser starts an `AudioContext` only from a user gesture (its autoplay
//! policy), so the page shows a start button and says why, and the context is
//! created inside the button's click handler ([`start_on_click`]); nothing of
//! the run starts before the click. Each frame then reads the context's clock
//! ([`Audio::reading`]) for [`crate::onbeat`], which owns every decision about
//! it, and schedules the clicks of the beats about to come on the audio clock
//! ([`Audio::schedule`]), where they play however late a frame runs.
//!
//! web-sys 0.3.106 binds neither `getOutputTimestamp()` nor `outputLatency`
//! nor `baseLatency`, so the three are imported here by name; a browser that
//! lacks one reads as `None`, and [`crate::onbeat::Reading::offset_us`] says
//! what is used instead.

use anyhow::{Context, Result, anyhow};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::prelude::wasm_bindgen;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{AudioContext, HtmlButtonElement};

use super::page;
use crate::onbeat::{Beats, Reading};

/// The `id` of the button `index.html` provides.
const START_ID: &str = "beatlm-start";

/// How far ahead of the audio clock clicks are scheduled, in seconds: half a
/// second, longer than any gap between frames of a page in the foreground, so
/// a click is never scheduled after its time; and shorter than [`crate::onbeat::LEAD`],
/// so the first is scheduled from a frame of the loop.
const LOOKAHEAD_S: f64 = 0.5;

/// A click's length, in seconds: an attack of 2 ms and a decay to silence by
/// 50 ms, short enough to mark an instant.
const CLICK_S: f64 = 0.06;

#[wasm_bindgen]
extern "C" {
    /// An `AudioContext` seen through the members web-sys does not bind; an
    /// `AudioContext` is cast to it unchecked, and every member is looked up
    /// by name on the object, so the type's own name is never used.
    type Unbound;

    /// `AudioContext.getOutputTimestamp()`.
    #[wasm_bindgen(catch, method, js_name = getOutputTimestamp)]
    fn get_output_timestamp(this: &Unbound) -> Result<JsValue, JsValue>;

    /// `AudioContext.outputLatency`, in seconds, or `undefined`.
    #[wasm_bindgen(method, getter = outputLatency)]
    fn output_latency(this: &Unbound) -> JsValue;

    /// `BaseAudioContext.baseLatency`, in seconds, or `undefined`.
    #[wasm_bindgen(method, getter = baseLatency)]
    fn base_latency(this: &Unbound) -> JsValue;
}

/// Shows the start button and waits for it to be clicked; returns the audio
/// context the click handler created and resumed.
///
/// # Errors
///
/// A page without the button, a browser that refuses to create the context,
/// or one that refuses to resume it.
pub async fn start_on_click() -> Result<AudioContext> {
    let button = page::window()?
        .document()
        .context("the window has no document")?
        .get_element_by_id(START_ID)
        .with_context(|| format!("the page has no element with id {START_ID:?}"))?
        .dyn_into::<HtmlButtonElement>()
        .ok()
        .with_context(|| format!("the element {START_ID:?} is not a button"))?;
    button.set_hidden(false);
    button.set_disabled(false);
    let promise =
        js_sys::Promise::new(&mut |resolve: js_sys::Function, reject: js_sys::Function| {
            let target = button.clone();
            // Runs once, inside the click: the gesture the context needs.
            let handler = Closure::once_into_js(move || {
                target.set_onclick(None);
                target.set_disabled(true);
                let settled = match AudioContext::new() {
                    Ok(context) => {
                        // Resuming inside the gesture is what lets it run; the
                        // promise it returns is awaited below.
                        let _resumed = context.resume();
                        resolve.call1(&JsValue::NULL, &context)
                    }
                    Err(error) => reject.call1(&JsValue::NULL, &error),
                };
                if let Err(error) = settled {
                    page::log_error(&format!("the start button's promise failed: {error:?}"));
                }
            });
            button.set_onclick(Some(handler.unchecked_ref()));
        });
    let context = JsFuture::from(promise)
        .await
        .map_err(|error| anyhow!("cannot create an audio context: {error:?}"))?
        .dyn_into::<AudioContext>()
        .map_err(|error| anyhow!("the start button gave no audio context: {error:?}"))?;
    let resumed = context
        .resume()
        .map_err(|error| anyhow!("cannot resume the audio context: {error:?}"))?;
    JsFuture::from(resumed)
        .await
        .map_err(|error| anyhow!("the audio context did not start: {error:?}"))?;
    button.set_hidden(true);
    Ok(context)
}

/// The run's audio context and its click track.
pub struct Audio {
    context: AudioContext,
    /// The next beat whose click is not scheduled yet.
    next_click: u64,
    /// Clicks scheduled.
    scheduled: u64,
    /// Beats whose time had passed before they could be scheduled, and so
    /// got no click.
    late: u64,
}

impl Audio {
    /// The click track of `context`, from beat 0.
    #[must_use]
    pub const fn new(context: AudioContext) -> Self {
        Self {
            context,
            next_click: 0,
            scheduled: 0,
            late: 0,
        }
    }

    /// `currentTime`, in seconds.
    #[must_use]
    pub fn current_time(&self) -> f64 {
        self.context.current_time()
    }

    /// The context's sample rate, in hertz.
    #[must_use]
    pub fn sample_rate(&self) -> f32 {
        self.context.sample_rate()
    }

    /// What the clock says now: the output timestamp, then `currentTime` and
    /// `performance.now()` right after each other, and the two latencies.
    #[must_use]
    pub fn reading(&self, performance: &web_sys::Performance) -> Reading {
        let field = |object: &JsValue, name: &str| {
            js_sys::Reflect::get(object, &JsValue::from_str(name))
                .ok()
                .and_then(|value| value.as_f64())
        };
        let unbound: &Unbound = self.context.unchecked_ref();
        let (context_time_s, performance_time_ms) = match unbound.get_output_timestamp() {
            Ok(stamp) => (
                field(&stamp, "contextTime"),
                field(&stamp, "performanceTime"),
            ),
            Err(_) => (None, None),
        };
        let current_time_s = self.context.current_time();
        let now_ms = performance.now();
        Reading {
            context_time_s,
            performance_time_ms,
            current_time_s,
            now_ms,
            output_latency_s: unbound.output_latency().as_f64(),
            base_latency_s: unbound.base_latency().as_f64(),
        }
    }

    /// Schedules the click of every beat of `beats` due before the lookahead,
    /// up to beat `last` when the run ends there; a beat already past gets no
    /// click and is counted.
    ///
    /// # Errors
    ///
    /// A node the context refuses to create or schedule.
    pub fn schedule(&mut self, beats: &Beats, last: Option<u64>) -> Result<()> {
        let now = self.context.current_time();
        let change = beats.tempo().change_beat();
        while last.is_none_or(|last| self.next_click <= last) {
            let Some(at) = beats.audio_time_s(self.next_click) else {
                break;
            };
            if at > now + LOOKAHEAD_S {
                break;
            }
            if at < now {
                self.late += 1;
            } else {
                // The change beat is marked higher, and every fourth beat
                // above the others.
                let pitch = if Some(self.next_click) == change {
                    2_000.0
                } else if self.next_click.is_multiple_of(4) {
                    1_500.0
                } else {
                    1_000.0
                };
                self.click(at, pitch)?;
                self.scheduled += 1;
            }
            self.next_click += 1;
        }
        Ok(())
    }

    /// One click at audio time `at`: a sine at `pitch` hertz with a 2 ms attack
    /// and an exponential decay, so its onset is the beat.
    fn click(&self, at: f64, pitch: f32) -> Result<()> {
        let failed = |what: &str, error: JsValue| anyhow!("cannot {what}: {error:?}");
        let oscillator = self
            .context
            .create_oscillator()
            .map_err(|e| failed("create a click's oscillator", e))?;
        oscillator.frequency().set_value(pitch);
        let gain = self
            .context
            .create_gain()
            .map_err(|e| failed("create a click's gain", e))?;
        let level = gain.gain();
        level
            .set_value_at_time(0.0, at)
            .and_then(|_| level.linear_ramp_to_value_at_time(0.4, at + 0.002))
            .and_then(|_| level.exponential_ramp_to_value_at_time(0.000_1, at + 0.05))
            .map_err(|e| failed("shape a click", e))?;
        oscillator
            .connect_with_audio_node(&gain)
            .and_then(|_| gain.connect_with_audio_node(&self.context.destination()))
            .map_err(|e| failed("connect a click", e))?;
        oscillator
            .start_with_when(at)
            .and_then(|()| oscillator.stop_with_when(at + CLICK_S))
            .map_err(|e| failed("schedule a click", e))?;
        Ok(())
    }

    /// Clicks scheduled, and beats that got none because their time had
    /// passed.
    #[must_use]
    pub const fn clicks(&self) -> (u64, u64) {
        (self.scheduled, self.late)
    }

    /// Closes the context, which stops the clicks still scheduled.
    pub fn close(&self) {
        if let Err(error) = self.context.close() {
            page::log_error(&format!("cannot close the audio context: {error:?}"));
        }
    }
}
