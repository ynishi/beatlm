//! What a run is configured with, read from the page's query string.
//!
//! The query is parsed by hand for the same reason the bench parses its
//! command line by hand: a handful of numbers. Parsing the text, rather than
//! asking the browser's `URLSearchParams`, keeps it testable without a
//! browser. Anything not understood is refused, so a mistyped parameter
//! cannot leave a run on its default without saying so.
//!
//! # Parameters
//!
//! - `frames`: frame indices the run lasts, at least one (default
//!   [`DEFAULT_FRAMES`]).
//! - `model`: the job, `synthetic` (the default; the bench's busy-loop slices)
//!   or `depth` (Depth Anything V2 Small, [`crate::depth`]).
//! - `side`: the depth model's input side in pixels, a positive multiple of 14
//!   up to 518 (default [`NATIVE_DEFAULT_SIDE`]).
//!   Only with `model=depth`.
//! - `latency`: the promised latency in frames, which the display's period
//!   turns into time.
//! - `latency_ms`: the promised latency in milliseconds, a decimal; the page
//!   measures the display's period before the run and promises the frames
//!   that take that long ([`crate::depth::frames_for`] has the rule). Not with
//!   `latency`. Without either, the synthetic workload promises
//!   [`DEFAULT_LATENCY`] frames and the depth model the native default's
//!   latency in time ([`NATIVE_DEFAULT_LATENCY`]).
//! - `arena_bytes`: plan one arena of that many bytes, create the device with
//!   the limits it needs and realise it before the run, to check the limit
//!   path. Only with the synthetic workload: the depth model's own plan sets
//!   the device's limits.
//!
//! **Why a latency in time.** A promise in frames is worth the display's
//! period, so the native default of 5 frames, a decision made at 60 Hz for
//! 83.3 ms and 12 results a second, is 41.7 ms on a 120 Hz display, a
//! different promise with half the frames for a job. The decision was about
//! time, so the page can be asked for it in time, and the depth model is
//! asked for it unless told otherwise; `latency` stays, in frames, for a run
//! that means frames.

use anyhow::{Context, Result, bail};
use beatlm_core::time::Micros;

use crate::depth::{MAX_SIDE, NATIVE_DEFAULT_LATENCY, NATIVE_DEFAULT_SIDE, PATCH};

/// Frames a run lasts when `frames` is not given: the bench's default.
pub const DEFAULT_FRAMES: u64 = 600;

/// The promised latency in frames of the synthetic workload when neither
/// `latency` nor `latency_ms` is given: the bench's default for it.
pub const DEFAULT_LATENCY: u32 = 2;

/// Slices in each frame's job of the synthetic workload: the bench's default
/// for it, which the page does not change.
pub const SLICES: u32 = 4;

/// The margin kept free in each frame's budget, in microseconds: the bench's
/// default, which the page does not change.
pub const MARGIN_US: u64 = 2_000;

/// The longest promised latency `latency_ms` accepts, in milliseconds: a
/// minute, far past any job here, so a typing slip is refused rather than
/// planned.
const MAX_LATENCY_MS: f64 = 60_000.0;

/// The parameters the query string accepts, as the error for an unknown one
/// lists them.
const PARAMETERS: &str = "frames, model, side, latency, latency_ms, arena_bytes";

/// The job each frame submits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Model {
    /// The bench's synthetic slices, `model=synthetic`.
    Synthetic,
    /// Depth Anything V2 Small at `side` pixels a side, `model=depth`.
    Depth {
        /// The input's side in pixels.
        side: u32,
    },
}

/// The promise of the fixed-latency policy, as the query gave it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Latency {
    /// A whole number of frames, `latency`.
    Frames(u32),
    /// A time, `latency_ms`, turned into frames at the display's period
    /// measured before the run.
    Time(Micros),
}

/// A run's configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    /// Frames to run, `frames`; at least one.
    pub frames: u64,
    /// The job, `model` and `side`.
    pub model: Model,
    /// The promised latency, `latency` or `latency_ms`.
    pub latency: Latency,
    /// `arena_bytes`: before the run, plan one arena of this many bytes
    /// against the adapter's limits, create the device with the limits it
    /// needs and realise it there, to check the limit path; `None` for a run
    /// that needs only the default limits.
    pub arena_bytes: Option<u64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            frames: DEFAULT_FRAMES,
            model: Model::Synthetic,
            latency: Latency::Frames(DEFAULT_LATENCY),
            arena_bytes: None,
        }
    }
}

/// Parses a query string such as `?frames=600&latency=2` or
/// `?model=depth&side=266&latency_ms=83.3`, with or without the leading `?`.
/// An empty query is the default configuration.
///
/// # Errors
///
/// An unknown or repeated parameter, one without a value, a value that is not
/// a number or not a model, zero frames or an arena of zero bytes, a side that
/// is not a positive multiple of 14 up to 518, `side` without `model=depth`,
/// `arena_bytes` with it, or both `latency` and `latency_ms`.
pub fn parse(query: &str) -> Result<Config> {
    let mut frames = DEFAULT_FRAMES;
    let mut depth = false;
    let mut side = None;
    let mut latency = None;
    let mut latency_ms = None;
    let mut arena_bytes = None;
    let mut seen: Vec<&str> = Vec::new();
    let query = query.strip_prefix('?').unwrap_or(query);
    for pair in query.split('&').filter(|pair| !pair.is_empty()) {
        let (name, value) = pair
            .split_once('=')
            .with_context(|| format!("query parameter {name:?} needs a value", name = pair))?;
        if seen.contains(&name) {
            bail!("query parameter {name:?} is given twice");
        }
        seen.push(name);
        match name {
            "frames" => frames = number(name, value)?,
            "model" => {
                depth = match value {
                    "synthetic" => false,
                    "depth" => true,
                    _ => bail!("query parameter model: {value:?} is not synthetic or depth"),
                };
            }
            "side" => side = Some(number::<u32>(name, value)?),
            "latency" => latency = Some(number(name, value)?),
            "latency_ms" => latency_ms = Some(milliseconds(value)?),
            "arena_bytes" => arena_bytes = Some(number(name, value)?),
            _ => bail!("unknown query parameter {name:?}; the page accepts {PARAMETERS}"),
        }
    }
    if frames == 0 {
        bail!("frames must be at least 1");
    }
    if arena_bytes == Some(0) {
        bail!("arena_bytes must be at least 1");
    }
    let model = if depth {
        if arena_bytes.is_some() {
            bail!(
                "arena_bytes checks the synthetic workload's limits; the depth model's plan sets its own"
            );
        }
        let side = side.unwrap_or(NATIVE_DEFAULT_SIDE);
        if side == 0 || side % PATCH != 0 || side > MAX_SIDE {
            bail!("side must be a positive multiple of {PATCH} up to {MAX_SIDE}, not {side}");
        }
        Model::Depth { side }
    } else {
        if side.is_some() {
            bail!("side is the depth model's input side; it needs model=depth");
        }
        Model::Synthetic
    };
    let latency = match (latency, latency_ms) {
        (Some(_), Some(_)) => bail!("give latency in frames or latency_ms in time, not both"),
        (Some(frames), None) => Latency::Frames(frames),
        (None, Some(time)) => Latency::Time(time),
        (None, None) => match model {
            Model::Synthetic => Latency::Frames(DEFAULT_LATENCY),
            Model::Depth { .. } => Latency::Time(NATIVE_DEFAULT_LATENCY),
        },
    };
    Ok(Config {
        frames,
        model,
        latency,
        arena_bytes,
    })
}

/// Parses the value of parameter `name` as a number.
fn number<T: std::str::FromStr>(name: &str, value: &str) -> Result<T> {
    value
        .parse()
        .ok()
        .with_context(|| format!("query parameter {name}: {value:?} is not a number in range"))
}

/// Parses `latency_ms`, a decimal number of milliseconds, into whole
/// microseconds.
fn milliseconds(value: &str) -> Result<Micros> {
    let ms: f64 = number("latency_ms", value)?;
    if !(0.0..=MAX_LATENCY_MS).contains(&ms) {
        bail!("query parameter latency_ms: {value:?} is not between 0 and {MAX_LATENCY_MS}");
    }
    // At most a minute in microseconds, well inside a u64; rounding keeps
    // `83.3` at 83 300 rather than one below it.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let us = (ms * 1_000.0).round() as u64;
    Ok(Micros::new(us))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_query_is_the_benchs_synthetic_defaults() {
        for query in ["", "?"] {
            assert_eq!(parse(query).unwrap(), Config::default());
        }
        assert_eq!(Config::default().frames, 600);
        assert_eq!(Config::default().latency, Latency::Frames(2));
        assert_eq!(Config::default().model, Model::Synthetic);
    }

    #[test]
    fn frames_latency_and_the_arena_are_read() {
        assert_eq!(
            parse("?frames=300&latency=5&arena_bytes=134217600").unwrap(),
            Config {
                frames: 300,
                model: Model::Synthetic,
                latency: Latency::Frames(5),
                arena_bytes: Some(134_217_600),
            }
        );
        assert_eq!(parse("latency=0").unwrap().latency, Latency::Frames(0));
        assert_eq!(parse("&frames=1&").unwrap().frames, 1);
        assert_eq!(
            parse("?model=synthetic&latency_ms=33.3").unwrap().latency,
            Latency::Time(Micros::new(33_300))
        );
    }

    #[test]
    fn the_depth_model_defaults_to_the_native_default_in_time() {
        let config = parse("?model=depth").unwrap();
        assert_eq!(config.model, Model::Depth { side: 266 });
        assert_eq!(config.latency, Latency::Time(Micros::new(83_335)));
        let config = parse("?model=depth&side=518&frames=1200&latency=40").unwrap();
        assert_eq!(config.model, Model::Depth { side: 518 });
        assert_eq!(config.latency, Latency::Frames(40));
        assert_eq!(config.frames, 1200);
        assert_eq!(
            parse("?latency_ms=83.3&side=182&model=depth").unwrap(),
            Config {
                frames: 600,
                model: Model::Depth { side: 182 },
                latency: Latency::Time(Micros::new(83_300)),
                arena_bytes: None,
            }
        );
    }

    #[test]
    fn anything_not_understood_is_refused() {
        for query in [
            "?frame=600",
            "?frames",
            "?frames=",
            "?frames=six",
            "?frames=-1",
            "?frames=0",
            "?latency=99999999999",
            "?arena_bytes=0",
            "?frames=1&frames=2",
            "?model=Depth",
            "?model=",
            "?side=266",
            "?model=synthetic&side=266",
            "?model=depth&side=0",
            "?model=depth&side=265",
            "?model=depth&side=532",
            "?model=depth&side=-14",
            "?model=depth&arena_bytes=1024",
            "?latency=5&latency_ms=83.3",
            "?latency_ms=-1",
            "?latency_ms=NaN",
            "?latency_ms=inf",
            "?latency_ms=60001",
            "?latency_ms=83,3",
        ] {
            assert!(parse(query).is_err(), "{query}");
        }
        let unknown = parse("?frame=600").unwrap_err().to_string();
        assert!(
            unknown.contains("frames, model, side, latency, latency_ms, arena_bytes"),
            "{unknown}"
        );
    }
}
