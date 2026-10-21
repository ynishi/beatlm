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
//! - `estimator`: the statistic the cost model estimates a slice with, the
//!   core's [`Estimator`], in the forms the next section lists. Default: the
//!   core's ([`Estimator::default`]).
//!
//! # The estimator parameter
//!
//! The value is a name and, after colons, its parameters. The bare names are
//! the bench's `--estimator` spellings (`quantile`, `mean-sigmas`, `capped`),
//! each meaning exactly the bench's parameters. The report's `config:` line
//! prints (`estimator=`, by [`estimator_name`]) the bare name when the
//! estimator is exactly what it means, as the bench's does, so the default
//! prints `estimator=quantile`, and the parameterised form otherwise; either
//! can be pasted back into a URL.
//!
//! | Value | The core's estimator |
//! |---|---|
//! | `quantile` | `WindowedQuantile`, quantile 0.9, window [`DEFAULT_WINDOW`] (32): the core's default |
//! | `quantile:<q>` |`WindowedQuantile`, window [`DEFAULT_WINDOW`] (32), quantile `q`, a number in `(0, 1]` |
//! | `quantile:<q>:<window>` | the same with a window of `window` measurements, `1..=`[`MAX_WINDOW`] |
//! | `mean-sigmas` | `MeanPlusSigmas`, `alpha` [`COST_ALPHA`] (0.2), `sigmas` [`COST_SIGMAS`] (2) |
//! | `mean-sigmas:<alpha>:<sigmas>` | the same with `alpha` in `(0, 1]` and `sigmas` not negative |
//! | `capped` | `CappedDeviation`, `alpha` 0.2, `sigmas` 2, `cap` [`COST_CAP`] (0.5) |
//! | `capped:<alpha>:<sigmas>:<cap>` | the same with `cap` not negative |
//!
//! The parameters of the bare names are the ones the core's choice of
//! estimator was measured with (`beatlm_core::cost`, "Choosing the
//! estimator"), which are also the bench's. The old spelling `mean` is
//! refused. A quantile of 0
//! (the window's minimum) is refused here although the core accepts it, since
//! it plans every slice for its fastest measurement. The remaining ranges are
//! the core's, and its error is quoted when one is out of range.
//!
//! It applies to both jobs: the cost model the loop plans with is built with
//! it. For the depth model that model is the one calibration measured the
//! split parts into, so calibration builds it with the estimator, as the
//! bench's does with `--estimator`; calibration's decision, how each kernel is
//! split, reads the median of its whole-kernel measurements and never the cost
//! model, so it is the same whichever estimator is chosen
//! ([`crate::depth::WholeRuns::durations`]). The render pass's own cost model
//! ([`crate::render_cost`]) is not an estimator choice and does not change.
//!
//! # Which estimator, measured in a browser
//!
//! The core chose its default on the native bench at 60 Hz ("Choosing the
//! estimator" in `beatlm_core::cost`). This is the same comparison in a
//! browser at 120 Hz, where the default does not keep the promise
//! ([`crate`], "What a browser run measured").
//!
//! Chrome 154 (WebGPU) on macOS, Apple M1 Pro, 120 Hz display (period
//! 8 300 µs), 2026-10-22, Depth Anything V2 Small at 266 pixels,
//! `?model=depth&side=266&frames=1500&latency_ms=200&estimator=<e>` (the URLs
//! said `quantile:0.5`, `quantile:0.75`, `quantile:0.9` and `capped`, and the
//! `config:` line of that build printed them in full; the table writes each
//! estimator as the `config:` line prints it now, the default as the bare
//! `quantile`, the same estimators): 25 frames,
//! about 207 ms (24 frames for `quantile:0.75:32`, whose run measured the
//! period as 8 400 µs at its start), free admission, margin 2 ms, render 50
//! iterations, one run per estimator, after #41 (a lone late job is kept). No
//! run was discarded.
//!
//! | Estimator | Submitted | Delivered | Missed | Rate | GPU frames over the period | `gpu_frame_us` p50 / p99 |
//! |---|---|---|---|---|---|---|
//! | `quantile:0.5:32` | 119 | 114 | 4 | 0.9661 | 408 of 1477 | 7 343 / 11 862 |
//! | `quantile:0.75:32` | 76 | 31 | 44 | 0.4133 | 165 of 1497 | 6 423 / 10 356 |
//! | `quantile` (default; 0.9, window 32) | 68 | 19 | 48 | 0.2836 | 57 of 1496 | 4 917 / 9 701 |
//! | `capped` (0.2, 2, 0.5) | 61| 7 | 53 | 0.1167 | 26 of 1486 | 4 523 / 8 848 |
//!
//! From the `quantile:0.5:32` run, the lines [`crate::diagnosis`] defines,
//! with the fields not quoted here elided as `…`:
//!
//! ```text
//! capacity_frames_needed: in_loop_run_us=68281 slice_budget_us=4631 frames=15 frames_promised=25 frames_available=26 … verdict=not_ruled_out
//! tail_loop_over_calibration: … median_mean_ratio=2.05 median_p50_ratio=1.57
//! tail_outlier_frames: … at_least_half_outliers=175 expected_if_independent=26.7
//! ```
//!
//! From an earlier 3000-frame run at the native default's 83 ms (10 frames)
//! with the default estimator: `tail_loop_over_calibration
//! median_mean_ratio=1.78 median_p50_ratio=1.50`; outlier frames 185 against
//! 98.5 expected if independent; first passes were 0.233 of the outliers
//! against 0.246 of the samples (H1, that first passes are the slow ones, is
//! refuted); the share of measurements that were outliers rose with the
//! previous frame's busyness, from 0.054 (0 to 25%) to 0.200 (75% and over).
//!
//! What the numbers say, and what they do not (one run per cell throughout):
//!
//! - **No estimator here keeps both promises.** The lighter estimates deliver
//!   (`quantile:0.5:32`: 0.97) by packing more into a frame than the frame
//!   holds: the GPU time of 28% of its frames exceeded the 8.3 ms period,
//!   which breaks the frame budget the renderer shares, the engine's first
//!   promise. The heavier estimates keep frames within the period (`capped`
//!   1.7%, `quantile:0.9:32` 3.8%) and deliver little.
//! - **Why, as far as these runs show.** The per-slice estimates are summed
//!   for a frame, but slow measurements cluster by frame (the frames with
//!   half or more of their measurements slow were 1.9 and 6.6 times as many
//!   as independent measurements would give, in the two runs above), so the
//!   sum of per-slice quantiles is too high on average and still not a bound
//!   on the slow frames. The clustering is measured; that it is the whole cause is not.
//! - **The default stays `quantile:0.9:32`.** The frame budget is the promise
//!   that must hold; delivery at 120 Hz is left to an estimate made for the
//!   frame rather than the slice (a follow-up issue).
//! - **At 83 ms (10 frames) the native default is infeasible on this display
//!   whatever the estimator.** The `quantile:0.5:32` run measured a run of the
//!   model in the loop of 68 ms against about 4.6 ms of slice budget per frame:
//!   15 frames needed against the 11 that 83 ms gives. That compares one
//!   run's figures with the 11 frames; it is not a run at 83 ms.
//! - **`capped` delivering least, though its estimates are closest to the
//!   means, is one run and unexplained.** Do not rank `capped` against
//!   `quantile:0.9:32` on it.
//!
//! **Why a latency in time.** A promise in frames is worth the display's
//! period, so the native default of 5 frames, a decision made at 60 Hz for
//! 83.3 ms and 12 results a second, is 41.7 ms on a 120 Hz display, a
//! different promise with half the frames for a job. The decision was about
//! time, so the page can be asked for it in time, and the depth model is
//! asked for it unless told otherwise; `latency` stays, in frames, for a run
//! that means frames.

use anyhow::{Context, Result, anyhow, bail};
use beatlm_core::cost::{CostModel, Estimator, MAX_WINDOW};
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

/// The window `estimator=quantile` and `quantile:<q>` read: the core's
/// default, and the bench's `COST_WINDOW`.
pub const DEFAULT_WINDOW: usize = 32;

/// The weight of a new measurement of `estimator=mean-sigmas` and `capped`:
/// the value the core's estimators were compared with, and the bench's.
pub const COST_ALPHA: f64 = 0.2;

/// The standard deviations `estimator=mean-sigmas` and `capped` add to the
/// mean.
pub const COST_SIGMAS: f64 = 2.0;

/// The most `estimator=capped` pads the mean by, as a multiple of it.
pub const COST_CAP: f64 = 0.5;

/// The parameters the query string accepts, as the error for an unknown one
/// lists them.
const PARAMETERS: &str = "frames, model, side, latency, latency_ms, arena_bytes, estimator";

/// The forms `estimator` accepts, as the error for another value lists them.
const ESTIMATOR_FORMS: &str = "quantile, quantile:<q>, quantile:<q>:<window>, mean-sigmas, \
                               mean-sigmas:<alpha>:<sigmas>, capped or capped:<alpha>:<sigmas>:<cap>";

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
#[derive(Debug, Clone, Copy, PartialEq)]
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
    /// `estimator`: the statistic the loop's cost model estimates a slice
    /// with.
    pub estimator: Estimator,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            frames: DEFAULT_FRAMES,
            model: Model::Synthetic,
            latency: Latency::Frames(DEFAULT_LATENCY),
            arena_bytes: None,
            estimator: Estimator::default(),
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
/// `arena_bytes` with it, both `latency` and `latency_ms`, or an `estimator`
/// that is not one of the forms [`estimator`] reads or has a parameter out of
/// range.
pub fn parse(query: &str) -> Result<Config> {
    let mut frames = DEFAULT_FRAMES;
    let mut depth = false;
    let mut side = None;
    let mut latency = None;
    let mut latency_ms = None;
    let mut arena_bytes = None;
    let mut chosen = None;
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
            "estimator" => chosen = Some(estimator(value)?),
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
        estimator: chosen.unwrap_or_default(),
    })
}

/// Parses the value of `estimator`: a name and its parameters after colons
/// (the module documentation has the table).
///
/// # Errors
///
/// A name or number of parameters that is none of the forms, a parameter that
/// is not a number, a quantile outside `(0, 1]`, a window outside
/// `1..=`[`MAX_WINDOW`], or an `alpha`, `sigmas` or `cap` the core refuses.
pub fn estimator(value: &str) -> Result<Estimator> {
    let mut parts = value.split(':');
    let kind = parts.next().unwrap_or_default();
    let args: Vec<&str> = parts.collect();
    let float = |text: &str| number::<f64>("estimator", text);
    let chosen = match (kind, args.as_slice()) {
        ("quantile", []) => bare_quantile(),
        ("mean-sigmas", []) => bare_mean_sigmas(),
        ("capped", []) => bare_capped(),
        ("quantile", [q]) => Estimator::WindowedQuantile {
            window: DEFAULT_WINDOW,
            quantile: float(q)?,
        },
        ("quantile", [q, window]) => Estimator::WindowedQuantile {
            window: number("estimator", window)?,
            quantile: float(q)?,
        },
        ("mean-sigmas", [alpha, sigmas]) => Estimator::MeanPlusSigmas {
            alpha: float(alpha)?,
            sigmas: float(sigmas)?,
        },
        ("capped", [alpha, sigmas, cap]) => Estimator::CappedDeviation {
            alpha: float(alpha)?,
            sigmas: float(sigmas)?,
            cap: float(cap)?,
        },
        _ => bail!("query parameter estimator: {value:?} is not {ESTIMATOR_FORMS}"),
    };
    if let Estimator::WindowedQuantile { window, quantile } = chosen {
        // `quantile > 0.0` is false for NaN, and `<= 1.0` for infinity.
        if !(quantile > 0.0 && quantile <= 1.0) {
            bail!("query parameter estimator: the quantile {quantile} is not in (0, 1]");
        }
        if !(1..=MAX_WINDOW).contains(&window) {
            bail!("query parameter estimator: the window {window} is not in 1..={MAX_WINDOW}");
        }
    }
    // The core owns the ranges of the other parameters; building a model is
    // how its check is asked.
    CostModel::new(Micros::ZERO, chosen)
        .map_err(|error| anyhow!("query parameter estimator: {value:?}: {error}"))?;
    Ok(chosen)
}

/// The estimator the bare `quantile` names: the core's default, which is the
/// bench's `--estimator quantile` (quantile 0.9, window [`DEFAULT_WINDOW`]).
const fn bare_quantile() -> Estimator {
    Estimator::WindowedQuantile {
        window: DEFAULT_WINDOW,
        quantile: 0.9,
    }
}

/// The estimator the bare `mean-sigmas` names, the bench's.
const fn bare_mean_sigmas() -> Estimator {
    Estimator::MeanPlusSigmas {
        alpha: COST_ALPHA,
        sigmas: COST_SIGMAS,
    }
}

/// The estimator the bare `capped` names, the bench's.
const fn bare_capped() -> Estimator {
    Estimator::CappedDeviation {
        alpha: COST_ALPHA,
        sigmas: COST_SIGMAS,
        cap: COST_CAP,
    }
}

/// `estimator` in a form [`estimator`] reads back: the bench's bare name when
/// the estimator is exactly what that name means, else the name with every
/// parameter written. What the report's `config:` line prints.
#[must_use]
pub fn estimator_name(estimator: Estimator) -> String {
    if estimator == bare_quantile() {
        return "quantile".to_owned();
    }
    if estimator == bare_mean_sigmas() {
        return "mean-sigmas".to_owned();
    }
    if estimator == bare_capped() {
        return "capped".to_owned();
    }
    match estimator {
        Estimator::MeanPlusSigmas { alpha, sigmas } => format!("mean-sigmas:{alpha}:{sigmas}"),
        Estimator::WindowedQuantile { window, quantile } => format!("quantile:{quantile}:{window}"),
        Estimator::CappedDeviation { alpha, sigmas, cap } => {
            format!("capped:{alpha}:{sigmas}:{cap}")
        }
    }
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
                estimator: Estimator::default(),
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
                estimator: Estimator::default(),
            }
        );
    }

    #[test]
    fn the_estimator_defaults_to_the_cores_and_applies_to_both_jobs() {
        assert_eq!(Config::default().estimator, Estimator::default());
        assert_eq!(
            parse("?model=depth").unwrap().estimator,
            Estimator::default()
        );
        // The forms spelled out for the default are the default; if the core's
        // default moves, this is where it shows.
        assert_eq!(estimator("quantile").unwrap(), Estimator::default());
        assert_eq!(estimator("quantile:0.9").unwrap(), Estimator::default());
        assert_eq!(estimator("quantile:0.9:32").unwrap(), Estimator::default());
        for query in [
            "?estimator=capped",
            "?model=depth&estimator=capped",
            "?estimator=capped&side=266&model=depth&frames=300",
        ] {
            assert_eq!(
                parse(query).unwrap().estimator,
                Estimator::CappedDeviation {
                    alpha: 0.2,
                    sigmas: 2.0,
                    cap: 0.5
                },
                "{query}"
            );
        }
        // Colons need no escaping in a query string.
        let config = parse("?model=depth&estimator=quantile:0.75:16&latency=10").unwrap();
        assert_eq!(
            config.estimator,
            Estimator::WindowedQuantile {
                window: 16,
                quantile: 0.75
            }
        );
        assert_eq!(config.model, Model::Depth { side: 266 });
    }

    #[test]
    fn every_form_of_the_estimator_is_read() {
        let quantile = |quantile, window| Estimator::WindowedQuantile { window, quantile };
        assert_eq!(estimator("quantile:0.75").unwrap(), quantile(0.75, 32));
        assert_eq!(estimator("quantile:1").unwrap(), quantile(1.0, 32));
        assert_eq!(estimator("quantile:0.5:1").unwrap(), quantile(0.5, 1));
        assert_eq!(estimator("quantile:.5:4096").unwrap(), quantile(0.5, 4096));
        assert_eq!(
            estimator("mean-sigmas").unwrap(),
            Estimator::MeanPlusSigmas {
                alpha: 0.2,
                sigmas: 2.0
            }
        );
        assert_eq!(
            estimator("mean-sigmas:1:0").unwrap(),
            Estimator::MeanPlusSigmas {
                alpha: 1.0,
                sigmas: 0.0
            }
        );
        assert_eq!(
            estimator("capped:0.1:3:0").unwrap(),
            Estimator::CappedDeviation {
                alpha: 0.1,
                sigmas: 3.0,
                cap: 0.0
            }
        );
    }

    #[test]
    fn an_estimator_not_understood_or_out_of_range_is_refused() {
        for value in [
            "",
            "quantile:",
            "quantile:0",
            "quantile:-0.5",
            "quantile:1.01",
            "quantile:NaN",
            "quantile:inf",
            "quantile:p90",
            "quantile:0.9:",
            "quantile:0.9:0",
            "quantile:0.9:4097",
            "quantile:0.9:-1",
            "quantile:0.9:3.5",
            "quantile:0.9:32:1",
            "mean",
            "mean:0.2:2",
            "mean-sigmas:0.2",
            "mean-sigmas:0:2",
            "mean-sigmas:1.5:2",
            "mean-sigmas:0.2:-1",
            "mean-sigmas:0.2:2:3",
            "capped:",
            "capped:0.2:2",
            "capped:0.2:2:-0.1",
            "capped:0.2:2:inf",
            "capped:0.2:2:0.5:1",
            "Capped",
            "Quantile",
            "mean-sigmas:",
            "quantile%3A0.9",
        ] {
            assert!(estimator(value).is_err(), "{value:?}");
            let query = format!("?estimator={value}");
            assert!(parse(&query).is_err(), "{query}");
        }
        assert!(parse("?estimator").is_err());
        assert!(parse("?estimator=mean-sigmas&estimator=capped").is_err());
        let said = estimator("quantile:0").unwrap_err().to_string();
        assert!(said.contains("not in (0, 1]"), "{said}");
        let said = estimator("quantile:0.9:4097").unwrap_err().to_string();
        assert!(said.contains("1..=4096"), "{said}");
        let said = estimator("mean-sigmas:0:2").unwrap_err().to_string();
        assert!(
            said.contains("alpha must be finite and in (0, 1]"),
            "{said}"
        );
        let said = estimator("p99").unwrap_err().to_string();
        assert!(said.contains("quantile:<q>:<window>"), "{said}");
    }

    #[test]
    fn the_estimator_is_named_in_a_form_that_reads_back() {
        assert_eq!(estimator_name(Estimator::default()), "quantile");
        for (value, name) in [
            ("quantile", "quantile"),
            ("quantile:0.9", "quantile"),
            ("quantile:0.9:32", "quantile"),
            ("quantile:0.75", "quantile:0.75:32"),
            ("quantile:0.75:16", "quantile:0.75:16"),
            ("quantile:0.9:16", "quantile:0.9:16"),
            ("quantile:1", "quantile:1:32"),
            ("mean-sigmas", "mean-sigmas"),
            ("mean-sigmas:0.2:2", "mean-sigmas"),
            ("mean-sigmas:0.3:2", "mean-sigmas:0.3:2"),
            ("capped", "capped"),
            ("capped:0.2:2:0.5", "capped"),
            ("capped:0.2:2:0.6", "capped:0.2:2:0.6"),
            ("capped:0.25:1.5:0.75", "capped:0.25:1.5:0.75"),
        ] {
            let chosen = estimator(value).unwrap();
            assert_eq!(estimator_name(chosen), name, "{value}");
            assert_eq!(estimator(name).unwrap(), chosen, "{name}");
        }
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
            unknown.contains("frames, model, side, latency, latency_ms, arena_bytes, estimator"),
            "{unknown}"
        );
    }
}
