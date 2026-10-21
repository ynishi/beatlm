//! The depth model's side of the page that needs no browser: where its files
//! are fetched from, what calibration keeps, how a latency in time becomes
//! frames, and whether a run kept the native default.
//!
//! The browser side (`web/model.rs`, compiled only for `wasm32`) does the
//! fetching and the GPU work; the crate documentation ("The depth model")
//! describes the whole path.

use anyhow::{Context, Result};
use beatlm::load::depth_anything_v2::{Config, reference};
use beatlm_core::cost::SliceId;
use beatlm_core::schedule::{FramePlan, JobId};
use beatlm_core::time::Micros;

use crate::pace::NOMINAL_PERIOD;
use crate::stats::median;

/// The URL prefix the models directory is served under, relative to the page:
/// `just serve-web` maps it to the models directory
/// (`scripts/serve_web.py`), so the weights never enter the page's build
/// output.
pub const MODELS_URL: &str = "models/";

/// The model's patch side, in pixels: a side must be a multiple of it.
pub const PATCH: u32 = 14;

/// The largest side the page accepts, in pixels: the side the position
/// embedding is stored for, the largest the native tests compare and time.
pub const MAX_SIDE: u32 = 518;

/// The side of the native default, in pixels: `beatlm-bench`'s default for
/// the depth workload, chosen on an Apple M1 Pro (its crate documentation,
/// "The depth workload", "The decision").
pub const NATIVE_DEFAULT_SIDE: u32 = 266;

/// The latency of the native default in frames of the bench's fixed 60 Hz
/// grid.
pub const NATIVE_DEFAULT_FRAMES: u32 = 5;

/// The latency of the native default in time: [`NATIVE_DEFAULT_FRAMES`]
/// periods of the bench's grid, [`NOMINAL_PERIOD`], so 83.3 ms. The decision
/// was made for this time, 12 results a second, and the page holds a browser
/// run to the same time whatever the display's rate.
pub const NATIVE_DEFAULT_LATENCY: Micros =
    Micros::new(NATIVE_DEFAULT_FRAMES as u64 * NOMINAL_PERIOD.get());

/// The longest a dispatch should take, in microseconds, which calibration
/// splits kernels to: the bench's default `--max-dispatch-us`, which the
/// native default was measured with.
pub const MAX_DISPATCH_US: u64 = 1_000;

/// Timed frames of whole jobs in each calibration pass, after one untimed
/// frame: the bench's (`calibrate.rs`), twice the window of the core's
/// default estimator, so the half the decision reads is as long as the window
/// the loop's estimate reads.
pub const CALIBRATION_FRAMES: u64 = 64;

/// The share of deadlines that must arrive with their result for the native
/// default to hold: the lower of its two native runs, 0.9916, rounded down to
/// two places. A run below it misses more than the default was accepted with.
pub const DELIVERY_FLOOR: f64 = 0.99;

/// A period's fraction, as a divisor, by which a latency in time may fall
/// short of the frames promised for it ([`frames_for`]): a twentieth.
const PERIOD_SLACK: u64 = 20;

/// The URL of the weights file, relative to the page.
#[must_use]
pub fn weights_url() -> String {
    format!(
        "{MODELS_URL}{}/{}",
        reference::MODEL_DIR,
        reference::WEIGHTS_FILE
    )
}

/// The URL of the reference directory made at `side` pixels, relative to the
/// page, ending in `/`.
#[must_use]
pub fn reference_url(side: u32) -> String {
    let side = usize::try_from(side).unwrap_or(usize::MAX);
    format!(
        "{MODELS_URL}{}/{}/",
        reference::MODEL_DIR,
        reference::dir(&Config::small(), side)
    )
}

/// The `just` recipe that writes the reference at `side` pixels.
#[must_use]
pub fn reference_recipe(side: u32) -> String {
    let side = usize::try_from(side).unwrap_or(usize::MAX);
    reference::recipe(&Config::small(), side)
}

/// The frames a latency of `latency` takes at a display `period`: the fewest
/// whole periods that last at least `latency`, less a twentieth of a period.
///
/// The slack is there because the two sides of the division are whole
/// microseconds that stand for rates which are not: the native default is 5
/// periods of 16 667 µs, 83 335 µs, and a 120 Hz display measures 8 333 µs,
/// so the exact quotient is 10.0006 and a strict ceiling would promise 11
/// frames, 91.7 ms, for what is 10 frames of the display. A promise may so
/// come out up to a twentieth of a period shorter than asked, 0.4 ms at
/// 120 Hz, and never by a frame. A zero period gives no frames.
#[must_use]
pub fn frames_for(latency: Micros, period: Micros) -> u32 {
    let (latency, period) = (latency.get(), period.get());
    if period == 0 {
        return 0;
    }
    // ceil((latency - period / 20) / period), in integers.
    let scaled = latency.saturating_mul(PERIOD_SLACK);
    let whole = period.saturating_mul(PERIOD_SLACK);
    let frames = scaled.saturating_sub(period).div_ceil(whole);
    u32::try_from(frames).unwrap_or(u32::MAX)
}

/// The latency `frames` periods of `period` promise, in milliseconds.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn latency_ms(frames: u32, period: Micros) -> f64 {
    f64::from(frames) * period.get() as f64 / 1_000.0
}

/// Whether a calibration frame's measurements count: those of the second half
/// of the timed frames, as the bench's calibration keeps them, after the GPU
/// has settled into the paced load. `frame` counts from 1 for the first timed
/// frame.
#[must_use]
pub fn counted(frame: u64) -> bool {
    frame > CALIBRATION_FRAMES / 2
}

/// What calibration's first pass keeps: each kernel's counted measurements,
/// and each counted run's GPU time, the sum of its kernels.
#[derive(Debug, Clone, Default)]
pub struct WholeRuns {
    samples: Vec<Vec<u64>>,
    runs: Vec<u64>,
}

impl WholeRuns {
    /// A record for a program of `kernels` kernels.
    #[must_use]
    pub fn new(kernels: usize) -> Self {
        Self {
            samples: vec![Vec::new(); kernels],
            runs: Vec::new(),
        }
    }

    /// Keeps what frame `frame` measured, each slice being a kernel's index,
    /// if the frame [`counted`]. A run is kept only when every kernel of it
    /// was measured, so a run's time is the whole program's.
    pub fn record(&mut self, frame: u64, compute: &[(SliceId, Micros)]) {
        if !counted(frame) {
            return;
        }
        let mut total = 0u64;
        for (slice, duration) in compute {
            let list = usize::try_from(slice.get())
                .ok()
                .and_then(|i| self.samples.get_mut(i));
            if let Some(list) = list {
                list.push(duration.get());
            }
            total = total.saturating_add(duration.get());
        }
        if compute.len() == self.samples.len() {
            self.runs.push(total);
        }
    }

    /// The GPU time of each counted run, in microseconds.
    #[must_use]
    pub fn runs(&self) -> &[u64] {
        &self.runs
    }

    /// Each kernel's duration: the median of its counted measurements, the
    /// statistic the decision reads (`beatlm-bench`'s `calibrate.rs`).
    ///
    /// # Errors
    ///
    /// A kernel that was never measured in a counted frame.
    pub fn durations(&mut self) -> Result<Vec<Micros>> {
        self.samples
            .iter_mut()
            .enumerate()
            .map(|(kernel, list)| {
                median(list)
                    .with_context(|| format!("calibration measured kernel {kernel} no time"))
            })
            .collect()
    }
}

/// The depth model's job that holds back the next input after `plan`, the bench's
/// `Admission::planned` for `--submit free`: the job submitted last, `job`
/// with its slice count when this frame submitted one or `busy` when it did
/// not, less the slices `plan` handed out, until none are left or it has left
/// the scheduler, delivered or missed. `None` when no job holds back the
/// next input.
#[must_use]
pub fn still_busy(
    busy: Option<(JobId, usize)>,
    job: Option<(JobId, usize)>,
    plan: &FramePlan,
) -> Option<(JobId, usize)> {
    let (id, left) = job.or(busy)?;
    let handed_out = plan.runs.iter().filter(|run| run.job == id).count();
    let left = left.saturating_sub(handed_out);
    let gone =
        plan.delivered.iter().any(|d| d.job == id) || plan.missed.iter().any(|m| m.job == id);
    (left > 0 && !gone).then_some((id, left))
}

/// What a run measured that decides whether the native default held.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Facts {
    /// The input side, in pixels.
    pub side: u32,
    /// The latency promised, in frames.
    pub latency: u32,
    /// The display's period as measured over the run; `None` when it was not.
    pub period: Option<Micros>,
    /// The 99th percentile of the GPU time per frame, render and slices;
    /// `None` when no frame was timed.
    pub gpu_frame_p99: Option<u64>,
    /// Deliveries at another latency than the promised one.
    pub violations: u64,
    /// Results delivered.
    pub delivered: u64,
    /// Deadlines that arrived without a result.
    pub missed: u64,
    /// The part of `missed` that were jobs abandoned before their first slice
    /// ran ([`crate::diagnosis`]).
    pub unstarted: u64,
    /// The sum of a job's slice estimates when the loop began and when it
    /// ended, in microseconds.
    pub job_estimate_us: (u64, u64),
    /// The median GPU time of one whole run of the model in calibration, in
    /// microseconds; `None` when it was not measured.
    pub run_us: Option<u64>,
}

/// Whether the native default held in this run: `yes`, `no: ` with each
/// criterion it failed, or `not tested: ` with why the run cannot say.
///
/// A `no` for delivery also says, from this run's diagnosis, how many misses
/// were jobs abandoned before their first slice (none, since #41, for a job
/// with nothing behind it) and what a job was estimated
/// at when the loop began and ended against what a whole run measured: the
/// facts the crate documentation ("What a browser run measured") reads the
/// 120 Hz result from.
///
/// The run must be the native default, [`NATIVE_DEFAULT_SIDE`] at the frames
/// [`NATIVE_DEFAULT_LATENCY`] takes at the display's period
/// ([`frames_for`]), and then it holds by the native criteria
/// (`beatlm-bench`'s crate documentation, "The depth workload"): GPU time per
/// frame within the display's period at the 99th percentile, no delivery at
/// another latency, and at least [`DELIVERY_FLOOR`] of the deadlines met.
#[must_use]
pub fn native_default(facts: &Facts) -> String {
    let Some(period) = facts.period else {
        return "not tested: the display's period was not measured".to_owned();
    };
    let wanted = frames_for(NATIVE_DEFAULT_LATENCY, period);
    if facts.side != NATIVE_DEFAULT_SIDE || facts.latency != wanted {
        return format!(
            "not tested: this run is size={} at latency={} ({:.1} ms); the native default is \
             size={NATIVE_DEFAULT_SIDE} at latency={wanted} ({:.1} ms) at this display's period",
            facts.side,
            facts.latency,
            latency_ms(facts.latency, period),
            latency_ms(wanted, period),
        );
    }
    let Some(p99) = facts.gpu_frame_p99 else {
        return "not tested: no frame's GPU time was measured".to_owned();
    };
    let deadlines = facts.delivered.saturating_add(facts.missed);
    if deadlines == 0 {
        return "not tested: no deadline arrived".to_owned();
    }
    let mut failed = Vec::new();
    if p99 > period.get() {
        failed.push(format!(
            "gpu_frame_us p99 {p99} is over the period {}",
            period.get()
        ));
    }
    if facts.violations > 0 {
        failed.push(format!("{} latency violations", facts.violations));
    }
    #[allow(clippy::cast_precision_loss)]
    let rate = facts.delivered as f64 / deadlines as f64;
    if rate < DELIVERY_FLOOR {
        failed.push(format!("delivery_rate {rate:.4} is below {DELIVERY_FLOOR}"));
        // Since #41 a job of the `--submit free` admission is no longer
        // abandoned before its first slice, so its misses are at the deadline
        // and `unstarted` is 0: the estimates are said either way.
        let run = facts
            .run_us
            .map_or_else(|| "unmeasured".to_owned(), |us| format!("{us} us"));
        failed.push(format!(
            "{} of {} misses were jobs abandoned before their first slice; a job was \
             estimated at {} us when the loop began and {} us when it ended, against a \
             whole run of {run} in calibration",
            facts.unstarted, facts.missed, facts.job_estimate_us.0, facts.job_estimate_us.1
        ));
    }
    if failed.is_empty() {
        "yes".to_owned()
    } else {
        format!("no: {}", failed.join("; "))
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn the_native_default_is_83_ms_at_266_pixels() {
        assert_eq!(NATIVE_DEFAULT_LATENCY, Micros::new(83_335));
        assert_eq!(NATIVE_DEFAULT_SIDE % PATCH, 0);
        assert_eq!(MAX_SIDE % PATCH, 0);
    }

    #[test]
    fn a_latency_in_time_takes_the_frames_that_last_that_long() {
        let at = |us, period| frames_for(Micros::new(us), Micros::new(period));
        // The native default: 5 frames at 60 Hz, 10 at 120 Hz, 12 at 144 Hz,
        // 5 on a 59.94 Hz panel, whose period is a little longer.
        assert_eq!(at(83_335, 16_667), 5);
        assert_eq!(at(83_335, 8_333), 10);
        assert_eq!(at(83_335, 8_334), 10);
        assert_eq!(at(83_335, 6_944), 12);
        assert_eq!(at(83_335, 16_683), 5);
        // Typed as 83.3 ms it is the same.
        assert_eq!(at(83_300, 8_333), 10);
        // A latency between whole periods rounds up, past the slack.
        assert_eq!(at(83_335, 11_111), 8);
        assert_eq!(at(8_800, 8_333), 2);
        assert_eq!(at(8_333, 8_333), 1);
        assert_eq!(at(0, 8_333), 0);
        assert_eq!(at(83_335, 0), 0);
        assert!((latency_ms(10, Micros::new(8_333)) - 83.33).abs() < 1e-9);
    }

    #[test]
    fn the_files_are_under_the_models_prefix() {
        assert_eq!(
            weights_url(),
            "models/depth-anything-v2-small/model.safetensors"
        );
        assert_eq!(
            reference_url(266),
            "models/depth-anything-v2-small/reference/266/"
        );
        assert_eq!(
            reference_url(518),
            "models/depth-anything-v2-small/reference/"
        );
        assert_eq!(reference_recipe(266), "reference-depth-anything 266");
    }

    fn timing(durations: &[u64]) -> Vec<(SliceId, Micros)> {
        durations
            .iter()
            .enumerate()
            .map(|(i, &d)| (SliceId::new(i as u32), Micros::new(d)))
            .collect()
    }

    #[test]
    fn calibration_keeps_the_second_half_and_takes_each_kernels_median() {
        let mut runs = WholeRuns::new(2);
        // The first half is the GPU settling: none of it counts.
        runs.record(1, &timing(&[9_000, 9_000]));
        runs.record(CALIBRATION_FRAMES / 2, &timing(&[9_000, 9_000]));
        assert!(runs.runs().is_empty());
        runs.record(CALIBRATION_FRAMES / 2 + 1, &timing(&[100, 300]));
        runs.record(CALIBRATION_FRAMES, &timing(&[120, 200]));
        runs.record(CALIBRATION_FRAMES, &timing(&[110, 250]));
        // A frame missing a kernel feeds the kernels it has, not a run.
        runs.record(CALIBRATION_FRAMES, &timing(&[105]));
        assert_eq!(runs.runs(), &[400, 320, 360]);
        assert_eq!(
            runs.durations().unwrap(),
            vec![Micros::new(110), Micros::new(250)]
        );
        let mut empty = WholeRuns::new(1);
        assert!(empty.durations().is_err());
    }

    fn facts() -> Facts {
        Facts {
            side: 266,
            latency: 10,
            period: Some(Micros::new(8_333)),
            gpu_frame_p99: Some(7_900),
            violations: 0,
            delivered: 99,
            missed: 1,
            unstarted: 0,
            job_estimate_us: (30_000, 30_000),
            run_us: Some(28_700),
        }
    }

    #[test]
    fn the_native_default_holds_by_the_native_criteria() {
        assert_eq!(native_default(&facts()), "yes");
        let over = Facts {
            gpu_frame_p99: Some(8_400),
            violations: 2,
            delivered: 97,
            missed: 3,
            ..facts()
        };
        let said = native_default(&over);
        assert!(said.starts_with("no: gpu_frame_us p99 8400"), "{said}");
        assert!(said.contains("2 latency violations"), "{said}");
        assert!(said.contains("delivery_rate 0.9700"), "{said}");
        // Missed at the deadline, none abandoned before it started: the
        // estimates are still said.
        assert!(
            said.contains("0 of 3 misses were jobs abandoned before their first slice"),
            "{said}"
        );
        let starved = Facts {
            delivered: 31,
            missed: 2_561,
            unstarted: 2_543,
            job_estimate_us: (34_236, 101_984),
            run_us: Some(28_716),
            ..facts()
        };
        let said = native_default(&starved);
        assert!(
            said.contains("delivery_rate 0.0120 is below 0.99"),
            "{said}"
        );
        assert!(
            said.contains(
                "2543 of 2561 misses were jobs abandoned before their first slice; a job was \
                 estimated at 34236 us when the loop began and 101984 us when it ended, against \
                 a whole run of 28716 us in calibration"
            ),
            "{said}"
        );
    }

    #[test]
    fn a_run_that_is_not_the_native_default_says_what_it_would_need() {
        let other = Facts {
            latency: 5,
            ..facts()
        };
        let said = native_default(&other);
        assert!(
            said.starts_with("not tested: this run is size=266 at latency=5"),
            "{said}"
        );
        assert!(said.contains("latency=10 (83.3 ms)"), "{said}");
        let at_60 = Facts {
            latency: 5,
            period: Some(Micros::new(16_667)),
            ..facts()
        };
        assert_eq!(native_default(&at_60), "yes");
        for unknown in [
            Facts {
                period: None,
                ..facts()
            },
            Facts {
                gpu_frame_p99: None,
                ..facts()
            },
            Facts {
                delivered: 0,
                missed: 0,
                ..facts()
            },
            Facts {
                side: 518,
                ..facts()
            },
        ] {
            assert!(
                native_default(&unknown).starts_with("not tested: "),
                "{unknown:?}"
            );
        }
    }
}
