//! The depth model's side of the page that needs no browser: where its files
//! are fetched from, what calibration keeps, how a latency in time becomes
//! frames, and whether a run kept the native default.
//!
//! The browser side (`web/model.rs`, compiled only for `wasm32`) does the
//! fetching and the GPU work; the crate documentation ("The depth model")
//! describes the whole path.

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use beatlm::load::depth_anything_v2::{Config, Role, reference};
use beatlm::model::Builder;
use beatlm::model::depth_anything_v2::model;
use beatlm_core::cost::SliceId;
use beatlm_core::kernel::{Fusion, Program};
use beatlm_core::memory::{MemoryPlan, TensorId};
use beatlm_core::schedule::{FramePlan, JobId};
use beatlm_core::time::Micros;

use crate::pace::NOMINAL_PERIOD;
use crate::slots::{self, Numbers};
use crate::stats::median;
use crate::workload::{DEFAULT_STAGE_STORAGE_BUFFERS, RenderOrder, written_fits};

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
/// the loop's estimate reads. An `estimator` with a longer window starts the
/// loop with the counted half, which is [`CALIBRATION_FRAMES`] / 2
/// measurements of each part, and fills the rest from the loop.
pub const CALIBRATION_FRAMES: u64 = 64;

/// The share of deadlines that must arrive with their result for the native
/// default to hold: the lower of its two native runs, 0.9916, rounded down to
/// two places. A run below it misses more than the default was accepted with.
pub const DELIVERY_FLOOR: f64 = 0.99;

/// A period's fraction, as a divisor, by which a latency in time may fall
/// short of the frames promised for it ([`frames_for`]): a twentieth.
const PERIOD_SLACK: u64 = 20;

/// The model at one side, built and lowered but without its numbers: what
/// the page knows before it fetches the weights, which is enough to plan its
/// memory and so to refuse a run the plan cannot serve before the download.
pub struct Graph {
    /// The side in pixels.
    pub side: u32,
    /// The builder the graph was made in, which lists the weights to read
    /// from the file once it is fetched.
    pub builder: Builder<Role>,
    /// The model lowered with fusion, not split.
    pub program: Program,
    /// The input image, a job's input.
    pub image: TensorId,
    /// The depth, a job's output.
    pub depth: TensorId,
}

impl Graph {
    /// The memory plan of the unsplit program with `slots` slots
    /// ([`slots::memory_plan`]), the one the device is created for.
    ///
    /// The plan reads only which tensors are a job's input and output, so it
    /// is made here from the ids, before any number exists.
    ///
    /// # Errors
    ///
    /// No slots or too many, or tensors that cannot be planned.
    pub fn memory_plan(&self, slots: u32) -> Result<MemoryPlan> {
        let ids = Numbers {
            input: self.image,
            output: self.depth,
            weights: Vec::new(),
            slot_input: Box::new(|_| Vec::new()),
        };
        slots::memory_plan(&self.program, &ids, slots)
    }
}

/// Whether `order` can be honoured for `plan`, the memory plan of the model
/// at `side` pixels with `slots` slots, before the device exists: under
/// [`RenderOrder::After`] the display reads every arena of the plan in its
/// vertex stage, and a device created with the limits the plan needs binds
/// at most [`DEFAULT_STAGE_STORAGE_BUFFERS`] storage buffers there
/// (`workload.rs`, "Ordering the render pass after the slices"). The page
/// asks before it fetches the weights, so a run that cannot be honoured
/// fails before the download.
///
/// # Errors
///
/// [`RenderOrder::After`] for a plan of more arenas than that, with what to
/// change.
pub fn check_render_order(
    order: RenderOrder,
    plan: &MemoryPlan,
    side: u32,
    slots: u32,
) -> Result<()> {
    if order == RenderOrder::Free {
        return Ok(());
    }
    let arenas = plan.arenas().len();
    written_fits(arenas, DEFAULT_STAGE_STORAGE_BUFFERS).with_context(|| {
        format!(
            "render_order=after cannot be honoured at side={side} with {slots} slots, whose memory \
             plan has {arenas} arenas: add render_order=free, or ask for a shorter latency or a \
             smaller side, whose plan has fewer"
        )
    })
}

/// The model at `side` pixels, built and lowered with fusion, without its
/// weights.
///
/// # Errors
///
/// A side the model cannot be built at, or a graph that does not lower.
pub fn graph(side: u32) -> Result<Graph> {
    let pixels = usize::try_from(side).context("the side does not fit this machine")?;
    let mut builder = Builder::new();
    let tensors = model(&mut builder, &Config::small(), pixels)
        .with_context(|| format!("cannot build Depth Anything V2 Small at {side} pixels"))?;
    let program = builder
        .graph()
        .lower(Fusion::On)
        .context("cannot lower the depth model")?;
    Ok(Graph {
        side,
        builder,
        program,
        image: tensors.image,
        depth: tensors.depth,
    })
}

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

/// The job whose depth the display copies in the frame of `plan`: the last
/// one `plan` delivers, `None` when it delivers none. The frame loop records
/// the copy after the frame's slices and before its render pass, so a result
/// is on screen from the frame that delivers it and never before
/// (`web/display.rs` says why the screen is held to that).
///
/// The job's output slot still holds it then. A slot's output is live for the
/// whole program and shares bytes with nothing else (`slots.rs`, "Jobs, slots
/// and the shared plan"), so only a later job on the same slot could
/// overwrite it, and none has had a slice by the copy:
///
/// - Under `FixedLatency { frames: k }` there are `k + 1` slots and at most
///   one submission a frame, so the job `k + 1` after job `j` is submitted at
///   least `k + 1` frames after `j`'s, past the frame `k` after it, the only
///   one that can deliver `j`.
/// - On beat, a job is not submitted while the job [`crate::onbeat::SLOTS`]
///   before it is pending, and a job stays pending until the plan that
///   delivers or misses it has been recorded, after that frame's submission
///   ([`crate::onbeat::Beats::slot_free`]).
#[must_use]
pub fn delivered_now(plan: &FramePlan) -> Option<JobId> {
    plan.delivered.last().map(|delivery| delivery.job)
}

/// The page's GPU memory as the frame loop writes it, for the tests of what
/// the display shows: which job each output slot was last written by, and
/// which job the display's buffer holds. It also follows the rule the copy at
/// delivery replaced, a copy into a buffer per slot after a job's last slice
/// with the slot of the job delivered last drawn, and counts the frames that
/// rule drew a job not yet delivered.
#[cfg(test)]
pub(crate) struct Screen {
    /// Per output slot, the job whose slice wrote it last.
    written: Vec<Option<u64>>,
    /// The job the display's buffer holds, copied at its delivery.
    shown: Option<u64>,
    /// The replaced rule's buffers: per slot, the job copied after its last
    /// slice.
    kept: Vec<Option<u64>>,
    /// The slot the replaced rule drew: the delivered last's.
    kept_slot: Option<usize>,
    /// Every job delivered so far.
    delivered: std::collections::BTreeSet<u64>,
    /// Frames in which the replaced rule drew a job not yet delivered.
    pub(crate) early_under_kept: u64,
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
impl Screen {
    /// Memory for `slots` output slots, nothing written and nothing shown.
    pub(crate) fn new(slots: u32) -> Self {
        let slots = usize::try_from(slots).unwrap();
        Self {
            written: vec![None; slots],
            shown: None,
            kept: vec![None; slots],
            kept_slot: None,
            delivered: std::collections::BTreeSet::new(),
            early_under_kept: 0,
        }
    }

    /// Runs the frame of `plan`, for jobs of `slices` slices, in the order
    /// the frame loop encodes it: the slices write their jobs' slots, the
    /// delivered job's depth is copied, and the render pass draws.
    ///
    /// # Panics
    ///
    /// When the copy finds its job's slot written by another job, or the
    /// screen draws a job not delivered.
    pub(crate) fn frame(&mut self, plan: &FramePlan, slices: usize) {
        let slots = self.written.len() as u64;
        let slot = |job: u64| usize::try_from(job % slots).unwrap();
        for run in &plan.runs {
            let job = run.job.get();
            self.written[slot(job)] = Some(job);
            if run.position + 1 == slices {
                self.kept[slot(job)] = Some(job);
            }
        }
        if let Some(job) = delivered_now(plan).map(JobId::get) {
            assert_eq!(
                self.written[slot(job)],
                Some(job),
                "job {job}'s slot was written by another job before its delivery"
            );
            self.shown = Some(job);
        }
        for delivery in &plan.delivered {
            self.delivered.insert(delivery.job.get());
            self.kept_slot = Some(slot(delivery.job.get()));
        }
        if let Some(job) = self.shown {
            assert!(self.delivered.contains(&job), "job {job} drawn undelivered");
        }
        let kept = self.kept_slot.and_then(|slot| self.kept[slot]);
        if kept.is_some_and(|job| !self.delivered.contains(&job)) {
            self.early_under_kept += 1;
        }
    }

    /// The job on screen after the last frame run.
    pub(crate) fn shown(&self) -> Option<u64> {
        self.shown
    }
}

/// Whether the frames a job is promised can hold its slices at all: the
/// report's `capacity_frames_needed` line.
///
/// A job's slices run in the frames from its submission to the frame that
/// keeps its deadline, and each frame takes slices only while their estimates
/// fit its budget (`beatlm_core::schedule`, "Projecting the queue onto frames
/// to come"). A promise of `k` frames therefore has `k + 1` frames to run a job
/// in, the deadline frame included, and each of them holds at most the
/// *expected* budget the scheduler plans the later frames with. If a whole job
/// takes more time than those frames have budget, no estimate, however exact,
/// lets it deliver on time; this says whether that is the case, from
/// measurements the run made.
///
/// The inputs are the run's own. `run_us` is the sum, over the slices measured
/// in the loop, of the slice's mean measured duration: what one whole job
/// costs in the loop, which is not calibration's `gpu_run_us` (the same model
/// measured alone, and the loop's slices were found slower). `budget_us` is
/// the median over the run's frames of the expected budget, the period less the
/// recent median render and the margin. Both are means and medians, the
/// statistics an exact estimator of the typical cost would plan with: an
/// estimator that pads (the default's 0.9 quantile) needs more room than this,
/// so `frames` is a lower bound. Slices cannot be split across frames, so even
/// at the exact mean the real count can be higher still.
///
/// So `infeasible` is a finding, and `not_ruled_out` is only the absence of
/// one: the capacity alone does not decide that the promise is kept.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Capacity {
    /// The sum of the measured slices' mean durations in the loop, in
    /// microseconds, rounded.
    pub run_us: u64,
    /// The median expected budget of the run's frames, in microseconds; `None`
    /// when no frame was run.
    pub budget_us: Option<u64>,
    /// The promised latency in frames.
    pub promised: u32,
    /// The slices of a job that were measured in the loop.
    pub measured: usize,
    /// The slices of a job.
    pub slices: usize,
}

impl Capacity {
    /// The capacity check of a run whose loop measured each slice's `(sum,
    /// count)` of durations in `measured`, with `slices` slices to a job, a
    /// median expected budget of `budget_us` and a promise of `promised`
    /// frames.
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn of(
        measured: &BTreeMap<SliceId, (u64, u64)>,
        slices: usize,
        budget_us: Option<u64>,
        promised: u32,
    ) -> Self {
        let means = measured
            .values()
            .filter(|(_, count)| *count > 0)
            .map(|&(sum, count)| sum as f64 / count as f64);
        let (total, seen) = means.fold((0.0, 0), |(total, n), mean| (total + mean, n + 1));
        Self {
            // A sum of at most a few hundred microsecond means: far inside u64.
            run_us: f64::round(total) as u64,
            budget_us,
            promised,
            measured: seen,
            slices,
        }
    }

    /// The frames the job's measured time needs at the budget: the fewest
    /// whole frames of `budget_us` that hold `run_us`. `None` without a
    /// positive budget.
    #[must_use]
    pub fn frames(&self) -> Option<u64> {
        self.budget_us
            .filter(|&budget| budget > 0)
            .map(|budget| self.run_us.div_ceil(budget))
    }

    /// The frames a job can run in: the promised latency's frames and the
    /// deadline frame.
    #[must_use]
    pub fn available(&self) -> u64 {
        u64::from(self.promised).saturating_add(1)
    }

    /// `infeasible` when the job needs more frames than it has, `not_ruled_out`
    /// when it does not, `none` when nothing was measured or there was no
    /// budget to compare with.
    #[must_use]
    pub fn verdict(&self) -> &'static str {
        match self.frames() {
            Some(frames) if self.measured > 0 => {
                if frames > self.available() {
                    "infeasible"
                } else {
                    "not_ruled_out"
                }
            }
            _ => "none",
        }
    }

    /// The report line's value.
    #[must_use]
    pub fn line(&self) -> String {
        let number =
            |value: Option<u64>| value.map_or_else(|| "none".to_owned(), |v| v.to_string());
        format!(
            "in_loop_run_us={} slice_budget_us={} frames={} frames_promised={} \
             frames_available={} slices_measured={} of {} verdict={}",
            self.run_us,
            number(self.budget_us),
            number(self.frames()),
            self.promised,
            self.available(),
            self.measured,
            self.slices,
            self.verdict(),
        )
    }
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
    use beatlm_core::budget::Budget;
    use beatlm_core::cost::{CostModel, Estimator};
    use beatlm_core::schedule::Scheduler;
    use beatlm_core::sync::{MissPolicy, SyncPolicy};
    use beatlm_core::time::{FrameIndex, FrameInfo, TimePoint};

    /// A fixed-latency depth run of `frames` frames at `latency` frames, as
    /// the page runs it: `latency + 1` slots, jobs of two 1 ms slices, an input
    /// only once the job before has had its slices (`still_busy`), and a frame
    /// budget of 10 ms, or none in the frames `starved` names. Returns the
    /// memory and the job on screen after each frame.
    fn fixed_run(
        latency: u32,
        frames: u64,
        starved: impl Fn(u64) -> bool,
    ) -> (Screen, Vec<Option<u64>>) {
        const PERIOD: u64 = 8_333;
        let full = Budget::new(Micros::new(10_000));
        let policy = SyncPolicy::FixedLatency { frames: latency };
        let mut scheduler = Scheduler::new(policy, MissPolicy::HoldPrevious, full);
        let cost = CostModel::new(Micros::new(1_000), Estimator::default()).unwrap();
        let slices = vec![SliceId::new(0), SliceId::new(1)];
        let mut screen = Screen::new(policy.max_in_flight().unwrap());
        let mut busy = None;
        let mut shown = Vec::new();
        for index in 0..frames {
            let frame = FrameInfo {
                index: FrameIndex::new(index),
                start: TimePoint::new(index * PERIOD),
                period: Micros::new(PERIOD),
            };
            let job = busy
                .is_none()
                .then(|| scheduler.submit(frame, slices.clone(), &cost).unwrap());
            let budget = if starved(index) {
                Budget::new(Micros::ZERO)
            } else {
                full
            };
            let plan = scheduler.plan_frame(frame, budget, &cost);
            screen.frame(&plan, slices.len());
            busy = still_busy(busy, job.map(|job| (job, slices.len())), &plan);
            shown.push(screen.shown());
        }
        (screen, shown)
    }

    #[test]
    fn a_result_is_on_screen_from_its_delivery_frame_and_not_before() {
        // Latency 2, three slots. Job 0, submitted at frame 0, finishes there
        // and is delivered at frame 2. Jobs 1 and 2 get no budget and miss, at
        // frames 3 and 6; job 3, on job 0's slot, is submitted at frame 7,
        // finishes there, and is delivered at frame 9.
        let (screen, shown) = fixed_run(2, 12, |frame| (1..=6).contains(&frame));
        assert_eq!(shown[1], None);
        assert_eq!(shown[2..9], [Some(0); 7]);
        assert_eq!(shown[9], Some(3));
        // The copy after the last slice, into job 0's slot's buffer, drew job
        // 3 at frames 7 and 8, before its delivery.
        assert_eq!(screen.early_under_kept, 2);
    }

    #[test]
    fn a_slot_is_never_reused_before_its_job_is_delivered() {
        // Runs of misses of every length between deliveries, at latencies
        // with one slot and with several: `Screen::frame` checks every copy
        // against its slot, and that only a delivered job is drawn.
        for latency in [0, 1, 2, 5] {
            let (screen, shown) = fixed_run(latency, 400, |frame| frame % 23 < frame % 11);
            assert!(shown.last().unwrap().is_some(), "latency {latency}");
            if latency > 0 {
                assert!(screen.early_under_kept > 0, "latency {latency}");
            }
        }
    }

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

    fn means(slices: &[(u64, u64)]) -> BTreeMap<SliceId, (u64, u64)> {
        slices
            .iter()
            .enumerate()
            .map(|(i, &measured)| (SliceId::new(i as u32), measured))
            .collect()
    }

    #[test]
    fn a_job_is_summed_from_the_loops_slice_means() {
        // Means 50, 300 and 10; the slice never measured adds nothing and is
        // said.
        let measured = means(&[(100, 2), (900, 3), (0, 0), (10, 1)]);
        let capacity = Capacity::of(&measured, 4, Some(4_300), 10);
        assert_eq!(capacity.run_us, 360);
        assert_eq!((capacity.measured, capacity.slices), (3, 4));
        assert_eq!(capacity.frames(), Some(1));
        assert_eq!(capacity.available(), 11);
        assert_eq!(capacity.verdict(), "not_ruled_out");
        assert_eq!(
            capacity.line(),
            "in_loop_run_us=360 slice_budget_us=4300 frames=1 frames_promised=10 \
             frames_available=11 slices_measured=3 of 4 verdict=not_ruled_out"
        );
        // The means are summed before rounding: four of 0.5 are 2, not four
        // times a rounded 1.
        let halves = means(&[(1, 2), (1, 2), (1, 2), (1, 2)]);
        assert_eq!(Capacity::of(&halves, 4, Some(10), 1).run_us, 2);
    }

    #[test]
    fn a_job_needing_more_frames_than_it_has_is_infeasible_at_any_estimate() {
        // A job of two slices at 27 ms each in the loop, against the 4 334 us
        // a frame of a 120 Hz display leaves: 12.46 frames, so 13.
        let measured = means(&[(54_000, 2), (27_000, 1)]);
        let at = |promised| Capacity::of(&measured, 2, Some(4_334), promised);
        assert_eq!(at(10).run_us, 54_000);
        assert_eq!(at(10).frames(), Some(13));
        assert_eq!(at(10).verdict(), "infeasible");
        // The deadline frame counts: 12 frames promised are 13 frames to run in.
        assert_eq!(at(11).verdict(), "infeasible");
        assert_eq!(at(12).verdict(), "not_ruled_out");
        assert!(
            at(10)
                .line()
                .contains("frames=13 frames_promised=10 frames_available=11")
        );
        assert!(at(10).line().ends_with("verdict=infeasible"));
        // An exact multiple needs no more frame.
        assert_eq!(
            Capacity::of(&measured, 2, Some(54_000), 0).frames(),
            Some(1)
        );
        assert_eq!(
            Capacity::of(&measured, 2, Some(27_000), 1).frames(),
            Some(2)
        );
        assert_eq!(
            Capacity::of(&measured, 2, Some(27_000), 1).verdict(),
            "not_ruled_out"
        );
    }

    #[test]
    fn without_a_budget_or_a_measurement_the_capacity_says_none() {
        let measured = means(&[(100, 1)]);
        for capacity in [
            Capacity::of(&measured, 1, None, 5),
            Capacity::of(&measured, 1, Some(0), 5),
            Capacity::of(&means(&[(0, 0)]), 1, Some(4_000), 5),
            Capacity::of(&BTreeMap::new(), 0, Some(4_000), 5),
        ] {
            assert_eq!(capacity.verdict(), "none", "{capacity:?}");
            assert!(capacity.line().ends_with("verdict=none"));
        }
        let said = Capacity::of(&measured, 1, None, 5).line();
        assert!(said.contains("slice_budget_us=none frames=none"), "{said}");
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

    /// The plan the page makes before the weights for `side` at `latency_us`
    /// on a 120 Hz display, with its slots.
    fn plan_at(side: u32, latency_us: u64) -> (MemoryPlan, u32) {
        let frames = frames_for(Micros::new(latency_us), Micros::new(8_333));
        let slots = SyncPolicy::FixedLatency { frames }.max_in_flight().unwrap();
        (graph(side).unwrap().memory_plan(slots).unwrap(), slots)
    }

    /// The runs `render_order=after` became the default with, 112 and 266
    /// px at 83 and 200 ms and 518 px at 600 ms at 120 Hz, keep it whatever
    /// their arena count, and one needing more arenas than the render can
    /// read is refused before the weights, naming the way out.
    #[test]
    fn the_measured_runs_fit_the_after_order_and_a_longer_one_is_refused() {
        let mut most = 0;
        for (side, latency_us) in [
            (112, 83_335),
            (266, 83_335),
            (112, 200_000),
            (266, 200_000),
            (518, 600_000),
        ] {
            let (plan, slots) = plan_at(side, latency_us);
            check_render_order(RenderOrder::After, &plan, side, slots)
                .unwrap_or_else(|e| panic!("{side} px at {latency_us} us: {e:#}"));
            most = most.max(plan.arenas().len());
        }
        // Several arenas among them, which the single-arena binding refused.
        assert!(most > 1, "{most}");
        // 266 px at 200 ms on 120 Hz: 25 slots and more than one arena.
        let (plan, slots) = plan_at(266, 200_000);
        assert_eq!(slots, 25);
        assert!(plan.arenas().len() > 1);

        // 518 px at 2 s: 241 slots, more arenas than a stage binds.
        let (plan, slots) = plan_at(518, 2_000_000);
        assert!(plan.arenas().len() > DEFAULT_STAGE_STORAGE_BUFFERS as usize);
        let refused = check_render_order(RenderOrder::After, &plan, 518, slots)
            .unwrap_err()
            .to_string();
        assert!(refused.contains("render_order=after cannot be honoured at side=518"));
        assert!(refused.contains(&format!("with {slots} slots")));
        assert!(refused.contains("render_order=free"), "{refused}");
        // `free` reads no arena, so it takes any plan.
        check_render_order(RenderOrder::Free, &plan, 518, slots).unwrap();
    }

    /// The plan made from the graph alone is the one `Net` realises from the
    /// numbers: the ids are all it reads.
    #[test]
    fn the_plan_before_the_weights_is_the_plan_of_the_ids() {
        let graph = graph(112).unwrap();
        let ids = Numbers {
            input: graph.image,
            output: graph.depth,
            weights: vec![(TensorId::new(0), vec![1.0])],
            slot_input: Box::new(|_| vec![0.5; 3]),
        };
        let from_numbers = slots::memory_plan(&graph.program, &ids, 3).unwrap();
        let before = graph.memory_plan(3).unwrap();
        assert_eq!(before.arenas(), from_numbers.arenas());
        assert_eq!(before.total_bytes(), from_numbers.total_bytes());
    }
}
