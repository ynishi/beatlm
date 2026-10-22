//! Whether a frame's render pass ran at the same time as its compute passes
//! on the GPU: the report's `pass_overlap_frames`, `pass_overlap_us`,
//! `pass_order`, `slice_vs_render_overlap` and `pass_overlap_compute` lines;
//! and where the next frame's first compute pass began against the render's
//! end, `pass_overlap_next_frame`.
//!
//! Both hosts use this module, so a page and the bench write these lines by
//! the same rules and in the same format, and
//! a browser report compares with a native one line by line. It is pure: a
//! host hands it what its timer returned ([`Frame::of`]), and nothing here
//! feeds the scheduler or the cost model.
//!
//! # What the lines test
//!
//! In Chrome 154 (WebGPU) on an Apple M1 Pro, 2026-10-22, Depth Anything V2
//! Small at 266 px, a minimal render pass (`render_iters=1`) about halved the
//! depth slices' own GPU times (slice 3 from 930 to 472 µs). Natively on the
//! same machine and day (Metal, `--submit free`, latency 5) they did not move
//! (545 and 548 µs) (issue #55). H7 (issue #61) reads this as Chrome, Dawn on
//! Metal, running a frame's render pass and its compute passes at the same
//! time, so that they slow each other, where wgpu natively runs them one
//! after the other. These lines measure whether they overlap; they change
//! nothing about what runs. Natively, the second half did not hold ("What it
//! found natively", below).
//!
//! # What the timestamps can say
//!
//! Each host encodes a frame as one command buffer, submitted once: the
//! compute passes in the order the plan handed out slices, then the render
//! pass (the page records an untimed copy of the delivered depth between
//! them). Every pass carries a begin and an end timestamp
//! ([`PassSpan`]) from the timer's one query set, resolved by the frame's own
//! submission on the one queue. Every comparison here but one is between
//! passes of one frame, so of one submission; that those timestamps share a
//! clock is taken, not checked, as for the page's `tail_gap_*` lines (issue
//! #43). `pass_overlap_next_frame` compares two frames, so two submissions,
//! and takes the same of them: that the queue's timestamps share one clock
//! across submissions, as the page's `tail_gap_before_first_us` does.
//!
//! A queue that runs the passes one after the other, in the order encoded,
//! gives spans that do not intersect: each begins at or after the end of the
//! one before. Two spans that intersect say that the GPU wrote one pass's
//! begin before the other's end: it had begun the later-encoded pass before
//! it finished the earlier one, or the timestamps of one submission do not
//! compare, which these lines cannot tell apart. Nor can they tell two passes
//! whose work ran together from one that began and then waited inside its
//! span; `slice_vs_render_overlap` is the line that says whether an overlap
//! cost the slice time.
//!
//! # When an intersection counts
//!
//! Two spans overlap when they intersect by more than the run's tolerance
//! ([`tolerance_ns`]): its timestamps' step ([`step_ns`]), and at least
//! [`MIN_TOLERANCE_NS`]. A timestamp cut down to a step moves by less than
//! one step, so an intersection read from four of them is off by less than
//! one step too, and only one longer than the step is certainly not two
//! passes that touch. Chrome's timestamps came in steps of 65 536 ns (#43).
//! An exact clock's step is a few nanoseconds; the floor of a microsecond is
//! a choice, not a measurement: far below any pass these workloads time, and
//! far above the nanoseconds by which a driver might sample two boundaries
//! that touch. A run with few timestamps can read a coarser step than its
//! clock's, so each line that depends on it prints the tolerance it used.
//!
//! # The lines
//!
//! - `pass_overlap_frames`: the timed frames (`frames`); of them, those with a
//!   timed render pass and at least one timed compute pass
//!   (`with_render_and_slices`), the only frames the next three lines read;
//!   of those, the frames whose render span overlaps any of their compute
//!   spans (`overlapping`, `share`); the tolerance (`tolerance_ns`); and the
//!   largest intersection of a render with a compute span, counted or not
//!   (`max_intersection_ns`), which shows how near the tolerance a run came.
//!   H7 predicts a large share in Chrome and none natively.
//! - `pass_overlap_us`: per overlapping frame, how long its render span shared
//!   with its compute spans together (the render's intersection with their
//!   union, so compute spans that overlap each other count once), in
//!   microseconds: p50, p99, max. Against a slice of a few hundred
//!   microseconds, how much of the slices ran beside the render.
//! - `pass_order`: where the render's begin fell, as [`Order`] defines: no
//!   later than the first compute pass's begin (`render_before_first_slice`),
//!   at or after the last one's end, within the tolerance
//!   (`render_after_last_slice`, the order encoded), or between them
//!   (`render_among_slices`). Every frame of a queue that runs passes in
//!   order is `after`; a render begun before or among the slices was begun
//!   ahead of work encoded before it.
//! - `slice_vs_render_overlap`: each compute pass's measured duration over
//!   the median of the same slice's in those frames (the slices
//!   [`slice_medians`] keeps), apart by whether the pass overlapped its frame's
//!   render: how many, and the ratios' mean, p50 and p90. H7's slowdown
//!   predicts the overlapped ratios above 1 and the others at or below it;
//!   overlap with ratios alike is overlap that does not slow the slices.
//! - `pass_overlap_compute`: within every timed frame, consecutive compute
//!   passes in encoding order (`pairs`); the pairs that overlap
//!   (`overlapping`, `share`); the frames with a pair (`frames_with_pairs`)
//!   and with an overlapping one (`frames_overlapping`); and the largest
//!   intersection of a pair, counted or not (`max_intersection_ns`). It tells
//!   overlap among the slices, which the render does not explain, from overlap
//!   with the render: compute passes that overlap one another as often as the
//!   render overlaps them point at the timestamps, or at how every pass is
//!   scheduled, rather than at the render.
//! - `pass_overlap_next_frame`: pairs of consecutive timed frames, the first
//!   with a timed render and the second with a timed compute pass (`pairs`),
//!   by where the second's first compute pass (the earliest begin) fell
//!   against the first's render end ([`Next`]): more than the tolerance
//!   before it (`first_slice_before_render_end`, `share` of the pairs),
//!   within it (`first_slice_at_render_end`), or more than it after
//!   (`first_slice_after_render_end`); the tolerance; and by how much, in
//!   microseconds, p50 and max, before (`before_us_*`) and after
//!   (`after_us_*`). See "Across frames".
//!
//! # Across frames
//!
//! Under the `after` render order (the bench's `--render-order`, the page's
//! `render_order`) the render reads every buffer the slices write
//! (`workload.rs`), so the next frame's slices write what this frame's render
//! read: a write after a read, which WebGPU orders after the read. Whether
//! that makes the next frame's slices wait for this render is what
//! `pass_overlap_next_frame` reads, beside the same line under `free`. Its
//! three counts say different things. A first slice that began before the
//! render ended did not wait for its end (WebGPU orders the write after the
//! vertex stage's read, not after the render's end, so this is no broken
//! rule), or the two submissions' timestamps do not compare. One that began
//! at the end, within the tolerance, is what a wait looks like when the next
//! frame's work was already queued. One that began later says nothing about
//! a wait: the GPU had nothing queued, because the next frame was submitted
//! after the render ended, which is how a frame whose GPU work is shorter than
//! its period usually ends.
//!
//! Natively that is every pair. Apple M1 Pro, Metal, wgpu 30.0.1, release,
//! 2026-10-22, Depth Anything V2 Small, `--workload depth --submit free
//! --frames 300 --render-iters 50`, one run each:
//!
//! | run | arenas | pairs | before / at / after the render's end | after, µs p50 / max |
//! |---|---|---|---|---|
//! | 266 px, `--latency 5`, `free` | 1 | 299 | 0 / 0 / 299 | 8316 / 17204 |
//! | 266 px, `--latency 5`, `after` | 1 | 299 | 0 / 0 / 299 | 1426 / 20715 |
//! | 266 px, `--latency 25`, `after` | 2 | 299 | 0 / 0 / 299 | 1391 / 17021 |
//! | 518 px, `--latency 10`, `after` | 2 | 299 | 0 / 0 / 299 | 1793 / 4701 |
//!
//! No first slice began before or at the previous render's end: the bench
//! submits a frame at most once per 16.7 ms period and its GPU work took
//! about 14 ms, so the GPU was idle when each frame came, and the line could
//! not have shown a wait natively. The gap is shorter under `after` because
//! the render is then the frame's last pass, where under `free` slices ran on
//! after it. Whether the next frame waits is for a run that keeps the GPU
//! busy across frames to show, as a browser at 120 Hz may.
//!
//! # What it found natively
//!
//! The bench on an Apple M1 Pro, Metal, wgpu 30.0.1, release build,
//! 2026-10-22, Depth Anything V2 Small, `--workload depth --size 266
//! --latency 5 --submit free --frames 600` (155 slices a job), one run each:
//!
//! | `--render-iters` | overlapping frames | `pass_overlap_us` p50 / p99 / max | render before / among / after the slices | overlapped: samples, ratio mean / p50 / p90 | not overlapped: samples, ratio mean / p50 / p90 | compute pairs overlapping | delivered / missed |
//! |---|---|---|---|---|---|---|---|
//! | 50 | 600 of 600 | 609 / 1579 / 2090 | 103 / 497 / 0 | 1229, 3.54 / 1.24 / 3.81 | 17971, 1.00 / 1.00 / 1.01 | 0 of 18600 | 119 / 0 |
//! | 1 | 600 of 600 | 73 / 197 / 569 | 2 / 598 / 0 | 659, 1.17 / 1.00 / 1.10 | 23226, 1.01 / 1.00 / 1.01 | 0 of 23285 | 148 / 0 |
//!
//! Natively too, then, the render pass's span intersects the slices' in every
//! frame, and in none did it begin after the last slice ended, the order
//! encoded; consecutive compute passes never intersected (their largest
//! intersection was 0 ns), so compute passes ran one after the other and the
//! render beside them. The tolerance did not decide a frame: the
//! intersections were hundreds of microseconds against its one. The overlap
//! grew with the render's work, and so did what it cost the slices it
//! overlapped: at 50 iterations those measured 1.24 times their slice's
//! median at the median and 3.54 in the mean, at 1 iteration 1.00 and 1.17.
//! That cost follows the render's work, which a clock offset between the
//! render and compute timestamps would not do. About two slices a frame
//! overlapped at 50 iterations, of 32, and one at 1, of 40, so a slice's
//! mean barely moved (slice 3 measured 546 µs at both settings), which is how the slices' times came to look
//! independent of the render natively (issue #55). That the render overlaps
//! the slices is therefore not what sets Chrome apart; how much of each
//! slice it overlaps, and what that costs there, is for the same lines in
//! Chrome to say. `gpu_frame_us`, a sum of pass durations, counts the
//! overlapped time twice.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use beatlm::gpu::{FrameTiming, PassSpan};
use beatlm_core::cost::SliceId;

use crate::stats::{Ratios, share, slice_medians, step_ns, summarize};

/// The least intersection, in nanoseconds, that can count as an overlap
/// whatever the timestamps' step: one microsecond. The module documentation,
/// "When an intersection counts", says why.
pub const MIN_TOLERANCE_NS: u64 = 1_000;

/// One timed compute pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pass {
    /// The slice it ran.
    pub slice: SliceId,
    /// Its duration as the cost model received it, in microseconds.
    pub us: u64,
    /// Its timestamps.
    pub span: PassSpan,
}

/// One timed frame's passes: its compute passes in the order they were
/// encoded, and its render pass, which was encoded after them.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Frame {
    /// The frame's index, which says which frame came before it.
    pub index: u64,
    /// The compute passes.
    pub compute: Vec<Pass>,
    /// The render pass's timestamps, when the frame timed one.
    pub render: Option<PassSpan>,
}

impl Frame {
    /// The frame `timing` measured, each compute pass named by the slice
    /// `slice` reads from its label.
    #[must_use]
    pub fn of<T>(timing: &FrameTiming<T>, slice: impl Fn(&T) -> SliceId) -> Self {
        Self {
            index: timing.frame.get(),
            compute: timing
                .compute
                .iter()
                .zip(&timing.compute_spans)
                .map(|((tag, duration), &span)| Pass {
                    slice: slice(tag),
                    us: duration.get(),
                    span,
                })
                .collect(),
            render: timing.render_span,
        }
    }

    /// Every timestamp of the frame.
    fn timestamps(&self) -> impl Iterator<Item = u64> + '_ {
        self.compute
            .iter()
            .map(|p| p.span)
            .chain(self.render)
            .flat_map(|s| [s.begin_ns, s.end_ns])
    }
}

/// How long `a` and `b` have in common, in nanoseconds: zero for spans that
/// are apart or only touch, and for a span whose end precedes its begin.
#[must_use]
pub fn intersection_ns(a: PassSpan, b: PassSpan) -> u64 {
    let begin = a.begin_ns.max(b.begin_ns);
    let end = a.end_ns.min(b.end_ns);
    end.saturating_sub(begin)
}

/// Whether `a` and `b` intersect by more than `tolerance_ns`.
#[must_use]
pub fn overlaps(a: PassSpan, b: PassSpan, tolerance_ns: u64) -> bool {
    intersection_ns(a, b) > tolerance_ns
}

/// The tolerance an intersection must exceed to count, for timestamps of a
/// step of `step_ns` ([`step_ns`]; `None` when there were too few to read
/// one): the step, and at least [`MIN_TOLERANCE_NS`].
#[must_use]
pub fn tolerance_ns(step_ns: Option<u64>) -> u64 {
    step_ns.unwrap_or(0).max(MIN_TOLERANCE_NS)
}

/// How long `render` shares with `compute` taken together, in nanoseconds:
/// its intersection with their union, so time two compute spans share with
/// each other and with the render counts once.
#[must_use]
pub fn covered_ns(render: PassSpan, compute: &[PassSpan]) -> u64 {
    let mut clipped: Vec<(u64, u64)> = compute
        .iter()
        .filter_map(|s| {
            let begin = s.begin_ns.max(render.begin_ns);
            let end = s.end_ns.min(render.end_ns);
            (end > begin).then_some((begin, end))
        })
        .collect();
    clipped.sort_unstable();
    // Sorted by begin, a span either extends the run of spans before it or
    // starts a new one; each run is counted once, when it ends.
    let mut total = 0_u64;
    let mut current: Option<(u64, u64)> = None;
    for (begin, end) in clipped {
        current = match current {
            Some((b, e)) if begin <= e => Some((b, e.max(end))),
            Some((b, e)) => {
                total = total.saturating_add(e.saturating_sub(b));
                Some((begin, end))
            }
            None => Some((begin, end)),
        };
    }
    if let Some((b, e)) = current {
        total = total.saturating_add(e.saturating_sub(b));
    }
    total
}

/// Where a frame's render pass began against its compute passes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    /// At or before the earliest compute pass's begin, and not within the
    /// tolerance of the latest one's end.
    BeforeFirst,
    /// After the earliest compute pass's begin and more than the tolerance
    /// before the latest one's end.
    Among,
    /// At or after the latest compute pass's end, or less than the tolerance
    /// before it: the order the passes were encoded in.
    AfterLast,
}

/// Where `render` began against `compute`, with `tolerance_ns` given to the
/// encoded order (see [`Order`]); `None` when there is no compute pass.
///
/// `AfterLast` is decided first, so a render that began in the same step as
/// the last compute pass ended is in order however the step cut them, and a
/// frame of one pass of no length is in order too.
#[must_use]
pub fn order(render: PassSpan, compute: &[PassSpan], tolerance_ns: u64) -> Option<Order> {
    let first = compute.iter().map(|s| s.begin_ns).min()?;
    let last = compute.iter().map(|s| s.end_ns).max()?;
    Some(if render.begin_ns.saturating_add(tolerance_ns) >= last {
        Order::AfterLast
    } else if render.begin_ns <= first {
        Order::BeforeFirst
    } else {
        Order::Among
    })
}

/// Where a frame's first compute pass began against the previous frame's
/// render end, with the distance in nanoseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Next {
    /// More than the tolerance before the render ended, by this much.
    Before(u64),
    /// Within the tolerance of the render's end, either side.
    At,
    /// More than the tolerance after the render ended, by this much.
    After(u64),
}

/// Where a compute pass that began at `first_begin_ns` fell against a render
/// that ended at `render_end_ns`, with `tolerance_ns` given to the end
/// (see [`Next`]).
#[must_use]
pub fn next(render_end_ns: u64, first_begin_ns: u64, tolerance_ns: u64) -> Next {
    if first_begin_ns.saturating_add(tolerance_ns) < render_end_ns {
        Next::Before(render_end_ns - first_begin_ns)
    } else if first_begin_ns > render_end_ns.saturating_add(tolerance_ns) {
        Next::After(first_begin_ns - render_end_ns)
    } else {
        Next::At
    }
}

/// Consecutive frames: the first's render against the second's first compute
/// pass ("Across frames" in the module documentation).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct NextFrame {
    /// Pairs of consecutive timed frames, the first with a timed render and
    /// the second with a timed compute pass.
    pub pairs: u64,
    /// Per pair whose first compute pass began before the render ended, by
    /// how much, in microseconds.
    pub before_us: Vec<u64>,
    /// Pairs whose first compute pass began within the tolerance of the
    /// render's end.
    pub at: u64,
    /// Per pair whose first compute pass began after the render ended, by
    /// how much, in microseconds.
    pub after_us: Vec<u64>,
}

/// [`NextFrame`] over `frames`, in any order, with `tolerance_ns`.
fn next_frame(frames: &[Frame], tolerance_ns: u64) -> NextFrame {
    let by_index: BTreeMap<u64, &Frame> = frames.iter().map(|f| (f.index, f)).collect();
    let mut out = NextFrame::default();
    for (&index, frame) in &by_index {
        let Some(render) = frame.render else {
            continue;
        };
        let first = index
            .checked_add(1)
            .and_then(|i| by_index.get(&i))
            .and_then(|f| f.compute.iter().map(|p| p.span.begin_ns).min());
        let Some(first) = first else {
            continue;
        };
        out.pairs += 1;
        match next(render.end_ns, first, tolerance_ns) {
            Next::Before(ns) => out.before_us.push(round_us(ns)),
            Next::At => out.at += 1,
            Next::After(ns) => out.after_us.push(round_us(ns)),
        }
    }
    out
}

/// Ratios of measurements to their slice's median: their median and 90th
/// percentile, and their mean.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Normalized {
    /// How many, their median and their 90th percentile.
    pub ratios: Ratios,
    /// Their mean.
    pub mean: f64,
}

impl Normalized {
    /// `None` when there are none.
    #[allow(clippy::cast_precision_loss)]
    fn of(values: Vec<f64>) -> Option<Self> {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        Some(Self {
            ratios: Ratios::of(values)?,
            mean,
        })
    }
}

/// Counts by [`Order`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Orders {
    /// [`Order::BeforeFirst`].
    pub before_first: u64,
    /// [`Order::AfterLast`].
    pub after_last: u64,
    /// [`Order::Among`].
    pub among: u64,
}

/// Consecutive compute passes of a frame against each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ComputePairs {
    /// Pairs of consecutive compute passes, over every timed frame.
    pub pairs: u64,
    /// Of them, pairs that overlap.
    pub overlapping: u64,
    /// Frames with at least one pair.
    pub frames_with_pairs: u64,
    /// Of them, frames with an overlapping pair.
    pub frames_overlapping: u64,
    /// The largest intersection of a pair, counted or not, in nanoseconds.
    pub max_intersection_ns: u64,
}

/// Everything the lines say.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PassOverlap {
    /// Timed frames.
    pub frames: u64,
    /// Of them, frames with a timed render pass and a timed compute pass.
    pub with_render_and_slices: u64,
    /// Of those, frames whose render overlaps one of their compute passes.
    pub overlapping: u64,
    /// The tolerance an intersection had to exceed, in nanoseconds.
    pub tolerance_ns: u64,
    /// The largest intersection of a render with a compute pass, counted or
    /// not, in nanoseconds.
    pub max_intersection_ns: u64,
    /// Per overlapping frame, what its render shared with its compute
    /// passes taken together ([`covered_ns`]), in microseconds.
    pub overlap_us: Vec<u64>,
    /// Where the render began, over the frames with a render and slices.
    pub order: Orders,
    /// Compute passes that overlapped their frame's render, over their
    /// slice's median.
    pub overlapped: Option<Normalized>,
    /// Compute passes of frames with a render that did not overlap it, over
    /// their slice's median.
    pub not_overlapped: Option<Normalized>,
    /// Consecutive compute passes against each other.
    pub compute: ComputePairs,
    /// Each render against the next frame's first compute pass.
    pub next_frame: NextFrame,
}

/// The statistics of the lines over `frames`, in any order.
#[must_use]
pub fn analyse(frames: &[Frame]) -> PassOverlap {
    let timestamps: Vec<u64> = frames.iter().flat_map(Frame::timestamps).collect();
    let tolerance = tolerance_ns(step_ns(&timestamps));
    let mut out = PassOverlap {
        frames: frames.len() as u64,
        tolerance_ns: tolerance,
        ..PassOverlap::default()
    };
    let mut by_slice: BTreeMap<SliceId, Vec<u64>> = BTreeMap::new();
    let mut flagged: Vec<(SliceId, u64, bool)> = Vec::new();
    for frame in frames {
        let spans: Vec<PassSpan> = frame.compute.iter().map(|p| p.span).collect();
        compute_pairs(&mut out.compute, &spans, tolerance);
        let Some(render) = frame.render else {
            continue;
        };
        let Some(order) = order(render, &spans, tolerance) else {
            continue;
        };
        out.with_render_and_slices += 1;
        match order {
            Order::BeforeFirst => out.order.before_first += 1,
            Order::Among => out.order.among += 1,
            Order::AfterLast => out.order.after_last += 1,
        }
        let mut overlapping = false;
        for pass in &frame.compute {
            out.max_intersection_ns = out
                .max_intersection_ns
                .max(intersection_ns(pass.span, render));
            let over = overlaps(pass.span, render, tolerance);
            overlapping |= over;
            by_slice.entry(pass.slice).or_default().push(pass.us);
            flagged.push((pass.slice, pass.us, over));
        }
        if overlapping {
            out.overlapping += 1;
            out.overlap_us.push(round_us(covered_ns(render, &spans)));
        }
    }
    let medians = slice_medians(&mut by_slice);
    let (mut with, mut without) = (Vec::new(), Vec::new());
    for (slice, us, over) in flagged {
        let Some(&median) = medians.get(&slice) else {
            continue;
        };
        #[allow(clippy::cast_precision_loss)]
        let ratio = us as f64 / median as f64;
        if over {
            with.push(ratio);
        } else {
            without.push(ratio);
        }
    }
    out.overlapped = Normalized::of(with);
    out.not_overlapped = Normalized::of(without);
    out.next_frame = next_frame(frames, tolerance);
    out
}

/// Adds one frame's consecutive compute `spans` to `pairs`.
fn compute_pairs(pairs: &mut ComputePairs, spans: &[PassSpan], tolerance_ns: u64) {
    if spans.len() < 2 {
        return;
    }
    pairs.frames_with_pairs += 1;
    let mut any = false;
    for (&a, &b) in spans.iter().zip(spans.iter().skip(1)) {
        pairs.pairs += 1;
        let shared = intersection_ns(a, b);
        pairs.max_intersection_ns = pairs.max_intersection_ns.max(shared);
        if shared > tolerance_ns {
            pairs.overlapping += 1;
            any = true;
        }
    }
    if any {
        pairs.frames_overlapping += 1;
    }
}

/// Nanoseconds in microseconds, rounded to nearest.
fn round_us(ns: u64) -> u64 {
    ns.saturating_add(500) / 1_000
}

/// Writes the lines, in the order the module documentation lists them.
///
/// # Errors
///
/// Only those of `out`, which for a `String` are none.
pub fn write_lines(out: &mut String, o: &PassOverlap) -> std::fmt::Result {
    writeln!(
        out,
        "pass_overlap_frames: frames={} with_render_and_slices={} overlapping={} share={} \
         tolerance_ns={} max_intersection_ns={}",
        o.frames,
        o.with_render_and_slices,
        o.overlapping,
        share(o.overlapping, o.with_render_and_slices),
        o.tolerance_ns,
        o.max_intersection_ns
    )?;
    match summarize(&o.overlap_us) {
        Some(s) => writeln!(
            out,
            "pass_overlap_us: p50={} p99={} max={}",
            s.p50, s.p99, s.max
        )?,
        None => writeln!(out, "pass_overlap_us: none")?,
    }
    writeln!(
        out,
        "pass_order: render_before_first_slice={} render_after_last_slice={} \
         render_among_slices={}",
        o.order.before_first, o.order.after_last, o.order.among
    )?;
    write!(out, "slice_vs_render_overlap:")?;
    write_normalized(out, "overlapped", o.overlapped)?;
    write_normalized(out, "not_overlapped", o.not_overlapped)?;
    writeln!(out)?;
    let c = o.compute;
    writeln!(
        out,
        "pass_overlap_compute: pairs={} overlapping={} share={} frames_with_pairs={} \
         frames_overlapping={} max_intersection_ns={}",
        c.pairs,
        c.overlapping,
        share(c.overlapping, c.pairs),
        c.frames_with_pairs,
        c.frames_overlapping,
        c.max_intersection_ns
    )?;
    let n = &o.next_frame;
    let before = n.before_us.len() as u64;
    write!(
        out,
        "pass_overlap_next_frame: pairs={} first_slice_before_render_end={before} share={} \
         first_slice_at_render_end={} first_slice_after_render_end={} tolerance_ns={}",
        n.pairs,
        share(before, n.pairs),
        n.at,
        n.after_us.len(),
        o.tolerance_ns
    )?;
    write_p50_max(out, "before_us", &n.before_us)?;
    write_p50_max(out, "after_us", &n.after_us)?;
    writeln!(out)
}

/// ` <name>_p50=.. <name>_max=..`, each `none` when there are no samples.
fn write_p50_max(out: &mut String, name: &str, samples: &[u64]) -> std::fmt::Result {
    match summarize(samples) {
        Some(s) => write!(out, " {name}_p50={} {name}_max={}", s.p50, s.max),
        None => write!(out, " {name}_p50=none {name}_max=none"),
    }
}

/// ` <name>_samples=n <name>_ratio_mean=.. <name>_ratio_p50=.. <name>_ratio_p90=..`,
/// each `none` when there are no samples.
fn write_normalized(out: &mut String, name: &str, n: Option<Normalized>) -> std::fmt::Result {
    match n {
        Some(n) => write!(
            out,
            " {name}_samples={} {name}_ratio_mean={:.2} {name}_ratio_p50={:.2} \
             {name}_ratio_p90={:.2}",
            n.ratios.samples, n.mean, n.ratios.median, n.ratios.p90
        ),
        None => write!(
            out,
            " {name}_samples=0 {name}_ratio_mean=none {name}_ratio_p50=none \
             {name}_ratio_p90=none"
        ),
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    /// Chrome's timestamp step.
    const Q: u64 = 65_536;

    fn span(begin_ns: u64, end_ns: u64) -> PassSpan {
        PassSpan { begin_ns, end_ns }
    }

    /// Frame 0, of compute passes `(slice, begin, end)` in nanoseconds, and a
    /// render.
    fn frame(compute: &[(u32, u64, u64)], render: Option<PassSpan>) -> Frame {
        Frame {
            index: 0,
            compute: compute
                .iter()
                .map(|&(slice, begin, end)| Pass {
                    slice: SliceId::new(slice),
                    us: (end - begin) / 1_000,
                    span: span(begin, end),
                })
                .collect(),
            render,
        }
    }

    #[test]
    fn an_intersection_is_what_two_spans_share() {
        // Apart, touching, partial, nested, in either order.
        assert_eq!(intersection_ns(span(0, 10), span(20, 30)), 0);
        assert_eq!(intersection_ns(span(0, 10), span(10, 30)), 0);
        assert_eq!(intersection_ns(span(10, 30), span(0, 10)), 0);
        assert_eq!(intersection_ns(span(0, 15), span(10, 30)), 5);
        assert_eq!(intersection_ns(span(0, 100), span(10, 30)), 20);
        assert_eq!(intersection_ns(span(10, 30), span(0, 100)), 20);
        // A reversed span is empty.
        assert_eq!(intersection_ns(span(30, 10), span(0, 100)), 0);
        // Quantized equal timestamps: two passes that both read as one step
        // of no length, or a pass of no length at another's end.
        assert_eq!(intersection_ns(span(5 * Q, 5 * Q), span(5 * Q, 5 * Q)), 0);
        assert_eq!(intersection_ns(span(4 * Q, 5 * Q), span(5 * Q, 5 * Q)), 0);
    }

    #[test]
    fn only_an_intersection_longer_than_the_tolerance_overlaps() {
        let tolerance = tolerance_ns(Some(Q));
        assert_eq!(tolerance, Q);
        // One step of intersection may be two touching passes cut to it.
        assert!(!overlaps(span(0, 3 * Q), span(2 * Q, 5 * Q), tolerance));
        assert!(overlaps(span(0, 3 * Q), span(Q, 5 * Q), tolerance));
        // An exact clock, or too few timestamps, gets the floor.
        assert_eq!(tolerance_ns(Some(1)), MIN_TOLERANCE_NS);
        assert_eq!(tolerance_ns(None), MIN_TOLERANCE_NS);
        assert!(!overlaps(
            span(0, 1_500),
            span(500, 2_000),
            MIN_TOLERANCE_NS
        ));
        assert!(overlaps(span(0, 1_501), span(500, 2_000), MIN_TOLERANCE_NS));
    }

    #[test]
    fn the_render_is_covered_by_the_union_of_the_compute_spans() {
        let render = span(100, 200);
        assert_eq!(covered_ns(render, &[]), 0);
        // Outside or touching: nothing.
        assert_eq!(covered_ns(render, &[span(0, 100), span(200, 300)]), 0);
        // Two compute spans overlapping each other inside the render count
        // once; one sticking out is clipped.
        assert_eq!(
            covered_ns(render, &[span(120, 160), span(140, 170), span(190, 400)]),
            50 + 10
        );
        // A compute span around the whole render covers it.
        assert_eq!(covered_ns(render, &[span(0, 400), span(150, 160)]), 100);
    }

    #[test]
    fn the_render_begins_before_among_or_after_the_slices() {
        let compute = [span(100, 200), span(200, 300)];
        assert_eq!(order(span(300, 400), &compute, 0), Some(Order::AfterLast));
        assert_eq!(order(span(350, 400), &compute, 0), Some(Order::AfterLast));
        // Within the tolerance of the last end is still in order.
        assert_eq!(order(span(290, 400), &compute, 10), Some(Order::AfterLast));
        assert_eq!(order(span(289, 400), &compute, 10), Some(Order::Among));
        assert_eq!(order(span(150, 400), &compute, 0), Some(Order::Among));
        // At the first begin, or before it.
        assert_eq!(order(span(100, 400), &compute, 0), Some(Order::BeforeFirst));
        assert_eq!(order(span(0, 50), &compute, 0), Some(Order::BeforeFirst));
        // No compute pass, no order.
        assert_eq!(order(span(0, 50), &[], 0), None);
        // Quantized equal timestamps: a slice and a render that both read as
        // one instant are in order.
        assert_eq!(
            order(span(5 * Q, 5 * Q), &[span(5 * Q, 5 * Q)], Q),
            Some(Order::AfterLast)
        );
    }

    /// Ten frames on Chrome's step, each slices 0 and 1 then a render. In
    /// the first five the passes run in order; in the last five the render
    /// begins inside slice 1, which measures twice as long.
    fn chrome_like() -> Vec<Frame> {
        (0..10)
            .map(|i| {
                // 201 steps apart, so that the timestamps' common step is one
                // of Chrome's and not a multiple of it.
                let at = i * 201 * Q;
                let f = if i < 5 {
                    frame(
                        &[(0, at, at + 10 * Q), (1, at + 10 * Q, at + 20 * Q)],
                        Some(span(at + 20 * Q, at + 50 * Q)),
                    )
                } else {
                    frame(
                        &[(0, at, at + 10 * Q), (1, at + 10 * Q, at + 30 * Q)],
                        Some(span(at + 15 * Q, at + 45 * Q)),
                    )
                };
                Frame { index: i, ..f }
            })
            .collect()
    }

    #[test]
    fn frames_whose_render_runs_inside_a_slice_overlap() {
        let o = analyse(&chrome_like());
        assert_eq!(o.tolerance_ns, Q);
        assert_eq!(
            (o.frames, o.with_render_and_slices, o.overlapping),
            (10, 10, 5)
        );
        assert_eq!(o.max_intersection_ns, 15 * Q);
        // 15 steps of 65.536 µs each.
        assert_eq!(o.overlap_us, vec![983; 5]);
        assert_eq!(
            o.order,
            Orders {
                before_first: 0,
                after_last: 5,
                among: 5,
            }
        );
        // Slice 1's median by nearest rank is the fifth of its ten
        // measurements, a fast one: the overlapped measure 2 times it, the
        // rest 1.
        let with = o.overlapped.unwrap();
        assert_eq!((with.ratios.samples, with.ratios.median), (5, 2.0));
        let without = o.not_overlapped.unwrap();
        assert_eq!((without.ratios.samples, without.ratios.p90), (15, 1.0));
        assert!((with.mean - 2.0).abs() < 1e-12);
        // Slices 0 and 1 touch in every frame: no compute overlap.
        assert_eq!(
            o.compute,
            ComputePairs {
                pairs: 10,
                overlapping: 0,
                frames_with_pairs: 10,
                frames_overlapping: 0,
                max_intersection_ns: 0,
            }
        );
    }

    #[test]
    fn a_frame_without_a_render_counts_only_for_its_compute_pairs() {
        let mut frames = chrome_like();
        let at = 10 * 201 * Q;
        // Overlapping slices, and no render span.
        frames.push(Frame {
            index: 10,
            ..frame(&[(0, at, at + 10 * Q), (1, at + 5 * Q, at + 20 * Q)], None)
        });
        // A render with no slice.
        frames.push(Frame {
            index: 11,
            ..frame(&[], Some(span(at + 100 * Q, at + 110 * Q)))
        });
        let o = analyse(&frames);
        assert_eq!((o.frames, o.with_render_and_slices), (12, 10));
        assert_eq!(o.compute.pairs, 11);
        assert_eq!(o.compute.overlapping, 1);
        assert_eq!(o.compute.frames_overlapping, 1);
        assert_eq!(o.compute.max_intersection_ns, 5 * Q);
        // Their measurements are not among the ratios.
        assert_eq!(o.overlapped.unwrap().ratios.samples, 5);
        assert_eq!(o.not_overlapped.unwrap().ratios.samples, 15);
    }

    #[test]
    fn the_next_slice_begins_before_at_or_after_the_render_end() {
        // Before by more than the tolerance; within it either side; after.
        assert_eq!(next(1_000, 500, 100), Next::Before(500));
        assert_eq!(next(1_000, 900, 100), Next::At);
        assert_eq!(next(1_000, 899, 100), Next::Before(101));
        assert_eq!(next(1_000, 1_100, 100), Next::At);
        assert_eq!(next(1_000, 1_101, 100), Next::After(101));
        // Quantized equal timestamps are at the end.
        assert_eq!(next(5 * Q, 5 * Q, Q), Next::At);
        // A render that ends at zero, a slice at the clock's end.
        assert_eq!(next(0, u64::MAX, 0), Next::After(u64::MAX));
        assert_eq!(next(u64::MAX, 0, 0), Next::Before(u64::MAX));
    }

    #[test]
    fn a_render_is_paired_with_the_next_frames_first_slice_only() {
        let at = |index: u64, compute: &[(u32, u64, u64)], render: Option<PassSpan>| Frame {
            index,
            ..frame(compute, render)
        };
        let frames = vec![
            // Given out of order: frame 2 first.
            at(
                2,
                &[(0, 2_500_000, 3_000_000)],
                Some(span(3_000_000, 4_000_000)),
            ),
            // Its render ends at 2 ms; frame 1's first slice, the earliest
            // begin and not the first encoded, begins 1 ms before.
            at(0, &[(0, 0, 500_000)], Some(span(500_000, 2_000_000))),
            at(
                1,
                &[(1, 1_200_000, 1_400_000), (0, 1_000_000, 1_200_000)],
                Some(span(1_400_000, 2_400_000)),
            ),
            // Frame 2's render ends at 4 ms and frame 3's slice begins there.
            at(
                3,
                &[(0, 4_000_000, 4_500_000)],
                Some(span(4_500_000, 5_000_000)),
            ),
            // Frame 4 has neither, so frame 3 has no pair and frame 5 none
            // before it; frame 7's previous frame, 6, was not timed. Its
            // render's odd end puts the run's step, and so the tolerance, at
            // the floor.
            at(4, &[], None),
            at(5, &[(0, 7_000_000, 7_500_000)], None),
            at(
                7,
                &[(0, 9_000_000, 9_500_000)],
                Some(span(9_500_000, 9_800_001)),
            ),
        ];
        let o = analyse(&frames);
        assert_eq!(o.tolerance_ns, MIN_TOLERANCE_NS);
        let n = o.next_frame;
        // (0, 1): 1 ms before; (1, 2): 0.1 ms after; (2, 3): at.
        assert_eq!(
            n,
            NextFrame {
                pairs: 3,
                before_us: vec![1_000],
                at: 1,
                after_us: vec![100],
            }
        );
    }

    #[test]
    fn the_lines_say_none_without_frames_and_the_figures_with_them() {
        let mut text = String::new();
        write_lines(&mut text, &analyse(&[])).unwrap();
        assert_eq!(
            text,
            "pass_overlap_frames: frames=0 with_render_and_slices=0 overlapping=0 share=none \
             tolerance_ns=1000 max_intersection_ns=0\n\
             pass_overlap_us: none\n\
             pass_order: render_before_first_slice=0 render_after_last_slice=0 \
             render_among_slices=0\n\
             slice_vs_render_overlap: overlapped_samples=0 overlapped_ratio_mean=none \
             overlapped_ratio_p50=none overlapped_ratio_p90=none not_overlapped_samples=0 \
             not_overlapped_ratio_mean=none not_overlapped_ratio_p50=none \
             not_overlapped_ratio_p90=none\n\
             pass_overlap_compute: pairs=0 overlapping=0 share=none frames_with_pairs=0 \
             frames_overlapping=0 max_intersection_ns=0\n\
             pass_overlap_next_frame: pairs=0 first_slice_before_render_end=0 share=none \
             first_slice_at_render_end=0 first_slice_after_render_end=0 tolerance_ns=1000 \
             before_us_p50=none before_us_max=none after_us_p50=none after_us_max=none\n"
        );
        let mut text = String::new();
        write_lines(&mut text, &analyse(&chrome_like())).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(
            lines[0],
            "pass_overlap_frames: frames=10 with_render_and_slices=10 overlapping=5 \
             share=0.5000 tolerance_ns=65536 max_intersection_ns=983040"
        );
        assert_eq!(lines[1], "pass_overlap_us: p50=983 p99=983 max=983");
        assert_eq!(
            lines[3],
            "slice_vs_render_overlap: overlapped_samples=5 overlapped_ratio_mean=2.00 \
             overlapped_ratio_p50=2.00 overlapped_ratio_p90=2.00 not_overlapped_samples=15 \
             not_overlapped_ratio_mean=1.00 not_overlapped_ratio_p50=1.00 \
             not_overlapped_ratio_p90=1.00"
        );
        // Each frame begins 201 steps after the one before, long after its
        // render ended: 151 steps after for the first five, 156 for the rest.
        assert_eq!(
            lines[5],
            "pass_overlap_next_frame: pairs=9 first_slice_before_render_end=0 share=0.0000 \
             first_slice_at_render_end=0 first_slice_after_render_end=9 tolerance_ns=65536 \
             before_us_p50=none before_us_max=none after_us_p50=9896 after_us_max=10224"
        );
        for line in lines {
            let (name, value) = line.split_once(": ").unwrap();
            assert!(!name.contains(' ') && !value.is_empty(), "{line}");
        }
    }

    #[test]
    fn a_frame_is_read_from_the_timers_timing() {
        let timing = FrameTiming {
            frame: beatlm_core::time::FrameIndex::new(3),
            compute: vec![
                (7_u32, beatlm_core::time::Micros::new(40)),
                (9, beatlm_core::time::Micros::new(60)),
            ],
            render: Some(beatlm_core::time::Micros::new(100)),
            compute_spans: vec![span(0, 40_000), span(40_000, 100_000)],
            render_span: Some(span(100_000, 200_000)),
        };
        let f = Frame::of(&timing, |&tag| SliceId::new(tag));
        assert_eq!(f.index, 3);
        assert_eq!(f.compute.len(), 2);
        assert_eq!(f.compute[1].slice, SliceId::new(9));
        assert_eq!(f.compute[1].us, 60);
        assert_eq!(f.compute[1].span, span(40_000, 100_000));
        assert_eq!(f.render, Some(span(100_000, 200_000)));
        let missing = FrameTiming {
            render: None,
            render_span: None,
            ..timing
        };
        assert_eq!(Frame::of(&missing, |&tag| SliceId::new(tag)).render, None);
    }
}
