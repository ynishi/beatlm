//! The `pass-cost` workload: what a compute pass costs apart from the work it
//! dispatches, with and without its timestamp writes.
//!
//! The engine encodes each slice as its own compute pass with a pair of
//! timestamp writes (`beatlm::gpu::SliceTimer`, which owns that structure).
//! Depth Anything V2 Small is about 155 such passes per job, so a fixed cost
//! per pass, paid on the GPU between or around the dispatches, is paid 155
//! times. This workload measures that cost; it schedules nothing, estimates
//! nothing and promises nothing. It is a measurement loop like calibration
//! (`calibrate.rs`): every frame encodes a fixed pattern and reads its time.
//!
//! The browser host (`beatlm-web`) compiles this file by path, so a page and
//! the bench encode the same passes and report the same lines. It reads no
//! clock: each host measures its CPU times with its own clock and hands them
//! in, as the hosts do with everything else they share.
//!
//! # What a frame encodes
//!
//! A run is a set of **cells**, every pairing of a variant with a pass count
//! `P`. A frame runs one cell: `P` dispatches, each of one kernel, encoded as
//! the cell's variant says, and nothing else unless `render` is on. The cells
//! are interleaved frame by frame, cell `n % cells` in frame `n`, so a drift
//! of the GPU's clock over the run lands on every cell alike rather than on
//! the cells that happen to run last. The first round, one frame of each
//! cell, is untimed: it pays for first use. Frames are paced at each host's
//! frame rate, as the loop's are.
//!
//! **One frame in flight.** A frame begins only once the frame before it has
//! finished on the GPU and its timings are back, as calibration's frames do.
//! Without that, a cell whose frame outlasts the period (the depth model's 128
//! kernels at 112 pixels took about 37 ms on an Apple M1 Pro, Metal, on
//! 2026-10-22) filled the timer's readback ring, and the frames after it were
//! not timed: since the cells run in a fixed order, the skips fell on the same
//! cells every round, which got none or few timings (306 of 1 800 frames
//! skipped, one cell with no timing at all). With one frame in flight no
//! frame is skipped and the GPU never runs one frame's passes behind
//! another's; a frame longer than the period gives up the periods it
//! overran, as the loop's do.
//!
//! **Why the render pass is off by default.** With `render` on, each frame
//! ends with the synthetic render pass, untimed, as the loop's frames do, to
//! keep the GPU as busy per frame as there (`calibrate.rs` says why an idle
//! GPU measures short). It made the passes before it slower, not faster: on an
//! Apple M1 Pro (Metal, wgpu 30.0.1, 2026-10-22, 100 frames per cell, one run
//! each) one trivial pass spanned 152 us at the median with it and 16 us
//! without, and 32 dispatches in one pass 1786 us against 190 us. That is
//! consistent with the render pass, which shares no resource with the
//! dispatches, running beside them, which was not confirmed; a span would
//! then measure the contention, not the passes. So the default measures the
//! passes alone, `render=on` measures them as the loop places them, and
//! `kernels=depth` is the case of a GPU kept busy by real work.
//!
//! | Variant | Spelling | Passes | Timestamp writes | What it isolates |
//! |---|---|---|---|---|
//! | A | `per-pass` | one per dispatch | a pair on every pass | what the engine does today |
//! | B | `ends` | one per dispatch | the first pass's beginning and the last pass's end | the passes without their timestamps |
//! | C | `one-pass` | one for all `P` | a pair on the one pass | the dispatches without their passes |
//! | D | `every:<k>` | one per dispatch | a pair on every `k`-th pass and on the last | how the cost of timestamps scales with how many passes carry them |
//!
//! Every variant's **span** is the same quantity: from the beginning of the
//! frame's first pass to the end of its last, read from the timestamps the
//! variant wrote. At `P = 1` the four variants encode the same thing, one pass
//! with a pair, which is a check that they are comparable.
//!
//! The dispatches are one of two kinds (`kernels`):
//!
//! - `trivial` (the default): [`SHADER`], `groups` workgroups of one
//!   invocation, each adding one to one word of a buffer every dispatch
//!   shares, so each dispatch depends on the one before it as a model's
//!   kernels do, and none can be removed.
//! - `depth`: the first `P` kernels of Depth Anything V2 Small, unsplit and
//!   fused (natively unless `--fusion off`), at `side` pixels (default [`DEFAULT_DEPTH_SIDE`]), on one slot,
//!   each recorded through `Net::record` into the pass the variant opens for
//!   it, as `Net::encode` records a slice into a pass of its own. Their work
//!   grows with `P`, so only the comparison of variants at one `P` means
//!   anything for them.
//!
//! # What is derived
//!
//! From the medians (nearest rank, the bench's `stats.rs`) of the cells'
//! spans, per pass:
//!
//! - **`per_pass_us`** (variants A, B, D) and **`per_dispatch_us`** (C):
//!   `(span(P) - span(1)) / (P - 1)` within one variant, the cost of one more
//!   pass and its dispatch, or of one more dispatch inside the pass. Only for
//!   `trivial`, whose dispatch does the same work at every `P`, and only when
//!   `1` is among the pass counts.
//! - **`over_one_pass_us`**: `(span_X(P) - span_C(P)) / (P - 1)`, what a pass
//!   boundary costs over dispatching inside one pass, at the same dispatches;
//!   for both kinds of kernel, when C ran at that `P`.
//! - **`over_ends_us`** (A): `(span_A(P) - span_B(P)) / (P - 1)`, what one
//!   pass's pair of timestamp writes adds, when B ran at that `P`.
//! - **`between_us`** (A): per frame, the span less the sum of the passes' own
//!   durations: the time no pass's timestamps cover, which a cost model
//!   reading per-pass durations never sees.
//!
//! The CPU side is per frame, each host's clock: `encode_us`, from creating
//! the command encoder to finishing it (the render pass, when on, and the
//! timestamps' resolve included, as the loop's `cpu_encode_us`), and
//! `submit_us`, inside `queue.submit`, with `encode_per_pass_us` and
//! `submit_per_pass_us` derived across `P` as `per_pass_us` is. A difference
//! of medians can come out negative when the cost is below the noise; it is
//! printed as measured.
//!
//! A browser quantises timestamps (the page's crate documentation, "What
//! differs from the native bench"), so a span of one pass may read as zero or
//! one step; the derived figures divide a span of many passes, and are read
//! at the large `P`. The results, with the adapter, backend and date they were
//! measured on, are in the bench's crate documentation ("The `pass-cost`
//! workload").

use std::fmt::Write as _;

use anyhow::{Context, Result, bail};
use beatlm::gpu::{FrameTimestamps, MAX_SLICES, PassSpan};
use beatlm_core::cost::SliceId;

use crate::slots::Net;
use crate::stats::{nearest_rank, summarize};

/// The pass counts a run measures when none are given: from one, which every
/// derivation across counts needs, to about the depth model's 155.
pub const DEFAULT_PASSES: [u32; 4] = [1, 8, 32, 128];

/// `k` of variant D when the variants are not given.
pub const DEFAULT_EVERY: u32 = 8;

/// The depth model's side for `kernels=depth` when none is given: the side
/// the browser holds on-beat delivery at (the page's crate documentation,
/// "On-beat delivery").
pub const DEFAULT_DEPTH_SIDE: u32 = 112;

/// The most workgroups a trivial dispatch may have: WebGPU's default
/// `maxComputeWorkgroupsPerDimension`.
pub const MAX_GROUPS: u32 = 65_535;

/// The trivial kernel: one invocation per workgroup, adding one to its word
/// of a buffer every dispatch shares, so each dispatch reads what the one
/// before it wrote.
pub const SHADER: &str = "\
@group(0) @binding(0) var<storage, read_write> sink: array<u32>;

@compute @workgroup_size(1)
fn pass_cost_main(@builtin(workgroup_id) wid: vec3<u32>) {
    sink[wid.x] = sink[wid.x] + 1u;
}
";

/// How a frame's dispatches are put into passes and which passes carry
/// timestamp writes; the module documentation's table says what each
/// isolates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// A: one pass per dispatch, a pair of timestamp writes on each.
    PerPass,
    /// B: one pass per dispatch; the first pass's beginning and the last
    /// pass's end are the only timestamps.
    Ends,
    /// C: every dispatch in one pass, which carries the pair.
    OnePass,
    /// D: one pass per dispatch, a pair on every `k`-th pass, the first
    /// included, and on the last.
    Every(u32),
}

impl Variant {
    /// The variants a run measures when none are given: all four.
    pub const DEFAULT: [Self; 4] = [
        Self::PerPass,
        Self::Ends,
        Self::OnePass,
        Self::Every(DEFAULT_EVERY),
    ];

    /// The spelling the flag and the query accept and the report prints.
    pub fn name(self) -> String {
        match self {
            Self::PerPass => "per-pass".to_owned(),
            Self::Ends => "ends".to_owned(),
            Self::OnePass => "one-pass".to_owned(),
            Self::Every(k) => format!("every:{k}"),
        }
    }

    /// Parses one spelling of [`name`](Self::name).
    ///
    /// # Errors
    ///
    /// Another word, or `every:<k>` with a `k` that is not a whole number
    /// from 1.
    pub fn parse(text: &str) -> Result<Self> {
        match text {
            "per-pass" => Ok(Self::PerPass),
            "ends" => Ok(Self::Ends),
            "one-pass" => Ok(Self::OnePass),
            _ => {
                let Some(k) = text.strip_prefix("every:") else {
                    bail!("{text:?} is not one of per-pass, ends, one-pass or every:<k>");
                };
                let k: u32 = k
                    .parse()
                    .ok()
                    .with_context(|| format!("{text:?}: {k:?} is not a whole number"))?;
                if k == 0 {
                    bail!("{text:?}: k must be at least 1");
                }
                Ok(Self::Every(k))
            }
        }
    }

    /// Whether dispatch `i` of `passes` is its own pass with a pair of
    /// timestamp writes. Only variants A and D time passes one by one.
    pub fn times_pass(self, i: u32, passes: u32) -> bool {
        match self {
            Self::PerPass => true,
            Self::Every(k) => i.is_multiple_of(k) || i + 1 == passes,
            Self::Ends | Self::OnePass => false,
        }
    }
}

/// What the dispatches run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dispatches {
    /// [`SHADER`] over `groups` workgroups.
    Trivial {
        /// Workgroups per dispatch, from 1 to [`MAX_GROUPS`].
        groups: u32,
    },
    /// The first kernels of Depth Anything V2 Small at `side` pixels.
    Depth {
        /// The input's side in pixels.
        side: u32,
    },
}

/// One pairing of a variant and a pass count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cell {
    /// How the dispatches are put into passes.
    pub variant: Variant,
    /// How many dispatches a frame of the cell encodes: `P`.
    pub passes: u32,
}

impl Cell {
    /// `[variant,P]`, the cell's key in the report.
    pub fn key(self) -> String {
        format!("[{},{}]", self.variant.name(), self.passes)
    }
}

/// What a run measures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Settings {
    /// The pass counts, in the order given.
    pub passes: Vec<u32>,
    /// The variants, in the order given.
    pub variants: Vec<Variant>,
    /// What each dispatch runs.
    pub dispatches: Dispatches,
    /// Whether each frame ends with the synthetic render pass, untimed; off
    /// unless asked for (the module documentation says why).
    pub render: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            passes: DEFAULT_PASSES.to_vec(),
            variants: Variant::DEFAULT.to_vec(),
            dispatches: Dispatches::Trivial { groups: 1 },
            render: false,
        }
    }
}

impl Settings {
    /// Every pairing of a variant with a pass count, variant by variant.
    pub fn cells(&self) -> Vec<Cell> {
        self.variants
            .iter()
            .flat_map(|&variant| {
                self.passes
                    .iter()
                    .map(move |&passes| Cell { variant, passes })
            })
            .collect()
    }

    /// The most timed passes one frame can have, which the timer is built
    /// for: the largest pass count.
    pub fn capacity(&self) -> u32 {
        self.passes.iter().copied().max().unwrap_or(1).max(1)
    }

    /// Refuses a pass count above the `available` kernels of the depth model;
    /// a trivial run has no such bound.
    ///
    /// # Errors
    ///
    /// A pass count above `available` with `kernels=depth`.
    pub fn check_kernels(&self, available: usize) -> Result<()> {
        if let Dispatches::Depth { side } = self.dispatches {
            let most = usize::try_from(self.capacity()).unwrap_or(usize::MAX);
            if most > available {
                bail!(
                    "the depth model at {side} pixels has {available} kernels; a pass count of \
                     {most} needs that many"
                );
            }
        }
        Ok(())
    }

    /// The settings as the report's `pass_cost_config` line prints them,
    /// after `frames_per_cell`.
    pub fn describe(&self) -> String {
        let passes: Vec<String> = self.passes.iter().map(u32::to_string).collect();
        let variants: Vec<String> = self.variants.iter().map(|v| v.name()).collect();
        let kernels = match self.dispatches {
            Dispatches::Trivial { groups } => format!("kernels=trivial groups={groups}"),
            Dispatches::Depth { side } => format!("kernels=depth side={side}"),
        };
        format!(
            "{kernels} passes={} variants={} render={}",
            passes.join(","),
            variants.join(","),
            if self.render { "on" } else { "off" }
        )
    }
}

/// Parses a list of pass counts, `1,8,32,128`: whole numbers from 1 to
/// `beatlm::gpu::MAX_SLICES`, the most passes one frame can time, each once.
///
/// # Errors
///
/// An empty list or entry, a count out of range, or one given twice.
pub fn parse_passes(text: &str) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    for entry in text.split(',') {
        let passes: u32 = entry
            .parse()
            .ok()
            .with_context(|| format!("{entry:?} is not a whole number"))?;
        if !(1..=MAX_SLICES).contains(&passes) {
            bail!("{passes} passes: a count is from 1 to {MAX_SLICES}");
        }
        if out.contains(&passes) {
            bail!("{passes} is given twice");
        }
        out.push(passes);
    }
    Ok(out)
}

/// Parses a list of variants, `per-pass,ends,one-pass,every:8`, each once.
///
/// # Errors
///
/// An empty list or entry, a word [`Variant::parse`] refuses, or a variant
/// given twice.
pub fn parse_variants(text: &str) -> Result<Vec<Variant>> {
    let mut out = Vec::new();
    for entry in text.split(',') {
        let variant = Variant::parse(entry)?;
        if out.contains(&variant) {
            bail!("{} is given twice", variant.name());
        }
        out.push(variant);
    }
    Ok(out)
}

/// Parses a workgroup count of a trivial dispatch, 1 to [`MAX_GROUPS`].
///
/// # Errors
///
/// Not a whole number, or out of range.
pub fn parse_groups(text: &str) -> Result<u32> {
    let groups: u32 = text
        .parse()
        .ok()
        .with_context(|| format!("{text:?} is not a whole number"))?;
    if !(1..=MAX_GROUPS).contains(&groups) {
        bail!("{groups} workgroups: a dispatch has from 1 to {MAX_GROUPS}");
    }
    Ok(groups)
}

/// The trivial kernel on a device: its pipeline, the buffer it writes and the
/// workgroups of each dispatch.
pub struct Trivial {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    groups: u32,
}

impl Trivial {
    /// Builds [`SHADER`] with a buffer of one word per workgroup of `groups`.
    pub fn new(device: &wgpu::Device, groups: u32) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("beatlm pass-cost"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let sink = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("beatlm pass-cost sink"),
            size: u64::from(groups.max(1)) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("beatlm pass-cost"),
            layout: None,
            module: &module,
            entry_point: Some("pass_cost_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("beatlm pass-cost bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sink.as_entire_binding(),
            }],
        });
        Self {
            pipeline,
            bind_group,
            groups,
        }
    }
}

/// What a frame's dispatches are recorded from.
#[derive(Clone, Copy)]
pub enum Source<'a> {
    /// The trivial kernel, every dispatch the same.
    Trivial(&'a Trivial),
    /// The depth model, dispatch `i` its kernel `i`, on slot 0.
    Net(&'a Net),
}

impl Source<'_> {
    /// Records dispatch `i` into `pass`. The pipeline and bindings are set
    /// for every dispatch, as for kernels that each have their own.
    fn record(self, pass: &mut wgpu::ComputePass<'_>, i: u32) -> Result<()> {
        match self {
            Self::Trivial(t) => {
                pass.set_pipeline(&t.pipeline);
                pass.set_bind_group(0, &t.bind_group, &[]);
                pass.dispatch_workgroups(t.groups, 1, 1);
                Ok(())
            }
            Self::Net(net) => net.record(pass, 0, SliceId::new(i)),
        }
    }
}

/// The next pair of timestamp writes of a timed frame, labelled with the
/// cell's index; `None` for an untimed frame.
fn pair<'a>(
    timestamps: &mut Option<&mut FrameTimestamps<'a, usize>>,
    cell: usize,
) -> Result<Option<wgpu::ComputePassTimestampWrites<'a>>> {
    match timestamps {
        Some(t) => Ok(t.compute_pass(cell)?),
        None => Ok(None),
    }
}

/// Of variant B's one pair, the writes pass `i` of `passes` carries: the
/// beginning on the first, the end on the last, both on a pass that is both.
fn ends_writes<'a>(
    pair: &wgpu::ComputePassTimestampWrites<'a>,
    i: u32,
    passes: u32,
) -> Option<wgpu::ComputePassTimestampWrites<'a>> {
    let (begin, end) = ends_indices(
        pair.beginning_of_pass_write_index,
        pair.end_of_pass_write_index,
        i,
        passes,
    );
    (begin.is_some() || end.is_some()).then_some(wgpu::ComputePassTimestampWrites {
        query_set: pair.query_set,
        beginning_of_pass_write_index: begin,
        end_of_pass_write_index: end,
    })
}

/// [`ends_writes`] on the query indices alone: `begin` for pass 0, `end` for
/// the last pass, both for a pass that is both, neither for the rest.
fn ends_indices(
    begin: Option<u32>,
    end: Option<u32>,
    i: u32,
    passes: u32,
) -> (Option<u32>, Option<u32>) {
    (begin.filter(|_| i == 0), end.filter(|_| i + 1 == passes))
}

/// Encodes one frame of `cell`, the run's cell number `index`, into
/// `encoder`: its dispatches from `source`, in passes and with timestamp
/// writes as the cell's variant says. `timestamps` is `None` for an untimed
/// frame, whose passes are the same and carry no writes.
///
/// # Errors
///
/// A timer whose capacity is below the cell's timed passes, or a dispatch the
/// source does not have.
pub fn encode(
    encoder: &mut wgpu::CommandEncoder,
    source: Source<'_>,
    cell: Cell,
    index: usize,
    mut timestamps: Option<&mut FrameTimestamps<'_, usize>>,
) -> Result<()> {
    let label = Some("beatlm pass-cost");
    let one = |encoder: &mut wgpu::CommandEncoder,
               writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
               i: u32| {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label,
            timestamp_writes: writes,
        });
        source.record(&mut pass, i)
    };
    match cell.variant {
        Variant::PerPass | Variant::Every(_) => {
            for i in 0..cell.passes {
                let writes = if cell.variant.times_pass(i, cell.passes) {
                    pair(&mut timestamps, index)?
                } else {
                    None
                };
                one(encoder, writes, i)?;
            }
        }
        Variant::Ends => {
            let pair = pair(&mut timestamps, index)?;
            for i in 0..cell.passes {
                let writes = pair.as_ref().and_then(|p| ends_writes(p, i, cell.passes));
                one(encoder, writes, i)?;
            }
        }
        Variant::OnePass => {
            let writes = pair(&mut timestamps, index)?;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label,
                timestamp_writes: writes,
            });
            for i in 0..cell.passes {
                source.record(&mut pass, i)?;
            }
        }
    }
    Ok(())
}

/// A frame's span: from the beginning of its first timed pass to the end of
/// its last, in nanoseconds; `None` for a frame with no timed pass. An end
/// before the beginning reads as zero.
pub fn frame_span_ns(spans: &[PassSpan]) -> Option<u64> {
    let (first, last) = (spans.first()?, spans.last()?);
    Some(last.end_ns.saturating_sub(first.begin_ns))
}

/// The time of a frame's span no pass's own timestamps cover: the span less
/// the sum of the passes' durations, in nanoseconds; zero when they overlap.
pub fn between_ns(spans: &[PassSpan]) -> Option<u64> {
    let inside: u64 = spans
        .iter()
        .map(|s| s.end_ns.saturating_sub(s.begin_ns))
        .fold(0, u64::saturating_add);
    Some(frame_span_ns(spans)?.saturating_sub(inside))
}

/// The cost of one more step from `few` to `many`, measured at `steps`
/// more: `(many - few) / steps`, in the unit of the two, which may be
/// negative; `None` for no steps.
#[allow(clippy::cast_precision_loss)]
pub fn per_step(few: u64, many: u64, steps: u32) -> Option<f64> {
    // Nanosecond medians this size are exact in an f64.
    (steps > 0).then(|| (many as f64 - few as f64) / f64::from(steps))
}

/// The median by nearest rank, the report's `p50`; `None` for no samples.
pub fn p50(samples: &[u64]) -> Option<u64> {
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    nearest_rank(&sorted, 50)
}

/// What one cell measured, every duration in nanoseconds.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Samples {
    /// Each timed frame's span ([`frame_span_ns`]).
    pub span_ns: Vec<u64>,
    /// Each timed frame's time between its passes ([`between_ns`]); variant
    /// A only.
    pub between_ns: Vec<u64>,
    /// Each frame's CPU time from creating its encoder to finishing it.
    pub encode_ns: Vec<u64>,
    /// Each frame's CPU time inside `queue.submit`.
    pub submit_ns: Vec<u64>,
}

/// Every cell's samples, and the order the frames run the cells in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Measurements {
    cells: Vec<Cell>,
    samples: Vec<Samples>,
    frames_per_cell: u64,
}

impl Measurements {
    /// Empty samples for each of `cells`, each to run `frames_per_cell`
    /// timed frames.
    pub fn new(cells: Vec<Cell>, frames_per_cell: u64) -> Self {
        let samples = vec![Samples::default(); cells.len()];
        Self {
            cells,
            samples,
            frames_per_cell,
        }
    }

    /// The cells, in the order the frames cycle through them.
    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }

    /// Frames a run of these cells encodes: one untimed round, then
    /// `frames_per_cell` of each.
    pub fn total_frames(&self) -> u64 {
        let cells = u64::try_from(self.cells.len()).unwrap_or(u64::MAX);
        self.frames_per_cell.saturating_add(1).saturating_mul(cells)
    }

    /// The cell the `n`th frame of the run runs, counting from zero, and
    /// whether it is in the untimed first round.
    pub fn cell_of(&self, n: u64) -> (usize, bool) {
        let cells = u64::try_from(self.cells.len()).unwrap_or(u64::MAX).max(1);
        let index = usize::try_from(n % cells).unwrap_or(0);
        (index, n < cells)
    }

    /// One frame of cell `index`'s CPU times.
    pub fn record_cpu(&mut self, index: usize, encode_ns: u64, submit_ns: u64) {
        if let Some(s) = self.samples.get_mut(index) {
            s.encode_ns.push(encode_ns);
            s.submit_ns.push(submit_ns);
        }
    }

    /// One timed frame of cell `index`: its timed passes' timestamps, in the
    /// order they were requested. A frame with fewer pairs than its variant
    /// writes, which the timer leaves out when it cannot read them, is not
    /// recorded.
    pub fn record_spans(&mut self, index: usize, spans: &[PassSpan]) {
        let (Some(cell), Some(s)) = (self.cells.get(index), self.samples.get_mut(index)) else {
            return;
        };
        let expected = match cell.variant {
            Variant::Ends | Variant::OnePass => 1,
            v => (0..cell.passes)
                .filter(|&i| v.times_pass(i, cell.passes))
                .count(),
        };
        if spans.len() != expected {
            return;
        }
        if let Some(span) = frame_span_ns(spans) {
            s.span_ns.push(span);
        }
        if cell.variant == Variant::PerPass {
            s.between_ns.extend(between_ns(spans));
        }
    }

    /// The samples of the cell of `variant` at `passes`, when the run has it.
    fn find(&self, variant: Variant, passes: u32) -> Option<&Samples> {
        self.cells
            .iter()
            .position(|c| c.variant == variant && c.passes == passes)
            .and_then(|i| self.samples.get(i))
    }

    /// The report's lines for the run: `pass_cost_config`, then for each
    /// cell `pass_cost[variant,P]` and `pass_cost_cpu[variant,P]`. The module
    /// documentation defines every value; a value that cannot be derived is
    /// `none`.
    ///
    /// # Errors
    ///
    /// Only the writer's.
    pub fn write_report(&self, out: &mut String, settings: &Settings) -> std::fmt::Result {
        writeln!(
            out,
            "pass_cost_config: frames_per_cell={} cells={} order=interleaved {}",
            self.frames_per_cell,
            self.cells.len(),
            settings.describe()
        )?;
        let trivial = matches!(settings.dispatches, Dispatches::Trivial { .. });
        for (cell, s) in self.cells.iter().zip(&self.samples) {
            let key = cell.key();
            let steps = cell.passes.saturating_sub(1);
            let one = self.find(cell.variant, 1);
            let across = |pick: fn(&Samples) -> &Vec<u64>| -> String {
                let derived = one.and_then(|one| per_step(p50(pick(one))?, p50(pick(s))?, steps));
                us2(derived)
            };
            let against = |other: Variant| -> String {
                let derived = self
                    .find(other, cell.passes)
                    .and_then(|o| per_step(p50(&o.span_ns)?, p50(&s.span_ns)?, steps));
                us2(derived)
            };
            write!(
                out,
                "pass_cost{key}: timings={} span_us {}",
                s.span_ns.len(),
                summary_us(&s.span_ns)
            )?;
            let per_step_name = if cell.variant == Variant::OnePass {
                "per_dispatch_us"
            } else {
                "per_pass_us"
            };
            let per_step_value = if trivial {
                across(|s| &s.span_ns)
            } else {
                "none".to_owned()
            };
            write!(out, " {per_step_name}={per_step_value}")?;
            if cell.variant != Variant::OnePass {
                write!(out, " over_one_pass_us={}", against(Variant::OnePass))?;
            }
            if cell.variant == Variant::PerPass {
                write!(
                    out,
                    " over_ends_us={} between_us {}",
                    against(Variant::Ends),
                    summary_us(&s.between_ns)
                )?;
            }
            writeln!(out)?;
            writeln!(
                out,
                "pass_cost_cpu{key}: frames={} encode_us {} submit_us {} encode_{per_step_name}={} \
                 submit_{per_step_name}={}",
                s.encode_ns.len(),
                summary_us(&s.encode_ns),
                summary_us(&s.submit_ns),
                across(|s| &s.encode_ns),
                across(|s| &s.submit_ns),
            )?;
        }
        Ok(())
    }
}

/// `p50=… p99=… max=…` of nanosecond samples in microseconds to one decimal,
/// by nearest rank as the bench's `summarize`; `none` when there are none.
fn summary_us(samples_ns: &[u64]) -> String {
    match summarize(samples_ns) {
        Some(s) => format!("p50={} p99={} max={}", us1(s.p50), us1(s.p99), us1(s.max)),
        None => "none".to_owned(),
    }
}

/// Nanoseconds as microseconds to one decimal.
#[allow(clippy::cast_precision_loss)]
fn us1(ns: u64) -> String {
    format!("{:.1}", ns as f64 / 1_000.0)
}

/// A derived figure in nanoseconds as microseconds to two decimals, or
/// `none`.
fn us2(ns: Option<f64>) -> String {
    ns.map_or_else(|| "none".to_owned(), |ns| format!("{:.2}", ns / 1_000.0))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn span(begin_ns: u64, end_ns: u64) -> PassSpan {
        PassSpan { begin_ns, end_ns }
    }

    #[test]
    fn variants_parse_their_own_names_and_nothing_else() {
        for v in Variant::DEFAULT {
            assert_eq!(Variant::parse(&v.name()).unwrap(), v);
        }
        assert_eq!(Variant::parse("every:1").unwrap(), Variant::Every(1));
        for bad in [
            "", "A", "per_pass", "every", "every:", "every:0", "every:-2", "every:x",
        ] {
            assert!(Variant::parse(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn lists_parse_in_order_and_refuse_repeats_gaps_and_ranges() {
        assert_eq!(parse_passes("1,8,32,128").unwrap(), vec![1, 8, 32, 128]);
        assert_eq!(parse_passes("64,1").unwrap(), vec![64, 1]);
        assert_eq!(
            parse_passes(&MAX_SLICES.to_string()).unwrap(),
            vec![MAX_SLICES]
        );
        for bad in ["", "0", "1,,8", "1,1", "8,", "x", "-1"] {
            assert!(parse_passes(bad).is_err(), "{bad:?}");
        }
        assert!(parse_passes(&(MAX_SLICES + 1).to_string()).is_err());
        assert_eq!(
            parse_variants("one-pass,every:4,per-pass").unwrap(),
            vec![Variant::OnePass, Variant::Every(4), Variant::PerPass]
        );
        assert!(parse_variants("ends,ends").is_err());
        assert!(parse_variants("every:4,every:4").is_err());
        assert!(parse_variants("ends,").is_err());
        assert_eq!(parse_groups("1").unwrap(), 1);
        assert_eq!(parse_groups("65535").unwrap(), MAX_GROUPS);
        assert!(parse_groups("0").is_err());
        assert!(parse_groups("65536").is_err());
    }

    #[test]
    fn every_kth_times_the_first_every_kth_and_the_last() {
        let timed: Vec<u32> = (0..10)
            .filter(|&i| Variant::Every(4).times_pass(i, 10))
            .collect();
        assert_eq!(timed, vec![0, 4, 8, 9]);
        assert!((0..3).all(|i| Variant::PerPass.times_pass(i, 3)));
        assert!(Variant::Every(1).times_pass(5, 9));
        assert!(Variant::Every(8).times_pass(0, 1));
    }

    #[test]
    fn ends_splits_one_pair_between_the_first_and_the_last_pass() {
        let (b, e) = (Some(6), Some(7));
        assert_eq!(ends_indices(b, e, 0, 1), (b, e));
        assert_eq!(ends_indices(b, e, 0, 3), (b, None));
        assert_eq!(ends_indices(b, e, 1, 3), (None, None));
        assert_eq!(ends_indices(b, e, 2, 3), (None, e));
        // An untimed frame's pair has no indices, so no pass gets any.
        assert_eq!(ends_indices(None, None, 0, 1), (None, None));
    }

    #[test]
    fn cells_pair_every_variant_with_every_count_variant_by_variant() {
        let settings = Settings {
            passes: vec![1, 8],
            variants: vec![Variant::PerPass, Variant::OnePass],
            ..Settings::default()
        };
        let keys: Vec<String> = settings.cells().into_iter().map(Cell::key).collect();
        assert_eq!(
            keys,
            [
                "[per-pass,1]",
                "[per-pass,8]",
                "[one-pass,1]",
                "[one-pass,8]"
            ]
        );
        assert_eq!(settings.capacity(), 8);
        assert!(settings.check_kernels(1).is_ok());
        let depth = Settings {
            dispatches: Dispatches::Depth { side: 112 },
            ..settings
        };
        assert!(depth.check_kernels(8).is_ok());
        assert!(depth.check_kernels(7).is_err());
        assert_eq!(
            Settings::default().describe(),
            "kernels=trivial groups=1 passes=1,8,32,128 variants=per-pass,ends,one-pass,every:8 \
             render=off"
        );
    }

    #[test]
    fn frames_cycle_through_the_cells_after_one_untimed_round() {
        let cells = Settings::default().cells();
        let m = Measurements::new(cells, 600);
        assert_eq!(m.total_frames(), 601 * 16);
        assert_eq!(m.cell_of(0), (0, true));
        assert_eq!(m.cell_of(15), (15, true));
        assert_eq!(m.cell_of(16), (0, false));
        assert_eq!(m.cell_of(16 * 3 + 5), (5, false));
    }

    #[test]
    fn a_span_runs_from_the_first_beginning_to_the_last_end() {
        assert_eq!(frame_span_ns(&[]), None);
        assert_eq!(frame_span_ns(&[span(100, 400)]), Some(300));
        let passes = [span(100, 200), span(250, 300), span(310, 500)];
        assert_eq!(frame_span_ns(&passes), Some(400));
        // 400 of span, 100 + 50 + 190 inside the passes.
        assert_eq!(between_ns(&passes), Some(60));
        assert_eq!(between_ns(&[span(0, 50), span(10, 60)]), Some(0));
        assert_eq!(frame_span_ns(&[span(500, 100)]), Some(0));
    }

    #[test]
    fn a_step_is_the_difference_over_the_steps_between() {
        assert_eq!(per_step(1_000, 8_000, 7), Some(1_000.0));
        assert_eq!(per_step(5_000, 4_000, 4), Some(-250.0));
        assert_eq!(per_step(1_000, 1_000, 0), None);
        assert_eq!(p50(&[30, 10, 20]), Some(20));
        assert_eq!(p50(&[]), None);
    }

    #[test]
    fn a_frame_with_fewer_pairs_than_its_variant_writes_is_left_out() {
        let cells = vec![
            Cell {
                variant: Variant::Every(4),
                passes: 10,
            },
            Cell {
                variant: Variant::Ends,
                passes: 10,
            },
        ];
        let mut m = Measurements::new(cells, 1);
        let four = [span(0, 1), span(4, 5), span(8, 9), span(9, 10)];
        m.record_spans(0, &four);
        m.record_spans(0, &four[..3]);
        m.record_spans(1, &[span(0, 10)]);
        m.record_spans(1, &four);
        m.record_spans(7, &four);
        assert_eq!(m.samples[0].span_ns, vec![10]);
        assert_eq!(m.samples[1].span_ns, vec![10]);
        assert!(m.samples[0].between_ns.is_empty());
    }

    #[test]
    fn the_report_derives_per_pass_costs_from_the_medians() {
        let settings = Settings {
            passes: vec![1, 5],
            variants: vec![Variant::PerPass, Variant::Ends, Variant::OnePass],
            render: true,
            ..Settings::default()
        };
        let mut m = Measurements::new(settings.cells(), 3);
        // A at 1 and 5 passes: spans of 10 us and 50 us, 2 us between passes.
        let a = |passes: u64, pass_ns: u64, gap_ns: u64| -> Vec<PassSpan> {
            (0..passes)
                .map(|i| {
                    let begin = i * (pass_ns + gap_ns);
                    span(begin, begin + pass_ns)
                })
                .collect()
        };
        for _ in 0..3 {
            m.record_spans(0, &a(1, 10_000, 0));
            m.record_spans(1, &a(5, 8_000, 2_000));
            m.record_spans(2, &[span(0, 10_000)]);
            m.record_spans(3, &[span(0, 30_000)]);
            m.record_spans(4, &[span(0, 10_000)]);
            m.record_spans(5, &[span(0, 14_000)]);
            // 100 us to encode one dispatch, 140 us for five.
            for cell in 0..6 {
                let encode = if cell % 2 == 1 { 140_000 } else { 100_000 };
                m.record_cpu(cell, encode, 20_000);
            }
        }
        let mut out = String::new();
        m.write_report(&mut out, &settings).unwrap();
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(
            lines[0],
            "pass_cost_config: frames_per_cell=3 cells=6 order=interleaved kernels=trivial \
             groups=1 passes=1,5 variants=per-pass,ends,one-pass render=on"
        );
        assert_eq!(
            lines[1],
            "pass_cost[per-pass,1]: timings=3 span_us p50=10.0 p99=10.0 max=10.0 \
             per_pass_us=none over_one_pass_us=none over_ends_us=none \
             between_us p50=0.0 p99=0.0 max=0.0"
        );
        // Span 48 us: (48 - 10) / 4, against one pass's (48 - 14) / 4 and
        // the ends' (48 - 30) / 4.
        assert_eq!(
            lines[3],
            "pass_cost[per-pass,5]: timings=3 span_us p50=48.0 p99=48.0 max=48.0 \
             per_pass_us=9.50 over_one_pass_us=8.50 over_ends_us=4.50 \
             between_us p50=8.0 p99=8.0 max=8.0"
        );
        assert_eq!(
            lines[4],
            "pass_cost_cpu[per-pass,5]: frames=3 encode_us p50=140.0 p99=140.0 max=140.0 \
             submit_us p50=20.0 p99=20.0 max=20.0 encode_per_pass_us=10.00 \
             submit_per_pass_us=0.00"
        );
        assert_eq!(
            lines[7],
            "pass_cost[ends,5]: timings=3 span_us p50=30.0 p99=30.0 max=30.0 \
             per_pass_us=5.00 over_one_pass_us=4.00"
        );
        assert_eq!(
            lines[11],
            "pass_cost[one-pass,5]: timings=3 span_us p50=14.0 p99=14.0 max=14.0 \
             per_dispatch_us=1.00"
        );
        assert!(lines[12].ends_with("encode_per_dispatch_us=10.00 submit_per_dispatch_us=0.00"));
    }

    #[test]
    fn the_real_kernels_derive_nothing_across_counts() {
        let settings = Settings {
            passes: vec![1, 5],
            variants: vec![Variant::PerPass, Variant::OnePass],
            dispatches: Dispatches::Depth { side: 112 },
            render: false,
        };
        let mut m = Measurements::new(settings.cells(), 1);
        m.record_spans(0, &[span(0, 1_000)]);
        m.record_spans(
            1,
            &(0..5)
                .map(|i| span(i * 2_000, i * 2_000 + 1_000))
                .collect::<Vec<_>>(),
        );
        m.record_spans(2, &[span(0, 1_000)]);
        m.record_spans(3, &[span(0, 5_000)]);
        let mut out = String::new();
        m.write_report(&mut out, &settings).unwrap();
        let line = out
            .lines()
            .find(|l| l.starts_with("pass_cost[per-pass,5]"))
            .unwrap();
        assert!(
            line.contains(" per_pass_us=none over_one_pass_us=1.00 "),
            "{line}"
        );
    }

    /// The trivial kernel is compiled only on a device, so it is validated
    /// here, by naga with every check on and no optional feature, as the
    /// generated kernels are.
    #[test]
    fn the_trivial_kernel_validates_without_optional_features() {
        beatlm::codegen::validate(
            "pass-cost",
            SHADER,
            beatlm_core::device::DeviceFeatures::default(),
        )
        .unwrap();
    }
}
