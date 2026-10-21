//! Per-pass GPU timing: timestamp writes in, measured durations out.
//!
//! A frame's timing follows one path. [`SliceTimer::begin_frame`] claims a slot
//! of the readback ring and hands out timestamp writes for the frame's passes.
//! [`FrameTimestamps::resolve`] records the resolve and the copy into the slot
//! at the end of the frame's encoder. After the host submits,
//! [`SliceTimer::start_readback`] requests the map. Once the host has polled the
//! device, [`SliceTimer::collect_ready`] returns every frame whose map has
//! completed, oldest first.
//!
//! Queries are allocated in the order passes are requested, two per pass, so the
//! queries a frame resolves are always the contiguous prefix it wrote. A frame
//! with fewer slices than the timer's capacity resolves fewer queries, and never
//! reads one nothing wrote.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use beatlm_core::time::{FrameIndex, Micros};
use thiserror::Error;

use super::ring::{Phase, Ring};

/// How many frames of timing can be in flight at once.
///
/// Three covers the frame being recorded, the one the GPU is running and the
/// one whose map is being collected, so a host that polls once per frame never
/// finds its slot busy while the GPU keeps up.
pub const RING_DEPTH: usize = 3;

/// The most timed compute passes one frame can have: one query set holds
/// [`wgpu::QUERY_SET_MAX_QUERIES`] queries, two per pass, and the frame's
/// render pass takes one pair. A host reads it to refuse a plan whose slices
/// cannot all be timed before it builds anything, instead of failing in
/// [`SliceTimer::new`].
pub const MAX_SLICES: u32 = (wgpu::QUERY_SET_MAX_QUERIES / 2).saturating_sub(1);

const MAP_PENDING: u8 = 0;
const MAP_OK: u8 = 1;
const MAP_FAILED: u8 = 2;

/// Why a timer could not be built or used.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum TimerError {
    /// Timestamps inside passes need the feature; asking the timer to work
    /// without it would return zeros that look like cheap slices.
    #[error("the device was created without wgpu::Features::TIMESTAMP_QUERY")]
    MissingFeature,
    /// A period that is zero, negative or not a number cannot convert ticks.
    #[error("the queue reports an unusable timestamp period: {period_ns} ns per tick")]
    InvalidPeriod {
        /// The period the queue reported, in nanoseconds per tick.
        period_ns: f32,
    },
    /// The capacity must be at least one slice and fit one query set.
    #[error("slice capacity {requested} is outside 1..={max}")]
    InvalidCapacity {
        /// The capacity asked for.
        requested: u32,
        /// The largest capacity one query set can hold.
        max: u32,
    },
    /// More compute passes were timed in a frame than the timer was built for.
    #[error("more than {capacity} compute passes timed in one frame")]
    TooManySlices {
        /// The capacity the timer was built with.
        capacity: u32,
    },
    /// A frame has one render pass to time.
    #[error("the render pass of this frame is already timed")]
    RenderPassAlreadyTimed,
}

/// When one pass began and ended, in nanoseconds on the queue's timestamp
/// clock: the two timestamps its duration was computed from, converted with
/// the queue's period but not otherwise changed.
///
/// A duration says how long a pass's boundaries were apart; only the
/// timestamps say where the pass sat against the passes before it, such as
/// whether it began the moment the previous one ended, or before. The clock's
/// origin is the implementation's, so only differences between timestamps
/// mean anything, and a host that compares timestamps of different frames
/// relies on their queries resolving on one clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PassSpan {
    /// The pass's beginning-of-pass timestamp, in nanoseconds.
    pub begin_ns: u64,
    /// The pass's end-of-pass timestamp, in nanoseconds.
    pub end_ns: u64,
}

/// What one frame's passes took on the GPU.
///
/// `T` is whatever the host attached to each compute pass, so a measurement
/// reaches the [`CostModel`](beatlm_core::cost::CostModel) under the same
/// identity the scheduler handed out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameTiming<T> {
    /// The frame the passes were recorded for.
    pub frame: FrameIndex,
    /// The duration of each timed compute pass, in the order they were
    /// requested.
    pub compute: Vec<(T, Micros)>,
    /// The duration of the render pass, when the frame timed one.
    pub render: Option<Micros>,
    /// The timestamps of each entry of [`compute`](Self::compute), in the
    /// same order.
    pub compute_spans: Vec<PassSpan>,
    /// The timestamps of the render pass, when the frame timed one.
    pub render_span: Option<PassSpan>,
}

/// Counts of what the timer could and could not measure.
///
/// A skipped frame is a measurement the cost model never saw, so the count is
/// what tells a reader how much to trust the estimates built from the rest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TimerStats {
    /// Frames not timed because their ring slot was still in flight.
    pub frames_skipped: u64,
    /// Frames whose timing was collected.
    pub frames_collected: u64,
    /// Frames whose map failed, so their timing was lost.
    pub frames_failed: u64,
}

#[derive(Debug)]
enum Entry<T> {
    Compute(T),
    Render,
}

#[derive(Debug)]
struct Slot<T> {
    readback: wgpu::Buffer,
    done: Arc<AtomicU8>,
    frame: FrameIndex,
    order: u64,
    entries: Vec<Entry<T>>,
}

/// Times the compute passes and the render pass of each frame on the GPU.
///
/// It owns a timestamp query set, a buffer the queries resolve into, and a ring
/// of [`RING_DEPTH`] readback buffers. The query set and the resolve buffer are
/// shared by all frames: a frame resolves and copies its queries inside its own
/// encoder, and the queue runs encoders in order, so the next frame may reuse
/// them before the previous frame's map completes. Only the readback buffers
/// must outlive a frame, which is why only they are in a ring.
///
/// Requires [`wgpu::Features::TIMESTAMP_QUERY`], which times a pass from its
/// boundaries; that is the granularity the scheduler plans in, since a slice is
/// one pass. `T` is the host's label for a compute pass, returned with its
/// duration.
///
/// # Driving it without blocking
///
/// The timer never polls the device. Natively, the map callbacks run when the
/// host calls [`wgpu::Device::poll`], and a poll that does not wait is
/// `device.poll(wgpu::PollType::Poll)`; the callbacks of every submission
/// already finished by then have run when it returns. In a browser `poll` does
/// nothing and the callbacks run from the event loop. Either way,
/// [`collect_ready`](Self::collect_ready) returns what has completed so far and
/// leaves the rest for a later call.
#[derive(Debug)]
pub struct SliceTimer<T> {
    query_set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    slots: Vec<Slot<T>>,
    ring: Ring,
    period_ns: f32,
    max_slices: u32,
    sequence: u64,
    stats: TimerStats,
}

impl<T> SliceTimer<T> {
    /// A timer for frames of up to `max_slices` timed compute passes plus one
    /// render pass.
    ///
    /// The timestamp period is read from `queue` once, here.
    ///
    /// # Errors
    ///
    /// [`TimerError::MissingFeature`] when `device` lacks
    /// [`wgpu::Features::TIMESTAMP_QUERY`]; [`TimerError::InvalidPeriod`] when
    /// the queue reports a period that cannot convert ticks;
    /// [`TimerError::InvalidCapacity`] when `max_slices` is zero or needs more
    /// queries than one query set holds.
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        max_slices: u32,
    ) -> Result<Self, TimerError> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return Err(TimerError::MissingFeature);
        }
        let period_ns = queue.get_timestamp_period();
        if !(period_ns.is_finite() && period_ns > 0.0) {
            return Err(TimerError::InvalidPeriod { period_ns });
        }
        if max_slices == 0 || max_slices > MAX_SLICES {
            return Err(TimerError::InvalidCapacity {
                requested: max_slices,
                max: MAX_SLICES,
            });
        }
        // One pair per slice and one for the render pass.
        let queries = (max_slices + 1) * 2;
        let bytes = u64::from(queries) * u64::from(wgpu::QUERY_SIZE);

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("beatlm timer queries"),
            ty: wgpu::QueryType::Timestamp,
            count: queries,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("beatlm timer resolve"),
            size: bytes,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let slots = (0..RING_DEPTH)
            .map(|_| Slot {
                readback: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("beatlm timer readback"),
                    size: bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                done: Arc::new(AtomicU8::new(MAP_PENDING)),
                frame: FrameIndex::new(0),
                order: 0,
                entries: Vec::new(),
            })
            .collect();

        Ok(Self {
            query_set,
            resolve,
            slots,
            ring: Ring::new(RING_DEPTH),
            period_ns,
            max_slices,
            sequence: 0,
            stats: TimerStats::default(),
        })
    }

    /// Starts timing `frame` and returns the handle that hands out its
    /// timestamp writes.
    ///
    /// When the ring slot the frame would use is still in flight the frame is
    /// not timed: the skip is counted in [`TimerStats::frames_skipped`], the
    /// handle hands out no writes, and nothing waits.
    pub fn begin_frame(&mut self, frame: FrameIndex) -> FrameTimestamps<'_, T> {
        let claimed = self.ring.claim();
        match claimed {
            Some(i) => {
                self.sequence = self.sequence.saturating_add(1);
                if let Some(slot) = self.slots.get_mut(i) {
                    slot.frame = frame;
                    slot.order = self.sequence;
                    slot.entries.clear();
                }
            }
            None => {
                self.stats.frames_skipped = self.stats.frames_skipped.saturating_add(1);
            }
        }
        FrameTimestamps {
            query_set: &self.query_set,
            resolve: &self.resolve,
            slots: &mut self.slots,
            ring: &mut self.ring,
            claimed,
            capacity: self.max_slices,
            compute_used: 0,
            render_timed: false,
        }
    }

    /// Requests the map of every frame whose resolve has been recorded.
    ///
    /// Call it after the submission carrying the frame's encoder: a buffer that
    /// a not-yet-submitted command buffer will write must not be mapped.
    pub fn start_readback(&mut self) {
        for i in 0..self.slots.len() {
            if !self.ring.advance(i, Phase::Encoded, Phase::Mapping) {
                continue;
            }
            let Some(slot) = self.slots.get(i) else {
                continue;
            };
            let bytes = readback_bytes(slot.entries.len());
            slot.done.store(MAP_PENDING, Ordering::Release);
            let done = Arc::clone(&slot.done);
            slot.readback
                .slice(0..bytes)
                .map_async(wgpu::MapMode::Read, move |result| {
                    let state = if result.is_ok() { MAP_OK } else { MAP_FAILED };
                    done.store(state, Ordering::Release);
                });
        }
    }

    /// Returns the timing of every frame whose map has completed, oldest first,
    /// and frees their slots.
    ///
    /// It never blocks and never polls the device: a frame whose map has not
    /// completed stays in flight for a later call. See the type documentation
    /// for how to poll without waiting.
    pub fn collect_ready(&mut self) -> Vec<FrameTiming<T>> {
        let mut ready: Vec<usize> = self
            .ring
            .slots_in(Phase::Mapping)
            .filter(|&i| {
                self.slots
                    .get(i)
                    .is_some_and(|s| s.done.load(Ordering::Acquire) != MAP_PENDING)
            })
            .collect();
        ready.sort_by_key(|&i| self.slots.get(i).map_or(0, |s| s.order));

        let mut out = Vec::with_capacity(ready.len());
        for i in ready {
            let Some(slot) = self.slots.get_mut(i) else {
                continue;
            };
            // A failed map leaves the buffer unmapped, and unmapping it again is
            // a validation error, which wgpu's default handler turns into a panic.
            let ticks = if slot.done.load(Ordering::Acquire) == MAP_OK {
                let ticks = read_ticks(&slot.readback, slot.entries.len());
                slot.readback.unmap();
                ticks
            } else {
                None
            };
            let entries = std::mem::take(&mut slot.entries);
            let frame = slot.frame;
            self.ring.release(i);
            match ticks {
                Some(ticks) => {
                    self.stats.frames_collected = self.stats.frames_collected.saturating_add(1);
                    out.push(decode(frame, entries, &ticks, self.period_ns));
                }
                None => {
                    self.stats.frames_failed = self.stats.frames_failed.saturating_add(1);
                }
            }
        }
        out
    }

    /// How many frames are recorded or requested and not yet collected.
    #[must_use]
    pub fn in_flight(&self) -> usize {
        self.ring.in_flight()
    }

    /// What the timer has and has not measured so far.
    #[must_use]
    pub fn stats(&self) -> TimerStats {
        self.stats
    }

    /// The most compute passes one frame may time.
    #[must_use]
    pub fn max_slices(&self) -> u32 {
        self.max_slices
    }
}

/// One frame's timestamp writes, and the resolve that ends it.
///
/// Each request for a pass's writes allocates the next pair of queries. The
/// writes must be used by the pass they were requested for; a pair that is
/// requested and never written leaves a stale value in the resolved data.
/// Dropping the handle without resolving gives the ring slot back, so an
/// abandoned frame costs nothing.
#[derive(Debug)]
#[must_use = "the frame's timing is lost unless the handle is resolved into the encoder"]
pub struct FrameTimestamps<'a, T> {
    query_set: &'a wgpu::QuerySet,
    resolve: &'a wgpu::Buffer,
    slots: &'a mut Vec<Slot<T>>,
    ring: &'a mut Ring,
    claimed: Option<usize>,
    capacity: u32,
    compute_used: u32,
    render_timed: bool,
}

impl<'a, T> FrameTimestamps<'a, T> {
    /// Whether this frame is being timed. A frame whose ring slot was busy is
    /// not, and then every request returns `None`.
    #[must_use]
    pub fn is_timed(&self) -> bool {
        self.claimed.is_some()
    }

    /// The timestamp writes for the next compute pass, labelled `tag`.
    ///
    /// Passes are measured in the order they are requested. `None` means the
    /// frame is not timed and the pass runs without timestamps.
    ///
    /// # Errors
    ///
    /// [`TimerError::TooManySlices`] when the frame already timed as many
    /// compute passes as the timer's capacity.
    pub fn compute_pass(
        &mut self,
        tag: T,
    ) -> Result<Option<wgpu::ComputePassTimestampWrites<'a>>, TimerError> {
        if self.compute_used >= self.capacity {
            return Err(TimerError::TooManySlices {
                capacity: self.capacity,
            });
        }
        self.compute_used += 1;
        let Some((begin, end)) = self.next_pair(Entry::Compute(tag)) else {
            return Ok(None);
        };
        Ok(Some(wgpu::ComputePassTimestampWrites {
            query_set: self.query_set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        }))
    }

    /// The timestamp writes for the frame's render pass.
    ///
    /// `None` means the frame is not timed.
    ///
    /// # Errors
    ///
    /// [`TimerError::RenderPassAlreadyTimed`] on a second request in a frame.
    pub fn render_pass(
        &mut self,
    ) -> Result<Option<wgpu::RenderPassTimestampWrites<'a>>, TimerError> {
        if self.render_timed {
            return Err(TimerError::RenderPassAlreadyTimed);
        }
        self.render_timed = true;
        let Some((begin, end)) = self.next_pair(Entry::Render) else {
            return Ok(None);
        };
        Ok(Some(wgpu::RenderPassTimestampWrites {
            query_set: self.query_set,
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        }))
    }

    /// Records the resolve of the frame's queries and their copy into the
    /// frame's readback slot at the end of `encoder`.
    ///
    /// Call it after the last timed pass of the frame, then submit the encoder,
    /// then call [`SliceTimer::start_readback`].
    pub fn resolve(mut self, encoder: &mut wgpu::CommandEncoder) {
        let Some(i) = self.claimed.take() else {
            return;
        };
        let Some(slot) = self.slots.get(i) else {
            self.ring.release(i);
            return;
        };
        let bytes = readback_bytes(slot.entries.len());
        let Ok(queries) = u32::try_from(slot.entries.len() * 2) else {
            self.ring.release(i);
            return;
        };
        if queries == 0 {
            self.ring.release(i);
            return;
        }
        encoder.resolve_query_set(self.query_set, 0..queries, self.resolve, 0);
        encoder.copy_buffer_to_buffer(self.resolve, 0, &slot.readback, 0, bytes);
        self.ring.advance(i, Phase::Recording, Phase::Encoded);
    }

    /// Allocates the next pair of queries to `entry`, or `None` when the frame
    /// is not timed.
    fn next_pair(&mut self, entry: Entry<T>) -> Option<(u32, u32)> {
        let slot = self.slots.get_mut(self.claimed?)?;
        let begin = u32::try_from(slot.entries.len() * 2).ok()?;
        slot.entries.push(entry);
        Some((begin, begin + 1))
    }
}

impl<T> Drop for FrameTimestamps<'_, T> {
    fn drop(&mut self) {
        if let Some(i) = self.claimed.take() {
            self.ring.release(i);
        }
    }
}

/// The bytes a frame with `entries` timed passes resolves: eight per query,
/// two queries per pass.
fn readback_bytes(entries: usize) -> u64 {
    (entries as u64) * 2 * u64::from(wgpu::QUERY_SIZE)
}

/// Reads the resolved timestamps out of a mapped readback buffer.
///
/// `None` when the range cannot be read, which the caller counts as a failed
/// frame.
fn read_ticks(readback: &wgpu::Buffer, entries: usize) -> Option<Vec<u64>> {
    let view = readback
        .slice(0..readback_bytes(entries))
        .get_mapped_range()
        .ok()?;
    Some(parse_ticks(&view))
}

/// Interprets `bytes` as little-endian `u64` timestamps; a trailing partial
/// value is dropped.
fn parse_ticks(bytes: &[u8]) -> Vec<u64> {
    bytes
        .chunks_exact(8)
        .filter_map(|chunk| <[u8; 8]>::try_from(chunk).ok())
        .map(u64::from_le_bytes)
        .collect()
}

/// Turns a frame's resolved timestamps into durations, pairing each pass with
/// the label it was requested under.
///
/// A pass whose pair is missing from `ticks` is left out rather than reported
/// as free.
fn decode<T>(
    frame: FrameIndex,
    entries: Vec<Entry<T>>,
    ticks: &[u64],
    period_ns: f32,
) -> FrameTiming<T> {
    let mut compute = Vec::new();
    let mut compute_spans = Vec::new();
    let mut render = None;
    let mut render_span = None;
    for (k, entry) in entries.into_iter().enumerate() {
        let (Some(&begin), Some(&end)) = (ticks.get(k * 2), ticks.get(k * 2 + 1)) else {
            continue;
        };
        let duration = ticks_to_micros(begin, end, period_ns);
        let span = PassSpan {
            begin_ns: ticks_to_nanos(begin, period_ns),
            end_ns: ticks_to_nanos(end, period_ns),
        };
        match entry {
            Entry::Compute(tag) => {
                compute.push((tag, duration));
                compute_spans.push(span);
            }
            Entry::Render => {
                render = Some(duration);
                render_span = Some(span);
            }
        }
    }
    FrameTiming {
        frame,
        compute,
        render,
        compute_spans,
        render_span,
    }
}

/// A timestamp of `ticks` in nanoseconds, rounded to nearest, at `period_ns`
/// nanoseconds per tick. One nanosecond per tick, what WebGPU reports, keeps
/// the value exact; a product past `u64` saturates.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp
)]
fn ticks_to_nanos(ticks: u64, period_ns: f32) -> u64 {
    const U64_LIMIT: f64 = 18_446_744_073_709_551_616.0; // 2^64
    if period_ns == 1.0 {
        return ticks;
    }
    let nanos = (ticks as f64 * f64::from(period_ns)).round();
    if nanos.is_nan() || nanos <= 0.0 {
        0
    } else if nanos >= U64_LIMIT {
        u64::MAX
    } else {
        nanos as u64
    }
}

/// The duration between two timestamps, in microseconds, rounded to nearest.
///
/// `period_ns` is the queue's nanoseconds per tick. A pair whose end precedes
/// its begin reads as zero rather than as a huge duration.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn ticks_to_micros(begin: u64, end: u64, period_ns: f32) -> Micros {
    const U64_LIMIT: f64 = 18_446_744_073_709_551_616.0; // 2^64
    let ticks = end.saturating_sub(begin);
    let micros = (ticks as f64 * f64::from(period_ns) / 1_000.0).round();
    if micros.is_nan() || micros <= 0.0 {
        Micros::ZERO
    } else if micros >= U64_LIMIT {
        Micros::new(u64::MAX)
    } else {
        Micros::new(micros as u64)
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn ticks_convert_with_the_period() {
        // 41.667 ns per tick (24 MHz): 24_000 ticks are a millisecond.
        assert_eq!(ticks_to_micros(0, 24_000, 41.666_668), Micros::new(1_000));
        // One tick per nanosecond.
        assert_eq!(ticks_to_micros(500, 2_500, 1.0), Micros::new(2));
    }

    #[test]
    fn conversion_rounds_to_nearest() {
        assert_eq!(ticks_to_micros(0, 1_499, 1.0), Micros::new(1));
        assert_eq!(ticks_to_micros(0, 1_500, 1.0), Micros::new(2));
        assert_eq!(ticks_to_micros(0, 499, 1.0), Micros::ZERO);
    }

    #[test]
    fn a_reversed_pair_reads_as_zero() {
        assert_eq!(ticks_to_micros(2_000, 1_000, 1.0), Micros::ZERO);
    }

    #[test]
    fn huge_durations_saturate() {
        assert_eq!(
            ticks_to_micros(0, u64::MAX, f32::MAX),
            Micros::new(u64::MAX)
        );
    }

    #[test]
    fn timestamps_parse_little_endian_and_drop_a_partial_tail() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&7u64.to_le_bytes());
        bytes.extend_from_slice(&0x0102_0304_0506_0708u64.to_le_bytes());
        bytes.extend_from_slice(&[1, 2, 3]);
        assert_eq!(parse_ticks(&bytes), vec![7, 0x0102_0304_0506_0708]);
    }

    #[test]
    fn readback_size_is_eight_bytes_per_query() {
        assert_eq!(readback_bytes(0), 0);
        assert_eq!(readback_bytes(1), 16);
        assert_eq!(readback_bytes(5), 80);
    }

    #[test]
    fn decode_pairs_each_pass_with_its_tag_in_request_order() {
        let entries = vec![Entry::Compute("a"), Entry::Compute("b"), Entry::Render];
        let ticks = [0, 1_000, 1_000, 4_000, 4_000, 9_000];
        let timing = decode(FrameIndex::new(5), entries, &ticks, 1_000.0);
        assert_eq!(timing.frame, FrameIndex::new(5));
        assert_eq!(
            timing.compute,
            vec![("a", Micros::new(1_000)), ("b", Micros::new(3_000))]
        );
        assert_eq!(timing.render, Some(Micros::new(5_000)));
        let span = |begin_ns, end_ns| PassSpan { begin_ns, end_ns };
        assert_eq!(
            timing.compute_spans,
            vec![span(0, 1_000_000), span(1_000_000, 4_000_000)]
        );
        assert_eq!(timing.render_span, Some(span(4_000_000, 9_000_000)));
    }

    #[test]
    fn decode_leaves_out_a_pass_with_no_timestamps() {
        let entries = vec![Entry::Compute(1u32), Entry::Compute(2u32)];
        let timing = decode(FrameIndex::new(0), entries, &[0, 1_000], 1_000.0);
        assert_eq!(timing.compute, vec![(1, Micros::new(1_000))]);
        assert_eq!(timing.compute_spans.len(), 1);
        assert_eq!(timing.render, None);
        assert_eq!(timing.render_span, None);
    }

    #[test]
    fn timestamps_convert_to_nanoseconds_exactly_at_one_per_tick() {
        // Past 2^53, where a round trip through f64 would lose the low bits.
        let large = (1_u64 << 60) + 1;
        assert_eq!(ticks_to_nanos(large, 1.0), large);
        assert_eq!(ticks_to_nanos(24_000, 41.666_668), 1_000_000);
        assert_eq!(ticks_to_nanos(u64::MAX, f32::MAX), u64::MAX);
        assert_eq!(ticks_to_nanos(0, 41.666_668), 0);
    }

    #[test]
    fn decode_without_a_render_pass_reports_none() {
        let timing = decode(
            FrameIndex::new(0),
            vec![Entry::Compute(())],
            &[0, 2_000],
            1_000.0,
        );
        assert_eq!(timing.render, None);
    }

    /// Needs a real adapter: one frame with one compute pass goes through the
    /// whole path, from timestamp writes to a collected measurement.
    #[test]
    #[ignore = "needs a GPU adapter with TIMESTAMP_QUERY"]
    fn timer_measures_a_compute_pass_on_a_real_device() {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .unwrap();
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::TIMESTAMP_QUERY,
            ..Default::default()
        }))
        .unwrap();

        let mut timer: SliceTimer<u32> = SliceTimer::new(&device, &queue, 2).unwrap();
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let mut frame = timer.begin_frame(FrameIndex::new(0));
        assert!(frame.is_timed());
        let writes = frame.compute_pass(7).unwrap();
        drop(encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: writes,
        }));
        frame.resolve(&mut encoder);
        queue.submit([encoder.finish()]);
        timer.start_readback();
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let ready = timer.collect_ready();
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].frame, FrameIndex::new(0));
        assert_eq!(ready[0].compute.len(), 1);
        assert_eq!(ready[0].compute[0].0, 7);
        assert_eq!(ready[0].compute_spans.len(), 1);
        let span = ready[0].compute_spans[0];
        assert!(span.end_ns >= span.begin_ns, "{span:?}");
        assert_eq!(timer.in_flight(), 0);
    }
}
