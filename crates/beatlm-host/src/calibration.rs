//! Which of a calibration pass's measurements count.
//!
//! Both hosts calibrate the same way: two paced passes of
//! [`CALIBRATION_FRAMES`] timed frames each, the first deciding the split from
//! each whole kernel's median, the second measuring the split program's parts
//! into the loop's cost model (`CostModel::record_calibration`). The bench's
//! `calibrate.rs` says what each pass measures and why both are paced. Each
//! host drives the passes itself, because driving them is waiting on the
//! device between frames (the bench blocks, a page awaits the next animation
//! frame); the rule both passes count by is [`counted`], and what the first
//! pass decides from is [`KernelSamples`].

use anyhow::{Context, Result};
use beatlm_core::cost::SliceId;
use beatlm_core::time::Micros;

use crate::presets::CALIBRATION_FRAMES;
use crate::stats::median;

/// Whether calibration keeps the measurements of a pass's timed frame
/// `frame`, counting from 1 for the first timed frame: those of the second
/// half of the [`CALIBRATION_FRAMES`].
///
/// The first half is the GPU settling into the paced load. Read in the first
/// pass, it would split a kernel for what it took while the GPU settled rather
/// than for what it usually takes; read in the second, an estimator that
/// keeps every sample would start the loop from it.
#[must_use]
pub const fn counted(frame: u64) -> bool {
    frame > CALIBRATION_FRAMES / 2
}

/// What calibration's first pass keeps of each whole kernel: its measurements
/// in the frames that [`counted`], and from them the median the split is
/// decided on.
#[derive(Debug, Clone, Default)]
pub struct KernelSamples {
    samples: Vec<Vec<u64>>,
}

impl KernelSamples {
    /// A record for a program of `kernels` kernels.
    #[must_use]
    pub fn new(kernels: usize) -> Self {
        Self {
            samples: vec![Vec::new(); kernels],
        }
    }

    /// The number of kernels this record keeps measurements for.
    #[must_use]
    pub fn kernels(&self) -> usize {
        self.samples.len()
    }

    /// Keeps `duration`, measured for kernel `slice` in timed frame `frame`,
    /// if the frame [`counted`] and the kernel is one of this record's.
    pub fn record(&mut self, frame: u64, slice: SliceId, duration: Micros) {
        if !counted(frame) {
            return;
        }
        let list = usize::try_from(slice.get())
            .ok()
            .and_then(|i| self.samples.get_mut(i));
        if let Some(list) = list {
            list.push(duration.get());
        }
    }

    /// Each kernel's duration: the median of its counted measurements, the
    /// statistic the decision reads (`Program::parts_within`).
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

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn only_the_second_half_of_the_timed_frames_counts() {
        assert!(!counted(1));
        assert!(!counted(CALIBRATION_FRAMES / 2));
        assert!(counted(CALIBRATION_FRAMES / 2 + 1));
        assert!(counted(CALIBRATION_FRAMES));
        let kept = (1..=CALIBRATION_FRAMES).filter(|&f| counted(f)).count();
        assert_eq!(kept as u64, CALIBRATION_FRAMES / 2);
    }

    #[test]
    fn a_kernel_s_duration_is_the_median_of_its_counted_measurements() {
        let mut first = KernelSamples::new(2);
        let half = CALIBRATION_FRAMES / 2;
        first.record(half, SliceId::new(0), Micros::new(9_000));
        for (frame, us) in [(half + 1, 100), (half + 2, 300), (half + 3, 200)] {
            first.record(frame, SliceId::new(0), Micros::new(us));
            first.record(frame, SliceId::new(1), Micros::new(us + 1));
        }
        first.record(half + 1, SliceId::new(2), Micros::new(5));
        assert_eq!(
            first.durations().unwrap(),
            vec![Micros::new(200), Micros::new(201)]
        );
    }

    #[test]
    fn a_kernel_never_measured_in_a_counted_frame_is_refused() {
        let mut first = KernelSamples::new(2);
        first.record(CALIBRATION_FRAMES, SliceId::new(0), Micros::new(100));
        first.record(1, SliceId::new(1), Micros::new(100));
        assert!(first.durations().is_err());
    }
}
