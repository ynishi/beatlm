//! Command-line options, parsed by hand: they are a handful of numbers, and a
//! parser dependency would outweigh the code it replaces.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use beatlm_core::kernel::Fusion;

/// The usage text, printed for `--help` and after a bad option.
pub const USAGE: &str = "\
usage: beatlm-bench [options]

  --frames N        frames to run                          (default 600)
  --latency K       promised delivery latency, in frames   (default 2)
  --slices S        slices per job, one job per frame      (default 4)
  --slice-iters N   loop iterations per invocation of a slice kernel
  --render-iters N  loop iterations per pixel of the render pass
  --margin-us N     safety margin in the frame budget, us  (default 2000)
  --workload W      synthetic or net                        (default synthetic)
  --fusion F        net only: fuse element-wise operations into the kernel
                    before them, on or off                  (default on)
  --dump-wgsl DIR   net only: write each generated kernel's WGSL into DIR
                    before the run
  --mode M          concurrency candidate: sync, thread or async (default sync)
  --runs N          repeat the whole run N times on one device (default 1)
  --estimator E     cost estimate: mean-sigmas, quantile or capped
                                                           (default quantile)
  --background-load run a competing GPU load from a second device on the
                    same adapter, from before the first frame to after the last
  --background-iters N  loop iterations per invocation of the competing load's
                    kernel                                  (default 4000)
  --help            print this text";

/// Loop iterations per slice-kernel invocation that put one slice at 0.8 to
/// 1.4 ms on an Apple M1 Pro (Metal); see the crate documentation.
pub const DEFAULT_SLICE_ITERS: u32 = 500;

/// Loop iterations per pixel that put the render pass at about 2 ms on an
/// Apple M1 Pro (Metal); see the crate documentation.
pub const DEFAULT_RENDER_ITERS: u32 = 50;

/// Loop iterations per invocation of the competing load's kernel; see the
/// crate documentation for what it did to the slices' measured times.
pub const DEFAULT_BACKGROUND_ITERS: u32 = 4_000;

/// Which statistic the cost model reads a slice's measurements with; the
/// parameters of each are constants in `engine.rs`. The default is the one the
/// measurements in `beatlm_core::cost` chose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimatorKind {
    /// `beatlm_core::cost::Estimator::MeanPlusSigmas`.
    MeanSigmas,
    /// `beatlm_core::cost::Estimator::WindowedQuantile`.
    Quantile,
    /// `beatlm_core::cost::Estimator::CappedDeviation`.
    Capped,
}

impl EstimatorKind {
    /// The spelling `--estimator` accepts and the report prints.
    pub const fn name(self) -> &'static str {
        match self {
            Self::MeanSigmas => "mean-sigmas",
            Self::Quantile => "quantile",
            Self::Capped => "capped",
        }
    }
}

/// Which concurrency candidate drives the frame; the crate documentation says
/// what each one does and what stays the same between them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// A synchronous `tick` on the frame-loop thread.
    Sync,
    /// A dedicated GPU thread fed frame events over a channel.
    Thread,
    /// An `async` `tick` awaited by the frame loop.
    Async,
}

impl Mode {
    /// The spelling `--mode` accepts and the report prints, so a report can be
    /// matched to the command that produced it.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Sync => "sync",
            Self::Thread => "thread",
            Self::Async => "async",
        }
    }
}

/// Which load the slices of a job run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkloadKind {
    /// The busy-loop kernel standing in for a slice (`workload.rs`).
    Synthetic,
    /// The planned network of generated kernels (`net.rs`): one slice per
    /// kernel, so the job has [`crate::net::kernel_count`] slices for the
    /// `--fusion` setting and `--slices` and `--slice-iters` do not apply.
    Net,
}

/// The spelling `--fusion` accepts and the report prints.
pub const fn fusion_name(fusion: Fusion) -> &'static str {
    match fusion {
        Fusion::On => "on",
        Fusion::Off => "off",
    }
}

impl WorkloadKind {
    /// The spelling `--workload` accepts and the report prints.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Synthetic => "synthetic",
            Self::Net => "net",
        }
    }
}

/// What a run is configured with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Args {
    /// Frames to run.
    pub frames: u64,
    /// The promised latency `k` of the L1 delivery policy, in frames.
    pub latency: u32,
    /// Slices in the one job submitted each frame.
    pub slices: u32,
    /// Loop iterations per invocation of the slice kernel.
    pub slice_iters: u32,
    /// Loop iterations per pixel of the render pass.
    pub render_iters: u32,
    /// The margin subtracted from each frame's budget, in microseconds.
    pub margin_us: u64,
    /// The load the slices run.
    pub workload: WorkloadKind,
    /// Whether the network's element-wise operations are fused; `net` only.
    pub fusion: Fusion,
    /// Where to write the network's generated WGSL before the run; `net` only.
    pub dump_wgsl: Option<PathBuf>,
    /// The concurrency candidate that drives the frame.
    pub mode: Mode,
    /// How many times the whole run is repeated in one process.
    pub runs: u32,
    /// The statistic the cost model estimates a slice with.
    pub estimator: EstimatorKind,
    /// Whether a second device on the same adapter competes for the GPU for
    /// the length of each run.
    pub background_load: bool,
    /// Loop iterations per invocation of the competing load's kernel.
    pub background_iters: u32,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            frames: 600,
            latency: 2,
            slices: 4,
            slice_iters: DEFAULT_SLICE_ITERS,
            render_iters: DEFAULT_RENDER_ITERS,
            margin_us: 2_000,
            workload: WorkloadKind::Synthetic,
            fusion: Fusion::On,
            dump_wgsl: None,
            mode: Mode::Sync,
            runs: 1,
            estimator: EstimatorKind::Quantile,
            background_load: false,
            background_iters: DEFAULT_BACKGROUND_ITERS,
        }
    }
}

/// What the command line asks for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Run the benchmark.
    Run(Args),
    /// Print the usage text.
    Help,
}

/// Parses the arguments after the program name.
///
/// Both `--name value` and `--name=value` are accepted.
///
/// # Errors
///
/// An unknown option, a missing or malformed value, a value out of range
/// (zero frames, zero slices or zero runs), or a `net` option without
/// `--workload net`.
pub fn parse<I>(args: I) -> Result<Command>
where
    I: IntoIterator<Item = String>,
{
    let mut out = Args::default();
    let mut slices_given = false;
    let mut background_iters_given = false;
    let mut fusion_given = false;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let (name, inline) = match arg.split_once('=') {
            Some((name, value)) => (name.to_owned(), Some(value.to_owned())),
            None => (arg, None),
        };
        if name == "--help" || name == "-h" {
            return Ok(Command::Help);
        }
        let mut value = || -> Result<String> {
            match inline.clone() {
                Some(v) => Ok(v),
                None => args.next().with_context(|| format!("{name} needs a value")),
            }
        };
        match name.as_str() {
            "--frames" => out.frames = number(&name, &value()?)?,
            "--latency" => out.latency = number(&name, &value()?)?,
            "--slices" => {
                out.slices = number(&name, &value()?)?;
                slices_given = true;
            }
            "--slice-iters" => out.slice_iters = number(&name, &value()?)?,
            "--render-iters" => out.render_iters = number(&name, &value()?)?,
            "--margin-us" => out.margin_us = number(&name, &value()?)?,
            "--workload" => out.workload = workload(&value()?)?,
            "--fusion" => {
                out.fusion = fusion(&value()?)?;
                fusion_given = true;
            }
            "--dump-wgsl" => out.dump_wgsl = Some(PathBuf::from(value()?)),
            "--mode" => out.mode = mode(&value()?)?,
            "--runs" => out.runs = number(&name, &value()?)?,
            "--estimator" => out.estimator = estimator(&value()?)?,
            "--background-load" => out.background_load = true,
            "--background-iters" => {
                out.background_iters = number(&name, &value()?)?;
                background_iters_given = true;
            }
            _ => bail!("unknown option {name}"),
        }
    }
    if out.frames == 0 {
        bail!("--frames must be at least 1");
    }
    if out.slices == 0 {
        bail!("--slices must be at least 1");
    }
    if out.runs == 0 {
        bail!("--runs must be at least 1");
    }
    if out.background_iters == 0 {
        bail!("--background-iters must be at least 1");
    }
    if background_iters_given && !out.background_load {
        bail!("--background-iters has no effect without --background-load");
    }
    if out.workload == WorkloadKind::Net {
        // The network's kernels are its slices.
        let kernels = crate::net::kernel_count(out.fusion)?;
        if slices_given && out.slices != kernels {
            bail!(
                "--slices is {kernels} for --workload net with --fusion {}: one slice per kernel",
                fusion_name(out.fusion)
            );
        }
        out.slices = kernels;
    } else {
        if fusion_given {
            bail!("--fusion has no effect without --workload net");
        }
        if out.dump_wgsl.is_some() {
            bail!("--dump-wgsl has no effect without --workload net");
        }
    }
    Ok(Command::Run(out))
}

/// Parses the value of `--fusion`.
fn fusion(text: &str) -> Result<Fusion> {
    [Fusion::On, Fusion::Off]
        .into_iter()
        .find(|&f| fusion_name(f) == text)
        .with_context(|| format!("--fusion: {text:?} is not one of on, off"))
}

/// Parses the value of `--workload`.
fn workload(text: &str) -> Result<WorkloadKind> {
    [WorkloadKind::Synthetic, WorkloadKind::Net]
        .into_iter()
        .find(|w| w.name() == text)
        .with_context(|| format!("--workload: {text:?} is not one of synthetic, net"))
}

/// Parses the value of `--estimator`.
fn estimator(text: &str) -> Result<EstimatorKind> {
    [
        EstimatorKind::MeanSigmas,
        EstimatorKind::Quantile,
        EstimatorKind::Capped,
    ]
    .into_iter()
    .find(|e| e.name() == text)
    .with_context(|| format!("--estimator: {text:?} is not one of mean-sigmas, quantile, capped"))
}

/// Parses the value of `--mode`.
fn mode(text: &str) -> Result<Mode> {
    [Mode::Sync, Mode::Thread, Mode::Async]
        .into_iter()
        .find(|m| m.name() == text)
        .with_context(|| format!("--mode: {text:?} is not one of sync, thread, async"))
}

fn number<T>(name: &str, text: &str) -> Result<T>
where
    T: std::str::FromStr,
    T::Err: std::error::Error + Send + Sync + 'static,
{
    text.parse()
        .with_context(|| format!("{name}: {text:?} is not a valid number"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    fn parse_strs(args: &[&str]) -> Result<Command> {
        parse(args.iter().map(|s| (*s).to_owned()))
    }

    #[test]
    fn no_arguments_gives_the_defaults() {
        assert_eq!(parse_strs(&[]).unwrap(), Command::Run(Args::default()));
    }

    #[test]
    fn both_spellings_of_a_value_are_accepted() {
        let Command::Run(a) =
            parse_strs(&["--frames", "300", "--latency=3", "--margin-us", "500"]).unwrap()
        else {
            panic!("expected a run");
        };
        assert_eq!((a.frames, a.latency, a.margin_us), (300, 3, 500));
    }

    #[test]
    fn help_wins() {
        assert_eq!(
            parse_strs(&["--frames", "5", "--help"]).unwrap(),
            Command::Help
        );
    }

    #[test]
    fn bad_input_is_rejected() {
        assert!(parse_strs(&["--nope"]).is_err());
        assert!(parse_strs(&["--frames"]).is_err());
        assert!(parse_strs(&["--frames", "x"]).is_err());
        assert!(parse_strs(&["--frames", "0"]).is_err());
        assert!(parse_strs(&["--slices", "0"]).is_err());
        assert!(parse_strs(&["--runs", "0"]).is_err());
        assert!(parse_strs(&["--mode", "fibers"]).is_err());
    }

    #[test]
    fn mode_and_runs_are_parsed() {
        let Command::Run(a) = parse_strs(&["--mode", "thread", "--runs=3"]).unwrap() else {
            panic!("expected a run");
        };
        assert_eq!((a.mode, a.runs), (Mode::Thread, 3));
        let Command::Run(a) = parse_strs(&["--mode=async"]).unwrap() else {
            panic!("expected a run");
        };
        assert_eq!(a.mode, Mode::Async);
    }

    #[test]
    fn the_default_workload_is_synthetic() {
        assert_eq!(Args::default().workload, WorkloadKind::Synthetic);
        let Command::Run(a) = parse_strs(&["--workload", "synthetic"]).unwrap() else {
            panic!("expected a run");
        };
        assert_eq!((a.workload, a.slices), (WorkloadKind::Synthetic, 4));
    }

    #[test]
    fn the_net_workload_has_one_slice_per_kernel() {
        let Command::Run(a) = parse_strs(&["--workload=net"]).unwrap() else {
            panic!("expected a run");
        };
        let fused = crate::net::kernel_count(Fusion::On).unwrap();
        assert_eq!(
            (a.workload, a.fusion, a.slices),
            (WorkloadKind::Net, Fusion::On, fused)
        );
        let Command::Run(a) = parse_strs(&["--workload=net", "--fusion", "off"]).unwrap() else {
            panic!("expected a run");
        };
        let unfused = crate::net::kernel_count(Fusion::Off).unwrap();
        assert_eq!((a.fusion, a.slices), (Fusion::Off, unfused));
        assert!(fused < unfused);
        let kernels = fused.to_string();
        assert!(parse_strs(&["--workload", "net", "--slices", &kernels]).is_ok());
        assert!(parse_strs(&["--workload", "net", "--slices", "3"]).is_err());
        assert!(parse_strs(&["--workload", "gpt"]).is_err());
        assert!(parse_strs(&["--workload", "net", "--fusion", "maybe"]).is_err());
    }

    #[test]
    fn the_net_options_need_the_net_workload() {
        assert!(parse_strs(&["--fusion", "off"]).is_err());
        assert!(parse_strs(&["--dump-wgsl", "kernels"]).is_err());
        let Command::Run(a) = parse_strs(&["--workload", "net", "--dump-wgsl=kernels"]).unwrap()
        else {
            panic!("expected a run");
        };
        assert_eq!(a.dump_wgsl, Some(PathBuf::from("kernels")));
        assert_eq!(Args::default().dump_wgsl, None);
    }

    #[test]
    fn the_estimator_and_the_background_load_are_parsed() {
        let a = Args::default();
        assert_eq!(a.estimator, EstimatorKind::Quantile);
        assert!(!a.background_load);
        for kind in [
            EstimatorKind::MeanSigmas,
            EstimatorKind::Quantile,
            EstimatorKind::Capped,
        ] {
            let Command::Run(a) = parse_strs(&["--estimator", kind.name()]).unwrap() else {
                panic!("expected a run");
            };
            assert_eq!(a.estimator, kind);
        }
        let Command::Run(a) = parse_strs(&["--background-load", "--background-iters=900"]).unwrap()
        else {
            panic!("expected a run");
        };
        assert_eq!((a.background_load, a.background_iters), (true, 900));
        assert!(parse_strs(&["--estimator", "median"]).is_err());
        assert!(parse_strs(&["--background-load", "--background-iters", "0"]).is_err());
        assert!(parse_strs(&["--background-iters", "900"]).is_err());
    }

    #[test]
    fn the_default_is_the_synchronous_candidate_run_once() {
        let a = Args::default();
        assert_eq!((a.mode, a.runs), (Mode::Sync, 1));
    }
}
