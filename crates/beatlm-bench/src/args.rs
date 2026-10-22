//! Command-line options, parsed by hand: they are a handful of numbers, and a
//! parser dependency would outweigh the code it replaces.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use beatlm_core::kernel::Fusion;

use crate::pass_cost::{self, DEFAULT_DEPTH_SIDE, Dispatches};
use crate::workload::{DEFAULT_RENDER_ITERS, DEFAULT_SLICE_ITERS, RenderOrder};

/// The usage text, printed for `--help` and after a bad option.
pub const USAGE: &str = "\
usage: beatlm-bench [options]

  --frames N        frames to run                          (default 600)
  --latency K       promised delivery latency, in frames   (default 2;
                    depth: 5)
  --slices S        slices per job                         (default 4)
  --slice-iters N   loop iterations per invocation of a slice kernel
  --render-iters N  loop iterations per pixel of the render pass
  --render-order O  free, or after: the render pass reads what the frame's
                    slices write, so it runs after them    (default free)
  --margin-us N     safety margin in the frame budget, us  (default 2000)
  --workload W      synthetic, net, depth or pass-cost     (default synthetic)
  --submit S        when a frame submits its input as a job: every frame,
                    or free, only once the last job's slices are all handed
                    out                                    (default every;
                    depth: free)
  --size N          depth, and pass-cost with --kernels depth: input side in
                    pixels, a multiple of 14   (default 266; pass-cost: 112)
  --fusion F        net, depth, and pass-cost with --kernels depth: fuse
                    element-wise operations into the kernel before them, on
                    or off                                 (default on)
  --dump-wgsl DIR   net only: write each generated kernel's WGSL into DIR
                    before the run
  --max-dispatch-us N  net and depth: calibrate each kernel before the run,
                    split those whose median is longer than N us into parts
                    and measure the parts; 0 turns splitting off
                                                           (default 1000)
  --split S         net and depth: on or off; off is --max-dispatch-us 0
                                                           (default on)
  --mode M          concurrency candidate: sync, thread or async (default sync)
  --runs N          repeat the whole run N times on one device (default 1)
  --estimator E     cost estimate: mean-sigmas, quantile, capped or
                    frame-scaled (a slice's median times a slowdown
                    measured over whole frames)            (default quantile)
  --background-load run a competing GPU load from a second device on the
                    same adapter, from before the first frame to after the last
  --background-iters N  loop iterations per invocation of the competing load's
                    kernel                                  (default 4000)

  pass-cost only (--frames is then frames per cell; --runs and
  --render-iters apply, the loop's other options do not):
  --passes LIST     dispatches per frame, comma-separated  (default 1,8,32,128)
  --variants LIST   per-pass, ends, one-pass, every:<k>    (default
                    per-pass,ends,one-pass,every:8)
  --kernels K       what each dispatch runs: trivial, or depth, the first
                    kernels of the depth model             (default trivial)
  --groups N        trivial only: workgroups per dispatch  (default 1)
  --render R        end each frame with the render pass, on or off
                                                           (default off)
  --help            print this text";

/// Loop iterations per invocation of the competing load's kernel; see the
/// crate documentation for what it did to the slices' measured times.
pub const DEFAULT_BACKGROUND_ITERS: u32 = 4_000;

/// The longest a dispatch of the network may take before calibration splits
/// its kernel, in microseconds; the crate documentation, under "Splitting",
/// has the sweep that chose it.
pub const DEFAULT_MAX_DISPATCH_US: u64 = 1_000;

/// The depth model's input side, in pixels, when `--size` is not given: the
/// resolution the sweep in the crate documentation ("The `depth` workload")
/// recommends for an Apple M1 Pro.
pub const DEFAULT_DEPTH_SIZE: u32 = 266;

/// The promised latency of the depth workload, in frames, when `--latency` is
/// not given: the one the same sweep recommends at [`DEFAULT_DEPTH_SIZE`], a
/// frame of slack above the smallest that met its criterion there.
pub const DEFAULT_DEPTH_LATENCY: u32 = 5;

/// When a frame submits its input as a job; the crate documentation, under
/// "The `depth` workload", says why the model needs `Free`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Submit {
    /// Every frame submits one job.
    Every,
    /// A frame submits a job only when the job submitted last has had all its
    /// slices handed out, or has left the scheduler; otherwise its input is
    /// not run, as a host feeding a model slower than its frame rate skips
    /// frames of the source.
    Free,
}

impl Submit {
    /// The spelling `--submit` accepts and the report prints.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Every => "every",
            Self::Free => "free",
        }
    }
}

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
    /// `beatlm_core::cost::Estimator::FrameScaled`.
    FrameScaled,
}

impl EstimatorKind {
    /// Every kind, in the order `--help` and the error for another value
    /// list them.
    pub const ALL: [Self; 4] = [
        Self::MeanSigmas,
        Self::Quantile,
        Self::Capped,
        Self::FrameScaled,
    ];

    /// The spelling `--estimator` accepts and the report prints.
    pub const fn name(self) -> &'static str {
        match self {
            Self::MeanSigmas => "mean-sigmas",
            Self::Quantile => "quantile",
            Self::Capped => "capped",
            Self::FrameScaled => "frame-scaled",
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
    /// Depth Anything V2 Small from its weights file (`depth.rs`), one slice
    /// per kernel like the network: [`crate::depth::kernel_count`] slices for
    /// `--size` and `--fusion`.
    Depth,
    /// The measurement of what a compute pass costs (`pass_cost.rs`): no job,
    /// no scheduler, no promise, so the loop's options do not apply.
    PassCost,
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
            Self::Depth => "depth",
            Self::PassCost => "pass-cost",
        }
    }

    /// Whether the slices are the kernels of a planned program, which is
    /// calibrated, split and given input and output slots.
    pub const fn is_planned(self) -> bool {
        matches!(self, Self::Net | Self::Depth)
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
    /// Whether the loop's render pass waits for the frame's slices
    /// (`workload.rs` says how).
    pub render_order: RenderOrder,
    /// The margin subtracted from each frame's budget, in microseconds.
    pub margin_us: u64,
    /// The load the slices run.
    pub workload: WorkloadKind,
    /// When a frame submits its input as a job.
    pub submit: Submit,
    /// The depth model's input side in pixels; `depth` only.
    pub size: u32,
    /// Whether the program's element-wise operations are fused; `net` and
    /// `depth` only.
    pub fusion: Fusion,
    /// Where to write the network's generated WGSL before the run; `net` only.
    pub dump_wgsl: Option<PathBuf>,
    /// The bound a calibrated kernel's dispatch is split to fit, in
    /// microseconds, or `None` when the program runs unsplit and is not
    /// calibrated; `net` and `depth` only.
    pub max_dispatch_us: Option<u64>,
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
    /// What `--workload pass-cost` measures; unused by the other workloads.
    pub pass_cost: pass_cost::Settings,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            frames: 600,
            latency: 2,
            slices: 4,
            slice_iters: DEFAULT_SLICE_ITERS,
            render_iters: DEFAULT_RENDER_ITERS,
            render_order: RenderOrder::Free,
            margin_us: 2_000,
            workload: WorkloadKind::Synthetic,
            submit: Submit::Every,
            size: DEFAULT_DEPTH_SIZE,
            fusion: Fusion::On,
            dump_wgsl: None,
            max_dispatch_us: Some(DEFAULT_MAX_DISPATCH_US),
            mode: Mode::Sync,
            runs: 1,
            estimator: EstimatorKind::Quantile,
            background_load: false,
            background_iters: DEFAULT_BACKGROUND_ITERS,
            pass_cost: pass_cost::Settings::default(),
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
/// (zero frames, zero slices or zero runs, a depth size that is not whole
/// patches), an option of the `net`, `depth` or `pass-cost` workload without
/// it, or an option of the loop with `pass-cost`.
pub fn parse<I>(args: I) -> Result<Command>
where
    I: IntoIterator<Item = String>,
{
    let mut out = Args::default();
    let mut slices_given = false;
    let mut background_iters_given = false;
    let mut fusion_given = false;
    let mut latency_given = false;
    let mut size_given = false;
    let mut submit_given = false;
    let mut bound_given: Option<u64> = None;
    let mut split_given: Option<bool> = None;
    let mut given: Vec<String> = Vec::new();
    let mut kernels_depth = false;
    let mut groups = 1;
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let (name, inline) = match arg.split_once('=') {
            Some((name, value)) => (name.to_owned(), Some(value.to_owned())),
            None => (arg, None),
        };
        if name == "--help" || name == "-h" {
            return Ok(Command::Help);
        }
        given.push(name.clone());
        let mut value = || -> Result<String> {
            match inline.clone() {
                Some(v) => Ok(v),
                None => args.next().with_context(|| format!("{name} needs a value")),
            }
        };
        match name.as_str() {
            "--frames" => out.frames = number(&name, &value()?)?,
            "--latency" => {
                out.latency = number(&name, &value()?)?;
                latency_given = true;
            }
            "--size" => {
                out.size = number(&name, &value()?)?;
                size_given = true;
            }
            "--submit" => {
                out.submit = submit(&value()?)?;
                submit_given = true;
            }
            "--slices" => {
                out.slices = number(&name, &value()?)?;
                slices_given = true;
            }
            "--slice-iters" => out.slice_iters = number(&name, &value()?)?,
            "--render-iters" => out.render_iters = number(&name, &value()?)?,
            "--render-order" => {
                let text = value()?;
                out.render_order = RenderOrder::from_name(&text).with_context(|| {
                    format!("--render-order: {text:?} is not one of free, after")
                })?;
            }
            "--margin-us" => out.margin_us = number(&name, &value()?)?,
            "--workload" => out.workload = workload(&value()?)?,
            "--fusion" => {
                out.fusion = fusion(&value()?)?;
                fusion_given = true;
            }
            "--dump-wgsl" => out.dump_wgsl = Some(PathBuf::from(value()?)),
            "--max-dispatch-us" => bound_given = Some(number(&name, &value()?)?),
            "--split" => split_given = Some(on_off(&name, &value()?)?),
            "--mode" => out.mode = mode(&value()?)?,
            "--runs" => out.runs = number(&name, &value()?)?,
            "--estimator" => out.estimator = estimator(&value()?)?,
            "--background-load" => out.background_load = true,
            "--background-iters" => {
                out.background_iters = number(&name, &value()?)?;
                background_iters_given = true;
            }
            "--passes" => {
                out.pass_cost.passes =
                    pass_cost::parse_passes(&value()?).with_context(|| name.clone())?;
            }
            "--variants" => {
                out.pass_cost.variants =
                    pass_cost::parse_variants(&value()?).with_context(|| name.clone())?;
            }
            "--kernels" => {
                kernels_depth = match value()?.as_str() {
                    "trivial" => false,
                    "depth" => true,
                    other => bail!("--kernels: {other:?} is not one of trivial, depth"),
                };
            }
            "--groups" => {
                groups = pass_cost::parse_groups(&value()?).with_context(|| name.clone())?;
            }
            "--render" => out.pass_cost.render = on_off(&name, &value()?)?,
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
    let seen =
        |names: &[&'static str]| names.iter().copied().find(|n| given.iter().any(|g| g == n));
    if out.workload == WorkloadKind::PassCost {
        // Nothing is scheduled, so the loop's options would be silently
        // ignored; they are refused instead.
        if let Some(name) = seen(&LOOP_ONLY) {
            bail!("{name} has no effect with --workload pass-cost, which schedules nothing");
        }
        if !kernels_depth && (size_given || fusion_given) {
            bail!("--size and --fusion have no effect without --kernels depth");
        }
        if kernels_depth && seen(&["--groups"]).is_some() {
            bail!("--groups has no effect with --kernels depth");
        }
        if kernels_depth {
            if !size_given {
                out.size = DEFAULT_DEPTH_SIDE;
            }
            out.pass_cost.dispatches = Dispatches::Depth { side: out.size };
            // A pass count above the model's kernels is refused here, before
            // the device opens and the weights load.
            let kernels = crate::depth::kernel_count(out.size, out.fusion)?;
            out.pass_cost
                .check_kernels(usize::try_from(kernels).unwrap_or(usize::MAX))?;
        } else {
            out.pass_cost.dispatches = Dispatches::Trivial { groups };
        }
        return Ok(Command::Run(out));
    }
    if let Some(name) = seen(&PASS_COST_ONLY) {
        bail!("{name} has no effect without --workload pass-cost");
    }
    if size_given && out.workload != WorkloadKind::Depth {
        bail!("--size has no effect without --workload depth");
    }
    if out.workload == WorkloadKind::Depth {
        // A model more than a frame long cannot take an input every frame;
        // the crate documentation says what happens when it is made to.
        if !submit_given {
            out.submit = Submit::Free;
        }
        if !latency_given {
            out.latency = DEFAULT_DEPTH_LATENCY;
        }
    }
    if out.workload.is_planned() {
        // The program's kernels are its slices.
        let kernels = match out.workload {
            WorkloadKind::Depth => crate::depth::kernel_count(out.size, out.fusion)?,
            _ => crate::net::kernel_count(out.fusion)?,
        };
        if slices_given && out.slices != kernels {
            bail!(
                "--slices is {kernels} for --workload {} with --fusion {}: one slice per kernel",
                out.workload.name(),
                fusion_name(out.fusion)
            );
        }
        out.slices = kernels;
        out.max_dispatch_us = match (split_given, bound_given) {
            (Some(false), Some(us)) if us > 0 => {
                bail!("--max-dispatch-us {us} has no effect with --split off")
            }
            (Some(true), Some(0)) => bail!("--split on needs a --max-dispatch-us above 0"),
            (Some(false), _) | (_, Some(0)) => None,
            (_, Some(us)) => Some(us),
            (_, None) => Some(DEFAULT_MAX_DISPATCH_US),
        };
    } else {
        if fusion_given {
            bail!("--fusion has no effect without --workload net or depth");
        }
        if bound_given.is_some() || split_given.is_some() {
            bail!("--max-dispatch-us and --split have no effect without --workload net or depth");
        }
    }
    if out.dump_wgsl.is_some() && out.workload != WorkloadKind::Net {
        bail!("--dump-wgsl has no effect without --workload net");
    }
    Ok(Command::Run(out))
}

/// The options of the loop, which `--workload pass-cost` refuses.
const LOOP_ONLY: [&str; 13] = [
    "--latency",
    "--render-order",
    "--slices",
    "--slice-iters",
    "--margin-us",
    "--submit",
    "--mode",
    "--estimator",
    "--background-load",
    "--background-iters",
    "--max-dispatch-us",
    "--split",
    "--dump-wgsl",
];

/// The options only `--workload pass-cost` reads.
const PASS_COST_ONLY: [&str; 5] = [
    "--passes",
    "--variants",
    "--kernels",
    "--groups",
    "--render",
];

/// Parses the value of `--submit`.
fn submit(text: &str) -> Result<Submit> {
    [Submit::Every, Submit::Free]
        .into_iter()
        .find(|s| s.name() == text)
        .with_context(|| format!("--submit: {text:?} is not one of every, free"))
}

/// Parses an `on` or `off` value.
fn on_off(name: &str, text: &str) -> Result<bool> {
    match text {
        "on" => Ok(true),
        "off" => Ok(false),
        _ => bail!("{name}: {text:?} is not one of on, off"),
    }
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
    [
        WorkloadKind::Synthetic,
        WorkloadKind::Net,
        WorkloadKind::Depth,
        WorkloadKind::PassCost,
    ]
    .into_iter()
    .find(|w| w.name() == text)
    .with_context(|| format!("--workload: {text:?} is not one of synthetic, net, depth, pass-cost"))
}

/// Parses the value of `--estimator`.
fn estimator(text: &str) -> Result<EstimatorKind> {
    EstimatorKind::ALL
        .into_iter()
        .find(|e| e.name() == text)
        .with_context(|| {
            let names: Vec<&str> = EstimatorKind::ALL.iter().map(|e| e.name()).collect();
            format!("--estimator: {text:?} is not one of {}", names.join(", "))
        })
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
        for kind in EstimatorKind::ALL {
            let Command::Run(a) = parse_strs(&["--estimator", kind.name()]).unwrap() else {
                panic!("expected a run");
            };
            assert_eq!(a.estimator, kind);
            let spelled = format!("--estimator={}", kind.name());
            let Command::Run(a) = parse_strs(&[&spelled]).unwrap() else {
                panic!("expected a run");
            };
            assert_eq!(a.estimator, kind);
        }
        assert!(parse_strs(&["--estimator", "frame"]).is_err());
        assert!(parse_strs(&["--estimator", "frame-scaled:0.9"]).is_err());
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
    fn splitting_is_on_by_default_and_turned_off_by_a_zero_bound_or_by_name() {
        let net = |extra: &[&str]| {
            let mut all = vec!["--workload", "net"];
            all.extend_from_slice(extra);
            match parse_strs(&all) {
                Ok(Command::Run(a)) => Ok(a.max_dispatch_us),
                Ok(Command::Help) => panic!("expected a run"),
                Err(e) => Err(e),
            }
        };
        assert_eq!(net(&[]).unwrap(), Some(DEFAULT_MAX_DISPATCH_US));
        assert_eq!(net(&["--max-dispatch-us", "500"]).unwrap(), Some(500));
        assert_eq!(net(&["--max-dispatch-us=0"]).unwrap(), None);
        assert_eq!(net(&["--split", "off"]).unwrap(), None);
        assert_eq!(
            net(&["--split", "off", "--max-dispatch-us", "0"]).unwrap(),
            None
        );
        assert_eq!(
            net(&["--split", "on", "--max-dispatch-us", "700"]).unwrap(),
            Some(700)
        );
        assert_eq!(net(&["--split=on"]).unwrap(), Some(DEFAULT_MAX_DISPATCH_US));
        assert!(net(&["--split", "off", "--max-dispatch-us", "700"]).is_err());
        assert!(net(&["--split", "on", "--max-dispatch-us", "0"]).is_err());
        assert!(net(&["--split", "sometimes"]).is_err());
        assert!(net(&["--max-dispatch-us", "-1"]).is_err());
        // Splitting does not change the slice count `args` knows: the parts
        // are decided by calibration, on the device.
        let Command::Run(a) = parse_strs(&["--workload", "net"]).unwrap() else {
            panic!("expected a run");
        };
        assert_eq!(a.slices, crate::net::kernel_count(Fusion::On).unwrap());
        assert!(parse_strs(&["--split", "off"]).is_err());
        assert!(parse_strs(&["--max-dispatch-us", "500"]).is_err());
    }

    #[test]
    fn the_depth_workload_has_its_own_size_latency_and_submission_defaults() {
        let Command::Run(a) = parse_strs(&["--workload", "depth"]).unwrap() else {
            panic!("expected a run");
        };
        assert_eq!(
            (a.size, a.latency, a.submit),
            (DEFAULT_DEPTH_SIZE, DEFAULT_DEPTH_LATENCY, Submit::Free)
        );
        assert_eq!(
            a.slices,
            crate::depth::kernel_count(DEFAULT_DEPTH_SIZE, Fusion::On).unwrap()
        );
        assert_eq!(a.max_dispatch_us, Some(DEFAULT_MAX_DISPATCH_US));
        let Command::Run(a) = parse_strs(&[
            "--workload=depth",
            "--size",
            "182",
            "--latency",
            "3",
            "--submit",
            "every",
            "--split",
            "off",
        ])
        .unwrap() else {
            panic!("expected a run");
        };
        assert_eq!(
            (a.size, a.latency, a.submit, a.max_dispatch_us),
            (182, 3, Submit::Every, None)
        );
        assert!(parse_strs(&["--workload", "depth", "--size", "100"]).is_err());
        assert!(parse_strs(&["--workload", "depth", "--slices", "3"]).is_err());
        assert!(parse_strs(&["--size", "266"]).is_err());
        assert!(parse_strs(&["--workload", "depth", "--dump-wgsl", "kernels"]).is_err());
        assert!(parse_strs(&["--submit", "sometimes"]).is_err());
        // The other workloads submit every frame unless told otherwise.
        assert_eq!(Args::default().submit, Submit::Every);
        let Command::Run(a) = parse_strs(&["--submit", "free"]).unwrap() else {
            panic!("expected a run");
        };
        assert_eq!((a.submit, a.latency), (Submit::Free, 2));
    }

    #[test]
    fn the_pass_cost_workload_has_its_own_options_and_refuses_the_loops() {
        let pass_cost = |extra: &[&str]| {
            let mut all = vec!["--workload", "pass-cost"];
            all.extend_from_slice(extra);
            match parse_strs(&all) {
                Ok(Command::Run(a)) => Ok(a),
                Ok(Command::Help) => panic!("expected a run"),
                Err(e) => Err(e),
            }
        };
        let a = pass_cost(&[]).unwrap();
        assert_eq!(a.workload, WorkloadKind::PassCost);
        assert_eq!(a.pass_cost, pass_cost::Settings::default());
        let a = pass_cost(&[
            "--passes=1,64",
            "--variants",
            "per-pass,every:4",
            "--groups",
            "16",
            "--render",
            "on",
            "--frames",
            "300",
            "--runs",
            "2",
            "--render-iters",
            "10",
        ])
        .unwrap();
        assert_eq!(a.pass_cost.passes, vec![1, 64]);
        assert_eq!(
            a.pass_cost.variants,
            vec![pass_cost::Variant::PerPass, pass_cost::Variant::Every(4)]
        );
        assert_eq!(a.pass_cost.dispatches, Dispatches::Trivial { groups: 16 });
        assert!(a.pass_cost.render);
        assert_eq!((a.frames, a.runs, a.render_iters), (300, 2, 10));
        let a = pass_cost(&["--kernels", "depth"]).unwrap();
        assert_eq!(
            a.pass_cost.dispatches,
            Dispatches::Depth {
                side: DEFAULT_DEPTH_SIDE
            }
        );
        let a = pass_cost(&["--kernels=depth", "--size", "154", "--fusion", "off"]).unwrap();
        assert_eq!(
            (a.pass_cost.dispatches, a.fusion),
            (Dispatches::Depth { side: 154 }, Fusion::Off)
        );
        for bad in [
            &["--passes", "0"][..],
            &["--passes", "1,1"],
            &["--variants", "every:0"],
            &["--variants", "fast"],
            &["--kernels", "net"],
            &["--groups", "0"],
            &["--render", "maybe"],
            &["--size", "112"],
            &["--fusion", "off"],
            &["--kernels", "depth", "--groups", "4"],
            &["--kernels", "depth", "--size", "100"],
            // More passes than the model at 112 pixels has kernels.
            &["--kernels", "depth", "--passes", "1,2000"],
            &["--latency", "3"],
            &["--mode", "sync"],
            &["--estimator", "capped"],
            &["--split", "off"],
            &["--background-load"],
        ] {
            assert!(pass_cost(bad).is_err(), "{bad:?}");
        }
        // Each with a value it accepts, so the refusal is for the workload.
        for (name, value) in PASS_COST_ONLY
            .into_iter()
            .zip(["1", "ends", "trivial", "1", "on"])
        {
            assert!(pass_cost(&[name, value]).is_ok(), "{name}");
            assert!(parse_strs(&[name, value]).is_err(), "{name}");
        }
    }

    #[test]
    fn the_render_order_is_free_unless_after_is_asked_for() {
        assert_eq!(Args::default().render_order, RenderOrder::Free);
        let order = |extra: &[&str]| match parse_strs(extra) {
            Ok(Command::Run(a)) => Ok(a.render_order),
            Ok(Command::Help) => panic!("expected a run"),
            Err(e) => Err(e),
        };
        assert_eq!(order(&[]).unwrap(), RenderOrder::Free);
        assert_eq!(
            order(&["--render-order", "after"]).unwrap(),
            RenderOrder::After
        );
        assert_eq!(order(&["--render-order=free"]).unwrap(), RenderOrder::Free);
        assert_eq!(
            order(&["--workload", "depth", "--render-order", "after"]).unwrap(),
            RenderOrder::After
        );
        assert!(order(&["--render-order", "before"]).is_err());
        assert!(order(&["--render-order"]).is_err());
        // The pass-cost measurement has no loop to order.
        assert!(order(&["--workload", "pass-cost", "--render-order", "after"]).is_err());
    }

    #[test]
    fn the_default_is_the_synchronous_candidate_run_once() {
        let a = Args::default();
        assert_eq!((a.mode, a.runs), (Mode::Sync, 1));
    }
}
