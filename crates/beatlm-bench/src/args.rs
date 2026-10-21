//! Command-line options, parsed by hand: they are a handful of numbers, and a
//! parser dependency would outweigh the code it replaces.

use anyhow::{Context, Result, bail};

/// The usage text, printed for `--help` and after a bad option.
pub const USAGE: &str = "\
usage: beatlm-bench [options]

  --frames N        frames to run                          (default 600)
  --latency K       promised delivery latency, in frames   (default 2)
  --slices S        slices per job, one job per frame      (default 4)
  --slice-iters N   loop iterations per invocation of a slice kernel
  --render-iters N  loop iterations per pixel of the render pass
  --margin-us N     safety margin in the frame budget, us  (default 2000)
  --help            print this text";

/// Loop iterations per slice-kernel invocation that put one slice at 0.8 to
/// 1.4 ms on an Apple M1 Pro (Metal); see the crate documentation.
pub const DEFAULT_SLICE_ITERS: u32 = 500;

/// Loop iterations per pixel that put the render pass at about 2 ms on an
/// Apple M1 Pro (Metal); see the crate documentation.
pub const DEFAULT_RENDER_ITERS: u32 = 50;

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
/// An unknown option, a missing or malformed value, or a value out of range
/// (zero frames or zero slices).
pub fn parse<I>(args: I) -> Result<Command>
where
    I: IntoIterator<Item = String>,
{
    let mut out = Args::default();
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
            "--slices" => out.slices = number(&name, &value()?)?,
            "--slice-iters" => out.slice_iters = number(&name, &value()?)?,
            "--render-iters" => out.render_iters = number(&name, &value()?)?,
            "--margin-us" => out.margin_us = number(&name, &value()?)?,
            _ => bail!("unknown option {name}"),
        }
    }
    if out.frames == 0 {
        bail!("--frames must be at least 1");
    }
    if out.slices == 0 {
        bail!("--slices must be at least 1");
    }
    Ok(Command::Run(out))
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
    }
}
