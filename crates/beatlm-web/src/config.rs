//! What a run is configured with, read from the page's query string.
//!
//! The query is parsed by hand for the same reason the bench parses its
//! command line by hand: a handful of numbers. Parsing the text, rather than
//! asking the browser's `URLSearchParams`, keeps it testable without a
//! browser. Anything not understood is refused, so a mistyped parameter
//! cannot leave a run on its default without saying so.

use anyhow::{Context, Result, bail};

/// Frames a run lasts when `frames` is not given: the bench's default.
pub const DEFAULT_FRAMES: u64 = 600;

/// The promised latency in frames when `latency` is not given: the bench's
/// default for the synthetic workload.
pub const DEFAULT_LATENCY: u32 = 2;

/// Slices in each frame's job: the bench's default for the synthetic workload,
/// which the page does not change.
pub const SLICES: u32 = 4;

/// The margin kept free in each frame's budget, in microseconds: the bench's
/// default, which the page does not change.
pub const MARGIN_US: u64 = 2_000;

/// The parameters the query string accepts, as the error for an unknown one
/// lists them.
const PARAMETERS: &str = "frames, latency, arena_bytes";

/// A run's configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Config {
    /// Frames to run, `frames`; at least one.
    pub frames: u64,
    /// The promised latency of the fixed-latency policy in frames, `latency`.
    pub latency: u32,
    /// `arena_bytes`: before the run, plan one arena of this many bytes
    /// against the adapter's limits, create the device with the limits it
    /// needs and realise it there, to check the limit path; `None` for a run
    /// that needs only the default limits.
    pub arena_bytes: Option<u64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            frames: DEFAULT_FRAMES,
            latency: DEFAULT_LATENCY,
            arena_bytes: None,
        }
    }
}

/// Parses a query string such as `?frames=600&latency=2`, with or without the
/// leading `?`. An empty query is the default configuration.
///
/// # Errors
///
/// An unknown or repeated parameter, one without a value, a value that is not
/// a number, zero frames or an arena of zero bytes.
pub fn parse(query: &str) -> Result<Config> {
    let mut config = Config::default();
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
            "frames" => config.frames = number(name, value)?,
            "latency" => config.latency = number(name, value)?,
            "arena_bytes" => config.arena_bytes = Some(number(name, value)?),
            _ => bail!("unknown query parameter {name:?}; the page accepts {PARAMETERS}"),
        }
    }
    if config.frames == 0 {
        bail!("frames must be at least 1");
    }
    if config.arena_bytes == Some(0) {
        bail!("arena_bytes must be at least 1");
    }
    Ok(config)
}

/// Parses the value of parameter `name` as a number.
fn number<T: std::str::FromStr>(name: &str, value: &str) -> Result<T> {
    value
        .parse()
        .ok()
        .with_context(|| format!("query parameter {name}: {value:?} is not a number in range"))
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_query_is_the_benchs_defaults() {
        for query in ["", "?"] {
            assert_eq!(parse(query).unwrap(), Config::default());
        }
        assert_eq!(Config::default().frames, 600);
        assert_eq!(Config::default().latency, 2);
    }

    #[test]
    fn frames_latency_and_the_arena_are_read() {
        assert_eq!(
            parse("?frames=300&latency=5&arena_bytes=134217600").unwrap(),
            Config {
                frames: 300,
                latency: 5,
                arena_bytes: Some(134_217_600),
            }
        );
        assert_eq!(parse("latency=0").unwrap().latency, 0);
        assert_eq!(parse("&frames=1&").unwrap().frames, 1);
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
        ] {
            assert!(parse(query).is_err(), "{query}");
        }
        let unknown = parse("?frame=600").unwrap_err().to_string();
        assert!(
            unknown.contains("frames, latency, arena_bytes"),
            "{unknown}"
        );
    }
}
