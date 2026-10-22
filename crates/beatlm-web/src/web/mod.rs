//! The browser side: the entry point, the device, the depth model's setup,
//! the frame loop and the page.
//!
//! Built only for `wasm32`; the crate documentation describes it.

mod audio;
mod device;
mod display;
mod fetch;
mod frame;
mod model;
mod page;
mod pass_cost;
mod wait;

use anyhow::{Context, Result};
use beatlm_core::sync::SyncPolicy;
use beatlm_core::time::Micros;
use beatlm_host::slots::Net;
use beatlm_host::workload::{DEFAULT_SLICE_ITERS, Workload};
use wasm_bindgen::prelude::wasm_bindgen;

use crate::config::{self, Config, Latency, Model};
use crate::depth;
use crate::onbeat;
use crate::pace::{Pace, Pacer, WINDOW, ms_to_us};
use crate::report::{DepthReport, Environment};
use device::Need;
use frame::{DepthLoad, Load, OnBeatRun, Run};

/// The module's entry point, run by the generated glue once the module is
/// instantiated: reads the configuration from the page's query string and
/// starts the setup as a task on the event loop, then returns, so the page is
/// never blocked.
#[wasm_bindgen(start)]
pub fn start() {
    std::panic::set_hook(Box::new(|info| {
        page::append_error(&format!(
            "error: panic: {}",
            page::one_line(&info.to_string())
        ));
        page::append_error("status: failed");
    }));
    let config = match query().and_then(|query| config::parse(&query)) {
        Ok(config) => config,
        Err(error) => {
            page::fail(&error);
            return;
        }
    };
    page::show(&format!("status: starting {}", describe(&config)));
    wasm_bindgen_futures::spawn_local(async move {
        if let Err(error) = setup(config).await {
            page::fail(&error);
        }
    });
}

/// The configuration in one line, for the status the page shows while it
/// sets up.
fn describe(config: &Config) -> String {
    let model = match &config.model {
        Model::Synthetic => "model=synthetic".to_owned(),
        Model::Depth { side } => format!("model=depth side={side}"),
        // No promise and no estimator: the measurement has only its own
        // settings.
        Model::PassCost(settings) => {
            return format!(
                "model=pass-cost frames={} {}",
                config.frames,
                settings.describe()
            );
        }
    };
    let latency = match (config.on_beat, config.latency) {
        (Some(on_beat), _) => {
            let mut text = format!(
                "sync=onbeat bpm={} margin_ms={}",
                on_beat.tempo.bpm,
                crate::report::milliseconds(on_beat.margin.get())
            );
            if let Some(change) = on_beat.tempo.change {
                text.push_str(&format!(" bpm_change={}@{}", change.bpm, change.after_s));
            }
            if let Some(beats) = on_beat.beats {
                text.push_str(&format!(" beats={beats}"));
            }
            text
        }
        (None, Latency::Frames(frames)) => format!("latency={frames}"),
        (None, Latency::Time(us)) => {
            format!("latency_ms={}", crate::report::milliseconds(us.get()))
        }
    };
    format!(
        "{model} frames={} {latency} estimator={} render_iters={} render_order={}",
        config.frames,
        crate::config::estimator_name(config.estimator),
        config.render_iters,
        config.render_order.name()
    )
}

/// The page's query string, `?` included.
fn query() -> Result<String> {
    page::window()?
        .location()
        .search()
        .map_err(|error| anyhow::anyhow!("cannot read the page's query string: {error:?}"))
}

/// The display's period, measured over the callbacks of half a measuring
/// window before anything else runs, for a latency asked in time.
///
/// It is the pacer's rule ([`crate::pace`]) over its own callbacks, so the
/// period the latency is turned into frames at is the one the run's frames
/// will be given; the nominal period when too few callbacks came on time to
/// measure one.
async fn display_period() -> Result<Micros> {
    let frames = u64::try_from(WINDOW / 2 + 1).unwrap_or(u64::MAX);
    let mut pacer = Pacer::new(frames);
    loop {
        let timestamp = wait::next_frame().await?;
        if let Pace::Done = pacer.frame_at(ms_to_us(timestamp)) {
            return Ok(pacer.period());
        }
    }
}

/// The display's period measured before the run, and the promised latency in
/// frames.
///
/// The period is measured for a latency asked in time, which needs it, and
/// for the depth model, whose first frames would otherwise be given the
/// nominal 60 Hz period and, on a faster display, overrun theirs while the
/// loop's pacer measures it; a run of the synthetic workload with a latency in
/// frames starts from the nominal period, as it always has. An on-beat run
/// measures it too, to turn `beats` into frames; its latency in frames is
/// unused.
async fn latency(config: &Config) -> Result<(Option<Micros>, u32)> {
    let measure = matches!(config.latency, Latency::Time(_))
        || matches!(config.model, Model::Depth { .. })
        || config.on_beat.is_some();
    let period = if measure {
        page::show("status: measuring the display's period");
        Some(display_period().await?)
    } else {
        None
    };
    let frames = match (config.latency, period) {
        (Latency::Frames(frames), _) => frames,
        (Latency::Time(time), Some(period)) => depth::frames_for(time, period),
        (Latency::Time(time), None) => depth::frames_for(time, crate::pace::NOMINAL_PERIOD),
    };
    Ok((period, frames))
}

/// Opens the device, builds and warms up the load, and starts the frame loop.
///
/// The warm-up runs a slice and a render pass once and waits for them, so
/// shader compilation and first use are not charged to frame 0; the wait is
/// an `await` on the event loop, before the first frame, where the bench
/// blocks on the device instead. The depth model's setup comes between the
/// device and the loop ([`depth_load`]).
async fn setup(mut config: Config) -> Result<()> {
    // The measurement has no job, promise or audio: it runs apart.
    if let Model::PassCost(settings) = &config.model {
        return pass_cost::run(&config, settings).await;
    }
    // The audio context starts only from a user gesture, so an on-beat run
    // waits for the start button before anything else, and says so.
    let audio = match config.on_beat {
        Some(_) => {
            page::show(&format!(
                "status: waiting for start: {} plays a click on every beat, and a browser \
                 starts audio only after a click on the page, so the run starts when \
                 the start button is pressed",
                describe(&config)
            ));
            Some(audio::Audio::new(audio::start_on_click().await?))
        }
        None => None,
    };
    let (period, latency) = latency(&config).await?;
    // A run asked in beats lasts the frames they take at the measured period.
    let in_beats = config
        .on_beat
        .and_then(|on_beat| Some((on_beat, on_beat.beats?)));
    if let Some((on_beat, beats)) = in_beats {
        config.frames = onbeat::run_frames(
            &on_beat.tempo,
            beats,
            period.unwrap_or(crate::pace::NOMINAL_PERIOD),
        );
    }
    // An on-beat run asks for no latency: the depth model's default
    // `latency_ms` is not one it asked for.
    let asked = match (config.on_beat, config.latency) {
        (None, Latency::Time(time)) => Some(time),
        _ => None,
    };
    let (gpu, workload, load) = match config.model {
        Model::PassCost(_) => anyhow::bail!("the pass-cost measurement runs apart"),
        Model::Synthetic => {
            let need = config.arena_bytes.map_or(Need::Defaults, Need::Arena);
            let gpu = device::open(page::canvas()?, need).await?;
            let mut workload = Workload::new(&gpu.device, DEFAULT_SLICE_ITERS, config.render_iters);
            // The synthetic slices write the load's own sink.
            workload.order_render(&gpu.device, config.render_order, &[])?;
            warm_up(&gpu, &workload).await?;
            (gpu, workload, Load::Synthetic)
        }
        Model::Depth { side } => {
            // One slot per job that can be in flight; the device is asked for
            // what the loop's plan of the unsplit model needs. On beat the
            // host bounds the jobs pending ([`onbeat::SLOTS`]).
            let slots = match config.on_beat {
                Some(_) => onbeat::SLOTS,
                None => SyncPolicy::FixedLatency { frames: latency }
                    .max_in_flight()
                    .context("the policy does not bound the jobs in flight")?,
            };
            // The plan needs only the graph, so a render order it cannot
            // serve is refused before the weights are downloaded.
            let graph = depth::graph(side)?;
            let plan = graph.memory_plan(slots)?;
            depth::check_render_order(config.render_order, &plan, side, slots)?;
            let url = depth::weights_url();
            page::show(&format!("status: fetching {url}"));
            let loaded = model::load(graph).await?;
            let gpu = device::open(page::canvas()?, Need::Plan(plan)).await?;
            let workload = Workload::new(&gpu.device, DEFAULT_SLICE_ITERS, config.render_iters);
            warm_up(&gpu, &workload).await?;
            let load = depth_load(&gpu, &workload, loaded, slots, url, &config).await?;
            (gpu, workload, load)
        }
    };

    let env = environment(&gpu)?;
    match config.on_beat {
        Some(_) => page::show(&format!("status: running {}", describe(&config))),
        None => page::show(&format!(
            "status: running {} at latency={latency}",
            describe(&config)
        )),
    }
    let on_beat = match (config.on_beat, audio) {
        (Some(config), Some(audio)) => Some(OnBeatRun { config, audio }),
        _ => None,
    };
    let run = Run {
        frames: config.frames,
        latency,
        asked,
        period,
        load,
        estimator: config.estimator,
        on_beat,
    };
    let host = frame::Host::new(gpu, workload, run).context("cannot start the frame loop")?;
    frame::run(host, env, config)
}

/// Where the run happens and what the device was created with, for the
/// report's first lines.
///
/// # Errors
///
/// A page whose user agent cannot be read.
fn environment(gpu: &device::Gpu) -> Result<Environment> {
    Ok(Environment {
        adapter: gpu.info.name.clone(),
        backend: gpu.info.backend.to_string(),
        user_agent: page::window()?
            .navigator()
            .user_agent()
            .map_err(|error| anyhow::anyhow!("cannot read the user agent: {error:?}"))?,
        date: String::from(js_sys::Date::new_0().to_iso_string()),
        timestamp_query: gpu.timestamp_query,
        adapter_limits: gpu.adapter_limits,
        device_limits: gpu.device_limits,
        arena_realised_bytes: gpu.arena_realised_bytes,
    })
}

/// Runs a synthetic slice and render pass once and waits for them.
async fn warm_up(gpu: &device::Gpu, workload: &Workload) -> Result<()> {
    let mut encoder = gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("beatlm-web warm-up"),
        });
    workload.encode_slice(&mut encoder, None);
    let canvas = match gpu.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(texture)
        | wgpu::CurrentSurfaceTexture::Suboptimal(texture) => texture,
        other => anyhow::bail!("the canvas has no texture to draw into: {other:?}"),
    };
    let view = canvas
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    workload.encode_render(&mut encoder, &view, None);
    gpu.queue.submit([encoder.finish()]);
    gpu.queue.present(canvas);
    wait::work_done(&gpu.queue).await;
    Ok(())
}

/// The depth model, from its weights to the loop: calibrated and split,
/// realised with `slots` slots, warmed up, checked against the reference, and
/// with its display.
async fn depth_load(
    gpu: &device::Gpu,
    workload: &Workload,
    mut loaded: model::Loaded,
    slots: u32,
    weights_url: String,
    config: &Config,
) -> Result<Load> {
    page::show(&format!(
        "status: calibrating {} kernels at {} pixels over {} frames, twice",
        loaded.program.kernels().len(),
        loaded.side,
        beatlm_host::presets::CALIBRATION_FRAMES + 1
    ));
    let calibration = model::calibrate(
        gpu,
        workload,
        &loaded.program,
        &loaded.numbers,
        config.estimator,
    )
    .await?;
    let net = device::accepted(&gpu.device, "the depth model's network", || {
        Net::new(
            &gpu.device,
            &gpu.queue,
            &loaded.numbers,
            slots,
            calibration.program.clone(),
        )
    })
    .await?;
    // The weights are on the GPU now; the display needs only the slots' ids.
    loaded.numbers.weights = Vec::new();
    page::show("status: checking the depth against the reference");
    model::warm_up(gpu, &net).await?;
    let check = model::check(gpu, &net, &loaded.numbers, loaded.side).await?;
    let display = display::Display::new(
        &gpu.device,
        &net,
        &loaded.numbers,
        loaded.side,
        check.depth_range,
        config.render_iters,
        config.render_order,
    )?;
    let plan = net.plan();
    let report = DepthReport {
        side: loaded.side,
        weights_url,
        weights_bytes: loaded.weights_bytes,
        tensors: loaded.tensors,
        parameters: loaded.parameters,
        kernels: u32::try_from(loaded.program.kernels().len()).unwrap_or(u32::MAX),
        slices: net.kernel_count(),
        calibrated: calibration.durations,
        runs_us: calibration.runs_us,
        parts_us: calibration.parts_us,
        arenas: plan.arenas().to_vec(),
        total_bytes: plan.total_bytes(),
        peak_live_bytes: plan.peak_live_bytes(),
        slots: net.slots(),
        check,
    };
    Ok(Load::Depth(Box::new(DepthLoad {
        net,
        display,
        cost: calibration.cost,
        report,
    })))
}
