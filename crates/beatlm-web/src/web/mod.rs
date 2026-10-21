//! The browser side: the entry point, the device, the frame loop and the page.
//!
//! Built only for `wasm32`; the crate documentation describes it.

mod device;
mod frame;
mod page;

use anyhow::{Context, Result};
use wasm_bindgen::prelude::wasm_bindgen;

use crate::config::{self, Config};
use crate::report::Environment;
use crate::workload::{DEFAULT_RENDER_ITERS, DEFAULT_SLICE_ITERS, Workload};

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
    page::show(&format!(
        "status: running frames={} latency={}",
        config.frames, config.latency
    ));
    wasm_bindgen_futures::spawn_local(async move {
        if let Err(error) = setup(config).await {
            page::fail(&error);
        }
    });
}

/// The page's query string, `?` included.
fn query() -> Result<String> {
    page::window()?
        .location()
        .search()
        .map_err(|error| anyhow::anyhow!("cannot read the page's query string: {error:?}"))
}

/// Opens the device, builds and warms up the load, and starts the frame loop.
///
/// The warm-up runs a slice and a render pass once and waits for them, so
/// shader compilation and first use are not charged to frame 0; the wait is
/// an `await` on the event loop, before the first frame, where the bench
/// blocks on the device instead.
async fn setup(config: Config) -> Result<()> {
    let gpu = device::open(page::canvas()?, config.arena_bytes).await?;
    let workload = Workload::new(&gpu.device, DEFAULT_SLICE_ITERS, DEFAULT_RENDER_ITERS);

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
    device::work_done(&gpu.queue).await;

    let env = Environment {
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
    };
    let host = frame::Host::new(gpu, workload, &config).context("cannot start the frame loop")?;
    frame::run(host, env, config)
}
