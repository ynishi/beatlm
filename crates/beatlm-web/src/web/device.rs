//! The adapter, the device and the canvas surface, requested from the browser.
//!
//! The device gets the WebGPU default limits unless a memory plan needs more,
//! and then exactly what the plan needs, through
//! [`beatlm::gpu::required_limits`]; a plan the adapter cannot hold is refused
//! there or by the planner, never cut down to fit. The plan is either the one
//! arena `arena_bytes` asks for or the depth model's ([`Need`]). Timestamp
//! queries are requested when the adapter has them, and their absence is
//! reported rather than refused: the run still keeps or misses its deadlines,
//! only the GPU times are not measured.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use anyhow::{Context, Result, bail};
use beatlm::gpu::{Arenas, profile, required_limits};
use beatlm_core::memory::{DType, MemoryPlan, TensorDesc, TensorId};
use beatlm_host::workload::{RENDER_HEIGHT, RENDER_WIDTH, TARGET_FORMAT};

use super::page;
use crate::report::SizeLimits;

/// Errors the device reported outside an error scope since it was created.
///
/// A static because wgpu requires the handler to be `Send` and `Sync`, which a
/// handle on the page or the run is not in a browser build; the page has one
/// run, so one count serves it.
static GPU_ERRORS: AtomicU64 = AtomicU64::new(0);

/// Why the device was lost, once it has been; a static for the reason
/// [`GPU_ERRORS`] is one.
static DEVICE_LOST: Mutex<Option<String>> = Mutex::new(None);

/// How many errors the device has reported so far.
pub fn gpu_errors() -> u64 {
    GPU_ERRORS.load(Ordering::Relaxed)
}

/// Why the device was lost, or `None` while it has not been.
pub fn device_lost() -> Option<String> {
    DEVICE_LOST
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone()
}

/// What the device must be able to hold.
pub enum Need {
    /// Nothing past the WebGPU defaults.
    Defaults,
    /// One arena of this many bytes, planned against the adapter's limits and
    /// realised on the device before the run (`arena_bytes`).
    Arena(u64),
    /// This plan, which the caller realises once the device exists: the depth
    /// model's, planned before the device so the device can be asked for it.
    Plan(MemoryPlan),
}

/// The device the run uses, with the surface on the page's canvas and what the
/// report says about them.
pub struct Gpu {
    /// The device that owns every resource.
    pub device: wgpu::Device,
    /// The queue the slices and the render pass share.
    pub queue: wgpu::Queue,
    /// The page's canvas, configured for the render pass.
    pub surface: wgpu::Surface<'static>,
    /// What the browser says about the adapter.
    pub info: wgpu::AdapterInfo,
    /// Whether the device was created with timestamp queries.
    pub timestamp_query: bool,
    /// The adapter's size limits.
    pub adapter_limits: SizeLimits,
    /// The device's size limits.
    pub device_limits: SizeLimits,
    /// The arena `arena_bytes` asked for, realised on the device.
    pub arena_realised_bytes: Option<u64>,
}

/// Requests a high-performance adapter compatible with `canvas`, a device with
/// the limits `need` needs and with timestamp queries when the adapter has
/// them, and configures the canvas for the render pass.
///
/// # Errors
///
/// No WebGPU in this browser or no adapter; a plan the adapter cannot hold;
/// a device the browser refuses; a canvas that cannot take the render
/// target's format.
pub async fn open(canvas: web_sys::HtmlCanvasElement, need: Need) -> Result<Gpu> {
    canvas.set_width(RENDER_WIDTH);
    canvas.set_height(RENDER_HEIGHT);
    let instance = wgpu::Instance::default();
    let surface = instance
        .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
        .context("cannot create a WebGPU surface on the canvas")?;
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            ..Default::default()
        })
        .await
        .context(
            "no WebGPU adapter: the browser has no navigator.gpu here, or it returned no adapter",
        )?;
    let info = adapter.get_info();
    let adapter_limits = adapter.limits();
    let timestamp_query = adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);

    // A plan may need more than the defaults; the device is asked for
    // exactly that. The arena is planned against what the adapter can do.
    let (plan, realise_here) = match need {
        Need::Defaults => (None, false),
        Need::Arena(bytes) => (Some(one_arena(bytes, &adapter_limits)?), true),
        Need::Plan(plan) => (Some(plan), false),
    };
    let required = match &plan {
        Some(plan) => required_limits(plan, &adapter_limits)?,
        None => wgpu::Limits::default(),
    };
    let required_features = if timestamp_query {
        wgpu::Features::TIMESTAMP_QUERY
    } else {
        wgpu::Features::empty()
    };
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("beatlm-web"),
            required_features,
            required_limits: required.clone(),
            ..Default::default()
        })
        .await
        .with_context(|| {
            format!(
                "the browser refused a device with max_storage_buffer_binding_size={} and \
                 max_buffer_size={}",
                required.max_storage_buffer_binding_size, required.max_buffer_size
            )
        })?;
    // Both handlers only log and record: the page is not written while frames
    // run, and the report counts the errors; a lost device fails the run at
    // the next callback.
    device.on_uncaptured_error(Arc::new(|error: wgpu::Error| {
        GPU_ERRORS.fetch_add(1, Ordering::Relaxed);
        page::log_error(&format!(
            "gpu_error: {}",
            page::one_line(&error.to_string())
        ));
    }));
    device.set_device_lost_callback(|reason, message| {
        let lost = format!("{reason:?} {}", page::one_line(&message));
        page::log_error(&format!("device_lost: {lost}"));
        *DEVICE_LOST.lock().unwrap_or_else(PoisonError::into_inner) = Some(lost);
    });

    let arena_realised_bytes = match &plan {
        Some(plan) if realise_here => Some(realise(&device, plan).await?),
        _ => None,
    };

    let mut config = surface
        .get_default_config(&adapter, RENDER_WIDTH, RENDER_HEIGHT)
        .context("the adapter cannot present to the canvas")?;
    if !surface
        .get_capabilities(&adapter)
        .formats
        .contains(&TARGET_FORMAT)
    {
        bail!("the canvas cannot be configured as {TARGET_FORMAT:?}");
    }
    config.format = TARGET_FORMAT;
    config.alpha_mode = wgpu::CompositeAlphaMode::Opaque;
    surface.configure(&device, &config);

    Ok(Gpu {
        timestamp_query: device.features().contains(wgpu::Features::TIMESTAMP_QUERY),
        device_limits: size_limits(&device.limits()),
        adapter_limits: size_limits(&adapter_limits),
        device,
        queue,
        surface,
        info,
        arena_realised_bytes,
    })
}

/// A plan of one tensor of at least `bytes` bytes, made against what an
/// adapter with `limits` can do.
fn one_arena(bytes: u64, limits: &wgpu::Limits) -> Result<MemoryPlan> {
    let tensor = TensorDesc::new(TensorId::new(0), DType::F32, bytes.div_ceil(4), 0, 0);
    MemoryPlan::new(&[tensor], &profile(limits, wgpu::Features::empty())).with_context(|| {
        format!(
            "an arena of {bytes} bytes does not fit this adapter, whose largest storage binding \
             is {} bytes and largest buffer {} bytes; it is refused, not cut down",
            limits.max_storage_buffer_binding_size, limits.max_buffer_size
        )
    })
}

/// Creates `plan`'s arenas on `device` and waits for the browser to accept
/// them, returning the largest arena's size.
async fn realise(device: &wgpu::Device, plan: &MemoryPlan) -> Result<u64> {
    let arenas = accepted(device, "the planned arena", || {
        Arenas::new(device, plan).map_err(anyhow::Error::from)
    })
    .await?;
    drop(arenas);
    Ok(plan.arenas().iter().copied().max().unwrap_or(0))
}

/// Runs `create`, which creates objects on `device`, and waits for the
/// browser to accept them; `what` names them in the error.
///
/// In a browser an object the device rejects still comes back as a handle and
/// the rejection arrives later as an error, so the creation runs inside an
/// error scope for validation and one for memory, and the scopes are awaited.
///
/// # Errors
///
/// `create`'s own error, or the first error either scope caught.
pub async fn accepted<T>(
    device: &wgpu::Device,
    what: &str,
    create: impl FnOnce() -> Result<T>,
) -> Result<T> {
    let out_of_memory = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let created = create();
    let invalid = validation.pop().await;
    let oom = out_of_memory.pop().await;
    let created = created?;
    if let Some(error) = invalid.or(oom) {
        bail!(
            "the device did not create {what}: {}",
            page::one_line(&error.to_string())
        );
    }
    Ok(created)
}

/// The two size limits the report shows.
fn size_limits(limits: &wgpu::Limits) -> SizeLimits {
    SizeLimits {
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        max_buffer_size: limits.max_buffer_size,
    }
}
