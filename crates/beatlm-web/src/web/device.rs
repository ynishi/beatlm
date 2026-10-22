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
use beatlm::gpu::{Arenas, FrameTiming, ResolvedFrame, SliceTimer, profile, required_limits};
use beatlm_core::memory::{DType, MemoryPlan, TensorDesc, TensorId};
use beatlm_host::workload::{RENDER_HEIGHT, RENDER_WIDTH, TARGET_FORMAT};

use super::page;
use crate::report::SizeLimits;

/// Errors the device reported outside an error scope since it was created:
/// what the `uncapturederror` handler counted.
///
/// A static because wgpu requires the handler to be `Send` and `Sync`, which a
/// handle on the page or the run is not in a browser build; the page has one
/// run, so one count serves it.
static UNCAPTURED_ERRORS: AtomicU64 = AtomicU64::new(0);

/// Errors a frame's error scope caught ([`Submission`]), which never reach
/// the handler; a static for the reason [`UNCAPTURED_ERRORS`] is one.
static SCOPED_ERRORS: AtomicU64 = AtomicU64::new(0);

/// Frames' error scopes popped whose promise has not settled yet
/// ([`Submission`]); a static for the reason [`UNCAPTURED_ERRORS`] is one.
static SCOPES_PENDING: AtomicU64 = AtomicU64::new(0);

/// Why the device was lost, once it has been; a static for the reason
/// [`UNCAPTURED_ERRORS`] is one.
static DEVICE_LOST: Mutex<Option<String>> = Mutex::new(None);

/// How many errors the device has reported so far: those raised outside any
/// error scope and those a frame's scope caught, so `gpu_errors` in the
/// report counts every rejected frame.
///
/// A frame's scope keeps only the first error it catches, as every error
/// scope does, so a rejected frame counts once however many of its calls
/// raised one (an invalid encoder's `finish` and the `submit` that refuses
/// it). A scope whose promise has not settled is not counted yet
/// ([`scopes_pending`]). It leaves out errors the scopes of [`accepted`]
/// caught, which fail the run before it starts.
pub fn gpu_errors() -> u64 {
    UNCAPTURED_ERRORS
        .load(Ordering::Relaxed)
        .saturating_add(SCOPED_ERRORS.load(Ordering::Relaxed))
}

/// How many frames' error scopes have not settled, and so are not yet in
/// [`gpu_errors`]: a run that reports the count waits for this to reach
/// zero, as it waits for its frames' timings.
pub fn scopes_pending() -> u64 {
    SCOPES_PENDING.load(Ordering::Relaxed)
}

/// How the page learns that the device rejected a frame it submitted, and
/// keeps the timer from reading it: the rule is **a frame is collected only
/// once the error scope around its own submission settled with no error, and
/// while no uncaptured error has been raised since it was submitted**. The
/// page submits a timed frame only through a [`Submission`] and collects
/// only through [`collect_ready`](Self::collect_ready), so no frame skips it.
///
/// **A frame rejected at its own submission is never read**, whatever order
/// the browser delivers its map and its error in. [`Submission`] pushes a
/// validation scope before the frame's `finish`, pops it straight after its
/// `queue.submit`, and hands the frame to the timer held
/// ([`SliceTimer::submitted_held`]); the frame is accepted once the scope's
/// promise settles with no error and rejected if it caught one, so only that
/// frame loses its timing. The scope's error is counted in `gpu_errors` and
/// logged as the handler logs, because a scoped error never reaches the
/// handler.
///
/// The scope takes `finish` and `queue.submit` because those are the calls
/// that report a frame's rejection: an encoder's recording errors, its
/// passes' included, are raised by `finish`, and a command buffer that is
/// invalid or uses a resource it may not is refused by `submit`; recording
/// itself raises nothing. Validation is the only filter, because a rejected
/// command buffer always raises a validation error, and neither call
/// allocates, so neither raises an out-of-memory one. The map is requested
/// after the pop, so its own errors stay outside the scope.
///
/// **An error outside every frame's scope** (an allocation that ran out of
/// memory, the map's own errors) reaches the page only as an
/// `uncapturederror`, which does not say which submission it was. A check of
/// that count just before [`SliceTimer::collect_ready`] marks everything the
/// timer holds in flight, held frames included
/// ([`SliceTimer::invalidate_submitted`]), once the handler has counted the
/// error; valid frames among it lose their timing. That guarantee is weaker:
/// the specification does not order the event before a map's completion (the
/// map's success steps are one task on the WebGPU task source, while
/// dispatching an uncaptured error queues a task that queues a second to fire
/// the event), and it lets a user agent throttle or not fire the event at
/// all. What it does guarantee is that a frame in flight when such an error
/// is counted, before the callback that would collect it, is never read.
///
/// The scope costs a push and a pop into the browser, a promise and a task
/// per frame (the loop charges the push to `cpu_encode_us` and so to
/// `cpu_start_to_submit_us`, the pop and the task's creation to
/// `cpu_start_to_tick_end_us`; `cpu_queue_submit_us` holds neither), and
/// holds each frame's collection until its scope settles. Frames are
/// collected in submission order (`SliceTimer::collect_ready` has the rule),
/// so a frame whose scope has not settled also holds back the frames after
/// it, and a rejected one holds back nothing; a held frame keeps its readback
/// slot as any frame in flight does, so a frame that needs the slot goes
/// untimed and counts in `timings_skipped`.
pub struct ErrorWatch {
    seen: u64,
}

impl ErrorWatch {
    /// A watch that counts uncaptured errors from those reported so far.
    pub fn from_now() -> Self {
        Self {
            seen: UNCAPTURED_ERRORS.load(Ordering::Relaxed),
        }
    }

    /// [`SliceTimer::collect_ready`], after marking everything `timer` was
    /// handed and has not returned when the device has raised an uncaptured
    /// error since the last call.
    pub fn collect_ready<T>(&mut self, timer: &mut SliceTimer<T>) -> Vec<FrameTiming<T>> {
        let errors = UNCAPTURED_ERRORS.load(Ordering::Relaxed);
        if errors != self.seen {
            self.seen = errors;
            timer.invalidate_submitted();
        }
        timer.collect_ready()
    }
}

/// One frame's submission, from [`open`](Self::open) to the frame's return to
/// the timer held: the frame's validation error scope. [`ErrorWatch`] has the
/// rule.
///
/// Dropped instead of handed back, on an error path, it pops its scope and
/// discards what the scope caught; the frame it was opened for is then never
/// handed to the timer either, so nothing is read.
///
/// **A pop the browser refuses accepts the frame.** wgpu's pop future yields
/// `None` both when the scope caught nothing and when the `popErrorScope`
/// promise rejects (the specification rejects it with an `OperationError`
/// when the device's scope stack is empty), so the two cannot be told apart
/// here, and such a frame is read as one the device ran.
#[must_use = "hand it back with Submission::submitted straight after queue.submit"]
pub struct Submission {
    scope: wgpu::ErrorScopeGuard,
}

impl Submission {
    /// Opens a frame's submission on `device`: call it after the frame is
    /// recorded and before its encoder is finished, and hand the result back
    /// straight after `queue.submit` ([`Submission::submitted`]), with no
    /// other call on the device in between. A host whose encode time is a
    /// measurement compared with the native bench's (`model=pass-cost`)
    /// opens it before the frame's encoder is created instead, so the push
    /// is not in the span: recording raises nothing, and an error the
    /// frame's own creation calls raise is the frame's.
    pub fn open(device: &wgpu::Device) -> Self {
        Self {
            scope: device.push_error_scope(wgpu::ErrorFilter::Validation),
        }
    }

    /// Closes the submission straight after `queue.submit`, and hands
    /// `resolved` to `timer` held until the scope settles. Either may be
    /// `None`, for a frame with no timer; the scope still counts what it
    /// caught.
    pub fn submitted<T>(self, timer: Option<&mut SliceTimer<T>>, resolved: Option<ResolvedFrame>) {
        // Popped before the map is requested, so the map's own errors stay
        // outside the frame's scope.
        let caught = self.scope.pop();
        let held = match (timer, resolved) {
            (Some(timer), Some(resolved)) => timer.submitted_held(resolved),
            _ => None,
        };
        SCOPES_PENDING.fetch_add(1, Ordering::Relaxed);
        wasm_bindgen_futures::spawn_local(async move {
            let caught = caught.await;
            // Counted before the frame is settled, so a run that sees the
            // frame collected or failed also sees its error counted.
            if caught.is_some() {
                SCOPED_ERRORS.fetch_add(1, Ordering::Relaxed);
            }
            SCOPES_PENDING.fetch_sub(1, Ordering::Relaxed);
            match caught {
                None => {
                    if let Some(held) = held {
                        held.accept();
                    }
                }
                Some(error) => {
                    page::log_error(&format!(
                        "gpu_error: {}",
                        page::one_line(&error.to_string())
                    ));
                    if let Some(held) = held {
                        held.reject();
                    }
                }
            }
        });
    }
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
        UNCAPTURED_ERRORS.fetch_add(1, Ordering::Relaxed);
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
