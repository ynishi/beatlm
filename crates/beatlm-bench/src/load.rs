//! A competing GPU load: a second device on the same adapter, driven by a
//! thread of its own, that keeps submitting a heavy compute kernel.
//!
//! The load makes slice times noisy the way another application on the GPU
//! does, so the cost model can be compared under noise it did not make. The
//! second device shares the adapter but no resource, queue or timer with the
//! run; it never reads a timestamp. The kernel is the slice kernel of
//! `shaders.wgsl` at a larger iteration count, so one number tunes how long the
//! load holds the GPU per dispatch.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::thread::{self, JoinHandle};

use anyhow::{Context, Result, anyhow};
use wgpu::util::DeviceExt;

use crate::device::Gpu;

const WORKGROUP_SIZE: u32 = 64;
/// Workgroups per dispatch: sixteen times the slice kernel's, so one dispatch
/// has far more invocations than the GPU runs at once and a slice has to wait
/// for execution units rather than run beside it.
const WORKGROUPS: u32 = 16_384;

/// A running competing load; stop it with [`BackgroundLoad::stop`].
pub struct BackgroundLoad {
    stop: Arc<AtomicBool>,
    worker: JoinHandle<Result<u64>>,
}

impl BackgroundLoad {
    /// Opens a second device on `gpu`'s adapter and starts a thread that
    /// submits a dispatch of `iters` loop iterations per invocation, keeping
    /// one dispatch queued behind the one running so the GPU is never idle for
    /// the load. Returns once the first dispatch has completed, so the load is
    /// running before the caller's first frame.
    ///
    /// # Errors
    ///
    /// A second device that cannot be opened, or a thread that ends before its
    /// first dispatch completes.
    pub fn start(gpu: &Gpu, iters: u32) -> Result<Self> {
        let (device, queue) =
            pollster::block_on(gpu.adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("beatlm-bench background load"),
                ..Default::default()
            }))
            .context("cannot open a second device for the background load")?;

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("beatlm-bench background shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders.wgsl").into()),
        });
        let mut params = Vec::with_capacity(16);
        for word in [iters, 0, 0, 0] {
            params.extend_from_slice(&word.to_le_bytes());
        }
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("beatlm-bench background params"),
            contents: &params,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let sink = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("beatlm-bench background sink"),
            size: u64::from(WORKGROUPS) * u64::from(WORKGROUP_SIZE) * 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("beatlm-bench background kernel"),
            layout: None,
            module: &module,
            entry_point: Some("slice_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("beatlm-bench background bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sink.as_entire_binding(),
                },
            ],
        });

        let stop = Arc::new(AtomicBool::new(false));
        let (ready_tx, ready_rx) = mpsc::channel::<()>();
        let worker = {
            let stop = Arc::clone(&stop);
            thread::spawn(move || -> Result<u64> {
                let dispatch = || {
                    let mut encoder =
                        device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("beatlm-bench background"),
                        });
                    {
                        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                            label: Some("beatlm-bench background"),
                            timestamp_writes: None,
                        });
                        pass.set_pipeline(&pipeline);
                        pass.set_bind_group(0, &bind_group, &[]);
                        pass.dispatch_workgroups(WORKGROUPS, 1, 1);
                    }
                    queue.submit([encoder.finish()])
                };
                let mut submitted: u64 = 0;
                let mut running = dispatch();
                submitted += 1;
                device
                    .poll(wgpu::PollType::Wait {
                        submission_index: Some(running.clone()),
                        timeout: None,
                    })
                    .context("background device poll failed")?;
                // The receiver waits for this; if it is gone, `start` failed
                // and the stop flag ends the loop below.
                let _ = ready_tx.send(());
                while !stop.load(Ordering::Relaxed) {
                    // Queue the next dispatch, then wait for the one before it:
                    // one is always waiting behind the one running.
                    let queued = dispatch();
                    submitted += 1;
                    device
                        .poll(wgpu::PollType::Wait {
                            submission_index: Some(running),
                            timeout: None,
                        })
                        .context("background device poll failed")?;
                    running = queued;
                }
                device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .context("background device poll failed")?;
                Ok(submitted)
            })
        };
        if ready_rx.recv().is_err() {
            stop.store(true, Ordering::Relaxed);
            return match worker.join() {
                Ok(Err(error)) => Err(error),
                Ok(Ok(_)) => Err(anyhow!("the background load ended before it started")),
                Err(_) => Err(anyhow!("the background load thread panicked")),
            };
        }
        Ok(Self { stop, worker })
    }

    /// Stops the load, waits for its last dispatch to finish, and returns how
    /// many dispatches it submitted.
    ///
    /// # Errors
    ///
    /// A device poll that failed, or a thread that panicked.
    pub fn stop(self) -> Result<u64> {
        self.stop.store(true, Ordering::Relaxed);
        self.worker
            .join()
            .map_err(|_| anyhow!("the background load thread panicked"))?
    }
}
