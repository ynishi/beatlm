//! A program's kernels realised once: a pipeline and its bind groups per
//! spec, created before the frame loop.
//!
//! The steady-state loop must create nothing (the rule is the core's; see
//! [`Arenas`]), so everything a kernel needs is made in [`Kernels::new`], and
//! [`Kernels::encode`] only records a pass over existing objects. A host that
//! keeps several copies of an input or output, one per job in flight, passes
//! one binding set per copy; a kernel that touches none of the substituted
//! tensors shares one bind group between all sets.
//!
//! A spec that is one part of a split kernel is a kernel here like any other,
//! with its own pipeline, bind groups and dispatch; the parts of one kernel
//! share its shader module, since their text is the same and only the
//! pipeline's [constants](super::constants) differ.

use beatlm_core::kernel::KernelSpec;
use beatlm_core::memory::{MemoryPlan, TensorId};

use super::{CodegenError, ENTRY_POINT, Generator, constants};
use crate::gpu::{Arenas, profile};

/// A tensor of a spec and the tensor bound in its place: `(original,
/// stand_in)`. The stand-in must have a placement of the same size, such as
/// one described by
/// [`Program::stand_in`](beatlm_core::kernel::Program::stand_in).
pub type Substitution = (TensorId, TensorId);

/// One kernel: its pipeline, its bind groups and its dispatch.
#[derive(Debug)]
struct Kernel {
    label: String,
    pipeline: wgpu::ComputePipeline,
    bind_groups: Vec<wgpu::BindGroup>,
    /// For each binding set, the index of its bind group.
    group_of_set: Vec<usize>,
    workgroups: [u32; 3],
}

/// The pipelines and bind groups of a list of kernel specs, created once.
#[derive(Debug)]
pub struct Kernels {
    kernels: Vec<Kernel>,
    sets: usize,
}

impl Kernels {
    /// Creates a pipeline per spec in `specs` and a bind group per distinct
    /// binding of it, over tensors placed by `plan` in `arenas`.
    ///
    /// Each spec's variant is the one
    /// [`KernelSpec::select_variant`] picks for the device's profile, and its
    /// text passes naga's validation ([`generate`](super::generate)) before
    /// any shader module is created (each distinct text once, every part's
    /// constants against it). `sets` lists the binding sets: each one
    /// rebinds some tensors to stand-ins. No sets means one set that binds
    /// every spec as it is.
    ///
    /// # Errors
    ///
    /// A spec no variant of which runs on the device, generated text naga
    /// rejects, a bound tensor with no placement in `plan`, a stand-in of a
    /// different size, or a placement the arenas cannot bind.
    pub fn new(
        device: &wgpu::Device,
        plan: &MemoryPlan,
        arenas: &Arenas,
        specs: &[KernelSpec],
        sets: &[Vec<Substitution>],
    ) -> Result<Self, CodegenError> {
        let device_profile = profile(&device.limits(), device.features());
        let identity = [Vec::new()];
        let sets = if sets.is_empty() { &identity[..] } else { sets };
        let placement = |id: TensorId| plan.placement(id).ok_or(CodegenError::MissingPlacement(id));
        // The generator, so the parts of a kernel validate its text once, and
        // shader modules by the generator's index of that text, so they compile
        // it once.
        let mut modules: Vec<Option<wgpu::ShaderModule>> = Vec::new();
        let mut generator = Generator::new();
        let kernels = specs
            .iter()
            .map(|spec| {
                let name = spec.name();
                let part = spec.part();
                let label = if part.is_whole() {
                    name.clone()
                } else {
                    format!("{name} part {}/{}", part.index() + 1, part.count())
                };
                let variant = spec.select_variant(&device_profile).map_err(|source| {
                    CodegenError::Kernel {
                        kernel: name.clone(),
                        source,
                    }
                })?;
                let text = generator.check(spec, variant)?;
                if modules.len() <= text {
                    modules.resize_with(text + 1, || None);
                }
                let module = &*modules[text].get_or_insert_with(|| {
                    device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(&name),
                        source: wgpu::ShaderSource::Wgsl(generator.text(text).into()),
                    })
                });
                let values = constants(spec, variant);
                let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(&label),
                    layout: None,
                    module,
                    entry_point: Some(ENTRY_POINT),
                    compilation_options: wgpu::PipelineCompilationOptions {
                        constants: &values,
                        ..wgpu::PipelineCompilationOptions::default()
                    },
                    cache: None,
                });
                let layout = pipeline.get_bind_group_layout(0);

                // The tensors each set binds, kept once per distinct list.
                let bindings = spec.bindings();
                let mut lists: Vec<Vec<TensorId>> = Vec::new();
                let mut group_of_set = Vec::with_capacity(sets.len());
                for set in sets {
                    let mut resolved = Vec::with_capacity(bindings.len());
                    for &original in &bindings {
                        let bound = set
                            .iter()
                            .find(|(from, _)| *from == original)
                            .map_or(original, |&(_, to)| to);
                        let (want, got) = (placement(original)?.size, placement(bound)?.size);
                        if want != got {
                            return Err(CodegenError::StandInSize {
                                original,
                                stand_in: bound,
                                original_size: want,
                                stand_in_size: got,
                            });
                        }
                        resolved.push(bound);
                    }
                    let index = match lists.iter().position(|l| *l == resolved) {
                        Some(index) => index,
                        None => {
                            lists.push(resolved);
                            lists.len() - 1
                        }
                    };
                    group_of_set.push(index);
                }
                let bind_groups = lists
                    .iter()
                    .map(|list| {
                        let entries = (0..)
                            .zip(list)
                            .map(|(binding, &id)| {
                                Ok(wgpu::BindGroupEntry {
                                    binding,
                                    resource: wgpu::BindingResource::Buffer(
                                        arenas.binding(&placement(id)?)?,
                                    ),
                                })
                            })
                            .collect::<Result<Vec<_>, CodegenError>>()?;
                        Ok(device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some(&label),
                            layout: &layout,
                            entries: &entries,
                        }))
                    })
                    .collect::<Result<Vec<_>, CodegenError>>()?;
                Ok(Kernel {
                    workgroups: spec.requirements(variant).workgroups,
                    label,
                    pipeline,
                    bind_groups,
                    group_of_set,
                })
            })
            .collect::<Result<Vec<_>, CodegenError>>()?;
        Ok(Self {
            kernels,
            sets: sets.len(),
        })
    }

    /// How many kernels there are: one dispatch each.
    #[must_use]
    pub fn len(&self) -> usize {
        self.kernels.len()
    }

    /// Whether there are no kernels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.kernels.is_empty()
    }

    /// How many binding sets the kernels were built with.
    #[must_use]
    pub const fn sets(&self) -> usize {
        self.sets
    }

    /// The label of kernel `kernel`: its spec's name, followed for a part of a
    /// split kernel by `part i/n`, counting from one.
    #[must_use]
    pub fn label(&self, kernel: usize) -> Option<&str> {
        self.kernels.get(kernel).map(|k| k.label.as_str())
    }

    /// Records kernel `kernel` as its own compute pass, with the bind group of
    /// binding set `set`, timed by `timestamp_writes` when given.
    ///
    /// # Errors
    ///
    /// A kernel or set index past those the kernels were built with.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        kernel: usize,
        set: usize,
        timestamp_writes: Option<wgpu::ComputePassTimestampWrites<'_>>,
    ) -> Result<(), CodegenError> {
        // Both indices are checked before the pass opens, so a bad one records
        // nothing, not an empty pass carrying the timestamp writes.
        let (k, group) = self.lookup(kernel, set)?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&k.label),
            timestamp_writes,
        });
        Self::dispatch(&mut pass, k, group);
        Ok(())
    }

    /// Records kernel `kernel`'s dispatch, with the bind group of binding set
    /// `set`, into `pass`, a compute pass the caller opened and may already
    /// have recorded other kernels into.
    ///
    /// [`encode`](Self::encode) is one pass per kernel, the granularity the
    /// timer measures and the scheduler plans in; this is the dispatch alone,
    /// so a host can put several kernels in one pass, as the bench's
    /// `pass-cost` workload does to measure what a pass costs beside its
    /// dispatch. Dispatches in one pass run in recording order, each seeing
    /// the writes of those before it, as across passes.
    ///
    /// # Errors
    ///
    /// A kernel or set index past those the kernels were built with.
    pub fn record(
        &self,
        pass: &mut wgpu::ComputePass<'_>,
        kernel: usize,
        set: usize,
    ) -> Result<(), CodegenError> {
        let (k, group) = self.lookup(kernel, set)?;
        Self::dispatch(pass, k, group);
        Ok(())
    }

    /// Kernel `kernel` and its bind group for binding set `set`.
    fn lookup(
        &self,
        kernel: usize,
        set: usize,
    ) -> Result<(&Kernel, &wgpu::BindGroup), CodegenError> {
        let k = self
            .kernels
            .get(kernel)
            .ok_or(CodegenError::NoSuchKernel(kernel))?;
        let group = k
            .group_of_set
            .get(set)
            .and_then(|&g| k.bind_groups.get(g))
            .ok_or(CodegenError::NoSuchSet(set))?;
        Ok((k, group))
    }

    /// Records `k`'s dispatch with `group` into `pass`.
    fn dispatch(pass: &mut wgpu::ComputePass<'_>, k: &Kernel, group: &wgpu::BindGroup) {
        pass.set_pipeline(&k.pipeline);
        pass.set_bind_group(0, group, &[]);
        let [x, y, z] = k.workgroups;
        pass.dispatch_workgroups(x, y, z);
    }
}
