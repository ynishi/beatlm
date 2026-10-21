//! Running a graph once on a GPU adapter, for the tests that compare
//! generated kernels with the CPU references in [`crate::reference`].
//!
//! Everything here waits for the GPU, which the frame loop never does; it is
//! test code only, and every test that calls it is `#[ignore]`d as needing an
//! adapter.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;

use beatlm::codegen::Kernels;
use beatlm::gpu::Arenas;
use beatlm_core::device::DeviceProfile;
use beatlm_core::kernel::{Fusion, Graph, OpKind, TensorRole};
use beatlm_core::memory::{MemoryPlan, TensorId};

/// The tensor the first operation of a [`chain`] writes; each later one
/// writes the next id.
pub(crate) const CHAIN: TensorId = TensorId::new(10);
/// The tensor the last operation of a [`chain`] writes: its output.
pub(crate) const CHAIN_OUT: TensorId = TensorId::new(20);

/// A graph of `ops` in order, the first reading `inputs` (declared as
/// weights), each writing the next tensor from [`CHAIN`] and the last writing
/// [`CHAIN_OUT`]. At most ten operations, so the ids stay apart.
pub(crate) fn chain(inputs: &[(TensorId, &Vec<f32>)], ops: &[OpKind]) -> Graph {
    let mut g = Graph::new();
    for &(id, v) in inputs {
        g.tensor(id, TensorRole::Weight, v.len() as u64).unwrap();
    }
    for (i, &op) in ops.iter().enumerate() {
        let last = i + 1 == ops.len();
        let out = if last {
            CHAIN_OUT
        } else {
            TensorId::new(CHAIN.get() + i as u32)
        };
        let role = if last {
            TensorRole::Output
        } else {
            TensorRole::Activation
        };
        g.tensor(out, role, op.output_elements()).unwrap();
        g.push(op, out).unwrap();
    }
    g
}

/// `graph` on the GPU, fused and unfused, whole and split into each of
/// `parts`, against the reference interpreter and against itself run whole;
/// compares [`CHAIN_OUT`].
pub(crate) fn assert_chain_matches(
    what: &str,
    graph: &Graph,
    inputs: &[(TensorId, &Vec<f32>)],
    parts: &[u32],
) {
    let (device, queue) = device();
    let values: BTreeMap<TensorId, Vec<f32>> =
        inputs.iter().map(|&(id, v)| (id, v.clone())).collect();
    let mut want = values.clone();
    crate::reference::run(graph, &mut want);
    for fusion in [Fusion::On, Fusion::Off] {
        let whole = run_on_gpu(&device, &queue, graph, fusion, 1, &values);
        for &count in parts {
            let got = run_on_gpu(&device, &queue, graph, fusion, count, &values);
            let case = format!("{what}, fusion {fusion:?}, {count} parts");
            assert_close(&got[&CHAIN_OUT], &want[&CHAIN_OUT], &case);
            assert_close(
                &got[&CHAIN_OUT],
                &whole[&CHAIN_OUT],
                &format!("{case}, against whole"),
            );
        }
    }
}

/// `count` uniform random numbers scaled by `fan_in`, as a trained layer's
/// weights are, so sums stay near one.
pub(crate) fn weights(seed: u64, count: u32, fan_in: u32) -> Vec<f32> {
    let s = 1.0 / (fan_in as f32).sqrt();
    crate::net::uniform(seed, count as usize, -s, s)
}

/// The default adapter's device, with default limits and no optional feature.
pub(crate) fn device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .unwrap();
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).unwrap()
}

/// Copies `count` floats from `offset` of `buffer` and waits for them.
pub(crate) fn read(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    offset: u64,
    count: usize,
) -> Vec<f32> {
    let bytes = (count * 4) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_buffer_to_buffer(buffer, offset, &staging, 0, bytes);
    queue.submit([encoder.finish()]);
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, |r| r.unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let mapped = staging.slice(..).get_mapped_range().unwrap();
    mapped
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

/// Fails naming the first element of `got` outside the tolerance of `want`.
pub(crate) fn assert_close(got: &[f32], want: &[f32], what: &str) {
    if let Some((i, g, w)) = crate::reference::first_mismatch(got, want) {
        panic!("{what}: element {i} is {g}, the reference says {w}");
    }
}

/// Runs `graph`, lowered with `fusion` and every kernel split into `parts`
/// parts, in planned memory through generated kernels, with the inputs and
/// weights in `values`, and returns every output tensor's numbers.
pub(crate) fn run_on_gpu(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    graph: &Graph,
    fusion: Fusion,
    parts: u32,
    values: &BTreeMap<TensorId, Vec<f32>>,
) -> BTreeMap<TensorId, Vec<f32>> {
    let whole = graph.lower(fusion).unwrap();
    let program = whole.split(&vec![parts; whole.kernels().len()]).unwrap();
    let plan = MemoryPlan::new(program.tensors(), &DeviceProfile::webgpu_baseline()).unwrap();
    let arenas = Arenas::new(device, &plan).unwrap();
    for (id, numbers) in values {
        let p = plan.placement(*id).unwrap();
        let bytes: Vec<u8> = numbers.iter().flat_map(|v| v.to_le_bytes()).collect();
        queue.write_buffer(arenas.buffer(p.arena).unwrap(), p.offset, &bytes);
    }
    let kernels = Kernels::new(device, &plan, &arenas, program.kernels(), &[]).unwrap();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    for k in 0..kernels.len() {
        kernels.encode(&mut encoder, k, 0, None).unwrap();
    }
    queue.submit([encoder.finish()]);
    program
        .tensors()
        .iter()
        .filter(|t| program.role(t.id) == Some(TensorRole::Output))
        .map(|t| {
            let p = plan.placement(t.id).unwrap();
            let got = read(
                device,
                queue,
                arenas.buffer(p.arena).unwrap(),
                p.offset,
                t.elements as usize,
            );
            (t.id, got)
        })
        .collect()
}
