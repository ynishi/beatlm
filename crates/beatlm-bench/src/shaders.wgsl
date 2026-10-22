// Synthetic load for beatlm-bench. Both kernels spend their time in a loop whose
// trip count comes from a uniform, so the compiler cannot remove or unroll it,
// and both feed the loop's result to an output, so it cannot be dead code.

struct Params {
    slice_iters: u32,
    render_iters: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> sink: array<u32>;

// One inference slice: a serial integer hash chain per invocation.
@compute @workgroup_size(64)
fn slice_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    var x = gid.x * 747796405u + 2891336453u;
    for (var i = 0u; i < params.slice_iters; i = i + 1u) {
        x = x * 1664525u + 1013904223u;
        x = x ^ (x >> 15u);
        x = x * 2246822519u;
    }
    sink[gid.x] = x;
}

// A full-screen triangle: three vertices cover the whole target.
fn triangle(vi: u32) -> vec4<f32> {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}

@vertex
fn render_vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return triangle(vi);
}

// Read by `render_vs_after` only: the first word of the buffer the frame's
// slices write (`sink`, or the depth model's arena), bound read-only. That the
// vertex stage reads what they wrote is what orders the render pass after
// them (`workload.rs`, "Ordering the render pass after the slices").
@group(1) @binding(0) var<storage, read> written: array<u32>;

// `render_vs`, made a reader of `written`. The word moves the vertex by at
// most 2^24 * 1e-30 of clip space, which no rasteriser shows, and from a
// value the compiler cannot know, so the read is not removed.
@vertex
fn render_vs_after(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    let p = triangle(vi);
    return vec4<f32>(p.x + f32(written[0] >> 8u) * 1e-30, p.y, p.z, p.w);
}

// The render pass: a serial float chain per pixel.
@fragment
fn render_fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    var a = p.x * 0.0013 + p.y * 0.0007;
    for (var i = 0u; i < params.render_iters; i = i + 1u) {
        a = fract(sin(a) * 43758.5453);
    }
    return vec4<f32>(a, a * 0.5, 1.0 - a, 1.0);
}
