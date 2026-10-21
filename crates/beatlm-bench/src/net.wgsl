// Hand-written kernels for the planned network of beatlm-bench. They are
// scaffolding until generated kernels replace them, and are written against the
// WebGPU baseline: at most 256 invocations per workgroup, at most 16 KiB of
// workgroup memory, at most 8 storage bindings.
//
// Every storage binding is `read_write`, inputs included. Tensors that share an
// arena share a buffer, and wgpu checks usage per buffer: an input bound
// read-only next to an output bound read-write from the same buffer would be
// rejected although their byte ranges are disjoint.
//
// Shapes arrive as pipeline-overridable constants, so they cost no uniform
// buffer and no binding; each kernel's constants have their own prefix, and the
// defaults exist only so every override has one.

// ---- 3x3 convolution, stride 1, zero padding of 1 -------------------------
//
// Layout: activations are NHWC with N = 1, i.e. [H][W][C] row-major, so the
// channels of one pixel are contiguous. Weights are HWIO, [3][3][CIN][COUT], so
// neighbouring output channels read neighbouring weights. One invocation
// computes one output element. Elements are numbered in memory order (channel
// fastest) over a grid CONV_ROW workgroups wide, because their count can exceed
// the limit on workgroups in one dimension.

override CONV_H: u32 = 1u;
override CONV_W: u32 = 1u;
override CONV_CIN: u32 = 1u;
override CONV_COUT: u32 = 1u;
// Set from `net.rs`, which sizes the dispatch from the same numbers.
override CONV_GROUP: u32 = 256u;
override CONV_ROW: u32 = 256u;

@group(0) @binding(0) var<storage, read_write> conv_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> conv_weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> conv_out: array<f32>;

@compute @workgroup_size(CONV_GROUP)
fn conv3x3(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let element = (wid.y * CONV_ROW + wid.x) * CONV_GROUP + lane;
    if (element >= CONV_H * CONV_W * CONV_COUT) {
        return;
    }
    let co = element % CONV_COUT;
    let pixel = element / CONV_COUT;
    let y = i32(pixel / CONV_W);
    let x = i32(pixel % CONV_W);
    var acc = 0.0;
    for (var ky = 0; ky < 3; ky = ky + 1) {
        let iy = y + ky - 1;
        if (iy < 0 || iy >= i32(CONV_H)) {
            continue;
        }
        for (var kx = 0; kx < 3; kx = kx + 1) {
            let ix = x + kx - 1;
            if (ix < 0 || ix >= i32(CONV_W)) {
                continue;
            }
            let src = (u32(iy) * CONV_W + u32(ix)) * CONV_CIN;
            let tap = u32(ky * 3 + kx) * CONV_CIN * CONV_COUT + co;
            for (var ci = 0u; ci < CONV_CIN; ci = ci + 1u) {
                acc = acc + conv_in[src + ci] * conv_weights[tap + ci * CONV_COUT];
            }
        }
    }
    conv_out[element] = acc;
}

// ---- layer normalisation over channels ------------------------------------
//
// y = (x - mean) / sqrt(var + eps) * gamma + beta, with mean and var taken over
// the channels of each pixel. `ln_params` holds gamma in [0, C) and beta in
// [C, 2C). One workgroup per pixel: its 64 invocations stride over the channels
// and reduce through workgroup memory. Pixels are numbered row-major over a
// grid LN_ROW workgroups wide, because the pixel count can exceed the limit on
// workgroups in one dimension.

override LN_C: u32 = 1u;
override LN_PIXELS: u32 = 1u;
// Set from `net.rs`, which sizes the dispatch from the same number.
override LN_ROW: u32 = 256u;
const LN_EPS: f32 = 1e-5;

@group(0) @binding(0) var<storage, read_write> ln_in: array<f32>;
@group(0) @binding(1) var<storage, read_write> ln_params: array<f32>;
@group(0) @binding(2) var<storage, read_write> ln_out: array<f32>;

var<workgroup> ln_scratch: array<f32, 64>;

fn ln_reduce(lane: u32) -> f32 {
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            ln_scratch[lane] = ln_scratch[lane] + ln_scratch[lane + stride];
        }
        workgroupBarrier();
    }
    let total = ln_scratch[0];
    // Everyone has read the total before the scratch is written again.
    workgroupBarrier();
    return total;
}

@compute @workgroup_size(64)
fn layernorm(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {
    let pixel = wid.y * LN_ROW + wid.x;
    // Uniform across the workgroup, so the barriers below are safe.
    if (pixel >= LN_PIXELS) {
        return;
    }
    let base = pixel * LN_C;

    var sum = 0.0;
    for (var c = lane; c < LN_C; c = c + 64u) {
        sum = sum + ln_in[base + c];
    }
    ln_scratch[lane] = sum;
    let mean = ln_reduce(lane) / f32(LN_C);

    var sq = 0.0;
    for (var c = lane; c < LN_C; c = c + 64u) {
        let d = ln_in[base + c] - mean;
        sq = sq + d * d;
    }
    ln_scratch[lane] = sq;
    let inv = inverseSqrt(ln_reduce(lane) / f32(LN_C) + LN_EPS);

    for (var c = lane; c < LN_C; c = c + 64u) {
        ln_out[base + c] = (ln_in[base + c] - mean) * inv * ln_params[c] + ln_params[LN_C + c];
    }
}

// ---- tiled matrix multiply ------------------------------------------------
//
// C[M][N] = A[M][K] * B[K][N], all row-major. Each 16x16 workgroup computes a
// 16x16 tile of C, staging a 16x16 tile of A and of B in workgroup memory per
// step along K: 2 tiles * 256 floats * 4 bytes = 2 KiB of the 16 KiB baseline.
// Tiles that run off the matrices load zeros, so every invocation reaches every
// barrier and M, K and N need not be multiples of 16.

override MM_M: u32 = 1u;
override MM_K: u32 = 1u;
override MM_N: u32 = 1u;
const MM_TILE: u32 = 16u;

@group(0) @binding(0) var<storage, read_write> mm_a: array<f32>;
@group(0) @binding(1) var<storage, read_write> mm_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> mm_c: array<f32>;

var<workgroup> mm_tile_a: array<f32, 256>;
var<workgroup> mm_tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn matmul(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.y * MM_TILE + lid.y;
    let col = wid.x * MM_TILE + lid.x;
    var acc = 0.0;
    let steps = (MM_K + MM_TILE - 1u) / MM_TILE;
    for (var t = 0u; t < steps; t = t + 1u) {
        let ka = t * MM_TILE + lid.x;
        let kb = t * MM_TILE + lid.y;
        var va = 0.0;
        if (row < MM_M && ka < MM_K) {
            va = mm_a[row * MM_K + ka];
        }
        var vb = 0.0;
        if (kb < MM_K && col < MM_N) {
            vb = mm_b[kb * MM_N + col];
        }
        mm_tile_a[lid.y * MM_TILE + lid.x] = va;
        mm_tile_b[lid.y * MM_TILE + lid.x] = vb;
        workgroupBarrier();
        for (var k = 0u; k < MM_TILE; k = k + 1u) {
            acc = acc + mm_tile_a[lid.y * MM_TILE + k] * mm_tile_b[k * MM_TILE + lid.x];
        }
        workgroupBarrier();
    }
    if (row < MM_M && col < MM_N) {
        mm_c[row * MM_N + col] = acc;
    }
}
