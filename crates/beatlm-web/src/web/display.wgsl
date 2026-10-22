// The depth model's render pass in the browser: the source image on the left
// and the depth on the right, each a square read straight from the slot's
// tensors in the memory plan, over the whole canvas.
//
// Every pixel first runs the synthetic render's loop (`render_fs` in
// beatlm-bench's shaders.wgsl, the same iterations), so the pass costs what
// the bench's render pass costs beside the depth model, and the frame budget
// is the one the native default was measured under. The loop's result feeds
// the colour at a weight no 8-bit target shows, so it cannot be removed.

struct Params {
    // The image processor's channel means and deviations, to undo the input's
    // normalisation; the fourth lane is unused.
    mean: vec4<f32>,
    deviation: vec4<f32>,
    side: u32,
    render_iters: u32,
    // 1 when the depth binding holds a delivered result, 0 before the first.
    shown: u32,
    pad0: u32,
    depth_low: f32,
    depth_high: f32,
    pad1: f32,
    pad2: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
// The slot's input, NHWC [side][side][3], normalised.
@group(0) @binding(1) var<storage, read> image: array<f32>;
// The slot's output, [side][side], relative depth.
@group(0) @binding(2) var<storage, read> depth: array<f32>;

// The canvas is 1280 x 720; each picture is a square of 600 pixels, centred in
// its half.
const SQUARE: f32 = 600.0;
const TOP: f32 = 60.0;
const LEFT_X: f32 = 20.0;
const RIGHT_X: f32 = 660.0;

// A full-screen triangle: three vertices cover the whole canvas.
fn triangle(vi: u32) -> vec4<f32> {
    let x = f32((vi << 1u) & 2u);
    let y = f32(vi & 2u);
    return vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
}

@vertex
fn display_vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    return triangle(vi);
}

// `display_vs_after`, for `render_order=after`, is `display_vs` made a reader
// of the first word of every arena of the plan, the buffers the slices write,
// so the pass runs after the frame's slices. It is appended to this text by
// beatlm-bench's `workload::after_vertex`, which binds one per arena and
// generates the synthetic render's `render_vs_after` the same way; its
// `workload.rs` says what that guarantees.

// The pixel of the model's image under canvas position `p` in the square whose
// left edge is `x0`, or -1 outside it.
fn texel(p: vec2<f32>, x0: f32) -> i32 {
    let u = (p - vec2<f32>(x0, TOP)) / SQUARE;
    if (any(u < vec2<f32>(0.0)) || any(u >= vec2<f32>(1.0))) {
        return -1;
    }
    let side = params.side;
    let x = min(u32(u.x * f32(side)), side - 1u);
    let y = min(u32(u.y * f32(side)), side - 1u);
    return i32(y * side + x);
}

@fragment
fn display_fs(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
    var a = p.x * 0.0013 + p.y * 0.0007;
    for (var i = 0u; i < params.render_iters; i = i + 1u) {
        a = fract(sin(a) * 43758.5453);
    }
    var colour = vec3<f32>(0.07);
    let source = texel(p.xy, LEFT_X);
    if (source >= 0) {
        let at = u32(source) * 3u;
        let rgb = vec3<f32>(image[at], image[at + 1u], image[at + 2u]);
        colour = clamp(rgb * params.deviation.xyz + params.mean.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
    }
    let result = texel(p.xy, RIGHT_X);
    if (result >= 0) {
        if (params.shown == 1u) {
            let span = max(params.depth_high - params.depth_low, 1e-6);
            let t = clamp((depth[u32(result)] - params.depth_low) / span, 0.0, 1.0);
            colour = vec3<f32>(t);
        } else {
            colour = vec3<f32>(0.0);
        }
    }
    return vec4<f32>(colour + vec3<f32>(a * 1e-6), 1.0);
}
