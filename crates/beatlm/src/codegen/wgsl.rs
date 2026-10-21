//! The templates: one per kind of main operation, sharing a header, the binding
//! declarations, the part's range and the epilogue hook.
//!
//! Every template computes one value per output element and stores it as
//! `out[i] = epilogue(value, i)`; [`epilogue`] writes that function from the
//! spec's steps, so the templates know nothing about fusion. The convolution
//! template reads its input through `prologue(value)` the same way
//! ([`prologue`]). Layouts are the ones [`beatlm_core::kernel`] documents.
//!
//! Every template also walks only its part's range of the output, read from
//! the constants [`part_constants`] names, so the text is the whole kernel's
//! and one shader module serves every part.
//!
//! Four kernels are one tiled matrix multiply ([`tiled`]) that differ only in
//! where an element of each operand is read and where a result is stored: the
//! matrix multiply itself, the patch embedding (the left operand gathered from
//! the image's patches), and the attention scores and the weighted sum of the
//! values (one product per head, the head being the grid's `z`). Each writes
//! those three accesses as WGSL functions, so the tiling, the barriers and the
//! bounds handling are written once.

use std::fmt::Write as _;

use beatlm_core::kernel::{
    AttentionScores, AttentionValues, Conv2d3x3, ConvTranspose2d, Epilogue, GELU_ERF_A, GELU_ERF_P,
    KernelSpec, LayerNorm, MainOp, MatMul, PatchEmbed, PrependToken, Prologue, Requirements,
    ResizeBilinear, Softmax, TokensToMap, Variant,
};
use beatlm_core::memory::TensorId;

use super::ENTRY_POINT;

/// First unit of the part: an output element for the convolution, the map and
/// the token copies, a row for the layer norm, the softmax and the tiled
/// kernels.
const PART_START: &str = "PART_START";
/// One past the last unit of the part, in the unit of [`PART_START`].
const PART_END: &str = "PART_END";
/// Width of the part's grid of workgroups, for the templates that number work
/// over rows of workgroups: the `x` of the part's dispatch.
const PART_GRID_ROW: &str = "PART_GRID_ROW";

/// The whole text of `variant` of the kernel `spec` is a part of. Every part of
/// a kernel has the same text.
pub(super) fn text(spec: &KernelSpec, variant: Variant) -> String {
    let spec = &spec.whole();
    let req = spec.requirements(variant);
    let names = Names {
        bindings: spec.bindings(),
    };
    let mut out = String::new();
    header(&mut out, spec, variant, &req, &names);
    overrides(&mut out, spec, variant);
    epilogue(&mut out, spec.epilogue(), &names);
    let y = names.of(spec.output());
    debug_assert!(
        matches!(spec.main(), MainOp::Conv2d3x3(_)) || spec.prologue().is_empty(),
        "only the convolution template applies a prologue: {}",
        spec.name()
    );
    match *spec.main() {
        MainOp::Conv2d3x3(c) => {
            prologue(&mut out, spec.prologue());
            conv(&mut out, &c, &req, &names, &y);
        }
        MainOp::ConvTranspose2d(c) => conv_transpose(&mut out, &c, &req, &names, &y),
        MainOp::ResizeBilinear(r) => resize_bilinear(&mut out, &r, &req, &names, &y),
        MainOp::LayerNorm(l) => layer_norm(&mut out, &l, &req, &names, &y),
        MainOp::MatMul(mm) => tiled(&mut out, &matmul(&mm, &names), &req, &y),
        MainOp::PatchEmbed(p) => tiled(&mut out, &patch_embed(&p, &names), &req, &y),
        MainOp::AttentionScores(s) => tiled(&mut out, &attention_scores(&s, &names), &req, &y),
        MainOp::AttentionValues(v) => tiled(&mut out, &attention_values(&v, &names), &req, &y),
        MainOp::Softmax(s) => softmax(&mut out, &s, &req, &names, &y),
        MainOp::PrependToken(t) => prepend_token(&mut out, &t, &req, &names, &y),
        MainOp::TokensToMap(t) => tokens_to_map(&mut out, &t, &req, &names, &y),
        MainOp::Map { input, .. } => map(&mut out, &names.of(input), &req, &y),
    }
    out
}

/// Whether the template walks its part in output elements, numbering one
/// invocation per element over rows of workgroups.
const fn by_element(main: &MainOp) -> bool {
    matches!(
        main,
        MainOp::Conv2d3x3(_)
            | MainOp::ConvTranspose2d(_)
            | MainOp::ResizeBilinear(_)
            | MainOp::PrependToken(_)
            | MainOp::TokensToMap(_)
            | MainOp::Map { .. }
    )
}

/// The pipeline-overridable constants of `variant` of `spec`'s template, with
/// the values that make a dispatch compute `spec`'s part, in declaration order.
///
/// Values are `u32`: the core bounds every output by `MAX_ELEMENTS`, below
/// `u32::MAX`, and a dispatch's grid by the workgroups per dimension.
pub(super) fn part_constants(spec: &KernelSpec, variant: Variant) -> Vec<(&'static str, u32)> {
    let narrow = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
    let req = spec.requirements(variant);
    let rows = spec.part().range();
    match spec.main() {
        main if by_element(main) => {
            let elements = spec.output_range();
            vec![
                (PART_START, narrow(elements.start)),
                (PART_END, narrow(elements.end)),
                (PART_GRID_ROW, req.workgroups[0]),
            ]
        }
        MainOp::LayerNorm(_) | MainOp::Softmax(_) => vec![
            (PART_START, rows.start),
            (PART_END, rows.end),
            (PART_GRID_ROW, req.workgroups[0]),
        ],
        _ => vec![(PART_START, rows.start), (PART_END, rows.end)],
    }
}

/// The declarations of the part's range, defaulting to the whole kernel, so
/// the text alone is the unsplit kernel and a pipeline that sets the
/// constants runs one part of it.
fn overrides(out: &mut String, whole: &KernelSpec, variant: Variant) {
    let unit = match whole.main() {
        main if by_element(main) => "output elements",
        MainOp::AttentionScores(_) | MainOp::AttentionValues(_) => "query rows",
        MainOp::PatchEmbed(_) => "patches",
        _ => "rows",
    };
    let _ = writeln!(
        out,
        "// The part of the output one dispatch computes, in {unit}, set per part as\n\
         // pipeline-overridable constants. The defaults are the whole kernel."
    );
    for (name, value) in part_constants(whole, variant) {
        let _ = writeln!(out, "override {name}: u32 = {value}u;");
    }
    let _ = writeln!(out);
}

/// Binding names: binding `i` is `t{i}`, in the order of
/// [`KernelSpec::bindings`].
struct Names {
    bindings: Vec<TensorId>,
}

impl Names {
    /// The name of `id`'s binding. A tensor the spec does not bind gets a name
    /// no binding has, so the text fails validation instead of reading the
    /// wrong tensor.
    fn of(&self, id: TensorId) -> String {
        match self.bindings.iter().position(|&b| b == id) {
            Some(i) => format!("t{i}"),
            None => format!("unbound_tensor_{}", id.get()),
        }
    }
}

/// An `f32` constant written exactly, as its bit pattern, with the decimal
/// value in a comment.
fn exact(value: f32) -> String {
    format!("bitcast<f32>(0x{:08x}u) /* {value:e} */", value.to_bits())
}

/// What the kernel is, and one binding declaration per tensor with what the
/// kernel uses it for.
fn header(
    out: &mut String,
    spec: &KernelSpec,
    variant: Variant,
    req: &Requirements,
    names: &Names,
) {
    let mut roles: Vec<(TensorId, String)> = Vec::new();
    let mut role = |id: TensorId, what: &str| roles.push((id, what.to_owned()));
    match *spec.main() {
        MainOp::Conv2d3x3(c) => {
            role(c.input, "convolution input");
            role(c.weights, "convolution weights");
            if let Some(b) = c.bias {
                role(b, "convolution bias");
            }
        }
        MainOp::ConvTranspose2d(c) => {
            role(c.input, "transposed convolution input");
            role(c.weights, "transposed convolution weights");
            if let Some(b) = c.bias {
                role(b, "transposed convolution bias");
            }
        }
        MainOp::ResizeBilinear(r) => role(r.input, "map to resample"),
        MainOp::LayerNorm(l) => {
            role(l.input, "layer norm input");
            role(l.gamma, "layer norm gamma");
            role(l.beta, "layer norm beta");
        }
        MainOp::MatMul(mm) => {
            role(mm.a, "matmul left");
            role(mm.b, "matmul right");
            if let Some(b) = mm.bias {
                role(b, "matmul bias");
            }
        }
        MainOp::PatchEmbed(p) => {
            role(p.input, "image");
            role(p.weights, "patch embedding weights");
            if let Some(b) = p.bias {
                role(b, "patch embedding bias");
            }
        }
        MainOp::PrependToken(t) => {
            role(t.token, "token put first");
            role(t.tokens, "sequence");
        }
        MainOp::TokensToMap(t) => role(t.input, "sequence, first token dropped"),
        MainOp::AttentionScores(s) => role(s.qkv, "packed queries, keys and values"),
        MainOp::Softmax(s) => role(s.input, "softmax input"),
        MainOp::AttentionValues(v) => {
            role(v.probs, "attention weights");
            role(v.qkv, "packed queries, keys and values");
        }
        MainOp::Map { input, .. } => role(input, "operand"),
    }
    for (step, e) in spec.epilogue().iter().enumerate() {
        match e {
            Epilogue::Add(id) => role(*id, &format!("addend of epilogue step {step}")),
            Epilogue::ChannelScale { scale, .. } => {
                role(
                    *scale,
                    &format!("per-channel multipliers of epilogue step {step}"),
                );
            }
            Epilogue::Relu | Epilogue::Gelu | Epilogue::GeluErf | Epilogue::Scale(_) => {}
        }
    }
    role(spec.output(), "output");

    let _ = writeln!(
        out,
        "// beatlm kernel {} ({variant:?} variant).",
        spec.name()
    );
    let _ = writeln!(
        out,
        "// Workgroup size {:?}, {} bytes of workgroup memory; the whole kernel is {:?} workgroups.",
        req.workgroup_size, req.workgroup_storage_bytes, req.workgroups
    );
    let _ = writeln!(out, "//");
    let _ = writeln!(
        out,
        "// Every storage binding is read_write, inputs included: tensors that share an"
    );
    let _ = writeln!(
        out,
        "// arena share a buffer, and wgpu checks usage per buffer, not per byte range."
    );
    let _ = writeln!(out);
    for (i, &id) in names.bindings.iter().enumerate() {
        let uses: Vec<&str> = roles
            .iter()
            .filter(|(t, _)| *t == id)
            .map(|(_, what)| what.as_str())
            .collect();
        let _ = writeln!(
            out,
            "// Tensor {}: {}.\n@group(0) @binding({i}) var<storage, read_write> {}: array<f32>;",
            id.get(),
            uses.join(", "),
            names.of(id)
        );
    }
    let _ = writeln!(out);
}

/// The hook every template stores through: the spec's epilogue steps, in
/// order, on the value of element `index`.
fn epilogue(out: &mut String, steps: &[Epilogue], names: &Names) {
    if steps.contains(&Epilogue::Gelu) {
        out.push_str(
            "// GELU, tanh approximation: 0.5 x (1 + tanh(sqrt(2 / pi) (x + 0.044715 x^3))).\n\
             // tanh is +-1 to f32 precision beyond +-15, so the clamp changes no result\n\
             // and keeps the argument in a range every implementation of tanh handles.\n\
             fn gelu_tanh(x: f32) -> f32 {\n\
             \x20   let u = 0.7978845608028654 * (x + 0.044715 * x * x * x);\n\
             \x20   return 0.5 * x * (1.0 + tanh(clamp(u, -15.0, 15.0)));\n\
             }\n\n",
        );
    }
    if steps.contains(&Epilogue::GeluErf) {
        gelu_erf(out);
    }
    out.push_str("fn epilogue(value: f32, index: u32) -> f32 {\n    var v = value;\n");
    for step in steps {
        let _ = match *step {
            Epilogue::Add(id) => writeln!(out, "    v = v + {}[index];", names.of(id)),
            Epilogue::Relu => writeln!(out, "    v = max(v, 0.0);"),
            Epilogue::Gelu => writeln!(out, "    v = gelu_tanh(v);"),
            Epilogue::GeluErf => writeln!(out, "    v = gelu_erf(v);"),
            Epilogue::Scale(factor) => writeln!(out, "    v = v * {};", exact(factor)),
            Epilogue::ChannelScale { scale, channels } => {
                writeln!(out, "    v = v * {}[index % {channels}u];", names.of(scale))
            }
        };
    }
    out.push_str("    return v;\n}\n\n");
}

/// GELU through erf, with the approximation and constants the core states at
/// [`GELU_ERF_P`]: `x * Phi(x)`, `Phi` from `erfc(|x| / sqrt 2) / 2`, so the
/// negative side is a product and loses nothing to cancellation.
fn gelu_erf(out: &mut String) {
    let [a1, a2, a3, a4, a5] = GELU_ERF_A.map(exact);
    let _ = write!(
        out,
        "\
// GELU as defined, x * Phi(x) with Phi(x) = (1 + erf(x / sqrt 2)) / 2. WGSL has no
// erf: erfc(z) for z >= 0 is Abramowitz and Stegun 7.1.26,
// (a1 t + a2 t^2 + a3 t^3 + a4 t^4 + a5 t^5) exp(-z^2) with t = 1 / (1 + p z),
// whose absolute error is at most 1.5e-7. q = erfc(|x| / sqrt 2) / 2 is Phi(x)
// for negative x and 1 - Phi(x) otherwise, so no side subtracts nearly equal
// numbers. exp(-z^2) underflows to zero for large |x|, which is the limit.
fn gelu_erf(x: f32) -> f32 {{
    let z = abs(x) * 0.70710678118654752;
    let t = 1.0 / (1.0 + {p} * z);
    let poly = t * ({a1} + t * ({a2} + t * ({a3} + t * ({a4} + t * {a5}))));
    let q = 0.5 * poly * exp(-z * z);
    return x * select(q, 1.0 - q, x >= 0.0);
}}

",
        p = exact(GELU_ERF_P),
    );
}

/// The entry point's attribute line for the variant's workgroup size.
fn entry(req: &Requirements) -> String {
    let [x, y, z] = req.workgroup_size;
    format!("@compute @workgroup_size({x}, {y}, {z})\nfn {ENTRY_POINT}(")
}

/// The hook the convolution template loads every input value through: the
/// spec's prologue steps, in order. An empty prologue returns the value as it
/// is, so a convolution without one has the same function and the same text
/// shape.
fn prologue(out: &mut String, steps: &[Prologue]) {
    out.push_str(
        "// The steps applied to every input value as it is loaded, before it is multiplied.\n\
         fn prologue(value: f32) -> f32 {\n    var v = value;\n",
    );
    for step in steps {
        match step {
            Prologue::Relu => out.push_str("    v = max(v, 0.0);\n"),
        }
    }
    out.push_str("    return v;\n}\n\n");
}

/// One invocation per output element of the part, numbered in memory order
/// (channel fastest) from the part's first element over a grid of rows
/// `PART_GRID_ROW` workgroups wide. Output pixel `(y, x)` is centred on input
/// pixel `(y * STRIDE, x * STRIDE)`; every input value goes through
/// `prologue`, which [`prologue`] writes.
fn conv(out: &mut String, c: &Conv2d3x3, req: &Requirements, names: &Names, y: &str) {
    let bias = c.bias.map_or(String::new(), |b| {
        format!("    acc = acc + {}[co];\n", names.of(b))
    });
    let _ = write!(
        out,
        "\
const H: u32 = {h}u;
const W: u32 = {w}u;
const OW: u32 = {ow}u;
const STRIDE: u32 = {stride}u;
const CIN: u32 = {cin}u;
const COUT: u32 = {cout}u;
const GROUP: u32 = {group}u;

// 3x3 convolution, zero padding of 1, stride STRIDE: NHWC input of H x W, output
// rows OW wide, HWIO weights.
{entry}
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {{
    let element = {PART_START} + (wid.y * {PART_GRID_ROW} + wid.x) * GROUP + lane;
    if (element >= {PART_END}) {{
        return;
    }}
    let co = element % COUT;
    let pixel = element / COUT;
    let py = i32((pixel / OW) * STRIDE);
    let px = i32((pixel % OW) * STRIDE);
    var acc = 0.0;
    for (var ky = 0; ky < 3; ky = ky + 1) {{
        let iy = py + ky - 1;
        if (iy < 0 || iy >= i32(H)) {{
            continue;
        }}
        for (var kx = 0; kx < 3; kx = kx + 1) {{
            let ix = px + kx - 1;
            if (ix < 0 || ix >= i32(W)) {{
                continue;
            }}
            let src = (u32(iy) * W + u32(ix)) * CIN;
            let tap = u32(ky * 3 + kx) * CIN * COUT + co;
            for (var ci = 0u; ci < CIN; ci = ci + 1u) {{
                acc = acc + prologue({x}[src + ci]) * {wt}[tap + ci * COUT];
            }}
        }}
    }}
{bias}    {y}[element] = epilogue(acc, element);
}}
",
        h = c.height,
        w = c.width,
        ow = c.out_width(),
        stride = c.stride,
        cin = c.cin,
        cout = c.cout,
        group = req.workgroup_size[0],
        entry = entry(req),
        x = names.of(c.input),
        wt = names.of(c.weights),
    );
}

/// One invocation per output element of the part, numbered as for [`conv`]:
/// output pixel `(oy, ox)` lies in the block input pixel
/// `(oy / F, ox / F)` writes, at tap `(oy % F, ox % F)` of the kernel, so the
/// element is that pixel's channels times that tap's weights, plus the bias.
fn conv_transpose(
    out: &mut String,
    c: &ConvTranspose2d,
    req: &Requirements,
    names: &Names,
    y: &str,
) {
    per_element(
        out,
        &[
            ("W", c.width),
            ("F", c.factor),
            ("OW", c.width * c.factor),
            ("CIN", c.cin),
            ("COUT", c.cout),
        ],
        "Transposed convolution, kernel and stride F, no padding: each NHWC input pixel\n\
         // writes its own F x F block of the output, through HWIO weights [F][F][CIN][COUT].",
        req,
    );
    let bias = c.bias.map_or(String::new(), |b| {
        format!("    acc = acc + {}[co];\n", names.of(b))
    });
    let _ = write!(
        out,
        "\
    let co = element % COUT;
    let pixel = element / COUT;
    let oy = pixel / OW;
    let ox = pixel % OW;
    let src = ((oy / F) * W + ox / F) * CIN;
    let tap = ((oy % F) * F + ox % F) * CIN * COUT + co;
    var acc = 0.0;
    for (var ci = 0u; ci < CIN; ci = ci + 1u) {{
        acc = acc + {x}[src + ci] * {wt}[tap + ci * COUT];
    }}
{bias}    {y}[element] = epilogue(acc, element);
}}
",
        x = names.of(c.input),
        wt = names.of(c.weights),
    );
}

/// One invocation per output element of the part, numbered as for [`conv`].
/// The sampling ratios are the core's ([`ResizeBilinear::ratios`]), written
/// exactly, and the source position is `ratio * index` in `f32`, as PyTorch
/// computes it; the two input rows and columns around it are the one it
/// truncates to and the next, clamped to the last, weighted by the fraction.
fn resize_bilinear(
    out: &mut String,
    r: &ResizeBilinear,
    req: &Requirements,
    names: &Names,
    y: &str,
) {
    per_element(
        out,
        &[
            ("IH", r.in_height),
            ("IW", r.in_width),
            ("OW", r.out_width),
            ("C", r.channels),
        ],
        "Bilinear resampling with the corners aligned, NHWC: output row oy samples input\n\
         // row oy * (IH - 1) / (OH - 1), columns likewise.",
        req,
    );
    let (ratio_y, ratio_x) = r.ratios();
    let _ = write!(
        out,
        "\
    let c = element % C;
    let pixel = element / C;
    let sy = {ratio_y} * f32(pixel / OW);
    let sx = {ratio_x} * f32(pixel % OW);
    let y0 = min(u32(sy), IH - 1u);
    let x0 = min(u32(sx), IW - 1u);
    let y1 = min(y0 + 1u, IH - 1u);
    let x1 = min(x0 + 1u, IW - 1u);
    let ly = clamp(sy - f32(y0), 0.0, 1.0);
    let lx = clamp(sx - f32(x0), 0.0, 1.0);
    let top = (1.0 - lx) * {x}[(y0 * IW + x0) * C + c] + lx * {x}[(y0 * IW + x1) * C + c];
    let bottom = (1.0 - lx) * {x}[(y1 * IW + x0) * C + c] + lx * {x}[(y1 * IW + x1) * C + c];
    {y}[element] = epilogue((1.0 - ly) * top + ly * bottom, element);
}}
",
        ratio_y = exact(ratio_y),
        ratio_x = exact(ratio_x),
        x = names.of(r.input),
    );
}

/// One workgroup per row of the part: its invocations stride over the
/// channels and reduce through workgroup memory. Rows are numbered from the
/// part's first row over a grid of rows `PART_GRID_ROW` workgroups wide.
fn layer_norm(out: &mut String, l: &LayerNorm, req: &Requirements, names: &Names, y: &str) {
    let lanes = req.workgroup_size[0];
    let _ = write!(
        out,
        "\
const C: u32 = {channels}u;
const LANES: u32 = {lanes}u;

var<workgroup> ln_scratch: array<f32, {lanes}>;

// The sum over the workgroup of what each invocation left in `ln_scratch`.
fn ln_reduce(lane: u32) -> f32 {{
    workgroupBarrier();
    for (var stride = LANES / 2u; stride > 0u; stride = stride >> 1u) {{
        if (lane < stride) {{
            ln_scratch[lane] = ln_scratch[lane] + ln_scratch[lane + stride];
        }}
        workgroupBarrier();
    }}
    let total = ln_scratch[0];
    // Everyone has read the total before the scratch is written again.
    workgroupBarrier();
    return total;
}}

// Layer norm over the last dimension: (x - mean) / sqrt(var + eps) * gamma + beta.
{entry}
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {{
    let row = {PART_START} + wid.y * {PART_GRID_ROW} + wid.x;
    // Uniform across the workgroup, so the barriers below are safe.
    if (row >= {PART_END}) {{
        return;
    }}
    let base = row * C;
    let eps = {eps};

    var sum = 0.0;
    for (var c = lane; c < C; c = c + LANES) {{
        sum = sum + {x}[base + c];
    }}
    ln_scratch[lane] = sum;
    let mean = ln_reduce(lane) / f32(C);

    var sq = 0.0;
    for (var c = lane; c < C; c = c + LANES) {{
        let d = {x}[base + c] - mean;
        sq = sq + d * d;
    }}
    ln_scratch[lane] = sq;
    let inv = inverseSqrt(ln_reduce(lane) / f32(C) + eps);

    for (var c = lane; c < C; c = c + LANES) {{
        let i = base + c;
        {y}[i] = epilogue(({x}[i] - mean) * inv * {gamma}[c] + {beta}[c], i);
    }}
}}
",
        channels = l.channels,
        entry = entry(req),
        eps = exact(l.epsilon),
        x = names.of(l.input),
        gamma = names.of(l.gamma),
        beta = names.of(l.beta),
    );
}

/// One workgroup per row of the part, as for the layer norm: the row's maximum
/// and then the sum of `exp(x - max)` are reduced through workgroup memory, and
/// every element is stored as its `exp(x - max)` over that sum.
fn softmax(out: &mut String, s: &Softmax, req: &Requirements, names: &Names, y: &str) {
    let lanes = req.workgroup_size[0];
    let _ = write!(
        out,
        "\
const C: u32 = {cols}u;
const LANES: u32 = {lanes}u;

var<workgroup> sm_scratch: array<f32, {lanes}>;

// The maximum over the workgroup of what each invocation left in `sm_scratch`.
fn sm_max(lane: u32) -> f32 {{
    workgroupBarrier();
    for (var stride = LANES / 2u; stride > 0u; stride = stride >> 1u) {{
        if (lane < stride) {{
            sm_scratch[lane] = max(sm_scratch[lane], sm_scratch[lane + stride]);
        }}
        workgroupBarrier();
    }}
    let total = sm_scratch[0];
    // Everyone has read the result before the scratch is written again.
    workgroupBarrier();
    return total;
}}

// The sum over the workgroup of what each invocation left in `sm_scratch`.
fn sm_sum(lane: u32) -> f32 {{
    workgroupBarrier();
    for (var stride = LANES / 2u; stride > 0u; stride = stride >> 1u) {{
        if (lane < stride) {{
            sm_scratch[lane] = sm_scratch[lane] + sm_scratch[lane + stride];
        }}
        workgroupBarrier();
    }}
    let total = sm_scratch[0];
    workgroupBarrier();
    return total;
}}

// Softmax over the last dimension, the row's maximum subtracted first, so every
// exp is at most one and none overflows.
{entry}
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {{
    let row = {PART_START} + wid.y * {PART_GRID_ROW} + wid.x;
    // Uniform across the workgroup, so the barriers below are safe.
    if (row >= {PART_END}) {{
        return;
    }}
    let base = row * C;

    // The lowest finite f32: a lane with no element of the row leaves it,
    // and it is below every element of a row of finite numbers.
    var m = {lowest};
    for (var c = lane; c < C; c = c + LANES) {{
        m = max(m, {x}[base + c]);
    }}
    sm_scratch[lane] = m;
    let row_max = sm_max(lane);

    var sum = 0.0;
    for (var c = lane; c < C; c = c + LANES) {{
        sum = sum + exp({x}[base + c] - row_max);
    }}
    sm_scratch[lane] = sum;
    let inv = 1.0 / sm_sum(lane);

    for (var c = lane; c < C; c = c + LANES) {{
        let i = base + c;
        {y}[i] = epilogue(exp({x}[i] - row_max) * inv, i);
    }}
}}
",
        cols = s.cols,
        entry = entry(req),
        lowest = exact(f32::MIN),
        x = names.of(s.input),
    );
}

/// What one tiled kernel reads and where it stores, for [`tiled`].
///
/// `load_a` and `load_b` are WGSL function bodies returning an operand
/// element; `index` is one returning the output index of a result. Each is
/// called only with indices inside the operands and the output.
struct Tiled {
    /// What the kernel computes, for the comment above the entry point.
    what: &'static str,
    /// Extent of the reduction: columns of the left operand, rows of the right.
    k: u32,
    /// Extent of the output's columns within one layer.
    n: u32,
    /// `const` declarations the functions use, one per line.
    consts: Vec<(&'static str, u32)>,
    /// `fn load_a(row: u32, ka: u32, layer: u32) -> f32`: the left operand at
    /// output row `row`, reduction index `ka`, in layer `layer`.
    load_a: String,
    /// `fn load_b(kb: u32, col: u32, layer: u32) -> f32`: the right operand at
    /// reduction index `kb`, output column `col`, in layer `layer`.
    load_b: String,
    /// `fn out_index(row: u32, col: u32, layer: u32) -> u32`.
    index: String,
    /// The bias binding added per output column, when there is one.
    bias: Option<String>,
}

/// One workgroup per output tile of the part, rows counted from the part's
/// first row, columns along `x` and layers (heads) along `z`, staging a tile of
/// each operand in workgroup memory per step along `k`. Tiles that run off the
/// part's rows or the operands load zeros, so every invocation reaches every
/// barrier, and store nothing.
fn tiled(out: &mut String, t: &Tiled, req: &Requirements, y: &str) {
    let tile = req.workgroup_size[0];
    let bias = t.bias.as_ref().map_or(String::new(), |b| {
        format!("        acc = acc + {b}[col];\n")
    });
    let consts: String = t
        .consts
        .iter()
        .map(|(name, value)| format!("const {name}: u32 = {value}u;\n"))
        .collect();
    let _ = write!(
        out,
        "\
const K: u32 = {k}u;
const N: u32 = {n}u;
const TILE: u32 = {tile}u;
{consts}
var<workgroup> mm_tile_a: array<f32, {cells}>;
var<workgroup> mm_tile_b: array<f32, {cells}>;

fn load_a(row: u32, ka: u32, layer: u32) -> f32 {{
{load_a}
}}

fn load_b(kb: u32, col: u32, layer: u32) -> f32 {{
{load_b}
}}

fn out_index(row: u32, col: u32, layer: u32) -> u32 {{
{index}
}}

// {what}
{entry}
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {{
    let row = {PART_START} + wid.y * TILE + lid.y;
    let col = wid.x * TILE + lid.x;
    let layer = wid.z;
    var acc = 0.0;
    let steps = (K + TILE - 1u) / TILE;
    for (var t = 0u; t < steps; t = t + 1u) {{
        let ka = t * TILE + lid.x;
        let kb = t * TILE + lid.y;
        var va = 0.0;
        if (row < {PART_END} && ka < K) {{
            va = load_a(row, ka, layer);
        }}
        var vb = 0.0;
        if (kb < K && col < N) {{
            vb = load_b(kb, col, layer);
        }}
        mm_tile_a[lid.y * TILE + lid.x] = va;
        mm_tile_b[lid.y * TILE + lid.x] = vb;
        workgroupBarrier();
        for (var i = 0u; i < TILE; i = i + 1u) {{
            acc = acc + mm_tile_a[lid.y * TILE + i] * mm_tile_b[i * TILE + lid.x];
        }}
        workgroupBarrier();
    }}
    if (row < {PART_END} && col < N) {{
{bias}        let i = out_index(row, col, layer);
        {y}[i] = epilogue(acc, i);
    }}
}}
",
        k = t.k,
        n = t.n,
        cells = tile * tile,
        load_a = t.load_a,
        load_b = t.load_b,
        index = t.index,
        what = t.what,
        entry = entry(req),
    );
}

/// `C[M][N] = A[M][K] * B[K][N]`, row-major.
fn matmul(mm: &MatMul, names: &Names) -> Tiled {
    Tiled {
        what: "C[M][N] = A[M][K] * B[K][N], row-major, tiled.",
        k: mm.k,
        n: mm.n,
        consts: Vec::new(),
        load_a: format!("    return {}[row * K + ka];", names.of(mm.a)),
        load_b: format!("    return {}[kb * N + col];", names.of(mm.b)),
        index: "    return row * N + col;".to_owned(),
        bias: mm.bias.map(|b| names.of(b)),
    }
}

/// The patches of the image times the HWIO weights: row `row` is patch
/// `(row / GW, row % GW)`, and reduction index `ka` is pixel `(ky, kx)` of the
/// patch and channel `ci`, in the weights' `[P][P][CIN]` order.
fn patch_embed(p: &PatchEmbed, names: &Names) -> Tiled {
    Tiled {
        what: "Patch embedding: [patches][P * P * CIN] gathered from the NHWC image, times\n// HWIO weights [P * P * CIN][COUT], tiled.",
        k: p.patch * p.patch * p.cin,
        n: p.cout,
        consts: vec![
            ("GW", p.grid_w),
            ("P", p.patch),
            ("CIN", p.cin),
            ("IMAGE_W", p.grid_w * p.patch),
        ],
        load_a: format!(
            "    let gy = row / GW;\n    let gx = row % GW;\n    let ky = ka / (P * CIN);\n    \
             let kx = (ka / CIN) % P;\n    let ci = ka % CIN;\n    \
             return {}[((gy * P + ky) * IMAGE_W + gx * P + kx) * CIN + ci];",
            names.of(p.input)
        ),
        load_b: format!("    return {}[kb * N + col];", names.of(p.weights)),
        index: "    return row * N + col;".to_owned(),
        bias: p.bias.map(|b| names.of(b)),
    }
}

/// The constants the two attention products share.
fn attention_consts(tokens: u32, heads: u32, head_dim: u32) -> Vec<(&'static str, u32)> {
    vec![
        ("TOKENS", tokens),
        ("HEADS", heads),
        ("HEAD_DIM", head_dim),
        ("QKV_ROW", 3 * heads * head_dim),
        ("KEYS_AT", heads * head_dim),
        ("VALUES_AT", 2 * heads * head_dim),
    ]
}

/// Per head (the layer), queries `[tokens][head_dim]` times the transposed
/// keys `[head_dim][tokens]`, stored query, head, key.
fn attention_scores(s: &AttentionScores, names: &Names) -> Tiled {
    let qkv = names.of(s.qkv);
    Tiled {
        what: "Attention scores per head (the grid's z): S[i][h][j] = Q[i][h] . K[j][h], tiled.",
        k: s.head_dim,
        n: s.tokens,
        consts: attention_consts(s.tokens, s.heads, s.head_dim),
        load_a: format!("    return {qkv}[row * QKV_ROW + layer * HEAD_DIM + ka];"),
        load_b: format!("    return {qkv}[col * QKV_ROW + KEYS_AT + layer * HEAD_DIM + kb];"),
        index: "    return (row * HEADS + layer) * TOKENS + col;".to_owned(),
        bias: None,
    }
}

/// Per head (the layer), the weights `[tokens][tokens]` of its rows times the
/// values `[tokens][head_dim]`, stored with the heads side by side.
fn attention_values(v: &AttentionValues, names: &Names) -> Tiled {
    Tiled {
        what: "Weighted sum of the values per head (the grid's z): O[i][h] = P[i][h] . V[h], tiled.",
        k: v.tokens,
        n: v.head_dim,
        consts: attention_consts(v.tokens, v.heads, v.head_dim),
        load_a: format!(
            "    return {}[(row * HEADS + layer) * TOKENS + ka];",
            names.of(v.probs)
        ),
        load_b: format!(
            "    return {}[kb * QKV_ROW + VALUES_AT + layer * HEAD_DIM + col];",
            names.of(v.qkv)
        ),
        index: "    return row * KEYS_AT + layer * HEAD_DIM + col;".to_owned(),
        bias: None,
    }
}

/// The opening of a template with one invocation per element of the part,
/// numbered over rows of workgroups: `element` is the output index, and an
/// invocation past the part returns.
fn per_element(out: &mut String, consts: &[(&str, u32)], what: &str, req: &Requirements) {
    for (name, value) in consts {
        let _ = writeln!(out, "const {name}: u32 = {value}u;");
    }
    let _ = write!(
        out,
        "\
const GROUP: u32 = {group}u;

// {what}
{entry}
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {{
    let element = {PART_START} + (wid.y * {PART_GRID_ROW} + wid.x) * GROUP + lane;
    if (element >= {PART_END}) {{
        return;
    }}
",
        group = req.workgroup_size[0],
        entry = entry(req),
    );
}

/// One invocation per output element: the first `C` elements are the token,
/// the rest the sequence's.
fn prepend_token(out: &mut String, t: &PrependToken, req: &Requirements, names: &Names, y: &str) {
    per_element(
        out,
        &[("C", t.channels)],
        "A token in front of a sequence: row 0 is the token, row r + 1 the sequence's row r.",
        req,
    );
    let _ = write!(
        out,
        "\
    var v = 0.0;
    if (element < C) {{
        v = {token}[element];
    }} else {{
        v = {tokens}[element - C];
    }}
    {y}[element] = epilogue(v, element);
}}
",
        token = names.of(t.token),
        tokens = names.of(t.tokens),
    );
}

/// One invocation per output element: element `i` of the map is element
/// `i + C` of the sequence, whose first token is dropped.
fn tokens_to_map(out: &mut String, t: &TokensToMap, req: &Requirements, names: &Names, y: &str) {
    per_element(
        out,
        &[("C", t.channels)],
        "A sequence without its first token, as the NHWC map of the rest.",
        req,
    );
    let _ = write!(
        out,
        "    {y}[element] = epilogue({x}[element + C], element);\n}}\n",
        x = names.of(t.input),
    );
}

/// One invocation per element of the part: the main operation of a standalone
/// element-wise kernel, whose work is all in the epilogue.
fn map(out: &mut String, x: &str, req: &Requirements, y: &str) {
    per_element(out, &[], "An element-wise operation on its own.", req);
    let _ = write!(
        out,
        "    {y}[element] = epilogue({x}[element], element);\n}}\n"
    );
}
