//! The templates: one per kind of main operation, sharing a header, the binding
//! declarations, the part's range and the epilogue hook.
//!
//! Every template stores each output value as `out[i] = epilogue(value, i)`;
//! [`epilogue`] writes that function from the spec's steps, so the templates
//! know nothing about fusion. The 3x3 convolution reads its input through
//! `prologue(value)` the same way ([`prologue`]). Layouts are the ones
//! [`beatlm_core::kernel`] documents.
//!
//! Every template also walks only its part's range of the output, read from
//! the constants [`part_constants`] names, so the text is the whole kernel's
//! and one shader module serves every part.
//!
//! Five kernels are one register-tiled matrix multiply ([`tiled`]) that differ
//! only in where an element of each operand is read and where a result is
//! stored: the matrix multiply itself (which the 1x1 convolution lowers to),
//! the patch embedding (the left operand gathered from the image's patches),
//! the 3x3 convolution (the left operand gathered from the image's 3x3
//! windows, an implicit GEMM), and the attention scores and the weighted sum
//! of the values (one product per head, the head being the grid's `z`). Each
//! writes those three accesses as WGSL functions, so the tiling, the barriers
//! and the bounds handling are written once.
//!
//! The register-tiled template replaced two kernels that each computed one
//! output per invocation: a 16 x 16 tiled matrix multiply for the four
//! products, and a convolution that read its input and weights straight from
//! memory, nine taps times the input channels per output. With real weights at
//! 518 pixels, on an Apple M1 Pro, Metal, it took Depth Anything V2 Small
//! from about 860 ms of GPU time per run to about 205 ms; the core's
//! [`Tile`](beatlm_core::kernel::Tile) has the measurements by kind of kernel,
//! and why its tiles and no others. Neither old kernel was kept: no shape of
//! the model ran faster on it.

use std::fmt::Write as _;

use beatlm_core::kernel::{
    AttentionScores, AttentionValues, Conv2d3x3, ConvTranspose2d, Epilogue, GELU_ERF_A, GELU_ERF_P,
    KernelSpec, LayerNorm, MainOp, MatMul, PatchEmbed, PrependToken, Prologue, Requirements,
    ResizeBilinear, Softmax, Tile, TokensToMap, Variant,
};
use beatlm_core::memory::TensorId;

use super::ENTRY_POINT;

/// First unit of the part: an output element for the transposed convolution,
/// the resampling, the map and the token copies, a row for the layer norm, the
/// softmax and the tiled kernels (an output pixel of the 3x3 convolution).
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
    let tiled_as = |t: Tiled, out: &mut String| match spec.tile() {
        Some(tile) => tiled(out, &t, tile, &req, &y),
        // Every main operation with a `Tiled` has a tile; a text without an
        // entry point fails validation rather than running something else.
        None => out.push_str("// no tile for a tiled kernel\n"),
    };
    match *spec.main() {
        MainOp::Conv2d3x3(c) => {
            prologue(&mut out, spec.prologue());
            tiled_as(conv(&c, &names), &mut out);
        }
        MainOp::ConvTranspose2d(c) => conv_transpose(&mut out, &c, &req, &names, &y),
        MainOp::ResizeBilinear(r) => resize_bilinear(&mut out, &r, &req, &names, &y),
        MainOp::LayerNorm(l) => layer_norm(&mut out, &l, &req, &names, &y),
        MainOp::MatMul(mm) => tiled_as(matmul(&mm, &names), &mut out),
        MainOp::PatchEmbed(p) => tiled_as(patch_embed(&p, &names), &mut out),
        MainOp::AttentionScores(s) => tiled_as(attention_scores(&s, &names), &mut out),
        MainOp::AttentionValues(v) => tiled_as(attention_values(&v, &names), &mut out),
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
        MainOp::ConvTranspose2d(_)
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
        MainOp::Conv2d3x3(_) => "output pixels",
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

/// The 3x3 convolution as a tiled matrix multiply, an implicit GEMM: row
/// `row` of the left operand is output pixel `(row / OW, row % OW)`, and
/// reduction index `ka` is tap `(ky, kx)` and input channel `ci`, in the HWIO
/// weights' `[3][3][CIN]` order, so the weights are the right operand
/// `[9 * CIN][COUT]` as they are stored. The left operand is never written
/// out: each value is read from the input pixel the tap lands on as it is
/// staged, through `prologue` ([`prologue`]), and is zero where the tap lands
/// in the padding. Output pixel `(y, x)` is centred on input pixel
/// `(y * STRIDE, x * STRIDE)`.
fn conv(c: &Conv2d3x3, names: &Names) -> Tiled {
    Tiled {
        what: "3x3 convolution, zero padding of 1, stride STRIDE: the 3x3 windows of the NHWC\n\
               // input of H x W, [output pixels][9 * CIN] with output rows OW wide, times the\n\
               // HWIO weights [9 * CIN][COUT], tiled.",
        k: 9 * c.cin,
        n: c.cout,
        consts: vec![
            ("H", c.height),
            ("W", c.width),
            ("OW", c.out_width()),
            ("STRIDE", c.stride),
            ("CIN", c.cin),
        ],
        load_a: format!(
            "    let tap = ka / CIN;\n    \
             let iy = i32((row / OW) * STRIDE + tap / 3u) - 1;\n    \
             let ix = i32((row % OW) * STRIDE + tap % 3u) - 1;\n    \
             if (iy < 0 || iy >= i32(H) || ix < 0 || ix >= i32(W)) {{\n        \
             return 0.0;\n    }}\n    \
             return prologue({}[(u32(iy) * W + u32(ix)) * CIN + ka % CIN]);",
            names.of(c.input)
        ),
        load_b: format!("    return {}[kb * N + col];", names.of(c.weights)),
        b_along_k: false,
        index: "    return row * N + col;".to_owned(),
        bias: c.bias.map(|b| names.of(b)),
    }
}

/// One invocation per output element of the part, numbered as in
/// [`per_element`]:
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

/// One invocation per output element of the part, numbered as in
/// [`per_element`].
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
    /// Whether neighbouring reduction indices of the right operand are
    /// neighbours in memory (the keys of the attention scores), rather than
    /// neighbouring columns (every other right operand): the invocations
    /// staging it then walk the reduction fastest, so neighbouring invocations
    /// read neighbouring values either way.
    b_along_k: bool,
    /// `fn out_index(row: u32, col: u32, layer: u32) -> u32`.
    index: String,
    /// The bias binding added per output column, when there is one.
    bias: Option<String>,
}

/// One workgroup per output tile of the part (the core's
/// [`Tile`](beatlm_core::kernel::Tile)), rows counted from the part's first
/// row, columns along `x` and layers (heads) along `z`.
///
/// The loop: for every step of `TILE_DEPTH` reduction indices, the workgroup
/// stages the step's `TILE_ROWS x TILE_DEPTH` block of the left operand and
/// `TILE_DEPTH x TILE_COLUMNS` block of the right one in workgroup memory,
/// reduction index outermost, so that a row of either is a run of `vec4`s;
/// then every invocation reads, per reduction index, the `vec4`s of its
/// `ROWS_PER x COLUMNS_PER` block's rows and columns and adds their outer
/// product to accumulators it keeps in registers. Each staged value is read by
/// a whole row or column of invocations, and each value read by an invocation
/// is used four times or more, which is what makes the tile pay: the template
/// it replaced read two values from workgroup memory per multiply-add.
///
/// Staging numbers the block's values so neighbouring invocations load
/// neighbouring values from memory: reduction index fastest for the left
/// operand, whose rows run along the reduction in every kernel; columns or
/// reduction index fastest for the right one, as [`Tiled::b_along_k`] says.
/// The rows of a staged block are padded ([`Tile::left_stride`]) so that the
/// invocations writing one step's values spread over the banks. Values past
/// the part's rows or the operands are staged as zeros, so every invocation
/// reaches every barrier and the extra products add nothing, and outputs past
/// them are not stored.
///
/// Each invocation stages whole `vec4`s: four rows (or columns) at one
/// reduction index, gathered in a function variable and written to workgroup
/// memory at once. The first version wrote each staged value into its
/// component of a workgroup `vec4` through a computed index
/// (`mm_a[at / 4u][at % 4u] = v`); naga accepted it, and on an Apple M1 Pro,
/// Metal, on 2026-10-22, every tiled kernel then computed wrong values, while
/// the same text staging whole `vec4`s matches the references. Do not go back
/// to writing single components of workgroup vectors.
///
/// Every output of an invocation's block is summed over the reduction in
/// index order, as one output per invocation was, so a part computes what the
/// whole kernel does (rule 6 of the core's
/// [module documentation](beatlm_core::kernel#rules)), and the real-weight
/// tests measured the same errors as with the template this replaced.
///
/// [`Tile::left_stride`]: beatlm_core::kernel::Tile::left_stride
fn tiled(out: &mut String, t: &Tiled, tile: Tile, req: &Requirements, y: &str) {
    let lanes = tile.invocations();
    let (per_row, per_col) = (tile.rows_per_invocation, tile.columns_per_invocation);
    debug_assert!(
        per_row % 4 == 0 && per_col % 4 == 0,
        "the template stages and reads vec4s of rows and columns: {tile:?}"
    );
    let consts: String = t
        .consts
        .iter()
        .map(|(name, value)| format!("const {name}: u32 = {value}u;\n"))
        .collect();
    let (b_kk, b_c) = if t.b_along_k {
        ("e % TILE_DEPTH", "(e / TILE_DEPTH) * 4u")
    } else {
        ("e / (TILE_COLUMNS / 4u)", "(e % (TILE_COLUMNS / 4u)) * 4u")
    };
    let bias = t.bias.as_ref().map_or(String::new(), |b| {
        format!("            v = v + {b}[col + c];\n")
    });
    // The accumulators: `acc_i_j` holds row `i` of the invocation's block,
    // columns `4 j` to `4 j + 3`.
    let (rows_v, cols_v) = (per_row / 4, per_col / 4);
    let mut declare = String::new();
    let mut multiply = String::new();
    let mut store = String::new();
    for v in 0..rows_v {
        let _ = writeln!(
            multiply,
            "            let a{v} = mm_a[kk * LEFT_VEC4S + a_at + {v}u];"
        );
    }
    for v in 0..cols_v {
        let _ = writeln!(
            multiply,
            "            let b{v} = mm_b[kk * RIGHT_VEC4S + b_at + {v}u];"
        );
    }
    for i in 0..per_row {
        let lane = ["x", "y", "z", "w"][(i % 4) as usize];
        for j in 0..cols_v {
            let _ = writeln!(declare, "    var acc_{i}_{j} = vec4<f32>(0.0);");
            let _ = writeln!(
                multiply,
                "            acc_{i}_{j} = acc_{i}_{j} + a{}.{lane} * b{j};",
                i / 4
            );
            let _ = writeln!(
                store,
                "    store(row_at + {i}u, col_at + {}u, layer, acc_{i}_{j});",
                4 * j
            );
        }
    }
    let _ = write!(
        out,
        "\
const K: u32 = {k}u;
const N: u32 = {n}u;
const TILE_ROWS: u32 = {rows}u;
const TILE_COLUMNS: u32 = {columns}u;
const TILE_DEPTH: u32 = {depth}u;
const ROWS_PER: u32 = {per_row}u;
const COLUMNS_PER: u32 = {per_col}u;
const LANES: u32 = {lanes}u;
// A staged row of each operand, in values and in vec4s.
const LEFT_STRIDE: u32 = {left}u;
const RIGHT_STRIDE: u32 = {right}u;
const LEFT_VEC4S: u32 = {left_v}u;
const RIGHT_VEC4S: u32 = {right_v}u;
{consts}
// One step of each operand, reduction index outermost: value (r, kk) of the
// left operand at kk * LEFT_STRIDE + r, (kk, c) of the right at
// kk * RIGHT_STRIDE + c.
var<workgroup> mm_a: array<vec4<f32>, {a_cells}>;
var<workgroup> mm_b: array<vec4<f32>, {b_cells}>;

fn load_a(row: u32, ka: u32, layer: u32) -> f32 {{
{load_a}
}}

fn load_b(kb: u32, col: u32, layer: u32) -> f32 {{
{load_b}
}}

fn out_index(row: u32, col: u32, layer: u32) -> u32 {{
{index}
}}

// Stores four neighbouring results of one row, those inside the part and the
// output.
fn store(row: u32, col: u32, layer: u32, values: vec4<f32>) {{
    if (row >= {PART_END}) {{
        return;
    }}
    for (var c = 0u; c < 4u; c = c + 1u) {{
        if (col + c < N) {{
            var v = values[c];
{bias}            let i = out_index(row, col + c, layer);
            {y}[i] = epilogue(v, i);
        }}
    }}
}}

// {what}
// Each workgroup computes a TILE_ROWS x TILE_COLUMNS tile of the output, each
// invocation a ROWS_PER x COLUMNS_PER block of it in registers.
{entry}
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) wid: vec3<u32>,
) {{
    let row0 = {PART_START} + wid.y * TILE_ROWS;
    let col0 = wid.x * TILE_COLUMNS;
    let layer = wid.z;
    // Where the invocation's block starts, in vec4s of a staged row.
    let a_at = lid.y * (ROWS_PER / 4u);
    let b_at = lid.x * (COLUMNS_PER / 4u);
{declare}    for (var k0 = 0u; k0 < K; k0 = k0 + TILE_DEPTH) {{
        // Four rows of the left operand at one reduction index per vec4.
        for (var e = lane; e < TILE_ROWS / 4u * TILE_DEPTH; e = e + LANES) {{
            let kk = e % TILE_DEPTH;
            let r = (e / TILE_DEPTH) * 4u;
            let ka = k0 + kk;
            var v = vec4<f32>(0.0);
            for (var q = 0u; q < 4u; q = q + 1u) {{
                let row = row0 + r + q;
                if (row < {PART_END} && ka < K) {{
                    v[q] = load_a(row, ka, layer);
                }}
            }}
            mm_a[kk * LEFT_VEC4S + r / 4u] = v;
        }}
        // Four columns of the right operand at one reduction index per vec4.
        for (var e = lane; e < TILE_DEPTH * TILE_COLUMNS / 4u; e = e + LANES) {{
            let kk = {b_kk};
            let c = {b_c};
            let kb = k0 + kk;
            var v = vec4<f32>(0.0);
            for (var q = 0u; q < 4u; q = q + 1u) {{
                let col = col0 + c + q;
                if (kb < K && col < N) {{
                    v[q] = load_b(kb, col, layer);
                }}
            }}
            mm_b[kk * RIGHT_VEC4S + c / 4u] = v;
        }}
        workgroupBarrier();
        for (var kk = 0u; kk < TILE_DEPTH; kk = kk + 1u) {{
{multiply}        }}
        workgroupBarrier();
    }}
    let row_at = row0 + lid.y * ROWS_PER;
    let col_at = col0 + lid.x * COLUMNS_PER;
{store}}}
",
        k = t.k,
        n = t.n,
        rows = tile.rows,
        columns = tile.columns,
        depth = tile.depth,
        left = tile.left_stride(),
        right = tile.right_stride(),
        left_v = tile.left_stride() / 4,
        right_v = tile.right_stride() / 4,
        a_cells = tile.depth * tile.left_stride() / 4,
        b_cells = tile.depth * tile.right_stride() / 4,
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
        b_along_k: false,
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
        b_along_k: false,
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
        b_along_k: true,
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
        b_along_k: false,
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
