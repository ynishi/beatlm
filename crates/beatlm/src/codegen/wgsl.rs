//! The templates: one per main operation, sharing a header, the binding
//! declarations, the part's range and the epilogue hook.
//!
//! Every template computes one value per output element and stores it as
//! `out[i] = epilogue(value, i)`; [`epilogue`] writes that function from the
//! spec's steps, so the templates know nothing about fusion. Layouts are the
//! ones [`beatlm_core::kernel`] documents.
//!
//! Every template also walks only its part's range of the output, read from
//! the constants [`part_constants`] names, so the text is the whole kernel's
//! and one shader module serves every part.

use std::fmt::Write as _;

use beatlm_core::kernel::{
    Conv2d3x3, Epilogue, KernelSpec, LAYER_NORM_EPSILON, LayerNorm, MainOp, MatMul, Requirements,
    Variant,
};
use beatlm_core::memory::TensorId;

use super::ENTRY_POINT;

/// First unit of the part: an output element for the convolution and the map,
/// a row for the layer norm and the matrix multiply.
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
    match *spec.main() {
        MainOp::Conv2d3x3(c) => conv(&mut out, &c, &req, &names, &y),
        MainOp::LayerNorm(l) => layer_norm(&mut out, &l, &req, &names, &y),
        MainOp::MatMul(mm) => matmul(&mut out, &mm, &req, &names, &y),
        MainOp::Map { input, .. } => map(&mut out, &names.of(input), &req, &y),
    }
    out
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
        MainOp::Conv2d3x3(_) | MainOp::Map { .. } => {
            let elements = spec.output_range();
            vec![
                (PART_START, narrow(elements.start)),
                (PART_END, narrow(elements.end)),
                (PART_GRID_ROW, req.workgroups[0]),
            ]
        }
        MainOp::LayerNorm(_) => vec![
            (PART_START, rows.start),
            (PART_END, rows.end),
            (PART_GRID_ROW, req.workgroups[0]),
        ],
        MainOp::MatMul(_) => vec![(PART_START, rows.start), (PART_END, rows.end)],
    }
}

/// The declarations of the part's range, defaulting to the whole kernel, so
/// the text alone is the unsplit kernel and a pipeline that sets the
/// constants runs one part of it.
fn overrides(out: &mut String, whole: &KernelSpec, variant: Variant) {
    let unit = match whole.main() {
        MainOp::Conv2d3x3(_) | MainOp::Map { .. } => "output elements",
        MainOp::LayerNorm(_) | MainOp::MatMul(_) => "rows",
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
    let mut role = |id: TensorId, what: String| roles.push((id, what));
    match *spec.main() {
        MainOp::Conv2d3x3(c) => {
            role(c.input, "convolution input".into());
            role(c.weights, "convolution weights".into());
            if let Some(b) = c.bias {
                role(b, "convolution bias".into());
            }
        }
        MainOp::LayerNorm(l) => {
            role(l.input, "layer norm input".into());
            role(l.gamma, "layer norm gamma".into());
            role(l.beta, "layer norm beta".into());
        }
        MainOp::MatMul(mm) => {
            role(mm.a, "matmul left".into());
            role(mm.b, "matmul right".into());
            if let Some(b) = mm.bias {
                role(b, "matmul bias".into());
            }
        }
        MainOp::Map { input, .. } => role(input, "operand".into()),
    }
    for (step, e) in spec.epilogue().iter().enumerate() {
        if let Epilogue::Add(id) = e {
            role(*id, format!("addend of epilogue step {step}"));
        }
    }
    role(spec.output(), "output".into());

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
    out.push_str("fn epilogue(value: f32, index: u32) -> f32 {\n    var v = value;\n");
    for step in steps {
        let _ = match *step {
            Epilogue::Add(id) => writeln!(out, "    v = v + {}[index];", names.of(id)),
            Epilogue::Relu => writeln!(out, "    v = max(v, 0.0);"),
            Epilogue::Gelu => writeln!(out, "    v = gelu_tanh(v);"),
            Epilogue::Scale(factor) => writeln!(out, "    v = v * {};", exact(factor)),
        };
    }
    out.push_str("    return v;\n}\n\n");
}

/// The entry point's attribute line for the variant's workgroup size.
fn entry(req: &Requirements) -> String {
    let [x, y, z] = req.workgroup_size;
    format!("@compute @workgroup_size({x}, {y}, {z})\nfn {ENTRY_POINT}(")
}

/// One invocation per output element of the part, numbered in memory order
/// (channel fastest) from the part's first element over a grid of rows
/// `PART_GRID_ROW` workgroups wide.
fn conv(out: &mut String, c: &Conv2d3x3, req: &Requirements, names: &Names, y: &str) {
    let bias = c.bias.map_or(String::new(), |b| {
        format!("    acc = acc + {}[co];\n", names.of(b))
    });
    let _ = write!(
        out,
        "\
const H: u32 = {h}u;
const W: u32 = {w}u;
const CIN: u32 = {cin}u;
const COUT: u32 = {cout}u;
const GROUP: u32 = {group}u;

// 3x3 convolution, stride 1, zero padding of 1: NHWC input, HWIO weights.
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
    let py = i32(pixel / W);
    let px = i32(pixel % W);
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
                acc = acc + {x}[src + ci] * {wt}[tap + ci * COUT];
            }}
        }}
    }}
{bias}    {y}[element] = epilogue(acc, element);
}}
",
        h = c.height,
        w = c.width,
        cin = c.cin,
        cout = c.cout,
        group = req.workgroup_size[0],
        entry = entry(req),
        x = names.of(c.input),
        wt = names.of(c.weights),
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
        eps = exact(LAYER_NORM_EPSILON),
        x = names.of(l.input),
        gamma = names.of(l.gamma),
        beta = names.of(l.beta),
    );
}

/// One workgroup per output tile of the part, rows counted from the part's
/// first row, staging a tile of each operand in workgroup memory per step
/// along `k`. Tiles that run off the part's rows or the matrices load zeros, so
/// every invocation reaches every barrier, and store nothing.
fn matmul(out: &mut String, mm: &MatMul, req: &Requirements, names: &Names, y: &str) {
    let tile = req.workgroup_size[0];
    let bias = mm.bias.map_or(String::new(), |b| {
        format!("        acc = acc + {}[col];\n", names.of(b))
    });
    let _ = write!(
        out,
        "\
const K: u32 = {k}u;
const N: u32 = {n}u;
const TILE: u32 = {tile}u;

var<workgroup> mm_tile_a: array<f32, {cells}>;
var<workgroup> mm_tile_b: array<f32, {cells}>;

// C[M][N] = A[M][K] * B[K][N], row-major, tiled.
{entry}
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {{
    let row = {PART_START} + wid.y * TILE + lid.y;
    let col = wid.x * TILE + lid.x;
    var acc = 0.0;
    let steps = (K + TILE - 1u) / TILE;
    for (var t = 0u; t < steps; t = t + 1u) {{
        let ka = t * TILE + lid.x;
        let kb = t * TILE + lid.y;
        var va = 0.0;
        if (row < {PART_END} && ka < K) {{
            va = {a}[row * K + ka];
        }}
        var vb = 0.0;
        if (kb < K && col < N) {{
            vb = {b}[kb * N + col];
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
{bias}        let i = row * N + col;
        {y}[i] = epilogue(acc, i);
    }}
}}
",
        k = mm.k,
        n = mm.n,
        cells = tile * tile,
        entry = entry(req),
        a = names.of(mm.a),
        b = names.of(mm.b),
    );
}

/// One invocation per element of the part: the main operation of a standalone
/// element-wise kernel, whose work is all in the epilogue.
fn map(out: &mut String, x: &str, req: &Requirements, y: &str) {
    let _ = write!(
        out,
        "\
const GROUP: u32 = {group}u;

// An element-wise operation on its own.
{entry}
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
) {{
    let element = {PART_START} + (wid.y * {PART_GRID_ROW} + wid.x) * GROUP + lane;
    if (element >= {PART_END}) {{
        return;
    }}
    {y}[element] = epilogue({x}[element], element);
}}
",
        group = req.workgroup_size[0],
        entry = entry(req),
    );
}
