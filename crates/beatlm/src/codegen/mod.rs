//! Kernel specs turned into WGSL text, checked by naga, and realised as
//! pipelines and bind groups.
//!
//! The core decides what a kernel computes and what it needs from a device
//! ([`KernelSpec`]); this module is where that becomes shader text, the one
//! place a shader language is spoken. It owns these rules:
//!
//! - **WGSL text is the interface.** A spec becomes text through one template
//!   per kind of main operation (transposed convolution; bilinear resampling;
//!   layer norm; softmax; one register-tiled matrix multiply serving the
//!   matrix multiply, the 1x1 and 3x3 convolutions, the patch embedding and
//!   the two attention products; the token copies; and a map for a standalone
//!   element-wise kernel), each of which stores every output
//!   value through one `epilogue(value, index)` function generated from the
//!   spec's [`Epilogue`](beatlm_core::kernel::Epilogue) steps. That function is
//!   how fusion reaches the shader: a fused kernel is the heavy template with a
//!   longer epilogue. The 3x3 convolution also loads every input value through
//!   a `prologue(value)` function generated from the spec's
//!   [`Prologue`](beatlm_core::kernel::Prologue) steps, which is how a ReLU
//!   fused forwards reaches it. naga IR is never built directly; the crate
//!   documentation says why.
//! - **Nothing reaches the device unvalidated.** [`generate`] parses the text
//!   with `naga::front::wgsl::parse_str` and validates the module with
//!   `naga::valid::Validator`, all checks on and only the capabilities the
//!   variant's features grant, before a shader module or pipeline is created.
//!   A failure is a [`CodegenError`] carrying naga's diagnostic and the
//!   generated text, so the kernel can be read where it broke.
//! - **The text is a pure function of the spec and the variant.** The same
//!   spec gives the same bytes, so generated kernels can be diffed between
//!   runs; [`wgsl`] returns them for inspection.
//! - **Every storage binding is `read_write`.** Tensors that share an arena
//!   share a buffer, and wgpu checks usage per buffer, not per byte range (see
//!   [`Arenas`](crate::gpu::Arenas)); inputs are declared `read_write` too.
//!   Binding `i` is tensor `i` of [`KernelSpec::bindings`], so the binding
//!   order is the core's, not a second list kept here.
//! - **Shapes are constants in the text.** A kernel is generated for one shape,
//!   so its dimensions are WGSL `const`s, costing no uniform buffer and no
//!   pipeline constant; the launch geometry (workgroup size and count) is the
//!   core's [`Requirements`](beatlm_core::kernel::Requirements), and the tile
//!   of a tiled kernel the core's [`Tile`](beatlm_core::kernel::Tile), which
//!   the templates read rather than restate.
//! - **A part's range is a pipeline-overridable constant.** A kernel the core
//!   split ([`Program::split`](beatlm_core::kernel::Program::split)) is the
//!   same text for every part: each template declares the range of the output
//!   it walks as WGSL `override`s defaulting to the whole kernel, and
//!   [`constants`] gives a part's values, which [`Kernels`] sets on that part's
//!   pipeline. [`generate`] checks them against the overrides naga parsed from
//!   the text, so every part is validated, not only the text; a [`Generator`]
//!   does the same for many specs while parsing each distinct text once, which
//!   is how [`Kernels`] builds a split kernel's parts.
//!
//! The part's range could have been a uniform instead. Overrides were chosen
//! because the range of a part never changes after the plan is made, and an
//! override makes it part of the pipeline: the steady-state loop binds and
//! dispatches what exists and writes no buffer, no buffer outside the memory
//! plan's arenas is created for it, no binding is added to kernels whose
//! storage bindings fusion already fills to the baseline's limit, and the
//! shader compiler sees the range as a constant. The price is one pipeline and
//! one bind group per part rather than per kernel, all created before the
//! loop; the parts of a kernel share one shader module, because the text is
//! the same. A uniform per part would have needed one pipeline per kernel, and
//! a buffer per part made and filled before the loop.
//!
//! [`Kernels`] creates every pipeline and bind group of a program once, before
//! the frame loop, so the loop only records passes. Why fusion and splitting
//! pay, with the measurements, is in the [`beatlm_core::kernel`]
//! documentation.

mod kernels;
mod wgsl;

use beatlm_core::device::DeviceFeatures;
use beatlm_core::kernel::{KernelError, KernelSpec, Variant};
use beatlm_core::memory::TensorId;
use thiserror::Error;

use crate::gpu::ArenaError;

pub use kernels::{Kernels, Substitution};

/// The entry point every generated kernel has.
pub const ENTRY_POINT: &str = "main";

/// Why a kernel could not be generated, validated or realised.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum CodegenError {
    /// naga's parser rejected the generated text: a template is wrong.
    #[error("generated WGSL for kernel {kernel} does not parse:\n{diagnostic}")]
    Parse {
        /// The kernel's name.
        kernel: String,
        /// naga's diagnostic, with the source location.
        diagnostic: String,
        /// The generated text.
        wgsl: String,
    },
    /// naga's validator rejected the parsed module: a template is wrong, or
    /// uses a capability its variant's features do not grant.
    #[error("generated WGSL for kernel {kernel} does not validate:\n{diagnostic}")]
    Validation {
        /// The kernel's name.
        kernel: String,
        /// naga's diagnostic, with the source location.
        diagnostic: String,
        /// The generated text.
        wgsl: String,
    },
    /// No variant of the kernel runs on the device.
    #[error("kernel {kernel}: {source}")]
    Kernel {
        /// The kernel's name.
        kernel: String,
        /// What the core's check reported.
        source: KernelError,
    },
    /// A tensor the kernel binds has no placement in the plan.
    #[error("tensor {} has no placement in the memory plan", .0.get())]
    MissingPlacement(TensorId),
    /// A stand-in's placement is not the size of the tensor it replaces.
    #[error("tensor {} stands in for tensor {} but its placement is {stand_in_size} bytes, not {original_size}", stand_in.get(), original.get())]
    StandInSize {
        /// The tensor of the spec.
        original: TensorId,
        /// What is bound in its place.
        stand_in: TensorId,
        /// The original's placement size.
        original_size: u64,
        /// The stand-in's placement size.
        stand_in_size: u64,
    },
    /// A placement cannot be bound from the arenas.
    #[error(transparent)]
    Arena(#[from] ArenaError),
    /// An index past the kernels of a [`Kernels`].
    #[error("there is no kernel {0}")]
    NoSuchKernel(usize),
    /// An index past the binding sets a [`Kernels`] was built with.
    #[error("there is no binding set {0}")]
    NoSuchSet(usize),
    /// A part's constants do not match the overrides the text declares: a
    /// value for a name the text lacks or declares as other than `u32`, or an
    /// override left without a value.
    #[error(
        "generated WGSL for kernel {kernel} does not declare `override {name}: u32` once, or declares it and the part gives it no value"
    )]
    PartConstant {
        /// The kernel's name.
        kernel: String,
        /// The constant's name.
        name: String,
        /// The generated text.
        wgsl: String,
    },
}

/// The WGSL text of `variant` of `spec`, unvalidated, for inspection.
///
/// A pure function of the kernel: the same kernel and variant give the same
/// text, and so does every part of a split kernel, whose range is in
/// [`constants`]. [`generate`] is the same text, validated.
#[must_use]
pub fn wgsl(spec: &KernelSpec, variant: Variant) -> String {
    wgsl::text(spec, variant)
}

/// The pipeline-overridable constants that make a dispatch of the text of
/// `variant` of `spec` compute `spec`'s part, as `(name, value)`, in the form
/// [`wgpu::PipelineCompilationOptions::constants`] takes.
///
/// A whole kernel's values are the text's defaults.
#[must_use]
pub fn constants(spec: &KernelSpec, variant: Variant) -> Vec<(&'static str, f64)> {
    wgsl::part_constants(spec, variant)
        .into_iter()
        .map(|(name, value)| (name, f64::from(value)))
        .collect()
}

/// The WGSL text of `variant` of `spec`, parsed and validated by naga with
/// the capabilities the variant's features grant, and with `spec`'s
/// [`constants`] checked against the overrides the parsed module declares:
/// each names one `u32` override, and each override has a value.
///
/// One call checks one spec; a caller with several, such as the parts of a
/// split kernel, uses a [`Generator`], which parses each distinct text once.
///
/// # Errors
///
/// [`CodegenError::Parse`] or [`CodegenError::Validation`], with the text, or
/// [`CodegenError::PartConstant`].
pub fn generate(spec: &KernelSpec, variant: Variant) -> Result<String, CodegenError> {
    Generator::new().generate(spec, variant).map(str::to_owned)
}

/// A text naga has validated, and the overrides the validated module declares.
#[derive(Debug)]
struct Checked {
    text: String,
    features: DeviceFeatures,
    /// Each declared override as `(name, is u32)`.
    overrides: Vec<(String, bool)>,
}

/// Generates and checks kernel texts, remembering each distinct text it has
/// validated.
///
/// The parts of a split kernel have the same text and differ only in their
/// [`constants`], so the text is parsed and validated by naga once, when the
/// first part asks for it, and every part's constants are still checked
/// against the overrides that validation found. "Nothing reaches the device
/// unvalidated" (the [module documentation](self)) then holds for every part
/// without parsing the same text once per part. A part's values change no type
/// or size in the text, so a text that validates with its defaults validates
/// with any part's; naga's override substitution is not in this crate's naga
/// features, and wgpu performs it, and validates the result again, when it
/// creates the part's pipeline.
#[derive(Debug, Default)]
pub struct Generator {
    checked: Vec<Checked>,
}

impl Generator {
    /// A generator that has validated nothing.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// How many distinct texts this generator has parsed and validated; a
    /// caller or a test reads it to see that parts sharing a text did not each
    /// parse it.
    #[must_use]
    pub fn validated(&self) -> usize {
        self.checked.len()
    }

    /// The text of `variant` of `spec`, validated (once per distinct text and
    /// feature set) and with `spec`'s [`constants`] checked against its
    /// overrides, as [`generate`] does for one spec.
    ///
    /// # Errors
    ///
    /// [`CodegenError::Parse`] or [`CodegenError::Validation`], with the text,
    /// or [`CodegenError::PartConstant`].
    pub fn generate(&mut self, spec: &KernelSpec, variant: Variant) -> Result<&str, CodegenError> {
        let index = self.check(spec, variant)?;
        Ok(self.text(index))
    }

    /// As [`generate`](Self::generate), returning the index of the distinct
    /// text instead of the text: specs that share a text and feature set share
    /// an index, so a caller can key what it builds from the text (a shader
    /// module) by it instead of comparing texts again.
    pub(crate) fn check(
        &mut self,
        spec: &KernelSpec,
        variant: Variant,
    ) -> Result<usize, CodegenError> {
        let kernel = spec.name();
        let text = wgsl(spec, variant);
        let features = spec.requirements(variant).features;
        let index = if let Some(index) = self
            .checked
            .iter()
            .position(|c| c.features == features && c.text == text)
        {
            index
        } else {
            let module = parse_and_validate(&kernel, &text, features)?;
            let overrides = module
                .overrides
                .iter()
                .map(|(_, o)| {
                    let is_u32 =
                        module.types[o.ty].inner == naga::TypeInner::Scalar(naga::Scalar::U32);
                    (o.name.clone().unwrap_or_default(), is_u32)
                })
                .collect();
            self.checked.push(Checked {
                text,
                features,
                overrides,
            });
            self.checked.len() - 1
        };
        check_constants(&kernel, &self.checked[index], &constants(spec, variant))?;
        Ok(index)
    }

    /// The validated text at `index`, as [`check`](Self::check) returned it.
    pub(crate) fn text(&self, index: usize) -> &str {
        &self.checked[index].text
    }
}

/// Checks that every override `checked` declares is a `u32` the part gives one
/// value, and that every value the part gives names one of them.
fn check_constants(
    kernel: &str,
    checked: &Checked,
    constants: &[(&'static str, f64)],
) -> Result<(), CodegenError> {
    let error = |name: &str| CodegenError::PartConstant {
        kernel: kernel.to_owned(),
        name: name.to_owned(),
        wgsl: checked.text.clone(),
    };
    for (name, is_u32) in &checked.overrides {
        if !is_u32 || constants.iter().filter(|(c, _)| c == name).count() != 1 {
            return Err(error(name));
        }
    }
    if let Some((name, _)) = constants
        .iter()
        .find(|(c, _)| !checked.overrides.iter().any(|(o, _)| o == c))
    {
        return Err(error(name));
    }
    Ok(())
}

/// Parses `wgsl` with naga and validates it with every check on and only the
/// capabilities `features` grant, naming the kernel `kernel` in an error.
///
/// # Errors
///
/// [`CodegenError::Parse`] when the text does not parse,
/// [`CodegenError::Validation`] when the module does not validate; both carry
/// naga's diagnostic and the text.
pub fn validate(kernel: &str, wgsl: &str, features: DeviceFeatures) -> Result<(), CodegenError> {
    parse_and_validate(kernel, wgsl, features).map(drop)
}

/// [`validate`], returning the parsed module.
fn parse_and_validate(
    kernel: &str,
    wgsl: &str,
    features: DeviceFeatures,
) -> Result<naga::Module, CodegenError> {
    let module = naga::front::wgsl::parse_str(wgsl).map_err(|e| CodegenError::Parse {
        kernel: kernel.to_owned(),
        diagnostic: e.emit_to_string(wgsl),
        wgsl: wgsl.to_owned(),
    })?;
    naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities(features))
        .validate(&module)
        .map_err(|e| CodegenError::Validation {
            kernel: kernel.to_owned(),
            diagnostic: e.emit_to_string(wgsl),
            wgsl: wgsl.to_owned(),
        })?;
    Ok(module)
}

/// The naga capabilities a kernel with `features` may use: none beyond the
/// baseline unless a feature grants them, so a kernel that uses a feature it
/// did not declare fails validation here rather than on a device.
fn capabilities(features: DeviceFeatures) -> naga::valid::Capabilities {
    use naga::valid::Capabilities as C;
    let mut caps = C::empty();
    if features.shader_f16 {
        caps |= C::SHADER_FLOAT16;
    }
    if features.subgroups {
        caps |= C::SUBGROUP | C::SUBGROUP_BARRIER;
    }
    caps
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use beatlm_core::kernel::{
        AttentionScores, AttentionValues, Conv2d1x1, Conv2d3x3, ConvTranspose2d, Fusion, Graph,
        LAYER_NORM_EPSILON, LayerNorm, MatMul, OpKind, PatchEmbed, PrependToken, Program,
        ResizeBilinear, Softmax, TensorRole, TokensToMap,
    };

    const X: TensorId = TensorId::new(0);
    const W: TensorId = TensorId::new(1);
    const P: TensorId = TensorId::new(2);
    const Q: TensorId = TensorId::new(3);
    const R: TensorId = TensorId::new(4);
    /// The per-channel multipliers of a LayerScale step.
    const S: TensorId = TensorId::new(5);
    const H: TensorId = TensorId::new(10);
    const Y: TensorId = TensorId::new(20);
    /// Activations of the epilogue chain.
    const CHAIN: [TensorId; 4] = [
        TensorId::new(11),
        TensorId::new(12),
        TensorId::new(13),
        TensorId::new(14),
    ];

    /// The epilogues every heavy op is generated with: none, each step alone,
    /// and chains of four. `channels` is the output's last dimension.
    fn epilogues(elements: u32, channels: u32) -> Vec<Vec<OpKind>> {
        let relu = |input| OpKind::Relu { input, elements };
        let gelu = |input| OpKind::Gelu { input, elements };
        let gelu_erf = |input| OpKind::GeluErf { input, elements };
        let layer_scale = |input| OpKind::ChannelScale {
            input,
            scale: S,
            rows: elements / channels,
            channels,
        };
        let scale = |input| OpKind::Scale {
            input,
            factor: -0.75,
            elements,
        };
        let add = |input| OpKind::Add {
            a: R,
            b: input,
            elements,
        };
        vec![
            vec![],
            vec![relu(H)],
            vec![gelu(H)],
            vec![scale(H)],
            vec![add(H)],
            vec![gelu_erf(H)],
            vec![layer_scale(H)],
            vec![add(H), relu(CHAIN[0]), gelu(CHAIN[1]), scale(CHAIN[2])],
            vec![
                layer_scale(H),
                add(CHAIN[0]),
                gelu_erf(CHAIN[1]),
                scale(CHAIN[2]),
            ],
        ]
    }

    /// `heavy` writing `H`, then `steps`, the last writing `Y`, lowered with
    /// `fusion`. `inputs` are declared with their element counts; `R`, the
    /// addend, has as many elements as the output, and `S`, the multipliers,
    /// one per channel of it.
    fn program(
        inputs: &[(TensorId, u64)],
        heavy: OpKind,
        out: (u64, u32),
        steps: &[OpKind],
        fusion: Fusion,
    ) -> Program {
        program_from(inputs, false, heavy, out, steps, fusion)
    }

    /// The ReLU of `X` a prologue test puts before its convolution.
    const RELU: TensorId = TensorId::new(30);

    /// [`program`], with a ReLU of `X` into `RELU` before `heavy` when
    /// `relu_first`; `heavy` then reads `RELU`.
    fn program_from(
        inputs: &[(TensorId, u64)],
        relu_first: bool,
        heavy: OpKind,
        (out, channels): (u64, u32),
        steps: &[OpKind],
        fusion: Fusion,
    ) -> Program {
        let mut g = Graph::new();
        for &(id, elements) in inputs {
            g.tensor(id, TensorRole::Weight, elements).unwrap();
        }
        if relu_first {
            let elements = inputs.iter().find(|(id, _)| *id == X).unwrap().1;
            g.tensor(RELU, TensorRole::Activation, elements).unwrap();
            g.push(
                OpKind::Relu {
                    input: X,
                    elements: u32::try_from(elements).unwrap(),
                },
                RELU,
            )
            .unwrap();
        }
        g.tensor(R, TensorRole::Input, out).unwrap();
        g.tensor(S, TensorRole::Weight, channels.into()).unwrap();
        let outputs: Vec<TensorId> = std::iter::once(H).chain(CHAIN).take(steps.len()).collect();
        for &t in &outputs {
            g.tensor(t, TensorRole::Activation, out).unwrap();
        }
        g.tensor(Y, TensorRole::Output, out).unwrap();
        let mut writes = outputs[..].iter().skip(1).copied().chain([Y]);
        g.push(heavy, if steps.is_empty() { Y } else { H }).unwrap();
        for &step in steps {
            g.push(step, writes.next().unwrap()).unwrap();
        }
        g.lower(fusion).unwrap()
    }

    /// A heavy op with the tensors it reads and their element counts, and the
    /// element count of its output and of the output's last dimension.
    type Heavy = (Vec<(TensorId, u64)>, OpKind, (u64, u32));

    /// Every heavy op of the tests: odd shapes, so no dimension is a multiple
    /// of a tile.
    fn heavies() -> Vec<Heavy> {
        let (h, w, cin, cout) = (7, 9, 3, 5);
        let conv = |bias| {
            OpKind::Conv2d3x3(Conv2d3x3 {
                input: X,
                weights: W,
                bias,
                height: h,
                width: w,
                cin,
                cout,
                stride: 1,
            })
        };
        let conv_out = (u64::from(h * w * cout), cout);
        let (m, k, n) = (37, 21, 19);
        let mm = |bias| {
            OpKind::MatMul(MatMul {
                a: X,
                b: W,
                bias,
                m,
                k,
                n,
            })
        };
        let (rows, c) = (11, 150);
        let mut all = vec![
            (
                vec![(X, u64::from(h * w * cin)), (W, u64::from(9 * cin * cout))],
                conv(None),
                conv_out,
            ),
            (
                vec![
                    (X, u64::from(h * w * cin)),
                    (W, u64::from(9 * cin * cout)),
                    (P, u64::from(cout)),
                ],
                conv(Some(P)),
                conv_out,
            ),
            (
                vec![(X, u64::from(rows * c)), (Q, c.into()), (P, c.into())],
                OpKind::LayerNorm(LayerNorm {
                    input: X,
                    gamma: Q,
                    beta: P,
                    rows,
                    channels: c,
                    epsilon: LAYER_NORM_EPSILON,
                }),
                (u64::from(rows * c), c),
            ),
            (
                vec![(X, u64::from(m * k)), (W, u64::from(k * n))],
                mm(None),
                (u64::from(m * n), n),
            ),
            (
                vec![(X, u64::from(m * k)), (W, u64::from(k * n)), (P, n.into())],
                mm(Some(P)),
                (u64::from(m * n), n),
            ),
        ];
        all.extend(encoder_heavies());
        all.extend(decoder_heavies());
        all.extend(wide_heavies());
        all
    }

    /// Tiled kernels whose outputs are wider than the narrow tile, so the
    /// other tile's text is generated too, at odd shapes: a matrix multiply, a
    /// strided convolution with a bias and the attention scores.
    fn wide_heavies() -> Vec<Heavy> {
        let (m, k, n) = (37, 21, 70);
        let (h, w, cin, cout) = (7, 9, 3, 37);
        let (tokens, heads, head_dim) = (41, 2, 5);
        let heavies: Vec<Heavy> = vec![
            (
                vec![(X, u64::from(m * k)), (W, u64::from(k * n)), (P, n.into())],
                OpKind::MatMul(MatMul {
                    a: X,
                    b: W,
                    bias: Some(P),
                    m,
                    k,
                    n,
                }),
                (u64::from(m * n), n),
            ),
            (
                vec![
                    (X, u64::from(h * w * cin)),
                    (W, u64::from(9 * cin * cout)),
                    (P, u64::from(cout)),
                ],
                OpKind::Conv2d3x3(Conv2d3x3 {
                    input: X,
                    weights: W,
                    bias: Some(P),
                    height: h,
                    width: w,
                    cin,
                    cout,
                    stride: 2,
                }),
                (u64::from(4 * 5 * cout), cout),
            ),
            (
                vec![(X, u64::from(tokens * 3 * heads * head_dim))],
                OpKind::AttentionScores(AttentionScores {
                    qkv: X,
                    tokens,
                    heads,
                    head_dim,
                }),
                (u64::from(tokens * heads * tokens), tokens),
            ),
        ];
        heavies
    }

    #[test]
    fn both_tiles_are_generated_and_read_from_the_core() {
        let mut columns = std::collections::BTreeSet::new();
        for (inputs, heavy, out) in heavies() {
            let p = program(&inputs, heavy, out, &[], Fusion::On);
            let spec = &p.kernels()[0];
            let Some(tile) = spec.tile() else {
                continue;
            };
            let text = generate(spec, Variant::Baseline).unwrap();
            columns.insert(tile.columns);
            let [x, y, z] = tile.workgroup_size();
            for line in [
                format!("const TILE_ROWS: u32 = {}u;", tile.rows),
                format!("const TILE_COLUMNS: u32 = {}u;", tile.columns),
                format!("const TILE_DEPTH: u32 = {}u;", tile.depth),
                format!("@workgroup_size({x}, {y}, {z})"),
                format!(
                    "var<workgroup> mm_a: array<vec4<f32>, {}>;",
                    tile.depth * tile.left_stride() / 4
                ),
            ] {
                assert!(text.contains(&line), "{line} in\n{text}");
            }
            // One accumulator per four outputs of the invocation's block.
            let accumulators = text.matches("= vec4<f32>(0.0);\n").count();
            let block = tile.rows_per_invocation * tile.columns_per_invocation / 4;
            assert_eq!(accumulators as u32, block + 2, "{}", spec.name());
        }
        assert_eq!(columns.len(), 2, "{columns:?}");
    }

    /// The DPT head's heavy ops, at odd shapes: a strided convolution with and
    /// without bias, a 1x1 convolution, a transposed convolution and a
    /// bilinear resampling.
    fn decoder_heavies() -> Vec<Heavy> {
        let (h, w, cin, cout) = (7, 9, 3, 5);
        let strided = |bias| {
            OpKind::Conv2d3x3(Conv2d3x3 {
                input: X,
                weights: W,
                bias,
                height: h,
                width: w,
                cin,
                cout,
                stride: 2,
            })
        };
        // 7 x 9 at stride 2 is 4 x 5.
        let strided_out = (u64::from(4 * 5 * cout), cout);
        let image = u64::from(h * w * cin);
        let factor = 3;
        let up = |bias| {
            OpKind::ConvTranspose2d(ConvTranspose2d {
                input: X,
                weights: W,
                bias,
                height: h,
                width: w,
                cin,
                cout,
                factor,
            })
        };
        let up_out = (u64::from(h * factor * w * factor * cout), cout);
        let up_weights = u64::from(factor * factor * cin * cout);
        vec![
            (
                vec![(X, image), (W, u64::from(9 * cin * cout))],
                strided(None),
                strided_out,
            ),
            (
                vec![
                    (X, image),
                    (W, u64::from(9 * cin * cout)),
                    (P, u64::from(cout)),
                ],
                strided(Some(P)),
                strided_out,
            ),
            (
                vec![(X, image), (W, u64::from(cin * cout)), (P, u64::from(cout))],
                OpKind::Conv2d1x1(Conv2d1x1 {
                    input: X,
                    weights: W,
                    bias: Some(P),
                    height: h,
                    width: w,
                    cin,
                    cout,
                }),
                (u64::from(h * w * cout), cout),
            ),
            (vec![(X, image), (W, up_weights)], up(None), up_out),
            (
                vec![(X, image), (W, up_weights), (P, u64::from(cout))],
                up(Some(P)),
                up_out,
            ),
            (
                vec![(X, image)],
                OpKind::ResizeBilinear(ResizeBilinear {
                    input: X,
                    in_height: h,
                    in_width: w,
                    out_height: 11,
                    out_width: 4,
                    channels: cin,
                }),
                (u64::from(11 * 4 * cin), cin),
            ),
        ]
    }

    /// The vision transformer's heavy ops, at odd shapes.
    fn encoder_heavies() -> Vec<Heavy> {
        let (gh, gw, patch, cin, cout) = (3, 4, 3, 2, 19);
        let patch_embed = |bias| {
            OpKind::PatchEmbed(PatchEmbed {
                input: X,
                weights: W,
                bias,
                grid_h: gh,
                grid_w: gw,
                patch,
                cin,
                cout,
            })
        };
        let image = u64::from(gh * patch * gw * patch * cin);
        let kernel = u64::from(patch * patch * cin * cout);
        let (tokens, heads, head_dim) = (17, 2, 5);
        let qkv = u64::from(tokens * 3 * heads * head_dim);
        let scores = u64::from(tokens * heads * tokens);
        let (seq, width) = (7, 5);
        vec![
            (
                vec![(X, image), (W, kernel)],
                patch_embed(None),
                (u64::from(gh * gw * cout), cout),
            ),
            (
                vec![(X, image), (W, kernel), (P, cout.into())],
                patch_embed(Some(P)),
                (u64::from(gh * gw * cout), cout),
            ),
            (
                vec![(W, width.into()), (X, u64::from(seq * width))],
                OpKind::PrependToken(PrependToken {
                    token: W,
                    tokens: X,
                    rows: seq,
                    channels: width,
                }),
                (u64::from((seq + 1) * width), width),
            ),
            (
                vec![(X, u64::from((1 + gh * gw) * width))],
                OpKind::TokensToMap(TokensToMap {
                    input: X,
                    height: gh,
                    width: gw,
                    channels: width,
                }),
                (u64::from(gh * gw * width), width),
            ),
            (
                vec![(X, qkv)],
                OpKind::AttentionScores(AttentionScores {
                    qkv: X,
                    tokens,
                    heads,
                    head_dim,
                }),
                (scores, tokens),
            ),
            (
                vec![(X, 19 * 300_u64)],
                OpKind::Softmax(Softmax {
                    input: X,
                    rows: 19,
                    cols: 300,
                }),
                (19 * 300_u64, 300),
            ),
            (
                vec![(W, scores), (X, qkv)],
                OpKind::AttentionValues(AttentionValues {
                    probs: W,
                    qkv: X,
                    tokens,
                    heads,
                    head_dim,
                }),
                (u64::from(tokens * heads * head_dim), heads * head_dim),
            ),
        ]
    }

    #[test]
    fn every_generated_kernel_parses_and_validates_with_and_without_epilogues() {
        let mut names = std::collections::BTreeSet::new();
        for (inputs, heavy, out) in heavies() {
            let elements = u32::try_from(out.0).unwrap();
            for steps in epilogues(elements, out.1) {
                for fusion in [Fusion::On, Fusion::Off] {
                    let p = program(&inputs, heavy, out, &steps, fusion);
                    let expected = if fusion == Fusion::On {
                        1
                    } else {
                        1 + steps.len()
                    };
                    assert_eq!(p.kernels().len(), expected);
                    for spec in p.kernels() {
                        let text =
                            generate(spec, Variant::Baseline).unwrap_or_else(|e| panic!("{e}"));
                        assert_eq!(
                            text.matches("var<storage, read_write>").count(),
                            spec.bindings().len(),
                            "{}",
                            spec.name()
                        );
                        names.insert(spec.name());
                    }
                }
            }
        }
        // Each template, alone and with each step and with the chain, was
        // generated.
        for name in [
            "conv2d3x3",
            "conv2d3x3+relu",
            "layernorm+gelu",
            "matmul+scale",
            "matmul+add+relu+gelu+scale",
            "matmul+channel_scale+add+gelu_erf+scale",
            "patch_embed+gelu_erf",
            "prepend_token+add",
            "tokens_to_map+channel_scale",
            "attention_scores+scale",
            "softmax+add+relu+gelu+scale",
            "attention_values+channel_scale+add+gelu_erf+scale",
            "map+relu",
            "map+gelu",
            "map+gelu_erf",
            "map+scale",
            "map+channel_scale",
            "map+add",
            "conv_transpose2d",
            "conv_transpose2d+add+relu+gelu+scale",
            "resize_bilinear+channel_scale+add+gelu_erf+scale",
        ] {
            assert!(names.contains(name), "{name} in {names:?}");
        }
    }

    #[test]
    fn a_convolution_with_a_relu_prologue_validates_whole_and_split_with_every_epilogue() {
        for (stride, (oh, ow)) in [(1, (7, 9)), (2, (4, 5))] {
            for bias in [None, Some(P)] {
                let (cin, cout) = (3, 5);
                let heavy = OpKind::Conv2d3x3(Conv2d3x3 {
                    input: RELU,
                    weights: W,
                    bias,
                    height: 7,
                    width: 9,
                    cin,
                    cout,
                    stride,
                });
                let mut inputs = vec![(X, 7 * 9 * 3), (W, 9 * 3 * 5)];
                if bias.is_some() {
                    inputs.push((P, 5));
                }
                let out = (u64::from(oh * ow * cout), cout);
                for steps in epilogues(oh * ow * cout, cout) {
                    let fused = program_from(&inputs, true, heavy, out, &steps, Fusion::On);
                    assert_eq!(fused.kernels().len(), 1);
                    let spec = &fused.kernels()[0];
                    assert!(spec.name().starts_with("relu-conv2d3x3"), "{}", spec.name());
                    let text = generate(spec, Variant::Baseline).unwrap_or_else(|e| panic!("{e}"));
                    assert!(
                        text.contains(
                            "fn prologue(value: f32) -> f32 {\n    var v = value;\n    \
                             v = max(v, 0.0);\n    return v;"
                        ),
                        "{text}"
                    );
                    for count in [2, 3, 7, 64] {
                        let split = fused.split(&[count]).unwrap();
                        for part in split.kernels() {
                            assert_eq!(
                                generate(part, Variant::Baseline).unwrap_or_else(|e| panic!("{e}")),
                                text
                            );
                        }
                    }
                    // Unfused, the ReLU is a map of its own and the
                    // convolution's prologue returns what it loads.
                    let unfused = program_from(&inputs, true, heavy, out, &steps, Fusion::Off);
                    assert_eq!(unfused.kernels()[0].name(), "map+relu");
                    let conv = generate(&unfused.kernels()[1], Variant::Baseline).unwrap();
                    assert!(!conv.contains("max(v, 0.0)"), "{conv}");
                    for spec in unfused.kernels() {
                        generate(spec, Variant::Baseline).unwrap_or_else(|e| panic!("{e}"));
                    }
                }
            }
        }
    }

    #[test]
    fn every_part_of_every_split_kernel_validates_and_shares_its_kernels_text() {
        for (inputs, heavy, out) in heavies() {
            let elements = u32::try_from(out.0).unwrap();
            for steps in epilogues(elements, out.1) {
                for fusion in [Fusion::On, Fusion::Off] {
                    let whole = program(&inputs, heavy, out, &steps, fusion);
                    // 64 is more parts than the convolution, the layer norm,
                    // the tiled kernels and the softmax allow, so it is
                    // clamped.
                    for count in [2, 3, 7, 64] {
                        let split = whole.split(&vec![count; whole.kernels().len()]).unwrap();
                        for part in split.kernels() {
                            let what = format!("{} split {count} ways", part.name());
                            let text = generate(part, Variant::Baseline)
                                .unwrap_or_else(|e| panic!("{what}: {e}"));
                            assert_eq!(text, wgsl(&part.whole(), Variant::Baseline), "{what}");
                            let values = constants(part, Variant::Baseline);
                            let start = values.iter().find(|(n, _)| *n == "PART_START").unwrap().1;
                            let end = values.iter().find(|(n, _)| *n == "PART_END").unwrap().1;
                            assert!(start < end, "{what}: {values:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_parts_of_a_kernel_validate_their_text_once_and_each_check_their_constants() {
        let (inputs, heavy, out) = heavies().remove(2);
        let whole = program(&inputs, heavy, out, &[], Fusion::On);
        let split = whole.split(&[3]).unwrap();
        assert_eq!(split.kernels().len(), 3);
        let mut generator = Generator::new();
        for part in split.kernels() {
            let text = generator
                .generate(part, Variant::Baseline)
                .unwrap()
                .to_owned();
            assert_eq!(text, wgsl(&part.whole(), Variant::Baseline));
        }
        assert_eq!(generator.validated(), 1);
        // A kernel with another text is validated on its own.
        let other = program(
            &inputs,
            heavy,
            out,
            &epilogues(u32::try_from(out.0).unwrap(), out.1)[1],
            Fusion::On,
        );
        generator
            .generate(&other.kernels()[0], Variant::Baseline)
            .unwrap();
        assert_eq!(generator.validated(), 2);
    }

    #[test]
    fn a_whole_kernels_constants_are_the_texts_defaults() {
        for (inputs, heavy, out) in heavies() {
            let p = program(&inputs, heavy, out, &[], Fusion::On);
            let spec = &p.kernels()[0];
            let text = generate(spec, Variant::Baseline).unwrap();
            for (name, value) in constants(spec, Variant::Baseline) {
                let declaration = format!("override {name}: u32 = {value}u;");
                assert!(text.contains(&declaration), "{declaration} in\n{text}");
            }
        }
    }

    #[test]
    fn the_parts_of_a_kernel_carry_their_own_ranges() {
        // A layer norm of 11 rows in three parts: rows 0..3, 3..7, 7..11.
        let (inputs, heavy, out) = heavies().remove(2);
        let p = program(&inputs, heavy, out, &[], Fusion::On);
        let split = p.split(&[3]).unwrap();
        let ranges: Vec<(f64, f64, f64)> = split
            .kernels()
            .iter()
            .map(|k| {
                let c = constants(k, Variant::Baseline);
                (c[0].1, c[1].1, c[2].1)
            })
            .collect();
        assert_eq!(
            ranges,
            vec![(0.0, 3.0, 3.0), (3.0, 7.0, 4.0), (7.0, 11.0, 4.0)]
        );
    }

    #[test]
    fn an_add_of_a_tensor_to_itself_binds_it_once() {
        let mut g = Graph::new();
        g.tensor(X, TensorRole::Input, 64).unwrap();
        g.tensor(Y, TensorRole::Output, 64).unwrap();
        g.push(
            OpKind::Add {
                a: X,
                b: X,
                elements: 64,
            },
            Y,
        )
        .unwrap();
        let p = g.lower(Fusion::On).unwrap();
        let text = generate(&p.kernels()[0], Variant::Baseline).unwrap();
        assert_eq!(text.matches("@binding(").count(), 2);
    }

    #[test]
    fn the_text_is_a_function_of_the_spec() {
        let (inputs, heavy, out) = heavies().remove(4);
        let steps = epilogues(u32::try_from(out.0).unwrap(), out.1).remove(7);
        let a = program(&inputs, heavy, out, &steps, Fusion::On);
        let b = program(&inputs, heavy, out, &steps, Fusion::On);
        assert_eq!(
            wgsl(&a.kernels()[0], Variant::Baseline),
            wgsl(&b.kernels()[0], Variant::Baseline)
        );
        // The scale's factor reaches the text exactly, as its bit pattern.
        let bits = format!("0x{:08x}u", (-0.75_f32).to_bits());
        assert!(wgsl(&a.kernels()[0], Variant::Baseline).contains(&bits));
    }

    #[test]
    fn the_erf_approximation_reaches_the_text_as_the_core_states_it() {
        let (inputs, heavy, out) = heavies().remove(4);
        let steps = epilogues(u32::try_from(out.0).unwrap(), out.1).remove(5);
        let p = program(&inputs, heavy, out, &steps, Fusion::On);
        let text = wgsl(&p.kernels()[0], Variant::Baseline);
        for value in
            std::iter::once(beatlm_core::kernel::GELU_ERF_P).chain(beatlm_core::kernel::GELU_ERF_A)
        {
            let bits = format!("0x{:08x}u", value.to_bits());
            assert!(text.contains(&bits), "{value} in\n{text}");
        }
    }

    #[test]
    fn a_layer_norms_epsilon_reaches_the_text() {
        let mut g = Graph::new();
        g.tensor(X, TensorRole::Input, 8).unwrap();
        g.tensor(Q, TensorRole::Weight, 4).unwrap();
        g.tensor(P, TensorRole::Weight, 4).unwrap();
        g.tensor(Y, TensorRole::Output, 8).unwrap();
        g.push(
            OpKind::LayerNorm(LayerNorm {
                input: X,
                gamma: Q,
                beta: P,
                rows: 2,
                channels: 4,
                epsilon: 1e-6,
            }),
            Y,
        )
        .unwrap();
        let text = generate(
            &g.lower(Fusion::On).unwrap().kernels()[0],
            Variant::Baseline,
        )
        .unwrap();
        assert!(text.contains(&format!("0x{:08x}u", 1e-6_f32.to_bits())));
    }

    #[test]
    fn a_text_that_does_not_parse_is_a_parse_error_carrying_the_text() {
        let text = "fn main( {";
        match validate("broken", text, DeviceFeatures::default()) {
            Err(CodegenError::Parse {
                kernel,
                diagnostic,
                wgsl,
            }) => {
                assert_eq!((kernel.as_str(), wgsl.as_str()), ("broken", text));
                assert!(!diagnostic.is_empty());
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_capability_the_features_do_not_grant_is_a_validation_error() {
        let text = "enable f16;\n@compute @workgroup_size(1)\nfn main() {\n    var h: f16 = 1.0h;\n    h = h + 1.0h;\n}\n";
        match validate("half", text, DeviceFeatures::default()) {
            Err(CodegenError::Validation { wgsl, .. }) => assert_eq!(wgsl, text),
            other => panic!("{other:?}"),
        }
        let f16 = DeviceFeatures {
            shader_f16: true,
            ..DeviceFeatures::default()
        };
        assert_eq!(validate("half", text, f16), Ok(()));
    }
}
