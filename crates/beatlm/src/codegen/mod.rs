//! Kernel specs turned into WGSL text, checked by naga, and realised as
//! pipelines and bind groups.
//!
//! The core decides what a kernel computes and what it needs from a device
//! ([`KernelSpec`]); this module is where that becomes shader text, the one
//! place a shader language is spoken. It owns these rules:
//!
//! - **WGSL text is the interface.** A spec becomes text through one template
//!   per main operation (convolution, layer norm, matrix multiply, and a map
//!   for a standalone element-wise kernel), each of which stores every output
//!   value through one `epilogue(value, index)` function generated from the
//!   spec's [`Epilogue`](beatlm_core::kernel::Epilogue) steps. That function is
//!   how fusion reaches the shader: a fused kernel is the heavy template with a
//!   longer epilogue. naga IR is never built directly; the crate
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
//!   core's [`Requirements`](beatlm_core::kernel::Requirements), which the
//!   template reads rather than restates.
//!
//! [`Kernels`] creates every pipeline and bind group of a program once, before
//! the frame loop, so the loop only records passes. Why fusion pays, with the
//! measurement, is in the [`beatlm_core::kernel`] documentation.

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
}

/// The WGSL text of `variant` of `spec`, unvalidated, for inspection.
///
/// A pure function: the same spec and variant give the same text. [`generate`]
/// is the same text, validated.
#[must_use]
pub fn wgsl(spec: &KernelSpec, variant: Variant) -> String {
    wgsl::text(spec, variant)
}

/// The WGSL text of `variant` of `spec`, parsed and validated by naga with
/// the capabilities the variant's features grant.
///
/// # Errors
///
/// [`CodegenError::Parse`] or [`CodegenError::Validation`], with the text.
pub fn generate(spec: &KernelSpec, variant: Variant) -> Result<String, CodegenError> {
    let text = wgsl(spec, variant);
    validate(&spec.name(), &text, spec.requirements(variant).features)?;
    Ok(text)
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
    Ok(())
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
        Conv2d3x3, Fusion, Graph, LayerNorm, MatMul, OpKind, Program, TensorRole,
    };

    const X: TensorId = TensorId::new(0);
    const W: TensorId = TensorId::new(1);
    const P: TensorId = TensorId::new(2);
    const Q: TensorId = TensorId::new(3);
    const R: TensorId = TensorId::new(4);
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
    /// and all four in a chain.
    fn epilogues(elements: u32) -> Vec<Vec<OpKind>> {
        let relu = |input| OpKind::Relu { input, elements };
        let gelu = |input| OpKind::Gelu { input, elements };
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
            vec![add(H), relu(CHAIN[0]), gelu(CHAIN[1]), scale(CHAIN[2])],
        ]
    }

    /// `heavy` writing `H`, then `steps`, the last writing `Y`, lowered with
    /// `fusion`. `inputs` are declared with their element counts; `R`, the
    /// addend, has as many elements as the output.
    fn program(
        inputs: &[(TensorId, u64)],
        heavy: OpKind,
        out: u64,
        steps: &[OpKind],
        fusion: Fusion,
    ) -> Program {
        let mut g = Graph::new();
        for &(id, elements) in inputs {
            g.tensor(id, TensorRole::Weight, elements).unwrap();
        }
        g.tensor(R, TensorRole::Input, out).unwrap();
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
    /// element count of its output.
    type Heavy = (Vec<(TensorId, u64)>, OpKind, u64);

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
            })
        };
        let conv_out = u64::from(h * w * cout);
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
        vec![
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
                }),
                u64::from(rows * c),
            ),
            (
                vec![(X, u64::from(m * k)), (W, u64::from(k * n))],
                mm(None),
                u64::from(m * n),
            ),
            (
                vec![(X, u64::from(m * k)), (W, u64::from(k * n)), (P, n.into())],
                mm(Some(P)),
                u64::from(m * n),
            ),
        ]
    }

    #[test]
    fn every_generated_kernel_parses_and_validates_with_and_without_epilogues() {
        let mut names = std::collections::BTreeSet::new();
        for (inputs, heavy, out) in heavies() {
            let elements = u32::try_from(out).unwrap();
            for steps in epilogues(elements) {
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
            "map+relu",
            "map+gelu",
            "map+scale",
            "map+add",
        ] {
            assert!(names.contains(name), "{name} in {names:?}");
        }
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
        let steps = epilogues(u32::try_from(out).unwrap()).remove(5);
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
