//! Graphs of models, built from their inventories: the last step of the
//! architecture table's row for model files ([crate](crate#architecture)).
//!
//! [`load`](crate::load) says which tensors a model's file holds and what each
//! is for. This module builds the [kernel IR](beatlm_core::kernel) graph that
//! computes with them, and says for every weight of the graph where its
//! numbers come from: which tensors of the file, by role, and how they are
//! rearranged into the IR's layouts. [`Builder`] is the part every model
//! shares; one module per model writes the model's structure with it:
//! [`depth_anything_v2`] for Depth Anything V2.
//!
//! Its graph is built in program order, and lowering fuses only neighbouring
//! operations ([`kernel`](beatlm_core::kernel#rules), rule 2), so a model
//! module pushes each operation right before the one that consumes it where
//! the two are meant to share a kernel.
//!
//! # Rules
//!
//! - **A weight is asked for by role, never by name.** A [`Source`] lists
//!   roles; [`Weight::numbers`] resolves them through the model's
//!   [`Inventory`], which owns the names. A graph is therefore built without a
//!   file, planned and checked before any weight is read, and given random
//!   numbers by a test that has no file at all.
//! - **Layouts change here, once, on the host.** Files store PyTorch's
//!   layouts (a linear layer's weight `[out][in]`, a convolution's
//!   `[out][in][kh][kw]`); kernels read the IR's (`[in][out]`, HWIO; the
//!   [`kernel`](beatlm_core::kernel#layouts) documentation). [`Weight::numbers`]
//!   converts a tensor when it is read, as its [`Convert`] says, so no kernel
//!   knows a file's layout and none pays for a transpose in a frame.
//! - **Every tensor of the graph comes from the builder.** It hands out ids in
//!   the order tensors are declared, so the same model builds the same graph,
//!   with the same ids, every time.

pub mod depth_anything_v2;

use std::fmt::Debug;

use beatlm_core::kernel::{Graph, KernelError, OpKind, TensorRole};
use beatlm_core::memory::TensorId;
use thiserror::Error;

use crate::load::{Archive, Inventory, LoadError, StoredType};

/// Why a graph could not be built or its weights could not be read.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ModelError {
    /// The graph refused an operation or a tensor: a shape the IR cannot
    /// hold.
    #[error(transparent)]
    Kernel(#[from] KernelError),
    /// The file lacks a tensor the inventory names.
    #[error(transparent)]
    Load(#[from] LoadError),
    /// The model's configuration does not describe a model that can be built;
    /// the source is the model's own configuration error.
    #[error("the configuration does not describe a model that can be built")]
    Config(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// A configuration value does not fit the IR's 32-bit dimensions.
    #[error("`{field}` is {value}, more than a kernel dimension holds")]
    TooLarge {
        /// The configuration field.
        field: &'static str,
        /// Its value.
        value: usize,
    },
    /// A decoder was given another number of feature maps than its
    /// configuration has stages.
    #[error("the decoder has {expected} stages and was given {given} feature maps")]
    FeatureCount {
        /// The stages, one per entry of `out_indices`.
        expected: usize,
        /// The feature maps given.
        given: usize,
    },
    /// A weight's source names a role the inventory does not list.
    #[error("the inventory lists no tensor for role {role}")]
    UnknownRole {
        /// The role, as `Debug` prints it.
        role: String,
    },
    /// A tensor in the file has another stored type or shape than the
    /// inventory says; [`Inventory::check`] reports every such tensor at once.
    #[error(
        "tensor `{name}` is stored as {found_type} {found:?}, the inventory says F32 {expected:?}"
    )]
    Layout {
        /// The tensor's name.
        name: String,
        /// The shape the inventory requires.
        expected: Vec<usize>,
        /// The stored type in the file.
        found_type: StoredType,
        /// The shape in the file.
        found: Vec<usize>,
    },
    /// A weight's source converts a tensor whose shape the conversion does not
    /// apply to: a linear layer's weight that is not two-dimensional, or a
    /// convolution's that is not four-dimensional.
    #[error("tensor `{name}` of shape {shape:?} cannot be converted as {convert:?}")]
    Convert {
        /// The tensor's name.
        name: String,
        /// Its shape.
        shape: Vec<usize>,
        /// The conversion its source asks for.
        convert: Convert,
    },
    /// The tensors of a source cannot be put side by side, or do not add up
    /// to the weight's size: the graph and the inventory disagree.
    #[error("weight {}: its source gives {given} numbers for {elements}", id.get())]
    SourceSize {
        /// The weight.
        id: TensorId,
        /// What the graph declared.
        elements: u64,
        /// What the source gives.
        given: u64,
    },
}

/// How a tensor of a file is rearranged into the layout a kernel reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Convert {
    /// As stored: vectors, embeddings and class tokens, whose stored layout is
    /// the IR's (a leading dimension of one is dropped by reading the bytes as
    /// they are).
    AsStored,
    /// A linear layer's weight, `[out][in]` in the file, as the right operand
    /// of a [`MatMul`](beatlm_core::kernel::MatMul), `[in][out]`.
    Linear,
    /// A convolution's weight, `[out][in][kh][kw]` in the file, as HWIO,
    /// `[kh][kw][in][out]`.
    Conv,
    /// A transposed convolution's weight, which PyTorch stores the other way
    /// round, `[in][out][kh][kw]`, as HWIO, `[kh][kw][in][out]`. A square
    /// weight has the same shape either way, so only this conversion, not the
    /// shape, says which dimension is which.
    ConvTranspose,
}

/// Where a weight's numbers come from: the tensors of these roles, each
/// converted, then put side by side along their last dimension.
///
/// One role is one tensor. Several are how a graph fuses layers the file
/// stores apart: attention's query, key and value projections become one
/// matrix multiply whose right operand is the three converted weights, `[in]`
/// rows of `[out_q | out_k | out_v]`. Every part must then have the same
/// leading dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Source<R> {
    /// The roles, in the order their columns are laid side by side.
    pub parts: Vec<R>,
    /// How each part is rearranged.
    pub convert: Convert,
}

impl<R> Source<R> {
    /// The tensor of one role, converted as `convert` says.
    pub fn one(role: R, convert: Convert) -> Self {
        Self {
            parts: vec![role],
            convert,
        }
    }
}

/// A weight of a graph: its tensor and where its numbers come from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Weight<R> {
    /// The tensor of the graph.
    pub id: TensorId,
    /// Its element count, as declared in the graph.
    pub elements: u64,
    /// Where its numbers come from.
    pub source: Source<R>,
}

impl<R: PartialEq + Debug> Weight<R> {
    /// The weight's numbers in the IR's layout, read from `archive` by way of
    /// the names `inventory` gives the source's roles.
    ///
    /// # Errors
    ///
    /// A role the inventory lacks, a tensor missing from the file or stored
    /// with another type or shape than the inventory says, or parts that do
    /// not add up to the weight.
    pub fn numbers(
        &self,
        inventory: &Inventory<R>,
        archive: &Archive<'_>,
    ) -> Result<Vec<f32>, ModelError> {
        let mut parts = Vec::with_capacity(self.source.parts.len());
        for role in &self.source.parts {
            let spec = inventory
                .by_role(role)
                .ok_or_else(|| ModelError::UnknownRole {
                    role: format!("{role:?}"),
                })?;
            let tensor = archive.require(&spec.name)?;
            if tensor.dtype() != StoredType::F32 || tensor.shape() != spec.shape.as_slice() {
                return Err(ModelError::Layout {
                    name: spec.name.clone(),
                    expected: spec.shape.clone(),
                    found_type: tensor.dtype(),
                    found: tensor.shape().to_vec(),
                });
            }
            let values: Vec<f32> = tensor
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            let converted =
                convert(values, tensor.shape(), self.source.convert).ok_or_else(|| {
                    ModelError::Convert {
                        name: spec.name.clone(),
                        shape: tensor.shape().to_vec(),
                        convert: self.source.convert,
                    }
                })?;
            parts.push(converted);
        }
        let size_error = |given: u64| ModelError::SourceSize {
            id: self.id,
            elements: self.elements,
            given,
        };
        let given = parts
            .iter()
            .map(|(v, _)| v.len() as u64)
            .fold(0, u64::saturating_add);
        let numbers = side_by_side(parts).ok_or_else(|| size_error(given))?;
        if numbers.len() as u64 != self.elements {
            return Err(size_error(numbers.len() as u64));
        }
        Ok(numbers)
    }
}

/// `values` of `shape` rearranged as `how` says, with the length of the
/// result's last dimension; `None` for a shape the conversion does not apply
/// to, so a wrong conversion is an error rather than numbers read untransposed.
fn convert(values: Vec<f32>, shape: &[usize], how: Convert) -> Option<(Vec<f32>, usize)> {
    Some(match (how, shape) {
        (Convert::Linear, &[out, inp]) => {
            let mut t = vec![0.0; values.len()];
            for o in 0..out {
                for i in 0..inp {
                    t[i * out + o] = values[o * inp + i];
                }
            }
            (t, out)
        }
        (Convert::Conv, &[out, inp, kh, kw]) => {
            let mut t = vec![0.0; values.len()];
            for o in 0..out {
                for i in 0..inp {
                    for y in 0..kh {
                        for x in 0..kw {
                            t[((y * kw + x) * inp + i) * out + o] =
                                values[((o * inp + i) * kh + y) * kw + x];
                        }
                    }
                }
            }
            (t, out)
        }
        (Convert::ConvTranspose, &[inp, out, kh, kw]) => {
            let mut t = vec![0.0; values.len()];
            for i in 0..inp {
                for o in 0..out {
                    for y in 0..kh {
                        for x in 0..kw {
                            t[((y * kw + x) * inp + i) * out + o] =
                                values[((i * out + o) * kh + y) * kw + x];
                        }
                    }
                }
            }
            (t, out)
        }
        (Convert::AsStored, _) => {
            let cols = shape.last().copied().unwrap_or(1);
            (values, cols)
        }
        (Convert::Linear | Convert::Conv | Convert::ConvTranspose, _) => return None,
    })
}

/// The parts laid side by side along their last dimension: row `r` of the
/// result is row `r` of each part in turn. `None` when the parts have
/// different numbers of rows.
fn side_by_side(mut parts: Vec<(Vec<f32>, usize)>) -> Option<Vec<f32>> {
    if parts.len() == 1 {
        return parts.pop().map(|(only, _)| only);
    }
    let rows_of = |(v, cols): &(Vec<f32>, usize)| v.len().checked_div(*cols);
    let rows = rows_of(parts.first()?)?;
    if parts.iter().any(|p| rows_of(p) != Some(rows)) {
        return None;
    }
    let mut out = Vec::with_capacity(parts.iter().map(|(v, _)| v.len()).sum());
    for r in 0..rows {
        for (v, cols) in &parts {
            out.extend_from_slice(&v[r * cols..(r + 1) * cols]);
        }
    }
    Some(out)
}

/// A graph under construction, with the ids it has handed out and the source
/// of every weight it has declared.
///
/// `R` is the model's role type, the one its inventory is keyed by.
#[derive(Debug, Clone)]
pub struct Builder<R> {
    graph: Graph,
    next: u32,
    weights: Vec<Weight<R>>,
}

impl<R> Default for Builder<R> {
    fn default() -> Self {
        Self {
            graph: Graph::new(),
            next: 0,
            weights: Vec::new(),
        }
    }
}

impl<R> Builder<R> {
    /// An empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares a tensor of `role` and `elements` under the next id.
    ///
    /// # Errors
    ///
    /// What [`Graph::tensor`] refuses, or more ids than a `u32` counts.
    pub fn tensor(&mut self, role: TensorRole, elements: u64) -> Result<TensorId, KernelError> {
        let id = TensorId::new(self.next);
        self.graph.tensor(id, role, elements)?;
        self.next = self.next.checked_add(1).ok_or(KernelError::TooManyOps)?;
        Ok(id)
    }

    /// Declares an input of `elements`, written by the host before a run.
    ///
    /// # Errors
    ///
    /// As [`tensor`](Self::tensor).
    pub fn input(&mut self, elements: u64) -> Result<TensorId, KernelError> {
        self.tensor(TensorRole::Input, elements)
    }

    /// Declares a weight of `elements` whose numbers come from `source`.
    ///
    /// # Errors
    ///
    /// As [`tensor`](Self::tensor).
    pub fn weight(&mut self, source: Source<R>, elements: u64) -> Result<TensorId, KernelError> {
        let id = self.tensor(TensorRole::Weight, elements)?;
        self.weights.push(Weight {
            id,
            elements,
            source,
        });
        Ok(id)
    }

    /// Appends `kind`, writing a new tensor of `role` (an activation or an
    /// output) of the size the operation gives, and returns that tensor.
    ///
    /// # Errors
    ///
    /// What [`Graph::push`] refuses.
    pub fn op(&mut self, kind: OpKind, role: TensorRole) -> Result<TensorId, KernelError> {
        let output = self.tensor(role, kind.output_elements())?;
        self.graph.push(kind, output)?;
        Ok(output)
    }

    /// The graph so far.
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Every weight declared so far, in declaration order.
    pub fn weights(&self) -> &[Weight<R>] {
        &self.weights
    }

    /// The graph and its weights.
    pub fn finish(self) -> (Graph, Vec<Weight<R>>) {
        (self.graph, self.weights)
    }
}

/// `value` as a kernel dimension, or the error naming `field`.
fn dimension(field: &'static str, value: usize) -> Result<u32, ModelError> {
    u32::try_from(value).map_err(|_| ModelError::TooLarge { field, value })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn a_linear_weight_is_transposed_and_a_convolution_becomes_hwio() {
        // [out 2][in 3].
        let (t, cols) =
            convert(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], Convert::Linear).unwrap();
        assert_eq!((t, cols), (vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 2));
        // [out 2][in 1][2][1]: element (o, 0, y, 0) = 10 o + y.
        let (t, cols) = convert(vec![0.0, 1.0, 10.0, 11.0], &[2, 1, 2, 1], Convert::Conv).unwrap();
        // HWIO [2][1][1][2]: element (y, 0, 0, o).
        assert_eq!((t, cols), (vec![0.0, 10.0, 1.0, 11.0], 2));
        // A transposed convolution's [in 2][out 3][1][2]: element (i, o, 0, x)
        // = 100 i + 10 o + x. HWIO [1][2][2][3]: element (0, x, i, o).
        let stored: Vec<f32> = (0..2)
            .flat_map(|i| {
                (0..3).flat_map(move |o| (0..2).map(move |x| (100 * i + 10 * o + x) as f32))
            })
            .collect();
        let (t, cols) = convert(stored, &[2, 3, 1, 2], Convert::ConvTranspose).unwrap();
        let hwio: Vec<f32> = (0..2)
            .flat_map(|x| {
                (0..2).flat_map(move |i| (0..3).map(move |o| (100 * i + 10 * o + x) as f32))
            })
            .collect();
        assert_eq!((t, cols), (hwio, 3));
        assert_eq!(convert(vec![1.0; 6], &[2, 3], Convert::ConvTranspose), None);
        let (t, cols) = convert(vec![1.0, 2.0], &[1, 1, 2], Convert::AsStored).unwrap();
        assert_eq!((t, cols), (vec![1.0, 2.0], 2));
        // A linear weight of another rank is refused, not read untransposed.
        assert_eq!(convert(vec![1.0; 6], &[1, 2, 3], Convert::Linear), None);
        assert_eq!(convert(vec![1.0; 6], &[2, 3], Convert::Conv), None);
    }

    #[test]
    fn parts_are_laid_side_by_side_row_by_row() {
        let a = (vec![1.0, 2.0, 3.0, 4.0], 2);
        let b = (vec![5.0, 6.0], 1);
        assert_eq!(
            side_by_side(vec![a.clone(), b]),
            Some(vec![1.0, 2.0, 5.0, 3.0, 4.0, 6.0])
        );
        assert_eq!(side_by_side(vec![a.clone(), (vec![1.0; 3], 1)]), None);
        assert_eq!(side_by_side(vec![a.clone()]), Some(a.0));
    }

    #[test]
    fn the_builder_hands_out_ids_in_order_and_records_weights() {
        let mut b: Builder<u8> = Builder::new();
        let x = b.input(4).unwrap();
        let w = b.weight(Source::one(7, Convert::AsStored), 4).unwrap();
        let y = b
            .op(
                OpKind::Add {
                    a: x,
                    b: w,
                    elements: 4,
                },
                TensorRole::Output,
            )
            .unwrap();
        assert_eq!((x.get(), w.get(), y.get()), (0, 1, 2));
        assert_eq!(b.weights()[0].source.parts, vec![7]);
        let (graph, weights) = b.finish();
        assert_eq!(graph.ops().len(), 1);
        assert_eq!(weights.len(), 1);
    }
}
