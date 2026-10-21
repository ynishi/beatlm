//! Model files, translated into the tensors a graph is built from.
//!
//! This is the umbrella crate's row for model files in the
//! [architecture table](crate#architecture): a weight file is read here, checked
//! here, and nothing past this module sees its format. The core never does;
//! what leaves this module is names, shapes, element types and bytes.
//!
//! Three parts, in the order a model is loaded:
//!
//! 1. [`reader`] opens a safetensors file held in memory and exposes its
//!    tensors by name, without copying them ([`Archive`]).
//! 2. [`inventory`] holds what a model needs, as a list of named, shaped
//!    tensors with roles ([`Inventory`]), and compares a file with it
//!    ([`Inventory::check`]). It knows no model.
//! 3. One module per model derives that list from the model's configuration:
//!    [`depth_anything_v2`] for Depth Anything V2 Small.
//!
//! # Rules
//!
//! - **The inventory comes from the configuration, never from the file.** A
//!   list copied from a file agrees with that file by construction and so
//!   checks nothing. Derived from the layer count, widths and heads, it can
//!   disagree, and the disagreement is the finding. This module's owners are
//!   [`depth_anything_v2::inventory`] and [`Inventory::check`].
//! - **A check reports every difference.** A missing tensor, an unexpected one
//!   and a wrong shape or stored type are all listed, with the parameter
//!   counts of both sides ([`CheckReport`]), because a wrong file is wrong in
//!   a pattern that one error hides.
//! - **Bytes are borrowed, not copied.** Tensors point into the buffer the
//!   archive was parsed from ([`Tensor::data`]). The reader works on bytes
//!   rather than a mapped file because the workspace forbids `unsafe` and a
//!   browser has no file to map; see [`reader`].
//! - **The reader builds for the browser.** Only [`read_file`], which needs a
//!   file system, is absent on `wasm` targets; a browser host passes the bytes
//!   of a fetched response to [`Archive::parse`].
//!
//! # Getting the Depth Anything V2 Small weights
//!
//! The weights are third-party material (Apache-2.0) and are never in the
//! repository; `just setup-depth-anything` fetches them, pinned to a commit and
//! verified by hash, into the ignored `models/depth-anything-v2-small/`. The
//! script states the source, the terms and the notices. Tests that read the
//! real file are `#[ignore]`d and name that recipe when the file is absent.
//!
//! `just reference-depth-anything` runs the original `transformers`
//! implementation on a fixed input and writes the input, the four encoder
//! features the head reads and the predicted depth next to the weights, as raw
//! little-endian `f32` files with a JSON manifest; given a side other than the
//! default 518 (`just reference-depth-anything 266`), it does the same at that
//! input size into a directory of its own. It runs in an isolated
//! Python environment and nothing in the engine's build or tests requires it;
//! it is what the graph is compared with. The input is defined by a formula in
//! integers and single-precision arithmetic so that Rust computes the same
//! bits without reading a file. That formula, where the files are kept, the
//! error measures and the tolerances of a comparison are
//! [`depth_anything_v2::reference`], which the native tests and the browser
//! host both read.

pub mod depth_anything_v2;
mod error;
pub mod inventory;
pub mod reader;
#[cfg(test)]
mod testing;

pub use error::LoadError;
pub use inventory::{CheckReport, Inventory, Layout, Mismatch, TensorSpec};
#[cfg(not(target_family = "wasm"))]
pub use reader::read_file;
pub use reader::{Archive, StoredType, Tensor};
