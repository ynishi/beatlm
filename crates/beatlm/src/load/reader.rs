//! The safetensors reader: a file's bytes as named, typed, shaped tensors.
//!
//! [`Archive::parse`] reads the header of a safetensors file and hands out
//! every tensor as a [`Tensor`] that borrows its bytes from the buffer it was
//! given. Nothing is copied: the only full-size allocation on the way from
//! disk to a tensor is the buffer itself, which [`read_file`] fills with one
//! read.
//!
//! # Why bytes, not a memory map
//!
//! The reader takes `&[u8]` and does not map the file. A memory map needs
//! `unsafe`, which the workspace forbids, and it does not exist in a browser,
//! where the bytes arrive from a `fetch` as one buffer. A reader over bytes
//! serves both, and the cost on native is one read of the file into memory,
//! which the weights need anyway on their way to the GPU.
//!
//! # What is checked
//!
//! The header's own consistency (offsets inside the buffer, in order, covering
//! it exactly, with shapes whose byte size matches the span) is validated by
//! the `safetensors` crate before any tensor is handed out, and a failure is
//! [`LoadError::Format`]. What this module adds is a vocabulary that does not
//! expose that crate: [`StoredType`] and [`Tensor`] are this crate's, so the parser
//! can change without a change to the callers.

use std::collections::BTreeMap;

use super::error::LoadError;

/// The element type a tensor is stored in.
///
/// This is the file's type, not the engine's: [`beatlm_core::memory::DType`] is
/// what a plan allocates and a kernel computes in, and a file may store a type
/// the engine does not compute in. [`StoredType::to_planned`] is the one place
/// the two meet.
///
/// Only the types a model's weights are stored in have a name; any other
/// element type the format can hold is [`StoredType::Other`], which compares
/// equal to no weight type and so surfaces as a type mismatch when a file is
/// checked against an inventory instead of being guessed at.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum StoredType {
    /// 32-bit IEEE 754 float, little endian.
    F32,
    /// 16-bit IEEE 754 float, little endian.
    F16,
    /// 16-bit brain float, little endian.
    Bf16,
    /// 64-bit IEEE 754 float, little endian.
    F64,
    /// 8-bit unsigned integer.
    U8,
    /// 32-bit signed integer, little endian.
    I32,
    /// 64-bit signed integer, little endian.
    I64,
    /// An element type the format has and this reader gives no name.
    Other,
}

impl StoredType {
    /// The size of one element in bytes, or `None` for [`StoredType::Other`],
    /// whose width this reader does not know.
    pub fn size_bytes(self) -> Option<usize> {
        match self {
            Self::F32 | Self::I32 => Some(4),
            Self::F16 | Self::Bf16 => Some(2),
            Self::F64 | Self::I64 => Some(8),
            Self::U8 => Some(1),
            Self::Other => None,
        }
    }

    /// The type the engine plans for a tensor stored as this one, when it
    /// plans for it as stored.
    ///
    /// `None` for every other type: converting `BF16` or `F64` to a computed
    /// type is a decision about precision, made where the graph is built, and
    /// not a default this reader may take.
    pub fn to_planned(self) -> Option<beatlm_core::memory::DType> {
        match self {
            Self::F32 => Some(beatlm_core::memory::DType::F32),
            Self::F16 => Some(beatlm_core::memory::DType::F16),
            _ => None,
        }
    }

    fn from_format(dtype: safetensors::Dtype) -> Self {
        use safetensors::Dtype;
        match dtype {
            Dtype::F32 => Self::F32,
            Dtype::F16 => Self::F16,
            Dtype::BF16 => Self::Bf16,
            Dtype::F64 => Self::F64,
            Dtype::U8 => Self::U8,
            Dtype::I32 => Self::I32,
            Dtype::I64 => Self::I64,
            _ => Self::Other,
        }
    }
}

impl std::fmt::Display for StoredType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::Bf16 => "BF16",
            Self::F64 => "F64",
            Self::U8 => "U8",
            Self::I32 => "I32",
            Self::I64 => "I64",
            Self::Other => "other",
        })
    }
}

/// One tensor of an [`Archive`]: its element type, its shape, and its bytes.
///
/// The bytes are the stored ones, row-major, in the file's byte order, and
/// borrow from the buffer the archive was parsed from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tensor<'a> {
    dtype: StoredType,
    shape: Vec<usize>,
    data: &'a [u8],
}

impl<'a> Tensor<'a> {
    /// The element type.
    pub fn dtype(&self) -> StoredType {
        self.dtype
    }

    /// The extent of each dimension, outermost first.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The stored bytes, borrowed from the buffer the archive was parsed from.
    pub fn data(&self) -> &'a [u8] {
        self.data
    }

    /// The number of elements, saturating at `u64::MAX`.
    ///
    /// A shape whose element count does not fit in 64 bits cannot come from a
    /// file the parser accepted, since the header's byte spans must fit in the
    /// buffer; saturating keeps this total instead of panicking on a value
    /// that cannot occur.
    pub fn element_count(&self) -> u64 {
        element_count(&self.shape)
    }
}

/// The product of a shape's extents, saturating at `u64::MAX`; the empty shape
/// is a scalar of one element.
pub(crate) fn element_count(shape: &[usize]) -> u64 {
    shape
        .iter()
        .fold(1u64, |acc, &extent| acc.saturating_mul(extent as u64))
}

/// The tensors of one safetensors file, by name.
///
/// Iteration is in name order, so a report built from an archive reads the
/// same on every run.
#[derive(Debug, Clone)]
pub struct Archive<'a> {
    tensors: BTreeMap<String, Tensor<'a>>,
}

impl<'a> Archive<'a> {
    /// Parses the whole file held in `bytes`.
    ///
    /// The returned tensors borrow from `bytes`, so the buffer outlives them.
    ///
    /// # Errors
    ///
    /// [`LoadError::Format`] when the header is malformed or inconsistent with
    /// the buffer.
    pub fn parse(bytes: &'a [u8]) -> Result<Self, LoadError> {
        let parsed = safetensors::SafeTensors::deserialize(bytes)
            .map_err(|source| LoadError::Format(Box::new(source)))?;
        let tensors = parsed
            .iter()
            .map(|(name, view)| {
                let tensor = Tensor {
                    dtype: StoredType::from_format(view.dtype()),
                    shape: view.shape().to_vec(),
                    data: view.data(),
                };
                (name.to_owned(), tensor)
            })
            .collect();
        Ok(Self { tensors })
    }

    /// The number of tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// Whether the file holds no tensor.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// The tensor called `name`, if the file has one.
    pub fn get(&self, name: &str) -> Option<&Tensor<'a>> {
        self.tensors.get(name)
    }

    /// The tensor called `name`.
    ///
    /// # Errors
    ///
    /// [`LoadError::MissingTensor`] when the file has no such tensor.
    pub fn require(&self, name: &str) -> Result<&Tensor<'a>, LoadError> {
        self.get(name).ok_or_else(|| LoadError::MissingTensor {
            name: name.to_owned(),
        })
    }

    /// Every tensor with its name, in name order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Tensor<'a>)> {
        self.tensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), tensor))
    }

    /// The number of elements across all tensors, saturating at `u64::MAX`.
    pub fn parameter_count(&self) -> u64 {
        self.tensors.values().fold(0u64, |acc, tensor| {
            acc.saturating_add(tensor.element_count())
        })
    }
}

/// Reads a whole file into memory.
///
/// One allocation of the file's size and one read: the buffer [`Archive::parse`]
/// borrows from. Not available on `wasm` targets, which have no file system;
/// a browser host passes the bytes of a fetched response to [`Archive::parse`]
/// instead.
///
/// # Errors
///
/// [`LoadError::Io`] when the file cannot be read.
#[cfg(not(target_family = "wasm"))]
pub fn read_file(path: impl AsRef<std::path::Path>) -> Result<Vec<u8>, LoadError> {
    let path = path.as_ref();
    std::fs::read(path).map_err(|source| LoadError::Io {
        path: path.to_path_buf(),
        source,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::load::testing::{f32_bytes, file_of};
    use safetensors::Dtype;

    #[test]
    fn tensors_come_back_with_name_dtype_shape_and_bytes() {
        let weights = f32_bytes(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let ids: Vec<u8> = vec![7, 8, 9];
        let file = file_of(&[
            ("w", Dtype::F32, vec![2, 3], weights.clone()),
            ("ids", Dtype::U8, vec![3], ids.clone()),
        ]);

        let archive = Archive::parse(&file).expect("a file the test wrote parses");

        assert_eq!(archive.len(), 2);
        let w = archive.get("w").expect("w is in the file");
        assert_eq!((w.dtype(), w.shape()), (StoredType::F32, &[2usize, 3][..]));
        assert_eq!(w.data(), weights.as_slice());
        assert_eq!(w.element_count(), 6);
        let i = archive.get("ids").expect("ids is in the file");
        assert_eq!((i.dtype(), i.shape()), (StoredType::U8, &[3usize][..]));
        assert_eq!(i.data(), ids.as_slice());
    }

    #[test]
    fn iteration_is_in_name_order() {
        let file = file_of(&[
            ("b", Dtype::F32, vec![1], f32_bytes(&[0.0])),
            ("a", Dtype::F32, vec![1], f32_bytes(&[0.0])),
            ("c", Dtype::F32, vec![1], f32_bytes(&[0.0])),
        ]);
        let archive = Archive::parse(&file).expect("parses");
        let names: Vec<&str> = archive.iter().map(|(name, _)| name).collect();
        assert_eq!(names, ["a", "b", "c"]);
    }

    #[test]
    fn tensor_bytes_borrow_from_the_buffer() {
        let file = file_of(&[("w", Dtype::F32, vec![4], f32_bytes(&[1.0; 4]))]);
        let archive = Archive::parse(&file).expect("parses");
        let data = archive.get("w").expect("w is in the file").data();

        let buffer = file.as_ptr_range();
        let span = data.as_ptr_range();
        assert!(buffer.start <= span.start && span.end <= buffer.end);
    }

    #[test]
    fn parameter_count_sums_every_element() {
        let file = file_of(&[
            ("a", Dtype::F32, vec![2, 3], f32_bytes(&[0.0; 6])),
            ("b", Dtype::F32, vec![4], f32_bytes(&[0.0; 4])),
            ("scalar", Dtype::F32, vec![], f32_bytes(&[0.0])),
        ]);
        let archive = Archive::parse(&file).expect("parses");
        assert_eq!(archive.parameter_count(), 11);
    }

    #[test]
    fn a_dtype_without_a_name_is_other() {
        let file = file_of(&[("h", Dtype::I16, vec![2], vec![0; 4])]);
        let archive = Archive::parse(&file).expect("parses");
        assert_eq!(archive.get("h").expect("h").dtype(), StoredType::Other);
    }

    #[test]
    fn a_missing_tensor_is_an_error_from_require_and_none_from_get() {
        let file = file_of(&[("a", Dtype::F32, vec![1], f32_bytes(&[0.0]))]);
        let archive = Archive::parse(&file).expect("parses");
        assert!(archive.get("nope").is_none());
        assert!(matches!(
            archive.require("nope"),
            Err(LoadError::MissingTensor { name }) if name == "nope"
        ));
    }

    #[test]
    fn a_truncated_file_is_a_format_error() {
        let file = file_of(&[("a", Dtype::F32, vec![4], f32_bytes(&[1.0; 4]))]);
        let cut = &file[..file.len() - 1];
        assert!(matches!(Archive::parse(cut), Err(LoadError::Format(_))));
    }

    #[test]
    fn bytes_that_are_not_a_file_are_a_format_error() {
        assert!(matches!(Archive::parse(&[]), Err(LoadError::Format(_))));
        assert!(matches!(
            Archive::parse(&[0xff; 64]),
            Err(LoadError::Format(_))
        ));
    }

    #[test]
    fn an_empty_archive_is_valid() {
        let file = file_of(&[]);
        let archive = Archive::parse(&file).expect("parses");
        assert!(archive.is_empty());
        assert_eq!(archive.parameter_count(), 0);
    }

    #[test]
    fn element_widths_are_known_for_named_types_only() {
        assert_eq!(StoredType::F32.size_bytes(), Some(4));
        assert_eq!(StoredType::Bf16.size_bytes(), Some(2));
        assert_eq!(StoredType::Other.size_bytes(), None);
    }

    #[test]
    fn only_types_the_engine_plans_as_stored_convert() {
        use beatlm_core::memory::DType;
        assert_eq!(StoredType::F32.to_planned(), Some(DType::F32));
        assert_eq!(StoredType::F16.to_planned(), Some(DType::F16));
        assert_eq!(StoredType::Bf16.to_planned(), None);
        assert_eq!(StoredType::F64.to_planned(), None);
    }

    #[cfg(not(target_family = "wasm"))]
    #[test]
    fn an_unreadable_path_is_an_io_error_naming_the_path() {
        let err =
            read_file("/definitely/not/a/file.safetensors").expect_err("the path does not exist");
        assert!(matches!(&err, LoadError::Io { path, .. }
            if path.ends_with("file.safetensors")));
    }
}
