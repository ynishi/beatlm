//! Helpers the tests of this module share: small safetensors files written by
//! the test itself, so no test needs a file from disk.

#![allow(clippy::unwrap_used, clippy::expect_used)]

use safetensors::Dtype;
use safetensors::tensor::TensorView;

/// One tensor to put in a test file: name, element type, shape, bytes.
pub(crate) type Entry<'a> = (&'a str, Dtype, Vec<usize>, Vec<u8>);

/// Serialises `entries` as a safetensors file.
pub(crate) fn file_of(entries: &[Entry<'_>]) -> Vec<u8> {
    let views: Vec<(String, TensorView<'_>)> = entries
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            let view = TensorView::new(*dtype, shape.clone(), bytes)
                .expect("the test's bytes match its shape");
            (name.to_string(), view)
        })
        .collect();
    safetensors::serialize(views, None).expect("the test's tensors serialise")
}

/// `values` as little-endian `f32` bytes.
pub(crate) fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}
