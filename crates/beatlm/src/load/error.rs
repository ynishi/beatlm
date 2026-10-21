//! The one error type of the [`load`](super) module.

use thiserror::Error;

/// Why a model file could not be read, or a model could not be described.
///
/// One type for the module because a caller loading a model meets these in one
/// sequence (read, parse, check) and handles them together; the variants say
/// which step failed.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum LoadError {
    /// The bytes are not a valid safetensors file: the header is malformed, or
    /// disagrees with the size of the buffer.
    ///
    /// The cause is boxed so that the parser behind the reader does not appear
    /// in this crate's public types.
    #[error("not a valid safetensors file")]
    Format(#[source] Box<dyn std::error::Error + Send + Sync>),
    /// The file could not be read from disk.
    #[cfg(not(target_family = "wasm"))]
    #[error("cannot read {}", path.display())]
    Io {
        /// The path that was read.
        path: std::path::PathBuf,
        /// What the operating system reported.
        #[source]
        source: std::io::Error,
    },
    /// A tensor the caller needs is not in the file.
    #[error("tensor `{name}` is not in the file")]
    MissingTensor {
        /// The name that was asked for.
        name: String,
    },
    /// Two entries of one inventory have the same tensor name, so the name
    /// would not say which of them a file's tensor is.
    #[error("tensor name `{name}` appears twice in the inventory")]
    DuplicateName {
        /// The repeated name.
        name: String,
    },
}
