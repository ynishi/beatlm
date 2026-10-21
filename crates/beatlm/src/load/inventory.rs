//! An inventory: the tensors a model's graph needs, and the check of a file
//! against it.
//!
//! An [`Inventory`] is a list of [`TensorSpec`]s: for each tensor a name, a
//! stored type, a shape, and a *role*, the graph's name for what the tensor is
//! for. The names and shapes are what the file must have; the roles are what
//! the code that builds the graph asks by, so it never parses a name. The role
//! type belongs to the model (see [`depth_anything_v2`](super::depth_anything_v2)),
//! which is why this module is generic over it: the check below is the same
//! for every model.
//!
//! # Why a check of its own
//!
//! A weight file that is almost right fails late and far from its cause: a
//! shape that is transposed or a layer that is missing shows up as a wrong
//! picture, or as a buffer binding error in the middle of a frame. Comparing a
//! file with the inventory before anything is built turns that into one
//! report, which names every disagreement at once. [`Inventory::check`] never
//! stops at the first, because a file is usually wrong in a pattern (a whole
//! layer missing, a renamed prefix) and the pattern is visible only in the
//! full list.

use std::collections::BTreeMap;
use std::fmt;

use super::error::LoadError;
use super::reader::{Archive, StoredType, element_count};

/// One tensor a model needs: what the file must hold, and what it is for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorSpec<R> {
    /// The tensor's name in the file.
    pub name: String,
    /// The type it must be stored in.
    pub dtype: StoredType,
    /// The shape it must have, outermost dimension first.
    pub shape: Vec<usize>,
    /// What the graph uses it for.
    pub role: R,
}

impl<R> TensorSpec<R> {
    /// The number of elements, saturating at `u64::MAX`.
    pub fn element_count(&self) -> u64 {
        element_count(&self.shape)
    }
}

/// The tensors a model needs, by name and by role.
#[derive(Debug, Clone)]
pub struct Inventory<R> {
    specs: Vec<TensorSpec<R>>,
    by_name: BTreeMap<String, usize>,
}

impl<R> Inventory<R> {
    /// Collects `specs`, keeping their order.
    ///
    /// # Errors
    ///
    /// [`LoadError::DuplicateName`] when two specs share a name.
    pub fn new(specs: Vec<TensorSpec<R>>) -> Result<Self, LoadError> {
        let mut by_name = BTreeMap::new();
        for (index, spec) in specs.iter().enumerate() {
            if by_name.insert(spec.name.clone(), index).is_some() {
                return Err(LoadError::DuplicateName {
                    name: spec.name.clone(),
                });
            }
        }
        Ok(Self { specs, by_name })
    }

    /// The number of tensors.
    pub fn len(&self) -> usize {
        self.specs.len()
    }

    /// Whether the inventory lists no tensor.
    pub fn is_empty(&self) -> bool {
        self.specs.is_empty()
    }

    /// Every spec, in the order the inventory was built in.
    pub fn iter(&self) -> impl Iterator<Item = &TensorSpec<R>> {
        self.specs.iter()
    }

    /// The spec for the tensor called `name`.
    pub fn get(&self, name: &str) -> Option<&TensorSpec<R>> {
        self.by_name.get(name).map(|&index| &self.specs[index])
    }

    /// The spec whose role is `role`, if any.
    ///
    /// A linear search: it serves building a graph once, not a frame loop.
    pub fn by_role(&self, role: &R) -> Option<&TensorSpec<R>>
    where
        R: PartialEq,
    {
        self.specs.iter().find(|spec| &spec.role == role)
    }

    /// The number of elements across all tensors, saturating at `u64::MAX`.
    pub fn parameter_count(&self) -> u64 {
        self.specs
            .iter()
            .fold(0u64, |acc, spec| acc.saturating_add(spec.element_count()))
    }

    /// Compares `archive` with this inventory and reports every difference.
    ///
    /// A tensor is *missing* when the inventory lists it and the file does
    /// not, *unexpected* when the file has it and the inventory does not, and
    /// *mismatched* when both have it but the stored type or the shape
    /// differs. Nothing is read from the tensors' bytes.
    pub fn check(&self, archive: &Archive<'_>) -> CheckReport {
        let mut missing = Vec::new();
        let mut mismatched = Vec::new();
        for spec in &self.specs {
            match archive.get(&spec.name) {
                None => missing.push(spec.name.clone()),
                Some(found) => {
                    if found.dtype() != spec.dtype || found.shape() != spec.shape.as_slice() {
                        mismatched.push(Mismatch {
                            name: spec.name.clone(),
                            expected: Layout {
                                dtype: spec.dtype,
                                shape: spec.shape.clone(),
                            },
                            found: Layout {
                                dtype: found.dtype(),
                                shape: found.shape().to_vec(),
                            },
                        });
                    }
                }
            }
        }
        let unexpected = archive
            .iter()
            .filter(|(name, _)| !self.by_name.contains_key(*name))
            .map(|(name, _)| name.to_owned())
            .collect();
        CheckReport {
            missing,
            unexpected,
            mismatched,
            expected_parameters: self.parameter_count(),
            file_parameters: archive.parameter_count(),
        }
    }
}

/// A stored type and a shape: what a tensor is, as far as a check can tell.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Layout {
    /// The stored type.
    pub dtype: StoredType,
    /// The shape, outermost dimension first.
    pub shape: Vec<usize>,
}

impl fmt::Display for Layout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {:?}", self.dtype, self.shape)
    }
}

/// A tensor both the inventory and the file have, with different layouts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mismatch {
    /// The tensor's name.
    pub name: String,
    /// What the inventory requires.
    pub expected: Layout,
    /// What the file holds.
    pub found: Layout,
}

/// Every difference between a file and an [`Inventory`], and the parameter
/// counts of both sides.
///
/// The counts are reported even when the lists are empty: a file that matches
/// exactly has the count the model is published with, and a count that
/// disagrees with the published one is the first sign of a wrong file.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct CheckReport {
    /// Names the inventory lists and the file lacks, in inventory order.
    pub missing: Vec<String>,
    /// Names the file has and the inventory does not list, in name order.
    pub unexpected: Vec<String>,
    /// Tensors whose stored type or shape differs, in inventory order.
    pub mismatched: Vec<Mismatch>,
    /// The elements the inventory's tensors add up to.
    pub expected_parameters: u64,
    /// The elements the file's tensors add up to.
    pub file_parameters: u64,
}

impl CheckReport {
    /// Whether the file has exactly the inventory's tensors, each with the
    /// stored type and shape the inventory requires.
    pub fn is_exact(&self) -> bool {
        self.missing.is_empty() && self.unexpected.is_empty() && self.mismatched.is_empty()
    }
}

impl fmt::Display for CheckReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_exact() {
            return write!(
                f,
                "the file matches the inventory exactly: {} parameters",
                self.file_parameters
            );
        }
        write!(
            f,
            "the file does not match the inventory: {} missing, {} unexpected, {} mismatched \
             (inventory {} parameters, file {})",
            self.missing.len(),
            self.unexpected.len(),
            self.mismatched.len(),
            self.expected_parameters,
            self.file_parameters,
        )?;
        for name in &self.missing {
            write!(f, "\n  missing    {name}")?;
        }
        for name in &self.unexpected {
            write!(f, "\n  unexpected {name}")?;
        }
        for mismatch in &self.mismatched {
            write!(
                f,
                "\n  mismatch   {}: expected {}, found {}",
                mismatch.name, mismatch.expected, mismatch.found
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::load::testing::{f32_bytes, file_of};
    use safetensors::Dtype;

    fn spec(name: &str, shape: &[usize]) -> TensorSpec<&'static str> {
        TensorSpec {
            name: name.to_owned(),
            dtype: StoredType::F32,
            shape: shape.to_vec(),
            role: "r",
        }
    }

    fn f32_entry(name: &'static str, shape: &[usize]) -> crate::load::testing::Entry<'static> {
        let n: usize = shape.iter().product();
        (name, Dtype::F32, shape.to_vec(), f32_bytes(&vec![0.0; n]))
    }

    fn inventory() -> Inventory<&'static str> {
        Inventory::new(vec![spec("a", &[2, 3]), spec("b", &[4])]).expect("names are distinct")
    }

    #[test]
    fn a_file_with_exactly_the_tensors_is_exact() {
        let file = file_of(&[f32_entry("a", &[2, 3]), f32_entry("b", &[4])]);
        let report = inventory().check(&Archive::parse(&file).expect("parses"));
        assert!(report.is_exact(), "{report}");
        assert_eq!(
            (report.expected_parameters, report.file_parameters),
            (10, 10)
        );
    }

    #[test]
    fn every_difference_is_reported_not_just_the_first() {
        let inv = Inventory::new(vec![
            spec("gone", &[1]),
            spec("also_gone", &[1]),
            spec("reshaped", &[2, 3]),
            spec("retyped", &[2]),
            spec("fine", &[5]),
        ])
        .expect("names are distinct");
        let file = file_of(&[
            f32_entry("reshaped", &[3, 2]),
            ("retyped", Dtype::F16, vec![2], vec![0; 4]),
            f32_entry("fine", &[5]),
            f32_entry("extra", &[7]),
            f32_entry("another_extra", &[1]),
        ]);

        let report = inv.check(&Archive::parse(&file).expect("parses"));

        assert!(!report.is_exact());
        assert_eq!(report.missing, ["gone", "also_gone"]);
        assert_eq!(report.unexpected, ["another_extra", "extra"]);
        assert_eq!(report.mismatched.len(), 2);
        assert_eq!(report.mismatched[0].name, "reshaped");
        assert_eq!(report.mismatched[0].expected.shape, [2, 3]);
        assert_eq!(report.mismatched[0].found.shape, [3, 2]);
        assert_eq!(report.mismatched[1].name, "retyped");
        assert_eq!(report.mismatched[1].found.dtype, StoredType::F16);
        assert_eq!(report.expected_parameters, 1 + 1 + 6 + 2 + 5);
        assert_eq!(report.file_parameters, 6 + 2 + 5 + 7 + 1);
    }

    #[test]
    fn the_rendered_report_names_every_difference() {
        let file = file_of(&[f32_entry("a", &[3, 2]), f32_entry("stray", &[1])]);
        let report = inventory().check(&Archive::parse(&file).expect("parses"));
        let text = report.to_string();
        assert!(text.contains("missing    b"), "{text}");
        assert!(text.contains("unexpected stray"), "{text}");
        assert!(
            text.contains("mismatch   a: expected F32 [2, 3], found F32 [3, 2]"),
            "{text}"
        );
    }

    #[test]
    fn a_repeated_name_is_refused() {
        let err = Inventory::new(vec![spec("a", &[1]), spec("a", &[2])])
            .expect_err("two specs share a name");
        assert!(matches!(err, LoadError::DuplicateName { name } if name == "a"));
    }

    #[test]
    fn specs_are_found_by_name_and_by_role() {
        let inv = Inventory::new(vec![
            TensorSpec {
                name: "x".to_owned(),
                dtype: StoredType::F32,
                shape: vec![1],
                role: 1u8,
            },
            TensorSpec {
                name: "y".to_owned(),
                dtype: StoredType::F32,
                shape: vec![2],
                role: 2u8,
            },
        ])
        .expect("names are distinct");
        assert_eq!(inv.get("y").map(|s| s.role), Some(2));
        assert_eq!(inv.by_role(&1).map(|s| s.name.as_str()), Some("x"));
        assert!(inv.get("z").is_none());
        assert!(inv.by_role(&3).is_none());
    }
}
