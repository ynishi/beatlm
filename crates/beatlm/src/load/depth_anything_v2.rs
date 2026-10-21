//! The tensors of Depth Anything V2, derived from its configuration.
//!
//! [`inventory`] lists every tensor the model's graph reads, with the shape it
//! must have and a [`Role`] saying what it is for. The list is computed from a
//! [`Config`], the way the model itself builds its layers from its
//! configuration, not copied from a file: so a file that disagrees with the
//! configuration is a finding, which [`Inventory::check`] reports, and a graph
//! builder asks for tensors by role instead of parsing names.
//!
//! # The model
//!
//! A DINOv2 vision transformer (patch size 14; for the Small variant hidden
//! size 384, 12 layers, 6 heads, an MLP of four times the hidden size,
//! LayerScale, layer-norm epsilon 1e-6) followed by a DPT head. The head takes
//! the transformer's output after four of its layers (`out_indices`), each
//! through the final layer norm with the class token dropped, and for each:
//!
//! 1. **reassembles** it: a 1×1 convolution to that stage's channel count, then
//!    a resampling ([`Resample`]) to the stage's resolution;
//! 2. brings it to the fusion width with a 3×3 convolution without bias;
//! 3. **fuses** the stages from the deepest to the shallowest, each fusion
//!    layer made of two pre-activation residual units and a 1×1 projection;
//! 4. and a three-convolution **head** turns the fused map into one channel of
//!    relative depth.
//!
//! This module describes the tensors of that structure and not its arithmetic;
//! building the graph from them is a separate step that consumes this
//! inventory.
//!
//! # Where the names come from
//!
//! The names are those the Small checkpoint
//! (`depth-anything/Depth-Anything-V2-Small-hf`, the repository the setup
//! recipe fetches) stores, which are the module paths of the `transformers`
//! implementation `DepthAnythingForDepthEstimation` at the time the checkpoint
//! was saved. They were read from, at the tags named:
//!
//! - `transformers` v5.18.0, `models/depth_anything/modeling_depth_anything.py`
//!   and `configuration_depth_anything.py`: the neck, fusion and head module
//!   names and shapes, and the configuration fields;
//! - `transformers` v5.18.0, `models/dinov2/modeling_dinov2.py` and
//!   `configuration_dinov2.py`: the backbone's modules and the configuration
//!   defaults (`layer_norm_eps` 1e-6, `qkv_bias`, `mlp_ratio` 4, the mask
//!   token);
//! - `transformers` v4.57.0, `models/dinov2/modeling_dinov2.py`: the
//!   attention's module names, `attention.attention.{query,key,value}` and
//!   `attention.output.dense`.
//!
//! The last is the reason for two tags. Version 5 renamed those modules to
//! `q_proj`, `k_proj`, `v_proj` and `o_proj`, and renames the stored names to
//! them when it loads a checkpoint (its `conversion_mapping.py` has the
//! rules). The file keeps the names it was saved with, so the inventory uses
//! the older ones: reading the v5 module definitions alone gives names that no
//! file of this model contains. The module paths of the neck and the head are
//! the same in both versions.
//!
//! The Small checkpoint's configuration (`config.json` at the pinned revision)
//! is [`Config::small`]. A test that reads the real weights file asserts that
//! this inventory matches it exactly; fields no tensor's shape depends on
//! (`out_indices`) are not checked by that test, and the reference script
//! checks `out_indices` against `config.json` when it runs.

use thiserror::Error;

use super::error::LoadError;
use super::inventory::{Inventory, TensorSpec};
use super::reader::StoredType;

/// How a stage's reassembled map is resampled to its resolution.
///
/// The resolution is relative to the transformer's patch grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Resample {
    /// Enlarge by an integer factor with a transposed convolution whose kernel
    /// size and stride are the factor.
    Up(usize),
    /// Keep the patch grid; the stage has no resampling tensors.
    Keep,
    /// Shrink by an integer factor with a strided 3×3 convolution.
    Down(usize),
}

/// The structure of a Depth Anything V2 model.
///
/// The fields are those of the `transformers` configuration the checkpoint was
/// published with, in the types the inventory needs. Start from
/// [`Config::small`] and change a field to describe another size.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct Config {
    /// Side of the square input the position embeddings are stored for, in
    /// pixels.
    pub image_size: usize,
    /// Side of one patch, in pixels.
    pub patch_size: usize,
    /// Channels of the input image.
    pub num_channels: usize,
    /// Width of the transformer.
    pub hidden_size: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention heads; the hidden size is split among them.
    pub num_heads: usize,
    /// The MLP's hidden width as a multiple of the hidden size.
    pub mlp_ratio: usize,
    /// Whether the embeddings hold a mask token. It is a stored parameter even
    /// though inference never reads it.
    pub mask_token: bool,
    /// The 1-based layers whose output the DPT head takes, shallowest first.
    pub out_indices: Vec<usize>,
    /// The channel count of each stage after reassembly, one per entry of
    /// `out_indices`.
    pub neck_hidden_sizes: Vec<usize>,
    /// How each stage is resampled, one per entry of `out_indices`.
    pub reassemble: Vec<Resample>,
    /// The width the stages are fused at.
    pub fusion_hidden_size: usize,
    /// The width of the head's second convolution.
    pub head_hidden_size: usize,
}

impl Config {
    /// The configuration of `depth-anything/Depth-Anything-V2-Small-hf`.
    pub fn small() -> Self {
        Self {
            image_size: 518,
            patch_size: 14,
            num_channels: 3,
            hidden_size: 384,
            num_layers: 12,
            num_heads: 6,
            mlp_ratio: 4,
            mask_token: true,
            out_indices: vec![3, 6, 9, 12],
            neck_hidden_sizes: vec![48, 96, 192, 384],
            reassemble: vec![
                Resample::Up(4),
                Resample::Up(2),
                Resample::Keep,
                Resample::Down(2),
            ],
            fusion_hidden_size: 64,
            head_hidden_size: 32,
        }
    }

    /// The side of the patch grid, in patches.
    pub fn grid(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// The transformer's sequence length: one token per patch, and the class
    /// token.
    pub fn tokens(&self) -> usize {
        self.grid() * self.grid() + 1
    }

    /// The MLP's hidden width.
    pub fn mlp_size(&self) -> usize {
        self.hidden_size * self.mlp_ratio
    }

    /// Checks that the fields describe a model that can be built.
    ///
    /// # Errors
    ///
    /// The [`ConfigError`] naming the first field that cannot be right.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let sizes = [
            ("image_size", self.image_size),
            ("patch_size", self.patch_size),
            ("num_channels", self.num_channels),
            ("hidden_size", self.hidden_size),
            ("num_layers", self.num_layers),
            ("num_heads", self.num_heads),
            ("mlp_ratio", self.mlp_ratio),
            ("fusion_hidden_size", self.fusion_hidden_size),
            ("head_hidden_size", self.head_hidden_size),
        ];
        if let Some(&(field, _)) = sizes.iter().find(|(_, value)| *value == 0) {
            return Err(ConfigError::Zero { field });
        }
        if !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(ConfigError::HeadsDoNotDivideHidden {
                hidden_size: self.hidden_size,
                num_heads: self.num_heads,
            });
        }
        if !self.image_size.is_multiple_of(self.patch_size) {
            return Err(ConfigError::ImageNotWholePatches {
                image_size: self.image_size,
                patch_size: self.patch_size,
            });
        }
        if !self.fusion_hidden_size.is_multiple_of(2) {
            return Err(ConfigError::FusionWidthOdd {
                fusion_hidden_size: self.fusion_hidden_size,
            });
        }
        let stages = self.out_indices.len();
        if stages == 0 || self.neck_hidden_sizes.len() != stages || self.reassemble.len() != stages
        {
            return Err(ConfigError::StageCountMismatch {
                out_indices: stages,
                neck_hidden_sizes: self.neck_hidden_sizes.len(),
                reassemble: self.reassemble.len(),
            });
        }
        if self.neck_hidden_sizes.contains(&0) {
            return Err(ConfigError::Zero {
                field: "neck_hidden_sizes",
            });
        }
        if self
            .reassemble
            .iter()
            .any(|r| matches!(r, Resample::Up(0) | Resample::Down(0)))
        {
            return Err(ConfigError::Zero {
                field: "reassemble",
            });
        }
        // `transformers` builds a factor of one as an identity with no tensors,
        // which is `Keep`; `Up(1)` or `Down(1)` would list tensors no file has.
        if let Some(stage) = self
            .reassemble
            .iter()
            .position(|r| matches!(r, Resample::Up(1) | Resample::Down(1)))
        {
            return Err(ConfigError::ResampleByOne { stage });
        }
        if let Some(&index) = self
            .out_indices
            .iter()
            .find(|&&i| i == 0 || i > self.num_layers)
        {
            return Err(ConfigError::OutIndexOutOfRange {
                index,
                num_layers: self.num_layers,
            });
        }
        if !self.out_indices.windows(2).all(|pair| pair[0] < pair[1]) {
            return Err(ConfigError::OutIndicesNotIncreasing);
        }
        Ok(())
    }
}

/// Why a [`Config`] cannot be built into an inventory.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ConfigError {
    /// A size that must be positive is zero.
    #[error("`{field}` must be positive")]
    Zero {
        /// The field.
        field: &'static str,
    },
    /// The hidden size cannot be split evenly among the heads.
    #[error("hidden size {hidden_size} is not divisible by {num_heads} heads")]
    HeadsDoNotDivideHidden {
        /// The hidden size.
        hidden_size: usize,
        /// The number of heads.
        num_heads: usize,
    },
    /// The image does not divide into whole patches, so the position
    /// embeddings have no grid.
    #[error("image size {image_size} is not a multiple of patch size {patch_size}")]
    ImageNotWholePatches {
        /// The image side.
        image_size: usize,
        /// The patch side.
        patch_size: usize,
    },
    /// The head's first convolution halves the fusion width.
    #[error("fusion width {fusion_hidden_size} is odd, and the head halves it")]
    FusionWidthOdd {
        /// The fusion width.
        fusion_hidden_size: usize,
    },
    /// The per-stage lists disagree on how many stages there are, or there are
    /// none.
    #[error(
        "stage lists disagree: {out_indices} out indices, {neck_hidden_sizes} neck sizes, \
         {reassemble} resamplings"
    )]
    StageCountMismatch {
        /// Length of `out_indices`.
        out_indices: usize,
        /// Length of `neck_hidden_sizes`.
        neck_hidden_sizes: usize,
        /// Length of `reassemble`.
        reassemble: usize,
    },
    /// A stage reads a layer the transformer does not have; layers count from
    /// one.
    #[error("out index {index} is not a layer of a {num_layers}-layer transformer")]
    OutIndexOutOfRange {
        /// The index.
        index: usize,
        /// The number of layers.
        num_layers: usize,
    },
    /// The stages do not read layers shallowest first, each once.
    #[error("out indices are not strictly increasing")]
    OutIndicesNotIncreasing,
    /// A stage resamples by a factor of one, which the model builds as no
    /// resampling at all; that stage is [`Resample::Keep`].
    #[error("stage {stage} resamples by a factor of one; use `Resample::Keep`")]
    ResampleByOne {
        /// The stage, from zero.
        stage: usize,
    },
    /// The inventory built from the configuration is not well formed.
    #[error("the inventory is not well formed")]
    Inventory(#[from] LoadError),
}

/// Whether a tensor is a layer's weight or its bias.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Param {
    /// The weight.
    Weight,
    /// The bias.
    Bias,
}

/// Which of two, in the order the model applies them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Ordinal {
    /// The one applied first.
    First,
    /// The one applied second.
    Second,
}

/// The head's three convolutions, in the order they are applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HeadConv {
    /// 3×3, from the fusion width to half of it; the upsampling to the input
    /// resolution follows it.
    First,
    /// 3×3, to the head width; a ReLU follows it.
    Second,
    /// 1×1, to one channel: the depth.
    Third,
}

/// One tensor of a transformer layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockPart {
    /// The layer norm before attention.
    Norm1(Param),
    /// The query projection of the attention.
    Query(Param),
    /// The key projection of the attention.
    Key(Param),
    /// The value projection of the attention.
    Value(Param),
    /// The projection after attention (`o_proj` in `transformers` 5).
    AttentionOutput(Param),
    /// LayerScale on the attention's output: a per-channel multiplier.
    LayerScale1,
    /// The layer norm before the MLP.
    Norm2(Param),
    /// The MLP's first linear layer, to the MLP width.
    MlpFc1(Param),
    /// The MLP's second linear layer, back to the hidden width.
    MlpFc2(Param),
    /// LayerScale on the MLP's output: a per-channel multiplier.
    LayerScale2,
}

/// One tensor of a stage's reassembly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReassemblePart {
    /// The 1×1 convolution from the hidden width to the stage's channel count.
    Projection(Param),
    /// The resampling convolution; stages that [`Resample::Keep`] have none.
    Resize(Param),
}

/// One tensor of a fusion layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionPart {
    /// The 1×1 convolution that ends the layer.
    Projection(Param),
    /// A convolution of one of the layer's two residual units.
    Residual {
        /// Which residual unit.
        unit: Ordinal,
        /// Which of the unit's two convolutions.
        conv: Ordinal,
        /// Weight or bias.
        param: Param,
    },
}

/// What a tensor of Depth Anything V2 is for.
///
/// A graph builder asks the inventory for a role and gets the tensor's name
/// and shape; stages and layers count from zero, in the order the model lists
/// them (stage 0 reads the shallowest of `out_indices`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// The class token prepended to the patch tokens.
    ClassToken,
    /// The mask token, unused at inference.
    MaskToken,
    /// The learned position embeddings, class token first.
    PositionEmbedding,
    /// The convolution that cuts the image into patch embeddings.
    PatchProjection(Param),
    /// A tensor of transformer layer `layer`.
    Block {
        /// The layer, from zero.
        layer: usize,
        /// Which of its tensors.
        part: BlockPart,
    },
    /// The layer norm applied to each layer output the head reads.
    FinalNorm(Param),
    /// A tensor of stage `stage`'s reassembly.
    Reassemble {
        /// The stage, from zero.
        stage: usize,
        /// Which of its tensors.
        part: ReassemblePart,
    },
    /// The 3×3 convolution (no bias) that brings stage `stage` to the fusion
    /// width.
    NeckConv {
        /// The stage, from zero.
        stage: usize,
    },
    /// A tensor of fusion layer `layer`. Layer 0 fuses the deepest stage and
    /// the last layer the shallowest, though they are numbered as the model
    /// stores them.
    Fusion {
        /// The fusion layer, from zero.
        layer: usize,
        /// Which of its tensors.
        part: FusionPart,
    },
    /// A tensor of the output head.
    Head {
        /// Which convolution.
        conv: HeadConv,
        /// Weight or bias.
        param: Param,
    },
}

/// Builds the inventory of the model `config` describes.
///
/// # Errors
///
/// [`ConfigError`] when `config` is not a model that can be built.
pub fn inventory(config: &Config) -> Result<Inventory<Role>, ConfigError> {
    config.validate()?;
    let mut specs = Vec::new();
    let mut push = |name: String, shape: Vec<usize>, role: Role| {
        specs.push(TensorSpec {
            name,
            dtype: StoredType::F32,
            shape,
            role,
        });
    };
    let hidden = config.hidden_size;
    let mlp = config.mlp_size();

    // The transformer.
    push(
        "backbone.embeddings.cls_token".to_owned(),
        vec![1, 1, hidden],
        Role::ClassToken,
    );
    if config.mask_token {
        push(
            "backbone.embeddings.mask_token".to_owned(),
            vec![1, hidden],
            Role::MaskToken,
        );
    }
    push(
        "backbone.embeddings.position_embeddings".to_owned(),
        vec![1, config.tokens(), hidden],
        Role::PositionEmbedding,
    );
    let patch = "backbone.embeddings.patch_embeddings.projection";
    push(
        format!("{patch}.weight"),
        vec![
            hidden,
            config.num_channels,
            config.patch_size,
            config.patch_size,
        ],
        Role::PatchProjection(Param::Weight),
    );
    push(
        format!("{patch}.bias"),
        vec![hidden],
        Role::PatchProjection(Param::Bias),
    );

    for layer in 0..config.num_layers {
        let p = format!("backbone.encoder.layer.{layer}");
        let mut block = |suffix: &str, shape: Vec<usize>, part: BlockPart| {
            push(format!("{p}.{suffix}"), shape, Role::Block { layer, part });
        };
        let vector = || vec![hidden];
        let square = || vec![hidden, hidden];
        block("norm1.weight", vector(), BlockPart::Norm1(Param::Weight));
        block("norm1.bias", vector(), BlockPart::Norm1(Param::Bias));
        block(
            "attention.attention.query.weight",
            square(),
            BlockPart::Query(Param::Weight),
        );
        block(
            "attention.attention.query.bias",
            vector(),
            BlockPart::Query(Param::Bias),
        );
        block(
            "attention.attention.key.weight",
            square(),
            BlockPart::Key(Param::Weight),
        );
        block(
            "attention.attention.key.bias",
            vector(),
            BlockPart::Key(Param::Bias),
        );
        block(
            "attention.attention.value.weight",
            square(),
            BlockPart::Value(Param::Weight),
        );
        block(
            "attention.attention.value.bias",
            vector(),
            BlockPart::Value(Param::Bias),
        );
        block(
            "attention.output.dense.weight",
            square(),
            BlockPart::AttentionOutput(Param::Weight),
        );
        block(
            "attention.output.dense.bias",
            vector(),
            BlockPart::AttentionOutput(Param::Bias),
        );
        block("layer_scale1.lambda1", vector(), BlockPart::LayerScale1);
        block("norm2.weight", vector(), BlockPart::Norm2(Param::Weight));
        block("norm2.bias", vector(), BlockPart::Norm2(Param::Bias));
        block(
            "mlp.fc1.weight",
            vec![mlp, hidden],
            BlockPart::MlpFc1(Param::Weight),
        );
        block("mlp.fc1.bias", vec![mlp], BlockPart::MlpFc1(Param::Bias));
        block(
            "mlp.fc2.weight",
            vec![hidden, mlp],
            BlockPart::MlpFc2(Param::Weight),
        );
        block("mlp.fc2.bias", vector(), BlockPart::MlpFc2(Param::Bias));
        block("layer_scale2.lambda1", vector(), BlockPart::LayerScale2);
    }
    push(
        "backbone.layernorm.weight".to_owned(),
        vec![hidden],
        Role::FinalNorm(Param::Weight),
    );
    push(
        "backbone.layernorm.bias".to_owned(),
        vec![hidden],
        Role::FinalNorm(Param::Bias),
    );

    // The neck: reassembly and the convolution to the fusion width, per stage.
    let fusion = config.fusion_hidden_size;
    for (stage, (&channels, resample)) in config
        .neck_hidden_sizes
        .iter()
        .zip(&config.reassemble)
        .enumerate()
    {
        let p = format!("neck.reassemble_stage.layers.{stage}");
        let mut part = |suffix: &str, shape: Vec<usize>, part: ReassemblePart| {
            push(
                format!("{p}.{suffix}"),
                shape,
                Role::Reassemble { stage, part },
            );
        };
        part(
            "projection.weight",
            vec![channels, hidden, 1, 1],
            ReassemblePart::Projection(Param::Weight),
        );
        part(
            "projection.bias",
            vec![channels],
            ReassemblePart::Projection(Param::Bias),
        );
        let kernel = match *resample {
            Resample::Up(factor) => Some(factor),
            Resample::Keep => None,
            Resample::Down(_) => Some(3),
        };
        if let Some(kernel) = kernel {
            part(
                "resize.weight",
                vec![channels, channels, kernel, kernel],
                ReassemblePart::Resize(Param::Weight),
            );
            part(
                "resize.bias",
                vec![channels],
                ReassemblePart::Resize(Param::Bias),
            );
        }
        push(
            format!("neck.convs.{stage}.weight"),
            vec![fusion, channels, 3, 3],
            Role::NeckConv { stage },
        );
    }

    // Fusion: one layer per stage.
    for layer in 0..config.out_indices.len() {
        let p = format!("neck.fusion_stage.layers.{layer}");
        let mut part = |suffix: String, shape: Vec<usize>, part: FusionPart| {
            push(format!("{p}.{suffix}"), shape, Role::Fusion { layer, part });
        };
        part(
            "projection.weight".to_owned(),
            vec![fusion, fusion, 1, 1],
            FusionPart::Projection(Param::Weight),
        );
        part(
            "projection.bias".to_owned(),
            vec![fusion],
            FusionPart::Projection(Param::Bias),
        );
        for (unit, unit_number) in [(Ordinal::First, 1), (Ordinal::Second, 2)] {
            for (conv, conv_number) in [(Ordinal::First, 1), (Ordinal::Second, 2)] {
                let stem = format!("residual_layer{unit_number}.convolution{conv_number}");
                part(
                    format!("{stem}.weight"),
                    vec![fusion, fusion, 3, 3],
                    FusionPart::Residual {
                        unit,
                        conv,
                        param: Param::Weight,
                    },
                );
                part(
                    format!("{stem}.bias"),
                    vec![fusion],
                    FusionPart::Residual {
                        unit,
                        conv,
                        param: Param::Bias,
                    },
                );
            }
        }
    }

    // The head.
    let half = fusion / 2;
    let head = config.head_hidden_size;
    let convs = [
        (HeadConv::First, "conv1", vec![half, fusion, 3, 3], half),
        (HeadConv::Second, "conv2", vec![head, half, 3, 3], head),
        (HeadConv::Third, "conv3", vec![1, head, 1, 1], 1),
    ];
    for (conv, name, weight_shape, bias_len) in convs {
        push(
            format!("head.{name}.weight"),
            weight_shape,
            Role::Head {
                conv,
                param: Param::Weight,
            },
        );
        push(
            format!("head.{name}.bias"),
            vec![bias_len],
            Role::Head {
                conv,
                param: Param::Bias,
            },
        );
    }

    Ok(Inventory::new(specs)?)
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::load::reader::Archive;
    use crate::load::testing::{Entry, file_of};
    use safetensors::Dtype;

    /// A model small enough to write out in full, with every kind of stage.
    fn tiny() -> Config {
        let mut config = Config::small();
        config.image_size = 4;
        config.patch_size = 2;
        config.hidden_size = 8;
        config.num_layers = 4;
        config.num_heads = 2;
        config.out_indices = vec![1, 2, 3, 4];
        config.neck_hidden_sizes = vec![2, 4, 6, 8];
        config.fusion_hidden_size = 4;
        config.head_hidden_size = 2;
        config
    }

    fn zero_file(inv: &Inventory<Role>, skip: &str) -> Vec<u8> {
        let entries: Vec<Entry<'_>> = inv
            .iter()
            .filter(|spec| spec.name != skip)
            .map(|spec| {
                let bytes = vec![0u8; spec.element_count() as usize * 4];
                (spec.name.as_str(), Dtype::F32, spec.shape.clone(), bytes)
            })
            .collect();
        file_of(&entries)
    }

    #[test]
    fn the_small_model_has_the_published_parameter_count() {
        let inv = inventory(&Config::small()).expect("the small config is valid");
        assert_eq!(inv.parameter_count(), 24_785_089);
    }

    #[test]
    fn names_are_distinct_and_roles_are_distinct() {
        let inv = inventory(&Config::small()).expect("valid");
        let mut roles: Vec<Role> = inv.iter().map(|s| s.role).collect();
        let total = roles.len();
        roles.sort_by_key(|r| format!("{r:?}"));
        roles.dedup();
        assert_eq!(roles.len(), total);
    }

    #[test]
    fn the_layers_follow_the_configuration() {
        let inv = inventory(&Config::small()).expect("valid");
        let q = inv
            .get("backbone.encoder.layer.11.attention.attention.query.weight")
            .expect("the last of 12 layers has a query");
        assert_eq!(q.shape, [384, 384]);
        assert_eq!(
            q.role,
            Role::Block {
                layer: 11,
                part: BlockPart::Query(Param::Weight)
            }
        );
        assert!(inv.get("backbone.encoder.layer.12.norm1.weight").is_none());

        let fc1 = inv
            .get("backbone.encoder.layer.0.mlp.fc1.weight")
            .expect("fc1");
        assert_eq!(fc1.shape, [1536, 384]);
        let pos = inv
            .get("backbone.embeddings.position_embeddings")
            .expect("pos");
        assert_eq!(pos.shape, [1, 37 * 37 + 1, 384]);
        let proj = inv
            .get("backbone.embeddings.patch_embeddings.projection.weight")
            .expect("patch projection");
        assert_eq!(proj.shape, [384, 3, 14, 14]);
    }

    #[test]
    fn the_resampling_tensors_follow_the_stage() {
        let inv = inventory(&Config::small()).expect("valid");
        let shape = |name: &str| inv.get(name).map(|s| s.shape.clone());
        assert_eq!(
            shape("neck.reassemble_stage.layers.0.resize.weight"),
            Some(vec![48, 48, 4, 4])
        );
        assert_eq!(
            shape("neck.reassemble_stage.layers.1.resize.weight"),
            Some(vec![96, 96, 2, 2])
        );
        assert_eq!(shape("neck.reassemble_stage.layers.2.resize.weight"), None);
        assert_eq!(shape("neck.reassemble_stage.layers.2.resize.bias"), None);
        assert_eq!(
            shape("neck.reassemble_stage.layers.3.resize.weight"),
            Some(vec![384, 384, 3, 3])
        );
    }

    #[test]
    fn the_neck_convolutions_have_no_bias_and_the_head_ends_in_one_channel() {
        let inv = inventory(&Config::small()).expect("valid");
        assert_eq!(
            inv.get("neck.convs.3.weight").map(|s| s.shape.clone()),
            Some(vec![64, 384, 3, 3])
        );
        assert!(inv.get("neck.convs.3.bias").is_none());
        assert_eq!(
            inv.get("head.conv3.weight").map(|s| s.shape.clone()),
            Some(vec![1, 32, 1, 1])
        );
        assert_eq!(
            inv.get("head.conv1.weight").map(|s| s.shape.clone()),
            Some(vec![32, 64, 3, 3])
        );
    }

    #[test]
    fn roles_find_the_tensor() {
        let inv = inventory(&Config::small()).expect("valid");
        let found = inv
            .by_role(&Role::Fusion {
                layer: 3,
                part: FusionPart::Residual {
                    unit: Ordinal::First,
                    conv: Ordinal::Second,
                    param: Param::Bias,
                },
            })
            .expect("fusion layer 3 has its first unit's second bias");
        assert_eq!(
            found.name,
            "neck.fusion_stage.layers.3.residual_layer1.convolution2.bias"
        );
        assert_eq!(found.shape, [64]);
    }

    #[test]
    fn without_a_mask_token_the_inventory_lacks_it() {
        let mut config = Config::small();
        config.mask_token = false;
        let inv = inventory(&config).expect("valid");
        assert!(inv.get("backbone.embeddings.mask_token").is_none());
        assert_eq!(inv.parameter_count(), 24_785_089 - 384);
    }

    #[test]
    fn a_file_written_from_the_inventory_checks_exact() {
        let inv = inventory(&tiny()).expect("the tiny config is valid");
        let file = zero_file(&inv, "");
        let report = inv.check(&Archive::parse(&file).expect("parses"));
        assert!(report.is_exact(), "{report}");
        assert_eq!(report.file_parameters, inv.parameter_count());
    }

    #[test]
    fn a_missing_tensor_is_named() {
        let inv = inventory(&tiny()).expect("valid");
        let gone = "backbone.encoder.layer.2.mlp.fc2.weight";
        let file = zero_file(&inv, gone);
        let report = inv.check(&Archive::parse(&file).expect("parses"));
        assert_eq!(report.missing, [gone]);
        assert!(report.unexpected.is_empty() && report.mismatched.is_empty());
    }

    #[test]
    fn the_v5_attention_names_are_unexpected_in_a_file_with_the_older_ones() {
        let inv = inventory(&tiny()).expect("valid");
        assert!(
            inv.get("backbone.encoder.layer.0.attention.q_proj.weight")
                .is_none()
        );
        assert!(
            inv.get("backbone.encoder.layer.0.attention.o_proj.weight")
                .is_none()
        );
    }

    #[test]
    fn invalid_configurations_are_refused() {
        let mut c = Config::small();
        c.num_heads = 5;
        assert!(matches!(
            inventory(&c),
            Err(ConfigError::HeadsDoNotDivideHidden { .. })
        ));

        let mut c = Config::small();
        c.image_size = 520;
        assert!(matches!(
            inventory(&c),
            Err(ConfigError::ImageNotWholePatches { .. })
        ));

        let mut c = Config::small();
        c.out_indices = vec![3, 6, 9, 13];
        assert!(matches!(
            inventory(&c),
            Err(ConfigError::OutIndexOutOfRange { index: 13, .. })
        ));

        let mut c = Config::small();
        c.out_indices = vec![0, 6, 9, 12];
        assert!(matches!(
            inventory(&c),
            Err(ConfigError::OutIndexOutOfRange { .. })
        ));

        let mut c = Config::small();
        c.neck_hidden_sizes.pop();
        assert!(matches!(
            inventory(&c),
            Err(ConfigError::StageCountMismatch { .. })
        ));

        let mut c = Config::small();
        c.fusion_hidden_size = 63;
        assert!(matches!(
            inventory(&c),
            Err(ConfigError::FusionWidthOdd { .. })
        ));

        let mut c = Config::small();
        c.hidden_size = 0;
        assert!(matches!(
            inventory(&c),
            Err(ConfigError::Zero {
                field: "hidden_size"
            })
        ));

        let mut c = Config::small();
        c.reassemble[0] = Resample::Up(0);
        assert!(matches!(
            inventory(&c),
            Err(ConfigError::Zero {
                field: "reassemble"
            })
        ));

        let mut c = Config::small();
        c.reassemble[3] = Resample::Down(1);
        assert!(matches!(
            inventory(&c),
            Err(ConfigError::ResampleByOne { stage: 3 })
        ));

        let mut c = Config::small();
        c.out_indices = vec![3, 9, 6, 12];
        assert!(matches!(
            inventory(&c),
            Err(ConfigError::OutIndicesNotIncreasing)
        ));

        let mut c = Config::small();
        c.out_indices = vec![3, 6, 6, 12];
        assert!(matches!(
            inventory(&c),
            Err(ConfigError::OutIndicesNotIncreasing)
        ));
    }
}
