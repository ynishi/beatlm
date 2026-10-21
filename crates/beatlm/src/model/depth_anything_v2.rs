//! The graph of Depth Anything V2, a DINOv2 vision transformer and a DPT head,
//! built from the tensors [`load::depth_anything_v2`](crate::load::depth_anything_v2)
//! lists.
//!
//! [`encoder`] builds the transformer from an image to the feature maps the
//! DPT head reads, and [`block`] is one transformer layer of it; [`decoder`]
//! builds the head from those maps to the depth, and [`fusion_layer`] is one
//! of its fusion layers; [`model`] is the encoder followed by the decoder.
//! Each writes into a [`Builder`] the caller owns, so the whole model is one
//! graph with one list of weights, the decoder reading the encoder's features
//! as activations. A test that has no weights file builds a [`block`] or a
//! [`fusion_layer`] the same way and gives its weights random numbers by role.
//!
//! # The encoder
//!
//! With `G = image_size / patch_size` and `T = G * G + 1` tokens:
//!
//! 1. the image, NHWC `[image_size][image_size][num_channels]`, becomes `G * G`
//!    patch tokens by a [`PatchEmbed`] with bias;
//! 2. the class token is put in front ([`PrependToken`]) and the position
//!    embedding added ([`OpKind::Add`] of a weight, which fuses into the
//!    previous kernel). The embedding is stored for `T` tokens and used as it
//!    is: at the size it was stored for, DINOv2 does not interpolate it;
//! 3. each layer up to the deepest of `out_indices` is a [`block`];
//! 4. after each layer of `out_indices`, the tokens go through the final layer
//!    norm and leave as a `[G][G][hidden_size]` map without the class token
//!    ([`TokensToMap`]), shallowest first.
//!
//! Layers past the deepest of `out_indices` feed no feature and are not built.
//! Every layer norm uses [`DINOV2_LAYER_NORM_EPSILON`].
//!
//! # The decoder
//!
//! The DPT head of `transformers`' `DepthAnythingForDepthEstimation`, as
//! version 5.18.0's `modeling_depth_anything.py` computes it (the version the
//! reference was made with), on NHWC maps. With `F = fusion_hidden_size`:
//!
//! 1. **Reassembly**, per stage `i` (shallowest first): a 1x1 convolution with
//!    bias ([`Conv2d1x1`]) from `hidden_size` to `neck_hidden_sizes[i]`, then
//!    the stage's [`Resample`]: `Up(f)` a [`ConvTranspose2d`] of kernel and
//!    stride `f` with bias (`G * f` pixels), `Keep` nothing, `Down(f)` a 3x3
//!    convolution of stride `f` with bias (`(G - 1) / f + 1` pixels). For the
//!    Small checkpoint at 518 pixels: 148, 74, 37 and 19 pixels a side.
//! 2. Per stage, a 3x3 convolution **without bias** to `F` channels.
//! 3. **Fusion**, deepest stage first: fusion layer `l` reads stage
//!    `n - 1 - l` ([`fusion_layer`]). Layer 0 runs only its second residual
//!    unit on its stage; every later layer adds its first residual unit of its
//!    stage to the previous layer's output, then runs its second on the sum.
//!    A residual unit is `x + conv2(relu(conv1(relu(x))))`, both 3x3 with bias:
//!    ReLU first, as `DepthAnythingPreActResidualLayer` applies it, with the
//!    residual the unit's input before that ReLU. The result is resampled
//!    bilinearly with aligned corners ([`ResizeBilinear`]) to the next stage's
//!    size, or to twice its own after the last stage, and projected by a 1x1
//!    convolution with bias.
//! 4. **The head**: a 3x3 convolution with bias to `F / 2`, bilinear
//!    resampling with aligned corners to `G * patch_size` pixels a side, a 3x3
//!    convolution with bias to `head_hidden_size`, a ReLU, a 1x1 convolution
//!    with bias to one channel and a ReLU: `[G * patch][G * patch][1]`, the
//!    relative depth.
//!
//! Three things in the `transformers` code have no operation here, because
//! they compute nothing for this model:
//!
//! - `DepthAnythingFeatureFusionLayer` resamples the stage's map to the
//!   previous layer's size, bilinearly *without* aligned corners, when the two
//!   differ. The previous layer is resampled to exactly this stage's size,
//!   whatever the configuration, so they never differ.
//! - Fusion layer 0's first residual unit is stored but never applied:
//!   `forward` calls it only when there is a previous layer.
//! - The head multiplies the depth by `max_depth`, which the Small
//!   checkpoint's `config.json` leaves unset and `DepthAnythingConfig` then
//!   sets to one; its `depth_estimation_type` is likewise the default,
//!   `"relative"`, whose final activation is the ReLU above.

use beatlm_core::kernel::{
    AttentionScores, AttentionValues, Conv2d1x1, Conv2d3x3, ConvTranspose2d, LayerNorm, MatMul,
    OpKind, PatchEmbed, PrependToken, ResizeBilinear, Softmax, TensorRole, TokensToMap,
};
use beatlm_core::memory::TensorId;

use super::{Builder, Convert, ModelError, Source, dimension};
use crate::load::depth_anything_v2::{
    BlockPart, Config, FusionPart, HeadConv, Ordinal, Param, ReassemblePart, Resample, Role,
};

/// The layer-norm epsilon of DINOv2: `layer_norm_eps` of `transformers`'
/// `Dinov2Config`, whose default the Small checkpoint's configuration does not
/// override (the sources are listed in
/// [`load::depth_anything_v2`](crate::load::depth_anything_v2)).
pub const DINOV2_LAYER_NORM_EPSILON: f32 = 1e-6;

/// The tensors of the encoder a caller binds or reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encoder {
    /// The input image, NHWC, `[image_size][image_size][num_channels]`.
    pub image: TensorId,
    /// One feature map per entry of `out_indices`, shallowest first, each
    /// `[G][G][hidden_size]`.
    pub features: Vec<TensorId>,
}

/// The dimensions of the transformer, as kernel dimensions.
#[derive(Debug, Clone, Copy)]
struct Dims {
    grid: u32,
    patch: u32,
    channels: u32,
    hidden: u32,
    heads: u32,
    head_dim: u32,
    mlp: u32,
    tokens: u32,
}

impl Dims {
    fn of(config: &Config) -> Result<Self, ModelError> {
        config
            .validate()
            .map_err(|e| ModelError::Config(Box::new(e)))?;
        let hidden = dimension("hidden_size", config.hidden_size)?;
        let heads = dimension("num_heads", config.num_heads)?;
        Ok(Self {
            grid: dimension("image_size / patch_size", config.grid())?,
            patch: dimension("patch_size", config.patch_size)?,
            channels: dimension("num_channels", config.num_channels)?,
            hidden,
            heads,
            head_dim: hidden / heads,
            mlp: dimension("mlp_ratio * hidden_size", config.mlp_size())?,
            tokens: dimension("tokens", config.tokens())?,
        })
    }
}

/// A weight of one tensor of `role`, converted as `convert` says.
fn weight(
    b: &mut Builder<Role>,
    role: Role,
    convert: Convert,
    elements: u64,
) -> Result<TensorId, ModelError> {
    Ok(b.weight(Source::one(role, convert), elements)?)
}

/// The product of `dims`, in `u64` so that a configuration whose dimensions
/// each fit a `u32` but whose sizes do not is refused by the graph rather
/// than overflowing here.
fn size(dims: &[u32]) -> u64 {
    dims.iter().map(|&d| u64::from(d)).product()
}

/// A layer norm over `rows` tokens of `d.hidden` channels.
fn norm(
    b: &mut Builder<Role>,
    input: TensorId,
    (gamma, beta): (TensorId, TensorId),
    d: &Dims,
) -> Result<TensorId, ModelError> {
    Ok(b.op(
        OpKind::LayerNorm(LayerNorm {
            input,
            gamma,
            beta,
            rows: d.tokens,
            channels: d.hidden,
            epsilon: DINOV2_LAYER_NORM_EPSILON,
        }),
        TensorRole::Activation,
    )?)
}

/// `a [tokens][k] * b [k][n] + bias`.
fn linear(
    b: &mut Builder<Role>,
    a: TensorId,
    (weights, bias): (TensorId, TensorId),
    (k, n): (u32, u32),
    d: &Dims,
) -> Result<TensorId, ModelError> {
    Ok(b.op(
        OpKind::MatMul(MatMul {
            a,
            b: weights,
            bias: Some(bias),
            m: d.tokens,
            k,
            n,
        }),
        TensorRole::Activation,
    )?)
}

/// Transformer layer `layer` (from zero) of the encoder `config` describes,
/// reading the tokens `x` (`[T][hidden_size]`) and returning its output, a
/// tensor of `role`:
///
/// ```text
/// x1  = x  + ls1 * proj(attention(norm1(x)))
/// out = x1 + ls2 * fc2(gelu(fc1(norm2(x1))))
/// ```
///
/// Attention is the decomposition the
/// [kernel documentation](beatlm_core::kernel#attention-materialised-scores)
/// gives: the query, key and value projections as one matrix multiply whose
/// weight is the three stored ones side by side, the scores scaled by
/// `1 / sqrt(head_dim)`, a softmax, the weighted sum of the values and the
/// output projection. GELU is the exact one, [`OpKind::GeluErf`].
///
/// # Errors
///
/// A configuration that does not validate or does not fit the IR's
/// dimensions, or a shape the graph refuses.
pub fn block(
    b: &mut Builder<Role>,
    config: &Config,
    layer: usize,
    x: TensorId,
    role: TensorRole,
) -> Result<TensorId, ModelError> {
    layer_of(b, &Dims::of(config)?, layer, x, role)
}

/// [`block`] with the configuration already validated into `d`.
fn layer_of(
    b: &mut Builder<Role>,
    d: &Dims,
    layer: usize,
    x: TensorId,
    role: TensorRole,
) -> Result<TensorId, ModelError> {
    use TensorRole::Activation as Act;
    let d = *d;
    let part = |part| Role::Block { layer, part };
    let vector = |b: &mut Builder<Role>, p| weight(b, part(p), Convert::AsStored, d.hidden.into());
    let (h, mlp) = (d.hidden, d.mlp);

    let norm1 = (
        vector(b, BlockPart::Norm1(Param::Weight))?,
        vector(b, BlockPart::Norm1(Param::Bias))?,
    );
    let qkv_weight = b.weight(
        Source {
            parts: [BlockPart::Query, BlockPart::Key, BlockPart::Value]
                .map(|p| part(p(Param::Weight)))
                .to_vec(),
            convert: Convert::Linear,
        },
        size(&[h, 3, h]),
    )?;
    let qkv_bias = b.weight(
        Source {
            parts: [BlockPart::Query, BlockPart::Key, BlockPart::Value]
                .map(|p| part(p(Param::Bias)))
                .to_vec(),
            convert: Convert::AsStored,
        },
        size(&[3, h]),
    )?;
    let projection = (
        weight(
            b,
            part(BlockPart::AttentionOutput(Param::Weight)),
            Convert::Linear,
            size(&[h, h]),
        )?,
        vector(b, BlockPart::AttentionOutput(Param::Bias))?,
    );
    let layer_scale1 = vector(b, BlockPart::LayerScale1)?;
    let norm2 = (
        vector(b, BlockPart::Norm2(Param::Weight))?,
        vector(b, BlockPart::Norm2(Param::Bias))?,
    );
    let fc1 = (
        weight(
            b,
            part(BlockPart::MlpFc1(Param::Weight)),
            Convert::Linear,
            size(&[h, mlp]),
        )?,
        weight(
            b,
            part(BlockPart::MlpFc1(Param::Bias)),
            Convert::AsStored,
            mlp.into(),
        )?,
    );
    let fc2 = (
        weight(
            b,
            part(BlockPart::MlpFc2(Param::Weight)),
            Convert::Linear,
            size(&[mlp, h]),
        )?,
        vector(b, BlockPart::MlpFc2(Param::Bias))?,
    );
    let layer_scale2 = vector(b, BlockPart::LayerScale2)?;

    // Attention.
    let n1 = norm(b, x, norm1, &d)?;
    let qkv = linear(b, n1, (qkv_weight, qkv_bias), (h, 3 * h), &d)?;
    let scores = b.op(
        OpKind::AttentionScores(AttentionScores {
            qkv,
            tokens: d.tokens,
            heads: d.heads,
            head_dim: d.head_dim,
        }),
        Act,
    )?;
    let score_elements = d.tokens * d.heads * d.tokens;
    let scaled = b.op(
        OpKind::Scale {
            input: scores,
            factor: 1.0 / (d.head_dim as f32).sqrt(),
            elements: score_elements,
        },
        Act,
    )?;
    let probs = b.op(
        OpKind::Softmax(Softmax {
            input: scaled,
            rows: d.tokens * d.heads,
            cols: d.tokens,
        }),
        Act,
    )?;
    let values = b.op(
        OpKind::AttentionValues(AttentionValues {
            probs,
            qkv,
            tokens: d.tokens,
            heads: d.heads,
            head_dim: d.head_dim,
        }),
        Act,
    )?;
    let projected = linear(b, values, projection, (h, h), &d)?;
    let scaled1 = b.op(
        OpKind::ChannelScale {
            input: projected,
            scale: layer_scale1,
            rows: d.tokens,
            channels: h,
        },
        Act,
    )?;
    let x1 = b.op(
        OpKind::Add {
            a: scaled1,
            b: x,
            elements: d.tokens * h,
        },
        Act,
    )?;

    // MLP.
    let n2 = norm(b, x1, norm2, &d)?;
    let hidden = linear(b, n2, fc1, (h, mlp), &d)?;
    let activated = b.op(
        OpKind::GeluErf {
            input: hidden,
            elements: d.tokens * mlp,
        },
        Act,
    )?;
    let out = linear(b, activated, fc2, (mlp, h), &d)?;
    let scaled2 = b.op(
        OpKind::ChannelScale {
            input: out,
            scale: layer_scale2,
            rows: d.tokens,
            channels: h,
        },
        Act,
    )?;
    Ok(b.op(
        OpKind::Add {
            a: scaled2,
            b: x1,
            elements: d.tokens * h,
        },
        role,
    )?)
}

/// The encoder `config` describes, from the image to its feature maps, which
/// are tensors of `features` (outputs, or activations a later stage reads).
///
/// # Errors
///
/// A configuration that does not validate or does not fit the IR's
/// dimensions, or a shape the graph refuses.
pub fn encoder(
    b: &mut Builder<Role>,
    config: &Config,
    features: TensorRole,
) -> Result<Encoder, ModelError> {
    use TensorRole::Activation as Act;
    let d = Dims::of(config)?;
    let image = b.input(size(&[d.grid, d.patch, d.grid, d.patch, d.channels]))?;
    let patch_weights = weight(
        b,
        Role::PatchProjection(Param::Weight),
        Convert::Conv,
        size(&[d.patch, d.patch, d.channels, d.hidden]),
    )?;
    let patch_bias = weight(
        b,
        Role::PatchProjection(Param::Bias),
        Convert::AsStored,
        d.hidden.into(),
    )?;
    let class_token = weight(b, Role::ClassToken, Convert::AsStored, d.hidden.into())?;
    let positions = weight(
        b,
        Role::PositionEmbedding,
        Convert::AsStored,
        size(&[d.tokens, d.hidden]),
    )?;

    let patches = b.op(
        OpKind::PatchEmbed(PatchEmbed {
            input: image,
            weights: patch_weights,
            bias: Some(patch_bias),
            grid_h: d.grid,
            grid_w: d.grid,
            patch: d.patch,
            cin: d.channels,
            cout: d.hidden,
        }),
        Act,
    )?;
    let tokens = b.op(
        OpKind::PrependToken(PrependToken {
            token: class_token,
            tokens: patches,
            rows: d.grid * d.grid,
            channels: d.hidden,
        }),
        Act,
    )?;
    let mut x = b.op(
        OpKind::Add {
            a: tokens,
            b: positions,
            elements: d.tokens * d.hidden,
        },
        Act,
    )?;

    let deepest = config.out_indices.iter().copied().max().unwrap_or(0);
    let mut taken = Vec::with_capacity(config.out_indices.len());
    for layer in 0..deepest {
        x = layer_of(b, &d, layer, x, Act)?;
        if config.out_indices.contains(&(layer + 1)) {
            taken.push(x);
        }
    }

    let final_norm = (
        weight(
            b,
            Role::FinalNorm(Param::Weight),
            Convert::AsStored,
            d.hidden.into(),
        )?,
        weight(
            b,
            Role::FinalNorm(Param::Bias),
            Convert::AsStored,
            d.hidden.into(),
        )?,
    );
    let mut maps = Vec::with_capacity(taken.len());
    for hidden_state in taken {
        let normed = norm(b, hidden_state, final_norm, &d)?;
        maps.push(b.op(
            OpKind::TokensToMap(TokensToMap {
                input: normed,
                height: d.grid,
                width: d.grid,
                channels: d.hidden,
            }),
            features,
        )?);
    }
    Ok(Encoder {
        image,
        features: maps,
    })
}

/// A map's height and width in pixels.
pub type Size = (u32, u32);

/// The tensors of the whole model a caller binds or reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DepthModel {
    /// The input image, NHWC, `[image_size][image_size][num_channels]`.
    pub image: TensorId,
    /// The encoder's feature maps, activations the decoder reads, shallowest
    /// first.
    pub features: Vec<TensorId>,
    /// The relative depth, `[G * patch_size][G * patch_size]` (one channel),
    /// an output.
    pub depth: TensorId,
}

/// What one fusion layer reads, and the size it resamples to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusionInputs {
    /// The previous fusion layer's output, of the stage's size; none for the
    /// first layer, which reads the deepest stage alone.
    pub previous: Option<TensorId>,
    /// The stage's map after its 3x3 convolution, `[size][fusion_hidden_size]`.
    pub stage: TensorId,
    /// The stage's size, which `previous` has too.
    pub size: Size,
    /// The size the layer's output is resampled to: the next stage's, or twice
    /// `size` after the last stage.
    pub out: Size,
}

/// `dims` multiplied, as an element count of an operation; a product past
/// `u32` is a configuration too large for the IR.
fn elements(dims: &[u32]) -> Result<u32, ModelError> {
    let total = size(dims);
    u32::try_from(total).map_err(|_| ModelError::TooLarge {
        field: "a map of the DPT head",
        value: usize::try_from(total).unwrap_or(usize::MAX),
    })
}

/// A 3x3 convolution of `input` (`size`, `cin` channels) to `cout` channels at
/// `stride`, with the weights of `conv` (converted from PyTorch's layout) and,
/// when `bias` is given, the bias of that role; returns its output, of `role`.
fn conv3x3(
    b: &mut Builder<Role>,
    input: TensorId,
    (conv, bias): (Role, Option<Role>),
    (height, width): Size,
    (cin, cout, stride): (u32, u32, u32),
    role: TensorRole,
) -> Result<TensorId, ModelError> {
    let weights = weight(b, conv, Convert::Conv, size(&[9, cin, cout]))?;
    let bias = bias
        .map(|r| weight(b, r, Convert::AsStored, cout.into()))
        .transpose()?;
    Ok(b.op(
        OpKind::Conv2d3x3(Conv2d3x3 {
            input,
            weights,
            bias,
            height,
            width,
            cin,
            cout,
            stride,
        }),
        role,
    )?)
}

/// A 1x1 convolution with bias, its weight and bias of roles `(conv, bias)`.
fn conv1x1(
    b: &mut Builder<Role>,
    input: TensorId,
    (conv, bias): (Role, Role),
    (height, width): Size,
    (cin, cout): (u32, u32),
    role: TensorRole,
) -> Result<TensorId, ModelError> {
    let weights = weight(b, conv, Convert::Conv, size(&[cin, cout]))?;
    let bias = weight(b, bias, Convert::AsStored, cout.into())?;
    Ok(b.op(
        OpKind::Conv2d1x1(Conv2d1x1 {
            input,
            weights,
            bias: Some(bias),
            height,
            width,
            cin,
            cout,
        }),
        role,
    )?)
}

/// `max(x, 0)` of a map of `size` and `channels`, an activation.
fn relu(
    b: &mut Builder<Role>,
    input: TensorId,
    (height, width): Size,
    channels: u32,
    role: TensorRole,
) -> Result<TensorId, ModelError> {
    Ok(b.op(
        OpKind::Relu {
            input,
            elements: elements(&[height, width, channels])?,
        },
        role,
    )?)
}

/// Bilinear resampling with aligned corners of `input` from `from` to `to`.
fn resize(
    b: &mut Builder<Role>,
    input: TensorId,
    (from, to): (Size, Size),
    channels: u32,
) -> Result<TensorId, ModelError> {
    Ok(b.op(
        OpKind::ResizeBilinear(ResizeBilinear {
            input,
            in_height: from.0,
            in_width: from.1,
            out_height: to.0,
            out_width: to.1,
            channels,
        }),
        TensorRole::Activation,
    )?)
}

/// The size of a stage of `grid` after `resample`, as the stage's
/// resampling layer computes it.
fn resampled(grid: u32, resample: Resample) -> Result<u32, ModelError> {
    Ok(match resample {
        Resample::Up(factor) => {
            let factor = dimension("reassemble", factor)?;
            grid.checked_mul(factor).ok_or(ModelError::TooLarge {
                field: "reassemble",
                value: usize::try_from(size(&[grid, factor])).unwrap_or(usize::MAX),
            })?
        }
        Resample::Keep => grid,
        Resample::Down(factor) => {
            let factor = dimension("reassemble", factor)?;
            Conv2d3x3 {
                input: TensorId::new(0),
                weights: TensorId::new(0),
                bias: None,
                height: grid,
                width: grid,
                cin: 1,
                cout: 1,
                stride: factor,
            }
            .out_height()
        }
    })
}

/// Stage `stage` reassembled from `feature` (`[grid][grid][hidden]`): the 1x1
/// projection to the stage's channels, then its resampling. Returns the map,
/// its size and its channels.
fn reassemble(
    b: &mut Builder<Role>,
    config: &Config,
    stage: usize,
    feature: TensorId,
    d: &Dims,
) -> Result<(TensorId, Size, u32), ModelError> {
    use TensorRole::Activation as Act;
    let part = |part| Role::Reassemble { stage, part };
    let channels = dimension("neck_hidden_sizes", config.neck_hidden_sizes[stage])?;
    let grid = (d.grid, d.grid);
    let projected = conv1x1(
        b,
        feature,
        (
            part(ReassemblePart::Projection(Param::Weight)),
            part(ReassemblePart::Projection(Param::Bias)),
        ),
        grid,
        (d.hidden, channels),
        Act,
    )?;
    let resample = config.reassemble[stage];
    let side = resampled(d.grid, resample)?;
    let resize_weight = part(ReassemblePart::Resize(Param::Weight));
    let resize_bias = part(ReassemblePart::Resize(Param::Bias));
    let map = match resample {
        Resample::Keep => projected,
        Resample::Up(factor) => {
            let factor = dimension("reassemble", factor)?;
            let weights = weight(
                b,
                resize_weight,
                Convert::ConvTranspose,
                size(&[factor, factor, channels, channels]),
            )?;
            let bias = weight(b, resize_bias, Convert::AsStored, channels.into())?;
            b.op(
                OpKind::ConvTranspose2d(ConvTranspose2d {
                    input: projected,
                    weights,
                    bias: Some(bias),
                    height: d.grid,
                    width: d.grid,
                    cin: channels,
                    cout: channels,
                    factor,
                }),
                Act,
            )?
        }
        Resample::Down(factor) => conv3x3(
            b,
            projected,
            (resize_weight, Some(resize_bias)),
            grid,
            (channels, channels, dimension("reassemble", factor)?),
            Act,
        )?,
    };
    Ok((map, (side, side), channels))
}

/// Residual unit `unit` of fusion layer `layer` on `x` (`size`, `fusion`
/// channels): `x + conv2(relu(conv1(relu(x))))`, as
/// `DepthAnythingPreActResidualLayer` computes it. The outer ReLU becomes the
/// first convolution's prologue and the inner one its epilogue; the residual
/// add fuses into the second convolution.
fn residual_unit(
    b: &mut Builder<Role>,
    (layer, unit): (usize, Ordinal),
    x: TensorId,
    hw: Size,
    fusion: u32,
) -> Result<TensorId, ModelError> {
    use TensorRole::Activation as Act;
    let roles = |conv| {
        let part = |param| Role::Fusion {
            layer,
            part: FusionPart::Residual { unit, conv, param },
        };
        (part(Param::Weight), Some(part(Param::Bias)))
    };
    let shape = (fusion, fusion, 1);
    let activated = relu(b, x, hw, fusion, Act)?;
    let first = conv3x3(b, activated, roles(Ordinal::First), hw, shape, Act)?;
    let activated = relu(b, first, hw, fusion, Act)?;
    let second = conv3x3(b, activated, roles(Ordinal::Second), hw, shape, Act)?;
    Ok(b.op(
        OpKind::Add {
            a: second,
            b: x,
            elements: elements(&[hw.0, hw.1, fusion])?,
        },
        Act,
    )?)
}

/// Fusion layer `layer` (from zero, deepest stage first) of the decoder
/// `config` describes, as `DepthAnythingFeatureFusionLayer` computes it:
///
/// ```text
/// x   = previous + unit1(stage)    (x = stage for the first layer)
/// out = projection(resize(unit2(x), out))
/// ```
///
/// with each unit `y + conv2(relu(conv1(relu(y))))` and the resampling
/// bilinear with aligned corners. Returns the output, `[out][F]`, a tensor of
/// `role`.
///
/// # Errors
///
/// A configuration that does not validate or does not fit the IR's
/// dimensions, or a shape the graph refuses.
pub fn fusion_layer(
    b: &mut Builder<Role>,
    config: &Config,
    layer: usize,
    inputs: FusionInputs,
    role: TensorRole,
) -> Result<TensorId, ModelError> {
    config
        .validate()
        .map_err(|e| ModelError::Config(Box::new(e)))?;
    let fusion = dimension("fusion_hidden_size", config.fusion_hidden_size)?;
    let mut x = inputs.stage;
    if let Some(previous) = inputs.previous {
        let unit = residual_unit(
            b,
            (layer, Ordinal::First),
            inputs.stage,
            inputs.size,
            fusion,
        )?;
        x = b.op(
            OpKind::Add {
                a: unit,
                b: previous,
                elements: elements(&[inputs.size.0, inputs.size.1, fusion])?,
            },
            TensorRole::Activation,
        )?;
    }
    let x = residual_unit(b, (layer, Ordinal::Second), x, inputs.size, fusion)?;
    let resized = resize(b, x, (inputs.size, inputs.out), fusion)?;
    let part = |param| Role::Fusion {
        layer,
        part: FusionPart::Projection(param),
    };
    conv1x1(
        b,
        resized,
        (part(Param::Weight), part(Param::Bias)),
        inputs.out,
        (fusion, fusion),
        role,
    )
}

/// The DPT head `config` describes, from the encoder's `features` (one per
/// entry of `out_indices`, shallowest first, each `[G][G][hidden_size]`) to the
/// relative depth, a tensor of `depth` (`[G * patch_size][G * patch_size]`,
/// one channel). The [module documentation](self#the-decoder) lists the
/// operations.
///
/// # Errors
///
/// A configuration that does not validate or does not fit the IR's
/// dimensions, a feature count other than the stage count, or a shape the
/// graph refuses.
pub fn decoder(
    b: &mut Builder<Role>,
    config: &Config,
    features: &[TensorId],
    depth: TensorRole,
) -> Result<TensorId, ModelError> {
    use TensorRole::Activation as Act;
    let d = Dims::of(config)?;
    if features.len() != config.out_indices.len() {
        return Err(ModelError::FeatureCount {
            expected: config.out_indices.len(),
            given: features.len(),
        });
    }
    let fusion = dimension("fusion_hidden_size", config.fusion_hidden_size)?;

    // Reassembly and the convolution to the fusion width, per stage.
    let mut stages = Vec::with_capacity(features.len());
    for (stage, &feature) in features.iter().enumerate() {
        let (map, hw, channels) = reassemble(b, config, stage, feature, &d)?;
        let neck = Role::NeckConv { stage };
        stages.push((
            conv3x3(b, map, (neck, None), hw, (channels, fusion, 1), Act)?,
            hw,
        ));
    }

    // Fusion, deepest stage first.
    let mut fused: Option<(TensorId, Size)> = None;
    for (layer, &(stage, hw)) in stages.iter().rev().enumerate() {
        // Saturating, so a size past the IR's is refused by the graph.
        let out = match stages.len().checked_sub(layer + 2) {
            Some(next) => stages[next].1,
            None => (hw.0.saturating_mul(2), hw.1.saturating_mul(2)),
        };
        let inputs = FusionInputs {
            previous: fused.map(|(t, _)| t),
            stage,
            size: hw,
            out,
        };
        fused = Some((fusion_layer(b, config, layer, inputs, Act)?, out));
    }
    // A validated configuration has at least one stage.
    let Some((fused, hw)) = fused else {
        return Err(ModelError::FeatureCount {
            expected: config.out_indices.len(),
            given: features.len(),
        });
    };

    // The head.
    let half = fusion / 2;
    let width = dimension("head_hidden_size", config.head_hidden_size)?;
    let side = d.grid.saturating_mul(d.patch);
    let head = |conv, param| Role::Head { conv, param };
    let first = conv3x3(
        b,
        fused,
        (
            head(HeadConv::First, Param::Weight),
            Some(head(HeadConv::First, Param::Bias)),
        ),
        hw,
        (fusion, half, 1),
        Act,
    )?;
    let resized = resize(b, first, (hw, (side, side)), half)?;
    let second = conv3x3(
        b,
        resized,
        (
            head(HeadConv::Second, Param::Weight),
            Some(head(HeadConv::Second, Param::Bias)),
        ),
        (side, side),
        (half, width, 1),
        Act,
    )?;
    let activated = relu(b, second, (side, side), width, Act)?;
    let third = conv1x1(
        b,
        activated,
        (
            head(HeadConv::Third, Param::Weight),
            head(HeadConv::Third, Param::Bias),
        ),
        (side, side),
        (width, 1),
        Act,
    )?;
    relu(b, third, (side, side), 1, depth)
}

/// The whole model `config` describes: the [`encoder`], its features as
/// activations, and the [`decoder`] reading them, the depth an output.
///
/// # Errors
///
/// As [`encoder`] and [`decoder`].
pub fn model(b: &mut Builder<Role>, config: &Config) -> Result<DepthModel, ModelError> {
    let enc = encoder(b, config, TensorRole::Activation)?;
    let depth = decoder(b, config, &enc.features, TensorRole::Output)?;
    Ok(DepthModel {
        image: enc.image,
        features: enc.features,
        depth,
    })
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::load::depth_anything_v2::inventory;
    use beatlm_core::device::DeviceProfile;
    use beatlm_core::kernel::{Fusion, KernelSpec, MainOp, Program, Variant};
    use beatlm_core::memory::MemoryPlan;

    fn small_encoder() -> (Builder<Role>, Encoder) {
        let mut b = Builder::new();
        let e = encoder(&mut b, &Config::small(), TensorRole::Output).unwrap();
        (b, e)
    }

    /// Checks that no two tensors live at the same kernel share a byte.
    fn assert_disjoint(program: &Program, plan: &MemoryPlan) {
        let tensors = program.tensors();
        for (i, a) in tensors.iter().enumerate() {
            for b in &tensors[i + 1..] {
                if a.first_use <= b.last_use && b.first_use <= a.last_use {
                    let (p, q) = (plan.placement(a.id).unwrap(), plan.placement(b.id).unwrap());
                    assert!(p.arena != q.arena || p.end() <= q.offset || q.end() <= p.offset);
                }
            }
        }
    }

    #[test]
    fn every_weight_of_the_encoder_is_a_role_of_the_inventory_of_its_size() {
        let (b, _) = small_encoder();
        let inv = inventory(&Config::small()).unwrap();
        for w in b.weights() {
            let stored: u64 = w
                .source
                .parts
                .iter()
                .map(|role| inv.by_role(role).unwrap().element_count())
                .sum();
            assert_eq!(stored, w.elements, "{:?}", w.source);
        }
        // Every tensor of the encoder's layers is read, and nothing of the
        // head: the mask token is unused at inference.
        let used: Vec<Role> = b
            .weights()
            .iter()
            .flat_map(|w| w.source.parts.clone())
            .collect();
        for spec in inv.iter() {
            let encoder_part = matches!(
                spec.role,
                Role::ClassToken
                    | Role::PositionEmbedding
                    | Role::PatchProjection(_)
                    | Role::Block { .. }
                    | Role::FinalNorm(_)
            );
            assert_eq!(used.contains(&spec.role), encoder_part, "{}", spec.name);
        }
    }

    #[test]
    fn the_small_encoder_lowers_to_its_kernels_and_plans_on_the_baseline() {
        let (b, e) = small_encoder();
        assert_eq!(e.features.len(), 4);
        let graph = b.graph();
        for &f in &e.features {
            assert_eq!(graph.elements(f), Some(37 * 37 * 384));
            assert_eq!(graph.role(f), Some(TensorRole::Output));
        }
        let baseline = DeviceProfile::webgpu_baseline();
        // The patch embedding, the class token with the position embedding
        // fused in, nine kernels per layer, and a layer norm and a copy per
        // feature.
        for (fusion, kernels) in [
            (Fusion::On, 2 + 12 * 9 + 4 * 2),
            (Fusion::Off, 3 + 12 * 15 + 4 * 2),
        ] {
            let program = graph.lower(fusion).unwrap();
            assert_eq!(program.kernels().len(), kernels, "{fusion:?}");
            let plan = MemoryPlan::new(program.tensors(), &baseline).unwrap();
            assert!(plan.valid_on(&baseline));
            assert_disjoint(&program, &plan);
        }
    }

    fn small_model() -> (Builder<Role>, DepthModel) {
        let mut b = Builder::new();
        let m = model(&mut b, &Config::small()).unwrap();
        (b, m)
    }

    /// Whether the model reads the tensor of `role`: everything but the mask
    /// token and the first residual unit of fusion layer 0, which
    /// `transformers` stores and never applies (the
    /// [module documentation](super#the-decoder)).
    fn read_by_the_model(role: &Role) -> bool {
        !matches!(
            role,
            Role::MaskToken
                | Role::Fusion {
                    layer: 0,
                    part: FusionPart::Residual {
                        unit: Ordinal::First,
                        ..
                    },
                }
        )
    }

    #[test]
    fn every_weight_of_the_model_is_a_role_of_the_inventory_of_its_size() {
        let (b, _) = small_model();
        let inv = inventory(&Config::small()).unwrap();
        for w in b.weights() {
            let stored: u64 = w
                .source
                .parts
                .iter()
                .map(|role| inv.by_role(role).unwrap().element_count())
                .sum();
            assert_eq!(stored, w.elements, "{:?}", w.source);
        }
        let used: Vec<Role> = b
            .weights()
            .iter()
            .flat_map(|w| w.source.parts.clone())
            .collect();
        for spec in inv.iter() {
            assert_eq!(
                used.contains(&spec.role),
                read_by_the_model(&spec.role),
                "{}",
                spec.name
            );
        }
        // Each role is one weight, read once.
        assert_eq!(used.len(), {
            let mut distinct = used.clone();
            distinct.sort_by_key(|r| format!("{r:?}"));
            distinct.dedup();
            distinct.len()
        });
        // The transposed convolutions' weights are converted as such.
        for w in b.weights() {
            if let [Role::Reassemble { stage, part }] = w.source.parts[..] {
                if part == ReassemblePart::Resize(Param::Weight) && stage < 2 {
                    assert_eq!(w.source.convert, Convert::ConvTranspose);
                }
            }
        }
    }

    #[test]
    fn the_small_decoder_lowers_to_the_kernels_the_fusion_rules_give() {
        let (b, m) = small_model();
        let graph = b.graph();
        assert_eq!(graph.elements(m.depth), Some(518 * 518));
        assert_eq!(graph.role(m.depth), Some(TensorRole::Output));
        for &f in &m.features {
            assert_eq!(graph.role(f), Some(TensorRole::Activation));
        }
        let program = graph.lower(Fusion::On).unwrap();
        let encoder_kernels = 2 + 12 * 9 + 4 * 2;
        let names: Vec<String> = program.kernels()[encoder_kernels..]
            .iter()
            .map(KernelSpec::name)
            .collect();
        let unit1 = ["relu-conv2d3x3+relu", "conv2d3x3+add+add"];
        let unit2 = ["relu-conv2d3x3+relu", "conv2d3x3+add"];
        let tail = ["resize_bilinear", "matmul"];
        let mut want: Vec<&str> = vec![
            // Reassembly and the neck's convolutions: 148, 74, 37, 19 pixels.
            "matmul",
            "conv_transpose2d",
            "conv2d3x3",
            "matmul",
            "conv_transpose2d",
            "conv2d3x3",
            "matmul",
            "conv2d3x3",
            "matmul",
            "conv2d3x3",
            "conv2d3x3",
        ];
        want.extend(unit2.iter().chain(&tail));
        for _ in 0..3 {
            want.extend(unit1.iter().chain(&unit2).chain(&tail));
        }
        want.extend([
            "conv2d3x3",
            "resize_bilinear",
            "conv2d3x3+relu",
            "matmul+relu",
        ]);
        assert_eq!(names, want);
        // The ReLUs before the residual units' first convolutions are all
        // prologues: seven units.
        let prologues = program
            .kernels()
            .iter()
            .filter(|k| !k.prologue().is_empty())
            .count();
        assert_eq!(prologues, 7);
        // Unfused, every operation is a kernel: 3 + 3 + 2 + 3 for the stages,
        // 7 for the first fusion layer and 13 for each other, 6 for the head.
        let unfused = graph.lower(Fusion::Off).unwrap();
        assert_eq!(
            unfused.kernels().len(),
            3 + 12 * 15 + 4 * 2 + 11 + 7 + 3 * 13 + 6
        );
        // The stride-2 convolution takes 37 pixels to 19, and the last fusion
        // layer doubles 148 to 296.
        let strided = &program.kernels()[encoder_kernels + 9];
        let MainOp::Conv2d3x3(c) = strided.main() else {
            panic!("{}", strided.name());
        };
        assert_eq!((c.stride, c.out_height(), c.cout), (2, 19, 384));
        let last_resize = &program.kernels()[program.kernels().len() - 6];
        let MainOp::ResizeBilinear(r) = last_resize.main() else {
            panic!("{}", last_resize.name());
        };
        assert_eq!((r.in_height, r.out_height, r.channels), (148, 296, 64));
    }

    /// Checks that every kernel's tensors are live at it.
    fn assert_live_at_their_kernels(program: &Program) {
        for (k, kernel) in program.kernels().iter().enumerate() {
            let k = k as u32;
            for t in kernel.bindings() {
                let d = program.tensors().iter().find(|d| d.id == t).unwrap();
                assert!(d.first_use <= k && k <= d.last_use, "tensor {}", t.get());
            }
        }
    }

    #[test]
    fn the_whole_model_at_518_pixels_plans_on_the_baseline_fused_unfused_and_split() {
        let (b, m) = small_model();
        let baseline = DeviceProfile::webgpu_baseline();
        let whole = b.graph().lower(Fusion::On).unwrap();
        // The decoder's largest tensors are the head's two 518 x 518 x 32
        // maps; the encoder's attention scores are larger. Each fits one
        // binding.
        let bytes = |t: &beatlm_core::memory::TensorDesc| t.elements * 4;
        let head_maps = whole
            .tensors()
            .iter()
            .filter(|t| bytes(t) == 518 * 518 * 32 * 4)
            .count();
        assert_eq!(head_maps, 2);
        let largest = whole.tensors().iter().map(bytes).max().unwrap();
        assert_eq!(largest, 45_045_600);
        assert!(largest <= baseline.max_arena_size());
        // The depth is an output: live for the whole program.
        let depth = whole.tensors().iter().find(|t| t.id == m.depth).unwrap();
        assert_eq!(
            (depth.first_use, depth.last_use),
            (0, whole.kernels().len() as u32 - 1)
        );
        // A feature map lives from its copy out of the tokens to its
        // reassembly's projection.
        for &f in &m.features {
            let t = whole.tensors().iter().find(|t| t.id == f).unwrap();
            let reader = whole.kernels()[t.last_use as usize].main();
            assert!(
                matches!(reader, MainOp::MatMul(mm) if mm.a == f),
                "{reader:?}"
            );
        }
        let unfused = b.graph().lower(Fusion::Off).unwrap();
        let split = whole.split(&vec![3; whole.kernels().len()]).unwrap();
        for (what, program) in [("fused", &whole), ("unfused", &unfused), ("split", &split)] {
            let plan = MemoryPlan::new(program.tensors(), &baseline).unwrap();
            assert!(plan.valid_on(&baseline), "{what}");
            assert!(
                plan.arenas()
                    .iter()
                    .all(|&a| a <= baseline.max_arena_size()),
                "{what}"
            );
            assert_disjoint(program, &plan);
            assert_live_at_their_kernels(program);
            println!(
                "{what}: {} kernels, {} tensors, arenas {:?} bytes ({} in all), {} bytes live at most",
                program.kernels().len(),
                program.tensors().len(),
                plan.arenas(),
                plan.total_bytes(),
                plan.peak_live_bytes()
            );
        }
    }

    #[test]
    fn every_kernel_of_the_whole_model_validates_fused_and_unfused_whole_and_split() {
        let (b, _) = small_model();
        let mut generator = crate::codegen::Generator::new();
        for fusion in [Fusion::On, Fusion::Off] {
            let whole = b.graph().lower(fusion).unwrap();
            for parts in [1, 3] {
                let program = whole.split(&vec![parts; whole.kernels().len()]).unwrap();
                for spec in program.kernels() {
                    generator
                        .generate(spec, Variant::Baseline)
                        .unwrap_or_else(|e| panic!("{e}"));
                }
            }
        }
    }

    #[test]
    fn a_decoder_given_the_wrong_number_of_features_is_refused() {
        let mut b = Builder::new();
        let e = encoder(&mut b, &Config::small(), TensorRole::Activation).unwrap();
        assert!(matches!(
            decoder(
                &mut b,
                &Config::small(),
                &e.features[..3],
                TensorRole::Output
            ),
            Err(ModelError::FeatureCount {
                expected: 4,
                given: 3
            })
        ));
    }

    #[test]
    fn layers_past_the_deepest_feature_are_not_built() {
        let mut config = Config::small();
        config.out_indices = vec![1, 2, 3, 4];
        let mut b = Builder::new();
        encoder(&mut b, &config, TensorRole::Output).unwrap();
        let p = b.graph().lower(Fusion::On).unwrap();
        assert_eq!(p.kernels().len(), 2 + 4 * 9 + 4 * 2);
        assert!(
            b.weights()
                .iter()
                .all(|w| w.source.parts.iter().all(|r| !matches!(
                    r,
                    Role::Block { layer, .. } if *layer >= 4
                )))
        );
    }
}
