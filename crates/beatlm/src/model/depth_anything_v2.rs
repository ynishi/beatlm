//! The graph of Depth Anything V2's encoder, a DINOv2 vision transformer,
//! built from the tensors [`load::depth_anything_v2`](crate::load::depth_anything_v2)
//! lists.
//!
//! [`encoder`] builds it from an image to the feature maps the DPT head reads;
//! [`block`] is one transformer layer of it. Both write into a [`Builder`] the
//! caller owns, so the graph of the whole model is this one continued: the
//! head's builder (issue #25) calls [`encoder`] with the features as
//! activations and reads them as its inputs, and its weights join the same
//! list. A test that has no weights file builds a [`block`] the same way and
//! gives its weights random numbers by role.
//!
//! # The graph
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

use beatlm_core::kernel::{
    AttentionScores, AttentionValues, LayerNorm, MatMul, OpKind, PatchEmbed, PrependToken, Softmax,
    TensorRole, TokensToMap,
};
use beatlm_core::memory::TensorId;

use super::{Builder, Convert, ModelError, Source, dimension};
use crate::load::depth_anything_v2::{BlockPart, Config, Param, Role};

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

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use crate::load::depth_anything_v2::inventory;
    use beatlm_core::device::DeviceProfile;
    use beatlm_core::kernel::{Fusion, Program};
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
