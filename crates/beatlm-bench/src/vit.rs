//! GPU tests of the vision transformer operations of the kernel IR, at the
//! shapes of DINOv2 Small at 518 pixels (1370 tokens, 384 channels, 6 heads of
//! 64), against the CPU references in [`crate::reference`].
//!
//! Each operation runs whole and split into parts, so every split kernel's
//! parts are checked to compute what the whole kernel does; then one complete
//! pre-norm encoder block, built by `beatlm::model::depth_anything_v2::block`
//! with random weights, runs fused, unfused and split against
//! [`reference::vit_block`], a forward pass written apart from any graph.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;

use beatlm::load::depth_anything_v2::{BlockPart, Config, Param, Role};
use beatlm::model::Builder;
use beatlm::model::depth_anything_v2::{DINOV2_LAYER_NORM_EPSILON, block};
use beatlm_core::kernel::{
    AttentionScores, AttentionValues, Fusion, GELU_ERF_MAX_ERROR, Graph, OpKind, PatchEmbed,
    PrependToken, Softmax, TensorRole, TokensToMap,
};
use beatlm_core::memory::TensorId;

use crate::harness::{
    CHAIN as H, CHAIN_OUT as Y, assert_chain_matches, assert_close, chain as graph, device,
    run_on_gpu, weights,
};
use crate::net::uniform;
use crate::reference;

const TOKENS: u32 = 1370;
const WIDTH: u32 = 384;
const HEADS: u32 = 6;
const HEAD_DIM: u32 = 64;
const GRID: u32 = 37;
const PATCH: u32 = 14;

const X: TensorId = TensorId::new(0);
const W: TensorId = TensorId::new(1);
const B: TensorId = TensorId::new(2);

/// Part counts each operation is split into: whole, counts that do and do not
/// divide the kernels' extents, the 22 tile rows of 64 of 1370 query rows
/// (one part per tile row), and a count past what the tiled kernels allow,
/// which clamps.
const PARTS: [u32; 5] = [1, 3, 7, 22, 200];

/// `graph` on the GPU against the reference, fused and unfused, whole and in
/// each of [`PARTS`].
fn assert_matches(what: &str, g: &Graph, inputs: &[(TensorId, &Vec<f32>)]) {
    assert_chain_matches(what, g, inputs, &PARTS);
}

#[test]
#[ignore = "needs a GPU adapter"]
fn the_patch_embedding_matches_the_reference_at_518_pixels() {
    let side = GRID * PATCH;
    let image = uniform(1, (side * side * 3) as usize, -2.0, 2.0);
    let w = weights(2, PATCH * PATCH * 3 * WIDTH, PATCH * PATCH * 3);
    let b = uniform(3, WIDTH as usize, -0.1, 0.1);
    let op = OpKind::PatchEmbed(PatchEmbed {
        input: X,
        weights: W,
        bias: Some(B),
        grid_h: GRID,
        grid_w: GRID,
        patch: PATCH,
        cin: 3,
        cout: WIDTH,
    });
    let inputs = [(X, &image), (W, &w), (B, &b)];
    assert_matches("patch embedding", &graph(&inputs, &[op]), &inputs);
}

#[test]
#[ignore = "needs a GPU adapter"]
fn the_class_token_position_add_and_token_drop_match_the_reference() {
    let patches = uniform(4, ((TOKENS - 1) * WIDTH) as usize, -1.0, 1.0);
    let token = uniform(5, WIDTH as usize, -1.0, 1.0);
    let positions = uniform(6, (TOKENS * WIDTH) as usize, -1.0, 1.0);
    let elements = TOKENS * WIDTH;
    let ops = [
        OpKind::PrependToken(PrependToken {
            token: W,
            tokens: X,
            rows: TOKENS - 1,
            channels: WIDTH,
        }),
        // Fuses into the class token's kernel with fusion on.
        OpKind::Add {
            a: H,
            b: B,
            elements,
        },
        OpKind::TokensToMap(TokensToMap {
            input: TensorId::new(H.get() + 1),
            height: GRID,
            width: GRID,
            channels: WIDTH,
        }),
    ];
    let inputs = [(X, &patches), (W, &token), (B, &positions)];
    let g = graph(&inputs, &ops);
    assert_eq!(g.lower(Fusion::On).unwrap().kernels().len(), 2);
    assert_matches("class token, position add, token drop", &g, &inputs);
}

#[test]
#[ignore = "needs a GPU adapter"]
fn the_scaled_attention_scores_match_the_reference() {
    let qkv = uniform(7, (TOKENS * 3 * WIDTH) as usize, -1.0, 1.0);
    let ops = [
        OpKind::AttentionScores(AttentionScores {
            qkv: X,
            tokens: TOKENS,
            heads: HEADS,
            head_dim: HEAD_DIM,
        }),
        OpKind::Scale {
            input: H,
            factor: 0.125,
            elements: TOKENS * HEADS * TOKENS,
        },
    ];
    let inputs = [(X, &qkv)];
    assert_matches("attention scores", &graph(&inputs, &ops), &inputs);
}

#[test]
#[ignore = "needs a GPU adapter"]
fn the_softmax_matches_the_reference_over_attention_rows() {
    // Scores of the spread a scaled product of unit-sized vectors has, and a
    // few far larger, which only the subtraction of the row maximum keeps
    // finite.
    let mut scores = uniform(8, (TOKENS * HEADS * TOKENS) as usize, -8.0, 8.0);
    for i in (0..scores.len()).step_by(9_973) {
        scores[i] = 120.0;
    }
    let op = OpKind::Softmax(Softmax {
        input: X,
        rows: TOKENS * HEADS,
        cols: TOKENS,
    });
    let inputs = [(X, &scores)];
    assert_matches("softmax", &graph(&inputs, &[op]), &inputs);
}

#[test]
#[ignore = "needs a GPU adapter"]
fn the_weighted_sum_of_the_values_matches_the_reference() {
    let qkv = uniform(9, (TOKENS * 3 * WIDTH) as usize, -1.0, 1.0);
    let raw = uniform(10, (TOKENS * HEADS * TOKENS) as usize, -4.0, 4.0);
    let probs = reference::softmax(&raw, TOKENS as usize);
    let op = OpKind::AttentionValues(AttentionValues {
        probs: W,
        qkv: X,
        tokens: TOKENS,
        heads: HEADS,
        head_dim: HEAD_DIM,
    });
    let inputs = [(X, &qkv), (W, &probs)];
    assert_matches("attention values", &graph(&inputs, &[op]), &inputs);
}

#[test]
#[ignore = "needs a GPU adapter"]
fn gelu_erf_and_layerscale_match_the_reference_at_the_mlp_width() {
    let mlp = 4 * WIDTH;
    let x = uniform(11, (TOKENS * mlp) as usize, -6.0, 6.0);
    let s = uniform(12, mlp as usize, 0.1, 1.0);
    let ops = [
        OpKind::GeluErf {
            input: X,
            elements: TOKENS * mlp,
        },
        OpKind::ChannelScale {
            input: H,
            scale: W,
            rows: TOKENS,
            channels: mlp,
        },
    ];
    let inputs = [(X, &x), (W, &s)];
    assert_matches("GELU (erf), LayerScale", &graph(&inputs, &ops), &inputs);
}

#[test]
#[ignore = "needs a GPU adapter"]
fn gelu_erf_on_the_gpu_is_within_its_stated_error() {
    // The grid the CPU test states the error over, through the generated
    // kernel, against GELU through an f64 erf: what the GPU's exp adds to the
    // approximation is in this number.
    let grid = reference::erf_error_grid();
    let n = grid.len() as u32;
    let op = OpKind::GeluErf {
        input: X,
        elements: n,
    };
    let inputs = [(X, &grid)];
    let g = graph(&inputs, &[op]);
    let (device, queue) = device();
    let values = BTreeMap::from([(X, grid.clone())]);
    let got = run_on_gpu(&device, &queue, &g, Fusion::On, 1, &values);
    let (worst, at) = grid
        .iter()
        .zip(&got[&Y])
        .map(|(&x, &y)| {
            let x64 = f64::from(x);
            let exact = 0.5 * x64 * (1.0 + reference::erf(x64 / std::f64::consts::SQRT_2));
            ((f64::from(y) - exact).abs(), x)
        })
        .fold((0.0, 0.0), |a, b| if b.0 > a.0 { b } else { a });
    println!("GELU (erf) on the GPU: max abs error {worst:e} at x = {at}");
    assert!(worst <= f64::from(GELU_ERF_MAX_ERROR), "{worst:e} at {at}");
}

/// Random numbers for one weight of a block, by what it is for: gamma near
/// one, biases near zero, linear weights scaled by their fan-in, LayerScale in
/// DINOv2's range.
fn block_weight(role: &Role, elements: usize, seed: u64, config: &Config) -> Vec<f32> {
    let Role::Block { part, .. } = role else {
        panic!("{role:?} is not a block's");
    };
    let fan_in = |f: usize| {
        let s = 1.0 / (f as f32).sqrt();
        uniform(seed, elements, -s, s)
    };
    match part {
        BlockPart::Norm1(Param::Weight) | BlockPart::Norm2(Param::Weight) => {
            uniform(seed, elements, 0.8, 1.2)
        }
        BlockPart::LayerScale1 | BlockPart::LayerScale2 => uniform(seed, elements, 0.1, 1.0),
        BlockPart::MlpFc2(Param::Weight) => fan_in(config.mlp_size()),
        BlockPart::Query(Param::Weight)
        | BlockPart::AttentionOutput(Param::Weight)
        | BlockPart::MlpFc1(Param::Weight) => fan_in(config.hidden_size),
        _ => uniform(seed, elements, -0.1, 0.1),
    }
}

#[test]
#[ignore = "needs a GPU adapter"]
fn one_encoder_block_with_random_weights_matches_the_reference_block() {
    let config = Config::small();
    let mut b: Builder<Role> = Builder::new();
    let tokens = config.tokens() as u64 * config.hidden_size as u64;
    let x = b.input(tokens).unwrap();
    let y = block(&mut b, &config, 0, x, TensorRole::Output).unwrap();
    let (g, weights) = b.finish();

    let mut values = BTreeMap::from([(x, uniform(20, tokens as usize, -2.0, 2.0))]);
    let mut by_role: BTreeMap<String, Vec<f32>> = BTreeMap::new();
    for (seed, w) in (100..).zip(&weights) {
        let numbers = block_weight(&w.source.parts[0], w.elements as usize, seed, &config);
        by_role.insert(format!("{:?}", w.source.parts[0]), numbers.clone());
        values.insert(w.id, numbers);
    }
    let get =
        |part: BlockPart| by_role[&format!("{:?}", Role::Block { layer: 0, part })].as_slice();
    let wts = reference::BlockWeights {
        norm1: (
            get(BlockPart::Norm1(Param::Weight)),
            get(BlockPart::Norm1(Param::Bias)),
        ),
        qkv: (
            get(BlockPart::Query(Param::Weight)),
            get(BlockPart::Query(Param::Bias)),
        ),
        projection: (
            get(BlockPart::AttentionOutput(Param::Weight)),
            get(BlockPart::AttentionOutput(Param::Bias)),
        ),
        layer_scale1: get(BlockPart::LayerScale1),
        norm2: (
            get(BlockPart::Norm2(Param::Weight)),
            get(BlockPart::Norm2(Param::Bias)),
        ),
        fc1: (
            get(BlockPart::MlpFc1(Param::Weight)),
            get(BlockPart::MlpFc1(Param::Bias)),
        ),
        fc2: (
            get(BlockPart::MlpFc2(Param::Weight)),
            get(BlockPart::MlpFc2(Param::Bias)),
        ),
        layer_scale2: get(BlockPart::LayerScale2),
    };
    let want = reference::vit_block(
        &values[&x],
        &wts,
        (TOKENS as usize, HEADS as usize, HEAD_DIM as usize),
        config.mlp_size(),
        DINOV2_LAYER_NORM_EPSILON,
    );

    let (device, queue) = device();
    for (fusion, parts) in [
        (Fusion::On, 1),
        (Fusion::Off, 1),
        (Fusion::On, 3),
        (Fusion::On, 22),
    ] {
        let got = run_on_gpu(&device, &queue, &g, fusion, parts, &values);
        assert_close(
            &got[&y],
            &want,
            &format!("encoder block, fusion {fusion:?}, {parts} parts"),
        );
    }
}
