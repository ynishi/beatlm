//! GPU tests of the DPT head operations of the kernel IR, at the shapes of
//! Depth Anything V2 Small at 518 pixels and at odd ones no workgroup or tile
//! divides, against the CPU references in [`crate::reference`].
//!
//! Each operation runs fused and unfused, whole and split into parts, as the
//! vision transformer's do (`vit.rs`); then one feature-fusion layer, built by
//! `beatlm::model::depth_anything_v2::fusion_layer` with random weights, runs
//! fused, unfused and split against [`reference::fusion_layer`], a forward
//! pass written apart from any graph.
#![allow(clippy::unwrap_used, clippy::expect_used)]

use std::collections::BTreeMap;

use beatlm::load::depth_anything_v2::{Config, FusionPart, Ordinal, Param, Role};
use beatlm::model::Builder;
use beatlm::model::depth_anything_v2::{FusionInputs, fusion_layer};
use beatlm_core::kernel::{
    Conv2d1x1, Conv2d3x3, ConvTranspose2d, Fusion, OpKind, ResizeBilinear, TensorRole,
};
use beatlm_core::memory::TensorId;

use crate::harness::{
    CHAIN as H, assert_chain_matches, assert_close, chain as graph, device, run_on_gpu, weights,
};
use crate::net::uniform;
use crate::reference;

const X: TensorId = TensorId::new(0);
const W: TensorId = TensorId::new(1);
const B: TensorId = TensorId::new(2);

/// Part counts each operation is split into: whole, counts that do and do not
/// divide the kernels' extents, and one past what a small tiled kernel
/// allows, which clamps.
const PARTS: [u32; 5] = [1, 2, 3, 7, 64];

#[test]
#[ignore = "needs a GPU adapter"]
fn strided_and_unbiased_3x3_convolutions_match_the_reference() {
    // The reassembly's downsampling at its real shape (37 to 19, 384
    // channels, bias), the neck's convolution without bias at an odd output
    // count, and a small stride-3 case; none of the output counts is a
    // multiple of the 256 invocations of a workgroup.
    for (seed, (h, w, cin, cout, stride), bias) in [
        (1, (37, 37, 384, 384, 2), true),
        (2, (37, 37, 192, 64, 1), false),
        (3, (7, 9, 5, 37, 2), false),
        (4, (8, 11, 3, 6, 3), true),
    ] {
        let x = uniform(seed, (h * w * cin) as usize, -1.0, 1.0);
        let wt = weights(seed + 10, 9 * cin * cout, 9 * cin);
        let b = uniform(seed + 20, cout as usize, -0.1, 0.1);
        let mut inputs = vec![(X, &x), (W, &wt)];
        if bias {
            inputs.push((B, &b));
        }
        let op = OpKind::Conv2d3x3(Conv2d3x3 {
            input: X,
            weights: W,
            bias: bias.then_some(B),
            height: h,
            width: w,
            cin,
            cout,
            stride,
        });
        let what = format!("conv3x3 {h}x{w}x{cin} to {cout}, stride {stride}, bias {bias}");
        assert_chain_matches(&what, &graph(&inputs, &[op]), &inputs, &PARTS);
    }
}

#[test]
#[ignore = "needs a GPU adapter"]
fn a_convolution_with_a_relu_prologue_and_epilogue_matches_the_reference() {
    // relu -> conv -> relu, the first half of a residual unit, at the second
    // fusion layer's 37 x 37 x 64, and at an odd size with stride 2. Fused, the
    // first ReLU is the convolution's prologue and the second its epilogue.
    for (seed, (h, w, c, stride)) in [(5, (37, 37, 64, 1)), (6, (9, 7, 3, 2))] {
        let x = uniform(seed, (h * w * c) as usize, -1.0, 1.0);
        let wt = weights(seed + 10, 9 * c * c, 9 * c);
        let b = uniform(seed + 20, c as usize, -0.1, 0.1);
        let conv = Conv2d3x3 {
            input: H,
            weights: W,
            bias: Some(B),
            height: h,
            width: w,
            cin: c,
            cout: c,
            stride,
        };
        let out = conv.out_height() * conv.out_width() * c;
        let ops = [
            OpKind::Relu {
                input: X,
                elements: h * w * c,
            },
            OpKind::Conv2d3x3(conv),
            OpKind::Relu {
                input: TensorId::new(H.get() + 1),
                elements: out,
            },
        ];
        let inputs = [(X, &x), (W, &wt), (B, &b)];
        let g = graph(&inputs, &ops);
        let fused = g.lower(Fusion::On).unwrap();
        assert_eq!(fused.kernels().len(), 1);
        assert_eq!(fused.kernels()[0].name(), "relu-conv2d3x3+relu");
        let what = format!("relu, conv3x3 {h}x{w}x{c} stride {stride}, relu");
        assert_chain_matches(&what, &g, &inputs, &PARTS);
    }
}

#[test]
#[ignore = "needs a GPU adapter"]
fn the_1x1_convolutions_match_the_reference() {
    // The reassembly's projection, the last fusion layer's at 296 x 296, the
    // head's last to one channel with its ReLU, and an odd one.
    for (seed, (h, w, cin, cout), relu) in [
        (7, (37, 37, 384, 48), false),
        (8, (296, 296, 64, 64), false),
        (9, (518, 518, 32, 1), true),
        (10, (5, 7, 3, 19), false),
    ] {
        let x = uniform(seed, (h * w * cin) as usize, -1.0, 1.0);
        let wt = weights(seed + 10, cin * cout, cin);
        let b = uniform(seed + 20, cout as usize, -0.1, 0.1);
        let mut ops = vec![OpKind::Conv2d1x1(Conv2d1x1 {
            input: X,
            weights: W,
            bias: Some(B),
            height: h,
            width: w,
            cin,
            cout,
        })];
        if relu {
            ops.push(OpKind::Relu {
                input: H,
                elements: h * w * cout,
            });
        }
        let inputs = [(X, &x), (W, &wt), (B, &b)];
        let what = format!("conv1x1 {h}x{w}x{cin} to {cout}, relu {relu}");
        assert_chain_matches(&what, &graph(&inputs, &ops), &inputs, &PARTS);
    }
}

#[test]
#[ignore = "needs a GPU adapter"]
fn the_transposed_convolutions_match_the_reference() {
    // The two upsampling stages at their real shapes and an odd one.
    for (seed, (h, w, cin, cout, factor), bias) in [
        (11, (37, 37, 48, 48, 4), true),
        (12, (37, 37, 96, 96, 2), true),
        (13, (3, 5, 2, 7, 3), false),
    ] {
        let x = uniform(seed, (h * w * cin) as usize, -1.0, 1.0);
        let wt = weights(seed + 10, factor * factor * cin * cout, cin);
        let b = uniform(seed + 20, cout as usize, -0.1, 0.1);
        let mut inputs = vec![(X, &x), (W, &wt)];
        if bias {
            inputs.push((B, &b));
        }
        let op = OpKind::ConvTranspose2d(ConvTranspose2d {
            input: X,
            weights: W,
            bias: bias.then_some(B),
            height: h,
            width: w,
            cin,
            cout,
            factor,
        });
        let what = format!("transposed conv {h}x{w}x{cin} to {cout}, factor {factor}");
        assert_chain_matches(&what, &graph(&inputs, &[op]), &inputs, &PARTS);
    }
}

#[test]
#[ignore = "needs a GPU adapter"]
fn aligned_corner_bilinear_resampling_matches_the_reference() {
    // Every resampling of the model at 518 pixels (each fusion layer's to the
    // next stage, the last one's doubling, the head's to the input size), and
    // odd shapes up, down, and to a single row.
    for (seed, (ih, iw, oh, ow, c)) in [
        (14, (19, 19, 37, 37, 64)),
        (15, (37, 37, 74, 74, 64)),
        (16, (74, 74, 148, 148, 64)),
        (17, (148, 148, 296, 296, 64)),
        (18, (296, 296, 518, 518, 32)),
        (19, (3, 5, 11, 4, 3)),
        (20, (9, 7, 4, 3, 2)),
        (21, (6, 4, 1, 5, 3)),
    ] {
        let x = uniform(seed, (ih * iw * c) as usize, -2.0, 2.0);
        let op = OpKind::ResizeBilinear(ResizeBilinear {
            input: X,
            in_height: ih,
            in_width: iw,
            out_height: oh,
            out_width: ow,
            channels: c,
        });
        let inputs = [(X, &x)];
        let what = format!("bilinear {ih}x{iw} to {oh}x{ow}, {c} channels");
        assert_chain_matches(&what, &graph(&inputs, &[op]), &inputs, &PARTS);
    }
}

/// Random numbers for a weight of a fusion layer, by role: 3x3 and 1x1
/// convolutions scaled by their fan-in, biases near zero.
fn fusion_weight(role: &Role, elements: usize, seed: u64, fusion: usize) -> Vec<f32> {
    let Role::Fusion { part, .. } = role else {
        panic!("{role:?} is not a fusion layer's");
    };
    let fan_in = |f: usize| {
        let s = 1.0 / (f as f32).sqrt();
        uniform(seed, elements, -s, s)
    };
    match part {
        FusionPart::Residual {
            param: Param::Weight,
            ..
        } => fan_in(9 * fusion),
        FusionPart::Projection(Param::Weight) => fan_in(fusion),
        _ => uniform(seed, elements, -0.1, 0.1),
    }
}

#[test]
#[ignore = "needs a GPU adapter"]
fn one_fusion_layer_with_random_weights_matches_the_reference_layer() {
    let config = Config::small();
    let c = config.fusion_hidden_size;
    let fusion = c as u32;
    // The second fusion layer at 518 pixels (37 x 37 to 74 x 74, with a
    // previous layer), and the first (19 x 19 to 37 x 37, alone).
    for (layer, size, out, with_previous) in [
        (1, (37, 37), (74, 74), true),
        (0, (19, 19), (37, 37), false),
    ] {
        let elements = u64::from(size.0 * size.1 * fusion);
        let mut b: Builder<Role> = Builder::new();
        let stage = b.input(elements).unwrap();
        let previous = with_previous.then(|| b.input(elements).unwrap());
        let inputs = FusionInputs {
            previous,
            stage,
            size,
            out,
        };
        let y = fusion_layer(&mut b, &config, layer, inputs, TensorRole::Output).unwrap();
        let (g, weights) = b.finish();

        let mut values = BTreeMap::from([(stage, uniform(30, elements as usize, -1.0, 1.0))]);
        if let Some(p) = previous {
            values.insert(p, uniform(31, elements as usize, -1.0, 1.0));
        }
        let mut by_role: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        for (seed, w) in (200..).zip(&weights) {
            let numbers = fusion_weight(&w.source.parts[0], w.elements as usize, seed, c);
            by_role.insert(format!("{:?}", w.source.parts[0]), numbers.clone());
            values.insert(w.id, numbers);
        }
        let get =
            |part: FusionPart| by_role[&format!("{:?}", Role::Fusion { layer, part })].as_slice();
        let unit = |unit| {
            let conv = |conv| {
                let p = |param| FusionPart::Residual { unit, conv, param };
                (get(p(Param::Weight)), get(p(Param::Bias)))
            };
            reference::UnitWeights {
                conv1: conv(Ordinal::First),
                conv2: conv(Ordinal::Second),
            }
        };
        let wts = reference::FusionWeights {
            unit1: with_previous.then(|| unit(Ordinal::First)),
            unit2: unit(Ordinal::Second),
            projection: (
                get(FusionPart::Projection(Param::Weight)),
                get(FusionPart::Projection(Param::Bias)),
            ),
        };
        let shape = (
            (size.0 as usize, size.1 as usize),
            (out.0 as usize, out.1 as usize),
            c,
        );
        let want = reference::fusion_layer(
            previous.map(|p| values[&p].as_slice()),
            &values[&stage],
            &wts,
            shape,
        );

        let fused = g.lower(Fusion::On).unwrap();
        let units = if with_previous { 2 } else { 1 };
        assert_eq!(
            fused
                .kernels()
                .iter()
                .filter(|k| !k.prologue().is_empty())
                .count(),
            units
        );
        let (device, queue) = device();
        for (fusion, parts) in [
            (Fusion::On, 1),
            (Fusion::Off, 1),
            (Fusion::On, 3),
            (Fusion::On, 64),
        ] {
            let got = run_on_gpu(&device, &queue, &g, fusion, parts, &values);
            assert_close(
                &got[&y],
                &want,
                &format!("fusion layer {layer}, fusion {fusion:?}, {parts} parts"),
            );
        }
    }
}
