#!/usr/bin/env -S uv run --isolated --no-project
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = [
#     "torch==2.14.0",
#     "transformers==5.18.0",
#     "numpy==2.5.3",
#     "safetensors==0.8.0",
# ]
#
# [tool.uv]
# exclude-newer = "2026-10-21T00:00:00Z"
# ///
"""Runs Depth Anything V2 Small in `transformers` and writes what a port must match.

Run it with `just reference-depth-anything`, after `just setup-depth-anything`.
It is a sidecar: nothing in the engine's build or tests needs it, and the tests
that read its output are `#[ignore]`d.

Environment. `uv` builds an isolated environment from the pins above (the
`exclude-newer` date also fixes everything they pull in), so the run does not
touch, and is not affected by, any other Python on the machine. `torch` is
large; the first run downloads it.

Terms. `torch`, `transformers`, `numpy`, `safetensors` and what they pull in
are fetched by `uv` into its cache under their own licences and are not
redistributed by this repository. The weights this script
reads are third-party material under their own terms, stated in
`scripts/setup-depth-anything.sh`; the files this script writes are computed
from them and stay under `models/`, which is not committed.

Input. A tensor of shape (1, 3, 518, 518) that is already in the model's input
space, defined by a formula that needs no random numbers, no image file and no
transcendental function, so Rust computes the same bits (the same formula is
`reference_input` in `crates/beatlm/tests/depth_anything_v2.rs`, and a test
compares the two). For channel c, row y and column x, with the integer level

    c = 0:  level = x * 255 // 517                  (horizontal ramp)
    c = 1:  level = y * 255 // 517                  (vertical ramp)
    c = 2:  level = ((x // 37 + y // 37) % 2) * 255 (checkerboard of 37-pixel cells)

the value is, in IEEE 754 single precision, one rounded operation after the
next in this order:

    value = (level / 255 - mean[c]) / std[c]

with mean = (0.485, 0.456, 0.406) and std = (0.229, 0.224, 0.225), the image
processor's constants in `preprocessor_config.json`. Integer-to-float
conversion, division and subtraction are correctly rounded in binary32, so any
conforming implementation gives identical bits. The image processor itself
(resizing, rescaling) is not run: the input is defined directly in the space
the network sees.

Output, in `models/depth-anything-v2-small/reference/`, as raw little-endian
float32 in C order with no header (shapes in `manifest.json`, which also
records the versions and the sha256 of the weights used):

    input.f32              (1, 3, 518, 518)
    encoder_stage_3.f32    (1, 1369, 384)   the four transformer outputs the
    encoder_stage_6.f32    (1, 1369, 384)   DPT head reads: the output of
    encoder_stage_9.f32    (1, 1369, 384)   layer 3, 6, 9 and 12, each after the
    encoder_stage_12.f32   (1, 1369, 384)   final layer norm, class token removed
    predicted_depth.f32    (1, 518, 518)    the model's output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

SIDE = 518
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
# The 1-based layers whose output the DPT head reads (`out_indices` in the
# backbone configuration of the checkpoint).
OUT_INDICES = (3, 6, 9, 12)
HIDDEN = 384
GRID = SIDE // 14

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = ROOT / "models" / "depth-anything-v2-small"


def reference_input() -> np.ndarray:
    """The input tensor (1, 3, 518, 518), float32, as the module docstring defines it."""
    ys, xs = np.meshgrid(np.arange(SIDE), np.arange(SIDE), indexing="ij")
    levels = np.stack(
        [
            xs * 255 // (SIDE - 1),
            ys * 255 // (SIDE - 1),
            ((xs // 37 + ys // 37) % 2) * 255,
        ]
    ).astype(np.float32)  # integers up to 255 are exact in float32
    mean = np.asarray(MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(STD, dtype=np.float32).reshape(3, 1, 1)
    unit = levels / np.float32(255.0)
    value = (unit - mean) / std
    assert value.dtype == np.float32
    return np.ascontiguousarray(value[None])


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_f32(directory: Path, name: str, array: np.ndarray) -> dict:
    """Writes `array` as raw little-endian float32 and returns its manifest entry."""
    array = np.ascontiguousarray(array, dtype="<f4")
    file = f"{name}.f32"
    array.tofile(directory / file)
    return {"name": name, "file": file, "shape": list(array.shape)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    model_dir: Path = args.model_dir
    weights = model_dir / "model.safetensors"
    if not weights.is_file() or not (model_dir / "config.json").is_file():
        print(
            f"missing {weights} or config.json: run `just setup-depth-anything` first",
            file=sys.stderr,
        )
        return 1

    import torch
    import transformers
    from transformers import DepthAnythingForDepthEstimation

    torch.set_grad_enabled(False)
    model = DepthAnythingForDepthEstimation.from_pretrained(
        model_dir, dtype=torch.float32, local_files_only=True
    ).eval()
    configured = tuple(model.config.backbone_config.out_indices)
    if configured != OUT_INDICES:
        print(f"unexpected out_indices {configured}, expected {OUT_INDICES}", file=sys.stderr)
        return 1

    pixel_values = torch.from_numpy(reference_input())

    # The features the head reads, exactly as the model's forward computes them:
    # the backbone's feature maps (final layer norm applied, class token kept
    # because the backbone does not reshape them), then the class token dropped.
    feature_maps = model.backbone.forward_with_filtered_kwargs(pixel_values).feature_maps
    if len(feature_maps) != len(OUT_INDICES):
        print(f"expected {len(OUT_INDICES)} feature maps, got {len(feature_maps)}", file=sys.stderr)
        return 1
    features = [f[:, 1:, :] for f in feature_maps]
    for f in features:
        if tuple(f.shape) != (1, GRID * GRID, HIDDEN):
            print(f"unexpected feature shape {tuple(f.shape)}", file=sys.stderr)
            return 1

    output = model(pixel_values=pixel_values, output_hidden_states=True)
    depth = output.predicted_depth

    # Cross-check the features against the model's own hidden states: layer k's
    # output through the final layer norm. A disagreement means the features
    # above are not what the head reads, and nothing is written.
    # The check is the reason these files can serve as ground truth, so a model
    # that returns no hidden states writes nothing rather than unchecked files.
    hidden_states = output.hidden_states
    if hidden_states is None:
        print("the model returned no hidden states; the feature cross-check cannot run", file=sys.stderr)
        return 1
    for stage, feature in zip(OUT_INDICES, feature_maps):
        expected = model.backbone.layernorm(hidden_states[stage])
        if not torch.allclose(expected, feature, rtol=0.0, atol=1e-5):
            worst = (expected - feature).abs().max().item()
            print(f"stage {stage} differs from the hidden state route by {worst}", file=sys.stderr)
            return 1

    out_dir = model_dir / "reference"
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = [write_f32(out_dir, "input", pixel_values.numpy())]
    for stage, feature in zip(OUT_INDICES, features):
        entries.append(write_f32(out_dir, f"encoder_stage_{stage}", feature.numpy()))
    entries.append(write_f32(out_dir, "predicted_depth", depth.numpy()))

    manifest = {
        "format": "raw little-endian float32, C order, no header",
        "weights": {"file": "model.safetensors", "sha256": sha256_of(weights)},
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
        },
        "input_formula": "value = (level / 255 - mean[c]) / std[c] in float32; "
        "level: c0 = x*255//517, c1 = y*255//517, c2 = ((x//37 + y//37) % 2) * 255",
        "tensors": entries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    depth_np = depth.numpy()
    print(f"wrote {len(entries)} tensors to {out_dir}")
    print(
        "predicted_depth: shape %s, min %.6g, max %.6g, mean %.6g"
        % (tuple(depth_np.shape), depth_np.min(), depth_np.max(), depth_np.mean())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
