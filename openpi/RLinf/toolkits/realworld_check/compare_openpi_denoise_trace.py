#!/usr/bin/env python3
"""Locate the first divergence in JAX/PyTorch OpenPI denoising traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


INPUT_KEYS = (
    "input_state",
    "input_tokens",
    "input_token_mask",
    "input_images",
    "input_image_masks",
)
MODEL_KEYS = (
    "prefix_tokens",
    "prefix_output",
    "first_suffix_tokens",
    "first_suffix_output",
)


def _numeric(array: np.ndarray) -> np.ndarray:
    """Decode JAX bfloat16 arrays persisted by NumPy as two-byte voids."""
    array = np.asarray(array)
    if array.dtype.kind == "V" and array.dtype.itemsize == 2:
        import ml_dtypes

        return array.view(ml_dtypes.bfloat16).astype(np.float32)
    return array


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        required = (*INPUT_KEYS, *MODEL_KEYS, "denoise_velocity", "denoise_state")
        missing = [key for key in required if key not in archive]
        if missing:
            raise ValueError(f"{path} is missing trace arrays: {missing}")
        return {key: np.asarray(archive[key]) for key in required}


def _error(left: np.ndarray, right: np.ndarray) -> dict[str, float | bool]:
    left = _numeric(left)
    right = _numeric(right)
    if left.shape != right.shape:
        return {
            "shape_equal": False,
            "left_shape": list(left.shape),
            "right_shape": list(right.shape),
        }
    if left.dtype.kind in "biu" and right.dtype.kind in "biu":
        equal = left == right
        return {
            "shape_equal": True,
            "exact_equal": bool(np.all(equal)),
            "mismatch_rate": float(1.0 - np.mean(equal)),
        }
    difference = np.abs(left.astype(np.float32) - right.astype(np.float32))
    return {
        "shape_equal": True,
        "mae": float(difference.mean()),
        "max_abs": float(difference.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jax-trace", type=Path, required=True)
    parser.add_argument("--torch-trace", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    jax_trace = _load(args.jax_trace)
    torch_trace = _load(args.torch_trace)
    if (
        jax_trace["input_images"].ndim == 5
        and torch_trace["input_images"].ndim == 5
        and jax_trace["input_images"].shape
        == torch_trace["input_images"].transpose(0, 1, 3, 4, 2).shape
    ):
        torch_trace["input_images"] = torch_trace["input_images"].transpose(
            0, 1, 3, 4, 2
        )
    report: dict[str, object] = {
        "inputs": {
            key: _error(jax_trace[key], torch_trace[key]) for key in INPUT_KEYS
        },
        "model_stages": {
            key: _error(jax_trace[key], torch_trace[key]) for key in MODEL_KEYS
        },
        "denoise_steps": [],
    }
    prefix_length = jax_trace["prefix_tokens"].shape[-2]
    # Piper uses three 224x224 views with 14x14 SigLIP patches: 256 tokens
    # per view, followed by language tokens.
    image_token_count = min(3 * 16 * 16, prefix_length)
    report["prefix_segments"] = {
        "image_tokens": {
            "embedding": _error(
                jax_trace["prefix_tokens"][..., :image_token_count, :],
                torch_trace["prefix_tokens"][..., :image_token_count, :],
            ),
            "output": _error(
                jax_trace["prefix_output"][..., :image_token_count, :],
                torch_trace["prefix_output"][..., :image_token_count, :],
            ),
        },
        "language_tokens": {
            "embedding": _error(
                jax_trace["prefix_tokens"][..., image_token_count:, :],
                torch_trace["prefix_tokens"][..., image_token_count:, :],
            ),
            "output": _error(
                jax_trace["prefix_output"][..., image_token_count:, :],
                torch_trace["prefix_output"][..., image_token_count:, :],
            ),
        },
    }
    jax_velocity = jax_trace["denoise_velocity"]
    torch_velocity = torch_trace["denoise_velocity"]
    jax_state = jax_trace["denoise_state"]
    torch_state = torch_trace["denoise_state"]
    if jax_velocity.shape != torch_velocity.shape:
        raise ValueError(
            f"velocity trace shapes differ: {jax_velocity.shape} != "
            f"{torch_velocity.shape}"
        )
    if jax_state.shape != torch_state.shape:
        raise ValueError(
            f"state trace shapes differ: {jax_state.shape} != {torch_state.shape}"
        )
    for step in range(jax_velocity.shape[1]):
        report["denoise_steps"].append(
            {
                "step": step + 1,
                "velocity": _error(
                    jax_velocity[:, step], torch_velocity[:, step]
                ),
                "state": _error(jax_state[:, step], torch_state[:, step]),
            }
        )

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
