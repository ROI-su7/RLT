#!/usr/bin/env python3
"""Fail-fast comparison of JAX and converted PyTorch Piper action rollouts.

Both input files must be ``.npz`` archives containing an ``actions`` array with
shape ``[samples, horizon, 14]``. Generate the two archives from the same
observations, prompt, normalization statistics, denoising noise, and number of
denoising steps. This tool intentionally does not compare model weights: action
parity is the deployment contract that matters for the robot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ARM_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
GRIPPER_INDICES = np.asarray([6, 13])


def load_actions(path: Path) -> np.ndarray:
    """Load and validate one physical-action archive."""
    with np.load(path) as archive:
        if "actions" not in archive:
            raise ValueError(f"{path} does not contain an 'actions' array")
        actions = np.asarray(archive["actions"], dtype=np.float32)
    if actions.ndim != 3 or actions.shape[-1] != 14:
        raise ValueError(
            f"{path} actions must have shape [samples, horizon, 14], got {actions.shape}"
        )
    if not np.isfinite(actions).all():
        raise ValueError(f"{path} actions contain NaN or infinity")
    return actions


def action_parity_metrics(
    jax_actions: np.ndarray,
    torch_actions: np.ndarray,
    *,
    prefix_steps: int,
    direction_epsilon: float = 1e-4,
) -> dict[str, float]:
    """Compute unit-aware error and temporal direction-agreement metrics."""
    if jax_actions.shape != torch_actions.shape:
        raise ValueError(
            f"JAX/PyTorch action shapes differ: {jax_actions.shape} != "
            f"{torch_actions.shape}"
        )
    if prefix_steps <= 0 or prefix_steps > jax_actions.shape[1]:
        raise ValueError(
            f"prefix_steps must be in [1, {jax_actions.shape[1]}], got {prefix_steps}"
        )
    jax_prefix = jax_actions[:, :prefix_steps]
    torch_prefix = torch_actions[:, :prefix_steps]
    error = np.abs(jax_prefix - torch_prefix)

    jax_delta = np.diff(jax_prefix[..., ARM_INDICES], axis=1)
    torch_delta = np.diff(torch_prefix[..., ARM_INDICES], axis=1)
    moving = (np.abs(jax_delta) >= direction_epsilon) | (
        np.abs(torch_delta) >= direction_epsilon
    )
    if np.any(moving):
        direction_agreement = np.mean(
            np.sign(jax_delta[moving]) == np.sign(torch_delta[moving])
        )
    else:
        direction_agreement = 1.0

    return {
        "arm_mae_rad": float(error[..., ARM_INDICES].mean()),
        "arm_max_abs_rad": float(error[..., ARM_INDICES].max()),
        "gripper_mae_m": float(error[..., GRIPPER_INDICES].mean()),
        "gripper_max_abs_m": float(error[..., GRIPPER_INDICES].max()),
        "arm_delta_direction_agreement": float(direction_agreement),
        "samples": float(jax_actions.shape[0]),
        "compared_prefix_steps": float(prefix_steps),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jax-actions", type=Path, required=True)
    parser.add_argument("--torch-actions", type=Path, required=True)
    parser.add_argument("--prefix-steps", type=int, default=5)
    parser.add_argument("--max-arm-mae-rad", type=float, default=0.01)
    parser.add_argument("--max-arm-error-rad", type=float, default=0.05)
    parser.add_argument("--max-gripper-mae-m", type=float, default=0.002)
    parser.add_argument("--max-gripper-error-m", type=float, default=0.01)
    parser.add_argument("--min-direction-agreement", type=float, default=0.90)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    metrics = action_parity_metrics(
        load_actions(args.jax_actions),
        load_actions(args.torch_actions),
        prefix_steps=args.prefix_steps,
    )
    limits = {
        "arm_mae_rad": args.max_arm_mae_rad,
        "arm_max_abs_rad": args.max_arm_error_rad,
        "gripper_mae_m": args.max_gripper_mae_m,
        "gripper_max_abs_m": args.max_gripper_error_m,
    }
    failures = [
        f"{name}={metrics[name]:.6g} > {limit:.6g}"
        for name, limit in limits.items()
        if metrics[name] > limit
    ]
    if metrics["arm_delta_direction_agreement"] < args.min_direction_agreement:
        failures.append(
            "arm_delta_direction_agreement="
            f"{metrics['arm_delta_direction_agreement']:.4f} < "
            f"{args.min_direction_agreement:.4f}"
        )
    report = {"passed": not failures, "metrics": metrics, "failures": failures}
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json is not None:
        args.output_json.write_text(rendered + "\n")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
