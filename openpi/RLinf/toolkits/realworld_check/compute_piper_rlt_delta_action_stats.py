#!/usr/bin/env python3
"""Quantile statistics for the RLT Stage-2 normalized delta action space (Piper or SO101).

Stage 2 acts on C-step chunks whose arm joints are expressed relative to the
proprioceptive state at the start of the chunk, while each gripper stays
absolute. This mirrors the joint-delta/absolute-gripper split that the pi05
VLA itself trains with, but in physical units (rad / m for Piper, degrees /
0-100 for SO101 with ``--action-dim 12``) so that the actor's normalization
stays linear and independent of the adapt_to_pi transform.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq

from rlinf.utils.arm_layout import arm_width, is_arm_joint


def _column(table: pq.Table, name: str, action_dim: int) -> np.ndarray:
    values = table.column(name).combine_chunks().flatten().to_numpy(zero_copy_only=False)
    return np.asarray(values, dtype=np.float32).reshape(-1, action_dim)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--chunk-len", type=int, default=10)
    parser.add_argument("--start-stride", type=int, default=2)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm-sigma", type=float, default=0.002)
    parser.add_argument("--gripper-sigma", type=float, default=0.0002)
    parser.add_argument("--action-dim", type=int, default=14, help="14 = Piper, 12 = dual SO101")
    args = parser.parse_args()
    action_dim = args.action_dim
    joint_mask = np.array([is_arm_joint(index, action_dim) for index in range(action_dim)])
    joints = arm_width(action_dim) - 1
    names = [
        f"{side}_{part}"
        for side in ("L", "R")[: action_dim // arm_width(action_dim)]
        for part in (*(f"j{i}" for i in range(1, joints + 1)), "grip")
    ]

    files = sorted(glob.glob(os.path.join(args.dataset_dir, "data", "chunk-*", "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No LeRobot parquet files under {args.dataset_dir}")

    windows = []
    absolute = []
    for path in files:
        table = pq.read_table(path, columns=["observation.state", "action"])
        state = _column(table, "observation.state", action_dim)
        action = _column(table, "action", action_dim)
        absolute.append(action)
        # Windows never cross an episode boundary: each parquet is one episode.
        for start in range(0, len(action), args.start_stride):
            chunk = action[start : start + args.chunk_len].copy()
            chunk[:, joint_mask] -= state[start, joint_mask]
            windows.append(chunk)

    relative = np.concatenate(windows)
    absolute = np.concatenate(absolute)
    q01 = np.quantile(relative, 0.01, axis=0)
    q99 = np.quantile(relative, 0.99, axis=0)
    span = q99 - q01
    sigma_phys = np.where(joint_mask, args.arm_sigma, args.gripper_sigma)
    sigma_norm = sigma_phys * 2.0 / np.maximum(span, 1e-6)
    beyond_unit = (np.abs(absolute) > 1.0).mean(axis=0)

    payload = {
        "description": "RLT Stage-2 delta action space: arm joints relative to chunk-start state, grippers absolute; physical units.",
        "dataset_dir": os.path.abspath(args.dataset_dir),
        "episodes": len(files),
        "frames": int(len(absolute)),
        "window_samples": int(len(relative)),
        "chunk_len": args.chunk_len,
        "start_stride": args.start_stride,
        "joint_mask": joint_mask.tolist(),
        "q01": q01.tolist(),
        "q99": q99.tolist(),
        "mean": relative.mean(axis=0).tolist(),
        "std": relative.std(axis=0).tolist(),
        "sigma_physical": sigma_phys.tolist(),
        "sigma_normalized": sigma_norm.tolist(),
        "absolute_min": absolute.min(axis=0).tolist(),
        "absolute_max": absolute.max(axis=0).tolist(),
        "absolute_fraction_beyond_unit": beyond_unit.tolist(),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(payload, handle, indent=2)

    print(f"episodes={len(files)} frames={len(absolute)} window_samples={len(relative)}")
    print(f"{'dim':<7}{'q01':>9}{'q99':>9}{'span':>8}{'sigma_n':>10}{'abs_min':>9}{'abs_max':>9}{'|a|>1':>8}")
    for i, name in enumerate(names):
        print(
            f"{name:<7}{q01[i]:>9.4f}{q99[i]:>9.4f}{span[i]:>8.4f}{sigma_norm[i]:>10.5f}"
            f"{absolute.min(axis=0)[i]:>9.3f}{absolute.max(axis=0)[i]:>9.3f}{beyond_unit[i] * 100:>7.1f}%"
        )
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
