#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Widen a Stage-2 RLT checkpoint's state input for new state features.

New state features (e.g. ``actor.model.head_pose_dim``) are appended to the
proprio branch of the Actor and both critics, so a checkpoint trained
without them has ``*proprio_proj.linear.weight`` of shape [F, S] where the
new model expects [F, S + extra]. This copies the checkpoint and appends
``extra`` zero columns to those weights in the model, the target model, the
full-weights export and the Adam moments, so the resumed networks compute
exactly what they did before and learn to use the new inputs from there.

    python toolkits/wire_head_pose/pad_stage2_checkpoint.py \
        --src logs/<run>/.../checkpoints/global_step_900 \
        --dst logs/<run>/.../checkpoints/global_step_900_head_pose --extra 4
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from typing import Any

import torch
from torch.distributed.checkpoint.format_utils import (
    dcp_to_torch_save,
    torch_save_to_dcp,
)

PADDED_SUFFIX = "proprio_proj.linear.weight"


def pad_columns(tensor: torch.Tensor, old_in: int, extra: int) -> torch.Tensor:
    if tensor.dim() != 2 or tensor.shape[1] != old_in:
        raise ValueError(
            f"Expected a [F, {old_in}] state projection, got {tuple(tensor.shape)}."
        )
    return torch.cat((tensor, tensor.new_zeros(tensor.shape[0], extra)), dim=1)


def pad_model_state(state: dict[str, Any], old_in: int, extra: int) -> int:
    """Pad every state projection weight in a flat model state dict in place."""
    count = 0
    for key in list(state):
        if key.endswith(PADDED_SUFFIX):
            state[key] = pad_columns(state[key], old_in, extra)
            count += 1
    return count


def pad_optimizer_state(state: dict[str, Any], old_in: int, extra: int) -> int:
    """Pad the Adam moments of state projection weights (``state/<param>/...``)."""
    count = 0
    for param, moments in state.get("state", {}).items():
        if not param.endswith(PADDED_SUFFIX):
            continue
        for name in ("exp_avg", "exp_avg_sq"):
            if name in moments:
                moments[name] = pad_columns(moments[name], old_in, extra)
                count += 1
    return count


def state_input_dim(state: dict[str, Any]) -> int:
    dims = {v.shape[1] for k, v in state.items() if k.endswith(PADDED_SUFFIX)}
    if len(dims) != 1:
        raise ValueError(f"Expected one state projection width, found {dims}.")
    return dims.pop()


def pad_checkpoint(src: str, dst: str, extra: int) -> dict[str, int]:
    if extra <= 0:
        raise ValueError("extra must be positive.")
    if os.path.exists(dst):
        raise FileExistsError(f"{dst} exists; choose a new destination.")
    shutil.copytree(src, dst)
    actor = os.path.join(dst, "actor")
    counts = {}

    full_path = os.path.join(actor, "model_state_dict", "full_weights.pt")
    full = torch.load(full_path, map_location="cpu", weights_only=False)
    old_in = state_input_dim(full)
    counts["full_weights"] = pad_model_state(full, old_in, extra)
    torch.save(full, full_path)

    target_dir = os.path.join(actor, "sac_components", "target_model")
    for name in sorted(os.listdir(target_dir)):
        path = os.path.join(target_dir, name)
        target = torch.load(path, map_location="cpu", weights_only=False)
        counts[f"target/{name}"] = pad_model_state(target, old_in, extra)
        torch.save(target, path)

    dcp_dir = os.path.join(actor, "dcp_checkpoint")
    with tempfile.TemporaryDirectory() as tmp:
        flat = os.path.join(tmp, "dcp.pt")
        dcp_to_torch_save(dcp_dir, flat)
        training = torch.load(flat, map_location="cpu", weights_only=False)
        state = training["fsdp_checkpoint"]
        counts["dcp/model"] = pad_model_state(state["model"], old_in, extra)
        counts["dcp/optimizer"] = pad_optimizer_state(
            state["optimizers"], old_in, extra
        )
        shutil.rmtree(dcp_dir)
        torch.save(training, flat)
        torch_save_to_dcp(flat, dcp_dir)
    counts["old_state_dim"] = old_in
    counts["new_state_dim"] = old_in + extra
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", required=True, help="checkpoints/global_step_N")
    parser.add_argument("--dst", required=True)
    parser.add_argument("--extra", type=int, default=4, help="appended state dims")
    args = parser.parse_args()
    for key, value in pad_checkpoint(args.src, args.dst, args.extra).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
