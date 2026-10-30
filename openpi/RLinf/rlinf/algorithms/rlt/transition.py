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

from typing import Any

import torch

from rlinf.envs import SupportedEnvType
from rlinf.utils.nested_dict_process import copy_dict_tensor

RLT_OBS_KEYS = ("z_rl", "proprio", "ref_chunk")
# Copied when present. ``bridge_delta`` (Piper execution-side RTC) is the last
# queued bridge command minus proprio; simulator RLT has no bridge.
# ``head_pose`` is the Piper connector-head keypoint state.
RLT_OPTIONAL_OBS_KEYS = ("bridge_delta", "head_pose")
RLT_TRANSITION_PREFIX = "rlt_transition_"


def finalize_keyboard_transition(trajectory: Any, reward: float) -> bool:
    """Attach an operator outcome once to the final real action of a row."""
    if torch.as_tensor(trajectory.dones).bool().any():
        return False
    mask = trajectory.forward_inputs.get("action_valid_mask")
    valid = (
        torch.ones_like(trajectory.rewards, dtype=torch.bool)
        if mask is None
        else torch.as_tensor(mask).bool()
    )
    positions = valid.reshape(-1).nonzero().reshape(-1)
    if positions.numel() == 0:
        return False
    last = int(positions[-1])
    trajectory.rewards = trajectory.rewards.clone()
    trajectory.rewards.reshape(-1)[last] = reward
    for field in ("dones", "terminations"):
        values = getattr(trajectory, field).clone()
        values.reshape(-1)[last] = True
        setattr(trajectory, field, values)
    return True


def use_simulator_transition_replay(cfg: Any) -> bool:
    """Return True for envs that store one replay row per env step."""
    train_env_cfg = cfg.env.get("train", None)
    if train_env_cfg is None:
        return False
    try:
        return (
            SupportedEnvType(train_env_cfg.get("env_type", ""))
            == SupportedEnvType.MANISKILL_RLT
        )
    except ValueError:
        return False


def use_transition_replay(cfg: Any) -> bool:
    """Return True when replay ingestion stores one file per transition.

    Simulator RLT has always used transition replay. Real-world deployments can
    opt in explicitly with ``algorithm.replay_buffer.save_transitions`` while
    retaining the real-world policy-switch route.
    """
    replay_cfg = cfg.algorithm.get("replay_buffer", {}) or {}
    return bool(replay_cfg.get("save_transitions", False)) or (
        use_simulator_transition_replay(cfg)
    )


def extract_rlt_obs_from_forward_inputs(
    forward_inputs: dict[str, Any],
    *,
    transition: bool = False,
) -> dict[str, Any]:
    prefix = RLT_TRANSITION_PREFIX if transition else ""
    missing = [
        f"{prefix}{key}"
        for key in RLT_OBS_KEYS
        if f"{prefix}{key}" not in forward_inputs
    ]
    if missing:
        raise ValueError(
            f"Missing RLT forward_inputs keys: {missing}. Ensure "
            "rollout.rlt_feature_model is configured and the rollout worker "
            "populates RLT features."
        )
    keys = RLT_OBS_KEYS + tuple(
        key for key in RLT_OPTIONAL_OBS_KEYS if f"{prefix}{key}" in forward_inputs
    )
    return copy_dict_tensor({key: forward_inputs[f"{prefix}{key}"] for key in keys})


def shift_reference_plan(ref_chunk: torch.Tensor, offset: int) -> torch.Tensor:
    """Re-anchor a [B, H, D] VLA plan at execution step ``offset``.

    With execution-side RTC the i-th action of a chunk is plan sample
    ``offset + i``, because ``offset`` bridge commands ran while the plan was
    being inferred. The shifted plan is the reference those actions, and the
    actor that produces them, must be compared with. The final sample is
    repeated so every replay row keeps H steps.
    """
    offset = int(offset)
    if offset <= 0:
        return ref_chunk
    if ref_chunk.dim() != 3:
        raise ValueError(
            f"Expected a [B, H, D] reference plan, got {tuple(ref_chunk.shape)}."
        )
    if offset >= ref_chunk.shape[1]:
        raise ValueError(
            f"Reference offset {offset} must be smaller than the plan length "
            f"{ref_chunk.shape[1]}."
        )
    tail = ref_chunk[:, offset:]
    return torch.cat((tail, tail[:, -1:].expand(-1, offset, -1)), dim=1)


def replace_reference_with_intervention(
    rlt_obs: dict[str, Any],
    intervene_actions: torch.Tensor,
    intervene_flags: torch.Tensor,
) -> None:
    """Paper Sec. IV-B: an intervention replaces the VLA reference in the buffer.

    Only steps the operator actually drove are replaced, so the actor input and
    the BC target both see the human action for those steps.
    """
    ref_chunk = rlt_obs["ref_chunk"]
    batch_size = ref_chunk.shape[0]
    flags = intervene_flags.reshape(batch_size, -1, 1).to(
        device=ref_chunk.device, dtype=torch.bool
    )
    human_actions = intervene_actions.reshape(batch_size, flags.shape[1], -1).to(
        device=ref_chunk.device, dtype=ref_chunk.dtype
    )
    reference = ref_chunk.reshape(batch_size, -1, human_actions.shape[-1]).clone()
    reference[:, : flags.shape[1]] = torch.where(
        flags, human_actions, reference[:, : flags.shape[1]]
    )
    rlt_obs["ref_chunk"] = reference.reshape_as(ref_chunk)


def update_rlt_transitions(
    stage_id: int,
    pending_obs: list[dict[str, Any] | None],
    rollout_results: list[Any],
    rollout_result: Any,
    *,
    cache_current: bool,
    intervene_actions: torch.Tensor | None = None,
    intervene_flags: torch.Tensor | None = None,
    curr_ref_offset: int = 0,
    next_ref_offset: int = 0,
    replace_human_reference: bool = True,
) -> None:
    """Close the pending transition and optionally open the next one.

    ``curr_ref_offset`` is the RTC offset actually applied to the chunk being
    opened. ``next_ref_offset`` is the offset the next chunk will run with,
    which only matters for the critic's bootstrap action. With
    ``replace_human_reference=False`` operator steps keep the VLA plan as the
    Actor input while BC still targets the executed operator action, so the
    Actor learns the operator's correction relative to the VLA.
    """
    if pending_obs[stage_id] is not None:
        if (
            replace_human_reference
            and intervene_actions is not None
            and intervene_flags is not None
        ):
            replace_reference_with_intervention(
                pending_obs[stage_id], intervene_actions, intervene_flags
            )
        next_obs = extract_rlt_obs_from_forward_inputs(
            rollout_result.forward_inputs,
            transition=True,
        )
        next_obs["ref_chunk"] = shift_reference_plan(
            next_obs["ref_chunk"], next_ref_offset
        )
        rollout_results[stage_id].append_transitions(
            pending_obs[stage_id],
            next_obs,
        )
        pending_obs[stage_id] = None

    if cache_current:
        curr_obs = extract_rlt_obs_from_forward_inputs(rollout_result.forward_inputs)
        curr_obs["ref_chunk"] = shift_reference_plan(
            curr_obs["ref_chunk"], curr_ref_offset
        )
        pending_obs[stage_id] = curr_obs
