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

from __future__ import annotations

from typing import Any

import torch

from rlinf.utils.arm_layout import is_arm_joint


def default_joint_mask(action_dim: int) -> list[bool]:
    """Arm joints are deltas; the last entry of every arm block is an absolute gripper."""
    return [is_arm_joint(index, action_dim) for index in range(action_dim)]


class RLTDeltaActionSpace(torch.nn.Module):
    """Map absolute joint targets to the actor's normalized delta space and back.

    Arm joints are expressed relative to the proprioceptive state at the start
    of the chunk, grippers stay absolute, and every dimension is mapped from its
    [q01, q99] range to [-1, 1] (paper App. C: "delta action space"). Without
    statistics the map is the identity, which keeps simulator configs that
    already act in normalized units unchanged.
    """

    def __init__(
        self,
        *,
        action_dim: int,
        q01: Any = None,
        q99: Any = None,
        joint_mask: Any = None,
    ):
        super().__init__()
        self.action_dim = int(action_dim)
        self.enabled = q01 is not None or q99 is not None
        if self.enabled:
            if q01 is None or q99 is None:
                raise ValueError("action_norm_stats needs both q01 and q99.")
            low = torch.as_tensor(list(q01), dtype=torch.float32)
            high = torch.as_tensor(list(q99), dtype=torch.float32)
            if low.shape != (self.action_dim,) or high.shape != (self.action_dim,):
                raise ValueError(
                    f"action_norm_stats must have {self.action_dim} values per bound, "
                    f"got q01={tuple(low.shape)} q99={tuple(high.shape)}."
                )
            if not bool(torch.all(high > low)):
                raise ValueError(
                    "action_norm_stats q99 must exceed q01 in every dimension."
                )
            mask_values = (
                default_joint_mask(self.action_dim)
                if joint_mask is None
                else list(joint_mask)
            )
            mask = torch.as_tensor(mask_values, dtype=torch.float32)
            if mask.shape != (self.action_dim,):
                raise ValueError(
                    f"joint_mask must have {self.action_dim} entries, got {tuple(mask.shape)}."
                )
        else:
            low = -torch.ones(self.action_dim)
            high = torch.ones(self.action_dim)
            mask = torch.zeros(self.action_dim)
        self.register_buffer("q01", low, persistent=False)
        self.register_buffer("q99", high, persistent=False)
        self.register_buffer("joint_mask", mask, persistent=False)

    @classmethod
    def from_config(cls, stats: Any, *, action_dim: int) -> RLTDeltaActionSpace:
        if stats is None:
            return cls(action_dim=action_dim)
        return cls(
            action_dim=action_dim,
            q01=stats.get("q01"),
            q99=stats.get("q99"),
            joint_mask=stats.get("joint_mask", None),
        )

    def _bounds(self, like: torch.Tensor):
        return (
            self.q01.to(device=like.device, dtype=like.dtype),
            self.q99.to(device=like.device, dtype=like.dtype),
            self.joint_mask.to(device=like.device, dtype=like.dtype),
        )

    def _state(self, state: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        state = state.reshape(state.shape[0], 1, -1)[..., : self.action_dim]
        return state.to(device=like.device, dtype=like.dtype)

    def normalize(self, actions: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        chunk = actions.reshape(actions.shape[0], -1, self.action_dim)
        low, high, mask = self._bounds(chunk)
        delta = chunk - mask * self._state(state, chunk)
        return ((delta - low) / (high - low) * 2.0 - 1.0).reshape(actions.shape)

    def denormalize(
        self, normalized: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        chunk = normalized.reshape(normalized.shape[0], -1, self.action_dim)
        low, high, mask = self._bounds(chunk)
        delta = (chunk + 1.0) * 0.5 * (high - low) + low
        return (delta + mask * self._state(state, chunk)).reshape(normalized.shape)

    def scale_sigma(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert a per-dimension physical sigma into normalized units."""
        return torch.as_tensor(sigma, dtype=torch.float32) * 2.0 / (self.q99 - self.q01)
