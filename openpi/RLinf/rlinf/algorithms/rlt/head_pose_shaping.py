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

"""Potential-based reward shaping from the Piper connector-head state.

With ``head_pose = [dx, dy, dangle, visible]`` (pixels / rad, see
``rlinf.envs.realworld.piper.wire_head_pose``) the potential is

    phi(s) = -weight * gate(|dy|) * min(|dx|, clip_px) / scale_px

where ``gate`` is 1 once the heads are within ``near_px`` along the
insertion axis, 0 beyond ``far_px`` and linear in between: a lateral offset
only matters when the heads are about to meet. A chunk transition gets
``F = gamma^k * phi(s') - phi(s)`` on top of its reward (Ng et al. 1999), so
the best policy is unchanged while the critic sees, chunk by chunk, whether
a correction reduced the measured offset. Transitions where either state has
no visible heads get no shaping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class HeadPoseShaping:
    weight: float = 0.05
    scale_px: float = 10.0
    clip_px: float = 30.0
    near_px: float = 15.0
    far_px: float = 60.0

    def __post_init__(self):
        if self.weight < 0 or self.scale_px <= 0 or self.clip_px <= 0:
            raise ValueError(
                "head_pose_shaping needs weight >= 0, scale_px > 0 and clip_px > 0."
            )
        if not 0 <= self.near_px < self.far_px:
            raise ValueError("head_pose_shaping needs 0 <= near_px < far_px.")

    @classmethod
    def from_config(cls, cfg: Any) -> HeadPoseShaping | None:
        """``algorithm.head_pose_shaping``; None when absent or disabled."""
        if cfg is None or not bool(cfg.get("enable", False)):
            return None
        return cls(
            **{
                key: float(cfg[key])
                for key in ("weight", "scale_px", "clip_px", "near_px", "far_px")
                if cfg.get(key) is not None
            }
        )

    @property
    def max_abs_potential(self) -> float:
        return self.weight * self.clip_px / self.scale_px

    def potential(self, head_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``phi`` [B, 1] and visibility [B, 1] of a [B, 4] head pose batch."""
        head_pose = head_pose.reshape(head_pose.shape[0], -1).float()
        dx, dy, visible = head_pose[:, 0:1], head_pose[:, 1:2], head_pose[:, 3:4]
        gate = ((self.far_px - dy.abs()) / (self.far_px - self.near_px)).clamp(0, 1)
        offset = dx.abs().clamp(max=self.clip_px) / self.scale_px
        return -self.weight * gate * offset, visible > 0.5

    def __call__(
        self,
        curr_head_pose: torch.Tensor,
        next_head_pose: torch.Tensor,
        bootstrap_discount: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Shaping term [B, 1] for chunk transitions and its metrics.

        ``bootstrap_discount`` is gamma raised to the executed steps, the
        same factor that discounts the next state's Q.
        """
        phi, visible = self.potential(curr_head_pose)
        next_phi, next_visible = self.potential(next_head_pose)
        both = visible & next_visible
        discount = bootstrap_discount.reshape(phi.shape).to(phi)
        shaping = torch.where(both, discount * next_phi - phi, torch.zeros_like(phi))
        shaped = both.float()
        metrics = {
            "head_pose_shaping_mean": float(shaping.mean()),
            "head_pose_shaping_abs_mean": float(shaping.abs().mean()),
            "head_pose_shaped_fraction": float(shaped.mean()),
            "head_pose_potential_mean": float(
                (phi * shaped).sum() / shaped.sum().clamp(min=1.0)
            ),
        }
        return shaping, metrics
