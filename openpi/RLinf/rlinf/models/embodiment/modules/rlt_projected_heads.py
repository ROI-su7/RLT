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

"""RLT Stage-2 heads with per-input projections (openpi-RLT ``networks.py``).

The 2048-D RL token would otherwise dominate the first layer of a plain
concatenating MLP: on the Piper campaign of 2026-10-14 the state-dependent z
signal in the critic's first layer was ~24x larger than the actor correction
it is supposed to score, and the critic fit its TD targets from z alone.
Each input is therefore projected on its own and LayerNorm-ed before the
trunk: z -> 256 (LayerNorm), proprio -> 64 and action/reference -> 256
(LayerNorm + tanh), then ``[Linear, LayerNorm, GELU] * L + Linear``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _linear(in_dim: int, out_dim: int) -> nn.Linear:
    layer = nn.Linear(in_dim, out_dim)
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class _Projection(nn.Module):
    """Linear projection, parameter-free LayerNorm, optional tanh."""

    def __init__(self, in_dim: int, out_dim: int, *, squash: bool):
        super().__init__()
        self.linear = _linear(in_dim, out_dim)
        self.norm = nn.LayerNorm(out_dim, elementwise_affine=False, eps=1e-6)
        self.squash = squash

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(self.linear(x))
        return torch.tanh(x) if self.squash else x


class _Trunk(nn.Module):
    """``[Linear, LayerNorm, GELU] * num_layers + Linear``."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, out_dim: int):
        super().__init__()
        if hidden_dim <= 0 or num_layers < 1:
            raise ValueError("Trunk needs hidden_dim > 0 and num_layers >= 1.")
        layers: list[nn.Module] = []
        for _ in range(num_layers):
            layers += [
                _linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6),
                nn.GELU(),
            ]
            in_dim = hidden_dim
        layers.append(_linear(in_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RLTProjectedInputs(nn.Module):
    """Project (z, proprio, chunk) separately and concatenate the features."""

    def __init__(
        self,
        z_dim: int,
        proprio_dim: int,
        chunk_dim: int,
        *,
        z_features: int = 256,
        proprio_features: int = 64,
        chunk_features: int = 256,
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.z_proj = _Projection(z_dim, z_features, squash=False)
        self.proprio_proj = _Projection(proprio_dim, proprio_features, squash=True)
        self.chunk_proj = _Projection(chunk_dim, chunk_features, squash=True)
        self.out_dim = z_features + proprio_features + chunk_features

    def forward(
        self, z: torch.Tensor, proprio: torch.Tensor, chunk: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat(
            (self.z_proj(z), self.proprio_proj(proprio), self.chunk_proj(chunk)),
            dim=-1,
        )


class RLTProjectedActor(nn.Module):
    """Full-output chunk actor: mean = trunk(proj(z), proj(proprio), proj(ref))."""

    def __init__(
        self,
        z_dim: int,
        proprio_dim: int,
        chunk_dim: int,
        *,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        self.inputs = RLTProjectedInputs(z_dim, proprio_dim, chunk_dim)
        self.trunk = _Trunk(self.inputs.out_dim, hidden_dim, num_layers, chunk_dim)

    def forward(
        self, z: torch.Tensor, proprio: torch.Tensor, reference: torch.Tensor
    ) -> torch.Tensor:
        return self.trunk(self.inputs(z, proprio, reference))


class RLTProjectedTwinQ(nn.Module):
    """Twin critic with the same projected inputs; returns ``[B, num_q_heads]``.

    Matches the ``q_head(state_features, action_features)`` interface of
    :class:`MultiQHead`, where ``state_features = cat(z, proprio)``.
    """

    def __init__(
        self,
        z_dim: int,
        proprio_dim: int,
        chunk_dim: int,
        *,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_q_heads: int = 2,
    ):
        super().__init__()
        self.z_dim = int(z_dim)
        self.proprio_dim = int(proprio_dim)
        self.num_q_heads = int(num_q_heads)
        self.inputs = nn.ModuleList(
            RLTProjectedInputs(z_dim, proprio_dim, chunk_dim)
            for _ in range(self.num_q_heads)
        )
        self.qs = nn.ModuleList(
            _Trunk(inputs.out_dim, hidden_dim, num_layers, 1) for inputs in self.inputs
        )

    def forward(
        self, state_features: torch.Tensor, action_features: torch.Tensor
    ) -> torch.Tensor:
        z = state_features[..., : self.z_dim]
        proprio = state_features[..., self.z_dim : self.z_dim + self.proprio_dim]
        return torch.cat(
            [
                q(inputs(z, proprio, action_features))
                for inputs, q in zip(self.inputs, self.qs)
            ],
            dim=-1,
        )
