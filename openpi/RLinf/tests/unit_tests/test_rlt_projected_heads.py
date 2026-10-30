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

"""Projected (openpi-RLT) actor and twin critic for RLT Stage 2."""

import pytest
import torch

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
from rlinf.models.embodiment.modules.rlt_projected_heads import RLTProjectedTwinQ

Z, P, C, D = 2048, 14, 10, 14
STATS = {"q01": [-0.2] * D, "q99": [0.2] * D}


def _policy(**kwargs) -> RLTMLPPolicy:
    options = {
        "z_dim": Z,
        "proprio_dim": P,
        "action_dim": D,
        "num_action_chunks": C,
        "fixed_std_is_normalized": True,
        "execute_actor_mean": True,
        "action_norm_stats": STATS,
        "q_head_type": "rlt_projected",
        "actor_architecture": "projected",
    }
    options.update(kwargs)
    return RLTMLPPolicy(**options)


def _obs(batch=4):
    torch.manual_seed(0)
    return {
        "z_rl": torch.randn(batch, Z) * 0.4 + 0.9,
        "proprio": torch.randn(batch, P) * 0.5,
        "ref_chunk": torch.randn(batch, 50, D) * 0.05,
    }


def test_projected_policy_shapes_and_parameter_groups():
    policy = _policy()
    obs = _obs()
    actions, _, _ = policy(forward_type=ForwardType.SAC, obs=obs, deterministic=True)
    assert actions.shape == (4, C * D)
    q = policy(forward_type=ForwardType.SAC_Q, obs=obs, actions=actions)
    assert q.shape == (4, 2)
    assert isinstance(policy.q_head, RLTProjectedTwinQ)
    names = [name for name, _ in policy.named_parameters()]
    # The SAC worker routes "q_head" parameters to the critic optimizer.
    actor = [n for n in names if "q_head" not in n]
    assert actor and all(n.startswith("actor_net.") for n in actor)
    assert not any(k in n for n in actor for k in ("encoder", "state_proj"))


def test_projected_critic_gives_the_action_equal_footing_with_z():
    """A bounded correction must move Q as much as typical z variation does.

    With raw concatenation the 2048-D token dominated the first layer; here each
    input is LayerNorm-ed on its own branch at initialization.
    """
    head = RLTProjectedTwinQ(Z, P, C * D)
    z = torch.randn(256, Z) * 0.4 + 0.9
    proprio = torch.randn(256, P) * 0.5
    action = torch.randn(256, C * D) * 0.3
    state = torch.cat((z, proprio), dim=-1)
    inputs = head.inputs[0]
    z_variation = (
        (inputs.z_proj(z) - inputs.z_proj(z.mean(0, keepdim=True))).norm(dim=-1).mean()
    )
    a_variation = (
        (inputs.chunk_proj(action + 0.05) - inputs.chunk_proj(action))
        .norm(dim=-1)
        .mean()
    )
    assert a_variation > 0.05 * z_variation
    assert head(state, action).shape == (256, 2)


def test_mlp_architecture_remains_the_default():
    policy = RLTMLPPolicy(
        z_dim=Z,
        proprio_dim=P,
        action_dim=D,
        num_action_chunks=C,
        action_norm_stats=STATS,
    )
    assert policy.actor_architecture == "mlp"
    assert not hasattr(policy, "actor_net")
    with pytest.raises(ValueError, match="actor_architecture"):
        _policy(actor_architecture="transformer")
