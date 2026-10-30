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

"""The RTC bridge end point is part of the Piper RLT state."""

import numpy as np
import torch

from rlinf.algorithms.rlt.subsample import StrideWindowAssembler
from rlinf.algorithms.rlt.transition import extract_rlt_obs_from_forward_inputs
from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
from rlinf.models.embodiment.native_jax_rlt import _with_bridge_delta
from rlinf.workers.env.env_worker import rlt_bridge_target

D, H, Z = 14, 50, 4


def _policy(bridge_delta_dim: int) -> RLTMLPPolicy:
    torch.manual_seed(0)
    return RLTMLPPolicy(
        z_dim=Z,
        proprio_dim=D,
        action_dim=D,
        num_action_chunks=10,
        q_head_type="rlt_projected",
        actor_architecture="projected",
        actor_output="residual",
        bridge_delta_dim=bridge_delta_dim,
    )


def _obs(batch: int = 3) -> dict:
    proprio = torch.rand(batch, D)
    return {
        "z_rl": torch.randn(batch, Z),
        "proprio": proprio,
        "ref_chunk": proprio[:, None] + torch.linspace(0, 0.05, H)[None, :, None],
    }


def test_bridge_target_is_the_last_queued_bridge_command() -> None:
    states = torch.zeros(2, D)
    tail = np.arange(2 * 25 * D, dtype=np.float32).reshape(2, 25, D)

    target = rlt_bridge_target(states, tail, 15)

    torch.testing.assert_close(target, torch.as_tensor(tail[:, 14]))


def test_without_a_bridge_the_chunk_starts_from_the_measured_state() -> None:
    states = torch.full((2, D), 0.5)
    tail = np.ones((2, 25, D), dtype=np.float32)

    torch.testing.assert_close(rlt_bridge_target(states, tail, 0), states)
    torch.testing.assert_close(rlt_bridge_target(states, None, 15), states)
    # A short suffix bridges only what was queued.
    short = np.arange(2 * 3 * D, dtype=np.float32).reshape(2, 3, D)
    torch.testing.assert_close(
        rlt_bridge_target(states, short, 15), torch.as_tensor(short[:, 2])
    )


def test_bridge_target_reaches_the_rollout_observation() -> None:
    obs = {
        "states": torch.zeros(1, D),
        "main_images": torch.zeros(1, 4, 4, 3),
        "rlt_bridge_target": torch.ones(1, D),
    }
    prepared = EnvOutput(obs=obs).prepare_observations(obs)
    torch.testing.assert_close(prepared["rlt_bridge_target"], torch.ones(1, D))
    del obs["rlt_bridge_target"]
    assert "rlt_bridge_target" not in EnvOutput(obs=obs).prepare_observations(obs)


def test_feature_bridge_delta_is_target_minus_state() -> None:
    states = np.full((2, D), 0.25, dtype=np.float32)
    rlt_obs = {"proprio": torch.as_tensor(states)}

    with_target = _with_bridge_delta(
        rlt_obs, {"rlt_bridge_target": torch.full((2, D), 0.30)}, states
    )
    without = _with_bridge_delta(rlt_obs, {}, states)

    torch.testing.assert_close(with_target["bridge_delta"], torch.full((2, D), 0.05))
    torch.testing.assert_close(without["bridge_delta"], torch.zeros(2, D))
    assert "bridge_delta" not in rlt_obs


def test_transition_obs_carry_bridge_delta_only_when_present() -> None:
    inputs = {
        "z_rl": torch.zeros(1, Z),
        "proprio": torch.zeros(1, D),
        "ref_chunk": torch.zeros(1, H, D),
    }
    assert "bridge_delta" not in extract_rlt_obs_from_forward_inputs(inputs)
    inputs["bridge_delta"] = torch.ones(1, D)
    obs = extract_rlt_obs_from_forward_inputs(inputs)
    torch.testing.assert_close(obs["bridge_delta"], torch.ones(1, D))


def test_actor_and_critic_see_the_bridge_delta() -> None:
    policy = _policy(D)
    obs = _obs()
    # Make the residual head non-zero so the actor output can differ.
    for parameter in policy.actor_net.parameters():
        torch.nn.init.normal_(parameter, std=0.1)
    actions = obs["ref_chunk"][:, :10].reshape(3, -1)

    zero = {**obs, "bridge_delta": torch.zeros(3, D)}
    moved = {**obs, "bridge_delta": torch.full((3, D), 0.04)}
    mean_zero, _, _ = policy.sac_forward(zero, deterministic=True)
    mean_moved, _, _ = policy.sac_forward(moved, deterministic=True)
    q_zero = policy.sac_q_forward(zero, actions)
    q_moved = policy.sac_q_forward(moved, actions)

    assert not torch.allclose(mean_zero, mean_moved)
    assert not torch.allclose(q_zero, q_moved)
    # A missing delta is a zero delta (first chunk, operator control).
    torch.testing.assert_close(
        policy.sac_forward(obs, deterministic=True)[0], mean_zero
    )


def test_bridge_delta_does_not_change_the_action_space() -> None:
    """Actions stay normalized against the raw proprio, not the delta."""
    policy = _policy(D)
    obs = {**_obs(), "bridge_delta": torch.full((3, D), 0.04)}
    reference = obs["ref_chunk"][:, :10].reshape(3, -1)

    mean, _, _ = policy.sac_forward(obs, deterministic=True)

    # The residual head starts at zero, so the actor reproduces the VLA plan.
    torch.testing.assert_close(mean, reference, atol=1e-5, rtol=0)


def test_disabled_bridge_delta_keeps_the_previous_network() -> None:
    assert _policy(0).actor_net.inputs.proprio_proj.linear.in_features == D
    assert _policy(D).actor_net.inputs.proprio_proj.linear.in_features == 2 * D


def test_command_after_the_episode_end_is_the_held_final_command() -> None:
    assembler = StrideWindowAssembler(chunk_len=10, stride=2, latency=15, horizon=H)
    chunk = assembler.add_chunk(
        bridge=torch.zeros(0, D),
        actions=torch.arange(10, dtype=torch.float32)[:, None].repeat(1, D),
        plan=torch.zeros(H, D),
        z_rl=torch.zeros(Z),
        proprio=torch.zeros(D),
        record=True,
    )
    assembler.add_outcome(chunk, rewards=torch.zeros(10), terminal_step=6)

    segment = assembler._chunks[chunk].segment
    torch.testing.assert_close(
        assembler._command_at(segment, 40), torch.full((D,), 6.0)
    )
    torch.testing.assert_close(assembler._command_at(segment, 8), torch.full((D,), 6.0))
    torch.testing.assert_close(assembler._command_at(segment, 3), torch.full((D,), 3.0))
