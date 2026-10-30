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

import torch

from rlinf.models.embodiment.mlp_policy.rlt_action_space import RLTDeltaActionSpace
from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy

# examples/embodiment/config/stats/piper_wire_all_data_rlt_delta_c10.json
Q01 = [-0.051495, -0.228377, -0.133604, -0.185569, -0.157921, -0.193, 0.0001,
       -0.182168, -0.260247, -0.189616, -0.155304, -0.188727, -0.161863, -0.0002]
Q99 = [0.046628, 0.194326, 0.154781, 0.158182, 0.152443, 0.211002, 0.062,
       0.154291, 0.216654, 0.256497, 0.156542, 0.154833, 0.160956, 0.0733]
STATS = {"q01": Q01, "q99": Q99}
ARM = [index for index in range(14) if index % 7 != 6]


def _obs(batch_size: int, chunk_len: int = 10, horizon: int = 50) -> dict:
    generator = torch.Generator().manual_seed(0)
    proprio = torch.rand(batch_size, 14, generator=generator) * 2.0 - 1.0
    proprio[:, [6, 13]] = 0.04
    ref_chunk = proprio[:, None, :] + torch.linspace(0.0, 0.1, horizon)[None, :, None]
    ref_chunk[..., [6, 13]] = 0.05
    return {
        "z_rl": torch.randn(batch_size, 4, generator=generator),
        "proprio": proprio,
        "ref_chunk": ref_chunk,
    }


def _policy() -> RLTMLPPolicy:
    # The deployed residual Actor: mean = reference + zero-initialized delta.
    return RLTMLPPolicy(
        z_dim=4,
        proprio_dim=14,
        action_dim=14,
        num_action_chunks=10,
        ref_num_action_chunks=10,
        action_norm_stats=STATS,
        actor_output="residual",
    )


def test_normalize_round_trip_is_exact() -> None:
    space = RLTDeltaActionSpace.from_config(STATS, action_dim=14)
    obs = _obs(3)
    actions = obs["ref_chunk"][:, :10].reshape(3, -1)

    normalized = space.normalize(actions, obs["proprio"])

    torch.testing.assert_close(space.denormalize(normalized, obs["proprio"]), actions)


def test_arm_joints_are_relative_to_state_and_grippers_absolute() -> None:
    space = RLTDeltaActionSpace.from_config(STATS, action_dim=14)
    state = torch.full((1, 14), 1.5)
    low, high = torch.tensor(Q01), torch.tensor(Q99)

    normalized = space.normalize(state.repeat(1, 10), state).reshape(10, 14)

    zero_delta = (0.0 - low) / (high - low) * 2.0 - 1.0
    absolute = (1.5 - low) / (high - low) * 2.0 - 1.0
    torch.testing.assert_close(normalized[0, ARM], zero_delta[ARM])
    torch.testing.assert_close(normalized[0, [6, 13]], absolute[[6, 13]])


def test_without_stats_the_action_space_is_the_identity() -> None:
    space = RLTDeltaActionSpace.from_config(None, action_dim=14)
    obs = _obs(2)
    actions = obs["ref_chunk"][:, :10].reshape(2, -1)

    torch.testing.assert_close(space.normalize(actions, obs["proprio"]), actions)


def test_normalized_actor_starts_at_reference_with_physical_sigma() -> None:
    policy = _policy()
    obs = _obs(4096)
    reference = obs["ref_chunk"][:, :10].reshape(4096, -1)

    mean, _, _ = policy.sac_forward(obs, deterministic=True)
    sample, _, _ = policy.sac_forward(obs)

    torch.testing.assert_close(mean, reference, atol=1e-5, rtol=0)
    noise = (sample - reference).reshape(-1, 14)
    torch.testing.assert_close(noise[:, ARM].std(), torch.tensor(0.002), rtol=0.05, atol=0)
    torch.testing.assert_close(noise[:, 6].std(), torch.tensor(0.0002), rtol=0.05, atol=0)


def test_critic_scores_actions_in_normalized_space() -> None:
    policy = _policy()
    obs = _obs(2)
    actions = obs["ref_chunk"][:, :10].reshape(2, -1)
    captured = []
    policy.q_head.register_forward_pre_hook(
        lambda module, args: captured.append(args[1])
    )

    policy.sac_q_forward(obs, actions)

    torch.testing.assert_close(
        captured[0], policy.action_space.normalize(actions, obs["proprio"])
    )


def test_sft_loss_is_measured_in_normalized_space() -> None:
    policy = _policy()
    obs = _obs(2)
    target = obs["ref_chunk"][:, :10].reshape(2, -1).clone()
    target[:, 0] += 0.01

    loss = policy.sft_forward({"obs": obs, "action": target})

    expected = (0.01 * 2.0 / (Q99[0] - Q01[0])) ** 2
    torch.testing.assert_close(loss[:, 0], torch.full((2,), expected), rtol=1e-3, atol=0)
