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
import torch.nn as nn

from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy


def _make_policy(chunk_len: int = 3) -> RLTMLPPolicy:
    return RLTMLPPolicy(
        z_dim=4,
        proprio_dim=14,
        action_dim=14,
        num_action_chunks=chunk_len,
        fixed_std=0.05,
        fixed_gripper_std=0.05,
        fixed_std_is_normalized=True,
    )


def _make_obs(batch_size: int = 2, chunk_len: int = 3) -> dict[str, torch.Tensor]:
    reference_step = torch.tensor(
        [
            -0.63,
            1.74,
            -1.07,
            0.72,
            1.21,
            1.29,
            0.05,
            0.72,
            2.44,
            -2.20,
            0.93,
            1.25,
            -1.06,
            0.06,
        ]
    )
    return {
        "z_rl": torch.randn(batch_size, 4),
        "proprio": torch.randn(batch_size, 14),
        "ref_chunk": reference_step.repeat(batch_size, chunk_len, 1),
    }


def test_rlt_actor_uses_random_full_output_instead_of_reference_skip() -> None:
    policy = _make_policy()
    obs = _make_obs()

    action, _, _ = policy.sac_forward(obs, deterministic=True)
    reference = obs["ref_chunk"].reshape(action.shape)

    assert not torch.allclose(action, reference)
    assert torch.count_nonzero(policy.actor_mean.weight).item() > 0


def test_rlt_actor_architecture_and_initialization_match_b() -> None:
    policy = _make_policy()

    assert policy.actor_hidden_dim == 256
    assert policy.actor_num_layers == 2
    assert isinstance(policy.backbone[0], nn.Linear)
    assert isinstance(policy.backbone[1], nn.ReLU)
    assert len(policy.backbone) == 2
    assert isinstance(policy.actor_mean, nn.Linear)
    assert policy.actor_logstd is None


def test_rlt_reference_is_only_an_actor_input() -> None:
    policy = _make_policy()
    obs = _make_obs()
    with torch.no_grad():
        policy.actor_mean.weight.zero_()
        policy.actor_mean.bias.fill_(0.3)

    dropped = policy._action_distribution(
        obs,
        apply_reference_dropout=True,
        reference_dropout_prob=1.0,
    ).mean
    kept = policy._action_distribution(
        obs,
        apply_reference_dropout=True,
        reference_dropout_prob=0.0,
    ).mean

    torch.testing.assert_close(dropped, torch.full_like(dropped, 0.3))
    torch.testing.assert_close(kept, torch.full_like(kept, 0.3))


def test_rlt_actor_full_output_is_not_structurally_tied_to_reference() -> None:
    policy = _make_policy(chunk_len=20)
    obs = _make_obs(batch_size=8, chunk_len=20)
    with torch.no_grad():
        policy.actor_mean.bias.fill_(5.0)

    action, _, _ = policy.sac_forward(obs, deterministic=True)
    reference = obs["ref_chunk"].reshape(action.shape)

    # The removed box capped this at 0.02 rad for arm joints.
    assert (action - reference).abs().max() > 1.0


def test_rlt_actor_sigma_matches_b_in_normalized_space() -> None:
    policy = _make_policy(chunk_len=20)
    obs = _make_obs(batch_size=4096, chunk_len=20)

    distribution = policy._action_distribution(
        obs, apply_reference_dropout=False, reference_dropout_prob=0.0
    )
    torch.testing.assert_close(
        distribution.stddev, torch.full_like(distribution.stddev, 0.05)
    )


def test_resume_uses_current_configured_sigma_not_checkpoint_buffer() -> None:
    old_policy = _make_policy()
    new_policy = RLTMLPPolicy(
        z_dim=4,
        proprio_dim=14,
        action_dim=14,
        num_action_chunks=3,
        fixed_std=0.005,
        fixed_gripper_std=0.005,
        fixed_std_is_normalized=True,
    )

    new_policy.load_state_dict(old_policy.state_dict(), strict=True)

    torch.testing.assert_close(
        new_policy.action_std,
        torch.full_like(new_policy.action_std, 0.005),
    )


def test_real_robot_executes_mean_while_sac_training_remains_stochastic() -> None:
    policy = _make_policy()
    policy.execute_actor_mean = True
    obs = _make_obs()

    torch.manual_seed(1)
    first, _ = policy.predict_action_batch(obs, mode="train")
    torch.manual_seed(2)
    second, _ = policy.predict_action_batch(obs, mode="train")
    torch.testing.assert_close(first, second)

    torch.manual_seed(1)
    sampled_first, _, _ = policy.sac_forward(obs)
    torch.manual_seed(2)
    sampled_second, _, _ = policy.sac_forward(obs)
    assert not torch.allclose(sampled_first, sampled_second)


def test_rlt_sft_initial_loss_is_nonzero_for_reference_target() -> None:
    policy = _make_policy()
    obs = _make_obs()
    target = obs["ref_chunk"].reshape(obs["ref_chunk"].shape[0], -1)

    loss = policy.sft_forward({"obs": obs, "action": target})

    assert loss.mean() > 0
