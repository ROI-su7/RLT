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

"""Residual Actor, teaching-row selection, split sampling, critic alignment."""

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import (
    TrajectoryReplayBuffer,
    human_reference_replaced,
    rlt_sampling_tags,
)
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACLossMixin

Z, P, C, D, H = 2048, 14, 10, 14, 50
STATS = {"q01": [-0.2] * D, "q99": [0.2] * D}


def _obs(batch=4):
    torch.manual_seed(0)
    return {
        "z_rl": torch.randn(batch, Z) * 0.4 + 0.9,
        "proprio": torch.randn(batch, P) * 0.3,
        "ref_chunk": torch.randn(batch, H, D) * 0.05,
    }


def _policy(**kwargs):
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
        "actor_output": "residual",
    }
    options.update(kwargs)
    return RLTMLPPolicy(**options)


# ---------------------------------------------------------------- A1 residual
@pytest.mark.parametrize("architecture", ["projected", "mlp"])
def test_untrained_residual_actor_reproduces_the_reference_exactly(architecture):
    policy = _policy(actor_architecture=architecture)
    obs = _obs()
    actions, _, _ = policy(forward_type=ForwardType.SAC, obs=obs, deterministic=True)
    torch.testing.assert_close(
        actions, obs["ref_chunk"][:, :C].reshape(4, -1), atol=1e-6, rtol=0
    )
    # Reference dropout hides the reference from the input, never the base.
    dropped, _, _ = policy(
        forward_type=ForwardType.SAC,
        obs=obs,
        deterministic=True,
        apply_reference_dropout=True,
        reference_dropout_prob=1.0,
    )
    torch.testing.assert_close(dropped, actions, atol=1e-6, rtol=0)


def test_residual_actor_learns_a_correction():
    policy = _policy()
    obs = _obs()
    target = obs["ref_chunk"][:, :C].reshape(4, -1) + 0.01
    optim = torch.optim.Adam(policy.actor_net.parameters(), lr=1e-3)
    for _ in range(200):
        actions, _, _ = policy(
            forward_type=ForwardType.SAC, obs=obs, deterministic=True
        )
        loss = (actions - target).square().mean()
        optim.zero_grad()
        loss.backward()
        optim.step()
    assert loss.item() < 1e-6
    with pytest.raises(ValueError, match="actor_output"):
        _policy(actor_output="delta")


# ------------------------------------------------------- A2 teaching rows
def _transition(index, *, human=False, replaced=False):
    ref = torch.full((1, 1, H, D), float(index))
    actions = torch.full((1, 1, C * D), float(index) + (0.0 if replaced else 0.01))
    return Trajectory(
        max_episode_length=1,
        model_weights_id="test",
        actions=actions,
        rewards=torch.zeros(1, 1, C),
        dones=torch.zeros(1, 1, C, dtype=torch.bool),
        intervene_flags=torch.full((1, 1, C * D), human),
        curr_obs={"z_rl": torch.full((1, 1, 1), float(index)), "ref_chunk": ref},
        next_obs={"z_rl": torch.full((1, 1, 1), float(index)), "ref_chunk": ref},
        forward_inputs={
            "reference_warmup": torch.zeros(1, 1, 1, dtype=torch.bool),
            "actor_switch": torch.zeros(1, 1, 1, dtype=torch.bool),
            "rlt_episode_id": torch.zeros(1, 1, 1, dtype=torch.long),
        },
    )


def test_only_takeover_rows_with_the_vla_reference_teach():
    assert human_reference_replaced(_transition(1, human=True, replaced=True))
    assert not human_reference_replaced(_transition(1, human=True, replaced=False))
    assert rlt_sampling_tags(_transition(1, human=True, replaced=False))[
        "rlt_human_teach"
    ]
    assert not rlt_sampling_tags(_transition(1, human=True, replaced=True))[
        "rlt_human_teach"
    ]

    buffer = TrajectoryReplayBuffer(
        seed=0,
        enable_cache=True,
        cache_size=1000,
        sample_window_size=1000,
        auto_save=False,
        sample_strategy="rlt_stratified",
        stratified_ratios={"human": 0.9},
    )
    buffer.add_trajectories(
        [_transition(i) for i in range(50)]
        + [_transition(100 + i, human=True, replaced=True) for i in range(30)]
        + [_transition(200 + i, human=True) for i in range(10)]
    )
    index = buffer.sample(num_chunks=100)["curr_obs"]["z_rl"].reshape(-1)
    teach = (index >= 200).float().mean().item()
    assert teach >= 0.85  # the human share draws only teaching rows
    # A3: the same tagged buffer can serve a uniform critic stream.
    uniform = buffer.sample(num_chunks=200, strategy="uniform")["curr_obs"]["z_rl"]
    assert (uniform.reshape(-1) < 100).float().mean().item() > 0.4


def test_stratified_sampling_requires_a_tagged_buffer():
    buffer = TrajectoryReplayBuffer(
        seed=0, enable_cache=True, cache_size=10, sample_window_size=10, auto_save=False
    )
    buffer.add_trajectories([_transition(0)])
    with pytest.raises(ValueError, match="rlt_stratified"):
        buffer.sample(num_chunks=1, strategy="rlt_stratified")


def _worker(exclude=True):
    worker = RLTACLossMixin()
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {"bc_exclude_replaced_human_reference": exclude},
            "actor": {"model": {"num_action_chunks": 1, "action_dim": 2}},
        }
    )
    return worker


def test_bc_skips_takeover_steps_whose_reference_was_replaced():
    pi = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    actions = torch.tensor([[0.2, 0.2], [0.2, 0.2]])
    ref = torch.tensor([[0.2, 0.2], [0.0, 0.0]])  # row 0: replaced; row 1: kept
    flags = torch.ones(2, 2, dtype=torch.bool)
    proprio = torch.zeros(2, 2)
    loss, metrics = _worker(True)._bc_metrics(pi, actions, ref, flags, proprio)
    assert metrics["human_reference_replaced_ratio"] == pytest.approx(0.5)
    keep_only = _worker(True)._bc_metrics(
        pi[1:], actions[1:], ref[1:], flags[1:], proprio[1:]
    )[0]
    torch.testing.assert_close(loss, keep_only)
    both = _worker(False)._bc_metrics(pi, actions, ref, flags, proprio)[0]
    assert both.item() == pytest.approx(loss.item())  # same error on both rows here
    loss_zero_pi, _ = _worker(True)._bc_metrics(actions, actions, ref, flags, proprio)
    assert loss_zero_pi.item() == pytest.approx(0.0)


# ------------------------------------------------------- A4 critic alignment
def test_critic_alignment_metric_reports_the_gradient_direction():
    worker = RLTACLossMixin()
    worker.cfg = OmegaConf.create(
        {"actor": {"model": {"num_action_chunks": 1, "action_dim": 7}}}
    )
    pi = torch.zeros(3, 7, requires_grad=True)
    direction = torch.tensor([1.0, -1.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    q = (pi * direction).sum(-1, keepdim=True)
    batch = {
        "actions": direction.repeat(3, 1) * 0.01,
        "intervene_flags": torch.tensor([[True] * 7, [True] * 7, [False] * 7]),
    }
    obs = {"ref_chunk": torch.zeros(3, 1, 7)}
    metrics = worker._critic_human_alignment(q, pi, batch, obs)
    assert metrics["critic_human_grad_cos"] == pytest.approx(1.0)
    assert metrics["critic_human_teach_rows"] == 2.0


# ------------------------------------------------------- A3 split sampling hook
def test_actor_batch_hook_defaults_to_the_critic_batch():
    worker = SimpleNamespace(_split_actor_sampling=False)
    from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACReplayMixin

    critic = [{"x": torch.zeros(1)}]
    assert RLTACReplayMixin._actor_micro_batches(worker, critic) is critic


def test_policy_rows_can_be_down_weighted_in_bc():
    def worker(weight):
        w = RLTACLossMixin()
        w.cfg = OmegaConf.create(
            {
                "algorithm": {"bc_policy_row_weight": weight},
                "actor": {"model": {"num_action_chunks": 1, "action_dim": 2}},
            }
        )
        return w

    pi = torch.zeros(2, 2)
    actions = torch.tensor([[0.0, 0.0], [0.4, 0.4]])  # row 1: operator correction
    ref = torch.zeros(2, 2)
    flags = torch.tensor([[False, False], [True, True]])
    proprio = torch.zeros(2, 2)
    full, _ = worker(1.0)._bc_metrics(pi, actions, ref, flags, proprio)
    human_only, _ = worker(0.0)._bc_metrics(pi, actions, ref, flags, proprio)
    tenth, _ = worker(0.1)._bc_metrics(pi, actions, ref, flags, proprio)
    # The operator row's error dominates as the policy-row weight shrinks.
    assert full.item() < tenth.item() < human_only.item()
    torch.testing.assert_close(human_only, 2 * full)

