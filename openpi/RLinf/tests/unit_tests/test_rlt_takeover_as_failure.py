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

"""An operator takeover ends the policy's segment with reward 0 (RLIF)."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer

STEPS = 3
DIM = 14


def row(tag, episode, human_steps=(), reward=0.0, terminal=False, valid=None):
    """One stored transition; ``tag`` identifies it in a sampled batch."""
    rewards = torch.zeros(1, 1, STEPS)
    rewards[0, 0, -1] = reward
    dones = torch.zeros(1, 1, STEPS, dtype=torch.bool)
    dones[0, 0, -1] = terminal
    flags = torch.zeros(1, 1, STEPS, DIM, dtype=torch.bool)
    for step in human_steps:
        flags[0, 0, step] = True
    mask = torch.ones(1, 1, STEPS, dtype=torch.bool)
    if valid is not None:
        mask = torch.as_tensor(valid, dtype=torch.bool).reshape(1, 1, STEPS)
    return Trajectory(
        max_episode_length=1,
        model_weights_id="test",
        actions=torch.full((1, 1, STEPS * DIM), float(tag)),
        intervene_flags=flags.reshape(1, 1, STEPS * DIM),
        rewards=rewards,
        dones=dones,
        terminations=dones.clone(),
        truncations=torch.zeros_like(dones),
        forward_inputs={
            "rlt_episode_id": torch.tensor([[[episode]]]),
            "action_valid_mask": mask,
        },
    )


def make_buffer(enabled=True, **kwargs):
    return TrajectoryReplayBuffer(
        auto_save=False,
        enable_cache=True,
        cache_size=50,
        sample_window_size=50,
        sample_strategy="rlt_stratified",
        takeover_as_failure=enabled,
        **kwargs,
    )


def sampled_rows(buffer, draws=60):
    """Tag -> (rewards, dones, terminations) of every sampled transition."""
    seen = {}
    for _ in range(draws):
        batch = buffer.sample(num_chunks=50, strategy="uniform")
        tags = batch["actions"].reshape(batch["actions"].shape[0], -1)[:, 0]
        for index, tag in enumerate(tags.tolist()):
            seen[int(tag)] = (
                batch["rewards"][index].reshape(-1).tolist(),
                batch["dones"][index].reshape(-1).tolist(),
                batch["terminations"][index].reshape(-1).tolist(),
            )
    return seen


def test_last_machine_row_before_takeover_becomes_terminal_zero():
    buffer = make_buffer()
    buffer.add_trajectories(
        [
            row(0, episode=1),
            row(1, episode=1),
            row(2, episode=1, human_steps=(0, 1, 2)),
            row(3, episode=1, human_steps=(0, 1, 2), reward=1.0, terminal=True),
        ]
    )
    seen = sampled_rows(buffer)
    assert seen[1] == ([0.0] * 3, [False, False, True], [False, False, True])
    # The policy's earlier rows and the operator's rescue are unchanged.
    assert seen[0] == ([0.0] * 3, [False] * 3, [False] * 3)
    assert seen[2] == ([0.0] * 3, [False] * 3, [False] * 3)
    assert seen[3] == ([0.0, 0.0, 1.0], [False, False, True], [False, False, True])


def test_mixed_row_counts_as_operator_row():
    buffer = make_buffer()
    buffer.add_trajectories(
        [
            row(0, episode=1),
            row(1, episode=1, human_steps=(2,)),
            row(2, episode=1, human_steps=(0, 1, 2)),
        ]
    )
    seen = sampled_rows(buffer)
    assert seen[0][1] == [False, False, True]
    assert seen[1][1] == [False] * 3


def test_terminal_lands_on_last_executed_step_of_partial_row():
    buffer = make_buffer()
    buffer.add_trajectories(
        [
            row(0, episode=1, valid=[True, True, False]),
            row(1, episode=1, human_steps=(0, 1, 2)),
        ]
    )
    assert sampled_rows(buffer)[0][1] == [False, True, False]


def test_each_takeover_of_an_episode_ends_one_policy_segment():
    buffer = make_buffer()
    buffer.add_trajectories(
        [
            row(0, episode=1, human_steps=(0, 1, 2)),  # takeover from the start
            row(1, episode=1),
            row(2, episode=1, human_steps=(0, 1, 2)),
            row(3, episode=2),
            row(4, episode=2, reward=1.0, terminal=True),  # autonomous success
        ]
    )
    seen = sampled_rows(buffer)
    assert seen[1][1] == [False, False, True]
    assert seen[0][1] == [False] * 3
    assert seen[3][1] == [False] * 3
    assert seen[4] == ([0.0, 0.0, 1.0], [False, False, True], [False, False, True])


def test_machine_windows_stored_after_the_first_operator_row_also_end():
    # Insertion order at a takeover: the operator's chunk row is stored before
    # the stride windows of the last machine chunk (released one chunk late).
    buffer = make_buffer()
    human = (0, 1, 2)
    buffer.add_trajectories(
        [
            row(0, episode=1),
            row(1, episode=1),
            row(2, episode=1, human_steps=human),
            row(3, episode=1),
            row(4, episode=1),
            row(5, episode=1, human_steps=human),
            row(6, episode=1, human_steps=human),
        ]
    )
    assert buffer._takeover_terminal_ids().tolist() == [1, 3, 4]


def test_policy_resumed_after_a_takeover_ends_only_at_the_next_one():
    buffer = make_buffer()
    human = (0, 1, 2)
    resumed = [row(tag, episode=1) for tag in range(3, 9)]  # longer than the lag
    buffer.add_trajectories(
        [
            row(0, episode=1),
            row(1, episode=1, human_steps=human),
            row(2, episode=1, human_steps=human),
            *resumed,
            row(9, episode=1, human_steps=human),
        ]
    )
    assert buffer._takeover_terminal_ids().tolist() == [0, 8]


def test_takeover_recorded_after_the_row_is_already_cached():
    buffer = make_buffer()
    buffer.add_trajectories([row(0, episode=1), row(1, episode=1)])
    assert sampled_rows(buffer)[1][1] == [False] * 3
    buffer.add_trajectories([row(2, episode=1, human_steps=(0, 1, 2))])
    assert sampled_rows(buffer)[1][1] == [False, False, True]


def test_stored_transition_keeps_its_recorded_flags():
    buffer = make_buffer()
    buffer.add_trajectories(
        [row(0, episode=1), row(1, episode=1, human_steps=(0, 1, 2))]
    )
    sampled_rows(buffer)
    stored = buffer._flat_trajectory_cache.get(0)
    assert not bool(stored["dones"].any())
    assert not bool(stored["terminations"].any())


def test_disabled_keeps_bootstrapping_through_the_takeover():
    buffer = make_buffer(enabled=False)
    buffer.add_trajectories(
        [row(0, episode=1), row(1, episode=1, human_steps=(0, 1, 2))]
    )
    assert sampled_rows(buffer)[0][1] == [False] * 3


def test_reloaded_replay_is_relabeled(tmp_path):
    source = make_buffer()
    source.add_trajectories(
        [row(0, episode=1), row(1, episode=1, human_steps=(0, 1, 2))]
    )
    source.save_checkpoint(str(tmp_path))
    restored = make_buffer()
    restored.load_checkpoint(str(tmp_path))
    assert sampled_rows(restored)[0][1] == [False, False, True]


def test_needs_the_operator_tags_of_the_stratified_index():
    with pytest.raises(ValueError, match="rlt_stratified"):
        TrajectoryReplayBuffer(
            auto_save=False, sample_strategy="uniform", takeover_as_failure=True
        )


def test_piper_stage2_config_enables_it():
    cfg = OmegaConf.load(
        Path(__file__).parents[2]
        / "examples/embodiment/config/realworld_piper_rlt_stage2_ac_mlp.yaml"
    )
    assert cfg.algorithm.replay_buffer.takeover_as_failure is True
    assert cfg.algorithm.replay_buffer.sample_strategy == "rlt_stratified"
