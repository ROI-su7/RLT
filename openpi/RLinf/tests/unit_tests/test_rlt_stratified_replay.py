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

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer


def _transition(index, *, warmup=False, online=False, human=False, episode_id=-1):
    action_dim = 2 * 14
    return Trajectory(
        max_episode_length=1,
        model_weights_id="test",
        actions=torch.zeros(1, 1, action_dim),
        rewards=torch.zeros(1, 1, 2),
        dones=torch.zeros(1, 1, 2, dtype=torch.bool),
        intervene_flags=torch.full((1, 1, action_dim), human),
        curr_obs={"z_rl": torch.full((1, 1, 1), float(index))},
        next_obs={"z_rl": torch.full((1, 1, 1), float(index))},
        forward_inputs={
            "reference_warmup": torch.full((1, 1, 1), warmup),
            "actor_switch": torch.full((1, 1, 1), online),
            "rlt_episode_id": torch.full((1, 1, 1), episode_id, dtype=torch.long),
        },
    )


def _buffer(strategy: str = "rlt_stratified") -> TrajectoryReplayBuffer:
    return TrajectoryReplayBuffer(
        seed=0,
        enable_cache=True,
        cache_size=1000,
        sample_window_size=1000,
        auto_save=False,
        sample_strategy=strategy,
    )


def test_batches_follow_the_stratified_shares() -> None:
    buffer = _buffer()
    warmup = [_transition(i, warmup=True) for i in range(100)]
    # 25 online episodes of 4 transitions; only the last 20 count as recent.
    online = [_transition(100 + i, online=True, episode_id=i // 4) for i in range(100)]
    human = [_transition(200 + i, human=True) for i in range(20)]
    buffer.add_trajectories(warmup + online + human)

    # The buffer caps a batch at the number of stored samples (220 here).
    batch = buffer.sample(num_chunks=200)
    index = batch["curr_obs"]["z_rl"].reshape(-1)

    fraction = lambda mask: mask.float().mean().item()  # noqa: E731
    assert index.numel() == 200
    assert fraction(index < 100) >= 0.3
    assert fraction((index >= 120) & (index < 200)) >= 0.4
    assert fraction(index >= 200) >= 0.2
    # Episodes 0..4 are reachable only through the 10% uniform remainder.
    assert fraction((index >= 100) & (index < 120)) < 0.05


def test_empty_pools_fall_back_to_uniform() -> None:
    buffer = _buffer()
    buffer.add_trajectories([_transition(i, warmup=True) for i in range(100)])

    batch = buffer.sample(num_chunks=64)

    # No online or human transitions: their 60% share is drawn uniformly.
    assert batch["curr_obs"]["z_rl"].shape[0] == 64


def test_uniform_strategy_does_not_tag_the_index() -> None:
    buffer = _buffer("uniform")
    buffer.add_trajectories([_transition(0, warmup=True)])

    assert "rlt_warmup" not in buffer._trajectory_index[0]


def test_loading_a_uniform_buffer_backfills_the_strata(tmp_path) -> None:
    saved = TrajectoryReplayBuffer(
        seed=0,
        enable_cache=True,
        cache_size=1000,
        sample_window_size=1000,
        auto_save=True,
        auto_save_path=str(tmp_path / "uniform"),
        sample_strategy="uniform",
    )
    saved.add_trajectories(
        [_transition(i, warmup=True) for i in range(80)]
        + [_transition(100 + i, human=True) for i in range(20)]
    )
    assert "rlt_human" not in saved._trajectory_index[0]

    loaded = TrajectoryReplayBuffer(
        seed=0,
        enable_cache=True,
        cache_size=1000,
        sample_window_size=1000,
        auto_save=True,
        auto_save_path=str(tmp_path / "stratified"),
        sample_strategy="rlt_stratified",
    )
    loaded.load_checkpoint(str(tmp_path / "uniform"))

    assert sum(info["rlt_human"] for info in loaded._trajectory_index.values()) == 20
    index = loaded.sample(num_chunks=100)["curr_obs"]["z_rl"].reshape(-1)
    # Human rows are 20% of the buffer and get a 20% share plus uniform draws.
    assert (index >= 100).float().mean().item() >= 0.2
