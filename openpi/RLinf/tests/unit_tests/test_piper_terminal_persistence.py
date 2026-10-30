"""C/X survives HOLD interruptions, chunk boundaries and disk reloads."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACReplayMixin
from rlinf.workers.env.env_worker import EnvWorker


def row(episode, terminal=False, reward=0.0):
    r = torch.zeros(1, 1, 3)
    r[0, 0, -1] = reward
    d = torch.zeros(1, 1, 3, dtype=torch.bool)
    d[0, 0, -1] = terminal
    return Trajectory(
        max_episode_length=1,
        model_weights_id="test",
        actions=torch.zeros(1, 1, 3, 14),
        rewards=r,
        dones=d,
        terminations=d.clone(),
        truncations=torch.zeros_like(d),
        forward_inputs={
            "record_transition": torch.ones(1, 1, 1, dtype=torch.bool),
            "rlt_episode_id": torch.tensor([[[episode]]]),
            "action_valid_mask": torch.ones(1, 1, 3, dtype=torch.bool),
        },
    )


def learner(buffer):
    worker = RLTACReplayMixin()
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {"replay_buffer": {"save_transitions": True}},
            "env": {"train": {"auto_reset": False, "env_type": "realworld"}},
        }
    )
    worker.replay_buffer = buffer
    worker.demo_buffer = None
    worker.log_info = Mock()
    return worker


def label(episode, reward):
    event = row(episode)
    event.forward_inputs["record_transition"].zero_()
    event.forward_inputs["action_valid_mask"].zero_()
    event.forward_inputs["rlt_keyboard_terminal"] = torch.ones(
        1, 1, 1, dtype=torch.bool
    )
    event.rewards = torch.zeros(2, 1, 3)
    event.rewards[1, 0, 0] = reward
    event.dones = torch.zeros(2, 1, 3, dtype=torch.bool)
    event.dones[1, 0] = True
    event.terminations = event.dones.clone()
    event.truncations = torch.zeros_like(event.dones)
    return event


@pytest.mark.parametrize("rewards", [(1.0, 1.0), (1.0, 0.0)])
def test_two_episodes_terminal_between_chunks_survives_reload(tmp_path, rewards):
    buffer = TrajectoryReplayBuffer(
        auto_save=True, auto_save_path=str(tmp_path), cache_size=2
    )
    worker = learner(buffer)
    with patch(
        "rlinf.workers.actor.fsdp_rlt_ac_policy_worker.collect_trajectory_replay_metrics",
        return_value={},
    ):
        for episode, reward in zip((101, 102), rewards):
            buffer.add_trajectories([row(episode)])
            added, completed = worker._ingest_rollout_trajectories(
                [label(episode, reward)]
            )
            assert added == 0 and completed == 1
            # A held/padded terminal event cannot award the same episode twice.
            assert worker._ingest_rollout_trajectories([label(episode, reward)]) == (
                0,
                0,
            )
    buffer.close()
    restored = TrajectoryReplayBuffer(auto_save=True, auto_save_path=str(tmp_path))
    restored.load_checkpoint(str(tmp_path))
    saved = [
        restored._load_trajectory(tid, "test") for tid in restored._trajectory_id_list
    ]
    assert len(saved) == 2
    assert [float(t.rewards.sum()) for t in saved] == list(rewards)
    assert all(bool(t.dones.any()) for t in saved)
    assert all(int(t.forward_inputs["action_valid_mask"].sum()) == 3 for t in saved)
    assert not restored.finalize_rlt_episode(999, 1.0)  # Never label another episode.
    restored.close()


def test_c_interrupting_an_action_tick_keeps_only_real_prefix():
    inputs = {"record_transition": torch.ones(1, 1, dtype=torch.bool)}
    result = SimpleNamespace(forward_inputs=[inputs], actions=[torch.ones(1, 42)])
    outcome = SimpleNamespace(
        rewards=torch.tensor([[0.0, 1.0, 0.0]]),
        dones=torch.tensor([[False, True, True]]),
        terminations=torch.tensor([[False, True, True]]),
        truncations=torch.zeros(1, 3, dtype=torch.bool),
        intervene_flags=None,
    )
    worker = SimpleNamespace(
        train_num_envs_per_stage=1,
        model_cfg=SimpleNamespace(num_action_chunks=3),
        _rlt_last_chunk_info=[
            {
                "execution_valid": torch.tensor([[True, False, False]]),
                "executed_actions": torch.zeros(1, 3, 14),
                "step_dones": outcome.dones,
                "keyboard_terminal": torch.tensor([[False, True, False]]),
            }
        ],
        rollout_results=[result],
        _rlt_bridge_tail=[None],
        _rlt_expected_rtc_offset=[0],
        _rlt_actor_plan_offset=0,
        _rlt_stride_enabled=False,
        _rlt_episode_ids=[torch.tensor([101])],
        _new_rlt_episode_ids=lambda count: torch.tensor([102]),
    )
    EnvWorker._rlt_end_chunk(worker, 0, outcome, {})
    assert inputs["record_transition"].item()
    assert inputs["action_valid_mask"].tolist() == [[True, False, False]]
    assert outcome.rewards.tolist() == [[1.0, 0.0, 0.0]]
    assert outcome.dones.tolist() == [[True, False, False]]
