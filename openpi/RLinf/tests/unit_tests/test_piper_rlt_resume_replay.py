"""A model resume cannot replace the explicit newer replay with its snapshot."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from omegaconf import OmegaConf
from test_piper_terminal_persistence import row

from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


def test_old_model_checkpoint_keeps_newer_explicit_replay(tmp_path):
    active = tmp_path / "active"
    checkpoint = tmp_path / "actor"
    snapshot = checkpoint / "sac_components/replay_buffer/rank_0"
    buffer = TrajectoryReplayBuffer(
        auto_save=True, auto_save_path=str(active), cache_size=2
    )
    buffer.add_trajectories([row(1)])
    buffer.save_checkpoint(str(snapshot))
    buffer.add_trajectories([row(2, terminal=True, reward=1)])
    buffer.save_checkpoint(str(active))
    target = checkpoint / "sac_components/target_model"
    target.mkdir()
    torch.save({}, target / "checkpoint_rank_0.pt")
    worker = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "algorithm": {
                    "loss_type": "rlt_ac",
                    "replay_buffer": {"load_path": str(active)},
                },
                "actor": {"fsdp_config": {"use_orig_params": True}},
            }
        ),
        _strategy=Mock(),
        model=Mock(),
        optimizer=Mock(),
        qf_optimizer=Mock(),
        lr_scheduler=Mock(),
        qf_lr_scheduler=Mock(),
        alpha_optimizer=None,
        target_model=Mock(),
        replay_buffer=buffer,
        demo_buffer=None,
        _rank=0,
        _world_size=1,
    )
    EmbodiedSACFSDPPolicy.load_checkpoint(worker, str(checkpoint))
    assert buffer.size == 2
    assert json.loads((active / "metadata.json").read_text())["size"] == 2
    assert json.loads((snapshot / "metadata.json").read_text())["size"] == 1


def test_distributed_load_preserves_quarantined_id_counter(tmp_path):
    buffer = TrajectoryReplayBuffer(
        auto_save=True, auto_save_path=str(tmp_path), cache_size=2
    )
    buffer.add_trajectories([row(1)])
    buffer._trajectory_counter = 100
    buffer.save_checkpoint(str(tmp_path))
    buffer.load_checkpoint(
        str(tmp_path), is_distributed=True, local_rank=0, world_size=1
    )
    assert buffer._trajectory_counter == 100
