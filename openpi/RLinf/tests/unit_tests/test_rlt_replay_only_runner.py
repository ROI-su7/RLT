"""Replay-only Stage-2 runner contracts without Ray, CUDA, ROS, or CAN."""

from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from rlinf.runners.rlt_replay_only_runner import RLTReplayOnlyRunner


class _Handle:
    def wait(self):
        return None


class _Actor:
    def __init__(self):
        self.steps = []
        self.saved = []

    def set_global_step(self, step):
        self.steps.append(step)

    def run_training(self):
        return _Handle()

    def save_checkpoint(self, path, step):
        self.saved.append((path, step))
        return _Handle()


def _cfg(tmp_path, *, max_steps=5, save_interval=2, resume_dir=None):
    return OmegaConf.create(
        {
            "runner": {
                "max_steps": max_steps,
                "save_interval": save_interval,
                "resume_dir": resume_dir,
                "logger": {
                    "log_path": str(tmp_path),
                    "experiment_name": "piper_wire_rlt_stage2",
                },
            }
        }
    )


def test_replay_only_runner_stops_and_saves_at_exact_max_step(tmp_path) -> None:
    actor = _Actor()
    runner = RLTReplayOnlyRunner(_cfg(tmp_path), actor)

    runner.run()

    assert actor.steps == [0, 1, 2, 3, 4]
    assert [step for _, step in actor.saved] == [2, 4, 5]
    assert actor.saved[-1][0].endswith("checkpoints/global_step_5/actor")


def test_replay_only_runner_rejects_resume_and_invalid_steps(tmp_path) -> None:
    with pytest.raises(ValueError, match="does not accept runner.resume_dir"):
        RLTReplayOnlyRunner(_cfg(tmp_path, resume_dir="old"), Mock())
    with pytest.raises(ValueError, match="max_steps > 0"):
        RLTReplayOnlyRunner(_cfg(tmp_path, max_steps=0), Mock())
