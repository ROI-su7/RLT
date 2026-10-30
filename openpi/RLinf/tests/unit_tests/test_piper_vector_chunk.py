"""Exercise four-CAN metadata through Gym vectorization and chunk batching."""

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.envs.realworld.venv import NoAutoResetSyncVectorEnv


class MetadataEnv(gym.Env):
    action_space = gym.spaces.Box(-1, 1, (14,), dtype=np.float32)
    observation_space = gym.spaces.Box(-1, 1, (14,), dtype=np.float32)

    def __init__(self, human: bool):
        self.human = human
        self.index = 0

    def step(self, action):
        self.index += 1
        executed = np.full(14, self.index / 10, dtype=np.float32)
        diagnostics = (
            {
                "piper_accepted_target": executed / 2,
                "piper_execution_filter_params": np.array(
                    [0.4, 0.4, 1 / 15], dtype=np.float32
                ),
            }
            if not self.human and self.index != 2
            else {}
        )
        return (
            np.zeros(14, np.float32),
            0.0,
            False,
            False,
            {
                "rlt_switch_flags": True,
                "rlt_execution_valid": self.index != 2,
                "rlt_executed_action": executed,
                "intervene_action": executed if self.human else None,
                **diagnostics,
            },
        )


class ObjectInfoVectorEnv(NoAutoResetSyncVectorEnv):
    """Emulate the deployed Gym version's ndarray-valued object infos."""

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = super().step(actions)
        for field in (
            "rlt_executed_action",
            "piper_accepted_target",
            "piper_execution_filter_params",
        ):
            if field not in infos:
                continue
            values = np.empty(self.num_envs, dtype=object)
            for index, action in enumerate(infos[field]):
                values[index] = action
            infos[field] = values
        return obs, rewards, terminated, truncated, infos


@pytest.mark.parametrize("vector_cls", [NoAutoResetSyncVectorEnv, ObjectInfoVectorEnv])
@pytest.mark.parametrize("human", [False, True])
def test_mux_metadata_becomes_dense_chunk_tensors(vector_cls, human):
    env = RealWorldEnv.__new__(RealWorldEnv)
    env.num_envs = 1
    env.env = vector_cls([lambda: MetadataEnv(human)])
    env._elapsed_steps = np.zeros(1, dtype=np.int32)
    env.cfg = SimpleNamespace(max_episode_steps=100)
    env.manual_episode_control_only = False
    env.ignore_terminations = False
    env.auto_reset = False
    env._wrap_obs = lambda obs: obs
    env._calc_step_reward = lambda reward: reward
    env._record_metrics = lambda *args: args[-1]
    try:
        _, _, _, _, infos = env.chunk_step(torch.zeros(1, 3, 14))
        final = infos[-1]
        assert final["rlt_executed_action"].shape == (1, 3, 14)
        assert final["rlt_executed_action"].dtype == torch.float32
        assert final["rlt_execution_valid"].tolist() == [[True, False, True]]
        torch.testing.assert_close(
            final["rlt_executed_action"][0, :, 0], torch.tensor([0.1, 0.2, 0.3])
        )
        assert final["intervene_flag"].tolist() == [[human] * 3]
        assert final["piper_accepted_target"].shape == (1, 3, 14)
        assert final["piper_accepted_target"].dtype == torch.float32
        assert final["piper_execution_filter_params"].shape == (1, 3, 3)
        if human:
            assert final["piper_accepted_target"].isnan().all()
        else:
            torch.testing.assert_close(
                final["piper_accepted_target"][0, [0, 2], 0], torch.tensor([0.05, 0.15])
            )
            assert final["piper_accepted_target"][0, 1].isnan().all()
        if human:
            torch.testing.assert_close(
                final["intervene_action"].reshape(1, 3, 14),
                final["rlt_executed_action"],
            )
    finally:
        env.env.close()


def test_diagnostic_object_infos_handle_mixed_policy_human_batch():
    env = RealWorldEnv.__new__(RealWorldEnv)
    env.num_envs = 2
    env.env = ObjectInfoVectorEnv(
        [lambda: MetadataEnv(False), lambda: MetadataEnv(True)]
    )
    env._elapsed_steps = np.zeros(2, dtype=np.int32)
    env.cfg = SimpleNamespace(max_episode_steps=100)
    env.manual_episode_control_only = env.ignore_terminations = env.auto_reset = False
    env._wrap_obs = lambda obs: obs
    env._calc_step_reward = lambda reward: reward
    env._record_metrics = lambda *args: args[-1]
    try:
        _, _, _, _, infos = env.chunk_step(torch.zeros(2, 3, 14))
        final = infos[-1]
        assert final["piper_accepted_target"].shape == (2, 3, 14)
        torch.testing.assert_close(
            final["piper_accepted_target"][0, [0, 2], 0], torch.tensor([0.05, 0.15])
        )
        assert final["piper_accepted_target"][1].isnan().all()
        assert final["piper_execution_filter_params"][1].isnan().all()
    finally:
        env.env.close()
