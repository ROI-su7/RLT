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

"""Dual-SO101 RLT environment and Evo-RLT-style leader takeover on simulated arms."""

import numpy as np
import pytest

from rlinf.envs.realworld.so101.hardware import SimSo101Arms
from rlinf.envs.realworld.so101.rlt_wrapper import FileKeyListener, So101RLTWrapper
from rlinf.envs.realworld.so101.so101_env import So101DualEnv
from rlinf.envs.realworld.so101.takeover import InterventionState, So101Takeover
from rlinf.utils.arm_layout import arm_width, gripper_slice, is_arm_joint


class QueueListener:
    """Keys "pressed" after reset() drains stale input (the first pop returns nothing)."""

    def __init__(self, keys=()):
        self.keys = list(keys)
        self._drained = False

    def pop_pressed_keys(self):
        if not self._drained:
            self._drained = True
            return []
        keys, self.keys = self.keys, []
        return keys


def make_env(keys=(), **cfg):
    arms = SimSo101Arms(max_relative_target=None, lag=1.0)
    cfg = {"hardware": "sim", "step_frequency": 1000.0, "task_description": "pick", "takeover_blend_s": 0.0, **cfg}
    listener = QueueListener(keys)
    env = So101RLTWrapper(So101DualEnv(cfg, arms=arms), listener=listener, start_keyboard_thread=False)
    return env, arms, listener


def press(env, *keys):
    for key in keys:
        env._last_press.clear()
        env.handle_key(key)


def test_arm_layout_keeps_piper_and_maps_so101():
    assert arm_width(14) == 7 and arm_width(7) == 7 and arm_width(8) == 7 and arm_width(2) == 7
    assert arm_width(12) == 6
    assert np.arange(12)[gripper_slice(12)].tolist() == [5, 11]
    assert np.arange(14)[gripper_slice(14)].tolist() == [6, 13]
    assert [i for i in range(12) if not is_arm_joint(i, 12)] == [5, 11]


def test_observation_layout_matches_piper():
    env, _, _ = make_env(keys=["i"])
    obs, _ = env.reset()
    assert obs["state"]["joint_position"].shape == (12,)
    assert set(obs["frames"]) == {"cam_high", "cam_left_wrist", "cam_right_wrist"}
    assert obs["frames"]["cam_high"].shape == (224, 224, 3)
    assert obs["frames"]["cam_high"].dtype == np.uint8


def test_policy_steps_mirror_the_stiff_leaders():
    env, arms, _ = make_env(keys=["i"])
    env.reset()
    press(env, "b")
    env.begin_action_chunk(10)
    target = np.arange(12, dtype=np.float32)
    _, reward, terminated, truncated, info = env.step(target)
    assert not (terminated or truncated) and reward == 0.0
    assert info["rlt_execution_valid"] and info["rlt_switch_flags"]
    assert info["intervene_action"] is None
    np.testing.assert_allclose(info["rlt_executed_action"], target)
    assert arms.leader_torque
    np.testing.assert_allclose(arms.read_leader(), target)  # Evo-RLT policy sync


def test_takeover_follows_limp_leaders_and_reports_human_actions():
    env, arms, _ = make_env(keys=["i"])
    env.reset()
    press(env, "b")
    env.begin_action_chunk(10)
    env.step(np.zeros(12, np.float32))
    press(env, "t")
    assert env.takeover.state == InterventionState.ACTIVE and not arms.leader_torque
    hand = np.full(12, 7.0, np.float32)
    arms.move_leader(hand)
    _, _, _, _, info = env.step(np.ones(12, np.float32))  # the policy action is ignored
    np.testing.assert_allclose(info["intervene_action"], hand)
    np.testing.assert_allclose(arms.follower_goal, hand)
    assert info["rlt_execution_valid"]

    press(env, "t")  # Evo-RLT toggle: release
    assert env.takeover.state == InterventionState.RELEASE and arms.leader_torque
    _, _, _, _, info = env.step(np.ones(12, np.float32))
    assert not info["rlt_execution_valid"]  # rest of the interrupted chunk dropped
    assert info["intervene_action"] is None
    np.testing.assert_allclose(arms.follower_goal, hand)  # held, not the stale policy action
    env.begin_action_chunk(10)
    _, _, _, _, info = env.step(np.ones(12, np.float32))
    assert info["rlt_execution_valid"]
    assert env.takeover.state == InterventionState.POLICY


def test_takeover_blend_starts_from_the_last_policy_command():
    arms = SimSo101Arms(max_relative_target=None)
    now = [0.0]
    takeover = So101Takeover(arms, blend_s=1.0, clock=lambda: now[0])
    takeover.policy_command(np.zeros(12, np.float32))
    takeover.start(np.zeros(12, np.float32))
    arms.move_leader(np.full(12, 10.0, np.float32))
    now[0] = 0.5
    np.testing.assert_allclose(takeover.human_command(), np.full(12, 5.0))
    now[0] = 1.5
    np.testing.assert_allclose(takeover.human_command(), np.full(12, 10.0))


def test_t_requires_collection_gate_and_success_terminates():
    env, arms, _ = make_env(keys=["i"])
    env.reset()
    press(env, "t")
    assert not env.takeover.active  # no B yet
    press(env, "b", "c")
    obs, reward, terminated, truncated, info = env.step(np.zeros(12, np.float32))
    assert reward == 1.0 and terminated and not truncated
    assert info["rlt_keyboard_terminal"] and info["manual_episode_reward"] == 1.0
    _, reward, _, _, info = env.step(np.zeros(12, np.float32))
    assert reward == 0.0 and not info["rlt_keyboard_terminal"]  # padding after the terminal


def test_failure_during_takeover_releases_the_leaders():
    env, arms, _ = make_env(keys=["i"])
    env.reset()
    press(env, "b", "t", "x")
    assert arms.leader_torque and not env.takeover.active
    _, reward, terminated, _, info = env.step(np.zeros(12, np.float32))
    assert reward == 0.0 and terminated and info["rlt_keyboard_terminal"]


def test_space_holds_until_i():
    env, arms, listener = make_env(keys=["i"])
    env.reset()
    env.begin_action_chunk(10)
    env.step(np.full(12, 3.0, np.float32))
    press(env, "space")
    env._last_press.clear()
    listener.keys = ["i"]  # consumed by the HOLD loop inside step
    _, _, _, _, info = env.step(np.full(12, 9.0, np.float32))
    assert not info["rlt_execution_valid"]
    np.testing.assert_allclose(arms.follower_goal, np.full(12, 3.0))


def test_home_moves_followers_back():
    env, arms, _ = make_env(keys=["i"], home_speed_deg_s=1000.0)
    env.reset()
    home = env.unwrapped.home_position.copy()
    env.begin_action_chunk(10)
    env.step(home + 20.0)
    press(env, "space", "h")
    env._go_home_if_requested()
    assert np.max(np.abs(arms.read_follower() - home)) <= 1.0


def test_file_key_listener(tmp_path):
    path = tmp_path / "keys"
    listener = FileKeyListener(str(path))
    path.write_text("i\nb\nspace\nt\n")
    assert listener.pop_pressed_keys() == ["i", "b", "space", "t"]
    assert listener.pop_pressed_keys() == []


def test_wrong_action_dim_is_rejected():
    env, _, _ = make_env(keys=["i"])
    env.reset()
    env.begin_action_chunk(10)
    with pytest.raises(ValueError):
        env.step(np.zeros(14, np.float32))
