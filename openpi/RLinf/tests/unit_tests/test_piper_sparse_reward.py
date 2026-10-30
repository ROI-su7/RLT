from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest

from rlinf.envs.realworld.piper.leader_takeover import TakeoverState
from rlinf.envs.realworld.piper.tasks import PiperRLTControlWrapper


class _FakeEnv(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, (14,), dtype=np.float32)
    observation_space = gym.spaces.Box(-1.0, 1.0, (14,), dtype=np.float32)

    def __init__(self, dense_reward: float = 0.5):
        self.dense_reward = dense_reward
        self.config = Mock(max_num_steps=100)
        self._logger = Mock()
        self.hold_position = Mock()

    def step(self, action):
        del action
        return np.zeros(14, dtype=np.float32), self.dense_reward, False, False, {}

    def step_external_control(self):
        return np.zeros(14, dtype=np.float32), self.dense_reward, False, False, {}


def _wrapper(*keys: str) -> PiperRLTControlWrapper:
    wrapper = PiperRLTControlWrapper.__new__(PiperRLTControlWrapper)
    wrapper.env = _FakeEnv()
    wrapper.listener = Mock()
    wrapper.listener.pop_pressed_keys.return_value = list(keys)
    wrapper._actor_active = True
    wrapper._last_press = {}
    wrapper._episode_step = 5
    wrapper._episode_reward = 0.0
    wrapper._insertions_scored = 0
    wrapper._drops_scored = 0
    wrapper._grasp_scored = False
    wrapper._actor_score_ready_step = 0
    wrapper._episode_ended = False
    wrapper._last_obs = None
    wrapper._mode2 = None
    wrapper._last_mode2_warning_at = -np.inf
    wrapper._last_mode2_state = None
    return wrapper


@pytest.mark.parametrize("key", ["g", "i", "f"])
def test_milestone_keys_are_diagnostic_only(key: str):
    wrapper = _wrapper(key)
    if key == "f":
        wrapper._grasp_scored = True

    _, reward, terminated, _, _ = wrapper.step(np.zeros(14, dtype=np.float32))

    assert reward == 0.0
    assert terminated is False


@pytest.mark.parametrize("key", ["x", "q"])
def test_failure_keys_are_zero_reward_terminal(key: str):
    wrapper = _wrapper(key)

    _, reward, terminated, _, _ = wrapper.step(np.zeros(14, dtype=np.float32))

    assert reward == 0.0
    assert terminated is True


def test_mode2_q_disables_bridge_before_holding() -> None:
    wrapper = _wrapper("q")
    wrapper._mode2 = Mock(state=TakeoverState.TELEOP)
    wrapper._mode2.action_for_step.return_value = (
        np.zeros(14, dtype=np.float32),
        True,
    )

    _, _, terminated, _, _ = wrapper.step(np.zeros(14, dtype=np.float32))

    assert terminated
    wrapper._mode2.emergency_stop.assert_called_once_with()
    wrapper.unwrapped.hold_position.assert_called_once_with()


def test_success_key_is_binary_positive_terminal():
    wrapper = _wrapper("c")

    _, reward, terminated, _, info = wrapper.step(np.zeros(14, dtype=np.float32))

    assert reward == 1.0
    assert terminated is True
    assert info["manual_episode_reward"] == 1.0


def test_mode2_hold_key_revokes_policy_and_holds_followers():
    wrapper = _wrapper("h")
    wrapper._actor_active = True
    wrapper._mode2 = Mock(state=TakeoverState.POLICY)
    wrapper._mode2.action_for_step.return_value = (
        np.zeros(14, dtype=np.float32),
        False,
    )

    _, _, terminated, _, info = wrapper.step(np.ones(14, dtype=np.float32))

    assert not terminated
    wrapper._mode2.request_hold.assert_called_once_with()
    wrapper.unwrapped.hold_position.assert_called_once_with()
    assert info["rlt_switch_flags"] is False
    assert info["rlt_policy_switch_event"] == "mode2_follower_hold"


def test_mode2_takeover_action_replaces_policy_action_in_replay():
    wrapper = _wrapper()
    human_action = np.linspace(-0.1, 0.1, 14, dtype=np.float32)
    wrapper._mode2 = Mock(state=TakeoverState.TELEOP)
    wrapper._mode2.action_for_step.return_value = (human_action, True)

    _, _, terminated, _, info = wrapper.step(np.ones(14, dtype=np.float32))

    assert not terminated
    np.testing.assert_array_equal(info["intervene_action"], human_action)
    np.testing.assert_array_equal(
        info["intervene_flag"],
        np.ones(1, dtype=np.float32),
    )
    assert info["rlt_switch_flags"] is True


def test_mode2_terminal_outcome_enters_terminal_hold():
    wrapper = _wrapper("x")
    wrapper._mode2 = Mock(state=TakeoverState.TELEOP)
    wrapper._mode2.action_for_step.return_value = (
        np.zeros(14, dtype=np.float32),
        True,
    )
    wrapper._mode2.finish_episode.side_effect = lambda: setattr(
        wrapper._mode2, "state", TakeoverState.TELEOP_RETURN
    )

    _, _, terminated, _, _ = wrapper.step(np.zeros(14, dtype=np.float32))

    assert terminated
    wrapper._mode2.finish_episode.assert_called_once_with()
    wrapper.unwrapped.hold_position.assert_not_called()
