"""Follower-only episode/replay semantics without robot hardware."""

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import numpy as np
import pytest
from test_piper_four_can_rlt import FakeEnv, FakeMux

from rlinf.envs.realworld.piper.follower_rlt_wrapper import PiperFollowerRLTWrapper


def start_followers():
    env, mux = FakeEnv(), FakeMux()
    wrapper = PiperFollowerRLTWrapper(
        env, mux, listener=Mock(), start_keyboard_thread=False
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        reset = executor.submit(wrapper.reset)
        assert env.reset_called.wait(1)
        wrapper.handle_key("i")
        reset.result(timeout=1)
    wrapper.begin_action_chunk(10)
    return wrapper, env, mux


@pytest.mark.parametrize(("outcome", "reward"), [("c", 1.0), ("x", 0.0)])
def test_follower_collection_reward_and_no_takeover(outcome, reward):
    wrapper, env, mux = start_followers()
    action = np.zeros(14, np.float32)
    _, r, done, _, info = wrapper.step(action)
    assert not info["rlt_switch_flags"] and not done and r == 0
    wrapper.handle_key("b")
    wrapper.handle_key("t")
    assert "TAKEOVER" not in mux.requests
    _, _, _, _, info = wrapper.step(action)
    assert info["rlt_execution_valid"] and info["rlt_switch_flags"]
    assert info["intervene_action"] is None and env.external_steps == 0
    wrapper.handle_key(outcome)
    _, r, done, truncated, info = wrapper.step(action)
    assert r == reward and done and not truncated
    _, repeated_reward, _, _, info = wrapper.step(action)
    assert repeated_reward == 0 and not info["rlt_execution_valid"]
    wrapper.handle_key("h")
    assert mux.requests[-1] == "HOME"


def test_actor_profile_replay_records_accepted_goal_and_preserves_terminal_reward():
    wrapper, env, _ = start_followers()
    env._actor_direct_execution = True
    env.config.actor_command_profile_enabled = True
    accepted = np.full(14, 0.002, np.float32)
    env.step = Mock(
        return_value=(
            env.read_observation(),
            0.0,
            False,
            False,
            {"piper_accepted_target": accepted},
        )
    )
    wrapper.handle_key("b")
    _, _, _, _, info = wrapper.step(np.full(14, 0.1, np.float32))
    np.testing.assert_array_equal(info["rlt_executed_action"], accepted)
    wrapper.handle_key("c")
    _, reward, done, _, info = wrapper.step(np.full(14, 0.1, np.float32))
    assert reward == 1 and done
    np.testing.assert_array_equal(info["rlt_executed_action"], accepted)


def test_space_keeps_collection_but_home_invalidates_old_chunk():
    wrapper, env, mux = start_followers()
    action = np.zeros(14, np.float32)
    wrapper.handle_key("b")
    wrapper.step(action)
    wrapper.handle_key("Key.space")
    assert wrapper._terminal is None and wrapper._recording
    wrapper.handle_key("h")
    assert mux.requests[-1] == "HOME"
    wrapper._last_press.clear()
    wrapper.handle_key("i")
    count = env.policy_steps
    _, _, _, _, info = wrapper.step(action)
    assert not info["rlt_execution_valid"] and env.policy_steps == count
    wrapper.begin_action_chunk(10)
    _, _, _, _, info = wrapper.step(action)
    assert info["rlt_execution_valid"] and info["rlt_switch_flags"]


def test_i_can_be_retried_after_mux_rejects_stale_feedback():
    wrapper, _, mux = start_followers()
    wrapper.handle_key("Key.space")
    mux.auto_confirm = False
    wrapper._last_press.clear()
    wrapper.handle_key("i")
    assert wrapper._resume_requested and mux.snapshot().hold
    first_count = mux.requests.count("INFERENCE")
    mux.auto_confirm = True
    wrapper._last_press.clear()
    wrapper.handle_key("i")
    assert mux.requests.count("INFERENCE") == first_count + 1
    assert mux.snapshot().policy_active
