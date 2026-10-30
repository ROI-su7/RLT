"""Four-CAN Stage-2 episode paths without ROS, CAN, or robot motion."""

import inspect
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
import torch

from rlinf.envs.realworld.piper.four_can_mux import MuxSnapshot
from rlinf.envs.realworld.piper.mux_rlt_wrapper import PiperMuxRLTWrapper
from rlinf.envs.realworld.piper.piper_env import PiperRobotConfig
from rlinf.models.embodiment.mlp_policy.rlt_action_space import RLTDeltaActionSpace
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACLossMixin
from rlinf.workers.env.env_worker import EnvWorker


class FakeEnv(gym.Env):
    action_space = gym.spaces.Box(-1, 1, (14,), dtype=np.float32)
    observation_space = gym.spaces.Box(-1, 1, (14,), dtype=np.float32)

    def __init__(self):
        self._logger = Mock()
        self.config = SimpleNamespace(step_frequency=1000)
        self.resume_policy_output = Mock()
        self.begin_action_chunk = Mock()
        self.reset_called = threading.Event()
        self.policy_steps = 0
        self.external_steps = 0

    def read_observation(self):
        return {"state": {"joint_position": np.zeros(14, np.float32)}}

    def reset(self, **kwargs):
        assert kwargs["options"]["preserve_suspended_policy_output"]
        self.reset_called.set()
        return self.read_observation(), {}

    def step(self, action):
        self.policy_steps += 1
        return self.read_observation(), 0.5, False, False, {}

    def step_external_control(self):
        self.external_steps += 1
        return self.read_observation(), 0.5, False, False, {}


class FakeMux:
    def __init__(self):
        self.status = MuxSnapshot("HOLD", "HOLD", False, True, False, 0)
        self.requests = []
        self.action = np.linspace(-0.1, 0.1, 14, dtype=np.float32)
        self.auto_confirm = True

    def snapshot(self):
        return self.status

    def request_hold(self):
        self.requests.append("HOLD")
        self.status = MuxSnapshot(
            "HOLD", "HOLD", False, True, False, self.status.generation + 1
        )

    def request_home(self):
        self.requests.append("HOME")

    def request_mode(self, mode):
        self.requests.append(mode)
        if self.auto_confirm:
            self.status = MuxSnapshot(
                mode,
                "ACTIVE",
                mode == "INFERENCE",
                False,
                False,
                self.status.generation + 1,
            )

    def executed_action(self):
        return self.action.copy()


def start_episode():
    env, mux = FakeEnv(), FakeMux()
    wrapper = PiperMuxRLTWrapper(env, mux, listener=Mock(), start_keyboard_thread=False)
    with ThreadPoolExecutor(max_workers=1) as executor:
        reset = executor.submit(wrapper.reset)
        assert env.reset_called.wait(1)
        wrapper.handle_key("i")
        reset.result(timeout=1)
    wrapper.begin_action_chunk(10)
    return wrapper, env, mux


@pytest.mark.parametrize(
    ("takeover", "outcome", "reward"),
    [
        (False, "c", 1.0),
        (False, "x", 0.0),
        (True, "c", 1.0),
        (True, "x", 0.0),
    ],
)
def test_i_b_optional_t_c_x(takeover, outcome, reward):
    wrapper, env, mux = start_episode()
    wrapper.handle_key("b")
    if takeover:
        wrapper.handle_key("t")
    _, step_reward, done, _, info = wrapper.step(np.ones(14, np.float32))
    assert step_reward == 0 and not done
    assert info["rlt_switch_flags"]
    assert info["rlt_execution_valid"]
    if takeover:
        np.testing.assert_array_equal(info["intervene_action"], mux.action)
        assert env.policy_steps == 0 and env.external_steps == 1
    else:
        assert info["intervene_action"] is None
        assert env.policy_steps == 1
    wrapper.handle_key(outcome)
    _, terminal_reward, done, truncated, info = wrapper.step(np.ones(14, np.float32))
    assert terminal_reward == reward and done and not truncated
    assert info["rlt_keyboard_terminal"]
    steps = env.policy_steps + env.external_steps
    _, repeated_reward, done, _, info = wrapper.step(np.ones(14, np.float32))
    assert repeated_reward == 0 and done
    assert not info["rlt_execution_valid"]
    assert not info["rlt_keyboard_terminal"]
    assert info["intervene_action"] is None
    assert env.policy_steps + env.external_steps == steps
    assert mux.requests[-1] == "HOLD"
    wrapper.handle_key("h")
    assert mux.requests[-1] == "HOME"


@pytest.mark.parametrize("outcome", ["c", "x"])
def test_terminal_without_b_does_not_emit_replay_label(outcome):
    wrapper, _, _ = start_episode()
    wrapper.step(np.ones(14, np.float32))
    wrapper.handle_key(outcome)
    _, _, done, _, info = wrapper.step(np.ones(14, np.float32))
    assert done
    assert not info["rlt_switch_flags"]
    assert not info["rlt_keyboard_terminal"]


def test_b_is_required_for_takeover_and_old_diagnostic_keys_are_inert():
    wrapper, _, mux = start_episode()
    wrapper.handle_key("t")
    assert "TAKEOVER" not in mux.requests
    before = list(mux.requests)
    for key in ("a", "g", "f", "r"):
        wrapper.handle_key(key)
    assert mux.requests == before
    assert not wrapper._recording


def test_space_preserves_episode_and_b_but_discards_old_chunk():
    wrapper, env, mux = start_episode()
    wrapper.handle_key("b")
    wrapper.step(np.ones(14, np.float32))
    wrapper.handle_key("Key.space")
    assert wrapper._terminal is None and wrapper._recording
    assert mux.requests[-1] == "HOLD"
    # The first I was recent; clear debounce to model an operator returning.
    wrapper._last_press.clear()
    wrapper.handle_key("i")
    _, reward, done, _, info = wrapper.step(np.ones(14, np.float32))
    assert reward == 0 and not done
    assert not info["rlt_execution_valid"]
    assert env.policy_steps == 1
    wrapper.begin_action_chunk(10)
    _, _, _, _, info = wrapper.step(np.ones(14, np.float32))
    assert info["rlt_execution_valid"] and env.policy_steps == 2


def test_takeover_waits_for_active_not_just_t_request():
    wrapper, env, mux = start_episode()
    wrapper.handle_key("b")
    mux.auto_confirm = False
    wrapper.handle_key("t")
    with ThreadPoolExecutor(max_workers=1) as executor:
        step = executor.submit(wrapper.step, np.ones(14, np.float32))
        assert env.policy_steps == 0 and env.external_steps == 0
        mux.status = MuxSnapshot("TAKEOVER", "ACTIVE", False, False, False, 2)
        info = step.result(timeout=1)[-1]
    assert info["intervene_action"] is not None
    assert env.policy_steps == 0


def test_i_after_human_replans_before_policy_execution():
    wrapper, env, _ = start_episode()
    wrapper.handle_key("b")
    wrapper.handle_key("t")
    wrapper.step(np.ones(14, np.float32))
    wrapper._last_press.clear()
    wrapper.handle_key("i")
    info = wrapper.step(np.ones(14, np.float32))[-1]
    assert not info["rlt_execution_valid"]
    assert wrapper._recording and wrapper._terminal is None
    assert env.policy_steps == 0
    wrapper.begin_action_chunk(10)
    wrapper.step(np.ones(14, np.float32))
    assert env.policy_steps == 1


def test_home_is_idle_only_and_does_not_capture_a_new_pose():
    wrapper, _, mux = start_episode()
    wrapper.handle_key("h")
    assert "HOME" not in mux.requests
    wrapper.handle_key("x")
    wrapper._last_press.clear()
    wrapper.handle_key("h")
    assert mux.requests[-1] == "HOME"


def test_space_then_h_uses_original_home_without_failing_episode():
    wrapper, _, mux = start_episode()
    wrapper.handle_key("b")
    wrapper.handle_key("Key.space")
    wrapper.handle_key("h")
    assert mux.requests[-1] == "HOME"
    assert wrapper._terminal is None and wrapper._recording


def test_mux_startup_seeds_policy_before_waiting_for_inference_active():
    env, mux = FakeEnv(), FakeMux()
    mux.auto_confirm = False
    wrapper = PiperMuxRLTWrapper(env, mux, listener=Mock(), start_keyboard_thread=False)
    env.resume_policy_output.side_effect = lambda: setattr(
        mux, "status", MuxSnapshot("INFERENCE", "IDLE", True, False, False, 2)
    )
    with ThreadPoolExecutor(max_workers=1) as executor:
        reset = executor.submit(wrapper.reset)
        assert env.reset_called.wait(1)
        wrapper.handle_key("i")
        # HOLD has released, but the original mux is still waiting for fresh
        # policy inputs and leader alignment before setting inference_active.
        mux.status = MuxSnapshot("INFERENCE", "IDLE", False, False, False, 2)
        reset.result(timeout=1)
    env.resume_policy_output.assert_called_once()


@pytest.mark.parametrize(
    "overrides",
    [
        {"isolated_control_process": False},
        {"left_command_topic": "/master/joint_left"},
    ],
)
def test_mux_rejects_competing_or_unacknowledged_policy_paths(overrides):
    cfg = {
        "four_can_mux_enabled": True,
        "isolated_control_process": True,
        "command_publish_hz": 200,
        "command_interpolation_duration_s": 1 / 15,
        "left_command_topic": "/teleop/policy/joint_left",
        "right_command_topic": "/teleop/policy/joint_right",
    }
    cfg.update(overrides)
    with pytest.raises(ValueError):
        PiperRobotConfig(**cfg)


@pytest.mark.parametrize(
    ("valid", "done", "record"),
    [
        ([True, True, False], [False, True, True], True),
        ([True, False, True], [False, False, False], False),
    ],
)
def test_execution_outcome_filters_hold_but_keeps_terminal_prefix(valid, done, record):
    inputs = {"record_transition": torch.ones(1, 1, dtype=torch.bool)}
    rollout = SimpleNamespace(forward_inputs=[inputs], actions=[torch.ones(1, 42)])
    worker = SimpleNamespace(
        train_num_envs_per_stage=1,
        model_cfg=SimpleNamespace(num_action_chunks=3),
        _rlt_last_chunk_info=[
            {
                "execution_valid": torch.tensor([valid]),
                "step_dones": torch.tensor([done]),
                "executed_actions": torch.zeros(1, 3, 14),
            }
        ],
        rollout_results=[rollout],
        _rlt_bridge_tail=[None],
        _rlt_expected_rtc_offset=[0],
        _rlt_actor_plan_offset=0,
        _rlt_stride_enabled=False,
        _rlt_episode_ids=[torch.zeros(1, dtype=torch.long)],
        _new_rlt_episode_ids=lambda count: torch.ones(count, dtype=torch.long),
    )
    outcome = SimpleNamespace(dones=torch.tensor([done]), intervene_flags=None)
    EnvWorker._rlt_end_chunk(worker, 0, outcome, {})
    assert inputs["record_transition"].item() == record
    torch.testing.assert_close(inputs["action_valid_mask"], torch.tensor([valid]))
    assert not rollout.actions[0].any()


def test_manual_non_auto_reset_reuses_live_observation_until_terminal() -> None:
    """Manual Piper reset waits for I only at bootstrap and after a terminal."""
    env = Mock()
    env.reset.side_effect = [({"step": 0}, {}), ({"step": 9}, {})]
    worker = SimpleNamespace(
        train_num_envs_per_stage=1,
        model_cfg=SimpleNamespace(num_action_chunks=2),
        cfg=SimpleNamespace(
            env=SimpleNamespace(train=SimpleNamespace(auto_reset=False))
        ),
        stage_num=1,
        env_list=[env],
        enable_rlt=True,
        enable_online_lerobot=False,
        _defer_train_reset=True,
        _train_reset_pending=[True],
        _reset_rlt_execution_state=Mock(),
        last_obs_list=[{"step": 1}],
        last_intervened_info_list=[(None, None)],
        train_last_rlt_switch_flags=[None],
    )

    bootstrap = inspect.unwrap(EnvWorker.bootstrap_step)
    first = bootstrap(worker)
    assert first[0].obs == {"step": 0}
    assert env.reset.call_count == 1
    assert worker._train_reset_pending == [False]

    second = bootstrap(worker)
    assert second[0].obs == {"step": 1}
    assert env.reset.call_count == 1

    worker._train_reset_pending = [True]
    third = bootstrap(worker)
    assert third[0].obs == {"step": 9}
    assert env.reset.call_count == 2


def test_terminal_padding_is_excluded_from_human_and_reference_bc():
    loss = RLTACLossMixin()
    # Default BC options: every valid row counts with weight 1.
    loss.cfg = SimpleNamespace(algorithm={})
    loss._chunk_shape = lambda: (3, 14)
    loss._rlt_action_space = lambda device: RLTDeltaActionSpace.from_config(
        None, action_dim=14
    )
    actions, reference = torch.zeros(1, 3, 14), torch.zeros(1, 3, 14)
    actions[:, 1] = 1.0
    reference[:, 2] = 100.0  # Post-terminal padding must not affect BC.
    value, metrics = loss._bc_metrics(
        pi=torch.zeros(1, 42),
        actions=actions,
        ref_chunk=reference,
        intervene_flags=torch.tensor([[False, True, False]]),
        proprio=torch.zeros(1, 14),
        valid_mask=torch.tensor([[True, True, False]]),
    )
    assert value.item() == 0.5
    assert metrics["human_mask_ratio"] == 0.5
    assert metrics["bc_human_loss"] == 1.0
    assert metrics["bc_ref_loss"] == 0.0


def test_mux_rtc_is_gated_by_current_observation_and_control_ownership():
    wrapper, env, mux = start_episode()
    env.queue_rtc_tail = Mock()
    env.start_rtc_tail = Mock()
    env.align_rtc_chunk = Mock(return_value=(np.zeros((10, 14)), 5))
    env._stop_rtc_tail = Mock()
    tail = np.ones((5, 14), np.float32)
    wrapper.queue_rtc_tail(tail)
    wrapper.start_rtc_tail(tail)
    assert env.queue_rtc_tail.call_count == 1
    assert env.start_rtc_tail.call_count == 1
    assert wrapper.align_rtc_chunk(np.zeros((10, 14)), tail)[1] == 5
    wrapper.handle_key("Key.space")
    wrapper.queue_rtc_tail(tail)
    wrapper.start_rtc_tail(tail)
    assert env.queue_rtc_tail.call_count == 1
    assert env.start_rtc_tail.call_count == 1
    assert wrapper.align_rtc_chunk(np.zeros((10, 14)), tail)[1] == 0
    env._stop_rtc_tail.assert_called_once()
    # Even after I, the old observation's suffix cannot be restarted.
    wrapper._last_press.clear()
    wrapper.handle_key("i")
    wrapper.queue_rtc_tail(tail)
    assert env.queue_rtc_tail.call_count == 1


def test_t_rejects_rtc_before_mux_transition_is_confirmed():
    wrapper, env, mux = start_episode()
    env.queue_rtc_tail = Mock()
    env.start_rtc_tail = Mock()
    wrapper.handle_key("b")
    mux.auto_confirm = False
    wrapper.handle_key("t")
    assert mux.status.policy_active
    wrapper.queue_rtc_tail(np.ones((5, 14)))
    wrapper.start_rtc_tail(np.ones((5, 14)))
    env.queue_rtc_tail.assert_not_called()
    env.start_rtc_tail.assert_not_called()


def test_hold_during_rtc_drain_discards_partial_alignment():
    wrapper, env, _ = start_episode()
    env._stop_rtc_tail = Mock()

    def interrupted_alignment(actions, plan):
        wrapper.handle_key("Key.space")
        return actions + 1, 2

    env.align_rtc_chunk = interrupted_alignment
    actions = np.zeros((10, 14), np.float32)
    aligned, offset = wrapper.align_rtc_chunk(actions, np.zeros((50, 14)))
    np.testing.assert_array_equal(aligned, actions)
    assert offset == 0
    wrapper.begin_action_chunk(10)
    assert wrapper._chunk_invalid
