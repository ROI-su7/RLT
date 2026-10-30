"""Run the actual isolated executor against a fake ROS transport, not CAN."""

import queue
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.envs.realworld.piper import control_process
from rlinf.envs.realworld.piper.piper_env import PiperRobotConfig


class Message:
    def __init__(self, data=None):
        self.data = data
        self.header = SimpleNamespace(stamp=None)
        self.position = []


@pytest.mark.parametrize("actor_profile", [False, True])
def test_mux_hold_ack_drains_publications_and_requires_explicit_fresh_reset(
    monkeypatch,
    actor_profile,
):
    history = []
    callbacks = {}
    shutdown = [False]

    class Publisher:
        def __init__(self, topic, *args, **kwargs):
            self.topic = topic

        def publish(self, message):
            history.append((self.topic, message.data, list(message.position)))

    def subscribe(topic, message_type, callback, **kwargs):
        callbacks[topic] = callback
        callback(Message(data=True))  # Latched initial HOLD from the mux.

    ros = SimpleNamespace(
        init_node=lambda *a, **k: None,
        Publisher=Publisher,
        Subscriber=subscribe,
        is_shutdown=lambda: shutdown[0],
        Time=SimpleNamespace(now=time.monotonic),
    )
    monkeypatch.setitem(sys.modules, "rospy", ros)
    monkeypatch.setitem(sys.modules, "sensor_msgs", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules, "sensor_msgs.msg", SimpleNamespace(JointState=Message)
    )
    monkeypatch.setitem(sys.modules, "std_msgs", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "std_msgs.msg", SimpleNamespace(Bool=Message))
    monkeypatch.setattr(control_process, "_exit_with_parent", lambda pid: None)
    cfg = PiperRobotConfig(
        four_can_mux_enabled=True,
        isolated_control_process=True,
        command_publish_hz=200,
        command_interpolation_duration_s=1 / 15,
        left_command_topic="/teleop/policy/joint_left",
        right_command_topic="/teleop/policy/joint_right",
        rtc_enabled=True,
        action_smoothing_alpha=0.4,
        max_arm_velocity_rad_s=0.4,
        actor_command_profile_enabled=actor_profile,
    )
    commands, responses = queue.Queue(), queue.Queue()
    action = np.zeros(14, np.float32)

    def request(identifier, operation, payload=None):
        commands.put((identifier, operation, payload))
        response = responses.get(timeout=1)
        assert response[0] == identifier
        return response

    def wait_ack(value, start):
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            if any(
                topic == "/teleop/policy_hold_ack" and data == value
                for topic, data, _ in history[start:]
            ):
                return
            time.sleep(0.001)
        raise AssertionError(f"No HOLD ACK={value}")

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            control_process._control_process_main, asdict(cfg), commands, responses, 0
        )
        try:
            assert responses.get(timeout=1)[1] == "ready"
            wait_ack(True, 0)
            assert request(1, "reset", action)[1] == "ok"
            assert request(2, "action", (action, action))[1] == "error"
            assert not any(
                topic.startswith("/teleop/policy/") for topic, _, _ in history
            )

            start = len(history)
            callbacks["/teleop/hold_active"](Message(data=False))
            wait_ack(False, start)
            # Releasing HOLD does not execute the queued pre-HOLD target.
            assert request(3, "action", (action, action))[1] == "error"
            assert request(4, "reset", action)[1] == "ok"
            assert any(topic.startswith("/teleop/policy/") for topic, _, _ in history)

            proposal = action.copy()
            proposal[0] = 0.1
            reply = request(41, "action_diagnostics", (proposal, action))
            assert reply[1] == "ok"
            assert reply[2]["accepted_target"][0] < proposal[0]
            assert reply[2]["accepted_target"][0] > 0
            np.testing.assert_allclose(
                reply[2]["accepted_target"][0], 0.4 / 15, atol=1e-6
            )
            assert reply[2]["current"].shape == (14,)

            # Actor bypasses the executor EMA while JAX keeps alpha=0.4.
            # The small proposal is below the unchanged velocity limit.
            small = action.copy()
            small[0] = 0.01
            assert request(42, "reset", action)[1] == "ok"
            direct = request(43, "actor_action_diagnostics", (small, action))
            np.testing.assert_allclose(
                direct[2]["accepted_target"][0],
                0.002 if actor_profile else 0.01,
                atol=1e-6,
            )
            if actor_profile:
                assert "actor_profile" in direct[2]
                for tick in range(10):
                    requested = action.copy()
                    requested[0] = 0.02 if tick % 2 else -0.02
                    reply = request(
                        430 + tick, "actor_action_diagnostics", (requested, action)
                    )
                    assert reply[2]["actor_profile"]["arm_amax"] <= 0.4 + 1e-7
                    assert reply[2]["actor_profile"]["arm_vmax"] <= 0.08 + 1e-7
            assert request(44, "reset", action)[1] == "ok"
            jax = request(45, "action_diagnostics", (small, action))
            np.testing.assert_allclose(jax[2]["accepted_target"][0], 0.004, atol=1e-6)

            # The real executor must continue its planned suffix while the
            # parent is waiting for inference, then drain/cancel on HOLD.
            tail = np.zeros((5, 14), np.float32)
            tail[:, 0] = np.linspace(0.01, 0.05, 5)
            assert request(5, "start_rtc", tail)[1] == "ok"
            time.sleep(0.08)
            assert any(
                position and position[0] > 0
                for topic, _, position in history
                if topic == "/teleop/policy/joint_left"
            )
            commands.put((6, "drain_rtc", None))
            time.sleep(0.01)
            start = len(history)
            callbacks["/teleop/hold_active"](Message(data=True))
            wait_ack(True, start)
            drained = responses.get(timeout=1)
            assert drained[:2] == (6, "ok")
            assert drained[2][0] < len(tail)
            count = sum(topic.startswith("/teleop/policy/") for topic, _, _ in history)
            time.sleep(0.03)
            assert (
                sum(topic.startswith("/teleop/policy/") for topic, _, _ in history)
                == count
            )
            assert request(7, "close")[1] == "ok"
            future.result(timeout=1)
        finally:
            shutdown[0] = True
