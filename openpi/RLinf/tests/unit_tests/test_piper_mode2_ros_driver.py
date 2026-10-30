"""Unit tests for the system-Python-compatible mode-2 CAN bridge."""

from __future__ import annotations

import importlib.util
import struct
import sys
import threading
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


def _load_driver(monkeypatch):
    fake_rospy = ModuleType("rospy")
    fake_rospy.is_shutdown = lambda: False
    fake_sensor = ModuleType("sensor_msgs.msg")
    fake_sensor.JointState = object
    fake_services = ModuleType("std_srvs.srv")
    fake_services.SetBool = object
    fake_services.SetBoolResponse = lambda **kwargs: SimpleNamespace(**kwargs)
    fake_services.Trigger = object
    fake_services.TriggerResponse = lambda **kwargs: SimpleNamespace(**kwargs)
    fake_can = ModuleType("can")
    fake_can.Message = lambda **kwargs: SimpleNamespace(**kwargs)
    monkeypatch.setitem(sys.modules, "can", fake_can)
    monkeypatch.setitem(sys.modules, "rospy", fake_rospy)
    monkeypatch.setitem(sys.modules, "sensor_msgs", ModuleType("sensor_msgs"))
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", fake_sensor)
    monkeypatch.setitem(sys.modules, "std_srvs", ModuleType("std_srvs"))
    monkeypatch.setitem(sys.modules, "std_srvs.srv", fake_services)
    path = (
        Path(__file__).parents[2]
        / "toolkits"
        / "realworld_check"
        / "piper_mode2_ros_driver.py"
    )
    spec = importlib.util.spec_from_file_location("piper_mode2_ros_driver", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _channel(module):
    channel = module.OffsetLeaderChannel.__new__(module.OffsetLeaderChannel)
    channel.can_name = "can_test"
    channel._lock = threading.Lock()
    channel._position = [0.0] * 7
    channel._seen = set()
    channel._timestamps = {}
    return channel


def _follower_channel(module):
    channel = module.FollowerFeedbackChannel.__new__(
        module.FollowerFeedbackChannel
    )
    channel.can_name = "can_test"
    channel._lock = threading.Lock()
    channel._position = [0.0] * 7
    channel._seen = set()
    channel._timestamps = {}
    return channel


def test_shifted_control_frames_decode_to_radians_and_metres(monkeypatch) -> None:
    module = _load_driver(monkeypatch)
    channel = _channel(module)
    frames = [
        (0x165, struct.pack(">ii", 90_000, -45_000)),
        (0x166, struct.pack(">ii", 10_000, 20_000)),
        (0x167, struct.pack(">ii", -30_000, 40_000)),
        (0x169, struct.pack(">i", 50_000) + bytes(4)),
    ]
    for arbitration_id, data in frames:
        channel._process_frame(
            SimpleNamespace(arbitration_id=arbitration_id, data=data),
            timestamp=12.0,
        )

    position, timestamp, complete = channel.sample()

    assert complete
    assert timestamp == 12.0
    np.testing.assert_allclose(
        position,
        [np.pi / 2, -np.pi / 4, np.pi / 18, np.pi / 9, -np.pi / 6, 2 * np.pi / 9, 0.05],
    )


def test_standard_master_frames_are_ignored_to_prevent_policy_conflict(
    monkeypatch,
) -> None:
    module = _load_driver(monkeypatch)
    channel = _channel(module)

    channel._process_frame(
        SimpleNamespace(arbitration_id=0x155, data=struct.pack(">ii", 1, 2)),
        timestamp=1.0,
    )

    position, timestamp, complete = channel.sample()
    np.testing.assert_array_equal(position, np.zeros(7))
    assert timestamp == 0.0
    assert not complete


def test_mode2_fails_closed_without_preisolated_leader_confirmation(
    monkeypatch,
) -> None:
    module = _load_driver(monkeypatch)
    module.rospy.get_param = lambda _name, default=None: default

    with pytest.raises(RuntimeError, match="requires physically verified 0x16x"):
        module.PiperMode2Bridge()


def test_gripper_is_clamped_to_physical_range(monkeypatch) -> None:
    module = _load_driver(monkeypatch)
    channel = _channel(module)

    channel._process_frame(
        SimpleNamespace(
            arbitration_id=0x169,
            data=struct.pack(">i", 100_000) + bytes(4),
        ),
        timestamp=1.0,
    )

    position, _, _ = channel.sample()
    assert position[6] == 0.08


def test_one_fresh_frame_does_not_mask_stale_joint_pairs(monkeypatch) -> None:
    module = _load_driver(monkeypatch)
    channel = _channel(module)
    frames = [
        (0x165, struct.pack(">ii", 1, 2)),
        (0x166, struct.pack(">ii", 3, 4)),
        (0x167, struct.pack(">ii", 5, 6)),
        (0x169, struct.pack(">i", 7) + bytes(4)),
    ]
    for arbitration_id, data in frames:
        channel._process_frame(
            SimpleNamespace(arbitration_id=arbitration_id, data=data),
            timestamp=1.0,
        )
    channel._process_frame(
        SimpleNamespace(
            arbitration_id=0x169,
            data=struct.pack(">i", 8) + bytes(4),
        ),
        timestamp=2.0,
    )

    _, timestamp, complete = channel.sample()

    assert complete
    assert timestamp == 1.0


def test_follower_feedback_decodes_native_frames(monkeypatch) -> None:
    module = _load_driver(monkeypatch)
    channel = _follower_channel(module)
    frames = [
        (0x2A5, struct.pack(">ii", 90_000, -45_000)),
        (0x2A6, struct.pack(">ii", 10_000, 20_000)),
        (0x2A7, struct.pack(">ii", -30_000, 40_000)),
        (0x2A8, struct.pack(">i", 50_000) + bytes(4)),
    ]
    for arbitration_id, data in frames:
        channel._process_frame(
            SimpleNamespace(arbitration_id=arbitration_id, data=data),
            timestamp=12.0,
        )

    position, timestamp, complete = channel.sample()

    assert complete
    assert timestamp == 12.0
    np.testing.assert_allclose(
        position,
        [
            np.pi / 2,
            -np.pi / 4,
            np.pi / 18,
            np.pi / 9,
            -np.pi / 6,
            2 * np.pi / 9,
            0.05,
        ],
    )


def test_native_restore_requires_aligned_leader_and_follower(monkeypatch) -> None:
    module = _load_driver(monkeypatch)
    now = time.monotonic()
    bridge = module.PiperMode2Bridge.__new__(module.PiperMode2Bridge)
    bridge.feedback_timeout = 0.1
    bridge.restore_joint_tolerance = 0.03
    bridge.restore_gripper_tolerance = 0.005
    aligned = [0.0, 0.1, -0.2, 0.3, -0.4, 0.5, 0.02]
    bridge.left = SimpleNamespace(sample=Mock(return_value=(aligned, now, True)))
    bridge.right = SimpleNamespace(sample=Mock(return_value=(aligned, now, True)))
    bridge.left_follower = SimpleNamespace(
        sample=Mock(return_value=(aligned, now, True))
    )
    bridge.right_follower = SimpleNamespace(
        sample=Mock(return_value=(aligned, now, True))
    )

    safe, failures = bridge._restoration_safe()
    assert safe
    assert failures == []

    mismatched = list(aligned)
    mismatched[3] += 0.2
    bridge.left_follower.sample.return_value = (mismatched, now, True)

    safe, failures = bridge._restoration_safe()
    assert not safe
    assert any("left joint error" in failure for failure in failures)


def test_recovery_alignment_target_is_velocity_bounded(monkeypatch) -> None:
    module = _load_driver(monkeypatch)
    current = [0.0] * 6 + [0.02]
    target = [0.5, -0.5, 0.01, -0.01, 0.2, -0.2, 0.08]

    result = module.PiperMode2Bridge._bounded_target(
        current,
        target,
        joint_step=0.002,
        gripper_step=0.0005,
    )

    np.testing.assert_allclose(
        result,
        [0.002, -0.002, 0.002, -0.002, 0.002, -0.002, 0.0205],
    )


def test_recovery_alignment_uses_restore_tolerances(monkeypatch) -> None:
    module = _load_driver(monkeypatch)
    bridge = module.PiperMode2Bridge.__new__(module.PiperMode2Bridge)
    bridge.restore_joint_tolerance = 0.03
    bridge.restore_gripper_tolerance = 0.005
    leader = [0.0] * 7
    within = [0.02] * 6 + [0.004]
    outside = [0.02] * 5 + [0.031, 0.004]

    assert bridge._aligned(leader, leader, within, within)
    assert not bridge._aligned(leader, leader, outside, within)


def test_recovery_alignment_snapshots_stationary_leader_pose(monkeypatch) -> None:
    module = _load_driver(monkeypatch)
    now = time.monotonic()
    bridge = module.PiperMode2Bridge.__new__(module.PiperMode2Bridge)
    bridge.feedback_timeout = 0.1
    bridge.alignment_joint_velocity = 0.05
    bridge.alignment_gripper_velocity = 0.01
    bridge._teleop_lock = threading.Lock()
    bridge._teleop_enabled = False
    bridge._alignment_enabled = False
    bridge._alignment_stable_count = 0
    bridge._alignment_targets = None
    leader = [0.1] * 6 + [0.02]
    follower = [0.0] * 7
    # A stationary leader's last complete pose can be old; followers must be fresh.
    bridge.left = SimpleNamespace(sample=Mock(return_value=(leader, 1.0, True)))
    bridge.right = SimpleNamespace(sample=Mock(return_value=(leader, 1.0, True)))
    bridge.left_follower = SimpleNamespace(
        sample=Mock(return_value=(follower, now, True))
    )
    bridge.right_follower = SimpleNamespace(
        sample=Mock(return_value=(follower, now, True))
    )

    response = bridge._start_alignment(None)

    assert response.success
    assert bridge._alignment_targets == (leader, leader)
    assert bridge._alignment_commands == (follower, follower)
    assert bridge._alignment_enabled


def test_alignment_status_reports_both_side_errors(monkeypatch) -> None:
    module = _load_driver(monkeypatch)
    targets = ([0.1] * 6 + [0.02], [0.2] * 6 + [0.03])
    message = module.PiperMode2Bridge._format_errors(
        targets,
        [0.0] * 7,
        [0.0] * 7,
    )

    assert "left joint=0.1000rad" in message
    assert "left" in message and "right" in message
    assert "gripper=0.0200m" in message
