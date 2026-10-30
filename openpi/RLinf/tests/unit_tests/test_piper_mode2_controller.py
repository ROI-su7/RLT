"""Tests for mode-2 command ownership and hardware side effects."""

from __future__ import annotations

import time

import numpy as np
import pytest

from rlinf.envs.realworld.piper.leader_takeover import (
    LeaderTakeoverConfig,
    LeaderTakeoverStateMachine,
    TakeoverState,
)
from rlinf.envs.realworld.piper.mode2_controller import (
    ARM_INDICES,
    Mode2Feedback,
    PiperMode2Controller,
)


class FakeHardware:
    def __init__(self) -> None:
        self.leader = np.zeros(14, dtype=np.float32)
        self.follower = np.zeros(14, dtype=np.float32)
        self.timestamp = 1.0
        self.commands: list[np.ndarray] = []
        self.drag_calls = 0
        self.drag_error: Exception | None = None
        self.command_error: Exception | None = None
        self.teleop_error: Exception | None = None
        self.teleop_enabled = False

    def feedback(self) -> Mode2Feedback:
        return Mode2Feedback(
            leader_action=self.leader.copy(),
            follower_action=self.follower.copy(),
            leader_timestamp=self.timestamp,
            follower_timestamp=self.timestamp,
        )

    def command_leader_joints(self, joints: np.ndarray) -> None:
        if self.command_error is not None:
            raise self.command_error
        self.commands.append(np.asarray(joints).copy())

    def enter_leader_drag_mode(self) -> None:
        self.drag_calls += 1
        if self.drag_error is not None:
            raise self.drag_error

    def set_follower_teleop_enabled(self, enabled: bool) -> None:
        if self.teleop_error is not None:
            raise self.teleop_error
        self.teleop_enabled = enabled


def _controller(*, stable_samples: int = 1) -> tuple[PiperMode2Controller, FakeHardware]:
    hardware = FakeHardware()
    machine = LeaderTakeoverStateMachine(
        LeaderTakeoverConfig(
            joint_count=12,
            approach_stable_samples=1,
            stable_samples=stable_samples,
            feedback_timeout_s=0.2,
        )
    )
    return PiperMode2Controller(machine, hardware), hardware


def _manual_controller() -> tuple[PiperMode2Controller, FakeHardware]:
    hardware = FakeHardware()
    machine = LeaderTakeoverStateMachine(
        LeaderTakeoverConfig(
            joint_count=12,
            manual_approach=True,
            approach_stable_samples=1,
            stable_samples=1,
            feedback_timeout_s=0.2,
        )
    )
    return PiperMode2Controller(machine, hardware), hardware


def test_policy_action_passes_through_without_hardware_command() -> None:
    controller, hardware = _controller()
    policy = np.arange(14, dtype=np.float32)

    selected, intervene = controller.action_for_step(policy)

    np.testing.assert_array_equal(selected, policy)
    assert not intervene
    assert hardware.commands == []


def test_hold_latches_measured_follower_action_including_grippers() -> None:
    controller, hardware = _controller()
    hardware.follower = np.linspace(0.0, 0.13, 14, dtype=np.float32)

    held = controller.request_hold()
    hardware.follower[:] = 9.0
    selected, intervene = controller.action_for_step(np.ones(14))

    np.testing.assert_array_equal(held, selected)
    assert not intervene
    assert controller.state == TakeoverState.FOLLOWER_HOLD


def test_alignment_commands_only_twelve_arm_joints() -> None:
    controller, hardware = _controller(stable_samples=2)
    hardware.follower[ARM_INDICES] = 1.0
    controller.request_hold()

    controller.request_alignment(now=1.0)

    assert len(hardware.commands) == 1
    assert hardware.commands[0].shape == (12,)
    np.testing.assert_allclose(hardware.commands[0], 0.002)


def test_ready_then_takeover_selects_leader_and_marks_intervention() -> None:
    controller, hardware = _controller()
    controller.request_hold()
    controller.request_alignment(now=1.0)
    assert controller.state == TakeoverState.LEADER_MANUAL_ALIGN
    controller.alignment_tick(now=1.0)
    assert controller.state == TakeoverState.READY

    controller.request_takeover(now=1.0)
    hardware.leader = np.linspace(-0.1, 0.1, 14, dtype=np.float32)
    hardware.timestamp = time.monotonic()
    selected, intervene = controller.action_for_step(np.ones(14))

    np.testing.assert_array_equal(selected, hardware.leader)
    assert intervene
    assert hardware.drag_calls == 1
    assert hardware.teleop_enabled


def test_automatic_approach_stops_near_then_requires_manual_fine_alignment() -> None:
    controller, hardware = _controller()
    hardware.follower[ARM_INDICES] = 0.5
    controller.request_hold()
    controller.request_alignment(now=1.0)
    assert controller.state == TakeoverState.LEADER_ALIGNING
    assert len(hardware.commands) == 1

    hardware.leader[ARM_INDICES] = 0.45
    controller.alignment_tick(now=1.0)
    assert controller.state == TakeoverState.LEADER_MANUAL_ALIGN
    assert hardware.drag_calls == 1
    assert len(hardware.commands) == 1

    controller.alignment_tick(now=1.0)
    assert controller.state == TakeoverState.LEADER_MANUAL_ALIGN

    hardware.leader[ARM_INDICES] = 0.5
    controller.alignment_tick(now=1.0)
    assert controller.state == TakeoverState.READY


def test_manual_approach_never_commands_leader_and_monitors_until_ready() -> None:
    controller, hardware = _manual_controller()
    hardware.follower[ARM_INDICES] = 0.5
    controller.request_hold()

    assert controller.request_alignment(now=1.0) is None
    assert controller.state == TakeoverState.LEADER_ALIGNING
    assert hardware.commands == []
    assert hardware.drag_calls == 1

    hardware.leader[ARM_INDICES] = 0.45
    controller.alignment_tick(now=1.0)
    assert controller.state == TakeoverState.LEADER_MANUAL_ALIGN
    hardware.leader[ARM_INDICES] = 0.5
    controller.alignment_tick(now=1.0)
    assert controller.state == TakeoverState.READY
    assert hardware.commands == []


def test_takeover_before_ready_never_enters_drag_mode() -> None:
    controller, hardware = _controller(stable_samples=2)
    controller.request_hold()

    with pytest.raises(RuntimeError, match="cannot enter teleoperation"):
        controller.request_takeover(now=1.0)

    assert hardware.drag_calls == 0


def test_takeover_service_failure_faults_without_policy_fallback() -> None:
    controller, hardware = _controller()
    controller.request_hold()
    controller.request_alignment(now=1.0)
    controller.alignment_tick(now=1.0)
    hardware.teleop_error = ConnectionError("bridge unavailable")

    with pytest.raises(RuntimeError, match="bridge unavailable"):
        controller.request_takeover(now=1.0)

    assert controller.state == TakeoverState.FAULT


def test_drag_mode_failure_faults_and_does_not_enable_teleop() -> None:
    controller, hardware = _controller()
    controller.request_hold()
    hardware.drag_error = OSError("CAN rejected request")

    with pytest.raises(RuntimeError, match="CAN rejected request"):
        controller.request_alignment(now=1.0)

    assert controller.state == TakeoverState.FAULT
    selected, intervene = controller.action_for_step(np.ones(14))
    np.testing.assert_array_equal(selected, np.zeros(14))
    assert not intervene


def test_non_finite_hardware_feedback_is_rejected() -> None:
    controller, hardware = _controller()
    hardware.follower[3] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        controller.request_hold()


def test_terminal_hold_can_resume_policy_from_measured_follower() -> None:
    controller, hardware = _controller()
    controller.capture_episode_start()
    controller.request_hold()
    controller.request_alignment(now=1.0)
    controller.alignment_tick(now=1.0)
    controller.request_takeover(now=1.0)
    controller.finish_episode()
    assert controller.state == TakeoverState.TELEOP_RETURN
    hardware.timestamp = time.monotonic()
    controller.complete_teleop_return()
    hardware.follower[ARM_INDICES] = 0.25

    seed = controller.resume_policy()

    np.testing.assert_allclose(seed, 0.25)
    assert controller.state == TakeoverState.POLICY
    assert not hardware.teleop_enabled


def test_teleop_return_rejects_r_until_both_pairs_reach_start() -> None:
    controller, hardware = _controller()
    controller.capture_episode_start()
    controller.request_hold()
    controller.request_alignment(now=1.0)
    controller.alignment_tick(now=1.0)
    controller.request_takeover(now=1.0)
    controller.finish_episode()
    hardware.leader[0] = 0.1
    hardware.timestamp = time.monotonic()

    with pytest.raises(RuntimeError, match="not at the recorded start"):
        controller.complete_teleop_return()

    assert controller.state == TakeoverState.TELEOP_RETURN
    assert hardware.teleop_enabled


def test_hold_from_teleop_disables_bridge_before_latching_hold() -> None:
    controller, hardware = _controller()
    controller.request_hold()
    controller.request_alignment(now=1.0)
    controller.alignment_tick(now=1.0)
    controller.request_takeover(now=1.0)

    controller.request_hold()

    assert controller.state == TakeoverState.FOLLOWER_HOLD
    assert not hardware.teleop_enabled


def test_emergency_stop_disables_bridge_and_faults() -> None:
    controller, hardware = _controller()
    controller.request_hold()
    controller.request_alignment(now=1.0)
    controller.alignment_tick(now=1.0)
    controller.request_takeover(now=1.0)

    controller.emergency_stop()

    assert controller.state == TakeoverState.FAULT
    assert not hardware.teleop_enabled


def test_stale_feedback_during_teleop_disables_bridge_and_faults() -> None:
    controller, hardware = _controller()
    controller.request_hold()
    now = hardware.timestamp
    controller.request_alignment(now=now)
    controller.alignment_tick(now=now)
    controller.request_takeover(now=now)
    hardware.timestamp = time.monotonic() - 1.0

    with pytest.raises(RuntimeError, match="feedback is stale"):
        controller.action_for_step(np.zeros(14, dtype=np.float32))

    assert controller.state == TakeoverState.FAULT
    assert not hardware.teleop_enabled


def test_disconnected_leader_target_latches_fault_and_preserves_hold() -> None:
    controller, hardware = _controller()
    hardware.follower[ARM_INDICES] = 1.0
    controller.request_hold()
    hardware.command_error = ConnectionError("no target subscriber")

    with pytest.raises(RuntimeError, match="no target subscriber"):
        controller.request_alignment(now=1.0)

    assert controller.state == TakeoverState.FAULT
    selected, intervene = controller.action_for_step(np.ones(14))
    np.testing.assert_array_equal(selected, hardware.follower)
    assert not intervene
