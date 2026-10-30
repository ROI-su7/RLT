"""Tests for the hardware-independent Piper mode-2 state machine."""

from __future__ import annotations

import numpy as np
import pytest

from rlinf.envs.realworld.piper.leader_takeover import (
    LeaderTakeoverConfig,
    LeaderTakeoverStateMachine,
    TakeoverState,
)


def _machine(**kwargs) -> LeaderTakeoverStateMachine:
    config_values = {
        "joint_count": 4,
        "approach_stable_samples": 1,
        "stable_samples": 3,
    }
    config_values.update(kwargs)
    return LeaderTakeoverStateMachine(LeaderTakeoverConfig(**config_values))


def test_policy_hold_latches_measured_follower_pose() -> None:
    machine = _machine()
    follower = np.array([0.1, 0.2, -0.3, 0.4])

    target = machine.request_hold(follower)
    follower[:] = 9.0

    assert machine.state == TakeoverState.FOLLOWER_HOLD
    np.testing.assert_allclose(target, [0.1, 0.2, -0.3, 0.4])
    np.testing.assert_allclose(
        machine.snapshot().follower_hold_target,
        [0.1, 0.2, -0.3, 0.4],
    )


def test_alignment_target_is_velocity_limited_per_tick() -> None:
    machine = _machine(alignment_rate_hz=50.0, max_velocity_rad_s=0.1)
    follower = np.array([1.0, -1.0, 0.5, -0.5])
    leader = np.zeros(4)
    machine.request_hold(follower)

    target = machine.request_alignment(
        leader_joints=leader,
        follower_joints=follower,
        leader_timestamp=1.0,
        follower_timestamp=1.0,
        now=1.0,
    )

    np.testing.assert_allclose(target, [0.002, -0.002, 0.002, -0.002])
    assert machine.state == TakeoverState.LEADER_ALIGNING


def test_approach_tolerance_must_exceed_fine_tolerance() -> None:
    with pytest.raises(ValueError, match="must be greater"):
        LeaderTakeoverConfig(
            approach_tolerance_rad=0.02,
            position_tolerance_rad=0.02,
        )


def test_ready_requires_consecutive_stable_samples() -> None:
    machine = _machine(position_tolerance_rad=0.02)
    follower = np.zeros(4)
    leader = np.full(4, 0.01)
    machine.request_hold(follower)
    machine.request_alignment(
        leader_joints=leader,
        follower_joints=follower,
        leader_timestamp=1.0,
        follower_timestamp=1.0,
        now=1.0,
    )

    assert machine.state == TakeoverState.LEADER_MANUAL_ALIGN
    for index in range(3):
        machine.manual_alignment_tick(
            leader_joints=leader,
            follower_joints=follower,
            leader_timestamp=1.01 + index * 0.01,
            follower_timestamp=1.01 + index * 0.01,
            now=1.01 + index * 0.01,
        )

    assert machine.state == TakeoverState.READY
    assert machine.snapshot().stable_samples == 3


def test_takeover_requires_ready_and_fresh_aligned_feedback() -> None:
    machine = _machine(stable_samples=1)
    follower = np.zeros(4)
    leader = np.zeros(4)
    machine.request_hold(follower)
    machine.request_alignment(
        leader_joints=leader,
        follower_joints=follower,
        leader_timestamp=1.0,
        follower_timestamp=1.0,
        now=1.0,
    )
    machine.manual_alignment_tick(
        leader_joints=leader,
        follower_joints=follower,
        leader_timestamp=1.0,
        follower_timestamp=1.0,
        now=1.0,
    )
    assert machine.state == TakeoverState.READY

    machine.request_takeover(
        leader_joints=leader,
        follower_joints=follower,
        leader_timestamp=1.01,
        follower_timestamp=1.01,
        now=1.01,
    )

    assert machine.state == TakeoverState.TELEOP


def test_stale_feedback_blocks_alignment() -> None:
    machine = _machine(feedback_timeout_s=0.1)
    follower = np.zeros(4)
    machine.request_hold(follower)

    with pytest.raises(RuntimeError, match="leader feedback is stale"):
        machine.request_alignment(
            leader_joints=np.zeros(4),
            follower_joints=follower,
            leader_timestamp=0.0,
            follower_timestamp=1.0,
            now=1.0,
        )

    assert machine.state == TakeoverState.FOLLOWER_HOLD


def test_follower_drift_during_alignment_latches_fault() -> None:
    machine = _machine(position_tolerance_rad=0.02)
    follower = np.zeros(4)
    machine.request_hold(follower)
    machine.request_alignment(
        leader_joints=np.ones(4),
        follower_joints=follower,
        leader_timestamp=1.0,
        follower_timestamp=1.0,
        now=1.0,
    )

    with pytest.raises(RuntimeError, match="follower left hold"):
        machine.next_alignment_target(
            leader_joints=np.ones(4),
            follower_joints=np.full(4, 0.03),
            leader_timestamp=1.01,
            follower_timestamp=1.01,
            now=1.01,
        )

    assert machine.state == TakeoverState.FAULT


def test_alignment_timeout_latches_fault() -> None:
    machine = _machine(alignment_timeout_s=1.0)
    follower = np.zeros(4)
    machine.request_hold(follower)
    machine.request_alignment(
        leader_joints=np.ones(4),
        follower_joints=follower,
        leader_timestamp=1.0,
        follower_timestamp=1.0,
        now=1.0,
    )

    with pytest.raises(TimeoutError, match="timed out"):
        machine.next_alignment_target(
            leader_joints=np.ones(4),
            follower_joints=follower,
            leader_timestamp=2.01,
            follower_timestamp=2.01,
            now=2.01,
        )

    assert machine.state == TakeoverState.FAULT


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        ("align", "request hold first"),
        ("takeover", "cannot enter teleoperation"),
    ],
)
def test_invalid_transitions_fail_closed(operation: str, message: str) -> None:
    machine = _machine()
    joints = np.zeros(4)

    with pytest.raises(RuntimeError, match=message):
        if operation == "align":
            machine.request_alignment(
                leader_joints=joints,
                follower_joints=joints,
                leader_timestamp=1.0,
                follower_timestamp=1.0,
                now=1.0,
            )
        else:
            machine.request_takeover(
                leader_joints=joints,
                follower_joints=joints,
                leader_timestamp=1.0,
                follower_timestamp=1.0,
                now=1.0,
            )


def test_resume_policy_is_forbidden_directly_from_teleop() -> None:
    machine = _machine(stable_samples=1)
    joints = np.zeros(4)
    machine.request_hold(joints)
    machine.request_alignment(
        leader_joints=joints,
        follower_joints=joints,
        leader_timestamp=1.0,
        follower_timestamp=1.0,
        now=1.0,
    )
    machine.manual_alignment_tick(
        leader_joints=joints,
        follower_joints=joints,
        leader_timestamp=1.0,
        follower_timestamp=1.0,
        now=1.0,
    )
    machine.request_takeover(
        leader_joints=joints,
        follower_joints=joints,
        leader_timestamp=1.0,
        follower_timestamp=1.0,
        now=1.0,
    )

    with pytest.raises(RuntimeError, match="cannot resume policy"):
        machine.resume_policy(joints)

    machine.finish_episode()
    assert machine.state == TakeoverState.TELEOP_RETURN
    with pytest.raises(RuntimeError, match="cannot resume policy"):
        machine.resume_policy(joints)
    machine.complete_teleop_return()
    seed = machine.resume_policy(joints)
    assert machine.state == TakeoverState.POLICY
    np.testing.assert_allclose(seed, joints)
