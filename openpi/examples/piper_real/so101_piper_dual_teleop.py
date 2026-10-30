"""Run synchronized dual SO-101 to dual Piper joint teleoperation."""

from __future__ import annotations

import argparse
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rospy
from sensor_msgs.msg import JointState

from lerobot.teleoperators.so_leader import SO101LeaderConfig, SOLeader
try:
    from .so101_piper_left_teleop import (
        LEADER_KEYS,
        _OneEuroFilter,
        _anchored_action,
        _profile_action,
    )
except ImportError:
    from so101_piper_left_teleop import (
        LEADER_KEYS,
        _OneEuroFilter,
        _anchored_action,
        _profile_action,
    )


@dataclass
class _Side:
    name: str
    profile: dict
    port: str
    identifier: str
    state_topic: str
    command_topic: str
    leader: SOLeader
    publisher: rospy.Publisher
    state: np.ndarray | None = None
    profile_anchor: np.ndarray | None = None
    piper_anchor: np.ndarray | None = None
    command: np.ndarray | None = None
    stable_desired: np.ndarray | None = None
    desired_filter: _OneEuroFilter | None = None


def _leader_action(leader: SOLeader) -> dict[str, float]:
    action = leader.get_action()
    return {key: float(action[f"{key}.pos"]) for key in LEADER_KEYS}


def _align_right_j5_to_left(
    left_anchor: np.ndarray,
    right_anchor: np.ndarray,
    max_difference_deg: float,
) -> tuple[np.ndarray, float]:
    """Return a right anchor whose J5 matches left, with a safety bound."""
    difference_rad = float(left_anchor[4] - right_anchor[4])
    difference_deg = float(np.rad2deg(difference_rad))
    if abs(difference_deg) > max_difference_deg:
        raise RuntimeError(
            "Refusing right J5 startup alignment: "
            f"difference={difference_deg:.2f}deg exceeds "
            f"{max_difference_deg:.2f}deg"
        )
    aligned = right_anchor.copy()
    aligned[4] = left_anchor[4]
    return aligned, difference_deg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--left-profile",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "so101_piper_left_profile.json",
    )
    parser.add_argument(
        "--right-profile",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "so101_piper_right_profile.json",
    )
    parser.add_argument(
        "--left-port",
        default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14031917-if00",
    )
    parser.add_argument(
        "--right-port",
        default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14112595-if00",
    )
    parser.add_argument("--left-id", default="piper_left_so101")
    parser.add_argument("--right-id", default="piper_right_so101")
    parser.add_argument("--left-state-topic", default="/puppet/joint_left")
    parser.add_argument("--right-state-topic", default="/puppet/joint_right")
    parser.add_argument("--left-command-topic", default="/master/joint_left")
    parser.add_argument("--right-command-topic", default="/master/joint_right")
    parser.add_argument("--rate-hz", type=float, default=30.0)
    parser.add_argument("--max-arm-delta-rad", type=float, default=0.03)
    parser.add_argument("--max-j3-delta-rad", type=float, default=0.04)
    parser.add_argument("--max-j5-delta-rad", type=float, default=0.05)
    parser.add_argument("--max-j6-delta-rad", type=float, default=0.04)
    parser.add_argument("--max-gripper-delta-m", type=float, default=0.003)
    parser.add_argument("--joint-deadband-rad", type=float, default=0.002)
    parser.add_argument("--gripper-deadband-m", type=float, default=0.0005)
    parser.add_argument("--filter-min-cutoff-hz", type=float, default=3.0)
    parser.add_argument("--filter-beta", type=float, default=0.8)
    parser.add_argument("--filter-derivative-cutoff-hz", type=float, default=1.0)
    parser.add_argument("--max-right-j5-alignment-deg", type=float, default=15.0)
    args = parser.parse_args()

    positive = (
        args.rate_hz,
        args.max_arm_delta_rad,
        args.max_j3_delta_rad,
        args.max_j5_delta_rad,
        args.max_j6_delta_rad,
        args.max_gripper_delta_m,
        args.filter_min_cutoff_hz,
        args.filter_derivative_cutoff_hz,
        args.max_right_j5_alignment_deg,
    )
    if min(positive) <= 0:
        raise ValueError("rate, deltas, and filter cutoffs must be positive")
    if min(args.joint_deadband_rad, args.gripper_deadband_m, args.filter_beta) < 0:
        raise ValueError("deadbands and filter beta must be non-negative")

    profiles = {
        "left": json.loads(args.left_profile.read_text()),
        "right": json.loads(args.right_profile.read_text()),
    }
    for name, profile in profiles.items():
        if profile.get("side") != name or profile.get("locked_piper_indices") != [3]:
            raise ValueError(f"{name} profile has the wrong side or does not lock Piper J4")

    rospy.init_node("so101_piper_dual_teleop", anonymous=True)
    _, _, system_state = rospy.get_master().getSystemState()
    command_topics = (args.left_command_topic, args.right_command_topic)
    conflicts = {
        topic: nodes for topic, nodes in system_state[0] if topic in command_topics
    }
    if conflicts:
        raise RuntimeError(f"Refusing dual teleop; command publishers already exist: {conflicts}")

    sides = {
        "left": _Side(
            "left",
            profiles["left"],
            args.left_port,
            args.left_id,
            args.left_state_topic,
            args.left_command_topic,
            SOLeader(SO101LeaderConfig(port=args.left_port, id=args.left_id)),
            rospy.Publisher(args.left_command_topic, JointState, queue_size=1, tcp_nodelay=True),
        ),
        "right": _Side(
            "right",
            profiles["right"],
            args.right_port,
            args.right_id,
            args.right_state_topic,
            args.right_command_topic,
            SOLeader(SO101LeaderConfig(port=args.right_port, id=args.right_id)),
            rospy.Publisher(args.right_command_topic, JointState, queue_size=1, tcp_nodelay=True),
        ),
    }
    state_lock = threading.Lock()

    def state_callback(message: JointState, side: _Side) -> None:
        values = np.asarray(message.position, dtype=np.float32)
        if values.shape == (7,) and np.isfinite(values).all():
            with state_lock:
                side.state = values

    _subscribers = [
        rospy.Subscriber(
            side.state_topic,
            JointState,
            state_callback,
            callback_args=side,
            queue_size=1,
            tcp_nodelay=True,
        )
        for side in sides.values()
    ]

    arm_indices = np.asarray([0, 1, 2, 4, 5])
    limits = np.asarray(
        [
            args.max_arm_delta_rad,
            args.max_arm_delta_rad,
            args.max_j3_delta_rad,
            0.0,
            args.max_j5_delta_rad,
            args.max_j6_delta_rad,
        ],
        dtype=np.float32,
    )
    try:
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            with state_lock:
                ready = all(side.state is not None for side in sides.values())
            if ready:
                break
            rospy.sleep(0.02)
        if not ready:
            missing = [side.state_topic for side in sides.values() if side.state is None]
            raise RuntimeError(f"Timed out waiting for follower states: {missing}")

        for side in sides.values():
            side.leader.connect(calibrate=False)
            if not side.leader.is_calibrated:
                raise RuntimeError(f"{side.name} SO-101 calibration is missing or mismatched")
            side.leader.bus.disable_torque(num_retry=3)

        timestamp = time.monotonic()
        for side in sides.values():
            leader_anchor = _leader_action(side.leader)
            with state_lock:
                measured = side.state.copy()
            side.profile_anchor = _profile_action(side.profile, leader_anchor)
            side.piper_anchor = measured.copy()
            side.command = measured.copy()
            side.stable_desired = measured.copy()
            initial_desired = _anchored_action(
                side.piper_anchor, side.profile_anchor, side.profile_anchor
            )
            side.desired_filter = _OneEuroFilter(
                initial_desired,
                timestamp,
                args.filter_min_cutoff_hz,
                args.filter_beta,
                args.filter_derivative_cutoff_hz,
            )

        right = sides["right"]
        right.piper_anchor, right_j5_offset_deg = _align_right_j5_to_left(
            sides["left"].piper_anchor,
            right.piper_anchor,
            args.max_right_j5_alignment_deg,
        )
        right_initial_desired = _anchored_action(
            right.piper_anchor,
            right.profile_anchor,
            right.profile_anchor,
        )
        right.desired_filter = _OneEuroFilter(
            right_initial_desired,
            timestamp,
            args.filter_min_cutoff_hz,
            args.filter_beta,
            args.filter_derivative_cutoff_hz,
        )
        rospy.logwarn(
            "Right Piper J5 startup target aligned to left J5: offset=%+.2fdeg",
            right_j5_offset_deg,
        )

        rospy.logwarn(
            "Dual SO-101 teleop active at %.1f Hz; Ctrl+C holds both Pipers. "
            "Both Piper J4 joints remain locked.",
            args.rate_hz,
        )
        rate = rospy.Rate(args.rate_hz)
        while not rospy.is_shutdown():
            timestamp = time.monotonic()
            messages: list[tuple[_Side, JointState]] = []
            for side in sides.values():
                with state_lock:
                    measured = None if side.state is None else side.state.copy()
                if measured is None:
                    raise RuntimeError(f"{side.name} Piper state became unavailable")
                profile_current = _profile_action(
                    side.profile, _leader_action(side.leader)
                )
                raw_desired = _anchored_action(
                    side.piper_anchor, side.profile_anchor, profile_current
                )
                desired = side.desired_filter.update(raw_desired, timestamp)
                changed = (
                    np.abs(desired[arm_indices] - side.stable_desired[arm_indices])
                    >= args.joint_deadband_rad
                )
                side.stable_desired[arm_indices[changed]] = desired[arm_indices[changed]]
                if abs(desired[6] - side.stable_desired[6]) >= args.gripper_deadband_m:
                    side.stable_desired[6] = desired[6]
                side.command[arm_indices] += np.clip(
                    side.stable_desired[arm_indices] - side.command[arm_indices],
                    -limits[arm_indices],
                    limits[arm_indices],
                )
                side.command[6] += float(
                    np.clip(
                        side.stable_desired[6] - side.command[6],
                        -args.max_gripper_delta_m,
                        args.max_gripper_delta_m,
                    )
                )
                side.command[3] = measured[3]
                message = JointState()
                message.position = side.command.tolist()
                messages.append((side, message))
            stamp = rospy.Time.now()
            for side, message in messages:
                message.header.stamp = stamp
                side.publisher.publish(message)
            rate.sleep()
    finally:
        with state_lock:
            holds = {
                name: None if side.state is None else side.state.copy()
                for name, side in sides.items()
            }
        for _ in range(3):
            stamp = rospy.Time.now()
            for name, side in sides.items():
                if holds[name] is None:
                    continue
                message = JointState()
                message.header.stamp = stamp
                message.position = holds[name].tolist()
                side.publisher.publish(message)
            time.sleep(0.05)
        for side in sides.values():
            if not side.leader.is_connected:
                continue
            try:
                side.leader.bus.disable_torque(num_retry=3)
                side.leader.disconnect()
            except Exception as error:  # Hardware cleanup must not mask the first failure.
                rospy.logwarn("Failed to cleanly disconnect %s SO-101: %s", side.name, error)


if __name__ == "__main__":
    main()
