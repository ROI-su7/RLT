"""Low-speed left SO-101 to Piper teleoperation using a calibrated profile."""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path

import numpy as np
import rospy
from sensor_msgs.msg import JointState

from lerobot.teleoperators.so_leader import SO101LeaderConfig, SOLeader


LEADER_KEYS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


class _OneEuroFilter:
    """Velocity-adaptive low-pass filter for responsive, quiet teleoperation."""

    def __init__(
        self,
        initial: np.ndarray,
        timestamp: float,
        min_cutoff_hz: float,
        beta: float,
        derivative_cutoff_hz: float,
    ) -> None:
        self._value = initial.astype(np.float32, copy=True)
        self._raw = initial.astype(np.float32, copy=True)
        self._derivative = np.zeros_like(initial, dtype=np.float32)
        self._timestamp = timestamp
        self._min_cutoff_hz = min_cutoff_hz
        self._beta = beta
        self._derivative_cutoff_hz = derivative_cutoff_hz

    @staticmethod
    def _alpha(cutoff_hz: np.ndarray | float, dt: float) -> np.ndarray:
        cutoff = np.asarray(cutoff_hz, dtype=np.float32)
        return 1.0 / (1.0 + 1.0 / (2.0 * np.pi * cutoff * dt))

    def update(self, raw: np.ndarray, timestamp: float) -> np.ndarray:
        dt = max(timestamp - self._timestamp, 1e-4)
        raw_derivative = (raw - self._raw) / dt
        derivative_alpha = self._alpha(self._derivative_cutoff_hz, dt)
        self._derivative += derivative_alpha * (raw_derivative - self._derivative)
        cutoff = self._min_cutoff_hz + self._beta * np.abs(self._derivative)
        value_alpha = self._alpha(cutoff, dt)
        self._value += value_alpha * (raw - self._value)
        self._raw = raw.astype(np.float32, copy=True)
        self._timestamp = timestamp
        return self._value.copy()


def _wrap_near(value: float, reference: float) -> float:
    return reference + ((value - reference + 180.0) % 360.0 - 180.0)


def _profile_action(profile: dict, leader: dict[str, float]) -> np.ndarray:
    """Map a leader reading into calibrated Piper coordinates (J4 is excluded)."""
    action = np.zeros(7, dtype=np.float32)
    for mapping in profile["mappings"]:
        key = mapping["leader_key"]
        if mapping.get("method") == "piecewise_linear_clamped":
            knots_x = np.asarray(mapping["leader_knots"], dtype=float)
            ref = float(knots_x[0])
            position = _wrap_near(leader[key], ref)
            action[mapping["piper_index"]] = np.interp(
                position,
                knots_x,
                np.asarray(mapping["piper_knots"], dtype=float),
            )
        elif mapping.get("method") == "endpoint_linear_clamped":
            leader_min, leader_max = mapping["leader_range"]
            piper_min, piper_max = mapping["piper_range"]
            position = float(np.clip(leader[key], leader_min, leader_max))
            action[mapping["piper_index"]] = float(
                np.interp(position, [leader_min, leader_max], [piper_min, piper_max])
            )
        elif mapping["name"] == "gripper":
            action[mapping["piper_index"]] = mapping["piper_reference"] + mapping["gain"] * (
                leader[key] - mapping["leader_reference"]
            )
        else:
            position = _wrap_near(leader[key], mapping["leader_reference"])
            delta_deg = position - mapping["leader_reference"]
            action[mapping["piper_index"]] = mapping["piper_reference"] + mapping["gain"] * np.deg2rad(delta_deg)
    return action


def _anchored_action(
    piper_anchor: np.ndarray,
    profile_anchor: np.ndarray,
    profile_current: np.ndarray,
) -> np.ndarray:
    """Apply calibrated leader motion relative to the pose at teleop start.

    A SO-101 and Piper do not need identical physical poses at startup.  This
    avoids an initial jump while retaining the calibrated sign and scale of
    every movable joint.  Piper J4 intentionally remains at its measured pose.
    """
    action = piper_anchor + (profile_current - profile_anchor)
    # The gripper is an absolute aperture, unlike the anchored arm joints.
    # Keeping it relative would consume part of its usable 0--80 mm stroke.
    action[6] = profile_current[6]
    action[3] = piper_anchor[3]
    return action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "so101_piper_left_profile.json",
    )
    parser.add_argument(
        "--port",
        default="/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14031917-if00",
    )
    parser.add_argument("--id", default="piper_left_so101")
    parser.add_argument("--state-topic", default="/puppet/joint_left")
    parser.add_argument("--command-topic", default="/master/joint_left")
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
    parser.add_argument("--alignment-tolerance-deg", type=float, default=10.0)
    args = parser.parse_args()
    if min(
        args.rate_hz,
        args.max_arm_delta_rad,
        args.max_j3_delta_rad,
        args.max_j5_delta_rad,
        args.max_j6_delta_rad,
        args.max_gripper_delta_m,
        args.filter_min_cutoff_hz,
        args.filter_derivative_cutoff_hz,
        args.alignment_tolerance_deg,
    ) <= 0:
        raise ValueError("rate, deltas, and alignment tolerance must be positive")
    if min(args.joint_deadband_rad, args.gripper_deadband_m, args.filter_beta) < 0:
        raise ValueError("deadbands and filter beta must be non-negative")
    profile = json.loads(args.profile.read_text())
    if profile.get("side") != "left" or profile.get("locked_piper_indices") != [3]:
        raise ValueError("--profile must be a left profile with locked Piper J4")

    rospy.init_node("so101_piper_left_teleop", anonymous=True)
    _, _, system_state = rospy.get_master().getSystemState()
    existing_publishers = [
        nodes for topic, nodes in system_state[0] if topic == args.command_topic
    ]
    if existing_publishers:
        raise RuntimeError(
            f"Refusing teleop: {args.command_topic} already has publishers "
            f"{existing_publishers}"
        )
    state_lock = threading.Lock()
    latest_state: np.ndarray | None = None

    def state_callback(message: JointState) -> None:
        nonlocal latest_state
        values = np.asarray(message.position, dtype=np.float32)
        if values.shape == (7,) and np.isfinite(values).all():
            with state_lock:
                latest_state = values

    rospy.Subscriber(args.state_topic, JointState, state_callback, queue_size=1, tcp_nodelay=True)
    publisher = rospy.Publisher(args.command_topic, JointState, queue_size=1, tcp_nodelay=True)
    leader = SOLeader(SO101LeaderConfig(port=args.port, id=args.id))
    try:
        deadline = time.monotonic() + 5.0
        while latest_state is None and time.monotonic() < deadline:
            rospy.sleep(0.02)
        if latest_state is None:
            raise RuntimeError(f"Timed out waiting for {args.state_topic}")
        leader.connect(calibrate=False)
        if not leader.is_calibrated:
            raise RuntimeError("SO-101 calibration is missing or mismatched")
        leader.bus.disable_torque(num_retry=3)
        leader_action = {key: float(leader.get_action()[f"{key}.pos"]) for key in LEADER_KEYS}
        with state_lock:
            measured = latest_state.copy()
        profile_anchor = _profile_action(profile, leader_action)
        piper_anchor = measured.copy()
        command = measured.copy()
        stable_desired = measured.copy()
        initial_desired = _anchored_action(piper_anchor, profile_anchor, profile_anchor)
        desired_filter = _OneEuroFilter(
            initial_desired,
            time.monotonic(),
            args.filter_min_cutoff_hz,
            args.filter_beta,
            args.filter_derivative_cutoff_hz,
        )
        rospy.logwarn(
            "Left SO-101 teleop active with a relative startup anchor; Ctrl+C "
            "holds Piper and releases leader. Piper J4 remains locked."
        )
        rate = rospy.Rate(args.rate_hz)
        while not rospy.is_shutdown():
            with state_lock:
                measured = None if latest_state is None else latest_state.copy()
            if measured is None:
                raise RuntimeError("Piper state became unavailable")
            leader_action = {key: float(leader.get_action()[f"{key}.pos"]) for key in LEADER_KEYS}
            profile_current = _profile_action(profile, leader_action)
            raw_desired = _anchored_action(piper_anchor, profile_anchor, profile_current)
            desired = desired_filter.update(raw_desired, time.monotonic())
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
            arm_indices = np.asarray([0, 1, 2, 4, 5])
            arm_changed = (
                np.abs(desired[arm_indices] - stable_desired[arm_indices])
                >= args.joint_deadband_rad
            )
            stable_desired[arm_indices[arm_changed]] = desired[arm_indices[arm_changed]]
            if abs(desired[6] - stable_desired[6]) >= args.gripper_deadband_m:
                stable_desired[6] = desired[6]
            command[arm_indices] += np.clip(
                stable_desired[arm_indices] - command[arm_indices],
                -limits[arm_indices],
                limits[arm_indices],
            )
            command[6] += float(
                np.clip(
                    stable_desired[6] - command[6],
                    -args.max_gripper_delta_m,
                    args.max_gripper_delta_m,
                )
            )
            command[3] = measured[3]
            message = JointState()
            message.header.stamp = rospy.Time.now()
            message.position = command.tolist()
            publisher.publish(message)
            rate.sleep()
    finally:
        with state_lock:
            hold = None if latest_state is None else latest_state.copy()
        if hold is not None:
            message = JointState()
            message.header.stamp = rospy.Time.now()
            message.position = hold.tolist()
            for _ in range(3):
                publisher.publish(message)
                time.sleep(0.05)
        if leader.is_connected:
            try:
                leader.bus.disable_torque(num_retry=3)
            finally:
                leader.disconnect()


if __name__ == "__main__":
    main()
