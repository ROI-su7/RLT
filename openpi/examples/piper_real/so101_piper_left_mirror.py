"""Low-speed left SO-101 follower mirror for profile validation.

This tool reads only ``/puppet/joint_left`` and drives only the left SO-101
leader. It never publishes a Piper command and is independent from pi0.5.
"""

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


def _write_targets(leader: SOLeader, targets: dict[str, float]) -> None:
    """Use acknowledged writes for validation; do not silently drop leader goals."""
    for key, value in targets.items():
        leader.bus.write("Goal_Position", key, value, num_retry=3)


def _wrap_near(value: float, reference: float) -> float:
    return reference + ((value - reference + 180.0) % 360.0 - 180.0)


def _inverse_piecewise(
    piper_value: float,
    leader_knots: np.ndarray,
    piper_knots: np.ndarray,
    current_leader: float,
) -> float:
    """Invert a possibly non-monotonic piecewise profile near current leader pose."""
    candidates: list[float] = []
    for start in range(len(leader_knots) - 1):
        x0, x1 = leader_knots[start : start + 2]
        y0, y1 = piper_knots[start : start + 2]
        lower, upper = sorted((y0, y1))
        if lower <= piper_value <= upper and not np.isclose(y0, y1):
            ratio = (piper_value - y0) / (y1 - y0)
            candidates.append(float(x0 + ratio * (x1 - x0)))
    if not candidates:
        endpoint = int(np.argmin(np.abs(piper_knots - piper_value)))
        return float(leader_knots[endpoint])
    return min(candidates, key=lambda value: abs(_wrap_near(value, current_leader) - current_leader))


def _leader_targets(profile: dict, piper: np.ndarray, current: dict[str, float]) -> dict[str, float]:
    """Invert profile mappings into calibrated SO-101 degree/unit targets."""
    targets: dict[str, float] = {}
    for mapping in profile["mappings"]:
        key = mapping["leader_key"]
        piper_value = float(piper[mapping["piper_index"]])
        if mapping.get("method") == "piecewise_linear_clamped":
            targets[key] = _inverse_piecewise(
                piper_value,
                np.asarray(mapping["leader_knots"], dtype=float),
                np.asarray(mapping["piper_knots"], dtype=float),
                current[key],
            )
        elif mapping["name"] == "gripper":
            targets[key] = mapping["leader_reference"] + (
                piper_value - mapping["piper_reference"]
            ) / mapping["gain"]
        else:
            target = mapping["leader_reference"] + np.rad2deg(
                (piper_value - mapping["piper_reference"]) / mapping["gain"]
            )
            target = _wrap_near(float(target), current[key])
            observed_range = mapping.get("leader_observed_range")
            if observed_range is not None:
                target = float(np.clip(target, observed_range[0], observed_range[1]))
            targets[key] = target
    return targets


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
    parser.add_argument("--topic", default="/puppet/joint_left")
    parser.add_argument("--rate-hz", type=float, default=40.0)
    parser.add_argument("--max-speed-deg-s", type=float, default=5.0)
    parser.add_argument("--max-gripper-units-s", type=float, default=10.0)
    parser.add_argument(
        "--target-smoothing-alpha",
        type=float,
        default=0.25,
        help="EMA coefficient for inverse-mapped leader targets in (0, 1].",
    )
    parser.add_argument("--max-tracking-error-deg", type=float, default=5.0)
    parser.add_argument("--max-tracking-error-gripper", type=float, default=10.0)
    args = parser.parse_args()
    if min(
        args.rate_hz,
        args.max_speed_deg_s,
        args.max_gripper_units_s,
        args.max_tracking_error_deg,
        args.max_tracking_error_gripper,
    ) <= 0:
        raise ValueError("rate and speed limits must be positive")
    if not 0 < args.target_smoothing_alpha <= 1:
        raise ValueError("--target-smoothing-alpha must be in (0, 1]")
    profile = json.loads(args.profile.read_text())
    if profile.get("side") != "left" or profile.get("locked_piper_indices") != [3]:
        raise ValueError("--profile must be a left profile with locked Piper J4")

    rospy.init_node("so101_piper_left_mirror", anonymous=True)
    leader = SOLeader(SO101LeaderConfig(port=args.port, id=args.id))
    piper_lock = threading.Lock()
    latest_piper: np.ndarray | None = None
    latest_piper_at = 0.0

    def piper_callback(message: JointState) -> None:
        nonlocal latest_piper, latest_piper_at
        values = np.asarray(message.position, dtype=float)
        if values.shape == (7,) and np.isfinite(values).all():
            with piper_lock:
                latest_piper = values
                latest_piper_at = time.monotonic()

    rospy.Subscriber(args.topic, JointState, piper_callback, queue_size=1, tcp_nodelay=True)
    target_pub = rospy.Publisher(
        "/so101_piper_left_mirror/leader_target",
        JointState,
        queue_size=1,
        tcp_nodelay=True,
    )
    state_pub = rospy.Publisher(
        "/so101_piper_left_mirror/leader_state",
        JointState,
        queue_size=1,
        tcp_nodelay=True,
    )
    try:
        leader.connect(calibrate=False)
        if not leader.is_calibrated:
            raise RuntimeError("SO-101 calibration is missing or mismatched")
        current = {key: float(leader.get_action()[f"{key}.pos"]) for key in LEADER_KEYS}
        _write_targets(leader, current)
        leader.bus.enable_torque(num_retry=3)
        torque = leader.bus.sync_read("Torque_Enable", normalize=False, num_retry=3)
        disabled = [key for key, value in torque.items() if int(value) != 1]
        if disabled:
            raise RuntimeError(f"SO-101 torque did not enable: {disabled}")
        rospy.logwarn(
            "Left follower mirror active at %.1f Hz; Ctrl+C releases SO-101 torque. "
            "No Piper command is ever published.",
            args.rate_hz,
        )
        rate = rospy.Rate(args.rate_hz)
        last_status_log = 0.0
        command_target = current.copy()
        filtered_desired = current.copy()
        while not rospy.is_shutdown():
            with piper_lock:
                piper = None if latest_piper is None else latest_piper.copy()
                piper_age = time.monotonic() - latest_piper_at
            if piper is None or piper_age > 1.0:
                if time.monotonic() - last_status_log > 1.0:
                    rospy.logwarn("No fresh Piper state on %s; SO-101 target held", args.topic)
                    last_status_log = time.monotonic()
                rate.sleep()
                continue
            try:
                leader_action = leader.get_action()
            except ConnectionError as exc:
                rospy.logerr("SO-101 feedback failed; mirror stopping: %s", exc)
                break
            measured = {key: float(leader_action[f"{key}.pos"]) for key in LEADER_KEYS}
            desired = _leader_targets(profile, piper, measured)
            for key in LEADER_KEYS:
                if key == "gripper":
                    filtered_desired[key] += args.target_smoothing_alpha * (
                        desired[key] - filtered_desired[key]
                    )
                else:
                    delta = _wrap_near(desired[key], filtered_desired[key]) - filtered_desired[key]
                    filtered_desired[key] = _wrap_near(
                        filtered_desired[key] + args.target_smoothing_alpha * delta,
                        filtered_desired[key],
                    )
            stamp = rospy.Time.now()
            target_message = JointState()
            target_message.header.stamp = stamp
            target_message.name = list(LEADER_KEYS)
            target_message.position = [filtered_desired[key] for key in LEADER_KEYS]
            target_pub.publish(target_message)
            state_message = JointState()
            state_message.header.stamp = stamp
            state_message.name = list(LEADER_KEYS)
            state_message.position = [measured[key] for key in LEADER_KEYS]
            state_pub.publish(state_message)
            next_target: dict[str, float] = {}
            arm_step = args.max_speed_deg_s / args.rate_hz
            gripper_step = args.max_gripper_units_s / args.rate_hz
            for key in LEADER_KEYS:
                tracking = command_target[key] - measured[key]
                if key != "gripper":
                    tracking = _wrap_near(command_target[key], measured[key]) - measured[key]
                    if abs(tracking) > args.max_tracking_error_deg:
                        next_target[key] = command_target[key]
                        continue
                    delta = _wrap_near(filtered_desired[key], command_target[key]) - command_target[key]
                    delta = float(np.clip(delta, -arm_step, arm_step))
                    next_target[key] = _wrap_near(command_target[key] + delta, command_target[key])
                else:
                    if abs(tracking) > args.max_tracking_error_gripper:
                        next_target[key] = command_target[key]
                        continue
                    delta = float(np.clip(filtered_desired[key] - command_target[key], -gripper_step, gripper_step))
                    next_target[key] = command_target[key] + delta
            try:
                # Continuous follower output benefits from simultaneous bus
                # writes; startup still uses acknowledged per-motor writes.
                leader.bus.sync_write("Goal_Position", next_target)
            except ConnectionError as exc:
                rospy.logerr("SO-101 command failed; mirror stopping: %s", exc)
                break
            command_target = next_target
            if time.monotonic() - last_status_log > 1.0:
                max_tracking = max(abs(command_target[key] - measured[key]) for key in LEADER_KEYS)
                rospy.loginfo("Left mirror max leader command tracking delta: %.2f", max_tracking)
                last_status_log = time.monotonic()
            rate.sleep()
    finally:
        if leader.is_connected:
            try:
                leader.bus.disable_torque(num_retry=3)
            finally:
                leader.disconnect()


if __name__ == "__main__":
    main()
