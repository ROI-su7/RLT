"""Capture read-only paired SO-101 and Piper joint samples for calibration."""

from __future__ import annotations

import argparse
import json
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
MAPPED_AXES = {
    "j1": 0,
    "j2": 1,
    "j3": 2,
    "j5": 4,
    "j6": 5,
    "gripper": 6,
}


def _piper_state(message: JointState, topic: str) -> list[float]:
    values = [float(value) for value in message.position]
    if len(values) != 7 or not np.isfinite(values).all():
        raise ValueError(f"{topic} must provide seven finite joint values")
    return values


def _leader_state(leader: SOLeader) -> dict[str, float]:
    action = leader.get_action()
    return {key: float(action[f"{key}.pos"]) for key in LEADER_KEYS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append new samples to an existing capture file.",
    )
    parser.add_argument(
        "--expected-axis",
        choices=tuple(MAPPED_AXES),
        help=(
            "Accept only a clean single-axis sample relative to sample 1. "
            "Use with --append after returning Piper to the baseline pose."
        ),
    )
    parser.add_argument("--min-target-arm-span-deg", type=float, default=3.0)
    parser.add_argument("--max-nontarget-arm-span-deg", type=float, default=1.5)
    parser.add_argument("--min-target-gripper-span-m", type=float, default=0.005)
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
    parser.add_argument("--left-topic", default="/puppet/joint_left")
    parser.add_argument("--right-topic", default="/puppet/joint_right")
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument(
        "--min-arm-span-deg",
        type=float,
        default=2.0,
        help="Minimum required Piper span for each mapped arm joint.",
    )
    parser.add_argument(
        "--min-gripper-span-m",
        type=float,
        default=0.003,
        help="Minimum required Piper gripper span.",
    )
    args = parser.parse_args()
    if args.samples < 1 or args.timeout_s <= 0:
        raise ValueError("--samples and --timeout-s must be positive")
    if not args.append and args.samples < 3:
        raise ValueError("--samples must be at least 3 unless --append is used")

    existing_samples: list[dict] = []
    if args.append:
        if not args.output.is_file():
            raise FileNotFoundError(f"Cannot append; missing {args.output}")
        existing = json.loads(args.output.read_text())
        if existing.get("format_version") != 1:
            raise ValueError("Unsupported existing capture format")
        existing_samples = list(existing.get("samples", []))
    if args.expected_axis and not existing_samples:
        raise ValueError("--expected-axis requires --append with a baseline sample")

    rospy.init_node("so101_piper_profile_capture", anonymous=True)
    leaders = (
        SOLeader(SO101LeaderConfig(port=args.left_port, id=args.left_id)),
        SOLeader(SO101LeaderConfig(port=args.right_port, id=args.right_id)),
    )
    samples: list[dict] = list(existing_samples)
    try:
        for leader in leaders:
            # Existing calibration only; never invoke the interactive calibration path.
            leader.connect(calibrate=False)
            if not leader.is_calibrated:
                raise RuntimeError("SO-101 calibration is missing or mismatched")
        print(
            "Read-only capture ready. Place both Piper followers and SO-101 leaders "
            "in matching safe poses. This program sends no Piper command."
        )
        captured_count = 0
        while captured_count < args.samples:
            input(
                f"New sample {captured_count + 1}/{args.samples}: press Enter when aligned. "
            )
            left_piper = _piper_state(
                rospy.wait_for_message(args.left_topic, JointState, args.timeout_s),
                args.left_topic,
            )
            right_piper = _piper_state(
                rospy.wait_for_message(args.right_topic, JointState, args.timeout_s),
                args.right_topic,
            )
            candidate = {
                "timestamp_monotonic_s": time.monotonic(),
                "left": {
                    "so101_deg": _leader_state(leaders[0]),
                    "piper_rad_gripper_m": left_piper,
                },
                "right": {
                    "so101_deg": _leader_state(leaders[1]),
                    "piper_rad_gripper_m": right_piper,
                },
            }
            if args.expected_axis:
                target_index = MAPPED_AXES[args.expected_axis]
                invalid_sides: list[str] = []
                for side in ("left", "right"):
                    baseline = np.asarray(
                        existing_samples[0][side]["piper_rad_gripper_m"],
                        dtype=np.float32,
                    )
                    current = np.asarray(
                        candidate[side]["piper_rad_gripper_m"],
                        dtype=np.float32,
                    )
                    delta = current - baseline
                    if args.expected_axis == "gripper":
                        target_ok = abs(delta[6]) >= args.min_target_gripper_span_m
                        arm_ok = np.all(
                            np.abs(np.rad2deg(delta[[0, 1, 2, 4, 5]]))
                            <= args.max_nontarget_arm_span_deg
                        )
                    else:
                        target_ok = abs(np.rad2deg(delta[target_index])) >= args.min_target_arm_span_deg
                        non_target_indices = [
                            value
                            for value in (0, 1, 2, 4, 5)
                            if value != target_index
                        ]
                        arm_ok = np.all(
                            np.abs(np.rad2deg(delta[non_target_indices]))
                            <= args.max_nontarget_arm_span_deg
                        ) and abs(delta[6]) <= args.min_target_gripper_span_m
                    if not (target_ok and arm_ok):
                        invalid_sides.append(side)
                if invalid_sides:
                    print(
                        "Rejected: the expected axis was not isolated for "
                        f"{', '.join(invalid_sides)}. Return to baseline, adjust "
                        "only the requested axis, then retry this sample."
                    )
                    continue
            samples.append(candidate)
            captured_count += 1
            print("Captured.")
    finally:
        for leader in leaders:
            if leader.is_connected:
                leader.disconnect()

    mapped_indices = np.asarray([0, 1, 2, 4, 5])
    piper_spans: dict[str, list[float]] = {}
    invalid_sides: list[str] = []
    for side in ("left", "right"):
        piper_values = np.asarray(
            [sample[side]["piper_rad_gripper_m"] for sample in samples],
            dtype=np.float32,
        )
        arm_spans_deg = np.rad2deg(np.ptp(piper_values[:, mapped_indices], axis=0))
        gripper_span_m = float(np.ptp(piper_values[:, 6]))
        piper_spans[side] = [float(value) for value in arm_spans_deg] + [
            gripper_span_m
        ]
        if (
            np.any(arm_spans_deg < args.min_arm_span_deg)
            or gripper_span_m < args.min_gripper_span_m
        ):
            invalid_sides.append(side)

    payload = {
        "format_version": 1,
        "notes": {
            "mapped_axes": "SO five arm axes map to Piper J1,J2,J3,J5,J6",
            "locked_axes": "Piper J4 is intentionally excluded",
        },
        "samples": samples,
        "piper_spans": piper_spans,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(samples)} paired samples to {args.output}")
    if invalid_sides:
        raise RuntimeError(
            "Insufficient Piper motion for "
            f"{', '.join(invalid_sides)}. Recollect with every mapped joint and "
            "the gripper moved through the requested span."
        )


if __name__ == "__main__":
    main()
