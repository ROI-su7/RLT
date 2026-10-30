"""Read-only single-side SO-101/Piper paired profile capture."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", choices=("left", "right"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument(
        "--port",
        help="SO-101 port; defaults to the calibrated port for --side.",
    )
    parser.add_argument("--id", help="SO-101 calibration ID.")
    parser.add_argument("--topic", help="Piper follower joint-state topic.")
    args = parser.parse_args()
    if args.samples < 2 or args.timeout_s <= 0:
        raise ValueError("--samples must be at least 2 and --timeout-s positive")

    defaults = {
        "left": {
            "port": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14031917-if00",
            "id": "piper_left_so101",
            "topic": "/puppet/joint_left",
        },
        "right": {
            "port": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5B14112595-if00",
            "id": "piper_right_so101",
            "topic": "/puppet/joint_right",
        },
    }[args.side]
    port = args.port or defaults["port"]
    identifier = args.id or defaults["id"]
    topic = args.topic or defaults["topic"]

    rospy.init_node(f"so101_piper_{args.side}_profile_capture", anonymous=True)
    leader = SOLeader(SO101LeaderConfig(port=port, id=identifier))
    samples: list[dict] = []
    try:
        leader.connect(calibrate=False)
        if not leader.is_calibrated:
            raise RuntimeError("SO-101 calibration is missing or mismatched")
        print(
            f"Read-only {args.side} capture ready. Move Piper {args.side} follower "
            "and SO-101 leader to matching safe poses; no command is sent."
        )
        for index in range(args.samples):
            input(f"Sample {index + 1}/{args.samples}: press Enter when aligned. ")
            message = rospy.wait_for_message(topic, JointState, args.timeout_s)
            piper = [float(value) for value in message.position]
            if len(piper) != 7 or not np.isfinite(piper).all():
                raise ValueError(f"{topic} must provide seven finite values")
            action = leader.get_action()
            samples.append(
                {
                    "timestamp_monotonic_s": time.monotonic(),
                    args.side: {
                        "so101_deg": {
                            key: float(action[f"{key}.pos"])
                            for key in LEADER_KEYS
                        },
                        "piper_rad_gripper_m": piper,
                    },
                }
            )
            print("Captured.")
    finally:
        if leader.is_connected:
            leader.disconnect()

    values = np.asarray(
        [sample[args.side]["piper_rad_gripper_m"] for sample in samples],
        dtype=np.float32,
    )
    payload = {
        "format_version": 1,
        "side": args.side,
        "notes": {
            "mapped_axes": "SO five arm axes map to Piper J1,J2,J3,J5,J6",
            "locked_axes": "Piper J4 is intentionally excluded",
        },
        "samples": samples,
        "piper_spans": {
            "mapped_arm_deg": np.rad2deg(
                np.ptp(values[:, [0, 1, 2, 4, 5]], axis=0)
            ).tolist(),
            "gripper_m": float(np.ptp(values[:, 6])),
        },
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {len(samples)} {args.side} samples to {args.output}")


if __name__ == "__main__":
    main()
