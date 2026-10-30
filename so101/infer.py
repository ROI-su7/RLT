#!/usr/bin/env python3
"""Run a pi0.5 policy (openpi websocket server) on the dual SO101 follower arms.

    # terminal 1 (GPU host):
    cd openpi && .venv/bin/python scripts/serve_policy.py --port 6006 policy:checkpoint \
        --policy.config pi05_so101_dual_lora --policy.dir <checkpoint dir>
    # terminal 2 (robot host):
    python so101/infer.py --host <gpu host> --port 6006 --task "..."

The leader arms are not used. Ctrl-C stops and leaves the follower holding its last position.
--dummy replaces the server by a policy that holds the current pose (hardware/pipeline check, with --mock).
"""
import argparse
import pathlib
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from robot_config import ROOT, make_robot_config  # noqa: E402

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")


class HoldPolicy:
    def infer(self, obs):
        return {"actions": np.repeat(np.asarray(obs["state"], np.float32)[None], 50, axis=0)}


def to_policy_obs(raw: dict, task: str) -> dict:
    images = {}
    for name in CAMERAS:
        key = f"observation.images.{name}"
        if key in raw:
            images[name] = np.ascontiguousarray(np.asarray(raw[key].numpy(), np.uint8).transpose(2, 0, 1))
    return {"images": images, "state": raw["observation.state"].numpy().astype(np.float32), "prompt": task}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "dual_arm.yaml"))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6006)
    ap.add_argument("--task", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--execute", type=int, default=25, help="actions executed from each chunk before re-planning")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--max-relative-target", type=float, default=10.0, help="per-step joint cap (degrees)")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--dummy", action="store_true")
    args = ap.parse_args()

    from lerobot.common.robot_devices.robots.utils import make_robot_from_config

    robot = make_robot_from_config(make_robot_config(args.config, mock=args.mock, max_relative_target=args.max_relative_target))
    if args.dummy:
        policy = HoldPolicy()
    else:
        from openpi_client import websocket_client_policy
        policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    robot.connect()
    step, period = 0, 1.0 / args.fps
    try:
        while step < args.max_steps:
            chunk = np.asarray(policy.infer(to_policy_obs(robot.capture_observation(), args.task))["actions"])
            for a in chunk[: args.execute]:
                t0 = time.perf_counter()
                robot.send_action(torch.from_numpy(a.astype(np.float32)))
                step += 1
                time.sleep(max(0.0, period - (time.perf_counter() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()
    print(f"ran {step} steps")


if __name__ == "__main__":
    main()
