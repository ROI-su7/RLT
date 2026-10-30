#!/usr/bin/env python3
"""Dual-SO101 data collection on top of LeRobot: calibrate / teleoperate / record / replay.

    python so101/cli.py calibrate              # once per arm, on the machine that owns the arms
    python so101/cli.py teleoperate
    python so101/cli.py record --task "..." --episodes 50 --root ~/rlt_data/lerobot_data/so101_dual
Add --mock to run everything without hardware (used by the tests).
Recording keys (LeRobot): Right arrow = finish episode, Left arrow = re-record, Esc = stop.
"""
import argparse
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from robot_config import ROOT, make_robot_config  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["calibrate", "teleoperate", "record", "replay"])
    ap.add_argument("--config", default=str(ROOT / "dual_arm.yaml"))
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--task", default="")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--episode-time", type=float, default=60)
    ap.add_argument("--reset-time", type=float, default=10)
    ap.add_argument("--warmup-time", type=float, default=3)
    ap.add_argument("--root", default=os.path.expanduser("~/rlt_data/lerobot_data/so101_dual"))
    ap.add_argument("--repo-id", default="local/so101_dual")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--episode", type=int, default=0, help="replay: episode index")
    args = ap.parse_args()

    from lerobot.common.robot_devices.control_configs import (
        CalibrateControlConfig, RecordControlConfig, ReplayControlConfig, TeleoperateControlConfig)
    from lerobot.common.robot_devices.robots.utils import make_robot_from_config
    from lerobot.scripts import control_robot as cr

    robot = make_robot_from_config(make_robot_config(args.config, mock=args.mock))
    if args.command == "calibrate":
        cr.calibrate(robot, CalibrateControlConfig(arms=robot.available_arms))
    elif args.command == "teleoperate":
        cr.teleoperate(robot, TeleoperateControlConfig(fps=args.fps, display_data=False))
    elif args.command == "record":
        if not args.task:
            sys.exit("--task is required")
        cr.record(robot, RecordControlConfig(
            repo_id=args.repo_id, single_task=args.task, root=args.root, fps=args.fps,
            warmup_time_s=args.warmup_time, episode_time_s=args.episode_time, reset_time_s=args.reset_time,
            num_episodes=args.episodes, video=True, push_to_hub=False, play_sounds=False, resume=args.resume))
    else:
        cr.replay(robot, ReplayControlConfig(
            repo_id=args.repo_id, root=args.root, episode=args.episode, fps=args.fps, play_sounds=False))


if __name__ == "__main__":
    main()
