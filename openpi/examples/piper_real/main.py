from __future__ import annotations

import dataclasses
import logging
import pathlib
import sys
import time

import numpy as np

from openpi_client import action_chunk_broker
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.runtime import runtime as _runtime
from openpi_client.runtime.agents import policy_agent as _policy_agent
import tyro

# Support running this file directly via `python examples/piper_real/main.py`.
if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from examples.piper_real import env as _env
from examples.piper_real import hold_aware_policy as _hold_aware_policy
from examples.piper_real import rtc_action_chunk_broker as _rtc_broker
from examples.piper_real import saver as _saver


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 6006

    action_horizon: int = 10
    rtc_enabled: bool = False
    rtc_prefix_len: int = 5
    rtc_boundary_blend_steps: int = 5
    num_episodes: int = 1
    max_episode_steps: int = 300
    max_hz: float = 15.0

    # Disabled by default. When enabled, the policy still runs at max_hz but
    # arm commands are linearly resampled to this ROS publish frequency.
    command_publish_hz: float = 0.0
    command_time_scale: float = 1.0
    action_smoothing_alpha: float = 1.0
    max_arm_velocity_rad_s: float = 0.0
    interpolate_gripper: bool = False

    ready_timeout_s: float = 10.0
    enable_on_reset: bool = True
    enable_pulses: int = 5
    enable_interval_s: float = 0.1

    # Optional safe return on normal exit, Ctrl+C, or an exception. This uses
    # the existing ROS command topics. With no explicit reset_home_action, it
    # returns to the first valid 14-D state captured at episode start. Provide
    # reset_home_action only for a separately validated fixture-safe pose.
    reset_on_exit: bool = False
    reset_home_action: str | None = None
    reset_duration_s: float = 5.0
    reset_timeout_s: float = 15.0
    reset_tolerance_rad: float = 0.05
    reset_publish_hz: float = 30.0

    record_video: bool = True
    video_out_path: str = "data/piper_real/videos"
    video_subsample: int = 1

    front_image_topic: str = "/camera_f/color/image_raw"
    left_image_topic: str = "/camera_l/color/image_raw"
    right_image_topic: str = "/camera_r/color/image_raw"
    left_state_topic: str = "/puppet/joint_left"
    right_state_topic: str = "/puppet/joint_right"
    left_command_topic: str = "/master/joint_left"
    right_command_topic: str = "/master/joint_right"
    enable_topic: str = "/enable_flag"
    hold_active_topic: str = "/teleop/hold_active"
    policy_hold_ack_topic: str = "/teleop/policy_hold_ack"


def main(args: Args) -> None:
    reset_home_action = None
    if args.reset_on_exit:
        if args.reset_home_action is not None:
            try:
                reset_home_action = np.fromstring(args.reset_home_action, sep=",", dtype=np.float32)
            except ValueError as exc:
                raise ValueError("--reset-home-action must contain only comma-separated numbers.") from exc
            if reset_home_action.shape != (14,):
                raise ValueError(
                    "--reset-home-action must contain 14 values: "
                    "six left joints, left gripper, six right joints, right gripper."
                )

    ws_client_policy = _websocket_client_policy.WebsocketClientPolicy(
        host=args.host,
        port=args.port,
    )
    metadata = ws_client_policy.get_server_metadata()
    logging.info("Server metadata: %s", metadata)

    subscribers = []
    if args.record_video:
        subscribers.append(
            _saver.VideoSaver(
                pathlib.Path(args.video_out_path),
                fps=round(args.max_hz) if args.max_hz > 0 else 15,
                subsample=args.video_subsample,
            )
        )

    # Stock OpenPI PI0.5 cannot consume a previous-action prefix.  When
    # enabled, this broker implements the execution-side part of RTC: it
    # requests the next chunk in the background while publishing the remaining
    # actions from the current chunk, avoiding a command gap at chunk rollover.
    policy = ws_client_policy
    if args.rtc_enabled:
        policy = _rtc_broker.RealtimeActionChunkBroker(
            ws_client_policy,
            action_horizon=args.action_horizon,
            prefix_len=args.rtc_prefix_len,
            control_hz=args.max_hz,
            boundary_blend_steps=args.rtc_boundary_blend_steps,
        )

    environment = _env.PiperRealEnvironment(
        topics=_env.PiperTopics(
            front_image=args.front_image_topic,
            left_image=args.left_image_topic,
            right_image=args.right_image_topic,
            left_state=args.left_state_topic,
            right_state=args.right_state_topic,
            left_command=args.left_command_topic,
            right_command=args.right_command_topic,
            enable=args.enable_topic,
            hold_active=args.hold_active_topic,
            policy_hold_ack=args.policy_hold_ack_topic,
        ),
        ready_timeout_s=args.ready_timeout_s,
        enable_on_reset=args.enable_on_reset,
        enable_pulses=args.enable_pulses,
        enable_interval_s=args.enable_interval_s,
        command_publish_hz=args.command_publish_hz,
        command_interpolation_duration_s=(args.command_time_scale / args.max_hz if args.max_hz > 0 else 0.0),
        action_smoothing_alpha=args.action_smoothing_alpha,
        max_arm_velocity_rad_s=args.max_arm_velocity_rad_s,
        interpolate_gripper=args.interpolate_gripper,
    )
    execution_policy = (
        policy
        if args.rtc_enabled
        else action_chunk_broker.ActionChunkBroker(
            policy=policy,
            action_horizon=args.action_horizon,
        )
    )
    execution_policy = _hold_aware_policy.HoldAwarePolicy(
        execution_policy, environment
    )
    runtime = _runtime.Runtime(
        environment=environment,
        agent=_policy_agent.PolicyAgent(
            policy=execution_policy
        ),
        subscribers=subscribers,
        max_hz=args.max_hz,
        num_episodes=args.num_episodes,
        max_episode_steps=args.max_episode_steps,
    )

    should_return_home = False
    try:
        runtime.run()
    except KeyboardInterrupt:
        environment.hold_current_position()
        if args.reset_on_exit:
            logging.warning(
                "First Ctrl+C stopped inference and is holding position. "
                "Press Ctrl+C once more to begin the configured safe return."
            )
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                should_return_home = True
                logging.info("Second Ctrl+C received; starting configured safe return.")
    finally:
        if should_return_home:
            try:
                home_action = reset_home_action if reset_home_action is not None else environment.captured_start_action()
                if home_action is None:
                    raise RuntimeError("No episode-start pose was captured; cannot perform automatic safe return.")
                environment.return_to_joint_home(
                    home_action,
                    duration_s=args.reset_duration_s,
                    timeout_s=args.reset_timeout_s,
                    tolerance_rad=args.reset_tolerance_rad,
                    publish_hz=args.reset_publish_hz,
                )
            except Exception:
                logging.exception("Piper safe return failed; inspect the robot before sending further commands.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
