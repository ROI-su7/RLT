"""Gym registration and real-world wrappers for the Piper wire task."""

from __future__ import annotations

import math
import time
from typing import Any, Mapping

import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener
from rlinf.envs.realworld.piper.leader_takeover import (
    LeaderTakeoverConfig,
    LeaderTakeoverStateMachine,
    TakeoverState,
)
from rlinf.envs.realworld.piper.mode2_controller import PiperMode2Controller
from rlinf.envs.realworld.piper.mode2_ros import Mode2RosConfig, RosMode2Hardware
from rlinf.envs.realworld.piper.piper_env import PiperWireEnv


class PiperRLTControlWrapper(gym.Wrapper):
    """Operator controls for RLT rollouts and safe between-episode reset."""

    IDLE_POLL_S = 0.05
    WAIT_HEARTBEAT_S = 10.0
    # Log once per second at the 15 Hz hardware control rate.  Logging every
    # control tick would itself reduce the effective robot frequency.
    STEP_LOG_INTERVAL = 15

    def __init__(
        self,
        env: gym.Env,
        *,
        manual_start_gate: bool = True,
        mode2_controller: PiperMode2Controller | None = None,
    ):
        super().__init__(env)
        self.listener = KeyboardListener()
        self.manual_start_gate = bool(manual_start_gate)
        self._actor_active = False
        self._last_press: dict[str, float] = {}
        self._episode_step = 0
        self._episode_reward = 0.0
        self._insertions_scored = 0
        self._drops_scored = 0
        self._grasp_scored = False
        self._actor_score_ready_step = None
        self._episode_ended = False
        self._last_obs = None
        self._mode2 = mode2_controller
        self._last_mode2_warning_at = -math.inf
        self._last_mode2_state = (
            None if mode2_controller is None else mode2_controller.state
        )

    def reset(self, **kwargs):
        mode2_returning = (
            self._mode2 is not None and self._mode2.state == TakeoverState.TELEOP_RETURN
        )
        if mode2_returning:
            options = dict(kwargs.get("options") or {})
            options["preserve_suspended_policy_output"] = True
            kwargs["options"] = options
        self._actor_active = False
        self._last_press.clear()
        self._episode_step = 0
        self._episode_reward = 0.0
        self._insertions_scored = 0
        self._drops_scored = 0
        self._grasp_scored = False
        self._actor_score_ready_step = None
        self._episode_ended = False
        self.listener.pop_pressed_keys()
        obs, info = self.env.reset(**kwargs)
        if self._mode2 is not None and self._mode2.state != TakeoverState.POLICY:
            if self._mode2.state == TakeoverState.FAULT:
                raise RuntimeError(
                    "Piper mode 2 is faulted; keep both arm pairs held and restart "
                    "the mode-2 driver after inspecting the fault."
                )
            if mode2_returning:
                self._wait_for_mode2_return()
                self.unwrapped.hold_position()
            self._mode2.resume_policy()
        if not self.manual_start_gate:
            return obs, info

        # PiperWireEnv.reset only reads state; it does not move the arms home.
        # Hold after every completed episode so the operator can re-thread and
        # arrange the fixture while the training process remains alive.
        self.unwrapped.hold_position()
        self._log_info(
            "Episode is idle and arms are holding position. Reset the wires/fixture, "
            "then press 'a' to start the next episode (Ctrl-C to stop training)."
        )
        last_heartbeat = time.monotonic()
        while True:
            time.sleep(self.IDLE_POLL_S)
            now = time.monotonic()
            if now - last_heartbeat >= self.WAIT_HEARTBEAT_S:
                last_heartbeat = now
                self._log_info("Still waiting for 'a' to start the next episode...")
            keys = self.listener.pop_pressed_keys()
            if "r" in keys:
                self._log_info(
                    "Returning both Piper arms to the captured episode start pose. "
                    "Press 'q' to stop and hold."
                )
                try:
                    completed = self.unwrapped.return_to_episode_start(
                        stop_requested=self._reset_stop_requested
                    )
                except (RuntimeError, TimeoutError) as exc:
                    self._log_info(f"Return-to-start failed safely: {exc}")
                    continue
                self._log_info(
                    "Piper returned to the episode start pose. Reset the wires/fixture, "
                    "then press 'a' to start."
                    if completed
                    else "Return-to-start interrupted; arms are holding position."
                )
                continue
            if "a" in keys:
                # Fetch fresh images/state after the operator has arranged the
                # scene, rather than returning the observation from before reset.
                obs, info = self.env.reset()
                self.unwrapped.hold_position()
                if self.unwrapped.capture_episode_start_position():
                    self._log_info(
                        "Captured the first episode's measured joint pose as the one-key reset target."
                    )
                if self._mode2 is not None and self._mode2.capture_episode_start():
                    self._log_info(
                        "Captured measured leader and follower poses as the "
                        "mode-2 return target."
                    )
                self._log_info("Starting next RLT episode.")
                self._last_obs = obs
                return obs, info

    def step(self, action):
        # The rollout interface supplies an action chunk. Once an operator
        # ends the episode, later actions from that already-issued chunk must
        # not overwrite q's hold or move the physical robot again.
        if self._episode_ended:
            if self._mode2 is None or self._mode2.state != TakeoverState.TELEOP_RETURN:
                self.unwrapped.hold_position()
            return (
                self._last_obs,
                0.0,
                True,
                False,
                {
                    "rlt_switch_flags": self._actor_active,
                    "rlt_policy_switch_event": "terminal_hold",
                    "manual_episode_reward": self._episode_reward,
                    "manual_insertions_scored": self._insertions_scored,
                    "manual_drops_scored": self._drops_scored,
                },
            )

        mode2_intervention = False
        if self._mode2 is not None:
            if self._mode2.state in {
                TakeoverState.LEADER_ALIGNING,
                TakeoverState.LEADER_MANUAL_ALIGN,
            }:
                try:
                    self._mode2.alignment_tick(time.monotonic())
                except (RuntimeError, TimeoutError, ValueError) as exc:
                    self.unwrapped.hold_position()
                    now = time.monotonic()
                    if now - self._last_mode2_warning_at >= 1.0:
                        self._last_mode2_warning_at = now
                        self._log_info(f"Mode-2 alignment stopped safely: {exc}")
                if (
                    self._last_mode2_state != TakeoverState.LEADER_MANUAL_ALIGN
                    and self._mode2.state == TakeoverState.LEADER_MANUAL_ALIGN
                ):
                    self._log_info(
                        "Mode 2: leaders reached the near region and entered "
                        "zero-force drag. Manually trim them to the follower pose; "
                        "the system will announce READY after stable fine alignment."
                    )
                if (
                    self._last_mode2_state != TakeoverState.READY
                    and self._mode2.state == TakeoverState.READY
                ):
                    self._log_info(
                        "Mode 2: leaders are aligned and stable (READY). Press "
                        "'t' to enter leader-follower takeover."
                    )
            self._last_mode2_state = self._mode2.state
            action, mode2_intervention = self._mode2.action_for_step(action)

        if mode2_intervention:
            # The mode-2 bridge is the sole 200-Hz follower command owner.
            # Continue collecting observations and replay actions without
            # waking the suspended policy publisher.
            obs, _, terminated, truncated, info = self.unwrapped.step_external_control()
        else:
            obs, _, terminated, truncated, info = self.env.step(action)
        if mode2_intervention:
            info["intervene_action"] = action.copy()
            info["intervene_flag"] = np.ones(1, dtype=np.float32)
            self._actor_active = True
        # RLT uses a sparse binary task reward. Diagnostic milestone keys never
        # alter this value; only a successful terminal ``c`` sets it to +1.
        reward = 0.0
        self._last_obs = obs
        self._episode_step += 1
        event = None
        for key in self.listener.pop_pressed_keys():
            now = time.monotonic()
            if now - self._last_press.get(key, -math.inf) < 0.2:
                continue
            self._last_press[key] = now
            if key == "h" and self._mode2 is not None:
                try:
                    self._mode2.request_hold()
                except (RuntimeError, ValueError) as exc:
                    self._log_info(f"Mode-2 hold request rejected safely: {exc}")
                    continue
                self.unwrapped.hold_position()
                self._actor_active = False
                event = "mode2_follower_hold"
                self._log_info(
                    "Mode 2: policy commands stopped; followers are holding their "
                    "measured pose. Press 'm' to align the leaders."
                )
            elif key == "m" and self._mode2 is not None:
                try:
                    self._mode2.request_alignment(now)
                except (RuntimeError, ValueError) as exc:
                    self.unwrapped.hold_position()
                    self._log_info(f"Mode-2 alignment request rejected safely: {exc}")
                    continue
                event = "mode2_leader_alignment"
                self._log_info(
                    "Mode 2: monitored manual leader approach started; followers "
                    "remain held. Move the powered leaders slowly to the follower "
                    "pose and wait for READY before pressing 't'."
                )
            elif key == "t" and self._mode2 is not None:
                try:
                    self.unwrapped.suspend_policy_output()
                    self._mode2.request_takeover(now)
                except (RuntimeError, ValueError) as exc:
                    self.unwrapped.hold_position()
                    self._log_info(f"Mode-2 takeover request rejected safely: {exc}")
                    continue
                event = "mode2_human_takeover"
                self._log_info(
                    "Mode 2: leader-follower takeover armed. Starting with the "
                    "next control step, leader actions replace policy actions in "
                    "replay; press 'c' for success or 'x' for failure."
                )
            elif key == "b":
                self._actor_active, event = True, "enter_actor"
                self._actor_score_ready_step = self._episode_step + 5
                self._log_info(
                    f"Episode step {self._episode_step}: entered the critical RLT "
                    "collection window. The router keeps native JAX control during "
                    "reference warmup and enables actor control only after the "
                    "learner-ready gate; "
                    f"reward keys become valid at step {self._actor_score_ready_step}."
                )
            elif key in {"g", "i", "f"} and (
                not self._actor_active
                or self._episode_step < (self._actor_score_ready_step or 0)
            ):
                self._log_info(
                    f"Episode step {self._episode_step}: ignored reward key '{key}' "
                    "because actor/RLT scoring is not ready; press 'b' and wait "
                    "for the logged ready step."
                )
            elif key == "g":
                if self._grasp_scored:
                    self._log_info(
                        f"Episode step {self._episode_step}: duplicate grasp score ignored."
                    )
                    continue
                self._grasp_scored, event = True, "dual_grasp"
                self._log_info(
                    f"Episode step {self._episode_step}: recorded dual-arm grasp "
                    "event (diagnostic only, reward unchanged)."
                )
            elif key == "i":
                if self._insertions_scored >= 4:
                    self._log_info(
                        f"Episode step {self._episode_step}: all four insertion scores "
                        "already recorded; duplicate ignored."
                    )
                    continue
                self._insertions_scored += 1
                self._grasp_scored = False
                event = "insertion"
                self._log_info(
                    f"Episode step {self._episode_step}: insertion "
                    f"{self._insertions_scored}/4 (diagnostic only, reward unchanged); "
                    "episode continues."
                )
            elif key == "f":
                if not self._grasp_scored:
                    self._log_info(
                        f"Episode step {self._episode_step}: slip/drop score ignored "
                        "because no coordinated grasp is currently credited."
                    )
                    continue
                self._drops_scored += 1
                self._grasp_scored = False
                event = "slip"
                self._log_info(
                    f"Episode step {self._episode_step}: recorded slip/drop event "
                    "(diagnostic only, reward unchanged); episode continues."
                )
            elif key == "c":
                reward = 1.0
                self._episode_reward = 1.0
                terminated, event = True, "success"
                self._log_info(
                    f"Episode step {self._episode_step}: objective complete +1.0; "
                    f"final episode reward={self._episode_reward:.1f}; ending episode."
                )
                break
            elif key == "a":
                self._log_info(
                    f"Episode step {self._episode_step}: 'a' is start-only and is "
                    "ignored while an episode is running."
                )
            elif key == "x":
                terminated, event = True, "abort"
                self._log_info(
                    f"Episode step {self._episode_step}: episode abandoned with no "
                    f"terminal penalty; final episode reward={self._episode_reward:.1f}."
                )
                break
            elif key == "q":
                if self._mode2 is not None:
                    self._mode2.emergency_stop()
                self.unwrapped.hold_position()
                terminated, event = True, "hold_and_abort"
                self._log_info(
                    f"Episode step {self._episode_step}: emergency hold and unsafe "
                    f"failure; terminal reward remains {self._episode_reward:.1f}."
                )
                break
        info["rlt_switch_flags"] = self._actor_active
        info["rlt_policy_switch_event"] = event
        info["manual_episode_reward"] = self._episode_reward
        info["manual_insertions_scored"] = self._insertions_scored
        info["manual_drops_scored"] = self._drops_scored
        self._episode_ended = bool(terminated or truncated)
        if self._episode_ended:
            # A suffix may already be running concurrently with final camera
            # decoding. Stop it immediately for every terminal outcome, not
            # only the explicit emergency-hold key.
            if self._mode2 is not None:
                self._mode2.finish_episode()
            if event != "hold_and_abort" and (
                self._mode2 is None or self._mode2.state != TakeoverState.TELEOP_RETURN
            ):
                self.unwrapped.hold_position()
        max_steps = self.unwrapped.config.max_num_steps
        if self._episode_step % self.STEP_LOG_INTERVAL == 0 or terminated or truncated:
            policy = (
                "critical RLT window requested (reference warmup or actor)"
                if self._actor_active
                else "reference/OpenPI"
            )
            outcome = " timeout" if truncated else ""
            self._log_info(
                f"Episode progress: step {self._episode_step}/{max_steps}, "
                f"policy={policy}.{outcome}"
            )
        return obs, reward, terminated, truncated, info

    def _log_info(self, message: str) -> None:
        logger = getattr(self.unwrapped, "_logger", None)
        if logger is not None:
            logger.info(message)

    def _reset_stop_requested(self) -> bool:
        """Consume idle keys while returning home; only ``q`` interrupts."""
        return "q" in self.listener.pop_pressed_keys()

    def _wait_for_mode2_return(self) -> None:
        """Keep bridge teleop active after scoring; return motion is not replay."""
        self._log_info(
            "Mode 2: reward/done committed. Teleop remains active for an "
            "unrecorded return. Move both arm pairs to the captured start and "
            "press 'r'; press 'q' to stop and hold."
        )
        last_heartbeat = time.monotonic()
        while self._mode2 is not None:
            time.sleep(self.IDLE_POLL_S)
            now = time.monotonic()
            if now - last_heartbeat >= self.WAIT_HEARTBEAT_S:
                last_heartbeat = now
                self._log_info(
                    "Still in unrecorded mode-2 return; press 'r' at the start pose."
                )
            keys = self.listener.pop_pressed_keys()
            if "q" in keys:
                self._mode2.hardware.set_follower_teleop_enabled(False)
                self._mode2.state_machine.fail("operator aborted teleop return")
                self.unwrapped.hold_position()
                raise RuntimeError("Mode-2 return aborted by operator.")
            if "r" not in keys:
                continue
            try:
                arm_error, gripper_error = self._mode2.complete_teleop_return()
            except RuntimeError as exc:
                self._log_info(f"Mode-2 return not accepted: {exc}")
                continue
            self._log_info(
                "Mode-2 return accepted: "
                f"arm_error={arm_error:.4f}rad, "
                f"gripper_error={gripper_error:.4f}m. Both followers are holding."
            )
            return


def create_piper_wire_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Mapping[str, Any],
) -> gym.Env:
    env = PiperWireEnv(override_cfg, worker_info, hardware_info, env_idx)
    if (
        not env.config.is_dummy
        and env_cfg.get("keyboard_reward_wrapper") == "piper_rlt"
    ):
        if getattr(env.config, "four_can_mux_enabled", False):
            if int(env_cfg.get("piper_control_mode", 1)) == 2:
                raise ValueError(
                    "Four-CAN mux and legacy mode2 takeover are mutually exclusive."
                )
            from rlinf.envs.realworld.piper.four_can_mux import FourCanMuxClient
            from rlinf.envs.realworld.piper.mux_rlt_wrapper import PiperMuxRLTWrapper

            wrapper_cls = PiperMuxRLTWrapper
            if env_cfg.get("piper_mux_followers_only", False):
                from rlinf.envs.realworld.piper.follower_rlt_wrapper import (
                    PiperFollowerRLTWrapper,
                )

                wrapper_cls = PiperFollowerRLTWrapper
            return wrapper_cls(env, FourCanMuxClient(env._rospy))
        control_mode = int(env_cfg.get("piper_control_mode", 1))
        if control_mode not in {0, 1, 2}:
            raise ValueError("piper_control_mode must be 0, 1, or 2.")
        mode2_controller = None
        if control_mode == 2:
            mode2_cfg = dict(env_cfg.get("mode2_takeover") or {})
            ros_keys = set(Mode2RosConfig.__dataclass_fields__)
            ros_cfg = Mode2RosConfig(
                **{key: mode2_cfg[key] for key in ros_keys if key in mode2_cfg}
            )
            safety_keys = set(LeaderTakeoverConfig.__dataclass_fields__) - {
                "joint_count"
            }
            safety_cfg = LeaderTakeoverConfig(
                joint_count=12,
                **{key: mode2_cfg[key] for key in safety_keys if key in mode2_cfg},
            )
            mode2_controller = PiperMode2Controller(
                LeaderTakeoverStateMachine(safety_cfg),
                RosMode2Hardware(ros_cfg),
            )
        env = PiperRLTControlWrapper(
            env,
            manual_start_gate=env_cfg.get("manual_start_gate", True),
            mode2_controller=mode2_controller,
        )
    return env


register(
    id="PiperWireEnv-v1",
    entry_point="rlinf.envs.realworld.piper.tasks:create_piper_wire_env",
)
