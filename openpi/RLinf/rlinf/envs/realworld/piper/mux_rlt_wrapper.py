"""Stage-2 episode/data control using the existing four-arm takeover mux."""

from __future__ import annotations

import threading
import time

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener
from rlinf.envs.realworld.piper.four_can_mux import FourCanMuxClient


class PiperMuxRLTWrapper(gym.Wrapper):
    """I starts/resumes, B records, T selects humans, C/X finish.

    Keyboard requests run independently of Ray/JAX inference. Space only
    pauses; HOME is handled by the mux using its launch-time four-arm poses.
    """

    def __init__(
        self,
        env: gym.Env,
        mux: FourCanMuxClient,
        *,
        listener=None,
        start_keyboard_thread: bool = True,
    ) -> None:
        super().__init__(env)
        self.mux = mux
        self.listener = listener if listener is not None else KeyboardListener()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._running = False
        self._recording = False
        self._terminal: str | None = None
        self._terminal_reported = False
        self._paused = True
        self._resume_requested = False
        self._resume_generation = -1
        self._resume_prepared_generation = -1
        self._takeover_requested = False
        self._last_press: dict[str, float] = {}
        self._last_obs = None
        self._last_action = np.zeros(14, dtype=np.float32)
        self._last_human = False
        self._observation_generation = -1
        self._chunk_generation = -1
        self._chunk_invalid = False
        self._rtc_interrupted = False
        self._thread = None
        if start_keyboard_thread:
            self._thread = threading.Thread(
                target=self._keyboard_loop, name="piper-rlt-mux-keyboard", daemon=True
            )
            self._thread.start()

    def _log(self, message: str) -> None:
        self.unwrapped._logger.info(message)

    def _keyboard_loop(self) -> None:
        while not self._stop.wait(0.02):
            for key in self.listener.pop_pressed_keys():
                try:
                    self.handle_key(key)
                except Exception:
                    # A broken control path must not silently leave an actor
                    # running while the operator believes HOLD was requested.
                    self.mux.request_hold()
                    self.unwrapped._logger.exception(
                        "Piper RLT keyboard request failed"
                    )
                    self._stop.set()

    def handle_key(self, key: str) -> None:
        """Handle one debounced edge, including during an inference RPC."""
        now = time.monotonic()
        with self._lock:
            if now - self._last_press.get(key, -1e9) < 0.2:
                return
            self._last_press[key] = now
            status = self.mux.snapshot()
            if key in {"Key.space", "space", " "}:
                self._paused = True
                self._resume_requested = False
                self._takeover_requested = False
                self._chunk_invalid = True
                self.mux.request_hold()
                self._log(
                    "Space: software HOLD; episode and B collection gate retained. I resumes."
                )
            elif key == "h":
                if (
                    self._running
                    and self._terminal is None
                    and not (self._paused and status.hold)
                ):
                    self._log(
                        "H requires idle/HOLD; press Space or finish with C/X first."
                    )
                else:
                    self._paused = True
                    self._resume_requested = False
                    self._chunk_invalid = True
                    self.mux.request_home()
                    self._log(
                        "H: returning all four arms to launch HOME; no replay is recorded."
                    )
            elif key == "i":
                if self._terminal is not None or status.home:
                    return
                if self._resume_requested or (
                    self._running and status.policy_active and not self._paused
                ):
                    return
                self._running = True
                self._paused = False
                self._resume_requested = True
                self._resume_generation = status.generation
                self._takeover_requested = False
                self.mux.request_mode("INFERENCE")
                self._log(
                    "I: requested INFERENCE; waiting for four-arm alignment and fresh planning."
                )
            elif key == "b":
                if self._running and self._terminal is None and not self._recording:
                    self._recording = True
                    self._log(
                        "B: collection requested from the next complete action chunk (warmup/actor gate retained)."
                    )
            elif key == "t":
                if (
                    not self._running
                    or not self._recording
                    or self._terminal is not None
                ):
                    self._log("T requires an active episode and B collection gate.")
                elif status.policy_active and not self._takeover_requested:
                    self._takeover_requested = True
                    self._resume_requested = False
                    self.mux.request_mode("TAKEOVER")
                    self._log(
                        "T: waiting for original mux HOLD ACK, two-second delay and both MASTER streams."
                    )
            elif key in {"c", "x", "q"}:
                if self._running and self._terminal is None:
                    self._terminal = "success" if key == "c" else "failure"
                    self._resume_requested = False
                    self._takeover_requested = False
                    self._paused = True
                    self.mux.request_hold()
                    self._log(
                        f"{key.upper()}: episode {self._terminal}; collection ends in HOLD."
                    )

    def reset(self, **kwargs):
        """Hold between episodes without redefining or moving HOME."""
        with self._lock:
            self._running = False
            self._recording = False
            self._terminal = None
            self._terminal_reported = False
            self._paused = True
            self._resume_requested = False
            self._takeover_requested = False
            self._last_human = False
            self._rtc_interrupted = False
            self._chunk_invalid = False
            self._last_press.clear()
            self.listener.pop_pressed_keys()
            self.mux.request_hold()
        options = dict(kwargs.get("options") or {})
        options["preserve_suspended_policy_output"] = True
        kwargs["options"] = options
        self.env.reset(**kwargs)
        self._log(
            "Idle/HOLD: I starts inference; B starts collection; H returns launch HOME."
        )
        while not self._stop.is_set():
            status = self.mux.snapshot()
            self._prepare_resume(status)
            with self._lock:
                if (
                    self._running
                    and self._resume_requested
                    and status.policy_active
                    and status.generation != self._resume_generation
                ):
                    self._resume_requested = False
                    break
            time.sleep(0.02)
        if self._stop.is_set():
            raise RuntimeError("Piper RLT keyboard stopped while waiting for I.")
        observation = self.unwrapped.read_observation()
        with self._lock:
            self._last_obs = observation
            # Space during camera decode must invalidate the pending plan.
            self._observation_generation = status.generation
            self._last_action = observation["state"]["joint_position"].copy()
        return observation, {}

    def _prepare_resume(self, status) -> None:
        """Seed the mux with measured pose after HOLD releases, before ACTIVE.

        Original mux INFERENCE confirmation requires fresh policy inputs.
        Waiting for ACTIVE before seeding those inputs would deadlock startup.
        Followers remain held by the mux until both leaders confirm alignment.
        """
        with self._lock:
            ready = (
                self._resume_requested
                and status.mode == "INFERENCE"
                and not status.hold
                and not status.home
                and status.generation != self._resume_generation
                and status.generation != self._resume_prepared_generation
            )
            if not ready:
                return
            self._resume_prepared_generation = status.generation
        self.unwrapped.resume_policy_output()

    def begin_action_chunk(self, num_actions: int) -> None:
        """Bind the pending plan to the generation of its input observation."""
        with self._lock:
            self._chunk_generation = self._observation_generation
            self._chunk_invalid = self._rtc_interrupted
            self._rtc_interrupted = False
        self.unwrapped.begin_action_chunk(num_actions)

    def _rtc_allowed(self) -> bool:
        status = self.mux.snapshot()
        with self._lock:
            return (
                self._running
                and self._terminal is None
                and not self._paused
                and not self._takeover_requested
                and not self._resume_requested
                and status.policy_active
                and status.generation == self._observation_generation
            )

    def queue_rtc_tail(self, reference_tail: np.ndarray) -> None:
        """Queue a suffix only for the current confirmed policy generation."""
        if self._rtc_allowed():
            self.unwrapped.queue_rtc_tail(reference_tail)

    def start_rtc_tail(self, reference_tail: np.ndarray) -> None:
        """Continue planned motion during inference, never during HOLD/teleop."""
        if self._rtc_allowed():
            self.unwrapped.start_rtc_tail(reference_tail)

    def align_rtc_chunk(self, actions: np.ndarray, reference_plan: np.ndarray):
        """Drain and align RTC only while the plan's generation remains current."""
        if not self._rtc_allowed():
            self.unwrapped._queued_rtc_tail = None
            self.unwrapped._stop_rtc_tail()
            return actions, 0
        result = self.unwrapped.align_rtc_chunk(actions, reference_plan)
        # A HOLD/T may arrive while the executor drains its bounded suffix.
        if not self._rtc_allowed():
            self.unwrapped._queued_rtc_tail = None
            self._rtc_interrupted = True
            return actions, 0
        return result

    def step(self, action):
        """Collect policy/human steps, never advance policy through a HOLD."""
        while not self._stop.is_set():
            status = self.mux.snapshot()
            self._prepare_resume(status)
            with self._lock:
                if self._terminal is not None:
                    return self._terminal_step()
                if (
                    status.human_active
                    and not self._paused
                    and not self._resume_requested
                ):
                    self._takeover_requested = False
                    break
                if (
                    self._resume_requested
                    and status.policy_active
                    and status.generation != self._resume_generation
                ):
                    self._resume_requested = False
                    self._chunk_invalid = True
                if (
                    status.policy_active
                    and not self._paused
                    and not self._takeover_requested
                ):
                    if status.generation != self._chunk_generation:
                        self._chunk_invalid = True
                    break
                # T's transition waits without inventing human or HOLD frames.
                # A spontaneous HOLD also requires an explicit I to resume.
                if not self._takeover_requested and not self._resume_requested:
                    self._paused = True
                    self._chunk_invalid = True
            time.sleep(0.02)
        else:
            raise RuntimeError("Piper RLT keyboard control stopped.")

        human = status.human_active
        if human:
            try:
                executed = self.mux.executed_action()
            except RuntimeError:
                self.mux.request_hold()
                with self._lock:
                    self._paused = True
                    self._chunk_invalid = True
                raise
            # External control must keep sampling at the policy rate; it does
            # not send policy commands or enable a competing follower driver.
            time.sleep(1.0 / self.unwrapped.config.step_frequency)
            obs, _, terminated, truncated, info = self.unwrapped.step_external_control()
        elif self._chunk_invalid:
            obs = self.unwrapped.read_observation()
            terminated = truncated = False
            info = {}
            executed = self._last_action.copy()
        else:
            try:
                obs, _, terminated, truncated, info = self.env.step(action)
            except RuntimeError as exc:
                if "suspended" not in str(exc) and self.mux.snapshot().policy_active:
                    raise
                # HOLD may arrive while the executor is waiting for its tick.
                with self._lock:
                    self._chunk_invalid = not self._takeover_requested
                return self.step(action)
            executed = np.asarray(action, dtype=np.float32).reshape(14).copy()
            if getattr(self.unwrapped, "_actor_direct_execution", False) and getattr(
                getattr(self.unwrapped, "config", None),
                "actor_command_profile_enabled",
                False,
            ):
                accepted = info.get("piper_accepted_target")
                if accepted is None:
                    raise RuntimeError(
                        "Actor command profile did not report its accepted goal."
                    )
                executed = np.asarray(accepted, dtype=np.float32).reshape(14).copy()
        with self._lock:
            self._last_obs = obs
            self._last_action = executed
            self._last_human = human
            self._observation_generation = self.mux.snapshot().generation
            if self._terminal is not None:
                return self._terminal_step()
            if terminated or truncated:
                self._terminal = "timeout"
                self.mux.request_hold()
                return self._terminal_step(truncated=True)
            info.update(self._info(human=human, valid=not self._chunk_invalid))
            return obs, 0.0, False, False, info

    def _info(self, *, human: bool, valid: bool) -> dict:
        return {
            "rlt_switch_flags": self._recording,
            "rlt_execution_valid": valid,
            "rlt_executed_action": self._last_action.copy(),
            "intervene_action": self._last_action.copy() if human else None,
            "manual_episode_reward": float(self._terminal == "success"),
            "rlt_policy_switch_event": self._terminal,
            "rlt_keyboard_terminal": False,
        }

    def _terminal_step(self, *, truncated: bool = False):
        truncated = truncated or self._terminal == "timeout"
        first = not self._terminal_reported
        self._terminal_reported = True
        # Padding repeats the last action, but is never marked as human data.
        info = self._info(
            human=self._last_human and first, valid=first and not self._chunk_invalid
        )
        info["rlt_keyboard_terminal"] = (
            first and self._recording and self._terminal in {"success", "failure"}
        )
        return (
            self._last_obs,
            float(first and self._terminal == "success"),
            not truncated,
            truncated,
            info,
        )

    def close(self) -> None:
        """Hold before stopping the keyboard and policy executor."""
        self.mux.request_hold()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.env.close()
        self.mux.close()
