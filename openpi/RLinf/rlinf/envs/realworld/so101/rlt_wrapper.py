# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Operator control and RLT replay signals for dual SO101.

Keys match the Piper RLT wrapper (I/B/T/C/X/H/Space), and every step reports
the same infos (``rlt_switch_flags``, ``rlt_execution_valid``,
``rlt_executed_action``, ``intervene_action``, ``rlt_keyboard_terminal`` ...),
so the RLT rollout, replay and learner are unchanged. The takeover itself is
Evo-RLT's leader-arm scheme (see :mod:`takeover`); unlike the Piper mux, T
toggles: a second T hands control back to the policy.
"""

from __future__ import annotations

import os
import threading
import time

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.so101.hardware import STATE_DIM
from rlinf.envs.realworld.so101.takeover import So101Takeover

_SPACE = {"Key.space", "space", " ", "KEY_SPACE"}


class FileKeyListener:
    """Keys appended to a text file (one key per line, e.g. ``echo t >> ~/.so101_rlt_keys``)."""

    def __init__(self, path: str):
        self.path = os.path.expanduser(path)
        open(self.path, "a").close()
        self._offset = os.path.getsize(self.path)

    def pop_pressed_keys(self) -> list[str]:
        with open(self.path) as f:
            f.seek(self._offset)
            text = f.read()
            self._offset = f.tell()
        keys = []
        for line in text.splitlines():
            key = line.strip().lower()
            if key in {"space", "key.space"} or (line and not line.strip()):
                keys.append("space")
            elif key:
                keys.append(key)
        return keys


def make_listener(kind: str, key_file: str):
    if kind == "file":
        return FileKeyListener(key_file)
    from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener

    return KeyboardListener()


class So101RLTWrapper(gym.Wrapper):
    """I starts/resumes, B records, T toggles the leader takeover, C/X finish, Space holds, H homes."""

    POLL_S = 0.02

    def __init__(self, env: gym.Env, *, listener=None, start_keyboard_thread: bool = True):
        super().__init__(env)
        base = env.unwrapped
        self.listener = listener if listener is not None else make_listener(base.config.keyboard, base.config.key_file)
        self.takeover = So101Takeover(
            base.arms, blend_s=base.config.takeover_blend_s, mirror_leader=base.config.mirror_leader
        )
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._last_press: dict[str, float] = {}
        self._running = False
        self._recording = False
        self._paused = True
        self._home_requested = False
        self._terminal: str | None = None
        self._terminal_reported = False
        self._chunk_invalid = False
        self._last_obs = None
        self._last_action = np.zeros(STATE_DIM, np.float32)
        self._last_human = False
        self._thread = None
        if start_keyboard_thread:
            self._thread = threading.Thread(target=self._keyboard_loop, name="so101-rlt-keyboard", daemon=True)
            self._thread.start()

    def _log(self, message: str) -> None:
        self.unwrapped._logger.info(f"[SO101] {message}")

    def _keyboard_loop(self) -> None:
        while not self._stop.wait(self.POLL_S):
            self.poll_keys()

    def poll_keys(self) -> None:
        for key in self.listener.pop_pressed_keys():
            self.handle_key(key)

    def _release_takeover(self) -> None:
        if self.takeover.active:
            self.takeover.release()
            # Evo-RLT: the interrupted policy chunk is dropped and recomputed.
            self._chunk_invalid = True

    def handle_key(self, key: str) -> None:
        key = "space" if key in _SPACE else key.lower().removeprefix("key_")
        now = time.monotonic()
        with self._lock:
            if now - self._last_press.get(key, -1e9) < 0.2:
                return
            self._last_press[key] = now
            if key == "space":
                self._paused = True
                self._chunk_invalid = True
                self._release_takeover()
                self._log("Space: HOLD; episode and B collection gate retained. I resumes.")
            elif key == "h":
                if self._running and self._terminal is None and not self._paused:
                    self._log("H requires HOLD; press Space or finish with C/X first.")
                else:
                    self._paused = True
                    self._home_requested = True
                    self._log("H: returning the followers to HOME; nothing is recorded.")
            elif key == "i":
                if self._terminal is not None or (self._running and not self._paused and not self.takeover.active):
                    return
                self._release_takeover()
                self._running = True
                self._paused = False
                self._chunk_invalid = True
                self._log("I: policy control (re-planning from a fresh observation).")
            elif key == "b":
                if self._running and self._terminal is None and not self._recording:
                    self._recording = True
                    self._log("B: collection starts with the next complete action chunk.")
            elif key == "t":
                if not self._running or not self._recording or self._terminal is not None:
                    self._log("T requires an active episode and the B collection gate.")
                elif self._paused:
                    self._log("T: press I first; takeover starts from policy control.")
                elif self.takeover.active:
                    self._release_takeover()
                    self._log("T: released; leaders hold, the policy re-plans.")
                else:
                    self.takeover.start(self.unwrapped.last_command)
                    self._log("T: takeover; leaders are limp, the followers track them.")
            elif key in {"c", "x", "q"}:
                if self._running and self._terminal is None:
                    self._terminal = "success" if key == "c" else "failure"
                    self._paused = True
                    self._release_takeover()
                    self._log(f"{key.upper()}: episode {self._terminal}; collection ends in HOLD.")

    def _go_home_if_requested(self) -> None:
        with self._lock:
            requested, self._home_requested = self._home_requested, False
        if requested:
            reached = self.unwrapped.go_home(should_stop=self._stop.is_set)
            self._log("HOME reached." if reached else "HOME interrupted.")

    def reset(self, **kwargs):
        with self._lock:
            self._running = self._recording = False
            self._terminal = None
            self._terminal_reported = False
            self._paused = True
            self._chunk_invalid = False
            self._last_human = False
            self._last_press.clear()
            self._release_takeover()
            self.listener.pop_pressed_keys()
        self.env.reset(**kwargs)
        self.takeover.hold_leader()
        self._log("Idle/HOLD: I starts the policy; B starts collection; H returns HOME.")
        while not self._stop.is_set():
            if self._thread is None:
                self.poll_keys()
            self._go_home_if_requested()
            with self._lock:
                if self._running and not self._paused:
                    break
            time.sleep(self.POLL_S)
        else:
            raise RuntimeError("SO101 RLT keyboard stopped while waiting for I.")
        observation = self.unwrapped.read_observation()
        with self._lock:
            self._last_obs = observation
            self._last_action = observation["state"]["joint_position"].copy()
        return observation, {}

    def begin_action_chunk(self, num_actions: int) -> None:
        with self._lock:
            if not self._paused and not self.takeover.active:
                self._chunk_invalid = False

    # Piper RTC hooks; SO101 runs without a bridge (rtc_enabled: false).
    def queue_rtc_tail(self, reference_tail: np.ndarray) -> None:
        return None

    def start_rtc_tail(self, reference_tail: np.ndarray) -> None:
        return None

    def align_rtc_chunk(self, actions: np.ndarray, reference_plan: np.ndarray):
        return actions, 0

    def step(self, action):
        base = self.unwrapped
        while True:
            if self._stop.is_set():
                raise RuntimeError("SO101 RLT keyboard control stopped.")
            if self._thread is None:
                self.poll_keys()
            self._go_home_if_requested()
            with self._lock:
                if self._terminal is not None:
                    return self._terminal_step()
                if not self._paused:
                    human = self.takeover.active
                    invalid = self._chunk_invalid
                    break
            base.hold()  # HOLD keeps the control clock and the followers still

        info: dict = {}
        terminated = truncated = False
        if human:
            executed = base.command(self.takeover.human_command())
            obs = base.read_observation()
        elif invalid:
            executed = base.hold()
            obs = base.read_observation()
        else:
            obs, _, terminated, truncated, info = self.env.step(action)
            executed = info.pop("so101_sent_action")
            self.takeover.policy_command(executed)
        with self._lock:
            self._last_obs = obs
            self._last_action = np.asarray(executed, np.float32).copy()
            self._last_human = human
            if self._terminal is not None:
                return self._terminal_step()
            if terminated or truncated:
                self._terminal = "timeout"
                self._paused = True
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
        info = self._info(human=self._last_human and first, valid=first and not self._chunk_invalid)
        info["rlt_keyboard_terminal"] = first and self._recording and self._terminal in {"success", "failure"}
        return self._last_obs, float(first and self._terminal == "success"), not truncated, truncated, info

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.takeover.hold_leader()
        self.env.close()
