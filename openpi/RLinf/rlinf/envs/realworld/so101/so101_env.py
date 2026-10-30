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

"""Dual-SO101 real-robot environment for RLT Stage 2 (no ROS).

Observation: ``{"state": {"joint_position": (12,)}, "frames": {cam_high,
cam_left_wrist, cam_right_wrist}}`` — the same layout as the Piper env, so
``RealWorldEnv`` and the native-JAX feature path are shared. Actions are 12-D
absolute follower targets in LeRobot units (degrees, gripper 0-100).
"""

from __future__ import annotations

import dataclasses
import os
import time
from typing import Any

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.so101.hardware import CAMERAS, STATE_DIM, So101Arms, make_arms
from rlinf.utils.logging import get_logger


@dataclasses.dataclass
class So101Config:
    # so101/dual_arm.yaml of the repository (ports by USB serial, cameras).
    arm_config: str = "${RLT_ROOT}/so101/dual_arm.yaml"
    # lerobot: real arms; lerobot_mock: LeRobot's mocked buses; sim: SimSo101Arms.
    hardware: str = "lerobot"
    task_description: str = ""
    # Must match the recording rate of the dataset the VLA was trained on.
    step_frequency: float = 30.0
    max_num_steps: int = 100000
    render_height: int = 224
    render_width: int = 224
    # Per-step follower safety cap (degrees), LeRobot ``max_relative_target``.
    max_relative_target: float | None = 10.0
    # Evo-RLT intervention_action_blend_time_s.
    takeover_blend_s: float = 0.5
    # Evo-RLT PolicySyncDualArmExecutor: stiff leaders follow the policy.
    mirror_leader: bool = True
    # H returns here; null captures the pose at the first reset.
    home_position: list[float] | None = None
    home_speed_deg_s: float = 30.0
    home_tolerance_deg: float = 1.0
    # evdev keyboard, or "file" to read keys appended to key_file (scripts, SSH).
    keyboard: str = "evdev"
    key_file: str = "~/.so101_rlt_keys"

    def __post_init__(self) -> None:
        if self.step_frequency <= 0:
            raise ValueError("step_frequency must be positive.")
        if self.home_position is not None and len(self.home_position) != STATE_DIM:
            raise ValueError(f"home_position needs {STATE_DIM} values.")


class So101DualEnv(gym.Env):
    """Synchronous control: one ``step`` = one follower command at ``step_frequency``."""

    def __init__(self, override_cfg: dict[str, Any] | None = None, arms: So101Arms | None = None):
        cfg = dict(override_cfg or {})
        fields = {f.name for f in dataclasses.fields(So101Config)}
        self.config = So101Config(**{k: v for k, v in cfg.items() if k in fields})
        self._logger = get_logger()
        self.arms = arms or make_arms(
            self.config.hardware,
            os.path.expandvars(os.path.expanduser(self.config.arm_config)),
            max_relative_target=self.config.max_relative_target,
        )
        self.arms.connect()
        self._period = 1.0 / self.config.step_frequency
        self._next_tick = time.monotonic()
        self._num_steps = 0
        self._last_command: np.ndarray | None = None
        self.home_position = (
            None if self.config.home_position is None else np.asarray(self.config.home_position, np.float32)
        )
        image = gym.spaces.Box(0, 255, (self.config.render_height, self.config.render_width, 3), np.uint8)
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {"joint_position": gym.spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32)}
                ),
                "frames": gym.spaces.Dict({name: image for name in CAMERAS}),
            }
        )
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (STATE_DIM,), np.float32)

    @property
    def task_description(self) -> str:
        return self.config.task_description

    @property
    def last_command(self) -> np.ndarray | None:
        return None if self._last_command is None else self._last_command.copy()

    def _resize(self, image: np.ndarray) -> np.ndarray:
        from openpi_client import image_tools

        image = image_tools.resize_with_pad(image, self.config.render_height, self.config.render_width)
        return image_tools.convert_to_uint8(image)

    def read_observation(self) -> dict[str, Any]:
        frames = self.arms.read_frames()
        missing = [name for name in CAMERAS if name not in frames]
        if missing:
            raise KeyError(f"SO101 cameras missing {missing}; check cameras in dual_arm.yaml")
        return {
            "state": {"joint_position": self.arms.read_follower()},
            "frames": {name: self._resize(frames[name]) for name in CAMERAS},
        }

    def _wait_tick(self) -> None:
        now = time.monotonic()
        if self._next_tick > now:
            time.sleep(self._next_tick - now)
        self._next_tick = max(self._next_tick + self._period, time.monotonic())

    def command(self, target: np.ndarray) -> np.ndarray:
        """One rate-limited follower command; returns the target actually sent."""
        target = np.asarray(target, np.float32).reshape(-1)
        if target.shape != (STATE_DIM,):
            raise ValueError(f"Expected a {STATE_DIM}-D SO101 joint action, got {target.shape}.")
        self._wait_tick()
        sent = self.arms.write_follower(target)
        self._last_command = sent.copy()
        self._num_steps += 1
        return sent

    def hold(self) -> np.ndarray:
        """Keep the followers at their last command (or measured pose)."""
        target = self._last_command if self._last_command is not None else self.arms.read_follower()
        return self.command(target)

    def step(self, action: np.ndarray):
        sent = self.command(action)
        observation = self.read_observation()
        truncated = self._num_steps >= self.config.max_num_steps
        return observation, 0.0, False, truncated, {"so101_sent_action": sent}

    def capture_home(self) -> None:
        if self.home_position is None:
            self.home_position = self.arms.read_follower()

    def go_home(self, should_stop=lambda: False) -> bool:
        """Move the followers to ``home_position`` at ``home_speed_deg_s``; True when reached."""
        if self.home_position is None:
            return True
        max_step = self.config.home_speed_deg_s / self.config.step_frequency
        current = self.arms.read_follower()
        for _ in range(int(10 * self.config.step_frequency * 60)):
            if should_stop():
                return False
            error = self.home_position - current
            if float(np.max(np.abs(error))) <= self.config.home_tolerance_deg:
                return True
            current = current + np.clip(error, -max_step, max_step)
            self.command(current)
        return False

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._num_steps = 0
        self.capture_home()
        return self.read_observation(), {}

    def close(self) -> None:
        self.arms.close()
