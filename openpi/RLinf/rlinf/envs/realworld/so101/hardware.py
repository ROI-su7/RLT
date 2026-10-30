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

"""Dual SO101 hardware access: LeRobot buses on the robot, a simulator for tests.

All joint vectors are 12-D ``[left arm (6), right arm (6)]`` in LeRobot's
calibrated units (degrees; gripper 0-100), matching the recorded dataset.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
from typing import Protocol

import numpy as np

SIDES = ("left", "right")
CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
ARM_DIM = 6
STATE_DIM = 2 * ARM_DIM


class So101Arms(Protocol):
    """Two leader/follower pairs and the cameras of one dual-SO101 station."""

    def connect(self) -> None: ...

    def close(self) -> None: ...

    def read_follower(self) -> np.ndarray: ...

    def read_leader(self) -> np.ndarray: ...

    def write_follower(self, target: np.ndarray) -> np.ndarray:
        """Command both followers; return the target actually sent (after the safety cap)."""

    def write_leader(self, target: np.ndarray) -> None:
        """Drive the torque-enabled leaders (policy mirroring)."""

    def set_leader_torque(self, enabled: bool) -> None: ...

    def read_frames(self) -> dict[str, np.ndarray]:
        """Latest HWC uint8 RGB frame of every camera."""


def _load_robot_config_module(arm_config: str):
    """Import ``robot_config.py`` that sits next to ``dual_arm.yaml`` (the repo's ``so101/``)."""
    path = pathlib.Path(arm_config).expanduser().resolve().parent / "robot_config.py"
    spec = importlib.util.spec_from_file_location("so101_robot_config", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"{path} not found next to {arm_config}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LeRobotSo101Arms:
    """Real (or LeRobot-mocked) dual SO101 through ``lerobot`` 0.1.0 Feetech buses."""

    def __init__(self, arm_config: str, *, mock: bool = False, max_relative_target: float | None = 10.0):
        module = _load_robot_config_module(arm_config)
        from lerobot.common.robot_devices.robots.utils import make_robot_from_config

        self.robot = make_robot_from_config(
            module.make_robot_config(arm_config, mock=mock, max_relative_target=max_relative_target)
        )
        self._leader_torque: bool | None = None

    def connect(self) -> None:
        self.robot.connect()  # follower torque on, leader torque off (LeRobot default)
        self._leader_torque = False

    def close(self) -> None:
        if self.robot.is_connected:
            self.set_leader_torque(False)
            self.robot.disconnect()

    def _read(self, arms) -> np.ndarray:
        return np.concatenate(
            [np.asarray(arms[side].read("Present_Position"), np.float32) for side in SIDES]
        )

    def read_follower(self) -> np.ndarray:
        return self._read(self.robot.follower_arms)

    def read_leader(self) -> np.ndarray:
        return self._read(self.robot.leader_arms)

    def write_follower(self, target: np.ndarray) -> np.ndarray:
        import torch

        sent = self.robot.send_action(torch.as_tensor(np.asarray(target, np.float32)))
        return np.asarray(sent, np.float32)

    def write_leader(self, target: np.ndarray) -> None:
        target = np.asarray(target, np.float32)
        for index, side in enumerate(SIDES):
            self.robot.leader_arms[side].write("Goal_Position", target[index * ARM_DIM : (index + 1) * ARM_DIM])

    def set_leader_torque(self, enabled: bool) -> None:
        if self._leader_torque == enabled:
            return
        from lerobot.common.robot_devices.motors.feetech import TorqueMode

        if enabled:
            # Start from the leader's own pose so enabling torque never jerks it.
            for side in SIDES:
                arm = self.robot.leader_arms[side]
                arm.write("Goal_Position", arm.read("Present_Position"))
        value = (TorqueMode.ENABLED if enabled else TorqueMode.DISABLED).value
        for side in SIDES:
            self.robot.leader_arms[side].write("Torque_Enable", value)
        self._leader_torque = enabled

    def read_frames(self) -> dict[str, np.ndarray]:
        return {name: np.asarray(camera.async_read()) for name, camera in self.robot.cameras.items()}


class SimSo101Arms:
    """Kinematic stand-in used by tests and the no-hardware pipeline check.

    Followers move toward their goal with a first-order lag. Torque-enabled
    leaders follow their goal; limp leaders stay where the (simulated)
    operator puts them with :meth:`move_leader`.
    """

    def __init__(self, *, max_relative_target: float | None = 10.0, lag: float = 0.5, image_size: int = 96):
        self.max_relative_target = max_relative_target
        self.lag = lag
        self.image_size = image_size
        self.follower = np.zeros(STATE_DIM, np.float32)
        self.follower[ARM_DIM - 1 :: ARM_DIM] = 30.0  # half-open grippers
        self.leader = self.follower.copy()
        self.follower_goal = self.follower.copy()
        self.leader_torque = False
        self.torque_log: list[bool] = []
        self.leader_writes = 0
        self.connected = False

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False

    def _tick(self) -> None:
        self.follower += self.lag * (self.follower_goal - self.follower)

    def read_follower(self) -> np.ndarray:
        self._tick()
        return self.follower.copy()

    def read_leader(self) -> np.ndarray:
        return self.leader.copy()

    def move_leader(self, target: np.ndarray) -> None:
        """Simulated operator hand on the limp leaders."""
        if not self.leader_torque:
            self.leader = np.asarray(target, np.float32).copy()

    def write_follower(self, target: np.ndarray) -> np.ndarray:
        target = np.asarray(target, np.float32).copy()
        if self.max_relative_target is not None:
            limit = float(self.max_relative_target)
            target = np.clip(target, self.follower - limit, self.follower + limit)
        self.follower_goal = target
        return target

    def write_leader(self, target: np.ndarray) -> None:
        self.leader_writes += 1
        if self.leader_torque:
            self.leader = np.asarray(target, np.float32).copy()

    def set_leader_torque(self, enabled: bool) -> None:
        if enabled != self.leader_torque:
            self.torque_log.append(enabled)
        self.leader_torque = enabled

    def read_frames(self) -> dict[str, np.ndarray]:
        size = self.image_size
        frames = {}
        for index, name in enumerate(CAMERAS):
            image = np.zeros((size, size, 3), np.uint8)
            image[..., index] = np.uint8(int(abs(self.follower[index * 4 % STATE_DIM])) % 256)
            frames[name] = image
        return frames


def make_arms(hardware: str, arm_config: str, *, max_relative_target: float | None) -> So101Arms:
    if hardware == "sim":
        return SimSo101Arms(max_relative_target=max_relative_target)
    if hardware in {"lerobot", "lerobot_mock"}:
        return LeRobotSo101Arms(
            os.path.expandvars(arm_config), mock=hardware == "lerobot_mock", max_relative_target=max_relative_target
        )
    raise ValueError(f"Unknown SO101 hardware {hardware!r}; use lerobot, lerobot_mock or sim.")
