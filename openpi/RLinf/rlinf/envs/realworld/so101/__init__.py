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

"""Dual SO101 (LeRobot, Feetech) real-robot environment for RLT Stage 2."""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
from gymnasium.envs.registration import register

from .so101_env import So101Config, So101DualEnv


def create_so101_env(
    override_cfg: dict[str, Any],
    worker_info: Any = None,
    hardware_info: Any = None,
    env_idx: int = 0,
    env_cfg: Mapping[str, Any] | None = None,
) -> gym.Env:
    env = So101DualEnv(override_cfg)
    if (env_cfg or {}).get("keyboard_reward_wrapper") == "so101_rlt":
        from .rlt_wrapper import So101RLTWrapper

        env = So101RLTWrapper(env)
    return env


register(id="So101DualEnv-v1", entry_point="rlinf.envs.realworld.so101:create_so101_env")

__all__ = ["So101Config", "So101DualEnv", "create_so101_env"]
