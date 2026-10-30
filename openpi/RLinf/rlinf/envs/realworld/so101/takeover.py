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

"""Leader-arm takeover for SO101, following Evo-RLT's human-in-the-loop recorder.

Evo-RLT (MINT-SJTU, ``adapters/lerobot/record/hil.py`` and ``loop.py``):

* While the policy drives, the leader arms hold torque and receive the same
  goal as the followers (``PolicySyncDualArmExecutor``), so at any moment the
  leader already sits where the follower is.
* Starting an intervention disables leader torque ("manual control"); the
  followers then track the leaders. The first ``blend_s`` seconds interpolate
  from the last policy action to the leader pose
  (``intervention_action_blend_time_s``).
* Releasing re-enables leader torque at its current pose and discards the rest
  of the interrupted policy chunk; the next action is recomputed.

States use Evo-RLT's codes: POLICY 0, ACTIVE 1, RELEASE 2.
"""

from __future__ import annotations

import enum
import time

import numpy as np

from rlinf.envs.realworld.so101.hardware import So101Arms


class InterventionState(enum.IntEnum):
    POLICY = 0
    ACTIVE = 1
    RELEASE = 2


class So101Takeover:
    def __init__(self, arms: So101Arms, *, blend_s: float = 0.5, mirror_leader: bool = True, clock=time.monotonic):
        if blend_s < 0:
            raise ValueError("blend_s must be >= 0")
        self.arms = arms
        self.blend_s = float(blend_s)
        self.mirror_leader = bool(mirror_leader)
        self.clock = clock
        self.state = InterventionState.POLICY
        self._blend_start_t: float | None = None
        self._blend_from: np.ndarray | None = None

    @property
    def active(self) -> bool:
        return self.state == InterventionState.ACTIVE

    def hold_leader(self) -> None:
        """Leaders stiff (torque on) whenever a human is not driving."""
        self.arms.set_leader_torque(self.mirror_leader)

    def policy_command(self, executed: np.ndarray) -> None:
        """Mirror a follower command onto the leaders (Evo-RLT policy sync)."""
        if self.state == InterventionState.ACTIVE:
            return  # T arrived during this step; the leaders are already limp
        self.state = InterventionState.POLICY
        if self.mirror_leader:
            self.arms.set_leader_torque(True)
            self.arms.write_leader(executed)

    def start(self, last_command: np.ndarray | None) -> None:
        self.state = InterventionState.ACTIVE
        self.arms.set_leader_torque(False)
        if self.blend_s > 0 and last_command is not None:
            self._blend_start_t = self.clock()
            self._blend_from = np.asarray(last_command, np.float32).copy()
        else:
            self._blend_start_t = self._blend_from = None

    def human_command(self) -> np.ndarray:
        """Follower target during an intervention: the leader pose, blended in."""
        target = self.arms.read_leader()
        if self._blend_start_t is None or self._blend_from is None:
            return target
        alpha = min((self.clock() - self._blend_start_t) / self.blend_s, 1.0)
        if alpha >= 1.0:
            self._blend_start_t = self._blend_from = None
            return target
        return ((1.0 - alpha) * self._blend_from + alpha * target).astype(np.float32)

    def release(self) -> None:
        self.state = InterventionState.RELEASE
        self._blend_start_t = self._blend_from = None
        self.hold_leader()
