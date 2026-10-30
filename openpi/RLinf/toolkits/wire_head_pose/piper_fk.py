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

"""Vectorized Piper forward kinematics (piper_sdk C_PiperForwardKinematics).

Modified DH parameters of piper_sdk with ``dh_is_offset=1``; positions in mm.
"""

import numpy as np

_A = (0.0, 0.0, 285.03, -21.98, 0.0, 0.0)
_ALPHA = (0.0, -np.pi / 2, 0.0, np.pi / 2, -np.pi / 2, np.pi / 2)
_THETA0 = (0.0, -np.pi * 172.22 / 180, -np.pi * 102.78 / 180, 0.0, 0.0, 0.0)
_D = (123.0, 0.0, 0.0, 250.75, 0.0, 91.0)

# Flange of the left arm minus flange of the right arm (each in its own base
# frame; the two bases are parallel) at a completed insertion. Mean of 69
# successful Stage-2 episodes on 2026-10-20, spread about 6 mm per axis
# because the heads sit differently in the grippers from grasp to grasp.
MATED_FLANGE_OFFSET_MM = np.array([82.7, -443.7, 21.1])


def _link(alpha: float, a: float, theta: np.ndarray, d: float) -> np.ndarray:
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    transform = np.zeros((theta.shape[0], 4, 4))
    transform[:, 0, 0], transform[:, 0, 1], transform[:, 0, 3] = ct, -st, a
    transform[:, 1, 0], transform[:, 1, 1] = st * ca, ct * ca
    transform[:, 1, 2], transform[:, 1, 3] = -sa, -sa * d
    transform[:, 2, 0], transform[:, 2, 1] = st * sa, ct * sa
    transform[:, 2, 2], transform[:, 2, 3] = ca, ca * d
    transform[:, 3, 3] = 1.0
    return transform


def flange_pose(joints: np.ndarray) -> np.ndarray:
    """(n, 6) joint angles in rad -> (n, 4, 4) flange transform in mm."""
    joints = np.atleast_2d(np.asarray(joints, dtype=np.float64))
    transform = np.tile(np.eye(4), (joints.shape[0], 1, 1))
    for i in range(6):
        transform = transform @ _link(
            _ALPHA[i], _A[i], joints[:, i] + _THETA0[i], _D[i]
        )
    return transform


def flange_offset(states: np.ndarray) -> np.ndarray:
    """(n, 14) Cobot Magic states -> left minus right flange position (n, 3)."""
    states = np.atleast_2d(np.asarray(states, dtype=np.float64))
    return (
        flange_pose(states[:, 0:6])[:, :3, 3] - flange_pose(states[:, 7:13])[:, :3, 3]
    )


def distance_to_mated(states: np.ndarray) -> np.ndarray:
    """Distance (mm) of the flange offset from a completed insertion."""
    return np.linalg.norm(flange_offset(states) - MATED_FLANGE_OFFSET_MM, axis=1)
