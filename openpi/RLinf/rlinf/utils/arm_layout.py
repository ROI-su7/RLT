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

"""Per-arm action layout shared by the RLT actor, critic and execution code."""

def arm_width(action_dim: int) -> int:
    """Width of one arm block (joints + gripper) in an action vector.

    Piper arms are 7-D (6 joints + gripper, e.g. 14-D dual arm). SO101 arms are
    6-D (5 joints + gripper, 12-D dual arm): used when ``action_dim`` divides by
    6 but not by 7. Every other width keeps the historical 7-D layout.
    """
    return 6 if action_dim % 6 == 0 and action_dim % 7 != 0 else 7


def gripper_slice(action_dim: int) -> slice:
    """Slice selecting each arm's gripper (the last entry of every arm block)."""
    width = arm_width(action_dim)
    return slice(width - 1, None, width)


def is_arm_joint(index: int, action_dim: int) -> bool:
    width = arm_width(action_dim)
    return index % width != width - 1
