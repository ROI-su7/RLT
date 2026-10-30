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

"""Wrist-camera pose of the two connector heads for the Piper wire task.

The Stage-2 RL token tells that an insertion is going wrong but not where
the heads are misaligned. These tools build a keypoint dataset from the
demonstration wrist videos (``extract_contact_frames``), let an operator
label it (``annotate``), and hold the shared geometry (``piper_fk``).
"""
