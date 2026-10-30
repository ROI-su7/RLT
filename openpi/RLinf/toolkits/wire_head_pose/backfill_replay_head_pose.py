#!/usr/bin/env python3
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

"""Add ``head_pose`` to every transition of a Stage-2 replay buffer.

The replay batch schema is fixed by its first row, so once the env reports
the connector-head state every stored transition needs it too. Each
transition's current and next observation is joined to its logged right
wrist frame (``rlinf.data.rlt_image_log.match_frame`` on the proprio) in the
``rlt_images`` of the given runs and run through the detector. Transitions
without a logged frame (collected before the image log existed) get zeros,
i.e. "not visible". The source replay is left untouched; a new replay
directory is written for ``RLINF_REPLAY_LOAD_PATH``.

    python toolkits/wire_head_pose/backfill_replay_head_pose.py \
        --replay logs/<run>/replay_buffer/rank_0 --runs logs/<run> \
        --dst logs/<run>/replay_buffer_head_pose/rank_0
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import shutil
import sys

import cv2
import numpy as np
import torch

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from rlinf.data.rlt_image_log import load_frames, match_frame  # noqa: E402
from rlinf.envs.realworld.piper.wire_head_pose import (  # noqa: E402
    HEAD_POSE_DIM,
    WireHeadPoseDetector,
)

CAMERA = "cam_right_wrist"


class FrameHeadPose:
    """Head pose of the logged right-wrist frame matching a proprio, if any."""

    def __init__(self, runs: list[str], detector):
        self.runs = runs
        self.detector = detector
        self._frames: dict[int, tuple[str, list] | None] = {}

    def _episode(self, episode_id: int) -> tuple[str, list] | None:
        if episode_id not in self._frames:
            self._frames[episode_id] = None
            for run in self.runs:
                directory = os.path.join(run, "rlt_images", f"episode_{episode_id}")
                if os.path.isdir(directory):
                    self._frames[episode_id] = (directory, load_frames(directory))
                    break
        return self._frames[episode_id]

    def __call__(self, episode_id: int, proprio: np.ndarray) -> np.ndarray | None:
        episode = self._episode(episode_id)
        if episode is None:
            return None
        directory, frames = episode
        hit = match_frame(frames, proprio)
        if hit is None or CAMERA not in hit["cameras"]:
            return None
        image = cv2.imread(os.path.join(directory, f"{hit['frame']:06d}_{CAMERA}.jpg"))
        return None if image is None else self.detector(image)


def backfill(replay: str, dst: str, head_pose: FrameHeadPose) -> dict[str, int]:
    if os.path.exists(dst):
        raise FileExistsError(f"{dst} exists; choose a new destination.")
    with open(os.path.join(replay, "trajectory_index.json")) as stream:
        index = json.load(stream)["trajectory_index"]
    os.makedirs(dst)
    counts = collections.Counter()
    for path in sorted(glob.glob(os.path.join(replay, "trajectory_*.pt"))):
        trajectory_id = os.path.basename(path).split("_")[1]
        episode_id = int(index[trajectory_id]["rlt_episode_id"])
        row = torch.load(path, map_location="cpu", weights_only=False)
        for side in ("curr_obs", "next_obs"):
            obs = row[side]
            proprio = obs["proprio"].reshape(-1).numpy()
            value = head_pose(episode_id, proprio)
            if value is None:
                counts[f"{side}_no_frame"] += 1
                value = np.zeros(HEAD_POSE_DIM, np.float32)
            else:
                counts[f"{side}_matched"] += 1
                counts[f"{side}_visible"] += int(value[3] > 0.5)
            shape = (*obs["proprio"].shape[:-1], HEAD_POSE_DIM)
            obs["head_pose"] = torch.as_tensor(value, dtype=torch.float32).reshape(
                shape
            )
        torch.save(row, os.path.join(dst, os.path.basename(path)))
        counts["transitions"] += 1
    for name in ("metadata.json", "trajectory_index.json"):
        shutil.copy2(os.path.join(replay, name), os.path.join(dst, name))
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--replay", required=True, help="replay_buffer/rank_0")
    parser.add_argument(
        "--runs", nargs="+", required=True, help="run dirs holding rlt_images"
    )
    parser.add_argument("--dst", required=True)
    parser.add_argument(
        "--weights", default=os.path.expanduser("~/rlt_data/wire_head_pose/head_pose_net.pt")
    )
    args = parser.parse_args()
    counts = backfill(
        args.replay,
        args.dst,
        FrameHeadPose(args.runs, WireHeadPoseDetector(args.weights)),
    )
    for key, value in sorted(counts.items()):
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
