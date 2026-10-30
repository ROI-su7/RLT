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

"""Full-resolution camera log written beside the RLT replay.

Replay transitions keep only the RL token of each observation, and the policy
already sees every camera downscaled to 224x224. This log keeps selected raw
camera frames of every executed step so features that need pixels (e.g. the
relative pose of the two connector heads in the wrist cameras) can be
computed later and joined to the transitions: a transition's
``curr_obs.proprio`` / ``next_obs.proprio`` equal the logged ``states`` of its
step bit for bit (see ``match_frame``).

Layout::

    <root>/episode_<rlt_episode_id>/frames.jsonl   one JSON object per step
    <root>/episode_<rlt_episode_id>/<frame:06d>_<camera>.jpg

Frames are JPEG-encoded on a background thread so the control loop never
waits on disk; when the queue is full, a step is dropped and counted.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from typing import Any

import cv2
import numpy as np

from rlinf.utils.logging import get_logger


class RLTImageLog:
    """Asynchronously store raw camera frames of executed steps per episode."""

    def __init__(
        self, root_dir: str, jpeg_quality: int = 90, max_queued_steps: int = 256
    ):
        self.root_dir = root_dir
        self._jpeg_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        self._queue: queue.Queue = queue.Queue(maxsize=max_queued_steps)
        self._frame_counters: dict[int, int] = {}
        self.dropped_steps = 0
        self.written_steps = 0
        self._logger = get_logger()
        os.makedirs(root_dir, exist_ok=True)
        self._thread = threading.Thread(
            target=self._run, name="rlt-image-log", daemon=True
        )
        self._thread.start()

    def submit(
        self, episode_id: int, states: np.ndarray, frames: dict[str, np.ndarray]
    ) -> None:
        """Queue one step: 14-D states and BGR uint8 frames by camera name."""
        item = (int(episode_id), time.time(), np.asarray(states, np.float32), frames)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self.dropped_steps += 1
            if self.dropped_steps == 1 or self.dropped_steps % 100 == 0:
                self._logger.warning(
                    f"RLT image log is behind; dropped {self.dropped_steps} steps."
                )

    def close(self, timeout: float = 30.0) -> None:
        """Write everything queued, then stop the writer thread."""
        self._queue.put(None)
        self._thread.join(timeout=timeout)

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            try:
                self._write(*item)
            except Exception as error:  # Never take the rollout down.
                self._logger.warning(f"RLT image log write failed: {error}")

    def _write(
        self,
        episode_id: int,
        wall_time: float,
        states: np.ndarray,
        frames: dict[str, np.ndarray],
    ) -> None:
        directory = os.path.join(self.root_dir, f"episode_{episode_id}")
        os.makedirs(directory, exist_ok=True)
        frame = self._frame_counters.get(episode_id, 0)
        self._frame_counters[episode_id] = frame + 1
        written = []
        for camera, image in frames.items():
            ok, encoded = cv2.imencode(
                ".jpg", np.ascontiguousarray(image), self._jpeg_params
            )
            if not ok:
                continue
            with open(os.path.join(directory, f"{frame:06d}_{camera}.jpg"), "wb") as f:
                f.write(encoded.tobytes())
            written.append(camera)
        row = {
            "frame": frame,
            "wall_time": wall_time,
            "states": states.reshape(-1).tolist(),
            "cameras": written,
        }
        with open(os.path.join(directory, "frames.jsonl"), "a") as stream:
            stream.write(json.dumps(row) + "\n")
        self.written_steps += 1


def load_frames(episode_dir: str) -> list[dict[str, Any]]:
    """Rows of one logged episode with ``states`` as a float32 array."""
    rows = []
    with open(os.path.join(episode_dir, "frames.jsonl")) as stream:
        for line in stream:
            row = json.loads(line)
            row["states"] = np.asarray(row["states"], dtype=np.float32)
            rows.append(row)
    return rows


def match_frame(rows: list[dict[str, Any]], proprio: Any) -> dict[str, Any] | None:
    """The logged step whose states equal a transition's proprio exactly.

    float32 -> JSON -> float32 is exact, so the comparison is bitwise. When the
    arms were still and several steps share the state, the last one is used.
    """
    target = np.asarray(proprio, dtype=np.float32).reshape(-1)
    found = None
    for row in rows:
        if row["states"].shape == target.shape and np.array_equal(
            row["states"], target
        ):
            found = row
    return found
