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

"""Extract wrist-camera frames around every insertion of the demonstrations.

Each demonstration mates three connector pairs. An insertion shows up as the
two flanges coming within ``--distance-mm`` of their relative position at a
completed insertion (``piper_fk.MATED_FLANGE_OFFSET_MM``). The heads are
already mated a few seconds before the closest point, so frames are picked by
geometry instead of time: for every insertion, the last frame before its
closest point at which the flanges were still ``--gaps-mm`` farther apart
than at the closest point (approach with a visible gap, first contact), plus
the closest point itself (gap 0, mated). Both wrist cameras are saved.

    source .venv-rlt/bin/activate
    python toolkits/wire_head_pose/extract_contact_frames.py \\
        --dataset ~/rlt_data/lerobot_data/all_wire_data \\
        --out ~/rlt_data/wire_head_pose --episodes 70 \\
        --events 0 --require-events 3 --cameras cam_right_wrist

``--events 0 --require-events 3`` keeps only the first of the three
connections (the Stage-2 RL connection) and only demos in which exactly three
insertions were found, so a retried insertion cannot shift the order.

Writes ``<out>/images/*.jpg`` and ``<out>/manifest.jsonl`` (one row per image).
Run again with another ``--seed`` to add more episodes; existing rows are kept.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import av
import cv2
import numpy as np
import pandas as pd

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from toolkits.wire_head_pose.piper_fk import distance_to_mated  # noqa: E402

CAMERAS = ("cam_left_wrist", "cam_right_wrist")


def insertion_events(
    distance_mm: np.ndarray, fps: float, threshold_mm: float, min_gap_s: float = 0.5
) -> list[int]:
    """Frame index of the closest approach of every insertion."""
    close = np.flatnonzero(distance_mm < threshold_mm)
    if close.size == 0:
        return []
    segments, start, prev = [], close[0], close[0]
    for index in close[1:]:
        if index - prev > min_gap_s * fps:
            segments.append((start, prev))
            start = index
        prev = index
    segments.append((start, prev))
    return [
        int(lo + np.argmin(distance_mm[lo : hi + 1]))
        for lo, hi in segments
        if hi - lo >= min_gap_s * fps
    ]


def approach_frame(
    distance_mm: np.ndarray, center: int, gap_mm: float, max_lookback: int
) -> int | None:
    """Last frame before ``center`` still ``gap_mm`` farther than at center."""
    if gap_mm <= 0:
        return center
    target = distance_mm[center] + gap_mm
    for index in range(center - 1, max(-1, center - max_lookback - 1), -1):
        if distance_mm[index] >= target:
            return index
    return None


def decode_frames(path: str, wanted: set[int]) -> dict[int, np.ndarray]:
    """BGR frames at the wanted indices of one video."""
    frames = {}
    if not wanted:
        return frames
    last = max(wanted)
    with av.open(path) as container:
        for index, frame in enumerate(container.decode(container.streams.video[0])):
            if index in wanted:
                frames[index] = frame.to_ndarray(format="bgr24")
            if index >= last:
                break
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--dataset", default=os.path.expanduser("~/rlt_data/lerobot_data/all_wire_data")
    )
    parser.add_argument("--out", default=os.path.expanduser("~/rlt_data/wire_head_pose"))
    parser.add_argument("--episodes", type=int, default=30, help="random demos to use")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distance-mm", type=float, default=60.0)
    parser.add_argument(
        "--gaps-mm",
        type=float,
        nargs="+",
        default=[0.0, 2.0, 5.0, 10.0, 20.0, 40.0],
        help="extra flange distance over the closest point of each insertion",
    )
    parser.add_argument(
        "--max-lookback-s",
        type=float,
        default=8.0,
        help="search window before the closest point",
    )
    parser.add_argument(
        "--events",
        type=int,
        nargs="*",
        default=None,
        help="insertion indices to keep in time order (default: all)",
    )
    parser.add_argument(
        "--require-events",
        type=int,
        default=0,
        help="only use demos with exactly this many insertions (0: any)",
    )
    parser.add_argument("--cameras", nargs="+", default=list(CAMERAS))
    args = parser.parse_args()

    info = json.load(open(os.path.join(args.dataset, "meta", "info.json")))
    fps = float(info["fps"])
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.jsonl")
    done = set()
    if os.path.exists(manifest_path):
        with open(manifest_path) as stream:
            done = {json.loads(line)["episode"] for line in stream}

    rng = np.random.default_rng(args.seed)
    candidates = [
        e for e in rng.permutation(int(info["total_episodes"])) if int(e) not in done
    ]
    written = used = 0
    with open(manifest_path, "a") as manifest:
        for episode in candidates:
            if used >= args.episodes:
                break
            episode = int(episode)
            chunk = episode // int(info["chunks_size"])
            data_path = os.path.join(
                args.dataset,
                info["data_path"].format(episode_chunk=chunk, episode_index=episode),
            )
            states = np.stack(
                pd.read_parquet(data_path)["observation.state"].to_numpy()
            )
            distance = distance_to_mated(states)
            picks = {}
            centers = insertion_events(distance, fps, args.distance_mm)
            if args.require_events and len(centers) != args.require_events:
                continue
            used += 1
            for event, center in enumerate(centers):
                if args.events is not None and event not in args.events:
                    continue
                for gap in args.gaps_mm:
                    frame = approach_frame(
                        distance, center, gap, int(args.max_lookback_s * fps)
                    )
                    if frame is not None:
                        picks[frame] = (event, gap)
            for camera in args.cameras:
                video = os.path.join(
                    args.dataset,
                    info["video_path"].format(
                        episode_chunk=chunk,
                        video_key=f"observation.images.{camera}",
                        episode_index=episode,
                    ),
                )
                for frame, image in sorted(decode_frames(video, set(picks)).items()):
                    name = f"ep{episode:06d}_f{frame:06d}_{camera}.jpg"
                    cv2.imwrite(
                        os.path.join(args.out, "images", name),
                        image,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                    )
                    event, gap = picks[frame]
                    row = {
                        "image": f"images/{name}",
                        "episode": episode,
                        "frame": frame,
                        "camera": camera,
                        "event": event,
                        "gap_mm": gap,
                        "offset_s": round((frame - centers[event]) / fps, 2),
                        "distance_mm": round(float(distance[frame]), 1),
                        "state": [round(float(x), 6) for x in states[frame]],
                    }
                    manifest.write(json.dumps(row) + "\n")
                    written += 1
            print(
                f"episode {episode}: {len(picks)} frames x {len(args.cameras)} cameras"
            )
    print(f"wrote {written} images to {args.out}")


if __name__ == "__main__":
    main()
