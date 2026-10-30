"""Lossless episode staging and non-destructive LeRobot export for Piper RECAP.

This module has no ROS, Ray, or model imports. Only completed C/X episodes are
exportable; interrupted recordings stay outside the training dataset.
"""

from __future__ import annotations

import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from rlinf.recap_piper.scoring import insertions_per_frame

CAMERAS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
# Lossless PNG at zlib level 1, one thread per camera: ~15 ms per 3x640x480
# frame on the robot host versus ~100 ms at the default level, which cannot
# sustain 15 fps.
PNG_COMPRESS_LEVEL = 1
# Per-frame shape-(1,) features written by the exporter.
SCALAR_FEATURES = (
    "intervention",
    "leader_valid",
    "policy_valid",
    "is_success",
    "done",
    "operator_reward",
    "source_timestamp",
    "insertions_done",
    "insertion_event",
)
JOINT_NAMES = [
    f"{side}_{joint}"
    for side in ("left", "right")
    for joint in ("j1", "j2", "j3", "j4", "j5", "j6", "gripper_m")
]


def vector14(value: Any) -> np.ndarray:
    """Validate physical absolute targets/state, without clipping or adaptation."""
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (14,) or not np.isfinite(value).all():
        raise ValueError("Expected a finite 14-D absolute joint/gripper vector.")
    return value


def atomic_json(path: Path, value: dict) -> None:
    """Publish a JSON file only after its complete contents reach disk."""
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class EpisodeWriter:
    """Stage one episode; publish its manifest after an explicit outcome."""

    def __init__(self, root: Path, task: str, fps: int = 15):
        if not task.strip() or fps <= 0:
            raise ValueError("A non-empty task and positive fps are required.")
        self.task, self.fps = task, int(fps)
        self.path = Path(root) / ("episode_" + uuid.uuid4().hex)
        self.path.mkdir(parents=True, exist_ok=False)
        self.stream = (self.path / "frames.jsonl").open("x", encoding="utf-8")
        self.count = 0
        self.closed = False
        self.last_timestamp = None
        self._pool = ThreadPoolExecutor(len(CAMERAS), thread_name_prefix="png")
        atomic_json(self.path / "recording.json", {"task": task, "fps": fps})

    def add_frame(self, frame: dict) -> None:
        """Persist synchronized observations, mux-selected targets and source flags."""
        if self.closed:
            raise RuntimeError("Episode is already closed.")
        row = {
            key: vector14(frame[key]).tolist()
            for key in ("state", "action", "policy_action", "leader_action")
        }
        timestamp = float(frame["timestamp"])
        if not np.isfinite(timestamp) or (
            self.last_timestamp is not None and timestamp <= self.last_timestamp
        ):
            raise ValueError("Episode timestamps must be finite and increasing.")
        images = {}
        for camera in CAMERAS:
            pixels = np.asarray(frame["images"][camera])
            if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 3:
                raise ValueError(f"{camera} must be RGB uint8 HWC.")
            images[camera] = pixels

        def save(camera: str) -> None:
            folder = self.path / camera
            folder.mkdir(exist_ok=True)
            Image.fromarray(images[camera]).save(
                folder / f"{self.count:06d}.png", compress_level=PNG_COMPRESS_LEVEL
            )

        # zlib releases the GIL, so the cameras encode in parallel; the row is
        # written only after all three images are on disk.
        for future in [self._pool.submit(save, camera) for camera in CAMERAS]:
            future.result()
        row.update(
            timestamp=timestamp,
            intervention=bool(frame["intervention"]),
            leader_valid=bool(frame.get("leader_valid", False)),
            policy_valid=bool(frame.get("policy_valid", False)),
        )
        self.stream.write(json.dumps(row, allow_nan=False) + "\n")
        self.stream.flush()
        self.last_timestamp = timestamp
        self.count += 1

    def finish(self, success: bool, score: dict | None = None) -> Path:
        """Commit one non-empty, explicitly labelled success/failure episode.

        ``score`` is the optional three-insertion progress record from
        :class:`rlinf.recap_piper.scoring.InsertionScore`.
        """
        if self.closed or self.count == 0:
            raise RuntimeError("Cannot label an empty or closed episode.")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        self._pool.shutdown(wait=True)
        self.closed = True
        atomic_json(
            self.path / "episode.json",
            {
                "schema_version": 1,
                "task": self.task,
                "fps": self.fps,
                "frames": self.count,
                "is_success": bool(success),
                "operator_reward": int(bool(success)),
                "action_units": "rad/metres",
                "action_source": "mux_arbitrated_command",
                "complete": True,
                **({"score": score} if score is not None else {}),
            },
        )
        return self.path

    def abort(self, reason: str) -> None:
        """Keep unlabelled data for inspection, excluding it from all exports."""
        if not self.closed:
            self.stream.close()
            self._pool.shutdown(wait=True)
            self.closed = True
            atomic_json(self.path / "aborted.json", {"reason": reason})


def completed_episodes(root: Path) -> list[Path]:
    """List committed episodes in stable order; ignore all partial recordings."""
    return sorted(path.parent for path in Path(root).glob("episode_*/episode.json"))


def intervention_segments(rows: list[dict], min_frames: int) -> list[tuple[int, int]]:
    """Contiguous human-controlled [start, end) ranges of at least min_frames."""
    segments, start = [], None
    for index, row in enumerate([*rows, {"intervention": False}]):
        if row["intervention"] and start is None:
            start = index
        elif not row["intervention"] and start is not None:
            if index - start >= min_frames:
                segments.append((start, index))
            start = None
    return segments


def export_lerobot(
    raw_root: Path,
    output: Path,
    repo_id: str,
    *,
    interventions_only: bool = False,
    min_segment_frames: int = 10,
    use_videos: bool = True,
) -> dict:
    """Export a new immutable dataset snapshot; never replace an existing one.

    Cameras are stored as AV1 videos by default, the same LeRobot encoding as
    the source demonstrations (``use_videos=False`` keeps per-frame PNGs).

    With ``interventions_only``, every contiguous human-controlled segment of a
    successful episode becomes its own all-success episode (the Evo-RL
    "SFT + interventions" data); autonomous frames and failures are dropped.
    """
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    output = Path(output).absolute()
    if output.exists():
        raise FileExistsError(
            f"Choose a new export directory; will not overwrite {output}"
        )
    if min_segment_frames < 1:
        raise ValueError("min_segment_frames must be positive.")
    episodes = completed_episodes(raw_root)
    if not episodes:
        raise ValueError("No C/X-labelled episodes to export.")
    first_meta = json.loads((episodes[0] / "episode.json").read_text())
    fps = first_meta["fps"]
    first_image = np.asarray(Image.open(episodes[0] / CAMERAS[0] / "000000.png"))
    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
        "action": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
        "policy_action": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
        "leader_action": {"dtype": "float32", "shape": (14,), "names": JOINT_NAMES},
        "intervention": {"dtype": "bool", "shape": (1,), "names": None},
        "leader_valid": {"dtype": "bool", "shape": (1,), "names": None},
        "policy_valid": {"dtype": "bool", "shape": (1,), "names": None},
        "is_success": {"dtype": "bool", "shape": (1,), "names": None},
        "done": {"dtype": "bool", "shape": (1,), "names": None},
        "operator_reward": {"dtype": "float32", "shape": (1,), "names": None},
        "source_timestamp": {"dtype": "float64", "shape": (1,), "names": None},
        # Operator-marked three-insertion progress (0 for unscored episodes).
        "insertions_done": {"dtype": "int64", "shape": (1,), "names": None},
        "insertion_event": {"dtype": "bool", "shape": (1,), "names": None},
    }
    for camera in CAMERAS:
        features[f"observation.images.{camera}"] = {
            "dtype": "video" if use_videos else "image",
            "shape": first_image.shape,
            "names": ["height", "width", "channel"],
        }
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=output,
        fps=fps,
        robot_type="piper",
        features=features,
        use_videos=use_videos,
        image_writer_processes=0,
        # PNG encoding releases the GIL; parallel writes cut export time ~3x.
        image_writer_threads=8,
    )
    atomic_json(output / "meta/recap_export_incomplete.json", {"source": str(raw_root)})
    exported, frames, successes = [], 0, 0
    for episode in episodes:
        meta = json.loads((episode / "episode.json").read_text())
        if meta["fps"] != fps or not meta["complete"]:
            raise ValueError(f"Inconsistent episode metadata: {episode}")
        rows = [
            json.loads(line)
            for line in (episode / "frames.jsonl").read_text().splitlines()
        ]
        if len(rows) != meta["frames"] or not rows:
            raise ValueError(f"Incomplete recording: {episode}")
        if not interventions_only:
            units = [(0, len(rows))]
        elif meta["is_success"]:
            units = intervention_segments(rows, min_segment_frames)
        else:
            units = []
        for start, end in units:
            success = meta["is_success"] or interventions_only
            progress = insertions_per_frame(meta.get("score"), len(rows))
            add_episode(
                dataset, episode, rows, start, end, meta["task"], success, progress
            )
            exported.append(
                {
                    "episode_index": len(exported),
                    "source": str(episode),
                    "frames": [start, end],
                }
            )
            frames += end - start
            successes += int(success)
    if hasattr(dataset, "stop_image_writer") and dataset.image_writer is not None:
        dataset.stop_image_writer()
    if not exported:
        raise ValueError(
            "No successful human-intervention segments to export."
            if interventions_only
            else "No frames were exported."
        )
    report = {
        "episodes": len(exported),
        "frames": frames,
        "successes": successes,
        "failures": len(exported) - successes,
        "interventions_only": interventions_only,
        "source_episodes": exported,
    }
    atomic_json(output / "meta" / "piper_recap_manifest.json", report)
    (output / "meta/recap_export_incomplete.json").unlink()
    return report


def add_episode(dataset, episode, rows, start, end, task, success, progress) -> None:
    """Append frames [start, end) of one raw episode as a LeRobot episode.

    ``progress`` is the per-raw-frame completed insertion count.
    """
    camera_shape = tuple(
        dataset.meta.features[f"observation.images.{CAMERAS[0]}"]["shape"]
    )
    for index in range(start, end):
        row = rows[index]
        terminal = index == end - 1
        frame = {
            "observation.state": vector14(row["state"]),
            **{
                key: vector14(row[key])
                for key in ("action", "policy_action", "leader_action")
            },
            "task": task,
            "intervention": np.array([row["intervention"]], dtype=bool),
            "leader_valid": np.array([row["leader_valid"]], dtype=bool),
            "policy_valid": np.array([row["policy_valid"]], dtype=bool),
            "is_success": np.array([terminal and success], dtype=bool),
            "done": np.array([terminal], dtype=bool),
            "operator_reward": np.array([float(terminal and success)], np.float32),
            "source_timestamp": np.array([row["timestamp"]], np.float64),
            "insertions_done": np.array([progress[index]], np.int64),
            "insertion_event": np.array(
                [index > 0 and progress[index] > progress[index - 1]], dtype=bool
            ),
        }
        for camera in CAMERAS:
            with Image.open(episode / camera / f"{index:06d}.png") as image:
                pixels = np.asarray(image.convert("RGB"))
            if pixels.shape != camera_shape:
                raise ValueError(f"Camera resolution changed in {episode}: {camera}")
            frame[f"observation.images.{camera}"] = pixels
        dataset.add_frame(frame)
    # LeRobot maps shape=(1,) to a scalar HF Value. Unwrap its required
    # per-frame [1] arrays locally for NumPy >=2, without site-package edits.
    for key in SCALAR_FEATURES:
        dataset.episode_buffer[key] = [
            np.asarray(value).reshape(-1)[0].item()
            for value in dataset.episode_buffer[key]
        ]
    dataset.save_episode()
