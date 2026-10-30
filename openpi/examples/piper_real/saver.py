from __future__ import annotations

import logging
import pathlib

import imageio
import numpy as np
from openpi_client.runtime import subscriber as _subscriber
from typing_extensions import override


class VideoSaver(_subscriber.Subscriber):
    """Saves rollout videos from Piper observations."""

    def __init__(self, out_dir: pathlib.Path, *, fps: int, subsample: int = 1) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self._out_dir = out_dir
        self._fps = fps
        self._subsample = max(1, subsample)
        self._frames: list[np.ndarray] = []

    @override
    def on_episode_start(self) -> None:
        self._frames = []

    def _make_frame(self, observation: dict) -> np.ndarray:
        image_names = ("cam_high", "cam_left_wrist", "cam_right_wrist")
        panels = []
        for name in image_names:
            image = observation["images"][name]  # [C, H, W], RGB uint8
            image = np.transpose(image, (1, 2, 0))  # [H, W, C]
            panels.append(np.asarray(image))
        return np.concatenate(panels, axis=1)

    @override
    def on_step(self, observation: dict, action: dict) -> None:
        del action
        self._frames.append(self._make_frame(observation))

    @override
    def on_episode_end(self) -> None:
        if not self._frames:
            logging.warning("No frames captured for Piper rollout; skipping video save.")
            return

        existing = list(self._out_dir.glob("episode_*.mp4"))
        next_idx = max((int(path.stem.split("_")[1]) for path in existing), default=-1) + 1
        out_path = self._out_dir / f"episode_{next_idx:04d}.mp4"

        logging.info("Saving Piper rollout video to %s", out_path)
        imageio.mimwrite(
            out_path,
            [np.asarray(frame) for frame in self._frames[:: self._subsample]],
            fps=max(1, self._fps // self._subsample),
        )
