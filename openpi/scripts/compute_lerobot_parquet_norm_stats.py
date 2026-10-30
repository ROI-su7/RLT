"""Compute pi05_wire normalization statistics without decoding LeRobot videos.

This is an optimized equivalent of ``compute_norm_stats.py --config-name pi05_wire``
for a LeRobot v2 dataset whose parquet files contain ``observation.state`` and
``action``. It applies pi05_wire's Aloha state/action and delta-action transforms,
but reads no video frames.
"""

import os
import pathlib

import numpy as np
import pyarrow.parquet as pq
import tqdm
import tyro

from openpi.policies import aloha_policy
from openpi.shared import normalize


def _transform_state(state: np.ndarray) -> np.ndarray:
    """Vectorized version of AlohaInputs' state conversion."""
    state = state.copy()
    state *= aloha_policy._joint_flip_mask()
    state[:, [6, 13]] = aloha_policy._gripper_to_angular(state[:, [6, 13]])
    return state


def _transform_actions(actions: np.ndarray, state: np.ndarray) -> np.ndarray:
    """Apply AlohaInputs and DeltaActions exactly as pi05_wire does."""
    actions = actions.copy()
    actions *= aloha_policy._joint_flip_mask()
    actions[..., [6, 13]] = aloha_policy._gripper_from_angular_inv(actions[..., [6, 13]])
    # pi05_wire uses ``make_bool_mask(6, -1, 6, -1)``: convert both
    # six-joint arms to deltas, while leaving both grippers absolute.
    actions[..., :6] -= state[:, None, :6]
    actions[..., 7:13] -= state[:, None, 7:13]
    return actions


def _action_windows(actions: np.ndarray, horizon: int, batch_size: int):
    """Yield LeRobot delta-timestamp action windows, clamped at episode ends."""
    frame_count = len(actions)
    offsets = np.arange(horizon)
    for start in range(0, frame_count, batch_size):
        indices = start + np.arange(min(batch_size, frame_count - start))[:, None] + offsets
        yield start, actions[np.minimum(indices, frame_count - 1)]


def main(
    dataset_root: pathlib.Path = pathlib.Path(os.path.expanduser("~/rlt_data/lerobot_data/new_total_data")),
    output_dir: pathlib.Path = pathlib.Path("assets/pi05_wire/new_total_data"),
    action_horizon: int = 10,
    batch_size: int = 4096,
):
    """Write norm_stats.json compatible with the pi05_wire training config."""
    files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet episodes found under {dataset_root / 'data'}")

    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    columns = ["observation.state", "action"]
    for path in tqdm.tqdm(files, desc="Computing stats", unit="episode"):
        table = pq.read_table(path, columns=columns)
        state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
        actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        state = _transform_state(state)
        stats["state"].update(state)
        for start, action_window in _action_windows(actions, action_horizon, batch_size):
            stats["actions"].update(_transform_actions(action_window, state[start : start + len(action_window)]))

    norm_stats = {key: value.get_statistics() for key, value in stats.items()}
    print(f"Writing stats to: {output_dir}")
    normalize.save(output_dir, norm_stats)


if __name__ == "__main__":
    tyro.cli(main)
