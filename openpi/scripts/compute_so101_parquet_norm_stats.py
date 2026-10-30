"""Normalization statistics for pi05_so101_dual(_lora) straight from the LeRobot parquet files (no video decoding).

Equivalent to compute_norm_stats.py for these configs: SO101 inputs are the raw 12-dim joint vectors, and the
actions are converted to deltas w.r.t. the current state for the 5 arm joints of each arm (grippers stay absolute).
"""

import pathlib

import numpy as np
import pyarrow.parquet as pq
import tqdm
import tyro

from openpi.shared import normalize


def _action_windows(actions: np.ndarray, horizon: int, batch_size: int):
    n = len(actions)
    offsets = np.arange(horizon)
    for start in range(0, n, batch_size):
        idx = start + np.arange(min(batch_size, n - start))[:, None] + offsets
        yield start, actions[np.minimum(idx, n - 1)]


def main(
    dataset_root: pathlib.Path,
    output_dir: pathlib.Path = pathlib.Path("assets/pi05_so101_dual/so101_dual"),
    action_horizon: int = 50,
    batch_size: int = 4096,
):
    files = sorted((dataset_root / "data").glob("chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet episodes found under {dataset_root / 'data'}")
    stats = {"state": normalize.RunningStats(), "actions": normalize.RunningStats()}
    for path in tqdm.tqdm(files, desc="Computing stats", unit="episode"):
        table = pq.read_table(path, columns=["observation.state", "action"])
        state = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
        actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
        stats["state"].update(state)
        for start, window in _action_windows(actions, action_horizon, batch_size):
            window = window.copy()
            s = state[start : start + len(window)]
            window[..., :5] -= s[:, None, :5]
            window[..., 6:11] -= s[:, None, 6:11]
            stats["actions"].update(window)
    normalize.save(output_dir, {k: v.get_statistics() for k, v in stats.items()})
    print("Wrote", output_dir)


if __name__ == "__main__":
    tyro.cli(main)
