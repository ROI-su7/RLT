from __future__ import annotations

from pathlib import Path

import torch

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from toolkits.merge_rlt_replay_buffers import merge_buffers


def _create_buffer(path: Path, values: list[float]) -> None:
    buffer = TrajectoryReplayBuffer(
        auto_save=True,
        auto_save_path=str(path),
        enable_cache=True,
    )
    for value in values:
        buffer.add_trajectories(
            [
                Trajectory(
                    max_episode_length=1,
                    model_weights_id=f"weights-{value}",
                    actions=torch.tensor([[[value]]]),
                    rewards=torch.zeros(1, 1, 1),
                )
            ]
        )
    buffer._save_executor.shutdown(wait=True)
    buffer._save_metadata()
    buffer._save_trajectory_index()


def test_merge_skips_inherited_index_entries_without_local_files(
    tmp_path: Path,
) -> None:
    source_a = tmp_path / "source_a"
    source_b = tmp_path / "source_b"
    destination = tmp_path / "merged"
    _create_buffer(source_a, [1.0])
    _create_buffer(source_b, [2.0, 3.0])

    next(source_b.glob("trajectory_0_*.pt")).unlink()
    added, duplicates, missing = merge_buffers(
        destination,
        [source_a, source_b],
    )

    assert (added, duplicates, missing) == (2, 0, 1)
    merged = TrajectoryReplayBuffer(auto_save=False, enable_cache=True)
    merged.load_checkpoint(str(destination))
    assert merged.total_samples == 2


def test_preloaded_auto_save_materializes_source_files_and_checkpoint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    checkpoint = tmp_path / "checkpoint"
    _create_buffer(source, [1.0, 2.0])

    buffer = TrajectoryReplayBuffer(
        auto_save=True,
        auto_save_path=str(destination),
        enable_cache=True,
    )
    buffer.load_checkpoint(str(source))

    assert len(list(destination.glob("trajectory_*.pt"))) == 2
    # Sampling must resolve the materialized destination, not the source.
    buffer.sample_chunks(num_chunks=1)

    buffer.save_checkpoint(str(checkpoint))
    assert len(list(checkpoint.glob("trajectory_*.pt"))) == 2
    restored = TrajectoryReplayBuffer(auto_save=False, enable_cache=True)
    restored.load_checkpoint(str(checkpoint))
    assert restored.total_samples == 2
