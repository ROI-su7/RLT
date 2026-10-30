#!/usr/bin/env python3
"""Merge persisted RLinf replay buffers into a new, validated buffer.

The source buffers are read only.  Every saved transition is loaded and added
to a fresh :class:`TrajectoryReplayBuffer`, which assigns new trajectory IDs
and writes a consistent ``metadata.json`` and ``trajectory_index.json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rlinf.data.replay_buffer import TrajectoryReplayBuffer


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_all(source: Path) -> TrajectoryReplayBuffer:
    """Load an index without eagerly opening files referenced by that index."""
    with (source / "trajectory_index.json").open() as stream:
        index_data = json.load(stream)
    buffer = TrajectoryReplayBuffer(
        auto_save=False,
        enable_cache=True,
        sample_window_size=1,
    )
    buffer._trajectory_index = {
        int(key): value for key, value in index_data.get("trajectory_index", {}).items()
    }
    buffer._trajectory_id_list = [
        int(value) for value in index_data.get("trajectory_id_list", [])
    ]
    buffer._trajectory_file_path = {
        trajectory_id: str(source) for trajectory_id in buffer._trajectory_id_list
    }
    return buffer


def merge_buffers(
    destination: Path,
    sources: list[Path],
    *,
    deduplicate: bool = False,
) -> tuple[int, int, int]:
    """Merge source buffers and return added, duplicate, and missing counts.

    A resumed replay index may retain entries that still live in its original
    preload directory. Those files are deliberately skipped here because the
    original preload directory should be supplied as a separate source.
    """
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    destination.mkdir(parents=True)
    merged = TrajectoryReplayBuffer(
        auto_save=True,
        auto_save_path=str(destination),
        enable_cache=True,
        cache_size=200,
        sample_window_size=20_000,
    )
    seen_hashes: set[str] = set()
    added = 0
    duplicates = 0
    missing = 0
    for source in sources:
        source_buffer = _load_all(source)
        for trajectory_id in source_buffer._trajectory_id_list:
            info = source_buffer._trajectory_index[trajectory_id]
            model_weights_id = info["model_weights_id"]
            source_file = Path(
                source_buffer._get_trajectory_path(
                    trajectory_id, model_weights_id, base_dir=str(source)
                )
            )
            if not source_file.is_file():
                missing += 1
                continue
            if deduplicate:
                content_hash = _file_sha256(source_file)
                if content_hash in seen_hashes:
                    duplicates += 1
                    continue
                seen_hashes.add(content_hash)
            merged.add_trajectories(
                [source_buffer._load_trajectory(trajectory_id, model_weights_id)]
            )
            added += 1
    merged._save_executor.shutdown(wait=True)
    merged._save_metadata()
    merged._save_trajectory_index()
    return added, duplicates, missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Drop byte-identical trajectory files. Disabled by default because "
        "identical robot transitions are valid replay samples.",
    )
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()

    for source in args.sources:
        if (
            not (source / "metadata.json").is_file()
            or not (source / "trajectory_index.json").is_file()
        ):
            raise FileNotFoundError(f"not a persisted replay buffer: {source}")
    added, duplicates, missing = merge_buffers(
        args.destination,
        args.sources,
        deduplicate=args.deduplicate,
    )
    print(
        f"merged transitions={added}, skipped_duplicates={duplicates}, "
        f"skipped_missing_index_entries={missing}"
    )


if __name__ == "__main__":
    main()
