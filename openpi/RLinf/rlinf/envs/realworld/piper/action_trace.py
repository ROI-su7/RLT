"""Opt-in, non-blocking action-chain diagnostics for Piper real-robot runs."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


class PiperActionTraceWriter:
    """Write a small, bounded set of diagnostic snapshots when explicitly enabled.

    The writer is disabled unless ``RLINF_PIPER_ACTION_TRACE_DIR`` is set.  Trace
    I/O is always dispatched to a daemon thread so an unavailable filesystem can
    never delay the real-time Piper command loop.  The process-local event cap
    defaults to eight and can be changed with
    ``RLINF_PIPER_ACTION_TRACE_MAX_EVENTS``.
    """

    def __init__(self, directory: str | os.PathLike[str] | None, component: str):
        self._directory = Path(directory) if directory else None
        self._component = component
        self._lock = threading.Lock()
        self._events_scheduled = 0
        try:
            self._max_events = max(
                1, int(os.environ.get("RLINF_PIPER_ACTION_TRACE_MAX_EVENTS", "8"))
            )
        except ValueError:
            self._max_events = 8

    @classmethod
    def from_environment(cls, component: str) -> "PiperActionTraceWriter":
        return cls(os.environ.get("RLINF_PIPER_ACTION_TRACE_DIR"), component)

    @property
    def enabled(self) -> bool:
        return self._directory is not None

    def schedule(self, kind: str, **arrays: Any) -> None:
        """Queue one snapshot without blocking the control caller."""
        if self._directory is None:
            return
        with self._lock:
            if self._events_scheduled >= self._max_events:
                return
            self._events_scheduled += 1
        thread = threading.Thread(
            target=self._write,
            args=(kind, arrays),
            name=f"piper-trace-{self._component}",
            daemon=True,
        )
        thread.start()

    def _write(self, kind: str, arrays: dict[str, Any]) -> None:
        """Persist one compressed trace; intentionally swallow diagnostics errors."""
        if self._directory is None:
            return
        temporary_path: Path | None = None
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            metadata = {
                "component": self._component,
                "kind": kind,
                "monotonic_ns": time.monotonic_ns(),
                "wall_time_ns": time.time_ns(),
            }
            payload = {
                name: np.ascontiguousarray(value)
                for name, value in arrays.items()
                if value is not None
            }
            payload["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
            path = self._directory / (
                f"{self._component}_{kind}_{metadata['monotonic_ns']}.npz"
            )
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=self._directory, suffix=".tmp", delete=False
            ) as handle:
                temporary_path = Path(handle.name)
                np.savez_compressed(handle, **payload)
            os.replace(temporary_path, path)
        except Exception:
            # This path is diagnostic only.  In particular, never let a full or
            # unavailable filesystem affect mechanical command publication.
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            return
