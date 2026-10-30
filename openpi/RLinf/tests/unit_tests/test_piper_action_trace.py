from __future__ import annotations

import json
import time

import numpy as np

from rlinf.envs.realworld.piper.action_trace import PiperActionTraceWriter


def test_action_trace_is_disabled_without_explicit_directory(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.delenv("RLINF_PIPER_ACTION_TRACE_DIR", raising=False)
    writer = PiperActionTraceWriter.from_environment("env")

    writer.schedule("rtc_alignment", raw_chunk=np.zeros((1, 14)))

    assert not writer.enabled
    assert list(tmp_path.iterdir()) == []


def test_action_trace_writes_bounded_snapshot(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RLINF_PIPER_ACTION_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("RLINF_PIPER_ACTION_TRACE_MAX_EVENTS", "1")
    writer = PiperActionTraceWriter.from_environment("env")

    raw = np.arange(28, dtype=np.float32).reshape(2, 14)
    writer.schedule("rtc_alignment", raw_chunk=raw, ignored_none=None)
    writer.schedule("rtc_alignment", raw_chunk=np.zeros_like(raw))

    deadline = time.monotonic() + 2.0
    traces = []
    while time.monotonic() < deadline:
        traces = list(tmp_path.glob("env_rtc_alignment_*.npz"))
        if traces:
            break
        time.sleep(0.01)
    assert len(traces) == 1
    with np.load(traces[0]) as trace:
        np.testing.assert_array_equal(trace["raw_chunk"], raw)
        assert "ignored_none" not in trace.files
        metadata = json.loads(str(trace["metadata"]))
    assert metadata["component"] == "env"
    assert metadata["kind"] == "rtc_alignment"
