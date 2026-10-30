"""Verify alignment commitment and native RTC lifecycle without ROS."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from rlinf.recap_piper.client import alignment_horizon, make_motion_broker


def test_only_stable_hold_extends_execution():
    actions = np.zeros((50, 14), dtype=np.float32)
    assert alignment_horizon(actions, True, 5, 10) == 30
    assert alignment_horizon(actions, False, 5, 10) == 10
    assert alignment_horizon(actions, True, 25, 10) == 10
    for grip in (6, 13):
        for step in (0, 12, 34):
            changed = actions.copy()
            changed[step, grip] = 0.05
            assert alignment_horizon(changed, True, 5, 10) == 10


def test_native_prefetch_promotion_release_and_reset(monkeypatch):
    native_root = Path(
        os.environ.get("RECAP_NATIVE_OPENPI", Path(__file__).resolve().parents[3])
    )
    path = native_root / "examples/piper_real/rtc_action_chunk_broker.py"
    if not path.exists():
        pytest.skip("Native OpenPI client not installed")
    spec = importlib.util.spec_from_file_location("native_recap_rtc_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    actions = np.zeros((50, 14), dtype=np.float32)

    class Policy:
        def infer(self, obs):
            return {"actions": actions.copy()}

        def reset(self):
            pass

    broker = make_motion_broker(module.RealtimeActionChunkBroker)(
        Policy(), action_horizon=10, prefix_len=4, control_hz=15, boundary_blend_steps=5
    )
    prefetches = []

    def prefetch(obs):
        prefetches.append((broker._action_horizon, broker._active_index))
        broker._pending_result = (actions.copy(), clock[0])

    monkeypatch.setattr(broker, "_start_prefetch", prefetch)

    def step(open_grippers=False):
        state = np.zeros(14, dtype=np.float32)
        state[[6, 13]] = 0.05 if open_grippers else 0.016
        clock[0] += 1 / 15
        broker.infer({"state": state})

    # Initially closed (home) must not be interpreted as acquired objects.
    for _ in range(11):
        step()
    assert broker._action_horizon == 10
    step(True)
    for _ in range(9):
        step()
    assert broker._action_horizon == 30
    for _ in range(25):
        step()
    assert (10, 6) in prefetches
    assert (30, 26) in prefetches
    # A planned release causes a short next chunk. No transition is skipped.
    broker._pending_result[0][20, 6] = 0.05
    for _ in range(5):
        step()
    assert broker._action_horizon == 10
    broker.reset()
    assert broker._action_horizon == 10
    assert not broker._seen_open.any()
    assert broker._stable_hold_count == 0
    assert broker._active_actions is None


def test_long_horizon_can_be_disabled_for_ab_tests():
    import numpy as np

    from rlinf.recap_piper.client import alignment_horizon

    closed = np.full((50, 14), 0.01, np.float32)
    assert alignment_horizon(closed, True, 4.0, 10) == 30
    assert alignment_horizon(closed, True, 4.0, 10, long=10) == 10
