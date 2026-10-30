"""Offline handover regressions without ROS imports or robot commands."""

import numpy as np
import pytest

from rlinf.recap_piper.client import blend_arm_boundary


def test_constant_velocity_does_not_restart_from_rest():
    delta = np.full(14, 0.01, dtype=np.float32)
    last = np.zeros(14, dtype=np.float32)
    actions = np.arange(1, 11, dtype=np.float32)[:, None] * delta
    blended = blend_arm_boundary(actions, last, delta, 5)
    np.testing.assert_allclose(blended, actions, atol=1e-8)
    # The former stationary-anchor blend reduced the first step to 20%.
    assert blended[0, 0] == pytest.approx(0.01)


def test_micro_approach_is_preserved_even_after_stationary_chunk():
    last = np.zeros(14, dtype=np.float32)
    actions = np.arange(1, 11, dtype=np.float32)[:, None] * np.full((1, 14), 0.001)
    blended = blend_arm_boundary(actions, last, np.zeros(14), 5)
    np.testing.assert_array_equal(blended, actions)


def test_larger_offset_keeps_new_trajectory_increments():
    last = np.zeros(14, dtype=np.float32)
    actions = np.full((10, 14), 0.02, dtype=np.float32)
    actions += np.arange(10, dtype=np.float32)[:, None] * 0.001
    blended = blend_arm_boundary(actions, last, np.zeros(14), 5)
    # Position correction decays; it must not attenuate the new plan's slope.
    assert np.all(np.diff(blended[:5, :6], axis=0) >= 0.001 - 1e-7)
    np.testing.assert_array_equal(blended[4:], actions[4:])


def test_each_arm_uses_its_own_offset_gate():
    last = np.zeros(14, dtype=np.float32)
    actions = np.full((10, 14), 0.001, dtype=np.float32)
    actions[:, 7:13] = 0.03
    blended = blend_arm_boundary(actions, last, np.zeros(14), 5)
    np.testing.assert_array_equal(blended[:, :6], actions[:, :6])
    assert np.all(blended[0, 7:13] == 0)


@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_new_plan_bounds_grippers_and_endpoint(direction):
    last = np.full(14, 0.2, dtype=np.float32)
    actions = last + direction * np.arange(1, 11)[:, None] * 0.002
    actions[:, [6, 13]] = np.arange(10)[:, None] % 2 * 0.05
    original = actions.copy()
    blended = blend_arm_boundary(actions, last, np.full(14, 0.02), 5)
    joints = np.r_[0:6, 7:13]
    lower = np.minimum(last[joints], actions[:5, joints].min(axis=0))
    upper = np.maximum(last[joints], actions[:5, joints].max(axis=0))
    assert np.all(blended[:5, joints] >= lower - 1e-7)
    assert np.all(blended[:5, joints] <= upper + 1e-7)
    np.testing.assert_array_equal(blended[:, [6, 13]], actions[:, [6, 13]])
    np.testing.assert_array_equal(blended[4:], actions[4:])
    np.testing.assert_array_equal(actions, original)


@pytest.mark.parametrize("steps", [0, 1, 5])
def test_stationary_restart_and_short_chunks(steps):
    actions = np.full((2, 14), 0.1, dtype=np.float32)
    result = blend_arm_boundary(actions, np.zeros(14), np.zeros(14), steps)
    assert np.isfinite(result).all()
    assert np.all(result >= 0)
    assert np.all(result <= 0.1 + 1e-7)
    np.testing.assert_array_equal(result[-1], actions[-1])
