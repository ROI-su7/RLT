import numpy as np
import pytest

from toolkits.realworld_check.compare_openpi_action_parity import (
    action_parity_metrics,
)


def test_action_parity_metrics_separates_arm_and_gripper_units():
    jax = np.zeros((2, 5, 14), dtype=np.float32)
    torch = jax.copy()
    torch[..., 0] = 0.02
    torch[..., 6] = 0.004

    metrics = action_parity_metrics(jax, torch, prefix_steps=5)

    assert metrics["arm_max_abs_rad"] == pytest.approx(0.02)
    assert metrics["gripper_max_abs_m"] == pytest.approx(0.004)


def test_action_parity_metrics_rejects_shape_drift():
    with pytest.raises(ValueError, match="action shapes differ"):
        action_parity_metrics(
            np.zeros((1, 5, 14), dtype=np.float32),
            np.zeros((1, 10, 14), dtype=np.float32),
            prefix_steps=5,
        )
