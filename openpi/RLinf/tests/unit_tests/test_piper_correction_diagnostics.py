import numpy as np
import pytest

from rlinf.envs.realworld.piper.correction_diagnostics import (
    format_actor_correction_chain,
    summarize_correction,
)


def test_correction_separates_units_and_excludes_human_and_padding():
    ref = np.ones((4, 14))
    actions = ref.copy()
    actions[0, :6] += 0.01
    actions[0, 6] += 0.002
    actions[0, 7:13] -= 0.02
    actions[1:] += 100  # Human or invalid targets must not pollute policy stats.
    message = summarize_correction(
        actions,
        ref,
        [True, True, False, False],
        [False, True, False, False],
        actor=True,
    )
    assert "source=MIXED(ACTOR+HUMAN)" in message
    assert "policy_steps=1 human_steps=1" in message
    assert "arm_max_abs=0.01000rad" in message
    assert "arm_max_abs=0.02000rad" in message
    assert "gripper_mean_signed=+0.00200m" in message


def test_reference_and_human_sources_are_not_labeled_actor():
    ref = np.ones((2, 14))
    message = summarize_correction(ref, ref, [True, True], [False, False], actor=False)
    assert "source=JAX_REFERENCE" in message
    assert "arm_max_abs=0.00000rad" in message
    message = summarize_correction(ref, ref, [True, True], [True, True], actor=True)
    assert "source=HUMAN" in message
    assert "correction=not_applicable" in message


def test_diagnostics_reject_different_time_windows():
    with pytest.raises(ValueError, match="time-aligned"):
        summarize_correction(
            np.ones((2, 14)), np.ones((3, 14)), [True, True], [False, False], actor=True
        )


def test_chain_compares_raw_peak_and_executor_at_same_step():
    ref = np.ones((3, 14), np.float32)
    stages = {
        key: np.zeros_like(ref) for key in ("raw", "clipped", "smoothed", "tapered")
    }
    stages["raw"][1, 0] = 0.06
    stages["clipped"][1, 0] = 0.03
    stages["smoothed"][1, 0] = 0.02
    stages["tapered"][1, 0] = 0.015
    stages["raw"][2, 0] = 0.02
    stages["coverage"] = np.ones(3, bool)
    stages["params"] = np.array([0.03, 0.003, 0.15, 4, 6000])
    rtc = ref.copy()
    rtc[1, 0] += 0.012
    accepted = ref.copy()
    accepted[1, 0] += 0.01
    accepted[2, 0] += 0.02  # Independent max would select the WRONG step.
    rows = format_actor_correction_chain(
        stages,
        ref,
        rtc,
        accepted,
        [True] * 3,
        [False] * 3,
        chunk_id=7,
        rtc_offset=5,
    )
    row = next(row for row in rows if "L.J1 " in row)
    assert "step=1 plan_step=6" in row
    assert "raw=+0.06000 clip=+0.03000 smooth=+0.02000 taper=+0.01500" in row
    assert "rtc=+0.01200 executor=+0.01000 rad retained=16.7%" in row
    assert "ref=+1.00000 target=+1.01000" in row
    assert "actor_alpha=0.150 taper_steps=4" in rows[0]


def test_chain_excludes_human_padding_and_nonactor_plan_prefix():
    ref = np.zeros((4, 14), np.float32)
    stages = {
        key: np.zeros_like(ref) for key in ("raw", "clipped", "smoothed", "tapered")
    }
    stages["raw"][:] = 100
    stages["raw"][2, 0] = 0.06
    stages["coverage"] = np.array([False, True, True, True])
    stages["params"] = np.array([0.03, 0.003, 0.15, 4, 6000])
    rows = format_actor_correction_chain(
        stages,
        ref,
        ref,
        None,
        [True, True, True, False],
        [False, True, False, False],
        chunk_id=1,
        rtc_offset=0,
    )
    row = next(row for row in rows if "L.J1 " in row)
    assert "step=2" in row and "executor=n/a" in row
    assert (
        format_actor_correction_chain(
            stages,
            ref,
            ref,
            None,
            [False] * 4,
            [False] * 4,
            chunk_id=1,
            rtc_offset=0,
        )
        == []
    )
