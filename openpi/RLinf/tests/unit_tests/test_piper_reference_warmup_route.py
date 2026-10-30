from __future__ import annotations

import torch

from rlinf.algorithms.rlt.route import RealworldRLTRoute, RLTRouteContext


def _context(*, version: int, critical: bool) -> RLTRouteContext:
    student = torch.full((1, 5, 14), 2.0)
    reference = torch.full((1, 5, 14), 1.0)
    return RLTRouteContext(
        env_obs={},
        rlt_obs={"ref_chunk": reference},
        student_actions=student,
        result={"forward_inputs": {"ref_chunk": reference}},
        mode="train",
        version=version,
        rlt_switch_flags=torch.tensor([[critical]]),
    )


def test_b_records_reference_transition_before_learner_ready() -> None:
    route = RealworldRLTRoute(
        use_reference_warmup=True,
        warmup_updates=400,
    )
    output = route.route(_context(version=399, critical=True))

    assert torch.all(output.actions == 1.0)
    assert output.result["forward_inputs"]["record_transition"].item() is True
    assert output.result["forward_inputs"]["actor_switch"].item() is False
    assert output.result["forward_inputs"]["reference_warmup"].item() is True


def test_b_switches_to_actor_after_learner_ready() -> None:
    route = RealworldRLTRoute(
        use_reference_warmup=True,
        warmup_updates=400,
    )
    output = route.route(_context(version=400, critical=True))

    assert torch.all(output.actions == 2.0)
    assert output.result["forward_inputs"]["record_transition"].item() is True
    assert output.result["forward_inputs"]["actor_switch"].item() is True
    assert output.result["forward_inputs"]["reference_warmup"].item() is False


def test_b_rejects_entire_chunk_when_actor_deviates_too_far() -> None:
    route = RealworldRLTRoute(
        use_reference_warmup=True,
        warmup_updates=400,
        max_actor_takeover_arm_deviation=0.05,
        max_actor_takeover_gripper_deviation=0.005,
    )
    context = _context(version=400, critical=True)
    output = route.route(context)

    torch.testing.assert_close(
        output.actions, context.result["forward_inputs"]["ref_chunk"]
    )
    assert output.result["forward_inputs"]["record_transition"].item() is True
    assert output.result["forward_inputs"]["actor_switch"].item() is False
    assert output.result["forward_inputs"]["reference_warmup"].item() is True


def test_b_accepts_near_reference_actor_chunk() -> None:
    route = RealworldRLTRoute(
        use_reference_warmup=True,
        warmup_updates=400,
        max_actor_takeover_arm_deviation=0.05,
        max_actor_takeover_gripper_deviation=0.005,
    )
    context = _context(version=400, critical=True)
    context.student_actions = context.result["forward_inputs"]["ref_chunk"] + 0.003
    output = route.route(context)

    torch.testing.assert_close(output.actions, context.student_actions)
    assert output.result["forward_inputs"]["actor_switch"].item() is True


def test_b_rejects_nonfinite_actor_chunk() -> None:
    route = RealworldRLTRoute(
        use_reference_warmup=True,
        warmup_updates=400,
        max_actor_takeover_arm_deviation=0.05,
    )
    context = _context(version=400, critical=True)
    context.student_actions[0, 0, 0] = float("nan")
    output = route.route(context)

    assert torch.isfinite(output.actions).all()
    assert output.result["forward_inputs"]["actor_switch"].item() is False


def test_actor_handover_is_bounded_around_reference() -> None:
    route = RealworldRLTRoute(
        use_reference_warmup=True,
        warmup_updates=2000,
        max_actor_reference_arm_delta=0.02,
        max_actor_reference_gripper_delta=0.002,
    )
    output = route.route(_context(version=2000, critical=True))

    expected = torch.full((1, 5, 14), 1.02)
    expected[..., 6] = 1.002
    expected[..., 13] = 1.002
    torch.testing.assert_close(output.actions, expected)


def test_actor_refinement_smooths_arm_residual_and_tapers_rtc_tail() -> None:
    route = RealworldRLTRoute(
        use_reference_warmup=True,
        warmup_updates=2000,
        max_actor_reference_arm_delta=0.02,
        max_actor_reference_gripper_delta=0.002,
        actor_refinement_smoothing_alpha=0.5,
        actor_refinement_tail_taper_steps=2,
    )
    context = _context(version=2000, critical=True)
    alternating = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0]).reshape(1, 5, 1)
    context.student_actions = (
        context.result["forward_inputs"]["ref_chunk"] + alternating
    )

    output = route.route(context)
    residual = output.actions - context.result["forward_inputs"]["ref_chunk"]

    expected_arm = torch.tensor([0.02, 0.0, 0.01, -0.0025, 0.0])
    torch.testing.assert_close(residual[0, :, 0], expected_arm)
    torch.testing.assert_close(residual[0, :, 7], expected_arm)
    torch.testing.assert_close(
        residual[0, :, 6],
        torch.tensor([0.002, -0.002, 0.002, -0.002, 0.002]),
    )


def test_actor_refinement_smoothing_config_is_validated() -> None:
    try:
        RealworldRLTRoute(
            use_reference_warmup=True,
            warmup_updates=2000,
            actor_refinement_smoothing_alpha=0.0,
        )
    except ValueError as exc:
        assert "smoothing_alpha" in str(exc)
    else:
        raise AssertionError("Expected invalid smoothing alpha to raise ValueError.")


def test_no_b_uses_reference_without_recording() -> None:
    route = RealworldRLTRoute(
        use_reference_warmup=True,
        warmup_updates=400,
    )
    output = route.route(_context(version=400, critical=False))

    assert torch.all(output.actions == 1.0)
    assert output.result["forward_inputs"]["record_transition"].item() is False
    assert output.result["forward_inputs"]["actor_switch"].item() is False
