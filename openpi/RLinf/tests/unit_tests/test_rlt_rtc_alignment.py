# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.rollout import predict_rlt_actions
from rlinf.algorithms.rlt.route import (
    RealworldRLTRoute,
    RLTRouteContext,
    actor_refinement_stages,
    bound_and_smooth_actor_refinement,
    build_rlt_route,
    rlt_actor_plan_offset,
    smooth_actor_refinement,
)
from rlinf.algorithms.rlt.transition import shift_reference_plan, update_rlt_transitions
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACLossMixin
from rlinf.workers.env.env_worker import EnvWorker, slice_rtc_reference_tail

C, H, D, OFFSET = 10, 50, 14, 5


def _plan(batch_size: int = 1) -> torch.Tensor:
    """A VLA plan whose sample value is its execution step."""
    return torch.arange(H, dtype=torch.float32)[None, :, None].repeat(batch_size, 1, D)


def _route() -> RealworldRLTRoute:
    return RealworldRLTRoute(
        use_reference_warmup=False, warmup_updates=0, actor_plan_offset=OFFSET
    )


def _context(
    plan: torch.Tensor, actor: torch.Tensor, critical: bool
) -> RLTRouteContext:
    return RLTRouteContext(
        env_obs={},
        rlt_obs={"ref_chunk": plan},
        student_actions=actor,
        result={"forward_inputs": {"ref_chunk": plan}},
        mode="train",
        rlt_switch_flags=torch.tensor([[critical]]),
    )


def _align(actions: np.ndarray, plan: np.ndarray, executed: int) -> np.ndarray:
    """PiperWireEnv.align_rtc_chunk without the boundary blend."""
    source = np.concatenate((actions, plan[:, C:]), axis=1)
    offset = min(executed, source.shape[1] - C)
    return source[:, offset : offset + C]


def test_shift_reference_plan_reanchors_and_keeps_length() -> None:
    shifted = shift_reference_plan(_plan(), OFFSET)

    assert shifted.shape == (1, H, D)
    assert shifted[0, 0, 0].item() == OFFSET
    assert (shifted[0, H - OFFSET :, 0] == H - 1).all()
    torch.testing.assert_close(shift_reference_plan(_plan(), 0), _plan())


def test_rtc_executes_exactly_the_actor_window() -> None:
    plan = _plan()
    actor = plan[:, OFFSET : OFFSET + C] + 0.5

    output = _route().route(_context(plan, actor.clone(), critical=True))
    execution_plan = output.result["forward_inputs"]["rtc_execution_plan"]

    torch.testing.assert_close(execution_plan[:, :OFFSET], plan[:, :OFFSET])
    torch.testing.assert_close(execution_plan[:, OFFSET + C :], plan[:, OFFSET + C :])
    executed = _align(output.actions.numpy(), execution_plan.numpy(), OFFSET)
    np.testing.assert_allclose(executed, actor.numpy())
    tail = slice_rtc_reference_tail(
        execution_plan.numpy(), chunk_len=C, plan_offset=OFFSET
    )
    np.testing.assert_allclose(tail, plan.numpy()[:, OFFSET + C :])
    # Without a bridge (first chunk of an episode) every command still lands on
    # the step it was planned for.
    np.testing.assert_allclose(
        _align(output.actions.numpy(), execution_plan.numpy(), 0),
        execution_plan.numpy()[:, :C],
    )


def test_reference_path_is_unchanged() -> None:
    plan = _plan()

    output = _route().route(_context(plan, plan[:, :C] + 0.5, critical=False))

    torch.testing.assert_close(
        output.result["forward_inputs"]["rtc_execution_plan"], plan
    )
    torch.testing.assert_close(output.actions, plan[:, :C])


def test_direct_actor_uses_unshifted_bounded_chunk_and_preserves_jax_plan() -> None:
    cfg = OmegaConf.create(
        {
            "algorithm": {
                "rlt_schedule": {
                    "actor_direct_execution": True,
                    "actor_plan_offset": OFFSET,
                    "max_actor_reference_arm_delta": 0.03,
                    "max_actor_reference_gripper_delta": 0.003,
                    "actor_refinement_smoothing_alpha": 0.15,
                    "actor_refinement_tail_taper_steps": 4,
                }
            },
            "env": {
                "train": {
                    "env_type": "realworld",
                    "override_cfg": {"rtc_enabled": True},
                }
            },
        }
    )
    route = build_rlt_route(cfg)
    assert route.actor_plan_offset == 0
    plan = _plan()
    actor = plan[:, :C] + 0.2
    actor_output = route.route(_context(plan, actor.clone(), critical=True))
    expected = plan[:, :C] + 0.03
    expected[..., 6::7] = plan[:, :C, 6::7] + 0.003
    torch.testing.assert_close(actor_output.actions, expected)
    reference_output = route.route(_context(plan, actor.clone(), critical=False))
    torch.testing.assert_close(reference_output.actions, plan[:, :C])
    torch.testing.assert_close(
        reference_output.result["forward_inputs"]["rtc_execution_plan"], plan
    )


def test_actor_is_conditioned_on_the_executed_reference_window() -> None:
    plan = _plan()
    seen = {}

    class _Features:
        def extract_rlt_obs(self, env_obs):
            return {
                "z_rl": torch.zeros(1, 4),
                "proprio": torch.zeros(1, 14),
                "ref_chunk": plan,
            }

    class _Actor:
        def predict_action_batch(self, env_obs, mode, return_obs):
            seen["ref_chunk"] = env_obs["ref_chunk"]
            # A zero-initialized actor reproduces its reference input.
            return env_obs["ref_chunk"][:, :C].clone(), {
                "forward_inputs": dict(env_obs)
            }

    _, result = predict_rlt_actions(
        policy_model=_Actor(),
        feature_model=_Features(),
        rlt_route=_route(),
        env_obs={},
        final_obs=None,
        mode="train",
        rlt_switch_flags=torch.tensor([[True]]),
    )

    assert seen["ref_chunk"][0, 0, 0].item() == OFFSET
    torch.testing.assert_close(result["forward_inputs"]["ref_chunk"], plan)
    torch.testing.assert_close(result["forward_inputs"]["rtc_execution_plan"], plan)


def test_replay_references_use_applied_offsets_and_operator_actions() -> None:
    plan = _plan()
    forward_inputs = {
        "z_rl": torch.zeros(1, 4),
        "proprio": torch.zeros(1, 14),
        "ref_chunk": plan,
        "rlt_transition_z_rl": torch.ones(1, 4),
        "rlt_transition_proprio": torch.zeros(1, 14),
        "rlt_transition_ref_chunk": plan + 100.0,
    }
    stored = []
    rollout_results = [
        SimpleNamespace(append_transitions=lambda curr, nxt: stored.append((curr, nxt)))
    ]
    pending = [None]
    rollout_result = SimpleNamespace(forward_inputs=forward_inputs)

    update_rlt_transitions(
        0,
        pending,
        rollout_results,
        rollout_result,
        cache_current=True,
        curr_ref_offset=OFFSET,
    )
    flags = torch.zeros(1, C)
    flags[0, :2] = 1.0
    update_rlt_transitions(
        0,
        pending,
        rollout_results,
        rollout_result,
        cache_current=False,
        intervene_actions=torch.full((1, C * D), -1.0),
        intervene_flags=flags,
        next_ref_offset=3,
    )

    curr, nxt = stored[0]
    assert (curr["ref_chunk"][0, :2] == -1.0).all()
    assert curr["ref_chunk"][0, 2, 0].item() == OFFSET + 2
    assert nxt["ref_chunk"][0, 0, 0].item() == 103.0


def test_actor_plan_offset_follows_the_rtc_bridge_config() -> None:
    def cfg(rtc_enabled: bool, override=None):
        schedule = {} if override is None else {"actor_plan_offset": override}
        return OmegaConf.create(
            {
                "algorithm": {"rlt_schedule": schedule},
                "env": {
                    "train": {
                        "override_cfg": {
                            "rtc_enabled": rtc_enabled,
                            "rtc_max_bridge_steps": 5,
                        }
                    }
                },
            }
        )

    assert rlt_actor_plan_offset(cfg(True)) == 5
    assert rlt_actor_plan_offset(cfg(False)) == 0
    assert rlt_actor_plan_offset(cfg(True, override=2)) == 2


def test_arm_is_unclamped_and_gripper_clamped() -> None:
    reference = torch.zeros(1, C, D)
    actor = torch.full((1, C, D), 0.1)

    executed, clipped = bound_and_smooth_actor_refinement(
        actor, reference, arm_limit=None, gripper_limit=0.002, alpha=None, taper_steps=0
    )

    torch.testing.assert_close(executed[..., 0], torch.full((1, C), 0.1))
    torch.testing.assert_close(executed[..., 6], torch.full((1, C), 0.002))
    assert clipped.item() is True


def test_diagnostic_stages_preserve_actions_gradients_and_arm_only_taper():
    raw = torch.randn(2, 10, 14, requires_grad=True)
    reference = torch.zeros_like(raw)
    stages, _ = actor_refinement_stages(
        raw,
        reference,
        arm_limit=0.03,
        gripper_limit=0.003,
        alpha=0.15,
        taper_steps=4,
    )
    limits = torch.full((14,), 0.03)
    limits[6::7] = 0.003
    original = smooth_actor_refinement(
        raw.clamp(-limits, limits), alpha=0.15, taper_steps=4
    )
    torch.testing.assert_close(stages["tapered"], original, rtol=0, atol=0)
    grad_new = torch.autograd.grad(stages["tapered"].sum(), raw, retain_graph=True)[0]
    grad_old = torch.autograd.grad(original.sum(), raw)[0]
    torch.testing.assert_close(grad_new, grad_old, rtol=0, atol=0)
    torch.testing.assert_close(
        stages["clipped"][..., 6::7], stages["tapered"][..., 6::7]
    )


def test_diagnostic_trace_uses_actual_rtc_offset_and_is_removed_from_replay():
    from collections import defaultdict

    plan = _plan()
    actor = plan[:, OFFSET : OFFSET + C] + 0.06
    route = RealworldRLTRoute(
        use_reference_warmup=False,
        warmup_updates=0,
        actor_plan_offset=OFFSET,
        max_actor_reference_arm_delta=0.03,
        max_actor_reference_gripper_delta=0.003,
        actor_refinement_smoothing_alpha=0.15,
        actor_refinement_tail_taper_steps=4,
    )
    output = route.route(_context(plan, actor, critical=True))
    inputs = output.result["forward_inputs"]
    aligned = _align(
        output.actions.numpy(), inputs["rtc_execution_plan"].numpy(), OFFSET
    )
    rollout = SimpleNamespace(forward_inputs=inputs, actions=torch.from_numpy(aligned))
    worker = SimpleNamespace(
        model_cfg=SimpleNamespace(action_dim=14, num_action_chunks=C),
        _rlt_episode_ids=[torch.tensor([123])],
        _rlt_expected_rtc_offset=[OFFSET],
        _rlt_stride_enabled=False,
    )
    EnvWorker._rlt_begin_chunk(worker, 0, rollout, OFFSET, defaultdict(list))
    trace = worker._rlt_correction_traces[0]
    torch.testing.assert_close(
        worker._rlt_correction_references[0], plan[:, OFFSET : OFFSET + C]
    )
    torch.testing.assert_close(trace["raw"], actor - plan[:, OFFSET : OFFSET + C])
    torch.testing.assert_close(trace["rtc_request"], torch.from_numpy(aligned))
    assert trace["coverage"].bool().all()
    assert not any(key.startswith("rlt_diag_") for key in inputs)


def test_smoothing_runs_without_any_clamp() -> None:
    reference = torch.zeros(1, C, D)
    actor = torch.zeros(1, C, D)
    actor[:, ::2, 0] = 1.0

    executed, clipped = bound_and_smooth_actor_refinement(
        actor, reference, arm_limit=None, gripper_limit=None, alpha=0.5, taper_steps=0
    )

    assert clipped.item() is False
    assert executed[0, 1, 0].item() == pytest.approx(0.5)


def test_learner_scores_the_executed_actor_actions() -> None:
    worker = RLTACLossMixin()
    schedule = {
        "max_actor_reference_arm_delta": None,
        "max_actor_reference_gripper_delta": 0.002,
        "actor_refinement_smoothing_alpha": 0.4,
        "actor_refinement_tail_taper_steps": 3,
    }
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {"rlt_schedule": schedule},
            "actor": {"model": {"num_action_chunks": C, "action_dim": D}},
        }
    )
    reference = torch.zeros(2, C * D)
    actor = torch.full((2, C * D), 0.1)

    executed = worker._apply_execution_transform(actor, reference)

    expected, _ = bound_and_smooth_actor_refinement(
        actor.reshape(2, C, D),
        reference.reshape(2, C, D),
        arm_limit=None,
        gripper_limit=0.002,
        alpha=0.4,
        taper_steps=3,
    )
    torch.testing.assert_close(executed, expected.reshape(2, -1))
    assert executed.reshape(2, C, D)[0, 0, 6].item() == pytest.approx(0.002)


def test_operator_steps_can_keep_the_vla_reference_in_chunk_rows() -> None:
    plan = _plan()
    forward_inputs = {
        "z_rl": torch.zeros(1, 4),
        "proprio": torch.zeros(1, 14),
        "ref_chunk": plan,
        "rlt_transition_z_rl": torch.ones(1, 4),
        "rlt_transition_proprio": torch.zeros(1, 14),
        "rlt_transition_ref_chunk": plan,
    }
    stored = []
    rollout_results = [
        SimpleNamespace(append_transitions=lambda curr, nxt: stored.append((curr, nxt)))
    ]
    pending = [None]
    rollout_result = SimpleNamespace(forward_inputs=forward_inputs)
    update_rlt_transitions(
        0, pending, rollout_results, rollout_result, cache_current=True
    )
    flags = torch.ones(1, C)
    update_rlt_transitions(
        0,
        pending,
        rollout_results,
        rollout_result,
        cache_current=False,
        intervene_actions=torch.full((1, C * D), -1.0),
        intervene_flags=flags,
        replace_human_reference=False,
    )
    torch.testing.assert_close(stored[0][0]["ref_chunk"], plan)
