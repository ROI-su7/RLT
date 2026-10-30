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

"""Low-frequency Actor-correction projection shared by route and learner."""

from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.route import (
    RealworldRLTRoute,
    RLTRouteContext,
    actor_refinement_stages,
    build_rlt_route,
    project_actor_refinement,
    temporal_projection_matrix,
)
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACLossMixin
from rlinf.workers.env.env_worker import EnvWorker

C, D = 10, 14
CONFIG_DIR = Path(__file__).parents[2] / "examples/embodiment/config"


def _curvature(chunk: torch.Tensor) -> float:
    """Largest step-to-step second difference along the chunk axis."""
    return torch.diff(chunk, n=2, dim=1).abs().max().item()


@pytest.mark.parametrize("basis_type", ["dct", "poly"])
@pytest.mark.parametrize("size", [1, 3, 4])
def test_projection_is_symmetric_idempotent_and_keeps_offsets(basis_type, size):
    projection = temporal_projection_matrix(C, size, basis_type=basis_type)
    torch.testing.assert_close(projection, projection.T)
    torch.testing.assert_close(projection @ projection, projection, atol=1e-6, rtol=0)
    offset = torch.full((1, C, D), 0.004)
    torch.testing.assert_close(
        project_actor_refinement(offset, basis_size=size, basis_type=basis_type),
        offset,
    )


def test_projection_removes_step_to_step_zigzag():
    steps = torch.arange(C, dtype=torch.float32)
    zigzag = (0.003 * (-1.0) ** steps)[None, :, None].expand(1, C, D)
    ramp = (0.001 * steps)[None, :, None].expand(1, C, D)

    projected = project_actor_refinement(ramp + zigzag, basis_size=3)

    # The input's second difference is 4 * 0.003; only a smooth window
    # leakage of the alternation may remain.
    assert _curvature(projected) < 0.1 * _curvature(zigzag)
    leakage = projected - project_actor_refinement(ramp, basis_size=3)
    assert leakage.abs().max().item() < 0.25 * 0.003
    # Full or missing basis is the identity.
    signal = ramp + zigzag
    assert project_actor_refinement(signal, basis_size=None) is signal
    assert project_actor_refinement(signal, basis_size=C) is signal


def test_projection_runs_before_the_exact_bound_and_passes_low_frequency_grads():
    raw = (torch.randn(2, C, D) * 0.02).requires_grad_()
    stages, _ = actor_refinement_stages(
        raw,
        torch.zeros_like(raw),
        arm_limit=0.01,
        gripper_limit=0.0015,
        alpha=None,
        taper_steps=0,
        basis_size=3,
    )
    projected = project_actor_refinement(raw.detach(), basis_size=3)
    torch.testing.assert_close(stages["projected"], projected)
    assert stages["tapered"][..., 0].abs().max() <= 0.01 + 1e-7
    assert stages["tapered"][..., 6].abs().max() <= 0.0015 + 1e-7

    # A Q gradient that alternates every step cannot reach the Actor.
    steps = torch.arange(C, dtype=torch.float32)
    weight = ((-1.0) ** steps)[None, :, None].expand_as(raw)
    unbounded, _ = actor_refinement_stages(
        raw,
        torch.zeros_like(raw),
        arm_limit=None,
        gripper_limit=None,
        alpha=None,
        taper_steps=0,
        basis_size=3,
    )
    (grad,) = torch.autograd.grad((unbounded["tapered"] * weight).sum(), raw)
    assert _curvature(grad) < 0.1 * _curvature(weight)
    assert grad.abs().max().item() < 0.25


def test_route_and_learner_execute_the_same_projected_correction():
    schedule = {
        "actor_direct_execution": True,
        "max_actor_reference_arm_delta": 0.01,
        "max_actor_reference_gripper_delta": 0.0015,
        "actor_correction_basis_size": 3,
        "actor_correction_basis_type": "dct",
    }
    cfg = OmegaConf.create(
        {
            "algorithm": {"rlt_schedule": schedule},
            "actor": {"model": {"num_action_chunks": C, "action_dim": D}},
            "env": {"train": {"override_cfg": {"rtc_enabled": True}}},
        }
    )
    route = build_rlt_route(cfg)
    assert isinstance(route, RealworldRLTRoute)
    assert route.actor_correction_basis_size == 3

    plan = torch.zeros(1, 50, D)
    actor = torch.randn(1, C, D) * 0.01
    output = route.route(
        RLTRouteContext(
            env_obs={},
            rlt_obs={"ref_chunk": plan},
            student_actions=actor,
            result={"forward_inputs": {"ref_chunk": plan}},
            mode="train",
            rlt_switch_flags=torch.tensor([[True]]),
        )
    )
    worker = RLTACLossMixin()
    worker.cfg = cfg
    learner = worker._apply_execution_transform(
        actor.reshape(1, -1), plan[:, :C].reshape(1, -1)
    )
    torch.testing.assert_close(output.actions.reshape(1, -1), learner)

    inputs = output.result["forward_inputs"]
    assert inputs["rlt_diag_params"][0, 7].item() == 3.0
    rollout = SimpleNamespace(forward_inputs=inputs, actions=output.actions)
    env = SimpleNamespace(
        model_cfg=SimpleNamespace(action_dim=D, num_action_chunks=C),
        _rlt_episode_ids=[torch.tensor([1])],
        _rlt_expected_rtc_offset=[0],
        _rlt_stride_enabled=False,
    )
    EnvWorker._rlt_begin_chunk(env, 0, rollout, 0, defaultdict(list))
    assert "projected" in env._rlt_correction_traces[0]
    assert not any(key.startswith("rlt_diag_") for key in inputs)


def test_invalid_basis_configuration_fails_closed():
    with pytest.raises(ValueError, match="basis_type"):
        temporal_projection_matrix(C, 3, basis_type="fft")
    with pytest.raises(ValueError, match="basis_size"):
        temporal_projection_matrix(C, 0)
    with pytest.raises(ValueError, match="actor_correction_basis_size"):
        RealworldRLTRoute(
            use_reference_warmup=False,
            warmup_updates=0,
            actor_correction_basis_size=0,
        )


def test_zero_correction_diagnostic_executes_the_reference():
    stage2 = OmegaConf.load(CONFIG_DIR / "realworld_piper_rlt_stage2_ac_mlp.yaml")
    diagnostic = OmegaConf.load(
        CONFIG_DIR / "realworld_piper_rlt_stage2_zero_correction.yaml"
    )
    assert diagnostic.defaults[0] == "realworld_piper_rlt_stage2_ac_mlp"
    merged = OmegaConf.merge(
        {"algorithm": stage2.algorithm},
        {"algorithm": diagnostic.algorithm},
    )
    schedule = merged.algorithm.rlt_schedule
    assert schedule.actor_direct_execution is True
    assert schedule.warmup_post_collect_updates == 0
    assert merged.algorithm.replay_buffer.auto_save is False
    assert merged.algorithm.replay_buffer.save_transitions is True

    route = build_rlt_route(
        OmegaConf.create(
            {
                "algorithm": {"rlt_schedule": OmegaConf.to_container(schedule)},
                "env": {"train": {"override_cfg": {"rtc_enabled": True}}},
            }
        )
    )
    reference = torch.randn(1, C, D)
    executed, _ = route._bound_actor_refinement(reference + 0.5, reference)
    torch.testing.assert_close(executed, reference)


def test_zero_correction_rtc_diagnostic_keeps_the_jax_executor():
    merged = OmegaConf.merge(
        *(
            {"algorithm": OmegaConf.load(CONFIG_DIR / name).algorithm}
            for name in (
                "realworld_piper_rlt_stage2_ac_mlp.yaml",
                "realworld_piper_rlt_stage2_zero_correction.yaml",
                "realworld_piper_rlt_stage2_zero_correction_rtc.yaml",
            )
        )
    )
    schedule = OmegaConf.to_container(merged.algorithm.rlt_schedule)
    assert schedule["actor_direct_execution"] is False
    route = build_rlt_route(
        OmegaConf.create(
            {
                "algorithm": {"rlt_schedule": schedule},
                "env": {
                    "train": {
                        "override_cfg": {"rtc_enabled": True, "rtc_max_bridge_steps": 5}
                    }
                },
            }
        )
    )
    # The Actor window follows the RTC bridge, as on the JAX path.
    assert route.actor_plan_offset == 5
    reference = torch.randn(1, C, D)
    executed, _ = route._bound_actor_refinement(reference + 0.5, reference)
    torch.testing.assert_close(executed, reference)


def test_per_joint_arm_bound_applies_to_both_arms():
    from rlinf.algorithms.rlt.route import (
        arm_limit_vector,
        bound_and_smooth_actor_refinement,
    )

    limits = [0.015, 0.015, 0.015, 0.04, 0.04, 0.04]
    vector = arm_limit_vector(limits, 14)
    assert vector[:6] == limits and vector[7:13] == limits
    assert vector[6] == vector[13] == float("inf")
    assert arm_limit_vector(0.02, 14) == [0.02] * 14
    with pytest.raises(ValueError):
        arm_limit_vector([0.1, 0.2], 14)

    ref = torch.zeros(1, 10, 14)
    actor = torch.full((1, 10, 14), 0.1)
    executed, clipped = bound_and_smooth_actor_refinement(
        actor, ref, arm_limit=limits, gripper_limit=0.001, alpha=None, taper_steps=0
    )
    step = executed[0, 0]
    torch.testing.assert_close(step[[0, 1, 2, 7, 8, 9]], torch.full((6,), 0.015))
    torch.testing.assert_close(step[[3, 4, 5, 10, 11, 12]], torch.full((6,), 0.04))
    torch.testing.assert_close(step[[6, 13]], torch.full((2,), 0.001))
    assert bool(clipped.all())
