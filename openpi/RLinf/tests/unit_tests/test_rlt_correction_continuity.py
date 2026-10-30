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

"""Actor correction held through the RTC bridge and slew-limited across chunks."""

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.route import (
    RealworldRLTRoute,
    RLTRouteContext,
    build_rlt_route,
)

C, D, H, OFFSET = 10, 14, 50, 5
CONFIG = Path(__file__).parents[2] / "examples/embodiment/config"


def _route(**kwargs) -> RealworldRLTRoute:
    options = {
        "use_reference_warmup": False,
        "warmup_updates": 0,
        "actor_plan_offset": OFFSET,
        "max_actor_reference_arm_delta": 0.01,
        "max_actor_reference_gripper_delta": 0.0015,
        "actor_correction_basis_size": 3,
        "actor_correction_slew_arm": 0.0003,
        "actor_correction_slew_gripper": 0.0001,
        "actor_correction_hold_through_bridge": True,
    }
    options.update(kwargs)
    return RealworldRLTRoute(**options)


def _step(route, correction=0.01, critical=True):
    plan = torch.zeros(1, H, D)
    actor = torch.full((1, C, D), float(correction))
    output = route.route(
        RLTRouteContext(
            env_obs={},
            rlt_obs={"ref_chunk": plan},
            student_actions=actor,
            result={"forward_inputs": {"ref_chunk": plan}},
            mode="train",
            rlt_switch_flags=torch.tensor([[critical]]),
        )
    )
    return output.result["forward_inputs"]["rtc_execution_plan"][0]


def test_correction_ramps_in_and_is_held_through_the_bridge():
    route = _route()
    plan = _step(route)
    window, bridge = plan[OFFSET : OFFSET + C, 0], plan[OFFSET + C :, 0]
    expected = 0.0003 * torch.arange(1, C + 1)
    torch.testing.assert_close(window, expected)
    # The suffix that feeds the next RTC bridge keeps the last correction.
    torch.testing.assert_close(bridge, torch.full_like(bridge, expected[-1].item()))
    # Grippers ramp with their own rate.
    torch.testing.assert_close(plan[OFFSET + C - 1, 6], torch.tensor(0.001))

    second = _step(route)[OFFSET : OFFSET + C, 0]
    torch.testing.assert_close(second, expected[-1] + expected)
    # The bound still applies to the accumulated correction.
    for _ in range(5):
        plan = _step(route)
    assert plan[OFFSET:, 0].max().item() == pytest.approx(0.01)


def test_correction_restarts_from_zero_after_the_actor_releases_control():
    route = _route()
    for _ in range(3):
        _step(route)
    jax_plan = _step(route, critical=False)
    assert jax_plan.abs().max().item() == 0.0
    torch.testing.assert_close(
        _step(route)[OFFSET : OFFSET + C, 0], 0.0003 * torch.arange(1, C + 1)
    )


def test_without_continuity_options_the_bridge_runs_the_raw_plan():
    route = _route(
        actor_correction_slew_arm=None,
        actor_correction_slew_gripper=None,
        actor_correction_hold_through_bridge=False,
    )
    plan = _step(route, correction=0.004)
    torch.testing.assert_close(plan[OFFSET : OFFSET + C, 0], torch.full((C,), 0.004))
    assert plan[OFFSET + C :].abs().max().item() == 0.0


def test_stage2_config_enables_hold_and_slew():
    stage2 = OmegaConf.load(CONFIG / "realworld_piper_rlt_stage2_ac_mlp.yaml")
    route = build_rlt_route(
        OmegaConf.create(
            {
                "algorithm": {
                    "rlt_schedule": OmegaConf.to_container(
                        stage2.algorithm.rlt_schedule
                    )
                },
                "env": {"train": {"override_cfg": {"rtc_enabled": True}}},
            }
        )
    )
    assert route.actor_correction_hold_through_bridge is True
    # Wrist joints J4-J6 ramp faster, matching their wider 0.04 rad bound.
    assert route.actor_correction_slew_arm == pytest.approx(
        [0.0008, 0.0008, 0.0008, 0.002, 0.002, 0.002]
    )
    assert route.actor_correction_slew_gripper == pytest.approx(0.0001)
    with pytest.raises(ValueError, match="actor_correction_slew_arm"):
        _route(actor_correction_slew_arm=0.0)


def test_per_joint_slew_ramps_wrist_joints_faster():
    route = _route(
        actor_correction_slew_arm=[0.001, 0.001, 0.001, 0.003, 0.003, 0.003],
        actor_correction_slew_gripper=None,
    )
    delta = torch.full((1, 10, 14), 0.1)
    out = route._slew_correction(delta, torch.ones(1, 10, 1, dtype=torch.bool))
    torch.testing.assert_close(out[0, 0, [0, 7]], torch.full((2,), 0.001))
    torch.testing.assert_close(out[0, 0, [3, 10]], torch.full((2,), 0.003))
    torch.testing.assert_close(out[0, 9, [5, 12]], torch.full((2,), 0.03))
    torch.testing.assert_close(out[0, 0, [6, 13]], torch.full((2,), 0.1))
    with pytest.raises(ValueError, match="actor_correction_slew_arm"):
        _route(actor_correction_slew_arm=[0.001, 0.0, 0.001, 0.003, 0.003, 0.003])
