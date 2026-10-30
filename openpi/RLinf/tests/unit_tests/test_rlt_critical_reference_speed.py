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

"""Slowing the VLA plan inside the critical window."""

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.rollout import predict_rlt_actions
from rlinf.algorithms.rlt.route import (
    RealworldRLTRoute,
    build_rlt_route,
    time_scale_reference_plan,
)

C, D, H, OFFSET = 10, 14, 50, 5


def _plan(batch: int = 1) -> torch.Tensor:
    """Sample value equals its plan step, so time scaling is easy to read."""
    return torch.arange(H, dtype=torch.float32)[None, :, None].repeat(batch, 1, D)


def test_time_scaling_resamples_the_same_path_more_slowly():
    slowed = time_scale_reference_plan(_plan(), 0.4)
    torch.testing.assert_close(
        slowed[0, :6, 0], torch.tensor([0, 0.4, 0.8, 1.2, 1.6, 2.0])
    )
    assert slowed.shape == (1, H, D)
    assert time_scale_reference_plan(_plan(), 1.0) is not None
    torch.testing.assert_close(time_scale_reference_plan(_plan(), 1.0), _plan())
    with pytest.raises(ValueError, match="speed"):
        time_scale_reference_plan(_plan(), 0.0)


def _predict(route, flags):
    plan = _plan(batch=len(flags))
    seen = {}

    class _Features:
        def extract_rlt_obs(self, env_obs):
            return {
                "z_rl": torch.zeros(len(flags), 4),
                "proprio": torch.zeros(len(flags), D),
                "ref_chunk": plan,
            }

    class _Actor:
        def predict_action_batch(self, env_obs, mode, return_obs):
            seen["ref_chunk"] = env_obs["ref_chunk"]
            return env_obs["ref_chunk"][:, :C].clone(), {
                "forward_inputs": dict(env_obs)
            }

    _, result = predict_rlt_actions(
        policy_model=_Actor(),
        feature_model=_Features(),
        rlt_route=route,
        env_obs={},
        final_obs=None,
        mode="train",
        rlt_switch_flags=torch.tensor([[f] for f in flags]),
    )
    return seen["ref_chunk"], result["forward_inputs"]


def test_only_critical_environments_are_slowed_everywhere_downstream():
    route = RealworldRLTRoute(
        use_reference_warmup=False,
        warmup_updates=0,
        actor_plan_offset=OFFSET,
        critical_reference_speed=0.5,
    )
    actor_ref, inputs = _predict(route, [True, False])
    # Critical env: the Actor sees the slowed plan from the bridge offset on.
    assert actor_ref[0, 0, 0].item() == pytest.approx(OFFSET * 0.5)
    assert actor_ref[1, 0, 0].item() == pytest.approx(OFFSET)
    # Replay reference and RTC execution plan carry the same slowed plan.
    assert inputs["ref_chunk"][0, 10, 0].item() == pytest.approx(5.0)
    assert inputs["ref_chunk"][1, 10, 0].item() == pytest.approx(10.0)
    assert inputs["rtc_execution_plan"][0, 20, 0].item() == pytest.approx(10.0)
    assert inputs["rtc_execution_plan"][1, 20, 0].item() == pytest.approx(20.0)


def test_config_wires_the_speed_and_defaults_to_unchanged():
    def route(schedule):
        return build_rlt_route(
            OmegaConf.create(
                {
                    "algorithm": {"rlt_schedule": schedule},
                    "env": {"train": {"override_cfg": {"rtc_enabled": True}}},
                }
            )
        )

    assert route({}).critical_reference_speed == 1.0
    assert route(
        {"critical_reference_speed": 0.4}
    ).critical_reference_speed == pytest.approx(0.4)
    with pytest.raises(ValueError, match="critical_reference_speed"):
        route({"critical_reference_speed": 1.5})
