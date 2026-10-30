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

from typing import Any, Literal

import numpy as np
import torch

from rlinf.algorithms.rlt.route import (
    RLTRoute,
    RLTRouteContext,
    time_scale_reference_plan,
)
from rlinf.algorithms.rlt.transition import (
    RLT_OBS_KEYS,
    RLT_OPTIONAL_OBS_KEYS,
    RLT_TRANSITION_PREFIX,
    shift_reference_plan,
)


def _append_rlt_transition_obs(
    *,
    feature_model: Any,
    result: dict[str, Any],
    rlt_obs: dict[str, torch.Tensor],
    final_obs: dict[str, Any] | None,
) -> None:
    transition_obs = rlt_obs
    if final_obs is not None:
        transition_obs = feature_model.extract_rlt_obs(final_obs)
    keys = RLT_OBS_KEYS + tuple(
        key for key in RLT_OPTIONAL_OBS_KEYS if key in transition_obs
    )
    for key in keys:
        result["forward_inputs"][f"{RLT_TRANSITION_PREFIX}{key}"] = transition_obs[key]


def _slow_critical_reference(
    rlt_obs: dict[str, torch.Tensor],
    rlt_switch_flags: torch.Tensor | None,
    speed: float,
) -> dict[str, torch.Tensor]:
    """Time-stretch the VLA plan of environments inside the critical window.

    Applied before the Actor, the route and the replay see the plan, so the
    Actor's input, its BC target, the RTC execution plan and the stored
    transition all describe the same slowed trajectory.
    """
    if speed >= 1.0 or rlt_switch_flags is None:
        return rlt_obs
    plan = rlt_obs["ref_chunk"]
    batch_size = plan.shape[0]
    critical = torch.as_tensor(rlt_switch_flags, device=plan.device).bool()
    critical = critical.reshape(batch_size, -1)[:, -1]
    if not bool(critical.any()):
        return rlt_obs
    slowed = time_scale_reference_plan(plan, speed)
    return {
        **rlt_obs,
        "ref_chunk": torch.where(critical[:, None, None], slowed, plan),
    }


def predict_rlt_actions(
    *,
    policy_model: Any,
    feature_model: Any,
    rlt_route: RLTRoute,
    env_obs: dict[str, Any],
    final_obs: dict[str, Any] | None,
    mode: Literal["train", "eval"],
    version: int = 0,
    rlt_switch_flags: torch.Tensor | None = None,
    intervene_requested: torch.Tensor | None = None,
    expert_model: Any | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    with torch.no_grad():
        rlt_obs = _slow_critical_reference(
            feature_model.extract_rlt_obs(env_obs),
            rlt_switch_flags,
            float(getattr(rlt_route, "critical_reference_speed", 1.0)),
        )
        # The actor's chunk starts executing after the RTC bridge, so it is
        # conditioned on the part of the VLA plan that covers those steps. The
        # learner sees the same shifted reference in replay.
        plan_offset = int(rlt_route.actor_plan_offset)
        actor_obs = rlt_obs
        if plan_offset > 0:
            actor_obs = {
                **rlt_obs,
                "ref_chunk": shift_reference_plan(rlt_obs["ref_chunk"], plan_offset),
            }
        actions, result = policy_model.predict_action_batch(
            env_obs=actor_obs,
            mode=mode,
            return_obs=True,
        )
        if plan_offset > 0:
            # forward_inputs keeps the VLA plan anchored at the observation
            # time; the route and EnvWorker index it by execution step.
            result["forward_inputs"]["ref_chunk"] = rlt_obs["ref_chunk"]
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        route_output = rlt_route.route(
            RLTRouteContext(
                env_obs=env_obs,
                rlt_obs=rlt_obs,
                student_actions=actions,
                result=result,
                mode=mode,
                rlt_switch_flags=rlt_switch_flags,
                intervene_requested=intervene_requested,
                expert_model=expert_model,
                version=version,
            )
        )
        actions = route_output.actions
        result = route_output.result

        _append_rlt_transition_obs(
            feature_model=feature_model,
            result=result,
            rlt_obs=rlt_obs,
            final_obs=final_obs,
        )

    return actions, result
