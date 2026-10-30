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

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch

from rlinf.algorithms.rlt.expert import predict_expert_actions
from rlinf.algorithms.rlt.transition import use_simulator_transition_replay
from rlinf.utils.arm_layout import arm_width, gripper_slice, is_arm_joint
from rlinf.utils.logging import get_logger


@dataclass(kw_only=True)
class RLTRouteContext:
    env_obs: dict[str, Any]
    rlt_obs: dict[str, torch.Tensor]
    student_actions: torch.Tensor
    result: dict[str, Any]
    mode: Literal["train", "eval"]
    rlt_switch_flags: torch.Tensor | None = None
    intervene_requested: torch.Tensor | None = None
    expert_model: Any | None = None
    version: int = 0
    default_actor_switch: bool = False


@dataclass(kw_only=True)
class RLTRouteOutput:
    actions: torch.Tensor
    result: dict[str, Any]


def _last_info_bool(
    value: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    default: bool,
) -> torch.Tensor:
    if value is None:
        return torch.full((batch_size,), bool(default), dtype=torch.bool, device=device)
    value = torch.as_tensor(value, device=device)
    if value.numel() == 1:
        return torch.full(
            (batch_size,),
            bool(value.reshape(-1)[0].item()),
            dtype=torch.bool,
            device=device,
        )
    return value.reshape(batch_size, -1).to(torch.bool)[:, -1]


def _flatten_action_chunk(actions: torch.Tensor) -> torch.Tensor:
    if actions.dim() <= 2:
        return actions
    return actions.reshape(actions.shape[0], -1)


def _normalize_rlt_switch_flags(
    actions: torch.Tensor,
    rlt_switch_flags: torch.Tensor | None,
    *,
    default: bool,
) -> torch.Tensor:
    if rlt_switch_flags is None:
        rlt_switch_flags = torch.full(
            (actions.shape[0], actions.shape[1]),
            bool(default),
            dtype=torch.bool,
            device=actions.device,
        )
    else:
        rlt_switch_flags = torch.as_tensor(
            rlt_switch_flags, device=actions.device
        ).bool()
    if rlt_switch_flags.dim() == 1:
        rlt_switch_flags = rlt_switch_flags[:, None]
    if rlt_switch_flags.shape[1] > 1:
        rlt_switch_flags = rlt_switch_flags[:, -1:]
    if actions.shape[1] > 1:
        rlt_switch_flags = rlt_switch_flags.expand(-1, actions.shape[1])
    return rlt_switch_flags.reshape(actions.shape[0], actions.shape[1], 1)


def arm_indices(action_dim: int, device: torch.device) -> torch.Tensor:
    """Return arm joints while excluding each arm's gripper."""
    return torch.tensor(
        [idx for idx in range(action_dim) if is_arm_joint(idx, action_dim)],
        dtype=torch.long,
        device=device,
    )


ACTOR_CORRECTION_BASIS_TYPES = ("dct", "poly")


def time_scale_reference_plan(plan: torch.Tensor, speed: float) -> torch.Tensor:
    """Re-sample a ``[B, H, D]`` VLA plan so it advances ``speed`` plan steps per step.

    Execution step ``k`` of the result is the plan linearly interpolated at
    ``speed * k``; the horizon is unchanged and the last sample is held. With
    ``speed < 1`` the robot follows the same path more slowly, which gives the
    Actor more decisions per millimetre of approach.
    """
    speed = float(speed)
    if not 0.0 < speed <= 1.0:
        raise ValueError(f"speed must be in (0, 1], got {speed}.")
    if speed == 1.0:
        return plan
    horizon = plan.shape[1]
    position = torch.arange(horizon, device=plan.device, dtype=torch.float64) * speed
    position = position.clamp(max=horizon - 1)
    lower = position.floor().long()
    upper = (lower + 1).clamp(max=horizon - 1)
    weight = (position - lower).to(plan.dtype)[None, :, None]
    return plan[:, lower] * (1.0 - weight) + plan[:, upper] * weight


def temporal_basis(
    chunk_len: int,
    basis_size: int,
    *,
    basis_type: str = "dct",
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a [C, K] orthonormal basis of the ``basis_size`` slowest modes.

    ``dct`` keeps the lowest DCT-II cosines (constant, half cosine, ...);
    ``poly`` keeps polynomials up to degree ``basis_size - 1``. Both bases
    include the constant, so a chunk-wide offset passes through unchanged.
    """
    chunk_len = int(chunk_len)
    basis_size = int(basis_size)
    if not 1 <= basis_size <= chunk_len:
        raise ValueError(f"basis_size must be in [1, {chunk_len}], got {basis_size}.")
    t = torch.arange(chunk_len, dtype=torch.float64)
    if basis_type == "dct":
        k = torch.arange(basis_size, dtype=torch.float64)
        basis = torch.cos(torch.pi * (t[:, None] + 0.5) * k[None, :] / chunk_len)
    elif basis_type == "poly":
        x = 2.0 * t / max(chunk_len - 1, 1) - 1.0
        basis = torch.stack([x**k for k in range(basis_size)], dim=1)
    else:
        raise ValueError(
            f"basis_type must be one of {ACTOR_CORRECTION_BASIS_TYPES}, "
            f"got {basis_type!r}."
        )
    q, _ = torch.linalg.qr(basis)
    return q.to(device=device, dtype=dtype)


def temporal_projection_matrix(
    chunk_len: int,
    basis_size: int,
    *,
    basis_type: str = "dct",
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return the [C, C] orthogonal projection onto ``basis_size`` slow modes."""
    q = temporal_basis(chunk_len, basis_size, basis_type=basis_type)
    return (q @ q.T).to(device=device, dtype=dtype)


def smooth_exploration_noise(
    batch_size: int,
    chunk_len: int,
    std: torch.Tensor,
    *,
    basis_size: int,
    basis_type: str = "dct",
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    coefficients: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gaussian noise that lives entirely in the slow correction subspace.

    Returns ``[B, C, D]`` noise whose per-step RMS equals ``std`` (one value per
    action dimension). Coefficients are drawn on the ``basis_size`` slowest
    modes and scaled by ``sqrt(C / K)`` so restricting the noise to K of C
    modes does not shrink it. The projection leaves it unchanged, so
    exploration can never add within-chunk jitter. ``coefficients`` supplies
    the standard-normal ``[B, K, D]`` draw, e.g. an AR(1) state kept across
    chunks; otherwise a fresh draw is used.
    """
    basis = temporal_basis(
        chunk_len, basis_size, basis_type=basis_type, device=device, dtype=dtype
    )
    std = torch.as_tensor(std, device=device, dtype=dtype)
    scale = (float(chunk_len) / float(basis_size)) ** 0.5
    if coefficients is None:
        coefficients = torch.randn(
            batch_size, basis.shape[1], std.shape[-1], device=device, dtype=dtype
        )
    coefficients = coefficients.to(device=device, dtype=dtype)
    return torch.einsum("tk,bkd->btd", basis, coefficients * std * scale)


def project_actor_refinement(
    delta: torch.Tensor,
    *,
    basis_size: int | None,
    basis_type: str = "dct",
) -> torch.Tensor:
    """Remove the within-chunk high-frequency part of an actor correction.

    A fixed linear, zero-phase projection along the time axis of a
    ``[B, C, D]`` correction. It is differentiable, so the learner's Q sees
    exactly the projected correction the robot executes, and Q gradients only
    reach the actor through the same slow modes.
    """
    if basis_size is None or int(basis_size) >= delta.shape[1]:
        return delta
    projection = temporal_projection_matrix(
        delta.shape[1],
        int(basis_size),
        basis_type=basis_type,
        device=delta.device,
        dtype=delta.dtype,
    )
    return torch.einsum("ts,bsd->btd", projection, delta)


def smooth_actor_refinement(
    delta: torch.Tensor,
    *,
    alpha: float | None,
    taper_steps: int,
) -> torch.Tensor:
    """Smooth the actor residual and fade it into an RTC reference tail.

    Differentiable and side-effect free so the actor objective can apply the
    exact transform the robot executes. Optimizing Q at the raw sample while
    the robot runs the smoothed one trains the critic on actions that are
    never taken.
    """
    taper_steps = min(int(taper_steps), delta.shape[1])
    if (alpha is None or alpha == 1.0) and taper_steps == 0:
        return delta

    indices = arm_indices(delta.shape[-1], delta.device)
    smoothed = delta
    if alpha is not None and alpha < 1.0 and delta.shape[1] > 1:
        columns = [delta[:, 0]]
        previous = delta[:, 0, indices]
        for step in range(1, delta.shape[1]):
            previous = alpha * delta[:, step, indices] + (1.0 - alpha) * previous
            column = delta[:, step].index_copy(-1, indices, previous)
            columns.append(column)
        smoothed = torch.stack(columns, dim=1)

    if taper_steps > 0:
        weights = torch.linspace(
            1.0 - 1.0 / taper_steps,
            0.0,
            taper_steps,
            dtype=delta.dtype,
            device=delta.device,
        )
        scale = torch.ones(smoothed.shape[1], dtype=delta.dtype, device=delta.device)
        scale[-taper_steps:] = weights
        tapered = smoothed.index_select(-1, indices) * scale[None, :, None]
        smoothed = smoothed.index_copy(-1, indices, tapered)
    return smoothed


def arm_limit_vector(
    arm_limit: float | Sequence[float] | None, action_dim: int
) -> list[float] | None:
    """Per-dimension correction bound for one action step.

    ``arm_limit`` is one bound for every arm joint, or one bound per joint of
    an arm (6 values for 7-D Piper arms, 5 for 6-D SO101 arms; applied to each
    arm) or per action dimension. Gripper slots get ``inf`` here; the gripper
    bound is applied separately.
    """
    if arm_limit is None:
        return None
    if isinstance(arm_limit, (int, float)):
        return [float(arm_limit)] * action_dim
    values = [float(v) for v in arm_limit]
    if len(values) == action_dim:
        return values
    width = arm_width(action_dim)
    if len(values) == width - 1 and action_dim % width == 0:
        return (values + [float("inf")]) * (action_dim // width)
    raise ValueError(
        f"arm_limit needs 1, {width - 1} or {action_dim} values, got {len(values)}."
    )


def actor_refinement_stages(
    actor_actions: torch.Tensor,
    ref_actions: torch.Tensor,
    *,
    arm_limit: float | Sequence[float] | None,
    gripper_limit: float | None,
    alpha: float | None,
    taper_steps: int,
    basis_size: int | None = None,
    basis_type: str = "dct",
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Return matched residuals at each stage of the execution transform.

    Order: raw -> temporal projection -> clamp -> EMA -> taper. Projecting
    before the clamp keeps the per-joint bound exact.
    """
    delta = actor_actions - ref_actions
    raw = delta
    delta = project_actor_refinement(
        delta, basis_size=basis_size, basis_type=basis_type
    )
    projected = delta
    clipped = torch.zeros(delta.shape[0], dtype=torch.bool, device=delta.device)
    if arm_limit is not None or gripper_limit is not None:
        per_dim = arm_limit_vector(arm_limit, delta.shape[-1])
        limits = torch.tensor(
            per_dim if per_dim is not None else [float("inf")] * delta.shape[-1],
            dtype=delta.dtype,
            device=delta.device,
        )
        if gripper_limit is not None:
            limits[gripper_slice(limits.shape[-1])] = float(gripper_limit)
        bounded = torch.clamp(delta, min=-limits, max=limits)
        clipped = (bounded != delta).reshape(delta.shape[0], -1).any(dim=-1)
        delta = bounded
    smoothed = smooth_actor_refinement(delta, alpha=alpha, taper_steps=0)
    tapered = smooth_actor_refinement(smoothed, alpha=None, taper_steps=taper_steps)
    return {
        "raw": raw,
        "projected": projected,
        "clipped": delta,
        "smoothed": smoothed,
        "tapered": tapered,
    }, clipped


def bound_and_smooth_actor_refinement(
    actor_actions: torch.Tensor,
    ref_actions: torch.Tensor,
    *,
    arm_limit: float | Sequence[float] | None,
    gripper_limit: float | None,
    alpha: float | None,
    taper_steps: int,
    basis_size: int | None = None,
    basis_type: str = "dct",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the shared differentiable rollout/learner execution transform."""
    stages, clipped = actor_refinement_stages(
        actor_actions,
        ref_actions,
        arm_limit=arm_limit,
        gripper_limit=gripper_limit,
        alpha=alpha,
        taper_steps=taper_steps,
        basis_size=basis_size,
        basis_type=basis_type,
    )
    return ref_actions + stages["tapered"], clipped


def _base_ref_actions(
    ref_chunk: torch.Tensor,
    *,
    chunk_len: int,
    action_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if ref_chunk.dim() == 2:
        ref_chunk = ref_chunk.reshape(ref_chunk.shape[0], -1, action_dim)
    return ref_chunk[:, :chunk_len, :action_dim].to(device=device, dtype=dtype)


class RLTRoute(ABC):
    # Execution step inside the VLA plan at which the actor's chunk starts.
    actor_plan_offset: int = 0
    # Plan steps advanced per executed step inside the critical window.
    critical_reference_speed: float = 1.0

    @abstractmethod
    def route(self, ctx: RLTRouteContext) -> RLTRouteOutput:
        """Route student actions among actor, reference, and optional expert."""


class RealworldRLTRoute(RLTRoute):
    """Actor/ref routing for realworld RLT (keyboard or env switch flags)."""

    def __init__(
        self,
        *,
        use_reference_warmup: bool,
        warmup_updates: int,
        max_actor_reference_arm_delta: float | Sequence[float] | None = None,
        max_actor_reference_gripper_delta: float | None = None,
        max_actor_takeover_arm_deviation: float | None = None,
        max_actor_takeover_gripper_deviation: float | None = None,
        actor_refinement_smoothing_alpha: float | None = None,
        actor_refinement_tail_taper_steps: int = 0,
        actor_correction_basis_size: int | None = None,
        actor_correction_basis_type: str = "dct",
        actor_plan_offset: int = 0,
        exploration_arm_std: float = 0.0,
        exploration_gripper_std: float = 0.0,
        exploration_eval_every_n_episodes: int = 0,
        exploration_chunk_correlation: float = 0.0,
        actor_correction_slew_arm: float | Sequence[float] | None = None,
        actor_correction_slew_gripper: float | None = None,
        actor_correction_hold_through_bridge: bool = False,
        critical_reference_speed: float = 1.0,
    ):
        self.use_reference_warmup = bool(use_reference_warmup)
        # With execution-side RTC a chunk starts after ``actor_plan_offset``
        # bridge commands, so the actor fills that window of the VLA plan.
        self.actor_plan_offset = int(actor_plan_offset)
        if self.actor_plan_offset < 0:
            raise ValueError(
                f"actor_plan_offset must be non-negative, got {self.actor_plan_offset}."
            )
        self.warmup_updates = int(warmup_updates)
        self.max_actor_reference_arm_delta = (
            None
            if max_actor_reference_arm_delta is None
            else float(max_actor_reference_arm_delta)
            if isinstance(max_actor_reference_arm_delta, (int, float))
            else [float(v) for v in max_actor_reference_arm_delta]
        )
        self.max_actor_reference_gripper_delta = (
            None
            if max_actor_reference_gripper_delta is None
            else float(max_actor_reference_gripper_delta)
        )
        self.max_actor_takeover_arm_deviation = max_actor_takeover_arm_deviation
        self.max_actor_takeover_gripper_deviation = max_actor_takeover_gripper_deviation
        for name, limit in (
            ("max_actor_takeover_arm_deviation", max_actor_takeover_arm_deviation),
            (
                "max_actor_takeover_gripper_deviation",
                max_actor_takeover_gripper_deviation,
            ),
        ):
            if limit is not None and float(limit) <= 0:
                raise ValueError(f"{name} must be positive, got {limit}.")
        self.actor_refinement_smoothing_alpha = (
            None
            if actor_refinement_smoothing_alpha is None
            else float(actor_refinement_smoothing_alpha)
        )
        if (
            self.actor_refinement_smoothing_alpha is not None
            and not 0.0 < self.actor_refinement_smoothing_alpha <= 1.0
        ):
            raise ValueError(
                "actor_refinement_smoothing_alpha must be in (0, 1], got "
                f"{self.actor_refinement_smoothing_alpha}."
            )
        self.actor_refinement_tail_taper_steps = int(actor_refinement_tail_taper_steps)
        if self.actor_refinement_tail_taper_steps < 0:
            raise ValueError(
                "actor_refinement_tail_taper_steps must be non-negative, got "
                f"{self.actor_refinement_tail_taper_steps}."
            )
        self.actor_correction_basis_size = (
            None
            if actor_correction_basis_size is None
            else int(actor_correction_basis_size)
        )
        if (
            self.actor_correction_basis_size is not None
            and self.actor_correction_basis_size < 1
        ):
            raise ValueError(
                "actor_correction_basis_size must be positive, got "
                f"{self.actor_correction_basis_size}."
            )
        self.actor_correction_basis_type = str(actor_correction_basis_type)
        if self.actor_correction_basis_type not in ACTOR_CORRECTION_BASIS_TYPES:
            raise ValueError(
                "actor_correction_basis_type must be one of "
                f"{ACTOR_CORRECTION_BASIS_TYPES}, got "
                f"{self.actor_correction_basis_type!r}."
            )
        # Smooth exploration on Actor chunks during training collection. Every
        # ``exploration_eval_every_n_episodes``-th Actor episode runs the mean
        # policy instead, so success rates can be measured without noise.
        self.exploration_arm_std = float(exploration_arm_std)
        self.exploration_gripper_std = float(exploration_gripper_std)
        self.exploration_eval_every_n_episodes = int(exploration_eval_every_n_episodes)
        if (
            self.exploration_arm_std < 0
            or self.exploration_gripper_std < 0
            or self.exploration_eval_every_n_episodes < 0
        ):
            raise ValueError("Actor exploration settings must be non-negative.")
        self.exploration_enabled = (
            self.exploration_arm_std > 0 or self.exploration_gripper_std > 0
        )
        if self.exploration_enabled and self.actor_correction_basis_size is None:
            raise ValueError(
                "Actor exploration requires actor_correction_basis_size so the "
                "noise stays in the smooth correction subspace."
            )
        # AR(1) correlation of the exploration draw between consecutive chunks
        # of one episode. 0 redraws every chunk (the correction then swings
        # between chunks); 0.9 keeps it for ~10 chunks with a per-chunk change
        # of only sqrt(2 * (1 - 0.9)) = 0.45 of its RMS.
        self.exploration_chunk_correlation = float(exploration_chunk_correlation)
        if not 0.0 <= self.exploration_chunk_correlation < 1.0:
            raise ValueError("exploration_chunk_correlation must be in [0, 1).")
        self._actor_episode_count = 0
        self._previous_actor_switch: torch.Tensor | None = None
        self._episode_explores: torch.Tensor | None = None
        self._exploration_state: torch.Tensor | None = None
        # Execution-side continuity of the Actor correction. The RTC bridge
        # that runs while the next chunk is inferred used to execute the raw
        # VLA suffix, so every cycle applied the correction for C steps and
        # dropped it for the bridge (2026-10-14: 32% of Actor steps carried no
        # correction), a ~1 Hz push-release. Holding the last correction
        # through the bridge removes that; the slew limit lets each step move
        # at most ``slew * steps`` away from the correction held before it,
        # starting from zero whenever the Actor takes over.
        # One rate for every arm joint, or one per joint of an arm (6 values,
        # see ``arm_limit_vector``), matching ``max_actor_reference_arm_delta``.
        self.actor_correction_slew_arm = (
            None
            if actor_correction_slew_arm is None
            else float(actor_correction_slew_arm)
            if isinstance(actor_correction_slew_arm, (int, float))
            else [float(v) for v in actor_correction_slew_arm]
        )
        self.actor_correction_slew_gripper = (
            None
            if actor_correction_slew_gripper is None
            else float(actor_correction_slew_gripper)
        )
        for name, rate in (
            ("actor_correction_slew_arm", self.actor_correction_slew_arm),
            ("actor_correction_slew_gripper", self.actor_correction_slew_gripper),
        ):
            if rate is not None and float(np.min(rate)) <= 0:
                raise ValueError(f"{name} must be positive, got {rate}.")
        self.actor_correction_hold_through_bridge = bool(
            actor_correction_hold_through_bridge
        )
        self._held_correction: torch.Tensor | None = None
        # 2026-10-14: in the last ~3 chunks before contact JAX moved joints at
        # 0.0019 rad/step, 4x faster than a human aligning, i.e. 0.029 rad per
        # 15-step decision cycle against a correction that can change by at
        # most 0.012. Slowing the plan inside the critical window (after B)
        # gives the Actor proportionally more decisions before contact.
        self.critical_reference_speed = float(critical_reference_speed)
        if not 0.0 < self.critical_reference_speed <= 1.0:
            raise ValueError(
                "critical_reference_speed must be in (0, 1], got "
                f"{self.critical_reference_speed}."
            )

    def _bound_actor_refinement(
        self,
        actor_actions: torch.Tensor,
        ref_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return bound_and_smooth_actor_refinement(
            actor_actions,
            ref_actions,
            arm_limit=self.max_actor_reference_arm_delta,
            gripper_limit=self.max_actor_reference_gripper_delta,
            alpha=self.actor_refinement_smoothing_alpha,
            taper_steps=self.actor_refinement_tail_taper_steps,
            basis_size=self.actor_correction_basis_size,
            basis_type=self.actor_correction_basis_type,
        )

    def _exploration_mask(self, actor_switch: torch.Tensor, mode: str) -> torch.Tensor:
        """Return [B] flags for chunks that receive exploration noise.

        A new Actor episode starts on the rising edge of the per-env actor
        switch (``b`` after the warmup gate). The explore/eval decision is
        made once per episode and logged, so the operator and the episode
        statistics know which episodes measured the mean policy.
        """
        active = actor_switch[:, 0, 0].detach().to("cpu", torch.bool)
        previous = self._previous_actor_switch
        if previous is None or previous.shape != active.shape:
            previous = torch.zeros_like(active)
            self._episode_explores = torch.zeros_like(active)
        for env_index in torch.nonzero(active & ~previous).reshape(-1).tolist():
            self._actor_episode_count += 1
            every = self.exploration_eval_every_n_episodes
            evaluate = every > 0 and self._actor_episode_count % every == 0
            explores = self.exploration_enabled and mode == "train" and not evaluate
            self._episode_explores[env_index] = explores
            if self._exploration_state is not None:
                self._exploration_state[env_index] = torch.randn_like(
                    self._exploration_state[env_index]
                )
            get_logger().info(
                "[RLT exploration] Actor episode %d env=%d: %s",
                self._actor_episode_count,
                env_index,
                f"EXPLORE arm_std={self.exploration_arm_std:.4f}rad "
                f"gripper_std={self.exploration_gripper_std:.5f}m"
                if explores
                else "EVAL mean actor, no exploration noise",
            )
        self._previous_actor_switch = active
        return active & self._episode_explores

    def _slew_correction(
        self, delta: torch.Tensor, actor_switch: torch.Tensor
    ) -> torch.Tensor:
        """Limit the correction's change from the one held before this chunk.

        ``delta`` is the bounded ``[B, C, D]`` correction. Step ``t`` may differ
        from the held correction by at most ``slew * (t + 1)``; environments
        without Actor control hold zero, so a takeover ramps in from zero.
        """
        batch_size, chunk_len, action_dim = delta.shape
        active = actor_switch[:, 0, 0].detach().to("cpu", torch.bool)
        held = self._held_correction
        if held is None or tuple(held.shape) != (batch_size, action_dim):
            held = torch.zeros(batch_size, action_dim)
        held = torch.where(active[:, None], held, torch.zeros_like(held))
        if (
            self.actor_correction_slew_arm is not None
            or self.actor_correction_slew_gripper is not None
        ):
            per_dim = arm_limit_vector(self.actor_correction_slew_arm, action_dim)
            rate = torch.tensor(
                per_dim if per_dim is not None else [float("inf")] * action_dim,
                device=delta.device,
                dtype=delta.dtype,
            )
            if self.actor_correction_slew_gripper is not None:
                rate[gripper_slice(rate.shape[-1])] = self.actor_correction_slew_gripper
            steps = torch.arange(
                1, chunk_len + 1, device=delta.device, dtype=delta.dtype
            )[:, None]
            limit = rate * steps
            start = held.to(device=delta.device, dtype=delta.dtype)[:, None, :]
            delta = start + torch.clamp(delta - start, min=-limit, max=limit)
        last = delta[:, -1].detach().to("cpu", torch.float32)
        self._held_correction = torch.where(
            active[:, None], last, torch.zeros_like(last)
        )
        return delta

    def _next_exploration_draw(self, batch_size: int, action_dim: int) -> torch.Tensor:
        """Advance the per-env AR(1) standard-normal draw by one chunk."""
        shape = (batch_size, int(self.actor_correction_basis_size), action_dim)
        state = self._exploration_state
        if state is None or tuple(state.shape) != shape:
            # Episode starts reseed their own rows; this only sizes the state.
            state = torch.randn(shape)
        else:
            rho = self.exploration_chunk_correlation
            state = rho * state + (1.0 - rho * rho) ** 0.5 * torch.randn(shape)
        self._exploration_state = state
        return state

    def _ready_for_actor(self, version: int) -> bool:
        return not self.use_reference_warmup or int(version) >= self.warmup_updates

    def route(self, ctx: RLTRouteContext) -> RLTRouteOutput:
        actions = ctx.student_actions
        result = ctx.result
        # On the real robot, ``b`` means "this is the critical RLT window".
        # During warmup that window is recorded while the native-JAX reference
        # still controls the robot. Only a learner-synchronized update version
        # may turn the same request into an actor takeover.
        critical_phase = _normalize_rlt_switch_flags(
            actions,
            ctx.rlt_switch_flags,
            default=ctx.default_actor_switch,
        )
        actor_ready = self._ready_for_actor(ctx.version)
        actor_switch = critical_phase & torch.full_like(critical_phase, actor_ready)
        batch_size, chunk_len, action_dim = actions.shape
        reference_plan = result["forward_inputs"]["ref_chunk"].to(
            device=actions.device, dtype=actions.dtype
        )[..., :action_dim]
        if reference_plan.shape[1] < chunk_len:
            raise ValueError(
                f"The VLA plan has {reference_plan.shape[1]} steps, fewer than the "
                f"{chunk_len}-step actor chunk."
            )
        offset = min(self.actor_plan_offset, reference_plan.shape[1] - chunk_len)
        ref_actions = reference_plan[:, offset : offset + chunk_len]
        # Reject a whole unsafe chunk instead of clipping every joint to an
        # oscillating boundary. B still records the reference transition.
        if (
            self.max_actor_takeover_arm_deviation is not None
            or self.max_actor_takeover_gripper_deviation is not None
        ):
            delta = actions - ref_actions
            safe = torch.isfinite(delta).reshape(batch_size, -1).all(dim=-1)
            if self.max_actor_takeover_arm_deviation is not None:
                arm_delta = delta.index_select(
                    -1, arm_indices(action_dim, delta.device)
                )
                safe &= (
                    (arm_delta.abs() <= float(self.max_actor_takeover_arm_deviation))
                    .reshape(batch_size, -1)
                    .all(dim=-1)
                )
            if self.max_actor_takeover_gripper_deviation is not None:
                gripper_delta = delta[..., gripper_slice(delta.shape[-1])]
                safe &= (
                    (
                        gripper_delta.abs()
                        <= float(self.max_actor_takeover_gripper_deviation)
                    )
                    .reshape(batch_size, -1)
                    .all(dim=-1)
                )
            actor_switch &= safe[:, None, None]
        explores = self._exploration_mask(actor_switch, ctx.mode)
        if bool(explores.any()):
            std = torch.full(
                (action_dim,), self.exploration_arm_std, device=actions.device
            )
            std[gripper_slice(std.shape[-1])] = self.exploration_gripper_std
            noise = smooth_exploration_noise(
                batch_size,
                chunk_len,
                std,
                basis_size=self.actor_correction_basis_size,
                basis_type=self.actor_correction_basis_type,
                device=actions.device,
                dtype=actions.dtype,
                coefficients=self._next_exploration_draw(batch_size, action_dim),
            )
            actions = actions + noise * explores.to(actions.device)[:, None, None]
        stages, _ = actor_refinement_stages(
            actions,
            ref_actions,
            arm_limit=self.max_actor_reference_arm_delta,
            gripper_limit=self.max_actor_reference_gripper_delta,
            alpha=self.actor_refinement_smoothing_alpha,
            taper_steps=self.actor_refinement_tail_taper_steps,
            basis_size=self.actor_correction_basis_size,
            basis_type=self.actor_correction_basis_type,
        )
        stages["slewed"] = self._slew_correction(stages["tapered"], actor_switch)
        bounded_actor_actions = ref_actions + stages["slewed"]
        if action_dim == 14:
            for name, delta in stages.items():
                trace = torch.zeros_like(reference_plan)
                trace[:, offset : offset + chunk_len] = delta.detach()
                result["forward_inputs"][f"rlt_diag_{name}"] = trace
            coverage = torch.zeros(
                reference_plan.shape[:2], dtype=torch.bool, device=actions.device
            )
            coverage[:, offset : offset + chunk_len] = actor_switch[:, :, 0]
            result["forward_inputs"]["rlt_diag_coverage"] = coverage
            arm_bound = self.max_actor_reference_arm_delta
            params = [
                # Largest per-joint bound (diagnostics log one number).
                float(np.max(arm_bound)) if arm_bound is not None else float("inf"),
                self.max_actor_reference_gripper_delta
                if self.max_actor_reference_gripper_delta is not None
                else float("inf"),
                self.actor_refinement_smoothing_alpha
                if self.actor_refinement_smoothing_alpha is not None
                else 1.0,
                float(self.actor_refinement_tail_taper_steps),
                float(ctx.version),
                float(offset),
                float(chunk_len),
                # 0 = no temporal projection.
                float(self.actor_correction_basis_size or 0),
                # 0 = mean actor (no exploration noise on this chunk).
                self.exploration_arm_std if bool(explores[0]) else 0.0,
            ]
            result["forward_inputs"]["rlt_diag_params"] = (
                actions.new_tensor(params)[None].expand(batch_size, -1).clone()
            )
        window = torch.where(actor_switch, bounded_actor_actions, ref_actions)
        # EnvWorker aligns by execution step: after ``offset`` bridge commands
        # the robot runs plan[offset : offset + C], which is exactly the actor
        # window, and the untouched VLA suffix still feeds the next bridge.
        suffix = reference_plan[:, offset + chunk_len :]
        if self.actor_correction_hold_through_bridge and suffix.shape[1] > 0:
            # The suffix feeds the next RTC bridge: keep the last applied
            # correction there instead of dropping back to the raw VLA plan.
            held = stages["slewed"][:, -1:, :] * actor_switch[:, -1:, :].to(
                stages["slewed"].dtype
            )
            suffix = suffix + held
        execution_plan = torch.cat(
            (reference_plan[:, :offset], window, suffix),
            dim=1,
        ).contiguous()
        routed_actions = execution_plan[:, :chunk_len].contiguous()
        result["forward_inputs"]["rtc_execution_plan"] = execution_plan
        result["forward_inputs"]["action"] = routed_actions.reshape(
            batch_size, -1
        ).contiguous()
        result["forward_inputs"]["record_transition"] = critical_phase.reshape(
            actions.shape[0], -1
        )[:, :1].to(torch.bool)
        result["forward_inputs"]["actor_switch"] = actor_switch.reshape(
            actions.shape[0], -1
        )[:, :1].to(torch.bool)
        result["forward_inputs"]["reference_warmup"] = (
            (critical_phase & ~actor_switch)
            .reshape(actions.shape[0], -1)[:, :1]
            .to(torch.bool)
        )
        # ``forward_inputs`` is persisted in the replay cache. Keep its schema
        # stable across recovered buffers; the filter is execution-only.
        return RLTRouteOutput(actions=routed_actions, result=result)


class SimulatorRLTRoute(RLTRoute):
    """Actor/ref/expert routing for ManiSkill RLT with schedule warmup."""

    def __init__(self, *, use_schedule: bool, warmup_updates: int):
        self.use_schedule = use_schedule
        self.warmup_updates = warmup_updates

    def _ready_for_online(self, version: int) -> bool:
        return not self.use_schedule or int(version) >= self.warmup_updates

    def route(self, ctx: RLTRouteContext) -> RLTRouteOutput:
        actions = ctx.student_actions
        result = ctx.result
        batch_size, chunk_len, action_dim = actions.shape
        ready_for_online = self._ready_for_online(ctx.version)

        critical_phase = _last_info_bool(
            ctx.rlt_switch_flags,
            batch_size=batch_size,
            device=actions.device,
            default=False,
        )
        actor_switch = critical_phase
        if self.use_schedule:
            actor_switch = actor_switch & torch.full(
                (batch_size,),
                ready_for_online,
                dtype=torch.bool,
                device=actions.device,
            )

        requested_expert_takeover = _last_info_bool(
            ctx.intervene_requested,
            batch_size=batch_size,
            device=actions.device,
            default=False,
        )
        expert_takeover = (
            requested_expert_takeover
            & ready_for_online
            & (ctx.mode == "train")
            & (ctx.expert_model is not None)
        )

        base_actions = _base_ref_actions(
            ctx.rlt_obs["ref_chunk"],
            chunk_len=chunk_len,
            action_dim=action_dim,
            device=actions.device,
            dtype=actions.dtype,
        )
        routed_actions = torch.where(
            actor_switch[:, None, None],
            actions,
            base_actions,
        ).contiguous()

        intervene_flags = torch.zeros(
            (batch_size, chunk_len),
            dtype=torch.bool,
            device=actions.device,
        )
        if expert_takeover.any():
            if ctx.expert_model is None:
                raise RuntimeError(
                    "ManiSkill RLT expert takeover was requested, but expert_model "
                    "is not configured."
                )
            expert_actions = predict_expert_actions(
                ctx.expert_model,
                ctx.env_obs,
                chunk_len=chunk_len,
                action_dim=action_dim,
                device=actions.device,
                dtype=actions.dtype,
            )
            routed_actions = torch.where(
                expert_takeover[:, None, None],
                expert_actions,
                routed_actions,
            ).contiguous()
            intervene_flags[expert_takeover] = True

        forward_inputs = result["forward_inputs"]
        forward_inputs["action"] = _flatten_action_chunk(routed_actions).detach()
        forward_inputs["record_transition"] = critical_phase[:, None]
        forward_inputs["actor_switch"] = (actor_switch & ~expert_takeover)[:, None]
        forward_inputs["intervention_requested"] = requested_expert_takeover[:, None]
        result["intervene_flags"] = intervene_flags
        return RLTRouteOutput(actions=routed_actions, result=result)


def rlt_actor_plan_offset(cfg: Any) -> int:
    """RTC bridge length preceding every real-robot chunk; 0 without RTC."""
    schedule_cfg = cfg.algorithm.get("rlt_schedule", {}) or {}
    if bool(schedule_cfg.get("actor_direct_execution", False)):
        return 0
    if schedule_cfg.get("actor_plan_offset", None) is not None:
        return int(schedule_cfg.get("actor_plan_offset"))
    train_cfg = cfg.env.get("train", {}) or {}
    override_cfg = train_cfg.get("override_cfg", {}) or {}
    if not bool(override_cfg.get("rtc_enabled", False)):
        return 0
    # PiperWireEnvConfig.rtc_max_bridge_steps defaults to 5. The control
    # process drains the whole bridge before a new chunk starts, so this is
    # the exact offset whenever a bridge ran.
    return int(override_cfg.get("rtc_max_bridge_steps", 5))


def build_rlt_route(cfg: Any) -> RLTRoute:
    schedule_cfg = cfg.algorithm.get("rlt_schedule", {}) or {}
    if bool(schedule_cfg.get("actor_direct_execution", False)):
        schedule_cfg = {
            **schedule_cfg,
            "actor_refinement_smoothing_alpha": None,
            "actor_refinement_tail_taper_steps": 0,
        }
    if use_simulator_transition_replay(cfg):
        return SimulatorRLTRoute(
            use_schedule=bool(schedule_cfg.get("enable", False)),
            warmup_updates=int(schedule_cfg.get("warmup_post_collect_updates", 0)),
        )
    exploration_cfg = schedule_cfg.get("actor_exploration", None) or {}
    exploration_enabled = bool(exploration_cfg.get("enable", False))
    return RealworldRLTRoute(
        use_reference_warmup=bool(schedule_cfg.get("enable", False)),
        warmup_updates=int(schedule_cfg.get("warmup_post_collect_updates", 0)),
        max_actor_reference_arm_delta=schedule_cfg.get(
            "max_actor_reference_arm_delta", None
        ),
        max_actor_reference_gripper_delta=schedule_cfg.get(
            "max_actor_reference_gripper_delta", None
        ),
        max_actor_takeover_arm_deviation=schedule_cfg.get(
            "max_actor_takeover_arm_deviation", None
        ),
        max_actor_takeover_gripper_deviation=schedule_cfg.get(
            "max_actor_takeover_gripper_deviation", None
        ),
        actor_refinement_smoothing_alpha=schedule_cfg.get(
            "actor_refinement_smoothing_alpha", None
        ),
        actor_refinement_tail_taper_steps=int(
            schedule_cfg.get("actor_refinement_tail_taper_steps", 0)
        ),
        actor_correction_basis_size=schedule_cfg.get(
            "actor_correction_basis_size", None
        ),
        actor_correction_basis_type=schedule_cfg.get(
            "actor_correction_basis_type", "dct"
        ),
        exploration_arm_std=exploration_cfg.get("arm_std", 0.0)
        if exploration_enabled
        else 0.0,
        exploration_gripper_std=exploration_cfg.get("gripper_std", 0.0)
        if exploration_enabled
        else 0.0,
        exploration_eval_every_n_episodes=exploration_cfg.get(
            "eval_every_n_episodes", 0
        ),
        exploration_chunk_correlation=exploration_cfg.get("chunk_correlation", 0.0),
        actor_correction_slew_arm=schedule_cfg.get("actor_correction_slew_arm", None),
        actor_correction_slew_gripper=schedule_cfg.get(
            "actor_correction_slew_gripper", None
        ),
        actor_correction_hold_through_bridge=bool(
            schedule_cfg.get("actor_correction_hold_through_bridge", False)
        ),
        critical_reference_speed=float(
            schedule_cfg.get("critical_reference_speed", 1.0)
        ),
        actor_plan_offset=rlt_actor_plan_offset(cfg),
    )
