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

import asyncio
import os
import queue

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rlinf.algorithms.rlt.head_pose_shaping import HeadPoseShaping
from rlinf.algorithms.rlt.route import (
    bound_and_smooth_actor_refinement,
    smooth_exploration_noise,
)
from rlinf.algorithms.rlt.subsample import iter_stride_windows
from rlinf.algorithms.rlt.transition import use_transition_replay
from rlinf.data.embodied_buffer_dataset import (
    ReplayBufferDataset,
    replay_buffer_collate_fn,
)
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.mlp_policy.rlt_action_space import RLTDeltaActionSpace
from rlinf.scheduler import Worker
from rlinf.utils.arm_layout import gripper_slice, is_arm_joint
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import (
    append_to_dict,
    collect_trajectory_replay_metrics,
    compute_split_num,
    trajectory_has_bool_tensor,
)
from rlinf.utils.nested_dict_process import (
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
    AsyncEmbodiedSACFSDPPolicy,
)
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

# Routing tags kept in replay rows. Rollout forward_inputs also carry
# observation copies and stride-window slots that are stored elsewhere.
RLT_REPLAY_FORWARD_INPUT_KEYS = (
    "record_transition",
    "actor_switch",
    "reference_warmup",
    "rlt_episode_id",
    "action_valid_mask",
)


class RLTACLossMixin:
    """RLT actor-critic losses on top of RLinf replay-buffer worker plumbing.

    Forward types follow the existing off-policy actor-critic API, while the
    RLT objective disables entropy/alpha and uses a fixed-std actor, min-Q
    critic target (optionally TD3-smoothed), a min-Q or Q1 actor objective
    (``algorithm.actor_agg_q``), and BC regularization.
    """

    @staticmethod
    def _flatten_chunk(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() <= 2:
            return tensor
        return tensor.reshape(tensor.shape[0], -1)

    def _chunk_shape(self) -> tuple[int, int]:
        chunk_len = int(self.cfg.actor.model.num_action_chunks)
        action_dim = int(self.cfg.actor.model.action_dim)
        return chunk_len, action_dim

    def get_rollout_sync_version(self) -> int:
        """Expose learner update count when RLT warmup gates actor rollout."""
        if not self.use_rlt_schedule:
            return int(self.version)
        return int(self.update_step)

    def _ref_chunk(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        chunk_len, action_dim = self._chunk_shape()
        ref_chunk = self._flatten_chunk(obs["ref_chunk"]).reshape(
            obs["ref_chunk"].shape[0], -1, action_dim
        )
        return ref_chunk[:, :chunk_len].reshape(ref_chunk.shape[0], -1)

    def _apply_execution_transform(
        self,
        actions: torch.Tensor,
        ref_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Apply the clamp, smoothing and taper the robot applies before execution.

        Apply the route's relative-action bounds before Q queries. Raw BC
        retains gradients beyond these bounds. Downstream
        hardware limits and interpolation are not modeled by this transform.
        """
        schedule_cfg = self.cfg.algorithm.get("rlt_schedule", {}) or {}
        direct_actor = bool(schedule_cfg.get("actor_direct_execution", False))
        chunk_len, action_dim = self._chunk_shape()
        alpha = schedule_cfg.get("actor_refinement_smoothing_alpha", None)
        executed, _ = bound_and_smooth_actor_refinement(
            actions.reshape(-1, chunk_len, action_dim),
            ref_chunk.reshape(-1, chunk_len, action_dim).to(dtype=actions.dtype),
            arm_limit=schedule_cfg.get("max_actor_reference_arm_delta", None),
            gripper_limit=schedule_cfg.get("max_actor_reference_gripper_delta", None),
            alpha=None if direct_actor or alpha is None else float(alpha),
            taper_steps=0
            if direct_actor
            else int(schedule_cfg.get("actor_refinement_tail_taper_steps", 0)),
            basis_size=schedule_cfg.get("actor_correction_basis_size", None),
            basis_type=schedule_cfg.get("actor_correction_basis_type", "dct"),
        )
        return executed.reshape(actions.shape)

    def _target_policy_smoothing(self, next_actions: torch.Tensor) -> torch.Tensor:
        """TD3 target policy smoothing: clipped Gaussian noise on target actions.

        Noise is in physical units (rad for arm joints, m for grippers) and is
        added before the execution transform, so the bootstrapped action still
        passes through the same projection and reference bound as the robot.
        Averaging the TD target over nearby actions keeps Q smooth in action
        space and limits exploitation of sharp critic errors.
        """
        smoothing_cfg = self.cfg.algorithm.get("target_policy_smoothing", None)
        if not smoothing_cfg or not bool(smoothing_cfg.get("enable", False)):
            return next_actions
        _, action_dim = self._chunk_shape()
        std = torch.full(
            (action_dim,),
            float(smoothing_cfg.get("arm_std", 0.0)),
            device=next_actions.device,
            dtype=next_actions.dtype,
        )
        clip = torch.full_like(std, float(smoothing_cfg.get("arm_clip", 0.0)))
        std[gripper_slice(std.shape[-1])] = float(smoothing_cfg.get("gripper_std", 0.0))
        clip[gripper_slice(clip.shape[-1])] = float(smoothing_cfg.get("gripper_clip", 0.0))
        if bool((std < 0).any()) or bool((clip < 0).any()):
            raise ValueError("target_policy_smoothing std/clip must be non-negative.")
        chunk = next_actions.reshape(next_actions.shape[0], -1, action_dim)
        noise = torch.clamp(torch.randn_like(chunk) * std, min=-clip, max=clip)
        return (chunk + noise).reshape(next_actions.shape)

    def _rlt_action_space(self, device: torch.device) -> RLTDeltaActionSpace:
        """The actor's normalized delta action space, shared by BC and metrics."""
        space = getattr(self, "_cached_rlt_action_space", None)
        if space is None:
            space = RLTDeltaActionSpace.from_config(
                self.cfg.actor.model.get("action_norm_stats", None),
                action_dim=int(self.cfg.actor.model.action_dim),
            )
            self._cached_rlt_action_space = space
        return space.to(device)

    @staticmethod
    def _require_twin_q(all_q_values: torch.Tensor) -> None:
        if all_q_values.shape[-1] < 2:
            raise ValueError(
                "RLT Stage 2 requires at least two Q heads for twin-Q training, "
                f"got Q shape {tuple(all_q_values.shape)}."
            )

    def _min_twin_q(self, all_q_values: torch.Tensor) -> torch.Tensor:
        self._require_twin_q(all_q_values)
        return torch.minimum(all_q_values[..., 0:1], all_q_values[..., 1:2])

    def _q1(self, all_q_values: torch.Tensor) -> torch.Tensor:
        self._require_twin_q(all_q_values)
        return all_q_values[..., 0:1]

    def _actor_q(self, all_q_values: torch.Tensor) -> torch.Tensor:
        """Select Q1 for openpi-RLT or conservative twin-Q for existing configs."""
        if self.cfg.algorithm.get("actor_agg_q", "min") == "first":
            return self._q1(all_q_values)
        return self._min_twin_q(all_q_values)

    def _discounted_chunk_return(
        self,
        rewards: torch.Tensor,
        valid_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return masked chunk rewards and gamma raised to executed steps.

        Full chunks retain the previous ``gamma ** chunk_len`` behavior. A
        partial terminal/interrupted chunk discounts and bootstraps using only
        the left-aligned control steps that actually executed.
        """
        rewards = rewards.reshape(rewards.shape[0], -1)
        rewards = rewards.to(self.torch_dtype)
        chunk_len = rewards.shape[-1]
        if valid_mask is None:
            valid = torch.ones_like(rewards, dtype=torch.bool)
        else:
            valid = valid_mask.to(device=rewards.device, dtype=torch.bool).reshape(
                rewards.shape[0], -1
            )
            if valid.shape != rewards.shape:
                raise ValueError(
                    "RLT action_valid_mask must match chunk rewards, got "
                    f"{tuple(valid.shape)} and {tuple(rewards.shape)}."
                )
        if not bool(valid.any(dim=-1).all()):
            raise ValueError("Every RLT transition must contain an executed action.")
        # Valid rows are a real prefix followed by terminal padding. Reject a
        # hole so gamma cannot silently count a non-executed control step.
        if chunk_len > 1 and bool((~valid[:, :-1] & valid[:, 1:]).any()):
            raise ValueError("RLT action_valid_mask must be left-aligned.")
        discounts = torch.pow(
            torch.as_tensor(self.cfg.algorithm.gamma, device=rewards.device),
            torch.arange(chunk_len, device=rewards.device, dtype=rewards.dtype),
        )
        reward_return = torch.sum(
            rewards * valid.to(rewards.dtype) * discounts,
            dim=-1,
            keepdim=True,
        )
        executed_steps = valid.sum(dim=-1, keepdim=True).to(rewards.dtype)
        bootstrap_discount = torch.pow(
            torch.as_tensor(
                self.cfg.algorithm.gamma,
                device=rewards.device,
                dtype=rewards.dtype,
            ),
            executed_steps,
        )
        return reward_return, bootstrap_discount

    @torch.no_grad()
    def _q_action_contrast(
        self,
        curr_obs: dict[str, torch.Tensor],
        raw_pi: torch.Tensor,
        q_pi: torch.Tensor,
    ) -> dict[str, float]:
        """How much min-Q changes under one smooth exploration perturbation.

        Near zero means the critic cannot tell nearby corrections apart, so a
        larger Q weight would only amplify noise. Uses the configured
        exploration scale and the shared execution transform.
        """
        schedule_cfg = self.cfg.algorithm.get("rlt_schedule", {}) or {}
        exploration_cfg = schedule_cfg.get("actor_exploration", None) or {}
        basis_size = schedule_cfg.get("actor_correction_basis_size", None)
        arm_std = float(exploration_cfg.get("arm_std", 0.0))
        if basis_size is None or arm_std <= 0:
            return {}
        chunk_len, action_dim = self._chunk_shape()
        std = torch.full((action_dim,), arm_std, device=raw_pi.device)
        std[gripper_slice(std.shape[-1])] = float(exploration_cfg.get("gripper_std", 0.0))
        noise = smooth_exploration_noise(
            raw_pi.shape[0],
            chunk_len,
            std,
            basis_size=int(basis_size),
            basis_type=schedule_cfg.get("actor_correction_basis_type", "dct"),
            device=raw_pi.device,
            dtype=raw_pi.dtype,
        )
        perturbed = self._apply_execution_transform(
            raw_pi.detach() + noise.reshape(raw_pi.shape), self._ref_chunk(curr_obs)
        )
        q_perturbed = self._min_twin_q(
            self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=perturbed,
                detach_encoder=True,
            )
        )
        contrast = (q_perturbed - q_pi.detach()).abs().mean()
        return {
            "q_action_contrast": contrast.item(),
            "q_action_contrast_rel": (
                contrast / q_pi.detach().abs().mean().clamp(min=1e-6)
            ).item(),
        }

    def _critic_human_alignment(
        self,
        qf_pi: torch.Tensor,
        pi: torch.Tensor,
        batch: dict,
        curr_obs: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Cosine between dQ/da and the operator's correction on teaching rows.

        Teaching rows are fully operator-driven windows whose reference still
        holds the VLA plan. Near zero means the critic does not know which way
        to correct, so a large Q weight would only add noise to the Actor.
        """
        flags = batch.get("intervene_flags")
        if flags is None or not pi.requires_grad:
            return {}
        chunk_len, action_dim = self._chunk_shape()
        reference = self._ref_chunk(curr_obs).to(pi.dtype)
        actions = self._flatten_chunk(batch["actions"]).to(pi.dtype)
        human = (
            self._flatten_chunk(flags).bool().reshape(-1, chunk_len, action_dim).any(-1)
        )
        correction = (actions - reference).reshape(-1, chunk_len, action_dim)
        teach = human.all(dim=-1) & (correction.abs().amax(dim=(1, 2)) > 1e-5)
        if not bool(teach.any()):
            return {}
        (grad,) = torch.autograd.grad(qf_pi.sum(), pi, retain_graph=True)
        arm = [index for index in range(action_dim) if is_arm_joint(index, action_dim)]
        g = grad.detach().reshape(-1, chunk_len, action_dim)[:, :, arm].mean(1)[teach]
        h = correction[:, :, arm].mean(1)[teach]
        cosine = (g * h).sum(-1) / (g.norm(dim=-1) * h.norm(dim=-1) + 1e-12)
        return {
            "critic_human_grad_cos": cosine.mean().item(),
            "critic_human_teach_rows": float(teach.sum().item()),
        }

    @staticmethod
    def _should_update_actor(update_step: int, critic_actor_ratio: int) -> bool:
        """Apply one actor update per configured number of critic updates."""
        if int(critic_actor_ratio) <= 0:
            raise ValueError("critic_actor_ratio must be positive.")
        return int(update_step) % int(critic_actor_ratio) == 0

    def _bc_metrics(
        self,
        pi: torch.Tensor,
        actions: torch.Tensor,
        ref_chunk: torch.Tensor,
        intervene_flags: torch.Tensor | None,
        proprio: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        chunk_len, action_dim = self._chunk_shape()
        space = self._rlt_action_space(pi.device)
        proprio = self._flatten_chunk(proprio)
        pi_chunk = space.normalize(self._flatten_chunk(pi), proprio).reshape(
            -1, chunk_len, action_dim
        )
        action_chunk = space.normalize(
            self._flatten_chunk(actions).to(dtype=pi.dtype), proprio
        ).reshape(-1, chunk_len, action_dim)
        bc_ref_chunk = self._flatten_chunk(ref_chunk).reshape(
            ref_chunk.shape[0], -1, action_dim
        )[:, :chunk_len]
        bc_ref_chunk = space.normalize(
            bc_ref_chunk.reshape(bc_ref_chunk.shape[0], -1).to(dtype=pi.dtype),
            proprio,
        ).reshape(-1, chunk_len, action_dim)
        if intervene_flags is None:
            human_mask = torch.zeros(
                pi_chunk.shape[:2], dtype=torch.bool, device=pi_chunk.device
            )
        else:
            human_mask = (
                self._flatten_chunk(intervene_flags)
                .to(device=pi_chunk.device)
                .bool()
                .reshape(pi_chunk.shape[0], chunk_len, -1)
                .any(dim=-1)
            )
        valid = (
            torch.ones_like(human_mask)
            if valid_mask is None
            else valid_mask.to(device=pi.device, dtype=torch.bool).reshape_as(
                human_mask
            )
        )
        # Operator steps stored with paper IV-B replacement carry the operator
        # action as their reference, so BC there teaches "copy the input" (a
        # zero correction exactly where the operator corrected). Skip them.
        replaced = human_mask & (
            (action_chunk - bc_ref_chunk).abs().amax(dim=-1) < 1e-5
        )
        if self.cfg.algorithm.get("bc_exclude_replaced_human_reference", False):
            valid = valid & ~replaced
        bc_target = torch.where(human_mask[..., None], action_chunk, bc_ref_chunk)
        bc_error = torch.mean(torch.square(pi_chunk - bc_target), dim=-1)
        # Non-operator rows target the VLA reference (a zero residual). Near
        # the takeover states they cancel the operator corrections, so they
        # can be down-weighted; the residual head already defaults to zero.
        policy_row_weight = float(self.cfg.algorithm.get("bc_policy_row_weight", 1.0))
        if policy_row_weight < 0:
            raise ValueError("bc_policy_row_weight must be non-negative.")
        weights = valid.to(bc_error.dtype) * torch.where(
            human_mask,
            torch.ones_like(bc_error),
            torch.full_like(bc_error, policy_row_weight),
        )
        bc_loss = (bc_error * weights).sum() / weights.sum().clamp(min=1e-6)

        replaced_ratio = (replaced.sum() / human_mask.sum().clamp(min=1)).item()
        human_mask = human_mask & valid
        policy_mask = ~human_mask & valid
        ref_error = torch.mean(torch.square(pi_chunk - bc_ref_chunk), dim=-1)
        human_error = torch.mean(torch.square(pi_chunk - action_chunk), dim=-1)
        bc_ref = torch.sum(ref_error * policy_mask.to(ref_error.dtype)) / torch.clamp(
            torch.sum(policy_mask.to(ref_error.dtype)), min=1.0
        )
        bc_human = torch.sum(
            human_error * human_mask.to(human_error.dtype)
        ) / torch.clamp(torch.sum(human_mask.to(human_error.dtype)), min=1.0)

        human_ratio = (human_mask.sum() / valid.sum().clamp(min=1)).item()
        metrics = {
            "bc_loss": bc_loss.detach().item(),
            "bc_ref_loss": bc_ref.detach().item(),
            "bc_human_loss": bc_human.detach().item(),
            "human_mask_ratio": human_ratio,
            "policy_mask_ratio": 1.0 - human_ratio,
            "human_reference_replaced_ratio": replaced_ratio,
        }
        return bc_loss, metrics

    def _delta_penalty(self, raw_pi, batch) -> tuple[torch.Tensor, dict[str, float]]:
        """openpi-RLT: match physical adjacent arm increments to BC targets.

        Apply to all six arm joints on each Piper side, excluding grippers.
        C's padding and source-switch pairs are masked out. Raw supervision
        retains gradients when the execution correction bound is saturated.
        """
        chunk_len, action_dim = self._chunk_shape()
        predicted = raw_pi.reshape(-1, chunk_len, action_dim)
        reference = self._ref_chunk(batch["curr_obs"]).reshape_as(predicted)
        recorded = batch["actions"].reshape_as(predicted)
        flags = batch.get("intervene_flags")
        human = torch.zeros(predicted.shape[:2], dtype=torch.bool, device=raw_pi.device)
        if flags is not None:
            human = (
                flags.to(device=raw_pi.device)
                .bool()
                .reshape(predicted.shape[0], chunk_len, -1)
                .any(dim=-1)
            )
        target = torch.where(human[..., None], recorded, reference).detach()
        valid = batch.get("forward_inputs", {}).get("action_valid_mask")
        valid = (
            torch.ones_like(human)
            if valid is None
            else valid.to(device=raw_pi.device, dtype=torch.bool).reshape_as(human)
        )
        indices = [i for i in range(action_dim) if is_arm_joint(i, action_dim)]
        increments = predicted[:, 1:, indices] - predicted[:, :-1, indices]
        target_increments = target[:, 1:, indices] - target[:, :-1, indices]
        error = (increments - target_increments).square().mean(dim=-1)
        pairs = valid[:, 1:] & valid[:, :-1] & (human[:, 1:] == human[:, :-1])
        penalty = (error * pairs).sum() / pairs.sum().clamp(min=1)
        return penalty, {
            "delta_penalty": penalty.detach().item(),
            "delta_valid_pairs": float(pairs.sum().item()),
        }

    def _actor_objective_weights(self) -> tuple[float, float, dict[str, float]]:
        """Resolve RLT actor-objective BC/Q weights."""
        schedule_cfg = self.cfg.algorithm.get("actor_weight_schedule", {})
        schedule_enabled = bool(schedule_cfg.get("enable", False))
        if not schedule_enabled:
            bc_weight = float(self.cfg.algorithm.get("bc_weight", 1.0))
            q_weight = float(self.cfg.algorithm.get("q_weight", 1.0))
            return (
                bc_weight,
                q_weight,
                {
                    "bc_weight": bc_weight,
                    "q_weight": q_weight,
                    "actor_weight_schedule_enabled": 0.0,
                    "actor_weight_in_warmup": 0.0,
                    "actor_weight_ramp_progress": 1.0,
                },
            )

        weight_warmup_updates = int(schedule_cfg.get("warmup_updates", 0))
        ramp_updates = int(schedule_cfg.get("ramp_updates", 0))
        in_warmup = int(self.update_step) < weight_warmup_updates
        warmup_bc_weight = float(
            schedule_cfg.get(
                "warmup_bc_weight",
                self.cfg.algorithm.get("bc_weight", 1.0),
            )
        )
        warmup_q_weight = float(
            schedule_cfg.get(
                "warmup_q_weight",
                self.cfg.algorithm.get("q_weight", 1.0),
            )
        )
        online_bc_weight = float(
            schedule_cfg.get(
                "online_bc_weight",
                self.cfg.algorithm.get("bc_weight", 1.0),
            )
        )
        online_q_weight = float(
            schedule_cfg.get(
                "online_q_weight",
                self.cfg.algorithm.get("q_weight", 1.0),
            )
        )
        if in_warmup:
            bc_weight = warmup_bc_weight
            q_weight = warmup_q_weight
            ramp_progress = 0.0
        elif ramp_updates > 0:
            ramp_progress = min(
                1.0,
                max(
                    0.0,
                    float(int(self.update_step) - weight_warmup_updates + 1)
                    / float(ramp_updates),
                ),
            )
            bc_weight = warmup_bc_weight + ramp_progress * (
                online_bc_weight - warmup_bc_weight
            )
            q_weight = warmup_q_weight + ramp_progress * (
                online_q_weight - warmup_q_weight
            )
        else:
            bc_weight = online_bc_weight
            q_weight = online_q_weight
            ramp_progress = 1.0

        metrics = {
            "bc_weight": bc_weight,
            "q_weight": q_weight,
            "actor_weight_schedule_enabled": 1.0,
            "actor_weight_in_warmup": float(in_warmup),
            "actor_weight_ramp_progress": ramp_progress,
        }
        return bc_weight, q_weight, metrics

    def _head_pose_shaping(self) -> HeadPoseShaping | None:
        """``algorithm.head_pose_shaping`` (potential-based), built once."""
        if not hasattr(self, "_head_pose_shaping_cache"):
            self._head_pose_shaping_cache = HeadPoseShaping.from_config(
                self.cfg.algorithm.get("head_pose_shaping", None)
            )
        return self._head_pose_shaping_cache

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"
        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")

        curr_obs = batch["curr_obs"]
        next_obs = batch["next_obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        done_source = batch["terminations"]
        if use_transition_replay(self.cfg):
            done_source = batch["dones"]
        done_source = done_source.to(self.torch_dtype)
        not_done = ~done_source.reshape(done_source.shape[0], -1).bool().any(
            dim=-1, keepdim=True
        )

        with torch.no_grad():
            # B-style delayed target actor: the existing C target_model is a
            # full EMA copy (actor + twin critic) updated at C's current
            # target_update_freq. Reference dropout remains disabled here.
            next_actions, _, _ = self.target_model(
                forward_type=ForwardType.SAC,
                obs=next_obs,
            )
            next_actions = self._target_policy_smoothing(next_actions)
            # Bootstrap with the action the robot would execute at next_obs.
            next_actions = self._apply_execution_transform(
                next_actions, self._ref_chunk(next_obs)
            )

            if not use_crossq:
                all_qf_next_target = self.target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=next_obs,
                    actions=next_actions,
                )
                q_next = self._min_twin_q(all_qf_next_target)
            else:
                _, all_qf_next = self.model(
                    forward_type=ForwardType.CROSSQ_Q,
                    obs=curr_obs,
                    actions=actions,
                    next_obs=next_obs,
                    next_actions=next_actions,
                )
                q_next = self._min_twin_q(all_qf_next.detach())

            valid_mask = batch.get("forward_inputs", {}).get("action_valid_mask")
            reward_target, bootstrap_discount = self._discounted_chunk_return(
                rewards,
                valid_mask,
            )
            shaping = self._head_pose_shaping()
            shaping_metrics = {}
            if (
                shaping is not None
                and curr_obs.get("head_pose") is not None
                and next_obs.get("head_pose") is not None
            ):
                shaping_term, shaping_metrics = shaping(
                    curr_obs["head_pose"], next_obs["head_pose"], bootstrap_discount
                )
                reward_target = reward_target + shaping_term.to(reward_target)
            if bootstrap_type == "always":
                target_q_values = reward_target + bootstrap_discount * q_next
            elif bootstrap_type == "standard":
                target_q_values = reward_target + not_done * bootstrap_discount * q_next
            else:
                raise NotImplementedError(f"{bootstrap_type=} is not supported!")
            # With terminal 0/1 rewards the true Q is in [0, 1]; bounding the TD
            # target stops a target critic from bootstrapping itself out of that
            # range (RL-Token-Pi05 clips to [-2, 2]).
            td_clip = self.cfg.algorithm.get("td_target_clip", None)
            if td_clip is not None:
                # The shaped critic learns Q - phi(s), with -phi in
                # [0, max_abs_potential].
                upper = float(td_clip[1]) + (
                    shaping.max_abs_potential if shaping is not None else 0.0
                )
                target_q_values = target_q_values.clamp(
                    min=float(td_clip[0]), max=upper
                )

        if not use_crossq:
            all_data_q_values = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=actions,
            )
        else:
            all_data_q_values, _ = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=actions,
                next_obs=next_obs,
                next_actions=next_actions,
            )

        target_q_values = target_q_values.to(dtype=all_data_q_values.dtype)
        critic_loss = F.mse_loss(
            all_data_q_values, target_q_values.expand_as(all_data_q_values)
        )
        return critic_loss, {
            "q_data": all_data_q_values.mean().item(),
            **shaping_metrics,
        }

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        use_crossq = self.cfg.algorithm.get("q_head_type", "default") == "crossq"

        curr_obs = batch["curr_obs"]
        reference_dropout_prob = float(
            self.cfg.algorithm.get("reference_dropout_prob", 0.0)
        )
        pi, log_pi, _ = self.model(
            forward_type=ForwardType.SAC,
            obs=curr_obs,
            apply_reference_dropout=True,
            reference_dropout_prob=reference_dropout_prob,
        )
        if log_pi.ndim == 1:
            log_pi = log_pi.unsqueeze(-1)
        log_pi = log_pi.sum(dim=-1, keepdim=True)
        # Q evaluates the bounded action; BC must retain a gradient to raw
        # outputs even when the execution clamp is saturated.
        raw_pi = pi
        pi = self._apply_execution_transform(raw_pi, self._ref_chunk(curr_obs))

        if not use_crossq:
            all_qf_pi = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=curr_obs,
                actions=pi,
                detach_encoder=True,
            )
        else:
            all_qf_pi, _ = self.model(
                forward_type=ForwardType.CROSSQ_Q,
                obs=curr_obs,
                actions=pi,
                next_obs=None,
                next_actions=None,
                detach_encoder=True,
            )

        num_q_values = all_qf_pi.shape[-1]
        metrics = {
            f"q_value_{q_id}": all_qf_pi[..., q_id].mean().item()
            for q_id in range(num_q_values)
        }
        qf_pi = self._actor_q(all_qf_pi)
        metrics["q_pi"] = qf_pi.mean().item()
        metrics.update(self._critic_human_alignment(qf_pi, pi, batch, curr_obs))
        if not use_crossq:
            metrics.update(
                self._q_action_contrast(curr_obs, raw_pi, self._min_twin_q(all_qf_pi))
            )

        ref_chunk = self._ref_chunk(curr_obs)
        bc_loss, rlt_metrics = self._bc_metrics(
            pi=raw_pi,
            actions=batch["actions"],
            ref_chunk=ref_chunk,
            intervene_flags=batch.get("intervene_flags", None),
            proprio=curr_obs["proprio"],
            valid_mask=batch.get("forward_inputs", {}).get("action_valid_mask"),
        )
        metrics.update(rlt_metrics)

        entropy = -log_pi.mean()
        bc_weight, q_weight, weight_metrics = self._actor_objective_weights()
        actor_loss = -q_weight * qf_pi.mean() + bc_weight * bc_loss
        delta_weight = float(self.cfg.algorithm.get("delta_weight", 0.0))
        if delta_weight < 0 or not torch.isfinite(torch.tensor(delta_weight)):
            raise ValueError("delta_weight must be finite and non-negative.")
        if delta_weight > 0:
            delta_penalty, delta_metrics = self._delta_penalty(raw_pi, batch)
            actor_loss = actor_loss + delta_weight * delta_penalty
            metrics.update(delta_metrics)
            metrics["weighted_delta"] = (delta_weight * delta_penalty).detach().item()
        metrics["delta_weight"] = delta_weight
        metrics.update(weight_metrics)
        action_ref_abs = (
            self._flatten_chunk(pi) - self._flatten_chunk(ref_chunk)
        ).abs()
        metrics["action_ref_abs_mean"] = action_ref_abs.mean().detach().item()
        metrics["action_ref_abs_max"] = action_ref_abs.max().detach().item()
        space = self._rlt_action_space(pi.device)
        proprio = self._flatten_chunk(curr_obs["proprio"])
        action_ref_norm_abs = (
            space.normalize(self._flatten_chunk(pi), proprio)
            - space.normalize(self._flatten_chunk(ref_chunk).to(pi.dtype), proprio)
        ).abs()
        metrics["action_ref_norm_abs_mean"] = action_ref_norm_abs.mean().detach().item()
        metrics["action_ref_norm_abs_max"] = action_ref_norm_abs.max().detach().item()
        metrics["weighted_q"] = (q_weight * qf_pi.mean()).detach().item()
        metrics["weighted_bc"] = (bc_weight * bc_loss).detach().item()
        metrics["reference_dropout_prob"] = reference_dropout_prob

        return actor_loss, entropy, metrics

    @Worker.timer("forward_alpha")
    def forward_alpha(self, batch):
        del batch
        raise NotImplementedError(
            "RLT AC disables entropy/alpha training. Use "
            "algorithm.entropy_tuning.alpha_type=fixed_alpha."
        )


class RLTACReplayMixin:
    def _setup_split_sampling(self) -> None:
        """Give the critic its own sampling stream when configured.

        ``algorithm.replay_buffer.critic_sample_strategy`` (e.g. ``uniform``)
        makes the critic sample the natural replay distribution while the
        actor keeps the buffer's stratified shares (operator rows
        over-weighted for BC). Without it both share one batch as before.
        """
        replay_cfg = self.cfg.algorithm.replay_buffer
        critic_strategy = replay_cfg.get("critic_sample_strategy", None)
        self._split_actor_sampling = critic_strategy is not None
        if not self._split_actor_sampling:
            return
        batch_size = self.cfg.actor.global_batch_size // self._world_size
        min_size = 1 if self.use_rlt_schedule else replay_cfg.min_buffer_size

        def loader(sample_kwargs):
            dataset = ReplayBufferDataset(
                replay_buffer=self.replay_buffer,
                demo_buffer=None,
                batch_size=batch_size,
                min_replay_buffer_size=min_size,
                min_demo_buffer_size=0,
                sample_kwargs=sample_kwargs,
            )
            return dataset, iter(
                DataLoader(
                    dataset,
                    batch_size=1,
                    num_workers=0,
                    drop_last=True,
                    collate_fn=replay_buffer_collate_fn,
                )
            )

        self.buffer_dataset, self.buffer_dataloader_iter = loader(
            {"strategy": str(critic_strategy)}
        )
        self.actor_buffer_dataset, self.actor_buffer_dataloader_iter = loader({})

    def _actor_micro_batches(self, critic_micro_batches: list) -> list:
        if not getattr(self, "_split_actor_sampling", False):
            return critic_micro_batches
        global_batch = next(self.actor_buffer_dataloader_iter)
        micro_batches = split_dict_to_chunk(
            global_batch,
            (self.cfg.actor.global_batch_size // self._world_size)
            // self.cfg.actor.micro_batch_size,
        )
        return [put_tensor_device(batch, device=self.device) for batch in micro_batches]

    """Shared rollout-to-replay ingestion for sync and async RLT AC workers."""

    @staticmethod
    def _infer_rlt_update_step(
        *,
        replay_size: int,
        warmup_min_size: int,
        warmup_updates: int,
        update_epoch: int,
    ) -> int:
        """Infer the strict-UTD update count for legacy checkpoints."""
        if int(replay_size) < int(warmup_min_size):
            return 0
        online_transitions = max(int(replay_size) - int(warmup_min_size), 0)
        return int(warmup_updates) + online_transitions * int(update_epoch)

    def _rlt_checkpoint_state_path(self, load_base_path: str) -> str:
        return os.path.join(
            load_base_path,
            "sac_components",
            f"rlt_schedule_state_rank_{self._rank}.pt",
        )

    def save_checkpoint(self, save_base_path, step):
        """Save SAC state plus RLT's Python-side strict-UTD counters."""
        super().save_checkpoint(save_base_path, step)
        if not getattr(self, "use_rlt_schedule", False):
            return
        state = {
            "update_step": int(self.update_step),
            "transitions_since_train": int(self.transitions_since_train),
            "episodes_since_train": int(self.episodes_since_train),
            "total_transitions_added": int(self.total_transitions_added),
            "total_episodes_added": int(self.total_episodes_added),
            "warmup_ready_total_transitions": self._warmup_ready_total_transitions,
            "warmup_ready_total_episodes": self._warmup_ready_total_episodes,
            "pending_update_budget": int(self.pending_update_budget),
        }
        state_path = self._rlt_checkpoint_state_path(save_base_path)
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        torch.save(state, state_path)

    def load_checkpoint(self, load_base_path):
        """Restore SAC state and keep the RLT update budget monotonic."""
        super().load_checkpoint(load_base_path)
        # Resuming must not silently restore an old, larger Actor learning rate.
        if self.cfg.actor.get("rlt_reset_actor_lr_on_resume", False):
            lr = float(self.cfg.actor.optim.lr)
            for group in self.optimizer.param_groups:
                group["lr"] = group["initial_lr"] = lr
            self.lr_scheduler.base_lrs = [lr] * len(self.optimizer.param_groups)
            self.lr_scheduler._last_lr = [lr] * len(self.optimizer.param_groups)
        replay_load_path = self.cfg.algorithm.replay_buffer.get("load_path", None)
        if replay_load_path is not None:
            # Real-robot checkpoints only snapshot transitions collected by the
            # current run. The historical warmup files remain in the explicit
            # replay load path, so restore that complete, merged buffer after
            # model/optimizer state is loaded instead of sampling a checkpoint
            # index whose preloaded files are absent.
            self.replay_buffer.load_checkpoint(
                replay_load_path,
                is_distributed=True,
                local_rank=self._rank,
                world_size=self._world_size,
            )
            self._logger.info(
                "Reloaded %d complete persisted RLT replay trajectories from %s "
                "after checkpoint restore.",
                self.replay_buffer.size,
                replay_load_path,
            )
        if not getattr(self, "use_rlt_schedule", False):
            return
        state_path = self._rlt_checkpoint_state_path(load_base_path)
        if os.path.isfile(state_path):
            state = torch.load(state_path, map_location="cpu", weights_only=True)
            self.update_step = int(state["update_step"])
            self.transitions_since_train = int(state["transitions_since_train"])
            self.episodes_since_train = int(state["episodes_since_train"])
            self.total_transitions_added = int(state["total_transitions_added"])
            self.total_episodes_added = int(state["total_episodes_added"])
            self._warmup_ready_total_transitions = state[
                "warmup_ready_total_transitions"
            ]
            self._warmup_ready_total_episodes = state["warmup_ready_total_episodes"]
            self.pending_update_budget = int(state["pending_update_budget"])
            return

        replay_cfg = self.cfg.algorithm.get("replay_buffer", {}) or {}
        schedule_cfg = self.cfg.algorithm.get("rlt_schedule", {}) or {}
        replay_size = int(self.replay_buffer.size)
        warmup_min_size = int(
            schedule_cfg.get("warmup_min_size", replay_cfg.get("min_buffer_size", 1))
        )
        self.update_step = self._infer_rlt_update_step(
            replay_size=replay_size,
            warmup_min_size=warmup_min_size,
            warmup_updates=int(schedule_cfg.get("warmup_post_collect_updates", 0)),
            update_epoch=int(self.cfg.algorithm.get("update_epoch", 1)),
        )
        self.total_transitions_added = replay_size
        self._warmup_ready_total_transitions = min(replay_size, warmup_min_size)
        self.transitions_since_train = 0
        self.pending_update_budget = 0
        self._logger.warning(
            "Legacy RLT checkpoint has no schedule state; inferred update_step=%d "
            "from replay_size=%d under strict UTD.",
            self.update_step,
            replay_size,
        )

    @staticmethod
    def _trajectory_transition_count(traj: Trajectory) -> int:
        if traj.actions is None:
            return 0
        return int(traj.actions.shape[0] * traj.actions.shape[1])

    @staticmethod
    def _trajectory_completed_episodes(traj: Trajectory) -> int:
        dones = traj.dones
        if dones is None:
            return 0
        return int(dones.reshape(dones.shape[0], dones.shape[1], -1).any(dim=-1).sum())

    @staticmethod
    def _transition_reward_value(traj: Trajectory) -> float | None:
        rewards = traj.rewards
        if not isinstance(rewards, torch.Tensor) or rewards.numel() == 0:
            return None
        return float(rewards.detach().float().reshape(-1).sum().item())

    @staticmethod
    def _transition_done_value(traj: Trajectory) -> bool | None:
        dones = traj.dones
        if not isinstance(dones, torch.Tensor) or dones.numel() == 0:
            return None
        return bool(dones.detach().to(torch.bool).reshape(-1).any().item())

    @staticmethod
    def _row_tensor(tensor: torch.Tensor, idx: int) -> torch.Tensor:
        return tensor[idx].detach().clone().unsqueeze(0).unsqueeze(0).cpu().contiguous()

    @staticmethod
    def _step_env_tensor(
        tensor: torch.Tensor, step_idx: int, env_idx: int
    ) -> torch.Tensor:
        return (
            tensor[step_idx, env_idx]
            .detach()
            .clone()
            .unsqueeze(0)
            .unsqueeze(0)
            .cpu()
            .contiguous()
        )

    @staticmethod
    def _action_outcome_step_idx(
        *,
        action_step_idx: int,
        action_steps: int,
        outcome_steps: int,
    ) -> int:
        """Map an action row to its following reward/done outcome row.

        ``EmbodiedRolloutResult`` stores one initial environment outcome before
        the actions in every rollout epoch. Consequently an epoch containing
        ``N`` actions has ``N + 1`` reward/done rows. This helper drops each
        epoch's initial row while preserving action/outcome alignment.
        """
        extra_outcomes = outcome_steps - action_steps
        if extra_outcomes == 0:
            return action_step_idx
        if extra_outcomes < 0 or action_steps % extra_outcomes != 0:
            raise ValueError(
                "Cannot align RLT action and outcome rows: "
                f"{action_steps=}, {outcome_steps=}."
            )

        actions_per_epoch = action_steps // extra_outcomes
        epoch_idx = action_step_idx // actions_per_epoch
        return action_step_idx + epoch_idx + 1

    def _row_tensor_dict(
        self,
        tensor_dict: dict[str, object],
        idx: int,
    ) -> dict[str, torch.Tensor]:
        row_dict = {}
        for key, value in tensor_dict.items():
            if isinstance(value, torch.Tensor) and idx < value.shape[0]:
                row_dict[key] = self._row_tensor(value, idx)
        return row_dict

    def _rlt_obs_from_flat_dict(
        self,
        flat: dict,
        dict_key: str,
        idx: int,
    ) -> dict[str, torch.Tensor] | None:
        value = flat.get(dict_key)
        if not isinstance(value, dict):
            return None
        obs = self._row_tensor_dict(value, idx)
        return obs if obs else None

    @staticmethod
    def _flat_record_transition(flat: dict, idx: int) -> bool:
        forward_inputs = flat.get("forward_inputs")
        if not isinstance(forward_inputs, dict):
            return False
        record_transition = forward_inputs.get("record_transition")
        if not isinstance(record_transition, torch.Tensor):
            return False
        if idx >= record_transition.shape[0]:
            return False
        return bool(record_transition[idx].detach().to(torch.bool).reshape(-1).all())

    def _replay_forward_inputs(
        self,
        forward_inputs: dict[str, object],
        idx: int,
    ) -> dict[str, torch.Tensor]:
        """Keep only routing tags once rollout rows carry RLT episode ids.

        Stride slots and observation copies are large and stored as transition
        fields already; keeping them would multiply the replay size and break
        the batch schema that chunk rows share with stride-window rows.
        """
        if "rlt_episode_id" not in forward_inputs:
            return self._row_tensor_dict(forward_inputs, idx)
        return {
            key: self._row_tensor(forward_inputs[key], idx)
            for key in RLT_REPLAY_FORWARD_INPUT_KEYS
            if isinstance(forward_inputs.get(key), torch.Tensor)
            and idx < forward_inputs[key].shape[0]
        }

    @staticmethod
    def _replay_cell(value: torch.Tensor) -> torch.Tensor:
        return value.detach().clone().reshape(1, 1, *value.shape).cpu().contiguous()

    def _stride_window_transitions(
        self,
        trajectory: Trajectory,
        flat: dict,
        idx: int,
    ) -> list[Trajectory]:
        """Turn the stride-window slots of one rollout row into replay rows."""
        forward_inputs = flat.get("forward_inputs")
        if not isinstance(forward_inputs, dict):
            return []
        cell = self._replay_cell
        transitions = []
        for window in iter_stride_windows(forward_inputs, idx):
            transition = Trajectory(
                max_episode_length=1,
                model_weights_id=trajectory.model_weights_id,
            )
            dones = window["dones"].to(torch.bool)
            transition.actions = cell(window["actions"])
            transition.intervene_flags = cell(window["intervene_flags"].to(torch.bool))
            transition.rewards = cell(window["rewards"])
            transition.dones = cell(dones)
            transition.terminations = cell(dones)
            transition.truncations = cell(torch.zeros_like(dones))
            for field_name in ("prev_logprobs", "prev_values", "versions"):
                value = flat.get(field_name)
                if isinstance(value, torch.Tensor) and idx < value.shape[0]:
                    setattr(transition, field_name, self._row_tensor(value, idx))
            transition.curr_obs = {
                "z_rl": cell(window["curr_z_rl"]),
                "proprio": cell(window["curr_proprio"]),
                "ref_chunk": cell(window["curr_ref_chunk"]),
                "bridge_delta": cell(window["curr_bridge_delta"]),
            }
            transition.next_obs = {
                "z_rl": cell(window["next_z_rl"]),
                "proprio": cell(window["next_proprio"]),
                "ref_chunk": cell(window["next_ref_chunk"]),
                "bridge_delta": cell(window["next_bridge_delta"]),
            }
            if "curr_head_pose" in window:
                transition.curr_obs["head_pose"] = cell(window["curr_head_pose"])
                transition.next_obs["head_pose"] = cell(window["next_head_pose"])
            transition.forward_inputs = {
                "record_transition": torch.ones(1, 1, 1, dtype=torch.bool),
                "actor_switch": window["actor_switch"].reshape(1, 1, 1).to(torch.bool),
                "reference_warmup": window["reference_warmup"]
                .reshape(1, 1, 1)
                .to(torch.bool),
                "rlt_episode_id": window["episode_id"].reshape(1, 1, 1).to(torch.long),
                "action_valid_mask": cell(
                    torch.cat(
                        (torch.ones_like(dones[:1]), ~dones[:-1].cummax(dim=0).values)
                    )
                ),
            }
            transitions.append(transition)
        return transitions

    def _transition_replay_trajectories(
        self,
        trajectory: Trajectory,
    ) -> tuple[list[Trajectory], int]:
        if (
            trajectory.actions is None
            or trajectory.rewards is None
            or self.replay_buffer is None
        ):
            return [], 0

        flat = self.replay_buffer._flatten_trajectory(trajectory)
        actions = flat.get("actions")
        rewards = flat.get("rewards")
        if not isinstance(actions, torch.Tensor) or not isinstance(
            rewards, torch.Tensor
        ):
            return [], 0

        tensor_fields = (
            "actions",
            "intervene_flags",
            "rewards",
            "terminations",
            "truncations",
            "dones",
            "prev_logprobs",
            "prev_values",
            "versions",
        )
        dict_fields = ("forward_inputs",)
        replay_trajectories = []
        completed_episodes = 0
        traj_len = int(trajectory.actions.shape[0])
        bsz = int(trajectory.actions.shape[1])
        num_rows = int(actions.shape[0])
        auto_reset = bool(self.cfg.env.train.get("auto_reset", False))

        for env_idx in range(bsz):
            for t in range(traj_len):
                idx = t * bsz + env_idx
                if idx >= num_rows:
                    break
                # Stride windows carry their own record flag from the chunk
                # they start in, so they are expanded before the chunk filter.
                replay_trajectories.extend(
                    self._stride_window_transitions(trajectory, flat, idx)
                )
                if not self._flat_record_transition(flat, idx):
                    flags = flat.get("forward_inputs", {})
                    label = flags.get("rlt_keyboard_terminal")
                    episode = flags.get("rlt_episode_id")
                    if (
                        label is not None
                        and episode is not None
                        and bool(label[idx].any())
                    ):
                        outcome_idx = self._action_outcome_step_idx(
                            action_step_idx=t,
                            action_steps=traj_len,
                            outcome_steps=int(trajectory.rewards.shape[0]),
                        )
                        value = float(trajectory.rewards[outcome_idx, env_idx].sum())
                        events = getattr(self, "_rlt_keyboard_labels", [])
                        events.append((int(episode[idx].reshape(-1)[0]), value))
                        self._rlt_keyboard_labels = events
                    continue

                transition = Trajectory(
                    max_episode_length=1,
                    model_weights_id=trajectory.model_weights_id,
                )
                for field_name in tensor_fields:
                    value = flat.get(field_name)
                    if isinstance(value, torch.Tensor) and idx < value.shape[0]:
                        setattr(transition, field_name, self._row_tensor(value, idx))
                for field_name in dict_fields:
                    value = flat.get(field_name)
                    if isinstance(value, dict):
                        setattr(
                            transition,
                            field_name,
                            self._replay_forward_inputs(value, idx),
                        )

                curr_obs = self._rlt_obs_from_flat_dict(flat, "curr_obs", idx)
                if curr_obs is None:
                    raise ValueError(
                        "RLT transition replay requires curr_obs. Ensure "
                        "update_rlt_transitions() populated transition obs "
                        f"before replay ingestion, got row index {idx}."
                    )
                transition.curr_obs = curr_obs
                if "action_valid_mask" not in transition.forward_inputs:
                    steps = transition.actions.numel() // curr_obs["proprio"].numel()
                    transition.forward_inputs["action_valid_mask"] = torch.ones(
                        1, 1, steps, dtype=torch.bool
                    )

                # Rewards and terminal flags are outcomes of the action, and
                # both include the same extra initial slot per rollout epoch.
                # Read them from one shared action-aligned outcome row so a
                # terminal C/X label cannot be attached to a different chunk.
                for outcome_field in (
                    "rewards",
                    "dones",
                    "terminations",
                    "truncations",
                ):
                    outcome_value = getattr(trajectory, outcome_field, None)
                    if isinstance(outcome_value, torch.Tensor):
                        outcome_idx = self._action_outcome_step_idx(
                            action_step_idx=t,
                            action_steps=traj_len,
                            outcome_steps=int(outcome_value.shape[0]),
                        )
                        if env_idx >= outcome_value.shape[1]:
                            continue
                        setattr(
                            transition,
                            outcome_field,
                            self._step_env_tensor(outcome_value, outcome_idx, env_idx),
                        )

                is_done = (
                    isinstance(transition.dones, torch.Tensor)
                    and transition.dones.reshape(-1).to(torch.bool).any()
                )
                if is_done:
                    next_obs = curr_obs
                else:
                    next_obs = self._rlt_obs_from_flat_dict(flat, "next_obs", idx)
                if next_obs is not None:
                    transition.next_obs = next_obs
                else:
                    raise ValueError(
                        "RLT transition replay requires next_obs for non-terminal "
                        "transitions. Ensure update_rlt_transitions() populated "
                        f"transition obs before replay ingestion, got row index {idx}."
                    )

                replay_trajectories.append(transition)
                if is_done:
                    completed_episodes += 1
                    if not auto_reset:
                        break

        return replay_trajectories, completed_episodes

    def _transition_replay_metrics(
        self,
        replay_trajectories: list[Trajectory],
    ) -> dict[str, float]:
        metrics = {"replay/transition_count": float(len(replay_trajectories))}
        reward_values = [
            reward
            for traj in replay_trajectories
            if (reward := self._transition_reward_value(traj)) is not None
        ]
        if reward_values:
            metrics["replay/reward_mean"] = float(
                sum(reward_values) / len(reward_values)
            )
            metrics["replay/reward_positive_rate"] = float(
                sum(reward > 0.0 for reward in reward_values) / len(reward_values)
            )
        done_values = [
            done
            for traj in replay_trajectories
            if (done := self._transition_done_value(traj)) is not None
        ]
        if done_values:
            metrics["replay/done_rate"] = float(
                sum(bool(done) for done in done_values) / len(done_values)
            )
        return metrics

    def _ingest_rollout_trajectories(
        self,
        recv_list: list[Trajectory],
    ) -> tuple[int, int]:
        self._last_replay_metrics = {}

        if use_transition_replay(self.cfg):
            self._rlt_keyboard_labels = []
            replay_list = []
            completed = 0
            for traj in recv_list:
                assert isinstance(traj, Trajectory)
                transition_trajs, completed_count = (
                    self._transition_replay_trajectories(traj)
                )
                replay_list.extend(transition_trajs)
                completed += completed_count
            from rlinf.algorithms.rlt.transition import finalize_keyboard_transition

            for episode, reward in self._rlt_keyboard_labels:
                candidate = next(
                    (
                        row
                        for row in reversed(replay_list)
                        if "rlt_episode_id" in row.forward_inputs
                        and bool(
                            (row.forward_inputs["rlt_episode_id"] == episode).all()
                        )
                    ),
                    None,
                )
                recovered = (
                    finalize_keyboard_transition(candidate, reward)
                    if candidate is not None
                    else self.replay_buffer.finalize_rlt_episode(episode, reward)
                )
                if recovered:
                    completed += 1
                    self.log_info(
                        f"[RLT terminal] persisted episode={episode} reward={reward:g} "
                        "on its final executed transition"
                    )
            self._last_replay_metrics = {
                **self._transition_replay_metrics(replay_list),
                **collect_trajectory_replay_metrics(recv_list, reducer=all_reduce_dict),
            }
            self.replay_buffer.add_trajectories(replay_list)

            if self.demo_buffer is not None:
                intervene_traj_list = [
                    traj
                    for traj in replay_list
                    if trajectory_has_bool_tensor(traj.intervene_flags)
                ]
                if len(intervene_traj_list) > 0:
                    self.demo_buffer.add_trajectories(intervene_traj_list)

            return len(replay_list), completed

        self.replay_buffer.add_trajectories(recv_list)

        if self.demo_buffer is not None:
            intervene_traj_list = []
            for traj in recv_list:
                assert isinstance(traj, Trajectory)
                intervene_trajs = traj.extract_intervene_traj()
                if intervene_trajs is not None:
                    intervene_traj_list.extend(intervene_trajs)

            if len(intervene_traj_list) > 0:
                self.demo_buffer.add_trajectories(intervene_traj_list)

        added = sum(self._trajectory_transition_count(traj) for traj in recv_list)
        completed = sum(self._trajectory_completed_episodes(traj) for traj in recv_list)
        self._last_replay_metrics = collect_trajectory_replay_metrics(
            recv_list, reducer=all_reduce_dict
        )
        return added, completed

    def _update_rollout_ingest_counters(self, added: int, completed: int) -> None:
        if not getattr(self, "use_rlt_schedule", False):
            return
        if not hasattr(self, "transitions_since_train"):
            return
        self.transitions_since_train += added
        self.episodes_since_train += completed
        self.total_transitions_added += added
        self.total_episodes_added += completed


class RLTACFSDPPolicy(RLTACLossMixin, RLTACReplayMixin, EmbodiedSACFSDPPolicy):
    """Synchronous RLT AC worker with transition replay and warmup scheduling."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self.rlt_schedule_cfg = cfg.algorithm.get("rlt_schedule", {}) or {}
        self.use_rlt_schedule = bool(self.rlt_schedule_cfg.get("enable", False))
        self.transitions_since_train = 0
        self.episodes_since_train = 0
        self.total_transitions_added = 0
        self.total_episodes_added = 0
        self._warmup_ready_total_transitions: int | None = None
        self._warmup_ready_total_episodes: int | None = None
        self.pending_update_budget = 0

    def setup_sac_components(self):
        """Initialize replay components and let RLT schedule own readiness."""
        super().setup_sac_components()
        if self.use_rlt_schedule:
            self.buffer_dataset.min_replay_buffer_size = 1
        self._setup_split_sampling()

    @Worker.timer("actor/recv_traj")
    async def recv_rollout_trajectories(self, input_channel):
        clear_memory(sync=False)

        send_num = self._component_placement.get_world_size("env") * self.stage_num
        recv_num = self._component_placement.get_world_size("actor")
        split_num = compute_split_num(send_num, recv_num)

        recv_list = []
        for _ in range(split_num):
            trajectory: Trajectory = await input_channel.get(async_op=True).async_wait()
            recv_list.append(trajectory)

        added, completed = self._ingest_rollout_trajectories(recv_list)
        self._update_rollout_ingest_counters(added, completed)

    def _global_rlt_counters(self) -> dict[str, float]:
        summed = all_reduce_dict(
            {
                "transitions_since_train": float(self.transitions_since_train),
                "episodes_since_train": float(self.episodes_since_train),
                "total_transitions_added": float(self.total_transitions_added),
                "total_episodes_added": float(self.total_episodes_added),
            },
            op=torch.distributed.ReduceOp.SUM,
        )
        minimums = all_reduce_dict(
            {
                "min_replay_size": float(self.replay_buffer.total_samples),
                "min_demo_size": float(
                    0 if self.demo_buffer is None else self.demo_buffer.total_samples
                ),
            },
            op=torch.distributed.ReduceOp.MIN,
        )
        summed.update(minimums)
        return summed

    def _rlt_updates_to_run(self) -> tuple[int, dict[str, float]]:
        replay_cfg = self.cfg.algorithm.replay_buffer
        schedule_cfg = self.rlt_schedule_cfg
        min_buffer_size = int(
            schedule_cfg.get("warmup_min_size", replay_cfg.get("min_buffer_size", 1))
        )
        counters = self._global_rlt_counters()
        buffer_ready = counters["min_replay_size"] >= min_buffer_size
        warmup_required_updates = int(
            schedule_cfg.get("warmup_post_collect_updates", 0)
        )
        if buffer_ready and self._warmup_ready_total_transitions is None:
            self._warmup_ready_total_transitions = int(
                counters["total_transitions_added"]
            )
            self._warmup_ready_total_episodes = int(counters["total_episodes_added"])

        train_every_transitions = int(schedule_cfg.get("train_every_transitions", 0))
        train_every_episodes = int(schedule_cfg.get("train_every_episodes", 0))
        update_epoch = int(self.cfg.algorithm.get("update_epoch", 1))
        max_updates = int(schedule_cfg.get("max_updates_per_train_step", 0))

        updates_to_run = 0
        skip_reason = 0
        desired_total_updates = 0
        pending_updates = 0
        updates_scheduled = 0
        if update_epoch <= 0:
            skip_reason = 3
        elif not buffer_ready:
            skip_reason = 1
        else:
            # Algorithm 1 ties learner work to environment data through UTD G.
            # Use replay growth beyond N_warm so preloaded/recovered buffers
            # receive the same update credit as freshly collected transitions.
            online_transitions = max(
                int(counters["min_replay_size"]) - min_buffer_size,
                0,
            )
            online_episodes = max(
                int(counters["total_episodes_added"])
                - int(self._warmup_ready_total_episodes or 0),
                0,
            )
            if train_every_transitions <= 0 and train_every_episodes <= 0:
                online_cycles = online_transitions
            else:
                transition_cycles = (
                    online_transitions // train_every_transitions
                    if train_every_transitions > 0
                    else 0
                )
                episode_cycles = (
                    online_episodes // train_every_episodes
                    if train_every_episodes > 0
                    else 0
                )
                online_cycles = max(transition_cycles, episode_cycles)
            desired_total_updates = (
                warmup_required_updates + online_cycles * update_epoch
            )
            pending_updates = max(desired_total_updates - int(self.update_step), 0)
            updates_scheduled = pending_updates
            updates_to_run = pending_updates
            if max_updates > 0:
                updates_to_run = min(updates_to_run, max_updates)
            if updates_to_run <= 0:
                skip_reason = 2
        self.pending_update_budget = int(pending_updates)

        metrics = {
            "rlt/update_step": float(self.update_step),
            "rlt/ready_for_online": float(
                int(self.update_step) >= warmup_required_updates
            ),
            "rlt/warmup_required_updates": float(warmup_required_updates),
            "rlt/update_epoch": float(update_epoch),
            "rlt/max_updates_per_train_step": float(max_updates),
            "rlt/train_every_transitions": float(train_every_transitions),
            "rlt/train_every_episodes": float(train_every_episodes),
            "rlt/desired_total_updates": float(desired_total_updates),
            "rlt/pending_update_budget": float(self.pending_update_budget),
            "rlt/updates_scheduled": float(updates_scheduled),
            "rlt/updates_to_run": float(updates_to_run),
            "rlt/critic_updates_run": 0.0,
            "rlt/actor_updates_run": 0.0,
            "rlt/should_train": float(updates_to_run > 0),
            "rlt/skip_reason": float(skip_reason),
            "rlt/global_min_replay_size": float(counters["min_replay_size"]),
            "rlt/min_replay_buffer_size": float(min_buffer_size),
            "rlt/global_transitions_since_train": float(
                counters["transitions_since_train"]
            ),
            "rlt/global_total_transitions_added": float(
                counters["total_transitions_added"]
            ),
        }
        metrics.update(getattr(self, "_last_replay_metrics", {}))
        return updates_to_run, metrics

    def run_training(self):
        if not self.use_rlt_schedule:
            mean_metric_dict = super().run_training()
            replay_metrics = getattr(self, "_last_replay_metrics", {})
            if replay_metrics:
                mean_metric_dict = {**mean_metric_dict, **replay_metrics}
            return mean_metric_dict

        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        updates_to_run, schedule_metrics = self._rlt_updates_to_run()
        if updates_to_run <= 0:
            mean_metric_dict = self.process_train_metrics(schedule_metrics)
            torch.cuda.synchronize()
            torch.distributed.barrier()
            torch.cuda.empty_cache()
            return mean_metric_dict

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}
        critic_updates_run = 0
        actor_updates_run = 0
        for _ in range(updates_to_run):
            update_actor = self._should_update_actor(
                self.update_step, self.critic_actor_ratio
            )
            metrics_data = self.update_one_epoch(train_actor=update_actor)
            append_to_dict(metrics, metrics_data)
            self.update_step += 1
            critic_updates_run += 1
            actor_updates_run += int(update_actor)

        schedule_metrics["rlt/critic_updates_run"] = float(critic_updates_run)
        schedule_metrics["rlt/actor_updates_run"] = float(actor_updates_run)
        self.pending_update_budget = max(
            int(self.pending_update_budget) - critic_updates_run,
            0,
        )
        schedule_metrics["rlt/pending_update_budget"] = float(
            self.pending_update_budget
        )
        append_to_dict(metrics, schedule_metrics)
        mean_metric_dict = self.process_train_metrics(metrics)
        self.transitions_since_train = 0
        self.episodes_since_train = 0

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return mean_metric_dict


class AsyncRLTACFSDPPolicy(
    RLTACLossMixin, RLTACReplayMixin, AsyncEmbodiedSACFSDPPolicy
):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.rlt_schedule_cfg = cfg.algorithm.get("rlt_schedule", {}) or {}
        self.use_rlt_schedule = bool(self.rlt_schedule_cfg.get("enable", False))
        self.transitions_since_train = 0
        self.episodes_since_train = 0
        self.total_transitions_added = 0
        self.total_episodes_added = 0
        self._warmup_ready_total_transitions: int | None = None
        self._warmup_ready_total_episodes: int | None = None
        self.pending_update_budget = 0

    def setup_sac_components(self):
        """Initialize replay components and let the RLT schedule own readiness."""
        super().setup_sac_components()
        if self.use_rlt_schedule:
            self.buffer_dataset.min_replay_buffer_size = 1
        self._setup_split_sampling()

    # The synchronous and asynchronous workers implement the same paper RLT
    # update budget.  Only the transport and yielding behavior differ.
    _global_rlt_counters = RLTACFSDPPolicy._global_rlt_counters
    _rlt_updates_to_run = RLTACFSDPPolicy._rlt_updates_to_run

    def _drain_received_trajectories(self, max_trajectories: int | None = None):
        if getattr(self, "_recv_queue", None) is None:
            return
        recv_list = []
        processed = 0
        while True:
            try:
                recv_list.append(self._recv_queue.get_nowait())
                processed += 1
                if max_trajectories is not None and processed >= max_trajectories:
                    break
            except queue.Empty:
                break
        if not recv_list:
            return

        added, completed = self._ingest_rollout_trajectories(recv_list)
        self._update_rollout_ingest_counters(added, completed)

    @Worker.timer("run_training")
    async def run_training(self):
        """Run paper-style asynchronous RLT updates with a strict UTD budget."""
        if not self.use_rlt_schedule:
            return await super().run_training()

        if self.cfg.actor.get("enable_offload", False):
            self.load_param_and_grad(self.device)
            self.load_optimizer(self.device)

        self._drain_received_trajectories(
            max_trajectories=self.cfg.actor.get("recv_drain_max_trajectories", 1024)
        )
        updates_to_run, schedule_metrics = self._rlt_updates_to_run()
        if updates_to_run <= 0:
            # Returning an empty result makes AsyncEmbodiedRunner back off for
            # one second instead of spinning learner/checkpoint steps without
            # receiving any new robot data.
            await asyncio.sleep(0)
            return {}

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        )
        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        self.model.train()
        metrics = {}
        critic_updates_run = 0
        actor_updates_run = 0
        for _ in range(updates_to_run):
            await asyncio.sleep(0)
            update_actor = self._should_update_actor(
                self.update_step, self.critic_actor_ratio
            )
            metrics_data = self.update_one_epoch(train_actor=update_actor)
            append_to_dict(metrics, metrics_data)
            self.update_step += 1
            critic_updates_run += 1
            actor_updates_run += int(update_actor)

        schedule_metrics["rlt/critic_updates_run"] = float(critic_updates_run)
        schedule_metrics["rlt/actor_updates_run"] = float(actor_updates_run)
        self.pending_update_budget = max(
            int(self.pending_update_budget) - critic_updates_run,
            0,
        )
        schedule_metrics["rlt/pending_update_budget"] = float(
            self.pending_update_budget
        )
        schedule_metrics["rlt/global_replay_samples"] = float(
            self.replay_buffer.total_samples
        )
        append_to_dict(metrics, schedule_metrics)
        mean_metric_dict = self.process_train_metrics(metrics)
        self.transitions_since_train = 0
        self.episodes_since_train = 0

        torch.cuda.synchronize()
        torch.distributed.barrier()
        torch.cuda.empty_cache()
        return mean_metric_dict
