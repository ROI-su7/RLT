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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy
from rlinf.models.embodiment.mlp_policy.rlt_action_space import RLTDeltaActionSpace
from rlinf.models.embodiment.modules.rlt_projected_heads import (
    RLTProjectedActor,
    RLTProjectedTwinQ,
)
from rlinf.utils.arm_layout import gripper_slice

ACTOR_ARCHITECTURES = ("mlp", "projected")
ACTOR_OUTPUTS = ("full", "residual")


class RLTMLPPolicy(MLPPolicy):
    """MLP actor-critic policy for RLT Stage 2 heads.

    Actor input follows RLT: reference action chunk, RL token feature, and
    proprioceptive state. Its B-compatible head directly predicts a full
    normalized action chunk. Critic input follows RLT: action chunk, RL token
    feature, and proprioceptive state.
    """

    def __init__(
        self,
        z_dim: int,
        proprio_dim: int,
        action_dim: int,
        num_action_chunks: int,
        ref_num_action_chunks: int | None = None,
        add_q_head: bool = True,
        q_head_type: str = "default",
        fixed_std: float = 0.002,
        fixed_gripper_std: float = 0.0002,
        fixed_std_is_normalized: bool = False,
        execute_actor_mean: bool = False,
        actor_hidden_dim: int = 256,
        actor_num_layers: int = 2,
        action_norm_stats=None,
        actor_architecture: str = "mlp",
        critic_hidden_dim: int = 256,
        critic_num_layers: int = 2,
        actor_output: str = "full",
        bridge_delta_dim: int = 0,
        bridge_delta_scale: float = 0.05,
        head_pose_dim: int = 0,
        head_pose_scale=None,
    ):
        if not add_q_head:
            raise ValueError(
                "RLTMLPPolicy requires add_q_head=True for actor-critic training."
            )
        z_dim = int(z_dim)
        proprio_dim = int(proprio_dim)
        # Execution-side RTC runs the queued VLA suffix between the observation
        # and the chunk, so the state also carries where that bridge leaves the
        # arm: obs["bridge_delta"] = last bridge command - proprio (rad / m).
        bridge_delta_dim = int(bridge_delta_dim)
        bridge_delta_scale = float(bridge_delta_scale)
        if bridge_delta_dim < 0:
            raise ValueError("bridge_delta_dim must be non-negative.")
        if bridge_delta_dim and bridge_delta_scale <= 0:
            raise ValueError("bridge_delta_scale must be positive.")
        # Piper connector-head state obs["head_pose"] = [dx px, dy px,
        # dangle rad, visible], divided by head_pose_scale and clipped.
        head_pose_dim = int(head_pose_dim)
        if head_pose_dim < 0:
            raise ValueError("head_pose_dim must be non-negative.")
        if head_pose_dim:
            scale = torch.as_tensor(
                [1.0] * head_pose_dim if head_pose_scale is None else head_pose_scale,
                dtype=torch.float32,
            ).reshape(-1)
            if scale.numel() != head_pose_dim or bool((scale <= 0).any()):
                raise ValueError(
                    f"head_pose_scale needs {head_pose_dim} positive values, got "
                    f"{scale.tolist()}."
                )
        # proprio branch width of actor and critic: proprio plus bridge delta
        # plus head pose.
        state_dim = proprio_dim + bridge_delta_dim + head_pose_dim
        step_action_dim = int(action_dim)
        chunk_len = int(num_action_chunks)
        ref_chunk_len = (
            chunk_len if ref_num_action_chunks is None else int(ref_num_action_chunks)
        )
        if ref_chunk_len < chunk_len:
            raise ValueError(
                "ref_num_action_chunks must be >= num_action_chunks, got "
                f"{ref_chunk_len} < {chunk_len}."
            )
        flat_action_dim = chunk_len * step_action_dim

        actor_obs_dim = z_dim + state_dim + flat_action_dim
        critic_obs_dim = z_dim + state_dim

        projected_critic = q_head_type == "rlt_projected"
        super().__init__(
            obs_dim=actor_obs_dim,
            action_dim=flat_action_dim,
            num_action_chunks=1,
            add_value_head=False,
            add_q_head=add_q_head,
            q_head_type="default" if projected_critic else q_head_type,
            critic_obs_dim=critic_obs_dim,
        )
        if projected_critic:
            # openpi-RLT critic: z, proprio and action are projected separately
            # so the 2048-D RL token cannot drown out the action correction.
            self.q_head = RLTProjectedTwinQ(
                z_dim,
                state_dim,
                flat_action_dim,
                hidden_dim=int(critic_hidden_dim),
                num_layers=int(critic_num_layers),
            )
        self.z_dim = z_dim
        self.proprio_dim = proprio_dim
        self.bridge_delta_dim = bridge_delta_dim
        self.bridge_delta_scale = bridge_delta_scale
        self.head_pose_dim = head_pose_dim
        if head_pose_dim:
            self.register_buffer("head_pose_scale", scale, persistent=False)
        self.step_action_dim = step_action_dim
        self.chunk_len = chunk_len
        self.ref_chunk_len = ref_chunk_len
        self.flat_action_dim = flat_action_dim
        self.actor_hidden_dim = int(actor_hidden_dim)
        self.actor_num_layers = int(actor_num_layers)
        if self.actor_hidden_dim <= 0:
            raise ValueError("actor_hidden_dim must be positive.")
        if self.actor_num_layers < 1:
            raise ValueError("actor_num_layers must be at least one.")

        self.actor_architecture = str(actor_architecture)
        if self.actor_architecture not in ACTOR_ARCHITECTURES:
            raise ValueError(
                f"actor_architecture must be one of {ACTOR_ARCHITECTURES}, "
                f"got {self.actor_architecture!r}."
            )
        if self.actor_architecture == "projected":
            # openpi-RLT actor: projected z / proprio / reference features and a
            # LayerNorm-GELU trunk with ``actor_num_layers`` hidden layers.
            self.actor_net = RLTProjectedActor(
                z_dim,
                state_dim,
                flat_action_dim,
                hidden_dim=self.actor_hidden_dim,
                num_layers=self.actor_num_layers,
            )
            self.backbone = nn.Identity()
            self.actor_mean = nn.Identity()
        else:
            # Match B's actor exactly: [Linear, ReLU] * (L - 1) + Linear, using
            # PyTorch's default random initialization.
            layers: list[nn.Module] = []
            in_dim = actor_obs_dim
            for _ in range(self.actor_num_layers - 1):
                layers.extend((nn.Linear(in_dim, self.actor_hidden_dim), nn.ReLU()))
                in_dim = self.actor_hidden_dim
            self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
            self.actor_mean = nn.Linear(in_dim, flat_action_dim)
        self.actor_output = str(actor_output)
        if self.actor_output not in ACTOR_OUTPUTS:
            raise ValueError(
                f"actor_output must be one of {ACTOR_OUTPUTS}, got {self.actor_output!r}."
            )
        if self.actor_output == "residual":
            # Residual chunk: mean = reference + delta with a zero-initialized
            # delta head, so an uninformed Actor reproduces the VLA exactly.
            # 2026-10-14: the full-output Actor's reproduction error on states
            # where JAX alone was right (0.0029 rad mean, p90 0.0065) was the
            # same size as the operator corrections it should learn.
            last = (
                self.actor_net.trunk.net[-1]
                if self.actor_architecture == "projected"
                else self.actor_mean
            )
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        self.actor_logstd = None
        self.fixed_std = float(fixed_std)
        if self.fixed_std <= 0:
            raise ValueError(f"fixed_std must be positive, got {self.fixed_std}.")
        self.fixed_gripper_std = float(fixed_gripper_std)
        if self.fixed_gripper_std <= 0:
            raise ValueError(
                f"fixed_gripper_std must be positive, got {self.fixed_gripper_std}."
            )
        self.fixed_std_is_normalized = bool(fixed_std_is_normalized)
        # Keep stochastic SAC updates, but allow deterministic robot execution.
        self.execute_actor_mean = bool(execute_actor_mean)

        # Actor and critic operate in the normalized delta action space; the
        # route, replay and robot keep absolute joint targets.
        self.action_space = RLTDeltaActionSpace.from_config(
            action_norm_stats, action_dim=step_action_dim
        )

        if self.fixed_std_is_normalized:
            # B contract: one fixed Gaussian sigma in normalized action space.
            action_std = torch.full((self.flat_action_dim,), self.fixed_std)
        else:
            # Backward-compatible physical-unit sigma for other RLT configs.
            per_step_std = torch.full((self.step_action_dim,), self.fixed_std)
            per_step_std[gripper_slice(per_step_std.shape[-1])] = self.fixed_gripper_std
            action_std = self.action_space.scale_sigma(per_step_std).repeat(
                self.chunk_len
            )
        self.register_buffer("action_std", action_std)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Keep the configured exploration scale when resuming old checkpoints.

        ``action_std`` is derived from the current config, but older Stage-2
        checkpoints stored it as a persistent buffer.  Drop that historical
        value so changing ``fixed_std`` (for example 0.05 -> 0.005) takes
        effect for both the live and target policy on resume.
        """
        state_dict.pop(prefix + "action_std", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        # The buffer is intentionally reconstructed from the current config,
        # so its absence from an old checkpoint is not a loading error.
        key = prefix + "action_std"
        if key in missing_keys:
            missing_keys.remove(key)

    def preprocess_env_obs(self, env_obs):
        device = next(self.parameters()).device
        processed = {}
        for key, value in env_obs.items():
            processed[key] = value.to(device) if torch.is_tensor(value) else value
        return processed

    @staticmethod
    def _flatten_batch(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() <= 2:
            return tensor
        return tensor.reshape(tensor.shape[0], -1)

    def _get_z(self, obs: dict) -> torch.Tensor:
        return self._flatten_batch(obs["z_rl"])

    def _get_proprio(self, obs: dict) -> torch.Tensor:
        return self._flatten_batch(obs["proprio"])

    def _get_state(self, obs: dict) -> torch.Tensor:
        """Network state: proprio, plus the scaled bridge delta and head pose.

        The action space keeps normalizing against the raw proprio. A missing
        ``bridge_delta`` (first chunk, operator control) means the chunk starts
        from the measured state, i.e. a zero delta. A missing ``head_pose``
        means "not visible" (all zeros).
        """
        proprio = self._get_proprio(obs)
        parts = [proprio]
        if self.bridge_delta_dim:
            parts.append(self._get_bridge_delta(obs, proprio))
        if self.head_pose_dim:
            parts.append(self._get_head_pose(obs, proprio))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    def _get_head_pose(self, obs: dict, proprio: torch.Tensor) -> torch.Tensor:
        head_pose = obs.get("head_pose")
        if head_pose is None:
            return proprio.new_zeros(proprio.shape[0], self.head_pose_dim)
        head_pose = self._flatten_batch(head_pose).to(proprio)
        if head_pose.shape[-1] != self.head_pose_dim:
            raise ValueError(
                f"Expected a {self.head_pose_dim}-D head_pose, got "
                f"{tuple(head_pose.shape)}."
            )
        return (head_pose / self.head_pose_scale.to(proprio)).clamp(-5.0, 5.0)

    def _get_bridge_delta(self, obs: dict, proprio: torch.Tensor) -> torch.Tensor:
        delta = obs.get("bridge_delta")
        if delta is None:
            delta = proprio.new_zeros(proprio.shape[0], self.bridge_delta_dim)
        else:
            delta = self._flatten_batch(delta).to(proprio)
        if delta.shape[-1] != self.bridge_delta_dim:
            raise ValueError(
                f"Expected a {self.bridge_delta_dim}-D bridge_delta, got "
                f"{tuple(delta.shape)}."
            )
        return delta / self.bridge_delta_scale

    def _get_ref_chunk(self, obs: dict) -> torch.Tensor:
        ref_chunk = self._flatten_batch(obs["ref_chunk"]).reshape(
            obs["ref_chunk"].shape[0], -1, self.step_action_dim
        )
        ref_chunk = ref_chunk[:, : self.chunk_len]
        return ref_chunk.reshape(ref_chunk.shape[0], -1)

    def _maybe_drop_reference(
        self,
        ref_chunk: torch.Tensor,
        reference_dropout_prob: float,
    ) -> torch.Tensor:
        if reference_dropout_prob <= 0:
            return ref_chunk
        keep_prob = 1.0 - float(reference_dropout_prob)
        keep_mask = (
            torch.rand((ref_chunk.shape[0], 1), device=ref_chunk.device) < keep_prob
        )
        return ref_chunk * keep_mask.to(dtype=ref_chunk.dtype)

    def _actor_state(self, ref_chunk: torch.Tensor, obs: dict) -> torch.Tensor:
        return torch.cat([ref_chunk, self._get_z(obs), self._get_state(obs)], dim=-1)

    def _critic_state(self, obs: dict) -> torch.Tensor:
        return torch.cat([self._get_z(obs), self._get_state(obs)], dim=-1)

    def _critic_actions(self, obs: dict, actions: torch.Tensor) -> torch.Tensor:
        return self.action_space.normalize(
            self._flatten_batch(actions), self._get_proprio(obs)
        )

    def _format_chunk_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return actions.reshape(-1, self.chunk_len, self.step_action_dim)

    def _action_distribution(
        self,
        obs: dict,
        *,
        apply_reference_dropout: bool,
        reference_dropout_prob: float,
    ) -> Normal:
        """Paper Eq.4: pi(a | x, a_ref) = N(mu(x, a_ref), sigma^2 I), in normalized units."""
        reference = self.action_space.normalize(
            self._get_ref_chunk(obs), self._get_proprio(obs)
        )
        base = reference
        if apply_reference_dropout:
            reference = self._maybe_drop_reference(reference, reference_dropout_prob)
        # B contract: reference is an input, not an additive output baseline.
        # The randomly initialized head directly predicts the full normalized
        # action chunk and BC supplies the reference/human anchor.
        if self.actor_architecture == "projected":
            mean = self.actor_net(self._get_z(obs), self._get_state(obs), reference)
        else:
            mean = self.actor_mean(self.backbone(self._actor_state(reference, obs)))
        if self.actor_output == "residual":
            # The base is never dropped: dropout only hides it from the input.
            mean = base + mean
        return Normal(mean, self.action_std.expand_as(mean))

    def sac_forward(
        self,
        obs,
        apply_reference_dropout: bool = False,
        reference_dropout_prob: float = 0.0,
        deterministic: bool = False,
        **kwargs,
    ):
        probs = self._action_distribution(
            obs,
            apply_reference_dropout=apply_reference_dropout,
            reference_dropout_prob=reference_dropout_prob,
        )
        normalized = probs.mean if deterministic else probs.rsample()
        action = self.action_space.denormalize(normalized, self._get_proprio(obs))
        return action, probs.log_prob(normalized), None

    def sac_q_forward(self, obs, actions, shared_feature=None, detach_encoder=False):
        del shared_feature
        critic_state = self._critic_state(obs)
        if detach_encoder:
            critic_state = critic_state.detach()
        return self.q_head(critic_state, self._critic_actions(obs, actions))

    def crossq_q_forward(
        self,
        obs,
        actions,
        next_obs=None,
        next_actions=None,
        shared_feature=None,
        detach_encoder=False,
    ):
        del shared_feature
        if next_actions is not None and next_obs is None:
            raise ValueError(
                "crossq_q_forward needs next_obs to normalize next_actions."
            )
        critic_state = self._critic_state(obs)
        next_critic_state = (
            self._critic_state(next_obs) if next_obs is not None else None
        )
        if detach_encoder:
            critic_state = critic_state.detach()
            if next_critic_state is not None:
                next_critic_state = next_critic_state.detach()
        return self.q_head(
            critic_state,
            self._critic_actions(obs, actions),
            next_state_features=next_critic_state,
            next_action_features=(
                self._critic_actions(next_obs, next_actions)
                if next_actions is not None
                else None
            ),
        )

    def crossq_forward(self, obs, **kwargs):
        return self.sac_forward(obs, **kwargs)

    def sft_forward(self, data, **kwargs):
        obs = data["obs"] if "obs" in data else data
        target_actions = self._flatten_batch(
            data["action"] if "action" in data else data["actions"]
        )
        pred_actions = self._action_distribution(
            obs,
            apply_reference_dropout=False,
            reference_dropout_prob=0.0,
        ).mean
        if pred_actions.numel() != target_actions.numel():
            raise ValueError(
                "RLT SFT targets must match the predicted action shape, "
                f"got predicted {pred_actions.shape} and target "
                f"{target_actions.shape}."
            )
        target = self.action_space.normalize(
            target_actions.reshape_as(pred_actions), self._get_proprio(obs)
        )
        return F.mse_loss(pred_actions, target, reduction="none")

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,
        return_obs=True,
        mode="train",
        **kwargs,
    ):
        del calculate_logprobs, calculate_values
        obs = self.preprocess_env_obs(env_obs=env_obs)
        action, chunk_logprobs, _ = self.sac_forward(
            obs, deterministic=(mode == "eval" or self.execute_actor_mean)
        )
        chunk_actions = self._format_chunk_actions(action)

        forward_inputs = {"action": action, "model_action": action}
        if return_obs:
            forward_inputs.update(obs)

        result = {
            "prev_logprobs": chunk_logprobs,
            "prev_values": torch.zeros_like(chunk_logprobs[..., :1]),
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result
