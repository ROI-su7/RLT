"""Piper-only RECAP CFG loading, preserving source LoRA and normalization.

The upstream cfg_model factory dispatches here only for pi05_piper_joint.
No RLT model, token, rollout, or ROS implementation is modified.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def policy_provenance(cfg) -> dict:
    """Identify the base policy and normalization required for this adapter."""
    base = json.loads((Path(cfg.model_path) / "config.json").read_text())
    norm = Path(cfg.openpi_data.norm_stats_path)
    return {
        "scheme": "piper_recap",
        "source_model_sha256": base["converted_model_sha256"],
        "norm_sha256": hashlib.sha256(norm.read_bytes()).hexdigest(),
        "task": str(cfg.openpi_data.default_prompt),
        "lora_only": bool(cfg.get("recap_lora_only", True)),
        "positive_only_conditional": bool(cfg.openpi.positive_only_conditional),
        "policy_mode": str(cfg.get("piper_policy_mode", "recap_cfg")),
    }


def validate_policy_provenance(checkpoint: Path, cfg) -> None:
    """Reject an unrelated checkpoint or mismatched base/task/stats at serving."""
    recorded = json.loads((checkpoint / "piper_recap_policy.json").read_text())
    if recorded != policy_provenance(cfg):
        raise ValueError(
            "RECAP checkpoint base/task/norm/adapter settings differ from the serving configuration."
        )


def configure_trainable_parameters(model, *, lora_only: bool) -> list[str]:
    """Train the action expert, optionally its native LoRA branches only."""
    model.requires_grad_(False)
    selected = []
    for name, parameter in model.named_parameters():
        if "gemma_expert" in name and (
            not lora_only or name.endswith((".lora_a", ".lora_b"))
        ):
            parameter.requires_grad_(True)
            selected.append(name)
        elif not lora_only and name.startswith(
            ("action_in_proj.", "action_out_proj.", "time_mlp_in.", "time_mlp_out.")
        ):
            parameter.requires_grad_(True)
            selected.append(name)
    if not selected:
        raise ValueError("No action-expert parameters selected for RECAP training.")
    return selected


def load_piper_cfg_model(cfg):
    """Load the 50-step three-camera policy with strictly checked base weights."""
    import dataclasses

    from openpi import transforms
    from safetensors.torch import load_model

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.native_lora import apply_native_jax_lora
    from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import (
        OpenPi0Config,
        OpenPi0ForCFGActionPrediction,
    )

    checkpoint = Path(cfg.model_path).expanduser().resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Local converted policy not found: {checkpoint}")
    metadata_file = checkpoint / "config.json"
    if not metadata_file.is_file():
        raise ValueError("RECAP base policy requires converter config.json provenance.")
    metadata = json.loads(metadata_file.read_text())
    if metadata.get("action_horizon") != 50 or metadata.get("action_dim") != 32:
        raise ValueError(
            "Expected the source Piper pi05 architecture: horizon=50, padded action_dim=32."
        )
    data_kwargs = cfg.get("openpi_data", None)
    train_config = get_openpi_config(
        "pi05_piper_joint", model_path=str(checkpoint), data_kwargs=data_kwargs
    )
    config = OpenPi0Config(**dataclasses.asdict(train_config.model))
    for key, value in cfg.openpi.items():
        object.__setattr__(config, key, value)
    if config.action_horizon != 50 or config.action_dim != 32:
        raise ValueError(
            "Do not change the source policy's 50-step/32-D attention layout."
        )
    model = OpenPi0ForCFGActionPrediction(config)
    source_native = metadata.get("lora_format") == "native_jax"
    lora_only = bool(cfg.get("recap_lora_only", True))
    if source_native or lora_only:
        if apply_native_jax_lora(model) != 252:
            raise ValueError("Unexpected native LoRA topology.")
        if not source_native:
            # A=B=0 would have zero gradients in both matrices. Random A and
            # zero B preserve the full checkpoint exactly and permit learning.
            for name, parameter in model.named_parameters():
                if name.endswith(".lora_a"):
                    torch.nn.init.normal_(parameter, std=0.01)
    weights = checkpoint / "model.safetensors"
    if not weights.is_file():
        raise FileNotFoundError(f"Converted base weights missing: {weights}")
    missing, unexpected = load_model(model, str(weights), strict=False)
    allowed_missing = {
        name
        for name in missing
        if not source_native and name.endswith((".lora_a", ".lora_b"))
    }
    if set(missing) - allowed_missing or unexpected:
        raise ValueError(
            f"Base policy mismatch: missing={sorted(set(missing) - allowed_missing)[:5]}, unexpected={sorted(unexpected)[:5]}"
        )
    # Native base weights and newly attached adapters are now established.
    # Serving may additionally overlay a *complete* RECAP FSDP checkpoint.
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    names = configure_trainable_parameters(model, lora_only=lora_only)
    logger.info(
        "Piper RECAP trainable tensors=%d, parameters=%d (LoRA-only=%s)",
        len(names),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
        lora_only,
    )
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if data_config.norm_stats is None:
        raise ValueError(
            "Set actor.model.openpi_data.norm_stats_path to source norm_stats.json."
        )
    model.setup_wrappers(
        transforms=[
            *data_config.data_transforms.inputs,
            transforms.Normalize(data_config.norm_stats, use_quantiles=True),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(data_config.norm_stats, use_quantiles=True),
            *data_config.data_transforms.outputs,
        ],
    )
    return model


class PiperCFGPolicy:
    """OpenPI websocket adapter: physical 14-D observations and action targets."""

    def __init__(self, model, *, task: str, device: str = "cuda:0"):
        from openpi import transforms

        from rlinf.data.datasets.recap.cfg_model import TokenizePromptWithGuidance

        self.model = model.to(device).eval()
        self.device, self.task = device, task
        steps = []
        for transform in model._input_transform.transforms:
            if type(transform).__name__ == "TokenizePrompt":
                transform = TokenizePromptWithGuidance(
                    transform.tokenizer,
                    discrete_state_input=transform.discrete_state_input,
                )
            steps.append(transform)
        self.input_transform = transforms.compose(steps)

    @torch.inference_mode()
    def infer(self, obs: dict) -> dict:
        """Return a physical absolute [50,14] plan compatible with piper_real."""
        from rlinf.models.embodiment.openpi_cfg.openpi_cfg_action_model import (
            Observation,
        )

        state = np.asarray(obs["state"], dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError(
                "Websocket observation.state must be finite physical 14-D."
            )
        images = obs["images"]
        # Use the same camera ordering and ALOHA adaptation as source training.
        raw = {
            "observation/state": state,
            "observation/image": images["cam_high"],
            "observation/extra_view_image-0": images["cam_left_wrist"],
            "observation/extra_view_image-1": images["cam_right_wrist"],
            "prompt": self.task,
        }
        inputs = self.input_transform(raw)

        def batch(value):
            if isinstance(value, dict):
                return {key: batch(item) for key, item in value.items()}
            return torch.as_tensor(np.asarray(value).copy(), device=self.device)[None]

        output = self.model.sample_actions(Observation.from_dict(batch(inputs)))
        actions = output["actions"][0].float().cpu().numpy()
        result = self.model._output_transform(
            {"actions": actions, "state": inputs["state"]}
        )
        actions = np.asarray(result["actions"], np.float32)
        if actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise ValueError(f"Invalid RECAP action plan: {actions.shape}")
        return {"actions": actions}

    def reset(self) -> None:
        """No recurrent state is retained between websocket requests."""
