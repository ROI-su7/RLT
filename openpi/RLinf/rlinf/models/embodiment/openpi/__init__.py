# Copyright 2025 The RLinf Authors.
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
# openpi model configs

import hashlib
import json
import os
import pathlib

import torch
from omegaconf import DictConfig


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_piper_converted_checkpoint(
    checkpoint_dir: str, *, verify_model_hash: bool = False
) -> None:
    """Reject Piper checkpoints that do not preserve the native JAX LoRA."""
    metadata_path = pathlib.Path(checkpoint_dir) / "config.json"
    if not metadata_path.is_file():
        raise ValueError(
            f"Piper converted checkpoint has no conversion metadata: {metadata_path}"
        )
    metadata = json.loads(metadata_path.read_text())
    if (
        metadata.get("lora_format") != "native_jax"
        or metadata.get("merged_lora_tensors") != 0
        or metadata.get("native_lora_projection_count") != 252
        or metadata.get("native_lora_parameter_count") != 504
        or metadata.get("lora_scaling") != 1.0
    ):
        raise ValueError(
            "Unsafe pi05_wire conversion: expected 252 native JAX LoRA "
            "projection branches (504 A/B parameters), no merged LoRA tensors, "
            f"and scaling 1.0; got {metadata}."
        )
    provenance = metadata.get("source_provenance_sha256")
    source_artifacts = metadata.get("source_artifact_sha256")
    if (
        not isinstance(provenance, str)
        or len(provenance) != 64
        or not isinstance(source_artifacts, dict)
        or not source_artifacts
        or not any("/d/" in path for path in source_artifacts)
    ):
        raise ValueError(
            "Unsafe pi05_wire conversion: source checkpoint weight provenance is "
            "missing. Re-run the fixed converter before Stage 1."
        )
    model_sha256 = metadata.get("converted_model_sha256")
    if not isinstance(model_sha256, str) or len(model_sha256) != 64:
        raise ValueError(
            "Unsafe pi05_wire conversion: converted model SHA-256 is missing."
        )
    if verify_model_hash:
        model_path = pathlib.Path(checkpoint_dir) / "model.safetensors"
        actual_sha256 = _sha256_file(model_path)
        if actual_sha256 != model_sha256:
            raise ValueError(
                "Unsafe pi05_wire conversion: model.safetensors SHA-256 mismatch: "
                f"{actual_sha256} != {model_sha256}"
            )


def get_model(cfg: DictConfig, torch_dtype=None):
    import glob

    import openpi.shared.download as download
    import openpi.transforms as transforms
    import safetensors
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.native_lora import (
        apply_native_jax_lora,
        native_lora_parameter_names,
    )
    from rlinf.models.embodiment.openpi.openpi_action_model import (
        OpenPi0Config,
        OpenPi0ForRLActionPrediction,
    )

    # config
    config_name = getattr(cfg.openpi, "config_name", None)
    data_kwargs = getattr(cfg, "openpi_data", None)
    actor_train_config = get_openpi_config(
        config_name, model_path=cfg.model_path, data_kwargs=data_kwargs
    )

    actor_model_config = actor_train_config.model
    actor_model_config = OpenPi0Config(**actor_model_config.__dict__)
    override_model_config_kwargs = cfg.openpi
    if override_model_config_kwargs is not None:
        for key, val in override_model_config_kwargs.items():
            actor_model_config.__dict__[key] = val

    # load model
    checkpoint_dir = download.maybe_download(str(cfg.model_path))

    # Check if this is a checkpoint directory (saved by FSDP)
    # Check for model_state_dict/full_weights.pt (direct checkpoint) or actor/model_state_dict/full_weights.pt (from runner)
    full_weights_path = os.path.join(
        checkpoint_dir, "model_state_dict", "full_weights.pt"
    )
    actor_full_weights_path = os.path.join(
        checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"
    )
    metadata_path = pathlib.Path(checkpoint_dir) / "config.json"
    source_metadata = (
        json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    )
    use_native_jax_lora = bool(
        getattr(actor_model_config, "native_jax_lora", False)
        or source_metadata.get("lora_format") == "native_jax"
    )

    model: OpenPi0ForRLActionPrediction = OpenPi0ForRLActionPrediction(
        actor_model_config
    )
    if use_native_jax_lora:
        injected = apply_native_jax_lora(model)
        if injected != 252:
            raise ValueError(f"Expected 252 native LoRA projections, got {injected}")
    # train expert only
    if actor_model_config.train_expert_only:
        model.freeze_vlm()

    # Load weights from checkpoint if it's a checkpoint directory, otherwise load from safetensors
    if os.path.exists(full_weights_path):
        # Direct checkpoint directory
        model_state_dict = torch.load(full_weights_path, map_location="cpu")
        incompatible = model.load_state_dict(model_state_dict, strict=False)
    elif os.path.exists(actor_full_weights_path):
        # Checkpoint directory from runner
        model_state_dict = torch.load(actor_full_weights_path, map_location="cpu")
        incompatible = model.load_state_dict(model_state_dict, strict=False)
    else:
        # Original model directory with safetensors files
        if config_name == "pi05_piper_joint":
            validate_piper_converted_checkpoint(checkpoint_dir)
        weight_paths = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
        if not weight_paths:
            weight_paths = [os.path.join(checkpoint_dir, "model.safetensors")]
        all_state_dict = {}
        for weight_path in weight_paths:
            state_dict = safetensors.torch.load_file(weight_path, device="cpu")
            all_state_dict.update(state_dict)
        incompatible = model.load_state_dict(all_state_dict, strict=False)

    if use_native_jax_lora:
        missing_native = native_lora_parameter_names(model).intersection(
            incompatible.missing_keys
        )
        if missing_native:
            raise ValueError(
                "Native JAX LoRA checkpoint is incomplete; missing "
                f"{sorted(missing_native)[:5]}"
            )

    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    if getattr(actor_model_config, "freeze_vla_for_rlt", False):
        if not actor_model_config.use_rlt:
            raise ValueError("freeze_vla_for_rlt requires openpi.use_rlt=True")
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.rlt_module.parameters():
            parameter.requires_grad = True
        trainable_names = [
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        ]
        if not trainable_names or any(
            not name.startswith("rlt_module.") for name in trainable_names
        ):
            raise ValueError(
                "Frozen-VLA Stage 1 must expose only rlt_module parameters"
            )
    # fsdp replace
    # model.paligemma_with_expert.replace_gemma_decoder_layers()
    # load data stats
    data_config = actor_train_config.data.create(
        actor_train_config.assets_dirs, actor_model_config
    )
    norm_stats_path = (
        data_kwargs.get("norm_stats_path") if data_kwargs is not None else None
    )
    if norm_stats_path is not None:
        norm_stats = data_config.norm_stats
        if norm_stats is None:
            norm_dir = pathlib.Path(norm_stats_path).expanduser()
            if norm_dir.is_file():
                norm_dir = norm_dir.parent
            norm_stats = _checkpoints.load_norm_stats(norm_dir.parent, norm_dir.name)
    else:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)
    # wrappers
    repack_transforms = transforms.Group()
    default_prompt = None
    model.setup_wrappers(
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
    )

    return model
