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

# convert hf model to middle_file
# middle_file:
#   params not in decoder: io.safetensors
#   params in decoder layer x: x.safetensors
#   use megatron name and style (fuse glu fc1)

"""
Load a JAX model and print all parameter keys, with optional conversion to PyTorch.

This script loads a JAX model checkpoint using orbax and can either:
1. Print out all the parameter keys in a hierarchical structure for inspection
2. Convert the JAX model to PyTorch format using our PI0Pytorch model

Usage:
python rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py
    --checkpoint_dir /path/to/jax/checkpoint
    --output_path /path/to/pytorch/checkpoint
    --config_name pi0_aloha_pen_uncap
"""

import hashlib
import json
import os
import pathlib
import shutil
from typing import Literal

import numpy as np
import openpi.models.gemma
import openpi.models.model
import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.training.config as _config
import orbax.checkpoint as ocp
import safetensors
import torch
import tyro
from flax.nnx import traversals
from openpi.training import utils

from rlinf.models.embodiment.openpi.native_lora import apply_native_jax_lora


def _sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_source_provenance(checkpoint_dir: str) -> dict[str, object]:
    """Build a stable fingerprint for the JAX checkpoint used for conversion."""
    root = pathlib.Path(checkpoint_dir).resolve()
    relative_paths = [
        "_CHECKPOINT_METADATA",
        "params/_METADATA",
        "params/_sharding",
        "params/manifest.ocdbt",
        "assets/total_data/norm_stats.json",
    ]
    # Orbax stores the actual tensor payload below params/d and
    # params/ocdbt.process_*/d. Fingerprinting metadata alone would not prove
    # which learned weights were converted.
    relative_paths.extend(
        path.relative_to(root).as_posix()
        for path in sorted((root / "params").glob("**/d/*"))
        if path.is_file()
    )
    artifacts = {}
    for relative_path in relative_paths:
        path = root / relative_path
        if path.is_file():
            artifacts[relative_path] = _sha256_file(path)
    if not artifacts:
        raise ValueError(
            f"Cannot fingerprint JAX checkpoint {root}: no checkpoint metadata found."
        )
    canonical = json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode()
    return {
        "source_checkpoint": str(root),
        "source_artifact_sha256": artifacts,
        "source_provenance_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def merge_lora_weights(
    state_dict: dict[str, np.ndarray], *, scaling: float = 1.0
) -> int:
    """Merge flattened OpenPI LoRA leaves into their base Gemma weights.

    OpenPI stores trained LoRA checkpoints as separate ``value``, ``lora_a``
    and ``lora_b`` leaves.  The PyTorch PI0 implementation used by RLinf has
    ordinary linear layers, so the adapter must be materialized as
    ``weight + scaling * (lora_a @ lora_b)`` before the existing JAX-to-Torch
    layout conversion runs.

    Both current OpenPI LoRA Gemma variants use ``alpha == rank``, hence a
    scaling value of 1.0.  The argument remains explicit so this helper cannot
    silently assume that for a future checkpoint.

    Args:
        state_dict: Flattened JAX parameter mapping. It is updated in place.
        scaling: OpenPI LoRA's ``alpha / rank`` scaling value.

    Returns:
        The number of base weights into which adapters were merged.
    """
    merged = 0
    lora_a_keys = [
        key
        for key in state_dict
        if key.endswith(("/lora_a/value", "_lora_a/value", "/lora_a", "_lora_a"))
    ]

    for lora_a_key in lora_a_keys:
        has_value_suffix = lora_a_key.endswith("/value")
        value_suffix = "/value" if has_value_suffix else ""
        key_without_value = lora_a_key.removesuffix(value_suffix)
        if key_without_value.endswith("/lora_a"):
            module_prefix = key_without_value.removesuffix("/lora_a")
            base_key = f"{module_prefix}/w{value_suffix}"
            lora_b_key = f"{module_prefix}/lora_b{value_suffix}"
        else:
            base_key = key_without_value.replace("_lora_a", "") + value_suffix
            lora_b_key = key_without_value.replace("_lora_a", "_lora_b") + value_suffix

        missing = [key for key in (base_key, lora_b_key) if key not in state_dict]
        if missing:
            raise KeyError(
                f"Incomplete LoRA parameters for {lora_a_key}: missing {missing}"
            )

        base = np.asarray(state_dict[base_key])
        lora_a = np.asarray(state_dict[lora_a_key])
        lora_b = np.asarray(state_dict[lora_b_key])
        expected_shape = (*lora_a.shape[:-2], lora_a.shape[-2], lora_b.shape[-1])
        if expected_shape != base.shape or lora_a.shape[:-2] != lora_b.shape[:-2]:
            raise ValueError(
                f"LoRA shape mismatch for {base_key}: base={base.shape}, "
                f"A={lora_a.shape}, B={lora_b.shape}, A@B={expected_shape}"
            )

        # Conversion restores weights as float32. Merge one leading-index slice
        # at a time: a full batched A@B for the 2B FFN is several GiB and can
        # otherwise exhaust RAM before the Torch model is even instantiated.
        merged_weight = base if base.dtype == np.float32 else base.astype(np.float32)
        for index in np.ndindex(base.shape[:-2]):
            update = np.matmul(lora_a[index], lora_b[index], dtype=np.float32)
            merged_weight[index] += float(scaling) * update
        state_dict[base_key] = merged_weight
        del state_dict[lora_a_key]
        del state_dict[lora_b_key]
        merged += 1

    dangling_lora = [key for key in state_dict if "lora_a" in key or "lora_b" in key]
    if dangling_lora:
        raise ValueError(f"Unmerged LoRA parameters remain: {dangling_lora[:5]}")

    return merged


def extract_native_lora_weights(
    state_dict: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Remove and return all JAX LoRA A/B leaves without materializing them."""
    adapters = {
        key: np.asarray(state_dict.pop(key))
        for key in list(state_dict)
        if "lora_a" in key or "lora_b" in key
    }
    if len(adapters) != 20:
        raise ValueError(
            f"Expected 20 stacked JAX LoRA A/B tensors, found {len(adapters)}"
        )
    return adapters


def convert_native_lora_state_dict(
    adapters: dict[str, np.ndarray],
) -> dict[str, torch.Tensor]:
    """Map stacked JAX Gemma adapters to exact per-layer PyTorch branches."""
    converted: dict[str, torch.Tensor] = {}

    def add_stack(*, expert: bool, rank: int) -> None:
        suffix = "_1" if expert else ""
        prefix = (
            "paligemma_with_expert.gemma_expert.model.layers"
            if expert
            else "paligemma_with_expert.paligemma.model.language_model.layers"
        )
        mlp_name = f"mlp{suffix}"

        q_a = adapters[f"llm/layers/attn/q_einsum{suffix}/lora_a"]
        q_b = adapters[f"llm/layers/attn/q_einsum{suffix}/lora_b"]
        kv_a = adapters[f"llm/layers/attn/kv_einsum{suffix}/lora_a"]
        kv_b = adapters[f"llm/layers/attn/kv_einsum{suffix}/lora_b"]
        o_a = adapters[f"llm/layers/attn/attn_vec_einsum{suffix}/lora_a"]
        o_b = adapters[f"llm/layers/attn/attn_vec_einsum{suffix}/lora_b"]
        gate_a = adapters[f"llm/layers/{mlp_name}/gating_einsum_lora_a"]
        gate_b = adapters[f"llm/layers/{mlp_name}/gating_einsum_lora_b"]
        down_a = adapters[f"llm/layers/{mlp_name}/linear_lora_a"]
        down_b = adapters[f"llm/layers/{mlp_name}/linear_lora_b"]

        if q_a.shape[0] != 18 or q_a.shape[-1] != rank:
            raise ValueError(f"Unexpected native LoRA shape: {q_a.shape}")
        for layer in range(18):
            root = f"{prefix}.{layer}"
            # Query and output adapters retain their independent head axis.
            converted[f"{root}.self_attn.q_proj.lora_a"] = torch.from_numpy(
                q_a[layer]
            )
            converted[f"{root}.self_attn.q_proj.lora_b"] = torch.from_numpy(
                q_b[layer]
            )
            converted[f"{root}.self_attn.o_proj.lora_a"] = torch.from_numpy(
                o_a[layer]
            )
            converted[f"{root}.self_attn.o_proj.lora_b"] = torch.from_numpy(
                o_b[layer]
            )
            # K/V and the three MLP projections are conventional linear LoRA.
            for index, name in enumerate(("k_proj", "v_proj")):
                converted[f"{root}.self_attn.{name}.lora_a"] = torch.from_numpy(
                    kv_a[layer, index, 0].T
                )
                converted[f"{root}.self_attn.{name}.lora_b"] = torch.from_numpy(
                    kv_b[layer, index, 0].T
                )
            for index, name in enumerate(("gate_proj", "up_proj")):
                converted[f"{root}.mlp.{name}.lora_a"] = torch.from_numpy(
                    gate_a[layer, index].T
                )
                converted[f"{root}.mlp.{name}.lora_b"] = torch.from_numpy(
                    gate_b[layer, index].T
                )
            converted[f"{root}.mlp.down_proj.lora_a"] = torch.from_numpy(
                down_a[layer].T
            )
            converted[f"{root}.mlp.down_proj.lora_b"] = torch.from_numpy(
                down_b[layer].T
            )

    add_stack(expert=False, rank=16)
    add_stack(expert=True, rank=32)
    if len(converted) != 504:
        raise ValueError(
            f"Expected 504 per-layer native LoRA parameters, got {len(converted)}"
        )
    return converted


def _get_model_config(config_name: str) -> openpi.models.pi0_config.Pi0Config:
    """Resolve a Pi0 model config, including local Cobot Magic training configs.

    RLinf's OpenPI fork does not carry every task-specific entry added to the
    local OpenPI checkout.  Conversion only needs the architecture fields, not
    the dataset transform.  Keep the wire entry here so the JAX LoRA checkpoint
    is converted with exactly the architecture used for its training.
    """
    if config_name == "pi05_wire":
        return openpi.models.pi0_config.Pi0Config(
            pi05=True,
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
        )

    model_config = _config.get_config(config_name).model
    if not isinstance(model_config, openpi.models.pi0_config.Pi0Config):
        raise ValueError(f"Config {config_name} is not a Pi0Config")
    return model_config


def slice_paligemma_state_dict(state_dict, config):
    """Convert PaliGemma JAX parameters to PyTorch format."""
    suffix = "/value" if "img/embedding/kernel/value" in state_dict else ""

    # patch embeddings
    jax_key = f"img/embedding/kernel{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose(3, 2, 0, 1)

    jax_key = f"img/embedding/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # positional embeddings
    jax_key = f"img/pos_embedding{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).reshape(
        -1, config.vision_config.hidden_size
    )

    # extract vision layers to be sliced at index 0. There are 27 layers in the base model.
    encoderblock_layernorm0_scale = state_dict.pop(
        f"img/Transformer/encoderblock/LayerNorm_0/scale{suffix}"
    )
    encoderblock_layernorm0_bias = state_dict.pop(
        f"img/Transformer/encoderblock/LayerNorm_0/bias{suffix}"
    )
    encoderblock_layernorm1_scale = state_dict.pop(
        f"img/Transformer/encoderblock/LayerNorm_1/scale{suffix}"
    )
    encoderblock_layernorm1_bias = state_dict.pop(
        f"img/Transformer/encoderblock/LayerNorm_1/bias{suffix}"
    )

    encoderblock_mlp_dense0_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/kernel{suffix}"
    )
    encoderblock_mlp_dense0_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MlpBlock_0/Dense_0/bias{suffix}"
    )
    encoderblock_mlp_dense1_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/kernel{suffix}"
    )
    encoderblock_mlp_dense1_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MlpBlock_0/Dense_1/bias{suffix}"
    )

    encoderblock_attention_0_key_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/kernel{suffix}"
    )
    encoderblock_attention_0_key_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/key/bias{suffix}"
    )
    encoderblock_attention_0_value_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/kernel{suffix}"
    )
    encoderblock_attention_0_value_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/value/bias{suffix}"
    )
    encoderblock_attention_0_query_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/kernel{suffix}"
    )
    encoderblock_attention_0_query_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/query/bias{suffix}"
    )
    encoderblock_attention_0_out_kernel = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/kernel{suffix}"
    )
    encoderblock_attention_0_out_bias = state_dict.pop(
        f"img/Transformer/encoderblock/MultiHeadDotProductAttention_0/out/bias{suffix}"
    )

    for i in range(config.vision_config.num_hidden_layers):
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.weight"
        ] = encoderblock_layernorm0_scale[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm1.bias"
        ] = encoderblock_layernorm0_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.weight"
        ] = encoderblock_layernorm1_scale[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.layer_norm2.bias"
        ] = encoderblock_layernorm1_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.weight"
        ] = encoderblock_mlp_dense0_kernel[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.bias"
        ] = encoderblock_mlp_dense0_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.weight"
        ] = encoderblock_mlp_dense1_kernel[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.mlp.fc2.bias"
        ] = encoderblock_mlp_dense1_bias[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.weight"
        ] = (
            encoderblock_attention_0_key_kernel[i]
            .reshape(-1, config.vision_config.hidden_size)
            .transpose()
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.k_proj.bias"
        ] = (
            encoderblock_attention_0_key_bias[i]
            .reshape(-1, config.vision_config.hidden_size)
            .reshape(-1)
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.weight"
        ] = (
            encoderblock_attention_0_value_kernel[i]
            .reshape(-1, config.vision_config.hidden_size)
            .transpose()
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.v_proj.bias"
        ] = (
            encoderblock_attention_0_value_bias[i]
            .reshape(-1, config.vision_config.hidden_size)
            .reshape(-1)
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.weight"
        ] = (
            encoderblock_attention_0_query_kernel[i]
            .reshape(-1, config.vision_config.hidden_size)
            .transpose()
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.q_proj.bias"
        ] = (
            encoderblock_attention_0_query_bias[i]
            .reshape(-1, config.vision_config.hidden_size)
            .reshape(-1)
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.weight"
        ] = (
            encoderblock_attention_0_out_kernel[i]
            .reshape(-1, config.vision_config.hidden_size)
            .transpose()
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers.{i}.self_attn.out_proj.bias"
        ] = (
            encoderblock_attention_0_out_bias[i]
            .reshape(-1, config.vision_config.hidden_size)
            .reshape(-1)
        )

    jax_key = f"img/Transformer/encoder_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()

    jax_key = f"img/Transformer/encoder_norm/bias{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # multimodal projector
    jax_key = f"img/head/kernel{suffix}"
    pytorch_key = (
        "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight"
    )
    state_dict[pytorch_key] = state_dict.pop(jax_key).transpose()

    jax_key = f"img/head/bias{suffix}"
    pytorch_key = (
        "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias"
    )
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # text decoder (gemma)
    jax_key = f"llm/embedder/input_embedding{suffix}"
    pytorch_key = (
        "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
    )
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    # pop the einsum attention + mlp representations
    llm_attention_attn_vec_einsum = state_dict.pop(
        f"llm/layers/attn/attn_vec_einsum/w{suffix}"
    )
    llm_attention_kv_einsum = state_dict.pop(f"llm/layers/attn/kv_einsum/w{suffix}")
    llm_attention_q_einsum = state_dict.pop(f"llm/layers/attn/q_einsum/w{suffix}")

    llm_mlp_gating_einsum = state_dict.pop(f"llm/layers/mlp/gating_einsum{suffix}")
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp/linear{suffix}")

    llm_input_layernorm = state_dict.pop(f"llm/layers/pre_attention_norm/scale{suffix}")
    llm_post_attention_layernorm = state_dict.pop(
        f"llm/layers/pre_ffw_norm/scale{suffix}"
    )

    for i in range(config.text_config.num_hidden_layers):
        q_proj_weight_reshaped = (
            llm_attention_q_einsum[i]
            .transpose(0, 2, 1)
            .reshape(
                config.text_config.num_attention_heads * config.text_config.head_dim,
                config.text_config.hidden_size,
            )
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.q_proj.weight"
        ] = q_proj_weight_reshaped

        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.k_proj.weight"
        ] = k_proj_weight_reshaped
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.v_proj.weight"
        ] = v_proj_weight_reshaped

        o_proj_weight_reshaped = (
            llm_attention_attn_vec_einsum[i]
            .transpose(2, 0, 1)
            .reshape(
                config.text_config.num_attention_heads * config.text_config.head_dim,
                config.text_config.hidden_size,
            )
        )
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.self_attn.o_proj.weight"
        ] = o_proj_weight_reshaped

        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.gate_proj.weight"
        ] = gate_proj_weight.transpose()
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.up_proj.weight"
        ] = up_proj_weight.transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.mlp.down_proj.weight"
        ] = llm_mlp_linear[i].transpose()
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.input_layernorm.weight"
        ] = llm_input_layernorm[i]
        state_dict[
            f"paligemma_with_expert.paligemma.model.language_model.layers.{i}.post_attention_layernorm.weight"
        ] = llm_post_attention_layernorm[i]

    jax_key = f"llm/final_norm/scale{suffix}"
    pytorch_key = "paligemma_with_expert.paligemma.model.language_model.norm.weight"
    state_dict[pytorch_key] = state_dict.pop(jax_key)

    expert_dict = {}
    final_state_dict = {}

    # Expert-related keys to extract (including pi05 Dense layer parameters)
    expert_keys = [
        f"llm/final_norm_1/scale{suffix}",
        f"llm/final_norm_1/Dense_0/bias{suffix}",
        f"llm/final_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/attn/attn_vec_einsum_1/w{suffix}",
        f"llm/layers/attn/kv_einsum_1/w{suffix}",
        f"llm/layers/attn/q_einsum_1/w{suffix}",
        f"llm/layers/mlp_1/gating_einsum{suffix}",
        f"llm/layers/mlp_1/linear{suffix}",
        f"llm/layers/pre_attention_norm_1/scale{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_attention_norm_1/Dense_0/kernel{suffix}",
        f"llm/layers/pre_ffw_norm_1/scale{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/bias{suffix}",
        f"llm/layers/pre_ffw_norm_1/Dense_0/kernel{suffix}",
    ]

    for key, value in state_dict.items():
        if key not in expert_keys:
            final_state_dict[key] = torch.from_numpy(value)
        else:
            expert_dict[key] = value

    return final_state_dict, expert_dict


def slice_gemma_state_dict(state_dict, config, *, num_expert, checkpoint_dir, pi05):
    """Convert Gemma JAX parameters to PyTorch format."""
    # Add missing attributes to config if they don't exist
    if not hasattr(config, "vocab_size"):
        config.vocab_size = 257152  # PALIGEMMA_VOCAB_SIZE
    if not hasattr(config, "hidden_size"):
        config.hidden_size = config.width
    if not hasattr(config, "num_hidden_layers"):
        config.num_hidden_layers = config.depth
    if not hasattr(config, "num_attention_heads"):
        config.num_attention_heads = config.num_heads

    suffix = (
        "/value"
        if f"llm/layers/attn/attn_vec_einsum_{num_expert}/w/value" in state_dict
        else ""
    )

    llm_attention_attn_vec_einsum = state_dict.pop(
        f"llm/layers/attn/attn_vec_einsum_{num_expert}/w{suffix}"
    )
    llm_attention_kv_einsum = state_dict.pop(
        f"llm/layers/attn/kv_einsum_{num_expert}/w{suffix}"
    )
    llm_attention_q_einsum = state_dict.pop(
        f"llm/layers/attn/q_einsum_{num_expert}/w{suffix}"
    )

    llm_mlp_gating_einsum = state_dict.pop(
        f"llm/layers/mlp_{num_expert}/gating_einsum{suffix}"
    )
    llm_mlp_linear = state_dict.pop(f"llm/layers/mlp_{num_expert}/linear{suffix}")

    # Check if we have Dense layers (for pi05/adaptive normalization) or scale layers (for regular pi0)
    if "pi05" in checkpoint_dir:
        # Pi05 with adaptive normalization
        llm_input_layernorm_bias = state_dict.pop(
            f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/bias{suffix}"
        )
        llm_post_attention_layernorm_bias = state_dict.pop(
            f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/bias{suffix}"
        )
        llm_input_layernorm_kernel = state_dict.pop(
            f"llm/layers/pre_attention_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
        llm_post_attention_layernorm_kernel = state_dict.pop(
            f"llm/layers/pre_ffw_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
    else:
        # Regular pi0 with standard RMSNorm
        llm_input_layernorm = state_dict.pop(
            f"llm/layers/pre_attention_norm_{num_expert}/scale{suffix}"
        )
        llm_post_attention_layernorm = state_dict.pop(
            f"llm/layers/pre_ffw_norm_{num_expert}/scale{suffix}"
        )

    for i in range(config.num_hidden_layers):
        q_proj_weight_reshaped = (
            llm_attention_q_einsum[i]
            .transpose(0, 2, 1)
            .reshape(config.num_attention_heads * config.head_dim, config.hidden_size)
        )
        state_dict[
            f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.q_proj.weight"
        ] = q_proj_weight_reshaped

        k_proj_weight_reshaped = llm_attention_kv_einsum[i, 0, 0].transpose()
        state_dict[
            f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.k_proj.weight"
        ] = k_proj_weight_reshaped
        v_proj_weight_reshaped = llm_attention_kv_einsum[i, 1, 0].transpose()
        state_dict[
            f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.v_proj.weight"
        ] = v_proj_weight_reshaped

        o_proj_weight_reshaped = (
            llm_attention_attn_vec_einsum[i]
            .reshape(config.num_attention_heads * config.head_dim, config.hidden_size)
            .transpose(1, 0)
        )
        state_dict[
            f"paligemma_with_expert.gemma_expert.model.layers.{i}.self_attn.o_proj.weight"
        ] = o_proj_weight_reshaped

        gate_proj_weight = llm_mlp_gating_einsum[i, 0]
        state_dict[
            f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.gate_proj.weight"
        ] = gate_proj_weight.transpose()
        up_proj_weight = llm_mlp_gating_einsum[i, 1]
        state_dict[
            f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.up_proj.weight"
        ] = up_proj_weight.transpose()
        state_dict[
            f"paligemma_with_expert.gemma_expert.model.layers.{i}.mlp.down_proj.weight"
        ] = llm_mlp_linear[i].transpose()

        if "pi05" in checkpoint_dir:
            # Pi05 with adaptive normalization - use Dense layer parameters directly
            state_dict[
                f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.bias"
            ] = llm_input_layernorm_bias[i]
            state_dict[
                f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.bias"
            ] = llm_post_attention_layernorm_bias[i]
            state_dict[
                f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.dense.weight"
            ] = llm_input_layernorm_kernel[i].transpose()
            state_dict[
                f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.dense.weight"
            ] = llm_post_attention_layernorm_kernel[i].transpose()
        else:
            # Regular pi0 with standard RMSNorm
            state_dict[
                f"paligemma_with_expert.gemma_expert.model.layers.{i}.input_layernorm.weight"
            ] = llm_input_layernorm[i]
            state_dict[
                f"paligemma_with_expert.gemma_expert.model.layers.{i}.post_attention_layernorm.weight"
            ] = llm_post_attention_layernorm[i]

    # Handle final norm layer
    if "pi05" in checkpoint_dir:
        # Pi05 with adaptive normalization - use Dense layer parameters directly
        final_norm_bias = state_dict.pop(
            f"llm/final_norm_{num_expert}/Dense_0/bias{suffix}"
        )
        final_norm_kernel = state_dict.pop(
            f"llm/final_norm_{num_expert}/Dense_0/kernel{suffix}"
        )
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.bias"] = (
            final_norm_bias
        )
        state_dict["paligemma_with_expert.gemma_expert.model.norm.dense.weight"] = (
            final_norm_kernel.transpose()
        )
    else:
        # Regular pi0 with standard RMSNorm
        state_dict["paligemma_with_expert.gemma_expert.model.norm.weight"] = (
            state_dict.pop(f"llm/final_norm_{num_expert}/scale{suffix}")
        )

        # state_dict["paligemma_with_expert.gemma_expert.lm_head.weight"] = embedding_vector # weights are tied.

    final_state_dict = {}
    for key, value in state_dict.items():
        if not isinstance(value, torch.Tensor):
            final_state_dict[key] = torch.from_numpy(value)
        else:
            final_state_dict[key] = value

    return final_state_dict


def slice_initial_orbax_checkpoint(
    checkpoint_dir: str, restore_precision: str | None = None
):
    """Load and process params by restoring via JAX model loader first.
    This respects dtype conversions that occur during model restore.
    """
    # Use repository restore utility to load a pure dict of params (value suffix removed)
    params = openpi.models.model.restore_params(
        f"{checkpoint_dir}/params/", restore_type=np.ndarray, dtype=restore_precision
    )

    return {
        "paligemma_params": traversals.flatten_mapping(params["PaliGemma"], sep="/"),
        "projection_params": params,
    }


def load_jax_model_and_print_keys(checkpoint_dir: str):
    """
    Load JAX model from checkpoint and print all parameter keys.

    Args:
        checkpoint_dir: Path to the checkpoint directory
    """
    checkpoint_dir = (
        os.path.abspath(checkpoint_dir)
        if not checkpoint_dir.startswith("gs://")
        else checkpoint_dir
    )
    # Initialize checkpointer
    checkpointer = ocp.PyTreeCheckpointer()
    metadata = checkpointer.metadata(f"{checkpoint_dir}/params")
    print(utils.array_tree_to_info(metadata))


def convert_pi0_checkpoint(
    checkpoint_dir: str,
    precision: str,
    output_path: str,
    model_config: openpi.models.pi0_config.Pi0Config,
    *,
    lora_mode: Literal["merge", "native"] = "merge",
):
    """
    Convert PI0 JAX checkpoint to PyTorch format.

    Args:
        checkpoint_dir: Path to the JAX checkpoint
        precision: Model precision (float32, bfloat16, float16)
        output_path: Path to save the converted PyTorch model
        model_config: Model config
    """
    print(f"Converting PI0 checkpoint from {checkpoint_dir} to {output_path}")
    print(f"Model config: {model_config}")

    # Break down orbax ckpts by restoring via JAX to respect dtype
    initial_params = slice_initial_orbax_checkpoint(
        checkpoint_dir=checkpoint_dir, restore_precision="float32"
    )

    paligemma_config = openpi.models.gemma.get_config(model_config.paligemma_variant)
    action_expert_config = openpi.models.gemma.get_config(
        model_config.action_expert_variant
    )
    lora_scalings = {
        config.scaling_value
        for gemma_config in (paligemma_config, action_expert_config)
        for config in gemma_config.lora_configs.values()
    }
    if len(lora_scalings) > 1:
        raise ValueError(
            f"Converter cannot merge mixed LoRA scaling values: {lora_scalings}"
        )
    lora_scaling = next(iter(lora_scalings), 1.0)
    native_lora_params: dict[str, torch.Tensor] = {}
    if lora_mode == "native":
        native_adapters = extract_native_lora_weights(
            initial_params["paligemma_params"]
        )
        native_lora_params = convert_native_lora_state_dict(native_adapters)
        merged_lora_count = 0
        print(
            "Preserving 252 JAX LoRA projection branches as 504 native "
            "PyTorch A/B parameters."
        )
    else:
        merged_lora_count = merge_lora_weights(
            initial_params["paligemma_params"], scaling=lora_scaling
        )
        if lora_scalings and merged_lora_count == 0:
            raise ValueError(
                "The model config enables LoRA, but the checkpoint contains no "
                "LoRA parameters."
            )
        print(
            f"Merged {merged_lora_count} OpenPI LoRA tensors into Gemma base "
            f"weights (scaling={lora_scaling})."
        )

    # Process projection params
    if model_config.pi05:
        keys = [
            "action_in_proj",
            "action_out_proj",
            "time_mlp_in",
            "time_mlp_out",
        ]
    else:
        keys = [
            "state_proj",
            "action_in_proj",
            "action_out_proj",
            "action_time_mlp_in",
            "action_time_mlp_out",
        ]

    projection_params = {}
    for key in keys:
        kernel_params = initial_params["projection_params"][key]["kernel"]
        bias_params = initial_params["projection_params"][key]["bias"]
        if isinstance(kernel_params, dict):
            weight = kernel_params["value"]
            bias = bias_params["value"]
        else:
            weight = kernel_params
            bias = bias_params

        pytorch_weight_key = f"{key}.weight"
        pytorch_bias_key = f"{key}.bias"

        projection_params[pytorch_weight_key] = torch.from_numpy(np.array(weight)).T
        projection_params[pytorch_bias_key] = torch.from_numpy(np.array(bias))

    # Create configs based on checkpoint path
    # All models use the same PaliGemma config structure
    class PaliGemmaConfig:
        def __init__(self):
            self.vision_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 1152,
                    "num_hidden_layers": 27,
                    "num_attention_heads": 16,
                    "intermediate_size": 4304,
                    "patch_size": 14,
                    "projection_dim": 2048,
                },
            )()
            self.text_config = type(
                "obj",
                (object,),
                {
                    "hidden_size": 2048,
                    "num_hidden_layers": 18,
                    "num_attention_heads": 8,
                    "head_dim": 256,
                    "intermediate_size": 16384,
                },
            )()

    paligemma_config = PaliGemmaConfig()
    # Process PaliGemma weights
    paligemma_params, expert_params = slice_paligemma_state_dict(
        initial_params["paligemma_params"], paligemma_config
    )

    # Process Gemma weights from expert_params
    gemma_params = slice_gemma_state_dict(
        expert_params,
        action_expert_config,
        num_expert=1,
        checkpoint_dir=checkpoint_dir,
        pi05=model_config.pi05,
    )

    # Instantiate model
    pi0_model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_config)
    if lora_mode == "native":
        injected = apply_native_jax_lora(pi0_model)
        if injected != 252:
            raise ValueError(f"Expected 252 injected LoRA projections, got {injected}")

    # Combine all parameters (no prefix needed for our model structure)
    all_params = {
        **paligemma_params,
        **gemma_params,
        **projection_params,
        **native_lora_params,
    }

    # The action expert consumes input embeddings supplied by PI0 and never
    # calls its causal LM head. It is the only parameter absent from the JAX
    # architecture. Refuse every other incomplete or unexpected conversion.
    incompatible = pi0_model.load_state_dict(all_params, strict=False)
    allowed_missing = {
        "paligemma_with_expert.gemma_expert.lm_head.weight",
        # Transformers ties this output head to the PaliGemma input embedding;
        # the JAX tree stores the shared tensor under the embedding name only.
        "paligemma_with_expert.paligemma.lm_head.weight",
    }
    unexpected_missing = set(incompatible.missing_keys) - allowed_missing
    if unexpected_missing or incompatible.unexpected_keys:
        raise ValueError(
            "Incomplete JAX-to-PyTorch conversion: "
            f"missing={sorted(unexpected_missing)}, "
            f"unexpected={sorted(incompatible.unexpected_keys)}"
        )
    with torch.no_grad():
        # Make the unused parameter deterministic instead of persisting the
        # random Transformers initialization in every converted checkpoint.
        pi0_model.paligemma_with_expert.gemma_expert.lm_head.weight.zero_()

    if precision == "float32":
        pi0_model = pi0_model.to(torch.float32)
    elif precision == "bfloat16":
        pi0_model = pi0_model.to(torch.bfloat16)
    else:
        raise ValueError(f"Invalid precision: {precision}")

    # Save the converted model using safetensors
    os.makedirs(output_path, exist_ok=True)

    # Save model weights as SafeTensors using save_model to handle tied weights
    safetensors.torch.save_model(
        pi0_model, os.path.join(output_path, "model.safetensors")
    )
    converted_model_sha256 = _sha256_file(
        pathlib.Path(output_path) / "model.safetensors"
    )

    # Fine-tuning checkpoints save assets inside the numbered checkpoint,
    # whereas base releases commonly keep them one directory above it.
    # Prefer the checkpoint-local copy so its norm_stats exactly match the
    # converted weights.
    checkpoint_path = pathlib.Path(checkpoint_dir)
    assets_source = checkpoint_path / "assets"
    if not assets_source.exists():
        assets_source = checkpoint_path.parent / "assets"
    if assets_source.exists():
        assets_dest = pathlib.Path(output_path) / "assets"
        if assets_dest.exists():
            shutil.rmtree(assets_dest)
        # Checkpoints can contain dataset symlinks (including self-referential
        # links). Preserve links rather than following them recursively.
        shutil.copytree(assets_source, assets_dest, symlinks=True)

    # Save config as JSON for reference
    config_dict = {
        "action_dim": model_config.action_dim,
        "action_horizon": model_config.action_horizon,
        "paligemma_variant": model_config.paligemma_variant,
        "action_expert_variant": model_config.action_expert_variant,
        "precision": precision,
        "merged_lora_tensors": merged_lora_count,
        "lora_format": "native_jax" if lora_mode == "native" else "merged",
        "native_lora_projection_count": 252 if lora_mode == "native" else 0,
        "native_lora_parameter_count": 504 if lora_mode == "native" else 0,
        "lora_scaling": lora_scaling,
        "converted_model_sha256": converted_model_sha256,
        **build_source_provenance(checkpoint_dir),
    }
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    print("Model conversion completed successfully!")
    print(f"Model saved to {output_path}")


def main(
    checkpoint_dir: str,
    config_name: str,
    output_path: str | None = None,
    precision: Literal["float32", "bfloat16", "float16"] = "bfloat16",
    *,
    inspect_only: bool = False,
    lora_mode: Literal["merge", "native"] = "merge",
):
    """Load JAX model and optionally convert to PyTorch.

    Args:
        checkpoint_dir: Path to the JAX checkpoint directory
        output_path: Path to save converted PyTorch model (required for conversion)
        precision: Precision for model conversion
        inspect_only: Only inspect parameter keys, don't convert
    """
    model_config = _get_model_config(config_name)
    if inspect_only:
        load_jax_model_and_print_keys(checkpoint_dir)
    else:
        if not output_path:
            print(
                "Error: --output_path is required for conversion. Use --inspect_only to only view keys."
            )
            return
        convert_pi0_checkpoint(
            checkpoint_dir,
            precision,
            output_path,
            model_config,
            lora_mode=lora_mode,
        )


if __name__ == "__main__":
    tyro.cli(main)
