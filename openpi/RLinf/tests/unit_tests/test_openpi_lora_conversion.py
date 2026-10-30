import numpy as np
import pytest

from rlinf.utils.ckpt_convertor.convert_openpi_jax_to_python import (
    build_source_provenance,
    merge_lora_weights,
)


def test_merge_lora_weights_handles_attention_and_feed_forward_leaves():
    attention_base = np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5)
    attention_a = np.full((2, 3, 4, 2), 0.25, dtype=np.float32)
    attention_b = np.full((2, 3, 2, 5), 0.5, dtype=np.float32)
    ffn_base = np.zeros((2, 4, 6), dtype=np.float32)
    ffn_a = np.arange(2 * 4 * 3, dtype=np.float32).reshape(2, 4, 3)
    ffn_b = np.ones((2, 3, 6), dtype=np.float32)

    params = {
        "llm/layers/attn/q_einsum/w/value": attention_base.copy(),
        "llm/layers/attn/q_einsum/lora_a/value": attention_a,
        "llm/layers/attn/q_einsum/lora_b/value": attention_b,
        "llm/layers/mlp/gating_einsum/value": ffn_base.copy(),
        "llm/layers/mlp/gating_einsum_lora_a/value": ffn_a,
        "llm/layers/mlp/gating_einsum_lora_b/value": ffn_b,
    }

    assert merge_lora_weights(params, scaling=0.5) == 2
    np.testing.assert_allclose(
        params["llm/layers/attn/q_einsum/w/value"],
        attention_base + 0.5 * np.matmul(attention_a, attention_b),
    )
    np.testing.assert_allclose(
        params["llm/layers/mlp/gating_einsum/value"],
        ffn_base + 0.5 * np.matmul(ffn_a, ffn_b),
    )
    assert not any("lora_" in key for key in params)


def test_merge_lora_weights_rejects_incomplete_adapter():
    params = {
        "llm/layers/attn/q_einsum/w/value": np.zeros((2, 3), dtype=np.float32),
        "llm/layers/attn/q_einsum/lora_a/value": np.zeros((2, 1), dtype=np.float32),
    }

    with pytest.raises(KeyError, match="Incomplete LoRA parameters"):
        merge_lora_weights(params)


def test_merge_lora_weights_handles_restore_params_without_value_suffix():
    params = {
        "llm/layers/attn/q_einsum/w": np.ones((2, 3), dtype=np.float32),
        "llm/layers/attn/q_einsum/lora_a": np.full((2, 1), 2.0, dtype=np.float32),
        "llm/layers/attn/q_einsum/lora_b": np.full((1, 3), 3.0, dtype=np.float32),
        "llm/layers/mlp/linear": np.ones((2, 3), dtype=np.float32),
        "llm/layers/mlp/linear_lora_a": np.full((2, 1), 4.0, dtype=np.float32),
        "llm/layers/mlp/linear_lora_b": np.full((1, 3), 5.0, dtype=np.float32),
    }

    assert merge_lora_weights(params) == 2
    np.testing.assert_allclose(params["llm/layers/attn/q_einsum/w"], 7.0)
    np.testing.assert_allclose(params["llm/layers/mlp/linear"], 21.0)


def test_merge_lora_weights_rejects_shape_mismatch():
    params = {
        "llm/layers/mlp/linear/value": np.zeros((2, 4), dtype=np.float32),
        "llm/layers/mlp/linear_lora_a/value": np.zeros((2, 3), dtype=np.float32),
        "llm/layers/mlp/linear_lora_b/value": np.zeros((3, 5), dtype=np.float32),
    }

    with pytest.raises(ValueError, match="LoRA shape mismatch"):
        merge_lora_weights(params)


def test_source_provenance_changes_with_checkpoint_metadata(tmp_path):
    (tmp_path / "params").mkdir()
    metadata = tmp_path / "params" / "_METADATA"
    metadata.write_bytes(b"checkpoint-a")
    first = build_source_provenance(str(tmp_path))
    metadata.write_bytes(b"checkpoint-b")
    second = build_source_provenance(str(tmp_path))

    assert first["source_checkpoint"] == str(tmp_path.resolve())
    assert first["source_provenance_sha256"] != second["source_provenance_sha256"]
