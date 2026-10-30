import json

import pytest

from rlinf.models.embodiment.openpi import validate_piper_converted_checkpoint


def _native_metadata():
    return {
        "lora_format": "native_jax",
        "merged_lora_tensors": 0,
        "native_lora_projection_count": 252,
        "native_lora_parameter_count": 504,
        "lora_scaling": 1.0,
        "converted_model_sha256": "d" * 64,
        "source_provenance_sha256": "a" * 64,
        "source_artifact_sha256": {
            "params/_METADATA": "b" * 64,
            "params/ocdbt.process_0/d/weights": "c" * 64,
        },
    }


def test_piper_checkpoint_rejects_merged_lora_metadata(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"merged_lora_tensors": 10, "lora_scaling": 1.0})
    )

    with pytest.raises(ValueError, match="expected 252 native JAX LoRA"):
        validate_piper_converted_checkpoint(str(tmp_path))


def test_piper_checkpoint_accepts_expected_lora_metadata(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(_native_metadata())
    )

    validate_piper_converted_checkpoint(str(tmp_path))


def test_piper_checkpoint_rejects_missing_source_provenance(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "lora_format": "native_jax",
                "merged_lora_tensors": 0,
                "native_lora_projection_count": 252,
                "native_lora_parameter_count": 504,
                "lora_scaling": 1.0,
            }
        )
    )

    with pytest.raises(
        ValueError, match="source checkpoint weight provenance is missing"
    ):
        validate_piper_converted_checkpoint(str(tmp_path))


def test_piper_checkpoint_rejects_model_hash_mismatch(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"not-the-converted-model")
    (tmp_path / "config.json").write_text(
        json.dumps(_native_metadata())
    )

    with pytest.raises(ValueError, match="model.safetensors SHA-256 mismatch"):
        validate_piper_converted_checkpoint(
            str(tmp_path), verify_model_hash=True
        )
