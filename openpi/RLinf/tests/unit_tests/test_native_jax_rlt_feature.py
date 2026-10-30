from __future__ import annotations

import numpy as np
import torch

from rlinf.models.embodiment.native_jax_rlt import NativeJaxRLTFeatureModel


class _MeanEncoder(torch.nn.Module):
    def forward(self, prefix, mask):
        del mask
        return prefix.mean(dim=1, keepdim=True)


class _FakeClient:
    def __init__(self):
        self.requests = []

    def infer(self, request):
        self.requests.append(request)
        return {
            "actions": np.arange(50 * 14, dtype=np.float32).reshape(50, 14),
            "rlt_prefix": np.ones((768, 2048), dtype=np.float16),
            "rlt_prefix_mask": np.ones((768,), dtype=np.bool_),
        }


def _initialize_timing_state(model: NativeJaxRLTFeatureModel) -> None:
    model._logger = __import__("logging").getLogger(__name__)
    model._timing_count = 0
    model._timing_jax_wall_ms = 0.0
    model._timing_jax_model_ms = 0.0
    model._timing_encoder_ms = 0.0


def test_native_jax_feature_uses_one_response_for_ref_and_prefix() -> None:
    model = NativeJaxRLTFeatureModel.__new__(NativeJaxRLTFeatureModel)
    torch.nn.Module.__init__(model)
    model.host = "robot"
    model.port = 6006
    model.prefix_seq_len = 768
    model.ref_num_action_chunks = 5
    model.action_dim = 14
    model.default_prompt = "insert the wire"
    _initialize_timing_state(model)
    model.encoder = _MeanEncoder()
    model.register_buffer("_device_anchor", torch.empty(0), persistent=False)
    client = _FakeClient()
    model._client = client

    obs = {
        "states": torch.zeros((1, 14)),
        "main_images": torch.zeros((1, 224, 224, 3), dtype=torch.uint8),
        "extra_view_images": torch.zeros((1, 2, 224, 224, 3), dtype=torch.uint8),
        "task_descriptions": ["insert the wire"],
    }
    result = model.extract_rlt_obs(obs)

    assert len(client.requests) == 1
    assert set(client.requests[0]["images"]) == {
        "cam_high",
        "cam_left_wrist",
        "cam_right_wrist",
    }
    for image in client.requests[0]["images"].values():
        assert image.shape == (3, 224, 224)
        assert image.dtype == np.uint8
    assert result["z_rl"].shape == (1, 2048)
    assert result["proprio"].shape == (1, 14)
    assert result["ref_chunk"].shape == (1, 5, 14)
    np.testing.assert_array_equal(
        result["ref_chunk"].cpu().numpy(),
        np.arange(50 * 14, dtype=np.float32).reshape(50, 14)[:5][None],
    )


def test_native_jax_feature_preserves_chw_camera_layout() -> None:
    model = NativeJaxRLTFeatureModel.__new__(NativeJaxRLTFeatureModel)
    torch.nn.Module.__init__(model)
    model.host = "robot"
    model.port = 6006
    model.prefix_seq_len = 768
    model.ref_num_action_chunks = 5
    model.action_dim = 14
    model.default_prompt = "insert the wire"
    _initialize_timing_state(model)
    model.encoder = _MeanEncoder()
    model.register_buffer("_device_anchor", torch.empty(0), persistent=False)
    client = _FakeClient()
    model._client = client

    obs = {
        "states": torch.zeros((1, 14)),
        "main_images": torch.zeros((1, 3, 224, 224), dtype=torch.uint8),
        "extra_view_images": torch.zeros((1, 2, 3, 224, 224), dtype=torch.uint8),
    }
    model.extract_rlt_obs(obs)

    for image in client.requests[0]["images"].values():
        assert image.shape == (3, 224, 224)
        assert image.flags.c_contiguous


def test_native_jax_feature_reuses_exact_rollout_boundary_observation() -> None:
    model = NativeJaxRLTFeatureModel.__new__(NativeJaxRLTFeatureModel)
    torch.nn.Module.__init__(model)
    model.host = "robot"
    model.port = 6006
    model.prefix_seq_len = 768
    model.ref_num_action_chunks = 5
    model.action_dim = 14
    model.default_prompt = "insert the wire"
    _initialize_timing_state(model)
    model.encoder = _MeanEncoder()
    model.register_buffer("_device_anchor", torch.empty(0), persistent=False)
    client = _FakeClient()
    model._client = client

    obs = {
        "states": torch.zeros((1, 14)),
        "main_images": torch.zeros((1, 224, 224, 3), dtype=torch.uint8),
        "extra_view_images": torch.zeros((1, 2, 224, 224, 3), dtype=torch.uint8),
    }
    first = model.extract_rlt_obs(obs)
    repeated = model.extract_rlt_obs(
        {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in obs.items()
        }
    )

    assert len(client.requests) == 1
    # The cached features are reused; only bridge_delta is rebuilt per call.
    assert repeated["z_rl"] is first["z_rl"]
    assert repeated["ref_chunk"] is first["ref_chunk"]
    torch.testing.assert_close(repeated["bridge_delta"], torch.zeros(1, 14))

    changed_obs = dict(obs)
    changed_obs["states"] = torch.ones((1, 14))
    model.extract_rlt_obs(changed_obs)
    assert len(client.requests) == 2


def test_native_jax_compact_response_avoids_gpu_node_prefix_encoding():
    model = NativeJaxRLTFeatureModel.__new__(NativeJaxRLTFeatureModel)
    torch.nn.Module.__init__(model)
    model.prefix_seq_len = 768
    model.ref_num_action_chunks = 10
    model.native_action_horizon = 50
    model.action_dim = 14
    model.default_prompt = "task"
    model.encoder_fingerprint = "same-token"
    model.require_compact_response = True
    _initialize_timing_state(model)
    model.register_buffer("_device_anchor", torch.empty(0), persistent=False)

    class Client:
        def infer(self, request):
            return {
                "actions": np.zeros((50, 14), np.float32),
                "rlt_z_rl": np.ones(2048, np.float32),
                "rlt_prefix_valid": np.ones(1, bool),
                "rlt_encoder_fingerprint": "same-token",
            }

    model._client = Client()
    result = model.extract_rlt_obs(
        {
            "states": torch.zeros(1, 14),
            "main_images": torch.zeros(1, 224, 224, 3, dtype=torch.uint8),
            "extra_view_images": torch.zeros(1, 2, 224, 224, 3, dtype=torch.uint8),
        }
    )
    assert result["z_rl"].shape == (1, 2048)
    assert result["z_rl"].sum().item() == 2048
    assert result["ref_chunk"].shape == (1, 50, 14)
