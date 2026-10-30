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

import numpy as np
import pytest
import torch

from rlinf.models.embodiment.modules.rlt_token_checkpoint import (
    encoder_state_fingerprint,
    load_stage1_rlt_encoder,
)
from rlinf.models.embodiment.native_jax_rlt import NativeJaxRLTOffsetFeatureClient

METADATA = {
    "rlt_native_jax_prefix": True,
    "rlt_prefix_only": True,
    "rlt_prefix_seq_len": 768,
    "rlt_encoder_fingerprint": "abc",
}


class _FakeServer:
    def __init__(self, metadata):
        self.metadata = metadata
        self.requests = []

    def get_server_metadata(self):
        return self.metadata

    def infer(self, request):
        batch = request["rlt_prefix_batch"]
        self.requests.append(batch)
        states = np.stack([item["state"] for item in batch])
        return {
            "rlt_z_rl": np.repeat(states[:, :1], 6, axis=1),
            "rlt_prefix_valid": np.ones(len(batch), dtype=bool),
            "policy_timing": {"infer_ms": 1.0},
        }


def _obs(value: float) -> dict:
    return {
        "states": torch.full((2, 14), value),
        "main_images": torch.zeros(2, 224, 224, 3, dtype=torch.uint8),
        "extra_view_images": torch.zeros(2, 2, 224, 224, 3, dtype=torch.uint8),
        "task_descriptions": ["insert the wire", "insert the wire"],
    }


def _client(metadata=None, expected="abc"):
    server = _FakeServer(dict(METADATA if metadata is None else metadata))
    client = NativeJaxRLTOffsetFeatureClient(
        host="unused",
        port=0,
        prefix_seq_len=768,
        action_dim=14,
        default_prompt="insert the wire",
        expected_encoder_fingerprint=expected,
        client=server,
    )
    return client, server


def test_one_request_per_env_carries_all_its_offsets() -> None:
    client, server = _client()

    features = client.extract_offset_features([_obs(1.0), _obs(2.0)])

    assert features["z_rl"].shape == (2, 2, 6)
    assert features["proprio"].shape == (2, 2, 14)
    assert len(server.requests) == 2
    assert len(server.requests[0]) == 2
    assert server.requests[0][0]["images"]["cam_high"].shape == (3, 224, 224)
    assert features["z_rl"][0, 1, 0].item() == 2.0


def test_mismatched_encoder_is_rejected() -> None:
    client, _ = _client(expected="other")

    with pytest.raises(RuntimeError, match="does not match"):
        client.extract_offset_features([_obs(1.0)])


def test_server_without_prefix_only_serving_is_rejected() -> None:
    client, _ = _client(metadata={**METADATA, "rlt_prefix_only": False})

    with pytest.raises(RuntimeError, match="rlt-token-checkpoint"):
        client.extract_offset_features([_obs(1.0)])


def test_rollout_rejects_a_server_encoding_with_another_token(monkeypatch) -> None:
    from rlinf.models.embodiment import native_jax_rlt

    model = native_jax_rlt.NativeJaxRLTFeatureModel.__new__(
        native_jax_rlt.NativeJaxRLTFeatureModel
    )
    model.host, model.port, model.prefix_seq_len = "unused", 0, 768
    model.encoder_fingerprint = "rollout-token"
    model._client = None
    monkeypatch.setattr(
        native_jax_rlt.websocket_client_policy,
        "WebsocketClientPolicy",
        lambda host, port: _FakeServer(dict(METADATA)),
    )

    with pytest.raises(RuntimeError, match="different Stage-1 token"):
        model._connect()


def test_legacy_stage1_tokens_are_rejected(tmp_path) -> None:
    path = tmp_path / "rlt_token.pt"
    torch.save(
        {"format_version": 1, "prefix_seq_len": 768, "image_only": True, "rlt_module": {}},
        path,
    )

    with pytest.raises(ValueError, match="legacy"):
        load_stage1_rlt_encoder(str(path), prefix_seq_len=768)


def test_encoder_fingerprint_depends_only_on_weights() -> None:
    state = {"a": torch.ones(2), "b": torch.zeros(3)}

    assert encoder_state_fingerprint(state) == encoder_state_fingerprint(
        dict(reversed(list(state.items())))
    )
    assert encoder_state_fingerprint(state) != encoder_state_fingerprint(
        {"a": torch.full((2,), 2.0), "b": torch.zeros(3)}
    )
