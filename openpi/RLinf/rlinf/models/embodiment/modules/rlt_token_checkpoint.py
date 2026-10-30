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

"""Stage-1 RL-token checkpoint loading shared by RLinf and the OpenPI server.

The OpenPI websocket server loads this file by path, outside the RLinf package
(importing ``rlinf.models`` pulls in Ray, which the serving environment does
not have), so nothing here may import RLinf by name.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import torch

if __package__:
    from .rlt_token_transformer import RLTTokenEncoder
else:
    _spec = importlib.util.spec_from_file_location(
        "rlt_token_transformer", Path(__file__).with_name("rlt_token_transformer.py")
    )
    _module = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_module)
    RLTTokenEncoder = _module.RLTTokenEncoder

# Version 2 tokens are trained with the autoregressive decoder and fp32 master
# weights. Earlier tokens used pure-bf16 AdamW, which rounds most updates away,
# so their encoders must not be served.
RLT_TOKEN_FORMAT_VERSION = 2


def _read_encoder_state(path: str, *, prefix_seq_len: int) -> dict[str, torch.Tensor]:
    # mmap: the file also holds the decoder and optimizer state, and the
    # robot-side EnvWorker only hashes the encoder.
    checkpoint = torch.load(
        str(path), map_location="cpu", weights_only=False, mmap=True
    )
    format_version = int(checkpoint.get("format_version", 1))
    if format_version < RLT_TOKEN_FORMAT_VERSION:
        raise ValueError(
            f"{path} is a legacy Stage-1 RL token (format_version={format_version}). "
            "Retrain it with toolkits/realworld_check/train_piper_rlt_token_native_jax.py."
        )
    if int(checkpoint.get("prefix_seq_len", -1)) != int(prefix_seq_len):
        raise ValueError(
            "Stage-1 RLT prefix length does not match Stage 2: "
            f"{checkpoint.get('prefix_seq_len')} != {prefix_seq_len}"
        )
    if not bool(checkpoint.get("image_only", False)):
        raise ValueError("Piper Stage-1 RLT checkpoint must be image_only=True.")
    return {
        name.removeprefix("encoder."): value
        for name, value in checkpoint["rlt_module"].items()
        if name.startswith("encoder.")
    }


def encoder_state_fingerprint(encoder_state: dict[str, torch.Tensor]) -> str:
    """Hash the encoder weights so separate processes can prove they match."""
    digest = hashlib.blake2b(digest_size=8)
    for name in sorted(encoder_state):
        tensor = encoder_state[name].detach().to(dtype=torch.float32).cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def rlt_encoder_fingerprint(path: str, *, prefix_seq_len: int) -> str:
    return encoder_state_fingerprint(
        _read_encoder_state(path, prefix_seq_len=prefix_seq_len)
    )


def load_stage1_rlt_encoder(
    path: str, *, prefix_seq_len: int
) -> tuple[RLTTokenEncoder, str]:
    """Load the frozen Stage-1 encoder and a fingerprint of its weights.

    The rollout worker and the robot-side JAX server both load the encoder, so
    the fingerprint is what proves they produce z_rl with the same weights.
    """
    encoder_state = _read_encoder_state(path, prefix_seq_len=prefix_seq_len)
    encoder = RLTTokenEncoder(prefix_seq_len=int(prefix_seq_len))
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.requires_grad_(False)
    return encoder.eval(), encoder_state_fingerprint(encoder_state)
