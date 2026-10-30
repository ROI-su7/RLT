"""Native-JAX OpenPI feature service used by real-robot RLT Stage 2."""

from __future__ import annotations

import hashlib
import time
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig
from openpi_client import websocket_client_policy

from rlinf.models.embodiment.modules.rlt_token_checkpoint import (
    RLT_TOKEN_FORMAT_VERSION,
    load_stage1_rlt_encoder,
    rlt_encoder_fingerprint,
)
from rlinf.utils.logging import get_logger

__all__ = [
    "RLT_TOKEN_FORMAT_VERSION",
    "NativeJaxRLTFeatureModel",
    "NativeJaxRLTOffsetFeatureClient",
    "build_native_jax_requests",
    "get_model",
    "load_stage1_rlt_encoder",
    "rlt_encoder_fingerprint",
]


def _numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _one(value: Any, index: int, batch_size: int) -> Any:
    if isinstance(value, (list, tuple)):
        return value[index]
    array = _numpy(value)
    if array.ndim > 0 and array.shape[0] == batch_size:
        return array[index]
    return array


def _chw_uint8(image: Any) -> np.ndarray:
    """Normalize one camera image to the CHW uint8 contract used by ALOHA.

    The native OpenPI ALOHA input transform unconditionally rearranges each
    image from CHW to HWC.  RealWorldEnv provides HWC images, so the websocket
    request must convert them to CHW before sending them to the JAX server.
    """
    image = _numpy(image)
    if image.ndim != 3:
        raise ValueError(f"Expected one image with 3 dimensions, got {image.shape}.")
    if image.shape[-1] == 3:
        image = np.moveaxis(image, -1, 0)
    elif image.shape[0] != 3:
        raise ValueError(
            f"Expected an RGB image in HWC or CHW layout, got {image.shape}."
        )
    if np.issubdtype(image.dtype, np.floating):
        if image.max(initial=0.0) <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0.0, 255.0)
    return np.ascontiguousarray(image, dtype=np.uint8)


def _request_fingerprint(requests: list[dict[str, Any]]) -> bytes:
    """Fingerprint the exact native-JAX inputs used by one rollout batch.

    RLinf asks for the final observation once to close an RLT transition, then
    asks for the same observation again at the start of the next rollout
    exchange.  Hashing the complete request lets us safely reuse the frozen
    JAX/Stage-1 feature result without relying on object identity.
    """
    digest = hashlib.blake2b(digest_size=16)
    for request in requests:
        digest.update(str(request["prompt"]).encode("utf-8"))
        for value in (
            request["state"],
            request["images"]["cam_high"],
            request["images"]["cam_left_wrist"],
            request["images"]["cam_right_wrist"],
        ):
            array = np.ascontiguousarray(value)
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            digest.update(memoryview(array).cast("B"))
    return digest.digest()


def build_native_jax_requests(
    env_obs: dict[str, Any], *, action_dim: int, default_prompt: str
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Build one websocket request per environment from a batched observation."""
    states = _numpy(env_obs["states"])
    if states.ndim == 1:
        states = states[None, :]
    if states.shape[-1] != action_dim:
        raise ValueError(f"Expected {action_dim}-D Piper state, got {states.shape}.")
    batch_size = states.shape[0]
    main_images = env_obs["main_images"]
    extra_images = env_obs.get("extra_view_images")
    if extra_images is None:
        raise ValueError("Piper native-JAX RLT requires both wrist cameras.")

    requests = []
    for index in range(batch_size):
        extra = _one(extra_images, index, batch_size)
        if extra.shape[0] != 2:
            raise ValueError(f"Expected left/right wrist images, got {extra.shape}.")
        task_descriptions = env_obs.get("task_descriptions")
        prompt = (
            _one(task_descriptions, index, batch_size)
            if task_descriptions is not None
            else default_prompt
        )
        if isinstance(prompt, bytes):
            prompt = prompt.decode()
        requests.append(
            {
                "state": np.ascontiguousarray(states[index], dtype=np.float32),
                "images": {
                    "cam_high": _chw_uint8(_one(main_images, index, batch_size)),
                    "cam_left_wrist": _chw_uint8(extra[0]),
                    "cam_right_wrist": _chw_uint8(extra[1]),
                },
                "prompt": str(prompt),
            }
        )
    return requests, states


def _with_bridge_delta(
    rlt_obs: dict[str, torch.Tensor], env_obs: dict[str, Any], states: np.ndarray
) -> dict[str, torch.Tensor]:
    """Add ``bridge_delta``: last queued RTC bridge command minus the state.

    The env worker attaches ``rlt_bridge_target`` for the chunk about to be
    decided. Without it (episode start, operator control, no bridge) the chunk
    starts from the measured state, so the delta is zero.
    """
    proprio = rlt_obs["proprio"]
    target = env_obs.get("rlt_bridge_target")
    if target is None:
        delta = torch.zeros_like(proprio)
    else:
        target = torch.as_tensor(_numpy(target), dtype=torch.float32).reshape(
            states.shape[0], -1
        )
        if target.shape != tuple(states.shape):
            raise ValueError(
                f"rlt_bridge_target {tuple(target.shape)} does not match states "
                f"{tuple(states.shape)}."
            )
        delta = (target - torch.as_tensor(states, dtype=torch.float32)).to(proprio)
    return {**rlt_obs, "bridge_delta": delta}


def _with_head_pose(
    rlt_obs: dict[str, torch.Tensor], env_obs: dict[str, Any]
) -> dict[str, torch.Tensor]:
    """Add the Piper connector-head state ``[dx, dy, dangle, visible]`` if any.

    The env computes it from the full-resolution right wrist frame
    (``PiperRobotConfig.head_pose_weights``); it is not part of the VLA input.
    """
    head_pose = env_obs.get("head_pose")
    if head_pose is None:
        return rlt_obs
    proprio = rlt_obs["proprio"]
    head_pose = torch.as_tensor(_numpy(head_pose), dtype=torch.float32)
    return {**rlt_obs, "head_pose": head_pose.reshape(proprio.shape[0], -1).to(proprio)}


class NativeJaxRLTFeatureModel(torch.nn.Module):
    """Build Stage-2 observations from one native-JAX action/prefix response.

    The remote OpenPI process performs all image, language and action-expert
    computation. This module owns only the frozen Stage-1 RLT token encoder;
    no converted PyTorch VLA is constructed.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.host = str(cfg.host)
        self.port = int(cfg.port)
        self.prefix_seq_len = int(cfg.prefix_seq_len)
        self.ref_num_action_chunks = int(cfg.ref_num_action_chunks)
        self.native_action_horizon = int(
            cfg.get("native_action_horizon", self.ref_num_action_chunks)
        )
        if self.native_action_horizon < self.ref_num_action_chunks:
            raise ValueError("native_action_horizon must be >= ref_num_action_chunks.")
        if bool(cfg.get("rtc_enabled", False)) and (
            self.native_action_horizon == self.ref_num_action_chunks
        ):
            raise ValueError(
                "RTC requires native_action_horizon > ref_num_action_chunks "
                "so the execution layer receives a non-empty VLA suffix."
            )
        self.action_dim = int(cfg.action_dim)
        self.default_prompt = str(cfg.default_prompt)
        self.require_compact_response = bool(cfg.get("require_compact_response", False))
        self._logger = get_logger()
        self._timing_count = 0
        self._timing_jax_wall_ms = 0.0
        self._timing_jax_model_ms = 0.0
        self._timing_encoder_ms = 0.0
        self._cached_request_fingerprint: bytes | None = None
        self._cached_rlt_obs: dict[str, torch.Tensor] | None = None
        self._cache_hit_count = 0

        if self.require_compact_response:
            self.encoder_fingerprint = rlt_encoder_fingerprint(
                str(cfg.rlt_token_checkpoint), prefix_seq_len=self.prefix_seq_len
            )
            encoder = torch.nn.Identity()
        else:
            encoder, self.encoder_fingerprint = load_stage1_rlt_encoder(
                str(cfg.rlt_token_checkpoint), prefix_seq_len=self.prefix_seq_len
            )
        self._logger.info(
            "Stage-1 RLT encoder fingerprint %s (the robot-side JAX server must match).",
            self.encoder_fingerprint,
        )
        self.encoder = encoder.to(dtype=torch.bfloat16)
        self.register_buffer("_device_anchor", torch.empty(0), persistent=False)
        self._client = None

    def _connect(self):
        if self._client is None:
            client = websocket_client_policy.WebsocketClientPolicy(
                host=self.host,
                port=self.port,
            )
            metadata = client.get_server_metadata()
            if getattr(self, "require_compact_response", False) and not metadata.get(
                "rlt_action_z_rl", False
            ):
                raise RuntimeError(
                    "This rollout requires compact action+z_rl responses. Restart the robot "
                    "service with toolkits/realworld_check/serve_piper_rlt_compact.py."
                )
            if not metadata.get("rlt_native_jax_prefix", False):
                raise RuntimeError(
                    "The OpenPI server does not expose native-JAX RLT prefixes. "
                    "Restart scripts/serve_policy.py with --return-rlt-prefix."
                )
            server_len = int(metadata.get("rlt_prefix_seq_len", -1))
            if server_len != self.prefix_seq_len:
                raise RuntimeError(
                    f"JAX server prefix length {server_len} != {self.prefix_seq_len}."
                )
            server_fingerprint = metadata.get("rlt_encoder_fingerprint")
            own_fingerprint = getattr(self, "encoder_fingerprint", None)
            if (
                server_fingerprint
                and own_fingerprint
                and server_fingerprint != own_fingerprint
            ):
                # Stride windows take z_rl from the server's encoder and chunk
                # rows from this one; two tokens would mix feature spaces in
                # one replay buffer.
                raise RuntimeError(
                    "The JAX server encodes z_rl with a different Stage-1 token: "
                    f"{server_fingerprint} != {own_fingerprint}. Start "
                    "serve_policy.py with the same --rlt-token-checkpoint as "
                    "RLINF_STAGE1_RLT_TOKEN."
                )
            self._client = client
        return self._client

    @torch.no_grad()
    def extract_rlt_obs(self, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        requests, states = build_native_jax_requests(
            env_obs, action_dim=self.action_dim, default_prompt=self.default_prompt
        )
        batch_size = len(requests)

        request_fingerprint = _request_fingerprint(requests)
        cached_fingerprint = getattr(self, "_cached_request_fingerprint", None)
        cached_rlt_obs = getattr(self, "_cached_rlt_obs", None)
        if cached_rlt_obs is not None and cached_fingerprint == request_fingerprint:
            self._cache_hit_count = getattr(self, "_cache_hit_count", 0) + 1
            if self._cache_hit_count % 25 == 0:
                self._logger.info(
                    "Native-JAX RLT exact-observation cache served %d repeated "
                    "rollout-boundary requests.",
                    self._cache_hit_count,
                )
            return _with_head_pose(
                _with_bridge_delta(cached_rlt_obs, env_obs, states), env_obs
            )

        prefixes = []
        masks = []
        ref_chunks = []
        compact_tokens = []
        client = self._connect()
        jax_wall_ms = 0.0
        jax_model_ms = 0.0
        for request in requests:
            jax_started = time.monotonic()
            response = client.infer(request)
            jax_wall_ms += (time.monotonic() - jax_started) * 1000.0
            jax_model_ms += float(
                response.get("policy_timing", {}).get("infer_ms", 0.0)
            )
            actions = np.asarray(response["actions"], dtype=np.float32)
            if "rlt_z_rl" in response:
                fingerprint = response.get("rlt_encoder_fingerprint")
                if fingerprint != self.encoder_fingerprint:
                    raise RuntimeError(
                        "Compact response encoder fingerprint does not match Stage 1."
                    )
                token = np.asarray(response["rlt_z_rl"], dtype=np.float32)
                if token.shape != (2048,) or not np.isfinite(token).all():
                    raise ValueError(
                        f"Invalid compact RLT token shape/value: {token.shape}."
                    )
                if not np.asarray(response.get("rlt_prefix_valid", False)).all():
                    raise ValueError(
                        "Compact response contains masked image prefix tokens."
                    )
                compact_tokens.append(token)
            else:
                if getattr(self, "require_compact_response", False):
                    raise RuntimeError("Compact server omitted action-request z_rl.")
                prefix = np.asarray(response["rlt_prefix"])
                mask = np.asarray(response["rlt_prefix_mask"])
                if prefix.shape != (self.prefix_seq_len, 2048):
                    raise ValueError(f"Unexpected JAX prefix shape: {prefix.shape}.")
                if mask.shape != (self.prefix_seq_len,):
                    raise ValueError(f"Unexpected JAX prefix mask shape: {mask.shape}.")
                prefixes.append(prefix)
                masks.append(mask)
            if actions.ndim != 2 or actions.shape[-1] != self.action_dim:
                raise ValueError(f"Unexpected JAX action shape: {actions.shape}.")
            ref_chunks.append(
                actions[
                    : getattr(
                        self,
                        "native_action_horizon",
                        self.ref_num_action_chunks,
                    )
                ]
            )

        device = self._device_anchor.device
        encoder_started = time.monotonic()
        if compact_tokens:
            if len(compact_tokens) != batch_size:
                raise RuntimeError(
                    "Mixed compact/raw responses in one batch are unsupported."
                )
            z_rl = torch.as_tensor(
                np.stack(compact_tokens), device=device, dtype=torch.float32
            )
        else:
            prefix_tensor = torch.as_tensor(
                np.stack(prefixes), device=device, dtype=torch.bfloat16
            )
            mask_tensor = torch.as_tensor(
                np.stack(masks), device=device, dtype=torch.bool
            )
            z_rl = self.encoder(prefix_tensor, mask_tensor).reshape(batch_size, -1)
        if z_rl.is_cuda:
            torch.cuda.synchronize(z_rl.device)
        encoder_ms = (time.monotonic() - encoder_started) * 1000.0
        self._timing_count += 1
        self._timing_jax_wall_ms += jax_wall_ms
        self._timing_jax_model_ms += jax_model_ms
        self._timing_encoder_ms += encoder_ms
        if self._timing_count >= 25:
            count = float(self._timing_count)
            self._logger.info(
                "Native-JAX RLT timing over %d requests: "
                "jax_wall=%.1fms, jax_model=%.1fms, rpc_and_prefix=%.1fms, "
                "rlt_encoder=%.1fms.",
                self._timing_count,
                self._timing_jax_wall_ms / count,
                self._timing_jax_model_ms / count,
                (self._timing_jax_wall_ms - self._timing_jax_model_ms) / count,
                self._timing_encoder_ms / count,
            )
            self._timing_count = 0
            self._timing_jax_wall_ms = 0.0
            self._timing_jax_model_ms = 0.0
            self._timing_encoder_ms = 0.0
        rlt_obs = {
            "z_rl": z_rl.to(dtype=torch.float32),
            "proprio": torch.as_tensor(states, device=device, dtype=torch.float32),
            "ref_chunk": torch.as_tensor(
                np.stack(ref_chunks), device=device, dtype=torch.float32
            ),
        }
        self._cached_request_fingerprint = request_fingerprint
        self._cached_rlt_obs = rlt_obs
        return _with_head_pose(_with_bridge_delta(rlt_obs, env_obs, states), env_obs)


class NativeJaxRLTOffsetFeatureClient:
    """Robot-side client that turns intermediate observations into z_rl.

    The JAX server on the robot computes the prefix and runs the Stage-1
    encoder itself, so only the 2048-D token crosses the websocket. This keeps
    stride subsampling off the cross-machine path the action request uses.
    """

    def __init__(
        self,
        *,
        host: str,
        port: int,
        prefix_seq_len: int,
        action_dim: int,
        default_prompt: str,
        expected_encoder_fingerprint: str | None = None,
        client: Any | None = None,
    ):
        self.host = str(host)
        self.port = int(port)
        self.prefix_seq_len = int(prefix_seq_len)
        self.action_dim = int(action_dim)
        self.default_prompt = str(default_prompt)
        self.expected_encoder_fingerprint = expected_encoder_fingerprint
        self.encoder_fingerprint: str | None = None
        self._client = client
        self._logger = get_logger()
        self._timing_count = 0
        self._timing_wall_ms = 0.0
        self._timing_model_ms = 0.0

    def _connect(self):
        if self._client is not None and self.encoder_fingerprint is not None:
            return self._client
        client = self._client or websocket_client_policy.WebsocketClientPolicy(
            host=self.host, port=self.port
        )
        metadata = client.get_server_metadata()
        if not metadata.get("rlt_native_jax_prefix", False) or not metadata.get(
            "rlt_prefix_only", False
        ):
            raise RuntimeError(
                "RLT stride subsampling needs z_rl at intermediate steps. Restart "
                "scripts/serve_policy.py with --return-rlt-prefix and "
                "--rlt-token-checkpoint."
            )
        server_len = int(metadata.get("rlt_prefix_seq_len", -1))
        if server_len != self.prefix_seq_len:
            raise RuntimeError(
                f"JAX server prefix length {server_len} != {self.prefix_seq_len}."
            )
        fingerprint = str(metadata.get("rlt_encoder_fingerprint", ""))
        if (
            self.expected_encoder_fingerprint
            and fingerprint != self.expected_encoder_fingerprint
        ):
            raise RuntimeError(
                "Robot-side RLT encoder does not match the rollout encoder: "
                f"{fingerprint} != {self.expected_encoder_fingerprint}."
            )
        self.encoder_fingerprint = fingerprint
        self._logger.info("RLT stride subsampling uses encoder %s.", fingerprint)
        self._client = client
        return client

    def extract_offset_features(
        self, offset_obs: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor]:
        """Return z_rl [B, W, Z] and proprio [B, W, P] for W batched observations."""
        if not offset_obs:
            raise ValueError("extract_offset_features needs at least one observation.")
        per_offset = [
            build_native_jax_requests(
                obs, action_dim=self.action_dim, default_prompt=self.default_prompt
            )
            for obs in offset_obs
        ]
        batch_size = len(per_offset[0][0])
        num_offsets = len(per_offset)
        client = self._connect()

        z_rows = []
        proprio_rows = []
        wall_ms = 0.0
        model_ms = 0.0
        for env_index in range(batch_size):
            started = time.monotonic()
            response = client.infer(
                {
                    "rlt_prefix_batch": [
                        requests[env_index] for requests, _ in per_offset
                    ]
                }
            )
            wall_ms += (time.monotonic() - started) * 1000.0
            model_ms += float(response.get("policy_timing", {}).get("infer_ms", 0.0))
            z_rl = np.asarray(response["rlt_z_rl"], dtype=np.float32)
            if z_rl.ndim != 2 or z_rl.shape[0] != num_offsets:
                raise ValueError(f"Unexpected prefix-only z_rl shape: {z_rl.shape}.")
            valid = np.asarray(
                response.get("rlt_prefix_valid", np.ones(num_offsets, dtype=bool))
            )
            if not bool(valid.all()):
                raise RuntimeError(
                    "Prefix-only response contained masked image tokens."
                )
            z_rows.append(z_rl)
            proprio_rows.append(
                np.stack([states[env_index] for _, states in per_offset]).astype(
                    np.float32
                )
            )

        self._timing_count += 1
        self._timing_wall_ms += wall_ms
        self._timing_model_ms += model_ms
        if self._timing_count >= 25:
            count = float(self._timing_count)
            self._logger.info(
                "RLT stride z_rl timing over %d chunks (%d offsets each): "
                "wall=%.1fms, model=%.1fms.",
                self._timing_count,
                num_offsets,
                self._timing_wall_ms / count,
                self._timing_model_ms / count,
            )
            self._timing_count = 0
            self._timing_wall_ms = 0.0
            self._timing_model_ms = 0.0

        return {
            "z_rl": torch.from_numpy(np.stack(z_rows)),
            "proprio": torch.from_numpy(np.stack(proprio_rows)),
        }


def get_model(cfg: DictConfig, torch_dtype=None) -> NativeJaxRLTFeatureModel:
    del torch_dtype
    return NativeJaxRLTFeatureModel(cfg)
