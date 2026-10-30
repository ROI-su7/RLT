from collections.abc import Sequence
import importlib.util
import logging
import os
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


def _rlt_token_checkpoint_module():
    """Load RLinf's Stage-1 checkpoint helpers without importing the RLinf package.

    Importing ``rlinf.models`` pulls in Ray, which this serving environment does
    not have; the helper module is written to be loadable by path.
    """
    rlinf_dir = pathlib.Path(
        os.environ.get("RLINF_DIR", pathlib.Path(__file__).resolve().parents[3] / "RLinf")
    )
    path = rlinf_dir / "rlinf" / "models" / "embodiment" / "modules" / "rlt_token_checkpoint.py"
    if not path.is_file():
        raise FileNotFoundError(f"RLT token checkpoint helpers not found at {path}; set RLINF_DIR.")
    spec = importlib.util.spec_from_file_location("rlt_token_checkpoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        return_rlt_prefix: bool = False,
        rlt_prefix_seq_len: int = 768,
        rlt_token_checkpoint: str | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
            rlt_token_checkpoint: Stage-1 RL-token checkpoint. Enables prefix-only
                requests that return z_rl for RLT stride subsampling.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = dict(metadata or {})
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._return_rlt_prefix = bool(return_rlt_prefix)
        self._rlt_prefix_seq_len = int(rlt_prefix_seq_len)
        if self._return_rlt_prefix:
            self._metadata["rlt_native_jax_prefix"] = True
            self._metadata["rlt_prefix_seq_len"] = self._rlt_prefix_seq_len

        if self._is_pytorch_model:
            if self._return_rlt_prefix:
                raise ValueError(
                    "return_rlt_prefix requires the native JAX policy path."
                )
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            sample_method = (
                model.sample_actions_with_prefix
                if self._return_rlt_prefix
                else model.sample_actions
            )
            self._sample_actions = nnx_utils.module_jit(sample_method)
            if self._return_rlt_prefix:
                self._extract_rlt_prefix = nnx_utils.module_jit(model.extract_rlt_prefix)
            self._rng = rng or jax.random.key(0)

        self._rlt_encoder = None
        self._rlt_encoder_device = None
        if rlt_token_checkpoint is not None:
            if not self._return_rlt_prefix:
                raise ValueError("rlt_token_checkpoint requires return_rlt_prefix=True.")
            encoder, fingerprint = _rlt_token_checkpoint_module().load_stage1_rlt_encoder(
                rlt_token_checkpoint, prefix_seq_len=self._rlt_prefix_seq_len
            )
            self._rlt_encoder_device = (
                torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
            )
            self._rlt_encoder = encoder.to(device=self._rlt_encoder_device, dtype=torch.bfloat16)
            self._metadata["rlt_prefix_only"] = True
            self._metadata["rlt_encoder_fingerprint"] = fingerprint
            logging.info("Serving RLT z_rl for prefix-only requests (encoder %s)", fingerprint)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        if "rlt_prefix_batch" in obs:
            return self._infer_rlt_prefix_batch(obs["rlt_prefix_batch"])
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        sample_result = self._sample_actions(
            sample_rng_or_pytorch_device, observation, **sample_kwargs
        )
        if self._return_rlt_prefix:
            actions, prefix_output, prefix_mask = sample_result
            if prefix_output.shape[1] < self._rlt_prefix_seq_len:
                raise ValueError(
                    "Native JAX prefix is shorter than the configured RLT "
                    f"image prefix: {prefix_output.shape[1]} < "
                    f"{self._rlt_prefix_seq_len}"
                )
        else:
            actions = sample_result
        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        if self._return_rlt_prefix:
            # NumPy/msgpack has no portable bfloat16 representation. float16
            # keeps the three-camera prefix payload at about 3 MiB while the
            # Stage-1 encoder converts it back to bfloat16 on the GPU node.
            outputs["rlt_prefix"] = np.asarray(
                prefix_output[0, : self._rlt_prefix_seq_len], dtype=np.float16
            )
            outputs["rlt_prefix_mask"] = np.asarray(
                prefix_mask[0, : self._rlt_prefix_seq_len], dtype=np.bool_
            )
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def _infer_rlt_prefix_batch(self, requests: Sequence[dict]) -> dict:
        """Encode several observations to z_rl without sampling actions.

        RLT stride subsampling needs z_rl every few control steps. Only the
        2048-D token leaves this process, so these requests never carry the
        3 MiB prefix across the network the way the action request does.
        """
        if self._rlt_encoder is None:
            raise ValueError(
                "rlt_prefix_batch requires the policy to be served with --rlt-token-checkpoint."
            )
        if not requests:
            raise ValueError("rlt_prefix_batch must contain at least one observation.")
        items = [self._input_transform(jax.tree.map(lambda x: x, request)) for request in requests]
        inputs = jax.tree.map(lambda *values: jnp.stack([jnp.asarray(value) for value in values]), *items)
        observation = _model.Observation.from_dict(inputs)

        start_time = time.monotonic()
        prefix_output, prefix_mask = self._extract_rlt_prefix(observation)
        if prefix_output.shape[1] < self._rlt_prefix_seq_len:
            raise ValueError(
                "Native JAX prefix is shorter than the configured RLT image prefix: "
                f"{prefix_output.shape[1]} < {self._rlt_prefix_seq_len}"
            )
        prefix = torch.utils.dlpack.from_dlpack(prefix_output[:, : self._rlt_prefix_seq_len])
        mask = torch.utils.dlpack.from_dlpack(prefix_mask[:, : self._rlt_prefix_seq_len])
        with torch.no_grad():
            z_rl = self._rlt_encoder(
                prefix.to(device=self._rlt_encoder_device, dtype=torch.bfloat16),
                mask.to(device=self._rlt_encoder_device, dtype=torch.bool),
            )
        return {
            "rlt_z_rl": z_rl.reshape(len(requests), -1).float().cpu().numpy(),
            "rlt_prefix_valid": mask.to(dtype=torch.bool).all(dim=1).cpu().numpy(),
            "policy_timing": {"infer_ms": (time.monotonic() - start_time) * 1000},
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
