#!/usr/bin/env python3
"""Generate deterministic OpenPI actions for JAX/PyTorch parity checks.

The script deliberately loads raw observations from the same LeRobot dataset
and injects NumPy-generated denoising noise into both runtimes.  Run it once
with ``--runtime jax`` and once with ``--runtime torch``, then compare the two
archives with ``compare_openpi_action_parity.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

import numpy as np


def _as_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        return value.numpy()
    return np.asarray(value)


def _parse_indices(value: str) -> list[int]:
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices:
        raise argparse.ArgumentTypeError("at least one sample index is required")
    if any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("sample indices must be non-negative")
    return indices


def _raw_observation(sample: dict, prompt: str) -> dict:
    """Convert one raw LeRobot row into the live Piper policy contract."""
    return {
        "state": np.asarray(sample["observation.state"], dtype=np.float32),
        "images": {
            "cam_high": np.asarray(sample["observation.images.cam_high"]),
            "cam_left_wrist": np.asarray(sample["observation.images.cam_left_wrist"]),
            "cam_right_wrist": np.asarray(sample["observation.images.cam_right_wrist"]),
        },
        "prompt": prompt,
    }


def _trace_jax_denoising(policy, raw_obs: dict, noise: np.ndarray, num_steps: int):
    """Return post-preprocessing inputs and every deterministic Euler step."""
    import jax
    import jax.numpy as jnp

    from openpi.models import model as openpi_model
    from openpi.models.pi0 import make_attn_mask

    inputs = policy._input_transform(  # noqa: SLF001
        jax.tree.map(lambda value: value, raw_obs)
    )
    inputs = jax.tree.map(lambda value: jnp.asarray(value)[None, ...], inputs)
    observation = openpi_model.Observation.from_dict(inputs)
    model = policy._model  # noqa: SLF001

    def trace_fn(observation, initial_noise):
        observation = openpi_model.preprocess_observation(
            None, observation, train=False
        )
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_output, _), kv_cache = model.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions,
        )
        batch_size = observation.state.shape[0]
        dt = jnp.asarray(-1.0 / num_steps, dtype=jnp.float32)

        def step(carry, _):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = (
                model.embed_suffix(
                    observation,
                    x_t,
                    jnp.broadcast_to(time, (batch_size,)),
                )
            )
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_to_suffix_mask = jnp.broadcast_to(
                prefix_mask[:, None, :],
                (
                    batch_size,
                    suffix_tokens.shape[1],
                    prefix_mask.shape[1],
                ),
            )
            full_attn_mask = jnp.concatenate(
                [prefix_to_suffix_mask, suffix_attn_mask], axis=-1
            )
            suffix_positions = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1)
                - 1
            )
            (_, suffix_out), _ = model.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=suffix_positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            suffix_out = suffix_out[:, -model.action_horizon :]
            velocity = model.action_out_proj(suffix_out)
            next_x = x_t + dt * velocity
            return (next_x, time + dt), (
                velocity,
                next_x,
                suffix_tokens,
                suffix_out,
            )

        (_, _), (velocities, states, suffix_tokens, suffix_outputs) = jax.lax.scan(
            step,
            (initial_noise, jnp.asarray(1.0, dtype=jnp.float32)),
            xs=None,
            length=num_steps,
        )
        return (
            observation,
            prefix_tokens,
            prefix_output,
            velocities,
            states,
            suffix_tokens,
            suffix_outputs,
        )

    (
        preprocessed,
        prefix_tokens,
        prefix_output,
        velocities,
        states,
        suffix_tokens,
        suffix_outputs,
    ) = jax.jit(trace_fn)(observation, jnp.asarray(noise[None, ...]))
    image_names = sorted(preprocessed.images)
    return {
        "input_state": _as_numpy(preprocessed.state[0]),
        "input_tokens": _as_numpy(preprocessed.tokenized_prompt[0]),
        "input_token_mask": _as_numpy(preprocessed.tokenized_prompt_mask[0]),
        "input_images": np.stack(
            [_as_numpy(preprocessed.images[name][0]) for name in image_names]
        ),
        "input_image_masks": np.stack(
            [_as_numpy(preprocessed.image_masks[name][0]) for name in image_names]
        ),
        "prefix_tokens": _as_numpy(prefix_tokens[0]),
        "prefix_output": _as_numpy(prefix_output[0]),
        "first_suffix_tokens": _as_numpy(suffix_tokens[0, 0]),
        "first_suffix_output": _as_numpy(suffix_outputs[0, 0]),
        "denoise_velocity": _as_numpy(velocities[:, 0]),
        "denoise_state": _as_numpy(states[:, 0]),
    }


def _trace_torch_denoising(
    policy,
    raw_obs: dict,
    noise: np.ndarray,
    num_steps: int,
    device: str,
):
    """PyTorch counterpart of :func:`_trace_jax_denoising`."""
    import jax
    import torch

    from openpi.models import model as openpi_model
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks

    inputs = policy._input_transform(  # noqa: SLF001
        jax.tree.map(lambda value: value, raw_obs)
    )
    inputs = jax.tree.map(
        lambda value: torch.from_numpy(np.asarray(value)).to(device)[None, ...],
        inputs,
    )
    observation = openpi_model.Observation.from_dict(inputs)
    model = policy._model  # noqa: SLF001
    images, image_masks, lang_tokens, lang_masks, state = (
        model._preprocess_observation(observation, train=False)  # noqa: SLF001
    )
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images, image_masks, lang_tokens, lang_masks
    )
    prefix_att_2d_masks = make_att_2d_masks(
        prefix_pad_masks, prefix_att_masks
    )
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_2d_masks_4d = model._prepare_attention_masks_4d(  # noqa: SLF001
        prefix_att_2d_masks
    )
    # Match PI0Pytorch.sample_actions exactly. The default SDPA route rejects
    # the float32 additive mask when q/k/v are bfloat16 and is not the runtime
    # used by the validated inference path.
    model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = (  # noqa: SLF001
        "eager"
    )
    model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = (  # noqa: SLF001
        "eager"
    )
    (prefix_outputs, past_key_values) = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_2d_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    prefix_output = prefix_outputs[0]

    x_t = torch.from_numpy(noise).to(device)[None, ...]
    dt = torch.tensor(-1.0 / num_steps, dtype=torch.float32, device=device)
    time = torch.tensor(1.0, dtype=torch.float32, device=device)
    velocities = []
    states = []
    first_suffix_tokens = None
    first_suffix_output = None
    with torch.no_grad():
        for step_index in range(num_steps):
            if step_index == 0:
                first_suffix_tokens = model.embed_suffix(
                    state, x_t, time.expand(state.shape[0])
                )[0][0].float().cpu().numpy()
                captured_suffix_output = []

                def capture_action_projection_input(_module, inputs):
                    captured_suffix_output.append(inputs[0].detach())

                hook = model.action_out_proj.register_forward_pre_hook(
                    capture_action_projection_input
                )
            velocity = model.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                time.expand(state.shape[0]),
            )
            if step_index == 0:
                hook.remove()
                first_suffix_output = (
                    captured_suffix_output[0][0].float().cpu().numpy()
                )
            x_t = x_t + dt * velocity
            velocities.append(velocity[0].float().cpu().numpy())
            states.append(x_t[0].float().cpu().numpy())
            time += dt

    return {
        "input_state": _as_numpy(state[0]),
        "input_tokens": _as_numpy(lang_tokens[0]),
        "input_token_mask": _as_numpy(lang_masks[0]),
        "input_images": np.stack([_as_numpy(image[0]) for image in images]),
        "input_image_masks": np.stack(
            [_as_numpy(mask[0]) for mask in image_masks]
        ),
        "prefix_tokens": _as_numpy(prefix_embs[0]),
        "prefix_output": _as_numpy(prefix_output[0]),
        "first_suffix_tokens": first_suffix_tokens,
        "first_suffix_output": first_suffix_output,
        "denoise_velocity": np.stack(velocities),
        "denoise_state": np.stack(states),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("jax", "torch"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-name", default="pi05_wire")
    parser.add_argument("--indices", type=_parse_indices, default=[0, 159424, 318848])
    parser.add_argument("--prompt", default="insert the wire")
    parser.add_argument("--noise-seed", type=int, default=20260724)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--torch-device", default="cuda")
    parser.add_argument(
        "--trace-denoise",
        action="store_true",
        help="Save post-preprocessing inputs plus velocity/state at every Euler step.",
    )
    parser.add_argument(
        "--torch-compile-mode",
        choices=("none", "default", "reduce-overhead", "max-autotune"),
        default="none",
    )
    args = parser.parse_args()

    # Import after parsing so --help remains lightweight.
    from openpi.policies import policy_config
    from openpi.policies import policy as openpi_policy
    from openpi import transforms
    from openpi.models_pytorch import pi0_pytorch
    from openpi.training import checkpoints
    from openpi.training import config as training_config
    from openpi.training import data_loader

    config = training_config.get_config(args.config_name)
    # Compilation only changes execution, not the parity contract, and
    # max-autotune can take several minutes on the first 4090 invocation.
    if args.runtime == "torch":
        config = dataclasses.replace(
            config,
            model=dataclasses.replace(
                config.model,
                pytorch_compile_mode=(
                    None
                    if args.torch_compile_mode == "none"
                    else args.torch_compile_mode
                ),
            ),
        )
    data_config = config.data.create(config.assets_dirs, config.model)
    data_config = dataclasses.replace(
        data_config, repo_id=str(args.dataset_dir.resolve())
    )
    dataset = data_loader.create_torch_dataset(
        data_config, config.model.action_horizon, config.model
    )

    if any(index >= len(dataset) for index in args.indices):
        raise IndexError(
            f"indices {args.indices} exceed dataset length {len(dataset)}"
        )

    is_torch_checkpoint = (args.checkpoint / "model.safetensors").is_file()
    if is_torch_checkpoint != (args.runtime == "torch"):
        raise ValueError(
            f"--runtime={args.runtime} does not match checkpoint layout at "
            f"{args.checkpoint}"
        )

    # Prevent JAX from reserving all GPU memory before the PyTorch comparison.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    metadata_path = args.checkpoint / "config.json"
    checkpoint_metadata = (
        json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    )
    if args.runtime == "torch" and checkpoint_metadata.get("lora_format") == "native_jax":
        import safetensors.torch

        from rlinf.models.embodiment.openpi.native_lora import apply_native_jax_lora

        model = pi0_pytorch.PI0Pytorch(config=config.model)
        if apply_native_jax_lora(model) != 252:
            raise ValueError("Failed to inject all 252 native JAX LoRA branches")
        safetensors.torch.load_model(
            model, str(args.checkpoint / "model.safetensors")
        )
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        policy_data_config = config.data.create(config.assets_dirs, config.model)
        if policy_data_config.asset_id is None:
            raise ValueError("asset_id is required for native LoRA parity")
        norm_stats = checkpoints.load_norm_stats(
            args.checkpoint / "assets", policy_data_config.asset_id
        )
        policy = openpi_policy.Policy(
            model,
            transforms=[
                transforms.InjectDefaultPrompt(args.prompt),
                *policy_data_config.data_transforms.inputs,
                transforms.Normalize(
                    norm_stats, use_quantiles=policy_data_config.use_quantile_norm
                ),
                *policy_data_config.model_transforms.inputs,
            ],
            output_transforms=[
                *policy_data_config.model_transforms.outputs,
                transforms.Unnormalize(
                    norm_stats, use_quantiles=policy_data_config.use_quantile_norm
                ),
                *policy_data_config.data_transforms.outputs,
            ],
            sample_kwargs={"num_steps": args.num_steps},
            is_pytorch=True,
            pytorch_device=args.torch_device,
        )
    else:
        policy = policy_config.create_trained_policy(
            config,
            args.checkpoint,
            sample_kwargs={"num_steps": args.num_steps},
            default_prompt=args.prompt,
            pytorch_device=args.torch_device if args.runtime == "torch" else None,
        )

    rng = np.random.default_rng(args.noise_seed)
    all_actions = []
    timings_ms = []
    episode_indices = []
    frame_indices = []
    traces: dict[str, list[np.ndarray]] = {}
    for index in args.indices:
        sample = dataset[index]
        noise = rng.standard_normal(
            (config.model.action_horizon, config.model.action_dim),
            dtype=np.float32,
        )
        raw_obs = _raw_observation(sample, args.prompt)
        if args.trace_denoise:
            trace = (
                _trace_jax_denoising(policy, raw_obs, noise, args.num_steps)
                if args.runtime == "jax"
                else _trace_torch_denoising(
                    policy,
                    raw_obs,
                    noise,
                    args.num_steps,
                    args.torch_device,
                )
            )
            for name, value in trace.items():
                traces.setdefault(name, []).append(np.asarray(value))
        result = policy.infer(raw_obs, noise=noise)
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.shape != (config.model.action_horizon, 14):
            raise ValueError(f"unexpected physical action shape {actions.shape}")
        all_actions.append(actions)
        timings_ms.append(float(result["policy_timing"]["infer_ms"]))
        episode_indices.append(int(sample["episode_index"]))
        frame_indices.append(int(sample["frame_index"]))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "runtime": args.runtime,
        "checkpoint": str(args.checkpoint.resolve()),
        "dataset_dir": str(args.dataset_dir.resolve()),
        "indices": args.indices,
        "episode_indices": episode_indices,
        "frame_indices": frame_indices,
        "prompt": args.prompt,
        "noise_seed": args.noise_seed,
        "num_steps": args.num_steps,
        "torch_compile_mode": args.torch_compile_mode,
        "timings_ms": timings_ms,
        "trace_denoise": args.trace_denoise,
    }
    arrays = {name: np.stack(values) for name, values in traces.items()}
    np.savez_compressed(
        args.output,
        actions=np.stack(all_actions),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        **arrays,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print(f"saved actions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
