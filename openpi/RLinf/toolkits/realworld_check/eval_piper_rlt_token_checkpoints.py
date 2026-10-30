#!/usr/bin/env python3
"""Compare Stage-1 RLT token checkpoints on identical fixed observations."""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jax-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--num-batches", type=int, default=32)
    args = parser.parse_args()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        devices = [item.strip() for item in visible.split(",")]
        os.environ["CUDA_VISIBLE_DEVICES"] = devices[rank]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

    import jax
    import jax.numpy as jnp
    from flax import nnx
    import torch

    from openpi.models import model as openpi_model
    from openpi.models.pi0 import make_attn_mask
    from openpi.training import config as training_config
    from openpi.training import data_loader
    from rlinf.models.embodiment.modules.rlt_token_transformer import (
        RLTTokenTransformer,
    )

    checkpoints = sorted(
        args.checkpoint_root.glob("step_*/rlt_token.pt"),
        key=lambda path: int(path.parent.name.removeprefix("step_")),
    )
    if rank >= len(checkpoints):
        return 0
    checkpoint_path = checkpoints[rank]
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    jax_device = jax.devices("gpu")[0]

    config = dataclasses.replace(training_config.get_config("pi05_wire"), batch_size=1)
    data_factory = dataclasses.replace(
        config.data,
        repo_id=str(args.dataset_dir.resolve()),
        assets=dataclasses.replace(
            config.data.assets,
            assets_dir=str((args.jax_checkpoint / "assets").resolve()),
        ),
    )
    data_config = data_factory.create(config.assets_dirs, config.model)
    batches = data_loader.create_torch_data_loader(
        data_config,
        config.model,
        config.model.action_horizon,
        1,
        shuffle=False,
        num_batches=args.num_batches,
        num_workers=0,
        seed=0,
        framework="pytorch",
    )

    with jax.default_device(jax_device):
        params = openpi_model.restore_params(
            args.jax_checkpoint / "params", dtype=jnp.bfloat16
        )
        vla = config.model.load(params)
        vla.eval()

    @nnx.jit
    def extract_image_prefix(model, observation):
        observation = openpi_model.preprocess_observation(
            None, observation, train=False
        )
        prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(observation)
        attention_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_output, _), _ = model.PaliGemma.llm(
            [prefix_tokens, None],
            mask=attention_mask,
            positions=positions,
        )
        return prefix_output[:, :768], prefix_mask[:, :768]

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    rlt = RLTTokenTransformer(prefix_seq_len=768)
    rlt.load_state_dict(checkpoint["rlt_module"], strict=True)
    rlt.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    del checkpoint
    gc.collect()

    mse_values = []
    cosine_values = []
    with torch.no_grad():
        for observation, _actions in batches:
            jax_observation = jax.tree.map(
                lambda value: jax.device_put(
                    np.asarray(value.detach().cpu()), jax_device
                ),
                observation,
            )
            jax_observation = dataclasses.replace(
                jax_observation,
                images={
                    name: (
                        jnp.transpose(image, (0, 2, 3, 1))
                        if image.ndim == 4
                        and image.shape[1] in (1, 3, 4)
                        and image.shape[-1] not in (1, 3, 4)
                        else image
                    )
                    for name, image in jax_observation.images.items()
                },
            )
            prefix, mask = extract_image_prefix(vla, jax_observation)
            prefix_torch = torch.utils.dlpack.from_dlpack(prefix).to(
                device=device, dtype=torch.bfloat16
            )
            mask_torch = torch.utils.dlpack.from_dlpack(mask).to(
                device=device, dtype=torch.bool
            )
            reconstructed, _ = rlt.reconstruct(prefix_torch, mask_torch)
            target = prefix_torch.float()
            prediction = reconstructed.float()
            valid = mask_torch[..., None].float()
            squared = ((prediction - target) ** 2) * valid
            mse = squared.sum() / (valid.sum() * target.shape[-1]).clamp_min(1)
            cosine = torch.nn.functional.cosine_similarity(
                prediction, target, dim=-1
            )
            cosine = (cosine * mask_torch.float()).sum() / mask_torch.sum().clamp_min(1)
            mse_values.append(float(mse))
            cosine_values.append(float(cosine))

    result = {
        "step": int(checkpoint_path.parent.name.removeprefix("step_")),
        "num_batches": len(mse_values),
        "fixed_data_mse": float(np.mean(mse_values)),
        "fixed_data_cosine": float(np.mean(cosine_values)),
        "checkpoint": str(checkpoint_path),
        "selection_scope": "fixed training-distribution observations; not held-out",
    }
    output_path = checkpoint_path.parent / "fixed_eval.json"
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
