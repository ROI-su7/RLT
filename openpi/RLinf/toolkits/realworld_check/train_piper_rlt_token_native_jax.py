#!/usr/bin/env python3
"""Train only the RLT encoder-decoder from frozen native-JAX π0.5 features."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path

import numpy as np


def _lr(step: int, *, peak: float, warmup: int, total: int, floor: float) -> float:
    if step < warmup:
        return peak * float(step + 1) / float(max(warmup, 1))
    progress = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jax-checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    # The config must describe the checkpoint's architecture and norm stats:
    # pi05 puts the discretized state in the prefix, so a mismatched config
    # shifts the very features the token is trained to reconstruct.
    parser.add_argument("--config-name", default="pi05_wire_all_data_full")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--min-lr", type=float, default=2.5e-6)
    parser.add_argument("--warmup", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Keep RL-token parameters, gradients and AdamW state on the CPU (FSDP "
        "CPUOffload) so a single 24 GB GPU fits next to the frozen VLA; slower.",
    )
    args = parser.parse_args()

    # torchrun exposes every GPU to every child by default.  Orbax restores a
    # single-device JAX checkpoint onto JAX device 0, so leaving all eight GPUs
    # visible makes all ranks restore onto the same physical GPU.  Pin each
    # process before importing either JAX or Torch; inside the process its
    # assigned physical GPU is then consistently logical device 0 for both
    # frameworks and for DLPack.
    local_rank = int(os.environ["LOCAL_RANK"])
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        devices = [device.strip() for device in visible_devices.split(",")]
        if local_rank >= len(devices):
            raise RuntimeError(
                f"LOCAL_RANK={local_rank} is outside CUDA_VISIBLE_DEVICES="
                f"{visible_devices!r}"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = devices[local_rank]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    import jax
    import jax.numpy as jnp
    from flax import nnx
    import torch
    import torch.distributed as dist
    from torch.distributed.fsdp import (
        CPUOffload,
        FullOptimStateDictConfig,
        FullStateDictConfig,
        FullyShardedDataParallel,
        MixedPrecision,
        ShardingStrategy,
        StateDictType,
    )

    from openpi.models import model as openpi_model
    from openpi.training import config as training_config
    from openpi.training import data_loader
    from rlinf.models.embodiment.modules.rlt_token_transformer import (
        RLTTokenTransformer,
    )
    from rlinf.models.embodiment.modules.rlt_token_checkpoint import (
        RLT_TOKEN_FORMAT_VERSION,
    )

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(0)
    torch_device = torch.device("cuda", 0)
    jax_device = jax.devices("gpu")[0]
    torch.manual_seed(args.seed + rank)

    config = training_config.get_config(args.config_name)
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
        world_size,
        shuffle=True,
        num_batches=args.steps * args.accumulate,
        num_workers=0,
        seed=args.seed,
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
        prefix_output, prefix_mask = model.extract_rlt_prefix(observation)
        # Three 224x224 cameras, 14x14 patches: 3 * 256 image tokens.
        return prefix_output[:, :768], prefix_mask[:, :768]

    # Master weights and optimizer state stay in fp32. With bf16 parameters,
    # an AdamW step of ~lr is smaller than half the bf16 spacing for most
    # weights (and all LayerNorm scales at 1.0), so it rounds away entirely.
    rlt = RLTTokenTransformer(prefix_seq_len=768).to(
        device=torch_device, dtype=torch.float32
    )
    rlt = FullyShardedDataParallel(
        rlt,
        device_id=torch_device,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            buffer_dtype=torch.float32,
        ),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=False,
        cpu_offload=CPUOffload(offload_params=args.cpu_offload),
    )
    optimizer = torch.optim.AdamW(
        rlt.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
    )
    iterator = iter(batches)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        # JAX keeps the frozen VLA executable/parameters resident while
        # PyTorch's allocator retains RLT activation blocks after backward.
        # Interleaving one JAX forward with one Torch backward therefore makes
        # the two allocators fight for the last few MiB on a 24-GiB 3090.
        # Extract the whole accumulation window first and stage the small
        # frozen features on CPU, then run the Torch backward passes.
        torch.cuda.empty_cache()
        feature_batches = []
        for _ in range(args.accumulate):
            observation, _actions = next(iterator)
            jax_observation = jax.tree.map(
                lambda value: jax.device_put(
                    np.asarray(value.detach().cpu()), jax_device
                ),
                observation,
            )
            # OpenPI's Torch loader represents normalized images as NCHW,
            # while the native JAX/SigLIP path consumes NHWC.
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
            prefix_torch = (
                torch.utils.dlpack.from_dlpack(prefix)
                .to(device="cpu", dtype=torch.bfloat16)
                .clone()
            )
            mask_torch = (
                torch.utils.dlpack.from_dlpack(mask)
                .to(device="cpu", dtype=torch.bool)
                .clone()
            )
            feature_batches.append((prefix_torch, mask_torch))
            del prefix, mask, prefix_torch, mask_torch, jax_observation

        for prefix_cpu, mask_cpu in feature_batches:
            prefix_torch = prefix_cpu.to(torch_device)
            mask_torch = mask_cpu.to(torch_device)
            loss, _ = rlt(prefix_torch, mask_torch)
            (loss / args.accumulate).backward()
            loss_sum += float(loss.detach())
            del loss, prefix_torch, mask_torch
        del feature_batches

        current_lr = _lr(
            step,
            peak=args.lr,
            warmup=args.warmup,
            total=args.steps,
            floor=args.min_lr,
        )
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        rlt.clip_grad_norm_(1.0)
        optimizer.step()

        mean_loss = torch.tensor(loss_sum / args.accumulate, device=torch_device)
        dist.all_reduce(mean_loss)
        mean_loss /= world_size
        if rank == 0 and (step % 10 == 0 or step + 1 == args.steps):
            print(
                json.dumps(
                    {"step": step + 1, "rlt_loss": mean_loss.item(), "lr": current_lr}
                ),
                flush=True,
            )
        if (step + 1) % args.save_every == 0 or step + 1 == args.steps:
            full_state_config = FullStateDictConfig(
                offload_to_cpu=True, rank0_only=True
            )
            full_optim_config = FullOptimStateDictConfig(
                offload_to_cpu=True, rank0_only=True
            )
            with FullyShardedDataParallel.state_dict_type(
                rlt,
                StateDictType.FULL_STATE_DICT,
                full_state_config,
                full_optim_config,
            ):
                rlt_state = rlt.state_dict()
                optimizer_state = FullyShardedDataParallel.optim_state_dict(
                    rlt, optimizer
                )
            if rank == 0:
                checkpoint_dir = args.output_dir / f"step_{step + 1}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "rlt_module": rlt_state,
                        "optimizer": optimizer_state,
                        "step": step + 1,
                        "source_jax_checkpoint": str(
                            args.jax_checkpoint.resolve()
                        ),
                        "config_name": args.config_name,
                        "format_version": RLT_TOKEN_FORMAT_VERSION,
                        "decoder": "autoregressive",
                        "master_dtype": "float32",
                        "image_only": True,
                        "prefix_seq_len": 768,
                        "sharding_strategy": "FULL_SHARD",
                    },
                    checkpoint_dir / "rlt_token.pt",
                )
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
