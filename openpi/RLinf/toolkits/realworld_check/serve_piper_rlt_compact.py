#!/usr/bin/env python3
"""Serve native OpenPI actions and compact RL tokens on the robot GPU."""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
from pathlib import Path

import numpy as np
import torch

from rlinf.models.embodiment.native_jax_rlt.compact_server import (
    ActionPriorityServer,
    CompactRLTPolicy,
)


def main() -> None:
    """Load the original checkpoint, warm both paths, and start compact serving."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_connect_strings")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--rlt-token-checkpoint", required=True)
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument(
        "--state-dim", type=int, default=14, help="Robot state size: 14 = Piper, 12 = dual SO101."
    )
    parser.add_argument(
        "--prompt", default=None, help="Task prompt when the config has no default_prompt (SO101 configs)."
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Warm and time model-only requests, then exit without listening or ROS/CAN.",
    )
    args = parser.parse_args()
    if not (Path(args.checkpoint_dir) / "params" / "_METADATA").is_file():
        parser.error("Checkpoint params/_METADATA is missing.")
    if not Path(args.rlt_token_checkpoint).is_file():
        parser.error("Stage-1 RL token is missing.")
    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(1)
    from openpi.policies import policy_config
    from openpi.training import config

    train_config = config.get_config(args.config)
    prompt = args.prompt or train_config.data.default_prompt
    if not prompt:
        parser.error("Pass --prompt or use a model config with default_prompt.")
    original = policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        default_prompt=prompt,
        return_rlt_prefix=True,
        rlt_prefix_seq_len=768,
        rlt_token_checkpoint=args.rlt_token_checkpoint,
    )
    policy = CompactRLTPolicy(original)
    # Model-only warmup: no ROS publisher or CAN is constructed by this script.
    observation = {
        "state": np.zeros(args.state_dim, np.float32),
        "prompt": prompt,
        "images": {
            name: np.zeros((3, 224, 224), np.uint8)
            for name in ("cam_high", "cam_left_wrist", "cam_right_wrist")
        },
    }
    policy.infer(observation)
    policy.infer({"rlt_prefix_batch": [observation]})
    logging.info(
        "Compact RLT ready: encoder=%s, prompt=%s; action+z_rl, raw prefix stays local",
        policy.metadata["rlt_encoder_fingerprint"],
        prompt,
    )
    if args.benchmark:
        action_ms, stride_ms = [], []
        for _ in range(5):
            start = time.monotonic()
            response = policy.infer(observation)
            action_ms.append((time.monotonic() - start) * 1000)
            start = time.monotonic()
            policy.infer({"rlt_prefix_batch": [observation]})
            stride_ms.append((time.monotonic() - start) * 1000)
        logging.info(
            "Compact benchmark (5 warmed requests): action+z_rl=%.1fms, single_stride=%.1fms, token_bytes=%d",
            np.mean(action_ms),
            np.mean(stride_ms),
            response["rlt_z_rl"].nbytes,
        )
        return
    asyncio.run(ActionPriorityServer(policy).run("0.0.0.0", args.port))


if __name__ == "__main__":
    main()
