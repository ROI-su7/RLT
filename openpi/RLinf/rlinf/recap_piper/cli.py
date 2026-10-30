"""CPU utilities, source conversion and separate websocket serving for RECAP."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from rlinf.recap_piper.source import native_config

# Training config used for each recorded policy mode.
SERVE_CONFIGS = {"recap_cfg": "piper_recap_cfg", "sft_int": "piper_recap_sft_int"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    task = commands.add_parser("task")
    task.add_argument("--config-file", type=Path, required=True)
    task.add_argument("--config-name", required=True)
    export = commands.add_parser("export")
    export.add_argument("--raw", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--repo-id", default="local/piper_recap")
    export.add_argument(
        "--interventions-only",
        action="store_true",
        help="Export only human segments of successful episodes (SFT+Int baseline)",
    )
    export.add_argument("--min-segment-frames", type=int, default=10)
    export.add_argument(
        "--png",
        action="store_true",
        help="Store per-frame PNG images instead of AV1 videos (much larger)",
    )
    score = commands.add_parser("score", help="Per-insertion success report")
    score.add_argument("--raw", type=Path, required=True)
    convert = commands.add_parser("convert")
    convert.add_argument("--source", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--config-file", type=Path, required=True)
    convert.add_argument("--config-name", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="RECAP global_step_N/actor or its parent global_step_N",
    )
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=6016)
    serve.add_argument("--guidance-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.command == "task":
        print(native_config(args.config_file, args.config_name)[0])
    elif args.command == "export":
        from rlinf.recap_piper.data import export_lerobot

        report = export_lerobot(
            args.raw,
            args.output,
            args.repo_id,
            interventions_only=args.interventions_only,
            min_segment_frames=args.min_segment_frames,
            use_videos=not args.png,
        )
        print(json.dumps(report, indent=2))
    elif args.command == "score":
        from rlinf.recap_piper.scoring import score_report

        print(json.dumps(score_report(args.raw), indent=2, ensure_ascii=False))
    elif args.command == "convert":
        if args.output.exists():
            raise FileExistsError(f"Conversion output already exists: {args.output}")
        if not (args.source / "params").is_dir():
            raise FileNotFoundError(f"Missing source params: {args.source}")
        prompt, architecture = native_config(args.config_file, args.config_name)
        # Conversion is CPU-only; no JAX allocations on the RLT test GPU.
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from openpi.models.pi0_config import Pi0Config

        from rlinf.recap_piper.data import atomic_json
        from rlinf.utils.ckpt_convertor.convert_openpi_jax_to_python import (
            convert_pi0_checkpoint,
        )

        model_config = Pi0Config(**architecture)
        native = any(
            "lora" in str(architecture.get(key, ""))
            for key in ("paligemma_variant", "action_expert_variant")
        )
        convert_pi0_checkpoint(
            str(args.source),
            "bfloat16",
            str(args.output),
            model_config,
            lora_mode="native" if native else "merge",
        )
        atomic_json(
            args.output / "piper_recap_source.json",
            {
                "task": prompt,
                "source_config": args.config_name,
                "source": str(args.source.resolve()),
                "architecture": architecture,
            },
        )
    elif args.command == "serve":
        serve_checkpoint(args)


def serve_checkpoint(args) -> None:
    """Serve only RECAP weights on its own port; no ROS or Ray is opened."""
    import hydra
    import torch
    from omegaconf import OmegaConf
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    from rlinf.recap_piper.policy import (
        PiperCFGPolicy,
        load_piper_cfg_model,
        validate_policy_provenance,
    )

    if args.port == 6006:
        raise ValueError(
            "Port 6006 is reserved for the current RLT/JAX service; use 6016."
        )
    checkpoint = args.checkpoint.resolve()
    weights = checkpoint / "actor/model_state_dict/full_weights.pt"
    if not weights.is_file():
        weights = checkpoint / "model_state_dict/full_weights.pt"
    if not weights.is_file():
        raise FileNotFoundError(f"No complete RECAP FSDP weights in {checkpoint}")
    root = Path(__file__).resolve().parents[2]
    recorded = json.loads(
        (weights.parent.parent / "piper_recap_policy.json").read_text()
    )
    mode = recorded.get("policy_mode", "recap_cfg")
    if mode not in SERVE_CONFIGS:
        raise ValueError(f"Unknown Piper policy mode in checkpoint: {mode}")
    if mode == "sft_int" and args.guidance_scale != 1.0:
        raise ValueError(
            "The SFT+Int baseline has no CFG branch; omit --guidance-scale."
        )
    with hydra.initialize_config_dir(
        config_dir=str(root / "examples/offline_rl/config"), version_base="1.1"
    ):
        cfg = hydra.compose(config_name=SERVE_CONFIGS[mode])
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    validate_policy_provenance(weights.parent.parent, cfg.actor.model)
    model = load_piper_cfg_model(cfg.actor.model)
    # Require every key: an RLT Actor checkpoint cannot be served accidentally.
    model.load_state_dict(
        torch.load(weights, map_location="cpu", weights_only=True), strict=True
    )
    object.__setattr__(model.config, "cfgrl_guidance_scale", args.guidance_scale)
    if mode == "sft_int":
        # Trained only on the unconditional prompt: sample that branch alone.
        object.__setattr__(model.config, "guidance_type", "no_guide")
    policy = PiperCFGPolicy(model, task=os.environ["RECAP_TASK"])
    metadata = {
        "scheme": "piper_recap",
        "action_horizon": 50,
        "action_dim": 14,
        "task": policy.task,
        "policy_mode": mode,
        "guidance_scale": None if mode == "sft_int" else args.guidance_scale,
        "checkpoint": str(checkpoint),
    }
    WebsocketPolicyServer(
        policy, host=args.host, port=args.port, metadata=metadata
    ).serve_forever()


if __name__ == "__main__":
    main()
