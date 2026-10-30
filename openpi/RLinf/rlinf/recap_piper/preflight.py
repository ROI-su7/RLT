"""Read-only RECAP checks. Never start Ray, contact ROS, or allocate CUDA."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from rlinf.recap_piper.data import CAMERAS
from rlinf.recap_piper.source import native_config


def check_dataset(root: Path, *, stage: str = "dataset", sft: bool = False) -> dict:
    """Validate physical Piper frames and exact sidecar key coverage.

    ``sft`` datasets (source demonstrations) carry no ``is_success`` column;
    every episode counts as successful.
    """
    import pyarrow.parquet as pq

    root = Path(root)
    if (root / "meta/recap_export_incomplete.json").exists():
        raise ValueError(
            "LeRobot export was interrupted; choose a fresh snapshot directory."
        )
    info = json.loads((root / "meta/info.json").read_text())
    features = info["features"]
    for key in ("observation.state", "action"):
        if list(features[key]["shape"]) != [14]:
            raise ValueError(f"{key} must be physical 14-D, got {features[key]}")
    for camera in CAMERAS:
        if f"observation.images.{camera}" not in features:
            raise ValueError(f"Missing camera: {camera}")
    seen, successes, episodes = set(), 0, set()
    for path in sorted((root / "data").rglob("*.parquet")):
        columns = ["episode_index", "frame_index", "action", "observation.state"]
        table = pq.read_table(path, columns=columns + ([] if sft else ["is_success"]))
        rows = table.to_pylist()
        for row in rows:
            key = int(row["episode_index"]), int(row["frame_index"])
            if key in seen:
                raise ValueError(f"Duplicate frame: {key}")
            seen.add(key)
            episodes.add(key[0])
            for name in ("action", "observation.state"):
                value = np.asarray(row[name])
                if value.shape != (14,) or not np.isfinite(value).all():
                    raise ValueError(f"Invalid {name} at {key}")
            if sft:
                continue
            flag = np.asarray(row["is_success"])
            if flag.size != 1 or flag.reshape(-1)[0] not in (True, False, 0, 1):
                raise ValueError(f"Invalid outcome at {key}")
            successes += int(bool(flag.reshape(-1)[0]))
    if len(seen) != info["total_frames"] or not seen:
        raise ValueError(
            "Frame count disagrees with dataset metadata, or dataset is empty."
        )
    for name, column in (
        ("returns_piper_recap_v1.parquet", "return"),
        ("advantages_piper_recap_adv_v1.parquet", "advantage"),
    ):
        required = (
            name.startswith("returns") and stage in {"value", "advantages", "cfg"}
        ) or (name.startswith("advantages") and stage == "cfg")
        if not required:
            continue
        sidecar = pq.read_table(root / "meta" / name).to_pydict()
        keys = list(zip(sidecar["episode_index"], sidecar["frame_index"], strict=True))
        if set(keys) != seen or len(keys) != len(seen):
            raise ValueError(f"{name} must cover every dataset frame exactly once.")
        if column == "return" and not np.isfinite(sidecar[column]).all():
            raise ValueError("Non-finite returns.")
    return {
        "frames": len(seen),
        "episodes": len(episodes),
        "positive_terminal_frames": len(episodes) if sft else successes,
    }


def check_assets(stage: str) -> list[str]:
    """List missing local assets, without triggering downloads or GPU loads."""
    errors = []
    if stage in {"value", "advantages"}:
        for variable in ("RECAP_SIGLIP_DIR", "RECAP_GEMMA_DIR"):
            folder = Path(os.environ.get(variable, "/missing"))
            if not (folder / "config.json").is_file() or not list(
                folder.glob("*.safetensors")
            ):
                errors.append(
                    f"{variable}: missing pretrained config/weights at {folder}"
                )
        gemma = Path(os.environ.get("RECAP_GEMMA_DIR", "/missing"))
        if not any(
            (gemma / name).is_file() for name in ("tokenizer.json", "tokenizer.model")
        ):
            errors.append(f"Missing Gemma tokenizer at {gemma}")
    if stage == "advantages":
        folder = Path(os.environ.get("RECAP_VALUE_CHECKPOINT", "/missing"))
        if not any(
            (folder / suffix).is_file()
            for suffix in (
                "full_weights.pt",
                "model_state_dict/full_weights.pt",
                "actor/model_state_dict/full_weights.pt",
            )
        ):
            errors.append(
                f"RECAP_VALUE_CHECKPOINT: no full value checkpoint at {folder}"
            )
    if stage in {"cfg", "sftint", "serve"}:
        folder = Path(os.environ.get("RECAP_PI05_TORCH_DIR", "/missing"))
        if (
            not (folder / "model.safetensors").is_file()
            or not (folder / "config.json").is_file()
        ):
            errors.append(
                f"RECAP_PI05_TORCH_DIR: run the isolated conversion first ({folder})"
            )
        if not Path(os.environ.get("RECAP_NORM_STATS", "/missing")).is_file():
            errors.append(
                "RECAP_NORM_STATS must point to the source checkpoint's norm_stats.json"
            )
    return errors


def gpu_busy_errors() -> list[str]:
    """Inspect driver-reported compute use without creating a CUDA context."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [f"Cannot inspect the GPU before model startup: {exc}"]
    if result.returncode:
        return ["nvidia-smi failed; model startup was not attempted."]
    for row in result.stdout.splitlines():
        columns = row.split(",")
        if (
            len(columns) == 2
            and columns[1].strip().isdigit()
            and int(columns[1]) > 1024
        ):
            return [
                f"GPU model process still active ({row.strip()}); finish RLT/stop the policy service before RECAP model startup."
            ]
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "cpu",
            "dataset",
            "returns",
            "value",
            "advantages",
            "cfg",
            "sftint",
            "serve",
        ),
        default="cpu",
    )
    parser.add_argument(
        "--gpu-idle",
        action="store_true",
        help="Read nvidia-smi only; reject an active model before startup",
    )
    args = parser.parse_args()
    errors = []
    for name in (
        "numpy",
        "PIL",
        "torch",
        "lerobot",
        "openpi",
        "pyarrow",
        "hydra",
        "transformers",
        "safetensors",
    ):
        if importlib.util.find_spec(name) is None:
            errors.append(f"Missing Python dependency: {name}")
    source_file = os.environ.get("RECAP_SOURCE_CONFIG_FILE")
    if source_file:
        try:
            prompt, architecture = native_config(
                Path(source_file), os.environ["RECAP_SOURCE_CONFIG"]
            )
            print(
                json.dumps(
                    {"source_task": prompt, "source_architecture": architecture},
                    ensure_ascii=False,
                )
            )
            if os.environ.get("RECAP_TASK") != prompt:
                errors.append(
                    "RECAP_TASK differs from the selected native source config prompt."
                )
        except (OSError, ValueError, KeyError) as exc:
            errors.append(str(exc))
    if args.stage == "sftint":
        for variable, sft in (
            ("RECAP_SFT_DATASET", True),
            ("RECAP_INT_DATASET", False),
        ):
            try:
                root = Path(os.environ.get(variable, "/missing"))
                print(json.dumps({variable: check_dataset(root, sft=sft)}))
            except (OSError, ValueError, KeyError) as exc:
                errors.append(f"{variable} check: {exc}")
    elif args.stage not in {"cpu", "serve"}:
        try:
            report = check_dataset(
                Path(os.environ.get("RECAP_DATASET", "/missing")), stage=args.stage
            )
            print(json.dumps(report))
        except (OSError, ValueError, KeyError) as exc:
            errors.append(f"Dataset check: {exc}")
    errors.extend(check_assets(args.stage))
    if args.gpu_idle:
        errors.extend(gpu_busy_errors())
    if errors:
        for error in errors:
            print("FAIL " + error)
        raise SystemExit(1)
    print(f"PASS Piper RECAP {args.stage} checks (no ROS/CUDA/Ray started)")


if __name__ == "__main__":
    main()
