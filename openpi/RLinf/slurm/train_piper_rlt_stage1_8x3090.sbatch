#!/bin/bash
# Official RLT Stage 1: jointly optimize VLA flow matching and RL-token loss
# from the validated LoRA-merged pi05_wire task checkpoint.
#SBATCH --job-name=piper_rlt_s1
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=72:00:00
#SBATCH --output=logs/piper_rlt_s1_%j.out
#SBATCH --error=logs/piper_rlt_s1_%j.err

set -euo pipefail

REPO_DIR="${RLINF_GPU_REPO_DIR:-$HOME/openpi/RLinf}"
: "${RLINF_PI05_TORCH_DIR:?Set RLINF_PI05_TORCH_DIR to the newly validated LoRA-merged conversion}"
: "${RLINF_PI05_NORM_STATS:?Set RLINF_PI05_NORM_STATS to its norm_stats.json}"
: "${RLINF_PIPER_DATASET_DIR:?Set RLINF_PIPER_DATASET_DIR to the 500-episode LeRobot dataset}"

cd "$REPO_DIR"
mkdir -p logs
source .venv-rlt/bin/activate

export PYTHONPATH="$REPO_DIR"
export EMBODIED_PATH="$REPO_DIR/examples/sft"
export RLINF_GPU_REPO_DIR="$REPO_DIR"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

python - <<'PY'
import os
from pathlib import Path

from rlinf.models.embodiment.openpi import validate_piper_converted_checkpoint

checkpoint = os.environ["RLINF_PI05_TORCH_DIR"]
norm_stats = Path(os.environ["RLINF_PI05_NORM_STATS"])
dataset = Path(os.environ["RLINF_PIPER_DATASET_DIR"])
validate_piper_converted_checkpoint(checkpoint, verify_model_hash=True)
if not norm_stats.is_file():
    raise FileNotFoundError(f"Missing norm stats: {norm_stats}")
if not dataset.is_dir():
    raise FileNotFoundError(f"Missing Piper dataset: {dataset}")
print("PASS: LoRA merge provenance, norm stats, and Stage-1 dataset")
PY

exec bash examples/sft/run_vla_sft.sh \
  realworld_piper_rlt_stage1_sft_openpi_pi05
