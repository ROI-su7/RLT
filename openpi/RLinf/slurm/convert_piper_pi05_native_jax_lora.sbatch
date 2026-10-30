#!/bin/bash
# Convert the task-adapted JAX pi0.5 checkpoint without merging LoRA into W.
#SBATCH --job-name=piper_lora_convert
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=220G
#SBATCH --time=06:00:00
#SBATCH --output=logs/piper_lora_convert_%j.out
#SBATCH --error=logs/piper_lora_convert_%j.err

set -euo pipefail

REPO_DIR="${RLINF_GPU_REPO_DIR:-$HOME/openpi/RLinf}"
: "${RLINF_PI05_JAX_DIR:?Set RLINF_PI05_JAX_DIR}"
: "${RLINF_PI05_NATIVE_TORCH_DIR:?Set RLINF_PI05_NATIVE_TORCH_DIR}"

cd "$REPO_DIR"
mkdir -p logs
source .venv-rlt/bin/activate

export PYTHONPATH="$REPO_DIR"
export PYTHONUNBUFFERED=1
# Conversion restores NumPy arrays and does not need to reserve GPU memory.
export JAX_PLATFORMS=cpu

python rlinf/utils/ckpt_convertor/convert_openpi_jax_to_python.py \
  --checkpoint-dir "$RLINF_PI05_JAX_DIR" \
  --config-name pi05_wire \
  --output-path "$RLINF_PI05_NATIVE_TORCH_DIR" \
  --precision bfloat16 \
  --lora-mode native

python - <<'PY'
import os
from rlinf.models.embodiment.openpi import validate_piper_converted_checkpoint

validate_piper_converted_checkpoint(
    os.environ["RLINF_PI05_NATIVE_TORCH_DIR"], verify_model_hash=True
)
print("PASS: native JAX LoRA checkpoint provenance and model hash")
PY
