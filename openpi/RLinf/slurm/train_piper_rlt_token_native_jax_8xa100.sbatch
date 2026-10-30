#!/bin/bash
# Native-JAX frozen VLA + PyTorch-only RL-token Stage 1 (8x A100-40G).
#SBATCH --job-name=piper_rlt_token_a100
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=24:00:00
#SBATCH --output=logs/piper_rlt_token_a100_%j.out
#SBATCH --error=logs/piper_rlt_token_a100_%j.err

set -euo pipefail
REPO_DIR="${RLINF_GPU_REPO_DIR:-$HOME/FluxVla/openpi/RLinf}"
RLINF_OPENPI_SOURCE_DIR="${RLINF_OPENPI_SOURCE_DIR:-$HOME/FluxVla/openpi}"
RLINF_PI05_JAX_DIR="${RLINF_PI05_JAX_DIR:-$HOME/FluxVla/openpi/checkpoints/pi05_connect_strings/pi05_connect_strings_20260923/30000}"
: "${RLINF_PIPER_DATASET_DIR:?Set RLINF_PIPER_DATASET_DIR}"
: "${RLINF_STAGE1_OUTPUT_DIR:?Set RLINF_STAGE1_OUTPUT_DIR}"

cd "$REPO_DIR"
source .venv-rlt/bin/activate
export PYTHONPATH="$REPO_DIR:$RLINF_OPENPI_SOURCE_DIR/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONUNBUFFERED=1

torchrun --standalone --nproc-per-node=8 \
  toolkits/realworld_check/train_piper_rlt_token_native_jax.py \
  --jax-checkpoint "$RLINF_PI05_JAX_DIR" \
  --dataset-dir "$RLINF_PIPER_DATASET_DIR" \
  --output-dir "$RLINF_STAGE1_OUTPUT_DIR" \
  --config-name "${RLINF_PI05_JAX_CONFIG:-pi05_connect_strings}" \
  --steps "${RLINF_STAGE1_STEPS:-10000}" \
  --accumulate "${RLINF_STAGE1_ACCUMULATE:-8}"
