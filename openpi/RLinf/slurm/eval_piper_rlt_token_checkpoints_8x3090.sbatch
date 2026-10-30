#!/bin/bash
#SBATCH --job-name=piper_rlt_eval
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=logs/piper_rlt_eval_%j.out
#SBATCH --error=logs/piper_rlt_eval_%j.err

set -euo pipefail

REPO_DIR="${RLINF_GPU_REPO_DIR:-$HOME/openpi/RLinf}"
: "${RLINF_OPENPI_SOURCE_DIR:?Set RLINF_OPENPI_SOURCE_DIR}"
: "${RLINF_PI05_JAX_DIR:?Set RLINF_PI05_JAX_DIR}"
: "${RLINF_STAGE1_DATASET:?Set RLINF_STAGE1_DATASET}"
: "${RLINF_STAGE1_CHECKPOINT_ROOT:?Set RLINF_STAGE1_CHECKPOINT_ROOT}"

cd "$REPO_DIR"
mkdir -p logs
source .venv-rlt/bin/activate
export PYTHONPATH="$REPO_DIR:$RLINF_OPENPI_SOURCE_DIR/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export PYTHONUNBUFFERED=1

torchrun --standalone --nproc-per-node=8 \
  toolkits/realworld_check/eval_piper_rlt_token_checkpoints.py \
  --jax-checkpoint "$RLINF_PI05_JAX_DIR" \
  --dataset-dir "$RLINF_STAGE1_DATASET" \
  --checkpoint-root "$RLINF_STAGE1_CHECKPOINT_ROOT" \
  --num-batches "${RLINF_STAGE1_EVAL_BATCHES:-32}"
