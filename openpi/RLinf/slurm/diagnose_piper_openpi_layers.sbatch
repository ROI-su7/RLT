#!/bin/bash
# One-sample layer-boundary trace for JAX/PyTorch OpenPI parity.
#SBATCH --job-name=piper_layer_trace
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=logs/piper_layer_trace_%j.out
#SBATCH --error=logs/piper_layer_trace_%j.err

set -euo pipefail

REPO_DIR="${RLINF_GPU_REPO_DIR:-$HOME/openpi/RLinf}"
: "${RLINF_OPENPI_SOURCE_DIR:?Set RLINF_OPENPI_SOURCE_DIR}"
: "${RLINF_PI05_JAX_DIR:?Set RLINF_PI05_JAX_DIR}"
: "${RLINF_PI05_NATIVE_TORCH_DIR:?Set RLINF_PI05_NATIVE_TORCH_DIR}"
: "${RLINF_PIPER_DATASET_DIR:?Set RLINF_PIPER_DATASET_DIR}"

cd "$REPO_DIR"
source .venv-rlt/bin/activate
export PYTHONPATH="$REPO_DIR:$RLINF_OPENPI_SOURCE_DIR/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

RESULT_DIR="logs/openpi_layer_trace_${SLURM_JOB_ID}"
mkdir -p "$RESULT_DIR"
COMMON_ARGS=(
  --dataset-dir "$RLINF_PIPER_DATASET_DIR"
  --indices 318848
  --noise-seed 20260724
  --num-steps 10
  --trace-denoise
)

if [[ -n "${RLINF_JAX_LAYER_TRACE_PATH:-}" ]]; then
  if [[ ! -s "$RLINF_JAX_LAYER_TRACE_PATH" ]]; then
    echo "Precomputed JAX layer trace is missing: $RLINF_JAX_LAYER_TRACE_PATH" >&2
    exit 1
  fi
  cp "$RLINF_JAX_LAYER_TRACE_PATH" "$RESULT_DIR/jax_trace.npz"
  echo "Reusing JAX layer trace from $RLINF_JAX_LAYER_TRACE_PATH"
else
  python toolkits/realworld_check/generate_openpi_action_parity.py \
    --runtime jax \
    --checkpoint "$RLINF_PI05_JAX_DIR" \
    --output "$RESULT_DIR/jax_trace.npz" \
    "${COMMON_ARGS[@]}"
fi

python toolkits/realworld_check/generate_openpi_action_parity.py \
  --runtime torch \
  --checkpoint "$RLINF_PI05_NATIVE_TORCH_DIR" \
  --torch-compile-mode none \
  --output "$RESULT_DIR/torch_trace.npz" \
  "${COMMON_ARGS[@]}"

python toolkits/realworld_check/compare_openpi_denoise_trace.py \
  --jax-trace "$RESULT_DIR/jax_trace.npz" \
  --torch-trace "$RESULT_DIR/torch_trace.npz" \
  --output-json "$RESULT_DIR/layer_trace_parity.json"

echo "RESULT_DIR=$RESULT_DIR"
