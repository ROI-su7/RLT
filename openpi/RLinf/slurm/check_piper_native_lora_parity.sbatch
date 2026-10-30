#!/bin/bash
# Deterministic JAX/native-PyTorch action parity and compiled latency benchmark.
#SBATCH --job-name=piper_lora_parity
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/piper_lora_parity_%j.out
#SBATCH --error=logs/piper_lora_parity_%j.err

set -euo pipefail

REPO_DIR="${RLINF_GPU_REPO_DIR:-$HOME/openpi/RLinf}"
: "${RLINF_OPENPI_SOURCE_DIR:?Set RLINF_OPENPI_SOURCE_DIR}"
: "${RLINF_PI05_JAX_DIR:?Set RLINF_PI05_JAX_DIR}"
: "${RLINF_PI05_NATIVE_TORCH_DIR:?Set RLINF_PI05_NATIVE_TORCH_DIR}"
: "${RLINF_PIPER_DATASET_DIR:?Set RLINF_PIPER_DATASET_DIR}"

cd "$REPO_DIR"
mkdir -p logs
source .venv-rlt/bin/activate
export PYTHONPATH="$REPO_DIR:$RLINF_OPENPI_SOURCE_DIR/src"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

RESULT_DIR="logs/native_lora_parity_${SLURM_JOB_ID}"
mkdir -p "$RESULT_DIR"
COMMON_ARGS=(
  --dataset-dir "$RLINF_PIPER_DATASET_DIR"
  --indices 0,159424,318848,400000
  --noise-seed 20260724
  --num-steps 10
  --trace-denoise
)

if [[ -n "${RLINF_JAX_TRACE_PATH:-}" ]]; then
  if [[ ! -s "$RLINF_JAX_TRACE_PATH" ]]; then
    echo "Precomputed JAX trace is missing: $RLINF_JAX_TRACE_PATH" >&2
    exit 1
  fi
  cp "$RLINF_JAX_TRACE_PATH" "$RESULT_DIR/jax_actions.npz"
  echo "Reusing deterministic JAX trace from $RLINF_JAX_TRACE_PATH"
else
  python toolkits/realworld_check/generate_openpi_action_parity.py \
    --runtime jax \
    --checkpoint "$RLINF_PI05_JAX_DIR" \
    --output "$RESULT_DIR/jax_actions.npz" \
    "${COMMON_ARGS[@]}"
fi

python toolkits/realworld_check/generate_openpi_action_parity.py \
  --runtime torch \
  --checkpoint "$RLINF_PI05_NATIVE_TORCH_DIR" \
  --torch-compile-mode none \
  --output "$RESULT_DIR/torch_native_eager_actions.npz" \
  "${COMMON_ARGS[@]}"

PARITY_FAILED=0
python toolkits/realworld_check/compare_openpi_action_parity.py \
  --jax-actions "$RESULT_DIR/jax_actions.npz" \
  --torch-actions "$RESULT_DIR/torch_native_eager_actions.npz" \
  --prefix-steps 5 \
  --output-json "$RESULT_DIR/action_parity.json" || PARITY_FAILED=1

python toolkits/realworld_check/compare_openpi_denoise_trace.py \
  --jax-trace "$RESULT_DIR/jax_actions.npz" \
  --torch-trace "$RESULT_DIR/torch_native_eager_actions.npz" \
  --output-json "$RESULT_DIR/denoise_trace_parity.json"

# Use default compilation for a bounded startup cost. The first timing is
# compilation; later entries are the steady-state values that matter.
python toolkits/realworld_check/generate_openpi_action_parity.py \
  --runtime torch \
  --checkpoint "$RLINF_PI05_NATIVE_TORCH_DIR" \
  --torch-compile-mode default \
  --output "$RESULT_DIR/torch_native_compiled_actions.npz" \
  "${COMMON_ARGS[@]}"

python toolkits/realworld_check/compare_openpi_action_parity.py \
  --jax-actions "$RESULT_DIR/jax_actions.npz" \
  --torch-actions "$RESULT_DIR/torch_native_compiled_actions.npz" \
  --prefix-steps 5 \
  --output-json "$RESULT_DIR/action_parity_compiled.json" || PARITY_FAILED=1

echo "RESULT_DIR=$RESULT_DIR"
if (( PARITY_FAILED )); then
  echo "FAIL: native JAX LoRA action parity gate"
  exit 1
fi
echo "PASS: native JAX LoRA action parity and compiled benchmark"
