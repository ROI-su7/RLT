#!/usr/bin/env bash
# Run on the GPU cluster host after the Piper node has joined the Ray head.
set -euo pipefail

REPO_DIR="${RLINF_GPU_REPO_DIR:-$HOME/openpi/RLinf}"
HEAD_IP="${RLINF_GPU_HEAD_IP:-192.168.85.129}"
: "${RLINF_STAGE1_RLT_TOKEN:?Set RLINF_STAGE1_RLT_TOKEN to the selected native-JAX Stage-1 rlt_token.pt}"
: "${RLINF_JAX_REF_HOST:?Set RLINF_JAX_REF_HOST to the robot-side native JAX server host}"
RLINF_JAX_REF_PORT="${RLINF_JAX_REF_PORT:-6006}"
STAGE1_RLT_TOKEN="$RLINF_STAGE1_RLT_TOKEN"
STAGE2_CONFIG="${RLINF_PIPER_STAGE2_CONFIG:-realworld_piper_rlt_stage2_ac_mlp_dual}"

cd "$REPO_DIR"
source .venv-rlt/bin/activate
export PYTHONPATH="$REPO_DIR"
export RLINF_GPU_REPO_DIR="$REPO_DIR"
export RLINF_STAGE1_RLT_TOKEN="$STAGE1_RLT_TOKEN"
export RLINF_JAX_REF_HOST
export RLINF_JAX_REF_PORT
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export RLINF_NODE_RANK=0
export RLINF_CODE_WORKING_DIR=auto

if [[ ! -s "$STAGE1_RLT_TOKEN" ]]; then
  echo "Stage-1 RLT token checkpoint is missing: $STAGE1_RLT_TOKEN" >&2
  exit 1
fi

if [[ -n "${RLINF_REPLAY_LOAD_PATH:-}" ]]; then
  if [[ "${RLINF_REPLAY_LOAD_PATH}" != /* ]]; then
    echo "RLINF_REPLAY_LOAD_PATH must be an absolute path." >&2
    exit 1
  fi
  if [[ ! -f "${RLINF_REPLAY_LOAD_PATH}/metadata.json" ]] ||
    [[ ! -f "${RLINF_REPLAY_LOAD_PATH}/trajectory_index.json" ]]; then
    echo "Persisted replay is incomplete: ${RLINF_REPLAY_LOAD_PATH}" >&2
    exit 1
  fi
  echo "Preloading Stage-2 replay: ${RLINF_REPLAY_LOAD_PATH}"
fi

echo "Using native-JAX Stage-1 RLT token: $STAGE1_RLT_TOKEN"
echo "Using native-JAX action/prefix server: ${RLINF_JAX_REF_HOST}:${RLINF_JAX_REF_PORT}"
echo "Using Stage-2 config: $STAGE2_CONFIG"

HYDRA_OVERRIDES=()
if [[ -n "${RLINF_STAGE2_RESUME_DIR:-}" ]]; then
  if [[ "${RLINF_STAGE2_RESUME_DIR}" != /* ]]; then
    echo "RLINF_STAGE2_RESUME_DIR must be an absolute path." >&2
    exit 1
  fi
  if [[ ! -d "${RLINF_STAGE2_RESUME_DIR}/actor" ]]; then
    echo "Stage-2 resume checkpoint is missing actor/: ${RLINF_STAGE2_RESUME_DIR}" >&2
    exit 1
  fi
  if [[ ! -f "${RLINF_STAGE2_RESUME_DIR}/actor/sac_components/replay_buffer/rank_0/metadata.json" ]]; then
    echo "Stage-2 resume checkpoint is missing its replay snapshot: ${RLINF_STAGE2_RESUME_DIR}" >&2
    exit 1
  fi
  HYDRA_OVERRIDES+=("runner.resume_dir=${RLINF_STAGE2_RESUME_DIR}")
  echo "Resuming the complete Stage 2 checkpoint from ${RLINF_STAGE2_RESUME_DIR}"
fi

# Cluster membership is handled by Ray placement: the Piper-pinned EnvWorker
# remains pending until the robot node joins. Avoid short-lived ray.init()
# probes, whose teardown can strand this site's older Slurm batch step.
echo "Starting trainer preflight; Ray will wait for the Piper node as needed."

python - <<'PY'
import transformers
from transformers.models.siglip import check
assert transformers.__version__ == "4.53.2", transformers.__version__
assert check.check_whether_transformers_replace_is_installed_correctly()
print("GPU OpenPI/Transformers preflight passed.")
PY

echo "Starting RLinf Stage-2 trainer."
exec bash examples/embodiment/run_realworld_async.sh \
  "$STAGE2_CONFIG" \
  "${HYDRA_OVERRIDES[@]}"
