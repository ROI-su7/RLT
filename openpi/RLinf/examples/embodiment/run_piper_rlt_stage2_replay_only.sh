#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${RLINF_GPU_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${RLINF_REPLAY_LOAD_PATH:?Set RLINF_REPLAY_LOAD_PATH to persisted replay/rank_0}"
: "${RLINF_STAGE1_RLT_TOKEN:?Set RLINF_STAGE1_RLT_TOKEN to rlt_token.pt}"

if [[ ! -f "${RLINF_REPLAY_LOAD_PATH}/metadata.json" ]] ||
  [[ ! -f "${RLINF_REPLAY_LOAD_PATH}/trajectory_index.json" ]]; then
  echo "Persisted replay is incomplete: ${RLINF_REPLAY_LOAD_PATH}" >&2
  exit 1
fi

cd "$REPO_DIR"
source .venv-rlt/bin/activate
export PYTHONPATH="$REPO_DIR"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RLINF_NODE_RANK=0
# These satisfy shared config interpolation only. Replay-only training never
# constructs RolloutGroup/EnvGroup and never opens this endpoint.
export RLINF_JAX_REF_HOST="${RLINF_JAX_REF_HOST:-127.0.0.1}"
export RLINF_JAX_PREFIX_HOST="${RLINF_JAX_PREFIX_HOST:-127.0.0.1}"
export RLINF_JAX_REF_PORT="${RLINF_JAX_REF_PORT:-6006}"

LOG_DIR="$REPO_DIR/logs/$(date +'%Y%m%d-%H:%M:%S')-realworld_piper_rlt_stage2_replay_only"
mkdir -p "$LOG_DIR"
MAX_STEPS="${RLINF_REPLAY_ONLY_MAX_STEPS:-2000}"
if ! [[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "RLINF_REPLAY_ONLY_MAX_STEPS must be a positive integer: $MAX_STEPS" >&2
  exit 1
fi

exec python examples/embodiment/train_piper_rlt_stage2_replay_only.py \
  --config-path "$REPO_DIR/examples/embodiment/config" \
  --config-name realworld_piper_rlt_stage2_ac_mlp \
  "runner.logger.log_path=$LOG_DIR" \
  "runner.max_steps=$MAX_STEPS" \
  runner.save_interval=50 \
  runner.resume_dir=null \
  algorithm.replay_buffer.auto_save=false
