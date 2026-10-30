#!/usr/bin/env bash
# Native-JAX pi0.5 service for dual-SO101 RLT: one request returns the reference
# action chunk and the Stage-1 z_rl. Same server as Piper, 12-D state.
set -euo pipefail

REPO_DIR="${RLINF_ROBOT_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PROJECT_DIR="${RLINF_OPENPI_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
: "${RLINF_PI05_JAX_DIR:?Set RLINF_PI05_JAX_DIR to the SO101 pi0.5 checkpoint step dir (e.g. checkpoints/pi05_so101_dual_lora/<exp>/<step>)}"
: "${RLINF_ROBOT_RLT_TOKEN:?Set RLINF_ROBOT_RLT_TOKEN to the SO101 Stage-1 rlt_token.pt}"
: "${RLINF_SO101_TASK:?Set RLINF_SO101_TASK to the task prompt used when recording}"

export RLINF_DIR="$REPO_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$REPO_DIR"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$PROJECT_DIR"
exec "$REPO_DIR/.venv-rlt/bin/python" -u "$REPO_DIR/toolkits/realworld_check/serve_piper_rlt_compact.py" \
  --config "${RLINF_PI05_JAX_CONFIG:-pi05_so101_dual_lora}" \
  --checkpoint-dir "$RLINF_PI05_JAX_DIR" \
  --rlt-token-checkpoint "$RLINF_ROBOT_RLT_TOKEN" \
  --state-dim 12 \
  --prompt "$RLINF_SO101_TASK" \
  --port "${RLINF_JAX_REF_PORT:-6006}"
