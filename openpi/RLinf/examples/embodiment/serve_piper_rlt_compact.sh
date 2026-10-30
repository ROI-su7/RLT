#!/usr/bin/env bash
# Robot-local JAX service; loads config prompt and encodes tokens before RPC.
set -euo pipefail

REPO_DIR="${RLINF_ROBOT_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PROJECT_DIR="${RLINF_OPENPI_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
CHECKPOINT_DIR="${RLINF_PI05_JAX_DIR:-$PROJECT_DIR/checkpoints/pi05_connect_strings/pi05_connect_strings_20260923/30000}"
TOKEN_PATH="${RLINF_ROBOT_RLT_TOKEN:-$PROJECT_DIR/checkpoints/rlt_token_connect_strings_a100_20260927/step_10000/rlt_token.pt}"

export RLINF_DIR="$REPO_DIR"
export PYTHONPATH="$PROJECT_DIR/src:$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$PROJECT_DIR"
exec "$REPO_DIR/.venv-rlt/bin/python" -u "$REPO_DIR/toolkits/realworld_check/serve_piper_rlt_compact.py" \
  --config "${RLINF_PI05_JAX_CONFIG:-pi05_connect_strings}" \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --rlt-token-checkpoint "$TOKEN_PATH" \
  --port "${RLINF_JAX_REF_PORT:-6006}"
