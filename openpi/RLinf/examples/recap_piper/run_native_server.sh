#!/usr/bin/env bash
# Baseline collection server on RECAP's port; no RLT token or GPU process stop.
set -euo pipefail
RECAP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RECAP_SCRIPT_DIR/env.sh"
"$RECAP_PYTHON" -m rlinf.recap_piper.preflight --gpu-idle
test -f "$RECAP_PI05_JAX_DIR/params/_METADATA"
cd "$RECAP_NATIVE_OPENPI"
export PYTHONPATH="$RECAP_NATIVE_OPENPI/src${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Keep the terminal output and a copy for diagnosing inference latency stalls.
LOG_DIR="$RECAP_REPO_DIR/recap_runs/robot_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/server_$(date +%Y%m%d-%H%M%S).log"
echo "Server log: $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1
exec "$RECAP_PYTHON" -u scripts/serve_policy.py \
  --env ALOHA --port 6016 --default_prompt "$RECAP_TASK" \
  policy:checkpoint --policy.config "$RECAP_SOURCE_CONFIG" \
  --policy.dir "$RECAP_PI05_JAX_DIR" "$@"
