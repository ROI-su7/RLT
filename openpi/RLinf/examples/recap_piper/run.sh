#!/usr/bin/env bash
set -euo pipefail
RECAP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RECAP_SCRIPT_DIR/env.sh"
cd "$RECAP_REPO_DIR"
STAGE="${1:-check}"
shift || true
CONFIG_DIR="$RECAP_REPO_DIR/examples/offline_rl/config"
case "$STAGE" in
  check)
    exec "$RECAP_PYTHON" -m rlinf.recap_piper.preflight "$@"
    ;;
  collect)
    source /opt/ros/noetic/setup.bash
    source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash
    export PYTHONPATH="$RECAP_REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
    exec "$RECAP_PYTHON" -m rlinf.recap_piper.collect \
      --output "$RECAP_RAW_DIR" --task "$RECAP_TASK" "$@"
    ;;
  export)
    exec "$RECAP_PYTHON" -m rlinf.recap_piper.cli export \
      --raw "$RECAP_RAW_DIR" --output "$RECAP_DATASET" "$@"
    ;;
  score)
    exec "$RECAP_PYTHON" -m rlinf.recap_piper.cli score --raw "$RECAP_RAW_DIR" "$@"
    ;;
  export-interventions)
    exec "$RECAP_PYTHON" -m rlinf.recap_piper.cli export \
      --raw "$RECAP_RAW_DIR" --output "$RECAP_INT_DATASET" --interventions-only "$@"
    ;;
  convert)
    exec "$RECAP_PYTHON" -m rlinf.recap_piper.cli convert \
      --source "$RECAP_PI05_JAX_DIR" --output "$RECAP_PI05_TORCH_DIR" \
      --config-file "$RECAP_SOURCE_CONFIG_FILE" --config-name "$RECAP_SOURCE_CONFIG" "$@"
    ;;
  returns)
    ENTRY="examples/offline_rl/advantage_labeling/recap/process/compute_returns.py"
    CONFIG=piper_recap_returns
    ;;
  value)
    ENTRY="examples/offline_rl/advantage_labeling/recap/train_value.py"
    CONFIG=piper_recap_value
    ;;
  advantages)
    ENTRY="examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py"
    CONFIG=piper_recap_advantages
    ;;
  cfg)
    ENTRY="examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py"
    CONFIG=piper_recap_cfg
    ;;
  sftint)
    ENTRY="examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py"
    CONFIG=piper_recap_sft_int
    ;;
  serve)
    "$RECAP_PYTHON" -m rlinf.recap_piper.preflight --stage serve --gpu-idle
    exec "$RECAP_PYTHON" -m rlinf.recap_piper.cli serve "$@"
    ;;
  *)
    echo "Usage: bash examples/recap_piper/run.sh {check|collect|score|export|export-interventions|convert|returns|value|advantages|cfg|sftint|serve} [args...]" >&2
    exit 2
    ;;
esac
CMD=("$RECAP_PYTHON" "$ENTRY" --config-path "$CONFIG_DIR" --config-name "$CONFIG")
if [[ "$STAGE" == value || "$STAGE" == cfg || "$STAGE" == sftint ]]; then
  CMD+=("runner.logger.log_path=$RECAP_RUN_DIR/$STAGE")
fi
# --dry-run composes/resolves the exact config without initializing workers/GPU.
if [[ "${1:-}" == --dry-run ]]; then
  shift
  # Keep Hydra overrides contiguous; argparse rejects positionals after options.
  exec "${CMD[@]}" "$@" --cfg job --resolve
fi
"$RECAP_PYTHON" -m rlinf.recap_piper.preflight --stage "$STAGE"
if [[ "$STAGE" != returns ]]; then
  "$RECAP_PYTHON" -m rlinf.recap_piper.preflight --gpu-idle
fi
if [[ "$STAGE" == value || "$STAGE" == cfg || "$STAGE" == sftint ]]; then
  if ! "$(dirname "$RECAP_PYTHON")/ray" status --address="$RECAP_RAY_ADDRESS" >/dev/null 2>&1; then
    echo "RECAP Ray head is not running at $RECAP_RAY_ADDRESS. See README_PIPER_RECAP.md; this script never stops/reuses the RLT cluster." >&2
    exit 1
  fi
fi
mkdir -p "$RECAP_RUN_DIR/$STAGE"
"${CMD[@]}" "$@" 2>&1 | tee "$RECAP_RUN_DIR/$STAGE/run.log"
