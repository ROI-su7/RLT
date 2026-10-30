#!/usr/bin/env bash
# Source this in a new terminal. No installation, Ray start/stop, or robot action.
RECAP_REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export REPO_PATH="$RECAP_REPO_DIR"
export PYTHONPATH="$RECAP_REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export RECAP_PYTHON="${RECAP_PYTHON:-$RECAP_REPO_DIR/.venv-rlt/bin/python}"
export RECAP_NATIVE_OPENPI="${RECAP_NATIVE_OPENPI:-/home/agilex/FluxVla/openpi}"
export RECAP_SOURCE_CONFIG_FILE="${RECAP_SOURCE_CONFIG_FILE:-$RECAP_NATIVE_OPENPI/src/openpi/training/config.py}"
export RECAP_SOURCE_CONFIG="${RECAP_SOURCE_CONFIG:-pi05_connect_strings}"
export RECAP_PI05_JAX_DIR="${RECAP_PI05_JAX_DIR:-$RECAP_NATIVE_OPENPI/checkpoints/pi05_connect_strings/pi05_connect_strings_20260923/30000}"
export RECAP_RAW_DIR="${RECAP_RAW_DIR:-$RECAP_REPO_DIR/recap_data/raw}"
export RECAP_DATASET="${RECAP_DATASET:-$RECAP_REPO_DIR/recap_data/lerobot_v1}"
# Isolated copy of the source demonstrations, and human segments of successful
# takeover episodes for the SFT+Int baseline (README section 6 / 8.5).
export RECAP_SFT_DATASET="${RECAP_SFT_DATASET:-$RECAP_REPO_DIR/recap_data/sft_all_wire_data}"
export RECAP_INT_DATASET="${RECAP_INT_DATASET:-$RECAP_REPO_DIR/recap_data/interventions_v1}"
export RECAP_PI05_TORCH_DIR="${RECAP_PI05_TORCH_DIR:-$RECAP_REPO_DIR/recap_artifacts/pi05_base}"
export RECAP_NORM_STATS="${RECAP_NORM_STATS:-$RECAP_PI05_JAX_DIR/assets/all_wire_data/norm_stats.json}"
export RECAP_SIGLIP_DIR="${RECAP_SIGLIP_DIR:-$RECAP_REPO_DIR/recap_artifacts/siglip2-so400m-patch14-224}"
export RECAP_GEMMA_DIR="${RECAP_GEMMA_DIR:-$RECAP_REPO_DIR/recap_artifacts/gemma-3-270m}"
export RECAP_RUN_DIR="${RECAP_RUN_DIR:-$RECAP_REPO_DIR/recap_runs/$(date +%Y%m%d-%H%M%S)}"
export RECAP_RAY_ADDRESS="${RECAP_RAY_ADDRESS:-127.0.0.1:6385}"
export HF_DATASETS_CACHE="${RECAP_DATASETS_CACHE:-$RECAP_REPO_DIR/recap_artifacts/datasets_cache}"
# Do not inherit the RLT cluster, resume path, token, or replay configuration.
export RAY_ADDRESS="$RECAP_RAY_ADDRESS"
export TOKENIZERS_PARALLELISM=false
if [[ -z "${RECAP_TASK:-}" ]]; then
  RECAP_TASK="$("$RECAP_PYTHON" -m rlinf.recap_piper.cli task \
    --config-file "$RECAP_SOURCE_CONFIG_FILE" --config-name "$RECAP_SOURCE_CONFIG")" || return 1
  export RECAP_TASK
fi
