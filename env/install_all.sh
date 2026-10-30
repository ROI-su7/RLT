#!/usr/bin/env bash
# One-click environment setup for RLT.
#
#   bash env/install_all.sh              # Cobot Magic (Ubuntu 20.04 + ROS Noetic): steps 01-05
#   bash env/install_all.sh --so101      # dual SO101 host (no ROS, no sudo): steps 03-06
#   bash env/install_all.sh --no-ros     # Python environments only (training server): 03-05
#   bash env/install_all.sh --only 03    # run a single step
#
# Steps (env/steps/):
#   01_system_deps.sh   apt packages (sudo)
#   02_ros.sh           ROS Noetic + catkin workspaces (Piper, cameras) (sudo for apt)
#   03_openpi_env.sh    openpi/.venv  (pi0.5 fine-tuning, norm stats, data conversion)
#   04_rlt_env.sh       openpi/RLinf/.venv-rlt (native JAX pi0.5 server, Stage 1, Stage 2)
#   05_base_assets.sh   public base weights (pi0.5 base checkpoint, PaliGemma tokenizer)
#   06_so101_env.sh     SO101 bus deps (Feetech SDK, pyserial) + so101/.venv probe env
# Datasets and fine-tuned weights are not part of this repository.
set -euo pipefail
RLT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RLT_ROOT
STEPS=(01_system_deps 02_ros 03_openpi_env 04_rlt_env 05_base_assets 06_so101_env)
SKIP_ROS=0
SO101=0
ONLY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-ros) SKIP_ROS=1 ;;
    --so101) SKIP_ROS=1; SO101=1 ;;
    --only) ONLY="$2"; shift ;;
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "Unknown option $1" >&2; exit 2 ;;
  esac
  shift
done

cd "$RLT_ROOT"
if [[ -d .git ]]; then
  git submodule update --init --recursive
fi

for step in "${STEPS[@]}"; do
  if [[ -n "$ONLY" && "$step" != "$ONLY"* ]]; then continue; fi
  if [[ $SKIP_ROS == 1 && ( "$step" == 01_system_deps || "$step" == 02_ros ) ]]; then
    echo "[RLT] skip $step (no ROS)"; continue
  fi
  if [[ -z "$ONLY" && $SO101 == 0 && "$step" == 06_so101_env ]]; then continue; fi
  echo "[RLT] ===== $step ====="
  bash "$RLT_ROOT/env/steps/$step.sh"
done
echo "[RLT] Done. Verify with: bash env/verify.sh"
