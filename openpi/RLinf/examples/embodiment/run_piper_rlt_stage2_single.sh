#!/usr/bin/env bash
# Start the single-machine Piper RLT Stage-2 stack after ROS/camera/Piper are up.
# The script sends no robot command before the read-only preflight passes.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RLT_ROOT="${RLT_ROOT:-$(cd "$REPO_DIR/../.." && pwd)}"
GPU_NODE_IP="${GPU_NODE_IP:-$(hostname -I | awk '{print $1}')}"
KEYBOARD_DEVICE="${RLINF_KEYBOARD_DEVICE:-}"
PIPER_WS="${PIPER_WS:-$RLT_ROOT/cobot_magic/Piper_ros_private-ros-noetic}"
: "${RLINF_STAGE1_RLT_TOKEN:?Set RLINF_STAGE1_RLT_TOKEN to the native-JAX Stage-1 rlt_token.pt}"
test -s "$RLINF_STAGE1_RLT_TOKEN"

unset PYTHONPATH
source /opt/ros/noetic/setup.bash
source "$PIPER_WS/devel/setup.bash"
source "$REPO_DIR/.venv-rlt/bin/activate"

cd "$REPO_DIR"
export RLINF_ROS_PYTHONPATH="$PYTHONPATH"
export PYTHONPATH="$REPO_DIR:$RLINF_ROS_PYTHONPATH"
export ROS_MASTER_URI="http://${GPU_NODE_IP}:11311"
export ROS_IP="$GPU_NODE_IP"
export CUDA_VISIBLE_DEVICES=0
export RLINF_NODE_RANK=0
if [[ -n "$KEYBOARD_DEVICE" && -e "$KEYBOARD_DEVICE" ]]; then
  export RLINF_KEYBOARD_DEVICE="$KEYBOARD_DEVICE"
elif [[ -n "$KEYBOARD_DEVICE" ]]; then
  echo "[Stage2] Keyboard override $KEYBOARD_DEVICE no longer exists; auto-detecting an evdev keyboard."
  unset RLINF_KEYBOARD_DEVICE
else
  unset RLINF_KEYBOARD_DEVICE
fi
export RLINF_PRESERVE_ROS_MASTER=1
export RLINF_JAX_REF_HOST="${RLINF_JAX_REF_HOST:-$GPU_NODE_IP}"
export RLINF_JAX_REF_PORT="${RLINF_JAX_REF_PORT:-6006}"
export RLINF_STAGE1_RLT_TOKEN

echo "[Stage2] ROS master: $ROS_MASTER_URI"
echo "[Stage2] Keyboard:   ${RLINF_KEYBOARD_DEVICE:-auto-detect}"

HYDRA_OVERRIDES=()
if [[ -n "${RLINF_REPLAY_LOAD_PATH:-}" ]]; then
  if [[ "${RLINF_REPLAY_LOAD_PATH}" != /* ]] ||
    [[ ! -f "${RLINF_REPLAY_LOAD_PATH}/metadata.json" ]] ||
    [[ ! -f "${RLINF_REPLAY_LOAD_PATH}/trajectory_index.json" ]]; then
    echo "RLINF_REPLAY_LOAD_PATH must be an absolute, complete replay/rank_0 directory." >&2
    exit 1
  fi
  echo "[Stage2] Loading replay: ${RLINF_REPLAY_LOAD_PATH}"
else
  echo "[Stage2] Replay:     empty (fresh collection)"
fi
# Workers resolve ${oc.env:...} in the Ray raylet's environment, captured at
# `ray start`. A head started from a shell that still exported an old
# RLINF_REPLAY_LOAD_PATH made a fresh run load (or fail on) that old replay.
# Pass the path explicitly so only this shell decides.
HYDRA_OVERRIDES+=("algorithm.replay_buffer.load_path=${RLINF_REPLAY_LOAD_PATH:-null}")
if [[ -n "${RLINF_STAGE2_RESUME_DIR:-}" ]]; then
  if [[ "${RLINF_STAGE2_RESUME_DIR}" != /* ]] ||
    [[ ! -f "${RLINF_STAGE2_RESUME_DIR}/actor/sac_components/rlt_schedule_state_rank_0.pt" ]] ||
    [[ ! -f "${RLINF_STAGE2_RESUME_DIR}/actor/sac_components/replay_buffer/rank_0/metadata.json" ]]; then
    echo "RLINF_STAGE2_RESUME_DIR must be an absolute, complete Stage-2 checkpoint." >&2
    exit 1
  fi
  HYDRA_OVERRIDES+=("runner.resume_dir=${RLINF_STAGE2_RESUME_DIR}")
  echo "[Stage2] Resuming checkpoint: ${RLINF_STAGE2_RESUME_DIR}"
else
  echo "[Stage2] Starting with fresh actor/critic weights."
fi

# Reuse an existing head when it is healthy. This deliberately does not call
# `ray stop`: a hardware operator may have long-lived ROS services on the same
# host, and tearing down processes blindly is not part of starting a new trial.
if ! ray status --address="$GPU_NODE_IP:6379" >/dev/null 2>&1; then
  ray start --head --port=6379 --node-ip-address="$GPU_NODE_IP" --disable-usage-stats
fi

python toolkits/realworld_check/preflight_piper_rlt_stage2.py \
  --ray-address "$GPU_NODE_IP:6379"

# RLINF_STAGE2_CONFIG_NAME selects a variant such as the zero-correction
# diagnostic (realworld_piper_rlt_stage2_zero_correction).
STAGE2_CONFIG_NAME="${RLINF_STAGE2_CONFIG_NAME:-realworld_piper_rlt_stage2_ac_mlp}"
echo "[Stage2] Config:     ${STAGE2_CONFIG_NAME}"

exec bash examples/embodiment/run_realworld_async.sh \
  "$STAGE2_CONFIG_NAME" "${HYDRA_OVERRIDES[@]}"
