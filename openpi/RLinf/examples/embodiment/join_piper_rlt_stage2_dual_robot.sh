#!/usr/bin/env bash
# Run on the Piper/ROS host (rank 1) after all five required ROS topics exist.
set -euo pipefail

REPO_DIR="${RLINF_ROBOT_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
HEAD_IP="${RLINF_GPU_HEAD_IP:-${GPU_HEAD_IP:-}}"
if [[ -z "$HEAD_IP" || "$HEAD_IP" == "192.168.85.129" ]]; then
  echo "Set RLINF_GPU_HEAD_IP (or GPU_HEAD_IP) to this job's compute-node Head IP, not the login-node address 192.168.85.129." >&2
  exit 1
fi
echo "[Stage2] Ray Head: ${HEAD_IP}:6379"
ROBOT_IP="${RLINF_ROBOT_IP:-192.168.209.107}"
KEYBOARD_DEVICE="${RLINF_KEYBOARD_DEVICE:-}"
PIPER_WS="${PIPER_WS:-${RLT_ROOT:-$HOME/RLT}/cobot_magic/Piper_ros_private-ros-noetic}"

unset PYTHONPATH
source /opt/ros/noetic/setup.bash
source "$PIPER_WS/devel/setup.bash"
source "$REPO_DIR/.venv-rlt/bin/activate"
cd "$REPO_DIR"

export RLINF_ROS_PYTHONPATH="$PYTHONPATH"
export PYTHONPATH="$REPO_DIR:$RLINF_ROS_PYTHONPATH"
export ROS_MASTER_URI="http://${ROBOT_IP}:11311"
export ROS_IP="$ROBOT_IP"
if [[ -n "$KEYBOARD_DEVICE" && -e "$KEYBOARD_DEVICE" ]]; then
  export RLINF_KEYBOARD_DEVICE="$KEYBOARD_DEVICE"
elif [[ -n "$KEYBOARD_DEVICE" ]]; then
  echo "[Stage2] Keyboard override $KEYBOARD_DEVICE no longer exists; auto-detecting an evdev keyboard."
  unset RLINF_KEYBOARD_DEVICE
else
  unset RLINF_KEYBOARD_DEVICE
fi
export RLINF_JAX_REF_HOST="${RLINF_JAX_REF_HOST:-$ROBOT_IP}"
export RLINF_JAX_REF_PORT="${RLINF_JAX_REF_PORT:-6006}"
export RLINF_PRESERVE_ROS_MASTER=1
export RLINF_PIPER_CONTROL_MODE="${RLINF_PIPER_CONTROL_MODE:-1}"
export RLINF_NODE_RANK=1
export CUDA_VISIBLE_DEVICES=0

# Pin PyTorch collectives to the same routed interface used by Ray. The Piper
# host currently resolves this to eno1; deriving it keeps the script portable.
ROBOT_IFACE="$(ip route get "$HEAD_IP" | awk '{for (i=1; i<=NF; i++) if ($i == "dev") {print $(i+1); exit}}')"
if [[ -z "$ROBOT_IFACE" ]]; then
  echo "Cannot determine the robot interface used to reach ${HEAD_IP}." >&2
  exit 1
fi
export GLOO_SOCKET_IFNAME="$ROBOT_IFACE"
export NCCL_SOCKET_IFNAME="$ROBOT_IFACE"
echo "[Stage2] Distributed interface: $ROBOT_IFACE"

PREFLIGHT_ARGS=(
  --ray-address "$HEAD_IP:6379"
  --skip-ray
)

python toolkits/realworld_check/preflight_piper_rlt_stage2.py \
  "${PREFLIGHT_ARGS[@]}"

# This stops Ray processes only on the robot host. ROS/camera/Piper are not Ray
# processes and remain running; RLINF_PRESERVE_ROS_MASTER prevents RLinf imports
# from terminating the externally managed ROS master.
ray stop
ray start --address="$HEAD_IP:6379" --node-ip-address="$ROBOT_IP"
ray status --address="$HEAD_IP:6379"

echo "Piper host joined ${HEAD_IP}:6379 as RLINF node rank 1."
