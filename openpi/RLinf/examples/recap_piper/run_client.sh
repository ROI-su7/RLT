#!/usr/bin/env bash
# Reuse README_PIPER_FOUR_CAN's unmodified OpenPI ROS client and mux protocol.
set -euo pipefail
RECAP_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RECAP_SCRIPT_DIR/env.sh"
PROFILE="${1:-takeover}"
shift || true
case "$PROFILE" in
  takeover|followers)
    # Both profiles use the Pi05 grasp-timing fix. A 15-step prefetch suffix
    # made even an 85 ms inference result about 1 s old at promotion.
    TIMING=(--action-horizon 10 --rtc-prefix-len 4 --rtc-boundary-blend-steps 5 \
      --action-smoothing-alpha 0.5 --max-arm-velocity-rad-s 0.6)
    if [[ "$PROFILE" == takeover ]]; then
      # Preserve the collection profile's original arm speed limit.
      TIMING[${#TIMING[@]}-1]=0.4
    fi
    ;;
  *) echo "Expected takeover or followers profile" >&2; exit 2 ;;
esac
CLIENT_PYTHON="${RECAP_CLIENT_PYTHON:-/home/agilex/miniconda3/envs/pi0/bin/python}"
CMD=("$CLIENT_PYTHON" -m rlinf.recap_piper.client
  --host 127.0.0.1 --port 6016 --max-hz 15 --rtc-enabled
  --command-publish-hz 200 --command-time-scale 1.0
  --num-episodes 1 --max-episode-steps 0 --no-record-video
  --left-command-topic /teleop/policy/joint_left
  --right-command-topic /teleop/policy/joint_right
  "${TIMING[@]}")
# Inspect the exact command without importing ROS or enabling the robot.
if [[ "${1:-}" == --dry-run ]]; then
  shift
  printf '%q ' "${CMD[@]}" "$@"
  printf '\n'
  exit 0
fi
CMD+=("$@")
source /opt/ros/noetic/setup.bash
source /home/agilex/cobot_magic/camera_ws/devel/setup.bash
source /home/agilex/cobot_magic/Piper_ros_private-ros-noetic/devel/setup.bash
export PYTHONPATH="$RECAP_NATIVE_OPENPI${PYTHONPATH:+:$PYTHONPATH}"
cd "$RECAP_NATIVE_OPENPI"
# Keep the terminal output and a copy for diagnosing policy-command stalls.
LOG_DIR="$RECAP_REPO_DIR/recap_runs/robot_logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/client_$(date +%Y%m%d-%H%M%S).log"
echo "Client log: $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1
export PYTHONUNBUFFERED=1
printf 'Client profile: %s\nCommand: ' "$PROFILE"
printf '%q ' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
