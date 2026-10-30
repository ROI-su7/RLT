#!/usr/bin/env bash
# Start persistent single-host ROS services for Piper RLT in tmux.
set -euo pipefail

GPU_NODE_IP="${GPU_NODE_IP:-$(hostname -I | awk '{print $1}')}"
RLT_ROOT="${RLT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PIPER_WS="${PIPER_WS:-$RLT_ROOT/cobot_magic/Piper_ros_private-ros-noetic}"
CAMERA_WS="${CAMERA_WS:-$RLT_ROOT/cobot_magic/camera_ws}"
MASTER_URI="http://${GPU_NODE_IP}:11311"
WAIT_SECONDS="${RLINF_ROS_WAIT_SECONDS:-45}"
TAKEOVER_BACKEND="${RLINF_PIPER_TAKEOVER_BACKEND:-four_can}"
case "$TAKEOVER_BACKEND" in
  four_can)
    PIPER_LAUNCH="start_inference_takeover.launch master_left_can:=can_l_leader master_right_can:=can_r_leader puppet_left_can:=can_l_follower puppet_right_can:=can_r_follower publish_hz:=200"
    COMMAND_TOPICS=(/teleop/command/joint_left /teleop/command/joint_right)
    ;;
  legacy)
    PIPER_LAUNCH="start_ms_piper.launch mode:=1 auto_enable:=true"
    COMMAND_TOPICS=(/master/joint_left /master/joint_right)
    ;;
  *)
    echo "RLINF_PIPER_TAKEOVER_BACKEND must be four_can or legacy" >&2
    exit 1
    ;;
esac

# The Piper ROS scripts are executed by the system Python selected by the
# shebang, not the RLinf virtual environment.  Fail before bringing up the
# robot if its mandatory SDK is absent.
if ! /usr/bin/python3 -c 'import piper_sdk' >/dev/null 2>&1; then
  echo "Missing system-Python dependency piper_sdk. Install it with:" >&2
  echo "  /usr/bin/python3 -m pip install --user piper_sdk" >&2
  exit 1
fi

TMUX_SESSION="rlt_stage2_ros"

show_pane() {
  local window="$1"
  echo "----- tmux ${TMUX_SESSION}:${window} (last 80 lines) -----" >&2
  if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    tmux capture-pane -p -t "${TMUX_SESSION}:${window}" -S -80 2>/dev/null >&2 || true
  else
    echo "session is not running" >&2
  fi
}

fail() {
  echo "Piper ROS startup failed: $*" >&2
  show_pane roscore
  show_pane camera
  show_pane piper
  exit 1
}

# Older revisions used three independent tmux sessions.  If one was cleaned
# up, the others continued running against a vanished master.  Keep all ROS
# processes in one session with named windows so their lifetime is explicit.
for name in rlt_roscore rlt_camera rlt_piper; do
  tmux has-session -t "$name" 2>/dev/null && tmux kill-session -t "$name" || true
done
tmux has-session -t "$TMUX_SESSION" 2>/dev/null && tmux kill-session -t "$TMUX_SESSION" || true

tmux new-session -d -s "$TMUX_SESSION" -n roscore \
  "bash -lc 'source /opt/ros/noetic/setup.bash && export ROS_IP=${GPU_NODE_IP} ROS_MASTER_URI=${MASTER_URI} && exec roscore'"
tmux set-option -t "${TMUX_SESSION}:roscore" remain-on-exit on

# Do not start hardware nodes until the master is actually reachable.  The old
# fixed five-second wait could silently continue with no master, leaving every
# later process permanently disconnected.
master_ready=false
for ((second = 1; second <= WAIT_SECONDS; second++)); do
  if source /opt/ros/noetic/setup.bash && ROS_MASTER_URI="$MASTER_URI" rosnode list >/dev/null 2>&1; then
    master_ready=true
    break
  fi
  sleep 1
done
"$master_ready" || fail "ROS master ${MASTER_URI} did not become reachable within ${WAIT_SECONDS}s"

tmux new-window -t "$TMUX_SESSION" -n camera \
  "bash -lc 'source /opt/ros/noetic/setup.bash && source ${CAMERA_WS}/devel/setup.bash && export ROS_IP=${GPU_NODE_IP} ROS_MASTER_URI=${MASTER_URI} && exec roslaunch astra_camera multi_camera.launch'"
tmux set-option -t "${TMUX_SESSION}:camera" remain-on-exit on
tmux new-window -t "$TMUX_SESSION" -n piper \
  "bash -lc 'source /opt/ros/noetic/setup.bash && source ${PIPER_WS}/devel/setup.bash && export ROS_IP=${GPU_NODE_IP} ROS_MASTER_URI=${MASTER_URI} && exec roslaunch piper ${PIPER_LAUNCH}'"
tmux set-option -t "${TMUX_SESSION}:piper" remain-on-exit on

source /opt/ros/noetic/setup.bash
export ROS_MASTER_URI="$MASTER_URI"
export ROS_IP="$GPU_NODE_IP"
required=(
  /puppet/joint_left /puppet/joint_right
  /camera_f/color/image_raw /camera_l/color/image_raw /camera_r/color/image_raw
)
topics_ready=false
for ((second = 1; second <= WAIT_SECONDS; second++)); do
  topics="$(rostopic list 2>/dev/null || true)"
  missing=""
  for topic in "${required[@]}"; do
    if ! grep -Fxq "$topic" <<<"$topics"; then
      missing+=" ${topic}"
    fi
  done
  if [[ -z "$missing" ]]; then
    topics_ready=true
    break
  fi
  sleep 1
done
"$topics_ready" || fail "required topics were not ready within ${WAIT_SECONDS}s:${missing}"

# Topic registrations may outlive crashed processes in a persistent master.
# Require live hardware messages and real Piper command subscribers.
for topic in "${required[@]}"; do
  timeout 5 rostopic echo -n 1 "$topic" >/dev/null 2>&1 \
    || fail "${topic} exists but is not publishing fresh messages"
done
for topic in "${COMMAND_TOPICS[@]}"; do
  rostopic info "$topic" 2>/dev/null | awk '
    /^Subscribers:/ {in_subscribers=1; next}
    in_subscribers && /^ \* \/piper_/ {found=1}
    END {exit !found}
  ' || fail "Piper driver is not subscribed to ${topic}"
done
tmux list-windows -t "$TMUX_SESSION"
echo "Piper ROS services are ready at ${MASTER_URI}."
