#!/usr/bin/env bash
# ROS Noetic + catkin workspaces: Piper (four-CAN mux, takeover) and cameras.
set -euo pipefail
RLT_ROOT="${RLT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if [[ ! -f /opt/ros/noetic/setup.bash ]]; then
  sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
  curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -
  sudo apt-get update
  sudo apt-get install -y ros-noetic-desktop-full
fi
sudo apt-get install -y python3-catkin-tools python3-rosdep \
  ros-noetic-image-geometry ros-noetic-camera-info-manager ros-noetic-image-transport \
  ros-noetic-image-publisher ros-noetic-libuvc-ros ros-noetic-realsense2-camera ros-noetic-librealsense2
set +u; source /opt/ros/noetic/setup.bash; set -u

# Local changes on top of the pinned submodule commits.
apply_patch() {  # $1 repo dir, $2 patch
  if [[ ! -s "$2" ]]; then return; fi
  if git -C "$1" apply --check "$2" 2>/dev/null; then
    git -C "$1" apply "$2" && echo "[RLT] applied $(basename "$2")"
  else
    echo "[RLT] $(basename "$2") already applied or not applicable; skipped"
  fi
}
apply_patch "$RLT_ROOT/cobot_magic/Piper_ros_private-ros-noetic" "$RLT_ROOT/patches/Piper_ros_private-ros-noetic.patch"
apply_patch "$RLT_ROOT/cobot_magic/collect_data/piper_sdk_demo" "$RLT_ROOT/patches/piper_sdk_demo.patch"

(cd "$RLT_ROOT/cobot_magic/Piper_ros_private-ros-noetic" && catkin_make)
(cd "$RLT_ROOT/cobot_magic/camera_ws" && catkin_make)
echo "[RLT] CAN udev rules / names: see cobot_magic/tools and cobot_magic/Piper_ros_private-ros-noetic/can_config.sh"
