sudo apt install libgflags-dev libgoogle-glog-dev libusb-1.0-0-dev libeigen3-dev -y
sudo apt install ros-$ROS_DISTRO-image-geometry ros-$ROS_DISTRO-camera-info-manager ros-$ROS_DISTRO-image-transport ros-$ROS_DISTRO-image-publisher ros-$ROS_DISTRO-libuvc-ros -y
sudo apt install can-utils net-tools libkdl-parser-dev -y

pip install empy==3.3.4 rospkg catkin_pkg