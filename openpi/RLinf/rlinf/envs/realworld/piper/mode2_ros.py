"""ROS transport for the opt-in Piper mode-2 hardware driver.

This transport does not open CAN and does not reuse mode-0/mode-1 command
topics.  A separately launched, hardware-specific mode-2 driver must publish
leader/follower feedback, subscribe to the leader-target topics, and expose
the drag-mode services.  Missing endpoints fail closed.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from rlinf.envs.realworld.piper.mode2_controller import Mode2Feedback


@dataclass(frozen=True)
class Mode2RosConfig:
    """ROS endpoints and readiness limits for an independent mode-2 driver."""

    leader_left_topic: str = "/rlinf/mode2/leader_left"
    leader_right_topic: str = "/rlinf/mode2/leader_right"
    follower_left_topic: str = "/puppet/joint_left"
    follower_right_topic: str = "/puppet/joint_right"
    leader_target_left_topic: str = "/rlinf/mode2/leader_target_left"
    leader_target_right_topic: str = "/rlinf/mode2/leader_target_right"
    leader_drag_left_service: str = "/rlinf/mode2/leader_drag_left"
    leader_drag_right_service: str = "/rlinf/mode2/leader_drag_right"
    follower_teleop_service: str = "/rlinf/mode2/follower_teleop"
    ready_timeout_s: float = 5.0

    def __post_init__(self) -> None:
        if self.ready_timeout_s <= 0:
            raise ValueError("ready_timeout_s must be positive.")


class RosMode2Hardware:
    """Mode-2 ROS client with fresh-feedback and live-subscriber checks."""

    def __init__(self, config: Mode2RosConfig | None = None) -> None:
        try:
            import rospy
            from sensor_msgs.msg import JointState
            from std_srvs.srv import SetBool, Trigger
        except ImportError as exc:  # pragma: no cover - requires robot ROS
            raise ModuleNotFoundError(
                "Piper mode 2 requires ROS1 sensor_msgs and std_srvs."
            ) from exc

        self.config = config or Mode2RosConfig()
        self._rospy = rospy
        self._JointState = JointState
        self._Trigger = Trigger
        self._SetBool = SetBool
        self._lock = threading.Lock()
        self._positions: dict[str, np.ndarray] = {}
        self._timestamps: dict[str, float] = {}
        topic_map = {
            "leader_left": self.config.leader_left_topic,
            "leader_right": self.config.leader_right_topic,
            "follower_left": self.config.follower_left_topic,
            "follower_right": self.config.follower_right_topic,
        }
        self._subscribers = [
            rospy.Subscriber(
                topic,
                JointState,
                self._joint_callback,
                callback_args=name,
                queue_size=1,
                tcp_nodelay=True,
            )
            for name, topic in topic_map.items()
        ]
        self._left_target_pub = rospy.Publisher(
            self.config.leader_target_left_topic,
            JointState,
            queue_size=1,
            tcp_nodelay=True,
        )
        self._right_target_pub = rospy.Publisher(
            self.config.leader_target_right_topic,
            JointState,
            queue_size=1,
            tcp_nodelay=True,
        )

    def _joint_callback(self, message, name: str) -> None:
        positions = np.asarray(message.position, dtype=np.float32).reshape(-1)
        if positions.shape != (7,) or not np.isfinite(positions).all():
            return
        # Use one monotonic clock domain for freshness checks. ROS header
        # stamps may use wall/sim time and cannot be compared to monotonic().
        stamp = time.monotonic()
        with self._lock:
            self._positions[name] = positions.copy()
            self._timestamps[name] = stamp

    def feedback(self) -> Mode2Feedback:
        """Return a coherent copy of the latest four arm messages."""
        with self._lock:
            missing = {
                name
                for name in (
                    "leader_left",
                    "leader_right",
                    "follower_left",
                    "follower_right",
                )
                if name not in self._positions
            }
            if missing:
                raise RuntimeError(
                    "mode-2 joint feedback is not ready: " + ", ".join(sorted(missing))
                )
            leader = np.concatenate(
                (self._positions["leader_left"], self._positions["leader_right"])
            )
            follower = np.concatenate(
                (self._positions["follower_left"], self._positions["follower_right"])
            )
            leader_timestamp = min(
                self._timestamps["leader_left"],
                self._timestamps["leader_right"],
            )
            follower_timestamp = min(
                self._timestamps["follower_left"],
                self._timestamps["follower_right"],
            )
        return Mode2Feedback(
            leader_action=leader,
            follower_action=follower,
            leader_timestamp=leader_timestamp,
            follower_timestamp=follower_timestamp,
        )

    def command_leader_joints(self, joints: np.ndarray) -> None:
        """Publish a leader-only target after verifying live driver subscribers."""
        target = np.asarray(joints, dtype=np.float32).reshape(-1)
        if target.shape != (12,) or not np.isfinite(target).all():
            raise ValueError("leader target must contain twelve finite joints.")
        if (
            self._left_target_pub.get_num_connections() < 1
            or self._right_target_pub.get_num_connections() < 1
        ):
            raise RuntimeError(
                "mode-2 leader target driver is not connected; followers remain held."
            )
        for publisher, values in (
            (self._left_target_pub, target[:6]),
            (self._right_target_pub, target[6:]),
        ):
            message = self._JointState()
            message.header.stamp = self._rospy.Time.now()
            message.name = [f"joint{index}" for index in range(6)]
            message.position = values.tolist()
            publisher.publish(message)

    def enter_leader_drag_mode(self) -> None:
        """Request zero-force drag on both leaders, requiring both acknowledgements."""
        for service_name in (
            self.config.leader_drag_left_service,
            self.config.leader_drag_right_service,
        ):
            self._rospy.wait_for_service(
                service_name,
                timeout=self.config.ready_timeout_s,
            )
            response = self._rospy.ServiceProxy(service_name, self._Trigger)()
            if not response.success:
                raise RuntimeError(
                    f"{service_name} rejected drag mode: {response.message}"
                )

    def set_follower_teleop_enabled(self, enabled: bool) -> None:
        """Switch the independent 200-Hz master-to-follower bridge."""
        service_name = self.config.follower_teleop_service
        self._rospy.wait_for_service(
            service_name,
            timeout=self.config.ready_timeout_s,
        )
        response = self._rospy.ServiceProxy(service_name, self._SetBool)(enabled)
        if not response.success:
            raise RuntimeError(
                f"{service_name} rejected teleop={enabled}: {response.message}"
            )
