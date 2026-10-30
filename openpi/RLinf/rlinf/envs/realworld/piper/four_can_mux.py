"""Client for the existing OpenPI four-CAN ROS mux (no CAN role control)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MuxSnapshot:
    """Confirmed hardware ownership, rather than a requested keyboard mode."""

    mode: str
    phase: str
    inference_active: bool
    hold: bool
    home: bool
    generation: int

    @property
    def human_active(self) -> bool:
        return self.mode == "TAKEOVER" and self.phase == "ACTIVE" and not self.home

    @property
    def policy_active(self) -> bool:
        return (
            self.mode == "INFERENCE"
            and self.inference_active
            and not self.hold
            and not self.home
        )


class FourCanMuxClient:
    """Use the unmodified takeover launch's request and feedback topics.

    The isolated policy executor owns the HOLD ACK. The mux alone switches
    leader roles, commands followers, captures HOME and interpolates HOME.
    """

    def __init__(self, rospy, *, feedback_timeout_s: float = 0.5) -> None:
        from sensor_msgs.msg import JointState
        from std_msgs.msg import Bool, Empty, String

        self._String, self._Empty = String, Empty
        self._lock = threading.RLock()
        self._mode = "UNKNOWN"
        self._phase = "UNKNOWN"
        self._inference_active = False
        self._hold = True
        self._home = False
        self._generation = 0
        self._feedback_timeout_s = float(feedback_timeout_s)
        self._actions: dict[str, tuple[float, np.ndarray]] = {}
        self._mode_pub = rospy.Publisher("/teleop/mode_request", String, queue_size=10)
        self._hold_pub = rospy.Publisher("/teleop/hold_request", Empty, queue_size=1)
        self._home_pub = rospy.Publisher("/teleop/home_request", Empty, queue_size=1)
        self._subscriptions = [
            rospy.Subscriber("/teleop/mode", String, self._mode_cb, queue_size=1),
            rospy.Subscriber(
                "/teleop/takeover_phase", String, self._phase_cb, queue_size=1
            ),
            rospy.Subscriber(
                "/teleop/inference_active", Bool, self._inference_cb, queue_size=1
            ),
            rospy.Subscriber("/teleop/hold_active", Bool, self._hold_cb, queue_size=1),
            rospy.Subscriber("/teleop/home_active", Bool, self._home_cb, queue_size=1),
        ]
        for side in ("left", "right"):
            self._subscriptions.append(
                rospy.Subscriber(
                    f"/master/joint_{side}",
                    JointState,
                    self._action_cb,
                    callback_args=side,
                    queue_size=1,
                    tcp_nodelay=True,
                )
            )

    def _mode_cb(self, message) -> None:
        with self._lock:
            self._mode = message.data.strip().upper()

    def _phase_cb(self, message) -> None:
        with self._lock:
            self._phase = message.data.strip().upper()

    def _inference_cb(self, message) -> None:
        with self._lock:
            self._inference_active = bool(message.data)

    def _hold_cb(self, message) -> None:
        with self._lock:
            active = bool(message.data)
            if active != self._hold:
                self._generation += 1
                self._hold = active

    def _home_cb(self, message) -> None:
        with self._lock:
            active = bool(message.data)
            if active != self._home:
                self._generation += 1
                self._home = active

    def _action_cb(self, message, side: str) -> None:
        action = np.asarray(message.position, dtype=np.float32)
        if action.shape != (7,) or not np.isfinite(action).all():
            return
        with self._lock:
            self._actions[side] = (time.monotonic(), action.copy())

    def snapshot(self) -> MuxSnapshot:
        """Return a consistent snapshot of the mux's latched status."""
        with self._lock:
            return MuxSnapshot(
                self._mode,
                self._phase,
                self._inference_active,
                self._hold,
                self._home,
                self._generation,
            )

    def executed_action(self) -> np.ndarray:
        """Read the fresh pair of mux-selected targets, in rad/metres."""
        with self._lock:
            now = time.monotonic()
            if any(
                side not in self._actions
                or now - self._actions[side][0] > self._feedback_timeout_s
                for side in ("left", "right")
            ):
                raise RuntimeError(
                    "Four-CAN mux executed action is incomplete or stale."
                )
            return np.concatenate(
                [self._actions[side][1] for side in ("left", "right")]
            )

    def request_mode(self, mode: str) -> None:
        """Request INFERENCE or TAKEOVER; callers must wait for confirmation."""
        if mode not in {"INFERENCE", "TAKEOVER"}:
            raise ValueError(f"Unsupported mux mode: {mode}")
        self._mode_pub.publish(self._String(data=mode))

    def request_hold(self) -> None:
        """Request the original software HOLD, never protocol quick-stop."""
        self._hold_pub.publish(self._Empty())

    def request_home(self) -> None:
        """Return all four arms to the poses captured by the ROS launch."""
        self._home_pub.publish(self._Empty())

    def close(self) -> None:
        """Release subscriptions after requesting HOLD."""
        self.request_hold()
        for subscription in self._subscriptions:
            subscription.unregister()
