from __future__ import annotations

from collections import deque
import dataclasses
import logging
import threading
import time

import numpy as np
from openpi_client import image_tools
from openpi_client.runtime import environment as _environment
from typing_extensions import override

try:
    import rospy
    from cv_bridge import CvBridge
    from sensor_msgs.msg import Image
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Bool
except ImportError as exc:  # pragma: no cover - depends on sourced ROS env
    raise SystemExit(
        "Missing ROS Python modules. Source /opt/ros/noetic/setup.bash and your robot workspaces first."
    ) from exc


@dataclasses.dataclass(frozen=True)
class PiperTopics:
    front_image: str = "/camera_f/color/image_raw"
    left_image: str = "/camera_l/color/image_raw"
    right_image: str = "/camera_r/color/image_raw"
    left_state: str = "/puppet/joint_left"
    right_state: str = "/puppet/joint_right"
    left_command: str = "/master/joint_left"
    right_command: str = "/master/joint_right"
    enable: str = "/enable_flag"
    hold_active: str = "/teleop/hold_active"
    policy_hold_ack: str = "/teleop/policy_hold_ack"


class PiperRealEnvironment(_environment.Environment):
    """Environment adapter for the Piper bimanual robot stack."""

    def __init__(
        self,
        *,
        topics: PiperTopics = PiperTopics(),
        render_height: int = 224,
        render_width: int = 224,
        ready_timeout_s: float = 10.0,
        enable_on_reset: bool = True,
        enable_pulses: int = 5,
        enable_interval_s: float = 0.1,
        command_publish_hz: float = 0.0,
        command_interpolation_duration_s: float = 0.0,
        action_smoothing_alpha: float = 1.0,
        max_arm_velocity_rad_s: float = 0.0,
        interpolate_gripper: bool = False,
        init_node: bool = True,
    ) -> None:
        self._topics = topics
        self._render_height = render_height
        self._render_width = render_width
        self._ready_timeout_s = ready_timeout_s
        self._enable_on_reset = enable_on_reset
        self._enable_pulses = enable_pulses
        self._enable_interval_s = enable_interval_s
        self._command_publish_hz = float(command_publish_hz)
        self._command_interpolation_duration_s = float(command_interpolation_duration_s)
        self._action_smoothing_alpha = float(action_smoothing_alpha)
        self._max_arm_velocity_rad_s = float(max_arm_velocity_rad_s)
        self._interpolate_gripper = bool(interpolate_gripper)
        if self._command_publish_hz < 0:
            raise ValueError("command_publish_hz must be non-negative")
        if self._command_publish_hz > 0 and self._command_interpolation_duration_s <= 0:
            raise ValueError(
                "command_interpolation_duration_s must be positive when command_publish_hz is enabled"
            )
        if not 0 < self._action_smoothing_alpha <= 1:
            raise ValueError("action_smoothing_alpha must be in (0, 1]")
        if self._max_arm_velocity_rad_s < 0:
            raise ValueError("max_arm_velocity_rad_s must be non-negative")

        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._hold_condition = threading.Condition(self._lock)

        self._front_images: deque[Image] = deque(maxlen=256)
        self._left_images: deque[Image] = deque(maxlen=256)
        self._right_images: deque[Image] = deque(maxlen=256)
        self._left_states: deque[JointState] = deque(maxlen=256)
        self._right_states: deque[JointState] = deque(maxlen=256)

        self._left_command_pub = None
        self._right_command_pub = None
        self._enable_pub = None
        self._policy_hold_ack_pub = None
        self._command_start: np.ndarray | None = None
        self._command_target: np.ndarray | None = None
        self._command_start_time = 0.0
        self._last_smoothed_action: np.ndarray | None = None
        self._episode_start_action: np.ndarray | None = None
        self._reset_in_progress = False
        self._hold_active = False
        self._hold_generation = 0
        self._command_thread: threading.Thread | None = None

        if init_node and not rospy.core.is_initialized():
            rospy.init_node("openpi_piper_runtime", anonymous=True, disable_signals=True)

        self._left_command_pub = rospy.Publisher(self._topics.left_command, JointState, queue_size=1, tcp_nodelay=True)
        self._right_command_pub = rospy.Publisher(
            self._topics.right_command, JointState, queue_size=1, tcp_nodelay=True
        )
        self._enable_pub = rospy.Publisher(self._topics.enable, Bool, queue_size=1, tcp_nodelay=True)
        self._policy_hold_ack_pub = rospy.Publisher(
            self._topics.policy_hold_ack,
            Bool,
            queue_size=1,
            latch=True,
            tcp_nodelay=True,
        )
        # Publish the initial state before subscribing to the mux's latched
        # HOLD. Otherwise a fast HOLD callback could ACK=true and then be
        # accidentally overwritten by a late initialization ACK=false.
        self._policy_hold_ack_pub.publish(Bool(data=False))

        rospy.Subscriber(self._topics.front_image, Image, self._front_image_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber(self._topics.left_image, Image, self._left_image_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber(self._topics.right_image, Image, self._right_image_cb, queue_size=1, buff_size=2**24)
        rospy.Subscriber(self._topics.left_state, JointState, self._left_state_cb, queue_size=50, tcp_nodelay=True)
        rospy.Subscriber(self._topics.right_state, JointState, self._right_state_cb, queue_size=50, tcp_nodelay=True)
        rospy.Subscriber(self._topics.hold_active, Bool, self._hold_active_cb, queue_size=1)

        if self._command_publish_hz > 0:
            self._command_thread = threading.Thread(
                target=self._run_command_interpolator,
                name="openpi-piper-command-interpolator",
                daemon=True,
            )
            self._command_thread.start()

    def _front_image_cb(self, msg: Image) -> None:
        with self._lock:
            self._front_images.append(msg)

    def _left_image_cb(self, msg: Image) -> None:
        with self._lock:
            self._left_images.append(msg)

    def _right_image_cb(self, msg: Image) -> None:
        with self._lock:
            self._right_images.append(msg)

    def _left_state_cb(self, msg: JointState) -> None:
        with self._lock:
            self._left_states.append(msg)

    def _right_state_cb(self, msg: JointState) -> None:
        with self._lock:
            self._right_states.append(msg)

    def _hold_active_cb(self, msg: Bool) -> None:
        """Stop publication and ACK only after the final publish has drained."""
        active = bool(msg.data)
        with self._hold_condition:
            changed = active != self._hold_active
            if changed:
                self._hold_active = active
                self._hold_generation += 1
                self._hold_condition.notify_all()
        if active:
            # _command_lock is also held around every policy ROS publish. Thus
            # entering this block waits for an already-started publish to
            # return, prevents another from starting, and clears every cached
            # target before ACK=true becomes observable by the mux.
            with self._command_lock:
                self._command_start = None
                self._command_target = None
                self._last_smoothed_action = None
            self._policy_hold_ack_pub.publish(Bool(data=True))
            if changed:
                logging.warning(
                    "OpenPI execution paused; final ROS publish drained, pending "
                    "actions discarded, policy HOLD ACK sent."
                )
        else:
            self._policy_hold_ack_pub.publish(Bool(data=False))
            if changed:
                logging.info("OpenPI HOLD released; waiting for a freshly replanned action chunk.")

    def wait_until_policy_execution_allowed(self) -> int:
        """Block model execution during HOLD and return the current epoch."""
        with self._hold_condition:
            while self._hold_active and not rospy.is_shutdown():
                self._hold_condition.wait(timeout=0.1)
            if rospy.is_shutdown():
                raise RuntimeError("ROS shut down while policy execution was paused")
            return self._hold_generation

    def policy_execution_snapshot(self) -> tuple[bool, int]:
        with self._lock:
            return (not self._hold_active), self._hold_generation

    def _wait_until_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline and not rospy.is_shutdown():
            with self._lock:
                ready = all(
                    (
                        self._front_images,
                        self._left_images,
                        self._right_images,
                        self._left_states,
                        self._right_states,
                    )
                )
            if ready:
                return
            time.sleep(0.05)
        raise RuntimeError(
            "Timed out waiting for Piper topics. Check cameras and joint-state publishers are running."
        )

    @staticmethod
    def _stamp_sec(msg: Image | JointState) -> float:
        return msg.header.stamp.to_sec()

    def _pop_synced(self, queue: deque[Image] | deque[JointState], frame_time: float):
        while len(queue) > 1 and self._stamp_sec(queue[0]) < frame_time:
            queue.popleft()
        if not queue or self._stamp_sec(queue[0]) < frame_time:
            return None
        return queue.popleft()

    def _get_synced_messages(self):
        with self._lock:
            required = [
                self._front_images,
                self._left_images,
                self._right_images,
                self._left_states,
                self._right_states,
            ]
            if not all(required):
                return None

            frame_time = min(self._stamp_sec(q[-1]) for q in required)
            front = self._pop_synced(self._front_images, frame_time)
            left = self._pop_synced(self._left_images, frame_time)
            right = self._pop_synced(self._right_images, frame_time)
            left_state = self._pop_synced(self._left_states, frame_time)
            right_state = self._pop_synced(self._right_states, frame_time)

        if None in (front, left, right, left_state, right_state):
            return None
        return front, left, right, left_state, right_state

    def _decode_image(self, msg: Image) -> np.ndarray:
        # The pi05 demonstrations were trained as RGB (LeRobot video decode) and
        # the Astra topics publish rgb8; bgr8 here fed the policy swapped colors.
        image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        image = image_tools.convert_to_uint8(image_tools.resize_with_pad(image, self._render_height, self._render_width))
        return np.transpose(image, (2, 0, 1))

    def _build_observation(self) -> dict:
        self._wait_until_ready(self._ready_timeout_s)

        synced = None
        deadline = time.monotonic() + self._ready_timeout_s
        while time.monotonic() < deadline and not rospy.is_shutdown():
            synced = self._get_synced_messages()
            if synced is not None:
                break
            time.sleep(0.01)

        if synced is None:
            raise RuntimeError("Timed out synchronizing Piper image/joint observations.")

        front, left, right, left_state, right_state = synced
        state = np.concatenate((np.asarray(left_state.position), np.asarray(right_state.position)), axis=0)
        if state.shape[0] != 14:
            raise ValueError(f"Expected 14-dim Piper state, got {state.shape[0]}")

        return {
            "state": state.astype(np.float32),
            "images": {
                "cam_high": self._decode_image(front),
                "cam_left_wrist": self._decode_image(left),
                "cam_right_wrist": self._decode_image(right),
            },
        }

    def _publish_enable(self, enabled: bool) -> None:
        msg = Bool(data=enabled)
        for _ in range(max(1, self._enable_pulses)):
            self._enable_pub.publish(msg)
            time.sleep(self._enable_interval_s)

    def _latest_state_reference(self) -> np.ndarray | None:
        """Return the latest 14-D feedback state for the first command ramp."""
        with self._lock:
            if not self._left_states or not self._right_states:
                return None
            reference = np.concatenate(
                (
                    np.asarray(self._left_states[-1].position, dtype=np.float32),
                    np.asarray(self._right_states[-1].position, dtype=np.float32),
                )
            )
        return reference if reference.shape == (14,) else None

    def _interpolated_command_locked(self, now: float) -> np.ndarray | None:
        if self._command_start is None or self._command_target is None:
            return None
        ratio = min(
            1.0,
            max(0.0, (now - self._command_start_time) / self._command_interpolation_duration_s),
        )
        command = self._command_start + ratio * (self._command_target - self._command_start)
        # Keep gripper commands piecewise constant by default: closing/opening
        # is discrete and should not be delayed by arm interpolation.
        if not self._interpolate_gripper:
            command[[6, 13]] = (
                self._command_start[[6, 13]] if ratio < 1.0 else self._command_target[[6, 13]]
            )
        return command.astype(np.float32, copy=False)

    def _publish_joint_command(self, actions: np.ndarray) -> None:
        timestamp = rospy.Time.now()
        left_msg = JointState()
        left_msg.header.stamp = timestamp
        left_msg.position = actions[:7].tolist()
        right_msg = JointState()
        right_msg.header.stamp = timestamp
        right_msg.position = actions[7:].tolist()
        self._left_command_pub.publish(left_msg)
        self._right_command_pub.publish(right_msg)

    def _run_command_interpolator(self) -> None:
        """Publish a continuously interpolated arm reference at command_publish_hz."""
        rate = rospy.Rate(self._command_publish_hz)
        while not rospy.is_shutdown():
            with self._lock:
                hold_active = self._hold_active
            with self._command_lock:
                actions = (
                    None
                    if self._reset_in_progress or hold_active
                    else self._interpolated_command_locked(time.monotonic())
                )
                # Keep publication inside the same barrier used by the HOLD
                # callback. ACK=true can only be published after this returns.
                if actions is not None:
                    self._publish_joint_command(actions)
            rate.sleep()

    @override
    def reset(self) -> None:
        if self._enable_on_reset:
            logging.info("Publishing enable flag to Piper controllers.")
            self._publish_enable(True)
        self._wait_until_ready(self._ready_timeout_s)
        if self._episode_start_action is None:
            self._episode_start_action = self._latest_state_reference()
            if self._episode_start_action is not None:
                logging.info("Captured Piper episode-start pose for safe return on exit.")

    def captured_start_action(self) -> np.ndarray | None:
        """Return a copy of the first valid 14-D state observed this run."""
        return None if self._episode_start_action is None else self._episode_start_action.copy()

    def hold_current_position(self) -> None:
        """Stop interpolation and command the current feedback pose as a hold target."""
        current = self._latest_state_reference()
        if current is None:
            logging.warning("Cannot publish Piper hold target: no 14-D joint-state feedback is available.")
            return
        with self._command_lock:
            self._command_start = None
            self._command_target = None
            self._last_smoothed_action = current.copy()
        # Publish several times so the ROS controller receives the hold target
        # even if its subscriber was temporarily busy with the final policy command.
        for _ in range(3):
            self._publish_joint_command(current)
            time.sleep(0.05)
        logging.info("Piper policy execution stopped; holding current feedback pose.")

    @override
    def is_episode_complete(self) -> bool:
        return False

    @override
    def get_observation(self) -> dict:
        return self._build_observation()

    @override
    def apply_action(self, action: dict) -> None:
        allowed, generation = self.policy_execution_snapshot()
        action_generation = action.get("_piper_hold_generation")
        if not allowed or (action_generation is not None and action_generation != generation):
            logging.warning(
                "Dropping a policy action from an obsolete HOLD epoch (action=%s current=%s).",
                action_generation,
                generation,
            )
            return
        actions = np.asarray(action["actions"], dtype=np.float32)
        if actions.shape[0] != 14:
            raise ValueError(f"Expected 14-dim Piper action, got {actions.shape}")
        if self._command_publish_hz <= 0:
            with self._command_lock:
                allowed, current_generation = self.policy_execution_snapshot()
                if not allowed or current_generation != generation:
                    logging.info("Dropped policy action at the HOLD publish barrier.")
                    return
                self._publish_joint_command(actions)
            return

        now = time.monotonic()
        first_reference = self._latest_state_reference()
        with self._command_lock:
            allowed, current_generation = self.policy_execution_snapshot()
            if not allowed or current_generation != generation:
                logging.info("Dropped policy target at the HOLD interpolation barrier.")
                return
            current = self._interpolated_command_locked(now)
            if current is None:
                current = first_reference if first_reference is not None else actions.copy()

            target = actions.copy()
            if self._last_smoothed_action is not None:
                arm_indices = np.r_[0:6, 7:13]
                alpha = self._action_smoothing_alpha
                target[arm_indices] = (
                    alpha * target[arm_indices]
                    + (1.0 - alpha) * self._last_smoothed_action[arm_indices]
                )
            else:
                arm_indices = np.r_[0:6, 7:13]

            # A hard per-joint velocity limit is more reliable than filtering
            # alone for suppressing a single erratic PI0.5 waypoint.  The
            # interpolation ramp then traverses this bounded delta over its
            # configured duration, so its slope cannot exceed this limit.
            if self._max_arm_velocity_rad_s > 0:
                max_delta = self._max_arm_velocity_rad_s * self._command_interpolation_duration_s
                target[arm_indices] = current[arm_indices] + np.clip(
                    target[arm_indices] - current[arm_indices], -max_delta, max_delta
                )
            self._last_smoothed_action = target.copy()
            self._command_start = current.copy()
            self._command_target = target
            self._command_start_time = now

    def return_to_joint_home(
        self,
        home_action: np.ndarray,
        *,
        duration_s: float,
        timeout_s: float,
        tolerance_rad: float,
        publish_hz: float = 30.0,
    ) -> None:
        """Smoothly return both arms to an explicitly configured safe joint pose.

        This deliberately publishes through the same ROS topics as policy actions.
        Do not open a second direct CAN client (for example pyAgxArm's ``move_p``)
        while the Piper ROS driver owns the CAN interface.
        """
        home_action = np.asarray(home_action, dtype=np.float32)
        if home_action.shape != (14,):
            raise ValueError(f"Expected a 14-D Piper home action, got {home_action.shape}")
        if duration_s <= 0 or timeout_s <= 0 or tolerance_rad <= 0 or publish_hz <= 0:
            raise ValueError("home duration, timeout, tolerance, and publish rate must be positive")

        self._wait_until_ready(self._ready_timeout_s)
        start_action = self._latest_state_reference()
        if start_action is None:
            raise RuntimeError("Cannot reset Piper: no 14-D joint-state feedback is available.")

        logging.info(
            "Returning Piper arms to configured home pose over %.1fs; target=%s",
            duration_s,
            np.array2string(home_action, precision=3),
        )
        # Pause the optional high-rate interpolator. Without this guard it would
        # publish its own short (usually 1 / max_hz) ramp and defeat the explicit
        # slow reset duration below.
        with self._command_lock:
            self._reset_in_progress = True
            self._command_start = None
            self._command_target = None

        period_s = 1.0 / publish_hz
        start_time = time.monotonic()
        try:
            while not rospy.is_shutdown():
                elapsed_s = time.monotonic() - start_time
                ratio = min(1.0, elapsed_s / duration_s)
                command = start_action + ratio * (home_action - start_action)
                self._publish_joint_command(command)
                if ratio >= 1.0:
                    break
                time.sleep(period_s)

            deadline = time.monotonic() + timeout_s
            arm_indices = np.r_[0:6, 7:13]
            while not rospy.is_shutdown() and time.monotonic() < deadline:
                self._publish_joint_command(home_action)
                measured = self._latest_state_reference()
                if measured is not None and np.max(np.abs(measured[arm_indices] - home_action[arm_indices])) <= tolerance_rad:
                    logging.info("Piper arms reached configured home pose.")
                    return
                time.sleep(period_s)

            raise RuntimeError(
                f"Piper did not reach the configured home pose within {timeout_s:.1f}s after the commanded motion."
            )
        finally:
            with self._command_lock:
                self._reset_in_progress = False
