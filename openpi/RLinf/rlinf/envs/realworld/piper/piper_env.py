"""ROS1 Gym environment for the bimanual Piper wire task.

The deployed policy and this environment use the same convention as
``examples/piper_real``: three BGR camera streams, 14-D joint feedback, and
14-D *absolute* joint-position targets (7 values per arm, including gripper).
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np
from openpi_client import image_tools

from rlinf.data.rlt_image_log import RLTImageLog
from rlinf.envs.realworld.piper.action_trace import PiperActionTraceWriter
from rlinf.envs.realworld.piper.control_process import PiperControlProcess
from rlinf.envs.realworld.piper.wire_head_pose import (
    HEAD_POSE_DIM,
    WireHeadPoseDetector,
)
from rlinf.utils.logging import get_logger


@dataclass
class PiperRobotConfig:
    """Runtime and safety configuration for a bimanual Piper ROS stack."""

    task_description: str = "insert the wire"
    is_dummy: bool = False
    step_frequency: float = 15.0
    ready_timeout_s: float = 10.0
    enable_on_reset: bool = True
    enable_pulses: int = 5
    enable_interval_s: float = 0.1
    render_height: int = 224
    render_width: int = 224
    max_num_steps: int = 300

    front_image_topic: str = "/camera_f/color/image_raw"
    left_image_topic: str = "/camera_l/color/image_raw"
    right_image_topic: str = "/camera_r/color/image_raw"
    left_state_topic: str = "/puppet/joint_left"
    right_state_topic: str = "/puppet/joint_right"
    left_command_topic: str = "/master/joint_left"
    right_command_topic: str = "/master/joint_right"
    enable_topic: str = "/enable_flag"

    # Cobot Magic/Piper ROS convention: six joints in rad + gripper stroke in m
    # for each arm.  The URDF is only a baseline; read the actual motor limits
    # from both CAN controllers before a real rollout.
    joint_position_limits_lower: list[float] = field(
        default_factory=lambda: [-2.618, 0.0, -2.697, -1.832, -1.22, -3.14, 0.0] * 2
    )
    joint_position_limits_upper: list[float] = field(
        default_factory=lambda: [2.618, 3.14, 0.0, 1.832, 1.22, 3.14, 0.08] * 2
    )
    # Applied only to the twelve revolute joints (rad per environment step).
    max_joint_delta: float = 0.05
    # ROS publishes/accepts the gripper as an aggregate stroke in metres.
    max_gripper_delta: float = 0.005
    # Keep enabled for generic direct-control safety. Disable only when
    # reproducing the validated OpenPI Piper client, which sends each bounded
    # absolute joint target once and lets the Piper controller track it.
    feedback_step_clamp_enabled: bool = True
    # Optional high-rate command interpolation. The policy/environment still
    # advances at step_frequency, while a dedicated thread continuously
    # publishes bounded intermediate arm targets to the Piper ROS controller.
    command_publish_hz: float = 0.0
    command_interpolation_duration_s: float = 0.0
    # Full-resolution frames of these cameras are logged for every executed
    # step of an RLT episode (rlinf.data.rlt_image_log); the policy only sees
    # 224x224. None disables the log.
    image_log_dir: str | None = None
    image_log_cameras: list[str] = field(
        default_factory=lambda: ["cam_left_wrist", "cam_right_wrist"]
    )
    image_log_jpeg_quality: int = 90
    # Connector-head keypoint detector (rlinf.envs.realworld.piper.
    # wire_head_pose) run on the full-resolution right wrist frame of every
    # decoded observation; adds obs["head_pose"] = [dx, dy, dangle, visible]
    # for the Stage-2 Actor/critic. None disables it.
    head_pose_weights: str | None = None
    head_pose_device: str = "cuda"
    action_smoothing_alpha: float = 1.0
    max_arm_velocity_rad_s: float = 0.0
    interpolate_gripper: bool = False
    actor_command_profile_enabled: bool = False
    actor_command_joint_step_rad: float = 0.002
    actor_command_gripper_step_m: float = 0.0002
    actor_command_joint_velocity_rad_s: float = 0.08
    actor_command_gripper_velocity_m_s: float = 0.004
    actor_command_joint_acceleration_rad_s2: float = 0.4
    actor_command_gripper_acceleration_m_s2: float = 0.02
    actor_command_response_time_s: float = 0.15
    # Keep the fixed-rate action scheduler and ROS publishers outside the Ray
    # EnvWorker. Ray serialization can hold the worker's GIL for >100 ms,
    # starving both Python control threads at every action-chunk boundary.
    isolated_control_process: bool = False
    # Reuse the original four-CAN mux; the policy executor must ACK HOLD.
    four_can_mux_enabled: bool = False
    # A new VLA prediction is requested after every action chunk. Blend only
    # the first few arm targets of the replacement chunk to remove its
    # boundary jump without delaying the remainder of the plan.
    action_chunk_size: int = 20
    action_chunk_boundary_blend_steps: int = 3
    # RLT "Subsampling Action Chunks": decode a real observation every N
    # control steps so the replay buffer can store <x_0, a_0:C>, <x_2, a_2:C+2>
    # ... instead of one row per chunk. 0 disables it and restores the
    # decode-only-at-chunk-end behaviour. Each extra offset costs one triple
    # camera decode inside the control period, so keep this at 2 (the paper's
    # stride) and watch the decode_mean warning in the observation timing log.
    rlt_subsample_stride: int = 0
    # Execution-side RTC.  The unused suffix of the native VLA H=50 plan
    # bridges the next synchronous inference request.  The returned plan is
    # shifted by the number of bridge commands already executed and blended
    # back onto the measured trajectory.
    rtc_enabled: bool = False
    rtc_max_bridge_steps: int = 5
    rtc_boundary_blend_steps: int = 5
    # One-key return-to-start settings.  The first episode's measured start
    # pose is captured and reused for the lifetime of this EnvWorker.
    reset_position_tolerance_rad: float = 0.01
    reset_gripper_tolerance_m: float = 0.001
    # During one-key reset both grippers remain fully open.  Only after both
    # arms have reached the captured pose are they closed at that pose.
    reset_open_gripper_position_m: float = 0.08
    reset_closed_gripper_position_m: float = 0.0
    reset_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        self.joint_position_limits_lower = np.asarray(
            self.joint_position_limits_lower, dtype=np.float32
        )
        self.joint_position_limits_upper = np.asarray(
            self.joint_position_limits_upper, dtype=np.float32
        )
        if self.joint_position_limits_lower.shape != (
            14,
        ) or self.joint_position_limits_upper.shape != (14,):
            raise ValueError("Piper joint-position limits must each contain 14 values.")
        if np.any(self.joint_position_limits_lower >= self.joint_position_limits_upper):
            raise ValueError(
                "Each Piper joint lower limit must be smaller than its upper limit."
            )
        if self.max_joint_delta <= 0:
            raise ValueError("max_joint_delta must be positive.")
        if self.max_gripper_delta <= 0:
            raise ValueError("max_gripper_delta must be positive.")
        if self.command_publish_hz < 0:
            raise ValueError("command_publish_hz must be non-negative.")
        if self.command_publish_hz > 0 and self.command_interpolation_duration_s <= 0:
            raise ValueError(
                "command_interpolation_duration_s must be positive when "
                "command_publish_hz is enabled."
            )
        if self.isolated_control_process and self.command_publish_hz <= 0:
            raise ValueError(
                "isolated_control_process requires command_publish_hz > 0."
            )
        if self.actor_command_profile_enabled:
            if not self.isolated_control_process:
                raise ValueError("Actor command profile requires isolated control.")
            for name in (
                "actor_command_joint_step_rad",
                "actor_command_gripper_step_m",
                "actor_command_joint_velocity_rad_s",
                "actor_command_gripper_velocity_m_s",
                "actor_command_joint_acceleration_rad_s2",
                "actor_command_gripper_acceleration_m_s2",
                "actor_command_response_time_s",
            ):
                value = float(getattr(self, name))
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f"{name} must be finite and positive.")
        if self.four_can_mux_enabled and not self.isolated_control_process:
            raise ValueError("four_can_mux_enabled requires isolated_control_process.")
        if self.four_can_mux_enabled and (
            self.left_command_topic != "/teleop/policy/joint_left"
            or self.right_command_topic != "/teleop/policy/joint_right"
        ):
            raise ValueError(
                "Four-CAN policy must publish only to /teleop/policy/joint_left/right."
            )
        if not 0 < self.action_smoothing_alpha <= 1:
            raise ValueError("action_smoothing_alpha must be in (0, 1].")
        if self.max_arm_velocity_rad_s < 0:
            raise ValueError("max_arm_velocity_rad_s must be non-negative.")
        if self.action_chunk_size <= 0:
            raise ValueError("action_chunk_size must be positive.")
        if not 0 <= self.action_chunk_boundary_blend_steps < self.action_chunk_size:
            raise ValueError(
                "action_chunk_boundary_blend_steps must be in [0, action_chunk_size)."
            )
        if not 0 <= self.rlt_subsample_stride <= self.action_chunk_size:
            raise ValueError(
                "rlt_subsample_stride must be in [0, action_chunk_size], got "
                f"{self.rlt_subsample_stride}."
            )
        if self.rtc_max_bridge_steps < 0:
            raise ValueError("rtc_max_bridge_steps must be non-negative.")
        if self.rtc_boundary_blend_steps < 0:
            raise ValueError("rtc_boundary_blend_steps must be non-negative.")
        if self.reset_position_tolerance_rad <= 0:
            raise ValueError("reset_position_tolerance_rad must be positive.")
        if self.reset_gripper_tolerance_m <= 0:
            raise ValueError("reset_gripper_tolerance_m must be positive.")
        gripper_lower = self.joint_position_limits_lower[[6, 13]]
        gripper_upper = self.joint_position_limits_upper[[6, 13]]
        for name, value in (
            ("reset_open_gripper_position_m", self.reset_open_gripper_position_m),
            ("reset_closed_gripper_position_m", self.reset_closed_gripper_position_m),
        ):
            if np.any(value < gripper_lower) or np.any(value > gripper_upper):
                raise ValueError(
                    f"{name} must be within both configured gripper limits."
                )
        if self.reset_timeout_s <= 0:
            raise ValueError("reset_timeout_s must be positive.")


class PiperWireEnv(gym.Env):
    """One physical Piper robot exposed as a Gym environment.

    Reset is intentionally non-motion-producing: the operator places the
    robot/task at its known start pose, then reset only re-enables the existing
    controller and waits for synchronized observations.
    """

    CONFIG_CLS = PiperRobotConfig

    def __init__(
        self,
        override_cfg: dict[str, Any],
        worker_info: Any,
        hardware_info: Any,
        env_idx: int,
    ):
        del worker_info, hardware_info, env_idx
        self.config = self.CONFIG_CLS(**override_cfg)
        self._logger = get_logger()
        self._num_steps = 0
        self._last_observation: dict[str, Any] | None = None
        self._observation_lock = threading.Lock()
        self._command_lock = threading.Lock()
        self._policy_action_lock = threading.Lock()
        self._next_policy_action_time = 0.0
        self._last_policy_dispatch_time = 0.0
        self._last_policy_dispatch_source: str | None = None
        self._slow_policy_interval_count = 0
        self._policy_dispatch_count = 0
        self._policy_interval_sum = 0.0
        self._policy_interval_max_error = 0.0
        self._command_stop = threading.Event()
        self._command_start: np.ndarray | None = None
        self._command_target: np.ndarray | None = None
        self._command_start_time = 0.0
        self._last_smoothed_action: np.ndarray | None = None
        self._last_raw_arm_action: np.ndarray | None = None
        self._last_raw_arm_delta: np.ndarray | None = None
        self._observation_metric_count = 0
        self._observation_sync_ms_sum = 0.0
        self._observation_decode_ms_sum = 0.0
        self._observation_total_ms_max = 0.0
        self._action_trace = PiperActionTraceWriter.from_environment("env")
        self._command_metric_count = 0
        self._raw_step_sum = 0.0
        self._raw_step_max = 0.0
        self._filtered_step_sum = 0.0
        self._filtered_step_max = 0.0
        self._raw_accel_max = 0.0
        self._last_policy_action: np.ndarray | None = None
        self._chunk_action_index = 0
        self._completed_action_chunks = 0
        self._remaining_actions_in_chunk = 0
        self._command_thread: threading.Thread | None = None
        self._image_log = (
            RLTImageLog(
                self.config.image_log_dir,
                jpeg_quality=self.config.image_log_jpeg_quality,
            )
            if self.config.image_log_dir and not self.config.is_dummy
            else None
        )
        self._image_log_episode: int | None = None
        self._head_pose = (
            WireHeadPoseDetector(
                self.config.head_pose_weights, self.config.head_pose_device
            )
            if self.config.head_pose_weights and not self.config.is_dummy
            else None
        )
        if self._head_pose is not None:
            # On CPU it costs tens of ms inside the control period; the
            # observation timing log reports head_pose_mean either way.
            log = (
                self._logger.info
                if self._head_pose.device == self.config.head_pose_device
                else self._logger.warning
            )
            log(
                "Head-pose detector %s on %s (requested %s).",
                self.config.head_pose_weights,
                self._head_pose.device,
                self.config.head_pose_device,
            )
        self._observation_head_pose_ms_sum = 0.0
        self._control_process: PiperControlProcess | None = None
        self._rtc_lock = threading.Lock()
        self._rtc_stop = threading.Event()
        self._rtc_thread: threading.Thread | None = None
        self._rtc_executed_steps = 0
        self._rtc_last_action: np.ndarray | None = None
        self._queued_rtc_tail: np.ndarray | None = None
        self._rtc_rollover_count = 0
        self._actor_direct_execution = False
        self._episode_start_position: np.ndarray | None = None
        self._init_spaces()
        if self.config.is_dummy:
            return
        self._init_ros()

    @property
    def task_description(self) -> str:
        return self.config.task_description

    def _init_spaces(self) -> None:
        self.action_space = gym.spaces.Box(
            self.config.joint_position_limits_lower,
            self.config.joint_position_limits_upper,
            dtype=np.float32,
        )
        image = gym.spaces.Box(
            0, 255, (self.config.render_height, self.config.render_width, 3), np.uint8
        )
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "joint_position": gym.spaces.Box(
                            -np.inf, np.inf, (14,), np.float32
                        )
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        "cam_high": image,
                        "cam_left_wrist": image,
                        "cam_right_wrist": image,
                    }
                ),
            }
        )
        if self.config.head_pose_weights:
            self.observation_space["head_pose"] = gym.spaces.Box(
                -np.inf, np.inf, (HEAD_POSE_DIM,), np.float32
            )

    def _init_ros(self) -> None:
        try:
            import cv2
            import rospy
            from cv_bridge import CvBridge
            from sensor_msgs.msg import Image, JointState
            from std_msgs.msg import Bool
        except ImportError as exc:  # pragma: no cover - depends on robot node
            raise ModuleNotFoundError(
                "PiperWireEnv requires ROS Noetic Python modules. Source ROS and the Piper workspace before starting Ray."
            ) from exc
        self._rospy, self._Image, self._JointState, self._Bool = (
            rospy,
            Image,
            JointState,
            Bool,
        )
        self._cv2 = cv2
        if not rospy.core.is_initialized():
            rospy.init_node(
                "rlinf_piper_wire_env", anonymous=True, disable_signals=True
            )
        self._bridge = CvBridge()
        self._front_images: deque[Any] = deque(maxlen=32)
        self._left_images: deque[Any] = deque(maxlen=32)
        self._right_images: deque[Any] = deque(maxlen=32)
        self._left_states: deque[Any] = deque(maxlen=32)
        self._right_states: deque[Any] = deque(maxlen=32)
        if not self.config.isolated_control_process:
            self._left_pub = rospy.Publisher(
                self.config.left_command_topic,
                JointState,
                queue_size=1,
                tcp_nodelay=True,
            )
            self._right_pub = rospy.Publisher(
                self.config.right_command_topic,
                JointState,
                queue_size=1,
                tcp_nodelay=True,
            )
        self._enable_pub = rospy.Publisher(
            self.config.enable_topic, Bool, queue_size=1, tcp_nodelay=True
        )
        rospy.Subscriber(
            self.config.front_image_topic,
            Image,
            self._front_image_cb,
            queue_size=1,
            buff_size=2**24,
        )
        rospy.Subscriber(
            self.config.left_image_topic,
            Image,
            self._left_image_cb,
            queue_size=1,
            buff_size=2**24,
        )
        rospy.Subscriber(
            self.config.right_image_topic,
            Image,
            self._right_image_cb,
            queue_size=1,
            buff_size=2**24,
        )
        rospy.Subscriber(
            self.config.left_state_topic,
            JointState,
            self._left_state_cb,
            queue_size=50,
            tcp_nodelay=True,
        )
        rospy.Subscriber(
            self.config.right_state_topic,
            JointState,
            self._right_state_cb,
            queue_size=50,
            tcp_nodelay=True,
        )
        if self.config.isolated_control_process:
            if self.config.command_publish_hz <= 0:
                raise ValueError(
                    "isolated_control_process requires command_publish_hz > 0."
                )
            self._control_process = PiperControlProcess(asdict(self.config))
            self._logger.info(
                "Piper fixed-rate scheduler and ROS command publishers are "
                "running in an isolated process."
            )
        elif self.config.command_publish_hz > 0:
            self._command_thread = threading.Thread(
                target=self._run_command_interpolator,
                name="rlinf-piper-command-interpolator",
                daemon=True,
            )
            self._command_thread.start()

    def _front_image_cb(self, msg: Any) -> None:
        with self._observation_lock:
            self._front_images.append(msg)

    def _left_image_cb(self, msg: Any) -> None:
        with self._observation_lock:
            self._left_images.append(msg)

    def _right_image_cb(self, msg: Any) -> None:
        with self._observation_lock:
            self._right_images.append(msg)

    def _left_state_cb(self, msg: Any) -> None:
        with self._observation_lock:
            self._left_states.append(msg)

    def _right_state_cb(self, msg: Any) -> None:
        with self._observation_lock:
            self._right_states.append(msg)

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.config.ready_timeout_s
        while time.monotonic() < deadline and not self._rospy.is_shutdown():
            with self._observation_lock:
                if (
                    self._front_images
                    and self._left_images
                    and self._right_images
                    and self._left_states
                    and self._right_states
                ):
                    return
            time.sleep(0.02)
        raise RuntimeError("Timed out waiting for Piper cameras or joint-state topics.")

    def set_image_log_episode(self, episode_id: int | None) -> None:
        """Log the frames of the following observations under this episode.

        The env worker sets the RLT episode id before an action chunk and
        clears it afterwards, so idle and reset observations are not logged.
        """
        self._image_log_episode = None if episode_id is None else int(episode_id)

    def _decode(self, msg: Any) -> np.ndarray:
        return self._resize_for_policy(self._decode_raw(msg))

    def _decode_raw(self, msg: Any) -> np.ndarray:
        """Full-resolution BGR uint8 frame of one ROS image message."""
        # ROS Noetic's binary cv_bridge/OpenCV build targets the system Python.
        # Under RLinf's Python 3.11 + NumPy environment it can receive a valid
        # 640x480 message but hand an empty Mat to cvtColor.  The Astra topics
        # are uncompressed rgb8/bgr8 sensor_msgs/Image messages, so decode
        # those formats directly from their documented row stride.
        encoding = str(msg.encoding).lower()
        if encoding in {"rgb8", "bgr8"}:
            height, width, step = int(msg.height), int(msg.width), int(msg.step)
            expected_row_bytes = width * 3
            if height <= 0 or width <= 0 or step < expected_row_bytes:
                raise ValueError(
                    f"Invalid {encoding} image geometry: {width}x{height}, step={step}."
                )
            raw = np.frombuffer(msg.data, dtype=np.uint8)
            required = height * step
            if raw.size < required:
                raise ValueError(
                    f"Truncated {encoding} image: got {raw.size} bytes, expected at least {required}."
                )
            image = raw[:required].reshape(height, step)[:, :expected_row_bytes]
            image = image.reshape(height, width, 3)
            if encoding == "rgb8":
                image = image[..., ::-1].copy()  # match desired bgr8 behavior
        else:
            image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        return image

    def _resize_for_policy(self, image: np.ndarray) -> np.ndarray:
        # Keep the policy input byte-compatible with examples/piper_real.
        # OpenCV INTER_LINEAR and PIL BILINEAR are not equivalent: on a live
        # Astra frame almost 30% of channel values differed and the maximum
        # error reached 96. Fine insertion behavior changed measurably even
        # though both images looked alike to a person. The fixed-rate Piper
        # scheduler now runs in a separate process, so using the validated PIL
        # preprocessing path here cannot starve the 15-Hz/200-Hz control loop.
        image = image_tools.resize_with_pad(
            image,
            self.config.render_height,
            self.config.render_width,
        )
        return image_tools.convert_to_uint8(image)

    @staticmethod
    def _stamp_sec(msg: Any) -> float:
        return float(msg.header.stamp.to_sec())

    def _pop_synced(self, queue: deque[Any], frame_time: float) -> Any | None:
        while len(queue) > 1 and self._stamp_sec(queue[0]) < frame_time:
            queue.popleft()
        if not queue or self._stamp_sec(queue[0]) < frame_time:
            return None
        return queue.popleft()

    def _get_synced_messages(self) -> tuple[Any, Any, Any, Any, Any] | None:
        """Pop one temporally aligned triple-camera/joint-state observation."""
        with self._observation_lock:
            queues = (
                self._front_images,
                self._left_images,
                self._right_images,
                self._left_states,
                self._right_states,
            )
            if not all(queues):
                return None
            frame_time = min(self._stamp_sec(queue[-1]) for queue in queues)
            messages = tuple(self._pop_synced(queue, frame_time) for queue in queues)
        if any(message is None for message in messages):
            return None
        return messages  # type: ignore[return-value]

    def _get_observation(self) -> dict[str, Any]:
        observation_started = time.monotonic()
        self._wait_ready()
        sync_started = time.monotonic()
        deadline = time.monotonic() + self.config.ready_timeout_s
        synced = None
        while time.monotonic() < deadline and not self._rospy.is_shutdown():
            synced = self._get_synced_messages()
            if synced is not None:
                break
            time.sleep(0.01)
        if synced is None:
            raise RuntimeError(
                "Timed out synchronizing Piper image and joint-state topics."
            )
        sync_ms = (time.monotonic() - sync_started) * 1000.0
        front, left, right, left_state, right_state = synced
        state = np.concatenate(
            (np.asarray(left_state.position), np.asarray(right_state.position))
        ).astype(np.float32)
        if state.shape != (14,):
            raise ValueError(f"Expected 14-D Piper joint state, got {state.shape}.")
        decode_started = time.monotonic()
        raw = {
            "cam_high": self._decode_raw(front),
            "cam_left_wrist": self._decode_raw(left),
            "cam_right_wrist": self._decode_raw(right),
        }
        frames = {name: self._resize_for_policy(image) for name, image in raw.items()}
        if self._image_log is not None and self._image_log_episode is not None:
            self._image_log.submit(
                self._image_log_episode,
                state,
                {
                    name: raw[name]
                    for name in self.config.image_log_cameras
                    if name in raw
                },
            )
        decode_ms = (time.monotonic() - decode_started) * 1000.0
        head_pose = None
        detector = getattr(self, "_head_pose", None)
        if detector is not None:
            head_pose = detector(raw["cam_right_wrist"])
            self._observation_head_pose_ms_sum += detector.last_ms
        total_ms = (time.monotonic() - observation_started) * 1000.0
        self._observation_metric_count += 1
        self._observation_sync_ms_sum += sync_ms
        self._observation_decode_ms_sum += decode_ms
        self._observation_total_ms_max = max(
            self._observation_total_ms_max,
            total_ms,
        )
        if self._observation_metric_count >= 25:
            count = float(self._observation_metric_count)
            self._logger.info(
                "Piper observation timing over %d frames: "
                "sync_mean=%.2fms, decode_mean=%.2fms, head_pose_mean=%.2fms, "
                "total_max=%.2fms.",
                self._observation_metric_count,
                self._observation_sync_ms_sum / count,
                self._observation_decode_ms_sum / count,
                getattr(self, "_observation_head_pose_ms_sum", 0.0) / count,
                self._observation_total_ms_max,
            )
            self._observation_metric_count = 0
            self._observation_sync_ms_sum = 0.0
            self._observation_decode_ms_sum = 0.0
            self._observation_head_pose_ms_sum = 0.0
            self._observation_total_ms_max = 0.0
        observation = {
            "state": {"joint_position": state},
            "frames": frames,
        }
        if head_pose is not None:
            observation["head_pose"] = head_pose
        return observation

    def _reuse_last_frames(self, current: np.ndarray) -> dict[str, Any]:
        """Fresh proprio with the last decoded frames (and their head pose)."""
        observation = {
            "state": {"joint_position": current.copy()},
            "frames": self._last_observation["frames"],
        }
        if "head_pose" in self._last_observation:
            observation["head_pose"] = self._last_observation["head_pose"]
        return observation

    def _get_joint_position(self) -> np.ndarray:
        """Read the latest 14-D joint feedback without decoding camera frames."""
        self._wait_ready()
        with self._observation_lock:
            left_state = self._left_states[-1]
            right_state = self._right_states[-1]
        state = np.concatenate(
            (np.asarray(left_state.position), np.asarray(right_state.position))
        ).astype(np.float32)
        if state.shape != (14,):
            raise ValueError(f"Expected 14-D Piper joint state, got {state.shape}.")
        return state

    def _publish_action(self, action: np.ndarray) -> None:
        if getattr(self, "_control_process", None) is not None:
            self._control_process.publish_immediate(action)
            return
        stamp = self._rospy.Time.now()
        left, right = self._JointState(), self._JointState()
        left.header.stamp = stamp
        right.header.stamp = stamp
        left.position, right.position = action[:7].tolist(), action[7:].tolist()
        self._left_pub.publish(left)
        self._right_pub.publish(right)

    def _interpolated_command_locked(self, now: float) -> np.ndarray | None:
        if self._command_start is None or self._command_target is None:
            return None
        ratio = min(
            1.0,
            max(
                0.0,
                (now - self._command_start_time)
                / self.config.command_interpolation_duration_s,
            ),
        )
        command = self._command_start + ratio * (
            self._command_target - self._command_start
        )
        if not self.config.interpolate_gripper:
            command[[6, 13]] = (
                self._command_start[[6, 13]]
                if ratio < 1.0
                else self._command_target[[6, 13]]
            )
        return command.astype(np.float32, copy=False)

    def _set_command_target(
        self,
        target: np.ndarray,
        *,
        current_reference: np.ndarray,
        smooth: bool,
    ) -> None:
        """Set one policy target or publish it directly when interpolation is off.

        Direct publishing is the normal Piper deployment mode.  It must apply
        the same arm-only command filtering as the optional interpolator;
        otherwise enabling ``action_smoothing_alpha`` or
        ``max_arm_velocity_rad_s`` has no effect because of the early direct
        publish return.
        """
        if getattr(self, "_control_process", None) is not None:
            if smooth:
                raise RuntimeError(
                    "Scheduled policy actions must use the isolated control "
                    "process policy clock."
                )
            self._control_process.apply_reset_target(target, current_reference)
            return
        target = np.asarray(target, dtype=np.float32).copy()
        raw_target = target.copy()
        now = time.monotonic()
        arm_indices = np.r_[0:6, 7:13]
        with self._command_lock:
            previous_target = (
                None
                if self._last_smoothed_action is None
                else self._last_smoothed_action.copy()
            )
            if self.config.command_publish_hz <= 0:
                # Measured feedback is noisy and lags the command.  Reusing
                # it for every policy step turns that noise into a target
                # oscillation.  The last command is the continuous reference
                # for direct absolute-position control.
                current = (
                    self._last_smoothed_action.copy()
                    if self._last_smoothed_action is not None
                    else np.asarray(current_reference, dtype=np.float32).copy()
                )
                transition_duration_s = 1.0 / self.config.step_frequency
            else:
                current = self._interpolated_command_locked(now)
                if current is None:
                    current = np.asarray(current_reference, dtype=np.float32).copy()
                transition_duration_s = self.config.command_interpolation_duration_s

            if (
                smooth
                and not getattr(self, "_actor_direct_execution", False)
                and self._last_smoothed_action is not None
            ):
                alpha = self.config.action_smoothing_alpha
                target[arm_indices] = (
                    alpha * target[arm_indices]
                    + (1.0 - alpha) * self._last_smoothed_action[arm_indices]
                )

            if self.config.max_arm_velocity_rad_s > 0:
                max_delta = self.config.max_arm_velocity_rad_s * transition_duration_s
                target[arm_indices] = current[arm_indices] + np.clip(
                    target[arm_indices] - current[arm_indices],
                    -max_delta,
                    max_delta,
                )

            self._last_smoothed_action = target.copy()
            if smooth and previous_target is not None:
                raw_arm_action = raw_target[arm_indices]
                raw_delta = (
                    np.zeros_like(raw_arm_action)
                    if self._last_raw_arm_action is None
                    else raw_arm_action - self._last_raw_arm_action
                )
                filtered_delta = target[arm_indices] - previous_target[arm_indices]
                raw_step = float(np.max(np.abs(raw_delta)))
                filtered_step = float(np.max(np.abs(filtered_delta)))
                self._command_metric_count += 1
                self._raw_step_sum += raw_step
                self._raw_step_max = max(self._raw_step_max, raw_step)
                self._filtered_step_sum += filtered_step
                self._filtered_step_max = max(self._filtered_step_max, filtered_step)
                if self._last_raw_arm_delta is not None:
                    self._raw_accel_max = max(
                        self._raw_accel_max,
                        float(np.max(np.abs(raw_delta - self._last_raw_arm_delta))),
                    )
                self._last_raw_arm_delta = raw_delta.copy()
                if self._command_metric_count >= 100:
                    count = float(self._command_metric_count)
                    self._logger.info(
                        "Piper action diagnostics over %d targets: "
                        "raw_step_mean=%.5f, raw_step_max=%.5f, "
                        "filtered_step_mean=%.5f, filtered_step_max=%.5f, "
                        "raw_accel_max=%.5f rad.",
                        self._command_metric_count,
                        self._raw_step_sum / count,
                        self._raw_step_max,
                        self._filtered_step_sum / count,
                        self._filtered_step_max,
                        self._raw_accel_max,
                    )
                    self._command_metric_count = 0
                    self._raw_step_sum = 0.0
                    self._raw_step_max = 0.0
                    self._filtered_step_sum = 0.0
                    self._filtered_step_max = 0.0
                    self._raw_accel_max = 0.0
                self._last_raw_arm_action = raw_arm_action.copy()
            if self.config.command_publish_hz <= 0:
                self._publish_action(target)
                return
            self._command_start = current.copy()
            self._command_target = target
            self._command_start_time = now

    def _apply_low_level_action(self, action: np.ndarray) -> np.ndarray:
        """Apply one bounded policy-rate target without acquiring an image."""
        current = self._get_joint_position()
        target = np.clip(
            np.asarray(action, dtype=np.float32),
            self.config.joint_position_limits_lower,
            self.config.joint_position_limits_upper,
        )
        if (
            self.config.command_publish_hz <= 0
            and self.config.feedback_step_clamp_enabled
        ):
            # Direct publishing has no command-space velocity limiter, so keep
            # the feedback-relative step clamp as its final safety bound.
            max_delta = np.full(14, self.config.max_joint_delta, dtype=np.float32)
            max_delta[[6, 13]] = self.config.max_gripper_delta
            target = current + np.clip(target - current, -max_delta, max_delta)
        # With the high-rate interpolator enabled, do not clamp every new
        # target around measured feedback. Piper feedback trails the currently
        # published trajectory; feeding that lag back into the target causes
        # alternating forward/backward corrections. _set_command_target()
        # already limits motion in command space using max_arm_velocity_rad_s,
        # matching examples/piper_real while retaining absolute joint bounds.
        self._set_command_target(
            target,
            current_reference=current,
            smooth=not getattr(self, "_actor_direct_execution", False),
        )
        return current

    def _apply_scheduled_policy_action(
        self,
        action: np.ndarray,
        *,
        stop_event: threading.Event | None = None,
        source: str = "main",
    ) -> np.ndarray | None:
        """Apply one policy target on the shared fixed-rate control clock.

        Main chunk execution and the RTC bridge both call this method. The
        shared lock and deadline guarantee that only one policy target updates
        the 200-Hz interpolator at a time and that updates cannot arrive
        back-to-back at a chunk handoff.
        """
        if getattr(self, "_control_process", None) is not None:
            if stop_event is not None and stop_event.is_set():
                return None
            current_reference = self._get_joint_position()
            if getattr(self, "_actor_direct_execution", False):
                current = self._control_process.apply_action(
                    action, current_reference, smooth=False
                )
            else:
                current = self._control_process.apply_action(action, current_reference)
            accepted = getattr(self._control_process, "last_accepted_target", None)
            self._last_accepted_policy_target = (
                accepted.copy() if isinstance(accepted, np.ndarray) else None
            )
            profile = getattr(
                self._control_process, "last_actor_profile_diagnostics", None
            )
            if isinstance(profile, dict):
                count = getattr(self, "_actor_profile_diagnostic_count", 0) + 1
                self._actor_profile_diagnostic_count = count
                if count == 1 or count % 15 == 0:
                    self._logger.info(
                        "[Piper Actor profile] command trajectory: "
                        "arm_vmax=%.5frad/s arm_amax=%.5frad/s2 "
                        "accepted_goal-minus-published_max=%.5frad; "
                        "these are command values, not motor feedback.",
                        profile["arm_vmax"],
                        profile["arm_amax"],
                        profile["arm_goal_lag"],
                    )
            return current

        period = 1.0 / self.config.step_frequency
        with self._policy_action_lock:
            now = time.monotonic()
            deadline = max(now, self._next_policy_action_time)
            delay = deadline - now
            if delay > 0:
                if stop_event is not None:
                    if stop_event.wait(delay):
                        return None
                else:
                    time.sleep(delay)
            if stop_event is not None and stop_event.is_set():
                return None

            dispatch_time = time.monotonic()
            current = self._apply_low_level_action(action)
            if self._last_policy_dispatch_time > 0:
                interval = dispatch_time - self._last_policy_dispatch_time
                if interval > 1.5 * period:
                    self._slow_policy_interval_count += 1
                    if (
                        self._slow_policy_interval_count <= 10
                        or self._slow_policy_interval_count % 25 == 0
                    ):
                        self._logger.warning(
                            "Piper slow policy interval %d: %.2fms, %s->%s.",
                            self._slow_policy_interval_count,
                            interval * 1000.0,
                            self._last_policy_dispatch_source,
                            source,
                        )
                self._policy_dispatch_count += 1
                self._policy_interval_sum += interval
                self._policy_interval_max_error = max(
                    self._policy_interval_max_error,
                    abs(interval - period),
                )
                if self._policy_dispatch_count >= 100:
                    mean_interval = (
                        self._policy_interval_sum / self._policy_dispatch_count
                    )
                    self._logger.info(
                        "Piper policy scheduler over %d intervals: "
                        "rate=%.2fHz, mean_interval=%.2fms, "
                        "max_period_error=%.2fms.",
                        self._policy_dispatch_count,
                        1.0 / mean_interval,
                        mean_interval * 1000.0,
                        self._policy_interval_max_error * 1000.0,
                    )
                    self._policy_dispatch_count = 0
                    self._policy_interval_sum = 0.0
                    self._policy_interval_max_error = 0.0
            self._last_policy_dispatch_time = dispatch_time
            self._last_policy_dispatch_source = source
            next_deadline = deadline + period
            self._next_policy_action_time = (
                next_deadline
                if dispatch_time <= next_deadline
                else dispatch_time + period
            )
            return current

    def begin_action_chunk(self, num_actions: int) -> None:
        """Mark a physical action chunk and reset its subsample counter."""
        if num_actions <= 0:
            raise ValueError("num_actions must be positive.")
        self._remaining_actions_in_chunk = int(num_actions)
        self._chunk_decode_index = 0

    def set_actor_execution_mode(self, active: bool) -> None:
        """Bypass optional policy filters only for an Actor-selected chunk."""
        active = bool(active)
        if active and not getattr(self, "_actor_direct_execution", False):
            self._queued_rtc_tail = None
            self._stop_rtc_tail()
            self._logger.info(
                "Direct Actor execution: residual/executor EMA and RTC bypassed; "
                "joint bounds, velocity limits and command interpolation retained."
            )
            if self.config.actor_command_profile_enabled:
                self._logger.info(
                    "Actor previous-command goal limits: %.5frad / %.5fm; "
                    "publish tracker vmax=%.3frad/s amax=%.3frad/s2 at %.1fHz.",
                    self.config.actor_command_joint_step_rad,
                    self.config.actor_command_gripper_step_m,
                    self.config.actor_command_joint_velocity_rad_s,
                    self.config.actor_command_joint_acceleration_rad_s2,
                    self.config.command_publish_hz,
                )
        self._actor_direct_execution = active

    def queue_rtc_tail(self, reference_tail: np.ndarray) -> None:
        """Queue the VLA suffix to start at the final action's next deadline.

        EnvWorker supplies the suffix before entering synchronous ``chunk_step``.
        Starting it from :meth:`step` prevents final camera decoding and result
        packaging from leaving a full policy-rate command hole at every chunk
        boundary.
        """
        if (
            not self.config.rtc_enabled
            or self.config.is_dummy
            or getattr(self, "_actor_direct_execution", False)
        ):
            return
        tail = np.asarray(reference_tail, dtype=np.float32)
        if tail.ndim != 2 or tail.shape[1] != 14:
            raise ValueError(f"Expected queued RTC tail [H, 14], got {tail.shape}.")
        self._queued_rtc_tail = tail.copy()

    def _blend_action_chunk_boundary(self, action: np.ndarray) -> np.ndarray:
        """Blend only arm targets at a newly predicted action-chunk boundary.

        Every within-chunk action remains unchanged. Unlike a per-step
        low-pass filter, this adds no persistent phase lag. Gripper targets
        retain their discrete timing.
        """
        action = np.asarray(action, dtype=np.float32).copy()
        blend_steps = self.config.action_chunk_boundary_blend_steps
        if getattr(self, "_actor_direct_execution", False):
            blend_steps = 0
        if (
            blend_steps > 0
            and self._completed_action_chunks > 0
            and self._chunk_action_index < blend_steps
            and self._last_policy_action is not None
        ):
            arm_indices = np.r_[0:6, 7:13]
            weight = (self._chunk_action_index + 1) / blend_steps
            action[arm_indices] = (1.0 - weight) * self._last_policy_action[
                arm_indices
            ] + weight * action[arm_indices]

        self._last_policy_action = action.copy()
        self._chunk_action_index += 1
        if self._chunk_action_index == self.config.action_chunk_size:
            self._chunk_action_index = 0
            self._completed_action_chunks += 1
        return action

    def _stop_rtc_tail(self, *, drain: bool = False) -> tuple[int, np.ndarray | None]:
        if getattr(self, "_control_process", None) is not None:
            return self._control_process.stop_rtc(drain=drain)
        self._rtc_stop.set()
        thread = self._rtc_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        with self._rtc_lock:
            executed = self._rtc_executed_steps
            last_action = (
                None if self._rtc_last_action is None else self._rtc_last_action.copy()
            )
            self._rtc_thread = None
            self._rtc_executed_steps = 0
            self._rtc_last_action = None
        return executed, last_action

    def start_rtc_tail(self, reference_tail: np.ndarray) -> None:
        """Execute a bounded VLA suffix while the next chunk is inferred."""
        if (
            not self.config.rtc_enabled
            or self.config.is_dummy
            or getattr(self, "_actor_direct_execution", False)
        ):
            return
        self._stop_rtc_tail()
        tail = np.asarray(reference_tail, dtype=np.float32)
        if tail.ndim != 2 or tail.shape[1] != 14:
            raise ValueError(f"Expected RTC tail [H, 14], got {tail.shape}.")
        tail = tail[: self.config.rtc_max_bridge_steps].copy()
        if tail.shape[0] == 0:
            return
        if getattr(self, "_control_process", None) is not None:
            self._control_process.start_rtc(tail)
            self._rtc_rollover_count += 1
            if self._rtc_rollover_count <= 3 or self._rtc_rollover_count % 25 == 0:
                self._logger.info(
                    "Piper isolated RTC started VLA bridge %d with %d "
                    "available tail steps.",
                    self._rtc_rollover_count,
                    tail.shape[0],
                )
            return

        self._rtc_stop.clear()
        with self._rtc_lock:
            self._rtc_executed_steps = 0
            self._rtc_last_action = None

        def _run() -> None:
            exhausted = True
            for action in tail:
                if self._rtc_stop.is_set() or self._rospy.is_shutdown():
                    exhausted = False
                    break
                current = self._apply_scheduled_policy_action(
                    action,
                    stop_event=self._rtc_stop,
                    source="rtc",
                )
                if current is None:
                    exhausted = False
                    with self._command_lock:
                        interpolated = self._interpolated_command_locked(
                            time.monotonic()
                        )
                    if interpolated is not None:
                        with self._rtc_lock:
                            self._rtc_last_action = interpolated.copy()
                    break
                with self._rtc_lock:
                    self._rtc_executed_steps += 1
                    self._rtc_last_action = action.copy()
            if exhausted and not self._rtc_stop.is_set():
                self._logger.warning(
                    "Piper RTC exhausted all %d bridge steps before the next "
                    "chunk arrived; the final target will be held.",
                    tail.shape[0],
                )

        self._rtc_thread = threading.Thread(
            target=_run,
            name="rlinf-piper-rtc-tail",
            daemon=True,
        )
        self._rtc_thread.start()
        self._rtc_rollover_count += 1
        if self._rtc_rollover_count <= 3 or self._rtc_rollover_count % 25 == 0:
            self._logger.info(
                "Piper RTC started VLA bridge %d with %d available tail steps.",
                self._rtc_rollover_count,
                tail.shape[0],
            )

    def align_rtc_chunk(
        self,
        actions: np.ndarray,
        reference_plan: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        """Stop the bridge and time-align the newly returned action chunk."""
        actions = np.asarray(actions, dtype=np.float32)
        reference_plan = np.asarray(reference_plan, dtype=np.float32)
        # Keep the full old VLA suffix running even when the next JAX response
        # is already available. This is the same temporal contract as the
        # validated examples/piper_real RTC broker: the new plan is aligned to
        # the complete bridge duration, not promoted midway through it.
        direct_actor = getattr(self, "_actor_direct_execution", False)
        executed, last_action = self._stop_rtc_tail(drain=not direct_actor)
        if direct_actor:
            return actions, 0
        if not self.config.rtc_enabled or executed <= 0 or actions.ndim != 2:
            action_trace = getattr(self, "_action_trace", None)
            if action_trace is not None:
                action_trace.schedule(
                    "rtc_alignment",
                    raw_chunk=actions,
                    reference_plan=reference_plan,
                    aligned_chunk=actions,
                    bridge_last_action=last_action,
                    bridge_executed_steps=np.asarray([executed], dtype=np.int64),
                    applied_offset_steps=np.asarray([0], dtype=np.int64),
                    blend_steps=np.asarray([0], dtype=np.int64),
                )
            return actions, 0

        chunk_len = actions.shape[0]
        if reference_plan.ndim != 2 or reference_plan.shape[1] != actions.shape[1]:
            raise ValueError(
                "RTC reference plan must match the action dimension, got "
                f"{reference_plan.shape} and {actions.shape}."
            )
        # Actor actions cover C steps.  The native VLA suffix supplies the
        # future samples required after shifting by the bridge duration.
        source = np.concatenate(
            (actions, reference_plan[chunk_len:]),
            axis=0,
        )
        offset = min(executed, max(0, source.shape[0] - chunk_len))
        aligned = source[offset : offset + chunk_len].copy()
        if aligned.shape[0] != chunk_len:
            raise RuntimeError(
                f"RTC plan is too short after offset {offset}: {source.shape}."
            )

        blend_steps = min(
            self.config.rtc_boundary_blend_steps,
            aligned.shape[0],
        )
        if blend_steps > 0 and last_action is not None:
            arm_indices = np.r_[0:6, 7:13]
            weights = np.arange(1, blend_steps + 1, dtype=np.float32) / blend_steps
            aligned[:blend_steps, arm_indices] = (1.0 - weights[:, None]) * last_action[
                arm_indices
            ] + weights[:, None] * aligned[:blend_steps, arm_indices]
        if self._rtc_rollover_count <= 3 or self._rtc_rollover_count % 25 == 0:
            self._logger.info(
                "Piper RTC aligned the new chunk by %d executed bridge steps "
                "and blended %d arm steps.",
                offset,
                blend_steps,
            )
        action_trace = getattr(self, "_action_trace", None)
        if action_trace is not None:
            action_trace.schedule(
                "rtc_alignment",
                raw_chunk=actions,
                reference_plan=reference_plan,
                aligned_chunk=aligned,
                bridge_last_action=last_action,
                bridge_executed_steps=np.asarray([executed], dtype=np.int64),
                applied_offset_steps=np.asarray([offset], dtype=np.int64),
                blend_steps=np.asarray([blend_steps], dtype=np.int64),
            )
        return aligned, offset

    def _run_command_interpolator(self) -> None:
        """Continuously publish the latest interpolated target at the ROS rate."""
        period = 1.0 / self.config.command_publish_hz
        next_publish = time.monotonic()
        while not self._command_stop.is_set() and not self._rospy.is_shutdown():
            with self._command_lock:
                command = self._interpolated_command_locked(time.monotonic())
            if command is not None:
                self._publish_action(command)
            next_publish += period
            remaining = next_publish - time.monotonic()
            if remaining <= 0:
                next_publish = time.monotonic()
                continue
            self._command_stop.wait(remaining)

    def hold_position(self) -> None:
        """Publish measured joint positions once; safe emergency stop primitive."""
        if not self.config.is_dummy:
            self._queued_rtc_tail = None
            self._stop_rtc_tail()
            current = self._get_joint_position()
            if self.config.command_publish_hz > 0:
                with self._command_lock:
                    self._command_start = current.copy()
                    self._command_target = current.copy()
                    self._command_start_time = time.monotonic()
                    self._last_smoothed_action = current.copy()
            self._publish_action(current)

    def suspend_policy_output(self) -> None:
        """Stop RTC and policy publishers before the mode-2 bridge takes over."""
        if self.config.is_dummy:
            return
        self._queued_rtc_tail = None
        self._stop_rtc_tail()
        if getattr(self, "_control_process", None) is not None:
            self._control_process.suspend()
        else:
            raise RuntimeError(
                "Piper mode 2 requires isolated_control_process so policy "
                "publication can be suspended without stopping ROS."
            )

    def resume_policy_output(self) -> None:
        """Reset the executor from feedback after confirmed mux INFERENCE."""
        self._queued_rtc_tail = None
        self._last_policy_action = None
        self._chunk_action_index = 0
        if not self.config.is_dummy:
            self._control_process.reset(self._get_joint_position())

    def read_observation(self) -> dict[str, Any]:
        """Read fresh cameras/joints without motion or an environment step."""
        observation = (
            self.observation_space.sample()
            if self.config.is_dummy
            else self._get_observation()
        )
        self._last_observation = observation
        return observation

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable both Piper slaves through the shared ROS topic."""
        if self.config.is_dummy:
            return
        for _ in range(max(1, self.config.enable_pulses)):
            self._enable_pub.publish(self._Bool(data=enabled))
            time.sleep(self.config.enable_interval_s)

    def capture_episode_start_position(self) -> bool:
        """Capture the first episode's measured pose as the persistent reset pose.

        Returns ``True`` only when a pose is captured.  Later episodes never
        overwrite it, so drift or an aborted rollout cannot redefine home.
        """
        if self._episode_start_position is not None:
            return False
        if self.config.is_dummy:
            self._episode_start_position = np.zeros(14, dtype=np.float32)
        else:
            self._episode_start_position = self._get_joint_position().copy()
        return True

    def return_to_episode_start(
        self, stop_requested: Callable[[], bool] | None = None
    ) -> bool:
        """Move both arms back to the captured start pose at the normal safe limits.

        This method is called only while the wrapper is idle, so its commands
        are excluded from rollout/replay data.  ``False`` means the operator
        aborted with ``q``; timeout or a missing start pose raises an error.
        """
        if self._episode_start_position is None:
            raise RuntimeError(
                "No episode start pose has been captured; start one episode with 'a' first."
            )
        if self.config.is_dummy:
            return True

        goal = self._episode_start_position.copy()
        arm_indices = np.r_[0:6, 7:13]
        gripper_indices = np.asarray([6, 13])
        open_goal = goal.copy()
        open_goal[gripper_indices] = self.config.reset_open_gripper_position_m
        closed_goal = goal.copy()
        closed_goal[gripper_indices] = self.config.reset_closed_gripper_position_m
        grippers_opened = False
        reached_open_reset_pose = False
        deadline = time.monotonic() + self.config.reset_timeout_s
        period = 1.0 / self.config.step_frequency
        next_step = time.monotonic()
        while time.monotonic() < deadline and not self._rospy.is_shutdown():
            if stop_requested is not None and stop_requested():
                self.hold_position()
                return False

            current = self._get_joint_position()
            arm_done = np.all(
                np.abs(goal[arm_indices] - current[arm_indices])
                <= self.config.reset_position_tolerance_rad
            )
            open_done = np.all(
                np.abs(open_goal[gripper_indices] - current[gripper_indices])
                <= self.config.reset_gripper_tolerance_m
            )
            if open_done:
                grippers_opened = True
            if grippers_opened and arm_done and open_done:
                reached_open_reset_pose = True

            closed_done = np.all(
                np.abs(closed_goal[gripper_indices] - current[gripper_indices])
                <= self.config.reset_gripper_tolerance_m
            )
            if reached_open_reset_pose and arm_done and closed_done:
                self._set_command_target(
                    closed_goal,
                    current_reference=current,
                    smooth=False,
                )
                return True

            max_delta = np.full(14, self.config.max_joint_delta, dtype=np.float32)
            max_delta[gripper_indices] = self.config.max_gripper_delta
            if reached_open_reset_pose:
                active_goal = closed_goal
            elif grippers_opened:
                # Keep both grippers fully open throughout the arm motion.
                active_goal = open_goal
            else:
                # First open both grippers while holding the measured arm pose;
                # only then begin the return motion.
                active_goal = current.copy()
                active_goal[gripper_indices] = open_goal[gripper_indices]
            requested = current + np.clip(active_goal - current, -max_delta, max_delta)
            self._set_command_target(
                requested,
                current_reference=current,
                smooth=False,
            )

            next_step += period
            remaining = next_step - time.monotonic()
            if remaining > 0:
                time.sleep(remaining)

        self.hold_position()
        raise TimeoutError(
            f"Piper did not reach the captured start pose within {self.config.reset_timeout_s:.1f}s."
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        preserve_suspended = bool(
            (options or {}).get("preserve_suspended_policy_output", False)
        )
        self._num_steps = 0
        if self.config.is_dummy:
            observation = self.observation_space.sample()
        else:
            if self.config.enable_on_reset:
                self.set_enabled(True)
            observation = self._get_observation()
            current = observation["state"]["joint_position"]
            with self._command_lock:
                self._last_smoothed_action = None
                self._last_raw_arm_action = None
                self._last_raw_arm_delta = None
                self._observation_metric_count = 0
                self._observation_sync_ms_sum = 0.0
                self._observation_decode_ms_sum = 0.0
                self._observation_head_pose_ms_sum = 0.0
                self._observation_total_ms_max = 0.0
                self._command_metric_count = 0
                self._raw_step_sum = 0.0
                self._raw_step_max = 0.0
                self._filtered_step_sum = 0.0
                self._filtered_step_max = 0.0
                self._raw_accel_max = 0.0
                self._last_policy_action = None
                self._chunk_action_index = 0
                self._completed_action_chunks = 0
                self._remaining_actions_in_chunk = 0
                self._queued_rtc_tail = None
                self._next_policy_action_time = 0.0
                self._last_policy_dispatch_time = 0.0
                self._last_policy_dispatch_source = None
                self._slow_policy_interval_count = 0
                self._policy_dispatch_count = 0
                self._policy_interval_sum = 0.0
                self._policy_interval_max_error = 0.0
                if self.config.command_publish_hz > 0:
                    self._command_start = current.copy()
                    self._command_target = current.copy()
                    self._command_start_time = time.monotonic()
            if (
                getattr(self, "_control_process", None) is not None
                and not preserve_suspended
            ):
                self._control_process.reset(current)
        self._last_observation = observation
        return observation, {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape != (14,):
            raise ValueError(
                f"Expected a 14-D absolute Piper joint action, got {action.shape}."
            )
        if self.config.is_dummy:
            observation = self.observation_space.sample()
        else:
            # Only joint feedback is needed for the safety clamp before
            # publishing. A full camera observation is decoded only at the end
            # of an action chunk.
            current = self._apply_scheduled_policy_action(
                self._blend_action_chunk_boundary(action),
            )
            assert current is not None
            remaining_actions = getattr(self, "_remaining_actions_in_chunk", 0)
            is_chunk_final_step = remaining_actions <= 1
            if remaining_actions > 0:
                self._remaining_actions_in_chunk -= 1
            # RLT subsamples action chunks: it needs a real observation every
            # ``rlt_subsample_stride`` control steps, not only at the chunk
            # boundary, so that <x_t, a_{t:t+C}> rows can be built.
            step_index = getattr(self, "_chunk_decode_index", 0)
            self._chunk_decode_index = step_index + 1
            stride = self.config.rlt_subsample_stride
            # The observation after action s is the start state of window s + 1.
            is_subsample_step = stride > 0 and (step_index + 1) % stride == 0
            if (
                is_chunk_final_step
                or is_subsample_step
                or self._last_observation is None
            ):
                if is_chunk_final_step:
                    # Start the first suffix target on the already-established
                    # policy clock before the synchronous triple-camera decode.
                    # Previously EnvWorker started RTC only after chunk_step
                    # returned, producing a 130-150 ms command interval every
                    # 20 actions despite a nominal 66.7 ms period. A stride
                    # decode must not start it: the suffix would then run
                    # alongside the remaining actions of this chunk.
                    queued_tail = self._queued_rtc_tail
                    self._queued_rtc_tail = None
                    if queued_tail is not None:
                        self.start_rtc_tail(queued_tail)
                observation = self._get_observation()
            else:
                # Intermediate camera decodes dominate the control period on
                # the Piper host. Reuse the most recent complete frame while
                # preserving the current proprioception; the final action in
                # every chunk always obtains a fresh synchronized observation.
                observation = self._reuse_last_frames(current)
        self._last_observation = observation
        self._num_steps += 1
        accepted = getattr(self, "_last_accepted_policy_target", None)
        if not self.config.is_dummy and getattr(self, "_control_process", None) is None:
            accepted = getattr(self, "_last_smoothed_action", None)
        info = {} if accepted is None else {"piper_accepted_target": accepted.copy()}
        if accepted is not None:
            info["piper_execution_filter_params"] = np.asarray(
                [
                    1.0
                    if getattr(self, "_actor_direct_execution", False)
                    else self.config.action_smoothing_alpha,
                    self.config.max_arm_velocity_rad_s,
                    self.config.command_interpolation_duration_s,
                ],
                dtype=np.float32,
            )
        return (
            observation,
            0.0,
            False,
            self._num_steps >= self.config.max_num_steps,
            info,
        )

    def step_external_control(self):
        """Observe one step without publishing while mode-2 owns the followers.

        The independent mode-2 ROS bridge publishes the leader pose directly
        to the followers at 200 Hz. Calling the normal :meth:`step` here would
        create a second command source and would also wake the suspended policy
        executor. The wrapper still supplies the measured leader action as
        ``intervene_action`` for replay.
        """
        if self.config.is_dummy:
            observation = self.observation_space.sample()
        else:
            current = self._get_joint_position()
            remaining_actions = getattr(self, "_remaining_actions_in_chunk", 0)
            is_chunk_final_step = remaining_actions <= 1
            if remaining_actions > 0:
                self._remaining_actions_in_chunk -= 1
            # Human-driven steps are replayed too, so stride windows need the
            # same fresh intermediate observations as policy-driven steps.
            step_index = getattr(self, "_chunk_decode_index", 0)
            self._chunk_decode_index = step_index + 1
            stride = self.config.rlt_subsample_stride
            is_subsample_step = stride > 0 and (step_index + 1) % stride == 0
            self._queued_rtc_tail = None
            if (
                is_chunk_final_step
                or is_subsample_step
                or self._last_observation is None
            ):
                observation = self._get_observation()
            else:
                observation = self._reuse_last_frames(current)
        self._last_observation = observation
        self._num_steps += 1
        return observation, 0.0, False, self._num_steps >= self.config.max_num_steps, {}

    def close(self) -> None:
        """Stop publishing new targets when the environment is closed."""
        if self.config.is_dummy:
            return
        if self._image_log is not None:
            self._image_log.close()
            self._image_log = None
        self._stop_rtc_tail()
        self._command_stop.set()
        if self._command_thread is not None:
            self._command_thread.join(timeout=1.0)
        if self._last_observation is not None and not self.config.four_can_mux_enabled:
            current = self._get_joint_position()
            self._publish_action(current)
        if getattr(self, "_control_process", None) is not None:
            self._control_process.close()
            self._control_process = None
