"""Process-isolated real-time command executor for the dual Piper robot."""

from __future__ import annotations

import ctypes
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
from collections import deque
from typing import Any

import numpy as np

from rlinf.envs.realworld.piper.action_trace import PiperActionTraceWriter
from rlinf.envs.realworld.piper.actor_command_profile import ActorCommandProfile


def _exit_with_parent(expected_parent_pid: int) -> None:
    """Ask Linux to terminate this executor when its EnvWorker exits."""
    libc = ctypes.CDLL(None, use_errno=True)
    pr_set_pdeathsig = 1
    if libc.prctl(pr_set_pdeathsig, signal.SIGTERM) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, "Failed to configure Piper parent-death signal.")
    # The parent can exit between Process.start() and prctl(). Detect that
    # race explicitly instead of leaving an orphan ROS command publisher.
    if os.getppid() != expected_parent_pid:
        raise RuntimeError(
            "Piper EnvWorker exited before the control executor initialized."
        )


def _control_process_main(
    config: dict[str, Any],
    command_queue: Any,
    response_queue: Any,
    expected_parent_pid: int,
) -> None:
    """Own the policy clock and ROS publishers outside the Ray EnvWorker."""
    _exit_with_parent(expected_parent_pid)

    import rospy
    from sensor_msgs.msg import JointState

    rospy.init_node(
        "rlinf_piper_control_executor",
        anonymous=True,
        disable_signals=True,
    )
    left_pub = rospy.Publisher(
        config["left_command_topic"],
        JointState,
        queue_size=1,
        tcp_nodelay=True,
    )
    right_pub = rospy.Publisher(
        config["right_command_topic"],
        JointState,
        queue_size=1,
        tcp_nodelay=True,
    )

    # Match OpenPI: HOLD ACK is sent only after the final policy publication
    # drains. Keep this gate in the executor, independent of Ray's GIL/RPCs.
    mux_enabled = bool(config.get("four_can_mux_enabled", False))
    hold_lock = threading.Lock()
    mux_hold = threading.Event()
    hold_changed = threading.Event()
    hold_ack = None
    if mux_enabled:
        from std_msgs.msg import Bool

        mux_hold.set()
        hold_ack = rospy.Publisher(
            "/teleop/policy_hold_ack", Bool, queue_size=1, latch=True
        )
        hold_ack.publish(Bool(data=False))

        def hold_callback(message) -> None:
            with hold_lock:
                if message.data:
                    mux_hold.set()
                else:
                    mux_hold.clear()
                hold_changed.set()

        rospy.Subscriber("/teleop/hold_active", Bool, hold_callback, queue_size=1)

    publish_period = 1.0 / float(config["command_publish_hz"])
    policy_period = 1.0 / float(config["step_frequency"])
    interpolation_duration = float(config["command_interpolation_duration_s"])
    arm_indices = np.r_[0:6, 7:13]
    lower = np.asarray(config["joint_position_limits_lower"], dtype=np.float32)
    upper = np.asarray(config["joint_position_limits_upper"], dtype=np.float32)
    smoothing_alpha = float(config["action_smoothing_alpha"])
    max_velocity = float(config["max_arm_velocity_rad_s"])
    interpolate_gripper = bool(config["interpolate_gripper"])

    command_start: np.ndarray | None = None
    command_target: np.ndarray | None = None
    command_start_time = 0.0
    last_smoothed: np.ndarray | None = None
    actor_profile: ActorCommandProfile | None = None
    next_policy_time = 0.0
    pending_main: deque[tuple[int, np.ndarray, np.ndarray, bool, bool]] = deque()
    rtc_actions: deque[np.ndarray] = deque()
    rtc_executed = 0
    rtc_last_action: np.ndarray | None = None
    action_trace = PiperActionTraceWriter.from_environment("control")
    bridge_capture_active = False
    bridge_initial_actions: np.ndarray | None = None
    bridge_started_ns = 0
    bridge_publish_times_ns: list[int] = []
    bridge_published_actions: list[np.ndarray] = []
    bridge_target_times_ns: list[int] = []
    bridge_targets: list[np.ndarray] = []
    drain_request_id: int | None = None
    publishing_enabled = True
    running = True

    def finish_rtc(request_id: int) -> None:
        """Return the completed bridge prefix and close its diagnostic capture."""
        nonlocal bridge_capture_active, rtc_executed, rtc_last_action
        if bridge_capture_active:
            action_trace.schedule(
                "rtc_bridge",
                bridge_started_ns=np.asarray([bridge_started_ns], dtype=np.int64),
                bridge_stopped_ns=np.asarray([time.monotonic_ns()], dtype=np.int64),
                bridge_initial_actions=bridge_initial_actions,
                bridge_executed_steps=np.asarray([rtc_executed], dtype=np.int64),
                bridge_last_action=rtc_last_action,
                target_times_ns=np.asarray(bridge_target_times_ns, dtype=np.int64),
                targets=np.asarray(bridge_targets, dtype=np.float32),
                publish_times_ns=np.asarray(bridge_publish_times_ns, dtype=np.int64),
                published_actions=np.asarray(
                    bridge_published_actions, dtype=np.float32
                ),
            )
        bridge_capture_active = False
        response_queue.put(
            (
                request_id,
                "ok",
                (
                    rtc_executed,
                    None if rtc_last_action is None else rtc_last_action.copy(),
                ),
            )
        )
        rtc_executed = 0
        rtc_last_action = None

    def interpolated(now: float) -> np.ndarray | None:
        if actor_profile is not None:
            return actor_profile.sample(now)
        if command_start is None or command_target is None:
            return None
        ratio = min(
            1.0,
            max(0.0, (now - command_start_time) / interpolation_duration),
        )
        result = command_start + ratio * (command_target - command_start)
        if not interpolate_gripper:
            result[[6, 13]] = (
                command_start[[6, 13]] if ratio < 1.0 else command_target[[6, 13]]
            )
        return result.astype(np.float32, copy=False)

    def publish(action: np.ndarray) -> None:
        with hold_lock:
            if mux_enabled and mux_hold.is_set():
                return
            publish_unlocked(action)

    def publish_unlocked(action: np.ndarray) -> None:
        if bridge_capture_active:
            bridge_publish_times_ns.append(time.monotonic_ns())
            bridge_published_actions.append(np.asarray(action, dtype=np.float32).copy())
        stamp = rospy.Time.now()
        left, right = JointState(), JointState()
        left.header.stamp = stamp
        right.header.stamp = stamp
        left.position = action[:7].tolist()
        right.position = action[7:].tolist()
        left_pub.publish(left)
        right_pub.publish(right)

    def set_target(
        raw_action: np.ndarray,
        current_reference: np.ndarray,
        now: float,
        *,
        smooth: bool,
        actor: bool = False,
    ) -> np.ndarray:
        nonlocal command_start, command_target, command_start_time, last_smoothed
        nonlocal actor_profile
        target = np.clip(raw_action, lower, upper).astype(np.float32, copy=True)
        current = interpolated(now)
        if current is None:
            current = current_reference.astype(np.float32, copy=True)
        if actor and bool(config.get("actor_command_profile_enabled", False)):
            if actor_profile is None:
                actor_profile = ActorCommandProfile(config, current, now)
            target = actor_profile.set_goal(target)
        else:
            actor_profile = None
        if smooth and last_smoothed is not None:
            target[arm_indices] = (
                smoothing_alpha * target[arm_indices]
                + (1.0 - smoothing_alpha) * last_smoothed[arm_indices]
            )
        if max_velocity > 0 and actor_profile is None:
            max_delta = max_velocity * interpolation_duration
            target[arm_indices] = current[arm_indices] + np.clip(
                target[arm_indices] - current[arm_indices],
                -max_delta,
                max_delta,
            )
        last_smoothed = target.copy()
        command_start = current.copy()
        command_target = target
        command_start_time = now
        if bridge_capture_active:
            bridge_target_times_ns.append(time.monotonic_ns())
            bridge_targets.append(target.copy())
        return current

    response_queue.put((0, "ready", None))
    next_publish = time.monotonic()
    while running and not rospy.is_shutdown():
        if mux_enabled and hold_changed.is_set():
            with hold_lock:
                hold_changed.clear()
                active = mux_hold.is_set()
                if active:
                    for pending_id, _, _, _, _ in pending_main:
                        response_queue.put(
                            (
                                pending_id,
                                "error",
                                "policy output is suspended by mux HOLD",
                            )
                        )
                    pending_main.clear()
                    rtc_actions.clear()
                    command_start = command_target = last_smoothed = None
                    actor_profile = None
                    publishing_enabled = False
                    if drain_request_id is not None:
                        finish_rtc(drain_request_id)
                        drain_request_id = None
                # No publisher can run concurrently with this ACK.
                hold_ack.publish(Bool(data=active))
        while True:
            try:
                request_id, operation, payload = command_queue.get_nowait()
            except queue.Empty:
                break

            if operation in {
                "action",
                "action_diagnostics",
                "actor_action_diagnostics",
            }:
                if not publishing_enabled or (mux_enabled and mux_hold.is_set()):
                    response_queue.put(
                        (request_id, "error", "policy output is suspended")
                    )
                    continue
                action, current_reference = payload
                pending_main.append(
                    (
                        request_id,
                        np.asarray(action, dtype=np.float32),
                        np.asarray(current_reference, dtype=np.float32),
                        operation != "action",
                        operation != "actor_action_diagnostics",
                    )
                )
            elif operation == "reset_target":
                action, current_reference = payload
                current = set_target(
                    np.asarray(action, dtype=np.float32),
                    np.asarray(current_reference, dtype=np.float32),
                    time.monotonic(),
                    smooth=False,
                )
                response_queue.put((request_id, "ok", current))
            elif operation == "start_rtc":
                if not publishing_enabled:
                    response_queue.put(
                        (request_id, "error", "policy output is suspended")
                    )
                    continue
                rtc_actions.clear()
                drain_request_id = None
                bridge_initial_actions = np.asarray(payload, dtype=np.float32)[
                    : int(config["rtc_max_bridge_steps"])
                ].copy()
                rtc_actions.extend(bridge_initial_actions)
                rtc_executed = 0
                rtc_last_action = None
                bridge_capture_active = action_trace.enabled
                bridge_started_ns = time.monotonic_ns()
                bridge_publish_times_ns.clear()
                bridge_published_actions.clear()
                bridge_target_times_ns.clear()
                bridge_targets.clear()
                response_queue.put((request_id, "ok", None))
            elif operation == "stop_rtc":
                rtc_actions.clear()
                drain_request_id = None
                finish_rtc(request_id)
            elif operation == "drain_rtc":
                # Match the validated Piper RTC broker: once a request has
                # started, finish the complete old suffix before promoting the
                # newly inferred chunk. Switching after only the first few
                # suffix targets makes two independently sampled VLA plans
                # fight at the boundary.
                if rtc_actions:
                    drain_request_id = request_id
                else:
                    finish_rtc(request_id)
            elif operation in {"immediate", "reset"}:
                actor_profile = None
                action = np.asarray(payload, dtype=np.float32)
                command_start = action.copy()
                command_target = action.copy()
                command_start_time = time.monotonic()
                last_smoothed = action.copy()
                publishing_enabled = not (mux_enabled and mux_hold.is_set())
                if operation == "reset":
                    pending_main.clear()
                    rtc_actions.clear()
                    rtc_executed = 0
                    rtc_last_action = None
                    next_policy_time = 0.0
                    bridge_capture_active = False
                    drain_request_id = None
                publish(action)
                response_queue.put((request_id, "ok", None))
            elif operation == "suspend":
                actor_profile = None
                pending_main.clear()
                rtc_actions.clear()
                rtc_executed = 0
                rtc_last_action = None
                bridge_capture_active = False
                drain_request_id = None
                publishing_enabled = False
                response_queue.put((request_id, "ok", None))
            elif operation == "close":
                pending_main.clear()
                rtc_actions.clear()
                drain_request_id = None
                running = False
                response_queue.put((request_id, "ok", None))
                break
            else:
                response_queue.put(
                    (request_id, "error", f"Unknown control operation: {operation}")
                )

        now = time.monotonic()
        if (pending_main or rtc_actions) and (
            next_policy_time <= 0 or now >= next_policy_time
        ):
            deadline = now if next_policy_time <= 0 else next_policy_time
            if pending_main:
                request_id, action, current_reference, diagnostics, smooth = (
                    pending_main.popleft()
                )
                current = set_target(
                    action,
                    current_reference,
                    now,
                    smooth=smooth,
                    actor=not smooth,
                )
                payload = (
                    {"current": current, "accepted_target": command_target.copy()}
                    if diagnostics
                    else current
                )
                if diagnostics and actor_profile is not None:
                    payload["actor_profile"] = actor_profile.diagnostics()
                response_queue.put((request_id, "ok", payload))
            else:
                action = rtc_actions.popleft()
                reference = action if command_target is None else command_target
                set_target(action, reference, now, smooth=True)
                rtc_executed += 1
                rtc_last_action = action.copy()
                if not rtc_actions and drain_request_id is not None:
                    request_id = drain_request_id
                    drain_request_id = None
                    finish_rtc(request_id)
            next_deadline = deadline + policy_period
            next_policy_time = (
                next_deadline if now <= next_deadline else now + policy_period
            )

        command = interpolated(time.monotonic())
        if publishing_enabled and command is not None:
            publish(command)
        next_publish += publish_period
        remaining = next_publish - time.monotonic()
        if remaining <= 0:
            next_publish = time.monotonic()
        else:
            time.sleep(remaining)


class PiperControlProcess:
    """Synchronous client for the process-isolated Piper control executor."""

    def __init__(self, config: dict[str, Any], timeout_s: float = 10.0) -> None:
        context = mp.get_context("spawn")
        self._commands = context.Queue()
        self._responses = context.Queue()
        self._process = context.Process(
            target=_control_process_main,
            args=(config, self._commands, self._responses, os.getpid()),
            name="rlinf-piper-control-executor",
            daemon=True,
        )
        self._request_lock = threading.Lock()
        self._next_request_id = 1
        self._timeout_s = timeout_s
        self._process.start()
        request_id, status, payload = self._responses.get(timeout=timeout_s)
        if request_id != 0 or status != "ready":
            raise RuntimeError(
                f"Piper control executor failed to start: {status}: {payload}"
            )

    def _request(self, operation: str, payload: Any = None) -> Any:
        with self._request_lock:
            request_id = self._next_request_id
            self._next_request_id += 1
            self._commands.put((request_id, operation, payload))
            try:
                response_id, status, result = self._responses.get(
                    timeout=self._timeout_s
                )
            except queue.Empty as exc:
                raise TimeoutError(
                    f"Timed out waiting for Piper control operation {operation!r}."
                ) from exc
            if response_id != request_id:
                raise RuntimeError(
                    f"Piper control response mismatch: {response_id} != {request_id}."
                )
            if status != "ok":
                raise RuntimeError(str(result))
            return result

    def apply_action(
        self,
        action: np.ndarray,
        current_reference: np.ndarray,
        *,
        smooth: bool = True,
    ) -> np.ndarray:
        """Schedule one main action and wait until its 15-Hz deadline."""
        operation = "action_diagnostics" if smooth else "actor_action_diagnostics"
        result = self._request(operation, (action, current_reference))
        self.last_accepted_target = np.asarray(
            result["accepted_target"], dtype=np.float32
        ).copy()
        self.last_actor_profile_diagnostics = result.get("actor_profile")
        return np.asarray(result["current"], dtype=np.float32)

    def start_rtc(self, actions: np.ndarray) -> None:
        """Queue a complete RTC suffix for independent fixed-rate execution."""
        self._request("start_rtc", actions)

    def apply_reset_target(
        self,
        action: np.ndarray,
        current_reference: np.ndarray,
    ) -> np.ndarray:
        """Apply one unsmoothed return-to-start interpolation target."""
        return np.asarray(
            self._request("reset_target", (action, current_reference)),
            dtype=np.float32,
        )

    def stop_rtc(self, *, drain: bool = False) -> tuple[int, np.ndarray | None]:
        """Return the RTC prefix, optionally after completing its queued suffix."""
        executed, last_action = self._request("drain_rtc" if drain else "stop_rtc")
        return int(executed), (
            None if last_action is None else np.asarray(last_action, dtype=np.float32)
        )

    def publish_immediate(self, action: np.ndarray) -> None:
        """Publish a direct reset/hold command without policy-rate scheduling."""
        self._request("immediate", action)

    def reset(self, action: np.ndarray) -> None:
        """Reset executor timing and filtering state at an episode boundary."""
        self._request("reset", action)

    def suspend(self) -> None:
        """Stop all policy/RTC publication until reset or immediate hold."""
        self._request("suspend")

    def close(self) -> None:
        """Stop the executor process."""
        if not self._process.is_alive():
            return
        try:
            self._request("close")
        finally:
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
