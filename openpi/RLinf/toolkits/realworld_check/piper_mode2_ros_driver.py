#!/usr/bin/env python3
"""ROS1 bridge for pre-isolated Piper leader teleoperation.

The physical leaders must already emit 0x16x before this bridge starts. Piper
firmware S-V1.8-8 acknowledges a role-neutral 0x470 request but does not apply
its linkage offset, so this node must never claim that a connected 0x15x pair
was dynamically isolated. During policy execution this node publishes leader
feedback and optionally forwards it to follower command topics.

Shutdown leaves the hardware pre-isolation unchanged. Restoring native 0x15x
requires the corresponding follower to be physically disconnected.

This file intentionally remains compatible with the robot's system Python 3.8.
"""

import math
import struct
import threading
import time

import can
import rospy
from sensor_msgs.msg import JointState
from std_srvs.srv import SetBool, SetBoolResponse, Trigger, TriggerResponse

JOINT_SCALE = math.pi / (180.0 * 1000.0)
PAIR_IDS = {0x165: (0, 1), 0x166: (2, 3), 0x167: (4, 5)}
GRIPPER_ID = 0x169
FOLLOWER_PAIR_IDS = {0x2A5: (0, 1), 0x2A6: (2, 3), 0x2A7: (4, 5)}
FOLLOWER_GRIPPER_ID = 0x2A8


def _open_bus(can_name):
    try:
        return can.interface.Bus(
            channel=can_name,
            interface="socketcan",
            receive_own_messages=False,
        )
    except TypeError:
        return can.interface.Bus(
            channel=can_name,
            bustype="socketcan",
            receive_own_messages=False,
        )


class OffsetLeaderChannel:
    """Assemble one leader pose from its offset 0x16x CAN frames."""

    def __init__(self, can_name):
        self.can_name = can_name
        self._lock = threading.Lock()
        self._position = [0.0] * 7
        self._seen = set()
        self._timestamps = {}
        self._bus = _open_bus(can_name)
        self._running = True
        self._thread = threading.Thread(
            target=self._read_loop,
            name="mode2-can-{}".format(can_name),
            daemon=True,
        )
        self._thread.start()

    def _read_loop(self):
        while self._running and not rospy.is_shutdown():
            message = self._bus.recv(timeout=0.1)
            if message is None:
                continue
            self._process_frame(message, time.monotonic())

    def _process_frame(self, message, timestamp):
        arbitration_id = int(message.arbitration_id)
        data = bytes(message.data)
        if arbitration_id in PAIR_IDS and len(data) >= 8:
            first, second = struct.unpack(">ii", data[:8])
            indices = PAIR_IDS[arbitration_id]
            with self._lock:
                self._position[indices[0]] = first * JOINT_SCALE
                self._position[indices[1]] = second * JOINT_SCALE
                self._seen.add(arbitration_id)
                self._timestamps[arbitration_id] = timestamp
        elif arbitration_id == GRIPPER_ID and len(data) >= 4:
            gripper = struct.unpack(">i", data[:4])[0] / 1_000_000.0
            with self._lock:
                self._position[6] = max(0.0, min(0.08, gripper))
                self._seen.add(arbitration_id)
                self._timestamps[arbitration_id] = timestamp

    def sample(self):
        with self._lock:
            required_ids = set(PAIR_IDS).union({GRIPPER_ID})
            complete = required_ids.issubset(self._seen)
            timestamp = (
                min(self._timestamps[frame_id] for frame_id in required_ids)
                if complete
                else 0.0
            )
            return list(self._position), timestamp, complete

    def close(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self._bus.shutdown()


class FollowerFeedbackChannel(OffsetLeaderChannel):
    """Assemble one follower pose from its native 0x2Ax feedback frames."""

    def _process_frame(self, message, timestamp):
        arbitration_id = int(message.arbitration_id)
        data = bytes(message.data)
        if arbitration_id in FOLLOWER_PAIR_IDS and len(data) >= 8:
            first, second = struct.unpack(">ii", data[:8])
            indices = FOLLOWER_PAIR_IDS[arbitration_id]
            with self._lock:
                self._position[indices[0]] = first * JOINT_SCALE
                self._position[indices[1]] = second * JOINT_SCALE
                self._seen.add(arbitration_id)
                self._timestamps[arbitration_id] = timestamp
        elif arbitration_id == FOLLOWER_GRIPPER_ID and len(data) >= 4:
            gripper = struct.unpack(">i", data[:4])[0] / 1_000_000.0
            with self._lock:
                self._position[6] = max(0.0, min(0.08, gripper))
                self._seen.add(arbitration_id)
                self._timestamps[arbitration_id] = timestamp

    def sample(self):
        with self._lock:
            required_ids = set(FOLLOWER_PAIR_IDS).union({FOLLOWER_GRIPPER_ID})
            complete = required_ids.issubset(self._seen)
            timestamp = (
                min(self._timestamps[frame_id] for frame_id in required_ids)
                if complete
                else 0.0
            )
            return list(self._position), timestamp, complete


class PiperMode2Bridge:
    """Publish leader feedback and arbitrate high-rate follower ownership."""

    def __init__(self):
        self.feedback_timeout = float(rospy.get_param("~feedback_timeout_s", 0.1))
        self.publish_hz = float(rospy.get_param("~publish_hz", 200.0))
        self.leaders_preisolated = bool(
            rospy.get_param("~leaders_preisolated", False)
        )
        if not self.leaders_preisolated:
            raise RuntimeError(
                "mode 2 requires physically verified 0x16x leaders; dynamic "
                "0x15x/0x16x switching is unsupported on Piper S-V1.8-8"
            )
        self.restore_joint_tolerance = float(
            rospy.get_param("~restore_joint_tolerance_rad", 0.03)
        )
        self.restore_gripper_tolerance = float(
            rospy.get_param("~restore_gripper_tolerance_m", 0.005)
        )
        self.alignment_joint_velocity = float(
            rospy.get_param("~alignment_joint_velocity_rad_s", 0.05)
        )
        self.alignment_gripper_velocity = float(
            rospy.get_param("~alignment_gripper_velocity_m_s", 0.01)
        )
        self.alignment_stable_samples = int(
            rospy.get_param("~alignment_stable_samples", 20)
        )
        self.alignment_timeout = float(
            rospy.get_param("~alignment_timeout_s", 30.0)
        )
        self.alignment_tracking_tolerance = float(
            rospy.get_param("~alignment_tracking_tolerance_rad", 0.05)
        )
        left_can = rospy.get_param("~left_can", "can_left")
        right_can = rospy.get_param("~right_can", "can_right")
        rospy.loginfo(
            "Mode 2 started in pre-isolated mode; hardware must already emit 0x16x."
        )
        self.left = OffsetLeaderChannel(left_can)
        self.right = OffsetLeaderChannel(right_can)
        self.left_follower = FollowerFeedbackChannel(left_can)
        self.right_follower = FollowerFeedbackChannel(right_can)
        self.left_feedback_pub = rospy.Publisher(
            "/rlinf/mode2/leader_left",
            JointState,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.right_feedback_pub = rospy.Publisher(
            "/rlinf/mode2/leader_right",
            JointState,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.left_follower_pub = rospy.Publisher(
            "/master/joint_left",
            JointState,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.right_follower_pub = rospy.Publisher(
            "/master/joint_right",
            JointState,
            queue_size=1,
            tcp_nodelay=True,
        )
        self._teleop_enabled = False
        self._alignment_enabled = False
        self._alignment_stable_count = 0
        self._alignment_targets = None
        self._alignment_commands = None
        self._alignment_started_at = 0.0
        self._teleop_lock = threading.Lock()
        self._stale_reported = False
        self._closed = False
        self._services = [
            rospy.Service(
                "/rlinf/mode2/follower_teleop",
                SetBool,
                self._set_teleop,
            ),
            rospy.Service(
                "/rlinf/mode2/leader_drag_left",
                Trigger,
                self._check_left,
            ),
            rospy.Service(
                "/rlinf/mode2/leader_drag_right",
                Trigger,
                self._check_right,
            ),
            rospy.Service(
                "/rlinf/mode2/align_followers_to_leaders",
                Trigger,
                self._start_alignment,
            ),
            rospy.Service(
                "/rlinf/mode2/alignment_status",
                Trigger,
                self._alignment_status,
            ),
            rospy.Service(
                "/rlinf/mode2/cancel_alignment",
                Trigger,
                self._cancel_alignment,
            ),
        ]

    def _fresh(self, channel):
        position, timestamp, complete = channel.sample()
        age = time.monotonic() - timestamp
        return position, complete and 0.0 <= age <= self.feedback_timeout, age

    def _check_channel(self, channel):
        _, fresh, age = self._fresh(channel)
        if not fresh:
            return TriggerResponse(
                success=False,
                message=(
                    "{} shifted leader feedback is incomplete/stale "
                    "(age={:.3f}s); verify persistent 0x16x offset"
                ).format(channel.can_name, age),
            )
        return TriggerResponse(
            success=True,
            message="{} leader is powered and publishing 0x16x".format(
                channel.can_name
            ),
        )

    def _check_left(self, _request):
        return self._check_channel(self.left)

    def _check_right(self, _request):
        return self._check_channel(self.right)

    def _set_teleop(self, request):
        if request.data:
            left_response = self._check_channel(self.left)
            right_response = self._check_channel(self.right)
            if not left_response.success or not right_response.success:
                return SetBoolResponse(
                    success=False,
                    message="; ".join(
                        (left_response.message, right_response.message)
                    ),
                )
        with self._teleop_lock:
            self._teleop_enabled = bool(request.data)
            if request.data:
                self._alignment_enabled = False
                self._alignment_stable_count = 0
        return SetBoolResponse(
            success=True,
            message="follower teleop {}".format(
                "enabled" if request.data else "disabled"
            ),
        )

    def _all_positions(self):
        left_leader, left_leader_fresh, _ = self._fresh(self.left)
        right_leader, right_leader_fresh, _ = self._fresh(self.right)
        left_follower, left_follower_fresh, _ = self._fresh(self.left_follower)
        right_follower, right_follower_fresh, _ = self._fresh(self.right_follower)
        fresh = (
            left_leader_fresh
            and right_leader_fresh
            and left_follower_fresh
            and right_follower_fresh
        )
        return (
            left_leader,
            right_leader,
            left_follower,
            right_follower,
            fresh,
        )

    @staticmethod
    def _bounded_target(current, target, joint_step, gripper_step):
        result = []
        for index, (source, destination) in enumerate(zip(current, target)):
            max_step = gripper_step if index == 6 else joint_step
            delta = max(-max_step, min(max_step, destination - source))
            result.append(source + delta)
        return result

    def _aligned(self, left_leader, right_leader, left_follower, right_follower):
        for leader, follower in (
            (left_leader, left_follower),
            (right_leader, right_follower),
        ):
            if (
                max(abs(leader[index] - follower[index]) for index in range(6))
                > self.restore_joint_tolerance
            ):
                return False
            if abs(leader[6] - follower[6]) > self.restore_gripper_tolerance:
                return False
        return True

    def _start_alignment(self, _request):
        left_leader, _, left_leader_complete = self.left.sample()
        right_leader, _, right_leader_complete = self.right.sample()
        left_follower, left_follower_fresh, _ = self._fresh(self.left_follower)
        right_follower, right_follower_fresh, _ = self._fresh(self.right_follower)
        if not (
            left_leader_complete
            and right_leader_complete
            and left_follower_fresh
            and right_follower_fresh
        ):
            return TriggerResponse(
                success=False,
                message="leader/follower feedback is incomplete or stale",
            )
        with self._teleop_lock:
            self._teleop_enabled = False
            self._alignment_enabled = True
            self._alignment_stable_count = 0
            self._alignment_targets = (left_leader, right_leader)
            self._alignment_commands = (left_follower, right_follower)
            self._alignment_started_at = time.monotonic()
        return TriggerResponse(
            success=True,
            message=(
                "bounded follower-to-leader alignment started at "
                "{:.3f} rad/s and {:.3f} m/s"
            ).format(
                self.alignment_joint_velocity,
                self.alignment_gripper_velocity,
            ),
        )

    def _alignment_status(self, _request):
        with self._teleop_lock:
            active = self._alignment_enabled
            targets = self._alignment_targets
        left_follower, left_fresh, _ = self._fresh(self.left_follower)
        right_follower, right_fresh, _ = self._fresh(self.right_follower)
        aligned = bool(
            targets is not None
            and left_fresh
            and right_fresh
            and self._aligned(
                targets[0],
                targets[1],
                left_follower,
                right_follower,
            )
        )
        return TriggerResponse(
            success=bool(aligned and not active),
            message="alignment {}; {}".format(
                "complete"
                if aligned and not active
                else "active"
                if active
                else "not ready",
                self._format_errors(
                    targets,
                    left_follower,
                    right_follower,
                ),
            ),
        )

    def _cancel_alignment(self, _request):
        with self._teleop_lock:
            was_active = self._alignment_enabled
            self._alignment_enabled = False
            self._alignment_stable_count = 0
        return TriggerResponse(
            success=True,
            message="alignment {}".format(
                "cancelled" if was_active else "was not active"
            ),
        )

    @staticmethod
    def _format_errors(targets, left_follower, right_follower):
        if targets is None:
            return "no alignment target"
        parts = []
        for label, leader, follower in (
            ("left", targets[0], left_follower),
            ("right", targets[1], right_follower),
        ):
            joint_error = max(
                abs(leader[index] - follower[index]) for index in range(6)
            )
            gripper_error = abs(leader[6] - follower[6])
            parts.append(
                "{} joint={:.4f}rad gripper={:.4f}m".format(
                    label, joint_error, gripper_error
                )
            )
        return "; ".join(parts)

    @staticmethod
    def _message(position):
        message = JointState()
        message.header.stamp = rospy.Time.now()
        message.name = ["joint{}".format(index) for index in range(7)]
        message.position = position
        return message

    def run(self):
        rate = rospy.Rate(self.publish_hz)
        joint_step = self.alignment_joint_velocity / self.publish_hz
        gripper_step = self.alignment_gripper_velocity / self.publish_hz
        while not rospy.is_shutdown():
            left_position, left_fresh, _ = self._fresh(self.left)
            right_position, right_fresh, _ = self._fresh(self.right)
            if left_fresh:
                self.left_feedback_pub.publish(self._message(left_position))
            if right_fresh:
                self.right_feedback_pub.publish(self._message(right_position))

            with self._teleop_lock:
                teleop_enabled = self._teleop_enabled
                alignment_enabled = self._alignment_enabled
                alignment_targets = self._alignment_targets
                alignment_commands = self._alignment_commands
                alignment_started_at = self._alignment_started_at
            if teleop_enabled and left_fresh and right_fresh:
                self.left_follower_pub.publish(self._message(left_position))
                self.right_follower_pub.publish(self._message(right_position))
                self._stale_reported = False
            elif teleop_enabled:
                with self._teleop_lock:
                    self._teleop_enabled = False
                if not self._stale_reported:
                    rospy.logerr(
                        "Mode-2 leader feedback went stale; follower teleop "
                        "disabled and the follower driver will hold its last target."
                    )
                    self._stale_reported = True
            elif alignment_enabled:
                left_follower, left_fresh, _ = self._fresh(self.left_follower)
                right_follower, right_fresh, _ = self._fresh(self.right_follower)
                if (
                    left_fresh
                    and right_fresh
                    and alignment_targets is not None
                    and alignment_commands is not None
                ):
                    left_leader, right_leader = alignment_targets
                    left_command, right_command = alignment_commands
                    tracking_error = max(
                        max(
                            abs(command[index] - follower[index])
                            for index in range(6)
                        )
                        for command, follower in (
                            (left_command, left_follower),
                            (right_command, right_follower),
                        )
                    )
                    if tracking_error <= self.alignment_tracking_tolerance:
                        left_command = self._bounded_target(
                            left_command,
                            left_leader,
                            joint_step,
                            gripper_step,
                        )
                        right_command = self._bounded_target(
                            right_command,
                            right_leader,
                            joint_step,
                            gripper_step,
                        )
                        with self._teleop_lock:
                            self._alignment_commands = (
                                left_command,
                                right_command,
                            )
                    self.left_follower_pub.publish(
                        self._message(left_command)
                    )
                    self.right_follower_pub.publish(
                        self._message(right_command)
                    )
                    if self._aligned(
                        left_leader,
                        right_leader,
                        left_follower,
                        right_follower,
                    ):
                        self._alignment_stable_count += 1
                    else:
                        self._alignment_stable_count = 0
                    if self._alignment_stable_count >= self.alignment_stable_samples:
                        with self._teleop_lock:
                            self._alignment_enabled = False
                        rospy.loginfo(
                            "Bounded follower-to-leader alignment completed; "
                            "native 0x15x restoration is now permitted."
                        )
                    elif time.monotonic() - alignment_started_at > self.alignment_timeout:
                        with self._teleop_lock:
                            self._alignment_enabled = False
                        rospy.logerr(
                            "Alignment timed out after %.1f seconds; stopped at %s.",
                            self.alignment_timeout,
                            self._format_errors(
                                alignment_targets,
                                left_follower,
                                right_follower,
                            ),
                        )
                else:
                    with self._teleop_lock:
                        self._alignment_enabled = False
                    rospy.logerr(
                        "Alignment stopped because leader/follower feedback "
                        "became incomplete or stale."
                    )
            rate.sleep()

    def _restoration_safe(self):
        failures = []
        for label, leader, follower in (
            ("left", self.left, self.left_follower),
            ("right", self.right, self.right_follower),
        ):
            leader_position, _, leader_complete = leader.sample()
            follower_position, follower_timestamp, follower_complete = follower.sample()
            follower_age = time.monotonic() - follower_timestamp
            if not leader_complete:
                failures.append("{} leader pose is incomplete".format(label))
                continue
            if not follower_complete or not 0.0 <= follower_age <= self.feedback_timeout:
                failures.append("{} follower feedback is stale".format(label))
                continue
            joint_error = max(
                abs(leader_position[index] - follower_position[index])
                for index in range(6)
            )
            gripper_error = abs(leader_position[6] - follower_position[6])
            if joint_error > self.restore_joint_tolerance:
                failures.append(
                    "{} joint error {:.4f} rad exceeds {:.4f}".format(
                        label, joint_error, self.restore_joint_tolerance
                    )
                )
            if gripper_error > self.restore_gripper_tolerance:
                failures.append(
                    "{} gripper error {:.4f} m exceeds {:.4f}".format(
                        label, gripper_error, self.restore_gripper_tolerance
                    )
                )
        return not failures, failures

    def close(self):
        if self._closed:
            return
        self._closed = True
        with self._teleop_lock:
            self._teleop_enabled = False
            self._alignment_enabled = False
        safe, failures = self._restoration_safe()
        if not safe:
            rospy.logerr(
                "Mode 2 stopped with unaligned leader/follower poses: %s. "
                "Hardware remains at pre-isolated 0x16x.",
                "; ".join(failures),
            )
        else:
            rospy.loginfo(
                "Mode 2 stopped with aligned poses. Hardware remains at "
                "pre-isolated 0x16x; restore 0x15x only with followers disconnected."
            )
        self.left.close()
        self.right.close()
        self.left_follower.close()
        self.right_follower.close()


def main():
    rospy.init_node("piper_mode2_bridge")
    bridge = PiperMode2Bridge()
    rospy.on_shutdown(bridge.close)
    bridge.run()


if __name__ == "__main__":
    main()
