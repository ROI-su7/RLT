"""Hardware-independent coordinator for Piper mode-2 takeover.

The coordinator is intentionally separate from the ROS/CAN adapter.  A mode-2
hardware adapter may command the leader arms, but the existing mode-0 and
mode-1 drivers never import or instantiate this class.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rlinf.envs.realworld.piper.leader_takeover import (
    LeaderTakeoverStateMachine,
    TakeoverState,
)

ARM_INDICES = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])


@dataclass(frozen=True)
class Mode2Feedback:
    """Latest measured joint state for both dual-arm pairs."""

    leader_action: np.ndarray
    follower_action: np.ndarray
    leader_timestamp: float
    follower_timestamp: float


class Mode2Hardware(Protocol):
    """Required capabilities of an explicitly selected mode-2 backend."""

    def feedback(self) -> Mode2Feedback:
        """Return fresh leader and follower joint feedback."""

    def command_leader_joints(self, joints: np.ndarray) -> None:
        """Command the twelve leader revolute joints."""

    def enter_leader_drag_mode(self) -> None:
        """Make the aligned leader arms safe for human dragging."""

    def set_follower_teleop_enabled(self, enabled: bool) -> None:
        """Enable or disable the independent high-rate follower bridge."""


class PiperMode2Controller:
    """Coordinate hold, leader alignment, and teleoperation action ownership."""

    def __init__(
        self,
        state_machine: LeaderTakeoverStateMachine,
        hardware: Mode2Hardware,
    ) -> None:
        if state_machine.config.joint_count != len(ARM_INDICES):
            raise ValueError("Piper mode 2 requires a 12-joint state machine.")
        self.state_machine = state_machine
        self.hardware = hardware
        self._follower_hold_action: np.ndarray | None = None
        self._leader_drag_enabled = False
        self._leader_start_action: np.ndarray | None = None
        self._follower_start_action: np.ndarray | None = None

    @property
    def state(self) -> TakeoverState:
        """Return the current exclusive command-owner state."""
        return self.state_machine.state

    @staticmethod
    def _action(value: np.ndarray, name: str) -> np.ndarray:
        action = np.asarray(value, dtype=np.float32).reshape(-1)
        if action.shape != (14,):
            raise ValueError(f"{name} must have shape (14,), got {action.shape}.")
        if not np.isfinite(action).all():
            raise ValueError(f"{name} contains a non-finite value.")
        return action

    def _require_fresh_feedback(
        self,
        feedback: Mode2Feedback,
        *,
        now: float | None = None,
    ) -> None:
        current = time.monotonic() if now is None else float(now)
        timeout = self.state_machine.config.feedback_timeout_s
        for name, timestamp in (
            ("leader", feedback.leader_timestamp),
            ("follower", feedback.follower_timestamp),
        ):
            age = current - float(timestamp)
            if age < 0 or age > timeout:
                raise RuntimeError(
                    f"{name} feedback is stale during mode-2 ownership "
                    f"({age:.3f}s)."
                )

    def request_hold(self) -> np.ndarray:
        """Latch measured follower pose and revoke policy ownership."""
        teleop_was_enabled = self.state in {
            TakeoverState.TELEOP,
            TakeoverState.TELEOP_RETURN,
        }
        feedback = self.hardware.feedback()
        follower = self._action(feedback.follower_action, "follower_action")
        self.state_machine.request_hold(follower[ARM_INDICES])
        self._follower_hold_action = follower.copy()
        if teleop_was_enabled:
            try:
                self.hardware.set_follower_teleop_enabled(False)
            except Exception as exc:
                self.state_machine.fail(f"failed to disable follower teleop: {exc}")
                raise RuntimeError(self.state_machine.snapshot().fault) from exc
        self._leader_drag_enabled = False
        return follower

    def request_alignment(self, now: float) -> np.ndarray | None:
        """Start the limited leader approach and send its first target."""
        feedback = self.hardware.feedback()
        leader = self._action(feedback.leader_action, "leader_action")
        follower = self._action(feedback.follower_action, "follower_action")
        if self.state_machine.config.manual_approach:
            self.state_machine.request_manual_approach(
                leader_joints=leader[ARM_INDICES],
                follower_joints=follower[ARM_INDICES],
                leader_timestamp=feedback.leader_timestamp,
                follower_timestamp=feedback.follower_timestamp,
                now=now,
            )
            self._enter_leader_drag_or_fault()
            return None
        target = self.state_machine.request_alignment(
            leader_joints=leader[ARM_INDICES],
            follower_joints=follower[ARM_INDICES],
            leader_timestamp=feedback.leader_timestamp,
            follower_timestamp=feedback.follower_timestamp,
            now=now,
        )
        if self.state == TakeoverState.LEADER_MANUAL_ALIGN:
            self._enter_leader_drag_or_fault()
            return None
        self._command_leader_or_fault(target)
        return target

    def alignment_tick(self, now: float) -> np.ndarray | None:
        """Advance automatic alignment by one bounded command tick."""
        if self.state not in {
            TakeoverState.LEADER_ALIGNING,
            TakeoverState.LEADER_MANUAL_ALIGN,
        }:
            return None
        feedback = self.hardware.feedback()
        leader = self._action(feedback.leader_action, "leader_action")
        follower = self._action(feedback.follower_action, "follower_action")
        if (
            self.state == TakeoverState.LEADER_ALIGNING
            and self.state_machine.config.manual_approach
        ):
            self.state_machine.manual_approach_tick(
                leader_joints=leader[ARM_INDICES],
                follower_joints=follower[ARM_INDICES],
                leader_timestamp=feedback.leader_timestamp,
                follower_timestamp=feedback.follower_timestamp,
                now=now,
            )
            return None
        if self.state == TakeoverState.LEADER_MANUAL_ALIGN:
            self.state_machine.manual_alignment_tick(
                leader_joints=leader[ARM_INDICES],
                follower_joints=follower[ARM_INDICES],
                leader_timestamp=feedback.leader_timestamp,
                follower_timestamp=feedback.follower_timestamp,
                now=now,
            )
            return None
        target = self.state_machine.next_alignment_target(
            leader_joints=leader[ARM_INDICES],
            follower_joints=follower[ARM_INDICES],
            leader_timestamp=feedback.leader_timestamp,
            follower_timestamp=feedback.follower_timestamp,
            now=now,
        )
        if self.state == TakeoverState.LEADER_MANUAL_ALIGN:
            self._enter_leader_drag_or_fault()
            return None
        self._command_leader_or_fault(target)
        return target

    def _command_leader_or_fault(self, target: np.ndarray) -> None:
        try:
            self.hardware.command_leader_joints(target)
        except Exception as exc:
            self.state_machine.fail(f"leader alignment command failed: {exc}")
            raise RuntimeError(self.state_machine.snapshot().fault) from exc

    def _enter_leader_drag_or_fault(self) -> None:
        try:
            self.hardware.enter_leader_drag_mode()
        except Exception as exc:
            self.state_machine.fail(f"failed to enter leader drag mode: {exc}")
            raise RuntimeError(self.state_machine.snapshot().fault) from exc
        self._leader_drag_enabled = True

    def request_takeover(self, now: float) -> None:
        """Enter drag/teleop only after the alignment gate is still valid."""
        feedback = self.hardware.feedback()
        leader = self._action(feedback.leader_action, "leader_action")
        follower = self._action(feedback.follower_action, "follower_action")
        self.state_machine.request_takeover(
            leader_joints=leader[ARM_INDICES],
            follower_joints=follower[ARM_INDICES],
            leader_timestamp=feedback.leader_timestamp,
            follower_timestamp=feedback.follower_timestamp,
            now=now,
        )
        if not self._leader_drag_enabled:
            self._enter_leader_drag_or_fault()
        try:
            self.hardware.set_follower_teleop_enabled(True)
        except Exception as exc:
            self.state_machine.fail(f"failed to enable follower teleop: {exc}")
            raise RuntimeError(self.state_machine.snapshot().fault) from exc

    def action_for_step(self, policy_action: np.ndarray) -> tuple[np.ndarray, bool]:
        """Select the sole command source for this environment step.

        Returns:
            A tuple of the selected 14-D action and whether it is a human
            intervention that must replace the policy action in replay.
        """
        policy = self._action(policy_action, "policy_action")
        if self.state == TakeoverState.POLICY:
            return policy, False
        if self.state == TakeoverState.TELEOP:
            feedback = self.hardware.feedback()
            try:
                self._require_fresh_feedback(feedback)
            except RuntimeError as exc:
                self.emergency_stop(str(exc))
                raise
            return self._action(feedback.leader_action, "leader_action").copy(), True
        if self._follower_hold_action is None:
            raise RuntimeError("mode-2 follower hold action is not latched.")
        return self._follower_hold_action.copy(), False

    def finish_episode(self) -> None:
        """Latch terminal hold after a scored or aborted teleoperation."""
        if self.state in {
            TakeoverState.FOLLOWER_HOLD,
            TakeoverState.LEADER_MANUAL_ALIGN,
            TakeoverState.READY,
            TakeoverState.TELEOP,
        }:
            self.state_machine.finish_episode()

    def emergency_stop(self, reason: str = "operator emergency hold") -> None:
        """Disable the bridge and latch a fault that requires a clean restart."""
        disable_error = None
        try:
            self.hardware.set_follower_teleop_enabled(False)
        except Exception as exc:  # Hold locally even if the service is unhealthy.
            disable_error = str(exc)
        fault = reason if disable_error is None else f"{reason}; {disable_error}"
        self.state_machine.fail(fault)

    def capture_episode_start(self) -> bool:
        """Capture measured leader and follower starts exactly once."""
        if self._leader_start_action is not None:
            return False
        feedback = self.hardware.feedback()
        self._leader_start_action = self._action(
            feedback.leader_action, "leader_action"
        ).copy()
        self._follower_start_action = self._action(
            feedback.follower_action, "follower_action"
        ).copy()
        return True

    def complete_teleop_return(
        self,
        *,
        arm_tolerance_rad: float = 0.03,
        gripper_tolerance_m: float = 0.005,
    ) -> tuple[float, float]:
        """Disable teleop only after both pairs return to captured starts."""
        if self.state != TakeoverState.TELEOP_RETURN:
            raise RuntimeError("mode 2 is not in post-terminal return.")
        if self._leader_start_action is None or self._follower_start_action is None:
            raise RuntimeError("mode-2 episode start has not been captured.")
        feedback = self.hardware.feedback()
        self._require_fresh_feedback(feedback)
        leader = self._action(feedback.leader_action, "leader_action")
        follower = self._action(feedback.follower_action, "follower_action")
        arm_error = float(
            max(
                np.max(
                    np.abs(
                        leader[ARM_INDICES]
                        - self._leader_start_action[ARM_INDICES]
                    )
                ),
                np.max(
                    np.abs(
                        follower[ARM_INDICES]
                        - self._follower_start_action[ARM_INDICES]
                    )
                ),
            )
        )
        gripper_indices = np.asarray([6, 13])
        gripper_error = float(
            max(
                np.max(
                    np.abs(
                        leader[gripper_indices]
                        - self._leader_start_action[gripper_indices]
                    )
                ),
                np.max(
                    np.abs(
                        follower[gripper_indices]
                        - self._follower_start_action[gripper_indices]
                    )
                ),
            )
        )
        if arm_error > arm_tolerance_rad or gripper_error > gripper_tolerance_m:
            raise RuntimeError(
                "both arm pairs are not at the recorded start: "
                f"arm_error={arm_error:.4f}rad, "
                f"gripper_error={gripper_error:.4f}m."
            )
        self.hardware.set_follower_teleop_enabled(False)
        self.state_machine.complete_teleop_return()
        return arm_error, gripper_error

    def resume_policy(self) -> np.ndarray:
        """Return to policy ownership using current measured follower joints."""
        feedback = self.hardware.feedback()
        follower = self._action(feedback.follower_action, "follower_action")
        seed = self.state_machine.resume_policy(follower[ARM_INDICES])
        self.hardware.set_follower_teleop_enabled(False)
        self._follower_hold_action = None
        self._leader_drag_enabled = False
        return seed
