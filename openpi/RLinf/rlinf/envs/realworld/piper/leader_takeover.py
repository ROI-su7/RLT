"""Safety state machine for Piper leader-follower takeover.

This module deliberately has no ROS or Piper SDK imports.  The hardware bridge
owns those dependencies and feeds measured leader/follower joints into this
state machine.  Keeping the transition logic pure makes every safety gate
testable without a connected robot.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np


class TakeoverState(StrEnum):
    """Exclusive owner states for the hybrid Piper controller."""

    POLICY = "policy"
    FOLLOWER_HOLD = "follower_hold"
    LEADER_ALIGNING = "leader_aligning"
    LEADER_MANUAL_ALIGN = "leader_manual_align"
    READY = "ready_for_takeover"
    TELEOP = "human_teleop"
    TELEOP_RETURN = "human_teleop_return"
    TERMINAL_HOLD = "terminal_hold"
    FAULT = "fault"


@dataclass(frozen=True)
class LeaderTakeoverConfig:
    """Safety limits for leader alignment and handover validation."""

    joint_count: int = 12
    alignment_rate_hz: float = 50.0
    max_velocity_rad_s: float = 0.10
    manual_approach: bool = False
    approach_tolerance_rad: float = 0.08
    approach_stable_samples: int = 3
    position_tolerance_rad: float = 0.02
    stable_samples: int = 10
    feedback_timeout_s: float = 0.10
    alignment_timeout_s: float = 30.0

    def __post_init__(self) -> None:
        if self.joint_count <= 0:
            raise ValueError("joint_count must be positive.")
        if self.alignment_rate_hz <= 0:
            raise ValueError("alignment_rate_hz must be positive.")
        if self.max_velocity_rad_s <= 0:
            raise ValueError("max_velocity_rad_s must be positive.")
        if self.approach_tolerance_rad <= 0:
            raise ValueError("approach_tolerance_rad must be positive.")
        if self.position_tolerance_rad <= 0:
            raise ValueError("position_tolerance_rad must be positive.")
        if self.approach_tolerance_rad <= self.position_tolerance_rad:
            raise ValueError(
                "approach_tolerance_rad must be greater than "
                "position_tolerance_rad."
            )
        if self.approach_stable_samples <= 0:
            raise ValueError("approach_stable_samples must be positive.")
        if self.stable_samples <= 0:
            raise ValueError("stable_samples must be positive.")
        if self.feedback_timeout_s <= 0:
            raise ValueError("feedback_timeout_s must be positive.")
        if self.alignment_timeout_s <= 0:
            raise ValueError("alignment_timeout_s must be positive.")

    @property
    def max_alignment_step_rad(self) -> float:
        """Maximum commanded leader-joint increment per alignment tick."""
        return self.max_velocity_rad_s / self.alignment_rate_hz


@dataclass(frozen=True)
class TakeoverSnapshot:
    """Immutable status returned to the ROS and RLT adapters."""

    state: TakeoverState
    follower_hold_target: np.ndarray | None
    max_alignment_error_rad: float | None
    approach_stable_samples: int
    stable_samples: int
    fault: str | None


class LeaderTakeoverStateMachine:
    """Guard policy, hold, alignment, and teleoperation ownership changes."""

    def __init__(self, config: LeaderTakeoverConfig | None = None) -> None:
        self.config = config or LeaderTakeoverConfig()
        self.state = TakeoverState.POLICY
        self._follower_hold_target: np.ndarray | None = None
        self._alignment_started_at: float | None = None
        self._approach_stable_samples = 0
        self._stable_samples = 0
        self._max_alignment_error_rad: float | None = None
        self._fault: str | None = None

    def _joints(self, value: np.ndarray, name: str) -> np.ndarray:
        joints = np.asarray(value, dtype=np.float64).reshape(-1)
        if joints.shape != (self.config.joint_count,):
            raise ValueError(
                f"{name} must have shape ({self.config.joint_count},), "
                f"got {joints.shape}."
            )
        if not np.isfinite(joints).all():
            raise ValueError(f"{name} contains a non-finite value.")
        return joints

    def _require_fresh_feedback(
        self,
        *,
        now: float,
        leader_timestamp: float,
        follower_timestamp: float,
    ) -> None:
        leader_age = now - float(leader_timestamp)
        follower_age = now - float(follower_timestamp)
        if leader_age < 0 or follower_age < 0:
            raise RuntimeError("joint feedback timestamps are in the future.")
        if leader_age > self.config.feedback_timeout_s:
            raise RuntimeError(f"leader feedback is stale ({leader_age:.3f}s).")
        if follower_age > self.config.feedback_timeout_s:
            raise RuntimeError(f"follower feedback is stale ({follower_age:.3f}s).")

    def snapshot(self) -> TakeoverSnapshot:
        """Return the current safety state without exposing mutable arrays."""
        return TakeoverSnapshot(
            state=self.state,
            follower_hold_target=(
                None
                if self._follower_hold_target is None
                else self._follower_hold_target.copy()
            ),
            max_alignment_error_rad=self._max_alignment_error_rad,
            approach_stable_samples=self._approach_stable_samples,
            stable_samples=self._stable_samples,
            fault=self._fault,
        )

    def request_hold(self, follower_joints: np.ndarray) -> np.ndarray:
        """Stop policy ownership and latch the measured follower hold target."""
        if self.state not in {
            TakeoverState.POLICY,
            TakeoverState.TELEOP,
            TakeoverState.TELEOP_RETURN,
        }:
            raise RuntimeError(f"cannot request hold from state {self.state.value}.")
        self._follower_hold_target = self._joints(
            follower_joints, "follower_joints"
        ).copy()
        self._alignment_started_at = None
        self._approach_stable_samples = 0
        self._stable_samples = 0
        self._max_alignment_error_rad = None
        self._fault = None
        self.state = TakeoverState.FOLLOWER_HOLD
        return self._follower_hold_target.copy()

    def request_alignment(
        self,
        *,
        leader_joints: np.ndarray,
        follower_joints: np.ndarray,
        leader_timestamp: float,
        follower_timestamp: float,
        now: float,
    ) -> np.ndarray:
        """Begin leader alignment while preserving the latched follower hold."""
        if self.state != TakeoverState.FOLLOWER_HOLD:
            raise RuntimeError(
                f"cannot align leader from state {self.state.value}; request hold first."
            )
        leader = self._joints(leader_joints, "leader_joints")
        follower = self._joints(follower_joints, "follower_joints")
        self._require_fresh_feedback(
            now=now,
            leader_timestamp=leader_timestamp,
            follower_timestamp=follower_timestamp,
        )
        if self._follower_hold_target is None:
            raise RuntimeError("follower hold target is missing.")
        hold_error = np.max(np.abs(follower - self._follower_hold_target))
        if hold_error > self.config.position_tolerance_rad:
            raise RuntimeError(
                "follower moved away from its hold target before alignment "
                f"({hold_error:.4f}rad)."
            )
        self.state = TakeoverState.LEADER_ALIGNING
        self._alignment_started_at = float(now)
        self._approach_stable_samples = 0
        self._stable_samples = 0
        self._max_alignment_error_rad = float(
            np.max(np.abs(leader - self._follower_hold_target))
        )
        return self.next_alignment_target(
            leader_joints=leader,
            follower_joints=follower,
            leader_timestamp=leader_timestamp,
            follower_timestamp=follower_timestamp,
            now=now,
        )

    def request_manual_approach(
        self,
        *,
        leader_joints: np.ndarray,
        follower_joints: np.ndarray,
        leader_timestamp: float,
        follower_timestamp: float,
        now: float,
    ) -> None:
        """Enter monitored manual approach while preserving follower hold."""
        if self.state != TakeoverState.FOLLOWER_HOLD:
            raise RuntimeError(
                "cannot start manual leader approach from state "
                f"{self.state.value}; request hold first."
            )
        leader = self._joints(leader_joints, "leader_joints")
        follower = self._joints(follower_joints, "follower_joints")
        self._require_fresh_feedback(
            now=now,
            leader_timestamp=leader_timestamp,
            follower_timestamp=follower_timestamp,
        )
        self._require_follower_hold(follower)
        self.state = TakeoverState.LEADER_ALIGNING
        self._alignment_started_at = float(now)
        self._approach_stable_samples = 0
        self._stable_samples = 0
        self._max_alignment_error_rad = float(
            np.max(np.abs(leader - self._follower_hold_target))
        )

    def manual_approach_tick(
        self,
        *,
        leader_joints: np.ndarray,
        follower_joints: np.ndarray,
        leader_timestamp: float,
        follower_timestamp: float,
        now: float,
    ) -> None:
        """Monitor an operator-moved leader until it enters the near region."""
        if self.state != TakeoverState.LEADER_ALIGNING:
            raise RuntimeError(
                f"manual approach tick is invalid in state {self.state.value}."
            )
        leader = self._joints(leader_joints, "leader_joints")
        follower = self._joints(follower_joints, "follower_joints")
        self._require_fresh_feedback(
            now=now,
            leader_timestamp=leader_timestamp,
            follower_timestamp=follower_timestamp,
        )
        if self._alignment_started_at is None or self._follower_hold_target is None:
            raise RuntimeError("manual approach state is incomplete.")
        if now - self._alignment_started_at > self.config.alignment_timeout_s:
            self.fail("leader manual approach timed out")
            raise TimeoutError("leader manual approach timed out.")
        self._require_follower_hold(follower)
        self._max_alignment_error_rad = float(
            np.max(np.abs(leader - self._follower_hold_target))
        )
        if self._max_alignment_error_rad <= self.config.approach_tolerance_rad:
            self._approach_stable_samples += 1
            if (
                self._approach_stable_samples
                >= self.config.approach_stable_samples
            ):
                self.state = TakeoverState.LEADER_MANUAL_ALIGN
        else:
            self._approach_stable_samples = 0

    def next_alignment_target(
        self,
        *,
        leader_joints: np.ndarray,
        follower_joints: np.ndarray,
        leader_timestamp: float,
        follower_timestamp: float,
        now: float,
    ) -> np.ndarray:
        """Return one limited leader target until it reaches the near region."""
        if self.state != TakeoverState.LEADER_ALIGNING:
            raise RuntimeError(
                f"alignment tick is invalid in state {self.state.value}."
            )
        leader = self._joints(leader_joints, "leader_joints")
        follower = self._joints(follower_joints, "follower_joints")
        self._require_fresh_feedback(
            now=now,
            leader_timestamp=leader_timestamp,
            follower_timestamp=follower_timestamp,
        )
        if self._alignment_started_at is None or self._follower_hold_target is None:
            raise RuntimeError("alignment state is incomplete.")
        if now - self._alignment_started_at > self.config.alignment_timeout_s:
            self.fail("leader alignment timed out")
            raise TimeoutError("leader alignment timed out.")
        self._require_follower_hold(follower)

        delta = self._follower_hold_target - leader
        self._max_alignment_error_rad = float(np.max(np.abs(delta)))
        if self._max_alignment_error_rad <= self.config.approach_tolerance_rad:
            self._approach_stable_samples += 1
            target = leader.copy()
            if (
                self._approach_stable_samples
                >= self.config.approach_stable_samples
            ):
                self.state = TakeoverState.LEADER_MANUAL_ALIGN
        else:
            self._approach_stable_samples = 0
            max_step = self.config.max_alignment_step_rad
            target = leader + np.clip(delta, -max_step, max_step)
        return target

    def manual_alignment_tick(
        self,
        *,
        leader_joints: np.ndarray,
        follower_joints: np.ndarray,
        leader_timestamp: float,
        follower_timestamp: float,
        now: float,
    ) -> None:
        """Validate the operator's final drag alignment without commanding it."""
        if self.state != TakeoverState.LEADER_MANUAL_ALIGN:
            raise RuntimeError(
                f"manual alignment tick is invalid in state {self.state.value}."
            )
        leader = self._joints(leader_joints, "leader_joints")
        follower = self._joints(follower_joints, "follower_joints")
        self._require_fresh_feedback(
            now=now,
            leader_timestamp=leader_timestamp,
            follower_timestamp=follower_timestamp,
        )
        if self._alignment_started_at is None or self._follower_hold_target is None:
            raise RuntimeError("manual alignment state is incomplete.")
        if now - self._alignment_started_at > self.config.alignment_timeout_s:
            self.fail("leader manual alignment timed out")
            raise TimeoutError("leader manual alignment timed out.")
        self._require_follower_hold(follower)
        self._max_alignment_error_rad = float(
            np.max(np.abs(leader - self._follower_hold_target))
        )
        if self._max_alignment_error_rad <= self.config.position_tolerance_rad:
            self._stable_samples += 1
            if self._stable_samples >= self.config.stable_samples:
                self.state = TakeoverState.READY
        else:
            self._stable_samples = 0

    def _require_follower_hold(self, follower: np.ndarray) -> None:
        if self._follower_hold_target is None:
            raise RuntimeError("follower hold target is missing.")
        hold_error = float(np.max(np.abs(follower - self._follower_hold_target)))
        if hold_error > self.config.position_tolerance_rad:
            self.fail(
                "follower left hold during leader alignment "
                f"({hold_error:.4f}rad)"
            )
            raise RuntimeError(self._fault)

    def request_takeover(
        self,
        *,
        leader_joints: np.ndarray,
        follower_joints: np.ndarray,
        leader_timestamp: float,
        follower_timestamp: float,
        now: float,
    ) -> None:
        """Transfer ownership only after a fresh, still-aligned READY state."""
        if self.state != TakeoverState.READY:
            raise RuntimeError(
                f"cannot enter teleoperation from state {self.state.value}."
            )
        leader = self._joints(leader_joints, "leader_joints")
        follower = self._joints(follower_joints, "follower_joints")
        self._require_fresh_feedback(
            now=now,
            leader_timestamp=leader_timestamp,
            follower_timestamp=follower_timestamp,
        )
        error = float(np.max(np.abs(leader - follower)))
        if error > self.config.position_tolerance_rad:
            raise RuntimeError(
                f"leader/follower alignment changed before takeover ({error:.4f}rad)."
            )
        self._max_alignment_error_rad = error
        self.state = TakeoverState.TELEOP

    def finish_episode(self) -> None:
        """End replay while preserving teleop only for an operator return."""
        if self.state not in {
            TakeoverState.FOLLOWER_HOLD,
            TakeoverState.LEADER_MANUAL_ALIGN,
            TakeoverState.READY,
            TakeoverState.TELEOP,
        }:
            raise RuntimeError(
                f"cannot finish takeover episode from state {self.state.value}."
            )
        self.state = (
            TakeoverState.TELEOP_RETURN
            if self.state == TakeoverState.TELEOP
            else TakeoverState.TERMINAL_HOLD
        )

    def complete_teleop_return(self) -> None:
        """Close post-terminal teleop after both arm pairs reach the start."""
        if self.state != TakeoverState.TELEOP_RETURN:
            raise RuntimeError(
                f"cannot complete return from state {self.state.value}."
            )
        self.state = TakeoverState.TERMINAL_HOLD

    def resume_policy(self, follower_joints: np.ndarray) -> np.ndarray:
        """Re-seed policy control from measured follower joints.

        The caller must clear policy and RTC queues before invoking this method.
        """
        if self.state not in {
            TakeoverState.FOLLOWER_HOLD,
            TakeoverState.LEADER_MANUAL_ALIGN,
            TakeoverState.READY,
            TakeoverState.TERMINAL_HOLD,
        }:
            raise RuntimeError(
                f"cannot resume policy from state {self.state.value}."
            )
        measured = self._joints(follower_joints, "follower_joints").copy()
        self.state = TakeoverState.POLICY
        self._follower_hold_target = None
        self._alignment_started_at = None
        self._approach_stable_samples = 0
        self._stable_samples = 0
        self._max_alignment_error_rad = None
        self._fault = None
        return measured

    def fail(self, reason: str) -> None:
        """Latch a fault; hardware adapters must continue follower hold."""
        self._fault = str(reason)
        self.state = TakeoverState.FAULT
