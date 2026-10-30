"""Actor-only command-space limits; independent of measured feedback noise."""

from __future__ import annotations

import numpy as np


class ActorCommandProfile:
    """Bound accepted goal increments and continuously track them at publish Hz.

    The goal limiter follows openpi-RLT's previous-sent-command contract.
    A damped, acceleration-bounded tracker additionally prevents sudden
    velocity reversal. Its state persists across chunks and is reset on HOLD.
    The profile operates on command positions, not motor feedback.
    """

    def __init__(self, config: dict, position: np.ndarray, now: float):
        self.position = np.asarray(position, dtype=np.float64).copy()
        self.velocity = np.zeros(14, dtype=np.float64)
        self.goal = self.position.copy()
        self.last_time = float(now)
        self.lower = np.asarray(config["joint_position_limits_lower"], np.float64)
        self.upper = np.asarray(config["joint_position_limits_upper"], np.float64)
        self.max_step = self._limits(
            config["actor_command_joint_step_rad"],
            config["actor_command_gripper_step_m"],
        )
        self.max_velocity = self._limits(
            config["actor_command_joint_velocity_rad_s"],
            config["actor_command_gripper_velocity_m_s"],
        )
        self.max_acceleration = self._limits(
            config["actor_command_joint_acceleration_rad_s2"],
            config["actor_command_gripper_acceleration_m_s2"],
        )
        self.omega = 2.0 / float(config["actor_command_response_time_s"])
        self.integration_period = 1.0 / float(config["command_publish_hz"])
        self.max_observed_velocity = np.zeros(14)
        self.max_observed_acceleration = np.zeros(14)

    @staticmethod
    def _limits(arm: float, gripper: float) -> np.ndarray:
        values = np.full(14, float(arm), dtype=np.float64)
        values[[6, 13]] = float(gripper)
        return values

    def set_goal(self, requested: np.ndarray) -> np.ndarray:
        """Bound each goal relative to the previous accepted goal, across chunks."""
        requested = np.asarray(requested, dtype=np.float64)
        if requested.shape != (14,) or not np.isfinite(requested).all():
            raise ValueError("Actor commands must be finite 14-D targets.")
        requested = np.clip(requested, self.lower, self.upper)
        self.goal += np.clip(requested - self.goal, -self.max_step, self.max_step)
        self.goal = np.clip(self.goal, self.lower, self.upper)
        return self.goal.astype(np.float32)

    def sample(self, now: float) -> np.ndarray:
        """Advance a damped tracker without a catch-up jump after scheduler stalls."""
        elapsed = max(0.0, float(now) - self.last_time)
        self.last_time = float(now)
        # A paused publisher must not integrate a large unseen displacement.
        dt = min(elapsed, self.integration_period)
        if dt > 0:
            acceleration = np.clip(
                self.omega**2 * (self.goal - self.position)
                - 2.0 * self.omega * self.velocity,
                -self.max_acceleration,
                self.max_acceleration,
            )
            next_velocity = np.clip(
                self.velocity + acceleration * dt,
                -self.max_velocity,
                self.max_velocity,
            )
            self.max_observed_velocity = np.maximum(
                self.max_observed_velocity, np.abs(next_velocity)
            )
            self.max_observed_acceleration = np.maximum(
                self.max_observed_acceleration,
                np.abs((next_velocity - self.velocity) / dt),
            )
            next_position = self.position + (self.velocity + next_velocity) * (0.5 * dt)
            bounded = np.clip(next_position, self.lower, self.upper)
            next_velocity[bounded != next_position] = 0.0
            self.position = bounded
            self.velocity = next_velocity
        return self.position.astype(np.float32)

    def diagnostics(self) -> dict[str, float]:
        """Describe published command tracking, not measured robot motion."""
        arm = np.r_[0:6, 7:13]
        return {
            "arm_vmax": float(self.max_observed_velocity[arm].max()),
            "arm_amax": float(self.max_observed_acceleration[arm].max()),
            "arm_goal_lag": float(np.abs(self.goal - self.position)[arm].max()),
        }
