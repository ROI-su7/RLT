"""Realtime action-chunk scheduler for the Piper OpenPI client.

This is an execution-side RTC implementation.  It starts the next policy
request while a suffix of the currently predicted action chunk is still being
executed.  Therefore the ROS control loop continues publishing commands while
the policy server is busy.

It is deliberately *not* called model-prefix RTC: stock OpenPI PI0.5 does not
accept ``prev_actions`` or ``prefix_len`` as model inputs, and the pi05_wire
checkpoint was not trained with those inputs.  Model-prefix RTC requires an
architecture and training change, not merely a client-side patch.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from typing import Any

import numpy as np

from openpi_client import base_policy as _base_policy


class RealtimeActionChunkBroker(_base_policy.BasePolicy):
    """Prefetch policy chunks and keep sending commands during inference.

    ``action_horizon`` actions are used from every model output.  Once only
    ``prefix_len`` old actions remain, the next request is made in a background
    thread using the then-current observation.  The old suffix is published
    while that request is pending.  If inference is slower than this suffix,
    the final command is held rather than leaving the robot with no command.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        *,
        action_horizon: int,
        prefix_len: int,
        control_hz: float,
        boundary_blend_steps: int = 0,
    ) -> None:
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        if not 0 < prefix_len < action_horizon:
            raise ValueError(
                "prefix_len must satisfy 0 < prefix_len < action_horizon; "
                f"got prefix_len={prefix_len}, action_horizon={action_horizon}"
            )
        if control_hz <= 0:
            raise ValueError("control_hz must be positive")
        if not 0 <= boundary_blend_steps < action_horizon:
            raise ValueError(
                "boundary_blend_steps must satisfy 0 <= steps < action_horizon; "
                f"got steps={boundary_blend_steps}, action_horizon={action_horizon}"
            )
        self._policy = policy
        self._action_horizon = action_horizon
        self._prefix_len = prefix_len
        self._control_hz = float(control_hz)
        self._boundary_blend_steps = int(boundary_blend_steps)
        self._lock = threading.Lock()
        self._active_actions: np.ndarray | None = None
        self._active_index = 0
        # Full model output and the monotonic time of the observation used to
        # request it.  The output is time-aligned only when it is promoted.
        self._pending_result: tuple[np.ndarray, float] | None = None
        self._request_thread: threading.Thread | None = None
        self._request_error: BaseException | None = None
        self._generation = 0
        self._last_action: np.ndarray | None = None
        self._hold_logged = False

    @staticmethod
    def _actions_from_result(result: Mapping[str, Any]) -> np.ndarray:
        if "actions" not in result:
            raise KeyError("Policy result has no 'actions' field")
        actions = np.asarray(result["actions"], dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError(
                "Expected policy actions with shape [chunk, action_dim], got "
                f"{actions.shape}"
            )
        return actions

    def _request_actions(self, observation: dict, generation: int, request_time: float) -> None:
        try:
            start_time = time.monotonic()
            actions = self._actions_from_result(self._policy.infer(observation))
            if actions.shape[0] < self._action_horizon:
                raise ValueError(
                    f"Policy returned {actions.shape[0]} actions, but "
                    f"action_horizon={self._action_horizon} was requested"
                )
            with self._lock:
                if generation == self._generation:
                    self._pending_result = (actions.copy(), request_time)
                    self._request_error = None
                    logging.info(
                        "RTC prefetch completed in %.0f ms; %d old actions remain.",
                        (time.monotonic() - start_time) * 1000,
                        self._prefix_len,
                    )
        except BaseException as exc:  # Preserve worker errors for the control thread.
            with self._lock:
                if generation == self._generation:
                    self._request_error = exc

    def _start_prefetch(self, observation: dict) -> None:
        if self._request_thread is not None and self._request_thread.is_alive():
            return
        self._request_error = None
        generation = self._generation
        request_time = time.monotonic()
        self._request_thread = threading.Thread(
            target=self._request_actions,
            args=(observation, generation, request_time),
            name="openpi-piper-rtc-prefetch",
            daemon=True,
        )
        self._request_thread.start()

    def _blocking_initial_request(self, observation: dict) -> None:
        result = self._policy.infer(observation)
        actions = self._actions_from_result(result)
        if actions.shape[0] < self._action_horizon:
            raise ValueError(
                f"Policy returned {actions.shape[0]} actions, but "
                f"action_horizon={self._action_horizon} was requested"
            )
        self._active_actions = actions[: self._action_horizon].copy()
        self._active_index = 0

    def _align_pending_actions(self, actions: np.ndarray, request_time: float, now: float) -> np.ndarray | None:
        """Select the future part of an action chunk that is still valid now.

        The policy chunk is referenced to the observation timestamp.  During
        server inference, the old chunk keeps moving the robot.  Starting the
        returned chunk from index zero would command a past pose and causes a
        visible extend-then-retract oscillation.  Interpolate the arm action
        chunk at the elapsed control-step offset before making it active.
        """
        offset = (now - request_time) * self._control_hz
        sample_times = offset + np.arange(self._action_horizon, dtype=np.float32)
        if sample_times[-1] > actions.shape[0] - 1:
            return None
        aligned = np.empty((self._action_horizon, actions.shape[1]), dtype=np.float32)
        source_times = np.arange(actions.shape[0], dtype=np.float32)
        for joint_idx in range(actions.shape[1]):
            aligned[:, joint_idx] = np.interp(sample_times, source_times, actions[:, joint_idx])
        # Gripper targets are discrete; preserve their timing with zero-order
        # hold rather than making a partially-open synthetic target.
        for joint_idx in (6, 13):
            if joint_idx < actions.shape[1]:
                indices = np.minimum(sample_times.astype(np.int64), actions.shape[0] - 1)
                aligned[:, joint_idx] = actions[indices, joint_idx]
        logging.info("RTC action chunk aligned by %.1f policy steps.", offset)
        return aligned

    def _blend_chunk_boundary(self, actions: np.ndarray) -> np.ndarray:
        """Blend arm joints from the prior chunk into a new action chunk."""
        if self._boundary_blend_steps == 0 or self._last_action is None:
            return actions
        steps = min(self._boundary_blend_steps, actions.shape[0])
        arm_indices = np.r_[0:6, 7:13]
        blended = actions.copy()
        weights = np.arange(1, steps + 1, dtype=np.float32) / steps
        blended[:steps, arm_indices] = (
            (1.0 - weights[:, None]) * self._last_action[arm_indices]
            + weights[:, None] * actions[:steps, arm_indices]
        )
        logging.info("RTC blended %d policy steps at action-chunk boundary.", steps)
        return blended

    def infer(self, obs: dict) -> dict:
        """Return one 14-D action without blocking at normal chunk rollover."""
        with self._lock:
            if self._active_actions is None:
                # The first action necessarily waits for the first model call.
                # Do not hold the lock over network inference.
                need_initial = True
            else:
                need_initial = False
        if need_initial:
            self._blocking_initial_request(obs)

        with self._lock:
            assert self._active_actions is not None

            # Once current actions are exhausted, atomically promote a prefetched
            # chunk.  This is the point that avoids ActionChunkBroker's blocking
            # server call in the real-time control loop.
            if self._active_index >= self._action_horizon and self._pending_result is not None:
                pending_actions, request_time = self._pending_result
                aligned_actions = self._align_pending_actions(pending_actions, request_time, time.monotonic())
                self._pending_result = None
                if aligned_actions is not None:
                    self._active_actions = self._blend_chunk_boundary(aligned_actions)
                    self._active_index = 0
                    self._hold_logged = False
                else:
                    logging.warning(
                        "RTC result is older than the model action horizon; discarding it and holding command."
                    )

            if self._active_index >= self._action_horizon:
                if self._request_error is not None:
                    error = self._request_error
                    self._request_error = None
                    raise RuntimeError("Background policy request failed") from error
                if self._last_action is None:
                    raise RuntimeError("No action is available to hold")
                if not self._hold_logged:
                    logging.warning(
                        "RTC prefetch did not finish before the action chunk ended; "
                        "holding the final command. Increase prefix_len or lower max_hz."
                    )
                    self._hold_logged = True
                return {"actions": self._last_action.copy()}

            action = self._active_actions[self._active_index].copy()
            self._active_index += 1
            self._last_action = action
            remaining = self._action_horizon - self._active_index
            if remaining == self._prefix_len:
                self._start_prefetch(obs)
            return {"actions": action}

    def reset(self) -> None:
        with self._lock:
            self._generation += 1
            request_thread = self._request_thread
            self._active_actions = None
            self._pending_result = None
            self._active_index = 0
            self._request_thread = None
            self._request_error = None
            self._last_action = None
            self._hold_logged = False
        # A websocket client is single-request-at-a-time.  Do not begin a new
        # episode on the same connection until a previous request has returned.
        if request_thread is not None and request_thread.is_alive():
            request_thread.join()
        self._policy.reset()
