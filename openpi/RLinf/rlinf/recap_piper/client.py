"""RECAP OpenPI client with motion-preserving chunk handovers."""

import logging
import os
from types import SimpleNamespace

import numpy as np


def blend_arm_boundary(
    actions: np.ndarray,
    last_action: np.ndarray,
    previous_delta: np.ndarray,
    steps: int,
) -> np.ndarray:
    """Remove a boundary offset without suppressing the new plan's motion.

    Continuous micro-motion needs no extra filtering beyond the environment's
    existing EMA. For larger discontinuities, decay a position correction
    rather than mixing the whole new trajectory with the old slow trajectory.
    Grippers retain the model's original timing. The final blend target exactly
    matches the new plan, and correction cannot overshoot the window's bounds.
    """
    steps = min(steps, len(actions))
    if steps <= 0:
        return actions.copy()
    joints = np.r_[0:6, 7:13]
    result = actions.copy()
    target = actions[:steps, joints]
    last = last_action[joints]
    lower = np.minimum(last, target.min(axis=0))
    upper = np.maximum(last, target.max(axis=0))
    first = target[0]
    incoming = np.clip(
        last + previous_delta[joints], np.minimum(last, first), np.maximum(last, first)
    )
    offset = incoming - first
    # Preserve already-continuous motion, particularly sub-millimetre approach
    # corrections. Judge each arm together to retain its joint coordination.
    for arm in (slice(0, 6), slice(6, 12)):
        if np.max(np.abs(offset[arm])) <= 0.002:
            offset[arm] = 0.0
    phase = np.linspace(0.0, 1.0, steps) if steps > 1 else np.ones(1)
    weight = (phase * phase * (3.0 - 2.0 * phase))[:, None]
    result[:steps, joints] = np.clip(target + (1.0 - weight) * offset, lower, upper)
    return result


def alignment_horizon(
    actions: np.ndarray, ready: bool, offset: float, short: int, long: int = 30
) -> int:
    """Extend a stable two-gripper hold, never a predicted gripper transition.

    ``long <= short`` disables the extension (always the short schedule).
    """
    if long <= short:
        return short
    end = int(np.ceil(offset + long))
    if not ready or offset < 0 or end >= len(actions):
        return short
    # Inspect the complete prefix too: an opening during prefetch is a reason
    # to keep the short schedule, even if both grippers close again later.
    grips = actions[: end + 1, [6, 13]]
    if not np.isfinite(grips).all() or np.any(grips > 0.03):
        return short
    return long


def make_motion_broker(base: type) -> type:
    """Add stable-hold scheduling to the native broker and its HOLD lifecycle."""

    class MotionContinuousBroker(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._short_horizon = self._action_horizon
            # A/B switch: RECAP_STABLE_HOLD_HORIZON=10 keeps the short schedule
            # during the dual-gripper approach (more frequent replanning).
            self._long_horizon = int(os.environ.get("RECAP_STABLE_HOLD_HORIZON", "30"))
            self._seen_open = np.zeros(2, dtype=bool)
            self._stable_hold_count = 0

        def infer(self, obs: dict) -> dict:
            state = np.asarray(obs.get("state", []))
            if state.shape == (14,) and np.isfinite(state).all():
                grips = state[[6, 13]]
                self._seen_open |= grips > 0.035
                stable = bool(self._seen_open.all() and np.all(grips <= 0.03))
            else:
                stable = False
            self._stable_hold_count = self._stable_hold_count + 1 if stable else 0
            return super().infer(obs)

        def _align_pending_actions(
            self, actions: np.ndarray, request_time: float, now: float
        ) -> np.ndarray | None:
            # Change length only at promotion, never while an old chunk is
            # executing (which could otherwise skip its prefetch trigger).
            horizon = alignment_horizon(
                actions,
                self._stable_hold_count >= 3,
                (now - request_time) * self._control_hz,
                self._short_horizon,
                self._long_horizon,
            )
            if horizon != self._action_horizon:
                logging.info(
                    "RTC execution horizon %d -> %d (stable two-gripper hold=%s, prefix=%d).",
                    self._action_horizon,
                    horizon,
                    self._stable_hold_count >= 3,
                    self._prefix_len,
                )
            self._action_horizon = horizon
            return super()._align_pending_actions(actions, request_time, now)

        def reset(self) -> None:
            super().reset()
            self._action_horizon = self._short_horizon
            self._seen_open[:] = False
            self._stable_hold_count = 0

        def _blend_chunk_boundary(self, actions: np.ndarray) -> np.ndarray:
            if self._boundary_blend_steps == 0 or self._last_action is None:
                return actions
            delta = np.zeros_like(self._last_action)
            # The base promotes a pending chunk before replacing active_actions.
            # A starvation HOLD must not resume with the pre-HOLD velocity.
            if (
                not self._hold_logged
                and self._active_actions is not None
                and len(self._active_actions) >= 2
            ):
                delta = self._active_actions[-1] - self._active_actions[-2]
            result = blend_arm_boundary(
                actions, self._last_action, delta, self._boundary_blend_steps
            )
            logging.info(
                "RTC offset-decay handover over %d policy steps; max arm correction %.5f rad.",
                self._boundary_blend_steps,
                float(
                    np.max(
                        np.abs(
                            result[:, np.r_[0:6, 7:13]] - actions[:, np.r_[0:6, 7:13]]
                        )
                    )
                ),
            )
            return result

    return MotionContinuousBroker


def main() -> None:
    """Run the native ROS client with a scoped broker replacement."""
    import tyro
    from examples.piper_real import main as native

    # Only this entrypoint uses the replacement; the native source is untouched.
    native._rtc_broker = SimpleNamespace(
        RealtimeActionChunkBroker=make_motion_broker(
            native._rtc_broker.RealtimeActionChunkBroker
        )
    )
    logging.basicConfig(level=logging.INFO, force=True)
    native.main(tyro.cli(native.Args))


if __name__ == "__main__":
    main()
