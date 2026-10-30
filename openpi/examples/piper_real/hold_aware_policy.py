"""HOLD-aware policy wrapper for the real Piper runtime.

Space/HOLD must stop both robot command consumption and policy progression.
On I/resume this wrapper resets the action broker and asks the model again
from a fresh camera/joint observation instead of replaying an old chunk.
"""

from __future__ import annotations

import logging
from typing import Any

from openpi_client import base_policy as _base_policy


class HoldAwarePolicy(_base_policy.BasePolicy):
    def __init__(self, policy: _base_policy.BasePolicy, environment: Any) -> None:
        self._policy = policy
        self._environment = environment
        self._generation: int | None = None

    def infer(self, observation: dict) -> dict:
        current_observation = observation
        while True:
            generation = self._environment.wait_until_policy_execution_allowed()
            if self._generation != generation:
                self._policy.reset()
                self._generation = generation
                # The observation passed by Runtime may predate a long HOLD.
                current_observation = self._environment.get_observation()
                logging.info(
                    "OpenPI action broker reset after HOLD; replanning from fresh observation (epoch %d).",
                    generation,
                )

            result = self._policy.infer(current_observation)
            allowed, current_generation = self._environment.policy_execution_snapshot()
            if allowed and current_generation == generation:
                tagged = dict(result)
                tagged["_piper_hold_generation"] = generation
                return tagged

            # HOLD arrived while a websocket inference was in flight.  Never
            # return that result to the robot; the next loop waits for I and
            # resets the broker before issuing a fresh request.
            logging.info("Discarded an in-flight policy result because HOLD became active.")

    def reset(self) -> None:
        self._policy.reset()
        _, self._generation = self._environment.policy_execution_snapshot()
