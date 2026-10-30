"""Replay-only Stage-2 runner for the real-world RLT actor and critics."""

from __future__ import annotations

import os
from typing import Any

from omegaconf import DictConfig

from rlinf.utils.logging import get_logger


class RLTReplayOnlyRunner:
    """Train RLT Actor/Critic from a persisted replay without an environment."""

    def __init__(self, cfg: DictConfig, actor: Any) -> None:
        self.cfg = cfg
        self.actor = actor
        self.logger = get_logger()
        self.max_steps = int(cfg.runner.max_steps)
        self.save_interval = int(cfg.runner.save_interval)
        if self.max_steps <= 0:
            raise ValueError("Replay-only Stage 2 requires runner.max_steps > 0.")
        if cfg.runner.get("resume_dir", None) is not None:
            raise ValueError(
                "Replay-only scratch training does not accept runner.resume_dir; "
                "load only algorithm.replay_buffer.load_path."
            )

    def _checkpoint_path(self, step: int) -> str:
        return os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            "checkpoints",
            f"global_step_{step}",
            "actor",
        )

    def _save_checkpoint(self, step: int) -> None:
        path = self._checkpoint_path(step)
        os.makedirs(path, exist_ok=True)
        self.logger.info("Saving replay-only checkpoint at global step %d.", step)
        self.actor.save_checkpoint(path, step).wait()

    def run(self) -> None:
        """Run exactly ``runner.max_steps`` learner loops and save checkpoints."""
        for step in range(1, self.max_steps + 1):
            self.actor.set_global_step(step - 1)
            self.actor.run_training().wait()
            if self.save_interval > 0 and step % self.save_interval == 0:
                self._save_checkpoint(step)

        if self.save_interval <= 0 or self.max_steps % self.save_interval != 0:
            self._save_checkpoint(self.max_steps)
        self.logger.info(
            "Replay-only Stage 2 finished at global step %d.", self.max_steps
        )
