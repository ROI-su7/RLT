from __future__ import annotations

from omegaconf import OmegaConf

from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import (
    AsyncRLTACFSDPPolicy,
    RLTACFSDPPolicy,
)


class _ScheduleHarness:
    _rlt_updates_to_run = RLTACFSDPPolicy._rlt_updates_to_run

    def __init__(self, *, replay_size: int, update_step: int):
        self.cfg = OmegaConf.create(
            {
                "algorithm": {
                    "update_epoch": 5,
                    "replay_buffer": {"min_buffer_size": 400},
                }
            }
        )
        self.rlt_schedule_cfg = {
            "warmup_min_size": 400,
            "warmup_post_collect_updates": 2000,
            "train_every_transitions": 1,
            "train_every_episodes": 0,
            "max_updates_per_train_step": 25,
        }
        self.update_step = update_step
        self.replay_size = replay_size
        self._warmup_ready_total_transitions = None
        self._warmup_ready_total_episodes = None
        self.pending_update_budget = 0
        self._last_replay_metrics = {}

    def _global_rlt_counters(self) -> dict[str, float]:
        return {
            "transitions_since_train": 0.0,
            "episodes_since_train": 0.0,
            "total_transitions_added": 0.0,
            "total_episodes_added": 0.0,
            "min_replay_size": float(self.replay_size),
            "min_demo_size": 0.0,
        }


def test_legacy_checkpoint_update_step_is_inferred_from_strict_utd() -> None:
    assert (
        RLTACFSDPPolicy._infer_rlt_update_step(
            replay_size=774,
            warmup_min_size=400,
            warmup_updates=2000,
            update_epoch=5,
        )
        == 3870
    )


def test_no_updates_before_paper_warmup_size() -> None:
    worker = _ScheduleHarness(replay_size=399, update_step=0)

    updates, metrics = worker._rlt_updates_to_run()

    assert updates == 0
    assert metrics["rlt/skip_reason"] == 1


def test_warmup_requires_five_updates_per_transition() -> None:
    worker = _ScheduleHarness(replay_size=400, update_step=0)

    updates, metrics = worker._rlt_updates_to_run()

    assert updates == 25
    assert metrics["rlt/desired_total_updates"] == 2000
    assert metrics["rlt/pending_update_budget"] == 2000


def test_preloaded_1051_transition_warmup_requires_5255_updates() -> None:
    worker = _ScheduleHarness(replay_size=1051, update_step=0)
    worker.cfg.algorithm.replay_buffer.min_buffer_size = 1051
    worker.rlt_schedule_cfg["warmup_min_size"] = 1051
    worker.rlt_schedule_cfg["warmup_post_collect_updates"] = 5255

    updates, metrics = worker._rlt_updates_to_run()

    assert updates == 25
    assert metrics["rlt/desired_total_updates"] == 5255
    assert metrics["rlt/pending_update_budget"] == 5255


def test_replay_growth_adds_exact_utd_five_credit() -> None:
    worker = _ScheduleHarness(replay_size=432, update_step=2000)

    updates, metrics = worker._rlt_updates_to_run()

    assert updates == 25
    assert metrics["rlt/desired_total_updates"] == 2160
    assert metrics["rlt/pending_update_budget"] == 160

    caught_up = _ScheduleHarness(replay_size=432, update_step=2160)
    updates, metrics = caught_up._rlt_updates_to_run()
    assert updates == 0
    assert metrics["rlt/skip_reason"] == 2


def test_async_worker_uses_the_same_strict_update_budget() -> None:
    assert (
        AsyncRLTACFSDPPolicy._rlt_updates_to_run is RLTACFSDPPolicy._rlt_updates_to_run
    )
