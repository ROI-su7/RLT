from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.subsample import pack_stride_windows
from rlinf.algorithms.rlt.transition import use_transition_replay
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACReplayMixin


class _ReplayStub:
    def __init__(self, flat: dict):
        self._flat = flat

    def _flatten_trajectory(self, trajectory: Trajectory) -> dict:
        del trajectory
        return self._flat


def test_realworld_piper_opts_into_transition_replay() -> None:
    cfg = OmegaConf.create(
        {
            "algorithm": {"replay_buffer": {"save_transitions": True}},
            "env": {"train": {"env_type": "realworld"}},
        }
    )

    assert use_transition_replay(cfg) is True


def test_piper_rlt_config_defers_manual_reset_until_after_terminal_replay() -> None:
    """A blocking mux reset must not consume a C/X/Q terminal action chunk."""
    config_path = (
        Path(__file__).parents[2]
        / "examples/embodiment/config/realworld_piper_rlt_stage2_ac_mlp.yaml"
    )
    cfg = OmegaConf.load(config_path)

    assert cfg.env.train.auto_reset is False
    assert cfg.env.eval.auto_reset is False


def test_actor_chunks_are_split_into_action_aligned_transitions() -> None:
    chunk = torch.zeros(2, 1, 5, 14)
    trajectory = Trajectory(
        max_episode_length=2,
        actions=chunk,
        # One initial empty outcome, followed by the outcomes of actions 0/1.
        rewards=torch.tensor([[[0.0]], [[0.0]], [[1.0]]]),
        dones=torch.tensor([[[False]], [[False]], [[True]]]),
        terminations=torch.tensor([[[False]], [[False]], [[True]]]),
        truncations=torch.zeros(3, 1, 1, dtype=torch.bool),
    )
    flat = {
        "actions": chunk[:, 0],
        "rewards": trajectory.rewards[:, 0],
        "forward_inputs": {
            "record_transition": torch.ones(2, 1, dtype=torch.bool),
        },
        "curr_obs": {
            "z_rl": torch.tensor([[1.0], [2.0]]),
            "proprio": torch.zeros(2, 14),
            "ref_chunk": torch.zeros(2, 5, 14),
        },
        "next_obs": {
            "z_rl": torch.tensor([[2.0], [3.0]]),
            "proprio": torch.zeros(2, 14),
            "ref_chunk": torch.zeros(2, 5, 14),
        },
    }
    worker = RLTACReplayMixin()
    worker.replay_buffer = _ReplayStub(flat)
    worker.cfg = SimpleNamespace(env=SimpleNamespace(train={"auto_reset": False}))

    transitions, completed = worker._transition_replay_trajectories(trajectory)

    assert len(transitions) == 2
    assert completed == 1
    assert transitions[0].actions.shape == (1, 1, 5, 14)
    assert transitions[0].dones.item() is False
    assert transitions[1].dones.item() is True
    assert transitions[0].curr_obs["z_rl"].item() == 1.0
    assert transitions[0].next_obs["z_rl"].item() == 2.0
    # Terminal rows deliberately reuse curr_obs; no post-reset observation can
    # leak across an episode boundary into the critic target.
    assert transitions[1].next_obs["z_rl"].item() == 2.0


@pytest.mark.parametrize(
    ("terminal_reward", "expected_positive"),
    [(1.0, True), (0.0, False)],
)
def test_only_b_to_terminal_window_is_persisted(
    terminal_reward: float,
    expected_positive: bool,
) -> None:
    """B opens replay collection; C/X closes its final transition."""
    chunk = torch.zeros(3, 1, 5, 14)
    trajectory = Trajectory(
        max_episode_length=3,
        actions=chunk,
        # The rollout has an initial outcome slot, hence both reward and done
        # for action row 2 are stored in outcome row 3.
        rewards=torch.tensor([[[0.0]], [[0.0]], [[0.0]], [[terminal_reward]]]),
        dones=torch.tensor([[[False]], [[False]], [[False]], [[True]]]),
        terminations=torch.tensor([[[False]], [[False]], [[False]], [[True]]]),
        truncations=torch.zeros(4, 1, 1, dtype=torch.bool),
    )
    flat = {
        "actions": chunk[:, 0],
        "rewards": trajectory.rewards[:, 0],
        # Row 0 precedes B. Rows 1 and 2 are the B-to-C/X critical window.
        "forward_inputs": {
            "record_transition": torch.tensor([[False], [True], [True]]),
        },
        "curr_obs": {
            "z_rl": torch.tensor([[1.0], [2.0], [3.0]]),
            "proprio": torch.zeros(3, 14),
            "ref_chunk": torch.zeros(3, 5, 14),
        },
        "next_obs": {
            "z_rl": torch.tensor([[2.0], [3.0], [4.0]]),
            "proprio": torch.zeros(3, 14),
            "ref_chunk": torch.zeros(3, 5, 14),
        },
    }
    worker = RLTACReplayMixin()
    worker.replay_buffer = _ReplayStub(flat)
    worker.cfg = SimpleNamespace(env=SimpleNamespace(train={"auto_reset": False}))

    transitions, completed = worker._transition_replay_trajectories(trajectory)

    assert len(transitions) == 2
    assert completed == 1
    assert transitions[0].curr_obs["z_rl"].item() == 2.0
    assert transitions[-1].dones.item() is True
    assert bool((transitions[-1].rewards > 0).any()) is expected_positive


def test_action_outcome_alignment_resets_for_each_rollout_epoch() -> None:
    """Each rollout epoch contributes its own initial empty outcome row."""
    chunk = torch.zeros(4, 1, 5, 14)
    trajectory = Trajectory(
        max_episode_length=4,
        actions=chunk,
        # Two epochs with two actions each:
        # [initial, action 0, action 1, initial, action 2, action 3].
        rewards=torch.tensor([[[0.0]], [[0.0]], [[1.0]], [[0.0]], [[0.0]], [[1.0]]]),
        dones=torch.tensor(
            [[[False]], [[False]], [[True]], [[False]], [[False]], [[True]]]
        ),
        terminations=torch.tensor(
            [[[False]], [[False]], [[True]], [[False]], [[False]], [[True]]]
        ),
        truncations=torch.zeros(6, 1, 1, dtype=torch.bool),
    )
    flat = {
        "actions": chunk[:, 0],
        "rewards": trajectory.rewards[:, 0],
        "forward_inputs": {
            "record_transition": torch.ones(4, 1, dtype=torch.bool),
        },
        "curr_obs": {
            "z_rl": torch.arange(4, dtype=torch.float32)[:, None],
            "proprio": torch.zeros(4, 14),
            "ref_chunk": torch.zeros(4, 5, 14),
        },
        "next_obs": {
            "z_rl": torch.arange(1, 5, dtype=torch.float32)[:, None],
            "proprio": torch.zeros(4, 14),
            "ref_chunk": torch.zeros(4, 5, 14),
        },
    }
    worker = RLTACReplayMixin()
    worker.replay_buffer = _ReplayStub(flat)
    worker.cfg = SimpleNamespace(env=SimpleNamespace(train={"auto_reset": True}))

    transitions, completed = worker._transition_replay_trajectories(trajectory)

    assert len(transitions) == 4
    assert completed == 2
    assert [transition.rewards.item() for transition in transitions] == [
        0.0,
        1.0,
        0.0,
        1.0,
    ]
    assert [transition.dones.item() for transition in transitions] == [
        False,
        True,
        False,
        True,
    ]


def test_piper_chunk_keeps_success_reward_and_done_on_same_control_step() -> None:
    """A C label inside a 20-step Piper chunk remains internally aligned."""
    chunk_size = 20
    success_step = 7
    actions = torch.zeros(2, 1, chunk_size * 14)
    rewards = torch.zeros(3, 1, chunk_size)
    dones = torch.zeros(3, 1, chunk_size, dtype=torch.bool)
    rewards[2, 0, success_step] = 1.0
    dones[2, 0, success_step] = True
    trajectory = Trajectory(
        max_episode_length=2,
        actions=actions,
        rewards=rewards,
        dones=dones,
        terminations=dones.clone(),
        truncations=torch.zeros_like(dones),
    )
    flat = {
        "actions": actions[:, 0],
        # This is deliberately the unaligned flatten result. The splitter must
        # use trajectory outcome row 2 for action row 1, not flat reward row 1.
        "rewards": rewards[:, 0],
        "forward_inputs": {
            "record_transition": torch.ones(2, 1, dtype=torch.bool),
        },
        "curr_obs": {
            "z_rl": torch.zeros(2, 1),
            "proprio": torch.zeros(2, 14),
            "ref_chunk": torch.zeros(2, chunk_size, 14),
        },
        "next_obs": {
            "z_rl": torch.ones(2, 1),
            "proprio": torch.zeros(2, 14),
            "ref_chunk": torch.zeros(2, chunk_size, 14),
        },
    }
    worker = RLTACReplayMixin()
    worker.replay_buffer = _ReplayStub(flat)
    worker.cfg = SimpleNamespace(env=SimpleNamespace(train={"auto_reset": False}))

    transitions, completed = worker._transition_replay_trajectories(trajectory)

    assert len(transitions) == 2
    assert completed == 1
    terminal = transitions[-1]
    assert terminal.rewards.shape == (1, 1, chunk_size)
    assert terminal.dones.shape == (1, 1, chunk_size)
    assert terminal.rewards[0, 0, success_step].item() == 1.0
    assert terminal.dones[0, 0, success_step].item() is True
    assert torch.equal(terminal.rewards.bool(), terminal.dones)


def test_stride_windows_become_replay_rows_with_the_chunk_row_schema() -> None:
    """Window rows and chunk rows are sampled into one batch, so they must match."""
    chunk_len, action_dim, horizon, z_dim = 10, 14, 50, 4
    window = {
        "actions": torch.ones(chunk_len, action_dim),
        "rewards": torch.zeros(chunk_len),
        "dones": torch.zeros(chunk_len, dtype=torch.bool),
        "intervene_flags": torch.zeros(chunk_len, dtype=torch.bool),
        "curr_z_rl": torch.full((z_dim,), 5.0),
        "curr_proprio": torch.zeros(14),
        "curr_ref_chunk": torch.zeros(horizon, action_dim),
        "curr_bridge_delta": torch.full((14,), 0.01),
        "next_z_rl": torch.full((z_dim,), 6.0),
        "next_proprio": torch.zeros(14),
        "next_ref_chunk": torch.zeros(horizon, action_dim),
        "next_bridge_delta": torch.full((14,), 0.02),
        "actor_switch": torch.tensor(False),
        "reference_warmup": torch.tensor(True),
        "episode_id": torch.tensor(42),
    }
    slots = pack_stride_windows(
        [[window]],
        slots=4,
        chunk_len=chunk_len,
        action_dim=action_dim,
        horizon=horizon,
        z_dim=z_dim,
        proprio_dim=14,
    )
    actions = torch.zeros(1, 1, chunk_len * action_dim)
    intervene_flags = torch.zeros(1, 1, chunk_len * action_dim, dtype=torch.bool)
    trajectory = Trajectory(
        max_episode_length=1,
        actions=actions,
        intervene_flags=intervene_flags,
        rewards=torch.zeros(2, 1, chunk_len),
        dones=torch.zeros(2, 1, chunk_len, dtype=torch.bool),
        terminations=torch.zeros(2, 1, chunk_len, dtype=torch.bool),
        truncations=torch.zeros(2, 1, chunk_len, dtype=torch.bool),
        versions=torch.zeros(1, 1, 1),
    )
    obs = {
        "z_rl": torch.zeros(1, z_dim),
        "proprio": torch.zeros(1, 14),
        "ref_chunk": torch.zeros(1, horizon, action_dim),
        "bridge_delta": torch.zeros(1, 14),
    }
    flat = {
        "actions": actions[:, 0],
        "intervene_flags": intervene_flags[:, 0],
        "rewards": trajectory.rewards[:, 0],
        "versions": torch.zeros(1, 1),
        "forward_inputs": {
            "record_transition": torch.ones(1, 1, dtype=torch.bool),
            "actor_switch": torch.zeros(1, 1, dtype=torch.bool),
            "reference_warmup": torch.ones(1, 1, dtype=torch.bool),
            "rlt_episode_id": torch.tensor([[42]]),
            "ref_chunk": torch.zeros(1, horizon, action_dim),
            **slots,
        },
        "curr_obs": obs,
        "next_obs": obs,
    }
    worker = RLTACReplayMixin()
    worker.replay_buffer = _ReplayStub(flat)
    worker.cfg = SimpleNamespace(env=SimpleNamespace(train={"auto_reset": True}))

    transitions, completed = worker._transition_replay_trajectories(trajectory)

    assert len(transitions) == 2
    assert completed == 0
    stride, chunk = transitions
    assert stride.curr_obs["z_rl"][0, 0, 0].item() == 5.0
    assert stride.next_obs["z_rl"][0, 0, 0].item() == 6.0
    assert stride.curr_obs["bridge_delta"].flatten()[0].item() == pytest.approx(0.01)
    assert stride.next_obs["bridge_delta"].flatten()[0].item() == pytest.approx(0.02)
    assert set(chunk.forward_inputs) == set(stride.forward_inputs)
    assert "rlt_stride_valid" not in chunk.forward_inputs
    for field_name in (
        "actions",
        "intervene_flags",
        "rewards",
        "dones",
        "terminations",
        "truncations",
        "versions",
    ):
        stride_value, chunk_value = getattr(stride, field_name), getattr(chunk, field_name)
        assert stride_value.shape == chunk_value.shape, field_name
        assert stride_value.dtype == chunk_value.dtype, field_name
    for group in ("curr_obs", "next_obs"):
        assert set(getattr(chunk, group)) == set(getattr(stride, group)), group
    for group in ("curr_obs", "next_obs", "forward_inputs"):
        for key, chunk_value in getattr(chunk, group).items():
            stride_value = getattr(stride, group)[key]
            assert stride_value.shape == chunk_value.shape, (group, key)
            assert stride_value.dtype == chunk_value.dtype, (group, key)
