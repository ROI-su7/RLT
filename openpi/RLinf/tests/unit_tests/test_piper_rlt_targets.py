"""Stage-2 target-network, discount, and delayed-actor contracts."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACLossMixin


def _worker() -> RLTACLossMixin:
    worker = RLTACLossMixin()
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {
                "gamma": 0.5,
                "bootstrap_type": "standard",
                "q_head_type": "default",
                "rlt_schedule": {},
                "replay_buffer": {"save_transitions": True},
            },
            "actor": {"model": {"num_action_chunks": 3, "action_dim": 2}},
            "env": {"train": {"env_type": "realworld"}},
        }
    )
    worker.torch_dtype = torch.float32
    return worker


def _obs() -> dict[str, torch.Tensor]:
    return {
        "z_rl": torch.zeros(1, 4),
        "proprio": torch.zeros(1, 2),
        "ref_chunk": torch.zeros(1, 3, 2),
    }


def test_partial_chunk_return_uses_only_executed_steps() -> None:
    worker = _worker()
    rewards = torch.tensor([[1.0, 2.0, 99.0]])
    valid = torch.tensor([[True, True, False]])

    value, bootstrap = worker._discounted_chunk_return(rewards, valid)

    torch.testing.assert_close(value, torch.tensor([[2.0]]))
    torch.testing.assert_close(bootstrap, torch.tensor([[0.25]]))


def test_full_chunk_without_mask_keeps_fixed_horizon_behavior() -> None:
    worker = _worker()
    value, bootstrap = worker._discounted_chunk_return(
        torch.tensor([[1.0, 2.0, 4.0]]), None
    )

    torch.testing.assert_close(value, torch.tensor([[3.0]]))
    torch.testing.assert_close(bootstrap, torch.tensor([[0.125]]))


@pytest.mark.parametrize(
    "valid",
    [torch.tensor([[False, False, False]]), torch.tensor([[True, False, True]])],
)
def test_invalid_execution_masks_fail_closed(valid: torch.Tensor) -> None:
    worker = _worker()
    with pytest.raises(ValueError):
        worker._discounted_chunk_return(torch.zeros(1, 3), valid)


def test_critic_bootstraps_from_ema_actor_and_target_twin_q() -> None:
    worker = _worker()
    calls = []

    class TargetModel:
        def __call__(self, *, forward_type, obs, actions=None, **kwargs):
            calls.append((forward_type, actions, kwargs))
            if forward_type == ForwardType.SAC:
                assert actions is None
                assert kwargs == {}
                return torch.full((1, 6), 2.0), None, None
            assert forward_type == ForwardType.SAC_Q
            torch.testing.assert_close(actions, torch.full((1, 6), 3.0))
            return torch.tensor([[2.0, 4.0]])

    class OnlineModel:
        def __call__(self, *, forward_type, actions=None, **kwargs):
            assert forward_type == ForwardType.SAC_Q
            assert actions is not None
            return torch.zeros(1, 2)

    worker.target_model = TargetModel()
    worker.model = OnlineModel()
    worker._apply_execution_transform = lambda actions, ref: actions + 1.0
    batch = {
        "curr_obs": _obs(),
        "next_obs": _obs(),
        "actions": torch.zeros(1, 6),
        "rewards": torch.tensor([[1.0, 2.0, 99.0]]),
        "dones": torch.tensor([[False, False, False]]),
        "terminations": torch.tensor([[False, False, False]]),
        "forward_inputs": {"action_valid_mask": torch.tensor([[True, True, False]])},
    }

    loss, _ = RLTACLossMixin.forward_critic.__wrapped__(worker, batch)

    # Return=2, gamma**2=.25, min target-Q=2 -> TD target=2.5.
    torch.testing.assert_close(loss, torch.tensor(6.25))
    assert [call[0] for call in calls] == [ForwardType.SAC, ForwardType.SAC_Q]


def test_actor_update_gate_is_exactly_two_to_one() -> None:
    observed = [RLTACLossMixin._should_update_actor(step, 2) for step in range(6)]
    assert observed == [True, False, True, False, True, False]
    with pytest.raises(ValueError, match="positive"):
        RLTACLossMixin._should_update_actor(0, 0)


def test_actor_objective_uses_minimum_twin_q() -> None:
    worker = _worker()
    values = torch.tensor([[3.0, 1.0], [-2.0, 4.0]])

    torch.testing.assert_close(worker._actor_q(values), torch.tensor([[1.0], [-2.0]]))


@pytest.mark.parametrize("human", [False, True])
def test_raw_actor_bc_keeps_gradient_beyond_execution_limit(human: bool) -> None:
    worker = _worker()
    worker.cfg.algorithm.bc_weight = 10.0
    worker.cfg.algorithm.q_weight = 1.0
    raw = torch.full((1, 6), 0.3, requires_grad=True)
    q_actions = []

    class IdentitySpace:
        @staticmethod
        def normalize(actions, proprio):
            return actions

    class Model:
        def __call__(self, *, forward_type, actions=None, **kwargs):
            if forward_type == ForwardType.SAC:
                return raw, torch.zeros_like(raw), None
            assert forward_type == ForwardType.SAC_Q
            q_actions.append(actions.detach().clone())
            return actions.mean(dim=-1, keepdim=True).expand(-1, 2)

    worker.model = Model()
    worker._rlt_action_space = lambda device: IdentitySpace()
    worker._apply_execution_transform = lambda actions, ref: actions.clamp(-0.03, 0.03)
    batch = {
        "curr_obs": _obs(),
        "actions": torch.full((1, 6), 0.1),
        "intervene_flags": torch.full((1, 6), human),
        "forward_inputs": {"action_valid_mask": torch.tensor([[True, True, False]])},
    }

    loss, _, metrics = RLTACLossMixin.forward_actor.__wrapped__(worker, batch)
    expected_bc = (0.3 - (0.1 if human else 0.0)) ** 2
    assert metrics["bc_loss"] == pytest.approx(expected_bc)
    torch.testing.assert_close(q_actions[0], torch.full((1, 6), 0.03))
    loss.backward()
    assert (raw.grad[:, :4] > 0).all()
    torch.testing.assert_close(raw.grad[:, 4:], torch.zeros(1, 2))


def test_piper_actor_weights_match_b_stage2() -> None:
    config_path = (
        Path(__file__).parents[2]
        / "examples/embodiment/config/realworld_piper_rlt_stage2_ac_mlp.yaml"
    )
    cfg = OmegaConf.load(config_path)

    assert cfg.algorithm.actor_agg_q == "min"
    assert cfg.algorithm.target_policy_smoothing.enable is True
    assert cfg.algorithm.rlt_schedule.actor_correction_basis_type == "dct"
    assert cfg.algorithm.rlt_schedule.actor_correction_basis_size == 3
    assert cfg.algorithm.bc_weight == 5.0
    # 2026-10-19: Q 0.1 in one step pinned J2 at the correction bound, so
    # it is ramped in; the 2026-10-23 fresh campaign ramps it after the 3000
    # post-collect updates.
    assert cfg.algorithm.q_weight == 0.01
    assert cfg.algorithm.actor_weight_schedule.warmup_q_weight == 0.0
    assert cfg.algorithm.actor_weight_schedule.online_bc_weight == 5.0
    assert cfg.algorithm.actor_weight_schedule.online_q_weight == 0.01
    assert cfg.algorithm.actor_weight_schedule.warmup_updates == 3000
    assert cfg.algorithm.actor_weight_schedule.ramp_updates == 2000
    # Bridge 15 (2026-10-15): bridges 5 and 10 left plain JAX crawling.
    assert cfg.env.train.override_cfg.rtc_max_bridge_steps == 15
    assert cfg.algorithm.reference_dropout_prob == 0.0
    assert cfg.algorithm.bc_exclude_replaced_human_reference is True
    assert cfg.algorithm.bc_policy_row_weight == 1.0
    # Actor and critic share one RL-Token-Pi05 stratified batch.
    assert "critic_sample_strategy" not in cfg.algorithm.replay_buffer
    assert cfg.actor.model.actor_output == "residual"
    assert list(cfg.algorithm.td_target_clip) == [0.0, 1.0]
    assert cfg.actor.model.q_head_type == "rlt_projected"
    assert cfg.actor.model.actor_architecture == "projected"
    assert cfg.algorithm.rlt_schedule.critical_reference_speed == 1.0
    assert cfg.algorithm.rlt_schedule.human_rows_keep_jax_reference is True
    assert cfg.algorithm.replay_buffer.sample_strategy == "rlt_stratified"
    assert dict(cfg.algorithm.replay_buffer.stratified_ratios) == {
        "recent_online": 0.4,
        "warmup": 0.3,
        "human": 0.2,
    }
    assert cfg.algorithm.delta_weight == 10.0
    assert cfg.algorithm.actor_weight_schedule.enable is True
    assert cfg.algorithm.replay_buffer.min_buffer_size == 1000
    assert cfg.algorithm.rlt_schedule.warmup_min_size == 1000
    assert cfg.algorithm.rlt_schedule.warmup_post_collect_updates == 3000
    # Fresh campaign (2026-10-23): Q ramps in after the post-collect warmup.
    assert cfg.algorithm.actor_weight_schedule.warmup_updates == 3000
    assert cfg.algorithm.rlt_schedule.actor_direct_execution is False
    # Wrist joints J4-J6 may correct up to 0.04 rad (operator wrist twist).
    assert list(cfg.algorithm.rlt_schedule.max_actor_reference_arm_delta) == [
        0.015,
        0.015,
        0.015,
        0.04,
        0.04,
        0.04,
    ]
    assert cfg.algorithm.rlt_schedule.max_actor_reference_gripper_delta == 0.0015
    assert cfg.algorithm.rlt_schedule.max_actor_takeover_arm_deviation is None
    assert cfg.algorithm.rlt_schedule.max_actor_takeover_gripper_deviation is None
    assert cfg.algorithm.rlt_schedule.actor_refinement_smoothing_alpha is None
    assert cfg.algorithm.rlt_schedule.actor_refinement_tail_taper_steps == 0
    assert cfg.actor.model.fixed_std == 0.002
    assert cfg.actor.model.fixed_std_is_normalized is True
    assert cfg.actor.model.execute_actor_mean is True
    assert cfg.actor.model.actor_hidden_dim == 256
    assert cfg.actor.model.actor_num_layers == 2
    assert cfg.env.train.override_cfg.actor_command_profile_enabled is True
    assert cfg.actor.optim.lr == 1e-4


def test_openpi_rlt_actor_uses_q1_and_phase_weights() -> None:
    worker = _worker()
    worker.cfg.algorithm.actor_agg_q = "first"
    torch.testing.assert_close(
        worker._actor_q(torch.tensor([[2.0, 1.0]])), torch.tensor([[2.0]])
    )
    worker.cfg.algorithm.actor_weight_schedule = {
        "enable": True,
        "warmup_updates": 5000,
        "warmup_bc_weight": 10.0,
        "warmup_q_weight": 0.1,
        "online_bc_weight": 5.0,
        "online_q_weight": 0.1,
    }
    worker.update_step = 4999
    assert worker._actor_objective_weights()[:2] == (10.0, 0.1)
    worker.update_step = 5000
    assert worker._actor_objective_weights()[:2] == (5.0, 0.1)


def test_openpi_rlt_delta_penalty_uses_physical_arms_and_excludes_grippers() -> None:
    worker = _worker()
    worker.cfg.actor.model.action_dim = 14
    raw = torch.full((1, 3, 14), 0.02)
    raw[:, 1] = -0.02
    raw[..., 6::7] = 1.0
    raw[:, 1, 6::7] = -1.0
    raw.requires_grad_()
    batch = {
        "curr_obs": {"ref_chunk": torch.zeros(1, 3, 14)},
        "actions": torch.zeros_like(raw),
    }
    penalty, _ = worker._delta_penalty(raw, batch)
    torch.testing.assert_close(penalty, torch.tensor(0.0016))
    penalty.backward()
    torch.testing.assert_close(
        raw.grad[..., 6::7], torch.zeros_like(raw.grad[..., 6::7])
    )
    assert raw.grad[..., :6].abs().min() > 0


@pytest.mark.parametrize("human", [False, True])
def test_delta_penalty_preserves_reference_or_human_motion(human: bool) -> None:
    worker = _worker()
    target = torch.tensor([[0.0, 0.0, 0.1, 0.1, 0.2, 0.2]])
    raw = (target + 0.01).requires_grad_()
    batch = {
        "curr_obs": {"ref_chunk": torch.zeros_like(target) if human else target},
        "actions": target,
        "intervene_flags": torch.full_like(target, human, dtype=torch.bool),
    }
    penalty, _ = worker._delta_penalty(raw, batch)
    torch.testing.assert_close(penalty, torch.tensor(0.0), atol=1e-7, rtol=0)


def test_delta_penalty_masks_padding_and_source_changes() -> None:
    worker = _worker()
    raw = torch.tensor([[100.0, 100.0, 1.0, 1.0, 2.0, 2.0]], requires_grad=True)
    batch = {
        "curr_obs": {"ref_chunk": torch.zeros_like(raw)},
        "actions": torch.tensor([[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]]),
        "intervene_flags": torch.tensor([[False, False, True, True, True, True]]),
        "forward_inputs": {"action_valid_mask": torch.tensor([[True, True, False]])},
    }
    penalty, metrics = worker._delta_penalty(raw, batch)
    assert penalty.item() == 0
    assert metrics["delta_valid_pairs"] == 0
    penalty.backward()
    torch.testing.assert_close(raw.grad, torch.zeros_like(raw))


def test_direct_actor_q_keeps_bounds_without_old_smoothing() -> None:
    worker = _worker()
    worker.cfg.algorithm.rlt_schedule = {
        "actor_direct_execution": True,
        "max_actor_reference_arm_delta": 0.01,
        "actor_refinement_smoothing_alpha": 0.15,
        "actor_refinement_tail_taper_steps": 3,
    }
    raw = torch.full((1, 6), 0.1, requires_grad=True)
    result = worker._apply_execution_transform(raw, torch.zeros_like(raw))
    torch.testing.assert_close(result, torch.full((1, 6), 0.01))
    result.sum().backward()
    torch.testing.assert_close(raw.grad, torch.zeros_like(raw))


def test_target_policy_smoothing_adds_clipped_noise_before_execution() -> None:
    worker = _worker()
    worker.cfg.algorithm.target_policy_smoothing = {
        "enable": True,
        "arm_std": 1.0,
        "arm_clip": 0.01,
        "gripper_std": 0.0,
        "gripper_clip": 0.0,
    }
    worker.cfg.actor.model.action_dim = 7
    worker.cfg.actor.model.num_action_chunks = 2
    actions = torch.zeros(64, 14)

    noisy = worker._target_policy_smoothing(actions).reshape(64, 2, 7)

    assert noisy[..., :6].abs().max().item() == pytest.approx(0.01)
    torch.testing.assert_close(noisy[..., 6], torch.zeros(64, 2))
    worker.cfg.algorithm.target_policy_smoothing.enable = False
    assert worker._target_policy_smoothing(actions) is actions


def test_critic_target_smooths_before_the_execution_transform() -> None:
    worker = _worker()
    order = []
    worker._target_policy_smoothing = lambda actions: order.append("noise") or actions
    worker._apply_execution_transform = lambda actions, ref: (
        order.append("transform") or actions
    )

    class Model:
        def __call__(self, *, forward_type, **kwargs):
            if forward_type == ForwardType.SAC:
                return torch.zeros(1, 6), None, None
            return torch.zeros(1, 2)

    worker.target_model = Model()
    worker.model = Model()
    batch = {
        "curr_obs": _obs(),
        "next_obs": _obs(),
        "actions": torch.zeros(1, 6),
        "rewards": torch.zeros(1, 3),
        "dones": torch.zeros(1, 3, dtype=torch.bool),
        "terminations": torch.zeros(1, 3, dtype=torch.bool),
        "forward_inputs": {},
    }
    RLTACLossMixin.forward_critic.__wrapped__(worker, batch)
    assert order == ["noise", "transform"]


def test_td_target_clip_bounds_the_bootstrapped_target() -> None:
    worker = _worker()
    worker.cfg.algorithm.td_target_clip = [0.0, 1.0]

    class Model:
        def __call__(self, *, forward_type, **kwargs):
            if forward_type == ForwardType.SAC:
                return torch.zeros(1, 6), None, None
            return torch.full((1, 2), 100.0)

    worker.target_model = Model()
    worker.model = lambda **kwargs: torch.zeros(1, 2)
    worker._apply_execution_transform = lambda actions, ref: actions
    batch = {
        "curr_obs": _obs(),
        "next_obs": _obs(),
        "actions": torch.zeros(1, 6),
        "rewards": torch.zeros(1, 3),
        "dones": torch.zeros(1, 3, dtype=torch.bool),
        "terminations": torch.zeros(1, 3, dtype=torch.bool),
        "forward_inputs": {},
    }
    loss, _ = RLTACLossMixin.forward_critic.__wrapped__(worker, batch)
    # gamma**3 * 100 = 12.5 would be the unclipped target; the clip caps it at 1.
    torch.testing.assert_close(loss, torch.tensor(1.0))


def test_head_pose_shaping_enters_the_critic_target_and_widens_the_clip() -> None:
    worker = _worker()
    worker.cfg.algorithm.td_target_clip = [0.0, 1.0]
    worker.cfg.algorithm.head_pose_shaping = {"enable": True, "weight": 0.05}

    class Model:
        def __call__(self, *, forward_type, **kwargs):
            if forward_type == ForwardType.SAC:
                return torch.zeros(1, 6), None, None
            return torch.full((1, 2), 100.0)

    worker.target_model = Model()
    worker.model = lambda **kwargs: torch.zeros(1, 2)
    worker._apply_execution_transform = lambda actions, ref: actions
    near = torch.tensor([[10.0, -5.0, 0.0, 1.0]])
    batch = {
        "curr_obs": {**_obs(), "head_pose": near},
        "next_obs": {**_obs(), "head_pose": torch.tensor([[0.0, -4.0, 0.0, 1.0]])},
        "actions": torch.zeros(1, 6),
        "rewards": torch.zeros(1, 3),
        "dones": torch.ones(1, 3, dtype=torch.bool),
        "terminations": torch.ones(1, 3, dtype=torch.bool),
        "forward_inputs": {},
    }
    loss, metrics = RLTACLossMixin.forward_critic.__wrapped__(worker, batch)
    # Terminal row: target = shaping only = 0 - (-0.05).
    torch.testing.assert_close(loss, torch.tensor(0.05**2))
    assert metrics["head_pose_shaped_fraction"] == 1.0

    batch["dones"] = batch["terminations"] = torch.zeros(1, 3, dtype=torch.bool)
    loss, _ = RLTACLossMixin.forward_critic.__wrapped__(worker, batch)
    # The upper clip grows by the largest |potential| (0.05 * 30 / 10).
    torch.testing.assert_close(loss, torch.tensor(1.15**2))


def test_piper_config_adds_head_pose_state_and_shaping() -> None:
    config_path = (
        Path(__file__).parents[2]
        / "examples/embodiment/config/realworld_piper_rlt_stage2_ac_mlp.yaml"
    )
    cfg = OmegaConf.load(config_path)

    assert cfg.env.train.override_cfg.head_pose_weights.endswith("head_pose_net_v2.pt")
    assert cfg.actor.model.head_pose_dim == 4
    assert list(cfg.actor.model.head_pose_scale) == [20.0, 50.0, 0.3, 1.0]
    shaping = cfg.algorithm.head_pose_shaping
    assert shaping.enable is True
    assert (shaping.weight, shaping.scale_px, shaping.clip_px) == (0.05, 10.0, 30.0)
    assert (shaping.near_px, shaping.far_px) == (15.0, 60.0)
