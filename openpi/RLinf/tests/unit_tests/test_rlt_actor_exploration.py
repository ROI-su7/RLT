# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Smooth Actor exploration, explore/eval episodes and episode statistics."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.route import (
    RealworldRLTRoute,
    RLTRouteContext,
    build_rlt_route,
    project_actor_refinement,
    smooth_exploration_noise,
)
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACLossMixin

C, D, H = 10, 14, 50
ROOT = Path(__file__).parents[2]
CONFIG_DIR = ROOT / "examples/embodiment/config"


def _route(**kwargs) -> RealworldRLTRoute:
    options = {
        "use_reference_warmup": False,
        "warmup_updates": 0,
        "max_actor_reference_arm_delta": 0.01,
        "max_actor_reference_gripper_delta": 0.0015,
        "actor_correction_basis_size": 3,
        "exploration_arm_std": 0.003,
        "exploration_gripper_std": 0.0005,
        "exploration_eval_every_n_episodes": 3,
    }
    options.update(kwargs)
    return RealworldRLTRoute(**options)


def _step(route, *, critical=True, mode="train", version=0):
    plan = torch.zeros(1, H, D)
    output = route.route(
        RLTRouteContext(
            env_obs={},
            rlt_obs={"ref_chunk": plan},
            # The mean Actor equals the reference: any change is exploration.
            student_actions=plan[:, :C].clone(),
            result={"forward_inputs": {"ref_chunk": plan}},
            mode=mode,
            rlt_switch_flags=torch.tensor([[critical]]),
            version=version,
        )
    )
    return output.actions


def test_noise_is_smooth_and_has_the_requested_per_step_rms():
    std = torch.full((D,), 0.003)
    std[6::7] = 0.0005
    noise = smooth_exploration_noise(20000, C, std, basis_size=3)
    torch.testing.assert_close(
        project_actor_refinement(noise, basis_size=3), noise, atol=1e-7, rtol=0
    )
    rms = noise.square().mean(dim=(0, 1)).sqrt()
    torch.testing.assert_close(rms, std, rtol=0.03, atol=0)


def test_episodes_explore_except_every_third_which_runs_the_mean_actor():
    route = _route()
    explored = []
    for _ in range(6):
        chunks = [_step(route) for _ in range(3)]
        _step(route, critical=False)  # C/X, reset, I before the next B
        moved = [bool(chunk.abs().max() > 0) for chunk in chunks]
        assert len(set(moved)) == 1  # decided once per episode
        explored.append(moved[0])
        for chunk in chunks:
            assert chunk[..., [0, 1, 2, 3, 4, 5]].abs().max() <= 0.01 + 1e-7
            assert chunk[..., 6].abs().max() <= 0.0015 + 1e-7
    assert explored == [True, True, False, True, True, False]
    assert route._actor_episode_count == 6


def test_no_noise_in_eval_mode_before_the_gate_or_when_disabled():
    assert _step(_route(), mode="eval").abs().max() == 0
    gated = _route(use_reference_warmup=True, warmup_updates=3000)
    assert _step(gated, version=10).abs().max() == 0
    assert gated._actor_episode_count == 0
    assert (
        _step(_route(exploration_arm_std=0.0, exploration_gripper_std=0.0)).abs().max()
        == 0
    )


def test_exploration_requires_the_smooth_basis():
    with pytest.raises(ValueError, match="actor_correction_basis_size"):
        _route(actor_correction_basis_size=None)


def test_configs_enable_exploration_only_for_training():
    stage2 = OmegaConf.load(CONFIG_DIR / "realworld_piper_rlt_stage2_ac_mlp.yaml")
    schedule = OmegaConf.to_container(stage2.algorithm.rlt_schedule)
    route = build_rlt_route(
        OmegaConf.create(
            {
                "algorithm": {"rlt_schedule": schedule},
                "env": {"train": {"override_cfg": {"rtc_enabled": True}}},
            }
        )
    )
    assert route.exploration_arm_std == pytest.approx(0.003)
    assert route.exploration_gripper_std == pytest.approx(0.0005)
    assert route.exploration_chunk_correlation == pytest.approx(0.9)
    assert route.exploration_eval_every_n_episodes == 3
    diagnostic = OmegaConf.load(
        CONFIG_DIR / "realworld_piper_rlt_stage2_zero_correction.yaml"
    )
    assert diagnostic.algorithm.rlt_schedule.actor_exploration.enable is False


def test_q_action_contrast_measures_min_q_change_under_smooth_noise():
    worker = RLTACLossMixin()
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {
                "rlt_schedule": {
                    "actor_correction_basis_size": 3,
                    "actor_exploration": {"arm_std": 0.003, "gripper_std": 0.0005},
                }
            },
            "actor": {"model": {"num_action_chunks": C, "action_dim": D}},
        }
    )

    class Model:
        def __call__(self, *, forward_type, actions, **kwargs):
            assert forward_type == ForwardType.SAC_Q
            q = actions.sum(dim=-1, keepdim=True)
            return torch.cat([q, q + 1.0], dim=-1)

    worker.model = Model()
    obs = {"ref_chunk": torch.zeros(4, C, D)}
    raw = torch.zeros(4, C * D)
    metrics = worker._q_action_contrast(obs, raw, torch.zeros(4, 1))
    assert metrics["q_action_contrast"] > 0
    worker.cfg.algorithm.rlt_schedule.actor_exploration.arm_std = 0.0
    assert worker._q_action_contrast(obs, raw, torch.zeros(4, 1)) == {}


def _stats_module():
    path = ROOT / "toolkits/realworld_check/rlt_episode_stats.py"
    spec = importlib.util.spec_from_file_location("rlt_episode_stats", path)
    module = importlib.util.module_from_spec(spec)
    # Dataclass annotations resolve through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_episode_stats_classify_modes_and_rescues():
    stats = _stats_module()
    log = """
[INFO 10:00:00 EnvGroup-Rank-0][mux_rlt_wrapper.py:60] I: requested INFERENCE
[INFO 10:00:20 EnvGroup-Rank-0][mux_rlt_wrapper.py:60] C: episode success; collection ends in HOLD.
[INFO 10:01:00 EnvGroup-Rank-0][mux_rlt_wrapper.py:60] I: requested INFERENCE
[INFO 10:01:05 EnvGroup-Rank-0][mux_rlt_wrapper.py:60] B: collection requested
[INFO 10:01:06 RolloutGroup-Rank-0][route.py:1] [RLT exploration] Actor episode 3 env=0: EVAL mean actor, no exploration noise
[INFO 10:01:30 EnvGroup-Rank-0][mux_rlt_wrapper.py:60] X: episode failure; collection ends in HOLD.
[INFO 10:02:00 EnvGroup-Rank-0][mux_rlt_wrapper.py:60] I: requested INFERENCE
[INFO 10:02:05 EnvGroup-Rank-0][mux_rlt_wrapper.py:60] B: collection requested
[INFO 10:02:06 RolloutGroup-Rank-0][route.py:1] [RLT exploration] Actor episode 4 env=0: EXPLORE arm_std=0.0030rad
[INFO 10:02:10 EnvGroup-Rank-0][mux_rlt_wrapper.py:60] Space: software HOLD
[INFO 10:02:12 EnvGroup-Rank-0][mux_rlt_wrapper.py:60] I: requested INFERENCE
[INFO 10:02:20 EnvGroup-Rank-0][mux_rlt_wrapper.py:60] T: waiting for original mux HOLD ACK
[INFO 10:02:40 EnvGroup-Rank-0][mux_rlt_wrapper.py:60] C: episode success; collection ends in HOLD.
""".splitlines()
    episodes = stats.parse_episodes(log)
    assert [e.mode for e in episodes] == ["JAX", "ACTOR_EVAL", "ACTOR_EXPLORE"]
    assert [e.outcome for e in episodes] == ["success", "failure", "success"]
    assert [e.takeover for e in episodes] == [False, False, True]
    assert episodes[2].duration_s == 35.0
    rows = stats.summarize(episodes, timeout_s=30.0)
    assert any(row.startswith("JAX ") and " 1 " in row for row in rows)


def test_correlated_exploration_drifts_instead_of_swinging_between_chunks():
    def chunk_offsets(correlation):
        route = _route(
            exploration_eval_every_n_episodes=0,
            max_actor_reference_arm_delta=None,
            max_actor_reference_gripper_delta=None,
            exploration_chunk_correlation=correlation,
        )
        torch.manual_seed(0)
        return torch.stack([_step(route)[0, :, 0].mean() for _ in range(4000)])

    for correlation in (0.0, 0.9):
        offsets = chunk_offsets(correlation)
        lag1 = torch.corrcoef(torch.stack((offsets[1:], offsets[:-1])))[0, 1]
        assert lag1.item() == pytest.approx(correlation, abs=0.05)
    iid, ar = chunk_offsets(0.0), chunk_offsets(0.9)
    swing = lambda x: (x[1:] - x[:-1]).abs().mean()  # noqa: E731
    assert swing(ar) < 0.5 * swing(iid)
    # The marginal size of the exploration is unchanged.
    assert ar.std().item() == pytest.approx(iid.std().item(), rel=0.15)


def test_new_episode_reseeds_the_exploration_draw():
    route = _route(
        exploration_eval_every_n_episodes=0, exploration_chunk_correlation=0.99
    )
    first = _step(route)
    _step(route, critical=False)
    second = _step(route)
    assert not torch.allclose(first, second)
    with pytest.raises(ValueError, match="exploration_chunk_correlation"):
        _route(exploration_chunk_correlation=1.0)
