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

"""Connector-head state: detector output, network input, stride windows, shaping."""

from unittest.mock import Mock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from test_rlt_image_log import _observing_env

from rlinf.algorithms.rlt.head_pose_shaping import HeadPoseShaping
from rlinf.algorithms.rlt.subsample import (
    StrideWindowAssembler,
    iter_stride_windows,
    pack_stride_windows,
)
from rlinf.algorithms.rlt.transition import extract_rlt_obs_from_forward_inputs
from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.envs.realworld.piper.piper_env import PiperRobotConfig, PiperWireEnv
from rlinf.envs.realworld.piper.wire_head_pose import HEAD_POSE_DIM, head_pose_state
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy
from rlinf.models.embodiment.native_jax_rlt import _with_head_pose
from toolkits.wire_head_pose.pad_stage2_checkpoint import (
    pad_model_state,
    pad_optimizer_state,
    state_input_dim,
)

Z, P, D, C, H, LATENCY = 16, 14, 14, 10, 50, 5
SCALE = [20.0, 50.0, 0.3, 1.0]


def _prediction(own_tip, other_tip, visible=0.9):
    return {
        "own_tip": {"xy": list(own_tip), "p_visible": visible},
        "own_back": {"xy": [own_tip[0], own_tip[1] + 50.0], "p_visible": 0.9},
        "other_tip": {"xy": list(other_tip), "p_visible": visible},
        "other_back": {"xy": [other_tip[0], other_tip[1] - 30.0], "p_visible": 0.9},
    }


def test_state_is_tip_offset_axis_angle_and_visibility() -> None:
    state = head_pose_state(_prediction((300.0, 250.0), (307.0, 246.0)))
    np.testing.assert_allclose(state, [7.0, -4.0, 0.0, 1.0], atol=1e-6)
    hidden = head_pose_state(_prediction((300.0, 250.0), (307.0, 246.0), 0.2))
    np.testing.assert_array_equal(hidden, np.zeros(HEAD_POSE_DIM))


def _policy(head_pose_dim: int) -> RLTMLPPolicy:
    torch.manual_seed(0)
    return RLTMLPPolicy(
        z_dim=Z,
        proprio_dim=P,
        action_dim=D,
        num_action_chunks=C,
        q_head_type="rlt_projected",
        actor_architecture="projected",
        actor_output="full",
        bridge_delta_dim=D,
        head_pose_dim=head_pose_dim,
        head_pose_scale=SCALE if head_pose_dim else None,
    )


def _obs(batch: int = 3) -> dict:
    return {
        "z_rl": torch.randn(batch, Z),
        "proprio": torch.randn(batch, P),
        "ref_chunk": torch.randn(batch, C, D),
        "bridge_delta": torch.randn(batch, D) * 0.02,
    }


def test_head_pose_joins_the_state_scaled_and_clipped() -> None:
    policy = _policy(HEAD_POSE_DIM)
    obs = {**_obs(1), "head_pose": torch.tensor([[10.0, -500.0, 0.15, 1.0]])}
    state = policy._get_state(obs)
    assert state.shape == (1, P + D + HEAD_POSE_DIM)
    torch.testing.assert_close(state[0, -4:], torch.tensor([0.5, -5.0, 0.5, 1.0]))
    missing = policy._get_state(_obs(1))
    torch.testing.assert_close(missing[0, -4:], torch.zeros(4))


def test_zero_padded_checkpoint_reproduces_the_old_networks() -> None:
    old, new = _policy(0), _policy(HEAD_POSE_DIM)
    state = old.state_dict()
    old_in = state_input_dim(state)
    assert old_in == P + D
    assert pad_model_state(state, old_in, HEAD_POSE_DIM) == 3
    new.load_state_dict(state)
    obs = _obs()
    seen = {**obs, "head_pose": torch.tensor([[12.0, -5.0, 0.1, 1.0]] * 3)}
    actions = torch.randn(3, C * D)
    with torch.no_grad():
        torch.testing.assert_close(
            new.actor_net(
                new._get_z(seen), new._get_state(seen), new._get_ref_chunk(seen)
            ),
            old.actor_net(
                old._get_z(obs), old._get_state(obs), old._get_ref_chunk(obs)
            ),
        )
        torch.testing.assert_close(
            new(forward_type=ForwardType.SAC_Q, obs=seen, actions=actions),
            old(forward_type=ForwardType.SAC_Q, obs=obs, actions=actions),
        )


def test_adam_moments_are_padded_like_the_weights() -> None:
    optim = {
        "state": {
            "actor_net.inputs.proprio_proj.linear.weight": {
                "exp_avg": torch.ones(64, 28),
                "exp_avg_sq": torch.ones(64, 28),
                "step": torch.tensor(5.0),
            },
            "actor_net.inputs.z_proj.linear.weight": {"exp_avg": torch.ones(2, 2)},
        }
    }
    assert pad_optimizer_state(optim, 28, 4) == 2
    moments = optim["state"]["actor_net.inputs.proprio_proj.linear.weight"]
    assert moments["exp_avg"].shape == (64, 32)
    assert moments["exp_avg_sq"][:, 28:].abs().sum() == 0
    assert optim["state"]["actor_net.inputs.z_proj.linear.weight"]["exp_avg"].shape == (
        2,
        2,
    )


def test_head_pose_flows_from_env_obs_into_transitions() -> None:
    obs = {
        "states": torch.zeros(1, D),
        "main_images": torch.zeros(1, 4, 4, 3),
        "head_pose": torch.tensor([[3.0, -4.0, 0.1, 1.0]]),
    }
    prepared = EnvOutput(obs=obs).prepare_observations(obs)
    torch.testing.assert_close(prepared["head_pose"], obs["head_pose"])
    rlt_obs = _with_head_pose({"proprio": torch.zeros(1, D)}, prepared)
    torch.testing.assert_close(rlt_obs["head_pose"], obs["head_pose"])
    assert "head_pose" not in _with_head_pose({"proprio": torch.zeros(1, D)}, {})
    inputs = {
        "z_rl": torch.zeros(1, Z),
        "proprio": torch.zeros(1, P),
        "ref_chunk": torch.zeros(1, H, D),
        "head_pose": rlt_obs["head_pose"],
    }
    torch.testing.assert_close(
        extract_rlt_obs_from_forward_inputs(inputs)["head_pose"], obs["head_pose"]
    )


def _steps(first: int, count: int) -> torch.Tensor:
    return torch.arange(first, first + count, dtype=torch.float32)[:, None].repeat(1, D)


def _chunk(assembler, start, bridge, *, record=True):
    return assembler.add_chunk(
        bridge=_steps(start, bridge),
        actions=_steps(start + bridge, C),
        plan=_steps(start, H),
        z_rl=torch.full((Z,), float(start)),
        proprio=torch.full((P,), float(start)),
        record=record,
        head_pose=torch.full((HEAD_POSE_DIM,), float(start)),
    )


def _finish(assembler, chunk_id, action_start):
    assembler.add_outcome(chunk_id, rewards=torch.zeros(C))
    times = torch.tensor(
        [float(action_start + offset) for offset in assembler.offsets]
    )[:, None]
    assembler.add_stride_features(
        chunk_id,
        times.repeat(1, Z),
        times.repeat(1, P),
        times.repeat(1, HEAD_POSE_DIM),
    )


def test_stride_windows_carry_the_head_pose_of_both_observations() -> None:
    assembler = StrideWindowAssembler(chunk_len=C, stride=2, latency=LATENCY, horizon=H)
    first = _chunk(assembler, 0, 0, record=False)
    _finish(assembler, first, 0)
    second = _chunk(assembler, 10, LATENCY)
    _finish(assembler, second, 15)
    third = _chunk(assembler, 25, LATENCY)
    _finish(assembler, third, 30)
    _chunk(assembler, 40, LATENCY)
    windows = assembler.pop_ready()
    assert len(windows) == 4
    for window in windows:
        # Head pose comes from the same observation as z_rl.
        torch.testing.assert_close(window["curr_head_pose"][0], window["curr_z_rl"][0])
        torch.testing.assert_close(window["next_head_pose"][0], window["next_z_rl"][0])

    packed = pack_stride_windows(
        [windows[:2]],
        slots=3,
        chunk_len=C,
        action_dim=D,
        horizon=H,
        z_dim=Z,
        proprio_dim=P,
        head_pose_dim=HEAD_POSE_DIM,
    )
    flat = {key: value.reshape(1, *value.shape[1:]) for key, value in packed.items()}
    unpacked = list(iter_stride_windows(flat, 0))
    assert len(unpacked) == 2
    torch.testing.assert_close(
        unpacked[1]["next_head_pose"], windows[1]["next_head_pose"]
    )
    without = pack_stride_windows(
        [windows[:1]],
        slots=3,
        chunk_len=C,
        action_dim=D,
        horizon=H,
        z_dim=Z,
        proprio_dim=P,
    )
    assert "rlt_stride_curr_head_pose" not in without
    assert "curr_head_pose" not in next(iter_stride_windows(without, 0))


def test_shaping_rewards_reducing_the_offset_near_contact_only() -> None:
    shaping = HeadPoseShaping.from_config(
        OmegaConf.create({"enable": True, "weight": 0.05, "scale_px": 10.0})
    )
    discount = torch.full((4, 1), 0.875)
    curr = torch.tensor(
        [
            [10.0, -5.0, 0.0, 1.0],  # at contact, 10 px off
            [10.0, -200.0, 0.0, 1.0],  # far: gate 0
            [10.0, -5.0, 0.0, 0.0],  # not visible
            [-40.0, -37.5, 0.0, 1.0],  # half gate, clipped at 30 px
        ]
    )
    nxt = torch.tensor(
        [
            [0.0, -4.0, 0.0, 1.0],
            [0.0, -200.0, 0.0, 1.0],
            [0.0, -4.0, 0.0, 1.0],
            [-40.0, -37.5, 0.0, 1.0],
        ]
    )
    term, metrics = shaping(curr, nxt, discount)
    expected = torch.tensor([[0.05], [0.0], [0.0], [0.075 * (1 - 0.875)]])
    torch.testing.assert_close(term, expected)
    assert metrics["head_pose_shaped_fraction"] == pytest.approx(0.75)
    assert shaping.max_abs_potential == pytest.approx(0.15)
    assert HeadPoseShaping.from_config(OmegaConf.create({"enable": False})) is None
    assert HeadPoseShaping.from_config(None) is None


def test_piper_env_reports_the_head_pose_of_the_raw_right_wrist_frame() -> None:
    env, frames = _observing_env(None)
    detector = Mock(return_value=np.array([1.0, -2.0, 0.1, 1.0], np.float32))
    detector.last_ms = 3.0
    env._head_pose = detector
    env._observation_head_pose_ms_sum = 0.0
    observation = env._get_observation()
    np.testing.assert_array_equal(detector.call_args.args[0], frames[2])
    np.testing.assert_allclose(observation["head_pose"], [1.0, -2.0, 0.1, 1.0])
    # Steps without a camera decode reuse the frames and their head pose.
    env._last_observation = observation
    reused = env._reuse_last_frames(np.zeros(14, np.float32))
    assert reused["head_pose"] is observation["head_pose"]
    env._head_pose = None
    assert "head_pose" not in env._get_observation()


def test_observation_space_declares_the_head_pose_only_when_enabled() -> None:
    env = PiperWireEnv.__new__(PiperWireEnv)
    env.config = PiperRobotConfig(is_dummy=True)
    env._init_spaces()
    assert "head_pose" not in env.observation_space.spaces
    env.config = PiperRobotConfig(is_dummy=True, head_pose_weights="net.pt")
    env._init_spaces()
    assert env.observation_space["head_pose"].shape == (HEAD_POSE_DIM,)
    assert env.observation_space.sample()["head_pose"].shape == (HEAD_POSE_DIM,)
