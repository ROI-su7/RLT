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

"""Full-resolution wrist frames are logged per RLT step and join to replay rows."""

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf
from test_piper_direct_control import _controller, _ImageMessage

from rlinf.data.rlt_image_log import RLTImageLog, load_frames, match_frame
from rlinf.envs.realworld.piper.piper_env import PiperRobotConfig
from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.envs.realworld.venv import NoAutoResetSyncVectorEnv
from rlinf.envs.utils import get_env_attr


def _frame(seed: int) -> np.ndarray:
    """A smooth image (JPEG-friendly) that differs per seed."""
    y, x = np.mgrid[0:480, 0:640]
    rgb = [(x + 37 * seed) % 256, (y + 11 * seed) % 256, (x + y + seed) % 256]
    return np.stack(rgb, axis=-1).astype(np.uint8)


def test_steps_are_written_per_episode_and_found_by_exact_state(tmp_path):
    log = RLTImageLog(str(tmp_path), jpeg_quality=95)
    states = [
        np.random.default_rng(i).normal(size=14).astype(np.float32) for i in range(3)
    ]
    for step, state in enumerate(states):
        log.submit(7, state, {"cam_right_wrist": _frame(step)})
    log.submit(8, states[0] + 1, {"cam_right_wrist": _frame(9)})
    log.close()

    rows = load_frames(str(tmp_path / "episode_7"))
    assert [row["frame"] for row in rows] == [0, 1, 2]
    found = match_frame(rows, states[1])
    assert found["frame"] == 1 and found["cameras"] == ["cam_right_wrist"]
    # A state that differs in the last float32 bit is a different step.
    nudged = states[1].copy()
    nudged[0] = np.nextafter(nudged[0], np.float32(np.inf))
    assert match_frame(rows, nudged) is None
    image = cv2.imread(str(tmp_path / "episode_7" / "000001_cam_right_wrist.jpg"))
    assert image.shape == (480, 640, 3)
    assert np.abs(image.astype(int) - _frame(1).astype(int)).mean() < 4
    assert len(load_frames(str(tmp_path / "episode_8"))) == 1


def test_non_contiguous_ros_view_is_encoded(tmp_path):
    log = RLTImageLog(str(tmp_path))
    padded = np.zeros((480, 700, 3), np.uint8)
    log.submit(1, np.zeros(14, np.float32), {"cam_left_wrist": padded[:, :640]})
    log.close()
    image = cv2.imread(str(tmp_path / "episode_1" / "000000_cam_left_wrist.jpg"))
    assert image.shape == (480, 640, 3)


def test_full_queue_drops_steps_instead_of_blocking(tmp_path, monkeypatch):
    release = threading.Event()
    original = RLTImageLog._write

    def slow_write(self, *args):
        release.wait(5)
        original(self, *args)

    monkeypatch.setattr(RLTImageLog, "_write", slow_write)
    log = RLTImageLog(str(tmp_path), max_queued_steps=1)
    for _ in range(5):
        log.submit(1, np.zeros(14, np.float32), {"cam_left_wrist": _frame(0)})
    assert log.dropped_steps >= 3
    release.set()
    log.close()


def _observing_env(image_log):
    env = _controller(PiperRobotConfig(render_height=224, render_width=224))
    env._wait_ready = Mock()
    env._rospy = SimpleNamespace(is_shutdown=lambda: False)
    frames = [_frame(i) for i in range(3)]
    left = SimpleNamespace(position=[0.1] * 7)
    right = SimpleNamespace(position=[0.2] * 7)
    env._get_synced_messages = Mock(
        return_value=(
            *(_ImageMessage(frame, "bgr8") for frame in frames),
            left,
            right,
        )
    )
    env._image_log = image_log
    env._image_log_episode = None
    return env, frames


def test_only_steps_inside_an_episode_chunk_are_logged():
    image_log = Mock()
    env, frames = _observing_env(image_log)

    observation = env._get_observation()
    image_log.submit.assert_not_called()
    assert observation["frames"]["cam_right_wrist"].shape == (224, 224, 3)

    env.set_image_log_episode(42)
    observation = env._get_observation()
    episode, state, logged = image_log.submit.call_args.args
    assert episode == 42
    np.testing.assert_array_equal(state, observation["state"]["joint_position"])
    assert sorted(logged) == ["cam_left_wrist", "cam_right_wrist"]
    np.testing.assert_array_equal(logged["cam_left_wrist"], frames[1])
    np.testing.assert_array_equal(logged["cam_right_wrist"], frames[2])

    env.set_image_log_episode(None)
    env._get_observation()
    assert image_log.submit.call_count == 1


def test_piper_stage2_config_logs_wrist_frames():
    cfg = OmegaConf.load(
        Path(__file__).parents[2]
        / "examples/embodiment/config/realworld_piper_rlt_stage2_ac_mlp.yaml"
    )
    override = cfg.env.train.override_cfg
    assert OmegaConf.to_container(override)["image_log_dir"].endswith("/rlt_images")
    config = PiperRobotConfig()
    assert config.image_log_dir is None
    assert config.image_log_cameras == ["cam_left_wrist", "cam_right_wrist"]


class _LoggingEnv(gym.Env):
    action_space = gym.spaces.Box(-1, 1, (14,), dtype=np.float32)
    observation_space = gym.spaces.Box(-1, 1, (14,), dtype=np.float32)

    def __init__(self):
        self.episodes = []

    def set_image_log_episode(self, episode_id):
        self.episodes.append(episode_id)


class _PlainEnv(gym.Env):
    action_space = gym.spaces.Box(-1, 1, (14,), dtype=np.float32)
    observation_space = gym.spaces.Box(-1, 1, (14,), dtype=np.float32)


def _vector_env(inner):
    env = RealWorldEnv.__new__(RealWorldEnv)
    # The deployed stack: RealWorldEnv -> vector env -> mux wrapper -> Piper env.
    env.env = NoAutoResetSyncVectorEnv([lambda: gym.Wrapper(inner)])
    return env


def test_env_worker_reaches_the_piper_env_through_the_vector_env():
    inner = _LoggingEnv()
    env = _vector_env(inner)
    hook = get_env_attr(env, "set_image_log_episode", None)
    assert callable(hook)
    hook(42)
    hook(None)
    assert inner.episodes == [42, None]


def test_envs_without_an_image_log_ignore_the_episode():
    env = _vector_env(_PlainEnv())
    env.set_image_log_episode(7)  # no AttributeError
