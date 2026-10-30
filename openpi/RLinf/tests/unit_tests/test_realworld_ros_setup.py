# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import Mock

from rlinf.envs.realworld.realworld_env import RealWorldEnv


def test_realworld_import_setup_preserves_ros_processes_by_default(
    monkeypatch,
):
    monkeypatch.delenv("RLINF_RESTART_ROS_MASTER", raising=False)
    monkeypatch.delenv("RLINF_PRESERVE_ROS_MASTER", raising=False)
    process = Mock()
    process.name.return_value = "rosmaster"
    process_iter = Mock(return_value=[process])
    monkeypatch.setattr(
        "rlinf.envs.realworld.realworld_env.psutil.process_iter", process_iter
    )

    RealWorldEnv.realworld_setup()

    process_iter.assert_not_called()
    process.kill.assert_not_called()


def test_realworld_setup_restarts_ros_only_when_explicitly_requested(
    monkeypatch,
):
    monkeypatch.setenv("RLINF_RESTART_ROS_MASTER", "1")
    monkeypatch.delenv("RLINF_PRESERVE_ROS_MASTER", raising=False)
    process = Mock()
    process.name.return_value = "rosmaster"
    monkeypatch.setattr(
        "rlinf.envs.realworld.realworld_env.psutil.process_iter",
        lambda: [process],
    )
    monkeypatch.setattr(
        "rlinf.envs.realworld.realworld_env.time.sleep", lambda _seconds: None
    )

    RealWorldEnv.realworld_setup()

    process.kill.assert_called_once_with()


def test_preserve_flag_overrides_explicit_restart(
    monkeypatch,
):
    monkeypatch.setenv("RLINF_RESTART_ROS_MASTER", "1")
    monkeypatch.setenv("RLINF_PRESERVE_ROS_MASTER", "1")
    process = Mock()
    process.name.return_value = "rosmaster"
    process_iter = Mock(return_value=[process])
    monkeypatch.setattr(
        "rlinf.envs.realworld.realworld_env.psutil.process_iter", process_iter
    )

    RealWorldEnv.realworld_setup()

    process_iter.assert_not_called()
    process.kill.assert_not_called()
