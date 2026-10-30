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

import copy
import os
import pathlib
import time
from functools import partial
from typing import OrderedDict

import gymnasium as gym
import numpy as np
import psutil
import torch
from filelock import FileLock
from omegaconf import OmegaConf

from rlinf.envs.realworld.venv import NoAutoResetSyncVectorEnv
from rlinf.envs.utils import to_tensor
from rlinf.scheduler import WorkerInfo


class RealWorldEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        assert num_envs == 1, (
            f"Currently, only 1 realworld env can be started per worker, but {num_envs=} is received."
        )

        self.cfg = cfg
        self.override_cfg = OmegaConf.to_container(
            cfg.get("override_cfg", OmegaConf.create({})), resolve=True
        )

        self.video_cfg = cfg.video_cfg

        self.seed = cfg.seed + seed_offset
        self.num_envs = num_envs
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.auto_reset = cfg.auto_reset
        self.ignore_terminations = cfg.ignore_terminations
        self.num_group = num_envs // cfg.group_size
        self.group_size = cfg.group_size
        self.main_image_key = cfg.main_image_key
        self.manual_episode_control_only = bool(
            self.override_cfg.get("manual_episode_control_only", False)
        )

        self._init_env()

        self._is_start = True
        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self._init_reset_state_ids()

    def _create_env(self, env_idx: int):
        worker_info: WorkerInfo = self.worker_info
        hardware_info = None
        if worker_info is not None and env_idx < len(worker_info.hardware_infos):
            hardware_info = worker_info.hardware_infos[env_idx]
        override_cfg = copy.deepcopy(self.override_cfg)
        env = gym.make(
            id=self.cfg.init_params.id,
            override_cfg=override_cfg,
            worker_info=worker_info,
            hardware_info=hardware_info,
            env_idx=env_idx,
            env_cfg=self.cfg,
        )
        return env

    @staticmethod
    def realworld_setup():
        """Setup RealWorld environment upon env class import.

        Existing ROS hardware sessions are externally managed and must survive
        ordinary RLinf imports.  Legacy cleanup is available only as an explicit
        opt-in for isolated environments that own the whole ROS process set.
        """
        preserve_requested = os.environ.get(
            "RLINF_PRESERVE_ROS_MASTER", ""
        ).lower() in {
            "1",
            "true",
            "yes",
        }
        if preserve_requested:
            return
        restart_requested = os.environ.get("RLINF_RESTART_ROS_MASTER", "").lower() in {
            "1",
            "true",
            "yes",
        }
        if not restart_requested:
            return

        # Concurrency control is needed for multiple processes on the same node
        node_lock_file = "/tmp/.realworld.lock"
        # Check if the path is valid
        if not os.path.exists(os.path.dirname(node_lock_file)):
            node_lock_file = os.path.join(pathlib.Path.home(), ".realworld.lock")
        node_lock = FileLock(node_lock_file)

        with node_lock:
            ros_proc_names = ["roscore", "rosmaster", "rosout"]
            for proc in psutil.process_iter():
                if proc.name() in ros_proc_names:
                    proc.kill()
                    time.sleep(0.5)

    def _init_env(self):
        env_fns = [
            partial(self._create_env, env_idx=env_idx)
            for env_idx in range(self.num_envs)
        ]
        self.env = NoAutoResetSyncVectorEnv(env_fns)
        self.task_descriptions = list(
            self.env.call("get_wrapper_attr", "task_description")
        )

    def set_actor_execution_mode(self, actor_flags) -> None:
        """Select direct execution per Piper env without changing JAX settings."""
        methods = self.env.call("get_wrapper_attr", "set_actor_execution_mode")
        flags = np.asarray(actor_flags, dtype=bool).reshape(-1)
        if len(methods) != len(flags):
            raise ValueError("Actor flags must match the number of Piper envs.")
        for method, active in zip(methods, flags, strict=True):
            method(bool(active))

    def align_rtc_chunk(self, actions, reference_plan):
        """Forward Piper RTC alignment through the one-env vector wrapper."""
        methods = self.env.call("get_wrapper_attr", "align_rtc_chunk")
        aligned, offset = methods[0](
            np.asarray(actions)[0],
            np.asarray(reference_plan)[0],
        )
        return np.asarray(aligned)[None], int(offset)

    def start_rtc_tail(self, reference_tail):
        """Forward the unused VLA suffix to the Piper execution layer."""
        methods = self.env.call("get_wrapper_attr", "start_rtc_tail")
        methods[0](np.asarray(reference_tail)[0])

    def queue_rtc_tail(self, reference_tail):
        """Queue the Piper suffix before synchronous chunk execution."""
        methods = self.env.call("get_wrapper_attr", "queue_rtc_tail")
        methods[0](np.asarray(reference_tail)[0])

    def begin_action_chunk(self, num_actions: int) -> None:
        """Tell the physical Piper env that one action chunk is starting."""
        methods = self.env.call("get_wrapper_attr", "begin_action_chunk")
        for method in methods:
            method(int(num_actions))

    def set_image_log_episode(self, episode_id: int | None) -> None:
        """Log the camera frames of the next steps under this RLT episode.

        Only Piper envs keep an image log; for others this is a no-op.
        """
        try:
            methods = self.env.call("get_wrapper_attr", "set_image_log_episode")
        except AttributeError:
            return
        for method in methods:
            method(episode_id)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def total_num_group_envs(self):
        return np.iinfo(np.uint8).max // 2  # TODO

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    def _init_metrics(self):
        self.prev_step_reward = np.zeros(self.num_envs)

        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)
        self.intervened_once = np.zeros(self.num_envs, dtype=bool)
        self.intervened_steps = np.zeros(self.num_envs, dtype=int)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self._elapsed_steps[mask] = 0
            self.intervened_once[mask] = False
            self.intervened_steps[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0
            self.intervened_once[:] = False
            self.intervened_steps[:] = 0

    def _record_metrics(
        self,
        step_reward,
        terminations,
        success_current_step,
        intervene_current_step,
        infos,
    ):
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | success_current_step
        self.intervened_once = self.intervened_once | intervene_current_step
        self.intervened_steps += intervene_current_step.astype(int)

        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        episode_info["intervened_once"] = self.intervened_once
        episode_info["intervened_steps"] = self.intervened_steps
        episode_info["success_no_intervened"] = self.success_once.copy() & (
            ~self.intervened_once
        )
        infos["episode"] = to_tensor(episode_info)
        return infos

    def reset(self, *, reset_state_ids=None, seed=None, options=None, env_idx=None):
        # TODO: handle partial reset
        raw_obs, infos = self.env.reset(seed=seed, options=options)

        extracted_obs = self._wrap_obs(raw_obs)
        if env_idx is not None:
            self._reset_metrics(env_idx)
        else:
            self._reset_metrics()
        return extracted_obs, infos

    def _wrap_obs(self, raw_obs):
        """
        raw_obs: Dict of list
        """
        obs = {}

        state = raw_obs["state"]
        full_states = np.concatenate([state[k] for k in sorted(state)], axis=-1)
        obs["states"] = full_states

        frames = raw_obs["frames"]
        if self.main_image_key not in frames:
            raise KeyError(
                f"main_image_key {self.main_image_key!r} not in {list(frames)}"
            )
        obs["main_images"] = frames[self.main_image_key]
        raw_images = OrderedDict(sorted(frames.items()))
        raw_images.pop(self.main_image_key)

        if raw_images:
            obs["extra_view_images"] = np.stack(list(raw_images.values()), axis=1)

        if "head_pose" in raw_obs:
            # Piper connector-head keypoints (PiperRobotConfig.head_pose_weights).
            obs["head_pose"] = raw_obs["head_pose"]

        obs = to_tensor(obs)
        obs["task_descriptions"] = self.task_descriptions
        return obs

    def step(self, actions=None, auto_reset=True):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, truncations, infos = self.env.step(actions)
        # max_episode_steps: null → external wrapper owns episode end.
        if self.cfg.max_episode_steps is None:
            timeout_truncations = np.zeros_like(truncations, dtype=bool)
        else:
            timeout_truncations = self.elapsed_steps >= self.cfg.max_episode_steps
        if not self.manual_episode_control_only:
            truncations = timeout_truncations

        obs = self._wrap_obs(raw_obs)
        step_reward = self._calc_step_reward(_reward)
        success_current_step = np.isclose(step_reward, 1.0)
        intervene_flag = np.zeros(self.num_envs, dtype=bool)
        if "intervene_action" in infos:
            for env_id in range(self.num_envs):
                if infos["intervene_action"][env_id] is not None:
                    intervene_flag[env_id] = True

        infos = self._record_metrics(
            step_reward,
            terminations,
            success_current_step,
            intervene_flag,
            infos,
        )
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        intervene_action = np.zeros_like(actions)
        if "intervene_action" in infos:
            for env_id in range(self.num_envs):
                env_intervene_action = infos["intervene_action"][env_id]
                if env_intervene_action is not None:
                    intervene_action[env_id] = env_intervene_action.copy()
        infos["intervene_action"] = to_tensor(intervene_action)
        infos["intervene_flag"] = to_tensor(intervene_flag)
        if "rlt_switch_flags" in infos:
            infos["rlt_switch_flags"] = to_tensor(
                np.asarray(infos["rlt_switch_flags"], dtype=bool)
            )
        if "rlt_execution_valid" in infos:
            infos["rlt_execution_valid"] = to_tensor(
                np.asarray(infos["rlt_execution_valid"], dtype=bool)
            )
            # Gym vector infos can wrap ndarray-valued fields in object
            # arrays. to_tensor(object_array) returns a list of tensors,
            # which cannot be stacked directly by chunk_step.
            executed_actions = np.stack(
                [
                    np.asarray(action, dtype=np.float32)
                    for action in infos["rlt_executed_action"]
                ]
            )
            if executed_actions.shape != (self.num_envs, actions.shape[-1]):
                raise ValueError(
                    "Invalid batched rlt_executed_action shape: "
                    f"{executed_actions.shape}"
                )
            infos["rlt_executed_action"] = to_tensor(executed_actions)
            for field, width in (
                ("piper_accepted_target", 14),
                ("piper_execution_filter_params", 3),
            ):
                if field not in infos:
                    continue
                # Gym may wrap each ndarray in an object array. Normalize
                # per-environment rows before to_tensor; object arrays are
                # intentionally converted to lists by the generic utility.
                present = infos.get(f"_{field}", np.ones(self.num_envs, dtype=bool))
                rows = []
                for env_id, value in enumerate(infos[field]):
                    if not present[env_id] or value is None:
                        rows.append(np.full(width, np.nan, dtype=np.float32))
                    elif isinstance(value, torch.Tensor):
                        rows.append(value.detach().cpu().numpy().astype(np.float32))
                    else:
                        rows.append(np.asarray(value, dtype=np.float32))
                values = np.stack(rows)
                if values.shape != (self.num_envs, width):
                    raise ValueError(f"Invalid batched {field} shape: {values.shape}")
                infos[field] = to_tensor(values)

        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []

        raw_chunk_intervene_actions = []
        raw_chunk_intervene_flag = []
        raw_chunk_rlt_switch_flags = []
        raw_chunk_execution_valid = []
        raw_chunk_keyboard_terminal = []
        raw_chunk_accepted_targets = []
        raw_chunk_filter_params = []
        raw_chunk_executed_actions = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)
            if "intervene_action" in infos:
                raw_chunk_intervene_actions.append(infos["intervene_action"])
                raw_chunk_intervene_flag.append(infos["intervene_flag"])
            if "rlt_switch_flags" in infos:
                raw_chunk_rlt_switch_flags.append(infos["rlt_switch_flags"])
            if "rlt_execution_valid" in infos:
                raw_chunk_execution_valid.append(
                    to_tensor(infos["rlt_execution_valid"])
                )
                raw_chunk_executed_actions.append(
                    to_tensor(infos["rlt_executed_action"])
                )
            if "rlt_keyboard_terminal" in infos:
                raw_chunk_keyboard_terminal.append(
                    to_tensor(infos["rlt_keyboard_terminal"])
                )
            if "rlt_execution_valid" in infos:
                raw_chunk_accepted_targets.append(
                    to_tensor(
                        infos.get(
                            "piper_accepted_target",
                            np.full((self.num_envs, 14), np.nan, dtype=np.float32),
                        )
                    )
                )
                raw_chunk_filter_params.append(
                    to_tensor(
                        infos.get(
                            "piper_execution_filter_params",
                            np.full((self.num_envs, 3), np.nan, dtype=np.float32),
                        )
                    )
                )

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        infos_last = infos_list[-1] if infos_list else {}
        if raw_chunk_accepted_targets:
            infos_last["piper_accepted_target"] = torch.stack(
                raw_chunk_accepted_targets, dim=1
            )
            infos_last["piper_execution_filter_params"] = torch.stack(
                raw_chunk_filter_params, dim=1
            )
        if raw_chunk_keyboard_terminal:
            infos_last["rlt_keyboard_terminal"] = torch.stack(
                raw_chunk_keyboard_terminal, dim=1
            )
        if raw_chunk_execution_valid:
            infos_last["rlt_execution_valid"] = torch.stack(
                raw_chunk_execution_valid, dim=1
            )
            infos_last["rlt_executed_action"] = torch.stack(
                raw_chunk_executed_actions, dim=1
            )
        if raw_chunk_intervene_actions:
            infos_last["intervene_action"] = torch.stack(
                raw_chunk_intervene_actions, dim=1
            ).reshape(self.num_envs, -1)
            infos_last["intervene_flag"] = torch.stack(raw_chunk_intervene_flag, dim=1)
            infos_list[-1] = infos_last
        if raw_chunk_rlt_switch_flags:
            infos_last["rlt_switch_flags"] = torch.stack(
                raw_chunk_rlt_switch_flags, dim=1
            )
            infos_list[-1] = infos_last
        # Per-step terminal flags survive the auto-reset collapse below, so RLT
        # stride windows can place a mid-chunk C/X/Q on its control step.
        infos_last["chunk_step_dones"] = torch.logical_or(
            raw_chunk_terminations, raw_chunk_truncations
        )
        infos_list[-1] = infos_last

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones.cpu().numpy(), obs_list[-1], infos_list[-1]
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        obs, infos = self.reset(
            env_idx=env_idx,
            reset_state_ids=(
                self.reset_state_ids[env_idx]
                if self.use_fixed_reset_state_ids
                else None
            ),
        )
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(self, reward: np.ndarray):
        return reward.astype(np.float32)

    def _get_random_reset_state_ids(self, num_reset_states):
        reset_state_ids = self._generator.integers(
            low=0, high=self.total_num_group_envs, size=(num_reset_states,)
        )
        return reset_state_ids

    def _init_reset_state_ids(self):
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self):
        reset_state_ids = torch.randint(
            low=0,
            high=self.total_num_group_envs,
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        )
