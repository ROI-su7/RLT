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

"""RLT "Subsampling Action Chunks".

The paper stores <x_0, a_0:C>, <x_2, a_2:C+2>, <x_4, a_4:C+4>, ... rather than
one transition per action chunk, which is what makes the replay buffer dense
enough for the low-data online regime.

On Piper every chunk is preceded by an RTC bridge: while the next chunk is
inferred, the robot keeps executing the unused suffix of the previous VLA
plan. Windows are therefore cut from the executed timeline, bridge commands
included, instead of by concatenating two chunks. The index arithmetic is the
part that silently corrupts a replay buffer when it is wrong, so it is kept
here, apart from the worker plumbing, and unit tested on its own.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

RLT_STRIDE_PREFIX = "rlt_stride_"
STRIDE_SLOT_KEYS = (
    "valid",
    "actions",
    "rewards",
    "dones",
    "intervene_flags",
    "curr_z_rl",
    "curr_proprio",
    "curr_ref_chunk",
    "curr_bridge_delta",
    "next_z_rl",
    "next_proprio",
    "next_ref_chunk",
    "next_bridge_delta",
    "actor_switch",
    "reference_warmup",
    "episode_id",
    # Only when the env reports the Piper connector-head state.
    "curr_head_pose",
    "next_head_pose",
)
_WAIT = object()


def stride_offsets(chunk_len: int, stride: int) -> list[int]:
    """Offsets inside a chunk whose observation starts an extra transition.

    Offset 0 is the chunk's own transition, built by the chunk-level path, so
    it is not listed. ``stride <= 0`` disables subsampling.
    """
    chunk_len = int(chunk_len)
    stride = int(stride)
    if chunk_len <= 0:
        raise ValueError(f"chunk_len must be positive, got {chunk_len}.")
    if stride <= 0:
        return []
    if stride > chunk_len:
        raise ValueError(
            f"stride must be <= chunk_len, got stride={stride} chunk_len={chunk_len}."
        )
    return list(range(stride, chunk_len, stride))


@dataclass
class _Chunk:
    chunk_id: int
    segment: int
    start: int
    bridge: torch.Tensor
    actions: torch.Tensor
    plan: torch.Tensor
    z_rl: torch.Tensor
    proprio: torch.Tensor
    record: bool
    actor_switch: bool
    reference_warmup: bool
    episode_id: int
    rewards: torch.Tensor | None = None
    human: torch.Tensor | None = None
    valid_steps: torch.Tensor | None = None
    terminal_step: int | None = None
    stride_z_rl: torch.Tensor | None = None
    stride_proprio: torch.Tensor | None = None
    features_missing: bool = False
    head_pose: torch.Tensor | None = None
    stride_head_pose: torch.Tensor | None = None

    @property
    def action_start(self) -> int:
        return self.start + self.bridge.shape[0]

    @property
    def end(self) -> int:
        return self.action_start + self.actions.shape[0]


def _cpu_float(value: Any) -> torch.Tensor:
    return torch.as_tensor(value).detach().to(device="cpu", dtype=torch.float32)


class StrideWindowAssembler:
    """Build <x_t, a_{t+L : t+L+C}> rows for the intermediate observations of one env.

    Time counts executed control steps. Chunk k is chosen from the observation
    at ``start``; RTC then runs ``e`` bridge commands before the chunk's ``C``
    actions, and the observation decoded after chunk action ``j - 1`` is at
    ``start + e + j``.

    Every row uses the observation-to-action latency ``L`` of the deployed
    policy (the RTC bridge length): its actions are the ``C`` commands executed
    from ``t + L``, and its next observation is the one decoded at
    ``t + L + C``. Its reference is the last plan chosen at or before ``t``,
    re-anchored at ``t + L``, so no later information reaches the actor input.
    Its bridge delta is the command executed just before ``t + L`` minus the
    proprio at ``t``: where the RTC bridge leaves the arm when the actions
    start. That command comes from plans chosen at or before ``t``.
    A window that reaches a terminal step is padded with its last action, zero
    reward and ``done``; a window whose next observation was never decoded, or
    that starts after the terminal step, is dropped.
    """

    def __init__(
        self,
        *,
        chunk_len: int,
        stride: int,
        latency: int,
        horizon: int,
        max_pending_chunks: int = 4,
        replace_human_reference: bool = True,
    ):
        self.chunk_len = int(chunk_len)
        # False keeps the VLA plan on operator steps (see update_rlt_transitions).
        self.replace_human_reference = bool(replace_human_reference)
        self.offsets = stride_offsets(self.chunk_len, stride)
        self.latency = int(latency)
        self.horizon = int(horizon)
        if self.latency < 0:
            raise ValueError(f"latency must be non-negative, got {self.latency}.")
        if self.horizon < self.chunk_len:
            raise ValueError(
                f"horizon {self.horizon} must be >= chunk_len {self.chunk_len}."
            )
        self.max_pending_chunks = int(max_pending_chunks)
        self.dropped_windows = 0
        self._chunks: dict[int, _Chunk] = {}
        self._pending: deque[tuple[int, int]] = deque()
        self._next_chunk_id = 0
        self._segment = 0
        self._next_start = 0
        self._segment_closed = False

    def reset(self) -> None:
        """Forget all chunks, e.g. after a reset that has no terminal outcome."""
        self._chunks.clear()
        self._pending.clear()
        self._start_segment()

    def _start_segment(self) -> None:
        self._segment += 1
        self._next_start = 0
        self._segment_closed = False

    def add_chunk(
        self,
        *,
        bridge: Any,
        actions: Any,
        plan: Any,
        z_rl: Any,
        proprio: Any,
        record: bool,
        actor_switch: bool = False,
        reference_warmup: bool = False,
        episode_id: int = 0,
        head_pose: Any = None,
    ) -> int:
        """Register an executed chunk; ``bridge`` holds the RTC commands before it.

        ``head_pose`` is the connector-head state of the chunk's observation,
        when the env provides one.
        """
        actions = _cpu_float(actions)
        if actions.dim() != 2 or actions.shape[0] != self.chunk_len:
            raise ValueError(
                f"Expected [{self.chunk_len}, D] executed actions, got "
                f"{tuple(actions.shape)}."
            )
        action_dim = actions.shape[1]
        bridge = _cpu_float(bridge).reshape(-1, action_dim)
        plan = _cpu_float(plan)
        if plan.shape != (self.horizon, action_dim):
            raise ValueError(
                f"Expected a [{self.horizon}, {action_dim}] plan, got {tuple(plan.shape)}."
            )
        if self._segment_closed:
            self._start_segment()
        chunk = _Chunk(
            chunk_id=self._next_chunk_id,
            segment=self._segment,
            start=self._next_start,
            bridge=bridge,
            actions=actions.clone(),
            plan=plan,
            z_rl=_cpu_float(z_rl).reshape(-1),
            proprio=_cpu_float(proprio).reshape(-1),
            record=bool(record),
            actor_switch=bool(actor_switch),
            reference_warmup=bool(reference_warmup),
            episode_id=int(episode_id),
            head_pose=None if head_pose is None else _cpu_float(head_pose).reshape(-1),
        )
        self._chunks[chunk.chunk_id] = chunk
        self._next_chunk_id += 1
        self._next_start = chunk.end
        if chunk.record:
            self._pending.extend(
                (chunk.chunk_id, row) for row in range(len(self.offsets))
            )
        return chunk.chunk_id

    def add_outcome(
        self,
        chunk_id: int,
        *,
        rewards: Any,
        terminal_step: int | None = None,
        human_actions: Any = None,
        human_flags: Any = None,
        valid_steps: Any = None,
        executed_actions: Any = None,
    ) -> None:
        """Attach per-step rewards, the terminal step and operator actions."""
        chunk = self._chunks[chunk_id]
        chunk.valid_steps = (
            torch.ones(self.chunk_len, dtype=torch.bool)
            if valid_steps is None
            else torch.as_tensor(valid_steps)
            .detach()
            .cpu()
            .reshape(self.chunk_len)
            .bool()
        )
        if executed_actions is not None:
            chunk.actions = _cpu_float(executed_actions).reshape_as(chunk.actions)
        rewards = _cpu_float(rewards).reshape(-1)
        if rewards.numel() != self.chunk_len:
            raise ValueError(
                f"Expected {self.chunk_len} rewards, got {rewards.numel()}."
            )
        human = torch.zeros(self.chunk_len, dtype=torch.bool)
        if human_flags is not None:
            human = (
                torch.as_tensor(human_flags).detach().cpu().reshape(-1).to(torch.bool)
            )
            if human.numel() != self.chunk_len:
                raise ValueError(
                    f"Expected {self.chunk_len} intervention flags, got {human.numel()}."
                )
            if bool(human.any()):
                if human_actions is None:
                    raise ValueError("Intervention flags need the operator actions.")
                operator = _cpu_float(human_actions).reshape(self.chunk_len, -1)
                chunk.actions = torch.where(human[:, None], operator, chunk.actions)
        if terminal_step is not None:
            terminal_step = int(terminal_step)
            if not 0 <= terminal_step < self.chunk_len:
                raise ValueError(
                    f"terminal_step must be in [0, {self.chunk_len}), got {terminal_step}."
                )
            if chunk.segment == self._segment:
                self._segment_closed = True
        chunk.rewards = rewards
        chunk.human = human
        chunk.terminal_step = terminal_step

    def needs_stride_features(self, chunk_id: int) -> bool:
        """Only origins of recorded windows and their next observations need z_rl."""
        chunk = self._chunks.get(chunk_id)
        if chunk is None or not self.offsets:
            return False
        previous = self._chunks.get(chunk_id - 1)
        return chunk.record or (
            previous is not None
            and previous.segment == chunk.segment
            and previous.record
        )

    def add_stride_features(
        self,
        chunk_id: int,
        z_rl: torch.Tensor | None,
        proprio: torch.Tensor | None,
        head_pose: torch.Tensor | None = None,
    ) -> None:
        """Attach [W, Z] and [W, P] features; ``None`` marks them unavailable.

        ``head_pose`` ([W, K]) is optional; windows carry it only when both
        their observations have one.
        """
        chunk = self._chunks.get(chunk_id)
        if chunk is None:
            return
        if z_rl is None or proprio is None:
            chunk.features_missing = True
            return
        z_rl = _cpu_float(z_rl)
        proprio = _cpu_float(proprio)
        if z_rl.shape[0] != len(self.offsets) or proprio.shape[0] != len(self.offsets):
            raise ValueError(
                f"Expected features for {len(self.offsets)} offsets, got "
                f"{tuple(z_rl.shape)} and {tuple(proprio.shape)}."
            )
        chunk.stride_z_rl = z_rl
        chunk.stride_proprio = proprio
        if head_pose is not None:
            head_pose = _cpu_float(head_pose)
            if head_pose.shape[0] != len(self.offsets):
                raise ValueError(
                    f"Expected head pose for {len(self.offsets)} offsets, got "
                    f"{tuple(head_pose.shape)}."
                )
            chunk.stride_head_pose = head_pose

    def pop_ready(self) -> list[dict[str, torch.Tensor]]:
        """Return every window that can be resolved, in collection order."""
        windows = []
        while self._pending:
            chunk_id, row = self._pending[0]
            if self._next_chunk_id - 1 - chunk_id > self.max_pending_chunks:
                result = None
            else:
                result = self._resolve(chunk_id, row)
            if result is _WAIT:
                break
            self._pending.popleft()
            if result is None:
                self.dropped_windows += 1
            else:
                windows.append(result)
        self._prune()
        return windows

    def _prune(self) -> None:
        keep_from = self._next_chunk_id - 2
        if self._pending:
            keep_from = min(keep_from, self._pending[0][0])
        for chunk_id in [cid for cid in self._chunks if cid < keep_from]:
            del self._chunks[chunk_id]

    def _chunk_at(self, segment: int, time: int) -> _Chunk | None:
        for chunk in self._chunks.values():
            if chunk.segment == segment and chunk.start <= time < chunk.end:
                return chunk
        return None

    def _timeline_may_grow(self, segment: int, time: int) -> bool:
        return (
            segment == self._segment
            and not self._segment_closed
            and time >= self._next_start
        )

    def _observation(self, segment: int, time: int):
        """Features of the observation decoded at ``time``, ``None`` or ``_WAIT``.

        Features are ``(z_rl, proprio, owner chunk, head_pose or None)``.
        """
        for chunk in self._chunks.values():
            if chunk.segment != segment:
                continue
            if chunk.start == time:
                return chunk.z_rl, chunk.proprio, chunk, chunk.head_pose
            if chunk.start < time <= chunk.action_start:
                # Nothing is decoded while the RTC bridge runs.
                return None
            if chunk.action_start < time < chunk.end:
                offset = time - chunk.action_start
                if offset not in self.offsets:
                    return None
                if chunk.rewards is None:
                    return _WAIT
                if not bool(chunk.valid_steps[offset - 1]):
                    return None
                if (
                    chunk.terminal_step is not None
                    and offset - 1 >= chunk.terminal_step
                ):
                    return None
                if chunk.features_missing:
                    return None
                if chunk.stride_z_rl is None:
                    return _WAIT
                row = self.offsets.index(offset)
                head_pose = (
                    None
                    if chunk.stride_head_pose is None
                    else chunk.stride_head_pose[row]
                )
                return (
                    chunk.stride_z_rl[row],
                    chunk.stride_proprio[row],
                    chunk,
                    head_pose,
                )
        return _WAIT if self._timeline_may_grow(segment, time) else None

    def _command_at(self, segment: int, time: int):
        """Command executed at ``time``, ``_WAIT``, or the held final command.

        After the episode's last executed step the arm holds that command.
        """
        chunk = self._chunk_at(segment, time)
        if chunk is None:
            if self._timeline_may_grow(segment, time):
                return _WAIT
            return self._final_command(segment)
        if time < chunk.action_start:
            return chunk.bridge[time - chunk.start]
        if chunk.rewards is None:
            return _WAIT
        step = time - chunk.action_start
        if chunk.terminal_step is not None and step > chunk.terminal_step:
            return chunk.actions[chunk.terminal_step]
        return chunk.actions[step]

    def _final_command(self, segment: int):
        chunks = [chunk for chunk in self._chunks.values() if chunk.segment == segment]
        if not chunks:
            return None
        last = max(chunks, key=lambda chunk: chunk.start)
        if last.rewards is None:
            return _WAIT
        step = (
            last.terminal_step
            if last.terminal_step is not None
            else last.actions.shape[0] - 1
        )
        return last.actions[step]

    def _reference(self, plan: torch.Tensor, index: int) -> torch.Tensor:
        tail = plan[index:]
        return torch.cat((tail, tail[-1:].expand(index, -1)), dim=0)

    def _resolve(self, chunk_id: int, row: int):
        origin = self._chunks.get(chunk_id)
        if origin is None:
            return None
        if origin.rewards is None:
            return _WAIT
        offset = self.offsets[row]
        if not bool(origin.valid_steps[offset - 1]):
            return None
        if origin.terminal_step is not None and offset - 1 >= origin.terminal_step:
            return None
        if origin.features_missing:
            return None
        if origin.stride_z_rl is None:
            return _WAIT

        first_action = origin.action_start + offset + self.latency
        reference_index = first_action - origin.start
        if reference_index >= self.horizon:
            return None

        actions, rewards, human, dones = [], [], [], []
        terminal = False
        for time in range(first_action, first_action + self.chunk_len):
            if terminal:
                actions.append(actions[-1])
                rewards.append(torch.zeros(()))
                human.append(False)
                dones.append(True)
                continue
            chunk = self._chunk_at(origin.segment, time)
            if chunk is None:
                return _WAIT if self._timeline_may_grow(origin.segment, time) else None
            if chunk.rewards is None:
                return _WAIT
            if time < chunk.action_start:
                actions.append(chunk.bridge[time - chunk.start])
                rewards.append(torch.zeros(()))
                human.append(False)
                dones.append(False)
                continue
            step = time - chunk.action_start
            if not bool(chunk.valid_steps[step]):
                return None
            if chunk.terminal_step is not None and step > chunk.terminal_step:
                return None
            actions.append(chunk.actions[step])
            rewards.append(chunk.rewards[step])
            human.append(bool(chunk.human[step]))
            terminal = chunk.terminal_step == step
            dones.append(terminal)

        actions = torch.stack(actions)
        human = torch.tensor(human, dtype=torch.bool)
        curr_ref = self._reference(origin.plan, reference_index)
        # Paper Sec. IV-B: an intervention replaces the reference in the buffer.
        if self.replace_human_reference:
            curr_ref[: self.chunk_len] = torch.where(
                human[:, None], actions, curr_ref[: self.chunk_len]
            )
        curr_z_rl = origin.stride_z_rl[row]
        curr_proprio = origin.stride_proprio[row]
        curr_head_pose = (
            None if origin.stride_head_pose is None else origin.stride_head_pose[row]
        )
        curr_command = self._command_at(origin.segment, first_action - 1)
        if curr_command is _WAIT or curr_command is None:
            return curr_command
        curr_bridge_delta = curr_command - curr_proprio
        if terminal:
            next_z_rl, next_proprio, next_ref = curr_z_rl, curr_proprio, curr_ref
            next_bridge_delta = curr_bridge_delta
            next_head_pose = curr_head_pose
        else:
            next_time = first_action + self.chunk_len
            observation = self._observation(origin.segment, next_time)
            if observation is _WAIT or observation is None:
                return observation
            next_z_rl, next_proprio, owner, next_head_pose = observation
            next_index = next_time + self.latency - owner.start
            if next_index >= self.horizon:
                return None
            next_ref = self._reference(owner.plan, next_index)
            next_command = self._command_at(
                origin.segment, next_time + self.latency - 1
            )
            if next_command is _WAIT or next_command is None:
                return next_command
            next_bridge_delta = next_command - next_proprio

        window = {
            "actions": actions,
            "rewards": torch.stack(rewards).to(torch.float32),
            "dones": torch.tensor(dones, dtype=torch.bool),
            "intervene_flags": human,
            "curr_z_rl": curr_z_rl,
            "curr_proprio": curr_proprio,
            "curr_ref_chunk": curr_ref,
            "curr_bridge_delta": curr_bridge_delta,
            "next_z_rl": next_z_rl,
            "next_proprio": next_proprio,
            "next_ref_chunk": next_ref,
            "next_bridge_delta": next_bridge_delta,
            "actor_switch": torch.tensor(origin.actor_switch),
            "reference_warmup": torch.tensor(origin.reference_warmup),
            "episode_id": torch.tensor(origin.episode_id, dtype=torch.long),
        }
        if curr_head_pose is not None and next_head_pose is not None:
            window["curr_head_pose"] = curr_head_pose
            window["next_head_pose"] = next_head_pose
        return window


def pack_stride_windows(
    windows_per_env: list[list[dict[str, torch.Tensor]]],
    *,
    slots: int,
    chunk_len: int,
    action_dim: int,
    horizon: int,
    z_dim: int,
    proprio_dim: int,
    head_pose_dim: int = 0,
) -> dict[str, torch.Tensor]:
    """Pack windows into fixed [B, slots, ...] forward_inputs tensors.

    Every chunk step carries the same keys and shapes so rollout steps still
    stack into one trajectory; ``valid`` marks the occupied slots. With
    ``head_pose_dim`` every slot carries a head pose; a window without one
    keeps zeros, i.e. "not visible".
    """
    batch_size = len(windows_per_env)

    def zeros(*shape: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros(batch_size, slots, *shape, dtype=dtype)

    packed = {
        "valid": zeros(dtype=torch.bool),
        "actions": zeros(chunk_len * action_dim),
        "rewards": zeros(chunk_len),
        "dones": zeros(chunk_len, dtype=torch.bool),
        "intervene_flags": zeros(chunk_len * action_dim, dtype=torch.bool),
        "curr_z_rl": zeros(z_dim),
        "curr_proprio": zeros(proprio_dim),
        "curr_ref_chunk": zeros(horizon, action_dim),
        "curr_bridge_delta": zeros(proprio_dim),
        "next_z_rl": zeros(z_dim),
        "next_proprio": zeros(proprio_dim),
        "next_ref_chunk": zeros(horizon, action_dim),
        "next_bridge_delta": zeros(proprio_dim),
        "actor_switch": zeros(dtype=torch.bool),
        "reference_warmup": zeros(dtype=torch.bool),
        "episode_id": zeros(dtype=torch.long),
    }
    if head_pose_dim:
        packed["curr_head_pose"] = zeros(head_pose_dim)
        packed["next_head_pose"] = zeros(head_pose_dim)
    for env_index, windows in enumerate(windows_per_env):
        if len(windows) > slots:
            raise ValueError(f"{len(windows)} windows do not fit in {slots} slots.")
        for slot, window in enumerate(windows):
            packed["valid"][env_index, slot] = True
            packed["actions"][env_index, slot] = window["actions"].reshape(-1)
            packed["intervene_flags"][env_index, slot] = (
                window["intervene_flags"][:, None].expand(-1, action_dim).reshape(-1)
            )
            for key in STRIDE_SLOT_KEYS[5:]:
                if key in packed and key in window:
                    packed[key][env_index, slot] = window[key]
            packed["rewards"][env_index, slot] = window["rewards"]
            packed["dones"][env_index, slot] = window["dones"]
    return {f"{RLT_STRIDE_PREFIX}{key}": value for key, value in packed.items()}


def iter_stride_windows(
    forward_inputs: dict[str, Any], row: int
) -> Iterator[dict[str, torch.Tensor]]:
    """Yield the occupied slots of one flattened forward_inputs row."""
    valid = forward_inputs.get(f"{RLT_STRIDE_PREFIX}valid")
    if not isinstance(valid, torch.Tensor) or row >= valid.shape[0]:
        return
    for slot in torch.nonzero(valid[row].reshape(-1)).reshape(-1).tolist():
        yield {
            key: forward_inputs[f"{RLT_STRIDE_PREFIX}{key}"][row, slot]
            for key in STRIDE_SLOT_KEYS[1:]
            if f"{RLT_STRIDE_PREFIX}{key}" in forward_inputs
        }
