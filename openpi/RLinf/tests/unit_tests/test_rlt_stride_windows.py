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

import pytest
import torch

from rlinf.algorithms.rlt.subsample import (
    StrideWindowAssembler,
    iter_stride_windows,
    pack_stride_windows,
    stride_offsets,
)

C, H, D, Z, P, LATENCY = 10, 50, 14, 3, 14, 5


def _steps(first: int, count: int) -> torch.Tensor:
    """Commands whose value is their execution step, so windows read as step ranges."""
    return torch.arange(first, first + count, dtype=torch.float32)[:, None].repeat(1, D)


def _add_chunk(assembler, start, bridge, *, record=True, episode_id=7):
    return assembler.add_chunk(
        bridge=_steps(start, bridge),
        actions=_steps(start + bridge, C),
        # A VLA plan anchored at its observation: sample i is meant for step start + i.
        plan=_steps(start, H),
        z_rl=torch.full((Z,), float(start)),
        proprio=torch.full((P,), float(start)),
        record=record,
        actor_switch=True,
        episode_id=episode_id,
    )


def _encode(assembler, chunk_id, action_start):
    times = torch.tensor([float(action_start + offset) for offset in assembler.offsets])
    assembler.add_stride_features(
        chunk_id, times[:, None].repeat(1, Z), times[:, None].repeat(1, P)
    )


def _finish(assembler, chunk_id, action_start, *, features=True, **outcome):
    outcome.setdefault("rewards", torch.zeros(C))
    assembler.add_outcome(chunk_id, **outcome)
    if features:
        _encode(assembler, chunk_id, action_start)


def _regular_windows():
    assembler = StrideWindowAssembler(chunk_len=C, stride=2, latency=LATENCY, horizon=H)
    # The episode's first chunk runs without a bridge; later chunks follow a
    # drained 5-step RTC bridge. Chunk 1 executes steps 15..24, chunk 2 30..39.
    first = _add_chunk(assembler, 0, 0, record=False)
    _finish(assembler, first, 0)
    second = _add_chunk(assembler, 10, LATENCY)
    _finish(assembler, second, 15)
    pending_after_origin = assembler.pop_ready()
    third = _add_chunk(assembler, 25, LATENCY)
    _finish(assembler, third, 30, features=False)
    pending_after_next_outcome = assembler.pop_ready()
    _encode(assembler, third, 30)
    # A next observation's bridge delta needs the command one step before its
    # actions: offsets 2 and 4 find it in chunk 2, offsets 6 and 8 in the
    # bridge that starts chunk 3.
    released_with_features = assembler.pop_ready()
    assert len(released_with_features) == 2
    _add_chunk(assembler, 40, LATENCY)
    return (
        assembler,
        pending_after_origin,
        pending_after_next_outcome,
        released_with_features + assembler.pop_ready(),
    )


def test_stride_offsets_leave_offset_zero_to_the_chunk_row() -> None:
    assert stride_offsets(10, 2) == [2, 4, 6, 8]
    assert stride_offsets(10, 0) == []
    with pytest.raises(ValueError):
        stride_offsets(10, 11)


def test_windows_follow_the_executed_timeline_across_the_rtc_bridge() -> None:
    assembler, before_next, before_features, windows = _regular_windows()

    assert before_next == []
    assert before_features == []
    assert len(windows) == 4
    for offset, window in zip(assembler.offsets, windows):
        observed = 15 + offset
        first_action = observed + LATENCY
        torch.testing.assert_close(
            window["actions"][:, 0],
            torch.arange(first_action, first_action + C, dtype=torch.float32),
        )
        assert window["curr_z_rl"][0].item() == observed
        assert window["next_z_rl"][0].item() == first_action + C
        # The last plan chosen before the observation, re-anchored at its first action.
        assert window["curr_ref_chunk"][0, 0].item() == first_action
        assert window["next_ref_chunk"][0, 0].item() == first_action + C + LATENCY
        assert window["curr_ref_chunk"].shape == (H, D)
        # Commands equal their execution step and proprio its observation
        # time, so the bridge delta is the latency minus one step.
        torch.testing.assert_close(
            window["curr_bridge_delta"], torch.full((P,), float(LATENCY - 1))
        )
        torch.testing.assert_close(
            window["next_bridge_delta"], torch.full((P,), float(LATENCY - 1))
        )
        assert not window["dones"].any()
        assert window["episode_id"].item() == 7


def test_windows_with_the_deployed_15_step_bridge() -> None:
    """Latency 15 > C: every window spans the next bridge and stays inside H."""
    latency = 15
    assembler = StrideWindowAssembler(chunk_len=C, stride=2, latency=latency, horizon=H)
    first = _add_chunk(assembler, 0, 0, record=False)
    _finish(assembler, first, 0)
    # Chunk 1 executes steps 25..34, chunk 2 runs its bridge on 35..49.
    second = _add_chunk(assembler, 10, latency)
    _finish(assembler, second, 25)
    third = _add_chunk(assembler, 35, latency)
    _finish(assembler, third, 50)
    assert assembler.pop_ready() == []
    _add_chunk(assembler, 60, latency)
    windows = assembler.pop_ready()

    assert len(windows) == 4
    for offset, window in zip(assembler.offsets, windows):
        observed = 25 + offset
        first_action = observed + latency
        torch.testing.assert_close(
            window["actions"][:, 0],
            torch.arange(first_action, first_action + C, dtype=torch.float32),
        )
        assert window["curr_z_rl"][0].item() == observed
        assert window["next_z_rl"][0].item() == first_action + C
        assert window["curr_ref_chunk"][0, 0].item() == first_action
        assert window["next_ref_chunk"][0, 0].item() == first_action + C + latency
        torch.testing.assert_close(
            window["curr_bridge_delta"], torch.full((P,), float(latency - 1))
        )
        torch.testing.assert_close(
            window["next_bridge_delta"], torch.full((P,), float(latency - 1))
        )
        assert not window["dones"].any()


def test_terminal_windows_are_padded_and_post_terminal_origins_dropped() -> None:
    assembler = StrideWindowAssembler(chunk_len=C, stride=2, latency=LATENCY, horizon=H)
    previous = _add_chunk(assembler, 0, 0)
    _finish(assembler, previous, 0)
    last = _add_chunk(assembler, 10, LATENCY)
    rewards = torch.zeros(C)
    rewards[3] = 1.0
    # Success on the chunk's fourth action, executed at step 18.
    _finish(assembler, last, 15, rewards=rewards, terminal_step=3)

    windows = assembler.pop_ready()

    assert len(windows) == 4
    non_terminal, *terminal = windows
    assert not non_terminal["dones"].any()
    assert non_terminal["next_z_rl"][0].item() == 17.0
    for window, position in zip(terminal, (9, 7, 5)):
        assert not window["dones"][:position].any()
        assert window["dones"][position:].all()
        assert window["rewards"][position].item() == 1.0
        assert window["rewards"].sum().item() == 1.0
        assert (window["actions"][position:, 0] == 18.0).all()
        torch.testing.assert_close(window["next_z_rl"], window["curr_z_rl"])
    # Every window of the terminal chunk starts after its success step.
    assert assembler.dropped_windows == 4
    new_episode = _add_chunk(assembler, 0, 0, episode_id=8)
    assert assembler._chunks[new_episode].start == 0


def test_operator_actions_replace_commands_and_reference() -> None:
    assembler = StrideWindowAssembler(chunk_len=C, stride=2, latency=LATENCY, horizon=H)
    first = _add_chunk(assembler, 0, 0)
    _finish(assembler, first, 0)
    flags = torch.zeros(C, dtype=torch.bool)
    flags[:2] = True
    second = _add_chunk(assembler, 10, LATENCY)
    _finish(
        assembler,
        second,
        15,
        human_actions=torch.full((C, D), -1.0),
        human_flags=flags,
    )

    window = assembler.pop_ready()[0]

    # Offset 2 of chunk 0 covers steps 7..16; the operator drove steps 15 and 16.
    assert window["intervene_flags"].tolist() == [False] * 8 + [True, True]
    assert (window["actions"][8:] == -1.0).all()
    assert (window["curr_ref_chunk"][8:10] == -1.0).all()
    assert window["curr_ref_chunk"][7, 0].item() == 14.0


def test_discarded_hold_actions_do_not_become_stride_transitions() -> None:
    assembler = StrideWindowAssembler(chunk_len=C, stride=2, latency=0, horizon=H)
    first = _add_chunk(assembler, 0, 0)
    _finish(assembler, first, 0)
    second = _add_chunk(assembler, C, 0)
    _finish(assembler, second, C, valid_steps=torch.zeros(C, dtype=torch.bool))
    assert assembler.pop_ready() == []


def test_zero_latency_human_terminal_window_uses_mux_actions():
    assembler = StrideWindowAssembler(chunk_len=C, stride=2, latency=0, horizon=H)
    first = _add_chunk(assembler, 0, 0)
    _finish(assembler, first, 0)
    second = _add_chunk(assembler, C, 0)
    rewards = torch.zeros(C)
    rewards[1] = 1
    _finish(
        assembler,
        second,
        C,
        terminal_step=1,
        rewards=rewards,
        human_flags=torch.ones(C, dtype=torch.bool),
        human_actions=torch.full((C, D), -2.0),
        executed_actions=torch.full((C, D), -2.0),
    )
    windows = assembler.pop_ready()
    assert windows
    for window in windows:
        assert window["rewards"].sum() == 1
        assert window["intervene_flags"].sum() == 2
        assert (window["actions"][window["intervene_flags"]] == -2).all()


def test_missing_features_drop_windows_instead_of_blocking() -> None:
    assembler = StrideWindowAssembler(chunk_len=C, stride=2, latency=LATENCY, horizon=H)
    first = _add_chunk(assembler, 0, 0)
    _finish(assembler, first, 0)
    second = _add_chunk(assembler, 10, LATENCY)
    _finish(assembler, second, 15, features=False)
    assembler.add_stride_features(second, None, None)

    assert assembler.pop_ready() == []
    # Chunk 0 lost its next observations and chunk 1 its origins.
    assert assembler.dropped_windows == 8


def test_packed_slots_round_trip() -> None:
    _, _, _, windows = _regular_windows()

    packed = pack_stride_windows(
        [windows[:2]],
        slots=12,
        chunk_len=C,
        action_dim=D,
        horizon=H,
        z_dim=Z,
        proprio_dim=P,
    )
    unpacked = list(iter_stride_windows(packed, 0))

    assert packed["rlt_stride_valid"].sum().item() == 2
    assert len(unpacked) == 2
    torch.testing.assert_close(
        unpacked[1]["actions"], windows[1]["actions"].reshape(-1)
    )
    torch.testing.assert_close(
        unpacked[1]["next_ref_chunk"], windows[1]["next_ref_chunk"]
    )
    assert unpacked[0]["intervene_flags"].shape == (C * D,)


def test_operator_steps_can_keep_the_vla_reference() -> None:
    assembler = StrideWindowAssembler(
        chunk_len=C,
        stride=2,
        latency=LATENCY,
        horizon=H,
        replace_human_reference=False,
    )
    first = _add_chunk(assembler, 0, 0)
    _finish(assembler, first, 0)
    flags = torch.zeros(C, dtype=torch.bool)
    flags[:2] = True
    second = _add_chunk(assembler, 10, LATENCY)
    _finish(
        assembler,
        second,
        15,
        human_actions=torch.full((C, D), -1.0),
        human_flags=flags,
    )

    window = assembler.pop_ready()[0]

    # The executed operator actions are still recorded for BC ...
    assert (window["actions"][8:] == -1.0).all()
    # ... but the Actor input keeps the VLA plan for those steps.
    assert window["curr_ref_chunk"][8, 0].item() == 15.0
    assert window["curr_ref_chunk"][9, 0].item() == 16.0
