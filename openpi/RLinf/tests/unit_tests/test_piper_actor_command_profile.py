"""Exercise command continuity without ROS, CUDA, or robot motion."""

from dataclasses import asdict

import numpy as np
import pytest

from rlinf.envs.realworld.piper.actor_command_profile import ActorCommandProfile
from rlinf.envs.realworld.piper.piper_env import PiperRobotConfig


def _profile(**overrides):
    cfg = PiperRobotConfig(
        command_publish_hz=200,
        command_interpolation_duration_s=1 / 15,
        **overrides,
    )
    initial = np.array([0, 1, -1, 0, 0, 0, 0.01] * 2, np.float64)
    return ActorCommandProfile(asdict(cfg), initial, 0), initial


def test_first_actor_goal_and_chunk_handoffs_share_the_previous_command() -> None:
    profile, initial = _profile()
    first = profile.set_goal(initial + 0.1)
    np.testing.assert_allclose(first - initial, profile.max_step, atol=1e-7)
    second = profile.set_goal(initial - 0.1)
    np.testing.assert_allclose(second - first, -profile.max_step, atol=1e-7)
    np.testing.assert_allclose(profile.sample(0), initial)


@pytest.mark.parametrize(
    "step,velocity,acceleration,response",
    [
        (0.002, 0.08, 0.4, 0.15),
        (0.004, 0.10, 0.6, 0.12),
    ],
)
def test_alternating_actor_proposals_cannot_reverse_command_velocity_abruptly(
    step,
    velocity,
    acceleration,
    response,
) -> None:
    profile, initial = _profile(
        actor_command_joint_step_rad=step,
        actor_command_joint_velocity_rad_s=velocity,
        actor_command_joint_acceleration_rad_s2=acceleration,
        actor_command_response_time_s=response,
    )
    positions = [initial.copy()]
    velocities = [profile.velocity.copy()]
    for tick in range(1, 2001):
        if tick % 13 == 1:
            sign = 1 if (tick // 13) % 2 else -1
            profile.set_goal(initial + sign * 0.02)
        profile.sample(tick / 200)
        positions.append(profile.position.copy())
        velocities.append(profile.velocity.copy())
    velocities = np.array(velocities)
    positions = np.array(positions)
    assert np.all(np.abs(velocities) <= profile.max_velocity + 1e-9)
    assert np.all(
        np.abs(np.diff(velocities, axis=0) * 200) <= profile.max_acceleration + 1e-8
    )
    assert np.all(
        np.abs(np.diff(positions, axis=0)) <= profile.max_velocity / 200 + 1e-9
    )
    # Oscillatory proposals become small continuous commands, not 0.04-rad flips.
    assert np.ptp(positions[:, 0]) < step


def test_tracker_converges_to_a_persistent_correction_without_resetting_each_chunk() -> (
    None
):
    profile, initial = _profile()
    for tick in range(1, 2001):
        if tick % 13 == 1:
            profile.set_goal(initial + 0.01)
        profile.sample(tick / 200)
    np.testing.assert_allclose(profile.position, initial + 0.01, atol=1e-5)


def test_publisher_stall_does_not_create_a_catch_up_jump() -> None:
    profile, initial = _profile()
    profile.set_goal(initial + 0.02)
    profile.sample(0.005)
    before = profile.position.copy()
    after = profile.sample(5)
    assert np.all(np.abs(after - before) <= profile.max_velocity / 200 + 1e-7)
    with pytest.raises(ValueError, match="finite"):
        profile.set_goal(np.full(14, np.nan))
