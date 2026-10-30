from unittest.mock import Mock

import numpy as np
import pytest
from openpi_client import image_tools

from rlinf.envs.realworld.piper.control_process import PiperControlProcess
from rlinf.envs.realworld.piper.piper_env import PiperRobotConfig, PiperWireEnv
from rlinf.workers.env.env_worker import slice_rtc_reference_tail


def _controller(config: PiperRobotConfig) -> PiperWireEnv:
    env = PiperWireEnv.__new__(PiperWireEnv)
    env.config = config
    env._logger = Mock()
    env._command_lock = __import__("threading").Lock()
    env._policy_action_lock = __import__("threading").Lock()
    env._next_policy_action_time = 0.0
    env._last_policy_dispatch_time = 0.0
    env._last_policy_dispatch_source = None
    env._slow_policy_interval_count = 0
    env._policy_dispatch_count = 0
    env._policy_interval_sum = 0.0
    env._policy_interval_max_error = 0.0
    env._last_smoothed_action = None
    env._last_raw_arm_action = None
    env._last_raw_arm_delta = None
    env._observation_metric_count = 0
    env._observation_sync_ms_sum = 0.0
    env._observation_decode_ms_sum = 0.0
    env._observation_total_ms_max = 0.0
    env._command_metric_count = 0
    env._raw_step_sum = 0.0
    env._raw_step_max = 0.0
    env._filtered_step_sum = 0.0
    env._filtered_step_max = 0.0
    env._raw_accel_max = 0.0
    env._last_policy_action = None
    env._chunk_action_index = 0
    env._completed_action_chunks = 0
    env._episode_start_position = None
    env._rtc_lock = __import__("threading").Lock()
    env._rtc_stop = __import__("threading").Event()
    env._rtc_thread = None
    env._rtc_executed_steps = 0
    env._rtc_last_action = None
    env._queued_rtc_tail = None
    env._rtc_rollover_count = 0
    return env


class _ImageMessage:
    def __init__(self, image: np.ndarray, encoding: str) -> None:
        self.encoding = encoding
        self.height, self.width = image.shape[:2]
        self.step = self.width * 3
        self.data = image.tobytes()


@pytest.mark.parametrize("encoding", ["bgr8", "rgb8"])
def test_camera_decode_is_byte_identical_to_validated_piper_real(
    encoding: str,
) -> None:
    """RLT must send JAX the same resized BGR pixels as piper_real."""
    env = _controller(PiperRobotConfig(render_height=224, render_width=224))
    source = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
    message = _ImageMessage(source, encoding)
    expected_bgr = source[..., ::-1].copy() if encoding == "rgb8" else source
    expected = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(expected_bgr, 224, 224)
    )

    decoded = env._decode(message)

    np.testing.assert_array_equal(decoded, expected)


def test_direct_actor_skips_rtc_and_boundary_blending_then_restores_jax() -> None:
    env = _controller(PiperRobotConfig(rtc_enabled=True))
    action = np.full((10, 14), 0.01, np.float32)
    plan = np.full((50, 14), 0.02, np.float32)
    env._stop_rtc_tail = Mock(return_value=(5, np.full(14, 0.03, np.float32)))
    env._completed_action_chunks = 1
    env._last_policy_action = np.full(14, 0.1, np.float32)
    env.set_actor_execution_mode(True)
    env.queue_rtc_tail(plan)
    assert env._queued_rtc_tail is None
    env.start_rtc_tail(plan)
    aligned, offset = env.align_rtc_chunk(action, plan)
    np.testing.assert_array_equal(aligned, action)
    assert offset == 0
    assert env._stop_rtc_tail.call_args.kwargs == {"drain": False}
    np.testing.assert_array_equal(
        env._blend_action_chunk_boundary(action[0]), action[0]
    )
    env.set_actor_execution_mode(False)
    env.queue_rtc_tail(plan)
    np.testing.assert_array_equal(env._queued_rtc_tail, plan)
    _, offset = env.align_rtc_chunk(action, plan)
    assert offset == 5
    assert env._stop_rtc_tail.call_args.kwargs == {"drain": True}


def test_direct_actor_rpc_requests_unsmoothed_execution() -> None:
    env = _controller(PiperRobotConfig())
    env._actor_direct_execution = True
    env._get_joint_position = Mock(return_value=np.zeros(14, np.float32))
    env._control_process = Mock()
    env._control_process.apply_action.return_value = np.zeros(14, np.float32)
    env._control_process.last_accepted_target = np.full(14, 0.01, np.float32)
    env._apply_scheduled_policy_action(np.full(14, 0.01, np.float32))
    assert env._control_process.apply_action.call_args.kwargs == {"smooth": False}


def test_episode_start_pose_is_captured_only_once() -> None:
    env = _controller(PiperRobotConfig())
    env.config.is_dummy = False
    first = np.arange(14, dtype=np.float32) / 100.0
    second = first + 1.0
    env._get_joint_position = Mock(side_effect=[first, second])

    assert env.capture_episode_start_position() is True
    assert env.capture_episode_start_position() is False

    np.testing.assert_array_equal(env._episode_start_position, first)
    assert env._get_joint_position.call_count == 1


def test_return_to_episode_start_can_be_interrupted_with_hold() -> None:
    env = _controller(PiperRobotConfig())
    env.config.is_dummy = False
    env._episode_start_position = np.ones(14, dtype=np.float32)
    env.hold_position = Mock()
    env._rospy = Mock()
    env._rospy.is_shutdown.return_value = False

    completed = env.return_to_episode_start(stop_requested=lambda: True)

    assert completed is False
    env.hold_position.assert_called_once_with()


def test_return_opens_grippers_while_moving_then_closes_at_start_pose() -> None:
    config = PiperRobotConfig(step_frequency=1000.0, reset_timeout_s=1.0)
    env = _controller(config)
    env.config.is_dummy = False
    goal = np.zeros(14, dtype=np.float32)
    moving_closed = goal.copy()
    moving_closed[0] = 0.1
    moving_open = moving_closed.copy()
    moving_open[[6, 13]] = config.reset_open_gripper_position_m
    at_goal_open = goal.copy()
    at_goal_open[[6, 13]] = config.reset_open_gripper_position_m
    env._episode_start_position = goal.copy()
    env._get_joint_position = Mock(
        side_effect=[moving_closed, moving_open, at_goal_open, goal.copy()]
    )
    env._publish_action = Mock()
    env._rospy = Mock()
    env._rospy.is_shutdown.return_value = False

    assert env.return_to_episode_start() is True

    first, second, third, final = [
        call.args[0] for call in env._publish_action.call_args_list
    ]
    assert first[0] == pytest.approx(moving_closed[0])
    np.testing.assert_allclose(first[[6, 13]], config.max_gripper_delta, atol=1e-6)
    assert second[0] < moving_open[0]
    np.testing.assert_allclose(
        second[[6, 13]], config.reset_open_gripper_position_m, atol=1e-6
    )
    assert np.all(third[[6, 13]] < at_goal_open[[6, 13]])
    np.testing.assert_allclose(
        final[[6, 13]], config.reset_closed_gripper_position_m, atol=1e-6
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"reset_open_gripper_position_m": 0.081}, "reset_open_gripper_position_m"),
        (
            {"reset_closed_gripper_position_m": -0.001},
            "reset_closed_gripper_position_m",
        ),
    ],
)
def test_piper_direct_control_rejects_invalid_reset_limits(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PiperRobotConfig(**kwargs)


def test_isolated_control_process_requires_high_rate_publisher() -> None:
    with pytest.raises(
        ValueError,
        match="isolated_control_process requires command_publish_hz > 0",
    ):
        PiperRobotConfig(isolated_control_process=True)


def test_return_target_is_forwarded_to_isolated_control_process() -> None:
    env = _controller(
        PiperRobotConfig(
            command_publish_hz=200.0,
            command_interpolation_duration_s=1.0 / 15.0,
            isolated_control_process=True,
        )
    )
    env._control_process = Mock()
    target = np.ones(14, dtype=np.float32)
    current = np.zeros(14, dtype=np.float32)

    env._set_command_target(target, current_reference=current, smooth=False)

    env._control_process.apply_reset_target.assert_called_once()
    np.testing.assert_array_equal(
        env._control_process.apply_reset_target.call_args.args[0],
        target,
    )
    np.testing.assert_array_equal(
        env._control_process.apply_reset_target.call_args.args[1],
        current,
    )


def test_mode2_suspend_stops_isolated_policy_publisher() -> None:
    env = _controller(
        PiperRobotConfig(
            command_publish_hz=200.0,
            command_interpolation_duration_s=1.0 / 15.0,
            isolated_control_process=True,
        )
    )
    env.config.is_dummy = False
    env._control_process = Mock()
    env._queued_rtc_tail = np.ones((2, 14), dtype=np.float32)
    env._stop_rtc_tail = Mock()

    env.suspend_policy_output()

    assert env._queued_rtc_tail is None
    env._stop_rtc_tail.assert_called_once_with()
    env._control_process.suspend.assert_called_once_with()


def test_control_process_suspend_uses_dedicated_operation() -> None:
    process = PiperControlProcess.__new__(PiperControlProcess)
    process._request = Mock()

    process.suspend()

    process._request.assert_called_once_with("suspend")


def test_control_process_can_drain_rtc_before_promoting_new_chunk() -> None:
    process = PiperControlProcess.__new__(PiperControlProcess)
    process._request = Mock(return_value=(15, np.zeros(14, dtype=np.float32)))

    executed, last_action = process.stop_rtc(drain=True)

    process._request.assert_called_once_with("drain_rtc")
    assert executed == 15
    assert last_action is not None


def test_external_control_step_observes_without_policy_publish() -> None:
    env = _controller(PiperRobotConfig())
    env.config.is_dummy = False
    env._num_steps = 0
    env._remaining_actions_in_chunk = 1
    env._queued_rtc_tail = np.ones((2, 14), dtype=np.float32)
    observation = {
        "state": {"joint_position": np.zeros(14, dtype=np.float32)},
        "frames": {},
    }
    env._get_joint_position = Mock(return_value=np.zeros(14, dtype=np.float32))
    env._get_observation = Mock(return_value=observation)
    env._apply_scheduled_policy_action = Mock()

    result, *_ = env.step_external_control()

    assert result is observation
    assert env._queued_rtc_tail is None
    env._apply_scheduled_policy_action.assert_not_called()


def test_interpolated_command_keeps_gripper_discrete() -> None:
    config = PiperRobotConfig(
        command_publish_hz=200.0,
        command_interpolation_duration_s=0.1,
    )
    env = _controller(config)
    env._command_start = np.zeros(14, dtype=np.float32)
    env._command_target = np.ones(14, dtype=np.float32)
    env._command_start_time = 10.0

    halfway = env._interpolated_command_locked(10.05)

    np.testing.assert_allclose(
        halfway[np.r_[0:6, 7:13]],
        0.5,
        atol=1e-6,
    )
    np.testing.assert_array_equal(halfway[[6, 13]], 0.0)


def test_command_target_applies_ema_and_velocity_limit() -> None:
    config = PiperRobotConfig(
        command_publish_hz=200.0,
        command_interpolation_duration_s=0.1,
        action_smoothing_alpha=0.4,
        max_arm_velocity_rad_s=0.4,
    )
    env = _controller(config)
    env._command_lock = __import__("threading").Lock()
    env._command_start = np.zeros(14, dtype=np.float32)
    env._command_target = np.zeros(14, dtype=np.float32)
    env._command_start_time = 0.0
    env._last_smoothed_action = np.full(14, 0.5, dtype=np.float32)

    env._set_command_target(
        np.ones(14, dtype=np.float32),
        current_reference=np.zeros(14, dtype=np.float32),
        smooth=True,
    )

    arm_indices = np.r_[0:6, 7:13]
    np.testing.assert_allclose(
        env._command_target[arm_indices],
        0.04,
        atol=1e-6,
    )
    np.testing.assert_allclose(env._command_target[[6, 13]], 1.0)


def test_interpolated_action_is_not_reclamped_around_lagging_feedback() -> None:
    config = PiperRobotConfig(
        command_publish_hz=200.0,
        command_interpolation_duration_s=0.1,
        action_smoothing_alpha=1.0,
        max_arm_velocity_rad_s=0.0,
        max_joint_delta=0.05,
    )
    env = _controller(config)
    feedback = np.zeros(14, dtype=np.float32)
    requested = np.full(14, 0.2, dtype=np.float32)
    requested[[2, 9]] = -0.2
    env._get_joint_position = Mock(return_value=feedback)
    env._set_command_target = Mock()

    env._apply_low_level_action(requested)

    applied = env._set_command_target.call_args.args[0]
    arm_indices = np.r_[0:6, 7:13]
    np.testing.assert_allclose(applied[arm_indices], requested[arm_indices])


def test_direct_action_keeps_feedback_relative_step_clamp() -> None:
    config = PiperRobotConfig(
        command_publish_hz=0.0,
        max_joint_delta=0.05,
    )
    env = _controller(config)
    feedback = np.zeros(14, dtype=np.float32)
    requested = np.full(14, 0.2, dtype=np.float32)
    requested[[2, 9]] = -0.2
    env._get_joint_position = Mock(return_value=feedback)
    env._set_command_target = Mock()

    env._apply_low_level_action(requested)

    applied = env._set_command_target.call_args.args[0]
    arm_indices = np.r_[0:6, 7:13]
    np.testing.assert_allclose(
        applied[arm_indices],
        np.sign(requested[arm_indices]) * 0.05,
    )


def test_openpi_direct_action_preserves_bounded_absolute_target() -> None:
    config = PiperRobotConfig(
        command_publish_hz=0.0,
        feedback_step_clamp_enabled=False,
    )
    env = _controller(config)
    requested = np.full(14, 0.2, dtype=np.float32)
    requested[[2, 9]] = -0.2
    env._get_joint_position = Mock(return_value=np.zeros(14, dtype=np.float32))
    env._set_command_target = Mock()

    env._apply_low_level_action(requested)

    applied = env._set_command_target.call_args.args[0]
    arm_indices = np.r_[0:6, 7:13]
    np.testing.assert_allclose(applied[arm_indices], requested[arm_indices])


def test_direct_command_applies_ema_and_velocity_limit_without_filtering_grippers() -> (
    None
):
    config = PiperRobotConfig(
        step_frequency=15.0,
        command_publish_hz=0.0,
        action_smoothing_alpha=0.4,
        max_arm_velocity_rad_s=0.4,
        feedback_step_clamp_enabled=False,
    )
    env = _controller(config)
    env._command_lock = __import__("threading").Lock()
    env._last_smoothed_action = np.full(14, 0.5, dtype=np.float32)
    env._publish_action = Mock()

    target = np.ones(14, dtype=np.float32)
    env._set_command_target(
        target,
        current_reference=np.zeros(14, dtype=np.float32),
        smooth=True,
    )

    arm_indices = np.r_[0:6, 7:13]
    expected_delta = config.max_arm_velocity_rad_s / config.step_frequency
    np.testing.assert_allclose(
        env._publish_action.call_args.args[0][arm_indices],
        0.5 + expected_delta,
        atol=1e-6,
    )
    np.testing.assert_allclose(env._publish_action.call_args.args[0][[6, 13]], 1.0)


def test_action_chunk_boundary_blends_only_arm_targets() -> None:
    env = _controller(
        PiperRobotConfig(action_chunk_size=4, action_chunk_boundary_blend_steps=2)
    )
    for value in range(4):
        action = np.full(14, value, dtype=np.float32)
        np.testing.assert_array_equal(env._blend_action_chunk_boundary(action), action)

    new_chunk_first = np.full(14, 12.0, dtype=np.float32)
    blended = env._blend_action_chunk_boundary(new_chunk_first)
    arm_indices = np.r_[0:6, 7:13]
    np.testing.assert_allclose(blended[arm_indices], 7.5)
    np.testing.assert_allclose(blended[[6, 13]], 12.0)

    new_chunk_second = np.full(14, 13.0, dtype=np.float32)
    np.testing.assert_array_equal(
        env._blend_action_chunk_boundary(new_chunk_second), new_chunk_second
    )


def test_step_decodes_only_the_returned_observation() -> None:
    config = PiperRobotConfig(step_frequency=1000.0)
    env = _controller(config)
    env.config.is_dummy = False
    env._num_steps = 0
    current = np.zeros(14, dtype=np.float32)
    observation = {
        "state": {"joint_position": current.copy()},
        "frames": {},
    }
    env._get_joint_position = Mock(return_value=current)
    env._get_observation = Mock(return_value=observation)
    env._set_command_target = Mock()

    returned, _, _, _, _ = env.step(current)

    assert returned is observation
    env._get_joint_position.assert_called_once_with()
    env._get_observation.assert_called_once_with()
    env._set_command_target.assert_called_once()


def test_action_chunk_decodes_cameras_only_for_its_final_action() -> None:
    config = PiperRobotConfig(step_frequency=1000.0)
    env = _controller(config)
    env.config.is_dummy = False
    env._num_steps = 0
    state = np.zeros(14, dtype=np.float32)
    previous = {"state": {"joint_position": state.copy()}, "frames": {"cam_high": 1}}
    final = {"state": {"joint_position": state.copy()}, "frames": {"cam_high": 2}}
    env._last_observation = previous
    env._get_joint_position = Mock(return_value=state)
    env._set_command_target = Mock()
    env._get_observation = Mock(return_value=final)

    env.begin_action_chunk(2)
    first, *_ = env.step(state)
    second, *_ = env.step(state)

    assert first["frames"] is previous["frames"]
    assert second is final
    env._get_observation.assert_called_once_with()


def test_final_chunk_action_starts_queued_rtc_before_camera_decode() -> None:
    config = PiperRobotConfig(step_frequency=1000.0, rtc_enabled=True)
    env = _controller(config)
    env.config.is_dummy = False
    env._num_steps = 0
    state = np.zeros(14, dtype=np.float32)
    previous = {"state": {"joint_position": state.copy()}, "frames": {}}
    events: list[str] = []
    env._last_observation = previous
    env._get_joint_position = Mock(return_value=state)
    env._set_command_target = Mock()
    env.start_rtc_tail = Mock(side_effect=lambda _tail: events.append("rtc"))
    env._get_observation = Mock(
        side_effect=lambda: events.append("observation") or previous
    )
    tail = np.ones((5, 14), dtype=np.float32)

    env.begin_action_chunk(1)
    env.queue_rtc_tail(tail)
    env.step(state)

    assert events == ["rtc", "observation"]
    np.testing.assert_array_equal(env.start_rtc_tail.call_args.args[0], tail)
    assert env._queued_rtc_tail is None


def test_stride_decodes_window_starts_but_starts_rtc_only_after_the_chunk() -> None:
    config = PiperRobotConfig(
        step_frequency=1000.0,
        rtc_enabled=True,
        action_chunk_size=10,
        rlt_subsample_stride=2,
    )
    env = _controller(config)
    env.config.is_dummy = False
    env._num_steps = 0
    state = np.zeros(14, dtype=np.float32)
    events: list[str] = []
    env._last_observation = {
        "state": {"joint_position": state.copy()},
        "frames": {"id": 0},
    }
    env._get_joint_position = Mock(return_value=state)
    env._set_command_target = Mock()
    env.start_rtc_tail = Mock(side_effect=lambda _tail: events.append("rtc"))

    def _decode():
        events.append("observation")
        decoded = events.count("observation")
        return {"state": {"joint_position": state.copy()}, "frames": {"id": decoded}}

    env._get_observation = Mock(side_effect=_decode)

    env.begin_action_chunk(10)
    env.queue_rtc_tail(np.ones((5, 14), dtype=np.float32))
    frames = [env.step(state)[0]["frames"]["id"] for _ in range(10)]

    # Fresh frames after actions 1, 3, 5 and 7 start the windows at 2, 4, 6
    # and 8; the one after action 9 starts the next chunk.
    assert frames == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5]
    # The RTC suffix must not run alongside the rest of the chunk.
    assert events == ["observation"] * 4 + ["rtc", "observation"]


def test_external_control_decodes_stride_observations() -> None:
    config = PiperRobotConfig(action_chunk_size=4, rlt_subsample_stride=2)
    env = _controller(config)
    env.config.is_dummy = False
    env._num_steps = 0
    state = np.zeros(14, dtype=np.float32)
    env._last_observation = {"state": {"joint_position": state.copy()}, "frames": {}}
    env._get_joint_position = Mock(return_value=state)
    env._get_observation = Mock(
        return_value={"state": {"joint_position": state.copy()}, "frames": {}}
    )

    env.begin_action_chunk(4)
    for _ in range(4):
        env.step_external_control()

    assert env._get_observation.call_count == 2


def test_main_and_rtc_actions_share_one_fixed_rate_scheduler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _controller(PiperRobotConfig(step_frequency=20.0))
    clock = [10.0]
    dispatches = []
    monkeypatch.setattr(
        "rlinf.envs.realworld.piper.piper_env.time.monotonic",
        lambda: clock[0],
    )
    monkeypatch.setattr(
        "rlinf.envs.realworld.piper.piper_env.time.sleep",
        lambda delay: clock.__setitem__(0, clock[0] + delay),
    )
    env._apply_low_level_action = Mock(
        side_effect=lambda _action: (
            dispatches.append(clock[0]) or np.zeros(14, dtype=np.float32)
        )
    )

    env._apply_scheduled_policy_action(np.zeros(14, dtype=np.float32))
    env._apply_scheduled_policy_action(np.ones(14, dtype=np.float32))

    np.testing.assert_allclose(dispatches, [10.0, 10.05], atol=1e-9)


def test_rtc_alignment_shifts_plan_and_blends_only_arm_joints() -> None:
    config = PiperRobotConfig(
        rtc_enabled=True,
        rtc_max_bridge_steps=5,
        rtc_boundary_blend_steps=3,
    )
    env = _controller(config)
    env._rtc_executed_steps = 2
    env._rtc_last_action = np.full(14, -1.0, dtype=np.float32)

    actions = np.arange(20 * 14, dtype=np.float32).reshape(20, 14)
    reference = np.arange(50 * 14, dtype=np.float32).reshape(50, 14)
    aligned, offset = env.align_rtc_chunk(actions, reference)

    assert offset == 2
    assert aligned.shape == (20, 14)
    # Grippers are time-aligned but never interpolated.
    np.testing.assert_array_equal(aligned[:3, [6, 13]], actions[2:5, [6, 13]])
    # The arm boundary starts between the bridge command and shifted plan.
    assert np.all(aligned[0, np.r_[0:6, 7:13]] < actions[2, np.r_[0:6, 7:13]])
    # The shifted actor is exhausted at source index 19, then VLA future
    # actions fill the end of the fixed-size C=20 transition.
    np.testing.assert_array_equal(aligned[-2:], reference[20:22])


def test_rtc_tail_does_not_replay_suffix_used_by_alignment() -> None:
    reference = np.arange(50 * 14, dtype=np.float32).reshape(1, 50, 14)

    tail = slice_rtc_reference_tail(reference, chunk_len=20, plan_offset=5)

    assert tail.shape == (1, 25, 14)
    np.testing.assert_array_equal(tail[:, 0], reference[:, 25])


def test_interrupted_rtc_target_is_not_counted_as_completed_step() -> None:
    class InterruptOnWait:
        def __init__(self) -> None:
            self.stopped = False

        def set(self) -> None:
            self.stopped = True

        def clear(self) -> None:
            self.stopped = False

        def is_set(self) -> bool:
            return self.stopped

        def wait(self, _timeout: float) -> bool:
            self.stopped = True
            return True

    config = PiperRobotConfig(
        rtc_enabled=True,
        command_publish_hz=200.0,
        command_interpolation_duration_s=1.0 / 15.0,
    )
    env = _controller(config)
    env._rtc_stop = InterruptOnWait()
    env._rospy = Mock()
    env._rospy.is_shutdown.return_value = False
    env._command_lock = __import__("threading").Lock()
    env._command_start = np.zeros(14, dtype=np.float32)
    env._command_target = np.ones(14, dtype=np.float32)
    env._command_start_time = __import__("time").monotonic()
    env._next_policy_action_time = __import__("time").monotonic() + 1.0
    env._apply_low_level_action = Mock()

    env.start_rtc_tail(np.ones((5, 14), dtype=np.float32))
    env._rtc_thread.join(timeout=1.0)

    assert env._rtc_executed_steps == 0
    assert env._rtc_last_action is not None
