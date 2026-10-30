"""Regression tests proving mode 2 is opt-in."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from rlinf.envs.realworld.piper import tasks

MODE2_ROS_DIR = (
    Path(__file__).parents[2] / "examples" / "embodiment" / "piper_mode2_ros"
)


@pytest.mark.parametrize("control_mode", [0, 1])
def test_existing_modes_never_construct_mode2_hardware(
    monkeypatch: pytest.MonkeyPatch,
    control_mode: int,
) -> None:
    base_env = SimpleNamespace(config=SimpleNamespace(is_dummy=False))
    wrapped_env = object()
    wrapper_factory = Mock(return_value=wrapped_env)
    mode2_factory = Mock(side_effect=AssertionError("mode 2 must remain opt-in"))
    monkeypatch.setattr(tasks, "PiperWireEnv", Mock(return_value=base_env))
    monkeypatch.setattr(tasks, "PiperRLTControlWrapper", wrapper_factory)
    monkeypatch.setattr(tasks, "RosMode2Hardware", mode2_factory)

    result = tasks.create_piper_wire_env(
        override_cfg={},
        worker_info=None,
        hardware_info=None,
        env_idx=0,
        env_cfg={
            "keyboard_reward_wrapper": "piper_rlt",
            "piper_control_mode": control_mode,
        },
    )

    assert result is wrapped_env
    mode2_factory.assert_not_called()
    assert wrapper_factory.call_args.kwargs["mode2_controller"] is None


def test_mode2_constructs_dedicated_hardware_only_when_selected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_env = SimpleNamespace(config=SimpleNamespace(is_dummy=False))
    wrapped_env = object()
    hardware = Mock()
    monkeypatch.setattr(tasks, "PiperWireEnv", Mock(return_value=base_env))
    monkeypatch.setattr(
        tasks,
        "PiperRLTControlWrapper",
        Mock(return_value=wrapped_env),
    )
    hardware_factory = Mock(return_value=hardware)
    monkeypatch.setattr(tasks, "RosMode2Hardware", hardware_factory)

    result = tasks.create_piper_wire_env(
        override_cfg={},
        worker_info=None,
        hardware_info=None,
        env_idx=0,
        env_cfg={
            "keyboard_reward_wrapper": "piper_rlt",
            "piper_control_mode": 2,
            "mode2_takeover": {"max_velocity_rad_s": 0.05},
        },
    )

    assert result is wrapped_env
    hardware_factory.assert_called_once()
    controller = tasks.PiperRLTControlWrapper.call_args.kwargs["mode2_controller"]
    assert controller.hardware is hardware
    assert controller.state_machine.config.max_velocity_rad_s == 0.05


def test_unknown_control_mode_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base_env = SimpleNamespace(config=SimpleNamespace(is_dummy=False))
    monkeypatch.setattr(tasks, "PiperWireEnv", Mock(return_value=base_env))

    with pytest.raises(ValueError, match="must be 0, 1, or 2"):
        tasks.create_piper_wire_env(
            override_cfg={},
            worker_info=None,
            hardware_info=None,
            env_idx=0,
            env_cfg={
                "keyboard_reward_wrapper": "piper_rlt",
                "piper_control_mode": 3,
            },
        )


def test_unified_ros_launch_adds_bridge_only_for_mode2() -> None:
    root = ET.parse(MODE2_ROS_DIR / "start_ms_piper.launch").getroot()
    groups = root.findall("group")
    assert len(groups) == 1
    assert groups[0].attrib["if"] == "$(eval int(arg('mode')) == 2)"
    bridge = groups[0].find("node")
    assert bridge is not None
    assert bridge.attrib["type"] == "piper_mode2_ros_driver.py"
    assert bridge.attrib["required"] == "true"


def test_candidate_ros_node_preserves_mode0_and_extends_mode1_to_mode2() -> None:
    source = (MODE2_ROS_DIR / "piper_start_ms_node.py").read_text()
    assert source.count("self.mode in (1, 2)") == 3
    assert source.count("self.mode == 0") == 2
    assert "self.mode not in (0, 1, 2)" in source


def test_compatibility_launch_delegates_to_unified_mode2() -> None:
    root = ET.parse(MODE2_ROS_DIR / "start_ms_piper_mode2.launch").getroot()
    include = root.find("include")
    assert include is not None
    args = {arg.attrib["name"]: arg.attrib["value"] for arg in include.findall("arg")}
    assert args["mode"] == "2"
    assert args["mode2_leaders_preisolated"] == "$(arg leaders_preisolated)"
