import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[2]
    / "toolkits"
    / "realworld_check"
    / "preflight_piper_rlt_stage2.py"
)
SPEC = importlib.util.spec_from_file_location("piper_preflight", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
PREFLIGHT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREFLIGHT)


def test_topic_nodes_returns_registered_nodes() -> None:
    entries = [
        ["/camera_f/color/image_raw", ["/camera"]],
        ["/master/joint_left", ["/piper_left"]],
    ]

    assert PREFLIGHT._topic_nodes(entries, "/master/joint_left") == ["/piper_left"]
    assert PREFLIGHT._topic_nodes(entries, "/missing") == []


def test_node_is_alive_treats_refused_stale_registration_as_dead() -> None:
    class RefusedRosnode:
        @staticmethod
        def rosnode_ping(*args, **kwargs):
            raise ConnectionRefusedError(111, "Connection refused")

    assert not PREFLIGHT._node_is_alive(RefusedRosnode, "/stale_env")


def test_native_jax_stage2_metadata_is_accepted() -> None:
    PREFLIGHT._validate_native_jax_metadata(
        {
            "rlt_native_jax_prefix": True,
            "rlt_prefix_seq_len": 768,
            "rlt_prefix_only": True,
        }
    )


def test_four_can_requires_compact_service_metadata():
    metadata = {
        "rlt_native_jax_prefix": True,
        "rlt_prefix_seq_len": 768,
        "rlt_prefix_only": True,
    }
    with pytest.raises(RuntimeError, match="serve_piper_rlt_compact"):
        PREFLIGHT._validate_native_jax_metadata(metadata, require_compact=True)
    metadata["rlt_action_z_rl"] = True
    PREFLIGHT._validate_native_jax_metadata(metadata, require_compact=True)


def test_old_action_only_server_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="return-rlt-prefix"):
        PREFLIGHT._validate_native_jax_metadata({})


def test_server_without_stride_z_rl_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="rlt-token-checkpoint"):
        PREFLIGHT._validate_native_jax_metadata(
            {"rlt_native_jax_prefix": True, "rlt_prefix_seq_len": 768}
        )


class _AliveRosnode:
    @staticmethod
    def rosnode_ping(node, **kwargs):
        del kwargs
        return not node.startswith("/stale")


def test_command_topics_require_one_live_driver_each() -> None:
    subscribers = [
        ["/master/joint_left", ["/stale_left", "/piper_left"]],
        ["/master/joint_right", ["/piper_right"]],
    ]

    assert PREFLIGHT._validate_command_subscribers(subscribers, _AliveRosnode) == {
        "/master/joint_left": "/piper_left",
        "/master/joint_right": "/piper_right",
    }


def test_duplicate_live_piper_drivers_are_rejected() -> None:
    subscribers = [
        ["/master/joint_left", ["/piper_left_a", "/piper_left_b"]],
        ["/master/joint_right", ["/piper_right"]],
    ]

    with pytest.raises(RuntimeError, match="exactly one live Piper driver"):
        PREFLIGHT._validate_command_subscribers(subscribers, _AliveRosnode)


def _mux_state():
    publications = [
        [topic, ["/piper_four_state_mux"]]
        for topic in PREFLIGHT.MUX_COMMAND_TOPICS + PREFLIGHT.MUX_STATUS_TOPICS
    ]
    subscriptions = [
        [topic, ["/piper_four_state_mux"]]
        for topic in PREFLIGHT.MUX_POLICY_TOPICS
        + ("/teleop/mode_request", "/teleop/home_request", "/teleop/hold_request")
    ]
    subscriptions += [
        [topic, ["/piper_driver"]] for topic in PREFLIGHT.MUX_COMMAND_TOPICS
    ]
    return publications, subscriptions


def test_four_can_mux_ownership_passes_without_policy_or_keyboard():
    publications, subscriptions = _mux_state()
    PREFLIGHT._validate_mux_ownership(publications, subscriptions, _AliveRosnode)
    PREFLIGHT._validate_command_subscribers(
        subscriptions, _AliveRosnode, PREFLIGHT.MUX_COMMAND_TOPICS
    )


def test_four_can_rejects_competing_openpi_client():
    publications, subscriptions = _mux_state()
    publications.append(["/teleop/policy/joint_left", ["/openpi_piper_runtime"]])
    with pytest.raises(RuntimeError, match="Stop old policy clients"):
        PREFLIGHT._validate_mux_ownership(publications, subscriptions, _AliveRosnode)


def test_four_can_allows_the_current_rlinf_policy_executor():
    publications, subscriptions = _mux_state()
    publications.append(
        [
            "/teleop/policy/joint_left",
            ["/rlinf_piper_control_executor_140616_1790664659164"],
        ]
    )
    publications.append(
        [
            "/teleop/policy/joint_right",
            ["/rlinf_piper_control_executor_140616_1790664659164"],
        ]
    )

    PREFLIGHT._validate_mux_ownership(publications, subscriptions, _AliveRosnode)


def test_four_can_rejects_standalone_keyboard():
    publications, subscriptions = _mux_state()
    publications.append(["/teleop/mode_request", ["/piper_mode_keyboard"]])
    with pytest.raises(RuntimeError, match="standalone keyboards"):
        PREFLIGHT._validate_mux_ownership(publications, subscriptions, _AliveRosnode)
