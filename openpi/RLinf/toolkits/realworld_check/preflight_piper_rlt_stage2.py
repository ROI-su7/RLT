#!/usr/bin/env python3
"""Read-only preflight for the single-machine Piper RLT Stage-2 deployment.

This script sends no robot commands. It validates the prerequisites that must
be true before launching the Stage-2 trainer: OpenPI fork replacement files,
the completed Stage-1 checkpoint, ROS master/topics, the physical keyboard
device used for i/b/t/c/x/Space/h, and the existing Ray head.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

REQUIRED_TOPICS = (
    "/puppet/joint_left",
    "/puppet/joint_right",
    "/camera_f/color/image_raw",
    "/camera_l/color/image_raw",
    "/camera_r/color/image_raw",
)
COMMAND_TOPICS = ("/master/joint_left", "/master/joint_right")
MUX_COMMAND_TOPICS = ("/teleop/command/joint_left", "/teleop/command/joint_right")
MUX_POLICY_TOPICS = ("/teleop/policy/joint_left", "/teleop/policy/joint_right")
RLINF_POLICY_EXECUTOR_PREFIX = "/rlinf_piper_control_executor_"
MUX_STATUS_TOPICS = (
    "/teleop/mode",
    "/teleop/takeover_phase",
    "/teleop/inference_active",
    "/teleop/hold_active",
    "/teleop/home_active",
    "/master/joint_left",
    "/master/joint_right",
)
MODE2_TOPICS = (
    "/rlinf/mode2/leader_left",
    "/rlinf/mode2/leader_right",
)
MODE2_SERVICES = (
    "/rlinf/mode2/follower_teleop",
    "/rlinf/mode2/leader_drag_left",
    "/rlinf/mode2/leader_drag_right",
)


def _topic_nodes(entries: list, topic: str) -> list[str]:
    """Return ROS node names registered for one system-state topic."""
    return next((list(nodes) for name, nodes in entries if name == topic), [])


def _node_is_alive(rosnode_module, node: str) -> bool:
    """Ping a ROS node while treating stale XMLRPC registrations as dead."""
    try:
        return bool(
            rosnode_module.rosnode_ping(
                node, max_count=1, verbose=False, skip_cache=True
            )
        )
    except (ConnectionError, OSError):
        # Noetic does not consistently convert a refused XMLRPC connection to
        # False on every supported Python version.
        return False


def _validate_native_jax_metadata(
    metadata: dict, *, require_compact: bool = False
) -> None:
    if not metadata.get("rlt_native_jax_prefix", False):
        raise RuntimeError("restart scripts/serve_policy.py with --return-rlt-prefix")
    if int(metadata.get("rlt_prefix_seq_len", -1)) != 768:
        raise RuntimeError(f"expected rlt_prefix_seq_len=768, got {metadata}")
    # Stride-2 windows take z_rl for intermediate observations from this
    # server's Stage-1 encoder.
    if not metadata.get("rlt_prefix_only", False):
        raise RuntimeError(
            "restart scripts/serve_policy.py with --rlt-token-checkpoint "
            "(RLT stride windows need prefix-only z_rl)"
        )
    if require_compact and not metadata.get("rlt_action_z_rl", False):
        raise RuntimeError(
            "start toolkits/realworld_check/serve_piper_rlt_compact.py for compact action+z_rl and action-priority scheduling"
        )


def _validate_command_subscribers(
    subscribers: list, rosnode_module, command_topics=COMMAND_TOPICS
) -> dict[str, str]:
    """Require exactly one live hardware driver on each command topic."""
    validated = {}
    for topic in command_topics:
        registered = _topic_nodes(subscribers, topic)
        live = [node for node in registered if _node_is_alive(rosnode_module, node)]
        if len(live) != 1:
            detail = ", ".join(live) if live else "none"
            raise RuntimeError(
                f"expected exactly one live Piper driver subscriber on {topic}, "
                f"found {len(live)}: {detail}"
            )
        validated[topic] = live[0]
    return validated


def _validate_mux_ownership(
    publishers: list, subscribers: list, rosnode_module
) -> None:
    """Require the original mux as sole command owner and no old policy client."""
    for topic in MUX_COMMAND_TOPICS + MUX_STATUS_TOPICS:
        live = [
            node
            for node in _topic_nodes(publishers, topic)
            if _node_is_alive(rosnode_module, node)
        ]
        if live != ["/piper_four_state_mux"]:
            raise RuntimeError(
                f"{topic} must have sole live publisher /piper_four_state_mux; got {live}"
            )
    for topic in MUX_POLICY_TOPICS + (
        "/teleop/mode_request",
        "/teleop/home_request",
        "/teleop/hold_request",
    ):
        live = [
            node
            for node in _topic_nodes(subscribers, topic)
            if _node_is_alive(rosnode_module, node)
        ]
        if live != ["/piper_four_state_mux"]:
            raise RuntimeError(
                f"{topic} must be consumed by the original four-CAN mux; got {live}"
            )
        live_publishers = [
            node
            for node in _topic_nodes(publishers, topic)
            if _node_is_alive(rosnode_module, node)
        ]
        if topic in MUX_POLICY_TOPICS:
            unexpected = [
                node
                for node in live_publishers
                if not node.startswith(RLINF_POLICY_EXECUTOR_PREFIX)
            ]
            executors = [
                node
                for node in live_publishers
                if node.startswith(RLINF_POLICY_EXECUTOR_PREFIX)
            ]
            if unexpected or len(executors) > 1:
                raise RuntimeError(
                    f"Stop old policy clients on {topic}: {unexpected or executors}"
                )
        elif live_publishers:
            raise RuntimeError(
                f"Stop standalone keyboards on {topic}: {live_publishers}"
            )


def check(name: str, fn) -> bool:
    """Run one read-only check and print a compact PASS/FAIL result."""
    try:
        detail = fn()
    except Exception as exc:  # Each failure should be actionable in one run.
        print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
        return False
    print(f"PASS  {name}: {detail}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ray-address",
        default="192.168.209.107:6379",
        help="Ray head address (default: %(default)s)",
    )
    parser.add_argument(
        "--skip-ray",
        action="store_true",
        help="Skip Ray validation (used on the robot immediately before it joins the remote head).",
    )
    args = parser.parse_args()
    results: list[bool] = []

    def transformers_check() -> str:
        import transformers
        from transformers.models.siglip import check as siglip_check

        if transformers.__version__ != "4.53.2":
            raise RuntimeError(
                f"expected transformers 4.53.2, got {transformers.__version__}"
            )
        if not siglip_check.check_whether_transformers_replace_is_installed_correctly():
            raise RuntimeError("OpenPI transformers_replace check returned false")
        return "transformers=4.53.2, replacement_ok=True"

    results.append(check("OpenPI Transformers replacement", transformers_check))

    def inference_contract_check() -> str:
        """Require the Stage-2 native-JAX dual-output websocket contract."""
        from openpi_client import websocket_client_policy

        host = os.environ.get("RLINF_JAX_REF_HOST", "192.168.209.107")
        port = int(os.environ.get("RLINF_JAX_REF_PORT", "6006"))
        client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        metadata = client.get_server_metadata()
        _validate_native_jax_metadata(
            metadata,
            require_compact=(
                os.environ.get("RLINF_PIPER_TAKEOVER_BACKEND", "four_can") == "four_can"
                and int(os.environ.get("RLINF_PIPER_CONTROL_MODE", "1")) != 2
            ),
        )
        return (
            f"{host}:{port}, compact={metadata.get('rlt_action_z_rl', False)}, prefix_seq_len=768, "
            f"stride z_rl encoder={metadata.get('rlt_encoder_fingerprint')}"
        )

    results.append(check("OpenPI/JAX Stage-2 interface", inference_contract_check))

    def keyboard_check() -> str:
        from rlinf.envs.realworld.common.keyboard.keyboard_listener import (
            KeyboardListener,
        )

        listener = KeyboardListener()
        return f"{listener.device.path} ({listener.device.name})"

    results.append(check("Keyboard/foot-pedal evdev device", keyboard_check))

    def ros_check() -> str:
        import rosgraph
        import rosnode
        import rosservice

        master = os.environ.get("ROS_MASTER_URI")
        if master != "http://192.168.209.107:11311":
            raise RuntimeError(
                f"expected ROS_MASTER_URI=http://192.168.209.107:11311, got {master!r}"
            )
        ros_master = rosgraph.Master("/rlinf_piper_preflight")
        publishers, subscribers, _ = ros_master.getSystemState()
        control_mode = int(os.environ.get("RLINF_PIPER_CONTROL_MODE", "1"))
        if control_mode not in {0, 1, 2}:
            raise RuntimeError(
                f"RLINF_PIPER_CONTROL_MODE must be 0, 1, or 2; got {control_mode}"
            )
        required_topics = REQUIRED_TOPICS + (MODE2_TOPICS if control_mode == 2 else ())
        four_can = (
            control_mode != 2
            and os.environ.get("RLINF_PIPER_TAKEOVER_BACKEND", "four_can") == "four_can"
        )
        if four_can:
            required_topics += MUX_STATUS_TOPICS
            _validate_mux_ownership(publishers, subscribers, rosnode)
        missing_publishers = [
            topic for topic in required_topics if not _topic_nodes(publishers, topic)
        ]
        if missing_publishers:
            raise RuntimeError(
                f"topics have no registered publisher: {', '.join(missing_publishers)}"
            )
        command_subscribers = _validate_command_subscribers(
            subscribers, rosnode, MUX_COMMAND_TOPICS if four_can else COMMAND_TOPICS
        )
        registered_env_publishers = sorted(
            {
                node
                for topic in COMMAND_TOPICS
                for node in _topic_nodes(publishers, topic)
                if node.startswith("/rlinf_piper_wire_env_")
            }
        )
        live_env_publishers = [
            node for node in registered_env_publishers if _node_is_alive(rosnode, node)
        ]
        if live_env_publishers:
            raise RuntimeError(
                "old RLinf EnvWorker publishers are still alive; stop old Ray "
                "workers before joining a new cluster: "
                + ", ".join(live_env_publishers)
            )
        if control_mode == 2:
            services = set(rosservice.get_service_list())
            missing_services = [
                service for service in MODE2_SERVICES if service not in services
            ]
            if missing_services:
                raise RuntimeError(
                    "mode-2 services are missing: " + ", ".join(missing_services)
                )
        for topic in required_topics:
            try:
                subprocess.run(
                    ["rostopic", "echo", "-n", "1", topic],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5.0,
                    check=True,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError(
                    f"timeout waiting for a fresh message on {topic}"
                ) from exc
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(
                    f"failed reading {topic}: {exc.stderr.strip()}"
                ) from exc
        stale_count = len(registered_env_publishers) - len(live_env_publishers)
        return (
            f"master={master}, mode={control_mode}, "
            f"{len(required_topics)} live sensor topics, "
            f"{len(command_subscribers)} unique Piper command subscribers, "
            f"ignored {stale_count} dead EnvWorker registrations"
        )

    results.append(check("ROS master and Piper/camera topics", ros_check))

    def ray_check() -> str:
        result = subprocess.run(
            ["ray", "status", "--address", args.ray_address],
            text=True,
            capture_output=True,
            check=True,
        )
        if "1.0/1.0 GPU" not in result.stdout and "0.0/1.0 GPU" not in result.stdout:
            raise RuntimeError("Ray status does not report one GPU")
        return args.ray_address

    if not args.skip_ray:
        results.append(check("Ray head and GPU", ray_check))

    if all(results):
        print("\nPiper RLT Stage-2 preflight PASSED. Safe to launch the trainer.")
        return 0
    print("\nPiper RLT Stage-2 preflight FAILED. Do not launch the trainer.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
