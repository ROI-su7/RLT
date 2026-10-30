"""Build a LeRobot dual-SO101 robot from ``dual_arm.yaml`` (ports by USB serial, cameras by device path/index)."""
import os
import pathlib
import re

import yaml

ROOT = pathlib.Path(__file__).resolve().parent
JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def _port(by_id: str) -> str:
    path = by_id if by_id.startswith("/") else f"/dev/serial/by-id/{by_id}"
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: plug the arm in or fix port_by_id in dual_arm.yaml")
    return path


def _camera_index(device) -> int:
    """Accept an int, '/dev/videoN' or a stable '/dev/v4l/by-id|by-path/...' link."""
    if isinstance(device, int):
        return device
    real = os.path.realpath(str(device))
    m = re.fullmatch(r"/dev/video(\d+)", real)
    if not m:
        raise ValueError(f"camera device {device!r} does not resolve to /dev/videoN (got {real})")
    return int(m.group(1))


def load(path=ROOT / "dual_arm.yaml") -> dict:
    return yaml.safe_load(open(path))


def make_robot_config(path=ROOT / "dual_arm.yaml", mock: bool = False, max_relative_target: float | None = 15.0):
    """Return a lerobot ``So101RobotConfig`` with arms 'left' and 'right' (state/action = left(6) + right(6))."""
    from lerobot.common.robot_devices.cameras.configs import OpenCVCameraConfig
    from lerobot.common.robot_devices.motors.configs import FeetechMotorsBusConfig
    from lerobot.common.robot_devices.robots.configs import So101RobotConfig

    cfg = load(path)
    calibration_dir = ROOT / "calibration"
    if mock:  # LeRobot's own hardware mocks live in its (vendored) tests package
        import sys
        import tempfile

        sys.path.insert(0, str(ROOT.parent / "openpi" / "third_party" / "lerobot"))
        from tests.utils import mock_calibration_dir

        calibration_dir = pathlib.Path(tempfile.mkdtemp(prefix="so101_mock_calib_"))
        mock_calibration_dir(calibration_dir)
    motors = {n: [i, "sts3215"] for i, n in enumerate(JOINTS, start=1)}

    def buses(role):
        return {
            side: FeetechMotorsBusConfig(port="mock" if mock else _port(cfg["arms"][side][role]["port_by_id"]), motors=motors)
            for side in ("left", "right")
        }

    cameras = {}
    for name, spec in (cfg.get("cameras") or {}).items():
        if not isinstance(spec, dict):
            continue
        cameras[name] = OpenCVCameraConfig(
            camera_index=0 if mock else _camera_index(spec["device"]),
            fps=spec.get("fps", 30), width=spec.get("width", 640), height=spec.get("height", 480),
        )
    return So101RobotConfig(
        calibration_dir=str(calibration_dir),
        max_relative_target=max_relative_target,  # safety cap per step (degrees)
        leader_arms=buses("leader"),
        follower_arms=buses("follower"),
        cameras=cameras,
        mock=mock,
    )
