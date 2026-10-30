"""Check grasp timing through the real launcher without ROS or robot motion."""

import os
import shlex
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("profile", ["takeover", "followers"])
def test_client_grasp_timing(profile):
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["bash", "examples/recap_piper/run_client.sh", profile, "--dry-run"],
        cwd=root,
        env={**os.environ, "RECAP_TASK": "test wire task"},
        capture_output=True,
        text=True,
        check=True,
    )
    command = shlex.split(result.stdout)
    assert command[command.index("-m") + 1] == "rlinf.recap_piper.client"
    for flag, value in {
        "--action-horizon": "10",
        "--rtc-prefix-len": "4",
        "--rtc-boundary-blend-steps": "5",
        "--action-smoothing-alpha": "0.5",
        "--max-arm-velocity-rad-s": "0.4" if profile == "takeover" else "0.6",
        "--port": "6016",
    }.items():
        assert command[command.index(flag) + 1] == value
    assert "--rtc-enabled" in command
