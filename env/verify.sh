#!/usr/bin/env bash
# Check an installed RLT tree: environments, ROS workspaces and the unit tests.
#
#   bash env/verify.sh            # environments + unit tests
#   bash env/verify.sh --no-ros   # skip the catkin workspace checks
set -uo pipefail
RLT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RLINF="$RLT_ROOT/openpi/RLinf"
CHECK_ROS=1
[[ "${1:-}" == "--no-ros" ]] && CHECK_ROS=0
FAIL=0
ok() { echo "  [ok]   $*"; }
bad() { echo "  [FAIL] $*"; FAIL=1; }

echo "== Python environments"
if "$RLT_ROOT/openpi/.venv/bin/python" -c "import openpi, jax, lerobot" 2>/dev/null; then
  ok "openpi/.venv (openpi, jax, lerobot)"
else
  bad "openpi/.venv: run env/steps/03_openpi_env.sh"
fi
if env -u PYTHONPATH "$RLINF/.venv-rlt/bin/python" - <<'PY'
import jax, torch, ray, hydra, openpi, rlinf
from rlinf.models.embodiment.modules.rlt_token_transformer import RLTTokenTransformer
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import RLTACFSDPPolicy
assert torch.cuda.is_available(), "torch cannot see a GPU"
assert jax.devices()[0].platform == "gpu", "jax cannot see a GPU"
PY
then
  ok "openpi/RLinf/.venv-rlt (torch+jax on GPU, RLinf RLT modules)"
else
  bad "openpi/RLinf/.venv-rlt: run env/steps/04_rlt_env.sh"
fi

if [[ $CHECK_ROS == 1 ]]; then
  echo "== ROS workspaces"
  for ws in cobot_magic/Piper_ros_private-ros-noetic cobot_magic/camera_ws; do
    if [[ -f "$RLT_ROOT/$ws/devel/setup.bash" ]]; then ok "$ws built"; else bad "$ws not built: run env/steps/02_ros.sh"; fi
  done
  if /usr/bin/python3 -c "import piper_sdk" 2>/dev/null; then ok "system python piper_sdk"; else bad "system python lacks piper_sdk"; fi
  # Four-CAN collection/inference/takeover mux and CAN guard (system Python, like the ROS nodes).
  if (cd "$RLT_ROOT/cobot_magic/Piper_ros_private-ros-noetic/src/piper" &&
      /usr/bin/python3 -m unittest discover -s test -p "test_*.py" >/dev/null 2>&1); then
    ok "Piper ROS takeover/mux tests"
  else
    bad "Piper ROS takeover/mux tests (run: cd cobot_magic/Piper_ros_private-ros-noetic/src/piper && python3 -m unittest discover -s test)"
  fi
fi

echo "== Unit tests (RLT / Piper)"
cd "$RLINF"
TESTS=$(ls tests/unit_tests/test_*rlt*.py tests/unit_tests/test_*piper*.py tests/unit_tests/test_*replay*.py | sort -u)
if env -u PYTHONPATH .venv-rlt/bin/python -m pytest -q -p no:cacheprovider $TESTS; then
  ok "RLinf RLT unit tests"
else
  bad "RLinf RLT unit tests"
fi

[[ $FAIL == 0 ]] && echo "All checks passed." || { echo "Some checks failed."; exit 1; }
