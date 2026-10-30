#!/usr/bin/env bash
# SO101 tools only: a small venv with pyserial + Feetech SDK (no ROS, no sudo, no GPU).
set -euo pipefail
RLT_ROOT="${RLT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$RLT_ROOT/so101"
uv venv -q --python 3.11 .venv
uv pip install -q --python .venv/bin/python -r requirements.txt
id -nG | grep -qw dialout || echo "[RLT] add yourself to dialout: sudo usermod -aG dialout \$USER (then re-login)"
echo "[RLT] so101 env ok"
