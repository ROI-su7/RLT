#!/usr/bin/env bash
# SO101 extras (no ROS, no sudo). Run after 03/04: `uv sync` in step 03 would remove them again.
#  - so101/.venv: tiny env for the read-only probe (pyserial + Feetech SDK).
#  - openpi/.venv and openpi/RLinf/.venv-rlt: add the Feetech SDK and pyserial that LeRobot's
#    SO101 buses need (recording in openpi/.venv, RLT Stage 2 in .venv-rlt). Nothing else changes,
#    so the Piper/Cobot Magic environments stay as pinned.
set -euo pipefail
RLT_ROOT="${RLT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
cd "$RLT_ROOT/so101"
uv venv -q --python 3.11 .venv
uv pip install -q --python .venv/bin/python -r requirements.txt
for venv in "$RLT_ROOT/openpi/.venv" "$RLT_ROOT/openpi/RLinf/.venv-rlt"; do
  if [[ -x "$venv/bin/python" ]]; then
    uv pip install -q --python "$venv/bin/python" "pyserial==3.5" "feetech-servo-sdk==1.0.0"
    echo "[RLT] SO101 bus deps added to $venv"
  else
    echo "[RLT] $venv missing; run steps 03/04 first, then this step again"
  fi
done
id -nG | grep -qw dialout || echo "[RLT] serial access: sudo usermod -aG dialout \$USER (then re-login)"
id -nG | grep -qw input || echo "[RLT] evdev keyboard for RLT keys: sudo usermod -aG input \$USER, or use RLINF_SO101_KEYBOARD=file"
echo "[RLT] so101 env ok"
