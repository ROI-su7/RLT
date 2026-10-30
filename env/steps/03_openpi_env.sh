#!/usr/bin/env bash
# openpi/.venv: pi0.5 fine-tuning, norm stats and LeRobot conversion (uv.lock pinned).
set -euo pipefail
RLT_ROOT="${RLT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PATH="$HOME/.local/bin:$PATH"
cd "$RLT_ROOT/openpi"
GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen
.venv/bin/python -c "import openpi, jax; print('[RLT] openpi env ok, jax', jax.__version__)"
