#!/usr/bin/env bash
# openpi/RLinf/.venv-rlt: native JAX pi0.5 server + Stage 1 RL token + Stage 2 RLT.
set -euo pipefail
RLT_ROOT="${RLT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PATH="$HOME/.local/bin:$PATH"
RLINF="$RLT_ROOT/openpi/RLinf"
cd "$RLINF"
if [[ ! -x .venv-rlt/bin/python ]]; then
  uv venv --python 3.11 --seed .venv-rlt
fi
# Exact versions of the reference machine; --no-deps reproduces them as-is.
uv pip install --python .venv-rlt/bin/python --no-deps -r "$RLT_ROOT/env/requirements-rlt.txt"
uv pip install --python .venv-rlt/bin/python --no-deps -e "$RLT_ROOT/openpi/packages/openpi-client"
uv pip install --python .venv-rlt/bin/python --no-deps -e "$RLT_ROOT/openpi"
uv pip install --python .venv-rlt/bin/python --no-deps -e "$RLINF"
env -u PYTHONPATH .venv-rlt/bin/python - <<'PY'
import jax, torch, ray, openpi, rlinf
print("[RLT] rlt env ok: torch", torch.__version__, "cuda", torch.cuda.is_available(),
      "| jax", jax.__version__, jax.devices()[0].platform, "| ray", ray.__version__)
PY
