#!/usr/bin/env bash
# Run on the GPU cluster host (rank 0). Starts Ray only; it does not train yet.
set -euo pipefail

REPO_DIR="${RLINF_GPU_REPO_DIR:-$HOME/openpi/RLinf}"
HEAD_IP="${RLINF_GPU_HEAD_IP:-192.168.85.129}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0,1}"

cd "$REPO_DIR"
source .venv-rlt/bin/activate
export PYTHONPATH="$REPO_DIR"
export RLINF_GPU_REPO_DIR="$REPO_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export RLINF_NODE_RANK=0

if ! hostname -I | tr ' ' '\n' | grep -Fxq "$HEAD_IP"; then
  echo "GPU head IP ${HEAD_IP} is not assigned to this host." >&2
  exit 1
fi

python - <<'PY'
import ray, torch
print(f"Ray={ray.__version__}, torch={torch.__version__}, CUDA GPUs={torch.cuda.device_count()}")
if torch.cuda.device_count() < 2:
    raise SystemExit("Dual Stage-2 requires CUDA_VISIBLE_DEVICES to expose at least two GPUs.")
PY

if ray status --address="$HEAD_IP:6379" >/dev/null 2>&1; then
  echo "Ray head is already reachable at ${HEAD_IP}:6379; leaving it unchanged."
else
  ray start --head --port=6379 --node-ip-address="$HEAD_IP" --disable-usage-stats
fi

ray status --address="$HEAD_IP:6379"
echo "GPU Ray head is ready. Now join the Piper host as node rank 1."
