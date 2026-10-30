#!/usr/bin/env bash
# Single-machine dual-SO101 RLT Stage 2 (no ROS). Start serve_so101_rlt_compact.sh first.
#
#   export RLINF_STAGE1_RLT_TOKEN=<rlt_token.pt> RLINF_SO101_TASK="<prompt>"
#   bash examples/embodiment/run_so101_rlt_stage2.sh [hydra overrides...]
#
# Keys (evdev keyboard, or RLINF_SO101_KEYBOARD=file and `echo t >> ~/.so101_rlt_keys`):
#   I start/resume policy, B start collection, T take over / hand back (leader arms),
#   C success, X failure, Space hold, H home.
# RLINF_SO101_HARDWARE=sim runs the whole stack on simulated arms (pipeline check).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RLT_ROOT="${RLT_ROOT:-$(cd "$REPO_DIR/../.." && pwd)}"
NODE_IP="${GPU_NODE_IP:-$(hostname -I | awk '{print $1}')}"
: "${RLINF_STAGE1_RLT_TOKEN:?Set RLINF_STAGE1_RLT_TOKEN to the SO101 Stage-1 rlt_token.pt}"
: "${RLINF_SO101_TASK:?Set RLINF_SO101_TASK to the task prompt used when recording}"
test -s "$RLINF_STAGE1_RLT_TOKEN"
export RLINF_SO101_HARDWARE="${RLINF_SO101_HARDWARE:-lerobot}"

if [[ "$RLINF_SO101_HARDWARE" == lerobot ]]; then
  # Read-only check that all four buses are plugged in (ports by USB serial).
  "$REPO_DIR/.venv-rlt/bin/python" - "$RLT_ROOT/so101/dual_arm.yaml" <<'PY'
import os, sys, yaml
cfg = yaml.safe_load(open(sys.argv[1]))
missing = [f"{side}/{role}: {arm['port_by_id']}" for side, roles in cfg["arms"].items() for role, arm in roles.items()
           if not os.path.exists(f"/dev/serial/by-id/{arm['port_by_id']}")]
bad_cams = [name for name, cam in (cfg.get("cameras") or {}).items() if "FILL_ME" in str(cam.get("device", ""))]
if missing or bad_cams:
    sys.exit(f"SO101 preflight failed. Missing ports: {missing}. Unset cameras: {bad_cams}")
print("[SO101] four buses present, cameras configured")
PY
fi

unset PYTHONPATH
source "$REPO_DIR/.venv-rlt/bin/activate"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR:$(cd "$REPO_DIR/.." && pwd)/src"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export RLINF_NODE_RANK=0
export RLINF_JAX_REF_HOST="${RLINF_JAX_REF_HOST:-$NODE_IP}"
export RLINF_JAX_REF_PORT="${RLINF_JAX_REF_PORT:-6006}"
export RLINF_STAGE1_RLT_TOKEN RLINF_SO101_TASK

# Workers resolve ${oc.env:...} in the raylet's environment (captured at `ray start`),
# so pass everything this shell decides explicitly.
HYDRA_OVERRIDES=(
  "algorithm.replay_buffer.load_path=${RLINF_REPLAY_LOAD_PATH:-null}"
  "env.train.override_cfg.arm_config=$RLT_ROOT/so101/dual_arm.yaml"
  "env.train.override_cfg.hardware=$RLINF_SO101_HARDWARE"
  "env.train.override_cfg.task_description='$RLINF_SO101_TASK'"
  "env.train.override_cfg.keyboard=${RLINF_SO101_KEYBOARD:-evdev}"
  "env.train.override_cfg.key_file=${RLINF_SO101_KEY_FILE:-$HOME/.so101_rlt_keys}"
  "env.train.rlt_stride_features.host=$RLINF_JAX_REF_HOST"
  "env.train.rlt_stride_features.port=$RLINF_JAX_REF_PORT"
  "rollout.rlt_feature_model.host=$RLINF_JAX_REF_HOST"
  "rollout.rlt_feature_model.port=$RLINF_JAX_REF_PORT"
  "rollout.rlt_feature_model.rlt_token_checkpoint=$RLINF_STAGE1_RLT_TOKEN"
)
if [[ -n "${RLINF_STAGE2_RESUME_DIR:-}" ]]; then
  HYDRA_OVERRIDES+=("runner.resume_dir=${RLINF_STAGE2_RESUME_DIR}")
fi
echo "[SO101] hardware=$RLINF_SO101_HARDWARE task=\"$RLINF_SO101_TASK\" JAX=$RLINF_JAX_REF_HOST:$RLINF_JAX_REF_PORT"

if ! ray status --address="$NODE_IP:6379" >/dev/null 2>&1; then
  ray start --head --port=6379 --node-ip-address="$NODE_IP" --disable-usage-stats
fi
exec bash examples/embodiment/run_realworld_async.sh \
  "${RLINF_STAGE2_CONFIG_NAME:-realworld_so101_rlt_stage2_ac_mlp}" "${HYDRA_OVERRIDES[@]}" "$@"
