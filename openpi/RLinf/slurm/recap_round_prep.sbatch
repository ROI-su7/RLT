#!/bin/bash
# Piper RECAP round preparation: isolated demo copy, JAX->PyTorch conversion and
# returns (CPU work; this account has no cpu-partition access, so it takes one
# 3090 slot). Submit from the cluster login node:
#   RECAP_ROUND=round_001 sbatch slurm/recap_round_prep.sbatch
# Every step is skipped when its output already exists, so resubmission resumes.
#SBATCH --job-name=recap_prep
#SBATCH --partition=3090
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/recap_prep_%j.out
#SBATCH --error=logs/recap_prep_%j.out

set -euo pipefail
REPO_DIR="${RLINF_GPU_REPO_DIR:-$HOME/FluxVla/openpi/RLinf}"
cd "$REPO_DIR"
source slurm/recap_server_env.sh

echo "Job $SLURM_JOB_ID on $(hostname); round dir $RECAP_RUN_DIR"
mkdir -p "$RECAP_RUN_DIR"

# The robot host writes this marker after the rollout dataset and the Value
# Model weights are fully synced, so the job can be queued before that.
READY="$RECAP_RUN_DIR/.inputs_ready"
for _ in $(seq 1 480); do
  [[ -f "$READY" ]] && break
  sleep 60
done
[[ -f "$READY" ]] || { echo "Timed out waiting for $READY" >&2; exit 1; }

# 1. Isolated demonstration copy: compute_returns rewrites meta/ in place, so
#    meta is copied while data/videos are hard links (no extra space).
if [[ ! -f "$RECAP_SFT_DATASET/meta/info.json" ]]; then
  SRC="$RECAP_NATIVE_OPENPI/lerobot_data/all_wire_data"
  mkdir -p "$RECAP_SFT_DATASET"
  cp -al "$SRC/data" "$SRC/videos" "$RECAP_SFT_DATASET/"
  cp -a "$SRC/meta" "$RECAP_SFT_DATASET/"
  echo "Created demo copy $RECAP_SFT_DATASET"
fi

# 2. JAX -> PyTorch conversion of the deployed base policy (CPU only).
if [[ ! -f "$RECAP_PI05_TORCH_DIR/model.safetensors" ]]; then
  bash examples/recap_piper/run.sh convert
fi

# 3. Returns for demos (sft) and rollouts with the RECAP reward settings in
#    piper_recap_returns.yaml (failure_reward, takeover_penalty).
if [[ ! -f "$RECAP_DATASET/meta/returns_piper_recap_v1.parquet" ]]; then
  "$RECAP_PYTHON" examples/offline_rl/advantage_labeling/recap/process/compute_returns.py \
    --config-path "$REPO_DIR/examples/offline_rl/config" --config-name piper_recap_returns \
    "$RET_DATA" data.num_workers=16 2>&1 | tee "$RECAP_RUN_DIR/returns.log"
fi

"$RECAP_PYTHON" -m rlinf.recap_piper.preflight --stage value
"$RECAP_PYTHON" -m rlinf.recap_piper.preflight --stage cfg || true  # advantages come later
touch "$RECAP_RUN_DIR/prep.done"
echo "PREP DONE"
