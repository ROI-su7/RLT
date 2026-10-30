#!/bin/bash
# Piper RECAP round on one 8x A100-40G node: Value Model -> advantages -> CFG.
# Needs recap_round_prep.sbatch outputs. Submit from the login node:
#   RECAP_ROUND=round_001 sbatch --dependency=afterok:<prep_job> slurm/recap_round_train_8xa100.sbatch
# Each stage writes $RECAP_RUN_DIR/<stage>.done; resubmission skips finished
# stages. Before each full run, a few-step smoke run picks the largest micro
# batch that fits; the global batch stays fixed via gradient accumulation.
# 8x RTX 3090 (24 GB) instead: shard CFG parameters and allow smaller micro
# batches (export the variables, then sbatch -p 3090 --export=ALL ...):
#   RECAP_FSDP_SHARDING=full_shard VALUE_MICRO_CANDIDATES="16 8 4 2" CFG_MICRO_CANDIDATES="8 4 2 1"
# full_shard is untested for CFG; the Value model supports no_shard only.
#SBATCH --job-name=recap_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=800G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/recap_train_%j.out
#SBATCH --error=logs/recap_train_%j.out

set -euo pipefail
REPO_DIR="${RLINF_GPU_REPO_DIR:-$HOME/FluxVla/openpi/RLinf}"
cd "$REPO_DIR"
source slurm/recap_server_env.sh
CONFIG_DIR="$REPO_DIR/examples/offline_rl/config"
RAY_BIN="$(dirname "$RECAP_PYTHON")/ray"

# Value: official RECAP setting (global batch 256, 8000 steps).
VALUE_GLOBAL_BATCH="${VALUE_GLOBAL_BATCH:-256}"
VALUE_STEPS="${VALUE_STEPS:-8000}"
# CFG: the original pi05_connect_strings training (batch 128, cosine 1k warmup,
# peak 2.5e-5 -> 2.5e-6, AdamW b2=0.95, clip 1.0), 30k steps, FSDP on 8 GPUs.
CFG_GLOBAL_BATCH="${CFG_GLOBAL_BATCH:-128}"
CFG_STEPS="${CFG_STEPS:-30000}"
# Memory knobs (defaults fit A100-40G); see the 3090 example above.
FSDP_SHARDING="${RECAP_FSDP_SHARDING:-no_shard}"  # CFG only
VALUE_MICRO_CANDIDATES="${VALUE_MICRO_CANDIDATES:-32 16 8}"
CFG_MICRO_CANDIDATES="${CFG_MICRO_CANDIDATES:-16 8 4 2}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-45m}"  # a hung collective must not stall the job

# Single node: keep collective bootstrap on loopback; NVLink carries NCCL data.
export RLINF_NODE_RANK=0
export GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo
RAY_TMP="/tmp/ray_recap_${SLURM_JOB_ID}"
cleanup() { env -u RAY_ADDRESS "$RAY_BIN" stop --force >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

echo "Job $SLURM_JOB_ID on $(hostname), GPUs ${CUDA_VISIBLE_DEVICES:-?}; round $RECAP_RUN_DIR; sharding $FSDP_SHARDING"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
mkdir -p "$RECAP_RUN_DIR"
[[ -f "$RECAP_RUN_DIR/prep.done" ]] || { echo "Run recap_round_prep.sbatch first" >&2; exit 1; }

start_ray() {
  cleanup
  env -u RAY_ADDRESS "$RAY_BIN" start --head --port=6385 --node-ip-address=127.0.0.1 \
    --num-cpus="${SLURM_CPUS_PER_TASK:-64}" --num-gpus=8 \
    --temp-dir="$RAY_TMP" --disable-usage-stats
}

latest_checkpoint() {  # <log dir> -> newest global_step_N directory
  find "$1" -maxdepth 4 -type d -name 'global_step_*' | sort -V | tail -1
}

# Try micro batches from large to small for a few steps; echo the first that
# completes. All candidates must divide global / 8. Only OOM or a timeout moves
# on to a smaller size; any other error is not batch-related and stops here.
pick_micro_batch() {  # <stage> <entry> <config> <global> <candidates> [overrides...]
  local stage=$1 entry=$2 config=$3 global=$4 candidates=$5; shift 5
  local mb
  for mb in $candidates; do
    local dir="$RECAP_RUN_DIR/smoke_${stage}_mb${mb}"
    rm -rf "$dir"
    echo "[$stage] smoke test micro_batch=$mb" >&2
    start_ray >&2
    if timeout "$SMOKE_TIMEOUT" "$RECAP_PYTHON" "$entry" --config-path "$CONFIG_DIR" --config-name "$config" \
      "$@" runner.logger.log_path="$dir" actor.micro_batch_size="$mb" \
      actor.global_batch_size="$global" runner.max_steps=3 runner.save_interval=1000000 \
      >"$dir.log" 2>&1; then
      echo "$mb"; return 0
    else
      local rc=$?
    fi
    echo "[$stage] micro_batch=$mb failed (exit $rc, tail of $dir.log):" >&2
    tail -5 "$dir.log" >&2 || true
    if [[ $rc -ne 124 ]] && ! grep -aqiE "out of memory|OutOfMemoryError" "$dir.log"; then
      echo "[$stage] not an OOM; see $dir.log" >&2; return 1
    fi
  done
  return 1
}

# ---- 1. Value Model ---------------------------------------------------------
if [[ ! -f "$RECAP_RUN_DIR/value.done" ]]; then
  ENTRY=examples/offline_rl/advantage_labeling/recap/train_value.py
  # The Value model calls its embedding outside the FSDP unit's forward, so it
  # stays on its config's no_shard; RECAP_FSDP_SHARDING applies to CFG only.
  VALUE_ARGS=("$VAL_DATA" data.train_num_workers=8
    actor.optim.total_training_steps="$VALUE_STEPS" actor.optim.lr_warmup_steps=500)
  MB=$(pick_micro_batch value "$ENTRY" piper_recap_value "$VALUE_GLOBAL_BATCH" "$VALUE_MICRO_CANDIDATES" "${VALUE_ARGS[@]}")
  echo "[value] full run: micro_batch=$MB global=$VALUE_GLOBAL_BATCH steps=$VALUE_STEPS"
  start_ray
  RESUME=$(latest_checkpoint "$RECAP_RUN_DIR/value" || true)
  "$RECAP_PYTHON" "$ENTRY" --config-path "$CONFIG_DIR" --config-name piper_recap_value \
    "${VALUE_ARGS[@]}" runner.logger.log_path="$RECAP_RUN_DIR/value" \
    actor.micro_batch_size="$MB" actor.global_batch_size="$VALUE_GLOBAL_BATCH" \
    runner.max_steps="$VALUE_STEPS" runner.save_interval=1000 \
    ${RESUME:+runner.resume_dir="$RESUME"} 2>&1 | tee -a "$RECAP_RUN_DIR/value.log"
  touch "$RECAP_RUN_DIR/value.done"
fi

# ---- 2. Advantages (8-GPU torchrun inference) -------------------------------
if [[ ! -f "$RECAP_RUN_DIR/advantages.done" ]]; then
  cleanup
  RECAP_VALUE_CHECKPOINT="$(latest_checkpoint "$RECAP_RUN_DIR/value")/actor/model_state_dict"
  export RECAP_VALUE_CHECKPOINT
  echo "[advantages] value checkpoint $RECAP_VALUE_CHECKPOINT"
  "$RECAP_PYTHON" -m torch.distributed.run --standalone --nproc-per-node=8 \
    examples/offline_rl/advantage_labeling/recap/process/compute_advantages.py \
    --config-path "$CONFIG_DIR" --config-name piper_recap_advantages \
    "$ADV_DATA" distributed.enabled=true advantage.batch_size=32 \
    advantage.num_dataloader_workers_per_gpu=4 2>&1 | tee "$RECAP_RUN_DIR/advantages.log"
  touch "$RECAP_RUN_DIR/advantages.done"
fi
"$RECAP_PYTHON" -m rlinf.recap_piper.preflight --stage cfg

# ---- 3. CFG policy training --------------------------------------------------
if [[ ! -f "$RECAP_RUN_DIR/cfg.done" ]]; then
  ENTRY=examples/offline_rl/policy_optimization/cfg_rl/train_cfg.py
  CFG_ARGS=("$CFG_DATA" data.num_workers=8 runner.max_epochs=1000000
    actor.fsdp_config.sharding_strategy="$FSDP_SHARDING"
    actor.optim.lr=2.5e-5 actor.optim.min_lr=2.5e-6 actor.optim.lr_scheduler=cosine
    actor.optim.lr_warmup_steps=1000 actor.optim.total_training_steps="$CFG_STEPS")
  MB=$(pick_micro_batch cfg "$ENTRY" piper_recap_cfg "$CFG_GLOBAL_BATCH" "$CFG_MICRO_CANDIDATES" "${CFG_ARGS[@]}")
  echo "[cfg] full run: micro_batch=$MB global=$CFG_GLOBAL_BATCH steps=$CFG_STEPS"
  start_ray
  RESUME=$(latest_checkpoint "$RECAP_RUN_DIR/cfg" || true)
  "$RECAP_PYTHON" "$ENTRY" --config-path "$CONFIG_DIR" --config-name piper_recap_cfg \
    "${CFG_ARGS[@]}" runner.logger.log_path="$RECAP_RUN_DIR/cfg" \
    actor.micro_batch_size="$MB" actor.global_batch_size="$CFG_GLOBAL_BATCH" \
    runner.max_steps="$CFG_STEPS" runner.save_interval=2500 \
    ${RESUME:+runner.resume_dir="$RESUME"} 2>&1 | tee -a "$RECAP_RUN_DIR/cfg.log"
  touch "$RECAP_RUN_DIR/cfg.done"
fi
echo "RECAP ROUND DONE: $(latest_checkpoint "$RECAP_RUN_DIR/cfg")"
