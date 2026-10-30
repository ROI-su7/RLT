#!/usr/bin/env bash
# One-screen status of a Piper RECAP round on the Slurm server. From the robot:
#   ssh myserver 'bash ~/FluxVla/openpi/RLinf/slurm/recap_status.sh'          # summary
#   ssh myserver 'bash ~/FluxVla/openpi/RLinf/slurm/recap_status.sh --gpu'    # + nvidia-smi
# RECAP_ROUND (default round_001) selects the round. Read-only.
REPO_DIR="${REPO_DIR:-$HOME/FluxVla/openpi/RLinf}"
RUN_DIR="${RECAP_RUN_DIR:-$REPO_DIR/recap_runs/${RECAP_ROUND:-round_001}}"
cd "$REPO_DIR" || exit 1

echo "== Slurm jobs ($(date '+%F %T'))"
squeue -u "$USER" -o "%.10i %.18j %.6P %.9T %.11M %.20S %R" | grep -E "JOBID|recap" || true

echo; echo "== Stages in $RUN_DIR"
for stage in prep value advantages cfg; do
  [[ -f "$RUN_DIR/$stage.done" ]] && state="done  $(date -r "$RUN_DIR/$stage.done" '+%F %T')" || state="-"
  printf '  %-11s %s\n' "$stage" "$state"
done

# Micro-batch smoke tests: the first passing size is used for the full run.
smokes=$(ls -d "$RUN_DIR"/smoke_*_mb*.log 2>/dev/null)
if [[ -n "$smokes" ]]; then
  echo; echo "== Micro-batch smoke tests"
  for f in $smokes; do
    if grep -qE "Traceback|OutOfMemory|out of memory|Error" "$f"; then r="FAILED: $(grep -m1 -oE '[A-Za-z]*(OutOfMemory|Error)[^ ]*' "$f")"
    else r="ok/running"; fi
    printf '  %-28s %s\n' "$(basename "$f")" "$r"
  done
fi

# tqdm writes "Global Step: 12%|...| 3600/30000 [1:02:03<7:34:00, 1.2s/it]" with \r.
echo; echo "== Training progress"
for stage in value cfg; do
  log="$RUN_DIR/$stage.log"
  [[ -f "$log" ]] || continue
  line=$(tail -c 20000 "$log" | tr '\r' '\n' | grep -a "Global Step" | tail -1)
  prog=$(grep -oE '[0-9]+/[0-9]+ \[[^]]*\]' <<<"$line" | tail -1)
  ckpt=$(find "$RUN_DIR/$stage" -maxdepth 4 -type d -name 'global_step_*' 2>/dev/null | sort -V | tail -1)
  printf '  %-6s %s\n' "$stage" "${prog:-(no step yet)}  (elapsed<remaining, time/step)"
  printf '  %-6s latest checkpoint: %s\n' "" "${ckpt:-none}"
done
[[ -f "$RUN_DIR/advantages.log" && ! -f "$RUN_DIR/advantages.done" ]] && {
  echo "  advantages: $(tail -c 3000 "$RUN_DIR/advantages.log" | tr '\r' '\n' | grep -a . | tail -1)"; }

echo; echo "== Latest Slurm log lines"
for f in $(ls -t logs/recap_prep_*.out 2>/dev/null | head -1) $(ls -t logs/recap_train_*.out 2>/dev/null | head -1); do
  echo "  -- $f"
  tail -c 4000 "$f" | tr '\r' '\n' | grep -a . | tail -4 | cut -c1-200 | sed 's/^/     /'
  grep -aqE "Traceback|Error|Timed out|exceeded" "$f" &&
    echo "     !! errors: $(grep -aE 'Traceback|Error|Timed out|exceeded' "$f" | tail -1 | cut -c1-160)"
done

if [[ "${1:-}" == "--gpu" ]]; then
  job=$(squeue -u "$USER" -h -t R -n recap_train,recap_train_3090 -o %i | head -1)
  echo; echo "== GPUs of running train job ${job:-none}"
  [[ -n "$job" ]] && srun --jobid="$job" -N1 -n1 \
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv
fi
