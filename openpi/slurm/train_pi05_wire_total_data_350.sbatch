#!/bin/bash
#SBATCH --job-name=pi05_wire_350
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

cd "$HOME/openpi"
mkdir -p logs

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export PYTHONPATH="$HOME/openpi/src${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_MODE=disabled

echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
nvidia-smi

"$HOME/openpi/envs/pi0/bin/python" scripts/train.py pi05_wire \
  --exp-name=pi05_wire_total_data_350_20260814 \
  --no-wandb-enabled \
  --resume
