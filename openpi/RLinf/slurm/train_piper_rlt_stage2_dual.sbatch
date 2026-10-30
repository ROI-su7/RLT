#!/bin/bash
# Slurm-owned GPU side of Piper RLT Stage 2. The job starts a Ray head, writes
# its reachable IP to logs/stage2_<jobid>.head, waits for the robot Ray node,
# then starts training. The robot host remains outside Slurm.
#SBATCH --job-name=piper_rlt_s2
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=72:00:00
#SBATCH --output=logs/piper_rlt_s2_%j.out
#SBATCH --error=logs/piper_rlt_s2_%j.err

set -euo pipefail

REPO_DIR="${RLINF_GPU_REPO_DIR:-$HOME/openpi/RLinf}"
cd "$REPO_DIR"
mkdir -p logs
source .venv-rlt/bin/activate

export PYTHONPATH="$REPO_DIR"
export RLINF_GPU_REPO_DIR="$REPO_DIR"
export RLINF_NODE_RANK=0
export RLINF_CODE_WORKING_DIR=auto
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1

# The 3090 and HGX compute nodes respectively use the 192.168.81.0/24 and
# 192.168.80.0/24 fabrics. Both are reachable from the robot host. Do not use
# the login-node address 192.168.85.129 here.
HEAD_IP="$(hostname -I | tr ' ' '\n' | awk '/^192\.168\.(80|81)\./ {print; exit}')"
if [[ -z "$HEAD_IP" ]]; then
  echo "Cannot find a robot-reachable compute-node 192.168.80.x/81.x address." >&2
  exit 1
fi
export RLINF_GPU_HEAD_IP="$HEAD_IP"

# Both hosts are multi-homed. Gloo otherwise may advertise the compute node's
# 192.168.73.x interface even though Ray uses the robot-reachable 192.168.80.x
# or 192.168.81.x interface, causing the first
# rollout collective to hang before any robot action is produced.
HEAD_IFACE="$(ip route get 192.168.209.107 | awk '{for (i=1; i<=NF; i++) if ($i == "dev") {print $(i+1); exit}}')"
if [[ -z "$HEAD_IFACE" ]]; then
  echo "Cannot determine the compute interface used to reach the Piper host." >&2
  exit 1
fi
export GLOO_SOCKET_IFNAME="$HEAD_IFACE"
export NCCL_SOCKET_IFNAME="$HEAD_IFACE"

HEAD_FILE="$REPO_DIR/logs/stage2_${SLURM_JOB_ID}.head"
printf '%s\n' "$HEAD_IP" > "$HEAD_FILE"
ln -sfn "$(basename "$HEAD_FILE")" "$REPO_DIR/logs/stage2_latest.head"

cleanup() {
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Ray head: $HEAD_IP:6379"
echo "Distributed interface: $HEAD_IFACE"
echo "Robot command:"
echo "  RLINF_GPU_HEAD_IP=$HEAD_IP bash examples/embodiment/join_piper_rlt_stage2_dual_robot.sh"
nvidia-smi

ray stop --force >/dev/null 2>&1 || true
ray start --head --port=6379 --node-ip-address="$HEAD_IP" \
  --num-cpus="${SLURM_CPUS_PER_TASK:-32}" \
  --num-gpus=2 --disable-usage-stats

# Do not create a disposable Python Ray driver just to poll membership here.
# On this Slurm/Ray combination, terminating that short driver can strand the
# batch step even after both nodes are visible. The long-lived RLinf driver can
# start immediately; Ray keeps the Piper-pinned EnvWorker pending until the
# robot node joins.
echo "Starting trainer; the Piper EnvWorker will wait for node 192.168.209.107."
exec bash examples/embodiment/run_piper_rlt_stage2_dual_train.sh
