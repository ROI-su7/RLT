#!/bin/bash
# Validate the offline RLinf/OpenPI environment on a Slurm GPU node.
# Submit from the cluster login node with:
#   sbatch slurm/setup_piper_rlt_stage2_env.sbatch
#
# The environment lives on the shared home filesystem and is reused by later
# Stage-2 jobs, regardless of which finaiXX node Slurm allocates.
#SBATCH --job-name=rlt_env
#SBATCH --partition=3090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/rlt_env_%j.out
#SBATCH --error=logs/rlt_env_%j.err

set -euo pipefail

REPO_DIR="${RLINF_GPU_REPO_DIR:-$HOME/openpi/RLinf}"
cd "$REPO_DIR"
mkdir -p logs

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Node IPs: $(hostname -I)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

# The compute nodes have no Internet access. Environment creation is performed
# by uploading and relocating the verified archive; this job is validation-only.
test -x .venv-rlt/bin/python
export PYTHONPATH="$REPO_DIR"
.venv-rlt/bin/python - <<'PY'
import openpi, ray, rlinf, torch, transformers
from transformers.models.siglip import check
print("openpi:", openpi.__file__)
print("rlinf:", rlinf.__file__)
print("ray:", ray.__version__)
print("torch:", torch.__version__, "GPUs:", torch.cuda.device_count())
print("transformers:", transformers.__version__)
assert ray.__version__ == "2.56.0"
assert torch.__version__.split("+")[0] == "2.6.0"
assert transformers.__version__ == "4.53.2"
assert check.check_whether_transformers_replace_is_installed_correctly()
assert torch.cuda.device_count() >= 2
PY
echo "PASS: cluster RLinf Stage-2 environment is ready"
