#!/usr/bin/env bash
# Server-side paths for Piper RECAP; sourced by the recap_round_*.sbatch jobs.
# Mirrors examples/recap_piper/env.sh, whose defaults point at the robot host.
REPO_DIR="${REPO_DIR:-$HOME/FluxVla/openpi/RLinf}"
export RECAP_NATIVE_OPENPI="${RECAP_NATIVE_OPENPI:-$HOME/FluxVla/openpi}"
export RECAP_PYTHON="${RECAP_PYTHON:-$REPO_DIR/.venv-rlt/bin/python}"
export RECAP_ROUND="${RECAP_ROUND:-round_001}"
export RECAP_RUN_DIR="${RECAP_RUN_DIR:-$REPO_DIR/recap_runs/$RECAP_ROUND}"
export RECAP_DATASET="${RECAP_DATASET:-$REPO_DIR/recap_data/lerobot_v1}"
export RECAP_SFT_DATASET="${RECAP_SFT_DATASET:-$REPO_DIR/recap_data/sft_all_wire_data}"
source "$REPO_DIR/examples/recap_piper/env.sh"
export PYTHONUNBUFFERED=1
# The cluster has no system FFmpeg 4-7 for torchcodec 0.2 (PyAV bundles its own).
export RECAP_VIDEO_BACKEND="${RECAP_VIDEO_BACKEND:-pyav}"

# Demonstrations (type sft, all positive) + rollouts (C/X, takeovers), the
# same four-stage data overrides as README_PIPER_RECAP.md section 8.
export RET_DATA="data.train_data_paths=[{dataset_path:$RECAP_SFT_DATASET,type:sft},{dataset_path:$RECAP_DATASET,type:rollout}]"
export VAL_DATA="data.train_data_paths=[{dataset_path:$RECAP_SFT_DATASET,type:sft,weight:1.0,robot_type:piper,model_type:pi05},{dataset_path:$RECAP_DATASET,type:rollout,weight:1.0,robot_type:piper,model_type:pi05}]"
export ADV_DATA="data.train_data_paths=[{dataset_path:$RECAP_SFT_DATASET,type:sft,weight:1.0,robot_type:piper},{dataset_path:$RECAP_DATASET,type:rollout,weight:1.0,robot_type:piper}]"
export CFG_DATA="data.train_data_paths=[{dataset_path:$RECAP_SFT_DATASET,type:sft,weight:1.0},{dataset_path:$RECAP_DATASET,type:rollout,weight:1.0}]"
