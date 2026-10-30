#!/usr/bin/env bash
# Start the locally downloaded full-parameter pi0.5 connect-strings checkpoint at step 30,000.
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
checkpoint_dir="$project_dir/checkpoints/pi05_connect_strings/pi05_connect_strings_20260923/30000"
python_bin="$project_dir/RLinf/.venv-rlt/bin/python"

if [[ ! -f "$checkpoint_dir/params/_METADATA" ]]; then
  echo "Checkpoint is incomplete or not found: $checkpoint_dir" >&2
  exit 1
fi

cd "$project_dir"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

exec "$python_bin" -u scripts/serve_policy.py \
  --env ALOHA \
  --port 6006 \
  --default_prompt "connect the light strings and the battery box to the splitter cable so the lights turn on." \
  policy:checkpoint \
  --policy.config pi05_connect_strings \
  --policy.dir "$checkpoint_dir"
