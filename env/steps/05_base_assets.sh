#!/usr/bin/env bash
# Public base assets. Fine-tuned checkpoints and robot datasets are NOT downloaded:
# collect data and fine-tune following docs/PIPELINE.md.
set -euo pipefail
RLT_ROOT="${RLT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$RLT_ROOT/openpi"
# pi0.5 base checkpoint (gs://openpi-assets/checkpoints/pi05_base) into ~/.cache/openpi.
.venv/bin/python - <<'PY'
from openpi.shared import download
url = "gs://openpi-assets/checkpoints/pi05_base"
path = download.maybe_download(url)
if not (path / "params").exists():  # an interrupted earlier download leaves an empty dir
    path = download.maybe_download(url, force_download=True)
assert (path / "params").exists(), f"pi05_base download incomplete: {path}"
print("[RLT] pi05_base at", path)
open(".pi05_base_path", "w").write(str(path))
PY
# The training configs load ./checkpoints/pi05_base/params relative to openpi/.
mkdir -p checkpoints
if [[ ! -e checkpoints/pi05_base ]]; then ln -s "$(cat .pi05_base_path)" checkpoints/pi05_base; fi
rm -f .pi05_base_path
# PaliGemma tokenizer used by pi0.5 (downloaded on first use by openpi as well).
.venv/bin/python - <<'PY'
from openpi.shared import download
print("[RLT] tokenizer at", download.maybe_download("gs://big_vision/paligemma_tokenizer.model", gs={"token": "anon"}))
PY
