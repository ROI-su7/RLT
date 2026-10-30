#!/usr/bin/env bash
# Public base assets. Fine-tuned checkpoints and robot datasets are NOT downloaded:
# collect data and fine-tune following docs/PIPELINE.md.
set -euo pipefail
RLT_ROOT="${RLT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$RLT_ROOT/openpi"
# pi0.5 base checkpoint (gs://openpi-assets/checkpoints/pi05_base) into ~/.cache/openpi.
.venv/bin/python - <<'PY'
import time
from openpi.shared import download
url = "gs://openpi-assets/checkpoints/pi05_base"
# An interrupted earlier download leaves <cache>/pi05_base.partial; fsspec would then copy into a
# nested pi05_base/ inside it. Move it aside (never deleted) so the download starts clean.
partial = download.get_cache_dir() / "openpi-assets/checkpoints/pi05_base.partial"
if partial.exists() and not (download.get_cache_dir() / "openpi-assets/checkpoints/pi05_base").exists():
    import filelock

    try:  # only when no other process is downloading right now (openpi holds this lock)
        with filelock.FileLock(partial.with_suffix(".lock"), timeout=0):
            stale = partial.with_name(f"pi05_base.partial.stale-{time.strftime('%Y%m%d-%H%M%S')}")
            partial.rename(stale)
            print(f"[RLT] moved an interrupted download aside: {stale} (delete it manually to free space)")
    except filelock.Timeout:
        print("[RLT] another process is downloading pi05_base; waiting for it")
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
