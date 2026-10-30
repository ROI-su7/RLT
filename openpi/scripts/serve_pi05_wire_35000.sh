#!/usr/bin/env bash
# Backward-compatible entry point for the full-parameter wire checkpoint at step 30,000.
set -euo pipefail

exec "$(dirname "$0")/serve_pi05_wire_all_data_full_30000.sh"
