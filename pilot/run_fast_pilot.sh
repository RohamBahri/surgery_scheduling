#!/usr/bin/env bash
set -euo pipefail
echo "Using fold-safe v2 pilot instead of obsolete in-sample pilot." >&2
exec bash pilot/run_fold_pilot.sh "$@"
