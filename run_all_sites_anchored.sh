#!/usr/bin/env bash
# Clean shared-policy experiment across all three UHN room groups.
# Reuses frozen rich features and baseline policies, never old tie-rule checkpoints.
set -euo pipefail
usage() {
  printf 'Usage: bash run_all_sites_anchored.sh {references|train|certify|evaluate|sensitivity} [--dry-run]\n' >&2
}
if [[ $# -lt 1 || $# -gt 2 ]]; then usage; exit 2; fi
STAGE="$1"
case "$STAGE" in references|train|certify|evaluate|sensitivity) ;; *) usage; exit 2 ;; esac
DRY_RUN=0
if [[ $# -eq 2 ]]; then
  [[ "$2" == "--dry-run" ]] || { usage; exit 2; }
  DRY_RUN=1
fi
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ "$(git branch --show-current)" != 'anchored-exact-adaptive' ]]; then
  echo 'Error: switch to anchored-exact-adaptive before running.' >&2
  exit 2
fi
WORKBOOK='data/UHNOperating_RoomScheduling2011-2013.xlsx'
ARTIFACTS='artifacts/weekly_rich_v2'
GAP_MODEL='artifacts/weekly_gap_v1/gap_progress.json'
OUT="${OUT:-artifacts/anchored_ALL_hash_v2}"
PYTHON="${PYTHON:-python3.14}"
WORKERS="${WORKERS:-15}"
for required in run_weekly_anchored.py "$WORKBOOK" \
  "$ARTIFACTS/learning_settings.json" "$ARTIFACTS/model_case.json" \
  "$ARTIFACTS/lasso_policy.json" "$ARTIFACTS/vf_progress.json" "$GAP_MODEL"; do
  if [[ ! -f "$required" ]]; then
    printf 'Error: required input not found: %s\n' "$required" >&2
    exit 2
  fi
done
if [[ "$STAGE" == train && ! -f "$OUT/anchored_references.json" ]]; then
  echo 'Error: run references first.' >&2
  exit 2
fi
if [[ "$STAGE" == certify || "$STAGE" == evaluate || "$STAGE" == sensitivity ]]; then
  if [[ ! -f "$OUT/anchored_progress.json" ]]; then
    echo 'Error: run train first.' >&2
    exit 2
  fi
fi
ARGS=(
  "$WORKBOOK"
  --artifacts "$ARTIFACTS"
  --gap-model "$GAP_MODEL"
  --output "$OUT"
  --site all
  --stage "$STAGE"
  --hash-seed 0
  --max-outer 15
  --iterations 20
  --closure-rounds 3
  --exact-sep-batch 384
  --workers "$WORKERS"
  --threads 1
  --master-threads 4
  --seconds 600
  --sep-seconds 90
  --master-seconds 300
)
if [[ "$DRY_RUN" -eq 1 ]]; then
  echo 'DRY RUN (no Python imports or solver calls):'
  printf '%q ' "$PYTHON" run_weekly_anchored.py "${ARGS[@]}"
  printf '\n'
  exit 0
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  printf 'Error: %s is not on PATH.\n' "$PYTHON" >&2
  exit 2
fi
if ! "$PYTHON" -c 'import gurobipy; import numpy; import pandas'; then
  echo 'Error: Python environment lacks gurobipy, numpy, or pandas.' >&2
  exit 2
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
printf 'Stage: %s\nOutput: %s\n' "$STAGE" "$OUT"
if [[ "$STAGE" == evaluate || "$STAGE" == sensitivity ]]; then
  echo '2012 development validation only; 2013 holdout is untouched.'
fi
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -i "$PYTHON" run_weekly_anchored.py "${ARGS[@]}"
else
  "$PYTHON" run_weekly_anchored.py "${ARGS[@]}"
fi
