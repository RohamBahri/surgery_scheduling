#!/usr/bin/env bash
# Reproducible all-site weekly anchored-regret learning and evaluation.
# Run on macOS with Python 3.14, Gurobi and the frozen training artifacts.
set -euo pipefail

usage() {
  printf 'Usage: bash run_all_sites_anchored.sh {references|diagnose-references|rescue-references|finalize-references|train|certify|evaluate} [--dry-run]\n' >&2
}
if [[ $# -lt 1 || $# -gt 2 ]]; then usage; exit 2; fi
STAGE="$1"
case "$STAGE" in
  references|diagnose-references|rescue-references|finalize-references|train|certify|evaluate) ;;
  *) usage; exit 2 ;;
esac
DRY_RUN=0
if [[ $# -eq 2 ]]; then
  if [[ "$2" != '--dry-run' ]]; then usage; exit 2; fi
  DRY_RUN=1
fi

# The script works from any current directory. Paths remain anchored at the repo root.
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [[ "$(git branch --show-current)" != 'anchored-exact-adaptive' ]]; then
  echo 'Error: switch to the anchored-exact-adaptive branch before running.' >&2
  exit 2
fi
WORKBOOK='data/UHNOperating_RoomScheduling2011-2013.xlsx'
ARTIFACTS='artifacts/weekly_rich_v2'
GAP_MODEL='artifacts/weekly_gap_v1/gap_progress.json'
OUT="${OUT:-artifacts/anchored_ALL_extended}"
PYTHON='python3.14'

for required in run_weekly_anchored.py "$WORKBOOK" \
                "$ARTIFACTS/learning_settings.json" \
                "$ARTIFACTS/model_case.json" \
                "$ARTIFACTS/lasso_policy.json" \
                "$ARTIFACTS/vf_progress.json" "$GAP_MODEL"; do
  if [[ ! -f "$required" ]]; then
    printf 'Error: required input not found: %s\n' "$required" >&2
    exit 2
  fi
done
if [[ ( "$STAGE" == diagnose-references || "$STAGE" == rescue-references || "$STAGE" == finalize-references ) && ! -f "$OUT/anchored_references.json" ]]; then
  echo "Error: run references before rescuing incomplete components." >&2
  exit 2
fi
if [[ "$STAGE" == train && ! -f "$OUT/anchored_references.json" ]]; then
  echo 'Error: run the references stage before training.' >&2
  exit 2
fi
if [[ "$STAGE" == certify || "$STAGE" == evaluate ]]; then
  if [[ ! -f "$OUT/anchored_progress.json" ]]; then
    echo 'Error: no trained model found; run train first.' >&2
    exit 2
  fi
fi

if [[ "$STAGE" == rescue-references || "$STAGE" == diagnose-references || "$STAGE" == finalize-references ]]; then
  if [[ ! -f rescue_anchored_references.py ]]; then
    echo "Error: missing rescue_anchored_references.py" >&2
    exit 2
  fi
  ENTRY=rescue_anchored_references.py
  SOLVE_SECONDS="${RESCUE_SECONDS:-1800}"
  if [[ "$STAGE" == finalize-references ]]; then
    SOLVE_SECONDS="${RESCUE_SECONDS:-120}"
  fi
  ARGS=(
    "$WORKBOOK" --artifacts "$ARTIFACTS" --gap-model "$GAP_MODEL"
    --output "$OUT" --seconds "$SOLVE_SECONDS" --threads "${RESCUE_THREADS:-4}"
  )
  if [[ "$STAGE" == diagnose-references ]]; then
    ARGS+=(--diagnose-only)
  fi
  if [[ "$STAGE" == finalize-references ]]; then
    ARGS+=(--allow-feasible)
  fi
else
  ENTRY=run_weekly_anchored.py
  ARGS=(
  "$WORKBOOK"
  --artifacts "$ARTIFACTS"
  --gap-model "$GAP_MODEL"
  --output "$OUT"
  --site all
  --stage "$STAGE"
  --max-outer 15
  --iterations 20
  --closure-rounds 3
  --exact-sep-batch 384
  --workers 15
  --threads 1
  --seconds 300
  --sep-seconds 90
  --master-seconds 300
  )
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf 'DRY RUN (no Python imports, solver calls, or artifacts written):\n'
  printf '%q ' "$PYTHON" "$ENTRY" "${ARGS[@]}"
  printf '\n'
  exit 0
fi
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  printf 'Error: %s is not on PATH.\n' "$PYTHON" >&2
  exit 2
fi
if ! "$PYTHON" -c 'import gurobipy; import numpy; import pandas'; then
  echo 'Error: Python 3.14 environment lacks gurobipy, numpy, or pandas.' >&2
  exit 2
fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
printf 'Stage: %s\nOutput: %s\n' "$STAGE" "$OUT"
if [[ "$STAGE" == evaluate ]]; then
  echo 'Evaluating 2012 development validation only; 2013 holdout is untouched.'
fi
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -i "$PYTHON" "$ENTRY" "${ARGS[@]}"
else
  "$PYTHON" "$ENTRY" "${ARGS[@]}"
fi
