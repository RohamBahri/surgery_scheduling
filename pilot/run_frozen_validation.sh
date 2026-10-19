#!/usr/bin/env bash
# Explicitly separated TRAIN/FREEZE and independent development-validation stages.
set -euo pipefail
PYTHON="${PYTHON:-python3.14}"
WORKBOOK="${WORKBOOK:-data/UHNOperating_RoomScheduling2011-2013.xlsx}"
ARTIFACTS="${ARTIFACTS:-artifacts/weekly_rich_leakfix_v1}"
ROOT="${ROOT:-artifacts/frozen_context_validation_v1}"
WORKERS="${WORKERS:-12}"
PHASE="${1:-help}"
common=(--root "$ROOT" --artifacts "$ARTIFACTS"
        --workbook "$WORKBOOK" --workers "$WORKERS")
run() { "$PYTHON" pilot/run_frozen_validation.py --stage "$1" "${common[@]}"; }
case "$PHASE" in
  train)
    run case
    run anchored
    run freeze
    echo "FROZEN. Inspect $ROOT/frozen_policies.json before running validation."
    ;;
  validate)
    if [[ ! -f "$ROOT/frozen_policies.json" ]]; then
      echo "No frozen policies. Run train before validate." >&2
      exit 2
    fi
    run evaluate
    ;;
  summary)
    run summary
    ;;
  *)
    echo "Usage: bash pilot/run_frozen_validation.sh {train|validate|summary}" >&2
    exit 2
    ;;
esac
