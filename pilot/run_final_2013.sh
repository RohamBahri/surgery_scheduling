#!/usr/bin/env bash
# Final paper experiment. Run from repository root. Never delete checkpoints.
set -euo pipefail
PYTHON="${PYTHON:-python3.14}"
WORKBOOK="${WORKBOOK:-data/UHNOperating_RoomScheduling2011-2013.xlsx}"
ROOT="${ROOT:-artifacts/final_2013_audited_v1}"
PHASE="${1:-help}"
run() { "$PYTHON" pilot/run_final_2013.py --stage "$1" --root "$ROOT" --workbook "$WORKBOOK" "${@:2}"; }
case "$PHASE" in
  prepare) run prepare ;;
  train)
    run prepare
    run case
    run anchored
    run vf
    run gap
    echo "Training finished. Next: bash pilot/run_final_2013.sh audit" ;;
  audit) run audit ;;
  freeze) run freeze "${@:2}" ;;
  test)
    [[ -f "$ROOT/FINAL_FREEZE.json" ]] || { echo 'Missing FINAL_FREEZE.json' >&2; exit 1; }
    run test ;;
  summary) run summary ;;
  *) echo 'Usage: bash pilot/run_final_2013.sh {prepare|train|audit|freeze|test|summary}' >&2; exit 2 ;;
esac