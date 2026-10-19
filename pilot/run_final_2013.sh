#!/usr/bin/env bash
# Run from repository root; existing checkpoints are reused.
set -euo pipefail
PYTHON="${PYTHON:-python3.14}"
WORKBOOK="${WORKBOOK:-data/UHNOperating_RoomScheduling2011-2013.xlsx}"
ROOT="${ROOT:-artifacts/final_2013_audited_v1}"
run() { "$PYTHON" pilot/run_final_2013.py --stage "$1" --root "$ROOT" --workbook "$WORKBOOK"; }
case "${1:-help}" in
  prepare) run prepare ;;
  train)
    run prepare
    run case
    run anchored
    run vf
    echo 'Primary training done. Optional: bash pilot/run_final_2013.sh gap' ;;
  gap) run gap ;;
  audit) run audit ;;
  test) run test ;;
  summary) run summary ;;
  *) echo 'Usage: bash pilot/run_final_2013.sh {prepare|train|gap|audit|test|summary}' >&2; exit 2 ;;
esac