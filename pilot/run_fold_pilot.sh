#!/usr/bin/env bash
# Fold-safe, training-period-only research pilot. Run at repository root.
set -euo pipefail
PYTHON="${PYTHON:-python3.14}"
ARTIFACTS="${ARTIFACTS:-artifacts/weekly_rich_leakfix_v1}"
WORKBOOK="${WORKBOOK:-data/UHNOperating_RoomScheduling2011-2013.xlsx}"
ROOT="${ROOT:-artifacts/decision_model_pilot_v2}"
WORKERS="${WORKERS:-12}"
PHASE="${1:-case}"
CONTEXTS="${CONTEXTS:-none}"
METHODS="${METHODS:-vf gap anchored}"
COMMON=(--root "$ROOT" --artifacts "$ARTIFACTS" --workbook "$WORKBOOK"
        --case-inner 500 --inner 35 --outer 2 --workers "$WORKERS"
        --seconds 300 --master-seconds 120 --sep-seconds 60
        --sep-batch 128 --closure-rounds 1 --seeds 0,1,2
        --ridge-grid 0.0003,0.001,0.003)
runner() {
  "$PYTHON" pilot/run_fold_pilot.py "${COMMON[@]}" "$@"
}
case "$PHASE" in
  case)
    # Stage I: original response without context; inverse with no/cohort/roster context.
    for fold in 0 1; do
      runner --stage case --fold "$fold" --link original --context none
      for ctx in none cohort roster; do
        runner --stage case --fold "$fold" --link inverse --context "$ctx"
      done
    done
    ;;
  decision)
    # Stage II: begin with CONTEXTS=none, then cohort/roster if warranted.
    for ctx in $CONTEXTS; do
      for fold in 0 1; do
        for method in $METHODS; do
          runner --stage decision --method "$method" --fold "$fold" \
                 --link inverse --context "$ctx"
        done
      done
    done
    ;;
  score)
    # Stage III: out-of-fold exact deployed planner; no selection during scoring.
    for fold in 0 1; do
      runner --stage score --fold "$fold" --link original --context none
      for ctx in none cohort roster; do
        runner --stage score --fold "$fold" --link inverse --context "$ctx"
      done
    done
    "$PYTHON" pilot/summarize_folds.py "$ROOT"
    ;;
  summary)
    "$PYTHON" pilot/summarize_folds.py "$ROOT"
    ;;
  *)
    echo "Usage: bash pilot/run_fold_pilot.sh {case|decision|score|summary}" >&2
    exit 2
    ;;
esac
