#!/usr/bin/env bash
# Run from the surgery_scheduling repository root.
set -euo pipefail
PYTHON="${PYTHON:-python3.14}"
ARTIFACTS="${ARTIFACTS:-artifacts/weekly_rich_leakfix_v1}"
WORKBOOK="${WORKBOOK:-data/UHNOperating_RoomScheduling2011-2013.xlsx}"
ROOT="${ROOT:-artifacts/decision_model_pilot_v1}"
REFS="${REFS:-artifacts/anchored_leakfix_v1/anchored_references.json}"
WORKERS="${WORKERS:-12}"
COMMON=(--artifacts "$ARTIFACTS" --workbook "$WORKBOOK" \
        --inner 35 --outer 2 --workers "$WORKERS" --seconds 300 \
        --master-seconds 120 --sep-seconds 60 --sep-batch 128 \
        --closure-rounds 1 --ridge .001 --seeds 0,1,2)

if [[ ! -f "$REFS" ]]; then
  echo "Cannot find training reference file: $REFS" >&2
  exit 2
fi
# Stage I: four statistical baselines.
for link in original inverse; do
  for context in none weekly; do
    OUT="$ROOT/${link}_${context}"
    "$PYTHON" pilot/run_pilot.py --stage case \
      --link "$link" --context "$context" --out "$OUT" "${COMMON[@]}"
    "$PYTHON" pilot/run_pilot.py --stage compare \
      --link "$link" --context "$context" --out "$OUT" \
      --reuse-refs "$REFS" "${COMMON[@]}"
  done
done
# Stage II: short optimization-method comparisons under the inverse link.
for context in none weekly; do
  OUT="$ROOT/inverse_${context}"
  for stage in vf gap anchored compare; do
    "$PYTHON" pilot/run_pilot.py --stage "$stage" \
      --link inverse --context "$context" --out "$OUT" \
      --reuse-refs "$REFS" "${COMMON[@]}"
  done
done
"$PYTHON" pilot/summarize.py "$ROOT"
echo "Upload $ROOT/pilot_summary.json and case_bic_path.json files."
