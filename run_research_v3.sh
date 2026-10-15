#!/usr/bin/env bash
# Reproducible full-retraining pipeline for the offline service-feature repair.
# Each distinct variant MUST have a distinct OUT; old hash-v2 outputs untouched.
set -euo pipefail
usage() {
  echo "Usage: bash run_research_v3.sh {features|lasso|case|vf|gap|references|train|certify|evaluate|sensitivity|audit|boosted-fit|boosted-evaluate|scenario-audit|flexible-audit} [--dry-run]" >&2
}
[[ $# -ge 1 && $# -le 2 ]] || { usage; exit 2; }
STAGE="$1"; DRY=0
case "$STAGE" in
  features|lasso|case|vf|gap|references|train|certify|evaluate|sensitivity|audit|boosted-fit|boosted-evaluate|scenario-audit|flexible-audit) ;;
  *) usage; exit 2 ;;
esac
if [[ $# -eq 2 ]]; then [[ "$2" == "--dry-run" ]] || { usage; exit 2; }; DRY=1; fi
cd "$(dirname "${BASH_SOURCE[0]}")"
if [[ "$(git branch --show-current)" != "anchored-research-v3" ]]; then
  echo "Error: switch to anchored-research-v3." >&2; exit 2
fi
PYTHON="${PYTHON:-python3.14}"
WORKBOOK="${WORKBOOK:-data/UHNOperating_RoomScheduling2011-2013.xlsx}"
ARTIFACTS="${ARTIFACTS:-artifacts/weekly_rich_leakfix_v1}"
GAP_DIR="${GAP_DIR:-artifacts/weekly_gap_leakfix_v1}"
OUT="${OUT:-artifacts/anchored_leakfix_v1}"
WORKERS="${WORKERS:-15}"
INITIAL="${INITIAL:-CASE}"
RIDGE="${RIDGE:-0}"
REFERENCE_PENALTY="${REFERENCE_PENALTY:-0}"
LINE_FRACTIONS="${LINE_FRACTIONS:-1,0.5,0.25,0.125}"
TRUST_GROWTH="${TRUST_GROWTH:-1}"
MAX_TERMINAL_RESTARTS="${MAX_TERMINAL_RESTARTS:-1}"
EXACT_SEP_BATCH="${EXACT_SEP_BATCH:-384}"
MAX_OUTER="${MAX_OUTER:-15}"
SCENARIO_GRID="${SCENARIO_GRID:-0.5:30,0.8:30,1:30,0.8:60}"
cmd=()
case "$STAGE" in
  features)
    cmd=("$PYTHON" run_weekly_rich.py "$WORKBOOK" --stage features --output "$ARTIFACTS");;
  lasso)
    cmd=("$PYTHON" run_weekly_rich.py "$WORKBOOK" --stage fit --output "$ARTIFACTS");;
  case|vf)
    cmd=("$PYTHON" run_weekly_train.py "$WORKBOOK" --artifacts "$ARTIFACTS"
       --stage "$STAGE" --alpha .8 --h 30 --lambda-l1 auto
       --iterations 250 --max-outer 6 --workers "$WORKERS" --threads 1 --seconds 600);;
  gap)
    cmd=("$PYTHON" run_weekly_gap.py "$WORKBOOK" --artifacts "$ARTIFACTS"
       --output "$GAP_DIR" --stage train --initial CASE
       --alpha .8 --h 30 --gamma 2 --max-outer 5 --iterations 150
       --trust 6 --temperature 1.5 --workers "$WORKERS" --threads 1 --seconds 600);;
  audit)
    cmd=("$PYTHON" run_anchored_research_diagnostics.py "$WORKBOOK"
       --artifacts "$ARTIFACTS" --output "$OUT" --stage audit);;
  boosted-fit)
    cmd=("$PYTHON" run_anchored_extras.py --stage boosted-fit --artifacts "$ARTIFACTS"
       --output "$OUT");;
  boosted-evaluate)
    cmd=("$PYTHON" run_anchored_extras.py --stage boosted-evaluate
       --artifacts "$ARTIFACTS" --output "$OUT" --workbook "$WORKBOOK"
       --workers "$WORKERS" --seconds 600);;
  flexible-audit)
    cmd=("$PYTHON" run_anchored_extras.py --stage flexible-audit
       --artifacts "$ARTIFACTS" --output "$OUT" --workbook "$WORKBOOK"
       --max-flex-components "${MAX_FLEX_COMPONENTS:-5}"
       --max-flex-cases "${MAX_FLEX_CASES:-3}"
       --grid-levels "${GRID_LEVELS:-3}"
       --workers "$WORKERS" --seconds 90);;
  scenario-audit)
    cmd=("$PYTHON" run_anchored_extras.py --stage scenario-audit
       --artifacts "$ARTIFACTS" --output "$OUT" --workbook "$WORKBOOK"
       --scenarios "$SCENARIO_GRID" --workers "$WORKERS" --seconds 600);;
  *)
    cmd=("$PYTHON" run_weekly_anchored.py "$WORKBOOK" --artifacts "$ARTIFACTS"
       --gap-model "$GAP_DIR/gap_progress.json" --output "$OUT"
       --site all --stage "$STAGE" --hash-seed 0
       --initial-method "$INITIAL" --max-outer "$MAX_OUTER" --iterations 20
       --closure-rounds 3 --exact-sep-batch "$EXACT_SEP_BATCH"
       --workers "$WORKERS" --threads 1 --master-threads 4
       --seconds 600 --sep-seconds 90 --master-seconds 300
       --ridge "$RIDGE" --reference-penalty "$REFERENCE_PENALTY"
       --line-fractions "$LINE_FRACTIONS" --trust-growth "$TRUST_GROWTH"
       --max-terminal-restarts "$MAX_TERMINAL_RESTARTS")
    if [[ "${TERMINAL_SEPARATION:-0}" == "1" ]]; then cmd+=(--terminal-separation); fi
    if [[ "${TIGHT_SURROGATE:-0}" == "1" ]]; then cmd+=(--tight-surrogate); fi
    ;;
esac
if [[ "$DRY" == 1 ]]; then printf 'DRY RUN: '; printf '%q ' "${cmd[@]}"; printf '\n'; exit 0; fi
if [[ ! -f "$WORKBOOK" ]]; then echo "Missing workbook: $WORKBOOK" >&2; exit 2; fi
case "$STAGE" in
  features) if [[ -d "$ARTIFACTS" ]] && [[ -n "$(ls -A "$ARTIFACTS" 2>/dev/null)" ]]; then
      echo "New feature artifact directory must be empty: $ARTIFACTS" >&2; exit 2; fi;;
  *) if [[ ! -f "$ARTIFACTS/manifest.json" ]]; then echo "Run features first." >&2; exit 2; fi;;
esac
if ! command -v "$PYTHON" >/dev/null; then echo "$PYTHON not found" >&2; exit 2; fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
printf 'Stage: %s\nFeature root: %s\nOutput: %s\n' "$STAGE" "$ARTIFACTS" "$OUT"
if command -v caffeinate >/dev/null; then caffeinate -i "${cmd[@]}"; else "${cmd[@]}"; fi
