#!/usr/bin/env bash
# One full-data, four-objective experiment; Unix hard stage watchdog.
set -euo pipefail
cd "$(dirname "$0")"
: "${PYTHON_BIN:=python}"
: "${WORKBOOK:=data/UHNOperating_RoomScheduling2011-2013.xlsx}"
: "${OUT:=artifacts/objective_selection_12h}"
: "${CORES:=15}"
: "${SETUP_SOLVE_SECONDS:=240}"
: "${TRAIN_SOLVE_SECONDS:=45}"
: "${EVAL_SOLVE_SECONDS:=40}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SURGERY_HASH_SEEDS=0
"$PYTHON_BIN" -m pytest -q tests/test_objective_selection.py tests/test_theory_solver.py tests/test_theory_paper.py
cmd=("$PYTHON_BIN" pilot/run_objective_selection.py
    --workbook "$WORKBOOK" --out "$OUT" --cores "$CORES"
    --setup-solve-seconds "$SETUP_SOLVE_SECONDS"
    --train-solve-seconds "$TRAIN_SOLVE_SECONDS"
    --eval-solve-seconds "$EVAL_SOLVE_SECONDS")
if command -v caffeinate >/dev/null 2>&1; then
    exec caffeinate -i "${cmd[@]}"
else
    exec "${cmd[@]}"
fi
