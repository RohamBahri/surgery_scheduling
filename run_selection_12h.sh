#!/usr/bin/env bash
# Single bounded full-training + one 2013 evaluation.
# Never run concurrently with the old full paper experiment.
set -euo pipefail
cd "$(dirname "$0")"
: "${PYTHON_BIN:=python}"
: "${WORKBOOK:=data/UHNOperating_RoomScheduling2011-2013.xlsx}"
: "${OUT:=artifacts/paper_selection_12h}"
: "${CORES:=15}"
: "${SETUP_SOLVE_SECONDS:=360}"
: "${TRAIN_SOLVE_SECONDS:=120}"
: "${EVAL_SOLVE_SECONDS:=90}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SURGERY_HASH_SEEDS=0
"$PYTHON_BIN" -m pytest -q tests/test_selection_12h.py tests/test_theory_solver.py tests/test_theory_paper.py
cmd=("$PYTHON_BIN" pilot/run_selection_12h.py --mode run
  --workbook "$WORKBOOK" --out "$OUT" --cores "$CORES"
  --setup-solver-seconds "$SETUP_SOLVE_SECONDS"
  --train-solver-seconds "$TRAIN_SOLVE_SECONDS"
  --eval-solver-seconds "$EVAL_SOLVE_SECONDS")
if command -v caffeinate >/dev/null 2>&1; then
  exec caffeinate -i "${cmd[@]}"
else
  exec "${cmd[@]}"
fi
