#!/usr/bin/env bash
# One chronological experiment: all 24,506 pre-2013 cases fit; 8,774 in 2013 test.
set -euo pipefail
cd "$(dirname "$0")"
: "${PYTHON_BIN:=python}"
: "${WORKBOOK:=data/UHNOperating_RoomScheduling2011-2013.xlsx}"
: "${OUT:=artifacts/theory_single_split_paper_v2}"
: "${CORES:=15}"
: "${SETUP_SECONDS:=14400}"
: "${STAT_SECONDS:=1800}"
: "${SOLVE_SECONDS:=120}"
: "${TASK_SECONDS:=1500}"
: "${ROUND_SECONDS:=7200}"
: "${SEARCH_SECONDS:=14400}"
: "${PREPROCESS_SECONDS:=14400}"
: "${GLOBAL_SECONDS:=14400}"
: "${SHARED_SECONDS:=14400}"
: "${PW_SECONDS:=480}"
: "${EVAL_COMPONENT_SECONDS:=1500}"
: "${MECHANISM_SECONDS:=600}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PYTHON_BIN" -m pytest -q tests/test_theory_exact.py tests/test_theory_solver.py tests/test_theory_regressions.py tests/test_theory_paper.py
cmd=("$PYTHON_BIN" pilot/run_theory_full.py
  --workbook "$WORKBOOK" --out "$OUT" --fold 2013 --acknowledge-final
  --methods STAT_PLUS,VF,SHARED,EG,OBS,ROBUST
  --stat-seconds "$STAT_SECONDS" --setup-seconds "$SETUP_SECONDS"
  --cores "$CORES" --seconds "$SOLVE_SECONDS" --task-seconds "$TASK_SECONDS"
  --round-seconds "$ROUND_SECONDS" --search-seconds "$SEARCH_SECONDS"
  --preprocess-seconds "$PREPROCESS_SECONDS"
  --pw-seconds "$PW_SECONDS" --eval-component-seconds "$EVAL_COMPONENT_SECONDS"
  --mechanism-seconds "$MECHANISM_SECONDS"
  --shared-seconds "$SHARED_SECONDS" --global-seconds "$GLOBAL_SECONDS"
  --global-nodes 40 --pw-nodes 100 --rounds 10 --radius 32)
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -i "${cmd[@]}"
else
  "${cmd[@]}"
fi
