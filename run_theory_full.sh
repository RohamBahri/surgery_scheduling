#!/usr/bin/env bash
# Full pre-2013 development experiment. Run from any working directory.
set -euo pipefail
cd "$(dirname "$0")"
: "${PYTHON_BIN:=python}"
: "${WORKBOOK:=data/UHNOperating_RoomScheduling2011-2013.xlsx}"
: "${OUT:=artifacts/theory_full_v3}"
: "${CORES:=15}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

"$PYTHON_BIN" -m pytest -q tests/test_theory_exact.py tests/test_theory_solver.py tests/test_theory_regressions.py

cmd=("$PYTHON_BIN" pilot/run_theory_full.py
  --workbook "$WORKBOOK" --out "$OUT" --fold both
  --methods VF,SHARED,OBS,EG,ROBUST,GLOBAL,ANCHORED
  --cores "$CORES" --seconds 120 --pw-seconds 900
  --shared-seconds 21600 --global-seconds 14400
  --global-nodes 40 --pw-nodes 100 --rounds 10 --radius 32
  --search-seconds 86400)
if command -v caffeinate >/dev/null 2>&1; then
  caffeinate -i "${cmd[@]}"
else
  "${cmd[@]}"
fi
