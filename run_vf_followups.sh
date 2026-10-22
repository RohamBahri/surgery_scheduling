#!/usr/bin/env bash
# Frozen VF follow-ups: independent resumable stages, no baseline retraining.
set -euo pipefail
cd "$(dirname "$0")"
: "${PYTHON_BIN:=python}"
: "${CORES:=15}"
: "${ARTIFACT_ROOT:=artifacts/objective_selection_12h}"
: "${OUT:=artifacts/vf_followups}"
export SURGERY_HASH_SEEDS=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$PYTHON_BIN" -m pytest -q tests/test_vf_followups.py tests/test_objective_selection.py tests/test_theory_solver.py
if command -v caffeinate >/dev/null 2>&1; then
    exec caffeinate -i "$PYTHON_BIN" pilot/run_vf_followups.py \
        --artifact-root "$ARTIFACT_ROOT" --out "$OUT" --cores "$CORES"
else
    exec "$PYTHON_BIN" pilot/run_vf_followups.py \
        --artifact-root "$ARTIFACT_ROOT" --out "$OUT" --cores "$CORES"
fi
