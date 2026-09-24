# Frozen final-paper experiment protocol

This document is the operational source of truth for the final experiment.
Do **not** run `run_final_vf_experiment.py` or `run_final_paper_experiment.py`
as final experiment entry points. They are implementation modules retained for
reuse/tests. The supported entry points are the scripts below.

## Frozen primary specification

- Sites: TGH + TWH; one pooled recommendation policy, physically separate site capacity.
- Split: 72 training weeks / 22 final holdout weeks; holdout begins 2013-01-28.
- Cohort: canonical non-cancelled, non-emergency OR cases with ordered positive timestamps; `0 < booked <= 480` minutes; exclude room duration `> 24 h` as a timestamp-data error; do not upper-cap otherwise valid realized duration.
- Frozen counts: 32,269 cleaned; 30,260 TGH/TWH weekdays; 20,951 training; 6,700 holdout.
- Features: 107 case-local predecision features; no calendar features; procedure/surgeon/service vocabularies selected from training only; no target encoding.
- Capacity: training-only `median_count_template` primary; `regular_template` required structural sensitivity.
- Planning: weekly day-flexible fixed-capacity assignment; no cross-site assignment; no deferral.
- Costs: overtime 15/min, idle 10/min; 480-minute blocks; 30-minute turnover primary.
- Turnover sensitivity: 0 minutes, required as a secondary structural check.
- Response: alpha=.8, h=30 primary; minimum displayed recommended duration 1 minute.
- Coefficient box: +/-100; intercept unpenalized.
- Regularization: Naive has its own minutes-scale L1 calibration; RA, RA_FULL, OS and VF share one common cost-scale lambda.
- Main policies: Booked, Naive, exposure-weighted RA, full-weight RA, site-shift baseline, OS, VF, realized oracle, implementable behavioral oracle.
- VF library: per-site TGH/TWH surfaces with implicit Cartesian-product combination.
- Holdout policy scheduling: one thread, fixed seed, deterministic Gurobi `WorkLimit`; wall-clock `TimeLimit` is fail-only.

## Required run order

### 1. Update and test

```bash
cd "/Users/roham/Desktop/Surgery project/code"
git checkout main
git pull origin main
source .venv/bin/activate

python -m pytest \
  tests/test_planning/test_fixed_capacity.py \
  tests/test_final_paper_experiment.py \
  tests/test_final_paper_runtime_fixes.py \
  tests/test_final_paper_scientific_fixes.py \
  -q
```

### 2. Real-data preflight

First run the solver-free structural/data check:

```bash
python run_final_paper_preflight.py \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx
```

Then run the short local solver/license check. It uses training data only:

```bash
python run_final_paper_preflight.py \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --cores 15 \
  --solver-check
```

Do not start Stage 1 unless both finish with `PREFLIGHT_OK`.

### 3. Stage 1: training only

Use a fresh artifact directory:

```bash
caffeinate -i python run_final_paper_training.py \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --artifact-root artifacts/final_paper_training_frozen_v2 \
  --cores 15 \
  --max-wall-minutes 720
```

Success is `TRAINING_COMPLETE_HOLDOUT_LOCKED`. Review the Stage-1 diagnostics
before consuming the holdout. In particular inspect oracle gaps, RA/RA_FULL/OS
pDCA histories, VF trajectory, library saturation, tie-seed audit, and clipping
rates/regularization records.

### 4. Stage 2: primary holdout, once

Stage 2 requires the exact Git commit recorded in `TRAINING_FREEZE.json` and a
clean tracked tree. If `main` has moved, create a worktree at the frozen SHA.
For example:

```bash
FROZEN_SHA=$(python -c 'import json; print(json.load(open("artifacts/final_paper_training_frozen_v2/TRAINING_FREEZE.json"))["git_head"])')
git worktree add ../surgery_final_eval "$FROZEN_SHA"
cd ../surgery_final_eval
source "/Users/roham/Desktop/Surgery project/code/.venv/bin/activate"
```

Run the primary evaluator once:

```bash
caffeinate -i python run_final_paper_evaluation.py \
  --data "/Users/roham/Desktop/Surgery project/code/data/UHNOperating_RoomScheduling2011-2013.xlsx" \
  --training-artifact-root "/Users/roham/Desktop/Surgery project/code/artifacts/final_paper_training_frozen_v2" \
  --artifact-root "/Users/roham/Desktop/Surgery project/code/artifacts/final_paper_holdout_frozen_v2" \
  --cores 15
```

If response-misspecification results are desired in the same one-shot run, add
`--run-response-sensitivities` before starting Stage 2. The predeclared response
scenarios are `(alpha,h)=(0.5,30),(0.8,15),(0.8,60)`.

### 5. Required structural sensitivities

These are predeclared and source-frozen by the Stage-1 commit. Run them only
after successful primary Stage 2, from the same frozen worktree:

```bash
caffeinate -i python run_final_paper_required_sensitivities.py \
  --data "/Users/roham/Desktop/Surgery project/code/data/UHNOperating_RoomScheduling2011-2013.xlsx" \
  --training-artifact-root "/Users/roham/Desktop/Surgery project/code/artifacts/final_paper_training_frozen_v2" \
  --primary-evaluation-root "/Users/roham/Desktop/Surgery project/code/artifacts/final_paper_holdout_frozen_v2" \
  --artifact-root "/Users/roham/Desktop/Surgery project/code/artifacts/final_paper_required_sensitivities_v2" \
  --cores 15
```

This runner evaluates frozen policies under the regular-template capacity proxy
and under zero turnover. It does not retrain anything.
