# Frozen final-paper experiment protocol

This document is the operational source of truth for the final experiment.
The **only supported final-paper entry point is `run_final_paper.py`**. The
older stage modules remain implementation modules used by tests and the wrapper;
do not execute them directly for the final paper run.

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
- Main deployable policies: Booked, Naive, exposure-weighted RA, full-weight RA, SITE_SHIFT, OS and VF.
- Retrospective benchmarks: realized oracle and the **projected hindsight benchmark** (`IMPLEMENTABLE_ORACLE`). The latter is not a scheduling-performance ceiling or lower bound.
- SITE_SHIFT is fit directly on its deployed clipped full-weight case-loss objective. The coarse and refined grids enforce the frozen coefficient box before minimization; the 1-minute duration floor is still applied case by case by the deployed clipping rule.
- VF library: per-site TGH/TWH surfaces with implicit Cartesian-product combination.
- Stage 1 is rejected if VF never attempts outer iteration 1. `VF_STATUS.json` records attempted iterations and termination reason.
- Holdout policy scheduling: one thread, fixed seed, deterministic Gurobi `WorkLimit`; the emergency wall cap is several times the work limit and is retried once if it fires first.
- Decomposed-site Phi accounting uses a scale-aware numerical audit: `abs(error) <= max(1e-2, 1e-6*scale)`. The independently recomputed feasible schedule cost is reported as the incumbent. The reviewed wrapper installs this guard on **all** runtime, deterministic-evaluation and sensitivity worker paths, including the late Stage-1 seed audit and the Stage-2 realized oracle. This tolerance remains far below the cost change from a one-minute idle/overtime accounting error, while leaving ample margin for solver feasibility round-off.
- The supported RA exposure cross-fit leaves scikit-learn's L2 penalty at its default instead of passing the deprecated `penalty="l2"` argument.
- Structural sensitivities evaluate the frozen primary policies under changed capacity/turnover and include a scenario-specific realized oracle and regret brackets. If BOOKED has zero regret in a scenario, percentage gap-closed is reported as undefined (`NaN`), not 100%.
- Optional response sensitivities solve BOOKED once at the **same reduced sensitivity planner budget** used by the learned policies and reuse that solve across the three response scenarios.

## Required run order

### 1. Update, clean the tracked tree, activate the environment

```bash
cd "/Users/roham/Desktop/Surgery project/code"
git checkout main
git pull origin main
git status --short
source .venv/bin/activate
```

The final run requires a clean **tracked** tree. Untracked data/artifact files
are allowed. If `git status --short` shows tracked modifications, stash or
revert them before continuing.

### 2. Focused tests

```bash
python -m pytest \
  tests/test_planning/test_fixed_capacity.py \
  tests/test_final_paper_experiment.py \
  tests/test_final_paper_runtime_fixes.py \
  tests/test_final_paper_scientific_fixes.py \
  tests/test_final_paper_finalization_fixes.py \
  tests/test_final_paper_numeric_guard.py \
  tests/test_final_paper_release_guard.py \
  tests/test_final_paper_wrapper_guards.py \
  -q
```

On the local Mac/scikit-learn version, also turn the warning discussed in the
audit into an error for the dedicated regression test:

```bash
python -W error::FutureWarning -m pytest tests/test_final_paper_numeric_guard.py -q
```

The regression suite contains the exact Phi values from the eight-hour failed
run and requires that this solver-scale discrepancy pass while a materially
larger accounting discrepancy still fails. It also verifies the wrapper installs
the guarded runtime worker for the Stage-2 oracle and the guarded deterministic
worker for the late Stage-1 seed audit before either stage main function starts.

### 3. Real-data preflight

Run the structural/data check:

```bash
python run_final_paper.py preflight \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx
```

Then the short solver/license check. This now explicitly exercises **both** the
runtime worker used by the Stage-2 oracle and the deterministic worker used by
the late Stage-1 seed audit, on training data only:

```bash
python run_final_paper.py preflight \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --cores 15 \
  --solver-check
```

Finally calibrate the actual deterministic Stage-2 planning path on the 15
largest **training** weeks. This can take materially longer than the short
preflight because it intentionally uses the frozen final WorkLimit:

```bash
python run_final_paper_deterministic_calibration.py \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --cores 15 \
  --weeks 15
```

Proceed only if this ends with `DETERMINISTIC_CALIBRATION_OK`.

### 4. Stage 1: training only

Use a fresh artifact directory:

```bash
caffeinate -i python run_final_paper.py train \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --artifact-root artifacts/final_paper_training_frozen_v5 \
  --cores 15 \
  --max-wall-minutes 720
```

Successful completion prints `TRAINING_COMPLETE_HOLDOUT_LOCKED_REVIEWED`.
Before consuming the holdout, review at least:

- `ORACLE_TRAIN.csv`;
- `RA_PDCA.csv`, `RA_FULL_PDCA.csv`, `OS_PDCA.csv`;
- `SITE_SHIFT_PDCA.csv`;
- `VF_TRAJECTORY.csv` and `VF_STATUS.json`;
- `TRAIN_LIBRARY_SURFACES.csv` and saturation status;
- `REGULARIZATION.json`;
- `TIE_SEED_AUDIT.csv` (a coarse training diagnostic only, not a deployment tie-equivalence proof);
- `FINALIZATION_FIXES.json`, `NUMERIC_GUARD.json`, and `RELEASE_GUARD.json`.

Do **not** run Stage 2 until the Stage-1 diagnostics have been reviewed.

### 5. Stage 2: primary holdout, once

Stage 2 requires the exact Git commit recorded in `TRAINING_FREEZE.json` and a
clean tracked tree. If `main` has moved, create a worktree at the frozen SHA:

```bash
FROZEN_SHA=$(python -c 'import json; print(json.load(open("artifacts/final_paper_training_frozen_v5/TRAINING_FREEZE.json"))["git_head"])')
git worktree add ../surgery_final_eval "$FROZEN_SHA"
cd ../surgery_final_eval
source "/Users/roham/Desktop/Surgery project/code/.venv/bin/activate"
```

Run the primary evaluator once:

```bash
caffeinate -i python run_final_paper.py evaluate \
  --data "/Users/roham/Desktop/Surgery project/code/data/UHNOperating_RoomScheduling2011-2013.xlsx" \
  --training-artifact-root "/Users/roham/Desktop/Surgery project/code/artifacts/final_paper_training_frozen_v5" \
  --artifact-root "/Users/roham/Desktop/Surgery project/code/artifacts/final_paper_holdout_frozen_v5" \
  --cores 15
```

The response-misspecification scenarios remain optional. If they are wanted,
add `--run-response-sensitivities` **before** the one-shot Stage-2 run. BOOKED is
solved once at the response-sensitivity planner budget and reused across all
three scenarios. The frozen scenarios are
`(alpha,h)=(0.5,30),(0.8,15),(0.8,60)`.

### 6. Required structural sensitivities

After successful primary Stage 2, from the same frozen worktree:

```bash
caffeinate -i python run_final_paper.py sensitivities \
  --data "/Users/roham/Desktop/Surgery project/code/data/UHNOperating_RoomScheduling2011-2013.xlsx" \
  --training-artifact-root "/Users/roham/Desktop/Surgery project/code/artifacts/final_paper_training_frozen_v5" \
  --primary-evaluation-root "/Users/roham/Desktop/Surgery project/code/artifacts/final_paper_holdout_frozen_v5" \
  --artifact-root "/Users/roham/Desktop/Surgery project/code/artifacts/final_paper_required_sensitivities_v5" \
  --cores 15
```

This evaluates the frozen primary policies under `regular_template` capacity and
under zero turnover. Each scenario includes its own realized-duration oracle,
regret brackets, and within-scenario gap-closed summaries. Nothing is retrained.
