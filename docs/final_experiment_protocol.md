# Final-paper experiment protocol

This document is the operational source of truth for the final experiment. The
**only supported final-paper entry point is `run_final_paper.py`**. The older
stage modules remain implementation modules used by tests and the wrapper; do
not execute them directly for the paper run.

## Scientific specification

- Sites: TGH + TWH; one pooled recommendation policy, physically separate site capacity.
- Split: 72 training weeks / 22 final holdout weeks; holdout begins 2013-01-28.
- Cohort: canonical non-cancelled, non-emergency OR cases with ordered positive timestamps; `0 < booked <= 480` minutes; exclude room duration `> 24 h` as a timestamp-data error; do not upper-cap otherwise valid realized duration.
- Frozen counts: 32,269 cleaned; 30,260 TGH/TWH weekdays; 20,951 training; 6,700 holdout.
- Features: 107 case-local predecision features; no calendar features; procedure/surgeon/service vocabularies selected from training only; no target encoding.
- Capacity: training-only `median_count_template` primary; `regular_template` required structural sensitivity.
- Planning: weekly day-flexible fixed-capacity assignment; no cross-site assignment; no deferral.
- Costs: overtime 15/min, idle 10/min; 480-minute blocks; 30-minute turnover primary.
- Turnover sensitivity: 0 minutes, required as a secondary structural check.
- Behavior: `(alpha,h)=(0.8,30)` remains the reference scenario, but Stage 1 now accepts any explicitly declared valid scenario with `0 <= alpha < 1` and `h > 0`. Each trained scenario is frozen separately in `BEHAVIOR_SCENARIO.json` and `FROZEN_SETTINGS.json`.
- Coefficient box: +/-100; intercept unpenalized.
- Regularization: Naive has its own minutes-scale L1 calibration; RA, RA_FULL, OS and VF share one common cost-scale lambda within a scenario.
- Main deployable policies: Booked, Naive, exposure-weighted RA, full-weight RA, SITE_SHIFT, OS and VF.
- Retrospective benchmarks: realized oracle and the projected hindsight benchmark (`IMPLEMENTABLE_ORACLE`). The latter is not a scheduling-performance ceiling or lower bound.
- VF library: per-site TGH/TWH surfaces with implicit Cartesian-product combination.
- Holdout policy scheduling: one thread, fixed seed, deterministic Gurobi `WorkLimit`; the emergency wall cap is several times the work limit and is retried once if it fires first.
- Decomposed-site Phi accounting uses `abs(error) <= max(1e-2, 1e-6*scale)`, with the independently recomputed feasible schedule cost reported as the incumbent.

## Shared behavior-independent weekly backbone

The training realized oracle and BOOKED schedules depend on the data, weekly
instances, eligibility, roster, turnover and cost specification, but **not** on
behavioral `alpha` or `h`. They must therefore be solved once and reused across
all behavior-specific training runs.

The shared-plan artifact stores, for every training week:

- the complete case-to-block assignment;
- feasible Phi incumbent;
- valid Phi lower bound;
- native-Psi gap;
- solver status, exactness flag and accumulated solve time;
- a cryptographic manifest over the input workbook and the exact weekly case,
  block and eligibility structure.

Loading a shared plan reconstructs the schedule, verifies eligibility, recomputes
its cost, verifies the stored bound, and rejects a changed data/weekly planning
instance. The shared artifact is not accepted merely because filenames match.

### Persistent Gurobi diagnostics

Every reviewed weekly site MILP now receives a unique `LogFile`, independent of
console verbosity. Logs are stored under the relevant run root at:

```text
gurobi_logs/<solve-label>/week_<###>/<site>__<duration-hash>__<budget>.log
```

This covers training oracle/BOOKED solves, VF/training weekly solves,
deterministic holdout scheduling and structural sensitivities, including macOS
spawned worker processes. These files are intended for later diagnosis of hard
weeks, root gaps, node growth, incumbent discovery and solver termination.

## Required run order

### 1. Update and test

```bash
cd "/Users/roham/Desktop/Surgery project/code"
git checkout main
git pull origin main
git status --short
source .venv/bin/activate

python -m pytest \
  tests/test_planning/test_fixed_capacity.py \
  tests/test_final_paper_experiment.py \
  tests/test_final_paper_runtime_fixes.py \
  tests/test_final_paper_scientific_fixes.py \
  tests/test_final_paper_finalization_fixes.py \
  tests/test_final_paper_numeric_guard.py \
  tests/test_final_paper_release_guard.py \
  tests/test_final_paper_shared_plans.py \
  tests/test_final_paper_wrapper_guards.py \
  -q

python -W error::FutureWarning -m pytest tests/test_final_paper_numeric_guard.py -q
```

The final run requires a clean **tracked** tree. Untracked data/artifact files
are allowed.

### 2. Real-data preflight and deterministic calibration

```bash
python run_final_paper.py preflight \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --cores 15 \
  --solver-check

python run_final_paper_deterministic_calibration.py \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --cores 15 \
  --weeks 15
```

Proceed only if calibration ends with `DETERMINISTIC_CALIBRATION_OK`.

### 3. Solve the shared realized oracle and BOOKED plans once

Use a fresh directory. `--verbose` is optional for the terminal; full Gurobi
logs are persisted to files either way.

```bash
caffeinate -i python run_final_paper.py shared-plans \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --artifact-root artifacts/final_paper_shared_plans_v1 \
  --cores 15 \
  --kind both \
  --oracle-seconds 1800 \
  --booked-seconds 300 \
  --booked-gap 0.01
```

Successful completion prints `SHARED_PLANS_COMPLETE`. Review
`SHARED_PLANS_MANIFEST.json`, `ORACLE_TRAIN.csv`, `BOOKED_TRAIN.csv`, and the
`gurobi_logs/` tree.

Oracle and BOOKED can also be run independently with `--kind oracle` or
`--kind booked`.

#### Give unresolved weeks more time later

Never mutate the first shared artifact. Create a new version and point
`--warm-start-root` at the previous one. Only weeks still above the requested
gap are re-solved, and their saved assignments are used as Gurobi MIP starts.
The refined artifact keeps the best feasible incumbent and strongest valid lower
bound across attempts.

```bash
caffeinate -i python run_final_paper.py shared-plans \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --artifact-root artifacts/final_paper_shared_plans_v2 \
  --warm-start-root artifacts/final_paper_shared_plans_v1 \
  --cores 15 \
  --kind both \
  --oracle-seconds 3600 \
  --booked-seconds 900 \
  --booked-gap 0.002
```

### 4. Train behavioral scenarios using the same shared plans

Each scenario gets a separate fresh artifact directory. `alpha` and `h` are not
restricted to the old three sensitivity values. The scenario name and parameters
are frozen into the training bundle.

Reference scenario:

```bash
caffeinate -i python run_final_paper.py train \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --shared-plans-root artifacts/final_paper_shared_plans_v2 \
  --scenario-name alpha080_h30 \
  --alpha 0.8 \
  --h 30 \
  --artifact-root artifacts/train_alpha080_h30 \
  --cores 15 \
  --max-wall-minutes 720
```

Examples of additional **trained** regimes (illustrative; the final scenario set
should be declared before holdout evaluation):

```bash
caffeinate -i python run_final_paper.py train \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --shared-plans-root artifacts/final_paper_shared_plans_v2 \
  --scenario-name alpha050_h30 --alpha 0.5 --h 30 \
  --artifact-root artifacts/train_alpha050_h30 \
  --cores 15 --max-wall-minutes 720

caffeinate -i python run_final_paper.py train \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --shared-plans-root artifacts/final_paper_shared_plans_v2 \
  --scenario-name alpha080_h60 --alpha 0.8 --h 60 \
  --artifact-root artifacts/train_alpha080_h60 \
  --cores 15 --max-wall-minutes 720
```

For every scenario, Stage 1 **loads** `oracle_train` and `booked_train` from the
validated shared artifact instead of re-solving them. Those schedules are then
added to the scenario's initial schedule library exactly as before. Remaining
weekly optimization is behavior-specific (especially VF enrichment) and is not
silently reused unless scientifically valid.

Successful completion prints `TRAINING_COMPLETE_HOLDOUT_LOCKED_REVIEWED`. Before
holdout use, review at least `BEHAVIOR_SCENARIO.json`, `RA_PDCA.csv`,
`RA_FULL_PDCA.csv`, `OS_PDCA.csv`, `SITE_SHIFT_PDCA.csv`, `VF_TRAJECTORY.csv`,
`VF_STATUS.json`, `TRAIN_LIBRARY_SURFACES.csv`, `REGULARIZATION.json`,
`TIE_SEED_AUDIT.csv`, and the run's `gurobi_logs/`.

### 5. Holdout evaluation

Do not evaluate holdout until the full set of trained scenarios has been fixed
and reviewed. Each Stage-1 bundle records its exact Git commit. Evaluate from a
clean worktree at that commit using only `python run_final_paper.py evaluate ...`.
The one-shot holdout-consumption rule remains unchanged.

### 6. Structural sensitivities

Run required `regular_template` capacity and zero-turnover checks after primary
holdout evaluation using `python run_final_paper.py sensitivities ...`. These are
deployment robustness checks for frozen policies; they do not retrain them.

## Efficiency boundary

This refactor intentionally removes work that is **provably behavior-independent**:
realized-oracle and BOOKED training MILPs. It also makes their schedules reusable
as warm starts for stronger later solves. It does **not** yet share behavior-
dependent VF schedule-library enrichment across scenarios. Cross-scenario VF
library sharing is a promising next optimization, but it should be added only
after the first multi-scenario timing/profile confirms the benefit and its
reproducibility rules are specified.
