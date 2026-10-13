# Weekly operating-room planning: rich-feature training

This branch is the weekly supported-day experiment. The weekly feasible set,
certified ordinary planner, hindsight oracle, and loss-augmented adversary are
in `surgery/weekly_data.py` and `surgery/weekly_planner.py`. They have NOT
been changed by the feature integration.

## Scientific protocol

- UHN 2011–2012 is the development/training period; January–June 2013 remains
  the **final held-out test**, not used to fit, select, or score weekly pilots.
- A case's recommendation is `booked + X @ w`: the coefficient of booked
  duration is fixed at one. The correction matrix does not contain an
  unrestricted booked-duration main effect. It does contain explicit
  booked-duration interactions with service, site, and historical bias.
- `surgery/rich_features.py` wraps the existing `surgery.data.Encoder`.
  The original leave-one-date-out procedure and shrinkage parameters are
  **unchanged**. The training matrix additionally uses same-date-excluded
  history aggregates, calendar, site, patient type, consultation/decision
  lead times, and shrunken surgeon–procedure interactions.
- Evaluation historical features use only the **fitted training cohort**,
  even for unseen surgeon, procedure, or combination IDs. Unseen effects
  shrink/fall back to wider service or global histories.
- Planned-room assignment, intraoperative timestamps, realized duration,
  postoperative outcomes, and held-out labels are not direct predictors.
  `actual` saved with the matrix is a training/evaluation target, not in X.
- To prevent within-development leakage, the feature encoder is fitted on
  selected training weeks, then applied unchanged to subsequent validation
  weeks. We require the maximum training surgery date to precede the
  minimum validation surgery date across the three room groups.
- For this stage, the regularized **LASSO predictive baseline** is selected
  by training-only BIC. The same X is intended for later direct decision-
  aware optimization. No separate feature models or artificial test split.

## Use

Install dependencies with `python -m pip install -r requirements-lock.txt`
and `python -m pip install -e . --no-deps`. Gurobi is required for the
exact planning stage.

```bash
python -m pytest -q

# Build training and later development-period validation matrices.
# This is deliberately slower than a naive date-blind encoder.
caffeinate -i python3.14 run_weekly_rich.py \
  data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --stage features --start-week-index 0 \
  --train-weeks 60 --validation-weeks 18 \
  --output artifacts/weekly_rich_v1

# Fit a single L1-regularized rich predictive correction on training cases only.
python3.14 run_weekly_rich.py \
  data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --stage fit --output artifacts/weekly_rich_v1

# Small **exact** pilot: two group-weeks, BOOKED vs LASSO, same response.
# Displays case-error/VF/GAP/SPO scores and realized regret; VF/GAP/SPO are NOT
# claimed as separately optimized policy methods.
caffeinate -i python3.14 run_weekly_rich.py \
  data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --stage pilot --output artifacts/weekly_rich_v1 \
  --limit-weeks 2 --seconds 120 --threads 1

# To expand to all development group-weeks use --limit-weeks 0;
# certified results from prior calls in this directory are reused.
```

The pilot writes `pilot.csv` after each group-week. To retry uncertified
solves, rerun **the same command in the same output directory** with a larger
`--seconds` budget. Certified policy-week results are reused, and
`pilot_settings.json` prevents mixing scenarios or policies. Use
`--limit-weeks 0` to expand a pilot to every development group-week.

Files: `manifest.json`, `train.npz`, `validation.npz`,
`lasso_policy.json`, `pilot_settings.json`, `pilot.csv`. The feature design and training
selection record the names, source hash, week windows, and policy coefficients.

## Method boundary and next phase

The exact `solve_week` / `solve_adversary` oracles remain the source of
truth. The runner **trains a predictive LASSO baseline**, and evaluates its VF, gamma-gap,
and SPO-style upper bounds with certified schedules. It does **not** yet
optimize distinct rich policy vectors against VF, GAP, or SPO. That requires
a proper optimizer over `w`, using the same frozen feature matrix and
consistent regularization, not another two-parameter advice grid. Do not
present pilot loss scores as a comparison of trained methods.

The original affine offset-and-slope grid and its specialized recovery
runner have been retired: those results were a restricted diagnostic and
are not evidence about rich-feature learning.

## Preserved weekly planner semantics

- Each site group and Monday–Friday week has a fixed roster of physical
  480-minute room-day sessions, inferred retrospectively from retained cases.
- All cases are assigned. Original surgeon operating days are preserved;
  additional weekdays require historical support. No sequencing, deferral,
  endogenous staffing, or unrestricted room access is modeled.
- Cost is idle minutes + 1.75 times overtime, with 30-minute turnover
  between cases on the same room-day, including unused staffed sessions.
- Planner minimizes predicted cost, then moved cases, then canonical
  assignment. Load-balance tie-breaking remains an **audit option** in
  `run_weekly_diagnostics.py`, not part of training.
- Solvers must certify every stage. An incomplete result is labeled
  incomplete, never treated as an optimum.

For the solver-only audit and benchmark use `run_weekly.py` (see
`python run_weekly.py --help`). The independent overload-diagnostics
script remains available because implausible realized block loads still
require investigation.
