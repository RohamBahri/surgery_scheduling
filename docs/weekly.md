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
  lead times, shrunken surgeon–procedure interactions, and four
  hierarchically shrunken proportional-error × booked-duration interactions
  (service/procedure/surgeon/joint). Annual seasonal sine/cosine and year
  progress are intentionally excluded.
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

## Updating an older local checkout

```bash
cd /path/to/surgery_scheduling
git status --short                   # preserve local edits before updating
git fetch origin
git switch weekly-supported-days
git pull --ff-only origin weekly-supported-days
python3.14 -m pip install -r requirements-lock.txt
python3.14 -m pip install -e . --no-deps
python3.14 -m pytest -q
```

Old rich-feature matrices are incompatible with the revised scientific-code
fingerprint. Start a **new** artifact directory; do not reuse v1 results.

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
  --output artifacts/weekly_rich_v2

# Fit a single L1-regularized rich predictive correction on training cases only.
python3.14 run_weekly_rich.py \
  data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --stage fit --output artifacts/weekly_rich_v2

# Small exact pilot: first validation week at each of three sites,
# BOOKED vs pooled LASSO, same response.
# Displays case-error/VF/GAP/SPO scores and realized regret; VF/GAP/SPO are NOT
# claimed as separately optimized policy methods.
caffeinate -i python3.14 run_weekly_rich.py \
  data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --stage pilot --output artifacts/weekly_rich_v2 \
  --limit-weeks 3 --seconds 120 --threads 1

# To expand to all development group-weeks use --limit-weeks 0;
# certified results from prior calls in this directory are reused.
```

The default three-week pilot covers the first validation week at each
site. The pilot writes `pilot.csv` after each group-week. To retry uncertified
solves, rerun **the same command in the same output directory** with a larger
`--seconds` budget (or more threads). A BOOKED/LASSO policy-week pair is
reused **only if both methods were certified against the identical hindsight
oracle assignment**, verified by `oracle_assignment_sha256`. Otherwise
**both methods for that week are recomputed** against the same freshly solved
hindsight oracle. This preserves paired SPO comparisons across partial
restarts even when the oracle's tie choice changes. Existing completed pairs
remain reusable. `pilot_settings.json` prevents mixing response scenarios
or fitted policies. Use `--limit-weeks 0` to expand a pilot to all
development group-weeks.

Files: `manifest.json`, `train.npz`, `validation.npz`,
`lasso_policy.json`, `pilot_settings.json`, `pilot.csv`. The feature design and training selection record the scientific-code
fingerprint, workbook hash, matrix hashes, week windows, and policy
coefficients. The fit stage refuses to overwrite an existing policy.

## Reuse completed rich-feature artifacts after oracle-pair fix

The source fingerprint intentionally includes the weekly runner. The
October 13 oracle-pair resume correction changed the runner, **not** the
feature builder or fitted Lasso. Consequently, files produced with code SHA
`dcdc7f7cfaae9cb5fe83a8c8a39da987ea118c9c6a965eb75288f2f2968b0fb9`
can be adopted **without regenerating X or refitting the policy**, after
checking matrix and policy hashes. This command is strictly limited to that
verified historical version; it rejects all other mismatched sources.

```bash
# From an updated checkout and the existing artifact directory:
python3.14 run_weekly_rich.py \
  data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --stage adopt --output artifacts/weekly_rich_v2
```

The command preserves the original manifest in
`manifest_before_oracle_pair_fix.json`, updates the manifest's code identity
with provenance fields, and renames the legacy SPO rows to
`pilot_before_oracle_pair_fix.csv`. The legacy pilot was fully solved but
its BOOKED/LASSO rows lack hindsight-oracle assignment fingerprints.
The feature arrays and Lasso policy remain byte-identical.

For a 15-core workstation, use three concurrent small pilot week jobs
with five Gurobi threads each (up to 15 solver threads):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 caffeinate -i python3.14 run_weekly_rich.py \
  data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --stage pilot --output artifacts/weekly_rich_v2 \
  --pilot-period validation --limit-weeks 3 \
  --workers 3 --threads 5 --seconds 120
```

Then evaluate **all 54 validation group-weeks**, using fifteen
independent one-thread Gurobi worker processes; the three certified
pairs are reused, and the 2013 test set remains untouched:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 caffeinate -i python3.14 run_weekly_rich.py \
  data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --stage pilot --output artifacts/weekly_rich_v2 \
  --pilot-period validation --limit-weeks 0 \
  --workers 15 --threads 1 --seconds 120
```

Increase `--seconds` if a solver is not certified and rerun the same
directory to retry only affected group-weeks. Each independent worker
solves **both** policy schedules against one shared hindsight oracle,
records `oracle_assignment_sha256`, and checkpoints as soon as a
group-week finishes. `--workers * --threads` cannot exceed 15.
Gurobi controls actual solver parallelism, so 15 is a concurrency
**capacity**, not a guarantee every core is busy at every instant.

These commands score BOOKED and the fitted *predictive* Lasso only.
They do not train the separate response-aware or decision-aware policies.

## Method boundary and next phase

The exact `solve_week` / `solve_adversary` oracles remain the source of
truth. The runner **trains a pooled squared-error LASSO predictive baseline**, and evaluates its VF, gamma-gap,
and SPO-style upper bounds with certified schedules. It does **not** yet
optimize distinct rich policy vectors against VF, GAP, or SPO. That requires
a proper optimizer over `w`, using the same frozen feature matrix and
consistent regularization, not another two-parameter advice grid. LASSO is
not a response-aware CASE policy and must not be reported as one. Do not
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
