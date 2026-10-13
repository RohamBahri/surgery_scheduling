# Full-scale rich weekly CASE and VF experiment

The 74-dimensional frozen feature matrices and LASSO baseline are reused from
`run_weekly_rich.py`. This entry point trains **two additional distinct
policies** using all 180 training group-weeks (60 weeks at each site), then
evaluates BOOKED, LASSO, CASE, and VF on **all 54 validation group-weeks**
(18 later weeks at each site). It does not create another validation period,
or use the 2013 test for selection.

`surgery/weekly_training.py` implements:

- CASE: minimizes asymmetric post-response duration error. The recommendation
  is `X @ w`, actual planner duration is `booked + response(clipped(X @ w))`.
- VF: minimizes a valid planning-regret upper bound (up to the constant
  realized hindsight optimal cost): `sum theta(actual - post) +
  min_feasible_schedule C(schedule, post)`. The current predicted minimum is
  obtained from **certified exact** weekly planning oracles. The fixed-plan
  schedule cost is an upper bound, tangent at the current policy. Smooth
  optimization proposes each update; only a decrease in the true *unsmoothed*
  fixed-plan bound is accepted. This guarantees non-increasing training VF
  bound across accepted iterations, **not** a global optimum certificate.
- Both objectives use the identical frozen `X`, responses, no train/test
  leakage, training-derived shared L1 penalty, and normalization per case.
  The intercept is not penalized. The anchor on booked duration stays exactly
  one. Bounded, piecewise-linear surgeon response is honored exactly for
  objective acceptance; smoothing affects only gradient computation.

The training settings pin the source hash of both training files independently\nfrom the unchanged feature encoder hash, so existing rich matrices remain\nreusable but new policies cannot silently mix trainer implementations.\n\nThe training optimizer is nonconvex because of the response map and
combinatorial weekly planner. Neither full-case nor VF training is described
as globally optimal. We do not claim to have separately trained GAP or SPO
models; those methods require dedicated valid optimizing algorithms.

## Update an older checkout

```bash
cd /path/to/surgery_scheduling
git status --short
git fetch origin
git switch weekly-supported-days
git pull --ff-only origin weekly-supported-days
python3.14 -m pip install -r requirements-lock.txt
python3.14 -m pip install -e . --no-deps
python3.14 -m pytest -q
```

Preserve uncommitted local changes first. No need to rebuild
`artifacts/weekly_rich_v2/train.npz`, `validation.npz`, or
`lasso_policy.json`; the feature code fingerprint is unchanged.

## Train CASE on all training cases

```bash
OPENBLAS_NUM_THREADS=15 OMP_NUM_THREADS=1 \
caffeinate -i python3.14 run_weekly_train.py \
  data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --artifacts artifacts/weekly_rich_v2 \
  --stage case --alpha 0.8 --h 30 \
  --lambda-l1 auto --iterations 250 --max-outer 6
```

CASE uses vectorized optimization with a multi-threaded BLAS library; it
does not launch 15 Gurobi jobs because no OR solver is involved.

## Train VF on ALL 180 training group-weeks, using 15 independent cores

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
caffeinate -i python3.14 run_weekly_train.py \
  data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --artifacts artifacts/weekly_rich_v2 \
  --stage vf --alpha 0.8 --h 30 \
  --lambda-l1 auto --iterations 250 --max-outer 6 \
  --workers 15 --threads 1 --seconds 180
```

This is full-cohort training, not a restricted policy grid or pilot. It
uses multiprocessing across the group-weeks; each Gurobi process uses one
thread. Certified plans are checkpointed in
`vf_plans_outer_00.json`, `vf_plans_outer_01.json`, etc. and the
policy/majorization history is checkpointed in `vf_progress.json`.

If any weekly primary-oracle solve fails certification, rerun the **same**
command with `--seconds 600`; completed plan solves are reused. No
uncertified solve is silently treated as optimal. Keep the alpha, h,
lambda, iterations, and max-outer settings identical when resuming.

## Evaluate on all 54 held-development group-weeks (15 cores)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
caffeinate -i python3.14 run_weekly_train.py \
  data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --artifacts artifacts/weekly_rich_v2 \
  --stage evaluate --alpha 0.8 --h 30 \
  --lambda-l1 auto --iterations 250 --max-outer 6 \
  --workers 15 --threads 1 --seconds 180
```

Every method in a group-week is scored against a *single* realized-duration
hindsight oracle. The cost, regret, VF certificate, case MAE, overtime, idle,
moved cases, and block-overload counts are recorded. The exact weekly
planner's existing tie-breaking policy is preserved. The validation summary
contains a paired calendar-week bootstrap interval, clustering all three
hospital groups within each of the 18 resampled calendar weeks.

Both weekly training and validation write finished jobs as they complete,\nnot in task-submission order; a slow group-week cannot block saving results\nfrom faster weeks. A completed VF optimization file carries an explicit\nrun-complete marker and stopping reason (not a global-optimality certificate).\n\nThe evaluation is checkpointed per group-week in
`evaluation_all_validation.csv`; running it again with `--seconds 600`
retries the incomplete weeks only. Final summaries are in
`evaluation_summary.json`. **The 2013 holdout is untouched**.

`--workers 15 --threads 1` allows 15 parallel Gurobi processes, subject to
hardware and licensing; it does not force all 15 cores to remain saturated.
