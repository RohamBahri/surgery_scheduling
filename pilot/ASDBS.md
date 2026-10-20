# ASDBS: full-cohort offline development experiment

Work on existing branch `full-2013-audited`. This experiment does **not** modify the
old VF/ANCHORED pipelines, create branches, or score any 2013 outcomes.

## Fixed method

* Canonical UHN cohort: 24,506 pre-2013 cases, 79 calendar Mondays.
* Odd/even calendar-week folds: each fit uses roughly half the pre-2013 cases;
  the other half is scored only after fitting. Both folds together evaluate
  the complete 24,506-case development cohort.
* Rich features: `leave_date_out` applied to **all** fitting-fold dates,
  earlier and later. Evaluation-case outcomes are never used in the encoder.
* Cohort booking context plus two predeclared dispersion x surgeon-load
  interactions, included in **both** CASE and ASDBS.
* Nominal inverse response: alpha=0.8, h=30; same exact hash-selected
  deployed planner for both methods, hash seed 0.
* Full-dimensional linear policy, initialized from fitted risk-aware CASE.
* Scenario model: full-offline surgeon-week log-residual bootstrap; donors
  for a target week come from *other fitting weeks*. Shared donor week per
  surgeon/replication. The conditional distribution is fixed through search.
* Scenario banks: A=50, fresh independent B=50 per iteration.
* Penalty = 0.001 ||X(w-w_CASE)||^2/(2N).
* Acceptance: paired bank-B gain > 1.645 standard errors, one proposed
  policy per iteration, ten iterations. This is a **Monte Carlo heuristic**,
  not a 95% population improvement certificate.
* Initialization attempts two reachable-box extreme schedules for each flexible
  component, plus at most 256 small SAA aspirations and 120 constructive
  per-case searches. These are **search accelerators**, not sampling the
  training objective. Every training case contributes to the objective and
  full-planner acceptance. Timed-out *optional* aspiration solves are skipped
  and counted; a timed-out **deployed** solve blocks progress, never silently
  removes the component.
* Decision-boundary search is a bounded, partial search over directions and
  candidate crossings, **not** exhaustive exact optimization of the bilevel
  objective. All accepted policy outcomes are determined by the certified
  full deployed planner.

## Exact commands (from repo root)

The following requires Python 3.14, the project's standard packages, and
a working Gurobi license. Run folds sequentially.

```bash
git switch full-2013-audited
git pull --ff-only
python3.14 -m unittest pilot.test_asdbs -v
mkdir -p logs
set -o pipefail

caffeinate -i python3.14 pilot/run_asdbs.py --stage fit --fold 0 --workers 12 --seconds 600 2>&1 | tee logs/asdbs_fold0_fit.log
caffeinate -i python3.14 pilot/run_asdbs.py --stage score --fold 0 --workers 12 --seconds 600 2>&1 | tee logs/asdbs_fold0_score.log

caffeinate -i python3.14 pilot/run_asdbs.py --stage fit --fold 1 --workers 12 --seconds 600 2>&1 | tee logs/asdbs_fold1_fit.log
caffeinate -i python3.14 pilot/run_asdbs.py --stage score --fold 1 --workers 12 --seconds 600 2>&1 | tee logs/asdbs_fold1_score.log
```

Outputs are under `artifacts/asdbs_79week_v1/fold_0/` and `fold_1/`.

Training checkpoints are saved in `asdbs_library_initialization.json`
and `asdbs_progress.json`; deployment decisions are cached in
`training_deploy/` and `heldout_deploy/`. Re-run the **same** command to
resume after an interruption. If a certified deployed solve times out, repeat
with `--seconds 1200` or `--seconds 1800` (without changing `--root`).
Do not interpret an unfinished run as a numerical result.

## Send back for review

* Both `fold_0/heldout_results.json` and `fold_1/heldout_results.json`.
* Both `fold_*/asdbs_progress.json` files.
* Both `fold_*/asdbs_library_initialization.json` files.
* Four `logs/asdbs_fold*_*.log` files, or the error/timeout log if stopped.

The decisive comparison is paired **held-out realized cost per case**, ASDBS
minus risk-aware CASE, in both folds. The fold bootstrap is descriptive and
does not select any tuning parameter. Do not claim a confirmation on 2013
until the method is frozen and separately evaluated.

## Scope of current verification

The new Python files have been committed, and solver-free unit tests have
been added. The authoring environment does not contain Gurobi or a GitHub
clone, so no real Gurobi deployment, end-to-end execution, or numerical ASDBS
improvement has been verified here. Run the unit tests before launching
the full two-fold experiment.
