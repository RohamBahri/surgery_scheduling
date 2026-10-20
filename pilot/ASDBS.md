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
* Within each of the ten rounds, search up to **two distinct** bank-A proposals.
  The full deployed planner is solved for each; new assignments enlarge the
  library before the second search. Choose only proposals that improve the
  full-planner bank-A objective. Screen the best with **one** fresh bank B.
* Acceptance: paired bank-B gain > 1.645 standard errors; ten outer rounds.
  This is a **Monte Carlo heuristic**,
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

This is a **repair rerun**. The original `artifacts/asdbs_79week_v1`
results remain untouched. This experiment requires Python 3.14 and Gurobi.

```bash
git switch full-2013-audited
git pull --ff-only origin full-2013-audited
python3.14 -m unittest pilot.test_asdbs -v

mkdir -p logs artifacts/asdbs_refined_v1/fold_0 artifacts/asdbs_refined_v1/fold_1
# The feature specification has not changed. Reuse the already-fitted CASE policies.
cp artifacts/asdbs_79week_v1/fold_0/case_policy.json artifacts/asdbs_refined_v1/fold_0/
cp artifacts/asdbs_79week_v1/fold_1/case_policy.json artifacts/asdbs_refined_v1/fold_1/
set -o pipefail

caffeinate -i python3.14 pilot/run_asdbs.py --stage fit --fold 0 --workers 12 --seconds 600 2>&1 | tee logs/asdbs_refined_fold0_fit.log
caffeinate -i python3.14 pilot/run_asdbs.py --stage score --fold 0 --workers 12 --seconds 600 2>&1 | tee logs/asdbs_refined_fold0_score.log

caffeinate -i python3.14 pilot/run_asdbs.py --stage fit --fold 1 --workers 12 --seconds 600 2>&1 | tee logs/asdbs_refined_fold1_fit.log
caffeinate -i python3.14 pilot/run_asdbs.py --stage score --fold 1 --workers 12 --seconds 600 2>&1 | tee logs/asdbs_refined_fold1_score.log
```

The revised artifact root is `artifacts/asdbs_refined_v1`. If you want to
refit CASE rather than reuse its frozen identical features, omit the two
`cp` commands; this is not needed for this algorithm-only repair.

Full-planner decisions and training progress are cached. An interrupted fit
can be resumed by repeating the same command; verified inner refinements
are checkpointed after each pass. An unresolved exact deployed solve
blocks the run; retry with `--seconds 1200` if necessary.

## Send back for review

```bash
tar -czf asdbs_refined_review.tgz \
  artifacts/asdbs_refined_v1/fold_*/heldout_results.json \
  artifacts/asdbs_refined_v1/fold_*/asdbs_progress.json \
  artifacts/asdbs_refined_v1/fold_*/asdbs_library_initialization.json \
  logs/asdbs_refined_fold*.log
```

Upload `asdbs_refined_review.tgz`. If training stops before either
fold is complete, send the last failed log instead.

The outcome remains paired held-out realized cost of ASDBS versus
risk-aware CASE in **both folds**, with no tuning on fold results.
The previous experiments informed this repair, so the rerun is exploratory
and not an independent confirmation. Do not run on 2013 yet.

## Scope of current verification

The new Python files have been committed, and solver-free unit tests have
been added. The authoring environment does not contain Gurobi or a GitHub
clone, so no real Gurobi deployment, end-to-end execution, or numerical ASDBS
improvement has been verified here. Run the unit tests before launching
the full two-fold experiment.
