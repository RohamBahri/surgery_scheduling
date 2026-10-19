# Full pre-2013 fitting, operational audit, frozen 2013 final test

**Status:** New experiment, independent of the 2012 development-validation pilot. The previous 2012 validation cases join the full training cohort; there is **no validation split or validation-based tuning**. Do not reuse fitted 60-week models, feature encoders or old validation score files in this experiment. The retained cohorts should contain 24,506 pre-2013 training cases over **79** calendar Mondays and 8,774 final 2013 cases. The reader fails closed if the counts differ.

## Preregistered analysis (unchanged after any 2013 scoring)

Response: inverse-link with alpha 0.8, h 30; gamma 2 for GAP/ANCHORED; same recovered week rosters and deterministic downstream planner. Policies: BOOKED, CASE-BIC no context, CASE-BIC with two booked-only surgeon-week features, a capped inverse-link LASSO-BIC with the same cohort features, ANCHORED with cohort context at ridge 0.001 (primary), ANCHORED at ridge 0.003 (sensitivity), VF with cohort context, VF-GAP with cohort context. Three deterministic planner-hash seeds 0/1/2 are *not* independent samples. CASE's asymmetric quasi-BIC penalty is fitted on **training only**, across the original grid.

Larger but finite budgets: CASE 650 inner iterations, ANCHORED four outer / 20 inner rounds with two separation rounds and a batch of 192, VF four outer / 180 inner iterations, GAP three outer / 120 inner iterations with dual-bound refinement. Planner 600s per independent MIP; final evaluation 900s, with a fixed 1800s escalation only for still-incomplete components; training reference building similarly retries unresolved components once at 1200s. Use 12 workers × one solver thread. This can take hours on a large machine. If training appears slow, do **not** reduce budgets midway or substitute an uncertified incumbent; interrupt gracefully and inspect the checkpoint/logs. A time-limited schedule without a completed primal and deterministic tie-breaking certificate is not scored as complete. The bound for ANCHORED may remain loose even with all deployment schedules certified; never claim global bilevel optimality.

## Exact commands

From repository root with Python 3.14 and working Gurobi license:

```bash
# Before starting final training
git fetch origin
git switch full-2013-audited
git pull --ff-only
python3.14 -m unittest discover -s pilot -p 'test_*.py' -v
python3.14 -m py_compile pilot/run_final_2013.py pilot/audit_final_2013.py

# Trains ALL pre-2013 cases. No 2013 metric is computed here.
caffeinate -i bash pilot/run_final_2013.sh train

# Training-only assessment, including extreme loads, availability of
# alternative room allocations, 0.5 and 2 minute perturbation sensitivity,
# exact solver completion and training budget hit rates.
caffeinate -i bash pilot/run_final_2013.sh audit
cat artifacts/final_2013_audited_v1/train_audit/audit_summary.json
```

**STOP HERE and review the training audit before opening the 2013 results.** An extreme-load flag does not disappear simply because an exact MIP solver certified its primary objective. The planner's modeled feasibility is not proof of a hospital-operationally feasible schedule. If a substantive feasibility/model defect is discovered, revise the *training* experiment, document a new model version, and redo its audit before the 2013 test. The old 2012 development results then become exploratory.

If the audit is complete and any operational limitations are acknowledged and accepted as conditions of this retrospective experiment:

```bash
caffeinate -i bash pilot/run_final_2013.sh freeze --acknowledge-audit
caffeinate -i bash pilot/run_final_2013.sh test
```

If the audit has no flags, the extra `--acknowledge-audit` option is unnecessary. The one-time frozen test computes all methods on the same 2013 cases, with exact planner certificates and calendar-Monday clustered paired confidence intervals. Do not select or tune on test outcomes. Repeating interrupted work with the exact same `ROOT` resumes checkpoints without changing policies.

## What to upload after training audit

Send `training_manifest.json`, `cohort/case_policy.json`, `train_audit/audit_summary.json`, and each model's `*_progress.json`. After test, also send `FINAL_FREEZE.json`, `test/test_summary.json`, and `test/test_scores.json`. A ZIP of the complete new artifact folder is easiest, but can be large from solver caches; a ZIP of these reports is enough.

## Interpretation and caveats

The workbook is parsed by the preexisting cohort builder, but no 2013 outcomes enter the training feature matrix, fitted model, response-parameter selection or training audit. Recovered operating-room rosters and eligibility rely on historical performed-room records, so ex-ante availability needs a separate institutional check. Capped-prediction stress tests and overload diagnoses use true durations only on training dates and cannot justify selecting a model after test. The paper must distinguish mathematical assignment feasibility, schedule-load realism and surgeon responses assumed rather than observed.