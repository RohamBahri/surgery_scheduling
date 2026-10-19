# Decision-model pilot v2 (training-only, held-out weeks)

**Run the new `pilot/run_fold_pilot.sh`; do not invoke the old in-sample
`pilot/run_pilot.py` directly.** The old launcher redirects to the new
runner. Source changes are already in the branch; no ZIP/installer is needed.

## Pull, test and run

From the repository root:

```bash
git fetch origin
git switch decision-model-pilot-v1
git pull --ff-only
python3.14 -m unittest discover -s pilot -p "test_*.py" -v

# Phase I: statistical CASE, original/inverse link, no/cohort/roster features
caffeinate -i bash pilot/run_fold_pilot.sh case
caffeinate -i bash pilot/run_fold_pilot.sh score

# Phase II: decision-aware learners, NO context first
CONTEXTS=none caffeinate -i bash pilot/run_fold_pilot.sh decision
caffeinate -i bash pilot/run_fold_pilot.sh score

# Phase III, only if warranted: cohort and reconstructed-roster context
CONTEXTS="cohort roster" caffeinate -i bash pilot/run_fold_pilot.sh decision
caffeinate -i bash pilot/run_fold_pilot.sh score

python3.14 pilot/summarize_folds.py artifacts/decision_model_pilot_v2
```

Send `artifacts/decision_model_pilot_v2/fold_heldout_summary.json` first.
Also useful: each `*/fold_*/case_bic_path.json`,
`*/fold_*/case_policy.json`, `*/fold_*/heldout_scores.json`,
`*/fold_*/fold_identity.json`, plus the VF/VF-GAP/ANCHORED progress JSONs.

The script uses 500 L-BFGS iterations per CASE smoothing pass; short
decision-aware budgets (VF/GAP 35 inner/2 outer; ANCHORED 12 pDCA,
one closure round, 2 outer, 128-component separation batch). It checks
three hash seeds (0,1,2). Override PYTHON, WORKBOOK, ARTIFACTS, ROOT, WORKERS
as needed. Set METHODS="vf anchored" to omit VF-GAP, for example.

## Cross-fold data protocol

Sort the original training cohort's 60 calendar Mondays, hold out odd
weeks versus even weeks, then swap. Each fit uses 30 calendar weeks
(90 site-weeks), evaluates on the other 30, and **refits the entire
historical feature encoder within the fitting fold**. It also refits
normalizers, CASE BIC penalty selection, and a fold-specific LASSO start.
Decision-aware training, its oracle references and ANCHORED's centered
ridge all use fitting-fold cases only. Evaluation-fold realized durations
are used solely to score already fitted policies.

No development-validation results or 2013 test results are used.
This is an offline odd/even screening comparison, not a time-forward
deployment simulation.

## Methods and important limitations

CASE L1 selection uses a training-only **asymmetric-Laplace quasi-BIC**:
`2*n*log(mean asymmetric error)+df*log(n)`. Counting nonzero effects
as effective degrees of freedom under a clipped link is approximate.

Inverse link: latent score represents desired effective correction;
display is clipped inverse of nominal surgeon response. The folded
LASSO anchor initializes the latent score **without alpha scaling**.
Original nonmonotone response remains the default outside the pilot.

ANCHORED is trained from CASE with CASE-centered coefficient ridge.
Three predeclared penalty strengths (0.0003, 0.001, 0.003) are all
reported as sensitivity experiments; choosing the winning strength from
these same outer-fold scores would create selection bias.

Feature specifications:
- none: refitted case-level features only;
- cohort: surgeon-week total booked minutes and surgeon-week case count;
- roster: cohort features plus site-week booked load/capacity and surgeon
  eligible room-day count.

Context features are standardized on fitting-fold data and transformed
using the same scalers for held-out weeks. **Staffed room-days and
eligibility are reconstructed retrospectively from performed surgeries,
not verified as known at recommendation time.** Thus roster-context
results are conditional on assumed advance roster knowledge.

VF, VF-GAP and ANCHORED are the actual existing optimizers with
shortened search budgets, not surrogate stand-ins. VF-GAP can pause
with overlapping exact-oracle intervals; do not rank incomplete methods.
Hash seeds alter deterministic tie-breaking, not independent datasets.
The summary bootstraps 60 calendar weeks as paired clusters. The
experiment is exploratory and does not establish global optimizer
convergence or superiority on future years.
