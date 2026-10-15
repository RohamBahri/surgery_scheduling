# ANCHORED research v3: repaired offline features and reproducible experiments

Branch: anchored-research-v3. Original main, anchored-exact-adaptive and
anchored_ALL_hash_v2 outputs remain unchanged. Use a new OUT directory for
every distinct experiment. This is a research toolkit, not a claim of improved
generalization or proof of coefficient optimality.

## 0. Update code

    git status --short
    git fetch origin
    git switch anchored-research-v3
    git pull --ff-only origin anchored-research-v3
    python3.14 -m pip install -r requirements-lock.txt
    python3.14 -m pip install -e . --no-deps
    python3.14 -m pytest -q
    bash run_research_v3.sh features --dry-run

## 1. Complete fresh feature and baseline fitting, in order

    bash run_research_v3.sh features
    bash run_research_v3.sh lasso
    bash run_research_v3.sh case
    bash run_research_v3.sh vf
    bash run_research_v3.sh gap

These operations produce a new feature root, by default
artifacts/weekly_rich_leakfix_v1, and a newly fitted
artifacts/weekly_gap_leakfix_v1. The feature pipeline DROPS exactly
three service-only supervised columns: hist_service_mean_error,
hist_service_mean_abs_error, hist_service_under_rate. Legacy procedure,
surgeon and pair encoding remains unchanged, under further audit. It
does not impose expanding chronological histories. The code and case
hashes invalidate incompatible checkpoints. Do not reuse old 74-feature
coefficients against the new feature matrix.

## 2. Run diagnostics without Gurobi

    bash run_research_v3.sh audit

The audit rebuilds the encoder, verifies the matrices, contrasts training
leave-date-out with deployment-like full-offline encodings on the same
training cases, and reports score shifts and response regimes for models
that already exist. The result is research_feature_audit.json under OUT.

## 3. Baseline ANCHORED using the repaired features

    bash run_research_v3.sh references
    bash run_research_v3.sh train
    bash run_research_v3.sh certify
    bash run_research_v3.sh evaluate
    bash run_research_v3.sh sensitivity
    bash run_research_v3.sh audit

Default OUT=artifacts/anchored_leakfix_v1. The training checkpoint stores
accepted coefficient vectors and separate completion/separation status.
Exact verified deployments are reused in certification. Certification
still requires valid separation-dual bounds and does not imply globally
optimal learned coefficients.

## 4. Predeclared algorithm-search experiment

Set a NEW OUT path and preserve ALL variables across stages:

    export OUT=artifacts/anchored_leakfix_search_v1
    export INITIAL=LASSO
    export LINE_FRACTIONS=1.5,1,0.5,0.25,0.125
    export TRUST_GROWTH=1.25
    export TERMINAL_SEPARATION=1
    export TIGHT_SURROGATE=1
    export MAX_TERMINAL_RESTARTS=1
    export REFERENCES_FROM=artifacts/anchored_leakfix_v1
    bash run_research_v3.sh references
    unset REFERENCES_FROM
    bash run_research_v3.sh train
    bash run_research_v3.sh certify
    bash run_research_v3.sh evaluate

REFERENCES_FROM is optional, but strongly recommended for multiple
variants. It copies only fully completed, fingerprint-compatible exact
reference libraries (same workbook, feature matrices, baseline
coefficients, hash seed and planner code). All new policy evaluations
and training oracles still run independently.

Terminal separation is over all active components and can be expensive.
Each component retains a strict solver budget. Unresolved components
are recorded explicitly. Search is bounded, not a convergence proof.

TIGHT_SURROGATE=1 activates the independently implemented S-hat
non-dominating-competitor restriction. Both the master library and
the exact MILP separation oracle exclude competitors whose realized
cost is strictly LOWER than the chosen anchor's cost. The anchor itself
is retained. This is a separate experimental variant; its valid
tie-safe regret bound is checked on enumerated toy schedules and the
full certificate stage uses the same restriction. It is not claimed to
converge globally or generalize automatically.

## 5. Ridge and LASSO anchoring ablation

    export OUT=artifacts/anchored_leakfix_ridge_v1
    export INITIAL=LASSO
    export RIDGE=0.001
    export REFERENCE_PENALTY=0
    REFERENCES_FROM=artifacts/anchored_leakfix_v1 bash run_research_v3.sh references
    bash run_research_v3.sh train
    bash run_research_v3.sh certify
    bash run_research_v3.sh evaluate

RIDGE adds nonintercept coefficient L2 regularization. Optional
REFERENCE_PENALTY penalizes score deviation from the frozen refitted
LASSO policy. The outer acceptance test still requires a strict
UNREGULARIZED exact training regret improvement; it does not guarantee
monotone regularized regret. These are ablations, not tuned defaults.
The +/-150 coefficient box remains unchanged for comparison.

## 6. Nonlinear predictive baseline

After feature generation and baseline fitting:

    export OUT=artifacts/anchored_leakfix_v1
    bash run_research_v3.sh boosted-fit
    bash run_research_v3.sh boosted-evaluate

Fixed-configuration HistGradientBoostingRegressor predicts booking
residuals. Training predictions are grouped 5-fold calendar-week OOF;
the predictor used for 2012 validation is fitted to all 18,705
training cases, without validation labels. This is a forecasting
baseline, not a decision-aware boosted correction policy.
The evaluation re-solves the exact deployed weekly planner.

## 7. Frozen-policy behavioral sensitivity

    export OUT=artifacts/anchored_leakfix_v1
    export SCENARIO_GRID=0.5:30,0.8:30,1:30,0.8:60
    bash run_research_v3.sh scenario-audit

This compares frozen ANCHORED, CASE, LASSO and VF policies across
response scenarios. It does NOT train one joint multi-scenario robust
policy. The 2012 validation has already been examined; do not
use repeated validation looks for unrestricted hyperparameter tuning.

## 8. Small-component case-level reachable-grid headroom

After ANCHORED training, optional training-only finite-grid enumeration:

    export OUT=artifacts/anchored_leakfix_v1
    export MAX_FLEX_COMPONENTS=5
    export MAX_FLEX_CASES=3
    export GRID_LEVELS=3
    bash run_research_v3.sh flexible-audit

The script selects small, high-gap training components and evaluates
every combination from the stated per-case reachable-duration GRID
through the exact hash-tie deployment planner. The result distinguishes
certified grid evaluations from timed-out candidates. This is an
exhaustive FINITE GRID diagnostic only; it is NOT an optimal solution
to the underlying continuous case-specific bilevel problem.

## 9. Remaining research directions

- Audit remaining supervised outcome encodings and proportional booked
  interactions; removing three columns does NOT prove all self-influence
  eliminated.
- Compare strong prediction baselines under the same response/planner,
  use paired calendar-week uncertainty, and preserve 2013 untouched.
- Solve exact inducibility diagnostics on small high-regret components
  before expanding policy expressiveness.
- Audit the tight S-hat master/oracle ablation empirically and retain
  its proof and enumeration checks with the final manuscript.
- Develop joint behavior-robust training only after fixed-policy
  scenario sensitivity demonstrates sufficient value.
- Preserve schedule-invariant, case-level recommendations and the
  unchanged original planner for all controlled comparisons.
