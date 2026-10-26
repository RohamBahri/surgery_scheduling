# VF-MM-pDCA with zero VF L1 penalty

One targeted diagnostic experiment: eliminate only lambda_VF (set it to
exactly 0.0) in the original VF MM-pDCA method to investigate whether L1
shrinkage explains the small original VF coefficient magnitude.

## Unchanged from the existing B/C grid

- Four original B response training scenarios alpha=.5/.8 x h=30/60.
- Original trained CASE coefficients (CASE lambda x1.0), never retrained.
- Same five CASE-draft features/scalers, BOOKED/CASE draft schedules,
  weekly planner, physical room roster, eligibility and objective costs.
- VF initial theta=0, coefficient box +/-64, outer MM and inner pDCA,
  default stopping logic and Gurobi solver. No ray/grid search.
- All old NAIVE, CASE, and original VF coefficients and evaluation cells.
- Six B and six non-monotone C deployment scenarios, applying each
  learned policy's frozen training-B display to the assumed response.
- Same content-addressed weekly planner cache.

Four NEW VF_ZERO policies are trained; they are evaluated along with the
ten old arms (BOOKED, NAIVE, four CASE, four original VF) on a shared
certified sample across all 12 evaluation scenarios.

For each new VF policy, we also calculate the direct historical realized
TRAINING scheduling cost from the deployed hash-zero planner. This cost is
a diagnostic output; it is NOT used to update theta. If complete
certification cannot be obtained, the reported value remains null.

## One command

Use the SAME original response_saturation_grid output folder and do not
run anything else that writes there simultaneously.

    git fetch origin
    git switch --track origin/feature/vf-zero-regularization
    python -m pytest -q tests/test_vf_zero_lambda.py

    set -o pipefail
    caffeinate -i python -u pilot/extend_vf_zero_lambda.py \
      --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
      --out artifacts/response_saturation_grid \
      --cores 15 \
      --plan-seconds 180 \
      2>&1 | tee -a artifacts/response_saturation_grid/run.log

If already locally checked out, instead git switch
feature/vf-zero-regularization and git pull --ff-only.

The new runner REQUIRES the existing B/C grid to have completed its
old policy fits and 120 evaluation cells; it will never replace or
retrain the original fitted policies. It stores every new training
checkpoint and component evaluation cell under vf_zero_extension/,
with a new scientific contract. Rerun the exact same command to
resume a completed stage or partial planner solve.

## Results

Under artifacts/response_saturation_grid/vf_zero_extension:

- training/<scenario>/vf/vf_library_progress.json: MM-pDCA ledger
- training/<scenario>/frozen_zero_policy.json: new coefficient, original
  coefficient, unregularized training surrogate, direct training cost
- selected_zero_policies.json: four fitted zero-L1 policies
- frozen_zero_displays.csv: canonical displayed advice
- evaluation_cells/<scenario>/VF_ZERO__*.json: realized cost, idle,
  overtime, MAE, planner certification
- summary.json and cost_grid_global_matched.csv: 12 B/C scenarios by
  14 arms, same certified cohort and paired week bootstrap differences
- stage_wall_times.json and run_contract.json

The nonconvex MM-pDCA fit remains a local algorithm; zero L1 does NOT
turn it into a globally optimal optimizer. Numerical guarantees come
from actual descent and planner bounds, not the penalty choice. The
previously reused 2013 outcomes are exploratory, and the response
scenarios are not observed advice-adoption records.
