# Saturation-only (B) surgeon-response grid

This is one controlled training-by-evaluation scenario experiment, using
only the monotone saturating response, with no C/non-monotone response,
scenario banks, SHARED, or additional optimizers.

## The four policy families

- BOOKED: original case durations, unchanged by response.
- NAIVE: a single response-unaware statistical recommendation trained
  using the same 1.75 underprediction / 1.0 overprediction loss and
  L1 calibration as CASE, but without modeling surgeon response.
  The raw predicted correction is displayed directly.
- CASE: train separate response-aware models using the exact convex-QP
  pDCA and the full saturating B link. The CASE penalty is 1.0 times
  the original training-only calibrated L1, restoring the earlier
  stronger-penalty fit rather than the later 0.75 setting.
- VF: separately train a schedule-aware five-feature policy ON TOP
  OF ITS OWN CASE in each training scenario; schedule-library MM and
  pDCA with lambda 1.0 times the training-calibrated VF penalty.
  Every VF theta starts at zero. No multidirection VF variant.

CASE and VF each have FOUR fitted policies; NAIVE has ONE.

## Response and scenarios

Under scenario (alpha,h), the effect of displayed advice u is

    F_B(u) = clip(alpha*u, -alpha*h, +alpha*h).
    reviewed = max(1, booked + F_B(u)).

No fitted shared raw advice score is constrained casewise to +/- h.

Four train scenarios (2x2):
- SAT_A50_H30: alpha=.5, h=30
- SAT_A50_H60: alpha=.5, h=60
- SAT_A80_H30: alpha=.8, h=30
- SAT_A80_H60: alpha=.8, h=60

Six evaluation scenarios (2x3): alpha=.5 or .8, each with h=20,
30, or 60 (SAT_A50_H20, SAT_A50_H30, SAT_A50_H60,
SAT_A80_H20, SAT_A80_H30, SAT_A80_H60).

Each learned policy creates displayed advice ONCE under its training
scenario. At evaluation, this SAME frozen display passes through each
evaluation scenario. Recomputing the display under the evaluation
scenario would incorrectly adapt the recommendation and invalidate
the training-versus-deployment mismatch comparison.

h=60 is included in training deliberately: a canonical display from
an h=30-trained policy is never larger than 30, so h=60 evaluation
alone would make little difference for that policy at fixed alpha.
h=60-trained policies can display up to 60, revealing the consequences
of narrower h=20/h=30 evaluation responses.

## Data, planner, comparisons

Reuse the existing objective_selection_12h/.../context.pkl, including
the UHN weekly room-day roster, eligibility, surgeon-days, 480-minute
capacity, 30-minute turnover, idle penalty 1 and overtime penalty 1.75.
No older manuscript roster. The evaluation tie rule uses hash seed 0.

Every evaluation response has BOOKED, NAIVE and all eight fitted CASE/VF
policies (ten arms). All comparisons within that scenario use exactly
the same certified components and case denominator. The reported
bootstrap resamples calendar weeks. A further globally matched grid
uses the intersection of certified components across ALL SIX evaluation
scenarios, allowing genuine across-response comparisons without
changing sample coverage.

The 2013 outcomes have been examined before: the results and bootstrap
intervals must be labeled exploratory, not a pristine holdout test or
observed savings from deployed surgeon advice.

## One-command run

Use a NEW output folder, never an older experiment's artifacts.
After fetching the branch, run:

    git fetch origin
    git switch --track origin/feature/response-scenario-grid
    python -m pytest -q tests/test_response_scenario_grid.py
    mkdir -p artifacts/response_saturation_grid
    set -o pipefail
    caffeinate -i python -u pilot/run_response_scenario_grid.py \
      --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
      --out artifacts/response_saturation_grid \
      --cores 15 --plan-seconds 180 \
      2>&1 | tee -a artifacts/response_saturation_grid/run.log

On existing local checkout use git switch feature/response-scenario-grid
followed by git pull --ff-only rather than --track.

This ONE process trains NAIVE, all four CASE and four VF policies,
freezes every display, evaluates ten arms under each of the six B
response scenarios, and generates all reports. The solve cache is
content-addressed by component + duration vector + time budget and
tie-rule; repeated identical planner problems across scenarios reuse
their exact saved certificates. Resuming the same run reuses completed
training and evaluation cells.

## Primary outputs

- run_contract.json: scientific configuration, input/source checksums
- naive_policy.json: one asymmetric response-unaware LP fit
- training/<training scenario>/case_policy.json: exact CASE pDCA history
- training/<training scenario>/vf/vf_library_progress.json: VF MM history
- training/<training scenario>/vf_features.json: frozen CASE draft scaler
- training/<training scenario>/frozen_policy.json: CASE and VF coefficients
- selected_policies.json: complete training policy bank
- frozen_advice/evaluation_displays.csv: unchanged displayed advice on 2013
- evaluation_cells/<evaluation scenario>/<method>.json: per-weekly-component
  realized modeled OR cost, idle and overtime, error and solver status
- summary.json: common certification per scenario and paired bootstraps
- cost_grid.csv and mae_grid.csv: ten-arm within-scenario cost/MAE
- cost_grid_global_matched.csv: across-scenario cost on identical components
- plans/content_addressed/: genuine cache for all distinct weekly problems
- run.log and stage_wall_times.json: progress, certification and runtime

No full-scale result has been run by GitHub CI; CI covers mathematical
and integration tests without relying on the user's licensed Gurobi.
