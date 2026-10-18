# Decision-model pilot: training-only experiment

This branch adds a **short-run, isolated experiment** comparing response-aware
CASE, VF, VF-GAP, and ANCHORED under identical training data and deterministic
weekly planners. It varies two policy links (original versus inverse-monotone)
and two predictor sets (71 case features versus those plus four retrospective
weekly-context features).

The response and ridge changes are already applied to the Python source files
in this branch. **Do not run an installer.** The original nonmonotone response
remains the default outside the explicitly configured pilot.

## Getting started

From the project root:

1. Pull/check out `decision-model-pilot-v1`.
2. Run `python3.14 -m unittest discover -s pilot -p 'test_*.py' -v`.
3. Run `PYTHON=python3.14 bash pilot/run_fast_pilot.sh`.
4. Run `python3.14 pilot/summarize.py artifacts/decision_model_pilot_v1`.

The runner expects the frozen training feature set in
`artifacts/weekly_rich_leakfix_v1/`, the training reference library at
`artifacts/anchored_leakfix_v1/anchored_references.json`, and the historical
workbook at `data/UHNOperating_RoomScheduling2011-2013.xlsx`.
Override their locations with `ARTIFACTS`, `REFS`, and `WORKBOOK`.
All new outputs use `ROOT=artifacts/decision_model_pilot_v1` by default.

## What it does

**Phase I:** fits four CASE models, choosing their L1 penalty along a
predeclared grid with a *quasi-BIC proxy* based on training asymmetric
post-response loss. Compares their verified realized training regret with
BOOKED under the same hash seeds.

**Phase II:** starts VF, VF-GAP, and ANCHORED from the corresponding CASE
coefficients using the inverse-monotone response link, once without weekly
context and once with it. ANCHORED uses a residual-to-CASE ridge penalty.
The run uses 2 outer iterations, 35 inner iterations for VF/GAP, 12 pDCA
iterations and 128 separation components for ANCHORED. The pilot uses three
hash seeds (0, 1, 2) for planning evaluation. These are short exploratory
budgets and **do not establish convergence**.

The four extra standardized weekly features are:
- Site-week booked workload divided by 480 times staffed room-days;
- Surgeon-week booked workload divided by 480 times eligible room-days;
- Surgeon-week number of booked cases;
- Number of eligible surgeon room-days.

**Information limitation:** the operative room-day roster and surgeon
eligibility are **retrospectively reconstructed from performed surgery
timestamps**. These weekly features use booked durations rather than realized
durations, but they are *not verified as prospectively observable*. Any gain is
conditional on the reconstructed roster; do not claim deployment feasibility
without prospectively observed roster data.

## Behavioral link

The inverse link interprets the shared score as a desired effective
correction and displays `clip(score/alpha, max(1-booked,-h), h)`.
With correctly specified response parameters, the effective correction is
`clip(score, alpha*max(1-booked,-h), alpha*h)`.
This removes the decreasing-response region without constraining the latent
linear coefficient model. It is not automatically robust if the actual
behavioral threshold differs from the assumed one. The exact ANCHORED hinge
decomposition and the CASE/VF/GAP training derivatives are adjusted together.

## Scientific limitations

- CASE's penalty selector is a **BIC-like approximation**, not the exact
  Gaussian LASSO BIC nor a fully justified effective-degrees-of-freedom
  estimate in a saturated model.
- Existing leave-date-out feature encodings retain the same
  training/deployment feature-stability considerations.
- Seeds represent arbitrary planner tie-breaking, not independent samples.
- VF-GAP may pause when solver intervals overlap. Missing or incomplete
  methods must remain reported as incomplete, not silently ranked.
- Reused reference assignments and hindsight costs are training-only.
- The code reads the workbook to reconstruct weekly cases, but only
  *training-site-week outcomes* are fitted or scored. It does not evaluate
  the 2012 validation weeks or the 2013 holdout.

## Files to report

Start with `artifacts/decision_model_pilot_v1/pilot_summary.json`.
For reproducibility also send the four `case_bic_path.json` and
`case_policy.json` files, plus the two inverse-link sets of:
`vf/vf_progress.json`, `gap/gap_progress.json`,
`anchored/anchored_progress.json`, and each
`pilot_training_comparison.json`.

Use the pilot to determine which learner merits a longer run. Do not use
these limited-iteration outcomes as a final manuscript comparison.
