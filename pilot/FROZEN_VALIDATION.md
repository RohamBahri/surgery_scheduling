# Frozen 2012 development-validation run

This is the next experiment after the odd/even training-fold pilot.
The scripts **do not modify any of the prior pilot artifacts**.
The untouched 2013 final test is not evaluated.

## Preregistered, fixed methods

All five policies are trained or evaluated under the nominal inverse response
(alpha 0.8, h 30) and deterministic exact downstream planner:

- BOOKED, no learned recommendation.
- CASE-BIC, case-only original 71 rich predictors.
- CASE-BIC, original predictors + cohort-only surgeon-week booked minutes
  and surgeon-week case count.
- ANCHORED, same cohort-only predictors, CASE-centered ridge **0.001** (primary).
- ANCHORED, same predictors, ridge **0.003** (sensitivity only).

CASE performs the same training-only asymmetric-loss BIC grid and receives
500 iterations per smoothing pass; ANCHORED has the same exploratory
2-outer-round, 12-pDCA-step/one-closure-round budget as before.
Ridge 0.001 was selected as the central value *before* this independent
validation experiment. **Do not change model selection after validation.**

## Commands (from repository root)

```bash
git fetch origin
git switch decision-model-pilot-v1
git pull --ff-only
python3.14 -m py_compile pilot/run_frozen_validation.py

# Fits on all 60 original training calendar weeks and creates the freeze.
caffeinate -i bash pilot/run_frozen_validation.sh train

# The following file must exist BEFORE the next command:
cat artifacts/frozen_context_validation_v1/frozen_policies.json

# Explicit one-time independent 18-week development validation.
caffeinate -i bash pilot/run_frozen_validation.sh validate
```

The TRAIN stage handles resumable ANCHORED references and model checkpoints
and writes `frozen_policies.json`. VALIDATE checks that the training
models, feature settings, source hashes, and workbook still match the
frozen file before loading `validation.npz`. No fit stage will run again
after the freeze. Models train on 18,705 cases/180 site-weeks and evaluate
on 5,779 validation cases/54 site-weeks.

No need to run a separate summary stage; validation writes it automatically.
If interrupted, rerun the same stage with the same output root. Run
`bash pilot/run_frozen_validation.sh summary` only after all validation
methods finish.

## Files to send

- `artifacts/frozen_context_validation_v1/frozen_policies.json`
- `artifacts/frozen_context_validation_v1/validation/validation_summary.json`
- `artifacts/frozen_context_validation_v1/validation/validation_scores.json`

For an optimization audit, also send
`cohort/sensitivity/ridge_0p001/anchored/anchored_progress.json`
and `cohort/sensitivity/ridge_0p003/anchored/anchored_progress.json`.

## Research cautions

No validation-based choices of ridge, features, or policy are permitted.
Three hash seeds represent alternative deterministic planner tie-breaks,
not independent observations. Pairwise uncertainties use the **18
calendar weeks** as bootstrap clusters; their intervals will be wide.
The two cohort features assume the full weekly booking list is known
at the advice time, but do not depend on reconstructed room eligibility.

This does not prove that surgeons would respond to advice as specified,
because their historical responses to displayed recommendations were not
observed. Report improvements as conditional on the scenario and the
retrospective planner. This development-validation experiment is not
an independent 2013 final-test confirmation.
