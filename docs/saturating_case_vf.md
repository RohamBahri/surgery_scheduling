# CASE(B) + VF(B): unrestricted-score saturating-link experiment

This is a **new**, deliberately narrow response-link comparison. It does not
replace the existing nominal non-monotone (C) four-method experiment. Use
its own Git branch and output folder. Only the two learned arms CASE_B
and VF_B are trained; BOOKED is the unchanged reference.

## Exact response specification

For each case, the fitted shared linear raw score s remains unrestricted
except for the same fixed coefficient box as earlier studies. No constraint
of the form -h <= s_i <= h is ever imposed on training or evaluation.

The new nominal link is
    F_B(s) = clip(alpha*s, -alpha*h, +alpha*h),
with alpha=0.8, h=30, hence a maximum nominal correction of +/-24 minutes.
This is **monotone saturation**, not a hard shared coefficient restriction
(A) and not the prior non-monotone adoption/attenuation/rejection (C):
    F_C(s) = sign(s)*min(alpha*abs(s), max(h-(1-alpha)*abs(s),0)).
For a raw score 60, B revises by +24 minutes, whereas C revises by
+18 minutes; at 150, B still revises +24 and C revises by zero.
The revised duration is max(1, booked+F_B(s)), applying the floor AFTER
response. The policy displays the minimal equivalent advice
F_effective(s)/alpha, so a raw score above h simply displays +h and
produces the saturated correction. There is no *casewise* bound on s.

All continuous piecewise-linear hinge coefficients P and N are compiled
from F_B by the same exact DC machinery. The CASE pDCA convex QP retains
the reusable sparse raw-score auxiliaries and every floor breakpoint;
no approximation or inverse-response shortcut is introduced. VF(B)
optimizes the same schedule-aware finite-library error envelope as the
preceding experiment and uses MM + pDCA from theta=0 on the frozen
CASE(B) raw scores and **one** CASE(B) draft. It is not initialized from
VF(C) or any prior coefficients.

## Fixed training-only regularization decisions

The earlier CASE penalty reduction from 1.0 to 0.75 of the
training-calibrated lambda is retained as-is for this link comparison.
VF(B) uses **0.50** of the same training-only Rademacher-calibrated
five-feature VF lambda. This is a single modestly weaker, fixed VF penalty
for the saturation experiment, not a grid search. Both lambda scales
are recorded in run_contract.json, and exact fitted values are saved
in case_policy.json and selected_policy.json.

These choices are *not validated* against an unseen outcome set. The
2013 evaluation period has already been inspected during research, so
its bootstrap intervals remain exploratory and must not be presented as
confirmatory evidence.

## Unchanged components

The input is the exact same 12h objective-selection context as the latest
SHARED/VF experiments. Week partitions, 480-minute rostered blocks,
30-minute turnover, eligibility and surgeon-day constraints, objective
weights (idle=1 and overtime=1.75), physical-room planner and hash-0
deployment tie rule are untouched. The obsolete manuscript roster is
not used. Initial VF libraries contain the BOOKED and CASE(B)-draft
schedules; VF expands them with the current primary planner and records
Delta/epsilon and pDCA inner progress. No response banks or extra methods.

## Run once end to end

    git fetch origin
    git switch feature/saturating-case-vf
    git pull --ff-only origin feature/saturating-case-vf

    python -m pytest -q tests/test_nonmonotone_learning.py tests/test_saturating_case_vf.py

    mkdir -p artifacts/saturating_case_vf
    set -o pipefail
    caffeinate -i python -u pilot/run_saturating_case_vf.py \
      --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
      --out artifacts/saturating_case_vf \
      --cores 15 --plan-seconds 180 \
      2>&1 | tee -a artifacts/saturating_case_vf/run.log

The one runner fits CASE(B), freezes its hash-0 provisional assignments,
creates the training-only standardized five features, trains VF(B) with
MM-pDCA at theta=0, freezes both policies and evaluates BOOKED, CASE(B)
and VF(B) on the 2013 components and their shared certified sample.

Checkpoints and exact policy parameters live under:
- run_contract.json (input hash, source hash and scientific parameters)
- case_policy.json (CASE(B) objective, weights, lambda, solver history)
- vf_features.json (single frozen train scaler)
- vf_b/vf_library_progress.json (VF(B) weights, MM/DCA history)
- selected_policy.json (final CASE(B) and VF(B) coefficients)
- cases_train.csv and cases_test.csv (raw unbounded scores, canonical
  displays, reviewed durations, score-outside-h indicators and five features)
- evaluation_2013.json (component-level realized cost and solver certification)
- summary.json (per-case cost/MAE, common certified coverage, calendar-week
  bootstrap for VF(B)-CASE(B) and both against BOOKED)
- stage_wall_times.json, run.log, plans/ (full solver audit trail)

This link is only a modeling assumption, not measured surgeon behavior.
Any observed benefit is a counterfactual scheduling result, not a causal
hospital saving. We must compare B versus C only on a matched, certified
component sample, and separately account for the fixed VF penalty change.
