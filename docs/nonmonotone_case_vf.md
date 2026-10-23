# Nominal CASE + VF: full-response pDCA and schedule-library MM

Implementation: draft PR #40; main and frozen earlier experiments are unchanged.

## Scientific specification

Only BOOKED, CASE, and VF are compared. One nominal assumed surgeon
response (alpha=0.8, h=30), not surgeon-specific or estimated, is used in
both learning and evaluation. NO response-scenario banks, heterogeneity,
robustness sweeps, or unrelated algorithm arms are in this run.

Use exactly the same historical UHN cohort, physical room-day roster and
eligibility components as the latest SHARED/VF experiments. The old paper
provides ONLY the VF envelope and pDCA/MM algorithm; it does not determine
the roster. All room-day blocks are charged for idle capacity, including
unused blocks. The planner and hash-0 deployment tie rule are reused.

CASE learns an affine RAW displayed-advice score Xw through the full
nonmonotone response (adoption, attenuation, rejection). The UI cap is
180 minutes; a positive-duration floor is applied AFTER response. The
operational displayed correction is the minimum-magnitude preimage of the
learned effective revision under the nominal response. The shared policy
weights are never constrained to a joint casewise reachable interval.

CASE first solves training-only regularized LAD, then uses pDCA with an
EXACT convex hinge-epigraph Gurobi QP, reused across iterations. This
eliminates the approximate 70-variable cutting-plane master that was
stalling on the component-normalized (~hundreds) objective. One LAD
start is used, not a second initialization from old inverse-link CASE.

Recompute ONE CASE provisional plan per component with hash seed 0 and
freeze five new standardized features: historical surgeon spread,
historical procedure spread, provisional block load deviation, and the
two spread-by-congestion interactions.

VF learns five shared display coefficients added to the frozen CASE
score. At zero increment VF exactly equals CASE. The Section 4
schedule-library surrogate is the minimum planned idle/overtime cost
over stored feasible schedules PLUS the full asymmetric realized
case-duration error envelope, normalized by eligibility component,
with an L1 coefficient penalty.

VF starts its feasible schedule libraries with the BOOKED and CASE
provisional assignments. At each outer iteration it adds planner
incumbents (even if not certified optimal), chooses the least-cost
library assignment, optimizes a TOUCHING MM majorizer through
five-variable pDCA with convex cutting planes, and accepts only
verified improvements. A cut-budget hit alone does NOT stop pDCA if
the candidate improves. Record the per-iteration Delta and epsilon.
Missing planner dual bounds use zero, a valid lower bound for
nonnegative idle/overtime cost. None of this implies globally
optimal learned coefficients, finite-run stationarity, or realized
hospital savings.

## Run

    git fetch origin
    git switch feature/nonmonotone-case-vf-mm
    python -m pip install -r requirements-lock.txt
    python -m pip install -e . --no-deps
    python -m pytest -q tests/test_nonmonotone_learning.py

Use a NEW output directory: older experiment results have a different
CASE solver and scientific settings.

First run CASE alone:

    python pilot/run_nonmonotone_learning.py \
      --stage case \
      --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
      --out artifacts/case_vf_exact_qp_v1 \
      --cores 15 --case-steps 200 --backend gurobi

The CASE file case_policy.json contains initial_objective,
objective, the LAD status, pDCA descent history, and inner statuses.
Confirm that pDCA actually improves CASE beyond its LAD initializer
before interpreting any later CASE/VF comparison.

Then train VF:

    caffeinate -i python pilot/run_nonmonotone_learning.py \
      --stage vf \
      --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
      --out artifacts/case_vf_exact_qp_v1 \
      --cores 15 --plan-seconds 180 \
      --case-steps 200 --vf-outers 30 --vf-pdca-steps 50 \
      --cuts 80 --backend gurobi

After fitting is frozen, do the single nominal 2013 evaluation:

    python pilot/run_nonmonotone_learning.py \
      --stage evaluate \
      --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
      --out artifacts/case_vf_exact_qp_v1 \
      --cores 15 --plan-seconds 180 \
      --case-steps 200 --vf-outers 30 --vf-pdca-steps 50 \
      --cuts 80 --backend gurobi

Required context can be generated, if missing, with the prior
objective-selection runner's --stage setup without rerunning
old optimization comparisons.

## Only essential outputs

case_policy.json (CASE coefficients, LAD and pDCA progress),
vf_features.json (frozen train feature normalization),
vf_library_progress.json (VF policy, libraries, Delta and epsilon),
selected_policy.json (frozen CASE and VF coefficients),
evaluation_2013.json (three-arm component-level outcomes), and
summary.json (common certified coverage, cost/case, MAE/case,
week-clustered paired cost bootstrap intervals and better/worse weeks).

No response banks, alternative methods, response-regime breakdowns,
cross-method library certificates, or extra termination gates.

## Resume completed CASE and see live progress

If CASE already finished successfully, with an objective reduction from
383.598 to 376.158 in 15 accepted steps, DO NOT repeat CASE.
Pull the new branch only after the running Python process has exited.
Use the SAME output folder containing case_policy.json. Code-only speed
and reporting revisions record their source history and keep its fitted
coefficients; scientific parameter changes still require a new directory.

    git fetch origin
    git switch feature/nonmonotone-case-vf-mm
    git pull --ff-only origin feature/nonmonotone-case-vf-mm
    mkdir -p artifacts/case_vf_exact_qp_v1
    set -o pipefail
    caffeinate -i python -u pilot/run_nonmonotone_learning.py \
      --stage vf \
      --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
      --out artifacts/case_vf_exact_qp_v1 \
      --cores 15 --plan-seconds 180 \
      --case-steps 200 --vf-outers 30 --vf-pdca-steps 50 \
      --cuts 80 --backend gurobi \
      2>&1 | tee -a artifacts/case_vf_exact_qp_v1/run.log

Python -u makes progress unbuffered; tee saves stdout and stderr.
Future fresh CASE fits print and persist every pDCA step's objective,
accepted status, solver status and elapsed seconds, in case_progress.jsonl.
Planner stages print completed component counts every 100 new solves.
VF logs each outer step's before and after objectives, Delta, epsilon,
library size and elapsed wall seconds. Completed stages record time
in stage_wall_times.json. Wall times for the CASE run completed before
these updates cannot be recovered retroactively.

Computational optimizations: the CASE QP uses one raw-score auxiliary
per case, reducing repeated feature-row nonzeros; VF scores previously
validated library schedules using NumPy room-load bincount operations.
The training objective and the latest SHARED/VF planner are unchanged.

New outputs: cases_train.csv and cases_test.csv preserve case ID,
booked and actual duration, CASE/VF raw score, minimum displayed advice,
reviewed duration, and all five standardized VF features. Evaluation
contains idle and overtime minutes per component and arm; summary gives
these per case. There is no behavioral scenario or robustness sweep.

The review's wall-time estimates are not measured guarantees.

## Verification scope

Small licensed-QP unit tests check the exact CASE epigraph formulation,
response-floor/DC identities, fixed-plan objective equality, and
monotone MM acceptance. These tests do NOT establish full-UHN Gurobi
runtime or that CASE actually improves from LAD at 24,506 cases.
The previously examined 2013 evaluation is exploratory.
