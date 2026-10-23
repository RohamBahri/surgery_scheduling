# Nominal CASE and VF: full response + library MM/pDCA

**Status:** research branch, draft PR #40. Old frozen artifacts and main remain unchanged.

## Agreed method

BOOKED (zero display), CASE (response-aware), and VF (CASE plus five planning
features) are the ONLY arms. Response is homogeneous nominal alpha=0.8, h=30.
No sweep, response uncertainty, profiles, cap optimization, or direct SHARED
method in this experiment.

CASE optimizes an unrestricted shared raw displayed-advice score Xw through
the TRUE nonmonotone adoption/attenuation/rejection map, except a nominal
UI cap of +/-180 minutes. The positive-duration floor f=1 is applied to the
REVIEWED booking, not as a lower bound on displayed advice; the compiler
includes extra breakpoints for floored cases. The operational display
F_eff(raw)/alpha is the minimum-magnitude display giving the same nominal
reviewed booking. Both CASE and VF optimize the raw score, not this
canonicalized display.

CASE's training objective is asymmetric post-response case error plus a
training-derived L1 coefficient penalty. It is solved by proximal DCA with
the exact convex/convex hinge decomposition and a convex cutting-plane QP,
starting from LAD and from a converted prior CASE warm start. Every accepted
step is checked against the exact unsmoothed objective.

After CASE training, compute ONE fixed provisional plan per component using
the deterministic hash-0 tie rule. Construct five standardized features:
history surgeon spread, history procedure spread, provisional load deviation
p = clip((load-480)/120,-2,2), and the two corresponding interactions.
No circular final-plan feature updates. Freeze CASE and fit five VF
incremental coefficients, initialized at zero, so VF(0) equals CASE.

VF optimizes the manuscript Section 4 full asymmetric training bound:
mean by training COMPONENT of (minimum library J(z,reviewed duration) plus
sum of 1.75*positive case error + 1.0*negative case error) with a
training-only L1 penalty on VF coefficients. Every staffed roster block
contributes idle cost even when unused. Initial schedule libraries contain
original, BOOKED and CASE-draft feasible assignments. Each outer step
requests new feasible planner incumbents, adds them, selects each library's
cheapest schedule, then uses pDCA to improve the touching MM majorizer.
A time-limited MIP can still add its feasible incumbent; missing dual
bounds are replaced by the valid zero lower bound, NEVER by a false optimum.
The run tracks Delta (certified library descent), epsilon (library minus
planner dual bound), inner solver status, coefficients and full training
certificate. MM does not certify global optimality or finite-run stationarity.

## One important manuscript inconsistency

The current weekly planner correctly accounts for all roster-block capacity,
including unoccupied blocks. BUT the existing cohort builder constructs the
list of physical room-day slots from rooms OBSERVED IN THAT WEEK, rather
than the fixed training-derived median-count roster template described in
the supplied implementation note. This new learner inherits the previous
context. The roster difference remains unresolved; do not describe it as
verified fixed-template planning. The branch does not silently change the
planning cohort or eligibility to match a manuscript sentence.

## Run

    git fetch origin
    git switch feature/nonmonotone-case-vf-mm
    python -m pip install -r requirements-lock.txt
    python -m pip install -e . --no-deps
    python -m pytest -q tests/test_nonmonotone_learning.py

Required prepared context:

    artifacts/objective_selection_12h/objective_selection_2013/context.pkl

If unavailable, create it WITHOUT rerunning all prior methods:

    python pilot/run_objective_selection.py \
      --stage setup \
      --workbook data/UHNOperating_RoomScheduling2011-2013.xlsx \
      --out artifacts/new_context \
      --cores 10 --setup-solve-seconds 180

Then use --context artifacts/new_context/objective_selection_2013/context.pkl.

Train only CASE first:

    python pilot/run_nonmonotone_learning.py \
      --stage case \
      --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
      --out artifacts/nominal_case_vf_mm \
      --case-steps 200 --cores 15 --backend gurobi

Continue to the draft and VF, using the SAME settings/output:

    caffeinate -i python pilot/run_nonmonotone_learning.py \
      --stage vf \
      --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
      --out artifacts/nominal_case_vf_mm \
      --case-steps 200 --vf-outers 30 --vf-pdca-steps 50 \
      --cuts 80 --cores 15 --plan-seconds 180 --backend gurobi

You may use --stage all to fit CASE, draft and VF together. Finished
CASE, feasible plans and VF outer steps are checkpointed. A changed source
or science configuration should use a NEW output directory.

Only AFTER training, retrospectively evaluate the three models:

    python pilot/run_nonmonotone_learning.py \
      --stage evaluate \
      --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
      --out artifacts/nominal_case_vf_mm \
      --case-steps 200 --vf-outers 30 --vf-pdca-steps 50 \
      --cuts 80 --cores 15 --plan-seconds 180 --backend gurobi

Outputs: case_policy.json, vf_features.json, plans/, vf_library_progress.json,
selected_policy.json, evaluation_2013.json and summary.json.

## Evidence still needed for manuscript

A full UHN run has NOT been executed. The 2013 sample is reused and any
result is exploratory, NOT an independently untouched confirmatory test.
Record solver coverage, time, CASE starts, library certificates, epsilon
gaps, pDCA cut gaps, and actual common-certified costs. Check line search,
tied schedules and stationarity only as diagnostics: finite-run termination
cannot establish full stationarity. The exact roster construction remains
an explicit operational mismatch. This study does not estimate actual
surgeon response parameters, and does not prove realized savings.
