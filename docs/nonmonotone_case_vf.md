# Non-monotone CASE + VF: research implementation (October 2026)

Branch: feature/nonmonotone-case-vf-mm. This is an additive research
implementation. It does NOT overwrite the frozen October 2026
objective-selection policies, comparison results, planner, or cohort.

## Scientific contract

1. **One displayed-advice policy.** CASE learns raw display scores X@w by
   post-response asymmetric error with pDCA. VF keeps CASE fixed and learns
   FIVE additional coefficients for s, q, congestion, s*congestion and
   q*congestion. The provisional congestion plan is recomputed from the
   newly learned CASE recommendations, once and without outcomes, using the
   same three deterministic hash ties as the earlier design.
2. **Non-monotone response.** For ANY unbounded shared raw display score,
   each case clips its displayed advice independently to
   [max(-c,1-booked),c]. The full response has an increasing adoption branch,
   decreasing attenuation branch, and zero rejection branch. No shared
   coefficient restriction is deduced from the intersection of individual
   reachable intervals.
3. **Minimum-magnitude actual display.** For a nominal response
   (alpha0,h0), the operational display is
   u = R_nom(clip(raw,...))/alpha0. This is the smallest-magnitude advice
   generating that nominal adopted correction. Crucially, the SAME canonical
   display enters every alternative response scenario DURING learning.
   Thus the learned method is mathematically non-monotone as a function
   of raw score, but its operational display does not show unnecessarily
   extreme advice. This canonicalization is conditional on the nominated
   response law; it does not make every scenario equally responsive.
4. **Behavioral uncertainty.** A predeclared family of response scenarios
   is pooled at equal weights by default, unless explicit weights are
   specified. These are hypothetical, not surgeon-specific estimates from
   UHN. No post-hoc real-surgeon profile assignment.
5. **Two-stage planning.** New CASE -> exactly solved provisional physical
   room/week assignments -> frozen five features -> VF display ->
   scenario-specific response -> final EXACT primary planner in training.
   Final evaluation uses the existing outcome-blind hash tie rule.
6. **VF objective.** For each response scenario, the primary reduced planner
   value Q(d) plus 2.75 times the sample stop loss E[(actual-d)+],
   averaged over cases; one shared ridge coefficient penalty. The duration
   scenario bank is FROZEN across coefficient iterations. No hindsight
   outcomes enter the final deployed planner. The current residual donor
   pool comes from the earlier training-only out-of-fold context, not from
   unobserved surgeon adoption.
7. **Optimizer.** For fixed schedules, the full clipped-response objective
   is exactly G-H, with convex G and H. pDCA forms a supporting hyperplane
   of H. The convex G-step uses a cutting-plane QP with globally valid
   supporting hyperplanes. Gurobi supplies certified convex masters; the
   difference G(candidate)-epigraph is checked and logged. Every accepted
   CASE pDCA step reduces the exact unsmoothed CASE objective. Every
   accepted VF inner pDCA step reduces its fixed-plan VF objective.
   A VF outer step exactly minimizes Q at the incumbent, freezes each
   scenario-specific room plan, and minimizes a touching upper majorizer.
   Therefore every accepted VF outer step cannot increase the full TRAINING
   VF objective. Local or iteration-cap termination is NOT global
   optimality. A SciPy backend exists ONLY for small smoke tests.
8. **Regularization differs from the previous frozen study.** CASE uses
   coefficient ridge and L1; VF uses ridge on its five incremental display
   coefficients. Previous regularization of deviations in implemented
   duration is NOT retained, because that nonlinear term is not convex
   in the displayed-advice coefficients. This is a declared new training
   specification. Every cap policy is optimized and compared using its
   fully recomputed same-form training VF objective.
9. **Selection.** If multiple display caps are supplied, one cap and one
   policy are selected using only the predeclared TRAINING VF objective.
   No 2013 outcomes are used in that choice. Repeated study of 2013 means
   subsequent evaluation is exploratory, not fresh external validation.
10. **No-harm claims.** Scenario-averaged VF is NOT a true-cost no-harm
    constraint. A scenario-wise surrogate bound does NOT imply a
    scenario-wise realized scheduling improvement. This implementation
    does not claim or silently enforce such a guarantee.
11. **Feasibility and incomplete solvers.** Every planner call must certify
    the complete primary or hash selection. An unresolved solve raises
    an error, leaving its completed per-component cache intact. Incomplete
    runs NEVER silently replace the learned policy with a baseline.
12. **Operational limitations.** Retrospectively inferred rosters, linear
    idle/overtime costs, potentially excessive room loads, synthetic
    behavioral scenarios, and independence approximations in residual
    draws remain limitations. This code is NOT a prospective clinical
    scheduling system.

## Prerequisites

- A local checkout of RohamBahri/surgery_scheduling with this branch.
- Python dependencies from requirements-lock.txt.
- A valid, licensed Gurobi installation for production pDCA and exact
  weekly planning.
- The original workbook and the previously prepared, training-only
  context.pkl from the October 2026 selection study. This implementation
  reads that context and re-trains CASE and rebuilds congestion features;
  it does NOT reuse the earlier fitted CASE coefficients or congestion
  features as final policies.

## Get the branch and run tests

    git fetch origin
    git switch feature/nonmonotone-case-vf-mm
    python -m pytest -q tests/test_nonmonotone_learning.py

## First small smoke test

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python pilot/run_nonmonotone_learning.py \
      --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
      --out artifacts/nonmonotone_smoke \
      --caps 45 \
      --scenario 0.8:30 \
      --draws 8 --cores 2 --case-steps 2 --vf-outers 1 \
      --vf-pdca-steps 2 --max-cuts 15 --plan-seconds 120

NOTE: This checks the full TRAINING cohort even with small iteration caps;
it is not a miniature synthetic instance, and the planner can remain
computationally expensive. The SciPy pDCA backend is available for unit
tests only. Gurobi is still required for all exact weekly plans.

## Full fit (no holdout)

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    caffeinate -i python pilot/run_nonmonotone_learning.py \
      --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
      --out artifacts/nonmonotone_case_vf_v1 \
      --caps 45 \
      --scenario 0.8:30 \
      --scenario 0.6:30 \
      --scenario 0.4:30 \
      --scenario 0.8:15 \
      --draws 50 --cores 10 --plan-seconds 180 \
      --case-steps 25 --vf-outers 8 --vf-pdca-steps 20 \
      --max-cuts 60 --backend gurobi

The default --caps 45 and the four response scenarios above are explicit
scientific settings, NOT calibrated adoption parameters. Set --caps 30 45
if you want a prespecified cap comparison; this can nearly double training.

Training saves: manifest.json, cap_45_0/case_pdca.json,
five_feature_normalization.json, vf_mm_state.json, fitted_policy.json,
and selected_policy.json. Primary and hash-optimal scheduling components
are checkpointed with duration signatures under plans/. A changed scientific
setting or source digest requires a NEW --out folder.

You may call the same command again to resume completed case/policy and
component checkpoints. It will not replay completed certified planner
subproblems with identical input durations.

## Optional retrospective 2013 comparison (after freezing training)

Add --evaluate to the IDENTICAL fitting command and --out folder.
It writes retrospective_2013.json with exactly solved per-case realized
modeled costs for BOOKED, CASE and VF under every response scenario. Every
evaluated scenario uses the SAME displayed advice for the relevant method.
The 2013 sample has already been examined repeatedly in this project,
so label the result EXPLORATORY_REUSED_HOLDOUT.

## Do not claim yet

- Full UHN run or licensed-Gurobi integration was executed by the authoring
  environment; neither has happened here.
- Global optimality of CASE or VF coefficients.
- A stationary-point theorem at assignment or response kinks.
- True-cost scenario-wise no-harm.
- Causal improvement over historical UHN operations.
- That the uncertain response distribution has been empirically identified.
- That the existing scenario donors correspond to an out-of-fold prediction
  distribution for the newly refitted CASE without extra validation.

These limitations should be resolved or acknowledged before using the
implementation as final journal evidence.
