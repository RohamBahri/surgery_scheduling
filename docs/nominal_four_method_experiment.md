# Four-method nominal non-monotone UHN experiment

This is ONE experiment with **four trained policies** and one reference:

| Arm | Learns | Optimizer |
| --- | --- | --- |
| BOOKED | Nothing | Historical surgeon-booked durations; reference only |
| CASE | Shared casewise raw advice | Regularized LAD start + exact convex-epigraph pDCA |
| VF_PDCA | Five coefficients added to CASE | Value-function schedule-library MM + pDCA |
| VF_DIRECTIONS | Five coefficients added to CASE | SAME VF objective, schedule-library MM + multi-direction search |
| SHARED | Five coefficients added to CASE | Actual planner-selected realized-cost training, with schedule-library screening and complete-training acceptance |

The two VF arms are **separate models**. They use identical training
data, frozen CASE provisional schedules, five standardized features,
nominal response, VF regularizer, coefficient bounds, weekly planner,
and cost objective; **only the optimizer differs**. VF_DIRECTIONS does
not secretly include pDCA and VF_PDCA does not scan multidirections.

CASE uses 0.75 times the original training-calibrated L1 coefficient,
the sole regularization change. There is no second VF penalty setting.

## Model and outcomes

The cohort and physical room roster come from the same
objective_selection_12h/.../context.pkl as the preceding experiments,
**not from the superseded manuscript roster**. The weekly planner
enforces the existing eligibility and required surgeon-days. Every
staffed block is charged for unused time; 480 minutes is the physical
room-day capacity, 30 minutes of turnover applies to occupied blocks,
and overtime costs 1.75 units/minute versus 1 unit/minute idle.

All learned policies use one nominal homogeneous non-monotone response
(alpha=.8, h=30), with one-minute duration floor AFTER surgeon
response. Its increasing, attenuating, and rejecting regions are
retained in fitting. All deployed displays are converted to the
minimum-magnitude equivalent advice, and the reviewed duration is
verified before evaluation. No behavior scenario banks or robustness
sweeps are generated.

The same frozen CASE draft provides all three 5-coefficient methods
their standardized historical error-spread/congestion interaction
features. Re-solving those drafts for each method would change the
feature definition and is explicitly not done.

The VF-pDCA and VF-directions arms both optimize the SAME
finite-library realized-case-error envelope plus the minimum
predicted idle/overtime cost in the schedule library and identical
L1 penalty. They start from VF theta=0 (CASE), not from each other.
In VF training, planner incumbents are added to each method's
own library; the MM objective is checked at its touching point.
The directional variant evaluates full-library loss along positive,
negative, and mixed feature directions with predetermined step sizes,
accepting only exact improvements of that same library loss.

SHARED optimizes something **different**: realized costs of the
nominal planner's hash-0 chosen schedules on HISTORICAL TRAINING cases,
plus the same 5-parameter L1 penalty. A fixed train-only mixed-site
subset and feasible schedule libraries screen candidate directions,
but only full-training actual planner solves can accept a step.
A solution with incomplete certification is never silently reported
as exact. SHARED starts at the direct-cost-better certified policy
among CASE and the two VF candidates and performs bounded
multi-direction search with full-data acceptance. Because the
direct planner value is discontinuous, CASE's convex pDCA surrogate
cannot be applied to this objective unchanged.

## Run exactly once (with resumable checkpoints)

Use a **new** output root; previous CASE/VF artifacts are different
scientific fits and must remain unchanged.

    git fetch origin
    git switch feature/nonmonotone-case-vf-mm
    git pull --ff-only origin feature/nonmonotone-case-vf-mm
    python -m pytest -q tests/test_nonmonotone_learning.py

    mkdir -p artifacts/nominal_four_methods
    set -o pipefail
    caffeinate -i python -u pilot/run_nominal_four_methods.py \
      --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
      --out artifacts/nominal_four_methods \
      --cores 15 \
      --plan-seconds 180 \
      2>&1 | tee -a artifacts/nominal_four_methods/run.log

This is one invocation. It sequentially executes:

1. CASE with its 25% weaker penalty;
2. one frozen CASE draft + BOOKED train plans + five shared features;
3. VF_PDCA (up to 30 MM iterations; 50 pDCA steps, max 160 inner cuts);
4. VF_DIRECTIONS (up to 30 MM iterations, predetermined multi-directions);
5. SHARED (up to 8 search rounds, full-train direct acceptance);
6. 2013 BOOKED/CASE/VF_PDCA/VF_DIRECTIONS/SHARED hash-0
   evaluation on the identical common-certified sample and week bootstrap.

Planner solutions are cached by component, duration values, tie setting
and solver budget. CASE coefficients, VF library progress, SHARED
search progress, and final policies are saved. Re-running the SAME
command with unchanged parameters and output directory reuses
completed training and cached plans, rather than launching five new
experiments. Changed scientific settings demand a new output root.

Progress is unbuffered and logged to run.log: CASE QP iterations,
planner completion counts, separate VF MM iterations (Delta, epsilon),
SHARED direct-cost search decisions and stage wall times.

## Results files

- case_policy.json: CASE coefficients, LAD start, lambda and exact-QP history
- vf_features.json: one train-only standardized five-feature map
- vf_pdca/vf_library_progress.json: VF-pDCA coefficients, library, history
- vf_directions/vf_library_progress.json: independent VF-directions fit
- shared/shared_progress.json: SHARED initializations, screen and full
  planner cost results, accepted steps and certification
- selected_policy.json: four frozen trained policies
- cases_train.csv and cases_test.csv: raw scores, displayed advice,
  reviewed durations and five standardized features for every method
- evaluation_2013.json: each arm's realized modeled cost, idle time,
  overtime, duration error and certified status per component
- summary.json: five-way common certified coverage, cost and MAE per
  case; paired week-bootstrap differences including direct comparison
  between the VF optimizers and SHARED versus CASE
- stage_wall_times.json and run.log

**Limits:** the 2013 outcomes were already inspected in prior research
iterations. Their confidence intervals and rankings are exploratory,
not pristine confirmatory holdout evidence. Response assumptions are
scenario inputs, not empirically observed surgeon adoption. A
training cost improvement does not establish a causal hospital saving.

Full-UHN runtime and SHARED full-training planner coverage are not
established by the size-limited-license GitHub tests. Training logs
record whether full-training candidates are certified and whether
SHARED actually improves on its starting policies.
