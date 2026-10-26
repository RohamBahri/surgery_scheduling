# Certified Parametric Ray Calibration — evaluation patch

**Scope: research implementation, not yet a completed exact full-UHN run.**

This patch adds a saturation-bounded, one-parameter ray solver to an existing
checkout of `RohamBahri/surgery_scheduling` (branch containing the completed
B/C response-grid experiment). It does not overwrite CASE, VF-MM-pDCA, VF-RAY,
zero-L1, or their evaluation cells. New outputs are isolated under
`artifacts/response_saturation_grid/cpro_SAT_A80_H30/`.

It has **two explicitly different certification levels**:

* `EXACT_ENUMERATION`: tiny components only. Exhaustive feasible-assignment
  enumeration, IEEE-float-as-exact-rational input interpretation, actual SHA-256
  seed-0/seed-1/lex tie order, exact rational partition roots, and explicit
  isolated boundary points. This is an exact-arithmetic *reference oracle*,
  with full closed-ray global optimality for those components.
* `NUMERIC_PARTITION`: larger components. Parametric Gurobi P1/P3 MILPs,
  exact binary-times-continuous McCormick rows, rational witness rechecks,
  exact rational P2 checks, and original anchored planner rechecks at boundaries.
  **Not an exact mathematical proof.** A Gurobi `OPTIMAL`, `INFEASIBLE`, or
  `ObjBound >= -mu` does not yield rational-arithmetic proof merely by
  reconstructing an incumbent. Numerical edge cases return `UNRESOLVED`.
* `UNRESOLVED`: timeouts, float-collapsed intervals, failed boundary checks,
  exceeded budgets, or missing certificates. These are never silently treated
  as globally optimal regions. Such runs will not select and deploy an
  allegedly exact global policy.

This is an **evaluation of computational feasibility and schedule discovery**.
The complete proof-carrying full-UHN backend (exact/verified separation
certificates and globally certified unresolved-interval lower-bound queue)
has NOT yet been implemented. Do not report its large-instance output as
"exact global calibration" in a manuscript. The mathematical finite-library
separation algorithm is exact *under exact optimization oracles*; its numerical
realization is a distinct, open implementation task.

## Mathematical domain

For each training case, `a_i = X_i w_CASE`, `p_i = H_i theta_VF`, and the
saturating B response with training `alpha,h`:

    d_i(s) = max(1, booked_i + clip(alpha*(a_i+s*p_i), -alpha*h, alpha*h)).

Define the final directional saturation threshold

    t_i = max(0,(h-a_i)/p_i)   if p_i>0
          max(0,(-h-a_i)/p_i)  if p_i<0
          0                   if p_i=0

and use `S=max(1,max_i t_i)` across the WHOLE training cohort. All reviewed
training durations are constant for `s>=S`. Keeping `s=1` inside the domain
preserves original VF even if the common saturation plateau begins earlier.
This **does not** use the VF-pDCA coefficient box to clip the calibration ray.

A case can cross both saturation thresholds on the way to its terminal
saturated state. The partition includes these crossings and the duration-floor
kink, not just the final saturation point.

## Install into repository

Unzip the patch in the ROOT of the local `surgery_scheduling` checkout,
preserving the `surgery/`, `pilot/`, and `tests/` relative paths. For example,
copy the provided ZIP there and run:

    unzip -o cpro_evaluation_patch.zip

Run the exact reference tests (no Gurobi required):

    python -m pytest -q tests/test_parametric_ray.py

Inspect intrinsic ray endpoint and input contracts without new solves:

    python -u pilot/run_exact_ray_calibration.py \
      --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
      --out artifacts/response_saturation_grid \
      --train-scenario SAT_A80_H30 \
      --cores 15 --plan-seconds 180 --dry-run

**First, pilot on real components.** This is a validation/computational-tractability
run, not the paper's full-method comparison:

    set -o pipefail
    caffeinate -i python -u pilot/run_exact_ray_calibration.py \
      --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
      --out artifacts/response_saturation_grid \
      --train-scenario SAT_A80_H30 \
      --pilot-components 40 \
      --cores 15 --plan-seconds 180 \
      --component-seconds 90 --separation-seconds 7 \
      --boundary-seconds 30 --wall-minutes 120 \
      2>&1 | tee -a artifacts/response_saturation_grid/cpro_run.log

Then, if the pilot establishes useful termination and boundary agreement,
run the complete training set. The same per-component checkpoints are reused:

    set -o pipefail
    caffeinate -i python -u pilot/run_exact_ray_calibration.py \
      --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
      --out artifacts/response_saturation_grid \
      --train-scenario SAT_A80_H30 \
      --cores 15 --plan-seconds 180 \
      --component-seconds 90 --separation-seconds 7 \
      --boundary-seconds 30 --wall-minutes 120 \
      --evaluate \
      2>&1 | tee -a artifacts/response_saturation_grid/cpro_run.log

The optional `--evaluate` flag replays **one selected training-only policy**
under all six B and six C response scenarios **only if** a complete component
partition exists and the incumbent is reproduced by the hash-0 planner on
all training components. Even then, a Gurobi numerical partition is not an
exact global-optimality certificate. The evaluation is exploratory because
2013 has been repeatedly accessed during earlier model development.

The time budget is finite; all completed components are checkpointed.
You can resume with the SAME command and settings; do not run two jobs
writing the same folder simultaneously. Changing certification settings or
source code requires a new output root (or archiving the old cpro folder).

## Outputs

* `cpro_SAT_A80_H30/intrinsic_domain.json` — saturation endpoint S.
* `cpro_SAT_A80_H30/run_contract.json` — hashes and solver specification.
* `cpro_SAT_A80_H30/components/<hash>.json` — per-component exact/numeric/
  unresolved status, discovered schedules, per-cell status and boundary costs.
* `cpro_SAT_A80_H30/summary.json` — component-level completion statistics;
  a universally valid (possibly weak) lower bound using exact small-component
  minima and zero for other components; endpoint training costs when complete;
  if and only if all components are covered, best reported ray value, step,
  boundary/stable status and independently replayed total training cost.
* `cpro_SAT_A80_H30/evaluation_component_records.json` — new B/C cells.
* `cpro_SAT_A80_H30/evaluation_cost_grid_global_matched.csv` and
  `evaluation_summary.json` — one common-certified cohort across original
  BOOKED, NAIVE, CASE, VF, optionally original VF-RAY and new CPRO, with
  paired calendar-week bootstrap comparisons.

`summary.json` uses a deliberately strict status vocabulary:

* `EXACT_CLOSED_RAY` — every component was exhaustively rationally solved;
  the global minimizer is mathematically exact for the stored inputs.
* `NUMERIC_PARTITION_CANDIDATE_NOT_EXACT` — a continuous partition candidate,
  validated numerically but without a rational global separation certificate.
* `UNRESOLVED_*` — no global-ray optimality claim is permitted.

## The remaining rigorous solver work

Before promoting this algorithm to a main method, implement **verified**
P1 dual lower bounds and P3 infeasibility proofs (for example with an
exact-arithmetic MILP backend or reconstructible proof certificates), and
add the valid interval lower-bound branch queue that can terminate with a
quarter-unit global objective certificate even if some regions remain
unresolved. Checking only a rational witness is not sufficient to certify
P1 optimality or P3 infeasibility. A stated numerical margin mu is not a
substitute for such a proof.

Numerical Gurobi interval-relaxation dual bounds for unresolved components
are recorded under each component's `full_ray_relaxation`. The interval
relaxation is mathematically valid, but its floating dual is **not** used
as a rational certified bound. A complete interval branch-and-bound priority
queue with exact verified dual bounds remains a separate implementation step.
