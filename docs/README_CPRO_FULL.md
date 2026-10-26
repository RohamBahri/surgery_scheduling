# Full-cohort CPRO: resumable parametric ray calibration

This command processes all 3,362 components of the original UHN training
cohort and keeps all previous CASE, VF, VF-RAY and B/C artifacts unchanged.
It trains no new VF direction. The intrinsic saturation domain is computed
from the existing fitted CASE score, frozen VF direction and B(.8,30)
response, without any 64-coefficient calibration cap.

The anchored weekly planner with hash seed 0 is the deployment oracle.
A component is EXACT_ENUMERATION only when fully enumerated in rational
arithmetic. Gurobi-based interval passes are NUMERIC_PARTITION, never a
proof-carrying real-arithmetic certificate. Unresolved components remain
clearly unresolved.

## Pull

    git fetch origin
    git switch feature/cpro-full-cohort
    git pull --ff-only origin feature/cpro-full-cohort

If the feature branch is not yet present locally, instead use:

    git switch --track origin/feature/cpro-full-cohort

## Full run — no pilot

    set -o pipefail
    caffeinate -i python -u pilot/run_exact_ray_calibration.py \
      --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
      --out artifacts/response_saturation_grid \
      --train-scenario SAT_A80_H30 \
      --cores 15 \
      --plan-seconds 180 \
      --component-seconds 150 \
      --separation-seconds 10 \
      --deployment-seconds 600 \
      --wall-minutes 600 \
      --max-rounds 25 \
      --max-cells 1500 \
      --max-schedules 120 \
      --exact-limit 64 \
      --evaluate \
      2>&1 | tee -a artifacts/response_saturation_grid/cpro_full_run.log

Use the existing artifacts/response_saturation_grid root. The new isolated
results are in cpro_full_SAT_A80_H30/. Do not run two CPRO processes on
that same output concurrently.

Work is sorted largest-components-first and submitted to a rolling bounded
pool of at most 2*cores outstanding futures. As soon as a worker finishes,
the next component can start; there is no 15-job batch barrier. Component
results are written atomically. On global timeout, not-yet-started jobs are
cancelled; active bounded jobs are allowed to finish and checkpointed.

Any old source/budget-pinned CPRO contract is migrated only if the fixed
scientific inputs, original experiment hashes, response, and numerical
tolerance match. Old numeric partition checkpoints from an earlier solver
revision are recomputed. EXACT_ENUMERATION rational results can be reused.

Original VF-RAY grid assignments already available in the content-addressed
cache are loaded as additional feasible schedule-library seeds, *without*
solving any new ray-grid planner problems. A revised discovery pass collects
multiple verified witnesses and retains P1-P3 region records valid on
subintervals with the same reference and selected schedule.

## Retry only unresolved components with larger budgets

After the full pass, the exact same command resumes unstarted components
and reuses all solved checkpoints. To explicitly retry previously
UNRESOLVED components with different solver budgets, run:

    set -o pipefail
    caffeinate -i python -u pilot/run_exact_ray_calibration.py \
      --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
      --out artifacts/response_saturation_grid \
      --train-scenario SAT_A80_H30 \
      --cores 15 \
      --plan-seconds 180 \
      --component-seconds 1800 \
      --separation-seconds 120 \
      --deployment-seconds 1800 \
      --max-rounds 200 \
      --max-cells 12000 \
      --max-schedules 1200 \
      --exact-limit 64 \
      --wall-minutes 600 \
      --retry-unresolved \
      --evaluate \
      2>&1 | tee -a artifacts/response_saturation_grid/cpro_full_retry.log

Component and separation limits, max_rounds, max_cells, max_schedules,
deployment time, and overall wall time are *budgets*, not scientific
model parameters. They are stored in each component's solver_budgets
record, and can change between invocations. The scientific contract
continues to protect input data and fitting definitions.

### Partial-cohort status does not block exploratory evaluation

When some components have unresolved global ray partitions, the runner
can still choose a **provisional training-only multiplier** from the
resolved component partitions. It then solves the anchored deployment
planner on **all 3,362 training components** at that multiplier, with
the independent --deployment-seconds budget.

Only if every training assignment at that multiplier is certified will
the runner record the actual complete training cost and, if --evaluate
was requested, replay its frozen B advice on the B/C evaluation scenarios.

This is *not* a globally optimal calibrated ray if any component ray
partition is unresolved. The summary explicitly says
EXPLORATORY_PARTIAL_CALIBRATION and global_ray_optimality_certified=false.
It is possible that a few hard anchored planner solves still block the
candidate deployment recheck; the runner reports the missing components.

## Output

    artifacts/response_saturation_grid/cpro_full_SAT_A80_H30/
        run_contract.json
        intrinsic_domain.json
        components/<component-digest>.json
        summary.json
        frozen_display.csv                         # when evaluated
        evaluation_component_records.json           # when evaluated
        evaluation_summary.json                     # when evaluated
        evaluation_cost_grid_global_matched.csv      # when evaluated

Each component JSON contains status, discovered assignments, execution
budgets, component dimensions, and (for unresolved numeric cases) the
resumable schedule library and previously checked cells.

- EXACT_ENUMERATION: exact-rational enumeration of all feasible schedules.
- NUMERIC_PARTITION: complete ray partition *numerically* checked by
  float Gurobi; not an exact-arithmetic global proof.
- UNRESOLVED: cannot certify the full ray for this component.
- EXPLORATORY_PARTIAL_CALIBRATION: incomplete global partitions, but the
  selected provisional step has certified deployed plans on all training
  components and can be evaluated out of sample.
- EXACT_CLOSED_RAY: reserved for a full exact-rational global certificate.

There is no promised wall time: solver runtime depends on actual
component complexity and the number of schedule switches. The 2013
evaluation period has been inspected during previous research and
should be treated as exploratory.
