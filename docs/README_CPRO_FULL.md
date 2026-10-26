# Full-cohort Certified Parametric Ray Optimization (CPRO)

This is the **full-size run**, not a small pilot. It uses every training
component in the original UHN 2011–2013 context, the pre-fitted original
CASE policy and VF-MM-pDCA direction, the exact-intrinsic saturation ray,
and the original anchored hash-0 deployment planner.

Scientific limits: this is still a research numerical prototype. The
small exact-enumeration reference provides rational certificates; for
larger components the Gurobi P1/P3 oracles produce **numeric-only** interval
certificates, never exact-arithmetic proofs. Unresolved regions are
logged, not promoted to completed global optimization. If the full run
does not yield a globally certified candidate, it does not manufacture
or evaluate one.

## Pull

    git fetch origin
    git switch --track origin/feature/cpro-full-cohort

If you already have the branch locally, instead:

    git switch feature/cpro-full-cohort
    git pull --ff-only

## Run all 3,362 training components for nominal B(.8,30)

Use the **existing** response_saturation_grid artifact root as the input
and cache base. The new results go to a separate
artifacts/response_saturation_grid/cpro_full_SAT_A80_H30/
folder. Do not run another job writing to this artifact root concurrently.

    set -o pipefail
    caffeinate -i python -u pilot/run_exact_ray_calibration.py \
      --context artifacts/objective_selection_12h/objective_selection_2013/context.pkl \
      --out artifacts/response_saturation_grid \
      --train-scenario SAT_A80_H30 \
      --cores 15 \
      --plan-seconds 180 \
      --component-seconds 150 \
      --separation-seconds 10 \
      --wall-minutes 600 \
      --max-rounds 25 \
      --max-cells 1500 \
      --max-schedules 120 \
      --exact-limit 64 \
      --evaluate \
      2>&1 | tee -a artifacts/response_saturation_grid/cpro_full_run.log

No --pilot-components argument is passed: ALL training components run.
The global wall budget prevents indefinite computation; when exceeded,
completed components remain checkpointed. Every component gets its own
wall limit and every numerical separation solve has a separate budget.
A full run can take hours. Runtime and certificate closure on 3,362
components are unknown beforehand.

**Resume after global wall exhaustion**, using the exact same command.
Completed (including unresolved) component files are reused, so the job
focuses on components not yet visited. To make a *separate* expensive
second pass on previously unresolved components, add
--retry-unresolved. Changing the numerical limits changes the immutable
experiment contract: use a newly named output directory or run a
controlled update rather than mixing incompatible certificates.

Output:
- run_contract.json: input hashes and scientific/numeric parameters
- intrinsic_domain.json: saturation ray endpoint S and frozen inputs
- components/*.json: checkpointed result and explicit certificate status
- summary.json: status counts, bounds, and any certified aggregate candidate
- evaluation_summary.json and evaluation_cost_grid_global_matched.csv:
  written only if a complete new candidate passes the training
  deployment recheck (all component plans certified)

Interpretation:
- EXACT_ENUMERATION: exact fixed-input rational result on a small component
- NUMERIC_PARTITION: approximate numerical Gurobi region checks, NOT
  a verified rational exactness certificate
- UNRESOLVED: ambiguous boundary, wall/budget cap, or incomplete solver proof
- GLOBAL result should NOT be called exact unless all required exact
  component/oracle checks and global objective certificates are present.

## Corrections versus previous pilot

1. Collisions in both hash objectives are tested in one MIP before
   launching the expensive case-indexed lexicographic fallback MIPs.
   A numerical infeasibility can skip lex tests only as a NUMERIC pass.
2. Every rational library boundary is checked with fixed-duration,
   fixed-parameter P1/P3 MILPs, without a floating ray variable or
   s*x linearization. A verified challenger enters the schedule library;
   ambiguous comparisons remain unresolved.
3. Results are kept separate from earlier 40-component pilot checkpoints.
   Previously unresolved components are not automatically retried every
   restart, avoiding repeated waste.

## Scope

This command trains **no new direction**: it calibrates original VF under
B(.8,30) using full original training data. The B/C response variants in
2013 are *evaluation assumptions*, not observed surgeon responses.
The 2013 outcomes have already been inspected and are exploratory.
