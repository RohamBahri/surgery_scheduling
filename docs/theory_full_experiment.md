# Full-cohort decision-aware method experiment

This is an **additive package for `RohamBahri/surgery_scheduling`, branch `behavioral-advice-comparison`**. It does not modify the existing planner, feature encoder, datasets, or previous experiment results. Copy the `pilot/`, `surgery/`, and `tests/` directories into the root of the existing checkout.

It implements the complete underlying **mathematical objectives**, rather than replacing difficult components with an easier toy model. Every fit and evaluation case is assigned to exactly one verified eligibility component; the runner asserts complete coverage. A time limit does **not** imply a globally solved method. All solver statuses and valid primal/dual bounds are retained, including incomplete solves.

## Quick start (no manual copying)

On this branch, run `bash run_theory_full.sh` from the repository root. It runs the licensed solver tests and then the complete pre-2013 two-fold development experiment. Override defaults with environment variables, for example `CORES=8 WORKBOOK=/path/to/UHNOperating_RoomScheduling2011-2013.xlsx bash run_theory_full.sh`. The 2013 holdout is not included.

## Methods

| Output arm | Training/decision problem | Optimization status |
|---|---|---|
| `BOOKED`, `CASE`, `RISK`, `STAT` | Existing, unchanged reference policies | Exact deployed Gurobi planner required |
| `PER_WEEK`, `PER_WEEK_ROBUST` | Existing full-size individual advice search, 1 or 3 planner hashes | Approximation search, exact deployment verification |
| `VF_SHARED` | Original VF in equivalent stop-loss form, **one coupled five-coefficient MIQP over every fitting component** | Solver gives a global primal/dual bound; `GAP_CERTIFIED` only if closed |
| `SHARED` | Direct conditional scenario expected deployed cost | Reproducible five-feature full-cohort directional approximation |
| `OBS` | Direct **observed training cost**, not simulated cost | Same approximation search, fit cases only |
| `EG` | Tolerance-aware Expected-GAP (gamma=1) | **Exact maximization MILP on every scored component**, outer training search remains approximate |
| `ROBUST` | Average scenario scheduling cost over four exactly implemented surgeon response functions | Same approximation search, all 3 hashes in each response model |
| `SHARED_GLOBAL` | Global **direct deployed** shared-cost objective on bounded five-dimensional domain | Coefficient-box search with unconditional SAA dual lower bounds and optional certified schedule stability; gap reported |
| `ANCHORED_TARGET` | Global restricted-inducibility problem: each fitting component must deploy its initial schedule or its SAA target, for each hash | Coefficient-box lower bounds and exact deployed feasibility checks; gap reported. **Not** the previous ANCHORED pDCA or the invalid proposed primary/hash cuts |
| `QFILL` | Exact constrained marginal stop-loss optimization given the current CASE draft plan | Convex LP (not a globally optimal deployed recommendation) |
| `VF_PW` | Full casewise VF with independent reachable reviewed durations | Global mixed-integer (quadratic if penalized), bounds retained |
| `PW_BOUNDED` | **Direct deployed** scenario expected cost over **all casewise durations**, 3 hashes | Full-dimensional branch-and-bound; valid but potentially weak SAA lower bound, optional stability pruning |
| `PW_ANCHORED` | Restricted version of the preceding problem: CASE or SAA target on each hash | Full-dimensional branch-and-bound with exact selected-schedule checks |
| `PW_VFCG` | Per-instance deployed scenario-cost bilevel problem for hash 0 | Full leader/witness assignment model, exact predicted cost graphs, primary value-function cuts, and necessary conditional hash cuts. Global dual/verified primal bounds retained; strict boundary cases marked unresolved |

**Do not conflate these certificates**. A globally optimal VF bound is *not* a bound on the optimal direct deployed regret. The unrestricted SAA dual is a valid lower bound for the direct scenario cost, but ignores inducibility. A value-function master cut only certifies deployed optimality when its global dual bound meets the objective of an independently verified deployed recommendation.

## Run

Requires the repository's original Python environment, a full licensed **Gurobi** installation, the UHN workbook, and sufficient solver memory. The package adds no new runtime dependencies beyond those used by that repository.

```bash
# From your surgery_scheduling checkout, after copying these files:
python -m pytest -q tests/test_theory_exact.py tests/test_theory_solver.py

# Complete, two-fold pre-2013 development experiment:
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
caffeinate -i python pilot/run_theory_full.py \
  --workbook data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --out artifacts/theory_full_v1 \
  --fold both \
  --methods VF,SHARED,OBS,EG,ROBUST,GLOBAL,ANCHORED \
  --cores 15 --seconds 120 --pw-seconds 900 \
  --shared-seconds 21600 --global-seconds 14400 \
  --global-nodes 40 --pw-nodes 100 --rounds 10 --radius 32
```

This is a **large optimization campaign**, not a quick benchmark. It fits every specified method separately on each training fold and evaluates all arms and all eligibility components. Extensive MIP timeouts are possible; the output quantifies how often. Increasing budgets requires a **new experiment directory**, preserving provenance. Repeating with the identical specification resumes saved component, SAA-bound, and directional-search records. An interrupted monolithic VF MIQP or global coefficient-region tree currently **restarts its solver/tree search**; its earlier saved progress remains available for diagnosis.

For the final forward 2013 evaluation, **only after methods and budgets are frozen**, use a new directory and explicitly acknowledge opening the holdout:

```bash
caffeinate -i python pilot/run_theory_full.py \
  --workbook data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --out artifacts/theory_final_2013_frozen \
  --fold 2013 --acknowledge-final \
  --methods VF,SHARED,OBS,EG,ROBUST,GLOBAL,ANCHORED \
  --cores 15 --seconds 120 --pw-seconds 900 \
  --shared-seconds 21600 --global-seconds 14400 \
  --global-nodes 40 --pw-nodes 100 --rounds 10 --radius 32
```

These flags are examples of **predeclared resource budgets**, not a statement that every problem will close. They do not silently lower the scenario count, drop large components, or change the planner objective.

## Outputs and how to interpret them

`manifest.json` records the exact workbook/source/configuration fingerprint. Under each `fold_0`, `fold_1`, or `fold_2013` directory:

- `vf_global.json`: global VF MIQP feasible objective and solver lower bound, gap, complete flag, and five coefficients. If there is no feasible solution before the time limit, VF training is unresolved.
- `<method>/progress.json`, `<method>/trained.json`, and round-wise `screen/*.json`, `all/*.json`: scored policy directions on all fitting components, their solver status, and the approximated outer objective. EG's inner maximization can time out; such candidates cannot be ranked as certified.
- `global_method/saa_bounds/*.json`: one SAA feasibility/dual audit for **each** training component, including failed or unfinished solves.
- `global_method/{direct,target}/result.json`: feasible policy upper bound, valid global lower bound, and exact **empirical** scenario objective gap. The bound can be loose when ties dominate.
- `full_evaluation/*.json`: individual component, method, seed, exact deployment assignments, realized cost, scenario cost, overloaded-room diagnostic, planned light/heavy/mixed imbalance regime, hash disagreement, and all optional MIP certificates.
- `theory_summary.json`: all-arm, full-denominator estimates (only if every component was certified for that arm), by-site/by-size breakdown, bootstrap pairs against CASE, and optimization gap counts/quantiles.
- `training_failures.json`: explicit names and reasons for any unsuccessful method training; they are not replaced with CASE while retaining the unsuccessful method's label.

The paired 20,000-resample bootstrap clusters by **calendar Monday**. All three hash outcomes are averaged *within* each Monday; they are not treated as independent weeks. Exploratory p-values are Holm-adjusted across every fully evaluated contrast to CASE. The 2013 cohort is not opened by the development command.

## What the code proves, and what it does not

1. `vf_shared_miqp` optimizes the **same finite-marginal VF upper bound** over one common five-dimensional clipped policy. Every fitting component participates in one Gurobi MIQP. A nonzero MIP gap means approximation, not a proof of outer optimality.
2. The inner `expected_gap_mip` is a true **maximization** over all feasible schedules with exact realized/predicted overtime hinges. It must complete to count as an exact candidate score. The *outer* EG search has no global optimization guarantee.
3. `value_function_exact` carries all primary and hash necessary conditions, with the **implemented value-dependent tolerance** and the exact two-hash order. Its hash cuts deliberately relax strict exclusion to `>=` rather than inventing a numerical epsilon. It certifies a global deployed optimum only after a verified incumbent matches a valid global lower bound. If a strict tie boundary prevents separation, its status stays `BOUNDED_UNRESOLVED`.
4. `ANCHORED_TARGET` exactly checks deployment to one of **two prespecified schedule alternatives** on each seed; its lower bound is a relaxation and the search can time out. It is a different, explicit approximation from the unimplemented target-induction theorem of the note. No mathematical descent guarantee is falsely inherited from old ANCHORED acceptance tests.
5. Scenarios are estimated from training residual donors with the existing casewise mechanism; they do not recover unknown within-room covariance or actual surgeon response. The code distinguishes simulation-model costs, observed realized costs, and computed training lower bounds. It does not claim that scenario-model guarantees hold for the true hospital distribution.
6. Exactness of *the model* is not evidence of roster realism. The retained eligibility reconstruction and idle/overtime objective permit extreme loads. Every arm records `over720` and maximum realized room load. The manuscript must discuss this assumption.
7. The full experiment has **not been executed in the artifact-creation environment** because `gurobipy` is not installed there. Pure mathematical tests pass; solver-level tests are supplied but require your Gurobi environment. Do not treat the code as fully solver-verified until those tests and a first completed experiment succeed.

## Important model limitations

The training policy in the global VF method has coefficients constrained to a five-dimensional `[-radius,radius]` correction box **around STAT**, not all unrestricted coefficients. This bound is part of the problem specification. The original STAT and CASE models are refitted within each fold; the 2013 run refits only on pre-2013 cases.

PW_VFCG optimizes **seed 0** but evaluates its obtained advice under all three seeds. Its returned training optimality gap is a seed-0 gap, not an optimality gap for the three-seed averaged performance. `PW_BOUNDED` instead uses all three hashes in its feasible objective and lower-bound optimization.

Some methods may require substantial additional strengthening before their global gaps close at realistic instance sizes. Reporting this is an intended outcome of the scalability experiment, not a reason to report partial results as exact. In particular, the coefficient-space lower bounds intentionally do **not** infer stability merely from multiple schedules having equal predicted cost; the second-best primary-gap certificate is sufficient but conservative. Also, the PW-VFCG strict hash-selection inequalities are represented by necessary non-strict cuts. Near-boundary cases can consequently remain unresolved even after extensive separation.

The code does not claim to reproduce Version 4's **incorrect** sufficient-but-not-necessary target-induction MILP constraints. `ANCHORED_TARGET` uses a mathematically valid but potentially slow restricted coefficient-region search instead. Its reported global gap is for the *restricted target set*, not for unrestricted direct SHARED or the old 74-coefficient ANCHORED method.

## 2026-10-21 audited restart (v2)

Use `bash run_theory_full.sh` with a NEW `artifacts/theory_full_v2` output root.
Old v1 checkpoints and certificates are not compatible: source hashes prevent
mixed-version recovery. The shared directional learner screens with Bank A and
confirms its proposed updates on a fresh Bank B each round; this screening
is not a claim of finite-sample improvement under the true duration law.
Global coefficient and per-instance searches retain dual bounds of every
region pruned under a nonzero tolerance. `GAP_CERTIFIED` means the stated
absolute gap is met; only `OPTIMAL_CERTIFIED` denotes near-zero closure.
Nested MILP time limits consume the method's remaining wall-clock budget.
Completed multiprocessing jobs are persisted as they finish, rather than in
submission order. `pooled_theory_summary.json` combines both development
folds, covers exactly 24,506 cases, and resamples calendar Mondays; unresolved
arms have no headline mean. Solver statuses and remaining gaps remain visible.

The GitHub CI suite includes solver integration tests, but no completed
24,506-case Gurobi experiment is claimed by CI. Large-model timeouts and
uncertified methods are scientific scalability results, not proofs of their
optima. The retrospective roster and assumed response law remain limitations.


## Version 3: full-scale comparative experiment (October 21)

**Branch:** `theory-full-experiment-v3`. Invoke `bash run_theory_full.sh`.
The default output is `artifacts/theory_full_v3`; version 2 outputs must not be
used as version 3 checkpoints. Manifest hashes include the exact-cache code.

**Stochastic oracle.** Each evaluation component computes one unrestricted
Bank-A SAA MILP and saves its assignment, objective value, dual bound, and
solver status. The reference is NOT an inducible recommendation and is never
passed through the deployment planner. Its fixed assignment is evaluated
using independent B/C banks and historical realized durations. The same
verified SAA solve is reused in PW_BOUNDED, PW_ANCHORED and PW_VFCG.
The pooled and fold reports show SAA model-A bounds, model-C and realized
costs, CASE-to-SAA opportunity intervals, and conservative inducibility
intervals. These intervals do **not** identify the shared-class gap, and
CASE-to-SAA is **not** the stochastic vs perfect-information gap. The
PW_VFCG global certificate remains seed-0-specific; three-seed deployed
averages are reported separately.

**Training reuse.** Certified deterministic planner assignments are memoized
in a WAL-enabled SQLite store keyed by the complete scheduling component,
exact float64 reviewed durations and tie seeds. No incomplete or inaccurate
planner solves are cached. Bank-A and independent Bank-B score evaluations
share only the *exact deployment*, never the scenario outcomes. Each score
record includes cache-hit and solver-call counts for cached hash deployments.
Global training still examines all specified candidates; caching changes
only redundant numerical solves, not the scientific objective.

**Interruption.** Completed directional rounds restart from `progress.json`;
uncompleted component tasks restart from their identity-checked cache.
The direct five-coefficient global branch-and-bound saves its entire active
queue, retired-region bounds, incumbent and processed node count in
`search_state.json`; a resumed run reprocesses at most the interrupted
node. Monolithic VF uses a disk-backed Gurobi node file and stores a MIP
start from any returned feasible solve. This is **not** full MIP-tree
checkpoint/restart and an operating-system-killed VF solve can lose its
unflushed search progress.

**Training budget.** `--search-seconds` limits directional training on a
per-method basis, checked between rounds. A budget-limited policy is labeled
`BUDGET_EXHAUSTED` in its `trained.json`, not globally optimal.
Individual candidate and lower-bound solvers also have explicit budgets.

**Interpretation.** No benchmark is substituted silently on failed solver
calls. All case-level records are retained, unresolved cases are counted,
and headline means are withheld for incomplete arms. No completed full-cohort
licensed Gurobi run is claimed by automated CI; the workstation campaign
remains necessary for empirical validation and scalability findings.
