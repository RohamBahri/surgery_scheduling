# Paper method selection — one 12-hour workstation window

Run **only** `bash run_selection_12h.sh` on the `paper-selection-12h`
branch. Do not run the older `run_theory_full.sh` simultaneously.

This is a **new**, deliberately focused experiment. It is not the old
15-arm experiment with shorter solve limits.

## Question and scope

* Train once on **24,506** pre-2013 UHN cases; evaluate the frozen policies
  on the **8,774** 2013 cases.
* Compare BOOKED, CASE, STAT_PLUS, SHARED_BASE, SHARED_PLUS, VF_PLUS,
  GAP_PLUS, and ANCHORED_PLUS.
* All methods use the **same response law** (alpha 0.8, h 30) and
  **one fixed deterministic primary hash seed (0)**. This intentionally
  differs from previous three-seed model averages. Do not compare
  the new absolute numerical results to earlier three-seed results
  without noting the estimand change.
* The 2013 cohort has been examined in previous project iterations.
  The resulting comparison is retrospective, **not pristine external
  validation**.
* The statistical outcome is **realized modeled operating-room cost per
  case**, paired by calendar Monday. Simulated Bank A/B scenarios and
  historical realized costs are not conflated.

## Interpretation of the learned policies

| Arm | What is actually fitted |
|---|---|
| SHARED_BASE | Bounded lightweight directional direct-cost search |
| SHARED_PLUS | Expanded direct-cost search initialized from SHARED_BASE; optionally compares VF coefficients as a warm start |
| VF_PLUS | Existing globally bounded single-level VF MIQP, returning a feasible coefficient incumbent and a bound whenever available |
| GAP_PLUS | Short EG-adversarial **interval-based candidate screen** followed by a direct-cost check against SHARED |
| ANCHORED_PLUS | Actual schedule-specific **anchored tangent surrogate**, optimized in the same five-coefficient class and directly confirmed against SHARED |
| STAT_PLUS | Bounded multistart asymmetric-duration-loss fit on the same five features |

**Important:** GAP_PLUS and ANCHORED_PLUS are bounded *proposal/refinement
implementations*, **not** fully optimized full-cohort EG or original 74-feature
anchored pDCA. They are a computationally affordable way to test whether
their mathematical mechanisms produce better SHARED-type advice. An
unsuccessful proposal cannot prove the original mathematical objective
is inferior.

Where a full training confirmation is incomplete, the algorithm may accept
a proposal using only a **paired, exactly evaluated training subset of at
least 98% of cases**, if its gain clears the explicit per-case threshold.
That is a **heuristic training decision**, recorded as
`PAIRED_TRAINING_HEURISTIC`, never a full-cohort optimization certificate.
These decisions use training outcomes only.

## Hard supervised time budget

The coordinator starts each phase in its own Unix process group, retains
completed artifacts, and terminates the process group at its **stage wall
deadline**. Total allocated stage ceilings sum to **11 h 16 min**, leaving
14 minutes before the global **11 h 30 min** controller ceiling (and 30
minutes before the user's 12-hour limit). The watchdog terminates
Gurobi and model-build subprocesses, not just cooperative loops.

| Phase | Maximum |
|---|---:|
| Data preparation and frozen feature context | 120 min |
| STAT_PLUS | 15 min |
| SHARED_BASE | 45 min |
| VF_PLUS | 75 min |
| SHARED_PLUS | 75 min |
| GAP_PLUS candidate generation | 45 min |
| ANCHORED_PLUS refinement | 75 min |
| Freeze policies | 1 min |
| Complete 2013 evaluation | 210 min |
| Report | 15 min |
| **Total** | **676 min** |

The exact solver time and number of optimizations completed **cannot** be
guaranteed. The global ceiling is guaranteed only under normal process
supervision on a Unix-like machine; power loss, SIGKILL of the supervisor,
filesystem failures, or resource-manager shutdowns can prevent a final
report. Avoid another CPU-intensive workload competing for the 15 cores.

## Performance and checkpoints

* Feature construction is executed **once**, and a prepared context is
  saved to `selection/context.pkl` for reuse by all later stages.
* A single SQLite exact-deployment cache is shared by candidate
  screening and direct-cost confirmation. Completed scoring shards,
  training state, per-component test results, and policy files are saved
  independently.
* Directional SHARED screening uses five coordinate directions,
  a smaller 32-component shortlist, and at most two full-cohort
  candidates per round. Unfinished solver tasks can receive longer
  retries. A single uncertified component no longer automatically
  converts every 98%-covered improvement into a nominal no-update.
* Main test arms use separate task budgets and a **90-second** initial
  solver allowance, and share certified identical advice deployments.
  Unresolved tasks remain explicitly unresolved.
* When an exact deployed recommendation is missing, a separate **all-case
  fallback replay** uses CASE if certified; otherwise it uses the
  workbook's recorded feasible original assignment. That performed
  assignment is **retrospective**, not available as an assumed prospective
  room plan. It is not represented as the advice method's exact
  selected plan or claimed to be an implementable operational fallback.
* A high-cost or unfinished solver does not stop subsequent policy
  families. A fatal mathematical inconsistency is logged and never
  converted to a certified plan.

### File locations

`artifacts/paper_selection_12h/selection/run_status.json`
: stage status, elapsed seconds and logs

`artifacts/paper_selection_12h/selection/policies/*.json`
: coefficient vector, optimization/provenance status and candidates

`artifacts/paper_selection_12h/selection/frozen.json`
: final pre-test policy freeze

`artifacts/paper_selection_12h/single_split_2013/full_evaluation/per_arm/`
: individual component-policy exact test records

`artifacts/paper_selection_12h/selection/selection_summary.json`
: final case-weighted result, coverage, operational fallback,
  simultaneous week-clustered bootstrap comparisons, and a provisional
  method-selection outcome.

The preliminary training libraries and fitted policies are never
overwritten because a later **evaluation-only** diagnostic is added.
For a new scientifically different training campaign use a **new**
`OUT` directory.

## Before claiming a winner

1. Report **exact case-weighted deployment coverage** for every learned
   arm. A candidate needs at least 98% exact coverage to be selected.
2. Confirm that SHARED_PLUS also has at least 98% exact coverage.
3. Use **simultaneous Bonferroni week-clustered bootstrap intervals**
   for seven predeclared comparisons against SHARED_PLUS. Nominate a
   challenger only if its upper interval is below **-0.5 modeled cost
   units per case**, with acceptable optimizer provenance.
4. Treat a `PAIRED_TRAINING_HEURISTIC` as conditional evidence, not
   a fully optimized method. If intervals overlap, select the simpler
   direct SHARED approach, or report that no unique winner was
   established. Never interpret an untrained GAP/VF/ANCHORED arm as
   proof against its objective.
5. If exact coverage is low, report the **fallback operational policy**
   separately and do not claim strong evidence about the original
   optimizer.

## Launch

```bash
git fetch origin
git switch paper-selection-12h
git pull --ff-only origin paper-selection-12h
bash run_selection_12h.sh
```

Override before starting only if necessary:
`CORES=12 OUT=artifacts/paper_selection_12h_custom bash run_selection_12h.sh`.

All scientific module fingerprints, workbook fingerprint, and time budgets
are stored in `selection/run_contract.json`; incompatible resumes are
rejected. Do not open 2013 results and then modify the training methods
within the same selection claim.
