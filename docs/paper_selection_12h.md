# Paper-selection experiment — four objectives, 12-hour workstation limit

Historical reproduction only: `bash run_selection_12h.sh` from `main` (the October 2026 comparison has completed).
This invokes `pilot/run_objective_selection.py`, not the older
proposal-only `pilot/run_selection_12h.py`. Do not run the other full
paper experiment concurrently.

## What is being compared

Train on 24,506 pre-2013 cases; evaluate frozen policies once on
8,774 2013 cases. The comparison uses one fixed deterministic planner
hash (seed 0), five shared policy features and nominal behavioral
response assumptions (alpha=0.8, h=30). The 2013 cohort was used in
earlier research and is retrospective, not pristine external validation.

The four separately fitted objectives begin at the same statistical
coefficients and use identical six-round coordinate-search directions
and steps 16, 8, 4, 2, 1, 0.5:

- **SHARED:** expected scenario scheduling cost following the deployed
  exact recommendation.
- **VF:** primary scheduling value plus scenario stop-loss surrogate.
  The global MIQP is optional diagnosis, not the training optimizer.
- **EG_LIB:** a bounded finite-library hinge-form Expected-GAP
  approximation. Its score is
  `max_z[g(z)-max(phi(z,d)-Q(d)-eps,0)]`, **without** an extra `Q(d)`.
  The current primary-optimal assignment is always included in the
  candidate maximum, even if it is absent from the precomputed library.
- **ANCHORED_T:** finite-library target-anchored surrogate.
  It is continuous piecewise linear in advice coefficients, not
  a hash-selected discrete objective.

The benchmark arms are BOOKED, CASE and STAT. All policies are frozen
before 2013 evaluation. Both full Bank A and Bank B must be exactly
scored over every fitting component for an update to be accepted;
partial-sample updates never count as training success. Exact
unresolved-component counts are logged after each screen and
confirmation round.

EG_LIB and ANCHORED_T are restricted finite-schedule approximations,
not unrestricted EG/Anchored optimization certificates. The libraries
start with recorded feasible assignments and attempt to incorporate
CASE, primary, and SAA schedules. Report library diversity and
coverage. An original-only one-schedule component may provide no
directional signal to Anchored. The original exact EG is evaluated on a
training-only sample to contextualize the finite-library approximation.

## Time budget

The stage ceilings total **10 h 36 min**, beneath the global
**11 h 30 min** Unix process-group supervisor (and the user's 12 h
limit). A hard kill retains previously checkpointed policies and
component results, but does not guarantee full optimization completion.

| Stage | Maximum |
|---|---:|
| Setup / features | 120 min |
| Library construction | 30 min |
| Training objective landscape scan | 30 min |
| SHARED | 75 min |
| VF | 45 min |
| EG_LIB | 45 min |
| ANCHORED_T | 30 min |
| Full-training own/deployed 2x2 and certificates | 100 min |
| Freeze | 1 min |
| Frozen 2013 evaluation | 150 min |
| Final report | 10 min |
| **Total** | **636 min (10 h 36 min)** |

Training components with >=25 cases are already scored one candidate
per shard for **all** methods. They now receive at least 120 seconds
for each primary/deployment solve and up to 240 seconds for the
bounded shard. Smaller components retain the default 45-second
solve allowance. A complete Bank A/B training comparison is required
even if this leads to an explicit unresolved status.

The training 2x2 reserves up to **1,200 s per method**, for both
its own and deployed-cost comparisons. The optional VF MIQP may use
only remaining certificate time. Test component-arm jobs each receive
an independent **300-second budget**, with a 120-second solver
allowance for components with at least 25 cases.

## Interpretation and files

The final report distinguishes certified-only method costs,
training objective vs deployed objective, exact case-weighted
coverage, solver failures, and retrospective fallback replay
(CASE, then the originally performed room assignment). Recorded
original assignments are **not** a deployable prospective fallback.

Calendar-week paired uncertainty summarizes retrospective comparisons;
neither a numerical winner nor a source of null results proves causal
improvement. If methods cannot be trained/evaluated credibly, report
the outcome as computationally unresolved.

Artifacts are written below `artifacts/objective_selection_12h/` by
default, including `objective_selection_2013/selection_summary.json`,
`objective_selection_2013/run_status.json`, frozen policy records,
training-round coverage, library diagnostics and per-arm test results.

```bash
git fetch origin
git switch main
git pull --ff-only origin paper-selection-12h
bash run_selection_12h.sh
```

Change the `OUT` directory whenever scientific source, time limits
or input data change. The run contract hashes the study's source and
workbook to prevent incompatible checkpoint reuse.
