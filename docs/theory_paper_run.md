# One full surgery-planning method comparison

**Branch:** `theory-single-split-budgeted`

Run `bash run_theory_full.sh` from the repository root. This version writes
only to `artifacts/theory_single_split_paper_v2`. Do not reuse earlier
v1/v3 artifacts or run two copies of the experiment simultaneously.

## Data and objective

Train once on all 24,506 pre-2013 UHN cases; evaluate the same frozen
method definitions on all 8,774 cases from 2013. The training-only
residual-fitting cross-fitting is not an outer two-fold policy comparison.
Previous research has inspected some 2013 outcomes, so describe this
as a retrospective method-selection study, not a pristine sealed test.

All available model families remain in the code. The **default** full
competition is:

- Fitted policies: STAT (current baseline), **STAT_PLUS** (bounded
  multistart case-loss control), VF_SHARED, SHARED, EG, OBS, ROBUST.
- Operational baselines: BOOKED, CASE, RISK.
- Flexible advice: PER_WEEK_ROBUST, QFILL and VF_PW.
- Nondeployable references: unrestricted SAA and realized hindsight.

GLOBAL, ANCHORED_TARGET, PW_VFCG, PW_BOUNDED, PW_ANCHORED,
legacy SHARED and single-seed PER_WEEK are **not deleted**. These are
optional algorithm/refinement experiments; running them alongside the
full default comparison would spend large budgets on overlapping solver
approaches. Use `--methods` for shared training methods and `--eval-arms`
to opt into alternative test methods. The default experiment intentionally
does not run expensive alternative planner/response mechanism solves.

## Budgets and CPU use

The default launcher uses 15 cores. Each independently scheduled
component or coefficient-shard worker has one Gurobi thread. The
monolithic full-cohort VF model is allowed 15 threads when running alone.

| Stage | Default |
|---|---:|
| Feature and duration setup | 4h cooperative stage limit |
| STAT_PLUS multistart training | 30 min |
| Each other shared method | 4h soft stage budget |
| Coupled shared VF build and optimization | 4h |
| Each optional direct/anchored global search | 4h |
| Single planner/solver call | 120s, adjusted to time remaining |
| Candidate shard | 1500s, inheriting the parent deadline |
| Candidate search round | 2h |
| One test component **per method** | 1500s |
| Per-instance search inside an arm | 480s |

These are optimization budgets, not predictions of completion time.
Gurobi time limits and cooperative Python deadline checks do not
guarantee a strictly enforced OS-level kill during a long construction
step or solver shutdown. Interrupted completed work is checkpointed,
and unresolved methods keep their reason, timings, and valid incumbents
where present. Incorrect primal/dual bounds are never accepted as
timeouts.

## Output semantics

- Each `full_evaluation/per_arm/<component>__<arm>.json` is saved as soon
  as its method completes, irrespective of other methods.
- Each method has an **independent wall budget**: late-running methods
  cannot inherit zero time from the earlier methods on the same component.
- When an optimizer returns a previously certified advice duration and
  three deployed assignments, those assignments are reused directly.
- The aggregated `theory_summary.json` reports certified coverage,
  cost on the entire cohort only when fully certified, and a *separate*
  predefined CASE-fallback operational policy for every arm.
- All attempted work, including unresolved task time and internal errors,
  contributes to resource accounting. Unexpected mathematical errors
  invalidate audit status but do not erase other completed tasks.
- `SAA` is never portrayed as inducible advice. Its primal/dual bounds
  refer to the specified scenario bank, not realized hospital outcomes.
- `QFILL` can change the selected partition; its effect is measured
  rather than assumed zero. The repository contains a regression for this.

A single independent test split cannot establish causal benefits from
behavioral responses. The response maps remain modeled scenarios. The
retrospective roster, component-level tolerance, and linear overtime
planner also remain fixed primary-study limitations.

## Reproduction

```bash
git fetch origin
git switch theory-single-split-budgeted
git pull --ff-only origin theory-single-split-budgeted
bash run_theory_full.sh
```

To change budgets, set environment variables before launching, such
as `CORES=12 EVAL_ARM_SECONDS=1800 bash run_theory_full.sh`. Use
`OUT=artifacts/a_new_unique_name` whenever the source, methods or
budget specification changes; a content-addressed manifest prevents
mixing incompatible checkpoints.
