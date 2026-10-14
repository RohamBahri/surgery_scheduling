# Anchored regret-surrogate learning: implementation and reproducibility

This is an implementation of the proposed **anchored regret-surrogate learning**
method on the existing frozen weekly rich-feature UHN cohort. It is not an
empirically validated improvement over CASE, VF, or VF-GAP until the local
Gurobi runs finish. Do not report proposed training gains as findings.

## What is preserved

* Identical 74-dimensional `X`, including the fixed booked-duration anchor,
  nonlinear surgeon response, display floor, attenuation, and rejection.
* Identical complete weekly physical-room feasibility and deterministic
  tie-breaking, including minimum moves and canonical tertiary assignment.
* 180 site-weeks (60 per site), 18,705 cases for full training; 54 later
  site-weeks, 5,779 cases for development **assessment only**. The 2013 final
  holdout is never fitted or evaluated by this runner.
* The original case, lasso, VF and (if supplied) VF-GAP coefficient files are
  read-only. This experiment writes only under `--output`.

## Mathematical design

For a fixed component, schedule `z`, and duration vector `d`, set

    phi(z,d) = 30*used_room_days(z) + 2.75*sum_j(max(load_j(z,d)-480,0)).

The full physical-room cost satisfies

    J(z,d) = 480*K - sum_i(d_i) - 30*n + phi(z,d).

## Exact schedule-specific anchored surrogate (v3)

The TGH experiments exposed a loss-approximation trade-off:
v1 trained on schedule-specific tangents and attained 9.8886 regret/case,
but its discovery oracle was inconsistent with that training loss; v2
used a common room-fixed-slope loss and exact routine closure but attained
only 10.7483 regret/case. These are **in-sample, prior-run** observations,
not results of this updated version.

For every component and chosen anchor `(y,beta)`, we train on

    S(d;y,beta) = gamma*phi(y,d)
       + max_{z, s in partial phi(z,beta)}
           [J(z,actual)-gamma*T_{z,s}^beta(d)] - V(actual),

where a physical-room subgradient has slope 0 on the idle side, OVER
(2.75) on the overtime side, and **either** at an exact 480-minute kink.
`T_{z,s}^beta(d)` is the exact touching supporting affine function
of `phi(z,d)` at `beta`. The maximization over these slopes at kinks
is important: the separation MILP may choose a different valid tangent
than a deterministic zero-at-kink convention. Both schedule assignments
**and the selected room slopes** are persisted and represented verbatim
in the training QP. When an anchor changes, prior cuts are re-tangentized
at the new beta; they are not falsely reused as touching tangents.

Each fixed library yields a convex piecewise-linear surrogate in d and
an exact response-aware DC function in the original 74 coefficients.
The proximal majorizer uses upper reviewed durations in the reference
term and lower reviewed durations in competitor cuts, so its QP contains
no scheduling or behavioral-regime binaries.

The exact separation MILP maximizes **this same tangent objective**.
Because it requires assignment-dependent overtime-regime binaries, it
can be expensive. Each outer-stage discovery round checks a budgeted,
deterministically selected batch, prioritized by apparent restricted-loss
improvement and rotating coverage of other components. A new exact
assignment/slope cut is stored even for a feasible time-limited oracle.
Closure is certified **only** for checked components whose maximization
dual bound is within tolerance of the known library maximum. Every
`adaptive_exact_tangent_separation` checkpoint records tested,
untested, closed, unresolved, added cuts, and maximum tested bound gap.
Full closure is never claimed when some components are untested.

This bound is valid for each fixed reference because all exact tangents
are global lower bounds on the corresponding fixed-schedule cost. For
the actual tolerance-lexicographic deployment use

    regret_deployed(d) <= S(d;y,beta) + gamma*delta(d),

with delta the certified primary-cost suboptimality of the deployed
assignment. Final `--stage certify` recomputes exact tangent separation
for all components and the appropriate primary planner bounds;
it does not turn restricted inner pDCA iterates into globally optimal
solutions.

The current deployed policy is a reference at every outer iteration,
winning ties in realized cost. A strictly better reference may be a
jointly attainable historical policy or a casewise box aspiration.
The latter is not evidence of reachability under one shared policy.
Unique-schedule and minimum-room/no-overtime certificates exclude
constant-cost components from training, while preserving their costs
and cases in all complete realized evaluations.

A strict decline in complete realized **training** cost is necessary
for accepting a candidate. Better surrogate values are not, by
themselves, accepted as better operating-room decisions. The entire
algorithm remains a finite-budget heuristic for a nonconvex bilevel
problem; it provides no global optimality or out-of-sample guarantee.

## Training and certification entry-point safeguards

`--exact-sep-batch` is accepted by both the runner and `learn()`, and is
validated as a nonnegative integer. An outer-loop integration test exercises
this argument rather than testing the mathematical modules in isolation.

For the final `--stage certify`, the complete deployed incumbent is loaded
from the same duration-hash cache used during training (or re-solved when
needed). The incumbent's exact post-response durations and selected schedules
are supplied to `prepare_components`, so references tied in realized cost
favor the current policy instead of alphabetical BOOKED precedence. The
reported bound remains conservative, but avoids this avoidable looseness.

For a **clean TGH v3-versus-v1 comparison**, omit `--initial-policy`
from **both** `references` and `train`, use CASE initialization, and
retain the same predeclared solver and pDCA budgets. Introducing v1 as a
warm start changes the reference library and creates a separate continuation
experiment, not a direct standalone comparison. This bug-fix commit does
**not** modify proximal coefficients, stopping heuristics or response
regime statistics; such changes require a separately fingerprinted run.

## Recovery from the Oct 14 interrupted extended-budget experiment

The extended-budget TGH run using 15 outer iterations, 20 pDCA steps,
three cut rounds, and exact-separation batch 192 reached **9.79714059
verified training regret per case** after four accepted updates. It then
stopped in outer iteration 5 because the former oracle accepted up to 0.02
absolute objective-reconstruction error but required the reconstructed
objective to lie below Gurobi's dual bound to within only 1e-6.

The revised oracle audits `checked`, `m.ObjVal`, and `m.ObjBound`
against the **same absolute/relative numeric tolerance**. A small floating
discrepancy is explicitly reported and the upper bound is reconciled to
the larger of the three values. A material discrepancy returns an
**uncertified** result with no usable upper bound, while still preserving
any valid feasible assignment. These flags are saved in the exact
separation checkpoint. This is a floating-point solver certificate, not
a claim of interval-arithmetic rigor.

Do **not** restart reference generation or delete the old directory.
For the specific interrupted experiment whose archived
`trainer_source_sha256` equals
`db51063aaaa62a4e7c17e4a76dd04706768a4848dcaf71d8a37a21e9a307c552`,
run the usual train command with the **same arguments** and add
`--resume-oracle-fix`. This one-time flag permits migration only when
the old source SHA is exactly allowlisted, the training state is
incomplete, and **all other saved fingerprints and hyperparameters**
match. The migration is recorded in
`anchored_source_migration.json`; no existing accepted coefficient,
reference, cut, or deployment-verification checkpoint is modified.
Existing completed separation components are reused, and the remaining
selected components are recomputed.

```bash
OUT=artifacts/anchored_TGH_extended
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
caffeinate -i python3.14 run_weekly_anchored.py \
  data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --artifacts artifacts/weekly_rich_v2 \
  --gap-model artifacts/weekly_gap_v1/gap_progress.json \
  --output "$OUT" --site TGH --stage train \
  --max-outer 15 --iterations 20 --closure-rounds 3 \
  --exact-sep-batch 192 --workers 15 --threads 1 \
  --seconds 300 --sep-seconds 90 --master-seconds 300 \
  --resume-oracle-fix
```

If the existing artifact directory uses a different code digest, a
different feature matrix, different baseline policies, or different
hyperparameters, migration intentionally fails. Start a newly
fingerprinted experiment rather than bypassing reproducibility checks.

## Warm starting and reproducibility

By default the learner starts from the frozen CASE coefficients, so
improvements can be attributed to this algorithm. Optionally supply
`--initial-policy /path/to/v1/anchored_progress.json` for a **separate**
continuation experiment that starts from the completed 74-coefficient
v1 policy (same site and training cases required). Do not credit the
seed's 11.50-to-9.89 reduction to the continuation algorithm.
Use a separate output directory for each run and pass the same
`--initial-policy` flag at **all** stages, because it is fingerprinted
as a read-only baseline, and contributes an additional SEED reference.
The same final score can be measured independently with `--stage evaluate`
on the frozen full-site model; validation is reserved for the completed
all-site run.

Use `--exact-sep-batch 96` for an adaptive full-site training pilot,
or `--exact-sep-batch 0` to ask exact separation on every active
component at every round (substantially more expensive). An
adaptive batch is a search strategy, **not** a full surrogate
optimality certificate. Final `--stage certify` is distinct from
the training-stage adaptive separation.


## Commands

Switch to the `anchored-regret-learning` branch of the existing GitHub
repository. Install the project's Python requirements including
Gurobi and a valid Gurobi license. Run `pytest -q tests/test_anchored_core.py`
and the original repository suite first.

The first computational experiment should cover a **full training site**:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
caffeinate -i python3.14 run_weekly_anchored.py \
  data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --artifacts artifacts/weekly_rich_v2 \
  --gap-model artifacts/weekly_gap_v1/gap_progress.json \
  --output artifacts/anchored_TGH_v1 \
  --site TGH --stage references --workers 15 --threads 1 --seconds 300
```

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
caffeinate -i python3.14 run_weekly_anchored.py \
  data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --artifacts artifacts/weekly_rich_v2 \
  --gap-model artifacts/weekly_gap_v1/gap_progress.json \
  --output artifacts/anchored_TGH_v1 \
  --site TGH --stage train --workers 15 --threads 1 \
  --seconds 300 --sep-seconds 90 --master-seconds 300
```

For full 180-week training, use **a fresh**
`--output artifacts/anchored_all_v1 --site all`; run `--stage references`,
then `--stage train`. Do **not** copy full-site model progress from the pilot:
the denominator, optimization coupling, and feasible policy problem differ.
The feature matrix is never refitted and the initial policy remains CASE.

After full training (still **before** accessing validation):

```bash
python3.14 run_weekly_anchored.py \
  data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --artifacts artifacts/weekly_rich_v2 \
  --gap-model artifacts/weekly_gap_v1/gap_progress.json \
  --output artifacts/anchored_all_v1 --site all --stage certify \
  --report-gap --workers 15 --threads 1 --seconds 600
```

Finally, evaluate **once** on development validation and paired calendar-week
bootstrap (2013 remains untouched):

```bash
python3.14 run_weekly_anchored.py \
  data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --artifacts artifacts/weekly_rich_v2 \
  --gap-model artifacts/weekly_gap_v1/gap_progress.json \
  --output artifacts/anchored_all_v1 --site all --stage evaluate \
  --workers 15 --threads 1 --seconds 600
```

### Artifacts and interpretation

* `anchored_settings.json`: source hashes, input hashes, all 74 feature names,
  response and solver-independent optimization settings. A mismatch refuses
  to use the same output directory.
* `anchored_references.json`: *actual tie-broken* reference assignments at
  booked, independently reachable probes and earlier model predictions.
  Probe vectors are **not** claimed jointly attainable by a shared policy.
  A reference needs an exact certified planner before it enters training.
* `anchored_progress.json`: coefficients, accepted actual-regret improvements,
  attempted candidate policies, pDCA status, non-intercept coefficient changes,
  objective history, discovered competitors and explicit stopping reason.
* `anchored_verify_<sha256>.json`: cached exact deployed selections across
  trial/current names, keyed by the exact post-response duration SHA. Increase `--seconds` on a timeout; neither
  a primary incumbent nor an incomplete tie-break can be used as acceptance.
* `anchored_exact_separation_*.json`: adaptive exact-tangent discovery,
  with primal-dual bounds on the **same exact tangent objective used for
  coefficient training**, for selected components. Untested and unresolved
  components are explicitly counted; optional final full closure uses
  `--stage certify`.
* `anchored_bound_certificates.json` and `anchored_bound_summary.json`:
  conditional full-tangent upper-bound and tie-safe correction; optional
  VF-GAP intervals. Inspect `separation_exact` / `vf_gap_all_exact` and
  incomplete components before using claims in a paper.
* `anchored_validation.csv` and `anchored_validation_summary.json`: all
  identical site-weeks across baselines, with a 95% bootstrap interval using
  **calendar weeks as paired resampling units**.

### Solver/learning behavior

The default `lambda_l1=0` is deliberate: this method uses recommendation-space
proximal stabilization and accepts only a strict realized-cost improvement.
The previous CASE noise-calibrated L1 rule is **not** silently repurposed for
an unrelated anchored loss. A justified new L1 rule would require a separate
score derivation, not simply multiplying CASE's penalty by 1/2.08. The defaults
`gamma=2`, `prox=.15`, `coef_prox=1e-5`, `iterations=12`,
`closure_rounds=2`, `exact_sep_batch=96`, `max_outer=5` are **predeclared implementation choices**,
not estimated best hyperparameters. Altering them creates a new experiment
and requires an isolated output directory.

This experiment is not certified to produce positive generalization gains.
The 2013 holdout cannot be used for iterative design; development validation
must not be repeatedly inspected to tune anchors or hyperparameters.
