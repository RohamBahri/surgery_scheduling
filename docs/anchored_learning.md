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

For an outer iteration, let `z_current` be the exact deployed schedule.
The anchor `(y,beta)` is chosen from certified references (including the
current policy's own schedule and post-response duration vector). A reference
must have **strictly smaller realized cost** to displace the current policy;
ties favor the current policy, not alphabetical source names. For each
physical room, set `s_j=2.75` when that room in `z_current` exceeds 480
minutes at `beta`, and `s_j=0` otherwise. These room slopes are constant
across **all** competitor schedules in one optimization stage.

For each feasible competing schedule `z`, define the affine lower bound

    T_s(z,d) = 30*used_room_days(z)
             + sum_j s_j*(load_j(z,d)-480*used_j(z)).

This is valid for every duration vector and every schedule, but it need not
touch the true scheduling cost for arbitrary competing schedules at `beta`.
The actual **training objective** is the fixed-room-slope anchored surrogate

    S_s(d;y,beta) = gamma*phi(y,d)
                   + max_z [J(z,actual)-gamma*T_s(z,d)]
                   - V(actual).

Because `T_s(z,d)<=phi(z,d)` and `min_z phi(z,d)<=phi(y,d)`,
`S_s` is a convex-in-duration upper bound on the primary VF-GAP surrogate.
For complete deterministic deployment, with primary-optimality discrepancy
`delta(d)`, the rigorous certificate is

    regret_deployed(d) <= S_s(d;y,beta) + gamma*delta(d).

No epsilon movement penalty replaces the lexicographic planner. The
`--stage certify` task separately computes an **exact schedule-specific
tangent** anchored bound; it is valid independently but is not identical to
the fixed-slope training loss. Both certify an upper bound on regret (with
the tie correction), but neither proves out-of-sample improvements.

For fixed anchors, `anchored_master.py` uses a restricted competitor
library and the original piecewise-linear surgeon response. Its touching
convex majorizer uses `b+p(w)-linearized(q)` in the reference term and
`b+linearized(p)-q(w)` in each competitor cut, with convex `p,q`. The
proximal term measures the change in raw recommendations `X(w-w_k)`,
and every original feature coefficient is retained. The resulting QP has
no scheduling or response-regime binaries. Inner pDCA descent refers to
the **fixed-library** surrogate; each added cut changes that objective.

Routine competitor separation solves the **same fixed-slope objective**
as the training master. Its maximization MILP provides both an incumbent
and a dual bound. A cut library is called closed at the candidate only when
the separation dual bound is within `1e-5` of its known library maximum;
unresolved components are explicitly reported. Closure at candidate
iterates is not a proof that the full fixed-slope objective was optimized
globally across all coefficients or all intermediate pDCA iterates.
Exact tangent separation remains available for independent final
certification and comparison to VF-GAP.

All components with a **provably unique feasible schedule** are excluded
from the learning master: their deployed realized costs are policy
independent. The previous no-overtime/minimum-room analytic certificate
is an additional sufficient condition for excluding other policy-locked
components. All excluded cases stay in the global per-case normalization
and the complete regret evaluation.

References may originate from shared-policy witnesses or independently
reachable box probes. The latter are aspirations, not proofs of joint
reachability through the shared 74-coefficient policy. Training-only
candidate selection is always checked using the complete deterministic
planner. Every accepted policy strictly decreases unregularized training
regret; a failed or timed-out planner is never accepted. Such empirical
descent does not imply local/global optimum or validation improvement.
An exhausted cut/search budget must not be described as policy-class
impossibility.


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
* `anchored_separation_*.json`: fast discovery of promising competitors,
  with primal-dual bounds on the **same fixed-slope objective used for
  coefficient training**. Exact closure at evaluated candidates is tracked
  separately from the exact-tangent final certificate.
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
`closure_rounds=2`, `max_outer=5` are **predeclared implementation choices**,
not estimated best hyperparameters. Altering them creates a new experiment
and requires an isolated output directory.

This experiment is not certified to produce positive generalization gains.
The 2013 holdout cannot be used for iterative design; development validation
must not be repeatedly inspected to tune anchors or hyperparameters.
