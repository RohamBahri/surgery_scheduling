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

For each **fixed reference pair** `(y,beta)`, every feasible competitor `z`
gets its own supporting tangent `T_z^beta(d)` to `phi(z,d)` at `beta`. The
full anchored surrogate is

    S(d;y,beta) = gamma*phi(y,d)
       + max_z[J(z,actual)-gamma*T_z^beta(d)] - V(actual).

`S` is convex in `d` and, for `gamma>=1`, bounds empirical regret when the
primary planner is exact. For actual deployed deterministic tie-breaking use

    regret_deployed(d) <= S(d;y,beta) + gamma*delta(d),

where `delta` is the deployed schedule's primary-cost excess above the exact
predicted optimum. The certification stage computes a rigorous upper bound on
`delta`, including Gurobi numerical tolerances. No arbitrary epsilon movement
penalty is substituted for the actual tertiary tie-break.

`anchored_master.py` minimizes the **restricted library** version of S, not the
full S. Each fixed-library objective has an exact DC representation in w by
composing a convex-in-d surrogate with the full piecewise-linear response.
A touching convex majorizer uses (p - q) with `p,q` convex, upper duration
`b + p(w) - linearized(q)` in the reference term, and lower duration
`b + linearized(p) - q(w)` in each competitor cut. A proximal squared change
in **all case-wise raw recommendation scores X@w** stabilizes the update.
The optimizer is a convex quadratic Gurobi program with linear epigraphs and
all 74 coefficient variables, **without schedule or response binaries**.

Routine competitor discovery uses a weaker, but globally valid, per-room
fixed-slope underestimator. Its maximization oracle is less elaborate than
schedule-specific tangent separation. **It does not certify closure of the
exact-tangent restricted library or full S.** The saved records are named
`anchored_separation_...json` and are used for discovery only; they must not
be cited as the exact anchored regret bound. The optional `--stage certify`
solves the full schedule-specific tangent oracle with binary room-regime
indicators and primal-dual solver bounds. With `--report-gap`, it separately
computes bounds on the primary VF-GAP quantity.

Fixed-reference pDCA subproblems may find only a critical point. Updating the
references changes the surrogate. The outer algorithm has only a **strict
realized-training-regret acceptance guarantee**, conditional on fully
certified deterministic planner solves. An exhausted search budget does not
prove local or global optimality, and rejection does not prove the shared
policy class is too restrictive. A failed MILP is never relabeled optimal.

## Commands

From the **root of the existing GitHub repository**, first apply the supplied
new-file patch (or copy the `surgery/`, `tests/`, `docs/` and runner files into
the matching locations). Install the project's Python requirements including
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
* `anchored_verify_*.json`: cached exact deployed selections, keyed by the
  exact post-response duration SHA. Increase `--seconds` on a timeout; neither
  a primary incumbent nor an incomplete tie-break can be used as acceptance.
* `anchored_separation_*.json`: fast discovery of promising competitors,
  with correct maximization primal-dual bounds on the *weaker fixed-slope*
  surrogate. Not a tight full-S certificate.
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
