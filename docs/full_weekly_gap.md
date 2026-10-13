# Full-training VF-GAP: exact-oracle-certified objective with schedule-library proposals

This is a **new fitted 74-dimensional correction policy**, not the VF-GAP
score of the existing LASSO or VF policy. Frozen rich matrices, surgeon
response, roster, and the exact two-oracle weekly MIPs are unchanged.

The training loss is

    (gamma * min_z J(z, post(w))
     + max_z {J(z, actual) - gamma J(z, post(w))}) / n
      + lambda * sum_{j > 0} |w_j|,

where `post(w)=booked+response(clip(X@w))`, `n=18705`,
`gamma=2`, and the hindsight optimum `min_z J(z,actual)` is
omitted as a constant. The loss is a valid upper bound on regret up to
that constant; it is pointwise no larger than the original VF bound.
The nonconvex objective is **not** claimed globally optimized.

The algorithm stores certified primary-optimal planner schedules and
certified adversarial maximizers. A **restricted schedule library**
gives differentiable, smoothed **candidate proposals only**. The
restricted-library objective has *no guaranteed bound direction*.
Every candidate, and every accepted update, is evaluated afresh with
both complete exact-oracle MIPs on **all 180 training group-weeks**.
Only a strict decrease in the UNSMOOTHED, fully certified,
regularized VF-GAP objective counts as progress. No uncertified result
is accepted. Failed weeks are checkpointed, with component status,
so increasing `--seconds` resumes them without rerunning other
certified weeks.

The step size has a 74-dimensional coefficient trust region; this is
not a common offset grid, intercept-only search, or a reduced set of
policy coefficients. Three line-search fractions (1, 0.5, 0.25) are
validated by the complete oracles; the resulting stage may take many
hours or longer depending on the maximization MIPs. The solver
concurrency budget is 15 CPU cores. Mac `caffeinate` is optional.

## 1. Update checkout (preserve local modifications)

```bash
cd "/Users/roham/Desktop/Surgery project/code"
git status --short
git fetch origin
git switch weekly-supported-days
git pull --ff-only origin weekly-supported-days
python3.14 -m pytest -q
```

No rerun of `features`, `fit`, `case`, or `vf` is required.
The new trainer is independently fingerprinted, and all output goes
into a distinct directory; do not overwrite `weekly_rich_v2`.

## 2. Train exact-certified VF-GAP (ALL 60 weeks/site; 15 cores)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
caffeinate -i python3.14 run_weekly_gap.py \
  data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --artifacts artifacts/weekly_rich_v2 \
  --output artifacts/weekly_gap_v1 \
  --stage train --initial CASE \
  --alpha 0.8 --h 30 --gamma 2 \
  --max-outer 5 --iterations 150 \
  --trust 6 --temperature 1.5 \
  --workers 15 --threads 1 --seconds 300
```

The 15-core budget is **15 independent one-thread Gurobi workers**
during oracle evaluation; the coefficient proposal uses a single
SciPy optimization process. If the run stops due to a nonoptimal
weekly planning/adversarial oracle, **run exactly the same command with
`--seconds 600`**, then `--seconds 1200` if needed. Do not change
other settings when resuming.

Output:
- `artifacts/weekly_gap_v1/gap_progress.json` — certified loss, accepted
  steps, REALIZED training cost diagnostics, and trained 74 coefficients.
- `gap_current_XX.json` — all training week assignments and exact-oracle
  scores at accepted policies.
- `gap_candidate_XX_Y.json` — full-training candidate certificates,
  resumable when interrupted.
- `gap_settings.json` — strict input/policy/source provenance guard.

Convergence means local progress stopped under the proposal/line-search
scheme, **not** that VF-GAP or regret reached global optimality.
Decreasing the VF-GAP bound also does not imply decreasing realized
regret. The training report prints both values separately.

## 3. Evaluate FIVE policies on ALL 18 later validation weeks (15 cores)

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
caffeinate -i python3.14 run_weekly_gap.py \
  data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --artifacts artifacts/weekly_rich_v2 \
  --output artifacts/weekly_gap_v1 \
  --stage evaluate --initial CASE \
  --alpha 0.8 --h 30 --gamma 2 \
  --max-outer 5 --iterations 150 \
  --trust 6 --temperature 1.5 \
  --workers 15 --threads 1 --seconds 300
```

A freshly shared realized hindsight oracle is used for BOOKED,
LASSO, CASE, VF, and VF_GAP in each group-week. Reports include
post-response MAE, realized OR cost, regret, overtime, overloaded
blocks, VF-GAP certificate for the learned VF_GAP policy, and paired
18-calendar-week bootstrap confidence intervals against every
baseline, holding all three sites together within each resample.

Outputs:
- `artifacts/weekly_gap_v1/gap_evaluation.csv`
- `artifacts/weekly_gap_v1/gap_evaluation_summary.json`

The 2013 test period remains untouched. This exploratory development
validation cannot by itself certify a positive population effect.

## Scope and limitations

The weekly planning and adversarial subproblems are solved exactly
when marked `complete`. The coefficient problem remains nonconvex.
The schedule-library gradients guide search; the *full-oracle*
certificates enforce correct acceptance. The trainer does not
silently substitute the library surrogate for VF-GAP.

The published planner tie-break is used for validation schedules.
Training optimizes primary optimal values, so exact optimal ties
can give distinct realized costs; this caveat matters when evaluating
direct regret. The bound itself is value-based and tie-invariant.
