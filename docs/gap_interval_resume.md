# Interval-certified VF-GAP — resume the existing 180-week run

This revision fixes the zero-MIP-gap bottleneck. A component's solver may time
out with a feasible incumbent and valid global bound. For a predicted planning
minimum P and adversarial maximum A, exact-oracle bounds are retained:

    Lp <= P <= Up, La <= A <= Ua
    gamma*Lp + La <= gamma*P + A <= gamma*Up + Ua.

The algorithm aggregates intervals for every case across all 180 group-weeks
and adds the same L1 penalty to both endpoints. It accepts only if

    candidate.UB < incumbent.LB - 1e-5  (per case).

Thus **every accepted step rigorously improves the exact VF-GAP objective**,
even if a component ends in TIME_LIMIT. Uncertified feasible assignments are
not described as optimal. When the intervals overlap, the runner selectively
retries the largest-uncertainty components, with additional solver time. If
still inconclusive it skips that candidate fraction; if all fractions are
inconclusive, training PAUSES, it does not falsely report convergence.

The checkpoint design retains old exact group-week records unchanged and
stores any newly incomplete groups as a list of component-level planner and
adversarial primal/dual certificates. Certified components are never re-solved
during a retry, and repeated time-limited solves merge the strongest valid
bounds and best incumbents. Existing CASE/VF artifacts, the frozen 74 features,
and the precise weekly planner implementation are unchanged.

**Important:** This fixes the certification and resumability architecture.
It does NOT implement DCA, gamma continuation, a new optimizer, or an exact
bilevel optimization of the 74 coefficients.

## Commands

First update the branch:

```bash
cd "/Users/roham/Desktop/Surgery project/code"
git status --short
git fetch origin
git switch weekly-supported-days
git pull --ff-only origin weekly-supported-days
python3.14 -m pytest -q
```

The code source fingerprint has legitimately changed. Adopt old run metadata
explicitly (one time only), preserving the completed weeks:

```bash
python3.14 run_weekly_gap.py \
  data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --artifacts artifacts/weekly_rich_v2 \
  --output artifacts/weekly_gap_v1 \
  --stage adopt --initial CASE \
  --alpha 0.8 --h 30 --gamma 2 \
  --max-outer 5 --iterations 150 \
  --trust 6 --temperature 1.5
```

The adopter independently reconstructs the SHA256 fingerprint of **the audited
previous Git commit `8b68f100f52f38455e5b4cd06ed79fc5fc50d9f3`** from
your local Git object database. It rejects any differences other than the
trainer source fingerprint. It verifies the headers of every old candidate
and current-policy checkpoint and retains `gap_settings_before_intervals.json`
(and the original progress file if one existed). If you have a shallow clone,
fetch the required commit before running adopt.

Resume training with a 300-second baseline solve and a 1200-second *selective*
refinement budget only if the full objective intervals overlap:

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
  --workers 15 --threads 1 \
  --seconds 300 --refine-seconds 1200 --refinement-budget 1
```

The 179 already certified group-week records in
`gap_candidate_00_0.json` are reused. Only the previously failed group-week
is reconstructed; newly generated certificate records save separate
components. Under the current cache layout the original incomplete record
does not contain component details, so that one week may need to solve its
other components once before individual reuse becomes available.

Inspect:
`artifacts/weekly_gap_v1/gap_progress.json` and
`artifacts/weekly_gap_v1/gap_candidate_00_0.json`.

If training reports `paused`, preserve the files and rerun the command with
`--refine-seconds 2400` (and, if needed, `--refinement-budget 2`). If an
oracle has neither a feasible incumbent nor a finite dual bound, the run
stops with an explicit error and its partial results remain on disk; increase
`--seconds` in that case.

Upon successful completion, use the same evaluation command in
`docs/full_weekly_gap.md`; the validation scorer independently requires
exact completed planners and adversaries. It never treats a time-limited
training incumbent as an exact validation oracle.

## Bound and result interpretation

`regularized_gap_lower_per_case` and
`regularized_gap_upper_per_case` are rigorous bounds on the true
regularized training surrogate (excluding the policy-invariant hindsight
constant). When a time-limited component remains,
`regularized_gap_without_hindsight_per_case` is **null**, not a fabricated
point estimate.

`realized_plan_cost_per_case` is calculated from feasible primary planning
incumbents under actual durations. It is diagnostic only and is not necessarily
the realized cost of the optimally solved tie-broken deployed scheduler.

A strict interval separation certifies descent of VF-GAP, not descent of
realized regret, global optimality in w, or an exact schedule for every
time-limited component.
