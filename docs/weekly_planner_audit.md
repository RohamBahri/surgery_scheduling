# Fixed-capacity weekly planner audit

This change evaluates a new operational planner before selecting a capacity proxy
or integrating it with VF. The legacy block-opening planner remains available.
`run_final_vf_experiment.py` is explicitly marked **pre-planner-audit**, and is not
the final paper pipeline until this audit is resolved.

## Run on the UHN workbook

From the repository root, with Python 3.10+ and a Gurobi license that supports the
full weekly models:

```bash
python -m pip install -e '.[dev]'
python -m pytest tests/test_planning/test_fixed_capacity.py -q
python scripts/run_weekly_planner_audit.py \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --quick --artifact-root artifacts/weekly_audit_quick
```

The new test file is a small mathematical regression suite, not another research
pipeline. It checks the cost identity against hand calculations and exhaustive
enumeration, formulation equivalence, safe symmetry, turnover, fixed idle
capacity, fallback, reporting, and legacy pricing compatibility.

Quick mode uses training weeks 0, 35, and 71, selected before examining outcomes.
All eligibility and template estimation still uses the same 72 training weeks.
Quick budgets are 2 seconds per ordinary solve and 10 seconds per zero-gap
attempt. There are 126 solves, so the nominal sum of budgets is nine minutes,
plus model construction and reporting. A requested time limit may be exceeded
slightly by Gurobi during termination.

For the complete training audit, use a fresh output directory:

```bash
python scripts/run_weekly_planner_audit.py \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --artifact-root artifacts/weekly_audit_full --threads 8
```

Full defaults are 10 seconds per ordinary solve and 60 seconds per exact attempt.
Across the full sensitivity matrix, the nominal serial budget is 20.4 hours;
solves that finish early reduce this. Use `--time-limit` and `--exact-time-limit`
to change computational budgets. The chosen budgets are recorded in MODEL_SPEC.
The scientific assumptions and primary settings cannot be changed through those
flags. Phi and Psi always receive matching budgets in their paired comparison.

To inspect all 72 weeks without any optimization:

```bash
python scripts/run_weekly_planner_audit.py \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --structure-only --artifact-root artifacts/weekly_audit_structure
```

This explicitly reports `STRUCTURE_ONLY`; it provides no solver-scale result.
Output directories must be empty to prevent overwriting a prior registration.
Source archives without a `.git` directory are supported; source hashes are
recorded alongside the input and training-ID fingerprints.

## Scientific model

- Capacity is exogenous. Every allocated block is fixed, has zero activation
  cost, and contributes 480 minutes even when unused.
- Every case is assigned exactly once. Day is flexible across all compatible
  room instances in the weekly roster. There is no deferral, hard overtime cap,
  or surgeon-room-count constraint.
- Primary eligibility is same site and raw service-room compatibility in at
  least three distinct training weeks. Thresholds 1, 2, 3, 5 are audited.
  Empty primary eligibility falls back to all allocated blocks at the same
  site. No allocated block at that site raises `CapacityModelError`.
- Primary turnover is 30 minutes, with 0/20/30/40 sensitivity. A block containing
  n cases incurs turnover for n-1 transitions; an empty block incurs none.
- Overtime costs 15/minute and idle time costs 10/minute. The legacy
  `CostConfig.max_overtime_minutes` does not constrain this new model.

The native assignment variables are x[i,b]. Occupation u[b] is exactly one iff
the block contains any cases; it does not allocate capacity. Load is

```math
L_b=\sum_i d_i x_{ib}+\tau(\sum_i x_{ib}-u_b).
```

The two formulations have identical feasible sets:

```math
\Phi=15\sum_b[L_b-H_b]^+ +10\sum_b[H_b-L_b]^+,
\qquad
\Psi=25\sum_b[L_b-H_b]^+ +10\tau\sum_bu_b.
```

For every complete schedule,

```math
K=10\left(\sum_b H_b-\sum_i d_i-\tau N\right),\qquad \Phi=K+\Psi.
```

Both upper and lower bounds shift by K. Relative gaps depend on the native
objective's scale; absolute gaps do not. `proven_optimal` means Gurobi returned
`OPTIMAL`, which remains tolerance-qualified. It does not mean the reported
absolute uncertainty is identically zero. Exact attempts set both MIPGap and
MIPGapAbs to zero and still report their actual termination.

Symmetry groups share site, weekday, capacity, and the complete eligible-case
set. Occupied blocks precede empty ones; minimum assigned case indices increase
within each group. Every feasible schedule has such a relabeling because those
blocks are interchangeable. Rooms with different eligibility cannot be grouped.

## Capacity proxies and data boundary

| Proxy | Construction | Interpretation |
|---|---|---|
| regular_template | Existing candidate-pool activation rule, threshold 0.25; legacy first/last-week exclusion retained and recorded | Fixed roster from recurring activity |
| observed_activity_proxy | Actual site/room/date blocks containing retained training cases | Retrospective calibration proxy; not observed master capacity |
| median_count_template | Median distinct active rooms per site/weekday, including zero-activity training weeks; round half up; choose most frequent rooms, lexical tie break | Count-calibrated template; identities may differ from history |

The split reproduces the final experiment's last 94 weeks with at least 50 cases,
using the first 72 for training: **9,289 cases, 2011-09-05 to 2013-01-21**.
Only aggregate weekly counts beyond training are used to preserve the
2013-01-28 holdout boundary. No holdout weekly instance is built. Raw service,
surgeon, and procedure identifiers are preserved before global rare-category
recoding; the audit uses raw service. Existing estimation features retain their
old recoding. Site imputation is restricted to pre-holdout history in the audit.

All 72 training weeks enter template/compatibility fitting, including each
retrospectively evaluated week. This is an in-sample structural calibration,
not a rolling-origin performance estimate.

## Outputs and interpretation

The output directory contains the required MODEL_SPEC.json, DATA_FREEZE.json,
TURNOVER_AUDIT.csv, CAPACITY_AUDIT.csv, ELIGIBILITY_AUDIT.csv,
COMPONENT_STRUCTURE.csv, PSI_SCALE_AUDIT.csv, FIXED_CAPACITY_EXACTNESS.csv,
HISTORICAL_BASELINE.csv, CALIBRATION.csv, AUDIT_SUMMARY.json, REPORT.md, and
run.log. Additional files expose per-case eligibility tiers, every native solve,
turnover sensitivity, a turnover summary, and a histogram CSV.

`SOLVE_RESULTS.csv` retains status, objective, bound, gaps, runtime, incumbent
count, objective mode, budget, and both shifted bound intervals. CSV infinite
gaps represent unavailable finite uncertainty. JSON uses null for nonfinite
numbers and includes coverage/status fields. No solver failure is converted to
an optimum. Missing incumbents or backend failures result in `INCOMPLETE` and
exit code 2, while preserving the available artifacts.

The scale table compares mean weekly Psi and mean/max absolute solver
uncertainty in the same cost units. The old epsilon_lib value of approximately
4,602 is recorded only as a reference; this audit does not compute a new VF
certificate or select the lowest-cost capacity proxy.

Historical placement comes only from actual start date, site, and room. If any
historical block is absent, historical cost is left unavailable, not computed
on a remapped or partially covered schedule. If all blocks exist, historical
cost is evaluated even when historical placement violates learned eligibility;
that feasibility flag is reported separately.

C_H is historical realized cost, C_B is booked-planner realized cost, and C_O
is a realized-duration optimization incumbent with a lower bound and uncertainty.
B/O overtime, occupation, cases per occupied block, and reassignment fractions
are reported. C_H-C_B is labeled assignment-optimization value; eligibility
feasibility must also be considered. Recommendation value C_B-C_R is reserved
for a later retrained-policy experiment.

Turnover calibration uses the gap from the preceding retained case's LEAVE_ROOM
to the next retained case's ENTER_ROOM, within site/OR/day. Plausible gaps are
preregistered as 0–240 minutes. Gaps can contain omitted noncohort activity and
must not be interpreted as identified physical turnover times. The primary
30-minute value is never changed by the empirical median.

## Validation in this change

The supplied workbook reproduced the cohort. The complete structural audit ran
on all 72 weeks; all 25 new planner tests passed. The matched Phi/Psi and symmetry
tests compare against exhaustively enumerated toy optima, including all four
turnover settings. A real zero-time-limit solve and controlled incumbent
snapshots verify termination/bound reporting.

| Proxy | Blocks min / mean / max | Mean weekly historical room-day coverage | Weeks with complete historical coverage |
|---|---:|---:|---:|
| regular_template | 94 / 94.00 / 94 | 99.19% | 46/72 |
| observed_activity_proxy | 29 / 64.82 / 77 | 100.00% | 72/72 |
| median_count_template | 67 / 67.00 / 67 | 78.82% | 0/72 |

Primary k=3 fallback is zero across all three proxies on this cohort. Each
weekly eligibility graph has one case-containing connected component, so the
day-flexible model does not acquire small independent case components.
Turnover calibration: 4,215 plausible gaps, median 28 minutes, Q25 22, Q75 36.
Primary turnover remains 30.

Two validation gates remain unresolved:

1. Dataset-sized Gurobi solves exceed the execution environment's restricted
   license. The quick audit writes all artifacts but correctly reports
   `INCOMPLETE`; mean optimal/near-optimal Psi and its absolute uncertainty are
   unavailable. Rerun the quick command with an unrestricted license before
   interpreting the new planner's optimization performance.
2. The pristine main commit `67e04be` already has 22 failing tests (62 passing),
   including a deleted `scripts.run_experiment` CLI, outdated feature expectations,
   and VF fixture/API mismatches. The changed tree has the same failure set and
   all 25 added checks pass. Those unrelated legacy failures are not hidden or
   repaired by changing the frozen VF method in this PR.

The only `src/vfcg` changes propagate real legacy oracle diagnostics/status;
there is no new planner integration or change to VF/pDCA mathematics.
