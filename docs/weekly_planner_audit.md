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
  --quick --artifact-root artifacts/weekly_audit_quick_v2 --threads 8
```

The new test file is a small mathematical regression suite, not another research
pipeline. It checks the cost identity against hand calculations and exhaustive
enumeration, formulation equivalence, safe symmetry, turnover, fixed idle
capacity, fallback, reporting, and legacy pricing compatibility.
The `fixed-capacity-planner-tests` GitHub Actions workflow runs this suite on
small models with the bundled restricted Gurobi license. Legacy estimation CI
remains separate.

Quick mode uses training weeks 0, 35, and 71, selected before examining outcomes.
All eligibility and template estimation still uses the same 72 training weeks.
The additional leave-one-week-out compatibility diagnostic always covers all
72 weeks, including in quick mode.
Quick budgets are 2 seconds per ordinary solve and 10 seconds per zero-gap
attempt. There are 162 solves, so the nominal sum of budgets is 12.6 minutes,
plus crossfit, construction, and reporting. A requested time limit may be exceeded
slightly by Gurobi during termination.

For the complete training audit, use a fresh output directory:

```bash
python scripts/run_weekly_planner_audit.py \
  --data data/UHNOperating_RoomScheduling2011-2013.xlsx \
  --artifact-root artifacts/weekly_audit_full --threads 8
```

Full defaults are 10 seconds per ordinary solve and 60 seconds per exact attempt.
Across the full sensitivity matrix, the nominal serial budget is 28.8 hours;
solves that finish early reduce this. Use `--time-limit` and `--exact-time-limit`
to change computational budgets. The chosen budgets are recorded in MODEL_SPEC.
The scientific assumptions and primary settings cannot be changed through those
flags. Run quick mode first. Phi, Psi-shifted, and Psi always receive matching
budgets, threads, and seed (0). Normal solves use MIPGap=0.01 and MIPGapAbs=1e-10;
exact attempts set both gaps to zero. All settings are recorded in MODEL_SPEC
and each solve row.

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

- Capacity is treated as exogenous by the optimization model; this is an
  assumption, not a property established for the activity-based proxies.
  Every allocated block is fixed, has zero activation
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

The two formulations have identical feasible assignment sets:

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

Three objective modes separate the two computational questions:

| Mode | Model and native objective | Purpose |
|---|---|---|
| `phi` | Direct overtime/idle formulation, Phi | Verify reformulation and its computational effect |
| `psi_shifted` | Reduced Psi formulation, K + Psi | Control for the objective constant |
| `psi` | Identical reduced formulation, Psi | Measure uncertainty on the reduced scale |

The two reduced models have identical variables, constraints, and variable
objective coefficients. A regression compares their actual solver matrices,
variable bounds/types, coefficients, and parameters; only `ObjCon` differs.
Both positive and negative K are covered. Bound translation treats
`psi_shifted` as Phi scale.

The native stopping allowance is `MIPGap * abs(native incumbent)`;
absolute uncertainty for the same bound interval is unaffected by K. See
[Gurobi's MIPGap definition](https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html#mipgap).
Compare `psi_shifted_rel_gap` with `psi_shifted_psi_rel_gap` and the absolute
gaps in PSI_SCALE_AUDIT. The allowance is not a measured error. Matched runs
need not follow identical search paths, so runtime alone cannot identify a
stopping mechanism. This new model's results cannot by themselves attribute
the old 5% VF certificate failure to scaling.

Symmetry groups share site, capacity, and the complete eligible-case
set. Occupied blocks precede empty ones; minimum assigned case indices increase
within each group. Every feasible schedule has such a relabeling because those
blocks are interchangeable. Rooms with different eligibility cannot be grouped.
Weekday is omitted because no day-dependent costs or constraints distinguish
otherwise identical blocks. If such restrictions are added, the key must also
respect them. Fixed-day eligibility alone already separates affected blocks by
their eligible-case sets. Stronger symmetry is valid; speed improvement is an
empirical question, not an established cause of the earlier bound difficulty.

## Capacity proxies and data boundary

| Proxy | Construction | Interpretation |
|---|---|---|
| regular_template | Existing candidate-pool activation rule, threshold 0.25; legacy first/last-week exclusion retained and recorded | Fixed roster from recurring activity |
| observed_activity_proxy | Actual site/room/date blocks containing retained training cases | Retrospective calibration proxy; not observed master capacity |
| median_count_template | Median distinct active rooms per site/weekday, including zero-activity training weeks; round half up; choose most frequent rooms, lexical tie break | Count-calibrated template; identities may differ from history |

`build_fixed_roster(..., source="observed_activity_proxy")` raises unless the
caller explicitly passes `allow_retrospective=True`. Even with that opt-in, the
requested week must occur in the supplied history. The audit opts in for
calibration; this does not authorize holdout or recommendation-policy use.

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

ELIGIBILITY_CROSSFIT_AUDIT excludes the evaluated week from compatibility fitting
and applies the resulting 71-week history at thresholds 1/2/3/5. It reports
in-sample and LOO primary coverage, fallback, eligible-block counts, historical
room compatibility, their deltas, and the fraction of cases whose eligible set
changes. Deltas mean LOO minus in-sample; report summaries are case-weighted.
Rosters stay fixed to isolate eligibility self-inclusion, including the
retrospective observed roster. This is not joint cross-validation of capacity
or cost performance. The final eligibility rule continues to use all 72 weeks.

## Outputs and interpretation

The output directory contains the required MODEL_SPEC.json, DATA_FREEZE.json,
TURNOVER_AUDIT.csv, CAPACITY_AUDIT.csv, ELIGIBILITY_AUDIT.csv,
ELIGIBILITY_CROSSFIT_AUDIT.csv, PSI_PROXY_COMPARISON.csv,
COMPONENT_STRUCTURE.csv, PSI_SCALE_AUDIT.csv, FIXED_CAPACITY_EXACTNESS.csv,
HISTORICAL_BASELINE.csv, CALIBRATION.csv, AUDIT_SUMMARY.json, REPORT.md, and
run.log. Additional files expose per-case eligibility tiers, every native solve,
turnover sensitivity, a turnover summary, and a histogram CSV.

`SOLVE_RESULTS.csv` retains status, objective, bound, gaps, runtime, incumbent
count, objective mode, budget, tolerances, threads, seed, both shifted bound
intervals, and gaps on both objective scales. CSV infinite
gaps represent unavailable finite uncertainty. JSON uses null for nonfinite
numbers and includes coverage/status fields. No solver failure is converted to
an optimum. Missing incumbents or backend failures result in `INCOMPLETE` and
exit code 2, while preserving the available artifacts.

The scale table compares mean weekly Psi and mean/max absolute solver
uncertainty in the same cost units. The old epsilon_lib value of approximately
4,602 is recorded only as a reference; this audit does not compute a new VF
certificate or select the lowest-cost capacity proxy.

PSI_PROXY_COMPARISON pairs the same weeks/durations across rosters. It uses
the tightest normal-budget interval across all three modes for each week and
reports mean K differences, mean signed and absolute paired Psi incumbent
differences, and bounds on the difference in mean optimal Psi. A pair of
intervals [L_A,U_A], [L_B,U_B] implies a difference in [L_A-U_B,U_A-L_B].
If any common week lacks finite intervals, Psi comparison means are unavailable.

This is a report headline because K can move substantially with allocated
capacity. But the roster also changes the feasible assignment set. Similar
mean Psi would support limited cost sensitivity only at the measured duration
vectors, not invariance of schedules, policy responses, or recommendation
value. Signed means can conceal opposing weekly changes. Similarity would
also not establish exogeneity of the observed-activity roster or solve the
historical-baseline identification problem.

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

Version 2 replaces the ambiguous `{B,O}_fraction_reassigned` column with two
explicit measures: `fraction_reassigned_raw` compares physical block IDs;
`fraction_reassigned_modulo_symmetry` minimizes case changes over valid block
bijections within each interchangeable group using maximum-overlap matching.
This handles both pure relabelings and different case partitions. Missing
historical blocks remain unmatched; C_H is never canonicalized or filled in.
The adjusted measure describes equivalence in this model, not clinical
interchangeability of actual rooms or weekdays.

Turnover calibration uses the gap from the preceding retained case's LEAVE_ROOM
to the next retained case's ENTER_ROOM, within site/OR/day. Plausible gaps are
preregistered as 0–240 minutes. Gaps can contain omitted noncohort activity and
must not be interpreted as identified physical turnover times. The primary
30-minute value is never changed by the empirical median.

## Validation in this change

The supplied workbook reproduced the cohort. The complete structural audit ran
on all 72 weeks in version 1; all 25 original planner checks passed. Version 2
has **50 passing checks**, including the new constant-only control, cross-day
symmetry, reassignment matching, crossfit, and roster opt-in. Phi/Psi and symmetry
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
day-flexible model does not acquire small independent case components. This
rules out useful decomposition by these eligibility components; it does not
prove that every exact decomposition or column-generation approach is impossible.
Turnover calibration: 4,215 plausible gaps, median 28 minutes, Q25 22, Q75 36.
Primary turnover remains 30.

Version 2 reran both the 72-week structural audit (`STRUCTURE_ONLY`) and the
three-week quick audit (`INCOMPLETE`, 162 solves rejected by the size-limited
license). All 216 primary week/roster graphs again have one case-containing
component. The LOO diagnostic contains 864 rows: 72 weeks, three rosters, and
four thresholds. At the primary three-week threshold:

| Roster | Full / LOO primary coverage | Full / LOO fallback | Cases with changed eligible set | Historical room compatibility delta |
|---|---:|---:|---:|---:|
| regular_template | 100% / 100% | 0% / 0% | 4.414% (410/9,289) | -0.431 percentage points |
| observed_activity_proxy | 100% / 100% | 0% / 0% | 4.414% (410/9,289) | -0.431 percentage points |
| median_count_template | 100% / 100% | 0% / 0% | 3.617% (336/9,289) | -0.431 percentage points |

Coverage is stable, but compatibility self-inclusion changes some constraint
sets. The diagnostic does not measure the resulting cost changes or establish
out-of-sample eligibility validity. At thresholds 1/2/3/5, the fallback deltas
are zero for all proxies; full eligible sets are not identical.

Two validation gates remain unresolved:

1. Dataset-sized Gurobi solves exceed the execution environment's restricted
   license. The quick audit writes all artifacts but correctly reports
   `INCOMPLETE`; mean optimal/near-optimal Psi and its absolute uncertainty are
   unavailable. Rerun the quick command with an unrestricted license before
   interpreting the new planner's optimization performance.
2. The pristine main commit `67e04be` already has 22 failing tests (62 passing),
   including a deleted `scripts.run_experiment` CLI, outdated feature expectations,
   and VF fixture/API mismatches. Version 2 has 112 passes and the same 22
   failures, with all 50 planner checks passing. Those
   unrelated legacy failures are not hidden or
   repaired by changing the frozen VF method in this PR.

The only `src/vfcg` changes propagate real legacy oracle diagnostics/status;
there is no new planner integration or change to VF/pDCA mathematics.

## Review decisions

Both reviews' six implementation requests are incorporated: cross-weekday
symmetry, explicit reassignment measures, LOO compatibility diagnostics,
Psi-shifted control, retrospective opt-in, and planner CI. The scientific
constraints and the final all-training eligibility model are unchanged.

The following stronger interpretations are not adopted: symmetry is proven to
explain the old epsilon; stable mean Psi makes every schedule-dependent claim
capacity-invariant; stable Psi validates exogeneity; one graph component rules
out all exact decompositions; or retained-case gaps identify physical turnover.
Institutional confirmation could motivate a different constraint model, but
observed weekdays are not imposed as constraints without it. No institutional
messages are sent and merge remains the repository owner's review decision.
