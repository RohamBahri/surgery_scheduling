# Implementation validation

## Workbook audit

The source workbook has 60,567 rows. Its SHA-256 is:

```
ce2830c04dba7b5a9444dea54ad6b56759e4cf260a902a85fbdbcfb351a4f308
```

The implemented rules produce 24,506 training and 8,774 test cases across the three included groups. The sorted retained-row list, encoded with the repository's `digest` function, has SHA-256:

```
69a594a89b5f87ac99af8b6d11460b5cd62099a84f8f5e4475904154a0b0a45e
```

The raw booking grid is verified. The training median morning-list count is 25, yielding 26 closure dates. There are 124 invalid records without a usable date in the included room registry. Every retained group-day is assignable and no surgeon-day uses the eligibility fallback. The private row manifests and features are generated locally, not committed.

## Numerical and recovery checks

The test suite covers:

- Contamination restricted to otherwise eligible records and keyed by room group.
- Nested room overlaps and the strict 15-minute boundary.
- No same-date or future-date outcome use in history scores; frozen test histories.
- Unequal-size ANOVA shrinkage and its nonpositive-variance case.
- Both exact formulations against exhaustive assignments, including unequal eligibility and LP pattern screening.
- The room-opening cost identity and outcome-blind deterministic ties.
- Reachable response intervals, the duration floor, and the DC decomposition.
- Response-limited oracle values in hand-checkable reachable/unreachable examples.
- A master without a proven optimum remaining inexact despite a successful follower check.
- Retrying incomplete cached plans and invalidating changed eligibility.
- Convex updates, training-bound descent, and comparison with the zero policy on identical libraries.
- Oracle-ratio brackets, shared calendar-week bootstrap indices, comparator percentages, and incomplete-report suppression.
- The full three-group/four-scenario pipeline on synthetic cases, including P3 reuse, Case-Error cap/resume flags, exact resume without additional optimizer calls, and rejection of changed test weights.
- Parallel/serial Shift agreement, worker failure recovery, and rejection of invalid solver incumbents.

The suite contains 32 passing tests. Run `python -m pytest -q` to reproduce them.

## Training-data solver validation

All 1,098 Booked plans completed the full minimum-cost, minimum-largest-load, alphabetical-assignment rule. All 1,098 realized-duration oracle plans reached primary optimality. The 2,196-call sweep used four worker processes, one solver thread per process, and 15-second budgets; none remained pending. Its wall time was 198.6 seconds. This measures planner computation separately from checkpoint I/O and is not an end-to-end training-time claim.

| Group | Calls | Median seconds | 95th percentile | Maximum seconds |
|---|---:|---:|---:|---:|
| TGH | 744 | 0.482 | 2.054 | 2.615 |
| TWH main | 744 | 0.074 | 1.161 | 2.053 |
| TWH day surgery | 708 | 0.002 | 0.004 | 0.022 |

The full Booked plan for TGH 2011-10-03 took 0.42 seconds. TGH 2012-09-19 took 2.04 seconds and proved cost 712.5; TGH 2011-12-06 took 2.26 seconds and proved cost 2,326.25. All 12 extreme-Shift checks on TGH 2011-10-03, 2011-12-06, and 2012-09-19 also completed; the slowest took 4.12 seconds. These are observed timings, not universal runtime guarantees.

The real CLI P1/P2 pilot completed: 28/28 daily seed plans were exact. With a one-second response-oracle budget, 31/56 oracle instances were exact and the other 25 retained valid brackets. The available license is still size limited, so larger response masters may also encounter license restrictions.

## Recovery and remaining limits

A separate four-worker checkpoint stress test completed 100 day jobs with 1,000 logged operations. Repeating it returned identical results without adding log records. Worker death recovery and consistent serial/parallel execution are also covered by tests.

Stress testing exposed memory growth from a large pattern model. A feasible LP dual screen reduces the hard positive-Shift model from 47,159 patterns to 208 without removing any possible optimum. The same tests exposed shared-database I/O problems; each day now has its own single-writer database with rollback journaling. Feasibility-query presolve is disabled after an invalid incumbent was detected, and every extracted assignment is checked before certification. These changes address observed failures without relaxed exactness or arbitrary subset-size limits.

The available Gurobi 13.0.3 license cannot solve the full policy-fitting model. Preflight detects this before learning. Full UHN Case-Error/Shift/VF training, P3, learned coefficient-bound checks, and test policy performance remain unrun here. No empirical policy-performance result is claimed. Longer or harder induced-duration solves can still remain pending and must be resumed; unfinished response oracles remain brackets.
